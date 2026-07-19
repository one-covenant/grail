"""S9: REAL-training-delta bench on HumanEval (goal.md Phase 2 S9).

Identical to the S8-v4/v5 two-node harness (trainer 8-rank FSDP on node A,
8 whole-model miners on node B, grail semantics with optimized consumer,
time + CPU/GPU peak instrumentation) with ONE change: the per-round weight
perturbation is a REAL SFT optimizer step instead of a synthetic bit-flip.

Training step (per round, untimed as before — it stands in for "training"):
  data   : openai_humaneval parquet, sample = prompt + canonical_solution
  loss   : causal LM (labels = input_ids, prompt tokens not masked — this is
           a delta GENERATOR, not a model-quality exercise)
  optim  : AdamW per user RL config — lr=3e-6 constant, wd=0.003,
           betas=(0.9,0.999), eps=1e-8, grad_clip=1.0; 1 step/round

The headline NEW metric: REAL density (nnz/total) and real delta size, as
produced by actual gradients through AdamW on bf16 weights.

Modes: S9_MODE=delta_optimized (nccl) | delta_disk
"""

import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")
sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail/docs/scripts")

from s8_nccl_bench_fsdp import _cpu_views, read_mem_peaks, reset_mem_peaks, stats  # noqa: E402

from grail.infrastructure.delta_checkpoint import (  # noqa: E402
    compute_sparse_delta,
    compute_weights_hash,
)
from grail.shared.safetensors_utils import load_model_state_dict  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

MODEL = os.getenv("S8_MODEL", "/storage/openpsi/models/Qwen__Qwen3-30B-A3B")
MODE = os.getenv("S9_MODE", os.getenv("S8_MODE", "delta_optimized"))
ROUNDS = int(os.getenv("S8_ROUNDS", "4"))
T = int(os.getenv("S8_TRAINER_RANKS", "8"))
DISK_DIR = os.getenv("S8_DISK_DIR", "/storage/openpsi/users/pengzai.pyq/grail-runs/tmp/s9_disk")
# S9_DISK_COMPRESS = "none" | "v3"   (v3 = grail production sparse_codec + zstd L1)
DISK_COMPRESS = os.getenv("S9_DISK_COMPRESS", "none").lower()
DATA = (os.getenv("S9_DATA") or
        "/storage/openpsi/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
LR = float(os.getenv("S9_LR") or "3e-6")          # user 07-13: RL 训练配置
WD = float(os.getenv("S9_WD", "0.003"))
GRAD_CLIP = float(os.getenv("S9_GRAD_CLIP", "1.0"))
BATCH = int(os.getenv("S9_BATCH", "2"))
SEQ = int(os.getenv("S9_SEQ", "1024"))
JSON_OUT = os.getenv("S8_JSON_OUT", "")
SEED = 20260713


def log(rank, msg):
    print(f"[s9 r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    is_trainer = rank < T
    is_miner = rank >= T
    trainer_pg = dist.new_group(ranks=list(range(T)))
    miner_bcast_pg = dist.new_group(ranks=[0] + list(range(T, world)))
    log(rank, f"mode={MODE} world={world} role={'trainer' if is_trainer else 'miner'}")

    def gather_full_state(model, force_cpu=False):
        # S16 (2026-07-16): S9_OFFLOAD_CPU=false forces gather to hold the full
        # state_dict on rank 0's GPU instead of the FSDP default pipelined
        # gather+D2H per module. This makes the "GPU all-gather + CPU delta"
        # cost visible in mem_gpu_alloc_peak_gb, at the price of +2-3s D2H
        # inside grail's compute_sparse_delta. Default True keeps S12 numbers
        # reproducible.
        #
        # force_cpu overrides the env var and pins gather to CPU. Used by init
        # snapshot (base_state semantics = load from disk/R2, always CPU) so
        # only the per-round `full` gather goes to GPU under S16.
        offload = force_cpu or (os.getenv("S9_OFFLOAD_CPU", "true").lower() != "false")
        cfg = FullStateDictConfig(offload_to_cpu=offload, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()

    def bcast_object_world(obj):
        box = [obj if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    # ---- setup ----
    model = optimizer = batches = replica = snapshot = pinned = None
    shapes_all = offsets = None
    if is_trainer:
        t0 = time.perf_counter()
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
                low_cpu_mem_usage=True)
        else:
            cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
        dist.barrier(group=trainer_pg)
        log(rank, f"HF load: {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        # per-layer auto_wrap is REQUIRED for real training: bcp's single-unit
        # wrap would materialize full 57GB params + 57GB grads in fwd/bwd -> OOM.
        # FULL_STATE_DICT gather API/semantics unchanged (per-unit transients).
        import functools
        layer_cls = {type(m) for m in model.modules()
                     if type(m).__name__.endswith("DecoderLayer")}
        assert layer_cls, "no DecoderLayer class found for auto_wrap"
        model = FSDP(
            model, process_group=trainer_pg,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=layer_cls),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            # 2026-07-17: reduce_dtype=fp32 tested but reverted — PyTorch AdamW state
            # dtype matches p.dtype (bf16 under MixedPrecision param_dtype=bf16),
            # NOT p.grad.dtype. So reduce_dtype=fp32 only affects intermediate
            # grad reduce, downcasts back to bf16 before optim.step(). State stays
            # bf16. See S19 iter 1 evidence: alloc peak unchanged 32.89 → 32.89.
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                           cast_forward_inputs=False),
            use_orig_params=True, sync_module_states=True,
            param_init_fn=lambda m: m.to_empty(
                device=torch.cuda.current_device(), recurse=False),
            device_id=torch.cuda.current_device(),
        )
        dist.barrier(group=trainer_pg)
        log(rank, f"FSDP wrap: {time.perf_counter() - t0:.1f}s")

        # user-specified RL optimizer config: lr=3e-6 constant, no warmup,
        # wd=0.003, betas=(0.9,0.999), eps=1e-8, grad clip 1.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                      betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=WD)

        # S16 v3 (2026-07-16): Pre-allocate AdamW state (exp_avg + exp_avg_sq,
        # fp32 each) for ALL params so trainer GPU baseline reflects steady-state
        # training, not lazy-alloc early phase. Qwen3-30B-A3B is MoE and PyTorch
        # AdamW only alloc-s state when p.grad is not None. Without this, first
        # few SFT steps only touch routed experts → optim state stays partial →
        # GPU peak is virtually low and doesn't reflect what a real production
        # trainer would see (all experts get routed to across many steps).
        #
        # Two modes (via S9_PREALLOC_OPTIM):
        #   - "bf16" (or "true"): pre-alloc AdamW state matching grad dtype (bf16
        #     under FSDP MixedPrecision). This is what grail production actually
        #     runs. Baseline +15 GB.
        #   - "fp32_shadow": also alloc a SHADOW fp32 buffer (2× per param) that
        #     is NOT used by the optimizer, purely to reserve GPU memory as if
        #     we were running fp32 master weights. Simulates the "if grail used
        #     fp32 optim state" baseline. Baseline +30 GB shadow +15 GB real bf16.
        # Default off keeps S12 numbers reproducible.
        prealloc_mode = os.getenv("S9_PREALLOC_OPTIM", "false").lower()
        if prealloc_mode in ("true", "bf16", "fp32_shadow"):
            # Safety log: baseline before prealloc so we can catch OOM risk
            # before gather (target: baseline < 65 GB, else abort likely).
            torch.cuda.synchronize()
            if rank == 0:
                pre_alloc = torch.cuda.memory_allocated() / 2**30
                pre_reserved = torch.cuda.memory_reserved() / 2**30
                log(rank, f"[S16 safety] pre-prealloc: alloc={pre_alloc:.2f} GB "
                          f"reserved={pre_reserved:.2f} GB (mode={prealloc_mode})")

            n_prealloced = 0
            # bf16 state matching grad dtype (real optimizer state)
            for p in model.parameters():
                if p.requires_grad:
                    state = optimizer.state[p]
                    if "exp_avg" not in state:
                        state["step"] = torch.zeros((), dtype=torch.float32,
                                                    device=p.device)
                        # Match param dtype (bf16 under MixedPrecision) so
                        # torch._foreach_lerp_ doesn't complain about dtype
                        # mismatch between exp_avg and grad.
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                        n_prealloced += 1

            # fp32 shadow: separate buffers, NOT used by optimizer, purely to
            # reserve GPU memory as if fp32 optim state were in use.
            # 2026-07-17 upgrade: 3 tensors per param, aligned with bcp
            # fake_train_state (master_weights_fp32 + adam_m_fp32 + adam_v_fp32).
            # See bytecheckpoint-runs/scripts/ais_submit/bench_qwen3_30b_bcp_v2.py:280-291.
            # Per rank size = 3 × 4 bytes × n_params / world_size = 12 bytes/param.
            fp32_shadow_state = None
            if prealloc_mode == "fp32_shadow":
                fp32_shadow_state = []
                for p in model.parameters():
                    if p.requires_grad:
                        fp32_shadow_state.append((
                            torch.zeros_like(p.data, dtype=torch.float32),  # master_weights_fp32
                            torch.zeros_like(p.data, dtype=torch.float32),  # adam_m_fp32
                            torch.zeros_like(p.data, dtype=torch.float32),  # adam_v_fp32
                        ))
                # Pin to model so it doesn't get GC'd across rounds
                model._fp32_shadow_state = fp32_shadow_state

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if rank == 0:
                post_alloc = torch.cuda.memory_allocated() / 2**30
                post_reserved = torch.cuda.memory_reserved() / 2**30
                shadow_note = (f" + fp32 shadow ({len(fp32_shadow_state)} pairs)"
                               if fp32_shadow_state else "")
                log(rank, f"[S16 safety] post-prealloc: alloc={post_alloc:.2f} GB "
                          f"reserved={post_reserved:.2f} GB "
                          f"(+{post_alloc - pre_alloc:.2f} GB, "
                          f"{n_prealloced} bf16 state pairs{shadow_note})")
                if post_alloc > 65:
                    log(rank, f"[S16 safety] ⚠️ post-prealloc alloc "
                              f"{post_alloc:.1f} GB > 65 GB, gather may OOM.")

        # HumanEval batches: prompt + canonical_solution, causal LM loss
        import pandas as pd
        if DATA.endswith(".jsonl"):
            df = pd.read_json(DATA, lines=True)
        elif DATA.endswith(".json"):
            df = pd.read_json(DATA)          # JSON array (e.g. LogiQA)
        else:
            df = pd.read_parquet(DATA)
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        cols = set(df.columns)
        if {"problem", "solution"} <= cols:     # MATH / MATH-500
            texts = [r["problem"] + "\n" + r["solution"] for _, r in df.iterrows()]
        elif "canonical_solution" in cols:      # HumanEval (HF schema)
            texts = [r["prompt"] + r["canonical_solution"] for _, r in df.iterrows()]
        elif "extra_info" in cols:              # GSM8K (verl schema)
            texts = [r["extra_info"]["question"] + "\n" + r["extra_info"]["answer"]
                     for _, r in df.iterrows()]
        elif {"context", "query", "options", "correct_option"} <= cols:  # LogiQA
            texts = [r["context"] + "\n" + r["query"] + "\n"
                     + "\n".join(r["options"])
                     + "\nAnswer: " + r["options"][r["correct_option"]]
                     for _, r in df.iterrows()]
        elif {"question", "answer"} <= cols:    # GSM8K (raw HF schema)
            texts = [r["question"] + "\n" + r["answer"] for _, r in df.iterrows()]
        else:
            raise ValueError(f"unknown dataset schema: {sorted(cols)}")
        rng = random.Random(SEED + rank)
        enc = [tok(t, truncation=True, max_length=SEQ, return_tensors="pt")
               for t in texts]
        batches = [e["input_ids"] for e in enc if e["input_ids"].shape[1] >= 32]
        log(rank, f"HumanEval: {len(batches)} usable samples "
                  f"(lr={LR}, B={BATCH}, seq<={SEQ})")

        def train_step():
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(BATCH):
                ids = batches[rng.randrange(len(batches))].cuda()
                out = model(input_ids=ids, labels=ids)
                (out.loss / BATCH).backward()
            model.clip_grad_norm_(GRAD_CLIP)   # FSDP-aware grad clipping
            optimizer.step()
            return float(out.loss.detach())

        snapshot = gather_full_state(model, force_cpu=True)
        if rank != 0:
            del snapshot
            snapshot = None
    else:
        t0 = time.perf_counter()
        cpu = load_model_state_dict(MODEL)
        replica = {k: cpu[k].cuda() for k in sorted(cpu)}
        del cpu
        shapes_all = {n: tuple(replica[n].shape) for n in sorted(replica)}
        offsets = [0]
        for n in sorted(replica):
            offsets.append(offsets[-1] + replica[n].numel())
        log(rank, f"miner replica on GPU: {time.perf_counter() - t0:.0f}s")
        t0 = time.perf_counter()
        pinned = torch.empty(offsets[-1], dtype=torch.bfloat16, pin_memory=True)
        log(rank, f"pinned verify buffer: {time.perf_counter() - t0:.0f}s")

    names = bcast_object_world(sorted(snapshot.keys()) if rank == 0 else None)
    dist.barrier()

    rounds = []
    for r in range(ROUNDS):
        if is_trainer:
            t0 = time.perf_counter()
            loss = train_step()
            torch.cuda.synchronize()
            if rank == 0:
                log(rank, f"round {r} train_step: loss={loss:.4f} "
                          f"({time.perf_counter() - t0:.1f}s, untimed)")
        dist.barrier()
        seg = {"round": r, "role": "trainer" if is_trainer else "miner"}
        reset_mem_peaks()
        t_round = time.perf_counter()

        if is_trainer:
            t0 = time.perf_counter()
            if rank == 0:
                log(rank, f"[dbg r{r}] gather_full_state START")
            full = gather_full_state(model)
            torch.cuda.synchronize()
            if rank == 0:
                seg["t_gather"] = time.perf_counter() - t0
                log(rank, f"[dbg r{r}] gather_full_state END wall={seg['t_gather']:.2f}s "
                          f"tensors={len(full)}")
        dist.barrier()
        if rank == 0:
            log(rank, f"[dbg r{r}] barrier after gather passed")

        if rank == 0:
            t0 = time.perf_counter()
            sparse, sh, st = compute_sparse_delta(full, snapshot)   # grail verbatim
            seg["t_encode"] = time.perf_counter() - t0
            seg["nnz"] = st["nonzero_params"]
            seg["density"] = st["nonzero_params"] / st["total_params"]
            log(rank, f"[dbg r{r}] compute_sparse_delta wall={seg['t_encode']:.2f}s "
                      f"nnz={seg['nnz']} density={seg['density']:.4%}")
            t0 = time.perf_counter()
            pub_hash = compute_weights_hash(full)                   # grail verbatim
            seg["t_hash_publish"] = time.perf_counter() - t0
            log(rank, f"[dbg r{r}] hash wall={seg['t_hash_publish']:.2f}s")
            t0 = time.perf_counter()
            # S16 (2026-07-16): under S9_OFFLOAD_CPU=false, `full` lives on
            # rank 0 GPU. Copying it to CPU here keeps `snapshot` in the
            # base_state semantics (CPU-resident, like a load-from-disk base)
            # and frees the GPU 60GB before the next round's gather. This D2H
            # is a real cost of the "GPU all-gather + CPU snapshot" pipeline;
            # under S9_OFFLOAD_CPU=true it's a no-op reference assignment (full
            # already on CPU).
            snapshot = {k: v.cpu() for k, v in full.items()}
            del full
            torch.cuda.synchronize()
            seg["t_snapshot_d2h"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            changed = sorted(n for n in names if f"{n}.indices" in sparse)
            counts = [int(sparse[f"{n}.indices"].shape[1]) for n in changed]
            idx_flat = torch.cat(
                [sparse[f"{n}.indices"].reshape(-1) for n in changed])
            val_flat = torch.cat([sparse[f"{n}.values"] for n in changed])
            # delta stays on CPU at ANY density (real density may be huge;
            # a 50% delta's int32 index stream alone is ~120 GB)
            torch.cuda.synchronize()
            seg["t_pack"] = time.perf_counter() - t0
            meta = (pub_hash, changed, counts, {n: sh[n] for n in changed})
            size = torch.tensor([idx_flat.numel(), val_flat.numel()],
                                dtype=torch.long, device="cuda")
            seg["delta_gb"] = (int(size[0]) * 4 + int(size[1]) * 2) / 1024**3
            # keep `sparse` when disk zstd v2/v3 (grail production encoder needs
            # the per-tensor sparse_tensors dict + shapes dict, not flat)
            if not (MODE == "delta_disk" and DISK_COMPRESS in ("v2", "v3")):
                del sparse

        if MODE == "delta_disk":
            use_zstd = DISK_COMPRESS in ("v2", "v3")
            fname = f"delta_r{r}.bin.zst" if use_zstd else f"delta_r{r}.safetensors"
            fpath = os.path.join(DISK_DIR, fname)
            mpath = os.path.join(DISK_DIR, f"delta_r{r}.meta.json")
            if rank == 0:
                os.makedirs(DISK_DIR, exist_ok=True)
                if use_zstd:
                    # grail production sparse codec (delta encoding + zstd L1).
                    # v2 = flat 1D indices, v3 = 2D COO rows+cols separate.
                    # Direct-import module file to bypass grail package __init__
                    # (which pulls dotenv and other prod-only deps).
                    import sys as _sys
                    _codec_dir = "/storage/openpsi/users/pengzai.pyq/grail/grail/infrastructure"
                    if _codec_dir not in _sys.path:
                        _sys.path.insert(0, _codec_dir)
                    import sparse_codec as _sc
                    if DISK_COMPRESS == "v2":
                        encode_fn = _sc.encode_sparse_delta_v2
                    else:
                        encode_fn = _sc.encode_sparse_delta_v3
                    log(rank, f"[dbg r{r}] encode_fn={DISK_COMPRESS} num_tensors={len(changed)} — building sparse_v dict")
                    t_c0 = time.perf_counter()
                    sparse_v = {}
                    for n in changed:
                        sparse_v[f"{n}.indices"] = sparse[f"{n}.indices"]
                        sparse_v[f"{n}.values"] = sparse[f"{n}.values"]
                    shapes_v = {n: list(sh[n]) for n in changed}
                    log(rank, f"[dbg r{r}] sparse_v dict built ({time.perf_counter()-t_c0:.2f}s), calling encoder ({DISK_COMPRESS})...")

                    # heartbeat thread: encoder can hang for long time on 30B moe
                    # (18867 tensors × per-tensor overhead + single-thread zstd).
                    # Print progress every 15s so we can localize hang.
                    import threading as _th
                    _stop = _th.Event()
                    _enc_start = time.perf_counter()
                    def _hb():
                        while not _stop.wait(15):
                            log(rank, f"[dbg r{r}] encoder heartbeat: still running {time.perf_counter()-_enc_start:.1f}s")
                    _hbt = _th.Thread(target=_hb, daemon=True)
                    _hbt.start()
                    try:
                        blob = encode_fn(sparse_v, shapes_v)
                    finally:
                        _stop.set()
                        _hbt.join(timeout=1)
                    del sparse, sparse_v
                    seg["t_disk_compress"] = time.perf_counter() - t_c0
                    seg["delta_gb_compressed"] = len(blob) / 1024**3
                    seg["compression_ratio"] = seg["delta_gb"] / seg["delta_gb_compressed"] \
                        if seg["delta_gb_compressed"] > 0 else 0
                    seg["disk_compress_variant"] = DISK_COMPRESS
                    log(rank, f"[dbg r{r}] encode DONE wall={seg['t_disk_compress']:.2f}s "
                              f"blob={seg['delta_gb_compressed']:.3f} GB ratio={seg['compression_ratio']:.2f}x")
                    t0 = time.perf_counter()
                    log(rank, f"[dbg r{r}] write START -> {fpath}")
                    with open(fpath, "wb") as f:
                        f.write(blob)
                    log(rank, f"[dbg r{r}] write done {time.perf_counter()-t0:.2f}s, fsync START")
                else:
                    t0 = time.perf_counter()
                    save_file({"idx": idx_flat, "val": val_flat}, fpath)
                fd = os.open(fpath, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                if use_zstd:
                    log(rank, f"[dbg r{r}] fsync done, writing meta.json")
                ph, ch, co, shp = meta
                with open(mpath, "w") as f:
                    json.dump({"hash": ph, "changed": ch, "counts": co,
                               "shapes": {k: list(v) for k, v in shp.items()}}, f)
                seg["t_disk_write"] = time.perf_counter() - t0
                if use_zstd:
                    log(rank, f"[dbg r{r}] rank0 disk write TOTAL wall={seg['t_disk_write']:.2f}s, entering barrier")
            dist.barrier()
            if rank == 0 and use_zstd:
                log(rank, f"[dbg r{r}] barrier after disk write passed")
            if is_miner:
                t0 = time.perf_counter()
                if use_zstd:
                    if rank == T:  # first miner logs (avoid 8x duplicates)
                        log(rank, f"[dbg r{r}] miner START disk read {fpath}")
                    import sys as _sys
                    _codec_dir = "/storage/openpsi/users/pengzai.pyq/grail/grail/infrastructure"
                    if _codec_dir not in _sys.path:
                        _sys.path.insert(0, _codec_dir)
                    import sparse_codec as _sc
                    if DISK_COMPRESS == "v2":
                        decode_fn = _sc.decode_sparse_delta_v2
                    else:
                        decode_fn = _sc.decode_sparse_delta_v3
                    _t_rd = time.perf_counter()
                    with open(fpath, "rb") as f:
                        blob = f.read()
                    if rank == T:
                        log(rank, f"[dbg r{r}] miner blob read {time.perf_counter()-_t_rd:.2f}s "
                                  f"({len(blob)/1024**3:.3f} GB), decode START")
                    t_d0 = time.perf_counter()

                    # heartbeat for miner decode (might hang similarly)
                    import threading as _th
                    _stop = _th.Event()
                    _dec_start = time.perf_counter()
                    _rank_local = rank
                    def _hb_m():
                        while not _stop.wait(15):
                            if _rank_local == T:
                                log(_rank_local, f"[dbg r{r}] miner decode heartbeat: still running {time.perf_counter()-_dec_start:.1f}s")
                    _hbt = _th.Thread(target=_hb_m, daemon=True)
                    _hbt.start()
                    try:
                        decoded, _ = decode_fn(blob)
                    finally:
                        _stop.set()
                        _hbt.join(timeout=1)
                    del blob
                    if rank == T:
                        log(rank, f"[dbg r{r}] miner decode DONE wall={time.perf_counter()-t_d0:.2f}s, reading meta")
                    with open(mpath) as f:
                        m = json.load(f)
                    ch_d = m["changed"]
                    shp_d = m["shapes"]
                    # v3 decoded indices are 2D (2, nnz) COO — reshape(-1) gives
                    # concat [rows, cols] per tensor, matching raw path layout.
                    # v2 decoded indices are 1D flat linear (row*ncols + col),
                    # need to unflatten to (rows, cols) COO before concat.
                    _t_u = time.perf_counter()
                    if DISK_COMPRESS == "v2":
                        idx_parts = []
                        for n in ch_d:
                            flat_idx = decoded[f"{n}.indices"].to(torch.int64)
                            shape_n = shp_d[n]
                            ncols = 1
                            for _d in shape_n[1:]:
                                ncols *= _d
                            if ncols == 1:
                                rows = flat_idx.to(torch.int32)
                                cols = torch.zeros_like(rows)
                            else:
                                rows = (flat_idx // ncols).to(torch.int32)
                                cols = (flat_idx %  ncols).to(torch.int32)
                            idx_parts.append(rows)
                            idx_parts.append(cols)
                        idx_flat = torch.cat(idx_parts)
                    else:
                        idx_flat = torch.cat(
                            [decoded[f"{n}.indices"].reshape(-1).to(torch.int32)
                             for n in ch_d])
                    val_flat = torch.cat([decoded[f"{n}.values"] for n in ch_d])
                    del decoded
                    if rank == T:
                        log(rank, f"[dbg r{r}] miner unflatten+concat wall={time.perf_counter()-_t_u:.2f}s")
                    seg["t_disk_decompress"] = time.perf_counter() - t_d0
                    meta = (m["hash"], m["changed"], m["counts"], m["shapes"])
                else:
                    d = load_file(fpath, device="cpu")
                    idx_flat = d["idx"].clone()
                    val_flat = d["val"].clone()
                    with open(mpath) as f:
                        m = json.load(f)
                    meta = (m["hash"], m["changed"], m["counts"], m["shapes"])
                    del d
                torch.cuda.synchronize()
                seg["t_disk_read"] = time.perf_counter() - t0
                if use_zstd and rank == T:
                    log(rank, f"[dbg r{r}] miner disk_read TOTAL wall={seg['t_disk_read']:.2f}s")
        else:
            if rank == 0 or is_miner:
                CH = 2_000_000_000  # 2B elems per chunk (8 GB i32 / 4 GB bf16 staging)
                if rank != 0:
                    size = torch.zeros(2, dtype=torch.long, device="cuda")
                    meta = None
                t0 = time.perf_counter()
                dist.broadcast(size, src=0, group=miner_bcast_pg)
                n_idx, n_val = int(size[0]), int(size[1])
                if rank != 0:
                    idx_flat = torch.empty(n_idx, dtype=torch.int32)
                    val_flat = torch.empty(n_val, dtype=torch.bfloat16)
                for tens, n_total, gdtype in ((idx_flat, n_idx, torch.int32),
                                              (val_flat, n_val, torch.bfloat16)):
                    gbuf = torch.empty(min(CH, max(n_total, 1)), dtype=gdtype, device="cuda")
                    for off in range(0, n_total, CH):
                        n = min(CH, n_total - off)
                        if rank == 0:
                            gbuf[:n].copy_(tens[off:off + n], non_blocking=True)
                        dist.broadcast(gbuf[:n], src=0, group=miner_bcast_pg)
                        if rank != 0:
                            tens[off:off + n].copy_(gbuf[:n], non_blocking=True)
                    torch.cuda.synchronize()
                    del gbuf
                box = [meta if rank == 0 else None]
                dist.broadcast_object_list(box, src=0, group=miner_bcast_pg)
                meta = box[0]
                torch.cuda.synchronize()
                seg["t_broadcast"] = time.perf_counter() - t0

        if is_miner:
            pub_hash, changed, counts, sh = meta
            t0 = time.perf_counter()
            io = vo = 0
            for n, c in zip(changed, counts):
                shp = sh[n]
                cols = 1
                for d_ in shp[1:]:
                    cols *= d_
                seg_idx = idx_flat[io:io + 2 * c].view(2, c).cuda(non_blocking=True).long()
                seg_val = val_flat[vo:vo + c].cuda(non_blocking=True)
                flat_local = seg_idx[0] * cols + seg_idx[1]
                replica[n].reshape(-1)[flat_local] = seg_val
                io += 2 * c
                vo += c
            torch.cuda.synchronize()
            seg["t_apply_gpu"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            for i, n in enumerate(names):
                pinned[offsets[i]:offsets[i + 1]].copy_(
                    replica[n].reshape(-1), non_blocking=True)
            torch.cuda.synchronize()
            seg["t_verify_d2h"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            ok = compute_weights_hash(
                _cpu_views(pinned, names, offsets, shapes_all)) == pub_hash
            seg["t_verify_hash"] = time.perf_counter() - t0
            seg["hash_ok"] = bool(ok)
            if not ok:
                raise RuntimeError(f"rank {rank} verify FAILED")

        if is_trainer and rank != 0:
            del full
        torch.cuda.empty_cache()
        dist.barrier()
        seg["t_sync_total_e2e"] = time.perf_counter() - t_round
        read_mem_peaks(seg)
        if MODE == "delta_disk" and rank == 0:
            for p in (fpath, mpath):
                if os.path.exists(p):
                    os.remove(p)

        allsegs = [None] * world
        dist.all_gather_object(allsegs, seg)
        if rank == 0:
            merged = dict(allsegs[0])
            miners = [s for s in allsegs if s["role"] == "miner"]
            trainers = [s for s in allsegs if s["role"] == "trainer"]
            for key in ("t_apply_gpu", "t_verify_d2h", "t_verify_hash",
                        "t_broadcast", "t_disk_read", "t_disk_decompress",
                        "mem_gpu_alloc_peak_gb",
                        "mem_gpu_reserved_peak_gb", "mem_cpu_hwm_gb",
                        "mem_gpu_delta_gb", "mem_cpu_delta_gb"):
                vals = [s[key] for s in miners if key in s]
                if vals:
                    merged[f"miner_{key}_max"] = max(vals)
                    merged[f"miner_{key}_mean"] = sum(vals) / len(vals)
            for key in ("mem_gpu_alloc_peak_gb", "mem_gpu_reserved_peak_gb",
                        "mem_cpu_hwm_gb", "mem_gpu_delta_gb", "mem_cpu_delta_gb"):
                vals = [s[key] for s in trainers if key in s]
                if vals:
                    merged[f"trainer_{key}_max"] = max(vals)
            merged["hash_ok_all"] = all(s.get("hash_ok", True) for s in miners)
            rounds.append(merged)
            log(rank, f"round {r}: " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(merged.items()) if k not in ("round", "role")))

    if rank == 0:
        steady = rounds[1:]
        summary = {}
        for key in sorted({k for x in steady for k in x}):
            vals = [float(x[key]) for x in steady
                    if isinstance(x.get(key), (int, float)) and key != "round"]
            if vals:
                summary[key] = stats(vals)
        print(f"\n### S9 REAL-delta (HumanEval SFT, lr={LR}) {MODE}, 2-node split, "
              f"Qwen3-30B, steady n={len(steady)}")
        for key, s in summary.items():
            if s.get("n"):
                line = f"  {key}: mean={s['mean']:.4f}"
                if "stdev" in s:
                    line += f" stdev={s['stdev']:.4f}"
                print(line + f" (n={s['n']})")
        if JSON_OUT:
            with open(JSON_OUT, "w") as f:
                json.dump({"mode": MODE, "lr": LR, "data": DATA, "rounds": rounds,
                           "steady_stats": summary}, f, indent=2, default=str)
            print(f"JSON written to {JSON_OUT}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
