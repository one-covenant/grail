"""S8-v3: grail algorithm + NCCL transport, aligned to ByteCheckpoint's
parallel strategy (goal.md Phase 2 S8-v3).

Topology (torchrun 8 ranks, 1 node, matches bcp bench_nccl_naive.py):
  - ALL ranks: Qwen3-30B wrapped in FSDP1 FULL_SHARD (bf16 MixedPrecision,
    use_orig_params, sync_module_states) — the "trainer", bcp-identical wrap
  - ranks 1..7: additionally hold a FULL 57 GB miner replica on their GPU
    (grail miners are whole-model, unsharded — source-verified)

Per round:
  perturb: each rank bit-flips its LOCAL FSDP shard at S8_DENSITY (simulated
           training step, untimed)
  (1) t_gather        FSDP FULL_STATE_DICT (offload_to_cpu, rank0_only) — the
                      real 8-way all-gather + D2H, same call bcp times
  (2) t_encode        grail compute_sparse_delta vs prev CPU snapshot [verbatim]
      t_hash_publish  grail compute_weights_hash over full state    [verbatim]
  (3) t_broadcast     pack -> H2D -> NCCL broadcast to ranks 1..7
  (4) per receiver:   t_unpack -> t_apply (grail apply_sparse_delta on CUDA
                      replica, verbatim: per-tensor GPU->CPU inside)
                      -> t_verify (grail hash, verbatim) -> t_load_back
  t_sync_total_e2e    barrier-to-barrier (slowest rank)

Modes: S8_MODE=delta_faithful (default) | dense_faithful | delta_optimized
  delta_optimized: grail SEMANTICS preserved (same COO wire format, same
  xxh3 bit-exact verification, rollback on mismatch) but consumer
  implemented for large-MoE + intranet: GPU flat scatter apply (no
  per-tensor CPU roundtrip), batched pinned D2H for the verify hash.
Faithful rule (user 07-13): grail-owned steps run grail's unmodified functions
with ALL verification costs; we only write the wire glue.
"""

import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

from grail.infrastructure.delta_checkpoint import (  # noqa: E402
    apply_sparse_delta,
    compute_sparse_delta,
    compute_weights_hash,
)
from grail.shared.safetensors_utils import load_model_state_dict  # noqa: E402

MODEL = os.getenv("S8_MODEL", "/storage/openpsi/models/Qwen__Qwen3-30B-A3B")
MODE = os.getenv("S8_MODE", "delta_faithful")
ROUNDS = int(os.getenv("S8_ROUNDS", "4"))
DENSITY = float(os.getenv("S8_DENSITY", "0.01"))
BUCKET_BYTES = int(os.getenv("S8_BUCKET_MB", "1024")) * 1024 * 1024
JSON_OUT = os.getenv("S8_JSON_OUT", "")
SEED = 20260713

INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def log(rank, msg):
    print(f"[s8v3 r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(vals):
    if not vals:
        return {"n": 0}
    d = {"n": len(vals), "mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
    if len(vals) >= 2:
        d["stdev"] = statistics.stdev(vals)
    return d


def perturb_local_shards(model, density, gen):
    """Bit-flip a fraction of each rank's LOCAL FSDP shard (training-step sim)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() == 0:
                continue
            ivd = INT_VIEW.get(p.dtype)
            if ivd is None:
                continue
            data = p.data
            mask = torch.rand(data.shape, generator=gen, device=data.device) < density
            if data.is_contiguous():
                data.view(ivd)[mask] ^= 1
            else:
                tmp = data.contiguous()
                tmp.view(ivd)[mask] ^= 1
                data.copy_(tmp)
            del mask


def gather_full_state(model):
    """bcp-identical FULL_STATE_DICT gather: rank0 gets full CPU state dict."""
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        return model.state_dict()


def bcast_object(obj, rank):
    box = [obj if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


_MEM_BASELINE = {}


def _read_vm(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field):
                return int(line.split()[1]) / 2**20
    return 0.0


def reset_mem_peaks():
    torch.cuda.reset_peak_memory_stats()
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")   # reset VmHWM (peak RSS high-water mark)
    except OSError:
        pass
    # baseline at window start -> per-round increment = peak - baseline
    _MEM_BASELINE["gpu"] = torch.cuda.memory_allocated() / 2**30
    _MEM_BASELINE["cpu"] = _read_vm("VmRSS")


def read_mem_peaks(seg):
    seg["mem_gpu_alloc_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    seg["mem_gpu_reserved_peak_gb"] = torch.cuda.max_memory_reserved() / 2**30
    seg["mem_cpu_hwm_gb"] = _read_vm("VmHWM")
    seg["mem_gpu_delta_gb"] = seg["mem_gpu_alloc_peak_gb"] - _MEM_BASELINE.get("gpu", 0.0)
    seg["mem_cpu_delta_gb"] = seg["mem_cpu_hwm_gb"] - _MEM_BASELINE.get("cpu", 0.0)


def _cpu_views(flat, names, offsets, shapes):
    """Per-tensor CPU views over one flat buffer; byte stream (sorted names,
    same shapes/dtype) is identical to grail's compute_weights_hash input."""
    return {n: flat[offsets[i]:offsets[i + 1]].view(shapes[n]) for i, n in enumerate(names)}


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    gen = torch.Generator(device="cuda").manual_seed(SEED + rank)

    log(rank, f"mode={MODE} rounds={ROUNDS} density={DENSITY} world={world}")

    # ---- trainer: FSDP FULL_SHARD, bcp-identical wrap ----
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
    dist.barrier()
    log(rank, f"HF load: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=False),
        use_orig_params=True,
        sync_module_states=True,
        param_init_fn=lambda m: m.to_empty(device=torch.cuda.current_device(), recurse=False),
        device_id=torch.cuda.current_device(),
    )
    dist.barrier()
    log(rank, f"FSDP wrap: {time.perf_counter() - t0:.1f}s")

    # ---- miners: full replica on every receiver rank's GPU ----
    replica = None
    if rank >= 1:
        t0 = time.perf_counter()
        cpu = load_model_state_dict(MODEL)
        replica = {k: cpu[k].cuda() for k in sorted(cpu)}
        del cpu
        log(rank, f"miner replica on GPU: {time.perf_counter() - t0:.0f}s")
    names = None
    # FULL_STATE_DICT gather is a COLLECTIVE: all ranks must enter it even
    # though rank0_only keeps the result only on rank 0 (first submit hung
    # 600s in NCCL watchdog because ranks 1-7 skipped this call).
    snapshot = gather_full_state(model)   # untimed init snapshot, ALL ranks
    if rank == 0:
        names = sorted(snapshot.keys())
    else:
        del snapshot
        snapshot = None
    names = bcast_object(names, rank)
    shapes_all = offsets = pinned = None
    if rank >= 1:
        shapes_all = {n: tuple(replica[n].shape) for n in names}
        offsets = [0]
        for n in names:
            offsets.append(offsets[-1] + replica[n].numel())
        if MODE == "delta_optimized":
            t0 = time.perf_counter()
            pinned = torch.empty(offsets[-1], dtype=torch.bfloat16, pin_memory=True)
            log(rank, f"pinned verify buffer ready: {time.perf_counter() - t0:.0f}s")
    dist.barrier()

    rounds = []
    for r in range(ROUNDS):
        perturb_local_shards(model, DENSITY, gen)
        torch.cuda.synchronize()
        dist.barrier()
        seg = {"round": r}
        reset_mem_peaks()
        t_round = time.perf_counter()

        # (1) gather — collective, all ranks participate; timing on rank0
        t0 = time.perf_counter()
        full = gather_full_state(model)
        torch.cuda.synchronize()
        dist.barrier()
        seg["t_gather"] = time.perf_counter() - t0

        if MODE in ("delta_faithful", "delta_optimized"):
            if rank == 0:
                t0 = time.perf_counter()
                sparse, sh, st = compute_sparse_delta(full, snapshot)      # grail verbatim
                seg["t_encode"] = time.perf_counter() - t0
                seg["nnz"] = st["nonzero_params"]
                t0 = time.perf_counter()
                pub_hash = compute_weights_hash(full)                      # grail verbatim
                seg["t_hash_publish"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                snapshot = full                                            # roll snapshot
                changed = sorted(n for n in names if f"{n}.indices" in sparse)
                counts = [int(sparse[f"{n}.indices"].shape[1]) for n in changed]
                idx_flat = torch.cat(
                    [sparse[f"{n}.indices"].reshape(-1) for n in changed]).cuda()
                val_flat = torch.cat([sparse[f"{n}.values"] for n in changed]).cuda()
                torch.cuda.synchronize()
                seg["t_pack"] = time.perf_counter() - t0
                meta = (pub_hash, changed, counts, {n: sh[n] for n in changed})
                size = torch.tensor([idx_flat.numel(), val_flat.numel()],
                                    dtype=torch.long, device="cuda")
            else:
                size = torch.zeros(2, dtype=torch.long, device="cuda")
                meta = None

            t0 = time.perf_counter()
            dist.broadcast(size, src=0)
            if rank != 0:
                idx_flat = torch.empty(int(size[0]), dtype=torch.int32, device="cuda")
                val_flat = torch.empty(int(size[1]), dtype=torch.bfloat16, device="cuda")
            dist.broadcast(idx_flat, src=0)
            dist.broadcast(val_flat, src=0)
            meta = bcast_object(meta, rank)
            torch.cuda.synchronize()
            seg["t_broadcast"] = time.perf_counter() - t0
            seg["delta_gb"] = (int(size[0]) * 4 + int(size[1]) * 2) / 1024**3

            if rank >= 1 and MODE == "delta_optimized":
                pub_hash, changed, counts, sh = meta
                # apply ON GPU: same wire format, flat scatter per tensor
                # segment (async kernel launches, no CPU roundtrip); keep
                # overwritten values for rollback-on-verify-failure
                t0 = time.perf_counter()
                io = vo = 0
                backups = []
                for n, c in zip(changed, counts):
                    shp = sh[n]
                    cols = 1
                    for d in shp[1:]:
                        cols *= d
                    seg_idx = idx_flat[io:io + 2 * c].view(2, c).long()
                    flat_local = seg_idx[0] * cols + seg_idx[1]
                    tflat = replica[n].reshape(-1)
                    backups.append((n, flat_local, tflat[flat_local].clone()))
                    tflat[flat_local] = val_flat[vo:vo + c]
                    io += 2 * c
                    vo += c
                torch.cuda.synchronize()
                seg["t_apply_gpu"] = time.perf_counter() - t0

                # verify: batched pinned D2H + grail hash (verbatim function,
                # identical byte stream -> digest comparable to publisher's)
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
                    for n, fl, bak in backups:   # rollback, grail-safe semantics
                        replica[n].reshape(-1)[fl] = bak
                    raise RuntimeError(f"rank {rank} hash verify FAILED (rolled back)")
                backups.clear()
            elif rank >= 1:
                pub_hash, changed, counts, sh = meta
                t0 = time.perf_counter()
                idx_cpu, val_cpu = idx_flat.cpu(), val_flat.cpu()
                sparse = {}
                io = vo = 0
                for n, c in zip(changed, counts):
                    sparse[f"{n}.indices"] = idx_cpu[io:io + 2 * c].view(2, c)
                    sparse[f"{n}.values"] = val_cpu[vo:vo + c]
                    io += 2 * c
                    vo += c
                seg["t_unpack"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                recon = apply_sparse_delta(replica, sparse, sh)            # grail verbatim
                seg["t_apply"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                ok = compute_weights_hash(recon) == pub_hash               # grail verbatim
                seg["t_verify"] = time.perf_counter() - t0
                seg["hash_ok"] = bool(ok)
                if not ok:
                    raise RuntimeError(f"rank {rank} hash verify FAILED")
                t0 = time.perf_counter()
                for n in names:
                    replica[n].copy_(recon[n].view(replica[n].shape), non_blocking=True)
                torch.cuda.synchronize()
                seg["t_load_back"] = time.perf_counter() - t0
                del recon, sparse, idx_cpu, val_cpu
            if rank == 0:
                del sparse
        else:  # dense_faithful
            if rank == 0:
                t0 = time.perf_counter()
                pub_hash = compute_weights_hash(full)                      # grail verbatim
                seg["t_hash_publish"] = time.perf_counter() - t0
                snapshot = full
            bucket = torch.empty(BUCKET_BYTES // 2, dtype=torch.bfloat16, device="cuda")
            t0 = time.perf_counter()
            src = full if rank == 0 else replica
            cap, group, used = bucket.numel(), [], 0
            for name in names:
                n = (src[name].numel() if rank == 0 else replica[name].numel())
                if used + n > cap and group:
                    _flush(group, src, replica, bucket, rank)
                    group, used = [], 0
                group.append(name)
                used += n
            if group:
                _flush(group, src, replica, bucket, rank)
            torch.cuda.synchronize()
            seg["t_broadcast"] = time.perf_counter() - t0
            pub_hash = bcast_object(pub_hash if rank == 0 else None, rank)
            if rank >= 1:
                t0 = time.perf_counter()
                cpu_copy = {n: replica[n].cpu() for n in names}            # grail-style pull
                ok = compute_weights_hash(cpu_copy) == pub_hash            # grail verbatim
                seg["t_verify"] = time.perf_counter() - t0
                seg["hash_ok"] = bool(ok)
                del cpu_copy
                if not ok:
                    raise RuntimeError(f"rank {rank} hash verify FAILED")
            del bucket

        if rank != 0:
            del full
        # return round-transient reservations to CUDA: the gather unshards a
        # 57 GB flat unit per rank; without this, round 1 OOMs on fragmentation
        # (60 GB "reserved but unallocated", job 323190366)
        torch.cuda.empty_cache()
        dist.barrier()
        seg["t_sync_total_e2e"] = time.perf_counter() - t_round
        read_mem_peaks(seg)

        allsegs = [None] * world
        dist.all_gather_object(allsegs, seg)
        if rank == 0:
            merged = dict(allsegs[0])
            recv = [s for s in allsegs[1:]]
            for key in ("t_unpack", "t_apply", "t_verify", "t_load_back",
                        "t_apply_gpu", "t_verify_d2h", "t_verify_hash",
                        "mem_gpu_alloc_peak_gb", "mem_gpu_reserved_peak_gb",
                        "mem_cpu_hwm_gb"):
                vals = [s[key] for s in recv if key in s]
                if vals:
                    merged[f"{key}_max"] = max(vals)
                    merged[f"{key}_mean"] = sum(vals) / len(vals)
            merged["hash_ok_all"] = all(s.get("hash_ok", True) for s in recv)
            rounds.append(merged)
            log(rank, f"round {r}: " + " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(merged.items()) if k != "round"))

    if rank == 0:
        steady = rounds[1:]
        summary = {}
        for key in sorted({k for x in steady for k in x}):
            vals = [float(x[key]) for x in steady
                    if isinstance(x.get(key), (int, float)) and key != "round"]
            if vals:
                summary[key] = stats(vals)
        print(f"\n### S8-v3 {MODE}+NCCL, FSDP 8-way (bcp-aligned), "
              f"Qwen3-30B, density={DENSITY}, steady n={len(steady)}")
        for key, s in summary.items():
            if s.get("n"):
                line = f"  {key}: mean={s['mean']:.3f}"
                if "stdev" in s:
                    line += f" stdev={s['stdev']:.3f}"
                print(line + f" (n={s['n']})")
        if JSON_OUT:
            with open(JSON_OUT, "w") as f:
                json.dump({"mode": MODE, "density": DENSITY, "world": world,
                           "rounds": rounds, "steady_stats": summary}, f,
                          indent=2, default=str)
            print(f"JSON written to {JSON_OUT}")

    dist.barrier()
    dist.destroy_process_group()


def _flush(group, src, replica, bucket, rank):
    off = 0
    if rank == 0:
        for name in group:
            t = src[name]
            bucket[off:off + t.numel()].copy_(t.reshape(-1).cuda() if not t.is_cuda else t.reshape(-1))
            off += t.numel()
    else:
        off = sum(replica[n].numel() for n in group)
    dist.broadcast(bucket[:off], src=0)
    if rank != 0:
        off = 0
        for name in group:
            t = replica[name]
            t.reshape(-1).copy_(bucket[off:off + t.numel()])
            off += t.numel()


if __name__ == "__main__":
    main()
