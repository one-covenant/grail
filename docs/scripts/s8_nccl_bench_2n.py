"""S8-v4: grail + NCCL, TWO-NODE split topology (goal.md Phase 2 S8-v4).

User requirement (07-13): no GPU sharing between roles. 16 ranks over 2 nodes:
  node A = ranks 0..7   : FSDP FULL_SHARD trainer ONLY (bcp-aligned wrap,
                          in its own process group)
  node B = ranks 8..15  : 8 pure miners, each a full 57 GB replica on its GPU
Broadcast rank0 -> miners crosses nodes over IB/RDMA in a dedicated group.

Modes: S8_MODE=delta_optimized (default) | delta_faithful | delta_disk
  delta_disk: identical to delta_optimized except transport = CPFS disk
  (rank0 writes safetensors+meta.json with fsync; each miner reads from
  CPFS — node B never wrote the file, so reads are naturally cold)
Reuses helpers from s8_nccl_bench_fsdp (same dir).
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
sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail/docs/scripts")

from s8_nccl_bench_fsdp import (  # noqa: E402
    _cpu_views,
    perturb_local_shards,
    read_mem_peaks,
    reset_mem_peaks,
    stats,
)

from grail.infrastructure.delta_checkpoint import (  # noqa: E402
    apply_sparse_delta,
    compute_sparse_delta,
    compute_weights_hash,
)
from grail.shared.safetensors_utils import load_model_state_dict  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

MODEL = os.getenv("S8_MODEL", "/storage/openpsi/models/Qwen__Qwen3-30B-A3B")
MODE = os.getenv("S8_MODE", "delta_optimized")
ROUNDS = int(os.getenv("S8_ROUNDS", "4"))
DENSITY = float(os.getenv("S8_DENSITY", "0.01"))
T = int(os.getenv("S8_TRAINER_RANKS", "8"))
DISK_DIR = os.getenv("S8_DISK_DIR", "/storage/openpsi/users/pengzai.pyq/grail-runs/tmp/s8_disk")
JSON_OUT = os.getenv("S8_JSON_OUT", "")
SEED = 20260713


def log(rank, msg):
    print(f"[s8v4 r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    gen = torch.Generator(device="cuda").manual_seed(SEED + rank)
    log(rank, f"mode={MODE} world={world} trainer_ranks={T} "
              f"role={'trainer' if is_trainer else 'miner'} local_rank={local_rank}")

    def gather_full_state(model):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()

    def bcast_object_world(obj):
        box = [obj if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    # ---- setup per role ----
    model = replica = snapshot = pinned = None
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
        model = FSDP(
            model,
            process_group=trainer_pg,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                           cast_forward_inputs=False),
            use_orig_params=True,
            sync_module_states=True,
            param_init_fn=lambda m: m.to_empty(
                device=torch.cuda.current_device(), recurse=False),
            device_id=torch.cuda.current_device(),
        )
        dist.barrier(group=trainer_pg)
        log(rank, f"FSDP wrap: {time.perf_counter() - t0:.1f}s")
        snapshot = gather_full_state(model)   # collective in trainer_pg only
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
        if MODE in ("delta_optimized", "delta_disk"):
            t0 = time.perf_counter()
            pinned = torch.empty(offsets[-1], dtype=torch.bfloat16, pin_memory=True)
            log(rank, f"pinned verify buffer ready: {time.perf_counter() - t0:.0f}s")

    names = bcast_object_world(sorted(snapshot.keys()) if rank == 0 else None)
    dist.barrier()

    rounds = []
    for r in range(ROUNDS):
        if is_trainer:
            perturb_local_shards(model, DENSITY, gen)
            torch.cuda.synchronize()
        dist.barrier()
        seg = {"round": r, "role": "trainer" if is_trainer else "miner"}
        reset_mem_peaks()
        t_round = time.perf_counter()

        # (1) gather — trainer_pg collective
        if is_trainer:
            t0 = time.perf_counter()
            full = gather_full_state(model)
            torch.cuda.synchronize()
            if rank == 0:
                seg["t_gather"] = time.perf_counter() - t0
        dist.barrier()

        # (2) encode + hash + pack on rank0 (grail verbatim)
        if rank == 0:
            t0 = time.perf_counter()
            sparse, sh, st = compute_sparse_delta(full, snapshot)
            seg["t_encode"] = time.perf_counter() - t0
            seg["nnz"] = st["nonzero_params"]
            t0 = time.perf_counter()
            pub_hash = compute_weights_hash(full)
            seg["t_hash_publish"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            snapshot = full
            changed = sorted(n for n in names if f"{n}.indices" in sparse)
            counts = [int(sparse[f"{n}.indices"].shape[1]) for n in changed]
            idx_flat = torch.cat(
                [sparse[f"{n}.indices"].reshape(-1) for n in changed])
            val_flat = torch.cat([sparse[f"{n}.values"] for n in changed])
            if MODE != "delta_disk":
                idx_flat, val_flat = idx_flat.cuda(), val_flat.cuda()
            torch.cuda.synchronize()
            seg["t_pack"] = time.perf_counter() - t0
            meta = (pub_hash, changed, counts, {n: sh[n] for n in changed})
            size = torch.tensor([idx_flat.numel(), val_flat.numel()],
                                dtype=torch.long, device="cuda")
            del sparse

        # (3a) transport = CPFS disk (delta_disk mode)
        if MODE == "delta_disk":
            import json as _json
            fpath = os.path.join(DISK_DIR, f"delta_r{r}.safetensors")
            mpath = os.path.join(DISK_DIR, f"delta_r{r}.meta.json")
            if rank == 0:
                t0 = time.perf_counter()
                os.makedirs(DISK_DIR, exist_ok=True)
                save_file({"idx": idx_flat, "val": val_flat}, fpath)
                fd = os.open(fpath, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                pub_hash_, changed_, counts_, sh_ = meta
                with open(mpath, "w") as f:
                    _json.dump({"hash": pub_hash_, "changed": changed_,
                                "counts": counts_,
                                "shapes": {k: list(v) for k, v in sh_.items()}}, f)
                seg["t_disk_write"] = time.perf_counter() - t0
                seg["delta_gb"] = (idx_flat.numel() * 4 + val_flat.numel() * 2) / 1024**3
            dist.barrier()
            if is_miner:
                t0 = time.perf_counter()
                d = load_file(fpath, device="cpu")
                idx_flat = d["idx"].cuda()
                val_flat = d["val"].cuda()
                with open(mpath) as f:
                    m = _json.load(f)
                meta = (m["hash"], m["changed"], m["counts"], m["shapes"])
                torch.cuda.synchronize()
                seg["t_disk_read"] = time.perf_counter() - t0
                del d

        # (3b) transport = NCCL broadcast (cross-node, dedicated group)
        if MODE != "delta_disk" and (rank == 0 or is_miner):
            if rank != 0:
                size = torch.zeros(2, dtype=torch.long, device="cuda")
                meta = None
            t0 = time.perf_counter()
            dist.broadcast(size, src=0, group=miner_bcast_pg)
            if rank != 0:
                idx_flat = torch.empty(int(size[0]), dtype=torch.int32, device="cuda")
                val_flat = torch.empty(int(size[1]), dtype=torch.bfloat16, device="cuda")
            dist.broadcast(idx_flat, src=0, group=miner_bcast_pg)
            dist.broadcast(val_flat, src=0, group=miner_bcast_pg)
            box = [meta if rank == 0 else None]
            dist.broadcast_object_list(box, src=0, group=miner_bcast_pg)
            meta = box[0]
            torch.cuda.synchronize()
            seg["t_broadcast"] = time.perf_counter() - t0
            seg["delta_gb"] = (int(size[0]) * 4 + int(size[1]) * 2) / 1024**3

        # (4) miners consume
        if is_miner:
            pub_hash, changed, counts, sh = meta
            if MODE in ("delta_optimized", "delta_disk"):
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
                    for n, fl, bak in backups:
                        replica[n].reshape(-1)[fl] = bak
                    raise RuntimeError(f"rank {rank} verify FAILED (rolled back)")
                backups.clear()
            else:  # delta_faithful consumer (grail verbatim, S7-identical)
                t0 = time.perf_counter()
                idx_cpu, val_cpu = idx_flat.cpu(), val_flat.cpu()
                sparse_d = {}
                io = vo = 0
                for n, c in zip(changed, counts):
                    sparse_d[f"{n}.indices"] = idx_cpu[io:io + 2 * c].view(2, c)
                    sparse_d[f"{n}.values"] = val_cpu[vo:vo + c]
                    io += 2 * c
                    vo += c
                seg["t_unpack"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                recon = apply_sparse_delta(replica, sparse_d, sh)
                seg["t_apply"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                ok = compute_weights_hash(recon) == pub_hash
                seg["t_verify"] = time.perf_counter() - t0
                seg["hash_ok"] = bool(ok)
                if not ok:
                    raise RuntimeError(f"rank {rank} verify FAILED")
                t0 = time.perf_counter()
                for n in names:
                    replica[n].copy_(recon[n].view(replica[n].shape), non_blocking=True)
                torch.cuda.synchronize()
                seg["t_load_back"] = time.perf_counter() - t0
                del recon, sparse_d, idx_cpu, val_cpu

        if is_trainer and rank != 0:
            del full
        torch.cuda.empty_cache()
        dist.barrier()
        seg["t_sync_total_e2e"] = time.perf_counter() - t_round
        if MODE == "delta_disk" and rank == 0:
            for p in (fpath, mpath):
                if os.path.exists(p):
                    os.remove(p)
        read_mem_peaks(seg)

        allsegs = [None] * world
        dist.all_gather_object(allsegs, seg)
        if rank == 0:
            merged = dict(allsegs[0])
            miners = [s for s in allsegs if s["role"] == "miner"]
            trainers = [s for s in allsegs if s["role"] == "trainer"]
            for key in ("t_unpack", "t_apply", "t_verify", "t_load_back",
                        "t_apply_gpu", "t_verify_d2h", "t_verify_hash",
                        "t_broadcast", "t_disk_read", "mem_gpu_alloc_peak_gb",
                        "mem_gpu_reserved_peak_gb", "mem_cpu_hwm_gb"):
                vals = [s[key] for s in miners if key in s]
                if vals:
                    merged[f"miner_{key}_max"] = max(vals)
                    merged[f"miner_{key}_mean"] = sum(vals) / len(vals)
            for key in ("mem_gpu_alloc_peak_gb", "mem_gpu_reserved_peak_gb",
                        "mem_cpu_hwm_gb"):
                vals = [s[key] for s in trainers if key in s]
                if vals:
                    merged[f"trainer_{key}_max"] = max(vals)
            merged["hash_ok_all"] = all(s.get("hash_ok", True) for s in miners)
            rounds.append(merged)
            log(rank, f"round {r}: " + " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(merged.items()) if k not in ("round", "role")))

    if rank == 0:
        steady = rounds[1:]
        summary = {}
        for key in sorted({k for x in steady for k in x}):
            vals = [float(x[key]) for x in steady
                    if isinstance(x.get(key), (int, float)) and key != "round"]
            if vals:
                summary[key] = stats(vals)
        print(f"\n### S8-v4 {MODE}+NCCL 2-node split (trainer 8 / miner 8), "
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
                           "trainer_ranks": T, "rounds": rounds,
                           "steady_stats": summary}, f, indent=2, default=str)
            print(f"JSON written to {JSON_OUT}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
