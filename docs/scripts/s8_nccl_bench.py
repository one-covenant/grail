"""S8: grail algorithm over NCCL transport (goal.md Phase 2 S8).

Two-rank torchrun bench: rank 0 = "trainer" (GPU 0), rank 1 = "miner" (GPU 1).
Both hold Qwen3-30B (57 GB bf16) on their GPU. Each round the trainer perturbs
its weights in-place on GPU (bit-flip at S8_DENSITY, simulating a training
step; not timed), then syncs to the miner via the selected mode.

Modes (S8_MODE):
  dense           transport floor: bucketed flat GPU->GPU broadcast, no checks
  delta           transport-optimized delta (flat diff, GPU apply, no per-round
                  hash) — kept for the A/B record, NOT grail-faithful
  dense_faithful  grail semantics on dense: trainer publishes xxh3 weights_hash
                  (grail compute_weights_hash, verbatim), broadcast, miner pulls
                  its full state to CPU and hash-verifies (grail verbatim)
  delta_faithful  ★ the user-requested end-to-end: all-gather(no-op, unsharded)
                  -> batched D2H snapshot (transport glue) -> grail
                  compute_sparse_delta (verbatim, per-tensor COO, threshold=0)
                  -> grail compute_weights_hash publish (verbatim)
                  -> pack + NCCL broadcast (transport glue)
                  -> miner: grail apply_sparse_delta on its CUDA state dict
                  (verbatim: per-tensor GPU->CPU pull inside, same as S7)
                  -> grail hash verify (verbatim) -> load back to GPU

Faithful rule (user 07-13): every grail-owned step runs grail's unmodified
functions with ALL verification costs. Only the wire itself is ours.
"""

import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

from grail.infrastructure.delta_checkpoint import (  # noqa: E402
    apply_sparse_delta,
    compute_sparse_delta,
    compute_weights_hash,
)
from grail.shared.safetensors_utils import load_model_state_dict  # noqa: E402

MODEL = os.getenv("S8_MODEL", "/storage/openpsi/models/Qwen__Qwen3-30B-A3B")
MODE = os.getenv("S8_MODE", "delta_faithful")
ROUNDS = int(os.getenv("S8_ROUNDS", "4"))  # round 0 = cold, skipped in stats
DENSITY = float(os.getenv("S8_DENSITY", "0.01"))
BUCKET_BYTES = int(os.getenv("S8_BUCKET_MB", "1024")) * 1024 * 1024
JSON_OUT = os.getenv("S8_JSON_OUT", "")
SEED = 20260713

INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def log(rank: int, msg: str) -> None:
    print(f"[s8 r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(vals):
    if not vals:
        return {"n": 0}
    d = {"n": len(vals), "mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
    if len(vals) >= 2:
        d["stdev"] = statistics.stdev(vals)
    return d


def perturb_gpu(state: dict, density: float, gen: torch.Generator) -> None:
    for t in state.values():
        ivd = INT_VIEW.get(t.dtype)
        if ivd is None or not t.numel():
            continue
        mask = torch.rand(t.shape, generator=gen, device=t.device) < density
        t.view(ivd)[mask] ^= 1
        del mask


# --------------------------------------------------------------------------- #
#                       transport-floor modes (v1, kept)                      #
# --------------------------------------------------------------------------- #

def _flush_dense(group, state, bucket, rank):
    off = 0
    if rank == 0:
        for name in group:
            t = state[name]
            bucket[off:off + t.numel()].copy_(t.reshape(-1))
            off += t.numel()
    else:
        off = sum(state[n].numel() for n in group)
    dist.broadcast(bucket[:off], src=0)
    if rank == 1:
        off = 0
        for name in group:
            t = state[name]
            t.reshape(-1).copy_(bucket[off:off + t.numel()])
            off += t.numel()


def broadcast_dense(state, names, rank, bucket) -> float:
    t0 = time.perf_counter()
    cap = bucket.numel()
    group, used = [], 0
    for name in names:
        n = state[name].numel()
        if used + n > cap and group:
            _flush_dense(group, state, bucket, rank)
            group, used = [], 0
        group.append(name)
        used += n
    if group:
        _flush_dense(group, state, bucket, rank)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def dense_sync(ctx, rank) -> dict:
    return {"t_broadcast": broadcast_dense(ctx["state"], ctx["names"], rank, ctx["bucket"])}


def delta_sync(ctx, rank) -> dict:
    """v1 transport-optimized delta (flat diff, GPU apply, no per-round hash)."""
    state, names, offsets = ctx["state"], ctx["names"], ctx["offsets"]
    seg = {}
    if rank == 0:
        t0 = time.perf_counter()
        for idx, name in enumerate(names):
            ctx["pinned"][offsets[idx]:offsets[idx + 1]].copy_(
                state[name].reshape(-1), non_blocking=True)
        torch.cuda.synchronize()
        seg["t_d2h"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        changed = (ctx["pinned"] != ctx["snapshot"]).nonzero(as_tuple=True)[0]
        values = ctx["pinned"][changed]
        seg["t_diff"] = time.perf_counter() - t0
        seg["nnz"] = int(changed.numel())
        t0 = time.perf_counter()
        ctx["snapshot"].copy_(ctx["pinned"])
        idx_gpu, val_gpu = changed.cuda(), values.cuda()
        torch.cuda.synchronize()
        seg["t_snapshot_pack"] = time.perf_counter() - t0
        meta = torch.tensor([idx_gpu.numel()], dtype=torch.long, device="cuda")
    else:
        meta = torch.zeros(1, dtype=torch.long, device="cuda")

    t0 = time.perf_counter()
    dist.broadcast(meta, src=0)
    nnz = int(meta.item())
    if rank == 1:
        idx_gpu = torch.empty(nnz, dtype=torch.long, device="cuda")
        val_gpu = torch.empty(nnz, dtype=torch.bfloat16, device="cuda")
    dist.broadcast(idx_gpu, src=0)
    dist.broadcast(val_gpu, src=0)
    torch.cuda.synchronize()
    seg["t_broadcast"] = time.perf_counter() - t0

    if rank == 1:
        t0 = time.perf_counter()
        bounds = torch.searchsorted(
            idx_gpu, torch.tensor(offsets[1:], dtype=torch.long, device="cuda"))
        start = 0
        for idx, name in enumerate(names):
            end = int(bounds[idx])
            if end > start:
                local = idx_gpu[start:end] - offsets[idx]
                state[name].reshape(-1)[local] = val_gpu[start:end]
            start = end
        torch.cuda.synchronize()
        seg["t_apply"] = time.perf_counter() - t0
    return seg


# --------------------------------------------------------------------------- #
#                    grail-faithful modes (all checks included)               #
# --------------------------------------------------------------------------- #

def _cpu_views(flat, names, offsets, shapes):
    return {n: flat[offsets[i]:offsets[i + 1]].view(shapes[n]) for i, n in enumerate(names)}


def dense_faithful_sync(ctx, rank) -> dict:
    """Dense broadcast + grail integrity semantics (publish hash, verify hash)."""
    state, names, offsets, shapes = ctx["state"], ctx["names"], ctx["offsets"], ctx["shapes"]
    seg = {}
    if rank == 0:
        t0 = time.perf_counter()
        for idx, name in enumerate(names):
            ctx["pinned"][offsets[idx]:offsets[idx + 1]].copy_(
                state[name].reshape(-1), non_blocking=True)
        torch.cuda.synchronize()
        seg["t_d2h"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pub_hash = compute_weights_hash(_cpu_views(ctx["pinned"], names, offsets, shapes))
        seg["t_hash_publish"] = time.perf_counter() - t0
    else:
        pub_hash = None

    seg["t_broadcast"] = broadcast_dense(state, names, rank, ctx["bucket"])
    obj = [pub_hash]
    dist.broadcast_object_list(obj, src=0)

    if rank == 1:
        t0 = time.perf_counter()
        for idx, name in enumerate(names):
            ctx["pinned"][offsets[idx]:offsets[idx + 1]].copy_(
                state[name].reshape(-1), non_blocking=True)
        torch.cuda.synchronize()
        got_hash = compute_weights_hash(_cpu_views(ctx["pinned"], names, offsets, shapes))
        seg["t_verify"] = time.perf_counter() - t0
        seg["hash_ok"] = bool(got_hash == obj[0])
        if not seg["hash_ok"]:
            raise RuntimeError("dense_faithful hash verify FAILED")
    return seg


def delta_faithful_sync(ctx, rank) -> dict:
    """User's 4-step end-to-end with EVERY grail check, grail functions verbatim:
    all-gather(no-op) -> D2H -> compute_sparse_delta -> weights_hash publish ->
    pack/broadcast -> apply_sparse_delta on CUDA dict (per-tensor CPU pull,
    S7-identical) -> hash verify -> load back to GPU."""
    state, names, offsets, shapes = ctx["state"], ctx["names"], ctx["offsets"], ctx["shapes"]
    seg = {"t_gather": 0.0}  # unsharded trainer: all-gather is a no-op, recorded honestly

    if rank == 0:
        t0 = time.perf_counter()
        for idx, name in enumerate(names):
            ctx["pinned"][offsets[idx]:offsets[idx + 1]].copy_(
                state[name].reshape(-1), non_blocking=True)
        torch.cuda.synchronize()
        seg["t_d2h"] = time.perf_counter() - t0

        cur_cpu = _cpu_views(ctx["pinned"], names, offsets, shapes)
        prev_cpu = _cpu_views(ctx["snapshot"], names, offsets, shapes)

        t0 = time.perf_counter()
        sparse, sh, st = compute_sparse_delta(cur_cpu, prev_cpu)   # grail verbatim
        seg["t_encode"] = time.perf_counter() - t0
        seg["nnz"] = st["nonzero_params"]

        t0 = time.perf_counter()
        pub_hash = compute_weights_hash(cur_cpu)                   # grail verbatim
        seg["t_hash_publish"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ctx["snapshot"].copy_(ctx["pinned"])
        changed = sorted(n for n in names if f"{n}.indices" in sparse)
        counts = torch.tensor([sparse[f"{n}.indices"].shape[1] for n in changed],
                              dtype=torch.long)
        idx_flat = torch.cat([sparse[f"{n}.indices"].reshape(-1) for n in changed]).cuda()
        val_flat = torch.cat([sparse[f"{n}.values"] for n in changed]).cuda()
        torch.cuda.synchronize()
        seg["t_snapshot_pack"] = time.perf_counter() - t0
        meta_obj = [pub_hash, changed, counts.tolist(),
                    {n: sh[n] for n in changed}]
        size = torch.tensor([idx_flat.numel(), val_flat.numel()],
                            dtype=torch.long, device="cuda")
    else:
        size = torch.zeros(2, dtype=torch.long, device="cuda")
        meta_obj = [None, None, None, None]

    t0 = time.perf_counter()
    dist.broadcast(size, src=0)
    if rank == 1:
        idx_flat = torch.empty(int(size[0]), dtype=torch.int32, device="cuda")
        val_flat = torch.empty(int(size[1]), dtype=torch.bfloat16, device="cuda")
    dist.broadcast(idx_flat, src=0)
    dist.broadcast(val_flat, src=0)
    dist.broadcast_object_list(meta_obj, src=0)
    torch.cuda.synchronize()
    seg["t_broadcast"] = time.perf_counter() - t0
    seg["delta_gb"] = (int(size[0]) * 4 + int(size[1]) * 2) / 1024**3

    if rank == 1:
        pub_hash, changed, counts, sh = meta_obj
        # unpack wire buffers back into grail's sparse dict format (CPU)
        t0 = time.perf_counter()
        idx_cpu = idx_flat.cpu()
        val_cpu = val_flat.cpu()
        sparse = {}
        io = vo = 0
        for n, c in zip(changed, counts):
            sparse[f"{n}.indices"] = idx_cpu[io:io + 2 * c].view(2, c)
            sparse[f"{n}.values"] = val_cpu[vo:vo + c]
            io += 2 * c
            vo += c
        seg["t_unpack"] = time.perf_counter() - t0

        # grail verbatim apply on the CUDA state dict: pulls every tensor to
        # CPU internally (18867 transfers) — identical code path to S7
        t0 = time.perf_counter()
        recon = apply_sparse_delta(state, sparse, sh)
        seg["t_apply"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        got_hash = compute_weights_hash(recon)                     # grail verbatim
        seg["t_verify"] = time.perf_counter() - t0
        seg["hash_ok"] = bool(got_hash == pub_hash)
        if not seg["hash_ok"]:
            raise RuntimeError("delta_faithful hash verify FAILED")

        t0 = time.perf_counter()
        for name in names:
            state[name].copy_(recon[name].view(state[name].shape), non_blocking=True)
        torch.cuda.synchronize()
        seg["t_load_back"] = time.perf_counter() - t0
        del recon, sparse
    return seg


SYNC_FN = {"dense": dense_sync, "delta": delta_sync,
           "dense_faithful": dense_faithful_sync, "delta_faithful": delta_faithful_sync}


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    gen = torch.Generator(device="cuda").manual_seed(SEED)

    log(rank, f"mode={MODE} rounds={ROUNDS} density={DENSITY}")
    t0 = time.perf_counter()
    cpu_state = load_model_state_dict(MODEL)
    names = sorted(cpu_state.keys())
    state = {k: cpu_state[k].cuda() for k in names}
    shapes = {k: tuple(state[k].shape) for k in names}
    del cpu_state
    total = sum(state[n].numel() for n in names)
    offsets = [0]
    for n in names:
        offsets.append(offsets[-1] + state[n].numel())
    log(rank, f"loaded {len(names)} tensors, {total / 1e9:.2f}B params in "
              f"{time.perf_counter() - t0:.0f}s")

    ctx = {"state": state, "names": names, "offsets": offsets, "shapes": shapes}
    if MODE in ("dense", "dense_faithful"):
        ctx["bucket"] = torch.empty(BUCKET_BYTES // 2, dtype=torch.bfloat16, device="cuda")
    if MODE in ("dense_faithful", "delta", "delta_faithful"):
        need_pin = (rank == 0) or MODE == "dense_faithful"
        if need_pin:
            ctx["pinned"] = torch.empty(total, dtype=torch.bfloat16, pin_memory=True)
        if rank == 0 and MODE in ("delta", "delta_faithful"):
            ctx["snapshot"] = torch.empty(total, dtype=torch.bfloat16)
            for i, n in enumerate(names):
                ctx["snapshot"][offsets[i]:offsets[i + 1]].copy_(state[n].reshape(-1).cpu())
            log(rank, "snapshot buffer ready")

    dist.barrier()
    rounds = []
    for r in range(ROUNDS):
        if rank == 0:
            perturb_gpu(state, DENSITY, gen)
            torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        seg = SYNC_FN[MODE](ctx, rank)
        dist.barrier()
        seg["t_sync_total_e2e"] = time.perf_counter() - t0
        seg["round"] = r

        # merge both ranks' segments (miner-side keys prefixed)
        both = [None, None]
        dist.all_gather_object(both, seg)
        merged = dict(both[0])
        merged.update({f"miner_{k}" if k in both[0] and both[0][k] != v else k: v
                       for k, v in both[1].items() if k != "round"})
        rounds.append(merged)
        if rank == 0:
            log(rank, f"round {r}: " + " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(merged.items()) if k != "round"))

        if r == 0 and MODE in ("dense", "delta"):  # faithful modes verify every round
            my_hash = compute_weights_hash({n: state[n] for n in names})
            hashes = [None, None]
            dist.all_gather_object(hashes, my_hash)
            ok = hashes[0] == hashes[1]
            log(rank, f"bit-exact after round 0: {ok}")
            if not ok:
                raise RuntimeError("bit-exact FAILED")

    if rank == 0:
        steady = rounds[1:]
        summary = {}
        for key in sorted({k for x in steady for k in x}):
            vals = [float(x[key]) for x in steady
                    if isinstance(x.get(key), (int, float)) and key != "round"]
            if vals:
                summary[key] = stats(vals)
        print(f"\n### S8 {MODE}+NCCL steady (skip cold), Qwen3-30B, density={DENSITY}")
        for key, s in summary.items():
            if s.get("n"):
                line = f"  {key}: mean={s['mean']:.3f}"
                if "stdev" in s:
                    line += f" stdev={s['stdev']:.3f}"
                print(line + f" (n={s['n']})")
        if JSON_OUT:
            with open(JSON_OUT, "w") as f:
                json.dump({"mode": MODE, "density": DENSITY, "rounds": rounds,
                           "steady_stats": summary}, f, indent=2, default=str)
            print(f"JSON written to {JSON_OUT}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
