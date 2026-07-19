"""Benchmark grail sparse delta checkpoint on Qwen3-30B-A3B (goal.md §5.2).

Semantics: pure algorithm + CPFS I/O benchmark of grail's delta path. Measures,
per density point, per iteration:
  T_encode = compute_sparse_delta(current, base) wall clock (CPU)
  T_write  = safetensors save_file + fsync + shapes.json to CPFS
  T_read   = safetensors load_file + shapes.json from CPFS
             (page cache dropped via posix_fadvise after write, best effort)
  T_apply  = apply_sparse_delta(base_clone, sparse, shapes)
  T_delta_sync = T_encode + T_write + T_read + T_apply
Does NOT include:
  - rollout engine reload (SGLang /update_weights_from_disk etc.)
  - R2 upload/download (replaced by CPFS)
→ Comparable-ish to slime delta+disk 15.15 s and ByteCheckpoint dense 71.53 s.

Density control: instead of goal.md's sigma-noise plan A (bf16 quantization
swallows small sigma, making density uncontrollable), we flip the lowest
mantissa bit at a random mask of fraction p — exact density, guaranteed
bit-change, realistic tiny-update magnitude.

CAUTION (source-verified): apply_sparse_delta mutates base_state in-place when
target_dtype == base dtype on CPU (reshape returns views). Production consumer
relies on this (apply_delta_in_place fast path). The bench therefore hands
apply a fresh clone of base each iter; T_clone is reported separately and
EXCLUDED from T_delta_sync because the real consumer path does not clone.

Usage (in sbatch, single process, CPU-bound):
    python3 bench_qwen3_30b_grail_delta.py \
        --work-dir /storage/.../grail_bench/bench_<JOBID> \
        --plan 0.01:11,0.05:11,0.30:4,0.60:3 \
        --json-output /storage/.../logs/grail_bench_<JOBID>.json
"""

import argparse
import gc
import glob
import json
import logging
import os
import shutil
import statistics
import sys
import time

import torch

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")
from grail.infrastructure.delta_checkpoint import (  # noqa: E402
    apply_sparse_delta,
    compute_sparse_delta,
    compute_weights_hash,
)
from safetensors.torch import load_file, save_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("grail_bench")


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    d = {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) >= 2:
        d["stdev"] = statistics.stdev(values)
    return d


def fmt_stats(d: dict) -> str:
    if d.get("n", 0) == 0:
        return "(no data)"
    parts = [f"n={d['n']}", f"mean={d['mean']:.3f}", f"median={d['median']:.3f}"]
    if "stdev" in d:
        parts.append(f"stdev={d['stdev']:.3f}")
    parts.extend([f"min={d['min']:.3f}", f"max={d['max']:.3f}"])
    return " ".join(parts)


def cpfs_free_gb() -> float:
    return shutil.disk_usage("/storage").free / 1024**3


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024**2
    return -1.0


def drop_page_cache(path: str) -> bool:
    """Best-effort page cache drop so T_read touches CPFS, not RAM."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True
    except OSError as e:
        logger.warning(f"posix_fadvise failed on {path}: {e}")
        return False


def load_base_state(model_path: str) -> dict[str, torch.Tensor]:
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {model_path}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(shard, device="cpu"))
    return state


def make_current(
    base: dict[str, torch.Tensor], density: float, seed: int
) -> dict[str, torch.Tensor]:
    """Clone base and flip the lowest mantissa bit at a fraction `density` of positions."""
    g = torch.Generator().manual_seed(seed)
    int_view = {
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
        torch.float32: torch.int32,
    }
    current = {}
    for name, t in base.items():
        c = t.clone()
        iv_dtype = int_view.get(c.dtype)
        if iv_dtype is not None and c.numel() > 0 and density > 0:
            mask = torch.rand(c.shape, generator=g) < density
            iv = c.view(iv_dtype)
            iv[mask] ^= 1
            del mask
        current[name] = c
    return current


def sparse_nbytes(sparse: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in sparse.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/storage/openpsi/models/Qwen__Qwen3-30B-A3B")
    parser.add_argument("--work-dir", required=True, help="CPFS dir for delta files")
    parser.add_argument(
        "--plan",
        default="0.01:11,0.05:11,0.30:4,0.60:3",
        help="Comma list of density:iters; iter #0 per density is cold, skipped in stats",
    )
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    plan = []
    for item in args.plan.split(","):
        d, n = item.split(":")
        plan.append((float(d), int(n)))

    os.makedirs(args.work_dir, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 64)))

    logger.info(f"===== grail delta bench: torch {torch.__version__} =====")
    logger.info(f"plan: {plan}, threads: {torch.get_num_threads()}")
    logger.info(f"CPFS free: {cpfs_free_gb():.1f} GB")

    results: dict = {
        "model": args.model_path.rsplit("/", 1)[-1],
        "plan": args.plan,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "densities": [],
    }

    def flush_json() -> None:
        if not args.json_output:
            return
        tmp = args.json_output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, args.json_output)

    # ---- HF load (once, reused as pristine base for all densities) ----
    t0 = time.perf_counter()
    base = load_base_state(args.model_path)
    t_hf_load = time.perf_counter() - t0
    total_params = sum(t.numel() for t in base.values())
    total_gb = sum(t.numel() * t.element_size() for t in base.values()) / 1024**3
    logger.info(
        f"HF load: {t_hf_load:.2f} s | {len(base)} tensors, "
        f"{total_params / 1e9:.2f}B params, {total_gb:.1f} GB | RSS {rss_gb():.0f} GB"
    )
    results["hf_load_s"] = t_hf_load
    results["total_params"] = total_params
    results["dense_size_gb"] = total_gb
    flush_json()

    delta_path = os.path.join(args.work_dir, "delta.safetensors")
    shapes_path = os.path.join(args.work_dir, "delta_shapes.json")

    for density, n_iters in plan:
        logger.info("=" * 70)
        logger.info(f"===== density {density:.2%}, {n_iters} iters =====")

        t0 = time.perf_counter()
        current = make_current(base, density, args.seed)
        t_perturb = time.perf_counter() - t0
        t0 = time.perf_counter()
        hash_current = compute_weights_hash(current)
        t_hash = time.perf_counter() - t0
        logger.info(
            f"perturb: {t_perturb:.1f} s, hash(current): {t_hash:.1f} s | RSS {rss_gb():.0f} GB"
        )

        dres: dict = {
            "density_target": density,
            "n_iters": n_iters,
            "perturb_s": t_perturb,
            "hash_current_s": t_hash,
            "iters": [],
        }
        results["densities"].append(dres)

        for i in range(n_iters):
            it: dict = {"iter": i, "cold": i == 0}

            # T_encode
            t0 = time.perf_counter()
            sparse, shapes, st = compute_sparse_delta(current, base)
            it["t_encode"] = time.perf_counter() - t0
            it["nnz"] = st["nonzero_params"]
            it["density_actual"] = st["nonzero_params"] / st["total_params"]
            it["delta_mem_gb"] = sparse_nbytes(sparse) / 1024**3

            # T_write (save + fsync so the bytes actually hit CPFS)
            t0 = time.perf_counter()
            save_file(sparse, delta_path)
            fd = os.open(delta_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            with open(shapes_path, "w") as f:
                json.dump(shapes, f)
            it["t_write"] = time.perf_counter() - t0
            it["delta_file_gb"] = os.path.getsize(delta_path) / 1024**3

            del sparse, shapes
            gc.collect()
            it["fadvise_ok"] = drop_page_cache(delta_path)

            # T_read
            t0 = time.perf_counter()
            sparse_loaded = load_file(delta_path, device="cpu")
            with open(shapes_path) as f:
                shapes_loaded = json.load(f)
            it["t_read"] = time.perf_counter() - t0

            # base clone for apply (apply mutates its base arg in-place; see header)
            t0 = time.perf_counter()
            work_base = {k: v.clone() for k, v in base.items()}
            it["t_clone_excluded"] = time.perf_counter() - t0

            # T_apply
            t0 = time.perf_counter()
            recon = apply_sparse_delta(work_base, sparse_loaded, shapes_loaded)
            it["t_apply"] = time.perf_counter() - t0

            it["t_delta_sync"] = it["t_encode"] + it["t_write"] + it["t_read"] + it["t_apply"]

            # bit-exact verify (cold iter only; xxh3 over 60 GB is ~seconds, not free)
            if i == 0:
                t0 = time.perf_counter()
                hash_recon = compute_weights_hash(recon)
                it["t_verify_excluded"] = time.perf_counter() - t0
                it["bit_exact"] = hash_recon == hash_current
                if not it["bit_exact"]:
                    logger.error(
                        f"BIT-EXACT FAIL at density {density}: "
                        f"current={hash_current} recon={hash_recon}"
                    )

            del sparse_loaded, shapes_loaded, work_base, recon
            gc.collect()
            for p in (delta_path, shapes_path):
                if os.path.exists(p):
                    os.remove(p)

            marker = " (cold)" if i == 0 else ""
            logger.info(
                f"[d={density:.2f} #{i}]{marker} "
                f"encode={it['t_encode']:.2f}s write={it['t_write']:.2f}s "
                f"read={it['t_read']:.2f}s apply={it['t_apply']:.2f}s "
                f"sync={it['t_delta_sync']:.2f}s | "
                f"delta={it['delta_file_gb']:.2f}GB nnz={it['nnz'] / 1e9:.2f}B "
                f"density={it['density_actual']:.4f} | "
                f"clone(excl)={it['t_clone_excluded']:.2f}s "
                f"RSS={rss_gb():.0f}GB cpfs_free={cpfs_free_gb():.0f}GB"
            )
            dres["iters"].append(it)
            flush_json()

        steady = [x for x in dres["iters"] if not x["cold"]] or dres["iters"]
        dres["steady_stats"] = {
            key: stats([x[f"t_{key}"] for x in steady])
            for key in ("encode", "write", "read", "apply", "delta_sync")
        }
        dres["bit_exact"] = dres["iters"][0].get("bit_exact")
        del current
        gc.collect()
        flush_json()

    # ---- Summary (goal.md §8 format) ----
    logger.info("=" * 70)
    logger.info("SUMMARY: grail delta sync, Qwen3-30B-A3B, CPFS, steady-state (cold skipped)")
    logger.info(
        f"{'density':>8} {'delta_GB':>9} {'%of60GB':>8} {'encode':>8} {'write':>8} "
        f"{'read':>8} {'apply':>8} {'SYNC':>8} {'bitexact':>8}"
    )
    for dres in results["densities"]:
        ss = dres["steady_stats"]
        d0 = dres["iters"][0]
        logger.info(
            f"{d0['density_actual']:>8.4f} {d0['delta_file_gb']:>9.2f} "
            f"{100 * d0['delta_file_gb'] / results['dense_size_gb']:>7.1f}% "
            f"{ss['encode']['mean']:>8.2f} {ss['write']['mean']:>8.2f} "
            f"{ss['read']['mean']:>8.2f} {ss['apply']['mean']:>8.2f} "
            f"{ss['delta_sync']['mean']:>8.2f} {str(dres['bit_exact']):>8}"
        )
    for dres in results["densities"]:
        ss = dres["steady_stats"]
        logger.info(f"--- density {dres['density_target']:.2%} detail ---")
        for key in ("encode", "write", "read", "apply", "delta_sync"):
            logger.info(f"  T_{key:<11} {fmt_stats(ss[key])}")
        sync = ss["delta_sync"]["mean"]
        logger.info(
            f"  vs slime delta+disk 15.15 s: {15.15 / sync:.2f}x | "
            f"vs slime delta+nccl 19.68 s: {19.68 / sync:.2f}x | "
            f"vs bcp dense CPFS 71.53 s: {71.53 / sync:.2f}x"
        )
    flush_json()
    logger.info(f"JSON written to {args.json_output}")


if __name__ == "__main__":
    main()
