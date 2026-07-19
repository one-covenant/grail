"""S6: per-segment disk-backend sync bench for Qwen3-30B (goal.md Phase 2 S6).

Measures every segment of grail's weight-transfer chain when the object store
is replaced by local CPFS disk (segments (5) SGLang reload measured in S5):

  (1) serialize + manifest: state dict -> safetensors staging write (+fsync),
      then publisher-prep = per-file sha256 manifest + xxh3 weights_hash
  (2) store write: staging dir -> store dir copy (+fsync)   ["upload" to disk]
  (3) fetch + verify: store -> cache copy (+fsync) + per-file sha256
      (grail consumer semantics), plus a verify-only variant (no copy)
  (4) proof reload: HF from_pretrained bf16 -> cuda:0, cold (fadvise) & warm

posix_fadvise(DONTNEED) before each cold read so CPFS is actually hit.
Repeats: N per segment via S6_ITERS (default 3).
"""

import gc
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

from grail.infrastructure.delta_checkpoint import compute_weights_hash  # noqa: E402
from grail.shared.safetensors_utils import load_model_state_dict  # noqa: E402

MODEL = Path(os.getenv("S6_MODEL", "/storage/openpsi/models/Qwen__Qwen3-30B-A3B"))
WORK = Path(os.getenv("S6_WORK", "/storage/openpsi/users/pengzai.pyq/grail_bench/s6"))
ITERS = int(os.getenv("S6_ITERS", "3"))
JSON_OUT = os.getenv("S6_JSON_OUT", "")

STAGING, STORE, CACHE = WORK / "staging", WORK / "store", WORK / "cache"


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    d = {"n": len(values), "mean": statistics.mean(values), "min": min(values), "max": max(values)}
    if len(values) >= 2:
        d["stdev"] = statistics.stdev(values)
    return d


def fadvise_dir(path: Path) -> None:
    for f in path.rglob("*"):
        if f.is_file():
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def copy_dir_fsync(src: Path, dst: Path) -> float:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    t0 = time.perf_counter()
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy(f, dst / f.name)
            fd = os.open(dst / f.name, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return time.perf_counter() - t0


def sha256_dir(path: Path) -> tuple[float, dict[str, str]]:
    t0 = time.perf_counter()
    manifest = {}
    for f in sorted(path.iterdir()):
        if f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                while chunk := fh.read(64 * 1024 * 1024):
                    h.update(chunk)
            manifest[f.name] = h.hexdigest()
    return time.perf_counter() - t0, manifest


def main() -> None:
    import faulthandler

    faulthandler.enable()
    WORK.mkdir(parents=True, exist_ok=True)
    results: dict = {"model": str(MODEL), "iters": ITERS}

    def flush_json() -> None:
        if JSON_OUT:
            tmp = JSON_OUT + ".tmp"
            with open(tmp, "w") as f:
                json.dump(results, f, indent=2)
            os.replace(tmp, JSON_OUT)

    def say(msg: str) -> None:
        print(f"[s6 {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    size_gb = sum(f.stat().st_size for f in MODEL.iterdir() if f.is_file()) / 1024**3
    results["payload_gb"] = size_gb
    say(f"payload {size_gb:.1f} GB, iters={ITERS}")

    # ---- (1) serialize + manifest ------------------------------------------
    # load state dict fully into RAM first (materialize; excluded from timing)
    state = load_model_state_dict(MODEL)
    for k in state:
        state[k] = state[k].clone()  # break mmap so staging write reads RAM, not CPFS
    say("state dict materialized in RAM")

    ser_write, prep = [], []
    from safetensors.torch import save_file

    for i in range(ITERS):
        if STAGING.exists():
            shutil.rmtree(STAGING)
        STAGING.mkdir(parents=True)
        t0 = time.perf_counter()
        save_file(state, STAGING / "model.safetensors")
        fd = os.open(STAGING / "model.safetensors", os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        ser_write.append(time.perf_counter() - t0)
        for small in ("config.json", "generation_config.json", "tokenizer.json",
                      "tokenizer_config.json", "vocab.json", "merges.txt"):
            if (MODEL / small).exists():
                shutil.copy(MODEL / small, STAGING / small)

        fadvise_dir(STAGING)
        t0 = time.perf_counter()
        _, _manifest = sha256_dir(STAGING)
        staged = load_model_state_dict(STAGING)
        _ = compute_weights_hash(staged)
        del staged
        gc.collect()
        prep.append(time.perf_counter() - t0)
        say(f"(1) iter {i}: staging_write={ser_write[-1]:.1f}s prep={prep[-1]:.1f}s")
        flush_json()
    del state
    gc.collect()
    results["seg1_raw"] = {"staging_write": ser_write, "prep": prep}
    results["seg1_staging_write"] = stats(ser_write)
    results["seg1_publish_prep"] = stats(prep)

    # ---- (2) store write ----------------------------------------------------
    store_w = []
    for i in range(ITERS):
        fadvise_dir(STAGING)
        store_w.append(copy_dir_fsync(STAGING, STORE))
        say(f"(2) iter {i}: store_write={store_w[-1]:.1f}s")
        flush_json()
    results["seg2_store_write"] = stats(store_w)

    # ---- (3) fetch + verify --------------------------------------------------
    fetch_copy, verify_after_copy, verify_only = [], [], []
    for i in range(ITERS):
        fadvise_dir(STORE)
        fetch_copy.append(copy_dir_fsync(STORE, CACHE))
        # cache pages hot after copy — grail verifies what it just downloaded
        t, _ = sha256_dir(CACHE)
        verify_after_copy.append(t)
        say(f"(3) iter {i}: fetch_copy={fetch_copy[-1]:.1f}s verify_hot={verify_after_copy[-1]:.1f}s")
        flush_json()
    for i in range(ITERS):
        fadvise_dir(STORE)
        t, _ = sha256_dir(STORE)
        verify_only.append(t)
        say(f"(3b) iter {i}: verify_only_cold={verify_only[-1]:.1f}s")
        flush_json()
    results["seg3_fetch_copy"] = stats(fetch_copy)
    results["seg3_verify_hot"] = stats(verify_after_copy)
    results["seg3_verify_only_cold"] = stats(verify_only)

    # ---- (4) proof reload ----------------------------------------------------
    from transformers import AutoModelForCausalLM

    proof_cold, proof_warm = [], []
    for i in range(ITERS):
        fadvise_dir(CACHE)
        t0 = time.perf_counter()
        m = AutoModelForCausalLM.from_pretrained(
            CACHE, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map={"": 0}, low_cpu_mem_usage=True,
        )
        proof_cold.append(time.perf_counter() - t0)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        m = AutoModelForCausalLM.from_pretrained(
            CACHE, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map={"": 0}, low_cpu_mem_usage=True,
        )
        proof_warm.append(time.perf_counter() - t0)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        say(f"(4) iter {i}: proof_cold={proof_cold[-1]:.1f}s proof_warm={proof_warm[-1]:.1f}s")
        flush_json()
    results["seg4_proof_reload_cold"] = stats(proof_cold)
    results["seg4_proof_reload_warm"] = stats(proof_warm)

    # ---- summary --------------------------------------------------------------
    print("\n### S6 per-segment (mean, disk backend, Qwen3-30B)")
    for key, val in results.items():
        if isinstance(val, dict) and val.get("n"):
            line = f"  {key}: mean={val['mean']:.1f}s"
            if "stdev" in val:
                line += f" stdev={val['stdev']:.1f}"
            print(line + f" (n={val['n']})")
    sglang_warm = 13.08  # S5 job 939679
    faithful = (
        results["seg1_staging_write"]["mean"] + results["seg1_publish_prep"]["mean"]
        + results["seg2_store_write"]["mean"]
        + results["seg3_fetch_copy"]["mean"] + results["seg3_verify_hot"]["mean"]
        + results["seg4_proof_reload_warm"]["mean"] + sglang_warm
    )
    optimal = (
        results["seg1_staging_write"]["mean"] + results["seg1_publish_prep"]["mean"]
        + results["seg3_verify_only_cold"]["mean"]
        + results["seg4_proof_reload_warm"]["mean"] + sglang_warm
    )
    results["total_grail_faithful"] = faithful
    results["total_disk_optimal"] = optimal
    print(f"\n  TOTAL grail-faithful (1+2+3copy+verify+4warm+5warm13.08): {faithful:.1f}s")
    print(f"  TOTAL disk-optimal   (1+verify-only+4warm+5warm, no copies): {optimal:.1f}s")

    if JSON_OUT:
        with open(JSON_OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON written to {JSON_OUT}")

    for d in (STAGING, STORE, CACHE):
        if d.exists():
            shutil.rmtree(d)


if __name__ == "__main__":
    main()
