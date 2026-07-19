"""S5a: SGLang reload microbench on Qwen3-30B-A3B (goal.md Phase 2 S5).

Measures /update_weights_from_disk latency for a 57 GB bf16 checkpoint —
the same call grail's miner makes on every weight sync (weight_sync.py),
against the same CPFS storage. No miner, no moto: the reload path never
touches S3 (miner reloads from its local CPFS cache dir).

Per iteration we posix_fadvise(DONTNEED) every safetensors shard first, so
each reload re-reads from CPFS like the miner does with a fresh checkpoint
directory (new dir => cold page cache). A few no-fadvise iterations at the
end separate "CPFS read" from "RAM -> GPU + engine swap".

Runs INSIDE grail-miner.sif (libnuma bind + CUDA_HOME fix required, see
s4_sync_bench.sbatch). Uses grail's own SGLangServerManager so server flags
match the miner exactly.
"""

import asyncio
import glob
import json
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

MODEL = os.getenv("S5_MODEL", "/storage/openpsi/models/Qwen__Qwen3-30B-A3B")
COLD_ITERS = int(os.getenv("S5_COLD_ITERS", "11"))
WARM_ITERS = int(os.getenv("S5_WARM_ITERS", "3"))
JSON_OUT = os.getenv("S5_JSON_OUT", "")


def stats(values: list[float]) -> dict[str, Any]:
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


def drop_page_cache(model_dir: str) -> tuple[int, float]:
    """fadvise(DONTNEED) all safetensors shards; returns (n_files, gb)."""
    total = 0
    files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    for path in files:
        total += os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return len(files), total / 1024**3


async def main() -> None:
    import httpx

    from grail.trainer.config import EvalConfig
    from grail.trainer.inference_server import ServerConfig, SGLangServerManager

    eval_config = EvalConfig(
        sglang_mem_fraction_static=0.9,
        sglang_context_length=12288,
        server_timeout=1800.0,  # 57 GB cold load takes a while
        stream_server_logs=True,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    manager = SGLangServerManager(
        config=ServerConfig(model_path=MODEL, timeout_s=1800.0, env=env),
        eval_config=eval_config,
    )

    print(f"[s5] starting SGLang with {MODEL}", flush=True)
    t0 = time.perf_counter()
    await manager.__aenter__()
    await manager.start_server()
    t_start = time.perf_counter() - t0
    print(f"[s5] server up in {t_start:.1f}s at {manager.base_url}", flush=True)

    results: dict[str, Any] = {"model": MODEL, "server_start_s": t_start, "cold": [], "warm": []}

    async with httpx.AsyncClient() as client:
        for mode, iters in (("cold", COLD_ITERS), ("warm", WARM_ITERS)):
            for i in range(iters):
                if mode == "cold":
                    n_files, gb = drop_page_cache(MODEL)
                    print(f"[s5] fadvise dropped {n_files} shards ({gb:.1f} GB)", flush=True)
                t0 = time.perf_counter()
                resp = await client.post(
                    f"{manager.base_url}/update_weights_from_disk",
                    json={"model_path": MODEL},
                    timeout=1800.0,
                )
                resp.raise_for_status()
                elapsed = time.perf_counter() - t0
                results[mode].append(elapsed)
                print(
                    f"[s5] reload {mode} #{i}: {elapsed:.2f}s "
                    f"(resp: {resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.status_code})",
                    flush=True,
                )

    await manager.__aexit__(None, None, None)

    results["cold_stats"] = stats(results["cold"])
    results["warm_stats"] = stats(results["warm"])
    print("\n### SGLang /update_weights_from_disk, Qwen3-30B-A3B (57 GB bf16, CPFS)")
    for mode in ("cold", "warm"):
        s = results[f"{mode}_stats"]
        label = "cold (fadvise, fresh CPFS read)" if mode == "cold" else "warm (page cache)"
        if s.get("n"):
            line = f"  {label}: n={s['n']} mean={s['mean']:.2f}s median={s['median']:.2f}s"
            if "stdev" in s:
                line += f" stdev={s['stdev']:.2f}"
            line += f" min={s['min']:.2f} max={s['max']:.2f}"
            print(line)
    if JSON_OUT:
        with open(JSON_OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON written to {JSON_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
