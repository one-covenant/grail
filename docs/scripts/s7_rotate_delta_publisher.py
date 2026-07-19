"""S7 delta rotator: bootstrap FULL then publish chained DELTA checkpoints.

Drives the miner's fast path (goal.md Phase 2 S7):
  bootstrap: materialize Qwen3-30B state -> staging -> upload_from_staging FULL
  each rotation: bit-flip perturb (density S7_DENSITY) -> staging write ->
                 upload_delta(prev_state=last published state) -> READY

State chain lives in this process's RAM (57 GB x2 during transition).
Per-rotation timings printed as ROTATOR_METRIC json lines.
"""

import asyncio
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

import bittensor as bt  # noqa: E402

from grail.shared.safetensors_utils import load_model_state_dict  # noqa: E402
from grail.trainer.checkpoint_publisher import CheckpointPublisher  # noqa: E402

MODEL = Path(os.getenv("S7_MODEL", "/storage/openpsi/models/Qwen__Qwen3-30B-A3B"))
STAGING = Path("/storage/openpsi/users/pengzai.pyq/grail_bench/s7/staging")
DENSITY = float(os.getenv("S7_DENSITY", "0.01"))
ROTATIONS = int(os.getenv("S7_ROTATIONS", "4"))
SEED = int(os.getenv("S7_SEED", "20260710"))
BLOCK_TIME = float(os.getenv("GRAIL_MOCK_BLOCK_TIME_S", "12"))
GENESIS = float(os.getenv("GRAIL_MOCK_GENESIS_TS", "1780000000"))
WINDOW_LEN = 30
# MODE: "bootstrap" = publish FULL and exit; "deltas" = skip FULL (needs
# S7_ANCHOR_WINDOW from the bootstrap run); "all" = original single-process flow.
# Split exists because run #2 (939775) showed the miner's 57GB moto download is
# too slow to overlap with rotations — sequence: bootstrap -> miner up -> deltas.
MODE = os.getenv("S7_MODE", "all")
ANCHOR_ENV = os.getenv("S7_ANCHOR_WINDOW", "")

SMALL_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
               "tokenizer.json", "vocab.json", "merges.txt")


def say(msg: str) -> None:
    print(f"[s7-rotator {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metric(**kw) -> None:
    print(f"ROTATOR_METRIC {json.dumps(kw)}", flush=True)


def current_window() -> int:
    return (int((time.time() - GENESIS) / BLOCK_TIME) // WINDOW_LEN) * WINDOW_LEN


async def wait_next_window(last: int) -> int:
    while True:
        w = current_window()
        if w > last:
            return w
        await asyncio.sleep(5)


def write_staging(state: dict[str, torch.Tensor]) -> float:
    from safetensors.torch import save_file

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
    elapsed = time.perf_counter() - t0
    for f in SMALL_FILES:
        if (MODEL / f).exists():
            shutil.copy(MODEL / f, STAGING / f)
    return elapsed


def perturb(state: dict[str, torch.Tensor], density: float, gen: torch.Generator) -> dict:
    """Clone + lowest-mantissa-bit flip at `density` fraction (Phase 1 method)."""
    int_view = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
                torch.float32: torch.int32}
    out = {}
    for name, t in state.items():
        c = t.clone()
        ivd = int_view.get(c.dtype)
        if ivd is not None and c.numel() and density > 0:
            mask = torch.rand(c.shape, generator=gen) < density
            c.view(ivd)[mask] ^= 1
            del mask
        out[name] = c
    return out


async def main() -> None:
    gen = torch.Generator().manual_seed(SEED)
    wallet = bt.wallet(name=os.getenv("BT_WALLET_COLD", "grailbench"), hotkey="default")
    wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    publisher = CheckpointPublisher(credentials=None, wallet=wallet)

    # ---- bootstrap FULL ----
    say(f"materializing {MODEL} state dict...")
    t0 = time.perf_counter()
    state = load_model_state_dict(MODEL)
    for k in state:
        state[k] = state[k].clone()
    say(f"materialized in {time.perf_counter() - t0:.0f}s")

    if MODE in ("all", "bootstrap"):
        t_stage = write_staging(state)
        anchor = current_window()
        say(f"bootstrap FULL at window {anchor} (staging write {t_stage:.1f}s), uploading...")
        t0 = time.perf_counter()
        res = await publisher.upload_from_staging(STAGING, {"timestamp": time.time()}, anchor)
        ok = await publisher.finalize_checkpoint_ready(anchor, anchor)
        metric(event="bootstrap_full", window=anchor, staging_write_s=round(t_stage, 2),
               upload_s=round(time.perf_counter() - t0, 2),
               net_upload_s=round(res.timing.network_upload_s, 2), ready=ok)
        if MODE == "bootstrap":
            say("bootstrap-only mode: exiting")
            return
    else:
        anchor = int(ANCHOR_ENV)
        say(f"deltas-only mode: anchor window {anchor} (state = unperturbed model)")
    prev_window, prev_state = anchor, state

    # ---- delta rotations ----
    for i in range(ROTATIONS):
        window = await wait_next_window(prev_window)
        t0 = time.perf_counter()
        current = perturb(prev_state, DENSITY, gen)
        t_perturb = time.perf_counter() - t0
        t_stage = write_staging(current)

        say(f"rotation {i + 1}/{ROTATIONS}: DELTA window {window} (prev {prev_window})")
        t0 = time.perf_counter()
        res = await publisher.upload_delta(
            STAGING, {"timestamp": time.time()},
            target_window=window, prev_window=prev_window,
            prev_state=prev_state, anchor_window=anchor,
        )
        t_upload = time.perf_counter() - t0
        ok = await publisher.finalize_checkpoint_ready(window, window)
        d = res.to_dict()
        metric(event="delta_publish", rotation=i, window=window, prev_window=prev_window,
               perturb_s=round(t_perturb, 2), staging_write_s=round(t_stage, 2),
               upload_total_s=round(t_upload, 2),
               compute_delta_s=round(d.get("timing_compute_delta_s", 0), 2),
               compression_s=round(d.get("timing_compression_s", 0), 2),
               net_upload_s=round(d.get("timing_network_upload_s", 0), 2),
               delta_mb=round(d.get("upload_total_mb", 0), 1),
               sparsity=d.get("sparsity_ratio"), ready=ok)

        del prev_state
        gc.collect()
        prev_window, prev_state = window, current

    say(f"done: FULL + {ROTATIONS} deltas published")


if __name__ == "__main__":
    asyncio.run(main())
