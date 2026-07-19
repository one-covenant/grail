"""S4 rotation publisher: publish a fresh FULL checkpoint every mock window.

Runs alongside the mocked miner (goal.md Phase 2 S4). Each rotation re-publishes
the same staged weights at the CURRENT mock window; the miner sees a new window
number, takes the slow path (download -> proof reload -> SGLang
/update_weights_from_disk) and our SYNC_METRIC instrumentation fires.

Env:
  GRAIL_ROTATIONS          number of checkpoints to publish (default 2)
  GRAIL_MOCK_BLOCK_TIME_S / GRAIL_MOCK_GENESIS_TS   must match the miner
  R2_* vars                moto endpoint (from moto_helper.sh)
"""

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

import bittensor as bt  # noqa: E402

from grail.trainer.checkpoint_publisher import CheckpointPublisher  # noqa: E402

MODEL_DIR = Path("/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct")
STAGING = Path("/storage/openpsi/users/pengzai.pyq/grail_bench/s1/staging")
BLOCK_TIME = float(os.getenv("GRAIL_MOCK_BLOCK_TIME_S", "12"))
GENESIS = float(os.getenv("GRAIL_MOCK_GENESIS_TS", "1780000000"))
ROTATIONS = int(os.getenv("GRAIL_ROTATIONS", "2"))
WINDOW_LEN = 30

MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]


def current_window() -> int:
    block = int((time.time() - GENESIS) / BLOCK_TIME)
    return (block // WINDOW_LEN) * WINDOW_LEN


async def main() -> None:
    if not (STAGING / "model.safetensors").exists():
        STAGING.mkdir(parents=True, exist_ok=True)
        for f in MODEL_FILES:
            shutil.copy(MODEL_DIR / f, STAGING / f)
        print(f"[rotator] staged model into {STAGING}", flush=True)

    wallet = bt.wallet(name=os.getenv("BT_WALLET_COLD", "grailbench"), hotkey="default")
    wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    publisher = CheckpointPublisher(credentials=None, wallet=wallet)

    last_published: int | None = None
    for i in range(ROTATIONS):
        window = current_window()
        if last_published is not None and window <= last_published:
            next_window = last_published + WINDOW_LEN
            wake_ts = GENESIS + next_window * BLOCK_TIME
            wait = max(0.0, wake_ts - time.time()) + 1.0
            print(f"[rotator] window {window} already published, sleeping {wait:.0f}s "
                  f"until window {next_window}", flush=True)
            await asyncio.sleep(wait)
            window = current_window()

        t0 = time.perf_counter()
        # upload_from_staging builds its manifest from EVERY file in staging,
        # then rewrites metadata.json — stale publisher artifacts from the
        # previous publish would get manifested with their OLD hashes and the
        # consumer's integrity check fails (S4a iter 1: "Checksum mismatch for
        # manifest.sig"). Publisher expects a clean staging: scrub its outputs.
        for stale in ("metadata.json", "manifest.sig", "FULL"):
            p = STAGING / stale
            if p.exists():
                p.unlink()
        result = await publisher.upload_from_staging(
            STAGING, {"timestamp": time.time()}, target_window=window
        )
        ok = await publisher.finalize_checkpoint_ready(window, window)
        elapsed = time.perf_counter() - t0
        print(
            f"[rotator] rotation {i + 1}/{ROTATIONS}: window {window} published in "
            f"{elapsed:.1f}s (upload {result.timing.network_upload_s:.1f}s, ready={ok})",
            flush=True,
        )
        last_published = window

    print(f"[rotator] done: {ROTATIONS} checkpoints published", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
