"""S1 acceptance: publish a FULL checkpoint from a local HF model into moto,
then consume it back with integrity verification (goal.md Phase 2 S1).

Flow (all against the moto endpoint from moto_helper.sh):
  1. Stage Qwen2.5-1.5B-Instruct files into a fresh staging dir (publisher
     writes metadata.json/manifest.sig/FULL into staging, so never point it
     at the pristine model dir).
  2. Local throwaway bittensor wallet (offline keypair, no chain access) for
     metadata signing.
  3. CheckpointPublisher.upload_from_staging -> finalize_checkpoint_ready.
  4. CheckpointManager.get_checkpoint downloads + SHA256-verifies.
  5. Bit-exact check: weights_hash(downloaded) == weights_hash(staged).

Runs INSIDE grail-miner.sif with writable HOME (bittensor mkdirs at import).
"""

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

import bittensor as bt  # noqa: E402

from grail.infrastructure.checkpoint_consumer import CheckpointManager  # noqa: E402
from grail.infrastructure.delta_checkpoint import compute_weights_hash  # noqa: E402
from grail.shared.safetensors_utils import load_model_state_dict  # noqa: E402
from grail.trainer.checkpoint_publisher import CheckpointPublisher  # noqa: E402

MODEL_DIR = Path(
    os.getenv("GRAIL_PUBLISH_MODEL", "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct")
)
WORK = Path("/storage/openpsi/users/pengzai.pyq/grail_bench/s1")
STAGING = WORK / "staging"
CACHE = WORK / "consumer_cache"

# Window selection: explicit int, or "auto" = current window of the mock chain
# (must use the same GRAIL_MOCK_BLOCK_TIME_S / GENESIS as the miner run).
_raw_window = os.getenv("GRAIL_PUBLISH_WINDOW", "30")
if _raw_window == "auto":
    _block_time = float(os.getenv("GRAIL_MOCK_BLOCK_TIME_S", "12"))
    _genesis = float(os.getenv("GRAIL_MOCK_GENESIS_TS", "1780000000"))
    _block = int((time.time() - _genesis) / _block_time)
    WINDOW = (_block // 30) * 30
else:
    WINDOW = int(_raw_window)

# GRAIL_PUBLISH_ONLY=1 skips the consumer round-trip (S3: the miner consumes)
PUBLISH_ONLY = os.getenv("GRAIL_PUBLISH_ONLY", "0") == "1"

MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]


async def main() -> None:
    # --- 1. staging ---
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    for f in MODEL_FILES:
        shutil.copy(MODEL_DIR / f, STAGING / f)
    print(f"staged {len(MODEL_FILES)} files -> {STAGING}")

    # --- 2. offline wallet ---
    wallet = bt.wallet(name="grailbench", hotkey="default")
    wallet.create_if_non_existent(
        coldkey_use_password=False, hotkey_use_password=False
    )
    print(f"wallet hotkey: {wallet.hotkey.ss58_address}")

    # --- 3. publish ---
    publisher = CheckpointPublisher(credentials=None, wallet=wallet)
    print(f"publishing to window {WINDOW} (env_id={os.getenv('GRAIL_ENV_ID', 'default')})")
    t0 = time.perf_counter()
    result = await publisher.upload_from_staging(
        STAGING, {"timestamp": time.time()}, target_window=WINDOW
    )
    t_pub = time.perf_counter() - t0
    print(f"upload_from_staging: {t_pub:.1f}s, result={result.to_dict()}")

    ok = await publisher.finalize_checkpoint_ready(WINDOW, WINDOW)
    print("finalize_checkpoint_ready:", ok)
    assert ok

    if PUBLISH_ONLY:
        print(f"PUBLISH-ONLY done (window {WINDOW})")
        return

    # --- 4. consume ---
    if CACHE.exists():
        shutil.rmtree(CACHE)
    cm = CheckpointManager(cache_root=CACHE, credentials=None)
    t0 = time.perf_counter()
    local = await cm.get_checkpoint(WINDOW)
    t_get = time.perf_counter() - t0
    print(f"get_checkpoint: {t_get:.1f}s -> {local}")
    assert local is not None, "consumer failed to fetch checkpoint"

    # --- 5. bit-exact ---
    staged_hash = compute_weights_hash(load_model_state_dict(STAGING))
    got_hash = compute_weights_hash(load_model_state_dict(Path(local)))
    print(f"staged={staged_hash}\ngot   ={got_hash}")
    assert staged_hash == got_hash, "weights hash mismatch after round-trip"

    print("S1 ACCEPTANCE: PASS")


if __name__ == "__main__":
    asyncio.run(main())
