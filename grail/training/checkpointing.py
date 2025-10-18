"""Checkpoint publishing utilities for the trainer."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import bittensor as bt

from ..infrastructure.checkpoints import (
    CHECKPOINT_PREFIX,
    CheckpointManager,
    CheckpointMetadata,
)
from ..infrastructure.comms import upload_file_chunked
from ..shared.constants import (
    MODEL_NAME,
    TRAINER_BATCH_SIZE,
    TRAINER_ENTROPY_COEF,
    TRAINER_EPOCHS,
    TRAINER_GRAD_CLIP,
    TRAINER_KL_COEF,
    TRAINER_LR,
    TRAINER_MAX_LENGTH,
    TRAINER_WARMUP_STEPS,
)

logger = logging.getLogger(__name__)


async def publish_checkpoint(
    model: Any,
    tokenizer: Any,
    target_window: int,
    trained_on_window: int,
    wallet: bt.wallet,
    credentials: Any,
    checkpoint_manager: CheckpointManager,
    seed: int,
) -> bool:
    """Publish a HF-style checkpoint to R2 and update metadata pointers."""

    temp_dir = Path(tempfile.mkdtemp(prefix=f"checkpoint-{target_window}-"))
    try:
        logger.info("Saving checkpoint to %s", temp_dir)
        model.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)

        file_manifest: dict[str, str] = {}
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(temp_dir))
                file_manifest[rel_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()

        training_config = {
            "lr": TRAINER_LR,
            "epochs": TRAINER_EPOCHS,
            "batch_size": TRAINER_BATCH_SIZE,
            "max_length": TRAINER_MAX_LENGTH,
            "grad_clip": TRAINER_GRAD_CLIP,
            "warmup_steps": TRAINER_WARMUP_STEPS,
            "kl_coef": TRAINER_KL_COEF,
            "entropy_coef": TRAINER_ENTROPY_COEF,
            "seed": seed,
        }
        config_hash = hashlib.sha256(
            json.dumps(training_config, sort_keys=True).encode()
        ).hexdigest()

        metadata = CheckpointMetadata(
            window=target_window,
            parent_window=trained_on_window,
            model_name=MODEL_NAME,
            file_manifest=file_manifest,
            training_config=training_config,
            git_commit=os.getenv("GIT_COMMIT", "unknown"),
            created_at=time.time(),
        )

        metadata_dict = {**metadata.__dict__, "config_hash": config_hash}
        metadata_path = temp_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2))

        canonical_metadata = json.dumps(metadata_dict, sort_keys=True, separators=(",", ":"))
        signature = wallet.hotkey.sign(data=canonical_metadata).hex()
        (temp_dir / "manifest.sig").write_text(signature)

        remote_prefix = f"{CHECKPOINT_PREFIX}checkpoint-{target_window}"
        semaphore = asyncio.Semaphore(4)

        async def upload_file(path: Path) -> bool:
            async with semaphore:
                rel_path = path.relative_to(temp_dir)
                remote_key = f"{remote_prefix}/{rel_path}"
                try:
                    return await upload_file_chunked(
                        remote_key,
                        path.read_bytes(),
                        credentials=credentials,
                        use_write=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to upload %s: %s", rel_path, exc)
                    return False

        upload_tasks = [upload_file(path) for path in temp_dir.rglob("*") if path.is_file()]
        results = await asyncio.gather(*upload_tasks)
        if not all(results):
            logger.error("Some checkpoint files failed to upload for window %s", target_window)
            return False

        await upload_file_chunked(
            f"{remote_prefix}/READY",
            b"",
            credentials=credentials,
            use_write=True,
        )

        logger.info("✅ Published checkpoint for window %s", target_window)

        try:
            await checkpoint_manager.cleanup_remote(target_window)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to perform remote checkpoint cleanup: %s", exc)

        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to publish checkpoint: %s", exc)
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
