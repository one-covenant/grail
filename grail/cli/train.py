#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bittensor as bt
import torch
import torch.nn.functional as F
import typer
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..infrastructure.checkpoints import (
    CHECKPOINT_PREFIX,
    CheckpointManager,
    CheckpointMetadata,
    default_checkpoint_cache_root,
)
from ..infrastructure.comms import (
    get_valid_rollouts,
    upload_file_chunked,
)
from ..infrastructure.credentials import load_r2_credentials
from ..infrastructure.network import create_subtensor
from ..monitoring import get_monitoring_manager
from ..monitoring.config import MonitoringConfig
from ..shared.constants import (
    MODEL_NAME,
    ROLLOUTS_PER_PROBLEM,
    WINDOW_LENGTH,
)
from . import console

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
logger = logging.getLogger("grail")

# Training hyperparameters (env-configurable with safe defaults)
TRAINER_LR = float(os.getenv("GRAIL_TRAINER_LR", "2e-6"))
TRAINER_EPOCHS = int(os.getenv("GRAIL_TRAINER_EPOCHS", "2"))
TRAINER_BATCH_SIZE = int(os.getenv("GRAIL_TRAINER_BATCH_SIZE", "4"))
TRAINER_MAX_LENGTH = int(os.getenv("GRAIL_TRAINER_MAX_LENGTH", "1024"))
TRAINER_GRAD_CLIP = float(os.getenv("GRAIL_TRAINER_GRAD_CLIP", "0.5"))
TRAINER_WARMUP_STEPS = int(os.getenv("GRAIL_TRAINER_WARMUP_STEPS", "10"))
TRAINER_KL_COEF = float(os.getenv("GRAIL_TRAINER_KL_COEF", "0.02"))
TRAINER_ENTROPY_COEF = float(os.getenv("GRAIL_TRAINER_ENTROPY_COEF", "0.001"))
TRAINER_ADV_CLIP_PERCENTILE = float(os.getenv("GRAIL_TRAINER_ADV_CLIP_PERCENTILE", "99.0"))
TRAINER_GROUP_ADV_SUM_TOL = float(os.getenv("GRAIL_TRAINER_GROUP_ADV_SUM_TOL", "0.01"))

# Watchdog
HEARTBEAT = time.monotonic()


# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #


def get_conf(key: str, default: Any = None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        console.print(f"[red]{key} not set.[/red]\nRun:\n    af set {key} <value>")
        raise typer.Exit(code=1)
    return v or default


# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR: bt.subtensor | None = None


async def get_subtensor() -> bt.subtensor:
    global SUBTENSOR
    if SUBTENSOR is None:
        logger.info("Making Bittensor connection...")
        SUBTENSOR = await create_subtensor()
        logger.info("Connected")
    return SUBTENSOR


# --------------------------------------------------------------------------- #
#                          GRPO Rollout Data                                  #
# --------------------------------------------------------------------------- #


@dataclass
class GRPORollout:
    """Single rollout from a GRPO group."""

    tokens: list[int]
    prompt_length: int
    completion_length: int
    advantage: float
    reward: float
    success: bool
    nonce: int
    rollout_group: str


@dataclass
class GRPOGroup:
    """A group of rollouts for a single SAT problem."""

    group_id: str
    rollouts: list[GRPORollout]

    def is_valid(self) -> bool:
        """Check if group size matches expected and advantages sum to ~0."""
        if len(self.rollouts) != ROLLOUTS_PER_PROBLEM:
            return False
        adv_sum = sum(r.advantage for r in self.rollouts)
        return abs(adv_sum) < TRAINER_GROUP_ADV_SUM_TOL


# --------------------------------------------------------------------------- #
#                         Data Ingestion                                      #
# --------------------------------------------------------------------------- #


async def load_grpo_groups(window: int) -> list[GRPOGroup]:
    """Load validated rollouts for window and organize into GRPO groups.

    Returns:
        List of valid GRPO groups with correct size and advantage sums.
    """
    rollouts_data = await get_valid_rollouts(window)
    if not rollouts_data:
        logger.warning(f"No valid rollouts found for window {window}")
        return []

    if not isinstance(rollouts_data, dict):
        logger.warning(f"Invalid rollouts data format for window {window}")
        return []

    rollouts = rollouts_data.get("rollouts", [])
    logger.info(f"Loaded {len(rollouts)} rollouts for window {window}")

    # Group by rollout_group
    groups_map: dict[str, list[dict]] = defaultdict(list)
    for rollout_dict in rollouts:
        group_id = str(rollout_dict.get("rollout_group", ""))
        if not group_id:
            continue
        groups_map[group_id].append(rollout_dict)

    # Build GRPOGroup objects
    grpo_groups: list[GRPOGroup] = []
    for group_id, group_rollouts in groups_map.items():
        parsed_rollouts: list[GRPORollout] = []
        for r_dict in group_rollouts:
            try:
                commit = r_dict.get("commit", {})
                rollout_meta = commit.get("rollout", {})
                parsed_rollouts.append(
                    GRPORollout(
                        tokens=commit.get("tokens", []),
                        prompt_length=int(rollout_meta.get("prompt_length", 0)),
                        completion_length=int(rollout_meta.get("completion_length", 0) or 0),
                        advantage=float(rollout_meta.get("advantage", 0.0)),
                        reward=float(rollout_meta.get("total_reward", 0.0)),
                        success=bool(rollout_meta.get("success", False)),
                        nonce=int(r_dict.get("nonce", 0)),
                        rollout_group=str(r_dict.get("rollout_group", "")),
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to parse rollout: {e}")
                continue

        if parsed_rollouts:
            group = GRPOGroup(group_id=group_id, rollouts=parsed_rollouts)
            grpo_groups.append(group)

    # Filter to valid groups
    valid_groups = [g for g in grpo_groups if g.is_valid()]
    invalid_count = len(grpo_groups) - len(valid_groups)
    if invalid_count > 0:
        logger.warning(
            f"Filtered out {invalid_count} invalid groups "
            f"(size != {ROLLOUTS_PER_PROBLEM} or bad advantage sum)"
        )

    logger.info(f"Loaded {len(valid_groups)} valid GRPO groups for window {window}")
    return valid_groups


# --------------------------------------------------------------------------- #
#                          GRPO Training Logic                                #
# --------------------------------------------------------------------------- #


def compute_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
) -> torch.Tensor:
    """Compute per-sequence log probabilities over completion tokens.

    Args:
        model: Language model
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        prompt_lengths: List of prompt lengths per sample
        completion_lengths: List of completion lengths per sample

    Returns:
        Tensor of shape [batch_size] with sum(log p(token)) over completions
    """
    with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # Shift for causal LM: logits[:, :-1] predict tokens[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log probs of actual tokens
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask to only completion tokens
    batch_size, seq_len_minus_1 = token_log_probs.shape
    seq_log_probs = []
    for i in range(batch_size):
        prompt_len = prompt_lengths[i]
        comp_len = completion_lengths[i]
        # Completion starts at prompt_len, length comp_len
        # After shift, completion region is [prompt_len-1 : prompt_len-1+comp_len]
        start_idx = max(0, prompt_len - 1)
        end_idx = min(seq_len_minus_1, start_idx + comp_len)
        if end_idx > start_idx:
            comp_logprobs = token_log_probs[i, start_idx:end_idx]
            seq_log_probs.append(comp_logprobs.sum())
        else:
            seq_log_probs.append(torch.tensor(0.0, device=token_log_probs.device))

    return torch.stack(seq_log_probs)


def compute_entropy(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
) -> torch.Tensor:
    """Compute mean entropy over completion tokens.

    Returns:
        Tensor of shape [batch_size]
    """
    with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy_per_token = -(probs * log_probs).sum(dim=-1)  # [batch, seq-1]

    batch_size = entropy_per_token.shape[0]
    seq_entropies = []
    for i in range(batch_size):
        prompt_len = prompt_lengths[i]
        comp_len = completion_lengths[i]
        start_idx = max(0, prompt_len - 1)
        end_idx = min(entropy_per_token.shape[1], start_idx + comp_len)
        if end_idx > start_idx:
            mean_ent = entropy_per_token[i, start_idx:end_idx].mean()
            seq_entropies.append(mean_ent)
        else:
            seq_entropies.append(torch.tensor(0.0, device=entropy_per_token.device))

    return torch.stack(seq_entropies)


async def train_grpo_epoch(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    groups: list[GRPOGroup],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    accelerator: Accelerator,
    monitor: Any,
    window: int,
) -> dict[str, float]:
    """Run one epoch of GRPO training.

    Returns:
        Dict of aggregated metrics for the epoch.
    """
    model.train()
    ref_model.eval()

    # Flatten groups into batches of rollouts
    all_rollouts: list[tuple[GRPORollout, str]] = []
    for group in groups:
        for rollout in group.rollouts:
            all_rollouts.append((rollout, group.group_id))

    # Sort deterministically by group_id then nonce for reproducibility
    all_rollouts.sort(key=lambda x: (x[1], x[0].nonce))

    batch_size = TRAINER_BATCH_SIZE
    num_batches = math.ceil(len(all_rollouts) / batch_size)

    epoch_metrics = defaultdict(list)

    for batch_idx in range(num_batches):
        global HEARTBEAT
        HEARTBEAT = time.monotonic()

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_rollouts))
        batch_rollouts = [all_rollouts[i][0] for i in range(start_idx, end_idx)]

        # Prepare batch
        batch_tokens = []
        batch_prompt_lens = []
        batch_comp_lens = []
        batch_advantages = []

        for rollout in batch_rollouts:
            tokens = rollout.tokens[:TRAINER_MAX_LENGTH]
            batch_tokens.append(tokens)
            batch_prompt_lens.append(rollout.prompt_length)
            batch_comp_lens.append(rollout.completion_length)
            batch_advantages.append(rollout.advantage)

        # Tokenize and pad
        # Already have token IDs; just pad
        max_len = max(len(t) for t in batch_tokens)
        input_ids = []
        attention_masks = []
        for tokens in batch_tokens:
            padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            input_ids.append(padded)
            attention_masks.append(mask)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).to(accelerator.device)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long).to(
            accelerator.device
        )
        advantages_tensor = torch.tensor(batch_advantages, dtype=torch.float32).to(
            accelerator.device
        )

        # Forward pass: current model
        logprobs_current = compute_logprobs(
            model,
            input_ids_tensor,
            attention_mask_tensor,
            batch_prompt_lens,
            batch_comp_lens,
        )

        # Forward pass: reference model (no grad)
        with torch.no_grad():
            logprobs_ref = compute_logprobs(
                ref_model,
                input_ids_tensor,
                attention_mask_tensor,
                batch_prompt_lens,
                batch_comp_lens,
            )

        # Compute proper KL divergence (not squared difference)
        # KL(current || ref) = ref_logprob - current_logprob
        # (for same sequence, this approximates KL between distributions)
        kl_div = logprobs_ref - logprobs_current

        # Normalize advantages within batch (GRPO best practice)
        if advantages_tensor.std() > 1e-8:
            advantages_normalized = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )
        else:
            advantages_normalized = advantages_tensor

        # Policy gradient loss with normalized advantages
        loss_pg = -(advantages_normalized * logprobs_current).mean()

        # KL penalty (keep model close to reference)
        loss_kl = TRAINER_KL_COEF * kl_div.mean()

        # Entropy
        entropies = compute_entropy(
            model,
            input_ids_tensor,
            attention_mask_tensor,
            batch_prompt_lens,
            batch_comp_lens,
        )
        loss_entropy = -TRAINER_ENTROPY_COEF * entropies.mean()

        loss_total = loss_pg + loss_kl + loss_entropy

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss_total).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINER_GRAD_CLIP)

        # Check for NaN/Inf
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logger.warning("NaN/Inf grad norm detected, skipping batch")
            continue

        scaler.step(optimizer)
        scaler.update()

        # Log metrics
        epoch_metrics["loss_total"].append(loss_total.item())
        epoch_metrics["loss_pg"].append(loss_pg.item())
        epoch_metrics["loss_kl"].append(loss_kl.item())
        epoch_metrics["loss_entropy"].append(loss_entropy.item())
        epoch_metrics["grad_norm"].append(grad_norm.item())
        epoch_metrics["advantage_mean"].append(advantages_tensor.mean().item())
        epoch_metrics["advantage_std"].append(advantages_tensor.std().item())
        epoch_metrics["entropy_mean"].append(entropies.mean().item())
        epoch_metrics["advantage_mean_normalized"].append(advantages_normalized.mean().item())
        epoch_metrics["advantage_std_normalized"].append(advantages_normalized.std().item())

        # KL divergence (proper)
        kl_divergence = kl_div.mean().item()
        epoch_metrics["kl_divergence"].append(kl_divergence)

    # Aggregate epoch metrics
    aggregated = {
        key: sum(vals) / len(vals) if vals else 0.0 for key, vals in epoch_metrics.items()
    }

    return aggregated


async def train_window(
    window: int,
    wallet: bt.wallet,
    credentials: Any,
    checkpoint_manager: CheckpointManager,
    monitor: Any,
) -> bool:
    """Train model on validated rollouts for window and publish checkpoint.

    Args:
        window: Training window
        wallet: Trainer wallet for signing
        credentials: R2 credentials
        checkpoint_manager: Checkpoint manager
        monitor: Monitoring manager

    Returns:
        True if training and publishing succeeded
    """
    global HEARTBEAT
    HEARTBEAT = time.monotonic()

    # Seed RNGs for determinism
    seed = window
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"🎓 Training window {window} with seed {seed}")

    # Load GRPO groups
    groups = await load_grpo_groups(window)
    if not groups:
        logger.warning(f"No valid groups for window {window}, publishing stable checkpoint")
        # TODO: publish previous stable as fallback
        return False

    # Compute stats
    total_rollouts = sum(len(g.rollouts) for g in groups)
    success_count = sum(1 for g in groups for r in g.rollouts if r.success)
    mean_reward = (
        sum(r.reward for g in groups for r in g.rollouts) / total_rollouts
        if total_rollouts > 0
        else 0.0
    )

    logger.info(
        f"📚 Training on {len(groups)} groups ({total_rollouts} rollouts), "
        f"{success_count} successful, mean reward: {mean_reward:.3f}"
    )

    # Load models
    accelerator = Accelerator(mixed_precision="fp16")

    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        use_safetensors=True,  # Use safetensors for security
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load reference model from latest_stable or base
    ref_model_path = None
    try:
        windows = await checkpoint_manager.list_remote_windows()
        if windows:
            latest_window = max(windows)
            # Time checkpoint download/retrieval
            timer_ctx = (
                monitor.timer("training/checkpoint_download")
                if monitor
                else contextlib.nullcontext()
            )
            with timer_ctx:
                ref_checkpoint = await checkpoint_manager.get_checkpoint(latest_window)
            if ref_checkpoint:
                ref_model_path = str(ref_checkpoint)
    except Exception as e:
        logger.debug(f"Failed to load reference checkpoint: {e}")

    if ref_model_path:
        logger.info("Loading reference model from %s", ref_model_path)
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    else:
        logger.info("Using base model as reference")
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

    # Prepare models with accelerator
    model, ref_model = accelerator.prepare(model, ref_model)
    ref_model.eval()

    # Optimizer and scheduler (prepare with accelerator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINER_LR)
    optimizer = accelerator.prepare(optimizer)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=TRAINER_WARMUP_STEPS
    )

    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(TRAINER_EPOCHS):
        HEARTBEAT = time.monotonic()
        logger.info(f"Epoch {epoch + 1}/{TRAINER_EPOCHS}")

        epoch_metrics = await train_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            groups,
            optimizer,
            scaler,
            accelerator,
            monitor,
            window,
        )

        scheduler.step()

        # Log to monitoring
        if monitor:
            for key, value in epoch_metrics.items():
                await monitor.log_gauge(f"training/{key}", value)
            await monitor.log_gauge("training/lr", scheduler.get_last_lr()[0])
            await monitor.log_counter("training/epochs_completed")

        logger.info(
            f"Epoch {epoch + 1} - "
            f"loss: {epoch_metrics['loss_total']:.4f}, "
            f"pg: {epoch_metrics['loss_pg']:.4f}, "
            f"kl: {epoch_metrics['loss_kl']:.4f}"
        )

    # Publish checkpoint for future window
    future_window = window + WINDOW_LENGTH
    logger.info(f"💾 Publishing checkpoint for window {future_window}")

    # Unwrap model from accelerator before saving
    unwrapped_model = accelerator.unwrap_model(model)

    success = await publish_checkpoint(
        unwrapped_model,
        tokenizer,
        future_window,
        window,
        wallet,
        credentials,
        checkpoint_manager,
        seed,
    )

    if success:
        logger.info(f"✅ Successfully published checkpoint for window {future_window}")
    else:
        logger.error(f"❌ Failed to publish checkpoint for window {future_window}")

    return success


async def publish_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    target_window: int,
    trained_on_window: int,
    wallet: bt.wallet,
    credentials: Any,
    checkpoint_manager: CheckpointManager,
    seed: int,
) -> bool:
    """Publish HF-style checkpoint with manifest and markers.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        target_window: Window this checkpoint is for
        trained_on_window: Window used for training
        wallet: Trainer wallet
        credentials: R2 credentials
        checkpoint_manager: Checkpoint manager
        seed: Training seed

    Returns:
        True if publishing succeeded
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=f"checkpoint-{target_window}-"))
    try:
        # Save model and tokenizer
        logger.info(f"Saving checkpoint to {temp_dir}")
        model.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)

        # Compute file manifest
        file_manifest: dict[str, str] = {}
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(temp_dir))
                file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
                file_manifest[rel_path] = file_hash

        # Build metadata
        git_commit = os.getenv("GIT_COMMIT", "unknown")
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
            git_commit=git_commit,
            created_at=time.time(),
        )

        metadata_dict = {
            "window": metadata.window,
            "parent_window": metadata.parent_window,
            "model_name": metadata.model_name,
            "file_manifest": metadata.file_manifest,
            "training_config": metadata.training_config,
            "git_commit": metadata.git_commit,
            "created_at": metadata.created_at,
            "config_hash": config_hash,
        }

        # Write metadata.json
        metadata_path = temp_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2))

        # Sign manifest
        metadata_canonical = json.dumps(metadata_dict, sort_keys=True, separators=(",", ":"))
        signature = wallet.hotkey.sign(data=metadata_canonical).hex()
        manifest_sig_path = temp_dir / "manifest.sig"
        manifest_sig_path.write_text(signature)

        # Upload all files
        remote_prefix = f"{CHECKPOINT_PREFIX}checkpoint-{target_window}"
        logger.info(f"Uploading checkpoint files to {remote_prefix}")

        # Upload files concurrently (limit concurrency to avoid overwhelming)
        semaphore = asyncio.Semaphore(4)

        async def upload_file(file_path: Path) -> bool:
            async with semaphore:
                rel_path = file_path.relative_to(temp_dir)
                remote_key = f"{remote_prefix}/{rel_path}"
                try:
                    content = file_path.read_bytes()
                    success = await upload_file_chunked(
                        remote_key, content, credentials=credentials, use_write=True
                    )
                    return success
                except Exception as e:
                    logger.error(f"Failed to upload {rel_path}: {e}")
                    return False

        upload_tasks = [upload_file(fp) for fp in temp_dir.rglob("*") if fp.is_file()]
        results = await asyncio.gather(*upload_tasks)

        if not all(results):
            logger.error("Some files failed to upload")
            return False

        # Upload READY marker
        ready_key = f"{remote_prefix}/READY"
        await upload_file_chunked(ready_key, b"", credentials=credentials, use_write=True)

        # Update latest_stable
        latest_stable_key = f"{CHECKPOINT_PREFIX}latest_stable"
        await upload_file_chunked(
            latest_stable_key,
            str(target_window).encode(),
            credentials=credentials,
            use_write=True,
        )

        logger.info(f"✅ Published checkpoint for window {target_window}")

        # Cleanup old checkpoints
        try:
            await checkpoint_manager.cleanup_remote(target_window)
            logger.info("Cleaned up old remote checkpoints")
        except Exception as e:
            logger.warning(f"Failed to cleanup remote checkpoints: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to publish checkpoint: {e}", exc_info=True)
        return False
    finally:
        # Cleanup temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #


async def watchdog(timeout: int = 600) -> None:
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #


def register(app: typer.Typer) -> None:
    app.command("train")(train)


def train() -> None:
    """Run the GRPO training process."""
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    logger.info(f"🔑 Trainer hotkey: {wallet.hotkey.ss58_address}")

    async def _run() -> None:
        subtensor = None
        last_processed_window = -1

        # Load credentials
        try:
            credentials = load_r2_credentials()
            logger.info("✅ Loaded R2 credentials")
        except Exception as e:
            logger.error(f"Failed to load R2 credentials: {e}")
            raise

        # Initialize checkpoint manager (trainer role)
        checkpoint_manager = CheckpointManager(
            cache_root=default_checkpoint_cache_root(),
            credentials=credentials,
            role="trainer",
        )

        # Initialize monitoring
        monitor = get_monitoring_manager()
        if monitor:
            training_config = MonitoringConfig.for_training(wallet.name)
            run_id = await monitor.start_run(
                f"trainer_{wallet.name}",
                training_config.get("hyperparameters", {}),
            )
            logger.info(f"Started monitoring run: {run_id}")

        while True:
            try:
                global HEARTBEAT
                HEARTBEAT = time.monotonic()

                if subtensor is None:
                    subtensor = await get_subtensor()

                current_block = await subtensor.get_current_block()
                current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH

                # Train on previous complete window
                target_window = current_window - WINDOW_LENGTH

                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(10)
                    continue

                logger.info(f"🎓 Processing training for window {target_window}")

                success = await train_window(
                    target_window,
                    wallet,
                    credentials,
                    checkpoint_manager,
                    monitor,
                )

                if success:
                    logger.info(f"✅ Completed training cycle for window {target_window}")
                    if monitor:
                        await monitor.log_counter("training/successful_windows")
                else:
                    logger.warning(f"⚠️ Training cycle had issues for window {target_window}")
                    if monitor:
                        await monitor.log_counter("training/failed_windows")

                last_processed_window = target_window

            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in trainer loop: {e}. Continuing...")
                subtensor = None
                await asyncio.sleep(30)
                continue

    async def _main() -> None:
        await asyncio.gather(_run(), watchdog(timeout=(60 * 15)))

    asyncio.run(_main())


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    train()
