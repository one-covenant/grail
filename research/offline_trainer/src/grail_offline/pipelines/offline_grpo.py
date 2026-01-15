"""Offline GRPO training pipeline.

Coordinates:
1. vLLM server setup for rollout generation (managed lifecycle)
2. Rollout generation via server
3. Training epochs using GRPO algorithm
4. Periodic evaluation using the same server
5. Checkpoint management
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig

from grail.environments.core import MultiTurnEnv
from grail.environments.gsm8k_env import GSM8KEnv
from grail.environments.sat_env import SATEnv
from grail.model.provider import get_model, get_tokenizer
from grail.shared.chat_templates import build_qwen_chat_template
from grail.shared.prompt_constants import SYSTEM_PROMPT
from grail.trainer.algorithms.grpo import GRPOAlgorithm
from grail.trainer.config import EvalConfig, TrainingConfig
from grail.trainer.eval_planner import EvaluationPlan
from grail.trainer.evaluator import EvaluatorService
from grail.trainer.inference_server import ServerConfig, VLLMServerManager
from grail_offline.data.offline_rollouts import OfflineRolloutGenerator, RolloutGenConfig

logger = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def _select_iteration_seeds(all_seeds: list[int], iteration: int, per_iter: int) -> list[int]:
    """Select seeds for current iteration with wraparound."""
    n = len(all_seeds)
    if n == 0:
        return []
    start = (iteration * per_iter) % n
    end = start + per_iter
    if end <= n:
        return all_seeds[start:end]
    return all_seeds[start:] + all_seeds[: end - n]


def _create_env_factory(environment: str) -> Any:
    """Create environment factory function."""

    def factory() -> MultiTurnEnv:
        env_type = environment.lower()
        if env_type == "gsm8k":
            return GSM8KEnv()
        return SATEnv()

    return factory


def _save_checkpoint_for_vllm(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    iteration: int,
) -> Path:
    """Save model and tokenizer checkpoint for vLLM server.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Base directory for checkpoints
        iteration: Current training iteration number

    Returns:
        Path to saved checkpoint directory
    """
    checkpoint_dir = output_dir / f"vllm_checkpoint_iter_{iteration:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)

    logger.info(
        "Saved vLLM checkpoint",
        extra={"iteration": iteration, "path": str(checkpoint_dir)},
    )
    return checkpoint_dir


def _delete_directory_recursive(directory: Path) -> None:
    """Recursively delete a directory and all its contents.

    Args:
        directory: Directory path to delete
    """
    for sub in directory.glob("**/*"):
        if sub.is_file():
            sub.unlink(missing_ok=True)
    for subdir in sorted(directory.glob("**/*"), reverse=True):
        if subdir.is_dir():
            subdir.rmdir()
    directory.rmdir()


def _prune_old_checkpoints(ckpt_dir: Path, keep_last_k: int) -> None:
    """Remove old checkpoints, keeping only the last K.

    Args:
        ckpt_dir: Directory containing checkpoints
        keep_last_k: Number of most recent checkpoints to keep
    """
    existing = sorted([p for p in ckpt_dir.iterdir() if p.is_dir()])
    if len(existing) <= keep_last_k:
        return

    logger.info(
        "Pruning old checkpoints",
        extra={"keep": keep_last_k, "existing": len(existing)},
    )
    for old in existing[:-keep_last_k]:
        try:
            _delete_directory_recursive(old)
            logger.info("Deleted old checkpoint", extra={"path": str(old)})
        except Exception as e:
            logger.warning(
                "Failed to delete checkpoint",
                extra={"path": str(old), "error": str(e)},
            )


def _prune_vllm_checkpoints(workdir: Path, keep_last_k: int) -> None:
    """Remove old vLLM checkpoints, keeping only the last K.

    Args:
        workdir: Working directory containing vLLM checkpoints
        keep_last_k: Number of most recent vLLM checkpoints to keep
    """
    existing_vllm = sorted(workdir.glob("vllm_checkpoint_iter_*"))
    if len(existing_vllm) <= keep_last_k:
        return

    logger.debug(
        "Pruning old vLLM checkpoints",
        extra={"keep": keep_last_k, "existing": len(existing_vllm)},
    )
    for old_vllm in existing_vllm[:-keep_last_k]:
        try:
            _delete_directory_recursive(old_vllm)
            logger.debug("Pruned vLLM checkpoint", extra={"path": str(old_vllm)})
        except Exception as e:
            logger.warning(
                "Failed to prune vLLM checkpoint",
                extra={"path": str(old_vllm), "error": str(e)},
            )


async def _log_rollout_stats(monitor: Any, groups: list[Any]) -> None:
    """Log rollout generation statistics to monitor."""
    if not monitor:
        return

    await monitor.log_gauge("training/num_groups", len(groups))
    await monitor.log_gauge("training/num_rollouts", sum(len(g.rollouts) for g in groups))

    all_rewards = [r.reward for g in groups for r in g.rollouts]
    all_successes = [r.success for g in groups for r in g.rollouts]
    if all_rewards:
        await monitor.log_gauge("training/rollout_reward_mean", float(np.mean(all_rewards)))
        await monitor.log_gauge("training/rollout_reward_std", float(np.std(all_rewards)))
        await monitor.log_gauge("training/rollout_success_rate", float(np.mean(all_successes)))


async def _log_training_metrics(monitor: Any, metrics: dict[str, float]) -> None:
    """Log training metrics to monitor."""
    if not monitor:
        return

    for metric_name, metric_value in metrics.items():
        await monitor.log_gauge(f"training/{metric_name}", float(metric_value))


async def _log_eval_metrics(monitor: Any, metrics: dict[str, float]) -> None:
    """Log evaluation metrics to monitor."""
    if not monitor:
        return

    for metric_name, metric_value in metrics.items():
        await monitor.log_gauge(f"eval/{metric_name}", float(metric_value))


async def _run_evaluation(
    cfg: DictConfig,
    tokenizer: Any,
    train_model: Any,
    training_device: str,
    model_name: str,
    iteration: int,
    monitor: Any,
    metrics_dir: Path,
    server_base_url: str,
) -> None:
    """Run evaluation cycle using vLLM server."""
    logger.info("Starting evaluation", extra={"iteration": iteration})

    eval_backend = str(cfg.eval.backend).lower()

    # Create evaluation config
    eval_cfg = EvalConfig(
        batch_size=int(cfg.eval.batch_size),
        do_sample=bool(cfg.eval.do_sample),
        max_new_tokens=int(cfg.eval.max_new_tokens),
        temperature=float(cfg.eval.temperature),
        top_p=float(cfg.eval.top_p),
        backend=eval_backend,
    )

    # Create environment factory
    env_factory = _create_env_factory(str(cfg.data.environment))

    # Create evaluator service using the managed server
    evaluator = EvaluatorService(
        model=None,  # Server-backed
        tokenizer=tokenizer,
        env_factory=env_factory,
        config=eval_cfg,
        monitor=monitor,
        device=training_device,
        server_base_url=server_base_url,
        server_model_name=model_name,
    )

    # Build evaluation IDs
    eval_ids = [
        str(i)
        for i in range(
            int(cfg.eval.id_seed_start),
            int(cfg.eval.id_seed_start) + int(cfg.eval.num_ids),
        )
    ]
    if cfg.eval.ids:
        eval_ids = [str(x) for x in cfg.eval.ids]

    plan = EvaluationPlan(
        ids=eval_ids,
        replicates=int(cfg.eval.replicates),
        cycle_index=iteration,
        seed_base=int(cfg.eval.seed_base),
    )

    try:
        eval_metrics = await evaluator.run_cycle(plan)

        await _log_eval_metrics(monitor, eval_metrics)

        logger.info(
            "Evaluation complete",
            extra={
                "iteration": iteration,
                "pass@1": eval_metrics.get("pass@1"),
                "mean@1": eval_metrics.get("mean@1"),
            },
        )

        (metrics_dir / f"eval_iter_{iteration:03d}.json").write_text(
            json.dumps(eval_metrics, indent=2)
        )
    finally:
        evaluator.shutdown()


async def run_training(cfg: DictConfig, workdir: Path, monitor: Any | None = None) -> None:
    """Run offline GRPO training pipeline with managed vLLM server.

    This pipeline coordinates:
    1. Initial model loading and vLLM server startup
    2. Rollout generation via vLLM server
    3. GRPO training epoch on generated rollouts
    4. Checkpoint save and vLLM server reload with updated weights
    5. Periodic evaluation and checkpoint management

    After each training epoch, the updated model weights are saved and the vLLM
    server is reloaded to ensure subsequent rollouts use the latest trained model.

    Args:
        cfg: Hydra configuration with training, generation, and evaluation settings
        workdir: Working directory for outputs (checkpoints, metrics, logs)
        monitor: Optional monitoring manager for metrics logging
    """
    _set_global_seed(int(cfg.seed))

    # Setup directories
    metrics_dir = workdir / "metrics"
    ckpt_dir = workdir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build tokenizer with chat template
    chat_template = build_qwen_chat_template(SYSTEM_PROMPT)
    train_id = str(cfg.model.train_id)
    ref_id = str(cfg.model.ref_id)

    # Determine training device
    gpu_strategy = str(cfg.gpu.strategy).lower()
    if gpu_strategy == "multi":
        training_device = f"cuda:{int(cfg.gpu.training_gpu)}"
        logger.info(
            "Multi-GPU mode enabled",
            extra={
                "training_gpu": int(cfg.gpu.training_gpu),
                "vllm_gpu": int(cfg.gpu.vllm_gpu),
            },
        )
    else:
        training_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Single-GPU mode enabled", extra={"device": training_device})

    # Load training models first
    logger.info("Loading training models...")
    train_model = get_model(train_id, device=training_device, eval_mode=False)
    tokenizer = get_tokenizer(train_id, chat_template=chat_template)

    kl_enabled = float(cfg.train.kl_coef) > 0.0
    ref_model = None
    if kl_enabled:
        ref_model = get_model(ref_id, device=training_device, eval_mode=True)
        logger.info("Loaded train and ref models", extra={"kl_coef": cfg.train.kl_coef})
    else:
        logger.info("Loaded train model only (KL disabled)")

    # Save initial model and tokenizer for vLLM server
    logger.info("Saving initial model checkpoint for vLLM server...")
    server_model_dir = _save_checkpoint_for_vllm(
        model=train_model,
        tokenizer=tokenizer,
        output_dir=workdir,
        iteration=-1,  # -1 indicates initial checkpoint
    )

    # Start vLLM server using the saved checkpoint
    logger.info("Starting vLLM inference server...")
    # Configure server environment (pin GPU for multi-GPU)
    server_env: dict[str, str] | None = None
    if gpu_strategy == "multi":
        try:
            server_env = {"CUDA_VISIBLE_DEVICES": str(int(cfg.gpu.vllm_gpu))}
        except Exception:
            server_env = None

    server_config = ServerConfig(
        host="127.0.0.1",
        port=int(cfg.generation.get("port", 30001)),
        timeout_s=180.0,
        trust_remote_code=True,
        model_path=str(server_model_dir),  # Use saved checkpoint
        chat_template_path=str(server_model_dir / "chat_template.jinja"),
        env=server_env,
    )
    # Pick memory utilization based on GPU strategy
    mem_util = (
        float(cfg.gpu.get("vllm_gpu_memory_utilization_single", 0.25))
        if gpu_strategy == "single"
        else float(cfg.gpu.get("vllm_gpu_memory_utilization", 0.75))
    )
    eval_config = EvalConfig(
        vllm_gpu_memory_utilization=mem_util,
        vllm_max_model_len=int(cfg.generation.get("max_model_len", 2048)),
        vllm_max_num_seqs=int(cfg.generation.get("max_num_seqs", 64)),
        stream_server_logs=True,
    )

    import sys

    server_manager = VLLMServerManager(
        config=server_config,
        eval_config=eval_config,
        python_executable=sys.executable,
    )

    # Use async context manager for server lifecycle
    async with server_manager:
        await server_manager.start_server()
        base_url = server_manager.base_url
        model_name = server_manager.model_name
        logger.info(f"✓ vLLM server ready at {base_url} serving {model_name}")

        # Create optimizer and accelerator
        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=float(cfg.train.lr),
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
        accelerator = Accelerator(mixed_precision="no")

        # Create rollout generator pointing to managed server
        gen_cfg = RolloutGenConfig(
            backend=str(cfg.generation.backend),
            base_url=base_url,
            model_name=model_name,
            batch_size=int(cfg.generation.batch_size),
            max_new_tokens=int(cfg.generation.max_new_tokens),
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            top_k=int(cfg.generation.top_k) if cfg.generation.top_k is not None else None,
            repetition_penalty=(
                float(cfg.generation.repetition_penalty)
                if cfg.generation.repetition_penalty is not None
                else None
            ),
            rollouts_per_problem=int(cfg.data.rollouts_per_problem),
            environment=str(cfg.data.environment),
        )
        # Pass HF model for computing ground-truth logprobs
        # This ensures behavior policy logprobs match training logprobs exactly
        generator = OfflineRolloutGenerator(
            tokenizer=tokenizer, config=gen_cfg, hf_model=train_model
        )

        # Create training config from hydra config
        train_cfg = TrainingConfig(
            # Basic training parameters
            lr=float(cfg.train.lr),
            batch_size=int(cfg.train.batch_size),
            grad_clip=float(cfg.train.grad_clip),
            warmup_steps=int(cfg.train.warmup_steps),
            max_length=int(cfg.train.max_length),
            # Loss coefficients
            kl_coef=float(cfg.train.kl_coef),
            entropy_coef=float(cfg.train.entropy_coef),
            kl_target=float(cfg.train.kl_target),
            kl_adapt_rate=float(cfg.train.kl_adapt_rate),
            kl_min=float(cfg.train.kl_min),
            kl_max=float(cfg.train.kl_max),
            # Gradient accumulation
            grad_accum_steps=int(cfg.train.grad_accum_steps),
            # Advantage normalization and PPO clipping
            adv_clip_percentile=float(cfg.train.adv_clip_percentile),
            ppo_clip_eps=float(cfg.train.ppo_clip_eps),
            ppo_clip_eps_upper=float(cfg.train.ppo_clip_eps_upper),
            logratio_clamp=float(cfg.train.logratio_clamp),
            # Importance sampling
            use_is=bool(cfg.train.use_is),
            is_ratio_max=float(cfg.train.is_ratio_max),
            # Data loading and filtering
            rollouts_per_problem=int(cfg.data.rollouts_per_problem),
            group_adv_sum_tolerance=float(cfg.train.group_adv_sum_tol),
        )

        # Prepare training seeds
        train_seed_start = int(cfg.data.train_seed_start)
        num_train_seeds = int(cfg.data.num_train_seeds)
        all_train_seeds = [train_seed_start + i for i in range(num_train_seeds)]

        # Initialize GRPO algorithm with config (CRITICAL: config must be passed!)
        adaptive_kl = bool(cfg.train.adaptive_kl) if hasattr(cfg.train, "adaptive_kl") else False
        algorithm = GRPOAlgorithm(adaptive_kl_enabled=adaptive_kl, config=train_cfg)

        # Log configuration
        logger.info("=" * 80)
        logger.info("OFFLINE TRAINER CONFIGURATION")
        logger.info("=" * 80)
        logger.info(
            "Training configuration",
            extra={
                "environment": cfg.data.environment,
                "model": train_id,
                "training_device": training_device,
                "vllm_server": base_url,
                "iterations": cfg.train.iterations,
                "problems_per_iteration": cfg.data.problems_per_iteration,
                "rollouts_per_problem": cfg.data.rollouts_per_problem,
                "batch_size": cfg.train.batch_size,
                "learning_rate": cfg.train.lr,
                "kl_coef": cfg.train.kl_coef,
                "adaptive_kl": adaptive_kl,
                "eval_enabled": cfg.eval.enabled,
                "eval_interval": cfg.eval.interval,
            },
        )
        logger.info("=" * 80)

        # Main training loop (inside server context)
        for it in range(int(cfg.train.iterations)):
            if monitor:
                monitor.set_block_context(block_number=it, window_number=it)

            logger.info(f"Starting iteration {it}/{cfg.train.iterations}")

            # Select seeds for this iteration
            iter_seeds = _select_iteration_seeds(
                all_train_seeds,
                it,
                int(cfg.data.problems_per_iteration),
            )

            # Generate rollouts
            logger.info("Generating rollouts", extra={"num_seeds": len(iter_seeds)})
            gen_batch_size = (
                int(cfg.data.generation_batch_size)
                if hasattr(cfg.data, "generation_batch_size")
                else 4
            )
            groups = await generator.generate_groups(
                iter_seeds,
                batch_size=gen_batch_size,
            )
            logger.info(
                "Rollouts generated",
                extra={
                    "num_groups": len(groups),
                    "total_rollouts": sum(len(g.rollouts) for g in groups),
                },
            )

            await _log_rollout_stats(monitor, groups)

            # Train one epoch
            logger.info("Starting training epoch")
            metrics = await algorithm.train_epoch(
                model=train_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                groups=groups,
                optimizer=optimizer,
                accelerator=accelerator,
                monitor=monitor,
                window=it,
                config=train_cfg,
            )

            await _log_training_metrics(monitor, metrics)

            logger.info(
                "Training epoch completed",
                extra={
                    "iteration": it,
                    "loss_total": metrics.get("loss_total"),
                    "loss_pg": metrics.get("loss_pg"),
                    "loss_kl": metrics.get("loss_kl"),
                    "reward_mean": metrics.get("reward_mean"),
                },
            )

            # Save training metrics
            (metrics_dir / f"train_iter_{it:03d}.json").write_text(json.dumps(metrics, indent=2))

            # Reload vLLM server with updated weights for next iteration's rollouts
            logger.info("Updating vLLM server with trained weights", extra={"iteration": it})
            updated_checkpoint = _save_checkpoint_for_vllm(
                model=train_model,
                tokenizer=tokenizer,
                output_dir=workdir,
                iteration=it,
            )
            await server_manager.reload_with_new_checkpoint(str(updated_checkpoint))

            # Prune old vLLM checkpoints to manage disk space
            keep_vllm = int(cfg.checkpoint.get("keep_last_vllm_k", 3))
            _prune_vllm_checkpoints(workdir, keep_vllm)

            # Periodic evaluation
            if bool(cfg.eval.enabled) and (it % int(cfg.eval.interval) == 0):
                await _run_evaluation(
                    cfg,
                    tokenizer,
                    train_model,
                    training_device,
                    model_name,
                    it,
                    monitor,
                    metrics_dir,
                    base_url,
                )

            # Save checkpoint
            if (it + 1) % int(cfg.checkpoint.save_interval) == 0:
                logger.info("Saving checkpoint", extra={"iteration": it})
                out = ckpt_dir / f"iter_{it:03d}"
                out.mkdir(parents=True, exist_ok=True)
                train_model.save_pretrained(out)
                tokenizer.save_pretrained(out)
                logger.info("Checkpoint saved", extra={"path": str(out)})

                _prune_old_checkpoints(ckpt_dir, int(cfg.checkpoint.keep_last_k))

    # Server context manager handles cleanup automatically
    logger.info("✓ Training complete, vLLM server shutting down...")
