from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Ensure repo root is importable when running as a subpackage tool
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_REPO_ROOT))

# Load environment variables from parent grail/.env for WandB API key and other configs
from dotenv import load_dotenv
_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
    logger_temp = logging.getLogger(__name__)
    logger_temp.debug(f"Loaded environment from {_env_file}")

from grail.model.provider import get_model, get_tokenizer
from grail.monitoring import MonitoringConfig, get_monitoring_manager, initialize_monitoring
from grail.shared.chat_templates import build_qwen_chat_template
from grail.shared.constants import TRAINER_USE_FLASH_ATTENTION
from grail.shared.prompt_constants import REASONING_START, SYSTEM_PROMPT
from grail.trainer.algorithms.grpo import GRPOAlgorithm
from grail.trainer.config import EvalConfig, TrainingConfig
from grail.trainer.eval_planner import EvaluationPlan
from grail.trainer.evaluator import EvaluatorService
from scripts.offline_trainer.offline_rollouts import OfflineRolloutGenerator, RolloutGenConfig

logger = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def _device_from_cfg(name: str) -> str:
    name_l = (name or "auto").lower()
    if name_l == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name_l in ("cuda", "cpu"):
        return name_l
    return "cpu"


def _select_iteration_seeds(all_seeds: list[int], iteration: int, per_iter: int) -> list[int]:
    n = len(all_seeds)
    if n == 0:
        return []
    start = (iteration * per_iter) % n
    end = start + per_iter
    if end <= n:
        return all_seeds[start:end]
    # wrap
    return all_seeds[start:] + all_seeds[: end - n]


async def _run(cfg: DictConfig) -> None:
    _set_global_seed(int(cfg.seed))

    # Initialize monitoring with WandB if enabled
    monitor = None
    if bool(cfg.logging.wandb.enabled):
        logger.info("Initializing monitoring system (WandB enabled)")
        monitoring_config = MonitoringConfig.for_training()
        monitoring_config.update(
            {
                "tags": ["offline_trainer", "grpo", str(cfg.data.environment)],
                "hyperparameters": {
                    "model": str(cfg.model.train_id),
                    "environment": str(cfg.data.environment),
                    "iterations": int(cfg.train.iterations),
                    "batch_size": int(cfg.train.batch_size),
                    "learning_rate": float(cfg.train.lr),
                    "kl_coef": float(cfg.train.kl_coef),
                    "entropy_coef": float(cfg.train.entropy_coef),
                    "rollouts_per_problem": int(cfg.data.rollouts_per_problem),
                    "problems_per_iteration": int(cfg.data.problems_per_iteration),
                },
            }
        )
        # Don't pass backend_type twice - it's already in monitoring_config
        backend_type = monitoring_config.pop("backend_type", "wandb")
        initialize_monitoring(backend_type=backend_type, **monitoring_config)
        monitor = get_monitoring_manager()
    else:
        logger.info("WandB disabled - using in-memory metrics only")

    # Build tokenizer with Qwen chat template for consistency
    chat_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)

    train_id = str(cfg.model.train_id)
    ref_id = str(cfg.model.ref_id)

    # GPU strategy configuration
    gpu_strategy = str(cfg.gpu.strategy).lower()
    is_multi_gpu = gpu_strategy == "multi"

    # Determine training device based on strategy
    if is_multi_gpu:
        training_gpu = int(cfg.gpu.training_gpu)
        training_device = f"cuda:{training_gpu}"
        vllm_gpu = int(cfg.gpu.vllm_gpu)
        vllm_memory_util = float(cfg.gpu.vllm_gpu_memory_utilization)
        logger.info(
            "Multi-GPU mode enabled",
            extra={"training_gpu": training_gpu, "vllm_gpu": vllm_gpu},
        )
    else:
        training_device = "cuda" if torch.cuda.is_available() else "cpu"
        vllm_gpu = None
        vllm_memory_util = float(cfg.gpu.vllm_gpu_memory_utilization_single)
        logger.info("Single-GPU mode enabled", extra={"device": training_device})

    tokenizer = get_tokenizer(train_id, chat_template=chat_template)

    # Create accelerator with specific device
    accelerator = Accelerator(mixed_precision="no")

    # Load models to accelerator's device (not config device, to avoid mismatch)
    # Enable Flash Attention for training model if configured
    train_model = get_model(train_id, device=training_device, eval_mode=False, use_flash_attention=TRAINER_USE_FLASH_ATTENTION)

    # Only load reference model if KL divergence is enabled (kl_coef > 0)
    kl_enabled = float(cfg.train.kl_coef) > 0.0
    if kl_enabled:
        ref_model = get_model(ref_id, device=training_device, eval_mode=True)
        logger.info(
            "Loaded train model and ref model",
            extra={"device": training_device, "kl_coef": cfg.train.kl_coef},
        )
    else:
        ref_model = None
        logger.info(
            "Loaded train model only (KL disabled)",
            extra={"device": training_device, "kl_coef": cfg.train.kl_coef},
        )

    # Optimizer after model placement
    optimizer = torch.optim.AdamW(
        train_model.parameters(), lr=float(cfg.train.lr), weight_decay=0.1
    )

    # Rollout generator (server-backed)
    gen_cfg = RolloutGenConfig(
        backend=str(cfg.generation.backend),
        base_url=str(cfg.generation.base_url),
        batch_size=int(cfg.generation.batch_size),
        max_new_tokens=int(cfg.generation.max_new_tokens),
        temperature=float(cfg.generation.temperature),
        top_p=float(cfg.generation.top_p),
        top_k=int(cfg.generation.top_k) if cfg.generation.top_k is not None else None,
        repetition_penalty=float(cfg.generation.repetition_penalty)
        if cfg.generation.repetition_penalty is not None
        else None,
        rollouts_per_problem=int(cfg.data.rollouts_per_problem),
        environment=str(cfg.data.environment),
    )
    generator = OfflineRolloutGenerator(tokenizer=tokenizer, config=gen_cfg)

    # Training config mirrors shared constants; override a subset from cfg
    # Note: grad_accum_steps is a constant in grail/shared/constants.py, not a TrainingConfig field
    train_cfg = TrainingConfig(
        lr=float(cfg.train.lr),
        batch_size=int(cfg.train.batch_size),
        grad_clip=float(cfg.train.grad_clip),
        kl_coef=float(cfg.train.kl_coef),
        entropy_coef=float(cfg.train.entropy_coef),
        warmup_steps=int(cfg.train.warmup_steps),
        max_length=int(cfg.train.max_length),
        group_adv_sum_tolerance=float(cfg.train.group_adv_sum_tol),
    )

    # Seeds for problem set
    train_seed_start = int(cfg.data.train_seed_start)
    num_train_seeds = int(cfg.data.num_train_seeds)
    all_train_seeds = [train_seed_start + i for i in range(num_train_seeds)]

    # Initialize GRPO algorithm with adaptive KL if enabled
    adaptive_kl = bool(cfg.train.adaptive_kl) if hasattr(cfg.train, "adaptive_kl") else False
    algorithm = GRPOAlgorithm(adaptive_kl_enabled=adaptive_kl)

    logger.info("=" * 80)
    logger.info("OFFLINE TRAINER CONFIGURATION")
    logger.info("=" * 80)
    logger.info(
        "Training configuration",
        extra={
            "environment": cfg.data.environment,
            "model": train_id,
            "training_device": training_device,
            "vllm_gpu": vllm_gpu if is_multi_gpu else "shared",
            "iterations": cfg.train.iterations,
            "problems_per_iteration": cfg.data.problems_per_iteration,
            "rollouts_per_problem": cfg.data.rollouts_per_problem,
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.train.lr,
            "grad_accum_steps": cfg.train.grad_accum_steps,
            "kl_coef": cfg.train.kl_coef,
            "adaptive_kl": adaptive_kl,
            "entropy_coef": cfg.train.entropy_coef,
            "eval_enabled": cfg.eval.enabled,
            "eval_interval": cfg.eval.interval,
        },
    )
    logger.info("=" * 80)

    workdir = Path(HydraConfig.get().runtime.output_dir)
    metrics_dir = workdir / "metrics"
    ckpt_dir = workdir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for it in range(int(cfg.train.iterations)):
        # Set monitoring context for this iteration
        if monitor:
            monitor.set_block_context(block_number=it, window_number=it)

        logger.info(f"Starting iteration {it}/{cfg.train.iterations}")

        iter_seeds = _select_iteration_seeds(
            all_train_seeds, it, int(cfg.data.problems_per_iteration)
        )

        # Generate fresh GRPO groups from server completions
        logger.info("Generating rollouts", extra={"num_seeds": len(iter_seeds)})
        groups = await generator.generate_groups(
            iter_seeds,
            batch_size=cfg.data.batch_size if hasattr(cfg.data, "batch_size") else 1,
        )
        logger.info(
            "Rollouts generated",
            extra={
                "num_groups": len(groups),
                "total_rollouts": sum(len(g.rollouts) for g in groups),
            },
        )

        # Log rollout statistics to WandB
        if monitor:
            await monitor.log_gauge("training/num_groups", len(groups))
            await monitor.log_gauge("training/num_rollouts", sum(len(g.rollouts) for g in groups))

            # Compute rollout statistics
            all_rewards = [r.reward for g in groups for r in g.rollouts]
            all_successes = [r.success for g in groups for r in g.rollouts]
            if all_rewards:
                await monitor.log_gauge("training/rollout_reward_mean", float(np.mean(all_rewards)))
                await monitor.log_gauge("training/rollout_reward_std", float(np.std(all_rewards)))
                await monitor.log_gauge("training/rollout_success_rate", float(np.mean(all_successes)))

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

        # Log training metrics to WandB
        if monitor:
            for metric_name, metric_value in metrics.items():
                await monitor.log_gauge(f"training/{metric_name}", float(metric_value))

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

        # Persist metrics snapshot
        (metrics_dir / f"train_iter_{it:03d}.json").write_text(json.dumps(metrics, indent=2))

        # Periodic evaluation using the same vLLM server
        if bool(cfg.eval.enabled) and (it % int(cfg.eval.interval) == 0):
            logger.info("Starting evaluation", extra={"iteration": it})
            eval_backend = str(cfg.eval.backend).lower()
            eval_base_url = str(cfg.eval.base_url)

            # Map EvalConfig to reuse existing EvaluatorService
            eval_cfg = EvalConfig(
                batch_size=int(cfg.eval.batch_size),
                do_sample=bool(cfg.eval.do_sample),
                max_new_tokens=int(cfg.eval.max_new_tokens),
                temperature=float(cfg.eval.temperature),
                top_p=float(cfg.eval.top_p),
                backend=eval_backend,
            )

            def _env_factory() -> Any:
                env_type = str(cfg.data.environment).lower()
                if env_type == "gsm8k":
                    from grail.environments.gsm8k_env import GSM8KEnv as _GSM8KEnv

                    return _GSM8KEnv()
                else:
                    from grail.environments.sat_env import SATEnv as _SATEnv

                    return _SATEnv()

            # When using server backend, pass model=None and let evaluator create the backend
            # For HF backend, pass the actual model
            use_server = eval_backend in ("vllm", "sglang")

            evaluator = EvaluatorService(
                model=None if use_server else train_model,
                tokenizer=tokenizer,
                env_factory=_env_factory,
                config=eval_cfg,
                monitor=monitor,
                device=training_device,
                server_base_url=eval_base_url if use_server else None,
                server_model_name=train_id if use_server else None,
            )

            # Build evaluation IDs
            eval_ids = [
                str(i)
                for i in range(
                    int(cfg.eval.id_seed_start), int(cfg.eval.id_seed_start) + int(cfg.eval.num_ids)
                )
            ]
            if cfg.eval.ids:
                eval_ids = [str(x) for x in cfg.eval.ids]
            plan = EvaluationPlan(
                ids=eval_ids,
                replicates=int(cfg.eval.replicates),
                cycle_index=it,
                seed_base=int(cfg.eval.seed_base),
            )

            try:
                eval_metrics = await evaluator.run_cycle(plan)

                # Log evaluation metrics to WandB
                if monitor:
                    for metric_name, metric_value in eval_metrics.items():
                        await monitor.log_gauge(f"eval/{metric_name}", float(metric_value))

                logger.info(
                    "Evaluation complete",
                    extra={
                        "iteration": it,
                        "pass@1": eval_metrics.get("pass@1"),
                        "mean@1": eval_metrics.get("mean@1"),
                    },
                )

                (metrics_dir / f"eval_iter_{it:03d}.json").write_text(
                    json.dumps(eval_metrics, indent=2)
                )
            finally:
                # Always cleanup evaluator
                evaluator.shutdown()

        # Save checkpoint and prune older ones
        if (it + 1) % int(cfg.checkpoint.save_interval) == 0:
            logger.info("Saving checkpoint", extra={"iteration": it})
            out = ckpt_dir / f"iter_{it:03d}"
            out.mkdir(parents=True, exist_ok=True)
            train_model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            logger.info("Checkpoint saved", extra={"path": str(out)})

            # Keep last K
            keep_k = int(cfg.checkpoint.keep_last_k)
            existing = sorted([p for p in ckpt_dir.iterdir() if p.is_dir()])
            if len(existing) > keep_k:
                logger.info("Pruning old checkpoints", extra={"keep": keep_k, "existing": len(existing)})
                for old in existing[:-keep_k]:
                    try:
                        for sub in old.glob("**/*"):
                            if sub.is_file():
                                sub.unlink(missing_ok=True)
                        for subdir in sorted(old.glob("**/*"), reverse=True):
                            if subdir.is_dir():
                                subdir.rmdir()
                        old.rmdir()
                        logger.info("Deleted old checkpoint", extra={"path": str(old)})
                    except Exception as e:
                        logger.warning("Failed to delete checkpoint", extra={"path": str(old), "error": str(e)})


@hydra.main(config_path="conf", config_name="offline_grpo", version_base="1.3")
def main(cfg: DictConfig) -> None:
  try:
      asyncio.run(_run(cfg))
  finally:
      # Cleanup monitoring on exit if enabled
      logger.info("Shutting down monitoring")
      try:
          monitor = get_monitoring_manager()
          if monitor:
              asyncio.run(monitor.shutdown())
      except Exception as e:
          logger.warning("Error during monitoring shutdown", extra={"error": str(e)})


if __name__ == "__main__":
    main()
