from __future__ import annotations

import asyncio
import json
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

from grail.model.provider import get_model, get_tokenizer
from grail.shared.chat_templates import build_qwen_chat_template
from grail.shared.constants import TRAINER_USE_FLASH_ATTENTION
from grail.shared.prompt_constants import REASONING_START, SYSTEM_PROMPT
from grail.trainer.algorithms.grpo import GRPOAlgorithm
from grail.trainer.config import EvalConfig, TrainingConfig
from grail.trainer.eval_planner import EvaluationPlan
from grail.trainer.evaluator import EvaluatorService
from scripts.offline_trainer.offline_rollouts import OfflineRolloutGenerator, RolloutGenConfig


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

    # Build tokenizer with Qwen chat template for consistency
    chat_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)

    train_id = str(cfg.model.train_id)
    ref_id = str(cfg.model.ref_id)
    device = _device_from_cfg(str(cfg.model.device))

    tokenizer = get_tokenizer(train_id, chat_template=chat_template)

    # Create accelerator first to determine actual device
    accelerator = Accelerator(mixed_precision="no")
    actual_device = str(accelerator.device)

    # Load models to accelerator's device (not config device, to avoid mismatch)
    # Enable Flash Attention for training model if configured
    train_model = get_model(train_id, device=actual_device, eval_mode=False, use_flash_attention=TRAINER_USE_FLASH_ATTENTION)
    ref_model = get_model(ref_id, device=actual_device, eval_mode=True)

    print(f"Loaded models to {actual_device} (config requested: {device})")

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
    )
    generator = OfflineRolloutGenerator(tokenizer=tokenizer, config=gen_cfg)

    # Training config mirrors shared constants; override a subset from cfg
    train_cfg = TrainingConfig(
        lr=float(cfg.train.lr),
        batch_size=int(cfg.train.batch_size),
        grad_clip=float(cfg.train.grad_clip),
        kl_coef=float(cfg.train.kl_coef),
        entropy_coef=float(cfg.train.entropy_coef),
    )

    # Seeds for problem set
    train_seed_start = int(cfg.data.train_seed_start)
    num_train_seeds = int(cfg.data.num_train_seeds)
    all_train_seeds = [train_seed_start + i for i in range(num_train_seeds)]

    algorithm = GRPOAlgorithm()

    workdir = Path(HydraConfig.get().runtime.output_dir)
    metrics_dir = workdir / "metrics"
    ckpt_dir = workdir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for it in range(int(cfg.train.iterations)):
        iter_seeds = _select_iteration_seeds(
            all_train_seeds, it, int(cfg.data.problems_per_iteration)
        )

        # Generate fresh GRPO groups from server completions
        groups = await generator.generate_groups(
            iter_seeds,
            batch_size=cfg.data.batch_size if hasattr(cfg.data, "batch_size") else 1,
        )

        # Train one epoch
        metrics = await algorithm.train_epoch(
            model=train_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            groups=groups,
            optimizer=optimizer,
            accelerator=accelerator,
            monitor=None,
            window=it,
            config=train_cfg,
        )

        # Persist metrics snapshot
        (metrics_dir / f"train_iter_{it:03d}.json").write_text(json.dumps(metrics, indent=2))

        # Periodic evaluation
        if bool(cfg.eval.enabled) and (it % int(cfg.eval.interval) == 0):
            eval_backend = str(cfg.eval.backend)
            eval_base_url = str(cfg.eval.base_url)
            # Map EvalConfig to reuse existing EvaluatorService
            eval_cfg = EvalConfig(
                batch_size=int(cfg.eval.batch_size),
                do_sample=bool(cfg.eval.do_sample),
                max_new_tokens=int(cfg.eval.max_new_tokens),
                temperature=float(cfg.eval.temperature),
                top_p=float(cfg.eval.top_p),
            )
            # Monkey-patch backend fields present in EvalConfig (without editing core dataclass)
            # These attributes are referenced in EvaluatorService
            eval_cfg.backend = eval_backend
            # Reuse sglang_host/port fields to pass base_url (Evaluator uses them for sglang)
            # We encode base_url into host:port in a simple way for compatibility
            # If eval_base_url is "http://host:port", we split
            try:
                from urllib.parse import urlparse

                parsed = urlparse(eval_base_url)
                host = parsed.hostname or "127.0.0.1"
                port = parsed.port or 30000
            except Exception:
                host = "127.0.0.1"
                port = 30000
            eval_cfg.sglang_host = host
            eval_cfg.sglang_port = port

            def _env_factory() -> Any:
                from grail.environments.sat_env import SATEnv as _SATEnv

                return _SATEnv()

            evaluator = EvaluatorService(
                model=train_model,
                tokenizer=tokenizer,
                env_factory=_env_factory,
                config=eval_cfg,
                monitor=None,
                device=actual_device,
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
            eval_metrics = await evaluator.run_cycle(plan)
            (metrics_dir / f"eval_iter_{it:03d}.json").write_text(
                json.dumps(eval_metrics, indent=2)
            )

        # Save checkpoint and prune older ones
        if (it + 1) % int(cfg.checkpoint.save_interval) == 0:
            out = ckpt_dir / f"iter_{it:03d}"
            out.mkdir(parents=True, exist_ok=True)
            train_model.save_pretrained(out)
            tokenizer.save_pretrained(out)

            # Keep last K
            keep_k = int(cfg.checkpoint.keep_last_k)
            existing = sorted([p for p in ckpt_dir.iterdir() if p.is_dir()])
            if len(existing) > keep_k:
                for old in existing[:-keep_k]:
                    try:
                        for sub in old.glob("**/*"):
                            if sub.is_file():
                                sub.unlink(missing_ok=True)
                        for subdir in sorted(old.glob("**/*"), reverse=True):
                            if subdir.is_dir():
                                subdir.rmdir()
                        old.rmdir()
                    except Exception:
                        pass


@hydra.main(config_path="conf", config_name="offline_grpo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
