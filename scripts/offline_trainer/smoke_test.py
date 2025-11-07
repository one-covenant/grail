from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

# Ensure repo root on sys.path before importing grail
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_REPO_ROOT))

import torch

from grail.environments.core import MultiTurnEnv
from grail.environments.loop import AgentEnvLoop
from grail.environments.sat_env import SATEnv
from grail.trainer.algorithms.grpo import GRPOAlgorithm
from grail.trainer.config import EvalConfig, TrainingConfig
from grail.trainer.eval_planner import EvaluationPlan
from grail.trainer.evaluator import EvaluatorService
from scripts.offline_trainer.offline_rollouts import OfflineRolloutGenerator, RolloutGenConfig
from tests.fixtures.fakes import DummyModel, DummyTokenizer, FakeBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ToyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_: Any
    ) -> Any:
        x = self.embed(input_ids)
        logits = self.lm_head(x)

        class Out:
            def __init__(self, logits: torch.Tensor) -> None:
                self.logits = logits

        return Out(logits)


def _make_generator(tokenizer: Any) -> OfflineRolloutGenerator:
    # Monkeypatch: replace server backend with FakeBackend for offline smoke
    import scripts.offline_trainer.offline_rollouts as orl

    class _FakeSGLServer(FakeBackend):
        def __init__(
            self, *, base_url: str, model_name: str, tokenizer: Any, timeout: float
        ) -> None:
            super().__init__(tokenizer=tokenizer)

    orl.SGLangServerBackend = _FakeSGLServer  # type: ignore[assignment]

    cfg = RolloutGenConfig(
        backend="sglang_server",
        base_url="http://127.0.0.1:0",
        batch_size=4,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        rollouts_per_problem=2,
    )
    return OfflineRolloutGenerator(tokenizer=tokenizer, config=cfg)


async def _train_epoch_smoke() -> dict[str, float]:
    logger.info("Starting training epoch smoke test")
    tokenizer = DummyTokenizer()
    generator = _make_generator(tokenizer)

    # Generate small set of groups
    seeds = [1001, 1002]
    logger.info("Generating rollout groups", extra={"num_seeds": len(seeds)})
    groups = generator.generate_groups(seeds)
    logger.info("Groups generated", extra={"num_groups": len(groups)})

    # Tiny toy models for train/ref
    model = ToyLM()
    ref_model = ToyLM()

    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device
    model = model.to(device)
    ref_model = ref_model.to(device)
    logger.info("Models loaded to device", extra={"device": str(device)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    algo = GRPOAlgorithm()
    train_cfg = TrainingConfig(lr=1e-3, batch_size=4)
    logger.info("Starting GRPO training epoch")
    metrics = await algo.train_epoch(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        groups=groups,
        optimizer=optimizer,
        accelerator=accelerator,
        monitor=None,
        window=0,
        config=train_cfg,
    )
    logger.info("Training epoch complete", extra={"metrics": metrics})
    return metrics


async def _eval_smoke() -> dict[str, float]:
    logger.info("Starting evaluation smoke test")
    model = DummyModel()
    tokenizer = DummyTokenizer()

    def env_factory() -> MultiTurnEnv:
        return SATEnv()

    eval_cfg = EvalConfig(batch_size=2, replicates=2, do_sample=True)
    svc = EvaluatorService(
        model=model, tokenizer=tokenizer, env_factory=env_factory, config=eval_cfg, device="cpu"
    )

    # Replace loop with FakeBackend
    svc._loop = AgentEnvLoop(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        do_sample=eval_cfg.do_sample,
        max_new_tokens=eval_cfg.max_new_tokens,
        temperature=eval_cfg.temperature,
        top_p=eval_cfg.top_p,
        gen_backend=FakeBackend(tokenizer=tokenizer),
    )

    plan = EvaluationPlan(ids=["1", "2"], replicates=2, cycle_index=0, seed_base=42)
    logger.info("Running evaluation cycle")
    metrics = await svc.run_cycle(plan)
    logger.info("Evaluation complete", extra={"metrics": metrics})
    return metrics


def main() -> None:
    logger.info("=" * 80)
    logger.info("Running offline trainer smoke tests")
    logger.info("=" * 80)
    
    logger.info("Running training smoke test")
    train_metrics = asyncio.run(_train_epoch_smoke())
    logger.info("Training smoke test complete", extra={"train_metrics": {k: float(v) for k, v in train_metrics.items()}})
    
    logger.info("Running evaluation smoke test")
    eval_metrics = asyncio.run(_eval_smoke())
    logger.info("Evaluation smoke test complete", extra={"eval_metrics": {k: float(v) for k, v in eval_metrics.items()}})
    
    logger.info("=" * 80)
    logger.info("All smoke tests passed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
