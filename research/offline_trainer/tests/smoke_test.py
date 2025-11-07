"""Quick smoke test to verify basic functionality."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure repo root and src on sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_SRC_DIR = _THIS_FILE.parents[1] / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch  # noqa: E402, I001
from grail.environments.core import MultiTurnEnv  # noqa: E402
from grail.environments.loop import AgentEnvLoop  # noqa: E402
from grail.environments.sat_env import SATEnv  # noqa: E402
from grail.trainer.algorithms.grpo import GRPOAlgorithm  # noqa: E402
from grail.trainer.config import EvalConfig, TrainingConfig  # noqa: E402
from grail.trainer.eval_planner import EvaluationPlan  # noqa: E402
from grail.trainer.evaluator import EvaluatorService  # noqa: E402
from grail_offline.data.offline_rollouts import (  # noqa: E402
    OfflineRolloutGenerator,
    RolloutGenConfig,
)
from tests.fixtures.fakes import (  # noqa: E402
    DummyModel,
    DummyTokenizer,
    FakeBackend,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ToyLM(torch.nn.Module):
    """Minimal language model for testing."""

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
    """Create rollout generator with FakeBackend."""
    import grail_offline.data.offline_rollouts as orl

    class _FakeSGLServer(FakeBackend):
        def __init__(
            self, *, base_url: str, model_name: str, tokenizer: Any, timeout: float, **_: Any
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
    """Quick training epoch smoke test."""
    logger.info("Starting training epoch smoke test")
    tokenizer = DummyTokenizer()
    generator = _make_generator(tokenizer)

    seeds = [1001, 1002]
    groups = await generator.generate_groups(seeds)

    model = ToyLM()
    ref_model = ToyLM()

    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device
    model = model.to(device)
    ref_model = ref_model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    algo = GRPOAlgorithm()
    train_cfg = TrainingConfig(lr=1e-3, batch_size=4)

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
    return metrics


async def _eval_smoke() -> dict[str, float]:
    """Quick evaluation smoke test."""
    logger.info("Starting evaluation smoke test")
    model = DummyModel()
    tokenizer = DummyTokenizer()

    def env_factory() -> MultiTurnEnv:
        return SATEnv()

    eval_cfg = EvalConfig(batch_size=2, do_sample=True)
    svc = EvaluatorService(
        model=model, tokenizer=tokenizer, env_factory=env_factory, config=eval_cfg, device="cpu"
    )

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
    metrics = await svc.run_cycle(plan)
    return metrics


def main() -> None:
    """Run smoke tests."""
    logger.info("=" * 80)
    logger.info("Running offline trainer smoke tests")
    logger.info("=" * 80)

    asyncio.run(_train_epoch_smoke())
    logger.info("✓ Training smoke test passed")

    asyncio.run(_eval_smoke())
    logger.info("✓ Evaluation smoke test passed")

    logger.info("=" * 80)
    logger.info("All smoke tests passed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
