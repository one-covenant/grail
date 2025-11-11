"""Concise tests for offline GRPO trainer focusing on main functionality."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Ensure repo root and src on sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_SRC_DIR = _THIS_FILE.parents[1] / "src"
if str(_REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_REPO_ROOT))
if _SRC_DIR.exists() and str(_SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_SRC_DIR))

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


def _make_fake_generator(tokenizer: Any) -> OfflineRolloutGenerator:
    """Create rollout generator with FakeBackend for testing."""
    import grail_offline.data.offline_rollouts as orl

    class _FakeSGLServer(FakeBackend):
        def __init__(
            self, *, base_url: str, model_name: str, tokenizer: Any, timeout: float, **_: Any
        ) -> None:
            super().__init__(tokenizer=tokenizer)

    original_sgl = orl.SGLangServerBackend
    orl.SGLangServerBackend = _FakeSGLServer  # type: ignore[assignment]

    try:
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
    finally:
        orl.SGLangServerBackend = original_sgl


async def test_rollout_generation() -> None:
    """Test that rollout generator produces valid GRPO groups."""
    logger.info("Testing rollout generation")
    tokenizer = DummyTokenizer()
    generator = _make_fake_generator(tokenizer)

    seeds = [1001, 1002]
    groups = await generator.generate_groups(seeds)

    assert len(groups) == 2, "Should produce one group per seed"
    for group in groups:
        assert len(group.rollouts) == 2, "Each group should have 2 rollouts"
        # Verify advantages sum to zero (GRPO requirement)
        adv_sum = sum(r.advantage for r in group.rollouts)
        assert abs(adv_sum) < 1e-6, f"Advantages should sum to zero, got {adv_sum}"
    logger.info("✓ Rollout generation works")


async def test_training_epoch() -> None:
    """Test GRPO training epoch completes successfully."""
    logger.info("Testing training epoch")
    tokenizer = DummyTokenizer()
    generator = _make_fake_generator(tokenizer)

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

    # Verify expected metrics are present
    assert "loss_total" in metrics
    assert "loss_pg" in metrics
    assert "reward_mean" in metrics
    logger.info("✓ Training epoch works")


async def test_evaluation() -> None:
    """Test evaluation service runs successfully."""
    logger.info("Testing evaluation")
    model = DummyModel()
    tokenizer = DummyTokenizer()

    def env_factory() -> MultiTurnEnv:
        return SATEnv()

    eval_cfg = EvalConfig(batch_size=2, do_sample=True)
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
    metrics = await svc.run_cycle(plan)

    # Verify expected metrics are present
    assert "pass@1" in metrics
    assert "mean@1" in metrics
    logger.info("✓ Evaluation works")


async def test_training_with_gpu() -> None:
    """Test training epoch on GPU if available."""
    if not torch.cuda.is_available():
        logger.info("SKIP: GPU not available")
        return

    logger.info("Testing training epoch on GPU")
    tokenizer = DummyTokenizer()
    generator = _make_fake_generator(tokenizer)

    seeds = [1001, 1002]
    groups = await generator.generate_groups(seeds)

    model = ToyLM(vocab_size=512, hidden_size=128)
    ref_model = ToyLM(vocab_size=512, hidden_size=128)

    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    model = model.to(device)
    ref_model = ref_model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    algo = GRPOAlgorithm()
    train_cfg = TrainingConfig(lr=1e-3, batch_size=8)

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

    assert "loss_total" in metrics
    assert next(model.parameters()).device.type == "cuda"
    logger.info("✓ GPU training works")
