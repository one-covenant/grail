"""Comprehensive tests for offline GRPO trainer.

Tests rollout generation, training loop, evaluation, and integration
with CPU and GPU backends.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

# Ensure repo root on sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_REPO_ROOT))

from grail.environments.core import MultiTurnEnv
from grail.environments.loop import AgentEnvLoop
from grail.environments.sat_env import SATEnv
from grail.trainer.algorithms.grpo import GRPOAlgorithm
from grail.trainer.config import EvalConfig, TrainingConfig
from grail.trainer.eval_planner import EvaluationPlan
from grail.trainer.evaluator import EvaluatorService
from scripts.offline_trainer.offline_rollouts import OfflineRolloutGenerator, RolloutGenConfig
from tests.fixtures.fakes import DummyModel, DummyTokenizer, FakeBackend


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

    import scripts.offline_trainer.offline_rollouts as orl

    class _FakeSGLServer(FakeBackend):
        def __init__(
            self, *, base_url: str, model_name: str, tokenizer: Any, timeout: float
        ) -> None:
            super().__init__(tokenizer=tokenizer)

    # Monkeypatch server backend with FakeBackend
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


async def test_rollout_generator_produces_valid_groups() -> None:
    """Test that rollout generator produces valid GRPO groups."""
    tokenizer = DummyTokenizer()
    generator = _make_fake_generator(tokenizer)

    seeds = [1001, 1002]
    groups = generator.generate_groups(seeds)

    assert len(groups) == 2, "Should produce one group per seed"
    for group in groups:
        assert len(group.rollouts) == 2, (
            f"Each group should have 2 rollouts, got {len(group.rollouts)}"
        )
        assert group.is_valid(advantage_tolerance=1e-6, rollouts_per_problem=2)
        # Check advantage zero-sum property
        adv_sum = sum(r.advantage for r in group.rollouts)
        assert abs(adv_sum) < 1e-6, f"Advantages should sum to zero, got {adv_sum}"


async def test_train_epoch_cpu() -> None:
    """Test GRPO training epoch on CPU with toy models."""
    tokenizer = DummyTokenizer()
    generator = _make_fake_generator(tokenizer)

    seeds = [1001, 1002]
    groups = generator.generate_groups(seeds)

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

    assert "loss_total" in metrics
    assert "loss_pg" in metrics
    assert "loss_kl" in metrics
    assert "loss_entropy" in metrics
    assert (
        metrics["loss_total"] >= 0 or abs(metrics["loss_total"]) < 1e6
    )  # Allow reasonable negatives
    assert isinstance(metrics["kl_divergence"], float)
    assert isinstance(metrics["reward_mean"], float)


async def test_train_epoch_gpu() -> None:
    if not torch.cuda.is_available():
        print("  SKIP: GPU not available")
        return
    """Test GRPO training epoch on GPU with toy models."""
    tokenizer = DummyTokenizer()
    generator = _make_fake_generator(tokenizer)

    seeds = [1001, 1002, 1003, 1004]
    groups = generator.generate_groups(seeds)

    model = ToyLM(vocab_size=512, hidden_size=128)
    ref_model = ToyLM(vocab_size=512, hidden_size=128)

    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    assert str(device).startswith("cuda"), "Should run on GPU"

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
    assert isinstance(metrics["kl_divergence"], float)
    assert isinstance(metrics["reward_mean"], float)
    # Verify GPU was used
    assert next(model.parameters()).device.type == "cuda"


async def test_evaluator_smoke() -> None:
    """Test evaluator service with fake backend."""
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
    metrics = await svc.run_cycle(plan)

    assert "pass@1" in metrics
    assert "mean@1" in metrics
    assert isinstance(metrics["pass@1"], float)
    assert isinstance(metrics["mean@1"], float)


async def test_evaluator_gpu() -> None:
    if not torch.cuda.is_available():
        print("  SKIP: GPU not available")
        return
    """Test evaluator service on GPU."""
    tokenizer = DummyTokenizer()
    model = ToyLM()
    model = model.to("cuda")

    def env_factory() -> MultiTurnEnv:
        return SATEnv()

    eval_cfg = EvalConfig(batch_size=4, replicates=3, do_sample=True)
    svc = EvaluatorService(
        model=model, tokenizer=tokenizer, env_factory=env_factory, config=eval_cfg, device="cuda"
    )

    # Replace loop with FakeBackend
    svc._loop = AgentEnvLoop(
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        do_sample=eval_cfg.do_sample,
        max_new_tokens=eval_cfg.max_new_tokens,
        temperature=eval_cfg.temperature,
        top_p=eval_cfg.top_p,
        gen_backend=FakeBackend(tokenizer=tokenizer),
    )

    plan = EvaluationPlan(ids=["1", "2", "3"], replicates=3, cycle_index=0, seed_base=99)
    metrics = await svc.run_cycle(plan)

    assert "pass@1" in metrics
    assert isinstance(metrics["pass@1"], float)
    # Verify GPU was used if HF backend is used (but we're using FakeBackend so this is mainly structure test)


def test_rollout_generator_advantage_computation() -> None:
    """Test that advantage computation is zero-mean and variance-normalized."""
    from scripts.offline_trainer.offline_rollouts import OfflineRolloutGenerator

    rewards = [1.0, 2.0, 3.0, 4.0]
    advantages = OfflineRolloutGenerator._compute_advantages(rewards)

    assert len(advantages) == len(rewards)
    # Check zero-mean
    mean_adv = sum(advantages) / len(advantages)
    assert abs(mean_adv) < 1e-6, f"Advantages should be zero-mean, got {mean_adv}"
    # Check variance normalization (std should be ~1)
    var_adv = sum(a * a for a in advantages) / len(advantages)
    std_adv = (var_adv) ** 0.5
    assert abs(std_adv - 1.0) < 1e-5, f"Advantages should have unit variance, got std={std_adv}"


def test_rollout_groups_are_valid() -> None:
    """Test that generated groups pass validation."""
    tokenizer = DummyTokenizer()
    generator = _make_fake_generator(tokenizer)

    seeds = [2001, 2002, 2003]
    groups = generator.generate_groups(seeds)

    assert len(groups) == 3
    for group in groups:
        assert group.is_valid(advantage_tolerance=1e-5, rollouts_per_problem=2)
        assert group.group_id in [str(s) for s in seeds]
        for rollout in group.rollouts:
            assert rollout.rollout_group == group.group_id
            assert rollout.nonce >= 0
            assert isinstance(rollout.reward, float)
            assert isinstance(rollout.advantage, float)
