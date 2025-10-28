from __future__ import annotations

import pytest

from grail.environments.core import MultiTurnEnv
from grail.environments.loop import AgentEnvLoop
from grail.trainer.config import EvalConfig
from grail.trainer.eval_planner import EvaluationPlan
from grail.trainer.evaluator import EvaluatorService
from tests.fixtures.fakes import DummyEnv, DummyModel, DummyTokenizer, FakeBackend


@pytest.mark.asyncio
async def test_evaluator_end_to_end_with_fake_backend() -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()

    def env_factory() -> MultiTurnEnv:
        return DummyEnv()

    cfg = EvalConfig(batch_size=2, replicates=2, do_sample=True)
    svc = EvaluatorService(
        model=model, tokenizer=tokenizer, env_factory=env_factory, config=cfg, device="cpu"
    )

    # Replace loop with one using FakeBackend
    svc._loop = AgentEnvLoop(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        batch_size=cfg.batch_size,
        do_sample=cfg.do_sample,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        gen_backend=FakeBackend(tokenizer=tokenizer),
    )

    plan = EvaluationPlan(ids=["x", "y"], replicates=2, cycle_index=0, seed_base=42)

    metrics1 = await svc.run_cycle(plan)
    metrics2 = await svc.run_cycle(plan)

    # Deterministic given same plan/seeds/backends
    assert metrics1 == metrics2
    # Basic sanity: expected keys exist
    assert any(k.startswith("pass@") for k in metrics1.keys())
