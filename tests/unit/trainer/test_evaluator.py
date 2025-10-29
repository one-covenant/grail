from __future__ import annotations

from typing import Any

import pytest

from grail.environments.core import MultiTurnEnv
from grail.trainer.config import EvalConfig
from grail.trainer.eval_planner import EvaluationPlan
from grail.trainer.evaluator import EvaluatorService
from tests.fixtures.fakes import DummyEnv, DummyModel, DummyTokenizer


@pytest.mark.asyncio
async def test_evaluator_calls_loop_render_and_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()

    def env_factory() -> MultiTurnEnv:
        return DummyEnv()

    cfg = EvalConfig(batch_size=2, replicates=1)
    svc = EvaluatorService(
        model=model, tokenizer=tokenizer, env_factory=env_factory, config=cfg, device="cpu"
    )

    calls: dict[str, Any] = {}

    def spy_render(batch: list[list[dict[str, str]]]) -> list[list[int]]:
        calls["render_batch"] = batch
        # Two prompts corresponding to two task ids
        return [[10], [20]]

    def spy_generate(
        prompt_ids_batch: list[list[int]],
        *,
        seeds: list[int] | None = None,
        trim_right_padding: bool = False,
    ) -> list[tuple[list[int], int]]:
        calls["generate_args"] = {
            "ids": prompt_ids_batch,
            "seeds": seeds,
            "trim": trim_right_padding,
        }
        # Return sequences and prompt lengths
        return [([10, 99], 1), ([20, 98], 1)]

    monkeypatch.setattr(svc._loop, "render_prompt_ids_batch", spy_render)
    monkeypatch.setattr(svc._loop, "generate_from_prompt_ids_batch", spy_generate)

    plan = EvaluationPlan(ids=["a", "b"], replicates=1, cycle_index=0, seed_base=123)
    _metrics = await svc.run_cycle(plan)

    # Assert delegation
    assert isinstance(calls.get("render_batch"), list)
    assert calls["generate_args"]["ids"] == [[10], [20]]
    assert calls["generate_args"]["trim"] is True


@pytest.mark.asyncio
async def test_evaluator_expands_per_replicate_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()

    def env_factory() -> MultiTurnEnv:
        return DummyEnv()

    cfg = EvalConfig(batch_size=2, replicates=3)
    svc = EvaluatorService(
        model=model, tokenizer=tokenizer, env_factory=env_factory, config=cfg, device="cpu"
    )

    captured: dict[str, Any] = {}

    def spy_render(batch: list[list[dict[str, str]]]) -> list[list[int]]:
        # Mirror number of inputs
        return [[42] for _ in batch]

    def spy_generate(
        prompt_ids_batch: list[list[int]],
        *,
        seeds: list[int] | None = None,
        trim_right_padding: bool = False,
    ) -> list[tuple[list[int], int]]:
        nonlocal captured
        captured = {"count": len(prompt_ids_batch), "seeds_len": 0 if seeds is None else len(seeds)}
        return [([7, 8], 1) for _ in prompt_ids_batch]

    monkeypatch.setattr(svc._loop, "render_prompt_ids_batch", spy_render)
    monkeypatch.setattr(svc._loop, "generate_from_prompt_ids_batch", spy_generate)

    plan = EvaluationPlan(ids=["t1", "t2"], replicates=3, cycle_index=0, seed_base=999)
    _ = await svc.run_cycle(plan)

    # Expect 2 tasks * 3 replicates expanded
    assert captured["count"] == 6
    assert captured["seeds_len"] == 6
