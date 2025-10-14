"""Single-turn GSM8K environment using HF datasets backend.

This environment serves a single math word problem per episode and computes
reward by exact-match against the dataset answer (with light normalization).
"""

from __future__ import annotations

import re
from typing import Any

from .base import Parser
from .core import ChatMessage, Observation, SingleTurnEnv
from .providers import GSM8KTaskSource, TaskSpec


class GSM8KAnswerParser(Parser):
    """Extract final numeric/free-form answer from completion text.

    Heuristics:
      - prefer the last '#### answer' style if present
      - otherwise, take the last number-like token
      - fallback to stripped tail line
    """

    _HASH_PATTERN = re.compile(r"####\s*(?P<ans>.+)")
    _NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")

    def parse(self, completion: str, context: Any) -> str:
        text = completion or ""
        match = None
        for m in self._HASH_PATTERN.finditer(text):
            match = m
        if match is not None:
            return match.group("ans").strip()
        nums = list(self._NUMBER_PATTERN.finditer(text))
        if nums:
            return nums[-1].group(0).replace(",", "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else ""


def _normalize_answer(s: str) -> str:
    return re.sub(r"[\s\.]+$", "", s.strip().lower())


class GSM8KEnv(SingleTurnEnv):
    """GSM8K single-turn environment inheriting template-method base."""

    def __init__(
        self,
        *,
        task_source: GSM8KTaskSource | None = None,
        parser: Parser | None = None,
    ) -> None:
        super().__init__()
        self._source = task_source or GSM8KTaskSource()
        self._parser = parser or GSM8KAnswerParser()
        self._task: TaskSpec | None = None

    def _do_reset(self, *, task_id: str | None = None, seed: int | None = None) -> Observation:
        self._task = self._source.next(seed=seed, task_id=task_id)
        q = self._task.payload["question"]

        obs = Observation(
            messages=[ChatMessage(role="user", content=q)],
            available_tools=[],
            turn_index=0,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )
        return obs

    def _do_step(self, action: ChatMessage) -> tuple[Observation, float, bool, dict[str, Any]]:
        assert self._task is not None

        completion_text = action.content or ""
        pred = self._parser.parse(completion_text, self._task.payload)
        gold = self._task.payload.get("answer", "")

        pred_n = _normalize_answer(str(pred))
        gold_n = _normalize_answer(str(gold))

        reward = 1.0 if pred_n == gold_n else 0.0
        components = {"exact_match": reward}

        obs = Observation(
            messages=[
                ChatMessage(role="user", content=self._task.payload["question"]),
                ChatMessage(role="assistant", content=completion_text),
            ],
            available_tools=[],
            turn_index=1,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )

        info = {
            "reward_components": components,
            "termination_cause": "final",
            "success": bool(reward == 1.0),
            "gold_answer": gold,
            "pred_answer": pred,
        }

        truncated = False
        return obs, float(reward), truncated, info
