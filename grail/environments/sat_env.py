"""SAT environment with all SAT logic consolidated (DRY principle).

This module contains:
- SAT problem generation, prompting, parsing, and reward functions (internalized)
- Public API for validators (generate_sat_problem, create_sat_prompt, etc.)
- SATEnv implementation inheriting SingleTurnEnv
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections.abc import Callable
from typing import Any, cast

from ..shared.prompt_constants import (
    SOLUTION_END_TOKEN,
    SOLUTION_START_TOKEN,
)
from .base import Parser, RewardVector, ThinkingParser
from .core import ChatMessage, Observation, SingleTurnEnv
from .providers import SATTaskSource, TaskSpec
from .rubric import RewardVectorRubric

# SAT Problem Configuration
_MIN_VARS = 4
_MAX_VARS = 15
_MIN_CLAUSES = 5
_MAX_CLAUSES = 20
_CLAUSE_LENGTH = 3  # 3-SAT

logger = logging.getLogger(__name__)


# ============================================================================
# INTERNAL SAT PROBLEM REPRESENTATION
# ============================================================================


class _SATProblem:
    """Internal SAT problem instance representation."""

    def __init__(self, num_vars: int, clauses: list[list[int]], seed: str):
        self.num_vars = num_vars
        self.clauses = clauses
        self.seed = seed
        self.solution = None

    def to_text(self) -> str:
        """Convert SAT problem to text format for LLM processing."""
        text = f"SAT Problem (seed: {self.seed[:8]}...):\n"
        text += f"Variables: {self.num_vars}\n"
        text += "Clauses:\n"
        for _i, clause in enumerate(self.clauses):
            clause_str = " OR ".join([f"{'NOT ' if lit < 0 else ''}x{abs(lit)}" for lit in clause])
            text += f"  ({clause_str})\n"
        return text

    def check_solution(self, assignment: list[bool]) -> bool:
        """Check if assignment satisfies all clauses."""
        if len(assignment) != self.num_vars:
            return False

        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if lit > 0 and assignment[var_idx]:
                    satisfied = True
                    break
                elif lit < 0 and not assignment[var_idx]:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True


def _generate_sat_problem(seed: str, difficulty: float = 0.5) -> _SATProblem:
    """Generate a SAT problem from seed with controlled difficulty."""
    rng = random.Random(hashlib.sha256(seed.encode()).digest())

    num_vars = rng.randint(
        _MIN_VARS + int((_MAX_VARS - _MIN_VARS) * difficulty * 0.5),
        _MIN_VARS + int((_MAX_VARS - _MIN_VARS) * difficulty),
    )
    num_clauses = rng.randint(
        _MIN_CLAUSES + int((_MAX_CLAUSES - _MIN_CLAUSES) * difficulty * 0.5),
        _MIN_CLAUSES + int((_MAX_CLAUSES - _MIN_CLAUSES) * difficulty),
    )

    clauses = []
    for _ in range(num_clauses):
        clause = []
        vars_in_clause = rng.sample(
            range(1, num_vars + 1),
            min(_CLAUSE_LENGTH, num_vars),
        )
        for var in vars_in_clause:
            if rng.random() < 0.5:
                clause.append(-var)
            else:
                clause.append(var)
        clauses.append(clause)

    problem = _SATProblem(num_vars, clauses, seed)
    logger.debug(
        ("Generated SAT problem: seed=%s difficulty=%.3f num_vars=%d num_clauses=%d"),
        seed[:12] if len(seed) > 12 else seed,
        difficulty,
        num_vars,
        num_clauses,
    )
    return problem


def _create_sat_prompt(problem: _SATProblem) -> str:
    """Create a SAT prompt for the given problem."""
    instructions = (
        "Provide your final assignment between <SOLUTION></SOLUTION> as "
        "space-separated 0/1 values for x1..xN (e.g., "
        "<SOLUTION>0 1 0 1</SOLUTION>).\n"
    )
    prompt = f"SAT Problem:\n{problem.to_text()}\n{instructions}"
    return prompt


# ============================================================================
# SAT PARSER (Inherits from ThinkingParser)
# ============================================================================


class SATParser(ThinkingParser):
    """Parser for extracting boolean assignments and formatting metadata.

    Inherits thinking and answer tag detection from ThinkingParser base class.
    """

    # Additional SAT-specific regex patterns (not in base class)
    _ANSWER_OPEN = re.compile(rf"<{SOLUTION_START_TOKEN}>", re.IGNORECASE)
    _ANSWER_CLOSE = re.compile(rf"</{SOLUTION_END_TOKEN}>", re.IGNORECASE)

    def _exact_token_count(self, answer_text: str) -> int:
        tokens = re.findall(r"[01]", answer_text)
        return len(tokens)

    def parse(self, completion: str, problem: _SATProblem) -> dict[str, Any]:
        text = completion or ""

        has_thinking = self._detect_thinking_block(text)
        has_answer = self._detect_answer_block(text)
        num_ans_opens = len(self._ANSWER_OPEN.findall(text))
        num_ans_closes = len(self._ANSWER_CLOSE.findall(text))

        answer_text = ""
        trailing_after_answer = 0
        strict_format_ok = False

        if has_answer:
            pair_match = None
            if has_thinking:
                pair_match = self._get_think_then_answer_pattern().search(text)
            if pair_match is None:
                pair_match = self._get_answer_pattern().search(text)

            if pair_match is not None:
                answer_text = pair_match.group("content")
                trailing_after_answer = len(text) - pair_match.end()
                strict_format_ok = (
                    num_ans_opens == 1
                    and num_ans_closes == 1
                    and trailing_after_answer == 0
                    and bool(re.fullmatch(r"[\s01]*", answer_text or ""))
                    and self._exact_token_count(answer_text) == problem.num_vars
                )
            else:
                has_answer = False
                answer_text = ""
                trailing_after_answer = 0

        answer_has_only_bits_spaces = bool(re.fullmatch(r"[\s01]*", answer_text or ""))

        if has_answer and answer_text:
            bits = re.findall(r"[01]", answer_text)
            actions = [int(b) for b in bits[: problem.num_vars]]
            while len(actions) < problem.num_vars:
                actions.append(0)
            assignment = [bool(x) for x in actions[: problem.num_vars]]
        else:
            assignment = []

        parsed: dict[str, Any] = {
            "assignment": assignment,
            "answer_text": answer_text,
            "has_thinking": has_thinking,
            "has_answer": has_answer,
            "strict_format_ok": strict_format_ok,
            "trailing_after_answer": int(trailing_after_answer),
            "answer_has_only_bits_spaces": answer_has_only_bits_spaces,
        }

        logger.debug(
            "SATParser: has_answer=%s, has_thinking=%s, strict=%s, trailing=%d",
            has_answer,
            has_thinking,
            strict_format_ok,
            trailing_after_answer,
        )

        return parsed


# ============================================================================
# INTERNAL REWARD FUNCTIONS
# ============================================================================


def _normalize_assignment(parsed: dict[str, Any], problem: _SATProblem) -> list[bool]:
    assignment_any = parsed.get("assignment", [])
    try:
        assignment_any = assignment_any[: problem.num_vars]
        return [bool(x) for x in assignment_any]
    except Exception:
        return [False] * problem.num_vars


def _sat_correctness_reward(parsed_or_assignment: Any, problem: _SATProblem) -> float:
    if not isinstance(parsed_or_assignment, dict):
        return -0.2

    has_answer = bool(parsed_or_assignment.get("has_answer"))
    only_bits_spaces = bool(parsed_or_assignment.get("answer_has_only_bits_spaces"))
    trailing_after = int(parsed_or_assignment.get("trailing_after_answer", 0))
    answer_text = parsed_or_assignment.get("answer_text", "")
    bits = re.findall(r"[01]", answer_text)

    well_formed = (
        has_answer and only_bits_spaces and trailing_after == 0 and len(bits) == problem.num_vars
    )

    if not well_formed:
        return -0.2

    assignment = _normalize_assignment(parsed_or_assignment, problem)
    return 1.0 if problem.check_solution(assignment) else 0.0


def _sat_strict_format_reward(parsed: Any, _: _SATProblem) -> float:
    if isinstance(parsed, dict) and parsed.get("strict_format_ok"):
        return 0.3
    return 0.0


def _sat_soft_format_reward(parsed: Any, _: _SATProblem) -> float:
    """Soft format reward for SAT - delegates to shared implementation."""
    from .reward_components import soft_format_reward

    return soft_format_reward(parsed, None)


def _sat_no_trailing_reward(parsed: Any, _: _SATProblem) -> float:
    """No trailing reward for SAT - delegates to shared implementation."""
    from .reward_components import no_trailing_reward

    return no_trailing_reward(parsed, None)


def _sat_answer_shape_reward(parsed: Any, problem: _SATProblem) -> float:
    if isinstance(parsed, dict):
        bits = re.findall(r"[01]", parsed.get("answer_text", ""))
        return 0.3 if len(bits) == problem.num_vars else 0.0
    return 0.0


def _sat_charset_reward(parsed: Any, _: _SATProblem) -> float:
    if (
        isinstance(parsed, dict)
        and parsed.get("has_answer")
        and parsed.get("answer_has_only_bits_spaces")
    ):
        return 0.1
    return 0.0


def _create_sat_reward_vector(
    correctness_weight: float = 1.0,
    strict_weight: float = 0.15,
    soft_weight: float = 0.10,
    shape_weight: float = 0.15,
    no_trailing_weight: float = 0.10,
    charset_weight: float = 0.05,
) -> RewardVector:
    """Create SAT reward vector with correctness + formatting rewards."""
    reward_functions = cast(
        list[Callable[[Any, Any], float]],
        [
            _sat_correctness_reward,
            _sat_strict_format_reward,
            _sat_soft_format_reward,
            _sat_answer_shape_reward,
            _sat_no_trailing_reward,
            _sat_charset_reward,
        ],
    )
    weights = [
        correctness_weight,
        strict_weight,
        soft_weight,
        shape_weight,
        no_trailing_weight,
        charset_weight,
    ]
    parser = SATParser()

    return RewardVector(
        reward_functions,
        weights,
        parser,
        bounds=[
            (-0.2, 1.0),  # correctness
            (0.0, 0.3),  # strict
            (0.0, 0.2),  # soft
            (0.0, 0.3),  # shape
            (0.0, 0.2),  # trailing
            (0.0, 0.1),  # charset
        ],
    )


# ============================================================================
# PUBLIC API FOR VALIDATORS (wrappers over internals)
# ============================================================================

# Type alias for compatibility
SATProblem = _SATProblem


def generate_sat_problem(seed: str, difficulty: float = 0.5) -> SATProblem:
    """Public wrapper for SAT problem generation (used by validators)."""
    return _generate_sat_problem(seed, difficulty)


def create_sat_prompt(problem: SATProblem) -> str:
    """Public wrapper for SAT prompt creation (used by validators)."""
    return _create_sat_prompt(problem)


def sat_correctness_reward(parsed_or_assignment: Any, problem: SATProblem) -> float:
    """Public wrapper for correctness reward (used by validators/exports)."""
    return _sat_correctness_reward(parsed_or_assignment, problem)


def create_sat_reward_vector(
    correctness_weight: float = 1.0,
    strict_weight: float = 0.15,
    soft_weight: float = 0.10,
    shape_weight: float = 0.15,
    no_trailing_weight: float = 0.10,
    charset_weight: float = 0.05,
) -> RewardVector:
    """Public wrapper for reward vector creation (used by validators)."""
    return _create_sat_reward_vector(
        correctness_weight,
        strict_weight,
        soft_weight,
        shape_weight,
        no_trailing_weight,
        charset_weight,
    )


# ============================================================================
# SAT ENVIRONMENT (SingleTurnEnv implementation)
# ============================================================================


class SATEnv(SingleTurnEnv):
    """Single-turn SAT environment using template-method pattern.

    Generates SAT problems deterministically, parses solutions, and computes
    rewards with decomposed components for RLVR training.
    """

    def __init__(
        self,
        *,
        task_source: SATTaskSource | None = None,
        parser: Parser | None = None,
        rubric: RewardVectorRubric | None = None,
    ) -> None:
        super().__init__()
        self._source = task_source or SATTaskSource()
        self._parser = parser or SATParser()
        self._rubric = rubric or RewardVectorRubric(_create_sat_reward_vector())
        self._task: TaskSpec | None = None
        self._problem: _SATProblem | None = None

    def _do_reset(self, *, task_id: str | None = None, seed: int | None = None) -> Observation:
        self._task = self._source.next(seed=seed, task_id=task_id)
        payload = self._task.payload
        assert isinstance(payload, _SATProblem)
        self._problem = payload

        user_prompt = _create_sat_prompt(self._problem)
        obs = Observation(
            messages=[ChatMessage(role="user", content=user_prompt)],
            available_tools=[],
            turn_index=0,
            task_meta={
                "task_id": self._task.id,
                "sat_problem": {
                    "seed": self._problem.seed,
                    "num_vars": self._problem.num_vars,
                    "clauses": self._problem.clauses,
                },
                **self._task.metadata,
            },
        )
        return obs

    def _do_step(self, action: ChatMessage) -> tuple[Observation, float, bool, dict[str, Any]]:
        assert self._problem is not None and self._task is not None

        completion_text = action.content or ""
        parsed = self._parser.parse(completion_text, self._problem)

        reward, components = self._rubric.step_reward(
            parsed=completion_text,
            context=self._problem,
            turn_index=1,
        )

        assignment: list[bool] = []
        try:
            if isinstance(parsed, dict):
                values = parsed.get("assignment", [])
                assignment = [bool(v) for v in values[: self._problem.num_vars]]
        except Exception:
            assignment = []

        success = False
        try:
            if isinstance(parsed, dict):
                success = _sat_correctness_reward(parsed, self._problem) == 1.0
        except Exception:
            success = False

        satisfied_clauses = 0
        if len(assignment) == self._problem.num_vars:
            for clause in self._problem.clauses:
                for lit in clause:
                    var_idx = abs(lit) - 1
                    if (lit > 0 and assignment[var_idx]) or (lit < 0 and not assignment[var_idx]):
                        satisfied_clauses += 1
                        break

        obs = Observation(
            messages=[
                ChatMessage(role="user", content=_create_sat_prompt(self._problem)),
                ChatMessage(role="assistant", content=completion_text),
            ],
            available_tools=[],
            turn_index=1,
            task_meta={
                "task_id": self._task.id,
                "sat_problem": {
                    "seed": self._problem.seed,
                    "num_vars": self._problem.num_vars,
                    "clauses": self._problem.clauses,
                },
                **self._task.metadata,
            },
        )

        info = {
            "reward_components": components,
            "termination_cause": "success" if success else "final",
            "success": success,
            "satisfied_clauses": satisfied_clauses,
            "total_clauses": len(self._problem.clauses),
            "assignment": assignment,
        }

        truncated = False
        return obs, float(reward), truncated, info
