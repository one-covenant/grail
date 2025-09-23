"""SAT Problem Solver for GRAIL RL System."""

import hashlib
import logging
import random
import re
from typing import Any, Callable, Optional, cast

from ..mining.rollout_generator import RolloutGenerator
from ..shared.constants import MAX_NEW_TOKENS
from .base import Parser, RewardVector

# SAT Problem Configuration
MIN_VARS = 4
MAX_VARS = 15
MIN_CLAUSES = 5
MAX_CLAUSES = 20
CLAUSE_LENGTH = 3  # 3-SAT


logger = logging.getLogger(__name__)


class SATProblem:
    """Represents a SAT problem instance."""

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


def create_sat_prompt(problem: "SATProblem") -> str:
    """Create a SAT prompt for the given problem."""
    instructions = (
        "Provide your final assignment between <SOLUTION></SOLUTION> as "
        "space-separated 0/1 values for x1..xN (e.g., "
        "<SOLUTION>0 1 0 1</SOLUTION>).\n"
    )
    prompt = f"SAT Problem:\n{problem.to_text()}\n{instructions}"
    return prompt


def generate_sat_problem(seed: str, difficulty: float = 0.5) -> SATProblem:
    """Generate a SAT problem from seed with controlled difficulty."""
    # Use seed for deterministic generation
    rng = random.Random(hashlib.sha256(seed.encode()).digest())

    # Scale problem size based on difficulty
    num_vars = rng.randint(
        MIN_VARS + int((MAX_VARS - MIN_VARS) * difficulty * 0.5),
        MIN_VARS + int((MAX_VARS - MIN_VARS) * difficulty),
    )
    num_clauses = rng.randint(
        MIN_CLAUSES + int((MAX_CLAUSES - MIN_CLAUSES) * difficulty * 0.5),
        MIN_CLAUSES + int((MAX_CLAUSES - MIN_CLAUSES) * difficulty),
    )

    clauses = []
    for _ in range(num_clauses):
        clause = []
        vars_in_clause = rng.sample(
            range(1, num_vars + 1),
            min(CLAUSE_LENGTH, num_vars),
        )
        for var in vars_in_clause:
            # Randomly negate
            if rng.random() < 0.5:
                clause.append(-var)
            else:
                clause.append(var)
        clauses.append(clause)

    return SATProblem(num_vars, clauses, seed)


class SATParser(Parser):
    """Parser for extracting boolean assignments and formatting metadata.

    Contract:
      Returns a dict with keys:
        - assignment: List[bool]
        - answer_text: str
        - has_thinking: bool
        - has_answer: bool
        - strict_format_ok: bool
        - trailing_after_answer: int
        - answer_has_only_bits_spaces: bool
    """

    # Strict tag names (no internal whitespace permitted)
    _TAG_SOLUTION_OPEN = "SOLUTION"
    _TAG_SOLUTION_CLOSE = "SOLUTION"
    _TAG_THINK_OPEN = "start_working_out"
    _TAG_THINK_CLOSE = "end_working_out"

    # Strict, reusable tag regexes built from the names above.
    # Note: case-insensitive, but no spaces allowed inside angle brackets.
    _ANSWER_OPEN = re.compile(rf"<{_TAG_SOLUTION_OPEN}>", re.IGNORECASE)
    _ANSWER_CLOSE = re.compile(rf"</{_TAG_SOLUTION_CLOSE}>", re.IGNORECASE)
    _THINK_OPEN = re.compile(rf"<{_TAG_THINK_OPEN}>", re.IGNORECASE)
    _THINK_CLOSE = re.compile(rf"</{_TAG_THINK_CLOSE}>", re.IGNORECASE)

    # First <SOLUTION>...</SOLUTION> anywhere.
    _ANSWER_PAIR = re.compile(
        (
            rf"<{_TAG_SOLUTION_OPEN}>"
            r"(?P<content>.*?)"
            rf"</{_TAG_SOLUTION_CLOSE}>"
        ),
        re.IGNORECASE | re.DOTALL,
    )

    # First <start_working_out>...</end_working_out> block.
    _THINK_BLOCK = re.compile(
        (
            rf"<{_TAG_THINK_OPEN}>"
            r".*?"
            rf"</{_TAG_THINK_CLOSE}>"
        ),
        re.IGNORECASE | re.DOTALL,
    )

    # First <SOLUTION> after the first thinking block.
    _THINK_THEN_ANSWER_PAIR = re.compile(
        (
            rf"<{_TAG_THINK_OPEN}>"
            r".*?"
            rf"</{_TAG_THINK_CLOSE}>"
            r".*?"
            rf"<{_TAG_SOLUTION_OPEN}>"
            r"(?P<content>.*?)"
            rf"</{_TAG_SOLUTION_CLOSE}>"
        ),
        re.IGNORECASE | re.DOTALL,
    )

    def _exact_token_count(self, answer_text: str) -> int:
        # Count 0/1 tokens separated by whitespace
        tokens = re.findall(r"[01]", answer_text)
        return len(tokens)

    def parse(self, completion: str, problem: SATProblem) -> dict[str, Any]:
        text = completion or ""

        # Presence flags and counts via strict regexes
        has_thinking = bool(self._THINK_BLOCK.search(text))
        has_answer = bool(self._ANSWER_PAIR.search(text))
        num_ans_opens = len(self._ANSWER_OPEN.findall(text))
        num_ans_closes = len(self._ANSWER_CLOSE.findall(text))

        answer_text = ""
        trailing_after_answer = 0
        strict_format_ok = False

        if has_answer:
            pair_match = None

            if has_thinking:
                # Prefer the first <SOLUTION> that appears after the first
                # thinking block.
                pair_match = self._THINK_THEN_ANSWER_PAIR.search(text)

            if pair_match is None:
                # Fallback: first <SOLUTION> anywhere
                pair_match = self._ANSWER_PAIR.search(text)

            if pair_match is not None:
                answer_text = pair_match.group("content")
                # Everything after </SOLUTION>
                trailing_after_answer = len(text) - pair_match.end()
                # Strict format: exactly one answer pair, no trailing,
                # bits-only, exact count
                strict_format_ok = (
                    num_ans_opens == 1
                    and num_ans_closes == 1
                    and trailing_after_answer == 0
                    and bool(re.fullmatch(r"[\s01]*", answer_text or ""))
                    and self._exact_token_count(answer_text) == problem.num_vars
                )
            else:
                # Malformed; treat as no answer
                has_answer = False
                answer_text = ""
                trailing_after_answer = 0

        answer_has_only_bits_spaces = bool(re.fullmatch(r"[\s01]*", answer_text or ""))

        # Determine assignment strictly from <SOLUTION> block;
        # otherwise default all-false
        if has_answer and answer_text:
            # Parse from answer_text strictly as bits
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

        # Best-effort: debug-only logging, no PII and no thinking text
        logger.debug(
            "SATParser: has_answer=%s, has_thinking=%s, strict=%s, trailing=%d",
            has_answer,
            has_thinking,
            strict_format_ok,
            trailing_after_answer,
        )

        return parsed


def _normalize_assignment(parsed: dict[str, Any], problem: SATProblem) -> list[bool]:
    """Accepts parsed dict from SATParser, returns List[bool]."""
    assignment_any = parsed.get("assignment", [])
    try:
        assignment_any = assignment_any[: problem.num_vars]
        return [bool(x) for x in assignment_any]
    except Exception:
        return [False] * problem.num_vars


def sat_correctness_reward(parsed_or_assignment: Any, problem: SATProblem) -> float:
    """Primary correctness reward.

    GRPO best-practice gating: only consider correctness when there is a
    <SOLUTION>...</SOLUTION> block whose contents are only 0/1 tokens
    (whitespace allowed), the number of tokens equals the number of
    variables, and there is no trailing text after the closing tag.
    Well-formed but wrong ⇒ 0.0; malformed ⇒ small negative.
    """
    # Enforce formatting gate before checking correctness
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
        # Small penalty to discourage malformed outputs
        return -0.2

    assignment = _normalize_assignment(parsed_or_assignment, problem)
    return 1.0 if problem.check_solution(assignment) else 0.0


# ----------------------------- Formatting Rewards ----------------------------


def sat_strict_format_reward(parsed: Any, _: SATProblem) -> float:
    if isinstance(parsed, dict) and parsed.get("strict_format_ok"):
        return 0.3
    return 0.0


def sat_soft_format_reward(parsed: Any, _: SATProblem) -> float:
    if isinstance(parsed, dict) and parsed.get("has_answer"):
        return 0.2
    return 0.0


def sat_answer_shape_reward(parsed: Any, problem: SATProblem) -> float:
    if isinstance(parsed, dict):
        bits = re.findall(r"[01]", parsed.get("answer_text", ""))
        return 0.3 if len(bits) == problem.num_vars else 0.0
    return 0.0


def sat_no_trailing_reward(parsed: Any, _: SATProblem) -> float:
    if isinstance(parsed, dict) and parsed.get("has_answer"):
        trailing = int(parsed.get("trailing_after_answer", 0))
        return max(0.0, 0.2 - 0.001 * trailing)
    return 0.0


def sat_charset_reward(parsed: Any, _: SATProblem) -> float:
    if (
        isinstance(parsed, dict)
        and parsed.get("has_answer")
        and parsed.get("answer_has_only_bits_spaces")
    ):
        return 0.1
    return 0.0


def create_sat_reward_vector(
    correctness_weight: float = 1.0,
    strict_weight: float = 0.15,
    soft_weight: float = 0.10,
    shape_weight: float = 0.15,
    no_trailing_weight: float = 0.10,
    charset_weight: float = 0.05,
) -> RewardVector:
    """Create SAT reward vector with correctness + small formatting rewards.

    Weights are small for formatting so correctness dominates.
    """
    reward_functions = cast(
        list[Callable[[Any, Any], float]],
        [
            sat_correctness_reward,
            sat_strict_format_reward,
            sat_soft_format_reward,
            sat_answer_shape_reward,
            sat_no_trailing_reward,
            sat_charset_reward,
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

    # Per-function bounds
    SAT_CORRECTNESS_BOUNDS = (-0.2, 1.0)
    STRICT_BOUNDS = (0.0, 0.3)
    SOFT_BOUNDS = (0.0, 0.2)
    SHAPE_BOUNDS = (0.0, 0.3)
    TRAIL_BOUNDS = (0.0, 0.2)
    CHARSET_BOUNDS = (0.0, 0.1)

    return RewardVector(
        reward_functions,
        weights,
        parser,
        bounds=[
            SAT_CORRECTNESS_BOUNDS,
            STRICT_BOUNDS,
            SOFT_BOUNDS,
            SHAPE_BOUNDS,
            TRAIL_BOUNDS,
            CHARSET_BOUNDS,
        ],
    )


class SATRolloutGenerator(RolloutGenerator):
    """SAT-specific rollout generator using RewardVector."""

    _current_problem: Optional[SATProblem]

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        rollouts_per_problem: int = 4,
        reward_vector: Optional[RewardVector] = None,
    ) -> None:
        """Initialize SAT rollout generator.

        Args:
            model: Language model for generation
            tokenizer: Tokenizer for the model
            device: Device to run on
            rollouts_per_problem: Number of rollouts per problem for GRPO
            reward_vector: Custom reward vector, creates default if None
        """
        super().__init__(model, tokenizer, device, rollouts_per_problem)
        self.reward_vector = reward_vector or create_sat_reward_vector()
        # Create a dummy environment for compatibility
        self._current_problem = None

    # TODO: in later versions this will return the env, not the problem,
    # supporting multi-turn RL
    def create_environment(self, problem: SATProblem) -> SATProblem:
        """Return the problem itself as 'environment' - no separate env
        needed.
        """
        self._current_problem = problem
        return problem

    def reset_environment(self, env: SATProblem) -> dict[str, Any]:
        """Return initial state - just the problem description."""
        return {
            "problem": env.to_text(),
            "num_vars": env.num_vars,
            "clauses": env.clauses,
        }

    def create_prompt(
        self,
        problem: SATProblem,
        env: SATProblem,
        state: dict[str, Any],
        trajectory: list,
    ) -> str:
        """Create minimal prompt with problem and solution format hint."""
        return create_sat_prompt(problem)

    def parse_action(self, text: str, env: SATProblem, state: dict[str, Any]) -> list[bool]:
        """Parse completion text into boolean assignment using reward parser
        output.

        Also caches last raw completion for reward computation.
        """
        # Cache the raw model output so step_environment can access formatting
        self._last_completion_text = text

        # Prefer using the configured parser; fallback to SATParser
        parser_obj: Parser = self.reward_vector.parser or SATParser()
        parsed_any = parser_obj.parse(text, env)

        if isinstance(parsed_any, dict):
            values_any = parsed_any.get("assignment", [])
            values_any = values_any[: env.num_vars]
            return [bool(x) for x in values_any]

        return []

    def step_environment(
        self, env: SATProblem, action: list[bool]
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Single-shot evaluation using reward vector.

        Uses the raw completion text when available to enable formatting
        rewards.
        """
        # Prefer the raw completion from the model to preserve formatting
        completion = getattr(self, "_last_completion_text", None)
        if not isinstance(completion, str) or not completion:
            # Fallback: reconstruct from action (legacy)
            completion = " ".join("1" if val else "0" for val in action)

        # Compute reward using reward vector
        reward = self.reward_vector.compute_reward(completion, env)
        # Optional: warn if reward outside declarative bounds
        try:
            if hasattr(self.reward_vector, "has_bounds") and self.reward_vector.has_bounds():
                low, high = self.reward_vector.reward_bounds()
                if reward < low or reward > high:
                    # Lazy import to avoid circular logger config
                    logger.warning(
                        "SAT reward %.4f outside composed bounds [%.4f, %.4f]",
                        reward,
                        low,
                        high,
                    )
        except Exception:
            # Non-fatal: continue without bounds warning
            pass

        # Check success — gate on strict answer formatting to avoid rewarding
        # malformed outputs that happen to contain a correct pattern elsewhere
        # Success iff correctness reward == 1.0 (correct and well-formed)
        success = False
        try:
            if self.reward_vector.parser:
                parsed = self.reward_vector.parser.parse(completion, env)
                if isinstance(parsed, dict):
                    success = sat_correctness_reward(parsed, env) == 1.0
        except Exception:
            success = False

        # Count satisfied clauses
        satisfied_clauses = 0
        if len(action) == env.num_vars:
            for clause in env.clauses:
                for lit in clause:
                    var_idx = abs(lit) - 1
                    if (lit > 0 and action[var_idx]) or (lit < 0 and not action[var_idx]):
                        satisfied_clauses += 1
                        break

        state = {
            "problem": env.to_text(),
            "assignment": action,
            "success": success,
        }

        info = {
            "success": success,
            "satisfied_clauses": satisfied_clauses,
            "total_clauses": len(env.clauses),
            "assignment": action,
        }

        return state, reward, True, info  # Always done after one step

    def get_final_info(
        self, env: SATProblem, trajectory: list, total_reward: float
    ) -> dict[str, Any]:
        """Get final SAT-specific information."""
        if trajectory:
            _, assignment, info = trajectory[-1]
            success = info.get("success", False)
            satisfied_clauses = 0
            if len(assignment) == env.num_vars:
                for clause in env.clauses:
                    for lit in clause:
                        var_idx = abs(lit) - 1
                        if (lit > 0 and assignment[var_idx]) or (
                            lit < 0 and not assignment[var_idx]
                        ):
                            satisfied_clauses += 1
                            break
        else:
            assignment = [False] * env.num_vars
            success = False
            satisfied_clauses = 0

        return {
            "success": success,
            "satisfied_clauses": satisfied_clauses,
            "assignment": assignment,
            "sat_problem": {
                "seed": env.seed,
                "num_vars": env.num_vars,
                "clauses": env.clauses,
            },
        }

    # Decoding controls
    def get_max_tokens(self) -> int:
        # Use project-wide default; can be overridden via GRAIL_MAX_NEW_TOKENS
        return int(MAX_NEW_TOKENS)
