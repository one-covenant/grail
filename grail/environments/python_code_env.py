"""Single-turn Python code generation environment using MBPP/HumanEval datasets.

This environment serves programming problems and evaluates generated code by:
- Executing test cases in a sandboxed subprocess
- Computing rewards based on test pass rate, syntax validity, and format adherence
- Supporting both MBPP (training) and HumanEval (evaluation) datasets

Key features:
- Safe code execution with timeout and privilege reduction
- Decomposed reward: correctness, syntax, format, thinking
- Flexible dataset backends (MBPP train/val/test, HumanEval test)
- Single test execution per step (reward functions use cached results)

Expected completion format:
    <start_working_out>
    Step-by-step reasoning about the problem...
    </end_working_out>
    <SOLUTION>
    def function_name(args):
        # Implementation here
        return result
    </SOLUTION>

RLVR/GRPO Design Principles:
- Rewards are deterministic given (completion, context) - no stochasticity
- Test execution happens exactly once per step to ensure reward/success consistency
- Reward bounds are explicit and achievable (max=1.0)
- Parser extracts all needed features in a single pass
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .base import Parser, RewardVector, ThinkingParser
from .core import ChatMessage, Observation, Rubric, SingleTurnEnv
from .execution import check_code_executes_fast
from .providers import HumanEvalTaskSource, MBPPTaskSource, TaskSource, TaskSpec


class PythonCodeParser(ThinkingParser):
    """Parser for Python code completions with <SOLUTION> tag detection.

    Inherits thinking tag detection from ThinkingParser base class.
    Extracts Python code from <SOLUTION>...</SOLUTION> tags.

    Expected format:
        <start_working_out>reasoning...</end_working_out>
        <SOLUTION>code...</SOLUTION>

    Detects:
        - Thinking blocks: inherited from ThinkingParser
        - Solution blocks: <SOLUTION>...</SOLUTION>
        - Trailing text: chars after closing tag
        - Syntax validity: Python compilation check
    """

    def parse(self, completion: str, context: Any) -> dict[str, Any]:
        """Parse completion for thinking tags, solution code, and syntax validity.

        This parser is designed to be robust and never raise exceptions.
        All edge cases return valid dict with appropriate default values.

        Returns dict with:
            - code: extracted code from <SOLUTION> tags (empty string if none)
            - has_thinking: bool, True if thinking block present
            - has_solution: bool, True if <SOLUTION> tags present
            - trailing_after_solution: int, chars after </SOLUTION>
            - syntax_valid: bool, True if code compiles without SyntaxError

        Edge cases:
            - None/empty completion: all False/0/""
            - Malformed tags: has_solution=False
            - Empty <SOLUTION></SOLUTION>: code="", syntax_valid=False
            - Whitespace-only code: code stripped, syntax checked on result
        """
        # Defensive: handle None and non-string inputs
        if completion is None:
            text = ""
        elif isinstance(completion, str):
            text = completion
        else:
            text = str(completion)  # Coerce to string as last resort

        # Use inherited methods from ThinkingParser
        has_thinking = self._detect_thinking_block(text)
        has_solution = self._detect_answer_block(text)  # Reuse answer block detection

        # Extract code using inherited method
        code = ""
        trailing_after_solution = 0
        syntax_valid = False

        if has_solution:
            try:
                content, trailing, _ = self._get_answer_with_thinking_check(text)
                if content is not None:
                    code = content.strip()
                    trailing_after_solution = max(0, trailing)  # Ensure non-negative

                    # Check syntax validity only for non-empty code
                    if code:
                        try:
                            compile(code, "<string>", "exec")
                            syntax_valid = True
                        except SyntaxError:
                            syntax_valid = False
            except Exception:
                # Parser should never crash - return safe defaults
                code = ""
                trailing_after_solution = 0
                syntax_valid = False

        return {
            "code": code,
            "has_thinking": has_thinking,
            "has_solution": has_solution,
            "trailing_after_solution": trailing_after_solution,
            "syntax_valid": syntax_valid,
        }


# =============================================================================
# Reward Functions for RLVR/GRPO
# =============================================================================
# These functions are designed with GRPO best practices:
# 1. They operate on parsed dict (not raw completion) - parsing happens once
# 2. Each function returns a value in a well-defined bounded range
# 3. The correctness function uses pre-computed test results from parsed dict
#    to avoid double execution
# =============================================================================


def _python_correctness_reward_from_parsed(parsed: dict[str, Any], context: Any) -> float:
    """Python code correctness reward based on cached test pass rate.

    IMPORTANT: This function expects parsed['test_result'] to contain
    pre-computed test execution results. This avoids executing tests twice
    (once for success determination, once for reward computation).

    Args:
        parsed: Parsed output with 'test_result' field containing execution results
        context: Task payload (unused, for API compatibility)

    Returns:
        Float in [0.0, 1.0] representing pass rate
    """
    if not isinstance(parsed, dict):
        return 0.0  # type: ignore[unreachable]

    # Use pre-computed test results if available
    test_result = parsed.get("test_result")
    if test_result is not None and isinstance(test_result, dict):
        total = test_result.get("total", 0)
        if total == 0:
            return 0.0
        passed = test_result.get("passed", 0)
        return float(passed) / float(total)

    # Fallback: no code extracted means 0 correctness
    return 0.0


def _python_syntax_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for syntactically valid Python code.

    Returns 1.0 if code compiles without SyntaxError, 0.0 otherwise.

    Args:
        parsed: Parsed output with 'syntax_valid' field
        context: Task payload (unused)

    Returns:
        1.0 if syntax is valid, 0.0 otherwise
    """
    if not isinstance(parsed, dict):
        return 0.0  # type: ignore[unreachable]

    return 1.0 if parsed.get("syntax_valid", False) else 0.0


def _python_solution_format_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for proper <SOLUTION> tag usage.

    Returns 1.0 if <SOLUTION> tags are present and minimal trailing text.

    Args:
        parsed: Parsed output with 'has_solution' and 'trailing_after_solution' fields
        context: Task payload (unused)

    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    if not isinstance(parsed, dict):
        return 0.0  # type: ignore[unreachable]

    has_solution = parsed.get("has_solution", False)
    trailing = int(parsed.get("trailing_after_solution", 0))

    # Allow small trailing text (e.g., closing remarks)
    if has_solution and trailing < 50:
        return 1.0

    return 0.0


def _python_thinking_format_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for having thinking/reasoning block.

    Returns 1.0 if thinking block present, 0.0 otherwise.

    NOTE: Unlike the shared thinking_format_reward which returns 0.5,
    this returns 1.0 to make the reward bounds clean (max total = 1.0).
    The weight on this component (0.1) controls its contribution.

    Args:
        parsed: Parsed output with 'has_thinking' field
        context: Task payload (unused)

    Returns:
        1.0 if thinking block present, 0.0 otherwise
    """
    if not isinstance(parsed, dict):
        return 0.0  # type: ignore[unreachable]

    return 1.0 if parsed.get("has_thinking", False) else 0.0


def _create_python_code_reward_vector() -> RewardVector:
    """Create Python code reward vector with 4 decomposed components.

    Components (all bounded [0.0, 1.0]):
        1. Correctness (0.7): Test pass rate (0.0-1.0)
        2. Syntax valid (0.1): Code compiles without errors
        3. Solution format (0.1): Proper <SOLUTION> tags
        4. Thinking (0.1): Presence of reasoning block

    Total weight: 1.0
    Max achievable reward: 1.0 (all components at max)

    GRPO Note: All component bounds are [0.0, 1.0] for clean normalization.
    """
    reward_functions = cast(
        list[Callable[[Any, Any], float]],
        [
            _python_correctness_reward_from_parsed,
            _python_syntax_reward,
            _python_solution_format_reward,
            _python_thinking_format_reward,
        ],
    )
    weights = [0.7, 0.1, 0.1, 0.1]

    return RewardVector(
        reward_functions,
        weights,
        parser=None,  # Parser is handled by env, not RewardVector
        bounds=[
            (0.0, 1.0),  # correctness (pass rate)
            (0.0, 1.0),  # syntax_valid
            (0.0, 1.0),  # solution_format
            (0.0, 1.0),  # thinking (now 0/1, not 0/0.5)
        ],
    )


class _PythonCodeRubric(Rubric):
    """Custom rubric that computes reward from pre-parsed dict with test results.

    This rubric avoids the double-execution problem by:
    1. Accepting a parsed dict that already contains test_result
    2. Computing component rewards from the parsed dict directly
    3. Never invoking test execution itself

    This is critical for GRPO: reward must be deterministic and consistent
    with the success flag computed in _do_step.
    """

    def __init__(self) -> None:
        self._reward_vector = _create_python_code_reward_vector()

    def step_reward(
        self, *, parsed: Any, context: Any, turn_index: int
    ) -> tuple[float, dict[str, float]]:
        """Compute reward from parsed dict containing test results.

        Args:
            parsed: Dict with parsing results AND test_result field
            context: Task payload (unused, passed to reward functions)
            turn_index: Turn number (unused for single-turn env)

        Returns:
            Tuple of (total_reward, component_dict)
        """
        if not isinstance(parsed, dict):
            return 0.0, {}

        try:
            # Compute individual rewards from parsed dict
            rewards = []
            for fn in self._reward_vector.reward_functions:
                rewards.append(fn(parsed, context))

            # Compute weighted sum
            total = sum(r * w for r, w in zip(rewards, self._reward_vector.weights, strict=False))

            # Build component dict with meaningful names
            components = {
                "correctness": rewards[0],
                "syntax": rewards[1],
                "format": rewards[2],
                "thinking": rewards[3],
            }

            return float(total), components
        except Exception:
            return 0.0, {}


def prepare_test_cases(payload: dict[str, Any]) -> list[str]:
    """Prepare test cases from task payload.

    Extracted as a standalone function for reuse.

    Args:
        payload: Task payload with test information

    Returns:
        List of executable test case strings
    """
    # MBPP format: test_list with optional setup/imports
    if "test_list" in payload:
        test_setup = payload.get("test_setup_code", "")
        test_imports = payload.get("test_imports", [])

        # Build setup code
        setup_code = "\n".join(test_imports) if test_imports else ""
        if test_setup:
            setup_code += f"\n{test_setup}"

        # Prepend setup to each test
        test_cases = []
        for test in payload["test_list"]:
            if setup_code:
                test_cases.append(f"{setup_code}\n{test}")
            else:
                test_cases.append(test)
        return test_cases

    # HumanEval format: single test function
    elif "test" in payload:
        test_func = payload["test"]
        entry_point = payload.get("entry_point", "")

        if entry_point and test_func:
            # Build complete test by calling check function
            test_code = f"{test_func}\ncheck({entry_point})"
            return [test_code]

    return []


class PythonCodeEnv(SingleTurnEnv):
    """Single-turn Python code generation environment with test execution.

    Serves programming problems from MBPP or HumanEval datasets and evaluates
    generated code by executing test cases in a sandboxed subprocess.

    Dataset backends:
        - MBPP: train (774), validation (190)
          Note: Original test set (500) redistributed 80/20 to train/validation.
          Requesting split='test' raises AssertionError.
        - HumanEval: test only (164)

    Completion format:
        <start_working_out>reasoning...</end_working_out>
        <SOLUTION>
        def function_name(args):
            return result
        </SOLUTION>

    Reward components:
        - Correctness (70%): Proportion of test cases passed
        - Syntax (10%): Code compiles without errors
        - Format (10%): Proper <SOLUTION> tags
        - Thinking (10%): Reasoning block present

    GRPO/RLVR Design:
        - Tests are executed exactly ONCE per step
        - Reward is computed from cached test results
        - success flag and reward are always consistent
        - Max achievable reward is 1.0

    Example:
        env = PythonCodeEnv(dataset="mbpp", split="train")
        obs = env.reset(seed=42)
        obs, reward, done, info = env.step(ChatMessage(role="assistant", content=completion))
        print(f"Tests passed: {info['tests_passed']}/{info['tests_total']}")
    """

    def __init__(
        self,
        *,
        dataset: str = "mbpp",
        split: str = "train",
        task_source: TaskSource | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
    ):
        """Initialize Python code generation environment.

        Args:
            dataset: Dataset to use ('mbpp' or 'humaneval')
            split: Dataset split ('train' or 'validation' for MBPP; 'test' for HumanEval)
            task_source: Custom task source (overrides dataset/split)
            parser: Custom parser (defaults to PythonCodeParser)
            rubric: Custom rubric (defaults to _PythonCodeRubric)

        Raises:
            AssertionError: If dataset='mbpp' and split='test' (test set redistributed)
            ValueError: If dataset/split combination is invalid
        """
        super().__init__()

        # Validate dataset/split combination
        if dataset not in ("mbpp", "humaneval"):
            raise ValueError(f"dataset must be 'mbpp' or 'humaneval', got '{dataset}'")

        if dataset == "humaneval" and split != "test":
            raise ValueError("HumanEval only supports split='test'")

        if dataset == "mbpp" and split == "test":
            raise AssertionError(
                "MBPP 'test' split does not exist. "
                "The original test set (500 examples) has been redistributed to "
                "train (80%) and validation (20%). Use 'train' or 'validation' instead."
            )
        if dataset == "mbpp" and split not in ("train", "validation"):
            raise ValueError(f"MBPP split must be 'train' or 'validation', got '{split}'")

        self._dataset = dataset
        self._split = split

        # Initialize components
        if task_source is None:
            if dataset == "mbpp":
                task_source = MBPPTaskSource(split=split)
            else:  # humaneval
                task_source = HumanEvalTaskSource()

        self._source = task_source
        self._parser = parser or PythonCodeParser()
        # Use custom rubric that works with pre-computed test results
        self._rubric = rubric or _PythonCodeRubric()
        self._task: TaskSpec | None = None

    def _do_reset(
        self,
        *,
        task_id: str | None = None,
        seed: int | None = None,
    ) -> Observation:
        """Reset environment and sample new programming problem.

        Args:
            task_id: Specific task ID to load
            seed: Random seed for deterministic sampling

        Returns:
            Initial observation with problem description
        """
        self._task = self._source.next(seed=seed, task_id=task_id)

        # Create initial observation with problem prompt
        obs = Observation(
            messages=[ChatMessage(role="user", content=self._task.payload["question"])],
            available_tools=[],
            turn_index=0,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )
        return obs

    def _do_step(self, action: ChatMessage) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute single turn: parse code, run tests ONCE, compute reward.

        GRPO Design: Tests are executed exactly once. The test_result is then
        passed to the rubric for reward computation, ensuring consistency
        between 'success' and 'reward'.

        Edge cases handled:
        - Empty completion: reward=0, success=False
        - No <SOLUTION> tags: correctness=0 (no code extracted)
        - Syntax error in code: syntax_valid=False, tests still run
        - No test cases in payload: success=False, tests_total=0
        - Test timeout: tracked but doesn't crash

        Args:
            action: Model's completion as ChatMessage

        Returns:
            Tuple of (observation, reward, truncated, info)
        """
        assert self._task is not None, "Must call reset() before step()"

        # Defensive: handle None content
        completion_text = action.content if action.content is not None else ""

        # Step 1: Parse completion (extracts code, checks syntax, detects tags)
        # Parser is defensive against empty strings
        parsed = self._parser.parse(completion_text, self._task.payload)
        code = parsed.get("code", "") or ""  # Ensure string, not None

        # Step 2: Execute tests ONCE and cache results
        tests_passed = 0
        tests_total = 0
        test_results: list[dict[str, Any]] = []
        test_result_dict: dict[str, Any] = {"passed": 0, "total": 0, "status": "no_code"}
        success = False

        if code:
            # Prepare test cases based on dataset format
            test_cases = prepare_test_cases(self._task.payload)

            if test_cases:
                # Execute tests ONCE - use fast pool if available
                test_result_dict = check_code_executes_fast(code, test_cases, timeout=5.0)
                tests_passed = test_result_dict["passed"]
                tests_total = test_result_dict["total"]
                test_results = test_result_dict.get("test_results", [])
                success = test_result_dict["status"] == "all_passed"

        # Step 3: Augment parsed dict with test results for reward computation
        # This is the key fix: reward functions use cached test results
        parsed_with_results = {
            **parsed,
            "test_result": test_result_dict,
        }

        # Step 4: Compute reward using rubric (no test execution here!)
        reward, components = self._rubric.step_reward(
            parsed=parsed_with_results,
            context=self._task.payload,
            turn_index=1,
        )

        # Build final observation
        obs = Observation(
            messages=[
                ChatMessage(role="user", content=self._task.payload["question"]),
                ChatMessage(role="assistant", content=completion_text),
            ],
            available_tools=[],
            turn_index=1,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )

        # Info dict with execution results
        info = {
            "reward_components": components,
            "termination_cause": "final",
            "success": success,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "test_results": test_results,
            "has_code": bool(code),
            "syntax_valid": parsed.get("syntax_valid", False),
        }

        truncated = False
        return obs, float(reward), truncated, info

    def _prepare_test_cases(self, payload: dict[str, Any]) -> list[str]:
        """Prepare test cases from task payload.

        Args:
            payload: Task payload with test information

        Returns:
            List of executable test case strings
        """
        return prepare_test_cases(payload)
