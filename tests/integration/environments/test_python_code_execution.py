"""Integration tests for Python code execution with MBPP/HumanEval environments.

Tests end-to-end execution flow:
1. Environment loads real problems from datasets
2. Correct reference solutions pass tests
3. Incorrect solutions fail tests
4. Reward computation accurately reflects test results
"""

from __future__ import annotations

import pytest

from grail.environments import ChatMessage, PythonCodeEnv, create_env

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _ensure_datasets() -> None:
    """Skip tests if HuggingFace datasets dependency is missing."""
    pytest.importorskip("datasets")


def _pull_task(env: object) -> dict[str, object]:
    """Extract task payload from environment."""
    task = getattr(env, "_task", None)
    assert task is not None, "Expected environment to hold current task after reset"
    return task.payload  # type: ignore[return-value]


def _build_solution(code: str, thinking: str = "") -> str:
    """Build completion with optional thinking block."""
    from grail.shared.thinking import get_thinking_config

    cfg = get_thinking_config()
    if thinking:
        return f"{cfg.thinking_open}\n{thinking}\n{cfg.thinking_close}\n{cfg.solution_open}\n{code}\n{cfg.solution_close}"
    return f"{cfg.solution_open}\n{code}\n{cfg.solution_close}"


# =============================================================================
# MBPP Environment Tests
# =============================================================================


class TestMBPPExecution:
    """Test MBPP environment execution with real dataset problems."""

    def test_reference_solution_passes(self) -> None:
        """Test that reference solution from dataset passes all tests."""
        env = create_env("mbpp", split="train")
        env.reset(seed=42)

        payload = _pull_task(env)
        reference = str(payload["reference_solution"])

        completion = _build_solution(reference, "Using the reference solution.")

        _obs, reward, terminated, truncated, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        assert terminated is True
        assert truncated is False
        assert info["success"] is True
        assert info["tests_passed"] == info["tests_total"]
        assert info["tests_total"] > 0
        assert reward >= 0.7  # Correctness weight is 0.7

    def test_incorrect_solution_fails(self) -> None:
        """Test that incorrect solution fails tests."""
        env = create_env("mbpp", split="train")
        env.reset(seed=42)

        wrong_code = "def wrong_func(): return None"
        completion = _build_solution(wrong_code)

        _obs, reward, terminated, truncated, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        assert terminated is True
        assert info["success"] is False
        assert reward < 0.7

    def test_syntax_error_detected(self) -> None:
        """Test that syntax errors are properly detected."""
        env = create_env("mbpp", split="train")
        env.reset(seed=10)

        syntax_error_code = "def broken_func(:\n    return 42"
        completion = _build_solution(syntax_error_code)

        _obs, reward, terminated, truncated, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        assert terminated is True
        assert info["syntax_valid"] is False
        assert info["success"] is False

    def test_multiple_problems_with_reference(self) -> None:
        """Test reference solutions across multiple problems."""
        env = create_env("mbpp", split="train")
        seeds = [10, 20, 30]
        results = []

        for seed in seeds:
            env.reset(seed=seed)
            payload = _pull_task(env)
            reference = str(payload["reference_solution"])

            completion = _build_solution(reference)
            _obs, reward, _terminated, _truncated, info = env.step(
                ChatMessage(role="assistant", content=completion)
            )

            results.append(
                {
                    "seed": seed,
                    "success": info["success"],
                    "passed": info["tests_passed"],
                    "total": info["tests_total"],
                    "reward": reward,
                }
            )

        # Most reference solutions should work (some may have dataset quirks)
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        assert success_rate >= 0.6, f"Too many reference solutions failed: {results}"


# =============================================================================
# HumanEval Environment Tests
# =============================================================================


class TestHumanEvalExecution:
    """Test HumanEval environment execution."""

    def test_basic_problem(self) -> None:
        """Test HumanEval environment with a simple problem."""
        env = create_env("humaneval")
        obs = env.reset(seed=0)

        # Check problem is loaded
        assert len(obs.messages) > 0
        assert obs.task_meta.get("task_id") is not None

        # Submit a simple solution
        code = """def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False"""

        completion = _build_solution(code, "Checking all pairs.")

        _obs, reward, terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        assert terminated is True
        assert info["has_code"] is True
        assert info["syntax_valid"] is True


# =============================================================================
# Direct Instantiation Tests
# =============================================================================


class TestDirectInstantiation:
    """Test direct environment instantiation patterns."""

    def test_mbpp_train_split(self) -> None:
        """Test MBPP with train split."""
        env = PythonCodeEnv(dataset="mbpp", split="train")
        obs = env.reset(seed=1)
        assert obs.task_meta.get("task_id") is not None

    def test_mbpp_validation_split(self) -> None:
        """Test MBPP with validation split."""
        env = PythonCodeEnv(dataset="mbpp", split="validation")
        obs = env.reset(seed=1)
        assert obs.task_meta.get("task_id") is not None

    def test_mbpp_test_split_raises(self) -> None:
        """Test that MBPP test split raises AssertionError."""
        with pytest.raises(AssertionError, match="does not exist"):
            PythonCodeEnv(dataset="mbpp", split="test")

    def test_humaneval_only_test_split(self) -> None:
        """Test that HumanEval only supports test split."""
        env = PythonCodeEnv(dataset="humaneval", split="test")
        obs = env.reset(seed=0)
        assert obs.task_meta.get("task_id") is not None

        with pytest.raises(ValueError, match="only supports"):
            PythonCodeEnv(dataset="humaneval", split="train")


# =============================================================================
# Security and Edge Cases
# =============================================================================


class TestSecurityAndEdgeCases:
    """Test security sandboxing and edge cases."""

    def test_code_execution_sandboxed(self) -> None:
        """Test that potentially dangerous code is sandboxed."""
        env = create_env("mbpp", split="train")
        env.reset(seed=10)

        # Try code that imports os (should still be contained)
        malicious_code = """import os
def malicious_func():
    return os.getcwd()"""

        completion = _build_solution(malicious_code)

        _obs, _reward, terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        # Should complete without crashing
        assert terminated is True

    def test_empty_completion_handled(self) -> None:
        """Test handling of empty completion."""
        env = create_env("mbpp", split="train")
        env.reset(seed=5)

        _obs, reward, terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content="")
        )

        assert terminated is True
        assert info["success"] is False
        assert info["has_code"] is False
        assert reward < 0.3

    def test_no_solution_tags_handled(self) -> None:
        """Test handling of completion without solution tags."""
        env = create_env("mbpp", split="train")
        env.reset(seed=5)

        _obs, reward, terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content="def foo(): return 42")
        )

        assert terminated is True
        assert info["success"] is False
        assert info["has_code"] is False


# =============================================================================
# Diverse Problem Types (from test_diverse_problems.py)
# =============================================================================


class TestDiverseProblemTypes:
    """Test execution across diverse MBPP problem types."""

    def test_diverse_problems_correct_vs_wrong(self) -> None:
        """Test correct reference solutions vs intentionally wrong solutions."""
        env = create_env("mbpp", split="train")

        test_seeds = [0, 5, 10, 15, 20]
        correct_results = []
        wrong_results = []

        for seed in test_seeds:
            # Test CORRECT reference solution
            env.reset(seed=seed)
            payload = _pull_task(env)
            reference = str(payload["reference_solution"])

            completion = _build_solution(reference)
            _obs, reward, _term, _trunc, info = env.step(
                ChatMessage(role="assistant", content=completion)
            )

            correct_results.append({"seed": seed, "success": info["success"], "reward": reward})

            # Test WRONG solution
            env.reset(seed=seed)
            wrong_code = "def wrong(*args, **kwargs): return None"
            completion = _build_solution(wrong_code)
            _obs, reward, _term, _trunc, info = env.step(
                ChatMessage(role="assistant", content=completion)
            )

            wrong_results.append({"seed": seed, "success": info["success"], "reward": reward})

        # Validation checks
        correct_success_rate = sum(1 for r in correct_results if r["success"]) / len(
            correct_results
        )
        wrong_success_rate = sum(1 for r in wrong_results if r["success"]) / len(wrong_results)
        avg_correct_reward = sum(r["reward"] for r in correct_results) / len(correct_results)
        avg_wrong_reward = sum(r["reward"] for r in wrong_results) / len(wrong_results)

        # Reference solutions should mostly pass
        assert correct_success_rate >= 0.6, f"Reference solutions: {correct_results}"

        # Wrong solutions should fail
        assert wrong_success_rate == 0.0, f"Wrong solutions passed: {wrong_results}"

        # Clear separation in rewards
        assert avg_correct_reward > avg_wrong_reward, (
            f"Correct avg: {avg_correct_reward:.3f}, Wrong avg: {avg_wrong_reward:.3f}"
        )
