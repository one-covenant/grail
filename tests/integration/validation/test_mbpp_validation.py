"""Integration tests for MBPP validation pipeline.

Verifies that the MBPP/Python code environment adapter integrates
correctly with the validation system.
"""

from __future__ import annotations

import pytest

from grail.environments.registry import PythonCodeEnvAdapter, get_adapter

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _ensure_deps() -> None:
    """Skip tests if required dependencies are missing."""
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer() -> object:
    """Load tokenizer for testing."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def mbpp_adapter() -> PythonCodeEnvAdapter:
    """Get MBPP adapter from registry."""
    adapter = get_adapter("mbpp")
    assert isinstance(adapter, PythonCodeEnvAdapter)
    return adapter


# =============================================================================
# Adapter Integration Tests
# =============================================================================


class TestMBPPAdapterIntegration:
    """Test MBPP adapter integration with validation pipeline."""

    def test_adapter_registered(self) -> None:
        """Test that MBPP adapter is in registry."""
        adapter = get_adapter("mbpp")
        assert adapter is not None
        assert isinstance(adapter, PythonCodeEnvAdapter)

    def test_humaneval_adapter_registered(self) -> None:
        """Test that HumanEval adapter is in registry."""
        adapter = get_adapter("humaneval")
        assert adapter is not None
        assert isinstance(adapter, PythonCodeEnvAdapter)

    def test_python_code_alias(self) -> None:
        """Test that python_code is an alias for mbpp."""
        adapter = get_adapter("python_code")
        assert isinstance(adapter, PythonCodeEnvAdapter)
        assert adapter.dataset == "mbpp"


class TestBuildPromptIds:
    """Test prompt ID generation for validation."""

    def test_build_prompt_ids(self, mbpp_adapter: PythonCodeEnvAdapter, tokenizer: object) -> None:
        """Test that build_prompt_ids generates valid token sequences."""
        seed = 42
        prompt_ids = mbpp_adapter.build_prompt_ids(seed, tokenizer)  # type: ignore[arg-type]

        assert isinstance(prompt_ids, list)
        assert len(prompt_ids) > 0
        assert all(isinstance(t, int) for t in prompt_ids)

    def test_prompt_ids_deterministic(
        self, mbpp_adapter: PythonCodeEnvAdapter, tokenizer: object
    ) -> None:
        """Test that same seed produces identical prompt."""
        seed = 42
        ids1 = mbpp_adapter.build_prompt_ids(seed, tokenizer)  # type: ignore[arg-type]
        ids2 = mbpp_adapter.build_prompt_ids(seed, tokenizer)  # type: ignore[arg-type]

        assert ids1 == ids2

    def test_different_seeds_different_prompts(
        self, mbpp_adapter: PythonCodeEnvAdapter, tokenizer: object
    ) -> None:
        """Test that different seeds produce different prompts."""
        ids1 = mbpp_adapter.build_prompt_ids(42, tokenizer)  # type: ignore[arg-type]
        ids2 = mbpp_adapter.build_prompt_ids(100, tokenizer)  # type: ignore[arg-type]

        assert ids1 != ids2


class TestEvaluateCompletion:
    """Test completion evaluation through adapter."""

    def test_evaluate_correct_solution(
        self, mbpp_adapter: PythonCodeEnvAdapter, tokenizer: object
    ) -> None:
        """Test evaluation of correct solution."""
        from grail.environments import create_env

        # Get reference solution for seed
        seed = 42
        env = create_env("mbpp", split="train")
        env.reset(seed=seed)
        payload = env._task.payload
        reference = payload["reference_solution"]

        completion = f"<SOLUTION>\n{reference}\n</SOLUTION>"

        result = mbpp_adapter.evaluate_completion(seed, completion, tokenizer)  # type: ignore[arg-type]

        assert result["success"] is True
        assert result["reward"] >= 0.7
        assert result["tests_passed"] == result["tests_total"]
        assert result["tests_total"] > 0

    def test_evaluate_wrong_solution(
        self, mbpp_adapter: PythonCodeEnvAdapter, tokenizer: object
    ) -> None:
        """Test evaluation of wrong solution."""
        seed = 42
        wrong_completion = "<SOLUTION>\ndef wrong(): return 42\n</SOLUTION>"

        result = mbpp_adapter.evaluate_completion(seed, wrong_completion, tokenizer)  # type: ignore[arg-type]

        assert result["success"] is False
        assert result["reward"] < 0.7

    def test_evaluate_includes_diagnostics(
        self, mbpp_adapter: PythonCodeEnvAdapter, tokenizer: object
    ) -> None:
        """Test that evaluation result includes diagnostic info."""
        seed = 10
        completion = "<SOLUTION>\ndef foo(): return 1\n</SOLUTION>"

        result = mbpp_adapter.evaluate_completion(seed, completion, tokenizer)  # type: ignore[arg-type]

        # Check diagnostic fields are present
        assert "syntax_valid" in result
        assert "has_code" in result
        assert "tests_passed" in result
        assert "tests_total" in result


# =============================================================================
# Validator Compatibility
# =============================================================================


class TestValidatorCompatibility:
    """Verify validators can use the adapter correctly."""

    def test_validators_import(self) -> None:
        """Test that all required validators can be imported."""
        from grail.validation.validators import (  # noqa: F401
            EnvironmentEvaluationValidator,
            EnvironmentPromptValidator,
            RewardValidator,
        )

    def test_validators_use_adapter_pattern(self) -> None:
        """Verify validators follow adapter pattern."""
        import inspect

        from grail.validation.validators import (
            EnvironmentEvaluationValidator,
            EnvironmentPromptValidator,
        )

        # Check source code uses get_adapter
        prompt_source = inspect.getsource(EnvironmentPromptValidator.validate)
        eval_source = inspect.getsource(EnvironmentEvaluationValidator.validate)

        assert "get_adapter" in prompt_source, "EnvironmentPromptValidator should use get_adapter"
        assert "get_adapter" in eval_source, "EnvironmentEvaluationValidator should use get_adapter"


# =============================================================================
# Cross-Dataset Consistency
# =============================================================================


class TestCrossDatasetConsistency:
    """Test consistency across MBPP and HumanEval adapters."""

    def test_both_adapters_have_same_interface(self, tokenizer: object) -> None:
        """Test that MBPP and HumanEval adapters have identical interface."""
        mbpp = get_adapter("mbpp")
        humaneval = get_adapter("humaneval")

        # Both should support the same methods
        assert hasattr(mbpp, "build_prompt_ids")
        assert hasattr(mbpp, "evaluate_completion")
        assert hasattr(humaneval, "build_prompt_ids")
        assert hasattr(humaneval, "evaluate_completion")

        # Both should generate valid prompts
        mbpp_ids = mbpp.build_prompt_ids(0, tokenizer)  # type: ignore[arg-type]
        humaneval_ids = humaneval.build_prompt_ids(0, tokenizer)  # type: ignore[arg-type]

        assert len(mbpp_ids) > 0
        assert len(humaneval_ids) > 0
