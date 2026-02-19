"""Unit tests for Triton kernel environment: parser, rewards, env flow.

No GPU required. Run locally with pytest.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from grail.environments.core import ChatMessage
from grail.environments.gpu_kernel.eval_backends import EvalResult
from grail.environments.gpu_kernel.env import TritonKernelEnv
from grail.environments.gpu_kernel.parser import TritonKernelParser
from grail.environments.gpu_kernel.rewards import (
    TritonKernelRubric,
    _compilation_reward,
    _correctness_reward,
    _gpu_compilation_reward,
    _solution_format_reward,
    _structure_reward,
    _thinking_format_reward,
    create_triton_kernel_reward_vector,
)
from grail.environments.providers import TaskSpec
from tests.fixtures.fakes import FakeEvalBackend
from tests.unit.environments.kernel_test_helpers import (
    MISSING_MODEL_NEW_CODE,
    SYNTAX_ERROR_CODE,
    VALID_TRITON_CODE,
    build_kernel_completion,
    make_task_payload,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def parser() -> TritonKernelParser:
    return TritonKernelParser()


@pytest.fixture(scope="module")
def rubric() -> TritonKernelRubric:
    return TritonKernelRubric()


# =============================================================================
# TestTritonKernelParser
# =============================================================================


class TestTritonKernelParser:
    """Parser extraction and validation."""

    def test_parse_full_completion(self, parser: TritonKernelParser) -> None:
        completion = build_kernel_completion("I will optimize this.", VALID_TRITON_CODE)
        parsed = parser.parse(completion, {})

        assert parsed["has_thinking"] is True
        assert parsed["has_solution"] is True
        assert parsed["code"] != ""
        assert parsed["syntax_valid"] is True
        assert parsed["structure_valid"] is True

    def test_parse_missing_solution(self, parser: TritonKernelParser) -> None:
        completion = "Just some text without solution tags."
        parsed = parser.parse(completion, {})

        assert parsed["has_solution"] is False
        assert parsed["code"] == ""

    def test_parse_empty_completion(self, parser: TritonKernelParser) -> None:
        parsed = parser.parse("", {})

        assert parsed["has_thinking"] is False
        assert parsed["has_solution"] is False
        assert parsed["code"] == ""
        assert parsed["syntax_valid"] is False

    @pytest.mark.parametrize(
        "code,expected_valid",
        [
            (VALID_TRITON_CODE, True),
            (MISSING_MODEL_NEW_CODE, False),
            (SYNTAX_ERROR_CODE, False),
            ("", False),
        ],
    )
    def test_structure_validation(
        self, parser: TritonKernelParser, code: str, expected_valid: bool
    ) -> None:
        completion = build_kernel_completion("thinking", code)
        parsed = parser.parse(completion, {})
        assert parsed["structure_valid"] is expected_valid

    def test_structure_partial(self, parser: TritonKernelParser) -> None:
        """Missing ModelNew but has triton imports and jit."""
        completion = build_kernel_completion("thinking", MISSING_MODEL_NEW_CODE)
        parsed = parser.parse(completion, {})

        assert parsed["has_model_new"] is False
        assert parsed["has_triton_jit"] is True
        assert parsed["has_triton_import"] is True
        assert parsed["structure_valid"] is False

    def test_syntax_error_detection(self, parser: TritonKernelParser) -> None:
        completion = build_kernel_completion("thinking", SYNTAX_ERROR_CODE)
        parsed = parser.parse(completion, {})
        assert parsed["syntax_valid"] is False

    def test_anti_hacking_signals(self) -> None:
        from grail.environments.gpu_kernel.parser import extract_anti_hacking_signals

        signals = extract_anti_hacking_signals(VALID_TRITON_CODE)
        assert isinstance(signals, dict)
        assert "has_try_except" in signals
        assert signals["has_try_except"] is False


# =============================================================================
# TestRewardComponents
# =============================================================================


class TestRewardComponents:
    """Individual reward functions."""

    def test_compilation_reward_valid_syntax(self) -> None:
        parsed: dict[str, Any] = {"syntax_valid": True}
        assert _compilation_reward(parsed, {}) == 1.0

    def test_compilation_reward_invalid_syntax(self) -> None:
        parsed: dict[str, Any] = {"syntax_valid": False}
        assert _compilation_reward(parsed, {}) == 0.0

    def test_structure_reward_all_present(self) -> None:
        parsed: dict[str, Any] = {
            "has_model_new": True,
            "has_triton_jit": True,
            "has_triton_import": True,
            "has_torch_import": True,
        }
        assert _structure_reward(parsed, {}) == pytest.approx(1.0)

    def test_structure_reward_partial(self) -> None:
        parsed: dict[str, Any] = {
            "has_model_new": True,
            "has_triton_jit": True,
            "has_triton_import": False,
            "has_torch_import": False,
        }
        assert _structure_reward(parsed, {}) == pytest.approx(0.5)

    def test_structure_reward_none_present(self) -> None:
        parsed: dict[str, Any] = {
            "has_model_new": False,
            "has_triton_jit": False,
            "has_triton_import": False,
            "has_torch_import": False,
        }
        assert _structure_reward(parsed, {}) == pytest.approx(0.0)

    def test_gpu_compilation_reward_compiled(self) -> None:
        parsed: dict[str, Any] = {"exec_result": {"compiled": True, "correct": False}}
        assert _gpu_compilation_reward(parsed, {}) == 1.0

    def test_gpu_compilation_reward_not_compiled(self) -> None:
        parsed: dict[str, Any] = {"exec_result": {"compiled": False, "correct": False}}
        assert _gpu_compilation_reward(parsed, {}) == 0.0

    def test_correctness_reward_correct(self) -> None:
        parsed: dict[str, Any] = {"exec_result": {"correct": True}}
        assert _correctness_reward(parsed, {}) == 1.0

    def test_correctness_reward_incorrect(self) -> None:
        parsed: dict[str, Any] = {"exec_result": {"correct": False}}
        assert _correctness_reward(parsed, {}) == 0.0

    def test_correctness_reward_no_exec(self) -> None:
        parsed: dict[str, Any] = {}
        assert _correctness_reward(parsed, {}) == 0.0

    def test_format_reward_proper_tags(self) -> None:
        parsed: dict[str, Any] = {"has_solution": True, "trailing_after_solution": 0}
        assert _solution_format_reward(parsed, {}) == 1.0

    def test_format_reward_excessive_trailing(self) -> None:
        parsed: dict[str, Any] = {"has_solution": True, "trailing_after_solution": 100}
        assert _solution_format_reward(parsed, {}) == 0.0

    def test_thinking_reward_present(self) -> None:
        parsed: dict[str, Any] = {"has_thinking": True}
        assert _thinking_format_reward(parsed, {}) == 1.0

    def test_thinking_reward_absent(self) -> None:
        parsed: dict[str, Any] = {"has_thinking": False}
        assert _thinking_format_reward(parsed, {}) == 0.0


# =============================================================================
# TestTritonKernelRubric
# =============================================================================


class TestTritonKernelRubric:
    """Rubric integration tests."""

    def test_rubric_full_reward_gpu_eval(self, rubric: TritonKernelRubric) -> None:
        """All components satisfied -> total near 1.0."""
        parsed: dict[str, Any] = {
            "syntax_valid": True,
            "has_model_new": True,
            "has_triton_jit": True,
            "has_triton_import": True,
            "has_torch_import": True,
            "has_solution": True,
            "trailing_after_solution": 0,
            "has_thinking": True,
            "exec_result": {"compiled": True, "correct": True},
        }
        reward, components = rubric.step_reward(parsed=parsed, context={}, turn_index=1)
        assert reward == pytest.approx(1.0)
        assert components["correctness"] == 1.0
        assert components["gpu_compilation"] == 1.0

    def test_rubric_structural_only(self, rubric: TritonKernelRubric) -> None:
        """No GPU eval -> max 0.35."""
        parsed: dict[str, Any] = {
            "syntax_valid": True,
            "has_model_new": True,
            "has_triton_jit": True,
            "has_triton_import": True,
            "has_torch_import": True,
            "has_solution": True,
            "trailing_after_solution": 0,
            "has_thinking": True,
            "exec_result": None,
        }
        reward, components = rubric.step_reward(parsed=parsed, context={}, turn_index=1)
        assert reward == pytest.approx(0.35)
        assert components["correctness"] == 0.0
        assert components["gpu_compilation"] == 0.0

    def test_rubric_zero_reward(self, rubric: TritonKernelRubric) -> None:
        """Empty completion -> 0.0."""
        parsed: dict[str, Any] = {
            "syntax_valid": False,
            "has_model_new": False,
            "has_triton_jit": False,
            "has_triton_import": False,
            "has_torch_import": False,
            "has_solution": False,
            "trailing_after_solution": 0,
            "has_thinking": False,
            "exec_result": None,
        }
        reward, _ = rubric.step_reward(parsed=parsed, context={}, turn_index=1)
        assert reward == pytest.approx(0.0)

    def test_rubric_weights_sum_to_one(self) -> None:
        rv = create_triton_kernel_reward_vector()
        assert sum(rv.weights) == pytest.approx(1.0)
        assert len(rv.reward_functions) == 6
        assert len(rv.weights) == 6


# =============================================================================
# TestTritonKernelEnv
# =============================================================================


class _FakeTaskSource:
    """Minimal TaskSource for unit tests."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload or make_task_payload()

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        return TaskSpec(
            id="test_task_0",
            payload=self._payload,
            metadata={"split": "test", "index": 0},
        )

    def size(self) -> int:
        return 1

    def iter_ids(self) -> list[str]:
        return ["test_task_0"]


class TestTritonKernelEnv:
    """Env with FakeEvalBackend (no GPU)."""

    def test_reset_returns_observation(self) -> None:
        env = TritonKernelEnv(task_source=_FakeTaskSource(), gpu_eval=False)
        obs = env.reset(seed=42)
        assert len(obs.messages) == 1
        assert obs.messages[0].role == "user"
        assert "Model" in obs.messages[0].content

    def test_step_structural_reward(self) -> None:
        """gpu_eval=False -> structural rewards only."""
        env = TritonKernelEnv(task_source=_FakeTaskSource(), gpu_eval=False)
        env.reset(seed=42)

        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)
        _, reward, terminated, _, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        assert terminated is True
        assert reward == pytest.approx(0.35)
        assert info["structure_valid"] is True
        assert info["exec_result"] is None

    def test_step_with_fake_backend(self) -> None:
        """gpu_eval=True + FakeEvalBackend -> uses backend."""
        backend = FakeEvalBackend(
            default_result=EvalResult(correct=True, compiled=True)
        )
        env = TritonKernelEnv(
            task_source=_FakeTaskSource(),
            gpu_eval=True,
            eval_backend=backend,
        )
        env.reset(seed=42)

        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)
        _, reward, _, _, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        assert reward == pytest.approx(1.0)
        assert info["exec_result"]["correct"] is True
        assert len(backend.call_log) == 1

    def test_step_backend_not_configured(self) -> None:
        """gpu_eval=True + no backend -> RuntimeError."""
        env = TritonKernelEnv(
            task_source=_FakeTaskSource(),
            gpu_eval=True,
            eval_backend=None,
        )
        env.reset(seed=42)

        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)

        # Mock out the global backend to be None
        with patch(
            "grail.environments.gpu_kernel.env.get_global_backend", return_value=None
        ):
            with pytest.raises(RuntimeError, match="no eval backend configured"):
                env.step(ChatMessage(role="assistant", content=completion))

    def test_step_skips_gpu_when_invalid(self) -> None:
        """Invalid structure -> skips backend call."""
        backend = FakeEvalBackend()
        env = TritonKernelEnv(
            task_source=_FakeTaskSource(),
            gpu_eval=True,
            eval_backend=backend,
        )
        env.reset(seed=42)

        # Missing ModelNew -> structure_valid=False -> no GPU eval
        completion = build_kernel_completion("thinking", MISSING_MODEL_NEW_CODE)
        _, _, _, _, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        assert info["exec_result"] is None
        assert len(backend.call_log) == 0

    def test_deterministic_rewards(self) -> None:
        """Same (seed, completion) -> same reward."""
        source = _FakeTaskSource()
        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)

        rewards = []
        for _ in range(3):
            env = TritonKernelEnv(task_source=source, gpu_eval=False)
            env.reset(seed=42)
            _, reward, _, _, _ = env.step(
                ChatMessage(role="assistant", content=completion)
            )
            rewards.append(reward)

        assert all(r == rewards[0] for r in rewards)

    def test_info_dict_complete(self) -> None:
        """All expected keys present in info."""
        env = TritonKernelEnv(task_source=_FakeTaskSource(), gpu_eval=False)
        env.reset(seed=42)

        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)
        _, _, _, _, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )

        expected_keys = {
            "reward_components",
            "termination_cause",
            "success",
            "has_code",
            "syntax_valid",
            "structure_valid",
            "has_model_new",
            "has_triton_jit",
            "gpu_eval",
            "exec_result",
            "hacking_signals",
        }
        assert expected_keys.issubset(info.keys())


# =============================================================================
# TestGpuConfigValidation
# =============================================================================


class TestGpuConfigValidation:
    """GPU configuration validation."""

    def test_validate_no_gpu_eval(self) -> None:
        """gpu_eval=False -> no validation needed."""
        from grail.environments.gpu_kernel.eval_backends import validate_gpu_config

        # Should not raise
        validate_gpu_config([], gpu_eval=False)

    def test_validate_no_cuda_raises(self) -> None:
        """gpu_eval=True + no CUDA -> RuntimeError."""
        from grail.environments.gpu_kernel.eval_backends import validate_gpu_config

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                validate_gpu_config([0], gpu_eval=True)

    def test_validate_invalid_gpu_id_raises(self) -> None:
        """GPU ID >= device_count -> RuntimeError."""
        from grail.environments.gpu_kernel.eval_backends import validate_gpu_config

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_name", return_value="MockGPU"),
        ):
            with pytest.raises(RuntimeError, match="only 2 GPUs detected"):
                validate_gpu_config([5], gpu_eval=True)

    def test_validate_no_gpu_ids_raises(self) -> None:
        """gpu_eval=True + empty gpu_ids -> RuntimeError."""
        from grail.environments.gpu_kernel.eval_backends import validate_gpu_config

        with patch("torch.cuda.is_available", return_value=True):
            with pytest.raises(RuntimeError, match="no KERNEL_EVAL_GPU_IDS"):
                validate_gpu_config([], gpu_eval=True)

    def test_error_message_actionable(self) -> None:
        """Error includes fix suggestions."""
        from grail.environments.gpu_kernel.eval_backends import validate_gpu_config

        with patch("torch.cuda.is_available", return_value=False):
            try:
                validate_gpu_config([0], gpu_eval=True)
            except RuntimeError as e:
                assert "gpu_eval=False" in str(e) or "modal" in str(e).lower()
