"""Unit tests for Triton kernel environment: parser, rewards, env flow.

No GPU required. Run locally with pytest.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from grail.environments.core import ChatMessage
from grail.environments.gpu_kernel.env import TritonKernelEnv
from grail.environments.gpu_kernel.eval_backends import EvalResult
from grail.environments.gpu_kernel.parser import TritonKernelParser
from grail.environments.gpu_kernel.rewards import (
    TritonKernelRubric,
    _kernel_quality_reward,
    _solution_format_reward,
    _thinking_format_reward,
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

    def test_kernel_quality_incorrect(self) -> None:
        parsed: dict[str, Any] = {"exec_result": {"correct": False}}
        assert _kernel_quality_reward(parsed, {}) == 0.0

    def test_kernel_quality_correct_no_speedup(self) -> None:
        parsed: dict[str, Any] = {"exec_result": {"correct": True}}
        reward = _kernel_quality_reward(parsed, {})
        assert reward == pytest.approx(0.310, abs=0.01)

    def test_kernel_quality_correct_with_speedup(self) -> None:
        parsed: dict[str, Any] = {"exec_result": {"correct": True, "speedup_ratio": 2.0}}
        reward = _kernel_quality_reward(parsed, {})
        assert reward == pytest.approx(0.769, abs=0.01)

    def test_kernel_quality_speedup_clipped(self) -> None:
        parsed: dict[str, Any] = {"exec_result": {"correct": True, "speedup_ratio": 10.0}}
        reward = _kernel_quality_reward(parsed, {})
        # Clipped to 3.0, same as speedup_ratio=3.0
        expected: dict[str, Any] = {"exec_result": {"correct": True, "speedup_ratio": 3.0}}
        assert reward == pytest.approx(_kernel_quality_reward(expected, {}))
        assert reward == pytest.approx(0.900, abs=0.01)

    def test_kernel_quality_no_exec(self) -> None:
        parsed: dict[str, Any] = {}
        assert _kernel_quality_reward(parsed, {}) == 0.0

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

    def test_rubric_correct_no_speedup(self, rubric: TritonKernelRubric) -> None:
        """Correct kernel, no timing data -> ~0.448."""
        parsed: dict[str, Any] = {
            "has_solution": True,
            "trailing_after_solution": 0,
            "has_thinking": True,
            "exec_result": {"compiled": True, "correct": True},
        }
        reward, components = rubric.step_reward(parsed=parsed, context={}, turn_index=1)
        # 0.80 * sigmoid(-0.8) + 0.10 * 1.0 + 0.10 * 1.0
        assert reward == pytest.approx(0.448, abs=0.01)
        assert components["kernel_quality"] == pytest.approx(0.310, abs=0.01)

    def test_rubric_correct_with_speedup(self, rubric: TritonKernelRubric) -> None:
        """Correct kernel with 3x speedup -> ~0.920."""
        parsed: dict[str, Any] = {
            "has_solution": True,
            "trailing_after_solution": 0,
            "has_thinking": True,
            "exec_result": {"compiled": True, "correct": True, "speedup_ratio": 3.0},
        }
        reward, components = rubric.step_reward(parsed=parsed, context={}, turn_index=1)
        assert reward == pytest.approx(0.920, abs=0.01)
        assert components["kernel_quality"] == pytest.approx(0.900, abs=0.01)

    def test_rubric_no_gpu_eval(self, rubric: TritonKernelRubric) -> None:
        """No exec_result -> 0.20 (format + thinking only)."""
        parsed: dict[str, Any] = {
            "has_solution": True,
            "trailing_after_solution": 0,
            "has_thinking": True,
            "exec_result": None,
        }
        reward, components = rubric.step_reward(parsed=parsed, context={}, turn_index=1)
        assert reward == pytest.approx(0.20)
        assert components["kernel_quality"] == 0.0

    def test_rubric_zero_reward(self, rubric: TritonKernelRubric) -> None:
        """Empty completion -> 0.0."""
        parsed: dict[str, Any] = {
            "has_solution": False,
            "trailing_after_solution": 0,
            "has_thinking": False,
            "exec_result": None,
        }
        reward, _ = rubric.step_reward(parsed=parsed, context={}, turn_index=1)
        assert reward == pytest.approx(0.0)

    def test_rubric_incorrect_with_format(self, rubric: TritonKernelRubric) -> None:
        """Incorrect kernel + format + thinking -> 0.20."""
        parsed: dict[str, Any] = {
            "has_solution": True,
            "trailing_after_solution": 0,
            "has_thinking": True,
            "exec_result": {"compiled": True, "correct": False},
        }
        reward, components = rubric.step_reward(parsed=parsed, context={}, turn_index=1)
        assert reward == pytest.approx(0.20)
        assert components["kernel_quality"] == 0.0
        assert components["format"] == 1.0
        assert components["thinking"] == 1.0


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
        """gpu_eval=False -> format + thinking rewards only."""
        env = TritonKernelEnv(task_source=_FakeTaskSource(), gpu_eval=False)
        env.reset(seed=42)

        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)
        _, reward, terminated, _, info = env.step(ChatMessage(role="assistant", content=completion))

        assert terminated is True
        assert reward == pytest.approx(0.20)
        assert info["structure_valid"] is True
        assert info["exec_result"] is None

    def test_step_with_fake_backend(self) -> None:
        """gpu_eval=True + FakeEvalBackend correct=True -> sigmoid reward."""
        backend = FakeEvalBackend(default_result=EvalResult(correct=True, compiled=True))
        env = TritonKernelEnv(
            task_source=_FakeTaskSource(),
            gpu_eval=True,
            eval_backend=backend,
        )
        env.reset(seed=42)

        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)
        _, reward, _, _, info = env.step(ChatMessage(role="assistant", content=completion))

        # correct=True, no speedup_ratio -> 0.80 * 0.310 + 0.10 + 0.10 = 0.448
        assert reward == pytest.approx(0.448, abs=0.01)
        assert info["exec_result"]["correct"] is True
        assert len(backend.call_log) == 1

    def test_step_with_fake_backend_speedup(self) -> None:
        """gpu_eval=True + FakeEvalBackend with speedup -> higher reward."""
        backend = FakeEvalBackend(
            default_result=EvalResult(
                correct=True,
                compiled=True,
                speedup_ratio=2.0,
                kernel_median_ms=0.5,
                reference_median_ms=1.0,
            )
        )
        env = TritonKernelEnv(
            task_source=_FakeTaskSource(),
            gpu_eval=True,
            eval_backend=backend,
        )
        env.reset(seed=42)

        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)
        _, reward, _, _, info = env.step(ChatMessage(role="assistant", content=completion))

        # correct=True, speedup=2.0 -> 0.80 * 0.769 + 0.10 + 0.10 = 0.815
        assert reward == pytest.approx(0.815, abs=0.01)
        assert info["exec_result"]["speedup_ratio"] == 2.0

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
        with patch("grail.environments.gpu_kernel.env.get_global_backend", return_value=None):
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
        _, _, _, _, info = env.step(ChatMessage(role="assistant", content=completion))

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
            _, reward, _, _, _ = env.step(ChatMessage(role="assistant", content=completion))
            rewards.append(reward)

        assert all(r == rewards[0] for r in rewards)

    def test_info_dict_complete(self) -> None:
        """All expected keys present in info."""
        env = TritonKernelEnv(task_source=_FakeTaskSource(), gpu_eval=False)
        env.reset(seed=42)

        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)
        _, _, _, _, info = env.step(ChatMessage(role="assistant", content=completion))

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
