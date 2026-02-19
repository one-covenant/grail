"""Unit tests for factory routing and registry adapters for kernel envs.

No GPU required.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from grail.environments.factory import clear_task_source_cache, create_env
from grail.environments.gpu_kernel.env import TritonKernelEnv
from grail.environments.providers import TaskSpec
from grail.environments.registry import get_adapter
from tests.fixtures.fakes import DummyTokenizer, FakeEvalBackend
from tests.unit.environments.kernel_test_helpers import (
    VALID_TRITON_CODE,
    build_kernel_completion,
    make_task_payload,
)


# =============================================================================
# Helpers
# =============================================================================


class _MockKernelBenchTaskSource:
    """Fake KernelBench source that doesn't download from HF."""

    def __init__(self, **kwargs: object) -> None:
        self._payload = make_task_payload()

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        return TaskSpec(
            id="mock_kb_0",
            payload=self._payload,
            metadata={"split": "train", "index": 0, "level": 1},
        )

    def size(self) -> int:
        return 1

    def iter_ids(self) -> list[str]:
        return ["mock_kb_0"]


# =============================================================================
# TestFactoryRouting
# =============================================================================


class TestFactoryRouting:
    """create_env() for kernel env_ids."""

    def setup_method(self) -> None:
        clear_task_source_cache()

    def test_create_triton_kernel_env(self) -> None:
        """env_id='triton_kernel' -> TritonKernelEnv."""
        source = _MockKernelBenchTaskSource()
        env = create_env("triton_kernel", task_source=source)
        assert isinstance(env, TritonKernelEnv)

    def test_kernelbench_env_id_removed(self) -> None:
        """env_id='kernelbench' is no longer valid."""
        source = _MockKernelBenchTaskSource()
        with pytest.raises(ValueError, match="Unknown environment ID"):
            create_env("kernelbench", task_source=source)

    def test_gpu_eval_from_env_params(self) -> None:
        """env_params={'gpu_eval': True} propagated."""
        source = _MockKernelBenchTaskSource()
        env = create_env(
            "triton_kernel",
            task_source=source,
            env_params={"gpu_eval": True},
        )
        assert isinstance(env, TritonKernelEnv)
        assert env._gpu_eval is True

    def test_default_gpu_eval_false(self) -> None:
        """No env_params -> gpu_eval=False (when env var not set)."""
        source = _MockKernelBenchTaskSource()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GRAIL_GPU_EVAL", None)
            env = create_env("triton_kernel", task_source=source)
            assert isinstance(env, TritonKernelEnv)
            assert env._gpu_eval is False

    def test_dataset_path_uses_unified_source(self) -> None:
        """env_params with dataset_path -> UnifiedKernelTaskSource."""
        from grail.environments.gpu_kernel.task_sources import UnifiedKernelTaskSource

        # Create a minimal JSONL file
        rows = [
            {
                "id": f"test_{i}",
                "source": "production_repos",
                "prompt": f"Optimize kernel {i}",
                "pytorch_reference": "class Model:\n  pass",
                "test_code": "def check_correctness(m): return True",
                "difficulty": "medium",
                "category": "elementwise",
            }
            for i in range(3)
        ]
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        try:
            UnifiedKernelTaskSource._cache.clear()
            env = create_env(
                "triton_kernel",
                env_params={"dataset_path": path},
            )
            assert isinstance(env, TritonKernelEnv)
        finally:
            os.unlink(path)
            UnifiedKernelTaskSource._cache.clear()

    def test_eval_backend_from_env_params(self) -> None:
        """eval_backend passed through env_params."""
        source = _MockKernelBenchTaskSource()
        backend = FakeEvalBackend()
        env = create_env(
            "triton_kernel",
            task_source=source,
            env_params={"gpu_eval": True, "eval_backend": backend},
        )
        assert isinstance(env, TritonKernelEnv)
        assert env._eval_backend is backend


# =============================================================================
# TestRegistryAdapters
# =============================================================================


class TestRegistryAdapters:
    """EnvAdapter for kernel envs."""

    def test_triton_kernel_adapter_registered(self) -> None:
        adapter = get_adapter("triton_kernel")
        assert adapter is not None

    def test_kernelbench_adapter_removed(self) -> None:
        """'kernelbench' is no longer a registered env_id."""
        with pytest.raises(KeyError):
            get_adapter("kernelbench")

    def test_unified_kernel_adapter_removed(self) -> None:
        """'unified_kernel' is no longer a registered env_id."""
        with pytest.raises(KeyError):
            get_adapter("unified_kernel")

    def test_evaluate_completion_structural(self) -> None:
        """gpu_eval=False -> structural reward."""
        adapter = get_adapter("triton_kernel")

        # create_env is imported locally in the adapter method via:
        #   from .factory import create_env
        # so we patch at the factory module level
        source = _MockKernelBenchTaskSource()
        mock_env = TritonKernelEnv(task_source=source, gpu_eval=False)

        with patch("grail.environments.factory.create_env", return_value=mock_env):
            tokenizer = DummyTokenizer()
            completion = build_kernel_completion("thinking", VALID_TRITON_CODE)
            result = adapter.evaluate_completion(42, completion, tokenizer)

            assert "reward" in result
            assert "success" in result
            assert isinstance(result["reward"], float)
            assert result["reward"] == pytest.approx(0.35)

    def test_evaluate_completion_deterministic(self) -> None:
        """Same inputs -> same reward."""
        adapter = get_adapter("triton_kernel")
        source = _MockKernelBenchTaskSource()

        tokenizer = DummyTokenizer()
        completion = build_kernel_completion("thinking", VALID_TRITON_CODE)

        rewards = []
        for _ in range(3):
            mock_env = TritonKernelEnv(task_source=source, gpu_eval=False)
            with patch("grail.environments.factory.create_env", return_value=mock_env):
                result = adapter.evaluate_completion(42, completion, tokenizer)
                rewards.append(result["reward"])

        assert all(r == rewards[0] for r in rewards)
