"""Tests for PipelineConfig.from_env() parsing."""

from __future__ import annotations

import os
from unittest.mock import patch

from grail.mining.config import PipelineConfig


def test_defaults() -> None:
    """Default config has pipeline disabled."""
    cfg = PipelineConfig()
    assert cfg.enabled is False
    assert cfg.backend == "vllm"
    assert cfg.vllm_gpu == 0
    assert cfg.proof_gpu == 1
    assert cfg.gpu_memory_utilization == 0.90


def test_from_env_enabled() -> None:
    """GRAIL_PIPELINE_ENABLED=1 enables pipeline."""
    env = {"GRAIL_PIPELINE_ENABLED": "1"}
    with patch.dict(os.environ, env, clear=False):
        cfg = PipelineConfig.from_env()
    assert cfg.enabled is True


def test_from_env_backend_sglang() -> None:
    """GRAIL_PIPELINE_BACKEND=sglang selects SGLang backend."""
    env = {"GRAIL_PIPELINE_BACKEND": "sglang"}
    with patch.dict(os.environ, env, clear=False):
        cfg = PipelineConfig.from_env()
    assert cfg.backend == "sglang"


def test_from_env_gpu_ids() -> None:
    """GPU IDs parsed from environment."""
    env = {
        "GRAIL_PIPELINE_VLLM_GPU": "2",
        "GRAIL_PIPELINE_PROOF_GPU": "3",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = PipelineConfig.from_env()
    assert cfg.vllm_gpu == 2
    assert cfg.proof_gpu == 3


def test_from_env_float_params() -> None:
    """Float params parsed correctly."""
    env = {
        "GRAIL_PIPELINE_GPU_MEM_UTIL": "0.85",
        "GRAIL_PIPELINE_SERVER_TIMEOUT": "120.5",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = PipelineConfig.from_env()
    assert cfg.gpu_memory_utilization == 0.85
    assert cfg.server_timeout == 120.5


def test_from_env_empty_values_use_defaults() -> None:
    """Empty env values fall back to class defaults."""
    env = {
        "GRAIL_PIPELINE_ENABLED": "",
        "GRAIL_PIPELINE_BACKEND": "",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = PipelineConfig.from_env()
    assert cfg.enabled is False
    assert cfg.backend == "vllm"


def test_from_env_bool_variants() -> None:
    """Various true/false string representations."""
    for true_val in ("1", "true", "True", "TRUE", "yes", "YES"):
        env = {"GRAIL_PIPELINE_ENABLED": true_val}
        with patch.dict(os.environ, env, clear=False):
            cfg = PipelineConfig.from_env()
        assert cfg.enabled is True, f"Failed for {true_val!r}"

    for false_val in ("0", "false", "no", "nope"):
        env = {"GRAIL_PIPELINE_ENABLED": false_val}
        with patch.dict(os.environ, env, clear=False):
            cfg = PipelineConfig.from_env()
        assert cfg.enabled is False, f"Failed for {false_val!r}"
