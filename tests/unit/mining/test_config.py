"""Tests for PipelineConfig.from_env() parsing."""

from __future__ import annotations

import os
from unittest.mock import patch

from grail.mining.config import PipelineConfig


def test_defaults() -> None:
    """Default config: SGLang on GPU 0, HF proof on GPU 1.

    SGLang is the production-validated mining backend (Stage 1 e2e); the
    default flipped from vllm to sglang along with the legacy-path removal.
    """
    cfg = PipelineConfig()
    assert cfg.backend == "sglang"
    assert cfg.vllm_gpu == 0
    assert cfg.proof_gpu == 1
    assert cfg.gpu_memory_utilization == 0.90


def test_from_env_backend_vllm() -> None:
    """GRAIL_PIPELINE_BACKEND=vllm overrides the sglang default."""
    env = {"GRAIL_PIPELINE_BACKEND": "vllm"}
    with patch.dict(os.environ, env, clear=False):
        cfg = PipelineConfig.from_env()
    assert cfg.backend == "vllm"


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
        "GRAIL_PIPELINE_BACKEND": "",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = PipelineConfig.from_env()
    assert cfg.backend == "sglang"
