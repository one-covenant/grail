"""Tests for SGLangWeightSync and VLLMWeightSync with mocked HTTP."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from grail.mining.config import PipelineConfig
from grail.mining.weight_sync import SGLangWeightSync, VLLMWeightSync


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(
        enabled=True,
        backend="vllm",
        vllm_gpu=0,
        proof_gpu=1,
        server_timeout=10.0,
        max_concurrent_requests=4,
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    return tok


class TestSGLangWeightSync:
    """Test SGLangWeightSync with mocked server manager."""

    def test_init(self, config: PipelineConfig, mock_tokenizer: MagicMock) -> None:
        sync = SGLangWeightSync(config, mock_tokenizer)
        assert sync._config is config

    def test_get_backend_raises_before_start(
        self, config: PipelineConfig, mock_tokenizer: MagicMock
    ) -> None:
        sync = SGLangWeightSync(config, mock_tokenizer)
        with pytest.raises(RuntimeError, match="not initialized"):
            sync.get_backend()

    @pytest.mark.asyncio
    async def test_sync_weights_raises_before_start(
        self, config: PipelineConfig, mock_tokenizer: MagicMock
    ) -> None:
        sync = SGLangWeightSync(config, mock_tokenizer)
        with pytest.raises(RuntimeError, match="not started"):
            await sync.sync_weights("/some/path")

    @pytest.mark.asyncio
    async def test_shutdown_no_op_when_not_started(
        self, config: PipelineConfig, mock_tokenizer: MagicMock
    ) -> None:
        sync = SGLangWeightSync(config, mock_tokenizer)
        # Should not raise
        await sync.shutdown()
        assert sync._manager is None


class TestVLLMWeightSync:
    """Test VLLMWeightSync with mocked server manager."""

    def test_init(self, config: PipelineConfig, mock_tokenizer: MagicMock) -> None:
        sync = VLLMWeightSync(config, mock_tokenizer)
        assert sync._config is config

    def test_get_backend_raises_before_start(
        self, config: PipelineConfig, mock_tokenizer: MagicMock
    ) -> None:
        sync = VLLMWeightSync(config, mock_tokenizer)
        with pytest.raises(RuntimeError, match="not initialized"):
            sync.get_backend()

    @pytest.mark.asyncio
    async def test_sync_weights_raises_before_start(
        self, config: PipelineConfig, mock_tokenizer: MagicMock
    ) -> None:
        sync = VLLMWeightSync(config, mock_tokenizer)
        with pytest.raises(RuntimeError, match="not started"):
            await sync.sync_weights("/some/path")

    @pytest.mark.asyncio
    async def test_shutdown_no_op_when_not_started(
        self, config: PipelineConfig, mock_tokenizer: MagicMock
    ) -> None:
        sync = VLLMWeightSync(config, mock_tokenizer)
        await sync.shutdown()
        assert sync._manager is None

    def test_resolve_tokenizer_from_metadata(
        self, config: PipelineConfig, mock_tokenizer: MagicMock, tmp_path: object
    ) -> None:
        ckpt = tmp_path / "ckpt"  # type: ignore[operator]
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(json.dumps({"model_name": "Qwen/Qwen3-8B"}))
        sync = VLLMWeightSync(config, mock_tokenizer)
        assert sync._resolve_tokenizer_name(str(ckpt)) == "Qwen/Qwen3-8B"

    def test_resolve_tokenizer_skips_no_name(
        self,
        config: PipelineConfig,
        mock_tokenizer: MagicMock,
        tmp_path: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ckpt = tmp_path / "ckpt"  # type: ignore[operator]
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(json.dumps({"model_name": "no_name"}))
        monkeypatch.setenv("GRAIL_TRAIN_MODEL_ID", "fallback/model")
        sync = VLLMWeightSync(config, mock_tokenizer)
        assert sync._resolve_tokenizer_name(str(ckpt)) == "fallback/model"

    def test_resolve_tokenizer_falls_back_to_env(
        self,
        config: PipelineConfig,
        mock_tokenizer: MagicMock,
        tmp_path: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ckpt = tmp_path / "ckpt"  # type: ignore[operator]
        ckpt.mkdir()
        # No metadata.json
        monkeypatch.setenv("GRAIL_TRAIN_MODEL_ID", "env/model-id")
        sync = VLLMWeightSync(config, mock_tokenizer)
        assert sync._resolve_tokenizer_name(str(ckpt)) == "env/model-id"

    def test_resolve_tokenizer_returns_none_when_nothing(
        self,
        config: PipelineConfig,
        mock_tokenizer: MagicMock,
        tmp_path: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ckpt = tmp_path / "ckpt"  # type: ignore[operator]
        ckpt.mkdir()
        monkeypatch.delenv("GRAIL_TRAIN_MODEL_ID", raising=False)
        sync = VLLMWeightSync(config, mock_tokenizer)
        assert sync._resolve_tokenizer_name(str(ckpt)) is None
