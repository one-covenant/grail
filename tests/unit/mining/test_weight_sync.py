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

    @pytest.mark.asyncio
    async def test_start_passes_max_model_len_to_backend(
        self, config: PipelineConfig, mock_tokenizer: MagicMock
    ) -> None:
        """Regression: SGLangServerBackend must receive PipelineConfig.max_model_len.

        Without it, the backend's _cap_max_tokens clamp is a no-op and overlong
        prompts hit the server, surface as failed retries, and produce empty
        completions instead of clean clamping. Was a real bug in the Stage 1
        refactor before being caught in code review.
        """
        from unittest.mock import AsyncMock, patch

        cfg = PipelineConfig(
            backend="sglang",
            vllm_gpu=0,
            proof_gpu=1,
            max_model_len=12288,
            server_timeout=10.0,
            max_concurrent_requests=4,
        )
        sync = SGLangWeightSync(cfg, mock_tokenizer)

        fake_manager = MagicMock()
        fake_manager.__aenter__ = AsyncMock(return_value=fake_manager)
        fake_manager.start_server = AsyncMock()
        fake_manager.base_url = "http://localhost:30000"
        fake_manager.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        with (
            patch(
                "grail.trainer.inference_server.SGLangServerManager",
                return_value=fake_manager,
            ),
            patch("grail.environments.backends.SGLangServerBackend") as fake_backend_cls,
        ):
            await sync.start("/some/checkpoint/path")

        fake_backend_cls.assert_called_once()
        kwargs = fake_backend_cls.call_args.kwargs
        assert kwargs.get("max_model_len") == 12288, (
            f"SGLangServerBackend was constructed without max_model_len: kwargs={kwargs}"
        )


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
