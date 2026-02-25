"""Backend-agnostic weight sync for pipelined mining.

Provides a strategy interface with two implementations:
- SGLangWeightSync: zero-downtime via /update_weights_from_disk
- VLLMWeightSync: sleep mode level 2 + restart
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class WeightSyncStrategy(ABC):
    """Abstract interface for generation-server weight synchronization."""

    @abstractmethod
    async def start(self, checkpoint_path: str) -> None:
        """Launch the generation server with initial weights."""

    @abstractmethod
    async def sync_weights(self, checkpoint_path: str) -> None:
        """Reload weights from disk into the running server."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Stop the generation server and release resources."""

    @abstractmethod
    def get_backend(self) -> Any:
        """Return a TextGenBackend instance connected to the server."""


class SGLangWeightSync(WeightSyncStrategy):
    """Zero-downtime weight sync via /update_weights_from_disk."""

    def __init__(self, config: PipelineConfig, tokenizer: Any) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._manager: Any | None = None
        self._backend: Any | None = None

    async def start(self, checkpoint_path: str) -> None:
        from ..environments.loop import SGLangServerBackend
        from ..trainer.config import EvalConfig
        from ..trainer.inference_server import ServerConfig, SGLangServerManager

        eval_config = EvalConfig(
            sglang_mem_fraction_static=self._config.gpu_memory_utilization,
            sglang_context_length=self._config.max_model_len,
            sglang_max_running_requests=self._config.max_num_seqs,
            sglang_max_concurrent_requests=self._config.max_concurrent_requests,
            server_timeout=self._config.server_timeout,
            stream_server_logs=True,
        )

        server_config = ServerConfig(
            model_path=checkpoint_path,
            timeout_s=self._config.server_timeout,
            env={"CUDA_VISIBLE_DEVICES": str(self._config.vllm_gpu)},
        )

        self._manager = SGLangServerManager(config=server_config, eval_config=eval_config)
        await self._manager.__aenter__()
        await self._manager.start_server()

        self._backend = SGLangServerBackend(
            base_url=self._manager.base_url,
            model_name=self._manager.model_name,
            tokenizer=self._tokenizer,
            timeout=self._config.server_timeout,
            max_concurrent_requests=self._config.max_concurrent_requests,
        )

        logger.info("SGLang generation server started at %s", self._manager.base_url)

    async def sync_weights(self, checkpoint_path: str) -> None:
        if self._manager is None:
            raise RuntimeError("SGLang server not started")

        import httpx

        base_url = self._manager.base_url
        logger.info("Syncing weights to SGLang server from %s", checkpoint_path)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/update_weights_from_disk",
                json={"model_path": str(checkpoint_path)},
                timeout=self._config.server_timeout,
            )
            resp.raise_for_status()

        logger.info("SGLang weight sync complete")

    async def shutdown(self) -> None:
        if self._manager is not None:
            try:
                await self._manager.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Error shutting down SGLang server: %s", exc)
            self._manager = None
        self._backend = None

    def get_backend(self) -> Any:
        if self._backend is None:
            raise RuntimeError("SGLang backend not initialized — call start() first")
        return self._backend


class VLLMWeightSync(WeightSyncStrategy):
    """Weight sync via sleep mode level 2 + restart."""

    def __init__(self, config: PipelineConfig, tokenizer: Any) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._manager: Any | None = None
        self._backend: Any | None = None

    @staticmethod
    def _is_hf_model_id(name: str) -> bool:
        """Check if *name* looks like a HuggingFace model ID (``org/model``).

        Checkpoint metadata stores internal names like ``"async_trainer_snapshot"``
        which are not valid HF repo IDs and must not be passed as ``--tokenizer``
        to vLLM — doing so triggers a 404 RepositoryNotFoundError.
        """
        return "/" in name

    def _resolve_tokenizer_name(self, checkpoint_path: str) -> str | None:
        """Derive tokenizer HF model ID from checkpoint metadata.

        Only returns the metadata ``model_name`` when it is a valid HuggingFace
        repo ID (e.g. ``Qwen/Qwen3-8B``).  Internal names like
        ``"async_trainer_snapshot"`` are ignored so we fall through to the
        ``GRAIL_TRAIN_MODEL_ID`` env-var fallback.
        """
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                name = meta.get("model_name")
                if name and self._is_hf_model_id(name):
                    return name
            except Exception:
                pass
        return os.environ.get("GRAIL_TRAIN_MODEL_ID")

    async def start(self, checkpoint_path: str) -> None:
        from ..environments.loop import VLLMServerBackend
        from ..trainer.config import EvalConfig
        from ..trainer.inference_server import ServerConfig, VLLMServerManager

        eval_config = EvalConfig(
            vllm_gpu_memory_utilization=self._config.gpu_memory_utilization,
            vllm_max_model_len=self._config.max_model_len,
            vllm_max_num_seqs=self._config.max_num_seqs,
            vllm_max_concurrent_requests=self._config.max_concurrent_requests,
            server_timeout=self._config.server_timeout,
            stream_server_logs=True,
        )

        # Use the HF model ID for tokenizer to avoid huggingface_hub repo_id
        # validation errors with local checkpoint paths (huggingface_hub >= 0.36)
        tokenizer_name = self._resolve_tokenizer_name(checkpoint_path)

        server_config = ServerConfig(
            model_path=checkpoint_path,
            timeout_s=self._config.server_timeout,
            tokenizer_name=tokenizer_name,
            env={
                "CUDA_VISIBLE_DEVICES": str(self._config.vllm_gpu),
                # Disable expandable_segments to avoid conflict with vLLM memory pool
                "PYTORCH_CUDA_ALLOC_CONF": "",
            },
        )

        self._manager = VLLMServerManager(
            config=server_config,
            eval_config=eval_config,
        )
        await self._manager.__aenter__()
        await self._manager.start_server()

        self._backend = VLLMServerBackend(
            base_url=self._manager.base_url,
            model_name=self._manager.model_name,
            tokenizer=self._tokenizer,
            timeout=self._config.server_timeout,
            max_concurrent_requests=self._config.max_concurrent_requests,
            strict_token_ids=True,
        )

        logger.info("vLLM generation server started at %s", self._manager.base_url)

    async def sync_weights(self, checkpoint_path: str) -> None:
        if self._manager is None:
            raise RuntimeError("vLLM server not started")

        import httpx

        base_url = self._manager.base_url
        logger.info("Syncing weights to vLLM server from %s", checkpoint_path)

        # 1. Put server to sleep (releases GPU memory instantly)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{base_url}/sleep",
                    params={"level": 2},
                    timeout=30.0,
                )
        except Exception as exc:
            logger.warning("Sleep request failed (may be unsupported): %s", exc)

        # 2. Update tokenizer identity and reload with new checkpoint
        self._manager._config.tokenizer_name = self._resolve_tokenizer_name(checkpoint_path)
        await self._manager.reload_with_new_checkpoint(checkpoint_path)

        # 3. Recreate backend with updated base_url
        from ..environments.loop import VLLMServerBackend

        self._backend = VLLMServerBackend(
            base_url=self._manager.base_url,
            model_name=self._manager.model_name,
            tokenizer=self._tokenizer,
            timeout=self._config.server_timeout,
            max_concurrent_requests=self._config.max_concurrent_requests,
            strict_token_ids=True,
        )

        logger.info("vLLM weight sync complete")

    async def shutdown(self) -> None:
        if self._manager is not None:
            try:
                await self._manager.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Error shutting down vLLM server: %s", exc)
            self._manager = None
        self._backend = None

    def get_backend(self) -> Any:
        if self._backend is None:
            raise RuntimeError("vLLM backend not initialized — call start() first")
        return self._backend
