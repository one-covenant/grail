"""Backend-agnostic weight sync for pipelined mining.

Provides a strategy interface with two implementations:
- SGLangWeightSync: zero-downtime via /update_weights_from_disk
- VLLMWeightSync: zero-downtime via sleep/wake/reload_weights API
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from .config import PipelineConfig

logger = logging.getLogger(__name__)

_SYMLINK_NAME = "vllm_active"


def _update_symlink(symlink_path: str, target: str) -> None:
    """Atomically update *symlink_path* to point to *target*.

    Uses create-tmp + rename to guarantee atomicity on POSIX filesystems.
    """
    tmp = f"{symlink_path}.tmp.{os.getpid()}"
    try:
        os.symlink(target, tmp)
        os.rename(tmp, symlink_path)
    except BaseException:
        # Clean up the tmp symlink if rename failed
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    """Zero-downtime weight sync via vLLM sleep/wake/reload API.

    Uses a stable symlink as the ``--model`` path so that vLLM's
    ``reload_weights`` (which always re-reads the original model path)
    picks up the new checkpoint after an atomic symlink update.
    """

    def __init__(self, config: PipelineConfig, tokenizer: Any) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._manager: Any | None = None
        self._backend: Any | None = None
        self._symlink_path: str | None = None

    # ------------------------------------------------------------------
    # Tokenizer resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_hf_model_id(name: str) -> bool:
        """Check if *name* looks like a HuggingFace model ID (``org/model``)."""
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

    # ------------------------------------------------------------------
    # Symlink management
    # ------------------------------------------------------------------

    def _resolve_symlink_dir(self, checkpoint_path: str) -> str:
        """Return the directory where the vLLM model symlink should live.

        Uses ``self._config.symlink_dir`` if set (e.g. an ephemeral/tmpfs
        mount for machines with limited main disk), otherwise falls back to
        the checkpoint's parent directory.
        """
        if self._config.symlink_dir:
            os.makedirs(self._config.symlink_dir, exist_ok=True)
            return self._config.symlink_dir
        return os.path.dirname(checkpoint_path)

    def _init_symlink(self, checkpoint_path: str) -> str:
        """Create the stable symlink pointing to *checkpoint_path*.

        Returns the symlink path for use as the vLLM ``--model`` argument.
        """
        symlink_dir = self._resolve_symlink_dir(checkpoint_path)
        symlink_path = os.path.join(symlink_dir, _SYMLINK_NAME)
        _update_symlink(symlink_path, checkpoint_path)
        logger.info("Created vLLM model symlink: %s -> %s", symlink_path, checkpoint_path)
        return symlink_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, checkpoint_path: str) -> None:
        from ..environments.loop import VLLMServerBackend
        from ..trainer.config import EvalConfig
        from ..trainer.inference_server import ServerConfig, VLLMServerManager

        # Create stable symlink so vLLM's reload_weights re-reads our target
        self._symlink_path = self._init_symlink(checkpoint_path)

        eval_config = EvalConfig(
            vllm_gpu_memory_utilization=self._config.gpu_memory_utilization,
            vllm_max_model_len=self._config.max_model_len,
            vllm_max_num_seqs=self._config.max_num_seqs,
            vllm_max_concurrent_requests=self._config.max_concurrent_requests,
            server_timeout=self._config.server_timeout,
            stream_server_logs=True,
        )

        tokenizer_name = self._resolve_tokenizer_name(checkpoint_path)

        server_config = ServerConfig(
            model_path=self._symlink_path,
            timeout_s=self._config.server_timeout,
            tokenizer_name=tokenizer_name,
            enable_sleep_mode=True,
            env={
                "CUDA_VISIBLE_DEVICES": str(self._config.vllm_gpu),
                "PYTORCH_CUDA_ALLOC_CONF": "",
                "VLLM_SERVER_DEV_MODE": "1",
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

        logger.info(
            "vLLM generation server started at %s (sleep mode enabled)", self._manager.base_url
        )

    async def sync_weights(self, checkpoint_path: str) -> None:
        """Reload weights via vLLM sleep/wake/reload API.

        1. Atomically update the model symlink to the new checkpoint
        2. Sleep (discard GPU weights, keep process alive)
        3. Wake (reallocate GPU memory)
        4. Reload weights from disk (re-reads symlink target)
        5. Reset prefix cache (clear stale KV entries)

        Falls back to full server restart if any API call fails.
        """
        if self._manager is None:
            raise RuntimeError("vLLM server not started")
        if self._symlink_path is None:
            raise RuntimeError("Symlink path not initialized — call start() first")

        import httpx

        base_url = self._manager.base_url
        t0 = time.monotonic()

        # Step 1: Update symlink to new checkpoint
        _update_symlink(self._symlink_path, checkpoint_path)
        logger.info(
            "vLLM weight sync: symlink updated -> %s (%.1fs)",
            checkpoint_path,
            time.monotonic() - t0,
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Step 2: Sleep — discard GPU weights
                resp = await client.post(f"{base_url}/sleep", params={"level": 2})
                resp.raise_for_status()
                logger.info("vLLM weight sync: sleep OK (%.1fs)", time.monotonic() - t0)

                # Step 3: Wake — reallocate GPU memory
                resp = await client.post(f"{base_url}/wake_up")
                resp.raise_for_status()
                logger.info("vLLM weight sync: wake_up OK (%.1fs)", time.monotonic() - t0)

                # Step 4: Reload weights from disk
                resp = await client.post(
                    f"{base_url}/collective_rpc",
                    json={"method": "reload_weights"},
                )
                resp.raise_for_status()
                logger.info("vLLM weight sync: reload_weights OK (%.1fs)", time.monotonic() - t0)

                # Step 5: Reset prefix cache
                resp = await client.post(f"{base_url}/reset_prefix_cache")
                resp.raise_for_status()

            elapsed = time.monotonic() - t0
            logger.info("vLLM weight sync complete in %.1fs", elapsed)

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "vLLM sleep/wake/reload failed after %.1fs: %s — falling back to full restart",
                elapsed,
                exc,
            )
            await self._fallback_full_restart(checkpoint_path)

    async def _fallback_full_restart(self, checkpoint_path: str) -> None:
        """Full server stop + restart when sleep/wake/reload fails."""
        assert self._manager is not None, "sync_weights checks _manager before calling"
        self._manager._config.tokenizer_name = self._resolve_tokenizer_name(checkpoint_path)
        await self._manager.reload_with_new_checkpoint(self._symlink_path)

        from ..environments.loop import VLLMServerBackend

        self._backend = VLLMServerBackend(
            base_url=self._manager.base_url,
            model_name=self._manager.model_name,
            tokenizer=self._tokenizer,
            timeout=self._config.server_timeout,
            max_concurrent_requests=self._config.max_concurrent_requests,
            strict_token_ids=True,
        )
        logger.info("vLLM fallback restart complete at %s", self._manager.base_url)

    async def shutdown(self) -> None:
        if self._manager is not None:
            try:
                await self._manager.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Error shutting down vLLM server: %s", exc)
            self._manager = None
        self._backend = None

        # Clean up symlink
        if self._symlink_path and os.path.islink(self._symlink_path):
            try:
                os.unlink(self._symlink_path)
            except OSError as exc:
                logger.debug("Failed to remove symlink %s: %s", self._symlink_path, exc)
            self._symlink_path = None

    def get_backend(self) -> Any:
        if self._backend is None:
            raise RuntimeError("vLLM backend not initialized — call start() first")
        return self._backend
