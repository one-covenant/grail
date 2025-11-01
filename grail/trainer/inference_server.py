"""Inference server managers for evaluation backends.

Provides unified lifecycle management (startup, health checks, cleanup) for
HuggingFace, vLLM, and SGLang inference backends via async context managers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from grail.trainer.config import EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for inference server initialization."""

    host: str = "127.0.0.1"
    port: int = 0  # 0 = auto-assign
    timeout_s: float = 180.0
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    model_name_override: str | None = None


class InferenceServerManager(ABC):
    """Abstract base class for inference server lifecycle management.

    Provides context manager interface for automatic resource cleanup.
    Subclasses implement backend-specific launch/shutdown logic.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: ServerConfig,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._checkpoint_dir: str | None = None
        self._bound_port: int | None = None
        self._process: subprocess.Popen[bytes] | None = None
        # Capture a stable model name before freeing references
        self._initial_model_name: str = getattr(model, "name_or_path", "model")
        # Background task that streams server stdout to logs during startup
        self._log_task: asyncio.Task[None] | None = None

    @abstractmethod
    async def _start_server(self) -> None:
        """Backend-specific server startup logic."""

    @abstractmethod
    async def _stop_server(self) -> None:
        """Backend-specific server shutdown logic."""

    async def __aenter__(self) -> InferenceServerManager:
        """Prepare resources and start server.

        Note: Checkpoint is saved here, GPU memory freed here (server_manager._model),
        but the server is NOT started yet. The caller must ensure trainer models
        are also freed before the server starts to prevent OOM.
        """
        await self._prepare_checkpoint()
        await self._free_gpu_memory()
        # Server will be started explicitly after trainer frees its models
        return self

    async def start_server(self) -> None:
        """Start the server subprocess (call after entering context manager)."""
        await self._start_server()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup server but defer checkpoint cleanup to trainer.

        Trainer needs to reload from the saved checkpoint before it's deleted.
        Call cleanup_checkpoint() after reload to remove temp directory.
        """
        await self._stop_server()
        await self._reload_models()
        # Checkpoint cleanup deferred to trainer for exact reload

    @property
    def base_url(self) -> str:
        """HTTP endpoint for OpenAI-compatible API."""
        port = self._bound_port or self._config.port
        return f"http://{self._config.host}:{port}"

    @property
    def model_name(self) -> str:
        """Model identifier for API requests."""
        if self._config.model_name_override:
            return self._config.model_name_override
        return self._initial_model_name

    @property
    def checkpoint_dir(self) -> str | None:
        """Path to saved checkpoint directory for trainer reload."""
        return self._checkpoint_dir

    async def _prepare_checkpoint(self) -> None:
        """Save model/tokenizer to temporary directory for server loading."""
        if self._model is None:
            return

        try:
            self._checkpoint_dir = tempfile.mkdtemp(prefix="grail_eval_ckpt_")
            self._model.save_pretrained(self._checkpoint_dir, safe_serialization=True)
            self._tokenizer.save_pretrained(self._checkpoint_dir)
            logger.info("Saved evaluation checkpoint: %s", self._checkpoint_dir)
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)
            if self._checkpoint_dir:
                shutil.rmtree(self._checkpoint_dir, ignore_errors=True)
                self._checkpoint_dir = None

    async def _free_gpu_memory(self) -> None:
        """Release GPU memory by dropping model references."""
        try:
            self._model = None  # type: ignore[assignment]
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logger.debug("GPU memory released for server startup")
        except Exception as exc:
            logger.debug("GPU memory release encountered issue: %s", exc)

    @abstractmethod
    async def _reload_models(self) -> None:
        """Reload training models after server shutdown (implemented by subclasses if needed)."""

    def cleanup_checkpoint(self) -> None:
        """Remove temporary checkpoint directory after trainer reload.

        Public method called by trainer after successfully reloading from
        the saved checkpoint to ensure exact model/tokenizer state.
        """
        if self._checkpoint_dir:
            try:
                shutil.rmtree(self._checkpoint_dir, ignore_errors=True)
                logger.info("Cleaned up eval checkpoint: %s", self._checkpoint_dir)
            except Exception as exc:
                logger.warning("Checkpoint cleanup failed: %s", exc)
            finally:
                self._checkpoint_dir = None

    async def _wait_for_server_ready(
        self,
        ready_url: str,
        timeout_s: float,
    ) -> bool:
        """Poll HTTP endpoint until server responds or timeout.

        Args:
            ready_url: Health check endpoint (typically /v1/models)
            timeout_s: Maximum wait time in seconds

        Returns:
            True if server became ready, False otherwise
        """
        import time as _time

        import requests

        deadline = _time.time() + timeout_s
        poll_count = 0
        last_error: str | None = None

        while _time.time() < deadline:
            # Check if process crashed
            if self._process and self._process.poll() is not None:
                logger.error(
                    "Server process exited unexpectedly (code=%s)",
                    self._process.returncode,
                )
                return False

            try:
                resp = requests.get(ready_url, timeout=3.0)
                if resp.status_code == 200:
                    logger.info("Server ready at %s", ready_url)
                    # Optional warmup request
                    await self._warmup_server(ready_url)
                    return True
                last_error = f"HTTP {resp.status_code}"
            except Exception as exc:
                last_error = str(exc)

            poll_count += 1
            if poll_count % 6 == 0:  # Log every 3s
                elapsed = _time.time() - (deadline - timeout_s)
                logger.debug("Waiting for server (%.1fs): %s", elapsed, last_error)

            await asyncio.sleep(0.5)

        logger.warning("Server not ready after %.1fs: %s", timeout_s, last_error)
        return False

    async def _consume_process_output(self, name: str) -> None:
        """Continuously read and log lines from the server process stdout.

        This runs as a background task during startup, ensuring that early
        initialization errors (imports, CUDA checks, argument parsing) are
        visible in our logs instead of being lost in the PIPE buffer.
        """
        if self._process is None or self._process.stdout is None:
            return

        loop = asyncio.get_running_loop()
        try:
            while True:
                if self._process is None or self._process.stdout is None:
                    break
                # Read one line without blocking the event loop
                line: bytes = await loop.run_in_executor(None, self._process.stdout.readline)
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        logger.debug("[%s] %s", name, text)
                except Exception:
                    # Best-effort logging; never raise from logger path
                    pass
        except Exception:
            # Swallow exceptions to avoid failing startup due to logger issues
            pass

    def _start_process_logger(self, name: str) -> None:
        """Begin background task to stream server stdout into logger."""
        # Avoid multiple log tasks
        if self._log_task is not None:
            return
        if self._process is None or self._process.stdout is None:
            return
        try:
            self._log_task = asyncio.create_task(self._consume_process_output(name))
        except RuntimeError:
            # No running loop (shouldn't happen in async context); skip
            self._log_task = None

    async def _stop_process_logger(self) -> None:
        """Stop background logger task if running."""
        if self._log_task is None:
            return
        try:
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                # Expected when cancelling the logger task; do not propagate
                pass
            except Exception:
                # Swallow any logger task exceptions; logging must be best-effort
                pass
        finally:
            self._log_task = None

    async def _warmup_server(self, base_url: str) -> None:
        """Send warmup request to initialize KV cache."""
        import time as _time

        import requests

        try:
            start = _time.time()
            completions_url = base_url.replace("/v1/models", "/v1/completions")
            payload = {"model": self.model_name, "prompt": "Test", "max_tokens": 1}
            requests.post(completions_url, json=payload, timeout=30.0)
            logger.info("Server warmup completed in %.2fs", _time.time() - start)
        except Exception as exc:
            logger.debug("Warmup failed (non-fatal): %s", exc)

    def _allocate_port(self) -> int:
        """Bind to an available port and return the port number."""
        if self._config.port > 0:
            return self._config.port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self._config.host, 0))
            port = sock.getsockname()[1]
            return port

    async def _terminate_process(
        self,
        proc: subprocess.Popen[bytes] | None,
        wait_for_gpu: bool = True,
    ) -> None:
        """Terminate subprocess gracefully and optionally wait for GPU memory release."""
        if proc is None:
            logger.debug("_terminate_process: proc is None, skipping")
            return

        import asyncio
        import time

        logger.info("_terminate_process: sending SIGTERM to pid=%s", proc.pid)
        try:
            proc.terminate()
            try:
                logger.debug("_terminate_process: waiting for process to exit (timeout=10s)...")
                proc.wait(timeout=10)
                logger.info("Server process terminated (pid=%s)", proc.pid)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't exit gracefully, force killing pid=%s", proc.pid)
                proc.kill()
                logger.debug(
                    "_terminate_process: waiting for killed process to exit (timeout=5s)..."
                )
                proc.wait(timeout=5)
                logger.info("Process killed and reaped (pid=%s)", proc.pid)
        except Exception as exc:
            logger.warning("Error terminating process: %s", exc)

        # Wait for GPU memory release if requested
        if wait_for_gpu and torch.cuda.is_available():
            logger.info("_terminate_process: waiting for GPU memory to be freed (max 30s)...")
            torch.cuda.empty_cache()
            start = time.time()
            max_wait = 30.0
            poll_count = 0

            while time.time() - start < max_wait:
                try:
                    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                    total_gb = torch.cuda.mem_get_info()[1] / (1024**3)
                    poll_count += 1

                    if free_gb > 25.0:  # Sufficient for training model reload
                        elapsed = time.time() - start
                        logger.info(
                            "GPU memory freed after %.1fs (poll #%d): %.2f / %.2f GB free",
                            elapsed,
                            poll_count,
                            free_gb,
                            total_gb,
                        )
                        return

                    if poll_count % 5 == 0:  # Log every 5 polls (5s)
                        logger.debug(
                            "GPU memory still waiting (poll #%d, %.1fs elapsed): %.2f / %.2f GB free",
                            poll_count,
                            time.time() - start,
                            free_gb,
                            total_gb,
                        )

                    # Use async sleep instead of blocking time.sleep()
                    logger.debug("Sleeping 1s before next GPU memory check...")
                    await asyncio.sleep(1.0)
                except Exception as exc:
                    logger.warning("Error checking GPU memory: %s", exc)
                    break

            # Log final state
            try:
                free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                total_gb = torch.cuda.mem_get_info()[1] / (1024**3)
                logger.warning(
                    "GPU memory may still be held after %.1fs (poll #%d): %.2f / %.2f GB free",
                    max_wait,
                    poll_count,
                    free_gb,
                    total_gb,
                )
            except Exception:
                pass
        else:
            logger.debug("_terminate_process: wait_for_gpu=%s or cuda not available", wait_for_gpu)


class HuggingFaceServerManager(InferenceServerManager):
    """No-op server manager for HuggingFace inference (model stays in memory)."""

    async def _start_server(self) -> None:
        """HF backend doesn't require external server."""
        logger.info("Using HuggingFace in-process backend")

    async def _stop_server(self) -> None:
        """No server to stop for HF backend."""

    async def _reload_models(self) -> None:
        """No-op for HuggingFace backend."""
        pass


class VLLMServerManager(InferenceServerManager):
    """vLLM server manager with isolated environment support."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: ServerConfig,
        eval_config: EvalConfig,
        python_executable: str = "tools/vllm-server/.venv/bin/python",
        module_entrypoint: str = "vllm.entrypoints.openai.api_server",
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, config=config)
        self._python_executable = python_executable
        self._module_entrypoint = module_entrypoint
        self._eval_config = eval_config

    async def _start_server(self) -> None:
        """Launch vLLM server in isolated environment."""
        if not self._checkpoint_dir:
            logger.warning("No checkpoint available, skipping vLLM server launch")
            return

        # Resolve Python executable to absolute path
        python_path = self._resolve_executable()
        if not python_path:
            return

        # Allocate port and build command
        self._bound_port = self._allocate_port()
        cmd = self._build_command(python_path)

        # Launch process
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
            )
            logger.info(
                "Launched vLLM server: pid=%s host=%s port=%s",
                self._process.pid,
                self._config.host,
                self._bound_port,
            )
            # Start streaming server logs immediately for easier diagnosis
            self._start_process_logger("vllm")
        except Exception as exc:
            logger.error("Failed to launch vLLM server: %s", exc)
            return

        # Wait for readiness
        ready_url = f"{self.base_url}/v1/models"
        is_ready = await self._wait_for_server_ready(ready_url, self._config.timeout_s)

        if not is_ready:
            await self._terminate_process(self._process, wait_for_gpu=False)
            self._process = None
            self._bound_port = None

    async def _stop_server(self) -> None:
        """Terminate vLLM server and wait for GPU memory release."""
        logger.info("VLLMServerManager._stop_server: starting server cleanup")
        try:
            logger.info("Stopping process logger task...")
            await self._stop_process_logger()
            logger.info("Process logger task stopped")
        except Exception as exc:
            logger.warning("Error stopping process logger: %s", exc)

        try:
            logger.info(
                "Terminating server process (pid=%s)...",
                self._process.pid if self._process else None,
            )
            await self._terminate_process(self._process, wait_for_gpu=True)
            logger.info("Server process terminated successfully")
        except Exception as exc:
            logger.error("Error terminating process: %s", exc)

        self._process = None
        self._bound_port = None
        logger.info("VLLMServerManager._stop_server: cleanup complete")

    async def _reload_models(self) -> None:
        """No-op for vLLM backend (models remain in external process)."""
        pass

    def _resolve_executable(self) -> str | None:
        """Resolve Python executable to absolute path and validate existence."""
        python_path = self._python_executable

        if not os.path.isabs(python_path):
            # Relative to project root (3 levels up from this file)
            # __file__ = /root/grail/grail/trainer/inference_server.py
            # 1 dirname: /root/grail/grail/trainer
            # 2 dirname: /root/grail/grail
            # 3 dirname: /root/grail (project root)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            python_path = os.path.join(project_root, python_path)

        if not os.path.isfile(python_path):
            logger.error(
                "vLLM Python executable not found: %s (run scripts/setup_vllm_env.sh)",
                python_path,
            )
            return None

        return python_path

    def _build_command(self, python_path: str) -> list[str]:
        """Build vLLM server launch command with optimized parameters."""
        cmd = [
            python_path,
            "-m",
            self._module_entrypoint,
            "--model",
            self._checkpoint_dir or "",
            "--served-model-name",
            self.model_name,  # Match client request model name to prevent 404 errors
            "--host",
            self._config.host,
            "--port",
            str(self._bound_port),
            "--dtype",
            self._config.dtype,
            "--tensor-parallel-size",
            "1",
            # Memory optimizations from EvalConfig to prevent OOM during KV cache allocation
            "--gpu-memory-utilization",
            str(self._eval_config.vllm_gpu_memory_utilization),
            "--max-model-len",
            str(self._eval_config.vllm_max_model_len),
            "--max-num-seqs",
            str(self._eval_config.vllm_max_num_seqs),
        ]

        if self._config.trust_remote_code:
            cmd.append("--trust-remote-code")

        return cmd


class SGLangServerManager(InferenceServerManager):
    """SGLang server manager with memory optimization."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: ServerConfig,
        eval_config: EvalConfig,
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, config=config)
        self._eval_config = eval_config

    async def _start_server(self) -> None:
        """Launch SGLang server with optimized memory settings."""
        if not self._checkpoint_dir:
            logger.warning("No checkpoint available, skipping SGLang server launch")
            return

        # Allocate port and build command
        self._bound_port = self._allocate_port()
        cmd = self._build_command()

        # Launch process
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
            )
            logger.info(
                "Launched SGLang server: pid=%s host=%s port=%s",
                self._process.pid,
                self._config.host,
                self._bound_port,
            )
            # Start streaming server logs immediately for easier diagnosis
            self._start_process_logger("sglang")
        except Exception as exc:
            logger.error("Failed to launch SGLang server: %s", exc)
            return

        # Wait for readiness
        ready_url = f"{self.base_url}/v1/models"
        is_ready = await self._wait_for_server_ready(ready_url, self._config.timeout_s)

        if not is_ready:
            await self._terminate_process(self._process, wait_for_gpu=False)
            self._process = None
            self._bound_port = None

    async def _stop_server(self) -> None:
        """Terminate SGLang server and wait for GPU memory release."""
        await self._stop_process_logger()
        await self._terminate_process(self._process, wait_for_gpu=True)
        self._process = None
        self._bound_port = None

    async def _reload_models(self) -> None:
        """No-op for SGLang backend (models remain in external process)."""
        pass

    def _build_command(self) -> list[str]:
        """Build SGLang server launch command with optimized parameters."""
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self._checkpoint_dir or "",
            "--host",
            self._config.host,
            "--port",
            str(self._bound_port),
            "--dtype",
            self._config.dtype,
            "--tp-size",
            "1",
            # Memory optimizations from EvalConfig to prevent OOM during KV cache allocation
            "--mem-fraction-static",
            str(self._eval_config.sglang_mem_fraction_static),
            "--max-running-requests",
            str(self._eval_config.sglang_max_running_requests),
            "--schedule-policy",
            "fcfs",  # Fair scheduling
            "--context-length",
            str(self._eval_config.sglang_context_length),
        ]

        if self._config.trust_remote_code:
            cmd.append("--trust-remote-code")

        return cmd


def create_inference_server(
    backend: str,
    model: Any,
    tokenizer: Any,
    eval_config: EvalConfig,
) -> InferenceServerManager:
    """Factory function to create appropriate server manager.

    Args:
        backend: One of 'hf', 'vllm', 'sglang'
        model: Training model to serve
        tokenizer: Tokenizer for the model
        eval_config: Evaluation configuration

    Returns:
        Configured server manager instance

    Raises:
        ValueError: If backend is unsupported
    """
    config = ServerConfig(
        host=eval_config.sglang_host,
        port=eval_config.sglang_port,
        timeout_s=eval_config.sglang_server_timeout_s,
        trust_remote_code=eval_config.sglang_trust_remote_code,
    )

    backend_lower = backend.lower()

    if backend_lower == "hf":
        return HuggingFaceServerManager(model=model, tokenizer=tokenizer, config=config)
    if backend_lower == "vllm":
        return VLLMServerManager(
            model=model,
            tokenizer=tokenizer,
            config=config,
            eval_config=eval_config,
            python_executable=eval_config.vllm_python_executable,
            module_entrypoint=eval_config.vllm_module_entrypoint,
        )
    if backend_lower == "sglang":
        return SGLangServerManager(
            model=model, tokenizer=tokenizer, config=config, eval_config=eval_config
        )

    msg = f"Unsupported backend: {backend} (choose 'hf', 'vllm', or 'sglang')"
    raise ValueError(msg)
