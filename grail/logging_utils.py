import asyncio
import contextlib
import contextvars
import functools
import logging
import multiprocessing
import os
import sys
import threading
import time
from collections.abc import Callable, Generator
from typing import Any, TypeVar

_uid_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "miner_uid",
    default=None,
)
_window_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "miner_window",
    default=None,
)

# Context variables for distributed tracing
correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id",
    default=None,
)
operation_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "operation_name",
    default=None,
)

T = TypeVar("T")


@contextlib.contextmanager
def miner_log_context(
    uid: object = None,
    window: object = None,
) -> Generator[None, None, None]:
    """Context manager to set miner uid/window for log prefixing.

    Usage:
        with miner_log_context(uid, window):
            logger.info("...")  # auto-prefixed
    """
    token_uid = _uid_ctx.set(None if uid is None else str(uid))
    token_win = _window_ctx.set(None if window is None else str(window))
    try:
        yield
    finally:
        _uid_ctx.reset(token_uid)
        _window_ctx.reset(token_win)


class MinerPrefixFilter(logging.Filter):
    """Logging filter that prefixes messages with miner uid/window when set.

    The prefix is added only if:
      - uid is present in the context, and
      - the message is a string, and
      - it does not already start with a standard prefix
        (avoids double-prefixing)
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        uid = _uid_ctx.get()
        window = _window_ctx.get()

        if uid and isinstance(record.msg, str):
            is_prefixed = record.msg.startswith("[MINER ") or record.msg.startswith("[GRAIL ")
            if is_prefixed:
                return True
            if window:
                prefix = f"[MINER uid={uid} window={window}] "
            else:
                prefix = f"[MINER uid={uid}] "
            record.msg = prefix + record.msg
        return True


# Patterns for harmless sandbox cleanup noise that should be suppressed
_SANDBOX_NOISE_PATTERNS: tuple[str, ...] = (
    "Task was destroyed but it is pending",
    "Exception in thread Thread-",
    "Exception ignored in atexit callback",
    "TemporaryDirectory",
    "'NoneType' object is not callable",
    "_monitor",
    "EOFError",
)


class ExecutionSandboxNoiseFilter(logging.Filter):
    """Suppress expected noise from Python code execution sandbox.

    The execution sandbox spawns child processes that produce harmless but
    noisy error messages during cleanup:
    - EOFError in multiprocessing logging queue monitor threads
    - TemporaryDirectory cleanup TypeError when os.unlink is disabled
    - asyncio "Task was destroyed" warnings for websocket connections

    These are expected behavior and don't indicate real problems.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Return False to suppress matching records."""
        try:
            msg = str(record.getMessage())
            for pattern in _SANDBOX_NOISE_PATTERNS:
                if pattern in msg:
                    return False
        except Exception:
            pass
        return True


def flush_all_logs() -> None:
    """Best-effort flush of all logging handlers and stdio.

    Used by CLI entry points and watchdog to ensure logs are written before
    exit.
    """
    try:
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                h.flush()
            except Exception:
                pass
        grail_logger = logging.getLogger("grail")
        for h in list(grail_logger.handlers):
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


async def await_with_stall_log(
    awaitable: Any,
    label: str,
    *,
    threshold_seconds: float = 120.0,
    log: logging.Logger | None = None,
) -> Any:
    """Await a coroutine and emit one stall warning if it exceeds threshold.

    Emits a single "[STALL] <label> running > Ns" and then awaits completion.
    """
    logger = log or logging.getLogger(__name__)
    task = awaitable if isinstance(awaitable, asyncio.Task) else asyncio.create_task(awaitable)
    timer = asyncio.create_task(asyncio.sleep(threshold_seconds))
    try:
        done, _ = await asyncio.wait({task, timer}, return_when=asyncio.FIRST_COMPLETED)
        if timer in done and not task.done():
            try:
                logger.warning("[STALL] %s running > %.0fs", label, threshold_seconds)
            except Exception:
                pass
            return await task
        return await task
    finally:
        if not timer.done():
            timer.cancel()
            # Swallow cancellation of the timer during shutdown to avoid
            # bubbling CancelledError from cleanup paths.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await timer


async def dump_asyncio_stacks(
    *,
    log: logging.Logger | None = None,
    max_tasks: int = 20,
    max_frames: int = 3,
    label: str = "WATCHDOG",
) -> None:
    """Emit a compact snapshot of running asyncio task stack tops.

    Logged once on watchdog expiry to pinpoint blocking await locations.
    """
    logger = log or logging.getLogger(__name__)
    try:
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
    except Exception:
        tasks = []

    if not tasks:
        try:
            logger.error("[%s] No active tasks to dump", label)
        except Exception:
            pass
        return

    try:
        logger.error("[%s] Task stack snapshot (%d tasks)", label, len(tasks))
        count = 0
        for t in list(tasks)[:max_tasks]:
            count += 1
            try:
                stack = t.get_stack(limit=max_frames)
            except Exception:
                stack = []
            if stack:
                top = stack[-1]
                fname = getattr(top.f_code, "co_filename", "<unknown>")
                lineno = getattr(top, "f_lineno", 0)
                func = getattr(top.f_code, "co_name", "<unknown>")
                logger.error(
                    "[%s] task=%s at %s:%s in %s()",
                    label,
                    t.get_name(),
                    fname,
                    lineno,
                    func,
                )
            else:
                logger.error("[%s] task=%s at <no-python-frame>", label, t.get_name())
        remaining = len(tasks) - count
        if remaining > 0:
            logger.error("[%s] … %d more tasks omitted", label, remaining)
    except Exception:
        pass


class StructuredFormatter(logging.Formatter):
    """Enhanced formatter with process/thread info, timing, and correlation IDs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._start_time = time.time()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with enhanced context."""
        # Add process info
        record.process_name = multiprocessing.current_process().name
        record.process_id = multiprocessing.current_process().pid
        record.thread_id = threading.get_ident()

        # Add correlation ID if set
        corr_id = correlation_id.get()
        record.correlation_id = f"[{corr_id}]" if corr_id else ""

        # Add operation context
        op = operation_name.get()
        record.operation = f"<{op}>" if op else ""

        # Add relative timestamp for timing analysis
        record.relative_time = f"+{time.time() - self._start_time:.3f}s"

        return super().format(record)


def configure_process_logging(
    process_label: str,
    level: int = logging.INFO,
    include_function: bool = True,
) -> None:
    """Configure enhanced logging for a child process.

    Args:
        process_label: Label for this process (e.g., 'training_process', 'upload_worker')
        level: Logging level (default: INFO, use DEBUG for troubleshooting)
        include_function: Include function name and line number in logs
    """
    # Quiet noisy libraries first (before creating handlers)
    for noisy in [
        "websockets",
        "bittensor",
        "bittensor-cli",
        "btdecode",
        "asyncio",
        "aiobotocore.regions",
        "botocore",
        "datasets",
        "filelock",
        "vllm",
        "httpcore",
        "httpx",
        "openai",
        "urllib3",
        "uvicorn",
        "uvicorn.access",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    handler = logging.StreamHandler(sys.stdout)

    if include_function:
        fmt = (
            "%(asctime)s.%(msecs)03d "
            "%(relative_time)s "
            "%(levelname)-8s "
            "[%(process_name)s:%(process_id)d:%(thread_id)d] "
            "%(correlation_id)s%(operation)s "
            "%(name)s:%(funcName)s:%(lineno)d: "
            "%(message)s"
        )
    else:
        fmt = (
            "%(asctime)s.%(msecs)03d "
            "%(levelname)-8s "
            "[%(process_name)s:%(process_id)d] "
            "%(correlation_id)s%(operation)s "
            "%(name)s: "
            "%(message)s"
        )

    handler.setFormatter(
        StructuredFormatter(
            fmt=fmt,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Conditionally suppress noisy but harmless errors from execution sandbox
    # Set GRAIL_SUPPRESS_SANDBOX_NOISE=1 to enable suppression
    suppress_sandbox_noise = os.getenv("GRAIL_SUPPRESS_SANDBOX_NOISE", "0") == "1"
    if suppress_sandbox_noise:
        root.addFilter(ExecutionSandboxNoiseFilter())

    # Force unbuffered output for immediate visibility
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


def log_blocking_operation(operation_label: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log blocking operations with timing.

    Usage:
        @log_blocking_operation("model_to_gpu")
        def move_model_to_gpu(model):
            return model.cuda()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger = logging.getLogger(func.__module__)
            start = time.time()
            logger.debug("🔒 Starting blocking operation: %s", operation_label)
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                if duration > 1.0:
                    logger.warning(
                        "⏱️  Blocking operation took %.3fs: %s", duration, operation_label
                    )
                else:
                    logger.info(
                        "✅ Blocking operation complete: %s (%.3fs)", operation_label, duration
                    )
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(
                    "❌ Blocking operation failed: %s (%.3fs): %s",
                    operation_label,
                    duration,
                    e,
                )
                raise

        return wrapper

    return decorator


async def monitor_event_loop_lag(interval: float = 5.0, threshold: float = 1.0) -> None:
    """Background task to detect event loop blocking.

    Args:
        interval: Check interval in seconds
        threshold: Lag threshold to trigger warning (seconds)
    """
    logger = logging.getLogger(__name__)
    last_check = time.time()

    while True:
        await asyncio.sleep(interval)
        now = time.time()
        actual_elapsed = now - last_check
        expected_elapsed = interval
        lag = actual_elapsed - expected_elapsed

        if lag > threshold:
            logger.warning(
                "⚠️ Event loop lag detected: expected %.1fs, actual %.1fs (lag: %.1fs)",
                expected_elapsed,
                actual_elapsed,
                lag,
            )

        last_check = now


def log_all_running_tasks() -> None:
    """Log all currently running async tasks for debugging."""
    logger = logging.getLogger(__name__)
    try:
        tasks = asyncio.all_tasks()
        logger.info("📋 Active async tasks: %d", len(tasks))
        for task in tasks:
            logger.debug("  - Task: %s (done=%s)", task.get_name(), task.done())
    except Exception as e:
        logger.warning("Failed to log tasks: %s", e)
