import asyncio
import contextlib
import contextvars
import logging
import sys
from collections.abc import Generator
from typing import Any

_uid_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "miner_uid",
    default=None,
)
_window_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "miner_window",
    default=None,
)


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

    def filter(self, record: logging.LogRecord) -> bool:
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
