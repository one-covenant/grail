import contextlib
import contextvars
import logging
from collections.abc import Generator
from typing import Optional

_uid_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("miner_uid", default=None)
_window_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "miner_window", default=None
)


@contextlib.contextmanager
def miner_log_context(
    uid: Optional[object] = None, window: Optional[object] = None
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
      - it does not already start with a standard prefix (avoids double-prefixing)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        uid = _uid_ctx.get()
        window = _window_ctx.get()

        if (
            uid
            and isinstance(record.msg, str)
            and not (record.msg.startswith("[MINER ") or record.msg.startswith("[GRAIL "))
        ):
            if window:
                prefix = f"[MINER uid={uid} window={window}] "
            else:
                prefix = f"[MINER uid={uid}] "
            record.msg = prefix + record.msg
        return True
