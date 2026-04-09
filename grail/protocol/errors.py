"""Protocol-level error types.

Raise these (instead of bare ``assert``) when an invariant the protocol
relies on is violated. Python ``-O`` strips ``assert`` silently; explicit
exceptions stay loud and carry structured context that surfaces in logs.
"""

from __future__ import annotations


class ProtocolViolationError(RuntimeError):
    """A protocol invariant was violated.

    Used by the miner, validator, and trainer when a contract that is
    required for miner-validator agreement is broken — for example, a
    missing checkpoint metadata field, an unexpected token shape, or a
    sketch dimension mismatch.

    The exception message should always include the offending values so
    that the failure is debuggable from logs alone.
    """
