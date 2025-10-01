"""Cryptographic primitives for GRAIL protocol.

Pure, deterministic functions for:
- Pseudorandom function (PRF) with domain separation
- Random sketch vector generation from randomness
- Deterministic index selection for proof challenges
- Modular inner product for hidden state sketches
- Proof package creation

These functions have no side effects and are easily unit-testable.
"""

from __future__ import annotations

import hashlib
import logging
import random
import struct
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore

from ..shared.constants import CHALLENGE_K, PRIME_Q, RNG_LABEL

logger = logging.getLogger(__name__)


def prf(label: bytes, *parts: bytes, out_bytes: int) -> bytes:
    """Pseudorandom function using SHA-256 in counter mode for arbitrary output length.

    Args:
        label: Domain separation label
        *parts: Variable number of byte strings to include in PRF input
        out_bytes: Number of output bytes required

    Returns:
        Deterministic pseudorandom bytes of length out_bytes

    Raises:
        ValueError: If out_bytes is negative or too large
        TypeError: If inputs are not bytes
    """
    # Input validation
    if out_bytes < 0:
        raise ValueError(f"out_bytes must be non-negative, got {out_bytes}")
    if out_bytes > 2**16:  # Reasonable upper limit (64KB)
        raise ValueError(f"out_bytes too large: {out_bytes} (max 65536)")
    if out_bytes == 0:
        return b""

    if not isinstance(label, bytes):
        raise TypeError(f"label must be bytes, got {type(label).__name__}")
    for i, part in enumerate(parts):
        if not isinstance(part, bytes):
            raise TypeError(f"parts[{i}] must be bytes, got {type(part).__name__}")

    # Use SHAKE256 for variable-length output if available (more efficient)
    try:
        if hasattr(hashlib, "shake_256"):
            shake = hashlib.shake_256()
            shake.update(label)
            shake.update(b"||")
            for part in parts[:-1] if parts else []:
                shake.update(part)
                shake.update(b"||")
            if parts:
                shake.update(parts[-1])
            return shake.digest(out_bytes)
    except Exception:
        pass  # Fall back to SHA256 method

    # SHA256-based expansion with counter mode
    hash_size = 32  # SHA256 output size
    num_blocks = (out_bytes + hash_size - 1) // hash_size

    # Build input once
    if parts:
        input_data = label + b"||" + b"||".join(parts)
    else:
        input_data = label

    # Use counter mode for expansion
    output = bytearray(num_blocks * hash_size)
    for i in range(num_blocks):
        block_input = input_data + i.to_bytes(4, "big")
        block_hash = hashlib.sha256(block_input).digest()
        output[i * hash_size : (i + 1) * hash_size] = block_hash

    return bytes(output[:out_bytes])


def r_vec_from_randomness(rand_hex: str, d_model: int) -> torch.Tensor:  # type: ignore[misc]
    """Generate random projection vector from drand randomness.

    Takes drand randomness (32 bytes hex) and expands it deterministically
    into a d_model-dimensional vector using a PRF. This ensures everyone
    with the same drand value generates the same projection vector.

    Args:
        rand_hex: Hex string of drand randomness (typically from drand beacon)
        d_model: Model hidden dimension size

    Returns:
        Random projection vector of shape (d_model,) with int32 values

    Raises:
        ValueError: If rand_hex is invalid or d_model is invalid

    Note:
        Uses big-endian byte order for cross-platform consistency.
        Results are cached to avoid recomputation.
    """
    if torch is None:
        raise ImportError("torch is required for r_vec_from_randomness")

    # Initialize cache on first call
    if not hasattr(r_vec_from_randomness, "_cache"):
        r_vec_from_randomness._cache = {}  # type: ignore[attr-defined]

    # Input validation
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if d_model > 100000:  # Reasonable upper limit
        raise ValueError(f"d_model too large: {d_model} (max 100000)")
    if not rand_hex:
        raise ValueError("rand_hex cannot be empty")

    # Normalize hex string
    clean_hex = rand_hex.strip().replace("0x", "").replace("0X", "")
    if not clean_hex:
        raise ValueError(f"Empty randomness hex string after cleaning: '{rand_hex}'")
    if len(clean_hex) % 2 != 0:
        clean_hex = "0" + clean_hex

    # Check cache
    cache_key = (clean_hex, d_model)
    cache: dict[tuple[str, int], torch.Tensor] = cast(
        dict[tuple[str, int], torch.Tensor],
        getattr(r_vec_from_randomness, "_cache", {}),
    )
    if cache_key in cache:
        logger.debug(f"Using cached sketch vector for d_model={d_model}")
        return cache[cache_key].clone()

    # Generate random vector using PRF
    try:
        raw = prf(
            RNG_LABEL["sketch"],
            bytes.fromhex(clean_hex),
            out_bytes=4 * d_model,
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}"
        ) from e

    # Unpack to int32 tensor (use numpy if available for efficiency)
    try:
        import numpy as np

        ints_array = np.frombuffer(raw, dtype=">i4").astype(np.int32, copy=False)
        tensor = torch.from_numpy(ints_array.copy())
    except ImportError:
        ints = struct.unpack(">" + "i" * d_model, raw)
        tensor = torch.tensor(ints, dtype=torch.int32)

    # Cache the result (limit cache size to prevent memory issues)
    if len(cache) < 100:
        cache[cache_key] = tensor.clone()
        r_vec_from_randomness._cache = cache  # type: ignore[attr-defined]

    logger.debug(
        f"Generated sketch vector with shape={tensor.shape}, first 4 values: {tensor[:4].tolist()}"
    )
    return tensor


def indices_from_root(tokens: list[int], rand_hex: str, seq_len: int, k: int) -> list[int]:
    """Generate deterministic indices for proof verification.

    Args:
        tokens: List of token IDs from the model output
        rand_hex: Randomness hex string (from drand/block hash)
        seq_len: Sequence length to sample from
        k: Number of indices to select

    Returns:
        Sorted list of k indices sampled deterministically

    Raises:
        ValueError: If rand_hex is invalid or k > seq_len
    """
    from .tokens import int_to_bytes  # Avoid circular import

    # Validate inputs early
    if k > seq_len:
        raise ValueError(f"Cannot sample {k} indices from sequence of length {seq_len}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not tokens:
        raise ValueError("tokens list cannot be empty")

    # Efficient token bytes conversion
    tokens_bytes = b"".join(int_to_bytes(token) for token in tokens)
    tokens_hash = hashlib.sha256(tokens_bytes).digest()

    # Normalize hex string
    clean_hex = rand_hex.strip().replace("0x", "").replace("0X", "")
    if not clean_hex:
        raise ValueError(f"Empty randomness hex string: '{rand_hex}'")
    if len(clean_hex) % 2 != 0:
        clean_hex = "0" + clean_hex

    # Generate deterministic sampling seed
    try:
        material = prf(
            RNG_LABEL["open"],
            tokens_hash,
            bytes.fromhex(clean_hex),
            out_bytes=32,
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}"
        ) from e

    # Use deterministic sampling with seed
    rnd = random.Random(material)

    # For small k relative to seq_len, use sample (more efficient)
    # For large k, use shuffle and slice (avoids rejection sampling overhead)
    if k < seq_len * 0.1:  # If selecting less than 10% of indices
        idxs = sorted(rnd.sample(range(seq_len), k))
    else:
        all_indices = list(range(seq_len))
        rnd.shuffle(all_indices)
        idxs = sorted(all_indices[:k])

    logger.debug(
        f"Selected {len(idxs)} indices from seq_len={seq_len}: {idxs[:5]}..."
        if len(idxs) > 5
        else f"Selected indices: {idxs}"
    )
    return idxs


def dot_mod_q(hidden: torch.Tensor, r_vec: torch.Tensor) -> int:  # type: ignore[misc]
    """Compute modular inner product of hidden state and random projection vector.

    Args:
        hidden: Hidden state tensor from model layer
        r_vec: Random projection vector

    Returns:
        Inner product modulo PRIME_Q
    """
    if torch is None:
        raise ImportError("torch is required for dot_mod_q")

    # Ensure both tensors are on the same device
    device = hidden.device
    r_vec = r_vec.to(device)

    # Scale and convert to float for computation (avoid int64 issues on CUDA)
    scaled = torch.round(hidden * 1024)
    prod = torch.dot(scaled, r_vec.float())

    # Convert to int and apply modulo
    return int(prod.item()) % PRIME_Q


def create_proof(
    tokens: list[int],
    randomness_hex: str,
    seq_len: int,
    k: int = CHALLENGE_K,
) -> dict:
    """Generate GRAIL proof package with deterministic indices.

    This replaces the stateful Prover.open() method with a pure function.

    Args:
        tokens: Token IDs from model generation
        randomness_hex: Challenge randomness (verifier-supplied)
        seq_len: Sequence length
        k: Number of indices to challenge

    Returns:
        Proof package with beacon and deterministically-derived indices
    """
    beacon_R1 = {"round": 2, "randomness": randomness_hex}
    idxs = indices_from_root(tokens, randomness_hex, seq_len, k)
    return {"round_R1": beacon_R1, "indices": idxs}
