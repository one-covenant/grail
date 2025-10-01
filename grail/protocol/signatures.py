"""Cryptographic signature functions for GRAIL protocol.

Functions for:
- Signing and verifying s_vals with Bittensor wallets
- Building and verifying commit bindings
- Deriving canonical SAT problem parameters
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bittensor as bt
else:
    try:
        import bittensor as bt
    except ImportError:
        bt = None  # type: ignore

from .tokens import hash_s_vals, hash_tokens, int_to_bytes

logger = logging.getLogger(__name__)

# Domain separation for commit bindings
COMMIT_DOMAIN = b"grail-commit-v1"
COMMIT_DOMAIN_MRS = b"grail-commit-v2-mrs"


def sign_s_vals(s_vals: list[int], wallet: bt.wallet) -> bytes:  # type: ignore[misc]
    """Sign the s_vals list using Bittensor wallet's cryptographic signature.

    Args:
        s_vals: List of s_vals to sign
        wallet: Bittensor wallet object (bt.wallet) with signing capability

    Returns:
        Signature bytes from Ed25519 signing

    Raises:
        TypeError: If wallet doesn't have signing capability
    """
    if bt is None:
        raise ImportError("bittensor is required for sign_s_vals")

    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError(f"Wallet must be a bt.wallet with hotkey.sign() method, got {type(wallet)}")

    s_vals_bytes = b"".join(int_to_bytes(val) for val in s_vals)
    signature: bytes = wallet.hotkey.sign(s_vals_bytes)
    logger.debug(f"Signed {len(s_vals)} s_vals with Bittensor wallet signature")
    return signature


def verify_s_vals_signature(s_vals: list[int], signature: bytes, wallet_address: str) -> bool:
    """Verify the signature of s_vals list using Bittensor wallet's public key.

    Args:
        s_vals: List of s_vals to verify
        signature: Signature to verify
        wallet_address: SS58 wallet address for public key verification

    Returns:
        True if signature is valid

    Raises:
        TypeError: If wallet_address is not a string
    """
    if bt is None:
        raise ImportError("bittensor is required for verify_s_vals_signature")

    if not isinstance(wallet_address, str):
        raise TypeError(f"wallet_address must be a string SS58 address, got {type(wallet_address)}")

    s_vals_bytes = b"".join(int_to_bytes(val) for val in s_vals)

    try:
        keypair = bt.Keypair(ss58_address=wallet_address)
        verified: bool = keypair.verify(data=s_vals_bytes, signature=signature)
        if not verified:
            logger.debug("Signature verification failed")
        return verified
    except Exception as e:
        logger.warning(f"Signature verification error: {e}")
        return False


def hash_mrs_commitments(mrs_commitments: list[dict]) -> bytes:
    """Return SHA-256 over a canonical JSON encoding of MRS commitments.

    Uses sort_keys=True and compact separators to avoid whitespace drift.
    """
    try:
        import json

        payload = json.dumps(mrs_commitments, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).digest()
    except Exception as e:
        logger.warning(f"Failed to hash MRS commitments: {e}")
        return hashlib.sha256(b"").digest()


def build_commit_binding(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    s_vals: list[int],
) -> bytes:
    """Build domain-separated commit binding to be signed.

    Format: SHA256(COMMIT_DOMAIN || len(x)||x for each x in
    [tokens_hash, rand_bytes, model_name_bytes, layer_index_be, s_vals_hash]).

    Args:
        tokens: Token IDs from model output
        randomness_hex: Challenge randomness hex string
        model_name: Model identifier
        layer_index: Hidden layer index used for sketches
        s_vals: Sketch values for each token

    Returns:
        SHA-256 hash of the domain-separated commit binding
    """

    def _len_bytes(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big")

    # Normalize randomness hex
    rand_clean = randomness_hex.strip().replace("0x", "").replace("0X", "")
    if len(rand_clean) % 2 != 0:
        rand_clean = "0" + rand_clean
    rand_bytes = bytes.fromhex(rand_clean)

    # Hash components
    tokens_h = hash_tokens(tokens)
    svals_h = hash_s_vals(s_vals)
    model_b = (model_name or "").encode("utf-8")
    layer_b = int(layer_index).to_bytes(4, "big", signed=True)

    # Build domain-separated commitment
    h = hashlib.sha256()
    h.update(COMMIT_DOMAIN)
    for part in (tokens_h, rand_bytes, model_b, layer_b, svals_h):
        h.update(_len_bytes(part))
        h.update(part)
    return h.digest()


def build_commit_binding_mrs(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    mrs_commitments: list[dict],
) -> bytes:
    """Build domain-separated commit binding for MRS commitments.

    Format: SHA256(COMMIT_DOMAIN_MRS || len(x)||x for each x in
    [tokens_hash, rand_bytes, model_name_bytes, layer_index_be, mrs_hash]).
    """

    def _len_bytes(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big")

    # Normalize randomness hex
    rand_clean = randomness_hex.strip().replace("0x", "").replace("0X", "")
    if len(rand_clean) % 2 != 0:
        rand_clean = "0" + rand_clean
    rand_bytes = bytes.fromhex(rand_clean)

    # Hash components
    tokens_h = hash_tokens(tokens)
    mrs_h = hash_mrs_commitments(mrs_commitments)
    model_b = (model_name or "").encode("utf-8")
    layer_b = int(layer_index).to_bytes(4, "big", signed=True)

    # Build domain-separated commitment
    h = hashlib.sha256()
    h.update(COMMIT_DOMAIN_MRS)
    for part in (tokens_h, rand_bytes, model_b, layer_b, mrs_h):
        h.update(_len_bytes(part))
        h.update(part)
    return h.digest()


def sign_commit_binding(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    s_vals: list[int],
    wallet: bt.wallet,  # type: ignore[misc]
) -> bytes:
    """Sign the commit-binding message with wallet hotkey.

    Args:
        tokens: Token IDs from model output
        randomness_hex: Challenge randomness hex string
        model_name: Model identifier
        layer_index: Hidden layer index
        s_vals: Sketch values
        wallet: Bittensor wallet for signing

    Returns:
        Ed25519 signature bytes

    Raises:
        TypeError: If wallet lacks signing capability
    """
    if bt is None:
        raise ImportError("bittensor is required for sign_commit_binding")

    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError("Wallet must provide hotkey.sign()")

    msg = build_commit_binding(tokens, randomness_hex, model_name, layer_index, s_vals)
    return wallet.hotkey.sign(msg)  # type: ignore[union-attr]


def sign_commit_binding_mrs(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    mrs_commitments: list[dict],
    wallet: bt.wallet,  # type: ignore[misc]
) -> bytes:
    """Sign the MRS commit-binding message with wallet hotkey."""
    if bt is None:
        raise ImportError("bittensor is required for sign_commit_binding_mrs")
    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError("Wallet must provide hotkey.sign()")
    msg = build_commit_binding_mrs(tokens, randomness_hex, model_name, layer_index, mrs_commitments)
    return wallet.hotkey.sign(msg)  # type: ignore[union-attr]


def verify_commit_signature(commit: dict, wallet_address: str) -> bool:
    """Verify commit signature binding tokens, randomness, model, layer, and proofs.

    Supports both legacy (s_vals) and MRS (mrs_commitments) proof formats.

    Args:
        commit: Commit data with tokens, proof data, beacon, model info, and signature
        wallet_address: SS58 address for public key verification

    Returns:
        True if signature is valid, False otherwise
    """
    if bt is None:
        raise ImportError("bittensor is required for verify_commit_signature")

    try:
        sig = bytes.fromhex(commit["signature"])
        proof_version = commit.get("proof_version", "v1")

        if proof_version == "v2_mrs":
            tokens = commit["tokens"]
            mrs_commitments = commit["mrs_commitments"]
            beacon = commit.get("beacon", commit.get("round_R", {}))
            randomness = beacon["randomness"]
            model_info = commit.get("model", {})
            model_name = model_info.get("name", "")
            layer_index = int(model_info.get("layer_index"))
            msg = build_commit_binding_mrs(
                tokens, randomness, model_name, layer_index, mrs_commitments
            )
        else:
            # Legacy proof: signature is over commit binding
            tokens = commit["tokens"]
            s_vals = commit["s_vals"]
            beacon = commit.get("beacon", commit.get("round_R", {}))
            randomness = beacon["randomness"]
            model_info = commit.get("model", {})
            model_name = model_info.get("name", "")
            layer_index = int(model_info.get("layer_index"))
            msg = build_commit_binding(tokens, randomness, model_name, layer_index, s_vals)

        keypair = bt.Keypair(ss58_address=wallet_address)
        return keypair.verify(data=msg, signature=sig)  # type: ignore[union-attr,return-value]
    except Exception as e:
        logger.debug(f"Signature verification failed: {e}")
        return False


def derive_canonical_sat(
    wallet_addr: str, window_hash: str, problem_index: int
) -> tuple[str, float]:
    """Derive canonical SAT seed and difficulty for miner/window/problem index.

    The seed binds problems to the miner hotkey and the window block hash.
    The difficulty is sampled ~uniformly in [0.3, 0.9] from a PRF of the
    same material to eliminate miner control while keeping a broad spread.

    Args:
        wallet_addr: Miner's SS58 wallet address
        window_hash: Block hash at window start
        problem_index: Problem index within the window

    Returns:
        Tuple of (seed_hex, difficulty_float)
    """
    try:
        idx = int(problem_index)
    except Exception:
        idx = 0

    material = f"{wallet_addr}:{window_hash}:{idx}".encode()
    seed = hashlib.sha256(b"seed|" + material).hexdigest()
    diff_digest = hashlib.sha256(b"diff|" + material).digest()
    u = int.from_bytes(diff_digest[:8], "big") / float(1 << 64)
    difficulty = 0.3 + 0.6 * u
    return seed, float(difficulty)
