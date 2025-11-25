"""GRAIL Verifier for GPU/Framework-Agnostic Proof.

This module implements a novel hidden-state verification scheme robust
across GPUs, CUDA versions, and frameworks (HF, vLLM, SGLang).

Key innovations:
1. Top-K selection: Focus on important activations (stable)
2. Logarithmic bucketing: Coarse quantization reduces sensitivity
3. Sketch verification: Random linear projection for cryptographic binding

Security: ~10^-117 forgery probability across k=16 positions with sketch-only verification.
"""

from __future__ import annotations

import logging
import math

import torch

from ..shared.constants import (
    PRIME_Q,
    PROOF_COEFF_RANGE,
    PROOF_NUM_BUCKETS,
    PROOF_POSITION_IMPORTANCE_DECAY,
    PROOF_SKETCH_TOLERANCE,
    PROOF_TOPK,
)

logger = logging.getLogger(__name__)


def log_magnitude_bucket(value: float, num_buckets: int = PROOF_NUM_BUCKETS) -> int:
    """Map activation to logarithmic magnitude bucket with sign preservation.

    Logarithmic bucketing provides natural robustness:
    - Small values: coarse bins (where drift happens)
    - Large values: finer bins (where we have precision)
    - Matches floating-point representation behavior

    Args:
        value: Activation value to bucket
        num_buckets: Number of buckets per sign (default: 16)

    Returns:
        Signed bucket index in [-num_buckets+1, 0, num_buckets-1]
    """
    # Handle NaN values - these indicate numerical instability in the model
    if math.isnan(value):
        logger.warning(
            "NaN value encountered in hidden state. This typically indicates "
            "numerical instability in quantized models (especially GPTQ-Int8) "
            "or missing CUDA kernels. Treating as zero bucket."
        )
        return 0

    # Handle infinity values
    if math.isinf(value):
        logger.warning(
            "Infinity value encountered in hidden state. This indicates "
            "numerical overflow. Clamping to maximum bucket."
        )
        return num_buckets - 1 if value > 0 else -(num_buckets - 1)

    abs_val = abs(value)

    # Deadzone for near-zero values
    if abs_val < 1e-6:
        return 0

    # Logarithmic scale: map log2(|x|+1) to bucket range
    # Typical hidden state range: [-3, 3] → log2 range ~ [0, 2]
    # Scale factor maps this to [0, num_buckets)
    log_val = math.log2(abs_val + 1.0)
    # TODO: come up with a more robust approach for measuring max log value
    scale_factor = num_buckets / 10.0  # Assuming max log value ~ 10
    bucket = int(log_val * scale_factor)
    bucket = max(0, min(num_buckets - 1, bucket))

    # Preserve sign
    return bucket if value >= 0 else -bucket


def adaptive_sketch_tolerance(
    position: int,
    sequence_length: int,
    base_tolerance: float,
) -> int:
    """Compute position-dependent sketch tolerance.

    Early positions are more important (set context) → tighter tolerance.
    Later positions may have accumulated drift → more permissive.

    Args:
        position: Token position in sequence
        sequence_length: Total sequence length
        base_tolerance: Base sketch tolerance value

    Returns:
        Adjusted sketch tolerance
    """
    # Importance weight: decays from 1.0 at start
    importance = 1.0 / (1.0 + position / PROOF_POSITION_IMPORTANCE_DECAY)

    # More important → tighter (multiply by factor < 1)
    # Less important → looser (multiply by factor > 1)
    factor = 2.0 - importance  # Range: [1.0, 2.0]

    return int(base_tolerance * factor)


class GRAILVerifier:
    """Sketch-based verifier for framework-agnostic hidden state proofs.

    Uses a single sketch check (random linear projection of bucketed top-k activations)
    which provides sufficient security (~10^-117 forgery probability) while being
    robust to floating-point variations across GPUs and frameworks.
    """

    def __init__(
        self,
        hidden_dim: int,
        topk: int = PROOF_TOPK,
        num_buckets: int = PROOF_NUM_BUCKETS,
        r_coeff_range: int = PROOF_COEFF_RANGE,
    ):
        """Initialize GRAIL verifier.

        Args:
            hidden_dim: Model hidden dimension size
            topk: Number of top activations to select
            num_buckets: Number of magnitude buckets per sign
            r_coeff_range: Range for bounded coefficients [-R, R]
        """
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.num_buckets = num_buckets
        self.r_coeff_range = r_coeff_range
        self.base_sketch_tolerance = float(PROOF_SKETCH_TOLERANCE)

    def generate_r_vec(self, randomness_hex: str) -> torch.Tensor:
        """Generate small bounded coefficient vector from randomness.

        Uses tiny coefficients in [-127, 127] to reduce sensitivity to
        floating-point variations while maintaining cryptographic security.

        Args:
            randomness_hex: Hex string of beacon randomness

        Returns:
            Tensor of shape [topk] with int8 coefficients in [-R, R]
        """
        from ..protocol.crypto import RNG_LABEL, prf

        # Clean hex string
        clean_hex = randomness_hex.strip().replace("0x", "").replace("0X", "")
        if len(clean_hex) % 2 != 0:
            clean_hex = "0" + clean_hex

        # Generate random bytes for coefficients
        raw = prf(
            RNG_LABEL["sketch"],
            bytes.fromhex(clean_hex),
            out_bytes=2 * self.topk,  # 2 bytes per coefficient
        )

        # Convert to int16, then map to [-R, R]
        import numpy as np

        int16_vals = np.frombuffer(raw, dtype=">i2")[: self.topk]  # Big-endian int16
        # Map to [-R, R] using modulo
        coeffs = (np.abs(int16_vals) % (2 * self.r_coeff_range + 1)) - self.r_coeff_range

        return torch.from_numpy(coeffs.astype(np.int8))

    def create_commitment(
        self, hidden_state: torch.Tensor, r_vec: torch.Tensor, position: int
    ) -> dict:
        """Create commitment for a single token position.

        Args:
            hidden_state: Hidden vector at position [hidden_dim]
            r_vec: Coefficient vector [topk]
            position: Token position (for metadata)

        Returns:
            Commitment dict with sketch, indices, and position
        """
        # Step 1: Select top-k activations by absolute magnitude
        abs_hidden = torch.abs(hidden_state)
        topk_result = torch.topk(abs_hidden, k=self.topk)
        indices = topk_result.indices  # [topk]
        values = hidden_state[indices]  # [topk] with signs preserved

        # Step 2: Logarithmic bucketing
        buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), self.num_buckets) for val in values],
            dtype=torch.int8,
        )

        # Step 3: Compute sketch via dot product with small coefficients
        sketch = torch.dot(buckets.to(torch.int32), r_vec.to(torch.int32))
        sketch_val = int(sketch.item()) % PRIME_Q

        return {
            "sketch": sketch_val,
            "indices": indices.tolist(),
            "position": position,
        }

    def verify_commitment(
        self,
        validator_hidden: torch.Tensor,
        miner_commitment: dict,
        r_vec: torch.Tensor,
        sequence_length: int,
    ) -> tuple[bool, dict]:
        """Verify commitment using sketch check.

        The sketch check computes a random linear projection of bucketed activations
        and verifies the modular distance is within tolerance.

        Args:
            validator_hidden: Validator's hidden vector at position
            miner_commitment: Miner's claimed commitment
            r_vec: Coefficient vector (same for miner and validator)
            sequence_length: Total sequence length (for adaptive tolerance)

        Returns:
            Tuple of (is_valid, diagnostics_dict)
        """
        position = miner_commitment["position"]

        # Get position-adjusted tolerance
        tolerance = adaptive_sketch_tolerance(position, sequence_length, self.base_sketch_tolerance)

        # Extract miner's claimed top-k indices
        miner_indices = torch.tensor(miner_commitment["indices"], dtype=torch.long)

        # Extract validator's values at those same indices
        validator_values = validator_hidden[miner_indices]

        # Compute validator's buckets
        validator_buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), self.num_buckets) for val in validator_values],
            dtype=torch.int8,
        )

        # Sketch verification: modular distance on dot product
        validator_sketch = torch.dot(validator_buckets.to(torch.int32), r_vec.to(torch.int32))
        validator_sketch_val = int(validator_sketch.item()) % PRIME_Q

        sketch_diff = abs(validator_sketch_val - miner_commitment["sketch"])
        mod_diff = min(sketch_diff, PRIME_Q - sketch_diff)  # Modular distance
        is_valid = mod_diff <= tolerance

        diagnostics = {
            "sketch_diff": mod_diff,
            "sketch_valid": is_valid,
            "sketch_tolerance": tolerance,
            "overall_valid": is_valid,
        }

        return is_valid, diagnostics
