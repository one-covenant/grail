"""MRS-based GRAIL cryptographic proof validator.

Verifies rollout tokens using Magnitude-Rank Sketch (MRS) for GPU/framework-agnostic
hidden state verification.

This validator replaces the legacy proof.py with a more robust approach that works
across HuggingFace Transformers, vLLM, SGLang, and different GPU/CUDA configurations.
"""

from __future__ import annotations

import logging

import torch

from ...protocol.mrs_verifier import MRSVerifier
from ...protocol.signatures import verify_commit_signature
from ...shared.constants import CHALLENGE_K, GRAIL_PROOF_VERSION_MRS, LAYER_INDEX
from ...shared.hf_compat import resolve_hidden_size, resolve_vocab_size
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class MRSProofValidator(Validator):
    """Verifies GRAIL cryptographic proof using MRS (Magnitude-Rank Sketch).

    Multi-layer verification:
    1. Signature binding (tokens, mrs_commitments, model, layer, randomness)
    2. Model inference to get hidden states
    3. MRS multi-check verification at challenged indices (sketch + rank + histogram)
    4. Caches logits for downstream validators (termination check)

    Security: ~10^-157 forgery probability with 3 independent checks at k=16 positions.
    """

    check_name = "proof_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        """Verify MRS proof and cache logits."""
        # Extract inputs
        try:
            tokens = ctx.commit["tokens"]
            # MRS commitments (list of dicts, one per position)
            mrs_commitments = ctx.commit.get("mrs_commitments")

            # Check proof version
            proof_version = ctx.commit.get(
                "proof_version", "v1"
            )  # Default to v1 for backward compat

            if proof_version != GRAIL_PROOF_VERSION_MRS:
                # Fall back to legacy validator if not MRS proof
                logger.debug(f"Non-MRS proof version: {proof_version}, skipping MRS validation")
                # For backward compatibility, we treat this as a soft failure
                # The system should use the legacy validator for v1 proofs
                ctx.checks[self.check_name] = False
                return False

            if not mrs_commitments:
                logger.debug("Missing mrs_commitments in commit")
                ctx.checks[self.check_name] = False
                return False

        except KeyError as e:
            logger.debug(f"Missing required field in commit: {e}")
            ctx.checks[self.check_name] = False
            return False

        # Validate structure
        if not isinstance(mrs_commitments, list) or len(tokens) != len(mrs_commitments):
            logger.debug(
                f"Invalid mrs_commitments: len(tokens)={len(tokens)}, "
                f"len(mrs_commitments)={len(mrs_commitments)}"
            )
            ctx.checks[self.check_name] = False
            return False

        # Minimum sequence length check
        seq_len = len(tokens)
        if seq_len < CHALLENGE_K:
            logger.debug(f"Sequence too short: {seq_len} < {CHALLENGE_K}")
            ctx.checks[self.check_name] = False
            return False

        # Verify commit signature binding
        if not verify_commit_signature(ctx.commit, ctx.prover_address):
            logger.debug("Commit signature verification failed")
            ctx.checks[self.check_name] = False
            return False

        # Verify model/layer binding
        model_info = ctx.commit.get("model", {})
        expected_model = ctx.model.name_or_path
        if model_info.get("name") != expected_model:
            logger.debug(f"Model mismatch: expected {expected_model}, got {model_info.get('name')}")
            ctx.checks[self.check_name] = False
            return False

        try:
            layer_claim = int(model_info.get("layer_index"))
        except (TypeError, ValueError):
            logger.debug("Invalid layer_index in commit")
            ctx.checks[self.check_name] = False
            return False

        if layer_claim != LAYER_INDEX:
            logger.debug(f"Layer mismatch: expected {LAYER_INDEX}, got {layer_claim}")
            ctx.checks[self.check_name] = False
            return False

        # Get beacon randomness
        beacon = ctx.commit.get("beacon", {})
        if not beacon or "randomness" not in beacon:
            logger.debug("Missing beacon randomness")
            ctx.checks[self.check_name] = False
            return False

        randomness_hex = beacon["randomness"]

        # Initialize MRS verifier
        hidden_dim = resolve_hidden_size(ctx.model.config)
        verifier = MRSVerifier(hidden_dim=hidden_dim)

        # Generate coefficient vector from randomness
        r_vec = verifier.generate_r_vec(randomness_hex)

        # Derive challenge indices deterministically
        from ...protocol.crypto import indices_from_root

        idxs = indices_from_root(tokens, ctx.challenge_randomness, seq_len, CHALLENGE_K)

        # Run model inference
        full_ids = torch.tensor(tokens, dtype=torch.long, device=ctx.device).unsqueeze(0)
        try:
            with torch.inference_mode():
                outs = ctx.model(full_ids, output_hidden_states=True)
        except RuntimeError as e:
            logger.error(f"Model inference failed: {e}")
            vocab_size = resolve_vocab_size(ctx.model.config)
            logger.error(f"Vocab={vocab_size}, tokens range=[{min(tokens)}, {max(tokens)}]")
            ctx.checks[self.check_name] = False
            return False

        h_layer = outs.hidden_states[LAYER_INDEX][0]

        # Cache logits for termination validator
        if outs.logits.size(1) >= 2:
            ctx.cached_logits = outs.logits[0, -2, :].detach().to("cpu")

        # Verify MRS commitments at challenged indices
        failed_checks = []
        for i in idxs:
            if i >= len(mrs_commitments):
                logger.debug(
                    f"Index {i} out of bounds for mrs_commitments length {len(mrs_commitments)}"
                )
                ctx.checks[self.check_name] = False
                return False

            is_valid, diagnostics = verifier.verify_commitment(
                h_layer[i], mrs_commitments[i], r_vec, seq_len
            )

            if not is_valid:
                failed_checks.append((i, diagnostics))
                logger.debug(
                    f"MRS verification failed at position {i}: "
                    f"sketch_diff={diagnostics['sketch_diff']} "
                    f"(tol={diagnostics['sketch_tolerance']}), "
                    f"rank_matches={diagnostics['rank_matches']}"
                    f"/{diagnostics['rank_tolerance']}, "
                    f"hist_diff={diagnostics['histogram_diff']}"
                    f"/{diagnostics['histogram_tolerance']}"
                )

        if failed_checks:
            logger.debug(
                f"MRS proof verification failed at {len(failed_checks)}/{CHALLENGE_K} positions"
            )
            ctx.checks[self.check_name] = False
            return False

        logger.debug("MRS proof verification successful")
        ctx.checks[self.check_name] = True
        return True
