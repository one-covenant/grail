"""GRAIL cryptographic proof validator.

Verifies rollout tokens using GPU/framework-agnostic hidden state verification.

This validator works across HuggingFace Transformers, vLLM, SGLang, and different
GPU/CUDA configurations using an activation sketch proof system.
"""

from __future__ import annotations

import logging

import torch

from ...protocol.grail_verifier import GRAILVerifier
from ...protocol.signatures import verify_commit_signature
from ...shared.constants import CHALLENGE_K, GRAIL_PROOF_VERSION, LAYER_INDEX
from ...shared.hf_compat import resolve_hidden_size, resolve_vocab_size
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class GRAILProofValidator(Validator):
    """Verifies GRAIL cryptographic proof using activation sketch verification.

    Multi-layer verification:
    1. Signature binding (tokens, commitments, model, layer, randomness)
    2. Model inference to get hidden states
    3. Multi-check verification at challenged indices (sketch + rank + histogram)
    4. Caches logits for downstream validators (termination check)

    Security: ~10^-157 forgery probability with 3 independent checks at k=16 positions.
    """

    check_name = "proof_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        """Verify GRAIL proof and cache logits."""
        # Extract inputs
        try:
            tokens = ctx.commit["tokens"]
            # Activation commitments (list of dicts, one per position)
            commitments = ctx.commit.get("commitments")

            # Check proof version
            proof_version = ctx.commit.get("proof_version")

            if not proof_version or proof_version != GRAIL_PROOF_VERSION:
                logger.debug(f"Invalid or missing proof version: {proof_version}")
                ctx.checks[self.check_name] = False
                return False

            if not commitments:
                logger.debug("Missing commitments in proof")
                ctx.checks[self.check_name] = False
                return False

        except KeyError as e:
            logger.debug(f"Missing required field in commit: {e}")
            ctx.checks[self.check_name] = False
            return False

        # Validate structure
        if not isinstance(commitments, list) or len(tokens) != len(commitments):
            logger.debug(
                f"Invalid commitments: len(tokens)={len(tokens)}, "
                f"len(commitments)={len(commitments)}"
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

        # Initialize GRAIL verifier
        hidden_dim = resolve_hidden_size(ctx.model)
        verifier = GRAILVerifier(hidden_dim=hidden_dim)

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

        # Verify proof commitments at challenged indices
        failed_checks = []
        for i in idxs:
            if i >= len(commitments):
                logger.debug(f"Index {i} out of bounds for commitments length {len(commitments)}")
                ctx.checks[self.check_name] = False
                return False

            is_valid, diagnostics = verifier.verify_commitment(
                h_layer[i], commitments[i], r_vec, seq_len
            )

            if not is_valid:
                failed_checks.append((i, diagnostics))
                logger.debug(
                    f"Proof verification failed at position {i}: "
                    f"sketch_diff={diagnostics['sketch_diff']} "
                    f"(tol={diagnostics['sketch_tolerance']}), "
                    f"rank_matches={diagnostics['rank_matches']}"
                    f"/{diagnostics['rank_tolerance']}, "
                    f"hist_diff={diagnostics['histogram_diff']}"
                    f"/{diagnostics['histogram_tolerance']}"
                )

        if failed_checks:
            logger.debug(
                f"Proof verification failed at {len(failed_checks)}/{CHALLENGE_K} positions"
            )
            ctx.checks[self.check_name] = False
            return False

        logger.debug("GRAIL proof verification successful")
        ctx.checks[self.check_name] = True
        return True
