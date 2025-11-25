"""GRAIL cryptographic proof validator.

Verifies rollout tokens using GPU/framework-agnostic hidden state verification.

This validator works across HuggingFace Transformers, vLLM, SGLang, and different
GPU/CUDA configurations using a sketch-based proof system (random linear projection
of bucketed top-k activations).
"""

from __future__ import annotations

import logging

import torch

from ...protocol.crypto import indices_from_root_in_range
from ...protocol.grail_verifier import GRAILVerifier
from ...protocol.signatures import verify_commit_signature
from ...shared.constants import CHALLENGE_K, GRAIL_PROOF_VERSION, LAYER_INDEX
from ...shared.hf_compat import resolve_hidden_size, resolve_vocab_size
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class GRAILProofValidator(Validator):
    """Verifies GRAIL cryptographic proof using sketch-based verification.

    Verification steps:
    1. Signature binding (tokens, commitments, model, layer, randomness)
    2. Model inference to get hidden states
    3. Sketch verification at challenged indices
    4. Caches logits for downstream validators (termination check)

    Security: ~10^-117 forgery probability with sketch check at k=16 positions.
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
                logger.debug(
                    f"[proof_valid] Proof version validation failed | "
                    f"Expected: {GRAIL_PROOF_VERSION} | "
                    f"Got: {proof_version}"
                )
                ctx.checks[self.check_name] = False
                return False

            if not commitments:
                logger.debug(
                    f"[proof_valid] Commitments missing or empty | commitments={commitments}"
                )
                ctx.checks[self.check_name] = False
                return False

        except KeyError as e:
            logger.debug(f"[proof_valid] Required field missing in commit | Missing field: {e}")
            ctx.checks[self.check_name] = False
            return False

        # Validate structure
        if not isinstance(commitments, list) or len(tokens) != len(commitments):
            logger.debug(
                f"[proof_valid] Commitments structure mismatch | "
                f"tokens_len={len(tokens)} | "
                f"commitments_len={len(commitments)} | "
                f"commitments_type={type(commitments).__name__}"
            )
            ctx.checks[self.check_name] = False
            return False

        # Determine effective domain: prefer completion region
        rollout = ctx.commit.get("rollout", {})
        prompt_len = int(rollout.get("prompt_length", 0))
        completion_len = int(rollout.get("completion_length", 0))

        # Enforce minimum completion length
        if completion_len < CHALLENGE_K:
            logger.debug(
                f"[proof_valid] Completion too short | required>={CHALLENGE_K} got={completion_len} | "
                f"prompt_len={prompt_len} seq_len={len(tokens)}"
            )
            ctx.checks[self.check_name] = False
            return False

        seq_len = len(tokens)
        start_idx = prompt_len
        end_idx = prompt_len + completion_len

        # Verify commit signature binding
        if not verify_commit_signature(ctx.commit, ctx.prover_address):
            logger.debug(
                f"[proof_valid] Signature verification failed | "
                f"prover_address={ctx.prover_address} | "
                f"tokens_hash={hash(tuple(tokens))} | "
                f"Possible causes: invalid signature, tampered data, wrong prover"
            )
            ctx.checks[self.check_name] = False
            return False

        # Verify model/layer binding
        model_info = ctx.commit.get("model", {})
        expected_model = getattr(ctx.model, "name_or_path", None)
        if model_info.get("name") != expected_model:
            logger.debug(
                f"[proof_valid] Model mismatch | "
                f"Expected: {expected_model} | "
                f"Got: {model_info.get('name')}"
            )
            ctx.checks[self.check_name] = False
            return False

        try:
            layer_claim = int(model_info.get("layer_index"))
        except (TypeError, ValueError):
            logger.debug(
                f"[proof_valid] Invalid layer_index in commit | "
                f"layer_index_value={model_info.get('layer_index')} | "
                f"layer_index_type={type(model_info.get('layer_index')).__name__}"
            )
            ctx.checks[self.check_name] = False
            return False

        if layer_claim != LAYER_INDEX:
            logger.debug(
                f"[proof_valid] Layer mismatch | Expected: {LAYER_INDEX} | Got: {layer_claim}"
            )
            ctx.checks[self.check_name] = False
            return False

        # Get beacon randomness
        beacon = ctx.commit.get("beacon", {})
        if not beacon or "randomness" not in beacon:
            logger.debug(
                f"[proof_valid] Beacon randomness missing | "
                f"beacon_present={bool(beacon)} | "
                f"beacon_keys={list(beacon.keys()) if beacon else []}"
            )
            ctx.checks[self.check_name] = False
            return False

        randomness_hex = beacon["randomness"]

        # Initialize GRAIL verifier
        hidden_dim = resolve_hidden_size(ctx.model)
        verifier = GRAILVerifier(hidden_dim=hidden_dim)

        # Generate coefficient vector from randomness
        r_vec = verifier.generate_r_vec(randomness_hex)

        # Derive challenge indices deterministically within selected domain
        idxs = indices_from_root_in_range(
            tokens,
            ctx.challenge_randomness,
            start_idx,
            end_idx,
            CHALLENGE_K,
        )

        # Run model inference with hidden states
        full_ids = torch.tensor(tokens, dtype=torch.long, device=ctx.device).unsqueeze(0)
        try:
            with torch.inference_mode():
                outs = ctx.model(full_ids, output_hidden_states=True)
        except RuntimeError as e:
            vocab_size = resolve_vocab_size(ctx.model.config)
            logger.error(
                f"[proof_valid] Model inference failed | "
                f"error={str(e)} | "
                f"vocab_size={vocab_size} | "
                f"token_range=[{min(tokens)}, {max(tokens)}] | "
                f"seq_len={seq_len} | "
                f"device={ctx.device}"
            )
            ctx.checks[self.check_name] = False
            return False

        h_layer = outs.hidden_states[LAYER_INDEX][0]

        # Log validator's hidden state computation for debugging
        logger.debug(
            "VALIDATOR HIDDEN STATE COMPUTATION: seq_len=%d "
            "tokens_first_4=%s tokens_last_4=%s attention_mask=None position_ids=None",
            seq_len,
            tokens[:4],
            tokens[-4:] if len(tokens) >= 4 else tokens,
        )

        # Cache full logits for downstream validators (termination + distribution)
        ctx.cached_logits = outs.logits[0].detach().to("cpu")

        # Verify proof commitments at challenged indices
        failed_checks = []
        for idx_num, i in enumerate(idxs):
            if i >= len(commitments):
                logger.debug(
                    f"[proof_valid] Challenge index out of bounds | "
                    f"index={i} | "
                    f"commitments_len={len(commitments)} | "
                    f"sequence_length={seq_len}"
                )
                ctx.checks[self.check_name] = False
                return False

            # Log sample validator commitments for debugging
            if idx_num == 0:
                miner_commit = commitments[i]
                logger.debug(
                    "VALIDATOR COMMITMENT pos=%d token_id=%d "
                    "miner_sketch_hash=%s miner_rank_hash=%s validator_hidden_norm=%.6f",
                    i,
                    tokens[i],
                    miner_commit.get("sketch_hash", "")[:16],
                    miner_commit.get("rank_hash", "")[:16],
                    float(h_layer[i].norm().item()),
                )

            is_valid, diagnostics = verifier.verify_commitment(
                h_layer[i], commitments[i], r_vec, seq_len
            )

            if not is_valid:
                failed_checks.append((i, diagnostics))
                logger.debug(
                    f"[proof_valid] Commitment verification failed at position {i} | "
                    f"sketch_diff={diagnostics['sketch_diff']:.6f} | "
                    f"tolerance={diagnostics['sketch_tolerance']:.6f}"
                )

        if failed_checks:
            failure_details = "; ".join(f"pos={i}" for i, _ in failed_checks)
            logger.debug(
                f"[proof_valid] FAILED | "
                f"failed_positions={len(failed_checks)}/{len(idxs)} | "
                f"positions=[{failure_details}] | "
                f"seq_len={seq_len} | "
                f"model={model_info.get('name')} | "
                f"layer={layer_claim}"
            )
            ctx.checks[self.check_name] = False
            return False

        logger.debug(
            f"[proof_valid] SUCCESS | "
            f"seq_len={seq_len} | "
            f"verified_positions={len(idxs)} | "
            f"model={model_info.get('name')} | "
            f"layer={layer_claim}"
        )
        ctx.checks[self.check_name] = True
        return True
