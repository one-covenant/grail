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
from ..forward_utils import forward_single_layer

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
        if not isinstance(model_info, dict):
            logger.debug(
                f"[proof_valid] Invalid model field in commit | "
                f"model_type={type(model_info).__name__}"
            )
            ctx.checks[self.check_name] = False
            return False

        # NOTE: Never hard-fail on raw name_or_path mismatches. Miners/validators
        # run on different machines with different local cache paths, so
        # `model.name_or_path` will often differ even when weights match.
        expected_model = getattr(ctx.model, "name_or_path", None)
        claimed_model = model_info.get("name")

        # Best-effort checkpoint identity. We intentionally avoid parsing strings
        # here; the canonical checkpoint window is attached at model load time
        # by `grail.model.provider.get_model()` as `model.grail_checkpoint_window`.
        expected_checkpoint_window: int | None = getattr(ctx.model, "grail_checkpoint_window", None)
        claimed_checkpoint_window: int | None = None
        claimed_raw = model_info.get("checkpoint_window")
        if claimed_raw is not None:
            try:
                claimed_checkpoint_window = int(claimed_raw)
            except (TypeError, ValueError):
                claimed_checkpoint_window = None

        # If both sides provide a comparable checkpoint window, enforce it.
        if (
            expected_checkpoint_window is not None
            and claimed_checkpoint_window is not None
            and claimed_checkpoint_window != expected_checkpoint_window
        ):
            logger.warning(
                "[proof_valid] CHECKPOINT WINDOW MISMATCH (model binding) | "
                "expected_checkpoint_window=%s | claimed_checkpoint_window=%s | "
                "expected_model=%s | claimed_model=%s",
                expected_checkpoint_window,
                claimed_checkpoint_window,
                expected_model,
                claimed_model,
            )
            ctx.checks[self.check_name] = False
            return False

        if claimed_model != expected_model:
            logger.debug(
                "[proof_valid] Non-fatal model identifier mismatch | expected=%s | got=%s",
                expected_model,
                claimed_model,
            )

        try:
            layer_index_value = model_info.get("layer_index")
            if layer_index_value is None:
                raise ValueError("layer_index is None")
            layer_claim = int(layer_index_value)
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

        # Use pre-computed results from batched forward pass if available,
        # otherwise run a single-sequence forward pass (backward compatible).
        if ctx.cached_hidden_states is not None and ctx.cached_logits is not None:
            h_layer = ctx.cached_hidden_states
            logger.debug(
                "VALIDATOR HIDDEN STATE: using pre-computed cache | seq_len=%d",
                seq_len,
            )
        else:
            full_ids = torch.tensor(tokens, dtype=torch.long, device=ctx.device).unsqueeze(0)
            try:
                with torch.inference_mode():
                    h_batch, logits_batch = forward_single_layer(
                        ctx.model,
                        full_ids,
                        None,
                        LAYER_INDEX,
                    )
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

            h_layer = h_batch[0]

            logger.debug(
                "VALIDATOR HIDDEN STATE: single forward pass | seq_len=%d "
                "tokens_first_4=%s tokens_last_4=%s",
                seq_len,
                tokens[:4],
                tokens[-4:] if len(tokens) >= 4 else tokens,
            )

            ctx.cached_logits = logits_batch[0].detach().to("cpu")

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
                # Get detailed miner commitment info
                miner_commit = commitments[i]
                miner_indices = miner_commit.get("indices", [])[:5]  # First 5

                logger.warning(
                    f"[proof_valid] Commitment verification FAILED at position {i} | "
                    f"sketch_diff={diagnostics['sketch_diff']} | "
                    f"tolerance={diagnostics['sketch_tolerance']} | "
                    f"validator_sketch={diagnostics.get('validator_sketch')} | "
                    f"miner_sketch={diagnostics.get('miner_sketch')} | "
                    f"miner_indices_sample={miner_indices} | "
                    f"validator_hidden_norm={float(h_layer[i].norm().item()):.4f} | "
                    f"token_id={tokens[i]} | "
                    f"uid={ctx.miner_uid}"
                )

        if failed_checks:
            failure_details = "; ".join(
                f"pos={i}(diff={d['sketch_diff']},tol={d['sketch_tolerance']})"
                for i, d in failed_checks
            )
            logger.warning(
                f"[proof_valid] FAILED | "
                f"failed_positions={len(failed_checks)}/{len(idxs)} | "
                f"details=[{failure_details}] | "
                f"seq_len={seq_len} | "
                f"model={model_info.get('name')} | "
                f"layer={layer_claim} | "
                f"uid={ctx.miner_uid} | "
                f"This failure typically means miner and validator have different model weights"
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
