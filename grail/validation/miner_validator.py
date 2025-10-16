"""Single miner validation for GRAIL protocol.

Handles fetching, validating, and scoring a single miner's window submission.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import bittensor as bt
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..infrastructure.chain import GrailChainManager
from ..infrastructure.comms import file_exists, get_file
from ..shared.constants import ROLLOUTS_PER_PROBLEM
from ..shared.digest import compute_completion_digest
from .context import ValidationContext
from .pipeline import ValidationPipeline, get_hard_check_keys, get_soft_check_keys
from .types import MinerResults

logger = logging.getLogger(__name__)

# Validation constants (aligned with pre-refactor validate.py)
MAX_SAMPLES_PER_MINER_THRESHOLD = 20  # If <= this many rollouts, check all
MAX_SAMPLES_PER_MINER = 40  # If > this many rollouts, sample GRPO groups
SAMPLE_RATE = 0.10  # Fraction of GRPO groups to spot-check
STOCHASTIC_CHECK_FAILURE_THRESHOLD = 0.51  # Soft-failure fraction to gate wallet
GRPO_ADV_SUM_TOLERANCE = 0.01  # Sum of advantages should be ~0
REWARD_REL_TOL = 0.02  # Relative tolerance on reward bounds
REWARD_ABS_TOL = 1e-6  # Absolute tolerance on reward bounds
FAILURE_FLAG_KEY = "had_failure"


@dataclass
class ValidationMetrics:
    """Metrics from validating a miner's window."""

    valid_count: int
    checked_count: int
    total_inferences: int
    estimated_valid: int
    successful_rollouts: int
    estimated_successful: int
    unique_rollouts: int
    estimated_unique: int
    prompt_valid_count: int
    prompt_mismatch_count: int
    failure_flag: int

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary format for weight computation."""
        return {
            "valid": self.valid_count,
            "checked": self.checked_count,
            "total": self.total_inferences,
            "estimated_valid": self.estimated_valid,
            "successful": self.successful_rollouts,
            "estimated_successful": self.estimated_successful,
            "unique": self.unique_rollouts,
            "estimated_unique": self.estimated_unique,
            "prompt_valid": self.prompt_valid_count,
            "prompt_mismatch": self.prompt_mismatch_count,
            FAILURE_FLAG_KEY: self.failure_flag,
        }


class MinerValidator:
    """Validates a single miner's window submission.

    Responsibilities:
    1. Fetch window file from miner's bucket
    2. Validate signatures and GRAIL proofs
    3. Check GRPO group constraints
    4. Compute metrics and extrapolations
    5. Return structured results for weight computation

    Design:
    - Deterministic sampling per miner+window
    - Early exit on hard failures
    - Soft failure accumulation with thresholding
    - GRPO group validation (4 rollouts per problem)
    """

    def __init__(
        self,
        pipeline: ValidationPipeline,
        text_log_limit: int = 5,
    ):
        """Initialize miner validator.

        Args:
            pipeline: Environment-agnostic validation pipeline
            text_log_limit: Max number of text samples to log per miner
        """
        self._pipeline = pipeline
        self._text_log_limit = text_log_limit

        # Derive check keys from pipeline validators (single source of truth)
        self._hard_check_keys = get_hard_check_keys(pipeline)
        soft_check_keys = get_soft_check_keys(pipeline)
        self._soft_check_key = soft_check_keys[0] if soft_check_keys else None

        logger.debug(
            f"MinerValidator initialized with {len(self._hard_check_keys)} hard checks "
            f"and {len(soft_check_keys)} soft checks"
        )

    async def validate_miner(
        self,
        miner_hotkey: str,
        window: int,
        window_hash: str,
        window_rand: str,
        validator_wallet: Any,  # bt.wallet
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        credentials: Any,
        chain_manager: GrailChainManager,
        monitor: Any | None,
        uid_by_hotkey: dict[str, int],
        text_logs_emitted: dict[str, int],
        heartbeat_callback: Any = None,
    ) -> MinerResults:
        """Validate a single miner's window submission.

        Args:
            miner_hotkey: Miner's hotkey address
            window: Window start block
            window_hash: Block hash at window start
            window_rand: Combined randomness for the window
            validator_wallet: Validator's wallet for derivations
            model: Language model for validation
            tokenizer: Tokenizer for text decoding
            credentials: Default R2 credentials
            chain_manager: Chain manager for miner bucket lookup
            monitor: Optional monitoring client
            uid_by_hotkey: Mapping of hotkey to UID
            text_logs_emitted: Counter of text logs per miner
            heartbeat_callback: Optional callback to update watchdog

        Returns:
            MinerResults with validation outcome and metrics
        """
        uid = uid_by_hotkey.get(miner_hotkey)

        # Step 1: Fetch window file
        file_data = await self._fetch_window_file(
            miner_hotkey, window, credentials, chain_manager, uid
        )

        if file_data is None:
            return self._create_not_found_result(miner_hotkey, uid)

        # Step 2: Validate file structure
        validation_result = self._validate_file_structure(file_data, miner_hotkey, window, uid)

        if not validation_result["valid"]:
            return self._create_invalid_file_result(miner_hotkey, uid, validation_result["reason"])

        inferences = file_data["inferences"]
        total_inferences = len(inferences)

        # Step 3: Determine which rollouts to check (all or sample)
        indices_to_check, groups_map, group_index_by_id = self._determine_rollouts_to_check(
            inferences, miner_hotkey, window_rand, validator_wallet, total_inferences
        )

        # Step 4: Validate selected rollouts
        validation_state = await self._validate_rollouts(
            inferences=inferences,
            indices_to_check=indices_to_check,
            group_index_by_id=group_index_by_id,
            miner_hotkey=miner_hotkey,
            window=window,
            window_hash=window_hash,
            window_rand=window_rand,
            model=model,
            tokenizer=tokenizer,
            monitor=monitor,
            uid=uid,
            text_logs_emitted=text_logs_emitted,
            heartbeat_callback=heartbeat_callback,
        )

        # Step 5: Check for early failures
        if validation_state["hard_failure"] or validation_state["soft_gate_triggered"]:
            return await self._create_failure_result(
                miner_hotkey, uid, total_inferences, validation_state, monitor
            )

        # Step 6: Validate GRPO groups
        grpo_valid = self._validate_grpo_groups(validation_state["rollout_groups"])

        if not grpo_valid:
            return await self._create_grpo_failure_result(
                miner_hotkey, uid, total_inferences, validation_state, monitor
            )

        # Step 7: Log aggregated distribution metrics to W&B
        self._log_aggregated_distribution_metrics(
            uid=uid, miner_hotkey=miner_hotkey, validation_state=validation_state
        )

        # Step 8: Compute final metrics and return
        return await self._create_success_result(
            miner_hotkey=miner_hotkey,
            uid=uid,
            total_inferences=total_inferences,
            validation_state=validation_state,
            inferences=inferences,
            monitor=monitor,
        )

    async def _fetch_window_file(
        self,
        miner_hotkey: str,
        window: int,
        credentials: Any,
        chain_manager: GrailChainManager,
        uid: int | None,
    ) -> dict | None:
        """Fetch and download miner's window file.

        Returns:
            Window data dict or None if not found/error
        """
        filename = f"grail/windows/{miner_hotkey}-window-{window}.json"
        miner_bucket = chain_manager.get_bucket_for_hotkey(miner_hotkey)
        bucket_to_use = miner_bucket if miner_bucket else credentials

        # Check existence
        exists = await file_exists(filename, credentials=bucket_to_use, use_write=False)

        if not exists:
            logger.debug(f"No file found at {filename}")
            return None

        uid_str = str(uid) if uid is not None else f"{miner_hotkey[:12]}..."
        logger.info(f"📁 Found file for miner {uid_str}")

        # Download file
        window_data = await get_file(filename, credentials=bucket_to_use, use_write=False)

        if not window_data:
            logger.warning(f"Could not download {filename}")
            return None

        return window_data

    def _validate_file_structure(
        self, file_data: dict, miner_hotkey: str, window: int, uid: int | None
    ) -> dict[str, Any]:
        """Validate basic file structure and consistency.

        Returns:
            Dict with 'valid' bool and optional 'reason' string
        """
        file_wallet = file_data.get("wallet")
        file_window = file_data.get("window_start")

        # Validate wallet matches
        if file_wallet != miner_hotkey:
            logger.warning(
                f"Wallet mismatch in file: expected {miner_hotkey[:12]}..., "
                f"got {file_wallet[:12] if file_wallet else 'None'}..."
            )
            return {"valid": False, "reason": "wallet_mismatch"}

        # Validate window matches
        if file_window != window:
            logger.warning(f"Window mismatch in file: expected {window}, got {file_window}")
            return {"valid": False, "reason": "window_mismatch"}

        return {"valid": True}

    def _determine_rollouts_to_check(
        self,
        inferences: list[dict],
        miner_hotkey: str,
        window_rand: str,
        validator_wallet: Any,
        total_inferences: int,
    ) -> tuple[list[int], dict[str, list[int]], dict[str, int]]:
        """Determine which rollouts to validate (all or deterministic sample).

        Returns:
            (indices_to_check, groups_map, group_index_by_id)
        """
        groups_map = defaultdict(list)
        group_index_by_id: dict[str, int] = {}

        # Build group membership
        for idx, inf in enumerate(inferences):
            raw_gid = inf.get("rollout_group")
            if raw_gid is not None:
                group_id = str(raw_gid)
                groups_map[group_id].append(idx)
                if group_id not in group_index_by_id:
                    group_index_by_id[group_id] = len(group_index_by_id)

        # Check all or sample?
        if total_inferences <= MAX_SAMPLES_PER_MINER_THRESHOLD:
            indices_to_check = list(range(total_inferences))
            logger.info(f"🔍 Verifying all {total_inferences} rollouts")
            return indices_to_check, groups_map, group_index_by_id

        # Deterministic sampling
        num_groups = len(groups_map)
        groups_to_check = max(1, min(num_groups, int(num_groups * SAMPLE_RATE)))
        groups_to_check = min(groups_to_check, MAX_SAMPLES_PER_MINER // ROLLOUTS_PER_PROBLEM)

        # Deterministic RNG seed
        seed_material = f"{miner_hotkey}:{window_rand}"
        seed_int = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
        rng = random.Random(seed_int)

        def _group_digest(gidxs: list[int]) -> str:
            dig = hashlib.sha256()
            for i in sorted(gidxs):
                commit_json = json.dumps(
                    inferences[i].get("commit", {}), sort_keys=True, separators=(",", ":")
                )
                dig.update(hashlib.sha256(commit_json.encode()).digest())
            return dig.hexdigest()

        # Canonicalize group ordering
        group_keys = sorted(groups_map.keys(), key=lambda gid: _group_digest(groups_map[gid]))

        # Sample groups
        selected_groups = rng.sample(group_keys, groups_to_check)
        indices_to_check = []
        for group_id in selected_groups:
            indices_to_check.extend(groups_map[group_id])

        indices_to_check.sort()

        logger.info(
            f"📊 Spot checking {len(indices_to_check)}/{total_inferences} "
            f"rollouts from {groups_to_check}/{num_groups} groups"
        )

        return indices_to_check, groups_map, group_index_by_id

    async def _validate_rollouts(
        self,
        inferences: list[dict],
        indices_to_check: list[int],
        group_index_by_id: dict[str, int],
        miner_hotkey: str,
        window: int,
        window_hash: str,
        window_rand: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        monitor: Any | None,
        uid: int | None,
        text_logs_emitted: dict[str, int],
        heartbeat_callback: Any,
    ) -> dict[str, Any]:
        """Validate selected rollouts and accumulate state.

        Returns:
            Validation state dict with counters and flags
        """
        from ..protocol.signatures import derive_env_seed

        # Initialize state
        state = {
            "valid_count": 0,
            "checked_count": 0,
            "successful_rollouts": 0,
            "unique_rollouts": set(),
            "nonces_seen": set(),
            "rollout_groups": defaultdict(list),
            "wallet_rollouts_buffer": [],
            "soft_failures": 0,
            "hard_failure": False,
            "soft_gate_triggered": False,
            "prompt_valid_count": 0,
            "prompt_mismatch_count": 0,
            "pr_total": 0,
            "pr_invalid_sig": 0,
            "pr_invalid_proof": 0,
            "pr_processing_err": 0,
            # Distribution metrics accumulation
            "distribution_samples": [],  # List of (metrics, initial_metrics, reasons) tuples
        }

        total_planned_checks = len(indices_to_check)
        soft_fail_cutoff = max(
            1, math.ceil(STOCHASTIC_CHECK_FAILURE_THRESHOLD * max(1, total_planned_checks))
        )

        uid_str = str(uid) if uid is not None else f"{miner_hotkey[:12]}..."

        for inference_idx in indices_to_check:
            # Update watchdog
            if heartbeat_callback:
                try:
                    heartbeat_callback()
                except Exception:
                    pass

            inference = inferences[inference_idx]
            state["checked_count"] += 1

            try:
                # Basic consistency checks
                if not self._check_inference_consistency(
                    inference, window, window_hash, state, miner_hotkey
                ):
                    break

                # Signature verification
                if not self._verify_rollout_signature(inference):
                    state["pr_invalid_sig"] += 1
                    state["hard_failure"] = True
                    logger.warning("Invalid signature; invalidating uid")
                    break

                # Extract commit data
                commit_data = inference["commit"]
                rollout_meta = commit_data.get("rollout", {})

                # Reward validation is now handled by RewardValidator in the pipeline

                # Track rollout group (skip if missing to avoid "None" grouping)
                rollout_group_raw = inference.get("rollout_group")
                if rollout_group_raw is None:
                    state["hard_failure"] = True
                    logger.warning("Missing rollout_group; invalidating uid")
                    break
                rollout_group = str(rollout_group_raw)
                state["rollout_groups"][rollout_group].append(inference)

                # Derive file-order group index for deterministic seed derivation
                group_index = int(group_index_by_id.get(rollout_group, 0))

                # Inject canonical env field (validator-derived, never trust miner)
                seed_int = derive_env_seed(miner_hotkey, window_hash, group_index)

                # Debug: trace seed derivation inputs
                logger.debug(
                    (
                        "VALIDATOR SEED DERIVATION: hotkey=%s window_hash=%s "
                        "group_index=%d -> seed=%d"
                    ),
                    miner_hotkey[:12],
                    window_hash[:12],
                    group_index,
                    seed_int,
                )
                # Run validation pipeline with trusted validator-derived values
                ctx = ValidationContext(
                    commit=commit_data,
                    prover_address=miner_hotkey,
                    challenge_randomness=window_rand,
                    window_hash=window_hash,
                    group_index=group_index,
                    miner_uid=uid_str,
                    model=model,
                    tokenizer=tokenizer,
                    device=model.device,
                )

                if monitor:
                    with monitor.timer("profiling/rollout_verification"):
                        is_valid, checks = self._pipeline.validate(ctx)
                else:
                    is_valid, checks = self._pipeline.validate(ctx)

                state["pr_total"] += 1

                # Collect distribution metrics from context metadata
                if "distribution_metrics" in ctx.metadata:
                    state["distribution_samples"].append(
                        (
                            ctx.metadata.get("distribution_metrics"),
                            ctx.metadata.get("distribution_initial_metrics"),
                            ctx.metadata.get("distribution_reasons"),
                        )
                    )

                # Check hard and soft validation results
                if not self._process_validation_results(
                    checks, state, soft_fail_cutoff, total_planned_checks, uid_str, monitor
                ):
                    break

                # Track prompt validity (non-gating metric)
                self._track_prompt_validity(checks, state)

                # Log success
                state["valid_count"] += 1

                # Log sample text (debug)
                nonce = inference.get("nonce")
                await self._log_sample_text(
                    commit_data,
                    rollout_meta,
                    nonce,
                    miner_hotkey,
                    uid_str,
                    window,
                    tokenizer,
                    text_logs_emitted,
                    monitor,
                )

                # Track success and uniqueness
                if rollout_meta.get("success", False):
                    state["successful_rollouts"] += 1

                state["unique_rollouts"] = self._update_unique_rollouts(
                    state["unique_rollouts"], commit_data, rollout_meta
                )

                # Recompute per-rollout advantage check placeholder (group-based check later)
                state["wallet_rollouts_buffer"].append(inference)

            except Exception as e:
                logger.debug(f"Error processing inference: {e}")
                state["pr_processing_err"] += 1
                continue

        return state

    def _verify_rollout_signature(self, rollout_data: dict[str, Any]) -> bool:
        """Verify the signature of a rollout (replicated from CLI logic).

        Expects fields: challenge, hotkey, signature (hex string).
        Challenge format: "{episode_seed}|{block_hash}|{nonce}" (with delimiters).
        """
        try:
            challenge = rollout_data.get("challenge")
            hotkey = rollout_data.get("hotkey")
            signature = rollout_data.get("signature")

            # Validate required fields
            if not all([challenge, hotkey, signature]):
                return False

            # Validate types
            if not isinstance(challenge, str):
                return False
            if not isinstance(signature, str):
                return False

            keypair = bt.Keypair(ss58_address=str(hotkey))
            # Encode challenge to bytes to match signing logic (explicit UTF-8)
            challenge_bytes = challenge.encode("utf-8")
            return bool(keypair.verify(data=challenge_bytes, signature=bytes.fromhex(signature)))
        except Exception:
            return False

    def _check_inference_consistency(
        self, inference: dict, window: int, window_hash: str, state: dict, miner_hotkey: str
    ) -> bool:
        """Check basic inference consistency (window, hash, nonce).

        Returns:
            False if hard failure detected
        """
        if inference.get("window_start") != window:
            state["hard_failure"] = True
            logger.warning("Window mismatch in inference; invalidating uid")
            return False

        if inference.get("block_hash") != window_hash:
            state["hard_failure"] = True
            logger.warning("Block hash mismatch in inference; invalidating uid")
            return False

        nonce = inference.get("nonce")
        if nonce in state["nonces_seen"]:
            state["hard_failure"] = True
            logger.warning(f"Duplicate nonce {nonce}; invalidating uid")
            return False

        state["nonces_seen"].add(nonce)
        return True

    def _process_validation_results(
        self,
        checks: dict[str, bool],
        state: dict,
        soft_fail_cutoff: int,
        total_planned_checks: int,
        uid_str: str,
        monitor: Any | None,
    ) -> bool:
        """Process validation pipeline results and update state.

        Returns:
            False if hard failure or soft gate triggered
        """
        hard_valid = all(checks.get(k, False) for k in self._hard_check_keys)
        soft_valid = checks.get(self._soft_check_key, True) if self._soft_check_key else True

        # Log failures
        if not hard_valid:
            failed_check = next(
                (k for k in self._hard_check_keys if not checks.get(k, False)), None
            )
            if failed_check:
                logger.debug(f"CHECK_FAILURE type=hard failed_check={failed_check}")

        if not soft_valid and self._soft_check_key:
            logger.debug(f"CHECK_FAILURE type=soft failed_check={self._soft_check_key}")

        # Handle hard failure
        if not hard_valid:
            state["pr_invalid_proof"] += 1
            state["hard_failure"] = True
            failed_check = next(
                (k for k in self._hard_check_keys if not checks.get(k, False)), None
            )
            if failed_check:
                logger.warning(
                    f"Hard verification failed on check '{failed_check}'; invalidating uid"
                )
            else:
                logger.warning("Hard verification failed; invalidating uid")
            return False

        # Handle soft failure
        if not soft_valid:
            state["soft_failures"] += 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"soft_failures={state['soft_failures']}/{total_planned_checks}; "
                    f"threshold={soft_fail_cutoff}"
                )

            # Soft failure observed; threshold gating handled below

            if state["soft_failures"] >= soft_fail_cutoff:
                state["soft_gate_triggered"] = True
                soft_name = self._soft_check_key or "soft_check"
                logger.warning(
                    f"Soft-check threshold reached for '{soft_name}' "
                    f"({state['soft_failures']}/{total_planned_checks}); invalidating uid"
                )
                return False

        return True

    def _track_prompt_validity(self, checks: dict[str, bool], state: dict) -> None:
        """Track prompt validity metrics (non-gating).

        Uses environment-agnostic check names.
        """
        try:
            if checks.get("tokens_valid") and checks.get("proof_valid"):
                if checks.get("env_prompt_valid"):
                    state["prompt_valid_count"] += 1
                else:
                    state["prompt_mismatch_count"] += 1
        except Exception:
            pass

    async def _log_sample_text(
        self,
        commit_data: dict,
        rollout_meta: dict,
        nonce: str,
        miner_hotkey: str,
        uid_str: str,
        window: int,
        tokenizer: AutoTokenizer,
        text_logs_emitted: dict[str, int],
        monitor: Any | None,
    ) -> None:
        """Log sample text for debugging (respects limit)."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        if text_logs_emitted.get(miner_hotkey, 0) >= self._text_log_limit:
            return

        try:
            tokens = commit_data.get("tokens", [])
            if not isinstance(tokens, list) or not tokens:
                return

            prompt_len = int(rollout_meta.get("prompt_length", 0))
            completion_len = int(rollout_meta.get("completion_length", 0) or 0)

            if completion_len > 0 and prompt_len >= 0:
                completion_ids = tokens[prompt_len : prompt_len + completion_len]
            else:
                completion_ids = tokens[prompt_len:]

            problem_text = tokenizer.decode(tokens[:prompt_len], skip_special_tokens=False)
            text = tokenizer.decode(completion_ids, skip_special_tokens=False)

            reward_val = rollout_meta.get("total_reward", float("nan"))
            adv_val = rollout_meta.get("advantage", float("nan"))
            success_val = rollout_meta.get("success", False)

            logger.debug(
                f"TEXT[validate] nonce={nonce} reward={float(reward_val):.3f} "
                f"adv={float(adv_val):.3f} success={bool(success_val)} text={text}"
            )

            if monitor:
                await monitor.log_artifact(
                    f"{uid_str}/validation/sample_text",
                    {
                        "window": window,
                        "group": uid_str,
                        "nonce": nonce,
                        "reward": float(reward_val),
                        "advantage": float(adv_val),
                        "success": bool(success_val),
                        "text": f"Problem:\n{problem_text}\n\nCompletion:\n{text}",
                    },
                    "text",
                )

            text_logs_emitted[miner_hotkey] = text_logs_emitted.get(miner_hotkey, 0) + 1

        except Exception:
            pass

    def _update_unique_rollouts(
        self, unique_rollouts: set[str], commit_data: dict, rollout_meta: dict
    ) -> set[str]:
        """Update set of unique rollout hashes."""
        try:
            tokens = commit_data.get("tokens", [])
            if not tokens:
                return unique_rollouts

            prompt_len = int(rollout_meta.get("prompt_length", 0) or 0)
            completion_ids = tokens[prompt_len:]

            digest_input = json.dumps(
                completion_ids, separators=(",", ":"), ensure_ascii=False
            ).encode()
            rollout_hash = hashlib.sha256(digest_input).hexdigest()

            unique_rollouts.add(rollout_hash)

        except Exception as e:
            logger.debug(f"Failed to hash rollout: {e}")

        return unique_rollouts

    def _log_aggregated_distribution_metrics(
        self, uid: int | None, miner_hotkey: str, validation_state: dict
    ) -> None:
        """Compute and log aggregated distribution metrics to W&B for this miner.

        Aggregates across all validated rollouts in the window:
        - Mean/median/std of distribution metrics
        - Trigger rates for each check type
        - Overall suspicious rollout percentage
        """
        try:
            import wandb  # type: ignore

            if getattr(wandb, "run", None) is None:
                return

            samples = validation_state.get("distribution_samples", [])
            if not samples:
                return

            uid_str = str(uid) if uid is not None else f"{miner_hotkey[:12]}..."

            # Separate metrics, initial_metrics, and reasons
            metrics_list = [s[0] for s in samples if s[0] is not None]
            initial_list = [s[1] for s in samples if s[1] is not None]
            reasons_list = [s[2] for s in samples if s[2] is not None]

            if not metrics_list:
                return

            # Aggregate full-sequence metrics
            import numpy as np

            agg_data: dict[str, float] = {}

            for key in ["mean", "median", "q10", "min", "bc", "low_frac", "high_frac", "mid_frac"]:
                values = [m.get(key, 0.0) for m in metrics_list if key in m]
                if values:
                    agg_data[f"{uid_str}/distribution/avg_{key}"] = float(np.mean(values))
                    agg_data[f"{uid_str}/distribution/std_{key}"] = float(np.std(values))

            # Aggregate initial-window metrics
            if initial_list:
                for key in ["q10", "bc", "median", "mean", "min"]:
                    values = [m.get(key, 0.0) for m in initial_list if key in m]
                    if values:
                        agg_data[f"{uid_str}/distribution/avg_init_{key}"] = float(np.mean(values))

            # Compute trigger rates (% of rollouts that triggered each check)
            n_samples = len(reasons_list)
            if n_samples > 0:
                trigger_keys = ["median_low", "min_low", "bimodal_full", "bimodal_initial"]
                for trigger_key in trigger_keys:
                    trigger_count = sum(1 for r in reasons_list if r and r.get(trigger_key, False))
                    trigger_rate = float(trigger_count) / float(n_samples)
                    agg_data[f"{uid_str}/distribution/trigger_rate_{trigger_key}"] = trigger_rate

                # Overall triggered rate
                any_triggered = sum(1 for r in reasons_list if r and any(r.values()))
                overall_rate = float(any_triggered) / float(n_samples)
                agg_data[f"{uid_str}/distribution/trigger_rate_overall"] = overall_rate

            # Add sample count
            agg_data[f"{uid_str}/distribution/n_rollouts"] = float(n_samples)

            wandb.log(agg_data)

        except Exception as e:
            # Never fail validation on logging errors
            logger.debug(f"Failed to log aggregated distribution metrics: {e}")

    def _validate_grpo_groups(self, rollout_groups: dict[str, list[dict]]) -> bool:
        """Validate GRPO group constraints.

        Returns:
            True if all groups valid, False otherwise
        """
        for group_id, group_rollouts in rollout_groups.items():
            # Check group size
            if len(group_rollouts) != ROLLOUTS_PER_PROBLEM:
                logger.warning(
                    f"GRPO group {group_id} has {len(group_rollouts)} rollouts, "
                    f"expected {ROLLOUTS_PER_PROBLEM}; invalidating uid"
                )
                return False

            # Check advantages sum to zero
            advantages = []
            rewards = []
            for r in group_rollouts:
                meta = r.get("commit", {}).get("rollout", {})
                advantages.append(float(meta.get("advantage", 0.0)))
                rewards.append(float(meta.get("total_reward", 0.0)))
            advantage_sum = sum(advantages)

            if abs(advantage_sum) > GRPO_ADV_SUM_TOLERANCE:
                logger.warning(
                    f"GRPO group {group_id} advantages don't sum to 0: {advantage_sum}; "
                    "invalidating uid"
                )
                return False

            # Optional: sanity check claimed advantages vs group rewards (zero-mean, normalized)
            try:
                n = len(rewards)
                if n > 0:
                    mean_r = sum(rewards) / n
                    centered = [r - mean_r for r in rewards]
                    # Avoid div-by-zero; compare shapes loosely
                    denom = max(1e-8, (sum(x * x for x in centered) / n) ** 0.5)
                    recomputed = [x / denom for x in centered]
                    # Allow small tolerance per element
                    for a, b in zip(advantages, recomputed, strict=False):
                        if abs(a - b) > 1e-3:
                            logger.warning(
                                "GRPO group %s advantage mismatch vs recomputed", group_id
                            )
                            return False
            except Exception:
                # Don't hard fail on numerical issues; already checked zero-sum
                pass

            # Check same base environment seed
            base_seeds = [r.get("commit", {}).get("env", {}).get("seed") for r in group_rollouts]

            if len(set(base_seeds)) != 1:
                logger.warning(
                    f"GRPO group {group_id} has different base env seeds; invalidating uid"
                )
                return False

        return True

    def _create_not_found_result(self, miner_hotkey: str, uid: int | None) -> MinerResults:
        """Create result for missing window file."""
        return MinerResults(
            hotkey=miner_hotkey,
            uid=uid,
            found_file=False,
            metrics=None,
            rollouts=[],
            processed_counts=(0, 0, 0, 0),
            digest_counter=None,
            total_inferences_in_file=0,
        )

    def _create_invalid_file_result(
        self, miner_hotkey: str, uid: int | None, reason: str
    ) -> MinerResults:
        """Create result for invalid file structure."""
        logger.warning(f"Invalid file structure: {reason}")
        return MinerResults(
            hotkey=miner_hotkey,
            uid=uid,
            found_file=True,
            metrics=None,
            rollouts=[],
            processed_counts=(0, 0, 0, 0),
            digest_counter=None,
            total_inferences_in_file=0,
        )

    async def _create_failure_result(
        self,
        miner_hotkey: str,
        uid: int | None,
        total_inferences: int,
        state: dict,
        monitor: Any | None,
    ) -> MinerResults:
        """Create result for validation failure."""
        uid_str = str(uid) if uid is not None else "unknown"

        metrics = ValidationMetrics(
            valid_count=0,
            checked_count=state["checked_count"],
            total_inferences=total_inferences,
            estimated_valid=0,
            successful_rollouts=0,
            estimated_successful=0,
            unique_rollouts=0,
            estimated_unique=0,
            prompt_valid_count=0,
            prompt_mismatch_count=state["prompt_mismatch_count"],
            failure_flag=1,
        )

        logger.info(
            f"❌ Rejected uid={uid_str} hard={state['hard_failure']} "
            f"soft={state['soft_failures']}/{state['checked_count']}"
        )

        # Log failure metrics to monitor
        if monitor:
            await monitor.log_gauge(f"{uid_str}/had_failure", 1.0)
            try:
                await monitor.log_gauge(f"{uid_str}/prompt_valid", 0.0)
                await monitor.log_gauge(
                    f"{uid_str}/prompt_mismatch", float(state["prompt_mismatch_count"])
                )
                # Log soft failure metrics
                await monitor.log_gauge(
                    f"{uid_str}/soft_failures_count", float(state["soft_failures"])
                )
                soft_failure_ratio = (
                    state["soft_failures"] / state["checked_count"]
                    if state["checked_count"] > 0
                    else 0.0
                )
                await monitor.log_gauge(f"{uid_str}/soft_failures_ratio", soft_failure_ratio)
            except Exception:
                pass

        return MinerResults(
            hotkey=miner_hotkey,
            uid=uid,
            found_file=True,
            metrics=metrics.to_dict(),
            rollouts=[],
            processed_counts=(
                state["pr_total"],
                state["pr_invalid_sig"],
                state["pr_invalid_proof"],
                state["pr_processing_err"],
            ),
            digest_counter=None,
            total_inferences_in_file=total_inferences,
        )

    async def _create_grpo_failure_result(
        self,
        miner_hotkey: str,
        uid: int | None,
        total_inferences: int,
        state: dict,
        monitor: Any | None,
    ) -> MinerResults:
        """Create result for GRPO validation failure."""
        return await self._create_failure_result(
            miner_hotkey, uid, total_inferences, state, monitor
        )

    async def _create_success_result(
        self,
        miner_hotkey: str,
        uid: int | None,
        total_inferences: int,
        validation_state: dict,
        inferences: list[dict],
        monitor: Any | None,
    ) -> MinerResults:
        """Create result for successful validation."""
        # Extrapolate from sample
        checked = validation_state["checked_count"]
        valid = validation_state["valid_count"]
        successful = validation_state["successful_rollouts"]
        unique = len(validation_state["unique_rollouts"])

        sample_pass_rate = (valid / checked) if checked > 0 else 0
        estimated_valid = int(total_inferences * sample_pass_rate)

        unique_rate = (unique / checked) if checked > 0 else 0
        estimated_unique = int(total_inferences * unique_rate)

        success_rate = (successful / checked) if checked > 0 else 0
        estimated_successful = int(total_inferences * success_rate)

        metrics = ValidationMetrics(
            valid_count=valid,
            checked_count=checked,
            total_inferences=total_inferences,
            estimated_valid=estimated_valid,
            successful_rollouts=successful,
            estimated_successful=estimated_successful,
            unique_rollouts=unique,
            estimated_unique=estimated_unique,
            prompt_valid_count=validation_state["prompt_valid_count"],
            prompt_mismatch_count=validation_state["prompt_mismatch_count"],
            failure_flag=0,
        )

        logger.info(
            f"✅ {valid}/{checked} checked, ~{estimated_valid}/{total_inferences} estimated valid, "
            f"{successful}/{total_inferences} successful, {unique}/{total_inferences} unique"
        )

        # Log metrics to monitor
        if monitor:
            uid_str = str(uid) if uid is not None else "unknown"
            await monitor.log_gauge(f"{uid_str}/had_failure", 0.0)
            try:
                await monitor.log_gauge(
                    f"{uid_str}/prompt_valid", float(validation_state["prompt_valid_count"])
                )
                await monitor.log_gauge(
                    f"{uid_str}/prompt_mismatch", float(validation_state["prompt_mismatch_count"])
                )
                # Log soft failure metrics (even for successful miners)
                await monitor.log_gauge(
                    f"{uid_str}/soft_failures_count", float(validation_state["soft_failures"])
                )
                soft_failure_ratio = (
                    validation_state["soft_failures"] / validation_state["checked_count"]
                    if validation_state["checked_count"] > 0
                    else 0.0
                )
                await monitor.log_gauge(f"{uid_str}/soft_failures_ratio", soft_failure_ratio)
            except Exception:
                pass

        # Build digest counter over ALL rollouts
        digest_counter: Counter[str] = Counter()
        for inf in inferences:
            commit_data = inf.get("commit", {})
            rollout_meta = commit_data.get("rollout", {})
            dig = compute_completion_digest(commit_data, rollout_meta)
            if dig:
                digest_counter[dig] += 1

        return MinerResults(
            hotkey=miner_hotkey,
            uid=uid,
            found_file=True,
            metrics=metrics.to_dict(),
            rollouts=validation_state["wallet_rollouts_buffer"],
            processed_counts=(
                validation_state["pr_total"],
                validation_state["pr_invalid_sig"],
                validation_state["pr_invalid_proof"],
                validation_state["pr_processing_err"],
            ),
            digest_counter=digest_counter,
            total_inferences_in_file=total_inferences,
        )
