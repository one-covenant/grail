"""Miner scoring from validation results."""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class MinerScorer:
    """Aggregates validation results into per-miner metrics.

    Converts list of validation results into metrics dict format
    compatible with weight computation.
    """

    @staticmethod
    def score_miner_window(
        validation_results: list[tuple[bool, dict[str, bool]]],
        rollouts: list[dict],
        checked_count: int,
        total_count: int,
    ) -> dict[str, int]:
        """Aggregate validation results into miner metrics.

        Args:
            validation_results: List of (is_valid, checks) tuples
            rollouts: Original rollout dicts (for extracting metadata)
            checked_count: Number of rollouts validated
            total_count: Total rollouts for this miner

        Returns:
            Metrics dict with keys: valid, successful, unique, estimated_*, etc.
        """
        valid_count = sum(1 for ok, _ in validation_results if ok)

        # Count successful (solved problem)
        successful = sum(
            1
            for (ok, _), r in zip(validation_results, rollouts)
            if ok and r.get("commit", {}).get("rollout", {}).get("success", False)
        )

        # Count unique (by hashing completion tokens)
        unique_hashes = set()
        for (ok, _), r in zip(validation_results, rollouts):
            if ok:
                digest = MinerScorer._compute_rollout_hash(r)
                if digest:
                    unique_hashes.add(digest)

        # Extrapolate from sample to estimate total
        sample_rate = checked_count / total_count if total_count > 0 else 0
        if sample_rate > 0:
            estimated_valid = int(valid_count / sample_rate)
            estimated_successful = int(successful / sample_rate)
            estimated_unique = int(len(unique_hashes) / sample_rate)
        else:
            estimated_valid = valid_count
            estimated_successful = successful
            estimated_unique = len(unique_hashes)

        # Count prompt validation (for monitoring)
        prompt_valid = sum(
            1 for _, checks in validation_results if checks.get("prompt_valid", False)
        )
        prompt_mismatch = sum(
            1 for _, checks in validation_results if not checks.get("prompt_valid", True)
        )

        return {
            "valid": valid_count,
            "checked": checked_count,
            "total": total_count,
            "estimated_valid": estimated_valid,
            "successful": successful,
            "estimated_successful": estimated_successful,
            "unique": len(unique_hashes),
            "estimated_unique": estimated_unique,
            "prompt_valid": prompt_valid,
            "prompt_mismatch": prompt_mismatch,
        }

    @staticmethod
    def _compute_rollout_hash(rollout: dict) -> str | None:
        """Hash completion tokens for uniqueness tracking."""
        try:
            commit = rollout.get("commit", {})
            tokens = commit.get("tokens", [])
            rollout_meta = commit.get("rollout", {})
            prompt_len = int(rollout_meta.get("prompt_length", 0))

            # Hash completion portion only
            completion_ids = tokens[prompt_len:]
            digest_input = json.dumps(
                completion_ids, separators=(",", ":"), ensure_ascii=False
            ).encode()
            return hashlib.sha256(digest_input).hexdigest()

        except Exception as e:
            logger.debug(f"Failed to hash rollout: {e}")
            return None
