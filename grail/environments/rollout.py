"""GRPO rollout dataclass and assembly."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GRPORollout:
    """Single rollout for GRPO training with GRAIL proof support."""

    tokens: list[int]
    token_logprobs: list[float]
    prompt_length: int
    completion_length: int
    reward: float
    advantage: float
    success: bool
    commitments: list[dict]
    signature: bytes
    beacon: dict
    proof_version: str


def assemble_rollouts(
    batch_data: list[tuple[list[int], int, float, dict]],
    proof_results: list[tuple[list[dict], list[float], bytes, dict, str]],
) -> list[GRPORollout]:
    """Assemble GRPORollout objects from generation + proof results.

    Args:
        batch_data: List of (all_ids, prompt_len, reward, info) tuples
        proof_results: Matching list of (commitments, logprobs, signature, beacon, proof_version)

    Returns:
        List of GRPORollout objects (advantages not yet computed)
    """
    rollouts: list[GRPORollout] = []
    for (all_ids, prompt_len, reward, info), (
        commitments,
        logprobs,
        signature,
        beacon,
        proof_version,
    ) in zip(batch_data, proof_results, strict=False):
        completion_ids = all_ids[prompt_len:]

        rollout = GRPORollout(
            tokens=all_ids,
            token_logprobs=[0.0] * prompt_len + logprobs,
            prompt_length=int(prompt_len),
            completion_length=int(len(completion_ids)),
            reward=reward,
            advantage=0.0,
            success=bool(info.get("success", False)),
            commitments=commitments,
            signature=signature,
            beacon=beacon,
            proof_version=proof_version,
        )
        logger.debug("Prompt length: %d", rollout.prompt_length)
        rollouts.append(rollout)
    return rollouts
