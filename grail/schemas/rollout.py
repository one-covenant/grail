"""Pydantic models for GRAIL rollout data validation.

Defines the expected schema for miner-submitted rollouts with:
- Type validation
- Range constraints
- Cross-field consistency checks
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModelInfo(BaseModel):
    """Model identification in commit."""

    name: str
    layer_index: int


class BeaconInfo(BaseModel):
    """Randomness beacon (drand or deterministic)."""

    randomness: str = Field(pattern=r"^[0-9a-fA-F]+$")  # Hex string
    # Note: beacon.round removed - never used in validation


class SATProblemInfo(BaseModel):
    """SAT problem specification."""

    seed: str
    num_vars: int = Field(gt=0)
    clauses: list[list[int]]
    difficulty: float = Field(ge=0.0, le=1.0)


class RolloutMetadata(BaseModel):
    """Rollout metadata and results."""

    # Required (validated)
    prompt_length: int = Field(ge=0)
    completion_length: int = Field(gt=0, le=1024)  # MAX_NEW_TOKENS
    success: bool
    total_reward: float
    advantage: float
    assignment: list[bool]

    # Training metadata (not validated, kept for future GRPO training)
    trajectory: list[Any] = Field(default_factory=list)
    token_logprobs: list[float] = Field(default_factory=list)
    satisfied_clauses: int = 0


class Commit(BaseModel):
    """Commit data with tokens, activation commitments, and metadata."""

    tokens: list[int] = Field(min_length=10)  # >= CHALLENGE_K
    commitments: list[dict]
    proof_version: str
    model: ModelInfo
    signature: str
    beacon: BeaconInfo
    sat_problem: SATProblemInfo
    rollout: RolloutMetadata

    @field_validator("commitments")
    @classmethod
    def validate_commitments_length(cls, v: list[dict], info) -> list[dict]:
        """Commitments must match tokens length."""
        tokens = info.data.get("tokens", [])
        if len(v) != len(tokens):
            raise ValueError(f"commitments length {len(v)} must equal tokens length {len(tokens)}")
        return v

    @field_validator("rollout")
    @classmethod
    def validate_lengths_sum(cls, v: RolloutMetadata, info) -> RolloutMetadata:
        """Prompt + completion must equal total tokens."""
        tokens = info.data.get("tokens", [])
        total_len = v.prompt_length + v.completion_length
        if total_len != len(tokens):
            raise ValueError(
                f"prompt_length({v.prompt_length}) + "
                f"completion_length({v.completion_length}) = {total_len} "
                f"must equal tokens length {len(tokens)}"
            )
        return v

    @field_validator("rollout")
    @classmethod
    def validate_assignment_if_success(cls, v: RolloutMetadata, info) -> RolloutMetadata:
        """Assignment length must match num_vars if success claimed."""
        if v.success:
            sat = info.data.get("sat_problem", {})
            if isinstance(sat, dict):
                num_vars = sat.get("num_vars", 0)
            elif isinstance(sat, SATProblemInfo):
                num_vars = sat.num_vars
            else:
                num_vars = 0

            if len(v.assignment) != num_vars:
                raise ValueError(
                    f"assignment length {len(v.assignment)} must equal "
                    f"num_vars {num_vars} when success=True"
                )
        return v


class RolloutData(BaseModel):
    """Complete rollout data from miner.

    Note: 'proof' field removed entirely - validator re-derives challenge indices
    deterministically from tokens + challenge_randomness. Miner-supplied indices
    were completely ignored.
    """

    # Required (validated)
    window_start: int
    nonce: int
    block_hash: str
    sat_seed: str
    rollout_group: int
    challenge: str
    hotkey: str
    signature: str
    commit: Commit

    # Optional metadata (not validated, for provenance/debugging)
    block: int | None = None
    randomness: str | None = None
    use_drand: bool | None = None
    difficulty: float | None = None
    timestamp: float | None = None
    rollout_index: int | None = None
    total_in_group: int | None = None
