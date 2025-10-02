"""Model-mismatch tests for GRAIL proof verification.
These tests ensure that when miner and validator use different models,
the GRAIL proof verification fails (as expected). We keep defaults tiny to run fast,
and gate larger models via environment flags.
"""

from __future__ import annotations

import os

import pytest
import torch

from .proof_test_utils import (
    create_proof,
    hash_hex,
    load_model_and_tokenizer,
    verify_proof,
)

# GPU requirement check
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for GRAIL proof model mismatch tests (CUDA not available)",
)


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def prompt() -> str:
    """Generate realistic SAT prompt for model mismatch tests."""
    from .proof_test_utils import generate_realistic_sat_prompt

    # Use raw prompt (no tokenizer) since we test different models
    return generate_realistic_sat_prompt("test_model_mismatch", 0.4)


def _model_pairs_tiny() -> list[tuple[str, str]]:
    """Tiny models that are quick to download and run for CI."""
    # Use publicly available tiny models with different architectures
    return [
        (
            "hf-internal-testing/tiny-random-gpt2",
            "hf-internal-testing/tiny-random-GPTNeoForCausalLM",
        ),
    ]


def _model_pairs_large_from_env() -> list[tuple[str, str]]:
    """Optional large-model pairs enabled via env flags (avoid OOM by default).

    Set GRAIL_TEST_LARGE_MODELS=1 to enable. Optionally select subsets via:
    - GRAIL_TEST_1B=1: Qwen3 0.6B-1.7B models (fast, low memory)
    - GRAIL_TEST_3B=1: Qwen2.5/3 3B-4B models (moderate memory)
    - GRAIL_TEST_4B=1: Qwen3 4B-14B models (high memory, slow)
    If GRAIL_TEST_LARGE_MODELS=1 but no size flags are set, defaults to tiny.
    """
    pairs: list[tuple[str, str]] = []

    # Only add pairs if large models are explicitly enabled
    if os.getenv("GRAIL_TEST_LARGE_MODELS") == "1":
        # Check if any specific size flags are set
        has_size_flags = any(
            [
                os.getenv("GRAIL_TEST_1B") == "1",
                os.getenv("GRAIL_TEST_3B") == "1",
                os.getenv("GRAIL_TEST_4B") == "1",
            ]
        )

        # If no size flags, default to 1B for basic testing
        if not has_size_flags:
            pairs += [
                (
                    "hf-internal-testing/tiny-random-gpt2",
                    "hf-internal-testing/tiny-random-GPTNeoForCausalLM",
                ),
            ]
        else:
            # 1B class - Small models for basic testing
            if os.getenv("GRAIL_TEST_1B") == "1":
                pairs += [
                    (
                        "Qwen/Qwen3-0.6B",
                        "Qwen/Qwen2.5-0.5B",
                    ),
                    (
                        "Qwen/Qwen3-1.7B",
                        "Qwen/Qwen2.5-1.5B-Instruct",
                    ),
                ]
            # 3B class - Medium models
            if os.getenv("GRAIL_TEST_3B") == "1":
                pairs += [
                    (
                        "Qwen/Qwen2.5-3B-Instruct",
                        "Qwen/Qwen3-4B",
                    ),
                ]
            # 4B+ class - Larger models (memory intensive)
            if os.getenv("GRAIL_TEST_4B") == "1":
                pairs += [
                    (
                        "Qwen/Qwen3-4B-Instruct-2507",
                        "Qwen/Qwen2.5-7B-Instruct",
                    ),
                    (
                        "Qwen/Qwen3-8B",
                        "Qwen/Qwen2.5-14B-Instruct",
                    ),
                ]

    return pairs


class TestProofModelMismatch:
    """Verification must fail when miner and validator models differ."""

    @requires_gpu
    @pytest.mark.parametrize("miner_name,validator_name", _model_pairs_tiny())
    def test_mismatch_tiny_models(
        self,
        miner_name: str,
        validator_name: str,
        device: str,
        prompt: str,
    ) -> None:
        miner, tokenizer_m = load_model_and_tokenizer(miner_name, device)
        validator, _ = load_model_and_tokenizer(validator_name, device)

        try:
            randomness = hash_hex(f"rand|tiny|{miner_name}|{validator_name}")
            challenge = hash_hex(f"chal|tiny|{miner_name}|{validator_name}")

            # Generate proof with miner model
            proof = create_proof(
                miner,
                tokenizer_m,
                prompt,
                randomness,
                device,
            )

            # Verify with different validator model
            is_valid, _ = verify_proof(validator, proof, challenge, device)

            assert not is_valid, (
                "Expected verification failure with different models: "
                f"{miner_name} vs {validator_name}"
            )
        finally:
            del miner
            del validator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # EOF

    @requires_gpu
    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("GRAIL_TEST_LARGE_MODELS") != "1",
        reason=("Large model tests disabled (set GRAIL_TEST_LARGE_MODELS=1 to enable)"),
    )
    @pytest.mark.parametrize(
        "miner_name,validator_name",
        _model_pairs_large_from_env(),
    )
    def test_mismatch_large_models(
        self,
        miner_name: str,
        validator_name: str,
        device: str,
        prompt: str,
    ) -> None:
        miner, tokenizer_m = load_model_and_tokenizer(miner_name, device)
        validator, _ = load_model_and_tokenizer(validator_name, device)

        try:
            randomness = hash_hex(f"rand|large|{miner_name}|{validator_name}")
            challenge = hash_hex(f"chal|large|{miner_name}|{validator_name}")

            proof = create_proof(
                miner,
                tokenizer_m,
                prompt,
                randomness,
                device,
            )
            is_valid, _ = verify_proof(
                validator,
                proof,
                challenge,
                device,
            )
            assert not is_valid, (
                "Expected verification failure with different large models: "
                f"{miner_name} vs {validator_name}"
            )
        finally:
            del miner
            del validator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
