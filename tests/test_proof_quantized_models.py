"""Quantized model tests for GRAIL proof verification.

Tests that GRAIL proofs fail when comparing quantized models against
full-precision models, as expected due to quantization-induced differences.

Dependencies for quantized models:
- GPTQ models: pip install optimum auto-gptq
- AWQ models: pip install autoawq
- FP8 models: Work with base transformers
"""

from __future__ import annotations

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
    reason="GPU required for quantized model tests (CUDA not available)",
)


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def prompt() -> str:
    """Generate realistic SAT prompt for quantized model tests."""
    from .proof_test_utils import generate_realistic_sat_prompt

    # Use raw prompt (no tokenizer) since we test different quantizations
    return generate_realistic_sat_prompt("test_quantized", 0.4)


class TestProofQuantizedModels:
    """GRAIL proof should reject proofs when quantization differs between systems."""

    @requires_gpu
    @pytest.mark.parametrize(
        "full_precision,quantized",
        [
            # Official Qwen GPTQ-Int4 (requires optimum+auto-gptq)
            (
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
            ),
            (
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
            ),
        ],
    )
    def test_gptq_int4_mismatch(
        self,
        full_precision: str,
        quantized: str,
        device: str,
        prompt: str,
    ) -> None:
        """Full vs GPTQ-Int4 should fail verification."""
        full_model, full_tok = load_model_and_tokenizer(full_precision, device)
        quant_model, _ = load_model_and_tokenizer(quantized, device)

        try:
            randomness = hash_hex(f"gptq4|{full_precision}|{quantized}")
            challenge = hash_hex(f"chal|gptq4|{full_precision}")

            proof = create_proof(
                full_model,
                full_tok,
                prompt,
                randomness,
                device,
            )

            is_valid, diagnostics = verify_proof(
                quant_model,
                proof,
                challenge,
                device,
            )

            assert not is_valid, f"Expected failure: {full_precision} vs {quantized}"

            failed_checks = [d for d in diagnostics if not d["overall_valid"]]
            assert len(failed_checks) > 0

        finally:
            del full_model
            del quant_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @requires_gpu
    @pytest.mark.parametrize(
        "full_precision,quantized",
        [
            # Official Qwen GPTQ-Int8 (requires optimum+auto-gptq)
            (
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
            ),
            (
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
            ),
        ],
    )
    def test_gptq_int8_mismatch(
        self,
        full_precision: str,
        quantized: str,
        device: str,
        prompt: str,
    ) -> None:
        """Full vs GPTQ-Int8 should fail verification."""
        full_model, full_tok = load_model_and_tokenizer(full_precision, device)
        quant_model, _ = load_model_and_tokenizer(quantized, device)

        try:
            randomness = hash_hex(f"gptq8|{full_precision}")
            challenge = hash_hex("chal_gptq8")

            proof = create_proof(
                full_model,
                full_tok,
                prompt,
                randomness,
                device,
            )

            is_valid, _ = verify_proof(
                quant_model,
                proof,
                challenge,
                device,
            )

            assert not is_valid, f"Expected failure: {full_precision} vs {quantized}"

        finally:
            del full_model
            del quant_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @requires_gpu
    @pytest.mark.parametrize(
        "full_precision,quantized",
        [
            # Official Qwen AWQ (requires autoawq)
            (
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            ),
            (
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            ),
            (
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct-AWQ",
            ),
        ],
    )
    def test_awq_mismatch(
        self,
        full_precision: str,
        quantized: str,
        device: str,
        prompt: str,
    ) -> None:
        """Full vs AWQ quantized should fail verification."""
        full_model, full_tok = load_model_and_tokenizer(full_precision, device)
        quant_model, _ = load_model_and_tokenizer(quantized, device)

        try:
            randomness = hash_hex(f"awq|{full_precision}")
            challenge = hash_hex("chal_awq")

            proof = create_proof(
                full_model,
                full_tok,
                prompt,
                randomness,
                device,
            )

            is_valid, _ = verify_proof(
                quant_model,
                proof,
                challenge,
                device,
            )

            assert not is_valid, f"Expected failure: {full_precision} vs {quantized}"

        finally:
            del full_model
            del quant_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @requires_gpu
    @pytest.mark.parametrize(
        "fp8,full_precision",
        [
            ("Qwen/Qwen3-4B-Instruct-2507-FP8", "Qwen/Qwen3-4B-Instruct-2507"),
            ("Qwen/Qwen3-4B-FP8", "Qwen/Qwen3-4B"),
            ("Qwen/Qwen3-4B-Thinking-2507-FP8", "Qwen/Qwen3-4B-Thinking-2507"),
        ],
    )
    def test_fp8_mismatch(
        self,
        fp8: str,
        full_precision: str,
        device: str,
        prompt: str,
    ) -> None:
        """FP8 vs full-precision should fail verification."""
        q_model, q_tok = load_model_and_tokenizer(fp8, device)
        f_model, _ = load_model_and_tokenizer(full_precision, device)

        try:
            randomness = hash_hex(f"fp8|{fp8}")
            challenge = hash_hex("chal_fp8")

            proof = create_proof(
                q_model,
                q_tok,
                prompt,
                randomness,
                device,
            )

            is_valid, _ = verify_proof(
                f_model,
                proof,
                challenge,
                device,
            )

            assert not is_valid, f"Expected failure: {fp8} (FP8) vs {full_precision}"

        finally:
            del q_model
            del f_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
