"""Cross-framework test for GRAIL proof verification.

Tests that GRAIL proofs are deterministic and work correctly with production-like
SAT prompts resembling actual GRAIL mining scenarios.
"""

from typing import cast

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from grail.shared.constants import (
    CHALLENGE_K,
    GRAIL_PROOF_VERSION,
    MODEL_NAME,
)

from .proof_test_utils import (
    create_proof,
    verify_proof,
)

# GPU requirement check
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for GRAIL proof cross-framework tests (CUDA not available)",
)


@pytest.fixture(scope="module")
def device() -> str:
    """Device fixture."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model(device: str) -> PreTrainedModel:
    """Model fixture (module-scoped to avoid reloading)."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map=device
    )


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    """Tokenizer fixture (module-scoped)."""
    return cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )


@pytest.fixture
def production_prompts() -> list[str]:
    """Production-like SAT prompt suite."""
    return [
        # SAT prompt 1: Full production format with instructions
        (
            "Problem:\n"
            "You are given a problem.\n"
            "Think about the problem and provide your working out.\n"
            "Place it between <start_working_out> and <end_working_out>.\n"
            "Then, provide your solution between <SOLUTION></SOLUTION>"
            "Keep the reasoning succinct (≤25 steps, ≤500 tokens).<|im_end|>\n"
            "SAT Problem:\n"
            "SAT Problem (seed: 4ba38907):\n"
            "Variables: 7\n"
            "Clauses:\n"
            "  (x7 OR NOT x2 OR x3)\n"
            "  (x4 OR x5 OR x2)\n"
            "  (NOT x2 OR x5 OR x7)\n"
            "  (x2 OR NOT x4 OR NOT x6)\n"
            "  (NOT x1 OR NOT x7 OR x4)\n"
            "  (NOT x4 OR NOT x5 OR x3)\n"
            "  (x3 OR x6 OR x5)\n"
            "  (NOT x2 OR x3 OR NOT x1)\n"
            "  (NOT x4 OR x7 OR NOT x1)\n\n"
            "Provide your final assignment between <SOLUTION></SOLUTION> as "
            "space-separated 0/1 values for x1..xN.\n"
            "<start_working_out>"
        ),
        # SAT prompt 2: Compact format
        (
            "You are a helpful AI. Solve the SAT instance described below. "
            "Think step by step within <start_working_out>..."
            "</end_working_out> "
            "and then give the final assignment inside <SOLUTION>..."
            "</SOLUTION>.\n\n"
            "SAT (seed: abcd1234): Vars=10, Clauses=12.\n"
            "(x1 OR x2 OR NOT x3)\n(x4 OR NOT x5 OR x6)\n"
            "(NOT x1 OR x7 OR x8)\n(x2 OR x9 OR NOT x10)\n"
            "(NOT x6 OR x3 OR x5)\n(x10 OR NOT x2 OR x7)\n"
            "<start_working_out>"
        ),
        # SAT prompt 3: System/user format
        (
            "System: You are given a problem. Follow the instructions "
            "strictly.\n"
            "User: Provide a valid assignment for the following CNF.\n"
            "Remember to include <start_working_out> and <SOLUTION> tags.\n\n"
            "CNF: (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (x2 ∨ ¬x3 ∨ x4) ∧ (¬x4 ∨ x1)\n"
            "<start_working_out>"
        ),
        # Generic math (non-SAT)
        (
            "You are a helpful assistant. Solve the arithmetic problem below.\n"
            "Show your work between <start_working_out> and "
            "<end_working_out>, then give the final numeric result inside "
            "<SOLUTION></SOLUTION>.\n\n"
            "Compute: sum_{i=1}^{50} i^2.\n"
            "<start_working_out>"
        ),
    ]


class TestProofCrossFramework:
    """Cross-framework compatibility tests for GRAIL proof."""

    @requires_gpu
    @pytest.mark.parametrize("prompt_idx", [0, 1, 2, 3])
    def test_hf_to_hf_multiple_prompts(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        device: str,
        production_prompts: list[str],
        prompt_idx: int,
    ) -> None:
        """Test GRAIL proof with production-like prompts (HF → HF, same run)."""
        prompt = production_prompts[prompt_idx]
        from .proof_test_utils import hash_hex

        randomness = hash_hex(f"rand|{prompt_idx}|{prompt[:64]}")
        challenge = hash_hex(f"chal|{prompt_idx}|{prompt[-64:]}")

        # Generate proof
        proof = create_proof(
            model,
            tokenizer,  # type: ignore[arg-type]
            prompt,
            randomness,
            device,
        )

        assert len(proof["tokens"]) >= CHALLENGE_K, "Not enough tokens for challenge"
        assert len(proof["commitments"]) == len(proof["tokens"])

        # Verify proof
        is_valid, diagnostics = verify_proof(model, proof, challenge, device)

        # All checks should pass perfectly (same run)
        assert is_valid, f"Prompt {prompt_idx} verification failed"

        # Verify perfect match
        avg_sketch_diff = sum(d["sketch_diff"] for d in diagnostics) / len(diagnostics)
        avg_rank_matches = sum(d["rank_matches"] for d in diagnostics) / len(diagnostics)
        avg_hist_diff = sum(d["histogram_diff"] for d in diagnostics) / len(diagnostics)

        assert avg_sketch_diff == 0.0, "Same run should have 0 sketch diff"
        assert avg_rank_matches == 5.0, "Same run should have 5/5 rank matches"
        assert avg_hist_diff == 0.0, "Same run should have 0 histogram diff"

    @requires_gpu
    def test_hf_to_hf_determinism(self, tokenizer: AutoTokenizer, device: str) -> None:
        """Test determinism across separate model loads."""
        prompt = (
            "Problem:\nYou are given a problem.\n"
            "Think about the problem and provide your working out.\n"
            "Place it between <start_working_out> and <end_working_out>.\n"
            "Then, provide your solution between <SOLUTION></SOLUTION>"
            "Keep the reasoning succinct.<|im_end|>\n"
            "SAT: (x1 OR x2) (¬x1 OR x3) (x2 OR ¬x3)\n"
            "<start_working_out>"
        )
        randomness = "feedbeefcafebabe1234567890abcdef"
        challenge = "deadc0de1337"

        # First run: generate proof
        model1 = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, device_map=device
        )
        proof = create_proof(
            model1,
            tokenizer,  # type: ignore[arg-type]
            prompt,
            randomness,
            device,
        )
        del model1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Second run: verify with fresh model
        model2 = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, device_map=device
        )
        is_valid, diagnostics = verify_proof(model2, proof, challenge, device)
        del model2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        assert is_valid, "Determinism test failed"

        # Verify perfect match across loads
        max_sketch_diff = max(d["sketch_diff"] for d in diagnostics)
        min_rank_matches = min(d["rank_matches"] for d in diagnostics)

        assert max_sketch_diff == 0, f"Sketch drift across loads: {max_sketch_diff}"
        assert min_rank_matches == 5, "Rank mismatch across loads"

    @requires_gpu
    def test_proof_structure(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        device: str,
        production_prompts: list[str],
    ) -> None:
        """Test that generated proofs have correct structure."""
        prompt = production_prompts[0]
        randomness = "feedbeef"
        proof = create_proof(
            model,
            tokenizer,  # type: ignore[arg-type]
            prompt,
            randomness,
            device,
        )

        assert "tokens" in proof
        assert "commitments" in proof
        assert "proof_version" in proof
        assert "randomness" in proof
        assert proof["proof_version"] == GRAIL_PROOF_VERSION

    @requires_gpu
    def test_commitment_integrity(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        device: str,
        production_prompts: list[str],
    ) -> None:
        """Test that each commitment in proof is well-formed."""
        prompt = production_prompts[0]
        randomness = "cafebabe"
        proof = create_proof(
            model,
            tokenizer,  # type: ignore[arg-type]
            prompt,
            randomness,
            device,
        )

        for idx, commitment in enumerate(proof["commitments"]):
            assert "sketch" in commitment, f"Position {idx} missing sketch"
            assert "indices" in commitment, f"Position {idx} missing indices"
            assert "top_5_ranks" in commitment, f"Position {idx} missing ranks"
            assert "histogram" in commitment, f"Position {idx} missing histogram"
            assert commitment["position"] == idx, f"Position mismatch at {idx}"

            # Type and size checks
            assert isinstance(commitment["sketch"], int)
            assert len(commitment["indices"]) == 256
            assert len(commitment["top_5_ranks"]) == 5
            assert len(commitment["histogram"]) == 33  # 2*16+1

    @requires_gpu
    @pytest.mark.slow
    def test_verification_metrics_distribution(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        device: str,
        production_prompts: list[str],
    ) -> None:
        """Test that verification metrics are reasonable across prompts."""
        all_sketch_diffs = []
        all_rank_matches = []
        all_hist_diffs = []

        from .proof_test_utils import hash_hex

        for idx, prompt in enumerate(production_prompts):
            randomness = hash_hex(f"test_metrics_{idx}")
            challenge = hash_hex(f"challenge_{idx}")

            proof = create_proof(
                model,
                tokenizer,  # type: ignore[arg-type]
                prompt,
                randomness,
                device,
            )
            is_valid, diagnostics = verify_proof(model, proof, challenge, device)

            assert is_valid, f"Verification failed for prompt {idx}"

            # Collect metrics
            for d in diagnostics:
                all_sketch_diffs.append(d["sketch_diff"])
                all_rank_matches.append(d["rank_matches"])
                all_hist_diffs.append(d["histogram_diff"])

        # All should be perfect for same-framework
        assert max(all_sketch_diffs) == 0, "Expected 0 sketch diff for HF→HF"
        assert min(all_rank_matches) == 5, "Expected 5/5 rank matches for HF→HF"
        assert max(all_hist_diffs) == 0, "Expected 0 histogram diff for HF→HF"
