"""Shared proof test utilities to keep tests DRY.

Functions here are imported by multiple test modules.
"""

from __future__ import annotations

import hashlib
import random

import pytest
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from grail.environments.sat import create_sat_prompt, generate_sat_problem
from grail.protocol.crypto import indices_from_root
from grail.protocol.grail_verifier import GRAILVerifier
from grail.shared.constants import (
    CHALLENGE_K,
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    PROOF_TOPK,
)
from grail.shared.hf_compat import resolve_hidden_size


def hash_hex(material: str) -> str:
    return hashlib.sha256(material.encode()).hexdigest()


def ensure_min_tokens(tokenizer: PreTrainedTokenizerBase, text: str, min_tokens: int) -> str:
    filler = (
        "\n\nNote: Think step by step and keep reasoning succinct. "
        "Provide only the required tags and final solution."
    )
    candidate = text
    while True:
        t = tokenizer.__call__(candidate, return_tensors="pt")
        if int(t["input_ids"].shape[1]) >= min_tokens:
            return candidate
        candidate += filler


def create_proof(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    randomness_hex: str,
    device: str,
) -> dict:
    """Generate proof using HF, robust to small model size variations."""
    prompt = ensure_min_tokens(tokenizer, prompt, max(CHALLENGE_K + 4, 24))
    inputs = tokenizer.__call__(prompt, return_tensors="pt").to(device)
    tokens = inputs["input_ids"][0].tolist()

    hidden_dim = resolve_hidden_size(model)
    topk = min(int(hidden_dim), int(PROOF_TOPK))
    verifier = GRAILVerifier(hidden_dim=hidden_dim, topk=topk)
    r_vec = verifier.generate_r_vec(randomness_hex)

    commitments = []
    with torch.inference_mode():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        layer_idx = min(LAYER_INDEX, len(outputs.hidden_states) - 1)
        h_layer = outputs.hidden_states[layer_idx][0]
        for pos in range(len(tokens)):
            commitment = verifier.create_commitment(h_layer[pos], r_vec, pos)
            commitments.append(commitment)

    return {
        "tokens": tokens,
        "commitments": commitments,
        "proof_version": GRAIL_PROOF_VERSION,
        "randomness": randomness_hex,
    }


def verify_proof(
    model: PreTrainedModel,
    proof: dict,
    challenge_randomness: str,
    device: str,
) -> tuple[bool, list[dict]]:
    """Verify proof with robust guards for mismatch cases."""
    tokens = proof["tokens"]
    commitments = proof["commitments"]
    randomness_hex = proof["randomness"]

    hidden_dim = resolve_hidden_size(model)
    commit_topk = int(len(commitments[0]["indices"]))
    verifier = GRAILVerifier(hidden_dim=hidden_dim, topk=commit_topk)
    r_vec = verifier.generate_r_vec(randomness_hex)

    # Early guard: ensure tokens exist in validator vocab
    try:
        embeddings = model.get_input_embeddings()
        vocab_size = int(embeddings.weight.shape[0])  # noqa
    except Exception:  # pragma: no cover
        vocab_size = None
    if vocab_size is not None and max(tokens) >= vocab_size:
        return False, [
            {
                "error": "token_id_out_of_vocab",
                "max_token": int(max(tokens)),
                "vocab_size": vocab_size,
            }
        ]

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(token_tensor.unsqueeze(0), output_hidden_states=True)
    layer_idx = min(LAYER_INDEX, len(outputs.hidden_states) - 1)
    h_layer = outputs.hidden_states[layer_idx][0]
    idxs = indices_from_root(tokens, challenge_randomness, len(tokens), CHALLENGE_K)

    all_valid = True
    diagnostics_list = []
    validator_hidden_dim = int(h_layer.size(-1))
    for i in idxs:
        miner_indices = commitments[i]["indices"]
        if max(miner_indices) >= validator_hidden_dim:
            diagnostics_list.append(
                {
                    "position": int(i),
                    "error": "hidden_index_out_of_bounds",
                    "validator_dim": validator_hidden_dim,
                    "max_commit_index": int(max(miner_indices)),
                }
            )
            all_valid = False
            continue
        is_valid, diagnostics = verifier.verify_commitment(
            h_layer[i], commitments[i], r_vec, len(tokens)
        )
        diagnostics_list.append({"position": i, **diagnostics})
        if not is_valid:
            all_valid = False

    return all_valid, diagnostics_list


def load_model_and_tokenizer(
    name: str, device: str
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model+tokenizer or skip gracefully if unavailable.

    Supports:
    - Standard FP16/BF16/FP8 via transformers
    - GPTQ via auto-gptq (AutoGPTQForCausalLM.from_quantized)
    - AWQ via autoawq (AutoAWQForCausalLM.from_quantized)
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_lower = name.lower()

    # GPTQ models
    if "gptq" in repo_lower:
        try:
            from auto_gptq import AutoGPTQForCausalLM  # type: ignore

            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            mdl = AutoGPTQForCausalLM.from_quantized(
                name,
                device_map="auto",
                trust_remote_code=True,
                use_safetensors=True,
            )
            return mdl, tok
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"skip (GPTQ): {name}: {exc}")

    # AWQ models
    if "awq" in repo_lower:
        try:
            from awq import AutoAWQForCausalLM  # type: ignore

            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            mdl = AutoAWQForCausalLM.from_quantized(
                name,
                device_map="auto",
                trust_remote_code=True,
                # Disable Triton: requires compute capability >= 8.6
                # Falls back to PyTorch ops (slower but compatible with sm80)
                use_triton=False,
            )
            return mdl, tok
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"skip (AWQ): {name}: {exc}")

    # Standard models (and FP8 repos that work with transformers)
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float32,
            device_map=device,
        )
        return mdl, tok
    except Exception:
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                device_map="auto",
                trust_remote_code=True,
            )
            return mdl, tok
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"skip: cannot load {name}: {exc}")


def generate_realistic_sat_prompt(
    seed: str | None = None,
    difficulty: float = 0.5,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> str:
    """Generate realistic SAT prompt using actual mining modules.

    Uses the same SAT problem generator, prompt creator, chat template,
    and system prompt as production mining for authentic test prompts.

    Args:
        seed: Deterministic seed for problem generation. Random if None.
        difficulty: Problem difficulty in [0.0, 1.0]. Defaults to 0.5.
        tokenizer: Optional tokenizer to apply chat template. If provided,
            uses the same Qwen-style chat template and system prompt as
            production mining. If None, returns raw SAT problem text.

    Returns:
        Production-quality SAT prompt string. If tokenizer provided,
        includes chat template and system prompt wrapper.
    """
    if seed is None:
        seed = hashlib.sha256(str(random.random()).encode()).hexdigest()
    problem = generate_sat_problem(seed, difficulty)
    base_prompt = create_sat_prompt(problem)

    # If no tokenizer, return raw prompt (backward compatible)
    if tokenizer is None:
        return base_prompt

    # Apply production chat template and system prompt
    from grail.mining.rollout_generator import (
        REASONING_START,
        SYSTEM_PROMPT,
    )
    from grail.shared.chat_templates import build_qwen_chat_template

    # Apply Qwen chat template (same as mining)
    try:
        original_template = getattr(tokenizer, "chat_template", None)
        tokenizer.chat_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
        messages = [{"role": "user", "content": base_prompt}]
        templated = str(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        # Restore original template
        if original_template is not None:
            tokenizer.chat_template = original_template
        return templated
    except Exception:
        # Fallback: return base prompt if template fails
        return base_prompt
