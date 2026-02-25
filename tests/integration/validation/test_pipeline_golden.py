"""Option A: Golden pipeline validation (fast, deterministic).

Runs the env-agnostic validation pipeline over a crafted commit dict using a
lightweight fake EnvAdapter and a stubbed proof validator.

Exercised checks: schema, tokens, proof (stubbed), env prompt/eval, reward,
logprobs, termination — without storage/chain/model forward passes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from grail.shared.constants import CURRENT_ENV_ID
from grail.validation import create_env_validation_pipeline
from grail.validation.context import ValidationContext


class _FakeAdapter:
    def __init__(self, prompt_ids: list[int], env_reward: float = 1.0) -> None:
        self._prompt_ids = prompt_ids
        self._reward = float(env_reward)

    def build_prompt_ids(
        self, seed: str, tokenizer: Any, env_params: dict | None = None
    ) -> list[int]:
        return list(self._prompt_ids)

    def evaluate_completion(
        self,
        seed: str,
        completion_text: str,
        tokenizer: Any,
        env_params: dict | None = None,
    ) -> dict:
        return {"success": True, "reward": self._reward}


def _stub_proof_validator(
    monkeypatch: pytest.MonkeyPatch,
    logits: torch.Tensor,
) -> None:
    from grail.validation.validators import proof as proof_mod

    def _ok(_self: Any, ctx: Any) -> bool:
        ctx.cached_logits = logits
        ctx.checks["proof_valid"] = True
        return True

    monkeypatch.setattr(
        proof_mod.GRAILProofValidator,
        "validate",
        _ok,
        raising=True,
    )


def _stub_registry(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    # Stub both the module and the validator import
    from grail.environments import registry as reg
    from grail.validation.validators import environment as env_mod

    def _get_adapter(_env_id: str) -> Any:
        return adapter

    monkeypatch.setattr(reg, "get_adapter", _get_adapter, raising=True)
    monkeypatch.setattr(env_mod, "get_adapter", _get_adapter, raising=True)


@pytest.mark.integration
def test_pipeline_golden_valid_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Canonical prompt/completion (min 10 tokens for schema)
    prompt_ids = [10, 11, 12, 13, 14, 15, 16]
    completion_ids = [17, 18, 19]
    tokens = prompt_ids + completion_ids
    prompt_len = len(prompt_ids)
    completion_len = len(completion_ids)

    # Fake adapter + registry
    adapter = _FakeAdapter(prompt_ids, env_reward=0.8)
    _stub_registry(monkeypatch, adapter)

    # Stub proof validator to set cached logits matching claimed logprobs
    vocab = 64
    logits = torch.full((len(tokens), vocab), -20.0, dtype=torch.float32)
    # Make chosen token likely at each step (logit 0 vs -20 elsewhere)
    for i, tok in enumerate(completion_ids):
        row = (prompt_len + i - 1) if (prompt_len + i - 1) >= 0 else 0
        logits[row, tok] = 0.0
    # Ensure EOS probability high on second-to-last step for termination check
    eos_id = completion_ids[-1]
    logits[-2, eos_id] = 5.0
    _stub_proof_validator(monkeypatch, logits)

    # Minimal tokenizer/model stubs
    class _Tok:
        eos_token_id = eos_id
        chat_template = ""

        def decode(
            self,
            ids: list[int],
            skip_special_tokens: bool = False,
        ) -> str:
            return " ".join(str(i) for i in ids)

        def apply_chat_template(
            self,
            messages: Any,
            tokenize: bool = False,
            add_generation_prompt: bool = True,
        ) -> str:
            return ""

        def __call__(self, text: str, **kwargs: Any) -> Any:
            # Return mock tensor-like object
            from types import SimpleNamespace

            return SimpleNamespace(input_ids=[prompt_ids])

    tokenizer = _Tok()

    model = SimpleNamespace(
        name_or_path="test-model",
        device=torch.device("cpu"),
        config=SimpleNamespace(vocab_size=vocab),
    )

    # Craft commit per schema
    commit = {
        "tokens": tokens,
        "commitments": [{} for _ in tokens],
        "proof_version": "v1",
        "model": {"name": model.name_or_path, "layer_index": 0},
        "signature": "deadbeef",
        "beacon": {"randomness": "abcdef01"},
        "rollout": {
            "prompt_length": prompt_len,
            "completion_length": completion_len,
            "success": True,
            "total_reward": 0.8,
            "advantage": 0.0,
            "assignment": [],
            "trajectory": [],
            "token_logprobs": [0.0] * prompt_len + [0.0 for _ in completion_ids],
            "satisfied_clauses": 0,
        },
    }

    # Inject env field like validator does (validator-derived, not miner-sent)
    from grail.protocol.signatures import derive_env_seed

    seed_int = derive_env_seed("miner_hotkey", "0xfakewindowhash", 0)
    commit["env"] = {"id": CURRENT_ENV_ID, "seed": seed_int}

    # Run pipeline
    pipeline = create_env_validation_pipeline()
    ctx = ValidationContext(
        commit=commit,
        prover_address="miner_hotkey",
        challenge_randomness="00",
        window_hash="0xfakewindowhash",
        group_index=0,
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,  # type: ignore[arg-type]
        device=model.device,
        miner_uid="1",
    )

    is_valid, checks = pipeline.validate(ctx)

    assert is_valid is True
    expected = [
        "schema_valid",
        "tokens_valid",
        "proof_valid",
        "env_prompt_valid",
        "termination_valid",
        "env_eval_valid",
        "reward_valid",
        "logprobs_valid",
    ]
    for key in expected:
        assert checks.get(key) is True
