"""Tests for loop.py pipeline-related changes.

Covers: _hidden_dim bug fix, lazy client, strict_token_ids,
generate_and_eval(), assemble_rollouts(), compute_advantages().
"""

from __future__ import annotations

from grail.environments.advantages import compute_advantages
from grail.environments.backends import VLLMServerBackend
from grail.environments.episode import AgentEnvLoop
from grail.environments.rollout import GRPORollout, assemble_rollouts
from tests.fixtures.fakes import DummyModel, DummyTokenizer, FakeBackend

# ──────────────────────────────────────────────────────────────────────
# 1.1 _hidden_dim initialization bug fix
# ──────────────────────────────────────────────────────────────────────


def test_hidden_dim_set_with_gen_backend_and_model() -> None:
    """hidden_dim should be set when model is provided, even with a gen_backend."""
    tokenizer = DummyTokenizer()
    model = DummyModel()
    backend = FakeBackend(tokenizer=tokenizer)

    loop = AgentEnvLoop(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        gen_backend=backend,
    )
    # Before the fix, _hidden_dim was None when gen_backend was set
    assert loop._hidden_dim is not None
    assert loop._hidden_dim == 32  # DummyModel._Cfg.hidden_size


def test_hidden_dim_none_without_model() -> None:
    """hidden_dim should be None when no model is provided."""
    tokenizer = DummyTokenizer()
    backend = FakeBackend(tokenizer=tokenizer)

    loop = AgentEnvLoop(
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        gen_backend=backend,
    )
    assert loop._hidden_dim is None


# ──────────────────────────────────────────────────────────────────────
# 1.2 Lazy AsyncOpenAI client
# ──────────────────────────────────────────────────────────────────────


def test_vllm_backend_lazy_client() -> None:
    """VLLMServerBackend should not create AsyncOpenAI in __init__."""
    backend = VLLMServerBackend(
        base_url="http://localhost:9999",
        model_name="test-model",
        tokenizer=DummyTokenizer(),
    )
    # Client should be None until first use
    assert backend._client is None


def test_vllm_backend_strict_token_ids_default_false() -> None:
    """strict_token_ids defaults to False."""
    backend = VLLMServerBackend(
        base_url="http://localhost:9999",
        model_name="test-model",
        tokenizer=DummyTokenizer(),
    )
    assert backend._strict_token_ids is False


def test_vllm_backend_strict_token_ids_true() -> None:
    """strict_token_ids can be set to True."""
    backend = VLLMServerBackend(
        base_url="http://localhost:9999",
        model_name="test-model",
        tokenizer=DummyTokenizer(),
        strict_token_ids=True,
    )
    assert backend._strict_token_ids is True


# ──────────────────────────────────────────────────────────────────────
# 6.1 generate_and_eval()
# ──────────────────────────────────────────────────────────────────────


def test_generate_and_eval_returns_batch_data() -> None:
    """generate_and_eval should return (all_ids, prompt_len, reward, info) tuples."""
    from tests.fixtures.fakes import DummyEnv

    tokenizer = DummyTokenizer()
    model = DummyModel()
    backend = FakeBackend(tokenizer=tokenizer)

    loop = AgentEnvLoop(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        gen_backend=backend,
    )

    batch_data = loop.generate_and_eval(
        DummyEnv,
        count=2,
        batch_size=2,
    )

    assert len(batch_data) == 2
    for all_ids, prompt_len, reward, info in batch_data:
        assert isinstance(all_ids, list)
        assert isinstance(prompt_len, int)
        assert prompt_len > 0
        assert isinstance(reward, float)
        assert isinstance(info, dict)


# ──────────────────────────────────────────────────────────────────────
# 6.2 assemble_rollouts()
# ──────────────────────────────────────────────────────────────────────


def test_assemble_rollouts_basic() -> None:
    """assemble_rollouts produces correct GRPORollout objects."""
    batch_data = [
        ([10, 20, 30, 40, 50], 2, 0.8, {"success": True, "assignment": [True, False]}),
        ([10, 20, 30, 40], 2, 0.2, {"success": False}),
    ]
    proof_results = [
        ([{"h": "a"}] * 5, [0.1, 0.2, 0.3], b"sig1", {"randomness": "abc"}, "v1"),
        ([{"h": "b"}] * 4, [0.4, 0.5], b"sig2", {"randomness": "abc"}, "v1"),
    ]

    rollouts = assemble_rollouts(batch_data, proof_results)

    assert len(rollouts) == 2

    r0 = rollouts[0]
    assert isinstance(r0, GRPORollout)
    assert r0.tokens == [10, 20, 30, 40, 50]
    assert r0.prompt_length == 2
    assert r0.completion_length == 3
    assert r0.reward == 0.8
    assert r0.success is True
    assert r0.advantage == 0.0  # Not yet computed

    r1 = rollouts[1]
    assert r1.prompt_length == 2
    assert r1.completion_length == 2
    assert r1.success is False


# ──────────────────────────────────────────────────────────────────────
# compute_advantages()
# ──────────────────────────────────────────────────────────────────────


def test_compute_advantages_zero_mean() -> None:
    """Advantages should be zero-mean."""
    advantages = compute_advantages([1.0, 2.0, 3.0, 4.0])
    total = sum(advantages)
    assert abs(total) < 1e-6


def test_compute_advantages_empty() -> None:
    """Empty rewards should return empty advantages."""
    assert compute_advantages([]) == []


def test_compute_advantages_single() -> None:
    """Single reward should give advantage of 0."""
    advantages = compute_advantages([5.0])
    assert len(advantages) == 1
    assert advantages[0] == 0.0
