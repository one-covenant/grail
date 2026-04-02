"""Tests for loop.py pipeline-related changes.

Covers: _hidden_dim bug fix, lazy client, strict_token_ids,
generate_and_eval().
"""

from __future__ import annotations

from grail.environments.backends import VLLMServerBackend
from grail.environments.episode import AgentEnvLoop
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
