"""Tests for the refactored environments modules.

Covers: backends imports, GenerationParams defaults, compute_advantages,
assemble_rollouts, AgentEnvLoop with FakeBackend.
"""

from __future__ import annotations

import asyncio

from grail.environments.advantages import compute_advantages
from grail.environments.backends import (
    GenerationParams,
    HFBackend,
    SGLangServerBackend,
    TextGenBackend,
    VLLMServerBackend,
)
from grail.environments.episode import AgentEnvLoop
from grail.environments.rollout import GRPORollout, assemble_rollouts
from tests.fixtures.fakes import DummyModel, DummyTokenizer, FakeBackend

# ── GenerationParams ──────────────────────────────────────────────────


def test_generation_params_defaults():
    """Verify dataclass defaults match expected constants."""
    from grail.protocol.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    p = GenerationParams()
    assert p.max_new_tokens == MAX_NEW_TOKENS_PROTOCOL_CAP
    assert p.temperature == 0.6
    assert p.do_sample is True
    assert p.top_p == 0.95
    assert p.top_k == 20
    assert p.repetition_penalty == 1.1
    assert p.trim_right_padding is False


# ── compute_advantages ────────────────────────────────────────────────


def test_compute_advantages_zero_mean():
    """Advantages should sum to approximately zero."""
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    advs = compute_advantages(rewards)
    assert len(advs) == 5
    assert abs(sum(advs)) < 1e-6


def test_compute_advantages_uniform_rewards():
    """All-same rewards should produce all-zero advantages."""
    rewards = [3.0, 3.0, 3.0]
    advs = compute_advantages(rewards)
    assert all(abs(a) < 1e-6 for a in advs)


def test_compute_advantages_empty():
    """Empty reward list returns empty."""
    assert compute_advantages([]) == []


# ── assemble_rollouts ─────────────────────────────────────────────────


def test_assemble_rollouts_correct_fields():
    """Assembled rollouts have all expected fields populated."""
    batch_data = [
        ([10, 20, 30, 40, 50], 2, 1.5, {"success": True}),
    ]
    proof_results = [
        ([{"sketch_hash": "abc"}], [-1.0, -2.0, -3.0], b"sig", {"randomness": "ff"}, "v1"),
    ]
    rollouts = assemble_rollouts(batch_data, proof_results)
    assert len(rollouts) == 1
    r = rollouts[0]
    assert isinstance(r, GRPORollout)
    assert r.tokens == [10, 20, 30, 40, 50]
    assert r.prompt_length == 2
    assert r.completion_length == 3
    assert r.reward == 1.5
    assert r.advantage == 0.0
    assert r.success is True
    assert r.signature == b"sig"
    assert r.proof_version == "v1"


def test_assemble_rollouts_logprob_padding():
    """Prompt region should be padded with 0.0 in token_logprobs."""
    batch_data = [
        ([1, 2, 3, 4], 2, 0.5, {"success": False}),
    ]
    proof_results = [
        ([], [-0.5, -0.7], b"s", {}, "v1"),
    ]
    rollouts = assemble_rollouts(batch_data, proof_results)
    lp = rollouts[0].token_logprobs
    assert lp[:2] == [0.0, 0.0]
    assert lp[2:] == [-0.5, -0.7]


# ── Backends importable ──────────────────────────────────────────────


def test_backends_importable():
    """All 5 public symbols importable from grail.environments.backends."""
    assert GenerationParams is not None
    assert TextGenBackend is not None
    assert HFBackend is not None
    assert VLLMServerBackend is not None
    assert SGLangServerBackend is not None


# ── AgentEnvLoop with FakeBackend ────────────────────────────────────


def _make_loop() -> AgentEnvLoop:
    model = DummyModel()
    tokenizer = DummyTokenizer()
    backend = FakeBackend(tokenizer=tokenizer, completion_len=4)
    return AgentEnvLoop(model, tokenizer, device="cpu", gen_backend=backend)


def test_episode_generate_batch_shapes():
    """generate_from_prompt_ids_batch returns correct tuple shapes."""
    loop = _make_loop()
    prompt_ids = [[10, 20, 30], [40, 50]]
    results = asyncio.run(loop.generate_from_prompt_ids_batch(prompt_ids, trim_right_padding=False))
    assert len(results) == 2
    for all_ids, prompt_len, lp in results:
        assert isinstance(all_ids, list)
        assert isinstance(prompt_len, int)
        assert prompt_len > 0
        assert len(all_ids) > prompt_len
        assert lp is None  # FakeBackend returns None logprobs


def test_episode_render_prompt_ids():
    """render_prompt_ids_batch returns list[list[int]]."""
    loop = _make_loop()
    messages_list = [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "world"}],
    ]
    results = loop.render_prompt_ids_batch(messages_list)
    assert len(results) == 2
    for ids in results:
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0
