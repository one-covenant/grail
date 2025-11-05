from __future__ import annotations

import pytest

from grail.environments.loop import AgentEnvLoop
from tests.fixtures.fakes import DummyModel, DummyTokenizer, FakeBackend


def make_loop(*, do_sample: bool) -> AgentEnvLoop:
    tokenizer = DummyTokenizer()
    model = DummyModel()
    backend = FakeBackend(tokenizer=tokenizer)
    loop = AgentEnvLoop(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        do_sample=do_sample,
        gen_backend=backend,
    )
    return loop


def test_render_prompt_ids_batch_shapes_and_order() -> None:
    loop = make_loop(do_sample=True)
    messages_list: list[list[dict[str, str]]] = [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "bye"}],
    ]
    out = loop.render_prompt_ids_batch(messages_list)

    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(x, list) and len(x) > 0 for x in out)


@pytest.mark.asyncio
async def test_generate_from_prompt_ids_batch_batching_and_lengths() -> None:
    loop = make_loop(do_sample=False)
    messages_list: list[list[dict[str, str]]] = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]
    prompt_ids_batch = loop.render_prompt_ids_batch(messages_list)

    results = await loop.generate_from_prompt_ids_batch(
        prompt_ids_batch, seeds=None, trim_right_padding=True
    )

    assert len(results) == len(prompt_ids_batch)
    for (seq, prompt_len), p in zip(results, prompt_ids_batch, strict=False):
        assert len(seq) > prompt_len
        assert prompt_len == len(p)
        # Should not end with pad when trim_right_padding=True
        assert seq[-1] != loop.tokenizer.pad_token_id


@pytest.mark.asyncio
async def test_generate_seed_determinism_do_sample_true_and_false() -> None:
    # do_sample=True: same seeds -> identical; different seeds -> different
    loop_sample = make_loop(do_sample=True)
    messages = [[{"role": "user", "content": "seedtest"}]]
    prompt_ids_batch = loop_sample.render_prompt_ids_batch(messages)

    out_a = await loop_sample.generate_from_prompt_ids_batch(
        prompt_ids_batch, seeds=[123], trim_right_padding=True
    )
    out_b = await loop_sample.generate_from_prompt_ids_batch(
        prompt_ids_batch, seeds=[123], trim_right_padding=True
    )
    out_c = await loop_sample.generate_from_prompt_ids_batch(
        prompt_ids_batch, seeds=[124], trim_right_padding=True
    )

    assert out_a == out_b
    assert out_a != out_c

    # do_sample=False: seeds shouldn't matter
    loop_nosample = make_loop(do_sample=False)
    prompt_ids_batch2 = loop_nosample.render_prompt_ids_batch(messages)

    out_d = await loop_nosample.generate_from_prompt_ids_batch(
        prompt_ids_batch2, seeds=[555], trim_right_padding=True
    )
    out_e = await loop_nosample.generate_from_prompt_ids_batch(
        prompt_ids_batch2, seeds=[556], trim_right_padding=True
    )

    assert out_d == out_e
