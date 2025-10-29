from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from grail.environments.core import ChatMessage, MultiTurnEnv, Observation
from grail.environments.loop import GenerationParams, TextGenBackend


class DummyTokenizer:
    """Minimal tokenizer implementing the subset used by AgentEnvLoop and evaluator.

    - apply_chat_template: concatenates role and content with separators
    - __call__: tokenizes characters into integer ids using a simple mapping
    - decode: converts token ids back to a simple string representation
    """

    pad_token_id: int = 0
    eos_token_id: int = 1

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts: list[str] = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        return_attention_mask: bool = False,
    ) -> Any:
        # Map each character to a token id: 2 + (ord(c) % 200)
        ids: list[int] = [2 + (ord(c) % 200) for c in text]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return type("TokOut", (), {"input_ids": input_ids})()

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        # Create a printable string from token ids, ignoring pad if requested
        out_chars: list[str] = []
        for t in token_ids:
            if skip_special_tokens and t in (self.pad_token_id, self.eos_token_id):
                continue
            # Map to lowercase letters cyclically for readability
            base = int(t) % 26
            out_chars.append(chr(ord("a") + base))
        return "".join(out_chars) or "x"


class DummyModel:
    """Minimal model exposing a config shape for resolve_hidden_size()."""

    class _Cfg:
        hidden_size: int = 32

    def __init__(self) -> None:
        self.config = DummyModel._Cfg()

    # No generate() required because tests use FakeBackend through AgentEnvLoop


@dataclass
class FakeBackend(TextGenBackend):
    """Simple deterministic backend for tests.

    - When params.do_sample is True: completion depends on provided seed
    - When params.do_sample is False: completion depends only on prompt ids
    - Trims right padding when params.trim_right_padding is True
    """

    tokenizer: DummyTokenizer
    completion_len: int = 4

    def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[list[int]]:
        results: list[list[int]] = []
        pad_id = self.tokenizer.pad_token_id
        for idx, p_ids in enumerate(prompt_ids_batch):
            if params.do_sample and seeds is not None and idx < len(seeds):
                rnd = random.Random(int(seeds[idx]))
                comp: list[int] = [2 + rnd.randint(0, 200) for _ in range(self.completion_len)]
            else:
                # Deterministic function of prompt ids
                base = sum(p_ids) % 127
                comp = [2 + ((base + i * 3) % 200) for i in range(self.completion_len)]

            seq = list(p_ids) + comp
            if not params.trim_right_padding:
                # Append a couple of pad tokens to exercise trimming behavior
                seq = seq + [pad_id, pad_id]
            results.append(seq)
        return results


class DummyEnv(MultiTurnEnv):
    """Single-turn dummy environment.

    - reset(task_id) returns a single user message mentioning the id
    - step(content) returns reward as text length; success if reward > 0
    """

    def __init__(self) -> None:
        self._last_id: str | None = None

    def reset(self, *, task_id: str | None = None, seed: int | None = None) -> Observation:
        self._last_id = str(task_id) if task_id is not None else "0"
        msg = ChatMessage(role="user", content=f"solve {self._last_id}")
        obs = Observation(messages=[msg], available_tools=[], turn_index=0, task_meta={})
        return obs

    def step(self, action: ChatMessage) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        reward = float(len(action.content))
        info = {"success": reward > 0, "reward_components": {"len": reward}}
        # Terminal after one step
        obs = Observation(messages=[], available_tools=[], turn_index=1, task_meta={})
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, info
