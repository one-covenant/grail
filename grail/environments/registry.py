"""Environment registry and adapter interfaces for env-agnostic validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@runtime_checkable
class EnvAdapter(Protocol):
    """Adapter interface for environment-specific logic used by validation."""

    def build_prompt_ids(
        self,
        seed: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]: ...

    def evaluate_completion(
        self,
        seed: int,
        completion_text: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict:
        """Return at least {"success": bool, "reward": float}.

        Additional keys may be included for metrics (e.g., assignment).
        """
        ...


@dataclass(frozen=True)
class SATEnvAdapter:
    """SAT adapter backed by SATEnv.

    Difficulty derived internally from seed.
    """

    def build_prompt_ids(
        self,
        seed: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]:
        # Lazy imports to avoid heavy deps in import graph
        from ..shared.chat_templates import build_qwen_chat_template
        from ..shared.prompt_constants import REASONING_START, SYSTEM_PROMPT
        from .sat_env import SATEnv

        env = SATEnv()
        # Convert hex seed string to int for env.reset()
        seed_int = seed
        obs = env.reset(seed=seed_int)
        messages = [{"role": m.role, "content": m.content} for m in obs.messages]

        # Validate canonical chat template (should be set by ValidationService)
        tpl = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
        try:
            current_tpl = getattr(tokenizer, "chat_template", None)
            if current_tpl != tpl:
                msg = "Tokenizer chat_template differs from expected Qwen template"
                logger.debug(msg)
        except Exception:
            logger.debug("Tokenizer chat_template check failed", exc_info=True)

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        toks = tokenizer(
            rendered,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids_tensor = toks.input_ids[0]
        ids: list[int] = [int(v) for v in input_ids_tensor.tolist()]

        # Debug: log rendered prompt text for comparison with miner
        logger.debug(
            ("VALIDATOR RENDERED PROMPT: length=%d chars, tokens=%d\n%s, seed=%d"),
            len(rendered),
            len(ids),
            rendered,
            seed_int,
        )

        return ids

    def evaluate_completion(
        self,
        seed: int,
        completion_text: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict:
        from .sat_env import SATEnv

        env = SATEnv()
        env.reset(seed=int(seed))
        # Single turn step
        from .core import ChatMessage

        _, reward, _terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content=completion_text)
        )
        success = bool(info.get("success", False))
        result = {"success": success, "reward": float(reward)}
        if "assignment" in info:
            result["assignment"] = info["assignment"]
        return result


@dataclass(frozen=True)
class GSM8KEnvAdapter:
    """GSM8K adapter backed by GSM8KEnv.

    Uses HF datasets-backed question/answer tasks; reward is exact match.
    """

    def build_prompt_ids(
        self,
        seed: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]:
        # Lazy imports to avoid heavy deps in import graph
        from ..shared.chat_templates import build_qwen_chat_template
        from ..shared.prompt_constants import REASONING_START, SYSTEM_PROMPT
        from .gsm8k_env import GSM8KEnv

        env = GSM8KEnv()
        obs = env.reset(seed=seed)
        messages = [{"role": m.role, "content": m.content} for m in obs.messages]

        # Validate canonical chat template (should be set by ValidationService)
        tpl = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
        try:
            current_tpl = getattr(tokenizer, "chat_template", None)
            if current_tpl != tpl:
                msg = "Tokenizer chat_template differs from expected Qwen template"
                logger.debug(msg)
        except Exception:
            logger.debug("Tokenizer chat_template check failed", exc_info=True)

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        toks = tokenizer(
            rendered,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids_tensor = toks.input_ids[0]
        ids: list[int] = [int(v) for v in input_ids_tensor.tolist()]

        # Debug: log rendered prompt text for comparison with miner
        logger.debug(
            ("VALIDATOR RENDERED PROMPT (GSM8K): length=%d chars, tokens=%d\n%s, seed=%d"),
            len(rendered),
            len(ids),
            rendered,
            int(seed),
        )

        return ids

    def evaluate_completion(
        self,
        seed: int,
        completion_text: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict:
        from .core import ChatMessage
        from .gsm8k_env import GSM8KEnv

        env = GSM8KEnv()
        env.reset(seed=int(seed))

        _obs, reward, _terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content=completion_text)
        )
        success = bool(info.get("success", False))
        result = {"success": success, "reward": float(reward)}
        # Include normalized answers for diagnostics if present
        if "gold_answer" in info:
            result["gold_answer"] = info["gold_answer"]
        if "pred_answer" in info:
            result["pred_answer"] = info["pred_answer"]
        return result


_REGISTRY: dict[str, EnvAdapter] = {
    "sat": cast(EnvAdapter, SATEnvAdapter()),
    "gsm8k": cast(EnvAdapter, GSM8KEnvAdapter()),
}


def get_adapter(env_id: str) -> EnvAdapter:
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown environment id: {env_id}")
    return _REGISTRY[env_id]
