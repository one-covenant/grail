"""Environment registry and adapter interfaces for env-agnostic validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@runtime_checkable
class EnvAdapter(Protocol):
    """Adapter interface for environment-specific logic used by validation."""

    def build_prompt_ids(
        self,
        seed: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]: ...

    def evaluate_completion(
        self,
        seed: str,
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

        # Validate canonical chat template without mutation
        # (expected to be provided via checkpoint/service)
        tpl = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
        try:
            current_tpl = getattr(tokenizer, "chat_template", None)
            if current_tpl != tpl:
                logger.warning(
                    "Tokenizer chat_template mismatch with expected Qwen "
                    "template; proceeding without mutation"
                )
        except Exception:
            logger.debug("Tokenizer chat_template check failed", exc_info=True)

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
            "VALIDATOR RENDERED PROMPT: length=%d chars, tokens=%d\n%s, seed=%d",
            len(rendered),
            len(ids),
            rendered,
            seed_int,
        )

        return ids

    def evaluate_completion(
        self,
        seed: str,
        completion_text: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict:
        from .sat_env import SATEnv

        env = SATEnv()
        # Convert hex seed string to int for env.reset()
        seed_int = int(seed, 16) if isinstance(seed, str) else int(seed)
        env.reset(seed=seed_int)
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


_REGISTRY: dict[str, EnvAdapter] = {
    "sat": SATEnvAdapter(),
}


def get_adapter(env_id: str) -> EnvAdapter:
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown environment id: {env_id}")
    return _REGISTRY[env_id]
