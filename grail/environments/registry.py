"""Environment registry and adapter interfaces for env-agnostic validation."""

from __future__ import annotations

import logging
import os
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
        from .factory import create_env

        env = create_env("sat")
        seed_int = seed
        obs = env.reset(seed=seed_int)
        messages = [{"role": m.role, "content": m.content} for m in obs.messages]

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(rendered, str), "Expected apply_chat_template to return string"
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
        from .core import ChatMessage
        from .factory import create_env

        env = create_env("sat")
        env.reset(seed=int(seed))

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
        from .factory import create_env

        env = create_env("gsm8k")
        obs = env.reset(seed=seed)
        messages = [{"role": m.role, "content": m.content} for m in obs.messages]

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(rendered, str), "Expected apply_chat_template to return string"
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
        from .factory import create_env

        env = create_env("gsm8k")
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


@dataclass(frozen=True)
class MATHEnvAdapter:
    """Hendrycks MATH adapter backed by MATHEnv.

    Uses HF datasets-backed math problems with multi-strategy validation.
    Supports filtering by level (1-5) and subject (7 domains).
    """

    def build_prompt_ids(
        self,
        seed: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]:
        from .factory import create_env

        env = create_env("math")
        obs = env.reset(seed=seed)
        messages = [{"role": m.role, "content": m.content} for m in obs.messages]

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(rendered, str), "Expected apply_chat_template to return string"
        toks = tokenizer(
            rendered,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids_tensor = toks.input_ids[0]
        ids: list[int] = [int(v) for v in input_ids_tensor.tolist()]

        # Debug: log rendered prompt text for comparison with miner
        logger.debug(
            ("VALIDATOR RENDERED PROMPT (MATH): length=%d chars, tokens=%d\n%s, seed=%d"),
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
        from .factory import create_env

        env = create_env("math")
        env.reset(seed=int(seed))

        _obs, reward, _terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content=completion_text)
        )
        success = bool(info.get("success", False))
        result = {"success": success, "reward": float(reward)}
        # Include metadata for diagnostics
        if "gold_answer" in info:
            result["gold_answer"] = info["gold_answer"]
        if "pred_answer" in info:
            result["pred_answer"] = info["pred_answer"]
        return result


@dataclass(frozen=True)
class PythonCodeEnvAdapter:
    """Python code generation adapter backed by PythonCodeEnv.

    Supports MBPP and HumanEval datasets for code generation tasks.
    Validates code by executing test cases in sandboxed subprocess.
    """

    dataset: str = "mbpp"  # "mbpp" or "humaneval"
    split: str = "train"  # "train", "validation", "test"

    def build_prompt_ids(
        self,
        seed: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]:
        from .factory import create_env

        env_id = "humaneval" if self.dataset == "humaneval" else "mbpp"
        env = create_env(env_id, split=self.split)
        obs = env.reset(seed=seed)
        messages = [{"role": m.role, "content": m.content} for m in obs.messages]

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(rendered, str), "Expected apply_chat_template to return string"
        toks = tokenizer(
            rendered,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids_tensor = toks.input_ids[0]
        ids: list[int] = [int(v) for v in input_ids_tensor.tolist()]

        # Debug: log rendered prompt text for comparison with miner
        logger.debug(
            ("VALIDATOR RENDERED PROMPT (PYTHON): length=%d chars, tokens=%d\n%s, seed=%d"),
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
        from .factory import create_env

        env_id = "humaneval" if self.dataset == "humaneval" else "mbpp"
        env = create_env(env_id, split=self.split)
        env.reset(seed=int(seed))

        _obs, reward, _terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content=completion_text)
        )
        success = bool(info.get("success", False))
        result = {
            "success": success,
            "reward": float(reward),
            "tests_passed": info.get("tests_passed", 0),
            "tests_total": info.get("tests_total", 0),
        }
        # Include diagnostics
        if "syntax_valid" in info:
            result["syntax_valid"] = info["syntax_valid"]
        if "has_code" in info:
            result["has_code"] = info["has_code"]
        return result


@dataclass(frozen=True)
class TritonKernelEnvAdapter:
    """Triton kernel generation adapter backed by TritonKernelEnv.

    Uses KernelBench dataset for GPU kernel optimization tasks.
    Validates generated Triton kernels for structure and optionally correctness.
    """

    split: str = "train"
    level: int | None = None

    @property
    def gpu_eval(self) -> bool:
        """Read gpu_eval from GRAIL_GPU_EVAL env var (default False)."""
        return os.environ.get("GRAIL_GPU_EVAL", "false").lower() in ("1", "true", "yes")

    def build_prompt_ids(
        self,
        seed: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]:
        from .factory import create_env

        env = create_env(
            "triton_kernel",
            split=self.split,
            env_params={"level": self.level, "gpu_eval": self.gpu_eval},
        )
        obs = env.reset(seed=seed)
        messages = [{"role": m.role, "content": m.content} for m in obs.messages]

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(rendered, str), "Expected apply_chat_template to return string"
        toks = tokenizer(
            rendered,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids_tensor = toks.input_ids[0]
        ids: list[int] = [int(v) for v in input_ids_tensor.tolist()]

        logger.debug(
            ("VALIDATOR RENDERED PROMPT (TRITON): length=%d chars, tokens=%d\n%s, seed=%d"),
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
        from .factory import create_env

        env = create_env(
            "triton_kernel",
            split=self.split,
            env_params={"level": self.level, "gpu_eval": self.gpu_eval},
        )
        env.reset(seed=int(seed))

        _obs, reward, _terminated, _truncated, info = env.step(
            ChatMessage(role="assistant", content=completion_text)
        )
        success = bool(info.get("success", False))
        result = {
            "success": success,
            "reward": float(reward),
            "reward_components": info.get("reward_components", {}),
            "has_code": info.get("has_code", False),
            "syntax_valid": info.get("syntax_valid", False),
            "structure_valid": info.get("structure_valid", False),
        }
        if info.get("exec_result") is not None:
            result["exec_result"] = info["exec_result"]
        return result


_REGISTRY: dict[str, EnvAdapter] = {
    "sat": cast(EnvAdapter, SATEnvAdapter()),
    "gsm8k": cast(EnvAdapter, GSM8KEnvAdapter()),
    "math": cast(EnvAdapter, MATHEnvAdapter()),
    "mbpp": cast(EnvAdapter, PythonCodeEnvAdapter(dataset="mbpp", split="train")),
    "python_code": cast(EnvAdapter, PythonCodeEnvAdapter(dataset="mbpp", split="train")),
    "humaneval": cast(EnvAdapter, PythonCodeEnvAdapter(dataset="humaneval", split="test")),
    "triton_kernel": cast(EnvAdapter, TritonKernelEnvAdapter()),
}


def get_adapter(env_id: str) -> EnvAdapter:
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown environment id: {env_id}")
    return _REGISTRY[env_id]
