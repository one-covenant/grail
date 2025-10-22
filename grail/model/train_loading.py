"""Env-driven training model loader for the Trainer.

Parses required environment variables, resolves checkpoints from R2, and
loads the training model, reference model, and tokenizer once at startup.

Strict behavior:
- No defaults: all required envs must be set; otherwise raise with guidance
- If a specific window is requested but missing, raise and list available

Trainer uses Qwen chat template for its tokenizer. Validators/miners must not
inject templates programmatically; they should rely on checkpoint/tokenizer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from grail.model.provider import get_model, get_tokenizer
from grail.shared.chat_templates import build_qwen_chat_template
from grail.shared.prompt_constants import REASONING_START, SYSTEM_PROMPT

ModelLoadMode = Literal["latest", "hf", "window"]


@dataclass(frozen=True)
class ModelLoadSpec:
    """Specification for loading a model for training/reference.

    Attributes:
        mode: One of "latest" | "hf" | "window"
        hf_id: Required when mode == "hf"
        window: Required when mode == "window"
    """

    mode: ModelLoadMode
    hf_id: str | None = None
    window: int | None = None


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            _guidance_missing_env(
                missing=name,
                context="trainer",
            )
        )
    return value


def parse_train_env() -> ModelLoadSpec:
    """Parse training model environment variables strictly.

    Required envs:
      - GRAIL_TRAIN_MODEL_MODE in {latest,hf,window}
      - If MODE=hf: GRAIL_TRAIN_MODEL_ID
      - If MODE=window: GRAIL_TRAIN_CHECKPOINT_WINDOW
    """
    mode = _require_env("GRAIL_TRAIN_MODEL_MODE").strip().lower()
    if mode not in {"latest", "hf", "window"}:
        raise ValueError("GRAIL_TRAIN_MODEL_MODE must be one of: latest | hf | window")

    if mode == "hf":
        hf_id = _require_env("GRAIL_TRAIN_MODEL_ID").strip()
        return ModelLoadSpec(mode="hf", hf_id=hf_id)

    if mode == "window":
        window_str = _require_env("GRAIL_TRAIN_CHECKPOINT_WINDOW").strip()
        try:
            window = int(window_str)
        except ValueError as err:
            raise ValueError("GRAIL_TRAIN_CHECKPOINT_WINDOW must be an integer") from err
        return ModelLoadSpec(mode="window", window=window)

    # latest
    return ModelLoadSpec(mode="latest")


def parse_ref_env() -> ModelLoadSpec:
    """Parse reference model environment variables strictly.

    Required envs:
      - GRAIL_REF_MODEL_MODE in {latest,hf,window}
      - If MODE=hf: GRAIL_REF_MODEL_ID
      - If MODE=window: GRAIL_REF_CHECKPOINT_WINDOW
    """
    mode = _require_env("GRAIL_REF_MODEL_MODE").strip().lower()
    if mode not in {"latest", "hf", "window"}:
        raise ValueError("GRAIL_REF_MODEL_MODE must be one of: latest | hf | window")

    if mode == "hf":
        hf_id = _require_env("GRAIL_REF_MODEL_ID").strip()
        return ModelLoadSpec(mode="hf", hf_id=hf_id)

    if mode == "window":
        window_str = _require_env("GRAIL_REF_CHECKPOINT_WINDOW").strip()
        try:
            window = int(window_str)
        except ValueError as err:
            raise ValueError("GRAIL_REF_CHECKPOINT_WINDOW must be an integer") from err
        return ModelLoadSpec(mode="window", window=window)

    # latest
    return ModelLoadSpec(mode="latest")


async def _resolve_checkpoint(spec: ModelLoadSpec, checkpoint_manager: Any) -> Path:
    """Resolve a checkpoint path from spec using CheckpointManager.

    For mode==latest: choose max available window.
    If none, raise with guidance. For mode==window: ensure the requested
    window exists remotely; else raise and include available windows.
    """
    if spec.mode == "latest":
        windows = await checkpoint_manager.list_remote_windows()
        if not windows:
            raise ValueError(
                "No remote checkpoints found. Ensure trainer has "
                "published at least one. Available windows: []"
            )
        target = max(windows)
        path: Path | None = await checkpoint_manager.get_checkpoint(target)
        if path is None:
            msg = "Failed to download latest checkpoint"
            raise ValueError(msg + f" (window {target}).")
        return path

    if spec.mode == "window":
        windows = await checkpoint_manager.list_remote_windows()
        requested = spec.window
        # Guard to satisfy type checkers
        if requested is None:
            raise ValueError("GRAIL_*_CHECKPOINT_WINDOW must be set for mode=window")
        if requested not in windows:
            raise ValueError(_guidance_missing_window(requested, windows))
        path = await checkpoint_manager.get_checkpoint(int(requested))
        if path is None:
            raise ValueError(f"Failed to download checkpoint for window {requested}.")
        return path

    raise RuntimeError("_resolve_checkpoint called for non-checkpoint mode")


async def load_training_artifacts(
    train_spec: ModelLoadSpec,
    ref_spec: ModelLoadSpec,
    checkpoint_manager: Any,
) -> tuple[Any, Any, Any]:
    """Load (train_model, ref_model, tokenizer) per provided specs.

    - Trainer tokenizer installs Qwen chat template.
    - Train model loads with eval_mode=False; Ref model with eval_mode=True.
    """
    # Build trainer tokenizer with Qwen chat template
    chat_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)

    # Resolve train source
    if train_spec.mode == "hf":
        train_model_id = train_spec.hf_id or ""
        train_model = get_model(train_model_id, eval_mode=False)
        tokenizer = get_tokenizer(train_model_id, chat_template=chat_template)
    else:
        train_ckpt = await _resolve_checkpoint(train_spec, checkpoint_manager)
        train_model = get_model(str(train_ckpt), eval_mode=False)
        tokenizer = get_tokenizer(str(train_ckpt), chat_template=chat_template)

    # Resolve ref source (independent)
    if ref_spec.mode == "hf":
        ref_model_id = ref_spec.hf_id or ""
        ref_model = get_model(ref_model_id, eval_mode=True)
    else:
        ref_ckpt = await _resolve_checkpoint(ref_spec, checkpoint_manager)
        ref_model = get_model(str(ref_ckpt), eval_mode=True)

    return train_model, ref_model, tokenizer


def _guidance_missing_env(*, missing: str, context: str) -> str:
    # Compact guidance with examples
    return (
        f"Missing required environment variable: {missing}.\n"
        "Specify training/ref model modes explicitly. Examples:\n\n"
        "# Train latest, Ref HF\n"
        "export GRAIL_TRAIN_MODEL_MODE=latest\n"
        "export GRAIL_REF_MODEL_MODE=hf\n"
        "export GRAIL_REF_MODEL_ID=Qwen/Qwen2.5-7B\n\n"
        "# Train HF, Ref window\n"
        "export GRAIL_TRAIN_MODEL_MODE=hf\n"
        "export GRAIL_TRAIN_MODEL_ID=Qwen/Qwen2.5-7B\n"
        "export GRAIL_REF_MODEL_MODE=window\n"
        "export GRAIL_REF_CHECKPOINT_WINDOW=72000\n"
    )


def _guidance_missing_window(
    requested: int | None,
    available: list[int],
) -> str:
    head = "Requested checkpoint window not found."
    avail = ", ".join(str(w) for w in available[-12:]) if available else ""
    line1 = f"{head} requested={requested}."
    line2 = "Available: [" + avail + "]"
    if len(line2) > 70:
        line2 = "Available: [...]"
    line3 = "Ensure the correct window is specified or publish the checkpoint."
    return f"{line1}\n{line2}\n{line3}"
