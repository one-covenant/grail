"""Utility for merging recent LoRA checkpoints into a base model."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

import torch
import typer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

app = typer.Typer(add_completion=False)

DEFAULT_TARGET_DIR = Path("/ephemeral")
DEFAULT_CHECKPOINT_COUNT = 10
_METADATA_FILES_TO_COPY = ("chat_template.jinja", "README.md")


def _normalize_model_source(base_model: str) -> str:
    """Expand local paths if they exist; otherwise return the original identifier."""

    candidate = Path(base_model).expanduser()
    if candidate.exists():
        return str(candidate)
    return base_model


def _resolve_precision(precision: str) -> torch.dtype:
    """Map a human readable precision string to a ``torch.dtype``."""

    precision_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "full": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = precision.lower()
    if key not in precision_map:
        supported = ", ".join(sorted(set(precision_map)))
        raise typer.BadParameter(
            f"Unsupported precision '{precision}'. Supported values: {supported}."
        )
    return precision_map[key]


def _discover_latest_checkpoints(root: Path, count: int) -> List[Path]:
    """Return the latest ``count`` LoRA checkpoint directories ordered newest first."""

    checkpoints = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("checkpoint-"):
            continue
        suffix = path.name.split("-")[-1]
        if not suffix.isdigit():
            continue
        checkpoints.append((int(suffix), path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint directories found in {root}.")
    checkpoints.sort(key=lambda item: item[0])
    latest = [entry[1] for entry in checkpoints[-count:]]
    latest.reverse()
    return latest


def _copy_metadata_files(source: Path, destination: Path) -> None:
    """Copy selected metadata files from source to destination if they exist."""

    for name in _METADATA_FILES_TO_COPY:
        file_path = source / name
        if file_path.exists() and file_path.is_file():
            shutil.copy2(file_path, destination / name)


def _load_base_model(base_model: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    """Load the base model with the requested precision."""

    kwargs: dict[str, object] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    except OSError as error:
        raise typer.BadParameter(
            f"Failed to load base model from '{base_model}'. "
            "Ensure the identifier or path is correct and accessible."
        ) from error
    return model


def _load_tokenizer(base_model: str) -> PreTrainedTokenizerBase:
    """Load tokenizer and provide a user friendly error if it fails."""

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except OSError as error:
        raise typer.BadParameter(
            f"Failed to load tokenizer from '{base_model}'. "
            "Provide a valid Hugging Face repo ID or a local path with tokenizer files."
        ) from error
    return tokenizer


def _merge_single_checkpoint(
    base_model: str,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint_dir: Path,
    output_dir: Path,
    dtype: torch.dtype,
) -> None:
    """Merge a single LoRA checkpoint into the base model and persist the result."""

    logging.info("Loading base model '%s'", base_model)
    model = _load_base_model(base_model, dtype)
    logging.info("Applying LoRA from '%s'", checkpoint_dir)
    peft_model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=False)
    merged_model = peft_model.merge_and_unload()
    if merged_model is None:
        merged_model = peft_model.base_model
    merged_model.to(device="cpu", dtype=dtype)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving merged model to '%s'", output_dir)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    _copy_metadata_files(checkpoint_dir, output_dir)
    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.command()
def merge_latest_loras(
    base_model: str = typer.Option(
        "unsloth/Qwen3-4B-Base",
        help="Base model identifier or local path to merge the LoRA adapters into.",
    ),
    loras_dir: Path = typer.Option(
        Path("/home/shadeform/grail/outputs"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing LoRA checkpoints named 'checkpoint-<step>'.",
    ),
    target_dir: Path = typer.Option(
        DEFAULT_TARGET_DIR,
        help="Directory to store merged models.",
    ),
    count: int = typer.Option(
        DEFAULT_CHECKPOINT_COUNT,
        min=1,
        help="Number of latest checkpoints to merge.",
    ),
    precision: str = typer.Option(
        "float16",
        help=(
            "Model precision to use when loading and saving. "
            "Accepts float16/fp16, float32/fp32/full, bfloat16/bf16."
        ),
    ),
) -> None:
    """Merge the most recent LoRA checkpoints into the base model."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    dtype = _resolve_precision(precision)
    normalized_model = _normalize_model_source(base_model)
    logging.info("Using base model '%s'", normalized_model)
    tokenizer = _load_tokenizer(normalized_model)
    checkpoints = _discover_latest_checkpoints(loras_dir, count)
    logging.info("Discovered %d checkpoints", len(checkpoints))
    for checkpoint_dir in checkpoints:
        merged_output_dir = target_dir / f"{checkpoint_dir.name}_merged"
        _merge_single_checkpoint(
            base_model=normalized_model,
            tokenizer=tokenizer,
            checkpoint_dir=checkpoint_dir,
            output_dir=merged_output_dir,
            dtype=dtype,
        )
    logging.info("Completed merging %d checkpoints.", len(checkpoints))


def main() -> None:
    """Entrypoint for execution without Typer auto-discovery."""

    typer.run(merge_latest_loras)


if __name__ == "__main__":  # pragma: no cover
    main()
