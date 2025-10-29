"""Typed training configuration, sourced from shared constants.

Keeps a single source of truth while enabling injection and testing.
"""

from __future__ import annotations

from dataclasses import dataclass

from grail.shared import constants


@dataclass
class TrainingConfig:
    """Hyperparameters and settings for training.

    Values default from `grail.shared.constants` to avoid duplication.
    """

    lr: float = constants.TRAINER_LR
    epochs: int = constants.TRAINER_EPOCHS
    batch_size: int = constants.TRAINER_BATCH_SIZE
    max_length: int = constants.TRAINER_MAX_LENGTH
    grad_clip: float = constants.TRAINER_GRAD_CLIP
    warmup_steps: int = constants.TRAINER_WARMUP_STEPS
    kl_coef: float = constants.TRAINER_KL_COEF
    entropy_coef: float = constants.TRAINER_ENTROPY_COEF
    group_adv_sum_tolerance: float = constants.TRAINER_GROUP_ADV_SUM_TOL
    min_aggregate_weight: float = constants.TRAINER_MIN_AGGREGATE_WEIGHT
    min_trusted_miners: int = constants.TRAINER_MIN_TRUSTED_MINERS


@dataclass
class EvalConfig:
    """Configuration for periodic evaluation cycles.

    Defaults chosen to be safe and reasonably fast for initial integration.
    """

    enabled: bool = True
    window_interval: int = 16
    split: str = "test"  # dataset-backed envs (e.g., GSM8K) #TODO: should be specified per env
    subset_size: int | None = 400  # generative envs or capped dataset eval
    seed_base: int = 2025
    batch_size: int = (
        32  # Increased for parallel server-based backends (8 tasks × 5 reps = 80 prompts/batch)
    )
    replicates: int = 5  # for pass@k / mean@k curves
    # Decoding configuration for evaluation (separate from training)
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    # Backend control: "hf" | "vllm" | "sglang"
    backend: str = "hf"
    # sgLang server options (used when backend == "sglang")
    sglang_host: str = "127.0.0.1"
    sglang_port: int = 30000
    sglang_start_server: bool = False  # Disabled: using offline async engine instead
    sglang_server_timeout_s: float = 120.0
    sglang_trust_remote_code: bool = False
    use_num_return_sequences: bool = False  # HF-only optimization
    # Metrics aggregation: which k to report (subset of 1..replicates)
    report_ks: tuple[int, ...] = (1, 5, 10)
    # Optional: path to store JSONL predictions (None disables)
    store_predictions_path: str | None = None
