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
