"""Typed training configuration, sourced from shared constants.

Keeps a single source of truth while enabling injection and testing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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
    # GRPO two-stage filtering controls
    grpo_max_groups: int = constants.GRPO_MAX_GROUPS
    grpo_max_completion_tokens: int = constants.GRPO_MAX_COMPLETION_TOKENS
    grpo_min_success_fraction: float = constants.GRPO_MIN_SUCCESS_FRACTION
    grpo_min_reward_per_token: float = constants.GRPO_MIN_REWARD_PER_TOKEN
    grpo_reward_per_token_drop_quantile: float = constants.GRPO_REWARD_PER_TOKEN_DROP_QUANTILE


@dataclass
class EvalConfig:
    """Configuration for periodic evaluation cycles.

    Defaults chosen to be safe and reasonably fast for initial integration.
    """

    enabled: bool = True
    window_interval: int = 16
    split: str = "test"  # dataset-backed envs (e.g., GSM8K) #TODO: should be specified per env
    subset_size: int | None = None  # generative envs or capped dataset eval
    seed_base: int = 2025
    batch_size: int = 32  # Conservative for vLLM server: 8 tasks × 5 reps = 40 prompts/batch (prevent queue timeout)
    replicates: int = 5  # for pass@k / mean@k curves
    # Decoding configuration for evaluation (separate from training)
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True
    # Backend control: "hf" | "vllm" | "sglang"
    backend: str = "vllm"  # Server mode with async API avoids Gloo socket issues
    # sgLang server options (used when backend == "sglang")
    sglang_host: str = "127.0.0.1"
    sglang_port: int = 30000
    sglang_start_server: bool = True  # Server runs in subprocess (avoids Gloo socket issues)
    sglang_server_timeout_s: float = 120.0
    sglang_trust_remote_code: bool = False
    # vLLM server options (used when backend == "vllm")
    # Path to isolated vLLM environment Python executable
    # Override via GRAIL_VLLM_PYTHON env var for custom deployments
    vllm_python_executable: str = field(
        default_factory=lambda: os.getenv("GRAIL_VLLM_PYTHON", "tools/vllm-server/.venv/bin/python")
    )
    # vLLM module entrypoint (version-specific, may change across vLLM releases)
    vllm_module_entrypoint: str = "vllm.entrypoints.openai.api_server"
    # vLLM server memory and concurrency tuning (optimized for single A100)
    # Best practices per vLLM docs: leave headroom for CUDA graph capture and KV cache
    # See: https://docs.vllm.ai/en/v0.10.2/configuration/optimization.html
    # - Lower gpu_memory_utilization (0.7–0.8) to leave room for graph allocation
    # - Set max_num_seqs low enough to fit in available KV cache (target ~24 for safety)
    # - Client concurrency at 50–70% of server max_num_seqs (avoid burst deadlock)
    vllm_gpu_memory_utilization: float = 0.80  # Conservative for graph capture safety
    vllm_max_model_len: int = (
        1536  # Sufficient: ~512 token prompt + 512 token completion (MAX_NEW_TOKENS)
    )
    vllm_max_num_seqs: int = 128  # Optimized for H200 with 141GB mem
    vllm_max_concurrent_requests: int = 96  # 75% of max_num_seqs for stability
    # SGLang server memory and concurrency tuning
    sglang_mem_fraction_static: float = 0.75  # Fraction of GPU memory for SGLang
    sglang_context_length: int = 1024  # Maximum sequence length
    sglang_max_running_requests: int = 4  # Server-side: max concurrent requests
    sglang_max_concurrent_requests: int = 4  # Client-side: max parallel HTTP requests
    use_num_return_sequences: bool = False  # HF-only optimization
    # Metrics aggregation: which k to report (subset of 1..replicates)
    report_ks: tuple[int, ...] = (1, 5, 10)
    # Optional: path to store JSONL predictions (None disables)
    store_predictions_path: str | None = None
    # Logging controls
    # - Disable noisy external server stdout (vLLM/SGLang) by default
    # - Optionally log a few sample completions per batch for visibility
    stream_server_logs: bool = False
    log_completions_n: int = 2
    log_completions_max_chars: int = 2048
