"""Example experiment configurations for distributed training.

This module contains example configurations for running hyperparameter
sweeps and ablation studies on Lium infrastructure.

GPU Allocation Strategy (8 GPUs per pod):
- Each experiment uses 1 GPU for VLLM + 1 GPU for training = 2 GPUs total
- Can run 4 parallel experiments per 8-GPU pod
- GPU assignment: Exp0: 0+1, Exp1: 2+3, Exp2: 4+5, Exp3: 6+7
"""

from experiment_runner import ExperimentConfig
from lium_manager import PodSpec


def assign_gpus_for_parallel(experiments: list[ExperimentConfig], base_port: int = 8000):
    """Assign GPUs and ports for parallel execution on 8-GPU pod.

    Args:
        experiments: List of experiment configs (max 4 for 8-GPU pod)
        base_port: Base port for VLLM servers (incremented for each experiment)

    Returns:
        Updated experiments with GPU assignments
    """
    if len(experiments) > 4:
        raise ValueError(f"Too many experiments ({len(experiments)}) for 8-GPU pod (max 4)")

    for i, exp in enumerate(experiments):
        # GPU pairs: 0+1, 2+3, 4+5, 6+7
        exp.vllm_gpu = str(i * 2)
        exp.train_gpu = str(i * 2 + 1)
        exp.vllm_port = base_port + i

    return experiments


# ============================================================================
# Example 1: Learning Rate Sweep on MATH (4 experiments on 1 pod)
# ============================================================================
def learning_rate_sweep_gsm8k() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Learning rate sweep for MATH dataset.

    Tests different learning rates: 1e-6, 3e-6, 5e-6, 1e-5
    All run in parallel on a single 8-GPU pod.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    # Define pod
    pods = [
        PodSpec(
            name="lr-sweep",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=6,
        ),
    ]

    # Define experiments
    learning_rates = [1e-6, 3e-6, 5e-6, 1e-5]
    experiments = []

    for lr in learning_rates:
        exp = ExperimentConfig(
            name=f"math_lr_{lr:.0e}",
            dataset="math",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            learning_rate=lr,
            batch_size=4,
            grad_accum_steps=128,
            total_steps=100,
            eval_every=20,
        )
        experiments.append(exp)

    # Assign GPUs for parallel execution
    experiments = assign_gpus_for_parallel(experiments)

    pod_experiments = {
        "lr-sweep": experiments,
    }

    return pods, pod_experiments


# ============================================================================
# Example 2: Multi-Dataset Comparison (3 pods, 1 experiment each)
# ============================================================================
def multi_dataset_comparison() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Compare performance across GSM8K, MATH, and MBPP datasets.

    Each dataset gets its own pod for isolation.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    # Define pods (one per dataset)
    pods = [
        PodSpec(
            name="dataset-gsm8k",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=8,
        ),
        PodSpec(
            name="dataset-math",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=8,
        ),
        PodSpec(
            name="dataset-mbpp",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=8,
        ),
    ]

    # Base config
    base_config = {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "learning_rate": 3e-6,
        "batch_size": 4,
        "grad_accum_steps": 128,
        "total_steps": 100,
        "eval_every": 20,
        "vllm_gpu": "0",  # First experiment on pod uses GPUs 0+1
        "train_gpu": "1",
        "vllm_port": 8000,
    }

    # Create experiments for each dataset
    pod_experiments = {
        "dataset-gsm8k": [
            ExperimentConfig(
                name="gsm8k_baseline",
                dataset="gsm8k",
                **base_config,
            )
        ],
        "dataset-math": [
            ExperimentConfig(
                name="math_baseline",
                dataset="math",
                **base_config,
            )
        ],
        "dataset-mbpp": [
            ExperimentConfig(
                name="mbpp_baseline",
                dataset="mbpp",
                **base_config,
            )
        ],
    }

    return pods, pod_experiments


# ============================================================================
# Example 3: Batch Size Grid Search (4 experiments on 1 pod)
# ============================================================================
def batch_size_grid_search() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Grid search over batch size and gradient accumulation steps.

    Keeps effective batch size constant at 512.
    All run in parallel on a single 8-GPU pod.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    # Define pod
    pods = [
        PodSpec(
            name="batch-grid",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=8,
        ),
    ]

    # Grid: (batch_size, grad_accum_steps) pairs that give effective_batch=512
    grid = [
        (2, 256),   # 2 * 256 = 512
        (4, 128),   # 4 * 128 = 512
        (8, 64),    # 8 * 64 = 512
        (16, 32),   # 16 * 32 = 512
    ]

    experiments = []
    for bs, gas in grid:
        exp = ExperimentConfig(
            name=f"math_bs{bs}_gas{gas}",
            dataset="math",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            learning_rate=3e-6,
            batch_size=bs,
            grad_accum_steps=gas,
            total_steps=100,
            eval_every=20,
        )
        experiments.append(exp)

    # Assign GPUs for parallel execution
    experiments = assign_gpus_for_parallel(experiments)

    pod_experiments = {
        "batch-grid": experiments,
    }

    return pods, pod_experiments


# ============================================================================
# Example 4: Large-Scale Hyperparameter Sweep (8 experiments on 2 pods)
# ============================================================================
def large_hyperparam_sweep() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Large hyperparameter sweep across learning rates and batch sizes.

    8 total experiments split across 2 pods (4 experiments per pod running in parallel).

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    # Define pods
    pods = [
        PodSpec(
            name="hyperparam-sweep-0",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=10,
        ),
        PodSpec(
            name="hyperparam-sweep-1",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=10,
        ),
    ]

    # Grid search
    learning_rates = [1e-6, 3e-6]
    batch_sizes = [(4, 128), (8, 64)]  # (bs, gas) keeping effective_batch=512

    all_experiments = []
    for lr in learning_rates:
        for bs, gas in batch_sizes:
            exp = ExperimentConfig(
                name=f"math_lr{lr:.0e}_bs{bs}",
                dataset="math",
                model_id="Qwen/Qwen2.5-1.5B-Instruct",
                learning_rate=lr,
                batch_size=bs,
                grad_accum_steps=gas,
                total_steps=100,
                eval_every=20,
            )
            all_experiments.append(exp)

    # Split experiments across pods (4 per pod)
    experiments_pod0 = assign_gpus_for_parallel(all_experiments[:4])
    experiments_pod1 = assign_gpus_for_parallel(all_experiments[4:])

    pod_experiments = {
        "hyperparam-sweep-0": experiments_pod0,
        "hyperparam-sweep-1": experiments_pod1,
    }

    return pods, pod_experiments


# ============================================================================
# Example 5: Custom Advanced Configuration
# ============================================================================
def custom_advanced_sweep() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Advanced example with custom environment variables and arguments.

    Demonstrates how to pass custom environment variables and arguments
    for more complex experiments.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(
            name="custom-exp",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=10,
        ),
    ]

    experiments = [
        ExperimentConfig(
            name="math_custom_warmup",
            dataset="math",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            learning_rate=3e-6,
            batch_size=4,
            grad_accum_steps=128,
            total_steps=100,
            eval_every=20,
            vllm_gpu="0",
            train_gpu="1",
            vllm_port=8000,
            # Custom environment variables (will be exported in the script)
            custom_env={
                "WANDB_PROJECT": "grail-hyperparam-sweep",
                "WANDB_TAGS": "custom,warmup-ablation",
                "GRAIL_TRAINER_WARMUP_STEPS": "50",  # Override warmup steps
            },
        ),
    ]

    pod_experiments = {
        "custom-exp": experiments,
    }

    return pods, pod_experiments


# ============================================================================
# Example 6: Single Test Run (Qwen2.5-0.5B)
# ============================================================================
def test_qwen_0_5b() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Single test run with Qwen2.5-0.5B-Instruct on 1 pod.

    This is a minimal configuration for testing the infrastructure
    before scaling to full experiments.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(
            name="test-qwen-0.5b",
            gpu_type="A100",
            gpu_count=8,
            # No bandwidth limits
            ttl_hours=24,
        ),
    ]

    # This will be run via nohup script, which handles 4 parallel seeds
    # We don't use ExperimentConfig here since nohup script manages seeds
    pod_experiments = {
        "test-qwen-0.5b": [],  # Empty - handled by deploy_parallel.py
    }

    return pods, pod_experiments


# ============================================================================
# Example 7: Multi-Model GRPO Sweep (8 pods)
# ============================================================================
def multi_model_grpo_sweep() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Multi-model GRPO experiments with num_iterations ablation.

    Runs 9 model configurations across 9 pods (each 8xA100):
    - Qwen2.5-Instruct: 0.5B, 1.5B, 7B (iter=1), 7B (iter=8), 7B (iter=16)
    - Llama-3.2-Instruct: 1B, 3B
    - Gemma-3-it: 1B, 4B

    Each pod runs 4 seeds in parallel via run_parallel_training_nohup.sh.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    # Define all model configurations
    model_configs = [
        {"name": "qwen2.5-0.5b-iter1", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_iterations": 1},
        {"name": "qwen2.5-1.5b-iter1", "model": "Qwen/Qwen2.5-1.5B-Instruct", "num_iterations": 1},
        {"name": "qwen2.5-7b-iter1", "model": "Qwen/Qwen2.5-7B-Instruct", "num_iterations": 1},
        {"name": "qwen2.5-7b-iter8", "model": "Qwen/Qwen2.5-7B-Instruct", "num_iterations": 8},
        {"name": "qwen2.5-7b-iter16", "model": "Qwen/Qwen2.5-7B-Instruct", "num_iterations": 16},
        {"name": "llama3.2-1b-iter1", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_iterations": 1},
        {"name": "llama3.2-3b-iter1", "model": "meta-llama/Llama-3.2-3B-Instruct", "num_iterations": 1},
        {"name": "gemma3-1b-iter1", "model": "google/gemma-3-1b-it", "num_iterations": 1},
        {"name": "gemma3-4b-iter1", "model": "google/gemma-3-4b-it", "num_iterations": 1},
    ]

    # Create one pod per model configuration
    pods = [
        PodSpec(
            name=config["name"],
            gpu_type="A100",
            gpu_count=8,
            # No bandwidth limits (as requested)
            ttl_hours=24,  # 24 hours should be plenty for any model
        )
        for config in model_configs
    ]

    # Empty experiments list - actual execution handled by deploy_parallel.py
    # which uses nohup_experiment_runner.py with the model configs
    pod_experiments = {
        config["name"]: [] for config in model_configs
    }

    return pods, pod_experiments


# ============================================================================
# Example 8: Active Models GRPO Sweep (6 pods - matches MODEL_CONFIGS)
# ============================================================================
def active_models_grpo_sweep() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Active models GRPO experiments (matches deploy_parallel.py MODEL_CONFIGS).

    Runs 6 model configurations across 6 pods (each 8xA100):
    - Qwen2.5-Instruct: 0.5B, 1.5B
    - Llama-3.2-Instruct: 1B, 3B
    - Gemma-3-it: 1B, 4B

    Each pod runs 4 seeds in parallel via run_parallel_training_nohup.sh.
    Note: Excludes test-qwen-0.5b and commented-out 7B models.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    # Only models that are active (non-commented) in deploy_parallel.py MODEL_CONFIGS
    # Excludes: test-qwen-0.5b, qwen2.5-7b-iter1, qwen2.5-7b-iter8, qwen2.5-7b-iter16
    model_configs = [
        {"name": "qwen2.5-0.5b-iter1", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_iterations": 1},
        {"name": "qwen2.5-1.5b-iter1", "model": "Qwen/Qwen2.5-1.5B-Instruct", "num_iterations": 1},
        {"name": "llama3.2-1b-iter1", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_iterations": 1},
        {"name": "llama3.2-3b-iter1", "model": "meta-llama/Llama-3.2-3B-Instruct", "num_iterations": 1},
        {"name": "gemma3-1b-iter1", "model": "google/gemma-3-1b-it", "num_iterations": 1},
        {"name": "gemma3-4b-iter1", "model": "google/gemma-3-4b-it", "num_iterations": 1},
    ]

    # Create one pod per model configuration
    pods = [
        PodSpec(
            name=config["name"],
            gpu_type="A100",
            gpu_count=8,
            ttl_hours=124,
        )
        for config in model_configs
    ]

    pod_experiments = {
        config["name"]: [] for config in model_configs
    }

    return pods, pod_experiments


# ============================================================================
# Helper Functions
# ============================================================================
def iter_ablation_1_5b() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Iteration ablation for Qwen2.5-1.5B (iter8 and iter16).

    Runs 2 model configurations across 2 pods (each 8xA100):
    - Qwen2.5-1.5B-Instruct with num_iterations=8
    - Qwen2.5-1.5B-Instruct with num_iterations=16

    Each pod runs 4 seeds in parallel via run_parallel_training_nohup.sh.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    model_configs = [
        {"name": "qwen2.5-1.5b-iter8"},
        {"name": "qwen2.5-1.5b-iter16"},
    ]

    pods = [
        PodSpec(
            name=config["name"],
            gpu_type="A100",
            gpu_count=8,
            ttl_hours=24,
        )
        for config in model_configs
    ]

    pod_experiments = {config["name"]: [] for config in model_configs}

    return pods, pod_experiments


def gemma_1b_4b() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Gemma 3 model experiments (1B and 4B).

    Runs 2 model configurations across 2 pods (each 8xA100):
    - google/gemma-3-1b-it with num_iterations=1
    - google/gemma-3-4b-it with num_iterations=1

    Each pod runs 4 seeds in parallel via run_parallel_training_nohup.sh.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    model_configs = [
        {"name": "gemma3-1b-iter1"},
        {"name": "gemma3-4b-iter1"},
    ]

    pods = [
        PodSpec(
            name=config["name"],
            gpu_type="A100",
            gpu_count=8,
            ttl_hours=24,
        )
        for config in model_configs
    ]

    pod_experiments = {config["name"]: [] for config in model_configs}

    return pods, pod_experiments


def gemma_4b_only() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Gemma 3 4B model only (separate deployment).

    Runs google/gemma-3-4b-it on a single 8xA100 pod.
    Use this config to deploy gemma-4b separately from gemma-1b.

    Hyperparameters:
    - batch_size: 2 (4B model)
    - grad_accum_steps: 256 (effective batch = 512)
    - learning_rate: 3e-6
    - num_instances: 4

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(
            name="gemma3-4b-iter1",
            gpu_type="A100",
            gpu_count=8,
            ttl_hours=72,  # ~3 days for 4B model training
        )
    ]

    pod_experiments = {"gemma3-4b-iter1": []}

    return pods, pod_experiments


def iter32_1_5b() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Iteration 32 experiment for Qwen2.5-1.5B.

    Runs Qwen2.5-1.5B-Instruct with num_iterations=32 on a single 8xA100 pod.
    All 4 seeds (42, 1337, 2024, 9999) run in parallel via run_parallel_training_nohup.sh.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(
            name="qwen2.5-1.5b-iter32",
            gpu_type="A100",
            gpu_count=8,
            ttl_hours=48,  # Longer TTL for iter32 (more training steps)
        )
    ]

    pod_experiments = {"qwen2.5-1.5b-iter32": []}

    return pods, pod_experiments


def lr_sweep_1_5b() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Learning rate sweep for Qwen2.5-1.5B (5e-7 vs 5e-6).

    Runs 2 model configurations across 2 pods (each 8xA100):
    - Qwen2.5-1.5B-Instruct with learning_rate=5e-7
    - Qwen2.5-1.5B-Instruct with learning_rate=5e-6

    Each pod runs 4 seeds in parallel via run_parallel_training_nohup.sh.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(name="qwen2.5-1.5b-lr5e-7", gpu_type="A100", gpu_count=8, ttl_hours=96),
        PodSpec(name="qwen2.5-1.5b-lr5e-6", gpu_type="A100", gpu_count=8, ttl_hours=96),
    ]

    pod_experiments = {
        "qwen2.5-1.5b-lr5e-7": [],
        "qwen2.5-1.5b-lr5e-6": [],
    }

    return pods, pod_experiments


def lr_1e6_1_5b() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Single LR experiment for Qwen2.5-1.5B with lr=1e-6.

    Runs Qwen2.5-1.5B-Instruct with learning_rate=1e-6 on a single 8xA100 pod.
    All 4 seeds (42, 1337, 2024, 9999) run in parallel.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(name="qwen2.5-1.5b-lr1e-6", gpu_type="A100", gpu_count=8, ttl_hours=96),
    ]

    pod_experiments = {"qwen2.5-1.5b-lr1e-6": []}

    return pods, pod_experiments


def qwen_1_5b_iter_sweep() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Qwen2.5-1.5B iteration sweep (8, 16, 32).

    Runs 3 model configurations across 3 pods (each 8xA100):
    - Qwen2.5-1.5B-Instruct with num_iterations=8
    - Qwen2.5-1.5B-Instruct with num_iterations=16
    - Qwen2.5-1.5B-Instruct with num_iterations=32

    Each pod runs 4 seeds in parallel via run_parallel_training_nohup.sh.

    Higher iterations = more off-policy steps reusing same rollouts.
    Should be faster than iter1 due to fewer vLLM inference calls.

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(name="qwen2.5-1.5b-iter8", gpu_type="A100", gpu_count=8, ttl_hours=24),
        PodSpec(name="qwen2.5-1.5b-iter16", gpu_type="A100", gpu_count=8, ttl_hours=24),
        PodSpec(name="qwen2.5-1.5b-iter32", gpu_type="A100", gpu_count=8, ttl_hours=24),
    ]

    pod_experiments = {p.name: [] for p in pods}

    return pods, pod_experiments


def qwen_7b_iter1() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Qwen2.5-7B experiment with num_iterations=1.

    Runs Qwen2.5-7B-Instruct on a single 8xA100 pod.
    All 4 seeds (42, 1337, 2024, 9999) run in parallel via run_parallel_training_nohup.sh.

    Hyperparameters:
    - batch_size: 2 (7B model with bf16)
    - grad_accum_steps: 256 (effective batch = 512)
    - learning_rate: 3e-6
    - num_instances: 4

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(
            name="qwen2.5-7b-iter1",
            gpu_type="A100",
            gpu_count=8,
            ttl_hours=132,  # ~5.5 days for 7B model training
        )
    ]

    pod_experiments = {"qwen2.5-7b-iter1": []}

    return pods, pod_experiments


def sft_qwen_1_5b() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """SFT experiment for Qwen2.5-1.5B on 8x A100.

    Runs supervised fine-tuning on a single 8xA100 pod.
    All 8 GPUs run independent SFT training (no vLLM needed).

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(
            name="qwen2.5-1.5b-sft",
            gpu_type="A100",
            gpu_count=8,
            ttl_hours=24,
        )
    ]

    pod_experiments = {"qwen2.5-1.5b-sft": []}

    return pods, pod_experiments


def sft_qwen_1_5b_4gpu() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """SFT experiment for Qwen2.5-1.5B on 4x A100.

    Runs supervised fine-tuning on a single 4xA100 pod (e.g., Hyperstack).
    All 4 GPUs run independent SFT training (no vLLM needed).

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(
            name="qwen2.5-1.5b-sft-4gpu",
            gpu_type="A100",
            gpu_count=4,
            ttl_hours=24,
        )
    ]

    pod_experiments = {"qwen2.5-1.5b-sft-4gpu": []}

    return pods, pod_experiments


def sft_qwen_1_5b_4gpu_lr3e6() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """SFT experiment for Qwen2.5-1.5B on 4x A100 with lr=3e-6.

    Runs supervised fine-tuning with lower learning rate and fewer steps:
    - max_steps: 100 (~2 epochs)
    - learning_rate: 3e-6

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    pods = [
        PodSpec(
            name="qwen2.5-1.5b-sft-4gpu-lr3e6",
            gpu_type="A100",
            gpu_count=4,
            ttl_hours=12,
        )
    ]

    pod_experiments = {"qwen2.5-1.5b-sft-4gpu-lr3e6": []}

    return pods, pod_experiments


def sft_all_models() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """SFT experiments for all supported models.

    Runs supervised fine-tuning across multiple model sizes:
    - Qwen2.5: 0.5B, 1.5B, 7B
    - Llama-3.2: 1B, 3B
    - Gemma-3: 1B, 4B

    Each pod runs 8 independent SFT training instances (one per GPU).

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)
    """
    model_configs = [
        {"name": "qwen2.5-0.5b-sft"},
        {"name": "qwen2.5-1.5b-sft"},
        {"name": "qwen2.5-7b-sft"},
        {"name": "llama3.2-1b-sft"},
        {"name": "llama3.2-3b-sft"},
        {"name": "gemma3-1b-sft"},
        {"name": "gemma3-4b-sft"},
    ]

    pods = [
        PodSpec(
            name=config["name"],
            gpu_type="A100",
            gpu_count=8,
            ttl_hours=24,
        )
        for config in model_configs
    ]

    pod_experiments = {config["name"]: [] for config in model_configs}

    return pods, pod_experiments


def get_config(name: str) -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """Get a predefined configuration by name.

    Args:
        name: Configuration name

    Returns:
        Tuple of (pod_specs, pod_to_experiments mapping)

    Raises:
        ValueError: If configuration name is unknown
    """
    configs = {
        "lr_sweep": learning_rate_sweep_gsm8k,
        "multi_dataset": multi_dataset_comparison,
        "batch_grid": batch_size_grid_search,
        "large_sweep": large_hyperparam_sweep,
        "custom_advanced": custom_advanced_sweep,
        "test_qwen_0.5b": test_qwen_0_5b,
        "multi_model": multi_model_grpo_sweep,
        "active_models": active_models_grpo_sweep,
        "iter_ablation_1_5b": iter_ablation_1_5b,
        "gemma_1b_4b": gemma_1b_4b,
        "gemma_4b_only": gemma_4b_only,
        "iter32_1_5b": iter32_1_5b,
        "lr_sweep_1_5b": lr_sweep_1_5b,
        "lr_1e6_1_5b": lr_1e6_1_5b,
        "qwen_1_5b_iter_sweep": qwen_1_5b_iter_sweep,
        "qwen_7b_iter1": qwen_7b_iter1,
        # SFT experiments
        "sft_qwen_1_5b": sft_qwen_1_5b,
        "sft_qwen_1_5b_4gpu": sft_qwen_1_5b_4gpu,
        "sft_qwen_1_5b_4gpu_lr3e6": sft_qwen_1_5b_4gpu_lr3e6,
        "sft_all_models": sft_all_models,
    }

    if name not in configs:
        raise ValueError(
            f"Unknown configuration: {name}. "
            f"Available: {list(configs.keys())}"
        )

    return configs[name]()


def list_configs() -> list[str]:
    """List all available predefined configurations.

    Returns:
        List of configuration names
    """
    return [
        # GRPO experiments
        "lr_sweep",
        "multi_dataset",
        "batch_grid",
        "large_sweep",
        "custom_advanced",
        "test_qwen_0.5b",
        "multi_model",
        "active_models",
        "iter_ablation_1_5b",
        "gemma_1b_4b",
        "gemma_4b_only",
        "iter32_1_5b",
        "lr_sweep_1_5b",
        "lr_1e6_1_5b",
        "qwen_1_5b_iter_sweep",
        "qwen_7b_iter1",
        # SFT experiments
        "sft_qwen_1_5b",
        "sft_qwen_1_5b_4gpu",
        "sft_qwen_1_5b_4gpu_lr3e6",
        "sft_all_models",
    ]
