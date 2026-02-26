#!/usr/bin/env python3
"""Deploy parallel training experiments on Basilica GPU instances.

Usage:
    python deploy_parallel_basilica.py --config test_qwen_0.5b
    python deploy_parallel_basilica.py --config multi_model --deploy-only
    python deploy_parallel_basilica.py --destroy --config test_qwen_0.5b
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from basilica_manager import BasilicaInfra, GpuConfig, Instance
from experiment_configs import get_config, list_configs
from nohup_experiment_runner import ExperimentConfig, NohupExperimentRunner

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# Model configurations: name -> (model_id, num_iterations, wandb settings)
MODEL_CONFIGS = {
    "test-qwen-0.5b": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "test,qwen-0.5b,basilica",
    },
    "qwen2.5-0.5b-iter1": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-0.5b,iter1,basilica",
    },
    "qwen2.5-1.5b-iter1": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,iter1,basilica",
    },
    "qwen2.5-7b-iter1": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-7b,iter1,basilica",
        "batch_size": 2,           # 7B model with bf16
        "grad_accum_steps": 256,   # 2 * 256 = 512 effective batch
        "num_instances": 4,        # 4 seeds (42, 1337, 2024, 9999)
        "learning_rate": 3e-6,     # Standard LR
    },
    "qwen2.5-1.5b-iter8": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 8,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,iter8,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,  # 4 seeds on full node
    },
    "qwen2.5-1.5b-iter16": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 16,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,iter16,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,  # 4 seeds on full node
    },
    "llama3.2-1b-iter1": {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "llama-1b,iter1,basilica",
    },
    "llama3.2-3b-iter1": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "llama-3b,iter1,basilica",
    },
    "gemma3-1b-iter1": {
        "model": "google/gemma-3-1b-it",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "gemma-1b,iter1,basilica",
        "batch_size": 4,  # 1B model can use larger batch
        "grad_accum_steps": 128,  # 4 * 128 = 512 effective batch
        "num_instances": 4,  # 4 instances on 8-GPU node (4 × 2 GPUs)
    },
    "gemma3-4b-iter1": {
        "model": "google/gemma-3-4b-it",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "gemma-4b,iter1,basilica",
        "batch_size": 2,  # 4B model needs smaller batch
        "grad_accum_steps": 256,  # 2 * 256 = 512 effective batch
        "num_instances": 4,  # 4 instances on 8-GPU node (4 × 2 GPUs)
        "learning_rate": 3e-6,  # Standard LR
    },
    "qwen2.5-1.5b-iter32": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 32,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,iter32,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,  # 4 seeds on one 8-GPU node (4 × 2 GPUs)
    },
    "qwen2.5-1.5b-lr5e-7": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,lr-sweep,lr5e-7,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,
        "learning_rate": 5e-7,
    },
    "qwen2.5-1.5b-lr5e-7-seed2024": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,lr-sweep,lr5e-7,basilica,seed2024-retry",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 1,  # Single seed on 2 GPUs
        "learning_rate": 5e-7,
        "seed": 2024,  # Only seed 2024
    },
    "qwen2.5-1.5b-lr5e-6": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,lr-sweep,lr5e-6,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,
        "learning_rate": 5e-6,
    },
    "qwen2.5-1.5b-lr1e-6": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,lr-sweep,lr1e-6,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,
        "learning_rate": 1e-6,
    },
    "qwen2.5-1.5b-lr5e-5": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,lr-sweep,lr5e-5,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,
        "learning_rate": 5e-5,
    },
    # ════════════════════════════════════════════════════════════════════════════
    # MBPP CONFIGURATIONS (Coding Tasks)
    # Note: MBPP requires CodeExecutionPool for test execution
    # ════════════════════════════════════════════════════════════════════════════
    "qwen2.5-1.5b-mbpp-iter1": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,mbpp,iter1,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,  # 4 seeds on 8-GPU node (4 × 2 GPUs)
        "learning_rate": 3e-6,
        "dataset": "mbpp",  # Use MBPP coding dataset
    },
    # ════════════════════════════════════════════════════════════════════════════
    # OPTIMIZER ABLATION CONFIGURATIONS
    # ════════════════════════════════════════════════════════════════════════════
    "qwen2.5-1.5b-beta2-0.95": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,iter1,beta2-0.95,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 4,
        "learning_rate": 3e-6,
        "adam_beta2": 0.95,
    },
    "qwen2.5-1.5b-fp32": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,iter1,fp32,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 1,
        "learning_rate": 3e-6,
        "dtype": "float32",
    },
    "qwen2.5-1.5b-bf16-analysis": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,iter1,bf16,analysis,grad-momentum,basilica",
        "batch_size": 4,
        "grad_accum_steps": 128,
        "num_instances": 1,
        "learning_rate": 3e-6,
    },
    # ════════════════════════════════════════════════════════════════════════════
    # SFT CONFIGURATIONS (Supervised Fine-Tuning)
    # Note: SFT uses 1 GPU per instance (no vLLM needed), so can run up to 8 instances
    # ════════════════════════════════════════════════════════════════════════════
    "qwen2.5-1.5b-sft-analysis": {
        "trainer_type": "sft",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,sft,analysis,lr3e-6,grad-momentum,basilica",
        "batch_size": 4,
        "grad_accum_steps": 32,
        "num_instances": 1,
        "learning_rate": 3e-6,
    },
    "qwen2.5-0.5b-sft": {
        "trainer_type": "sft",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-0.5b,sft,basilica",
        "batch_size": 4,
        "grad_accum_steps": 32,
        "num_instances": 8,  # SFT can use all 8 GPUs
        "learning_rate": 2e-5,
    },
    "qwen2.5-1.5b-sft": {
        "trainer_type": "sft",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,sft,basilica",
        "batch_size": 4,
        "grad_accum_steps": 32,
        "num_instances": 8,  # SFT can use all 8 GPUs
        "learning_rate": 2e-5,
    },
    "qwen2.5-1.5b-sft-4gpu": {
        "trainer_type": "sft",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,sft,4gpu,basilica",
        "batch_size": 4,
        "grad_accum_steps": 32,
        "num_instances": 4,  # 4x A100 node (e.g., Hyperstack)
        "learning_rate": 2e-5,
    },
    "qwen2.5-1.5b-sft-4gpu-lr3e6": {
        "trainer_type": "sft",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_steps": 100,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-1.5b,sft,4gpu,lr3e-6,basilica",
        "batch_size": 4,
        "grad_accum_steps": 32,
        "num_instances": 4,  # 4x A100 node (e.g., Hyperstack)
        "learning_rate": 3e-6,
    },
    "qwen2.5-7b-sft": {
        "trainer_type": "sft",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "qwen-7b,sft,basilica",
        "batch_size": 2,  # Smaller batch for 7B model
        "grad_accum_steps": 64,
        "num_instances": 8,  # SFT can use all 8 GPUs
        "learning_rate": 2e-5,
    },
    "llama3.2-1b-sft": {
        "trainer_type": "sft",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "llama-1b,sft,basilica",
        "batch_size": 4,
        "grad_accum_steps": 32,
        "num_instances": 8,
        "learning_rate": 2e-5,
    },
    "llama3.2-3b-sft": {
        "trainer_type": "sft",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "llama-3b,sft,basilica",
        "batch_size": 4,
        "grad_accum_steps": 32,
        "num_instances": 8,
        "learning_rate": 2e-5,
    },
    "gemma3-1b-sft": {
        "trainer_type": "sft",
        "model": "google/gemma-3-1b-it",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "gemma-1b,sft,basilica",
        "batch_size": 4,
        "grad_accum_steps": 32,
        "num_instances": 8,
        "learning_rate": 2e-5,
    },
    "gemma3-4b-sft": {
        "trainer_type": "sft",
        "model": "google/gemma-3-4b-it",
        "max_steps": 400,
        "wandb_project": "grail-basilica-sweep",
        "wandb_tags": "gemma-4b,sft,basilica",
        "batch_size": 2,  # Smaller batch for 4B model
        "grad_accum_steps": 64,
        "num_instances": 8,
        "learning_rate": 2e-5,
    },
}


def lium_to_basilica_config(lium_spec) -> GpuConfig:
    """Convert lium PodSpec to GpuConfig."""
    return GpuConfig(
        name=lium_spec.name,
        gpu_type=lium_spec.gpu_type,
        gpu_count=lium_spec.gpu_count,
        country=getattr(lium_spec, "country", None),
    )


def get_r2_config() -> dict[str, str] | None:
    """Load R2 configuration from environment."""
    bucket = os.getenv("R2_BUCKET_NAME") or os.getenv("R2_BUCKET_ID")

    config = {
        "bucket_id": bucket,
        "account_id": os.getenv("R2_ACCOUNT_ID"),
        "access_key": os.getenv("R2_WRITE_ACCESS_KEY_ID"),
        "secret_key": os.getenv("R2_WRITE_SECRET_ACCESS_KEY"),
    }

    if not all(config.values()):
        logger.error("Missing R2 config. Required: R2_BUCKET_NAME, R2_ACCOUNT_ID, R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY")
        return None

    return config


MAX_CONCURRENT_SSH = 5  # Limit concurrent SSH connections


async def run_experiment(
    instance: Instance,
    r2_config: dict[str, str],
    dataset: str = "math",
    eval_every: int = 40,
    sync_code: bool = True,
    setup_env: bool = True,
    semaphore: asyncio.Semaphore | None = None,
) -> bool:
    """Run experiment on a single instance.

    Args:
        instance: The GPU instance to run on
        r2_config: R2 storage configuration
        dataset: Dataset to use for training
        eval_every: Evaluation frequency
        sync_code: Whether to sync code to remote
        setup_env: Whether to set up environment
        semaphore: Optional semaphore to limit concurrent connections
    """
    # Use semaphore if provided to limit concurrent connections
    if semaphore:
        async with semaphore:
            return await _run_experiment_impl(
                instance, r2_config, dataset, eval_every, sync_code, setup_env
            )
    return await _run_experiment_impl(
        instance, r2_config, dataset, eval_every, sync_code, setup_env
    )


async def _run_experiment_impl(
    instance: Instance,
    r2_config: dict[str, str],
    dataset: str,
    eval_every: int,
    sync_code: bool,
    setup_env: bool,
) -> bool:
    """Internal implementation of run_experiment."""
    name = instance.name

    if name not in MODEL_CONFIGS:
        logger.error(f"No model config for: {name}")
        return False

    model_cfg = MODEL_CONFIGS[name]

    # Determine trainer type and defaults
    trainer_type = model_cfg.get("trainer_type", "grpo")
    # Default num_instances: 4 for GRPO (2 GPUs each), 4 for SFT (1 GPU each)
    default_num_instances = 4

    # Use dataset from model config if specified, otherwise use command-line default
    effective_dataset = model_cfg.get("dataset", dataset)

    config = ExperimentConfig(
        name=name,
        model_id=model_cfg["model"],
        num_iterations=model_cfg.get("num_iterations", 1),
        trainer_type=trainer_type,
        max_steps=model_cfg.get("max_steps", 400),
        dataset=effective_dataset,
        eval_every=eval_every,
        wandb_project=model_cfg.get("wandb_project", "grail-basilica-sweep"),
        wandb_tags=model_cfg.get("wandb_tags", ""),
        batch_size=model_cfg.get("batch_size"),
        grad_accum_steps=model_cfg.get("grad_accum_steps"),
        num_instances=model_cfg.get("num_instances", default_num_instances),
        # Port isolation for running multiple experiments on same node
        run_prefix=model_cfg.get("run_prefix"),
        seed=model_cfg.get("seed"),
        start_instance=model_cfg.get("start_instance", 0),
        base_port=model_cfg.get("base_port", 8000),
        base_group_port=model_cfg.get("base_group_port", 51200),
        vllm_nixl_port_base=model_cfg.get("vllm_nixl_port_base", 5557),
        vllm_master_port_base=model_cfg.get("vllm_master_port_base", 29500),
        learning_rate=model_cfg.get("learning_rate"),
        adam_beta2=model_cfg.get("adam_beta2"),
        dtype=model_cfg.get("dtype"),
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting: {name} ({trainer_type.upper()})")
    logger.info(f"  Model: {config.model_id}")
    if trainer_type == "grpo":
        logger.info(f"  Iterations: {config.num_iterations}")
    else:
        logger.info(f"  Max Steps: {config.max_steps}")
    logger.info(f"  Num instances: {config.num_instances} (on 8-GPU node)")
    if config.batch_size and config.grad_accum_steps:
        logger.info(f"  Batch size: {config.batch_size} | Grad accum: {config.grad_accum_steps} | Effective: {config.batch_size * config.grad_accum_steps}")
    logger.info(f"{'='*60}\n")

    runner = NohupExperimentRunner(
        ssh_host=instance.ssh.host,
        ssh_port=instance.ssh.port,
        ssh_user=instance.ssh.user,
        r2_config=r2_config,
    )

    success = await runner.run_experiment(
        config=config,
        local_code_path=PROJECT_ROOT,
        local_env_path=PROJECT_ROOT / ".env",
        sync_code=sync_code,
        setup_env=setup_env,
        upload_to_r2=True,
        cleanup_local=False,
    )

    status = "✓" if success else "✗"
    logger.info(f"{status} Experiment {name} {'completed' if success else 'failed'}")
    return success


async def deploy_and_run(
    config_name: str,
    dataset: str = "math",
    eval_every: int = 40,
    deploy_only: bool = False,
    no_deploy: bool = False,
    sync_code: bool = True,
    setup_env: bool = True,
    state_file: str | None = None,
) -> bool:
    """Deploy instances and run experiments."""
    state_file = state_file or f".basilica_state_{config_name}.json"

    # Load experiment config and convert to GpuConfig
    logger.info(f"Loading config: {config_name}")
    lium_pods, _ = get_config(config_name)
    configs = [lium_to_basilica_config(p) for p in lium_pods]

    # Validate configuration
    if not configs:
        logger.error(f"No instances defined in config '{config_name}'")
        return False

    logger.info(f"Configurations:")
    missing_model_configs = []
    for c in configs:
        logger.info(f"  - {c.name}: {c.gpu_count}x {c.gpu_type}")
        if c.name not in MODEL_CONFIGS:
            missing_model_configs.append(c.name)

    if missing_model_configs:
        logger.error(f"Missing MODEL_CONFIGS entries for: {missing_model_configs}")
        logger.error("Add these to MODEL_CONFIGS dict before deploying")
        return False

    infra = BasilicaInfra(state_file=state_file)

    # Deploy
    if not no_deploy:
        logger.info(f"\n{'='*60}")
        logger.info(f"Deploying {len(configs)} instances...")
        logger.info(f"{'='*60}\n")

        instances = infra.apply(configs)
        if not instances:
            logger.error("Deployment failed - no instances provisioned")
            return False

        if len(instances) < len(configs):
            logger.warning(f"⚠️  Partial deployment: {len(instances)}/{len(configs)} instances")
            failed_names = [c.name for c in configs if c.name not in instances]
            logger.warning(f"   Failed: {failed_names}")

        logger.info(f"\n✓ Deployed {len(instances)}/{len(configs)} instances")

        # Verify health
        logger.info(f"\n{'='*60}")
        logger.info("Verifying instance health...")
        logger.info(f"{'='*60}\n")

        instances = infra.verify_health(configs, max_retries=3)
        if not instances:
            logger.error("No healthy instances")
            return False

        logger.info(f"\n✓ {len(instances)} healthy instances")

        for name, inst in instances.items():
            logger.info(f"  {name}: {inst.ssh.host}:{inst.ssh.port} ({inst.config.gpu_count}x {inst.config.gpu_type})")

        if deploy_only:
            logger.info("\n✓ Deploy-only mode complete")
            return True
    else:
        instances = infra.list_instances()
        logger.info(f"Using {len(instances)} existing instances")

    # Get R2 config
    r2_config = get_r2_config()
    if not r2_config:
        return False

    # Run experiments in parallel
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {len(instances)} experiments...")
    logger.info(f"{'='*60}\n")

    # Create named tasks with concurrency limit
    instance_names = list(instances.keys())
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SSH)
    logger.info(f"Using concurrency limit of {MAX_CONCURRENT_SSH} simultaneous connections")

    tasks = [
        run_experiment(
            instance=inst,
            r2_config=r2_config,
            dataset=dataset,
            eval_every=eval_every,
            sync_code=sync_code,
            setup_env=setup_env,
            semaphore=semaphore,
        )
        for inst in instances.values()
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results with detailed error logging
    success_count = 0
    failed_experiments = []

    for name, result in zip(instance_names, results):
        if result is True:
            success_count += 1
        elif isinstance(result, Exception):
            failed_experiments.append((name, str(result)))
            logger.error(f"❌ {name} raised exception: {type(result).__name__}: {result}")
        else:
            failed_experiments.append((name, "returned False"))
            logger.error(f"❌ {name} failed (returned False)")

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {success_count} succeeded, {len(failed_experiments)} failed")
    if failed_experiments:
        logger.info("Failed experiments:")
        for name, reason in failed_experiments:
            logger.info(f"  - {name}: {reason}")
    logger.info(f"{'='*60}\n")

    return len(failed_experiments) == 0


def destroy(config_name: str | None, state_file: str | None) -> bool:
    """Destroy all managed instances."""
    if not state_file and not config_name:
        print("Error: --destroy requires --config or --state-file")
        return False

    state_file = state_file or f".basilica_state_{config_name}.json"
    logger.info(f"Destroying instances from {state_file}")

    infra = BasilicaInfra(state_file=state_file)
    return infra.terminate_all()


def setup_logging():
    """Configure logging for visibility."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    # Unbuffered output for nohup
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy training experiments on Basilica")

    parser.add_argument("--config", choices=list_configs(), help="Experiment configuration")
    parser.add_argument("--list-configs", action="store_true", help="List available configs")
    parser.add_argument("--destroy", action="store_true", help="Destroy instances")

    parser.add_argument("--dataset", default="math", choices=["gsm8k", "math", "mbpp"])
    parser.add_argument("--eval-every", type=int, default=40)

    parser.add_argument("--deploy-only", action="store_true", help="Only deploy, don't run")
    parser.add_argument("--no-deploy", action="store_true", help="Use existing instances")
    parser.add_argument("--no-sync", action="store_true", help="Skip code sync")
    parser.add_argument("--no-setup", action="store_true", help="Skip env setup")

    parser.add_argument("--state-file", help="State file path")

    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    setup_logging()

    if args.list_configs:
        print("\nAvailable configurations:")
        for name in list_configs():
            print(f"  - {name}")
        return 0

    if args.destroy:
        success = destroy(args.config, args.state_file)
        return 0 if success else 1

    if not args.config:
        print("Error: --config required")
        return 1

    success = await deploy_and_run(
        config_name=args.config,
        dataset=args.dataset,
        eval_every=args.eval_every,
        deploy_only=args.deploy_only,
        no_deploy=args.no_deploy,
        sync_code=not args.no_sync,
        setup_env=not args.no_setup,
        state_file=args.state_file,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
