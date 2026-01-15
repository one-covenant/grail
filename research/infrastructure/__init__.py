"""Lium infrastructure for distributed training experiments.

This package provides declarative infrastructure management and experiment
orchestration for running distributed training on Lium cloud infrastructure.

Main components:
- lium_manager: Infrastructure management with bandwidth filtering
- experiment_runner: SSH-based experiment orchestration
- experiment_configs: Predefined experiment configurations
- deploy: Main CLI for deployment and execution

Quick start:
    >>> from lium_manager import LiumInfra, PodSpec
    >>> from experiment_runner import ExperimentConfig
    >>>
    >>> # Define infrastructure
    >>> pods = [PodSpec(name="trainer-0", gpu_count=8)]
    >>>
    >>> # Deploy pods
    >>> infra = LiumInfra()
    >>> infra.apply(pods)
"""

__version__ = "0.1.0"

from .experiment_configs import ExperimentConfig, get_config, list_configs
from .experiment_runner import ExperimentRunner, run_experiments_on_pod, run_experiments_parallel
from .lium_manager import LiumInfra, PodSpec

__all__ = [
    "LiumInfra",
    "PodSpec",
    "ExperimentConfig",
    "ExperimentRunner",
    "run_experiments_on_pod",
    "run_experiments_parallel",
    "get_config",
    "list_configs",
]
