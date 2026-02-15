#!/usr/bin/env python3
from __future__ import annotations

__version__ = "0.0.49"

from importlib import import_module
from typing import Any

from dotenv import load_dotenv

# Load environment variables as early as possible so any module-level
# reads (e.g., in shared.constants) see updated values from .env.
load_dotenv(override=True)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Environments / rollouts
    "AgentEnvLoop": ("grail.environments.loop", "AgentEnvLoop"),
    "GRPORollout": ("grail.environments.loop", "GRPORollout"),
    # Legacy reward system exports
    "Parser": ("grail.environments.base", "Parser"),
    "RewardVector": ("grail.environments.base", "RewardVector"),
    # SAT exports
    "SATEnv": ("grail.environments.sat_env", "SATEnv"),
    "SATParser": ("grail.environments.sat_env", "SATParser"),
    "SATProblem": ("grail.environments.sat_env", "SATProblem"),
    "create_sat_reward_vector": ("grail.environments.sat_env", "create_sat_reward_vector"),
    "generate_sat_problem": ("grail.environments.sat_env", "generate_sat_problem"),
    # Comms exports (bittensor-dependent; imported only on attribute access)
    "PROTOCOL_VERSION": ("grail.infrastructure.comms", "PROTOCOL_VERSION"),
    "download_file_chunked": ("grail.infrastructure.comms", "download_file_chunked"),
    "download_from_huggingface": ("grail.infrastructure.comms", "download_from_huggingface"),
    "file_exists": ("grail.infrastructure.comms", "file_exists"),
    "get_file": ("grail.infrastructure.comms", "get_file"),
    "get_valid_rollouts": ("grail.infrastructure.comms", "get_valid_rollouts"),
    "list_bucket_files": ("grail.infrastructure.comms", "list_bucket_files"),
    "login_huggingface": ("grail.infrastructure.comms", "login_huggingface"),
    "sink_window_inferences": ("grail.infrastructure.comms", "sink_window_inferences"),
    "upload_file_chunked": ("grail.infrastructure.comms", "upload_file_chunked"),
    "upload_to_huggingface": ("grail.infrastructure.comms", "upload_to_huggingface"),
    "upload_valid_rollouts": ("grail.infrastructure.comms", "upload_valid_rollouts"),
    # Drand exports
    "get_drand_beacon": ("grail.infrastructure.drand", "get_drand_beacon"),
    "get_round_at_time": ("grail.infrastructure.drand", "get_round_at_time"),
    # CLI entrypoint (bittensor-dependent; imported only on access)
    "main": ("grail.cli", "main"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute access to keep `import grail` lightweight.

    This avoids importing bittensor-heavy modules during offline usage.
    """
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_EXPORTS.keys())))


__all__ = sorted(
    {
        "Parser",
        "RewardVector",
        "SATProblem",
        "SATParser",
        "SATEnv",
        "create_sat_reward_vector",
        "generate_sat_problem",
        "AgentEnvLoop",
        "GRPORollout",
        "main",
    }
)
