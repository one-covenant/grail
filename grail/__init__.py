#!/usr/bin/env python3
__version__ = "0.0.18"

from dotenv import load_dotenv

# Load environment variables as early as possible so any module-level
# reads (e.g., in shared.constants) see updated values from .env.
load_dotenv(override=True)

from .environments import (  # noqa: F401, E402, E501, F403, F405
    # New reward system
    Parser,
    RewardVector,
    SATParser,
    # Existing classes
    SATProblem,
    SATRolloutGenerator,
    create_sat_reward_vector,
    generate_sat_problem,
)
from .grail import Prover, Verifier  # noqa: F401, E402, E501, F403, F405
from .infrastructure.comms import (  # noqa: F401, E402, E501, F403, F405
    PROTOCOL_VERSION,
    download_file_chunked,
    download_from_huggingface,
    file_exists,
    get_file,
    get_valid_rollouts,
    list_bucket_files,
    login_huggingface,
    sink_window_inferences,
    upload_file_chunked,
    # NEW: Hugging Face dataset upload
    upload_to_huggingface,
    # TODO(v2): Re-enable model state management for training
    # save_model_state, load_model_state, model_state_exists,
    upload_valid_rollouts,
)
from .infrastructure.drand import (
    get_drand_beacon,
    get_round_at_time,
)  # noqa: F401, E402, E501, F403, F405
from .mining.rollout_generator import RolloutGenerator  # noqa: F401, E402, E501, F403, F405

# flake8: noqa: E402,E501,F401,F403,F405

__all__ = [
    # Core classes
    "Prover",
    "Verifier",
    # New reward system
    "Parser",
    "RewardVector",
    "SATParser",
    "create_sat_reward_vector",
    # Existing SAT classes
    "SATProblem",
    "generate_sat_problem",
    "SATRolloutGenerator",
    # Entry points
    "main",
]

from .cli import main  # noqa: E402,F401
