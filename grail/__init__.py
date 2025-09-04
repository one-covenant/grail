#!/usr/bin/env python3
__version__ = "0.1.0.dev0"

from dotenv import load_dotenv

# Load environment variables as early as possible so any module-level
# reads (e.g., in shared.constants) see updated values from .env.
try:
    load_dotenv(override=True)
except Exception:
    pass

from .grail import Prover, Verifier  # noqa: F401
from .infrastructure.drand import get_drand_beacon, get_round_at_time  # noqa: F401
from .environments import (  # noqa: F401
    # New reward system
    Parser,
    RewardVector,
    SATParser,
    create_sat_reward_vector,
    # Existing classes
    SATProblem,
    generate_sat_problem,
    SATRolloutGenerator,
)
from .mining.rollout_generator import RolloutGenerator  # noqa: F401
from .infrastructure.comms import (  # noqa: F401
    upload_file_chunked,
    download_file_chunked,
    file_exists,
    list_bucket_files,
    get_file,
    sink_window_inferences,
    # TODO(v2): Re-enable model state management for training
    # save_model_state, load_model_state, model_state_exists,
    upload_valid_rollouts,
    get_valid_rollouts,
    # NEW: Hugging Face dataset upload
    upload_to_huggingface,
    download_from_huggingface,
    login_huggingface,
    PROTOCOL_VERSION,
)

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
