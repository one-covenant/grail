#!/usr/bin/env python3
__version__ = "0.1.0.dev0"

from .grail import Prover, Verifier
from .drand import get_drand_beacon, get_round_at_time
from .environments import (
    # New reward system
    Parser, RewardVector, SATParser, 
    create_sat_reward_vector,
    # Existing classes
    SATProblem, generate_sat_problem, SATRolloutGenerator
)
from .rollout import RolloutGenerator
from .comms import (
    upload_file_chunked, download_file_chunked, file_exists, list_bucket_files,
    get_file, sink_window_inferences, 
    # TODO(v2): Re-enable model state management for training
    # save_model_state, load_model_state, model_state_exists,
    upload_valid_rollouts, get_valid_rollouts,
    # NEW: Hugging Face dataset upload
    upload_to_huggingface, download_from_huggingface, login_huggingface, PROTOCOL_VERSION
)

__all__ = [
    # Core classes
    "Prover", "Verifier", 
    # New reward system
    "Parser", "RewardVector", "SATParser",
    "create_sat_reward_vector",
    # Existing SAT classes
    "SATProblem", "generate_sat_problem", "SATRolloutGenerator",
    # Entry points
    "main"
]

from .cli import main  # noqa: E402,F401
