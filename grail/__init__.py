#!/usr/bin/env python3
__version__ = "0.0.35"

from dotenv import load_dotenv

# Load environment variables as early as possible so any module-level
# reads (e.g., in shared.constants) see updated values from .env.
load_dotenv(override=True)

from .environments import (  # noqa: F401, E402, E501, F403, F405
    # Loop and rollouts
    AgentEnvLoop,
    GRPORollout,
    # Core reward system
    Parser,
    RewardVector,
    SATEnv,
    SATParser,
    # SAT exports
    SATProblem,
    create_sat_reward_vector,
    generate_sat_problem,
)

# Stage 3c: Verifier class deleted
# Use grail.validation.create_sat_validation_pipeline() instead
try:
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
except Exception:  # pragma: no cover - optional in offline mode
    # Allow importing grail package without comms/bittensor installed
    pass
from .infrastructure.drand import (
    get_drand_beacon,
    get_round_at_time,
)  # noqa: F401, E402, E501, F403, F405

# RolloutGenerator removed; use AgentEnvLoop instead

# flake8: noqa: E402,E501,F401,F403,F405

__all__ = [
    # Core reward system
    "Parser",
    "RewardVector",
    # SAT
    "SATProblem",
    "SATParser",
    "SATEnv",
    "create_sat_reward_vector",
    "generate_sat_problem",
    # Loop
    "AgentEnvLoop",
    "GRPORollout",
    # Entry points
    "main",
]
try:
    from .cli import main  # noqa: E402,F401
except Exception:  # pragma: no cover - optional in offline mode
    # CLI includes bittensor-dependent trainer; keep import optional offline
    def main() -> None:  # type: ignore[override]
        raise RuntimeError("grail CLI unavailable in offline mode without bittensor")
