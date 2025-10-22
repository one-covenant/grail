"""Shared pytest fixtures and configuration for validation service tests.

This module provides reusable fixtures for testing the validation service layer.
Fixtures follow pytest best practices with proper scoping and dependency injection.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from accelerate import Accelerator

    from grail.validation.copycat_service import CopycatTracker

from .proof_test_utils import generate_realistic_sat_prompt

# ============================================================================
# Test Constants
# ============================================================================

TEST_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# ============================================================================
# Test Environment Setup
# ============================================================================


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide safe default environment for tests.

    Keeps tests deterministic and avoids accidental network/monitoring.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("GRAIL_MONITORING_BACKEND", "null")
    monkeypatch.setenv("BT_NETWORK", "test")
    monkeypatch.setenv("NETUID", "1")
    monkeypatch.setenv("GRAIL_ROLLOUTS_PER_PROBLEM", "4")
    # Ensure project root is on path when running from different CWDs
    root = str(pathlib.Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)


# ============================================================================
# Mock Objects - Reusable across tests
# ============================================================================


@pytest.fixture
def mock_subtensor() -> AsyncMock:
    """Mock bittensor subtensor for testing."""
    subtensor = AsyncMock()
    subtensor.get_current_block = AsyncMock(return_value=1000)
    subtensor.get_block_hash = AsyncMock(
        return_value="0x" + hashlib.sha256(b"block_1000").hexdigest()
    )
    subtensor.metagraph = AsyncMock()
    subtensor.set_weights = AsyncMock()
    return subtensor


@pytest.fixture
def mock_metagraph() -> MagicMock:
    """Mock bittensor metagraph with sample miners."""
    metagraph = MagicMock()
    metagraph.hotkeys = [f"hotkey_{i}" for i in range(20)]
    metagraph.uids = list(range(20))
    return metagraph


@pytest.fixture
def mock_chain_manager() -> MagicMock:
    """Mock GrailChainManager for credential access."""
    manager = MagicMock()
    manager.get_bucket = MagicMock(return_value=None)
    manager.stop = MagicMock()
    return manager


@pytest.fixture
def mock_credentials() -> MagicMock:
    """Mock bucket credentials."""
    credentials = MagicMock()
    credentials.access_key = "test_key"
    credentials.secret_key = "test_secret"
    return credentials


@pytest.fixture
def mock_wallet() -> MagicMock:
    """Mock bittensor wallet."""
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "validator_hotkey_test"
    return wallet


@pytest.fixture
def mock_monitor() -> AsyncMock:
    """Mock monitoring client."""
    monitor = AsyncMock()
    monitor.log_gauge = AsyncMock()
    monitor.timer = MagicMock()
    monitor.set_block_context = MagicMock()
    return monitor


# ============================================================================
# Test Data - Sample validation data
# ============================================================================


@pytest.fixture
def sample_window_data() -> dict[str, str | int]:
    """Sample window metadata."""
    return {
        "window": 1000,
        "window_hash": "0x" + hashlib.sha256(b"window_1000").hexdigest(),
        "window_rand": "0x" + hashlib.sha256(b"rand_1000").hexdigest(),
    }


@pytest.fixture
def sample_miner_rollouts() -> list[dict[str, str | int | float]]:
    """Sample rollouts for a miner."""
    return [
        {
            "completion_digest": f"digest_{i}",
            "hotkey": "miner_1",
            "problem_id": i // 4,  # 4 rollouts per problem
            "advantage": 0.1 * (i % 4 - 1.5),  # Sum to ~0 per group
            "reward": 0.5 + 0.1 * i,
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_validation_metrics() -> dict[str, dict[str, int]]:
    """Sample validation metrics for multiple miners."""
    return {
        "miner_1": {"valid": 10, "checked": 12, "total": 20, "successful": 8, "unique": 10},
        "miner_2": {"valid": 8, "checked": 10, "total": 15, "successful": 6, "unique": 8},
        "miner_3": {"valid": 5, "checked": 8, "total": 10, "successful": 3, "unique": 5},
    }


@pytest.fixture
def sample_rollout_counters() -> dict[str, Counter[str]]:
    """Sample rollout counters for copycat detection."""
    return {
        "miner_1": Counter(["digest_1", "digest_2", "digest_3", "digest_4", "digest_5"]),
        "miner_2": Counter(["digest_1", "digest_2", "digest_6", "digest_7"]),  # Copies 2
        "miner_3": Counter(["digest_8", "digest_9", "digest_10"]),  # Unique
    }


# ============================================================================
# Parametrized Test Data
# ============================================================================


@pytest.fixture(
    params=[
        (10, 0.2, 5, None, 5),  # (active, rate, min, max, expected)
        (100, 0.2, 5, 50, 20),
        (200, 0.2, 5, 20, 20),  # Hits max
        (0, 0.2, 5, None, 0),  # No active miners
        (5, 0.2, 10, None, 10),  # Hits min
    ]
)
def sample_size_cases(request: pytest.FixtureRequest) -> tuple[int, float, int, int | None, int]:
    """Parametrized test cases for sample size calculation."""
    return request.param


# ============================================================================
# SAT Prompts for Proof Tests (Original fixtures)
# ============================================================================


@pytest.fixture(scope="session")
def sat_prompts() -> list[str]:
    """Pregenerated realistic SAT prompts for proof tests.

    Uses deterministic seeds and production chat templates to ensure
    prompts match actual mining scenarios.
    """
    from transformers import AutoTokenizer

    # Load tokenizer for chat template application
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_ID)

    seeds = [
        "test_seed_easy_01",
        "test_seed_medium_02",
        "test_seed_hard_03",
        "test_seed_compact_04",
        "test_seed_large_05",
    ]
    difficulties = [0.3, 0.5, 0.7, 0.5, 0.8]
    return [
        generate_realistic_sat_prompt(seed, diff, tokenizer)
        for seed, diff in zip(seeds, difficulties, strict=False)
    ]


@pytest.fixture
def sat_prompt_tokens(sat_prompts: list[str]) -> list[int]:
    """Tokenized SAT prompts for testing."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_ID)
    return [tokenizer.encode(prompt, add_special_tokens=False) for prompt in sat_prompts]


@pytest.fixture(scope="session")
def tracker() -> CopycatTracker:
    """CopycatTracker instance for testing."""
    from grail.validation.copycat_service import CopycatTracker

    return CopycatTracker()


# ============================================================================
# TRAINER-SPECIFIC FIXTURES
# ============================================================================


@pytest.fixture
def seeded_torch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set all random seeds for reproducibility and force CPU execution.

    Enables deterministic behavior in torch and ensures tests run on CPU
    for consistent results across runs.
    """
    import random

    import numpy as np
    import torch

    # Set Python random seed
    random.seed(42)
    # Set NumPy seed
    np.random.seed(42)
    # Set torch seed on CPU
    torch.manual_seed(42)

    # Force CPU execution
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Enable deterministic algorithms in torch (may reduce performance)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


@pytest.fixture
def tiny_qwen_model_and_tokenizer() -> tuple[Any, Any]:
    """Load Qwen 1.5B model and tokenizer, ensure pad_token_id is set.

    Uses a small model suitable for fast CPU-based testing. Lazy-loads
    to avoid slow imports in unrelated tests.

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-1.5B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    model.eval()

    return model, tokenizer


@pytest.fixture
def monkeypatch_trainer_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override trainer constants for fast tests.

    Shrinks batch size, max length, and gradient accumulation steps
    to speed up test execution while maintaining correctness.
    """
    import grail.shared.constants as constants

    monkeypatch.setattr(constants, "TRAINER_MAX_LENGTH", 256)
    monkeypatch.setattr(constants, "TRAINER_BATCH_SIZE", 4)
    monkeypatch.setattr(constants, "TRAINER_GRAD_ACCUM_STEPS", 2)
    monkeypatch.setattr(constants, "ROLLOUTS_PER_PROBLEM", 4)


@pytest.fixture
def accelerator_cpu() -> Accelerator:
    """Create Accelerator instance configured for CPU execution.

    Returns a deterministic, CPU-based accelerator suitable for testing.
    """
    from accelerate import Accelerator

    return Accelerator(
        mixed_precision="no",
        device_placement=False,
    )


@pytest.fixture
def gsm8k_env_factory() -> Callable[[], Any]:
    """Factory function for creating GSM8KEnv instances.

    Returns a callable that creates fresh GSM8KEnv instances for tests.
    """
    from grail.environments.gsm8k_env import GSM8KEnv

    def _make_env() -> Any:
        return GSM8KEnv()

    return _make_env
