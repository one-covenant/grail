"""Shared pytest fixtures and configuration for validation service tests.

This module provides reusable fixtures for testing the validation service layer.
Fixtures follow pytest best practices with proper scoping and dependency injection.
"""

import hashlib
import pathlib
import sys
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest

from .proof_test_utils import generate_realistic_sat_prompt

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
def mock_subtensor():
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
def mock_metagraph():
    """Mock bittensor metagraph with sample miners."""
    metagraph = MagicMock()
    metagraph.hotkeys = [f"hotkey_{i}" for i in range(20)]
    metagraph.uids = list(range(20))
    return metagraph


@pytest.fixture
def mock_chain_manager():
    """Mock GrailChainManager for credential access."""
    manager = MagicMock()
    manager.get_bucket = MagicMock(return_value=None)
    manager.stop = MagicMock()
    return manager


@pytest.fixture
def mock_credentials():
    """Mock bucket credentials."""
    credentials = MagicMock()
    credentials.access_key = "test_key"
    credentials.secret_key = "test_secret"
    return credentials


@pytest.fixture
def mock_wallet():
    """Mock bittensor wallet."""
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "validator_hotkey_test"
    return wallet


@pytest.fixture
def mock_monitor():
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
def sample_window_data():
    """Sample window metadata."""
    return {
        "window": 1000,
        "window_hash": "0x" + hashlib.sha256(b"window_1000").hexdigest(),
        "window_rand": "0x" + hashlib.sha256(b"rand_1000").hexdigest(),
    }


@pytest.fixture
def sample_miner_rollouts():
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
def sample_validation_metrics():
    """Sample validation metrics for multiple miners."""
    return {
        "miner_1": {"valid": 10, "checked": 12, "total": 20, "successful": 8, "unique": 10},
        "miner_2": {"valid": 8, "checked": 10, "total": 15, "successful": 6, "unique": 8},
        "miner_3": {"valid": 5, "checked": 8, "total": 10, "successful": 3, "unique": 5},
    }


@pytest.fixture
def sample_rollout_counters():
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
def sample_size_cases(request):
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

    from grail.shared.constants import MODEL_NAME

    # Load tokenizer for chat template application
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    from grail.shared.constants import MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return [tokenizer.encode(prompt, add_special_tokens=False) for prompt in sat_prompts]


@pytest.fixture(scope="session")
def tracker():
    """CopycatTracker instance for testing."""
    from grail.validation.copycat_service import CopycatTracker

    return CopycatTracker()
