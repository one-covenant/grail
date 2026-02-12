"""Integration tests for ValidationService.

Tests focus on core functionality:
- Rolling history management (miner selection tracking)
- Service lifecycle (cleanup)

These tests verify behavior, not just component existence.
"""

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock

from grail.infrastructure.checkpoint_consumer import CheckpointManager
from grail.scoring import WeightComputer
from grail.validation import ValidationService, create_env_validation_pipeline


@pytest.fixture
def validation_service(
    mock_wallet: "MagicMock", mock_credentials: "MagicMock", mock_monitor: "AsyncMock"
) -> ValidationService:
    """Create a ValidationService instance for testing."""
    validation_pipeline = create_env_validation_pipeline()
    weight_computer = WeightComputer(
        rolling_windows=12,
        window_length=20,
        superlinear_exponent=1.5,
        burn_uid=0,
        burn_percentage=0.05,
        cap_enabled=True,
    )
    checkpoint_manager = CheckpointManager(
        cache_root=Path("/tmp/test_checkpoints"),
        credentials=mock_credentials,
        keep_limit=3,
    )

    return ValidationService(
        wallet=mock_wallet,
        netuid=42,
        validation_pipeline=validation_pipeline,
        weight_computer=weight_computer,
        credentials=mock_credentials,
        checkpoint_manager=checkpoint_manager,
        monitor=mock_monitor,
    )


@pytest.mark.integration
class TestRollingHistoryManagement:
    """Test rolling history tracking for miner selection and inference counts."""

    def test_updates_counts_on_first_window(self, validation_service: ValidationService) -> None:
        """Given first window, should add all miners to counts."""
        history = deque(maxlen=3)
        counts = {}

        validation_service._update_rolling(history, counts, {"miner_1", "miner_2"})

        assert counts == {"miner_1": 1, "miner_2": 1}
        assert len(history) == 1
        assert history[0] == {"miner_1", "miner_2"}

    def test_accumulates_counts_across_windows(self, validation_service: ValidationService) -> None:
        """Given multiple windows, should accumulate counts correctly."""
        history = deque(maxlen=3)
        counts = {}

        validation_service._update_rolling(history, counts, {"miner_1", "miner_2"})
        validation_service._update_rolling(history, counts, {"miner_2", "miner_3"})

        assert counts == {"miner_1": 1, "miner_2": 2, "miner_3": 1}
        assert len(history) == 2

    def test_evicts_oldest_window_when_maxlen_exceeded(
        self, validation_service: ValidationService
    ) -> None:
        """Given maxlen windows, should evict oldest and adjust counts."""
        history = deque(maxlen=3)
        counts = {}

        # Fill to maxlen
        validation_service._update_rolling(history, counts, {"miner_1", "miner_2"})
        validation_service._update_rolling(history, counts, {"miner_2", "miner_3"})
        validation_service._update_rolling(history, counts, {"miner_1"})

        assert len(history) == 3
        assert counts == {"miner_1": 2, "miner_2": 2, "miner_3": 1}

        # Add fourth window - should evict oldest (miner_1, miner_2)
        validation_service._update_rolling(history, counts, {"miner_4"})

        assert len(history) == 3
        # Oldest window (miner_1, miner_2) removed
        assert counts["miner_1"] == 1  # Was 2, decreased by 1
        assert counts["miner_2"] == 1  # Was 2, decreased by 1
        assert counts["miner_3"] == 1  # Unchanged
        assert counts["miner_4"] == 1  # New entry

    def test_handles_empty_window_gracefully(self, validation_service: ValidationService) -> None:
        """Given empty miner set, should not crash."""
        history = deque(maxlen=3)
        counts = {}

        validation_service._update_rolling(history, counts, set())

        assert len(history) == 1
        assert counts == {}

    def test_handles_overlapping_miners_correctly(
        self, validation_service: ValidationService
    ) -> None:
        """Given overlapping miner sets, should maintain correct counts."""
        history = deque(maxlen=2)
        counts = {}

        validation_service._update_rolling(history, counts, {"miner_1", "miner_2", "miner_3"})
        validation_service._update_rolling(history, counts, {"miner_2", "miner_3", "miner_4"})

        assert counts == {"miner_1": 1, "miner_2": 2, "miner_3": 2, "miner_4": 1}

        # Evict first window - should decrease miner_1, miner_2, miner_3 counts
        validation_service._update_rolling(history, counts, {"miner_5"})

        # miner_1 count goes to 0 (still in dict, but zeroed)
        assert counts.get("miner_1", 0) == 0  # Removed from first window
        assert counts["miner_2"] == 1  # Decreased from 2 to 1
        assert counts["miner_3"] == 1  # Decreased from 2 to 1
        assert counts["miner_4"] == 1  # Unchanged
        assert counts["miner_5"] == 1  # New


@pytest.mark.integration
class TestServiceLifecycle:
    """Test service lifecycle management."""

    def test_cleanup_stops_chain_manager(self, validation_service: ValidationService) -> None:
        """Given cleanup call, should stop chain manager worker process."""
        from unittest.mock import MagicMock

        mock_chain = MagicMock()
        validation_service._chain_manager = mock_chain

        validation_service.cleanup()

        # Should call stop on chain manager
        mock_chain.stop.assert_called_once()

    def test_cleanup_handles_missing_chain_manager(
        self, validation_service: ValidationService
    ) -> None:
        """Given no chain manager, cleanup should not crash."""
        validation_service._chain_manager = None

        # Should not raise
        validation_service.cleanup()
