"""Unit tests for CopycatService component.

Tests focus on:
- Interval reset behavior
- Rollout filtering logic
- Service initialization

Note: CopycatService is a thin wrapper around COPYCAT_TRACKER singleton.
Most complex logic (detection, gating) is tested in test_copycat_detection.py.

Uses pytest best practices:
- Minimal, focused tests
- Tests public API only
- Clear test names
"""


from grail.validation.copycat import COPYCAT_TRACKER
from grail.validation.copycat_service import COPYCAT_SERVICE, CopycatService


class TestServiceInitialization:
    """Test that service initializes correctly."""

    def test_creates_service_instance(self):
        """Given no arguments, service should initialize successfully."""
        service = CopycatService()
        assert service is not None

    def test_singleton_exists(self):
        """Given global singleton, it should be accessible."""
        assert COPYCAT_SERVICE is not None
        assert isinstance(COPYCAT_SERVICE, CopycatService)


class TestIntervalManagement:
    """Test interval reset behavior."""

    def test_reset_interval_callable(self):
        """Given interval ID, reset should be callable without error."""
        service = CopycatService()

        # Should not raise
        service.reset_interval(interval_id=42)

        # Tracker should be updated
        assert COPYCAT_TRACKER.current_interval_id == 42

    def test_reset_different_intervals(self):
        """Given multiple resets, should update each time."""
        service = CopycatService()

        service.reset_interval(interval_id=1)
        assert COPYCAT_TRACKER.current_interval_id == 1

        service.reset_interval(interval_id=2)
        assert COPYCAT_TRACKER.current_interval_id == 2

        service.reset_interval(interval_id=10)
        assert COPYCAT_TRACKER.current_interval_id == 10


class TestRolloutFiltering:
    """Test rollout filtering logic."""

    def test_filters_cheater_rollouts(self):
        """Given rollouts and cheaters, should filter correctly."""
        service = CopycatService()

        rollouts = [
            {"hotkey": "miner_1", "data": "a"},
            {"hotkey": "miner_2", "data": "b"},
            {"hotkey": "miner_3", "data": "c"},
            {"hotkey": "miner_1", "data": "d"},
        ]

        cheaters = {"miner_1", "miner_3"}

        filtered = service.filter_cheater_rollouts(rollouts, cheaters)

        # Only miner_2's rollouts should remain
        assert len(filtered) == 1
        assert filtered[0]["hotkey"] == "miner_2"

    def test_empty_cheaters_returns_all(self):
        """Given no cheaters, all rollouts should be returned."""
        service = CopycatService()

        rollouts = [
            {"hotkey": "miner_1", "data": "a"},
            {"hotkey": "miner_2", "data": "b"},
        ]

        filtered = service.filter_cheater_rollouts(rollouts, set())

        assert filtered == rollouts

    def test_empty_rollouts_returns_empty(self):
        """Given empty rollouts, should return empty."""
        service = CopycatService()

        filtered = service.filter_cheater_rollouts([], {"miner_1"})

        assert filtered == []

    def test_all_cheaters_returns_empty(self):
        """Given all rollouts from cheaters, should return empty list."""
        service = CopycatService()

        rollouts = [
            {"hotkey": "miner_1", "data": "a"},
            {"hotkey": "miner_1", "data": "b"},
        ]

        filtered = service.filter_cheater_rollouts(rollouts, {"miner_1"})

        assert filtered == []

    def test_preserves_rollout_structure(self):
        """Given complex rollouts, should preserve structure."""
        service = CopycatService()

        rollouts = [
            {
                "hotkey": "keeper",
                "data": "complex",
                "nested": {"key": "value"},
                "list": [1, 2, 3],
            },
            {"hotkey": "cheater", "simple": True},
        ]

        filtered = service.filter_cheater_rollouts(rollouts, {"cheater"})

        assert len(filtered) == 1
        assert filtered[0]["hotkey"] == "keeper"
        assert filtered[0]["nested"]["key"] == "value"
        assert filtered[0]["list"] == [1, 2, 3]


class TestServiceIntegration:
    """Test service integration with global tracker."""

    def test_uses_global_tracker(self):
        """Given service, it should use the global COPYCAT_TRACKER."""
        service = CopycatService()

        # Service and singleton should interact with same tracker
        service.reset_interval(interval_id=99)
        assert COPYCAT_TRACKER.current_interval_id == 99

        # Using singleton should affect same state
        COPYCAT_SERVICE.reset_interval(interval_id=100)
        assert COPYCAT_TRACKER.current_interval_id == 100

    def test_multiple_instances_share_state(self):
        """Given multiple service instances, they share global tracker."""
        service1 = CopycatService()
        service2 = CopycatService()

        service1.reset_interval(interval_id=5)
        assert COPYCAT_TRACKER.current_interval_id == 5

        # service2 should see the same state
        service2.reset_interval(interval_id=6)
        assert COPYCAT_TRACKER.current_interval_id == 6
