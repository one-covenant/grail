"""Tests for window timing utilities."""

import time
from datetime import timedelta
from unittest import mock

from grail.shared.window_utils import (
    WindowWaitTracker,
    calculate_blocks_to_window,
    calculate_next_window,
    estimate_time_to_window,
    format_duration,
    log_window_wait_initial,
    log_window_wait_periodic,
)


class TestCalculateBlocksToWindow:
    """Test block calculation logic."""

    def test_blocks_to_future_window(self) -> None:
        """Test calculation when window is in the future."""
        blocks = calculate_blocks_to_window(100, 150)
        assert blocks == 50

    def test_blocks_to_past_window(self) -> None:
        """Test calculation when window is in the past."""
        blocks = calculate_blocks_to_window(150, 100)
        assert blocks == 0

    def test_blocks_at_exact_window(self) -> None:
        """Test calculation when at the exact window block."""
        blocks = calculate_blocks_to_window(100, 100)
        assert blocks == 0


class TestEstimateTimeToWindow:
    """Test time estimation."""

    def test_estimate_with_default_block_time(self) -> None:
        """Test estimation with default 12s per block."""
        duration, eta = estimate_time_to_window(60)  # 60 blocks * 12s = 720s = 12min
        assert duration == timedelta(seconds=720)
        assert isinstance(eta.timestamp(), float)

    def test_estimate_with_custom_block_time(self) -> None:
        """Test estimation with custom block time."""
        duration, eta = estimate_time_to_window(30, secs_per_block=10)
        assert duration == timedelta(seconds=300)

    def test_estimate_zero_blocks(self) -> None:
        """Test estimation when no blocks to wait."""
        duration, eta = estimate_time_to_window(0)
        assert duration == timedelta(seconds=0)


class TestCalculateNextWindow:
    """Test next window calculation."""

    def test_initial_state(self) -> None:
        """Test calculation when never processed (last = -1)."""
        next_win = calculate_next_window(-1, 100)
        assert next_win == 0

    def test_first_window_processed(self) -> None:
        """Test calculation after processing window 0."""
        next_win = calculate_next_window(0, 100)
        assert next_win == 100

    def test_later_window(self) -> None:
        """Test calculation for later windows."""
        next_win = calculate_next_window(500, 100)
        assert next_win == 600


class TestFormatDuration:
    """Test duration formatting."""

    def test_format_simple_duration(self) -> None:
        """Test formatting a simple duration."""
        duration = timedelta(hours=1, minutes=30, seconds=45)
        assert format_duration(duration) == "1:30:45"

    def test_format_less_than_minute(self) -> None:
        """Test formatting duration under a minute."""
        duration = timedelta(seconds=30)
        assert format_duration(duration) == "0:00:30"

    def test_format_many_hours(self) -> None:
        """Test formatting with many hours."""
        duration = timedelta(hours=10, minutes=5, seconds=0)
        assert format_duration(duration) == "10:05:00"


class TestWindowWaitTracker:
    """Test window wait tracking state machine."""

    def test_initial_state(self) -> None:
        """Test initial state - should log on first call."""
        tracker = WindowWaitTracker()
        assert tracker.should_log_initial() is True
        # get_elapsed_seconds returns int, so may be 0 if very fast
        assert tracker.get_elapsed_seconds() >= 0
        assert tracker.get_elapsed_seconds() < 2  # Should be very fast

    def test_no_periodic_log_immediately(self) -> None:
        """Test that periodic log doesn't trigger immediately."""
        tracker = WindowWaitTracker(log_interval_secs=120)
        assert tracker.should_log_initial() is True
        assert tracker.should_log_periodic() is False

    def test_periodic_log_after_interval(self) -> None:
        """Test that periodic log triggers after interval."""
        tracker = WindowWaitTracker(log_interval_secs=1)  # 1 second interval
        assert tracker.should_log_initial() is True
        time.sleep(1.1)  # Wait slightly more than interval
        assert tracker.should_log_periodic() is True

    def test_elapsed_seconds_increases(self) -> None:
        """Test that elapsed seconds increases over time."""
        tracker = WindowWaitTracker()
        tracker.should_log_initial()
        time.sleep(0.1)  # Sleep first to allow measurable time to pass
        elapsed1 = tracker.get_elapsed_seconds()
        time.sleep(0.5)
        elapsed2 = tracker.get_elapsed_seconds()
        assert elapsed2 >= elapsed1  # Should be non-decreasing

    def test_reset_clears_state(self) -> None:
        """Test that reset clears state."""
        tracker = WindowWaitTracker()
        tracker.should_log_initial()
        tracker.reset()
        assert tracker.should_log_initial() is True  # Should return True again


class TestWindowWaitLogging:
    """Test window wait logging functions."""

    @mock.patch("grail.shared.window_utils.logger")
    def test_log_window_wait_initial(self, mock_logger: mock.Mock) -> None:
        """Test initial window wait logging."""
        log_window_wait_initial(
            current_block=950,
            last_processed_window=900,
            window_length=100,
        )
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        # Should show next_window = 900 + 100 = 1000
        assert "1000" in str(call_args)
        assert "900" in str(call_args)
        assert "950" in str(call_args)

    @mock.patch("grail.shared.window_utils.logger")
    def test_log_window_wait_periodic(self, mock_logger: mock.Mock) -> None:
        """Test periodic window wait logging."""
        log_window_wait_periodic(next_window=1000, elapsed_seconds=120)
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "1000" in str(call_args)
        assert "0:02:00" in str(call_args)  # 120 seconds = 0:02:00
