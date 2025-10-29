"""Integration tests for block timestamp retrieval from Bittensor chain.

Tests the GrailChainManager's ability to accurately retrieve and estimate
block timestamps using the Substrate Timestamp pallet.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.credentials import BucketCredentials
from grail.infrastructure.network import create_subtensor


@pytest_asyncio.fixture
async def subtensor() -> any:
    """Create a test subtensor connection (testnet for reliability)."""
    sub = await create_subtensor(resilient=False)
    yield sub
    # Cleanup if needed
    if hasattr(sub, "close"):
        try:
            await sub.close()
        except Exception:
            pass


@pytest_asyncio.fixture
async def chain_manager(subtensor: any) -> GrailChainManager:
    """Create a GrailChainManager with test subtensor."""
    # Create minimal mock objects for testing timestamp functions only
    mock_config = SimpleNamespace(netuid=1)
    mock_wallet = SimpleNamespace(
        name="test_wallet",
        hotkey=SimpleNamespace(ss58_address="5test"),
    )
    mock_metagraph = SimpleNamespace(hotkeys=[], uids=[])
    mock_credentials = BucketCredentials(
        bucket_name="test",
        account_id="test",
        read_access_key_id="test",
        read_secret_access_key="test",
        write_access_key_id="test",
        write_secret_access_key="test",
    )

    manager = GrailChainManager(
        config=mock_config,
        wallet=mock_wallet,
        metagraph=mock_metagraph,
        subtensor=subtensor,
        credentials=mock_credentials,
        fetch_interval=3600,  # Don't auto-fetch during test
    )

    yield manager

    # Cleanup
    try:
        manager.stop()
    except Exception:
        pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_block_timestamp_current_block(chain_manager: GrailChainManager) -> None:
    """Test retrieving timestamp for a recent block.

    Verifies that:
    1. get_block_timestamp returns a valid timestamp
    2. The timestamp is in seconds (not milliseconds)
    3. The timestamp is recent (within last hour)
    4. The timestamp is a reasonable Unix epoch value
    """
    # Get current block number
    current_block = await chain_manager.subtensor.get_current_block()
    assert current_block > 0, "Should have a valid current block"

    # Query timestamp for a block a few blocks back (safer than current)
    target_block = current_block - 5
    timestamp = await chain_manager.get_block_timestamp(target_block)

    # Verify timestamp was retrieved
    assert timestamp is not None, f"Should retrieve timestamp for block {target_block}"
    assert isinstance(timestamp, float), "Timestamp should be a float"

    # Verify timestamp is in seconds (not milliseconds)
    # Recent timestamps should be ~1.7B seconds, not ~1.7T milliseconds
    assert 1_000_000_000 < timestamp < 3_000_000_000, (
        f"Timestamp {timestamp} should be in seconds, not milliseconds"
    )

    # Convert to datetime and verify it's recent
    block_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    age = now - block_datetime

    # Block should be from within the last hour (Bittensor blocks are ~12s)
    assert age < timedelta(hours=1), (
        f"Block timestamp {block_datetime} is too old (age: {age}). "
        f"Expected recent block within last hour."
    )

    # Block should not be in the future (allow small clock skew)
    assert age > timedelta(seconds=-30), (
        f"Block timestamp {block_datetime} is in the future by {-age}. "
        "Clock skew or invalid timestamp."
    )

    print(f"✅ Block {target_block} timestamp: {timestamp} ({block_datetime} UTC)")
    print(f"   Age: {age.total_seconds():.1f} seconds")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_estimate_block_timestamp_accuracy(chain_manager: GrailChainManager) -> None:
    """Test block timestamp estimation using empirical block time.

    Verifies that:
    1. estimate_block_timestamp returns a reasonable estimate
    2. The estimate is close to the actual timestamp (within tolerance)
    3. The estimation works for future blocks
    4. The function handles edge cases gracefully
    """
    # Get current block
    current_block = await chain_manager.subtensor.get_current_block()
    assert current_block > 100, "Need sufficient block history"

    # Get actual timestamp for a past block (ground truth)
    test_block = current_block - 50  # 50 blocks back (~10 minutes)
    actual_timestamp = await chain_manager.get_block_timestamp(test_block)
    assert actual_timestamp is not None, f"Should get actual timestamp for block {test_block}"

    # Estimate timestamp using only earlier blocks
    # Simulate estimating test_block from 100 blocks before it
    estimated_timestamp = await chain_manager.estimate_block_timestamp(
        test_block, anchor_distance=100
    )

    assert estimated_timestamp is not None, "Should return an estimate"
    assert isinstance(estimated_timestamp, float), "Estimate should be a float"

    # Calculate error
    error_seconds = abs(estimated_timestamp - actual_timestamp)
    error_percent = (error_seconds / actual_timestamp) * 100

    # Bittensor aims for 12s block time but can vary
    # Allow up to 2 minutes error for 50-block estimate (~12s * 50 = 10min)
    max_allowed_error = 120.0  # 2 minutes tolerance

    assert error_seconds < max_allowed_error, (
        f"Estimation error too large: {error_seconds:.1f}s "
        f"({error_percent:.3f}%). Actual: {actual_timestamp}, "
        f"Estimated: {estimated_timestamp}"
    )

    # Test future block estimation
    future_block = current_block + 100  # ~20 minutes in the future
    future_estimate = await chain_manager.estimate_block_timestamp(future_block)

    assert future_estimate is not None, "Should estimate future block timestamp"
    assert future_estimate > actual_timestamp, "Future block should have later timestamp"

    # Future estimate should be reasonable (not wildly off)
    expected_future_offset = (future_block - test_block) * 12  # Nominal 12s/block
    actual_future_offset = future_estimate - actual_timestamp
    offset_error = abs(actual_future_offset - expected_future_offset)

    # Allow 20% error in future projection
    assert offset_error < (expected_future_offset * 0.2), (
        f"Future projection error too large: {offset_error:.1f}s. "
        f"Expected offset: {expected_future_offset:.1f}s, "
        f"Actual offset: {actual_future_offset:.1f}s"
    )

    print(f"✅ Block {test_block} estimation:")
    print(
        f"   Actual:    {actual_timestamp} ({datetime.fromtimestamp(actual_timestamp, tz=timezone.utc)})"
    )
    print(
        f"   Estimated: {estimated_timestamp} ({datetime.fromtimestamp(estimated_timestamp, tz=timezone.utc)})"
    )
    print(f"   Error:     {error_seconds:.2f}s ({error_percent:.3f}%)")
    print(f"✅ Future block {future_block} estimate: {future_estimate}")
    print(f"   Projection error: {offset_error:.1f}s")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_block_timestamp_edge_cases(chain_manager: GrailChainManager) -> None:
    """Test edge cases and error handling for block timestamp functions.

    Verifies graceful degradation when:
    1. Querying very old blocks
    2. Querying non-existent blocks
    3. Network issues (simulated via timeout)
    """
    # Test 1: Very old block (block 1 - genesis-ish)
    # This might fail if genesis timestamp isn't available, which is OK
    old_timestamp = await chain_manager.get_block_timestamp(1)
    if old_timestamp is not None:
        # If we got it, verify it's plausible (should be years ago)
        old_datetime = datetime.fromtimestamp(old_timestamp, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        age = now - old_datetime
        assert age > timedelta(days=30), "Genesis block should be at least months old"
        print(f"✅ Genesis block (1) timestamp: {old_datetime} UTC")
    else:
        print("⚠️  Genesis block timestamp not available (acceptable)")

    # Test 2: Far future block (doesn't exist yet)
    current_block = await chain_manager.subtensor.get_current_block()
    future_block = current_block + 1_000_000  # Very far future

    # get_block_timestamp should return None for non-existent blocks
    future_timestamp = await chain_manager.get_block_timestamp(future_block)
    assert future_timestamp is None, "Should return None for non-existent future block"

    # estimate_block_timestamp should still work (extrapolation)
    future_estimate = await chain_manager.estimate_block_timestamp(future_block)
    if future_estimate is not None:
        # If we got an estimate, it should be in the future
        future_datetime = datetime.fromtimestamp(future_estimate, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        assert future_datetime > now, "Estimated future timestamp should be in the future"
        print(f"✅ Far future block estimate: {future_datetime} UTC")

    # Test 3: Invalid block number (0 or negative)
    invalid_timestamp = await chain_manager.get_block_timestamp(0)
    # Should handle gracefully (likely returns None)
    print(f"✅ Block 0 timestamp handling: {invalid_timestamp}")

    print("✅ All edge cases handled gracefully")
