#!/bin/bash
# Quick script to test block timestamp functions against Bittensor testnet
#
# Usage:
#   ./scripts/test_block_timestamps.sh

set -e

cd "$(dirname "$0")/.."

echo "=================================================="
echo "Testing Block Timestamp Retrieval Functions"
echo "=================================================="
echo ""
echo "These tests verify that GrailChainManager can:"
echo "  1. Retrieve exact block timestamps from Substrate"
echo "  2. Estimate timestamps using empirical block times"
echo "  3. Handle edge cases gracefully"
echo ""
echo "Connecting to Bittensor testnet..."
echo ""

# Run the integration tests
uv run pytest \
    tests/integration/infrastructure/test_block_timestamp.py \
    -v \
    --tb=short \
    -s \
    "$@"

echo ""
echo "=================================================="
echo "✅ Block timestamp tests complete!"
echo "=================================================="

