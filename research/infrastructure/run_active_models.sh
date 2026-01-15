#!/bin/bash
# run_active_models.sh
# Runs deploy_parallel.py with active_models config in nohup mode
#
# This deploys and runs all 6 active model configurations:
#   - qwen2.5-0.5b-iter1
#   - qwen2.5-1.5b-iter1
#   - llama3.2-1b-iter1
#   - llama3.2-3b-iter1
#   - gemma3-1b-iter1
#   - gemma3-4b-iter1
#
# Usage:
#   ./run_active_models.sh           # Run in foreground
#   nohup ./run_active_models.sh &   # Run in background with nohup
#
# Each model runs on its own 8xA100 pod with 4 seeds in parallel.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/deploy_runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/active_models_${TIMESTAMP}.log"
STATE_FILE="${SCRIPT_DIR}/.lium_state_active_models.json"

echo "=========================================="
echo "Starting active_models deployment"
echo "Timestamp: ${TIMESTAMP}"
echo "Log file: ${LOG_FILE}"
echo "State file: ${STATE_FILE}"
echo "=========================================="
echo ""
echo "Models to deploy:"
echo "  - qwen2.5-0.5b-iter1"
echo "  - qwen2.5-1.5b-iter1"
echo "  - llama3.2-1b-iter1"
echo "  - llama3.2-3b-iter1"
echo "  - gemma3-1b-iter1"
echo "  - gemma3-4b-iter1"
echo ""

# Activate venv and run deployment with unbuffered output for nohup visibility
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python -u deploy_parallel.py \
    --config active_models \
    --state-file "$STATE_FILE" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Deployment completed successfully"
else
    echo "✗ Deployment failed with exit code: $EXIT_CODE"
fi
echo "Log saved to: ${LOG_FILE}"
echo "=========================================="

exit $EXIT_CODE
