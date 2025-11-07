#!/bin/bash
# Monitor offline training progress

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/training.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Training log not found: $LOG_FILE"
    exit 1
fi

echo "================================"
echo "  Training Progress Monitor"
echo "================================"
echo ""

# Check if process is running
if [ -f "$SCRIPT_DIR/training.pid" ]; then
    PID=$(cat "$SCRIPT_DIR/training.pid")
    if kill -0 $PID 2>/dev/null; then
        echo "✓ Training process running (PID: $PID)"
    else
        echo "✗ Training process not running (PID: $PID exited)"
    fi
else
    echo "? No PID file found"
fi

echo ""
echo "Recent iterations:"
grep -E "Starting iteration|iteration.*/" "$LOG_FILE" | tail -5

echo ""
echo "Recent training metrics:"
grep -E "Training epoch completed|loss_total|reward_mean" "$LOG_FILE" | tail -10

echo ""
echo "Evaluation results:"
grep -E "Evaluation complete|pass@|mean@" "$LOG_FILE" | tail -10

echo ""
echo "Recent errors/warnings:"
grep -E "ERROR|torch.OutOfMemoryError" "$LOG_FILE" | tail -5

echo ""
echo "================================"
echo "Live tail (Ctrl+C to exit):"
echo "================================"
tail -f "$LOG_FILE"



