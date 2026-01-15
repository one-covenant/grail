#!/bin/bash
# Check status of parallel training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="logs/parallel_training/launcher.pid"

echo "=================================================="
echo "PARALLEL TRAINING STATUS"
echo "=================================================="

# Check launcher
if [ -f "$PID_FILE" ]; then
    LAUNCHER_PID=$(cat "$PID_FILE")
    if ps -p "$LAUNCHER_PID" > /dev/null 2>&1; then
        echo "✓ Launcher running (PID: $LAUNCHER_PID)"
    else
        echo "✗ Launcher not running (stale PID: $LAUNCHER_PID)"
    fi
else
    echo "✗ No launcher PID file found"
fi

echo ""

# Check VLLM processes
VLLM_COUNT=$(pgrep -f "vllm.entrypoints.openai.api_server" | wc -l)
echo "VLLM servers: $VLLM_COUNT running"
if [ $VLLM_COUNT -gt 0 ]; then
    echo "  PIDs: $(pgrep -f 'vllm.entrypoints.openai.api_server' | tr '\n' ' ')"
fi

# Check training processes
TRAIN_COUNT=$(pgrep -f "train_trl_grpo.py" | wc -l)
echo "Training processes: $TRAIN_COUNT running"
if [ $TRAIN_COUNT -gt 0 ]; then
    echo "  PIDs: $(pgrep -f 'train_trl_grpo.py' | tr '\n' ' ')"
fi

echo ""

# Check ports
echo "Port status:"
for port in 8000 8001 8002 8003; do
    if lsof -i :$port > /dev/null 2>&1; then
        PID=$(lsof -ti :$port)
        echo "  Port $port: OPEN (PID: $PID)"
    else
        echo "  Port $port: CLOSED"
    fi
done

echo ""

# Check recent logs
echo "Recent launcher log (last 10 lines):"
LATEST_LOG=$(ls -t logs/parallel_training/launcher_*.log 2>/dev/null | head -1)
if [ ! -z "$LATEST_LOG" ]; then
    echo "  File: $LATEST_LOG"
    tail -10 "$LATEST_LOG" | sed 's/^/  /'
else
    echo "  No launcher logs found"
fi

echo ""
echo "=================================================="
echo "Detailed logs:"
echo "  Launcher: tail -f $LATEST_LOG"
echo "  Training: tail -f logs/parallel_training/training_instance*.log"
echo "  VLLM:     tail -f logs/parallel_training/vllm_instance*.log"
echo ""
echo "GPU status: nvidia-smi"
echo "Stop all:   ./stop_parallel_training.sh"
echo "=================================================="
