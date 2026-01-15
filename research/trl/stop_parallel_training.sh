#!/bin/bash
# Stop all parallel training processes gracefully

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="logs/parallel_training/launcher.pid"

echo "=================================================="
echo "STOPPING PARALLEL TRAINING"
echo "=================================================="

# Check PID file
if [ -f "$PID_FILE" ]; then
    LAUNCHER_PID=$(cat "$PID_FILE")

    if ps -p "$LAUNCHER_PID" > /dev/null 2>&1; then
        echo "Found launcher process (PID: $LAUNCHER_PID)"
        echo "Sending SIGTERM for graceful shutdown..."
        kill -TERM "$LAUNCHER_PID"

        echo "Waiting for shutdown (max 30s)..."
        for i in {1..30}; do
            if ! ps -p "$LAUNCHER_PID" > /dev/null 2>&1; then
                echo "✓ Launcher stopped gracefully"
                rm -f "$PID_FILE"
                break
            fi
            sleep 1
            echo -n "."
        done
        echo ""

        # Force kill if still running
        if ps -p "$LAUNCHER_PID" > /dev/null 2>&1; then
            echo "⚠️  Launcher still running, force killing..."
            kill -KILL "$LAUNCHER_PID" 2>/dev/null || true
            rm -f "$PID_FILE"
        fi
    else
        echo "Launcher PID $LAUNCHER_PID not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found at $PID_FILE"
fi

# Clean up any stray processes
echo ""
echo "Checking for stray VLLM/training processes..."

VLLM_PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server" || true)
TRAIN_PIDS=$(pgrep -f "train_trl_grpo.py" || true)

if [ ! -z "$VLLM_PIDS" ]; then
    echo "Found VLLM processes: $VLLM_PIDS"
    echo "Stopping..."
    pkill -TERM -f "vllm.entrypoints.openai.api_server" || true
    sleep 2
    pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
fi

if [ ! -z "$TRAIN_PIDS" ]; then
    echo "Found training processes: $TRAIN_PIDS"
    echo "Stopping..."
    pkill -TERM -f "train_trl_grpo.py" || true
    sleep 2
    pkill -KILL -f "train_trl_grpo.py" 2>/dev/null || true
fi

echo ""
echo "✓ All processes stopped"
echo ""
echo "Verify with:"
echo "  pgrep -f vllm"
echo "  pgrep -f train_trl_grpo"
echo "  nvidia-smi"
