#!/bin/bash
# Wrapper script to run parallel training in nohup mode
# This allows training to continue even after logout

set -e

DATASET=${1:-math}
EVAL_EVERY=${2:-40}
MODEL=${3:-Qwen/Qwen2.5-1.5B-Instruct}
NUM_ITERATIONS=${4:-1}

# W&B configuration (inherited from environment or defaults)
WANDB_PROJECT_VAL=${WANDB_PROJECT:-grail-lium-sweep}
WANDB_TAGS_VAL=${WANDB_TAGS:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p logs/parallel_training

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LAUNCHER_LOG="logs/parallel_training/launcher_${TIMESTAMP}.log"
PID_FILE="logs/parallel_training/launcher.pid"

echo "=================================================="
echo "PARALLEL TRAINING LAUNCHER (NOHUP MODE)"
echo "=================================================="
echo "Dataset: $DATASET"
echo "Eval every: $EVAL_EVERY steps"
echo "Model: $MODEL"
echo "Num Iterations: $NUM_ITERATIONS"
echo "W&B Project: $WANDB_PROJECT_VAL"
echo "W&B Tags: $WANDB_TAGS_VAL"
echo "Launcher log: $LAUNCHER_LOG"
echo "PID file: $PID_FILE"
echo "=================================================="

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Launcher already running with PID $OLD_PID"
        echo "   To stop it: kill $OLD_PID"
        echo "   Or wait for it to complete"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

# Run in nohup mode with W&B config passed as CLI args (takes precedence over env)
echo ""
echo "Starting parallel training in background..."
nohup python -u run_parallel_training.py \
    --dataset "$DATASET" \
    --eval-every "$EVAL_EVERY" \
    --model "$MODEL" \
    --num-iterations "$NUM_ITERATIONS" \
    --wandb-project "$WANDB_PROJECT_VAL" \
    --wandb-tags "$WANDB_TAGS_VAL" \
    > "$LAUNCHER_LOG" 2>&1 &

LAUNCHER_PID=$!
echo $LAUNCHER_PID > "$PID_FILE"

echo "✓ Launcher started with PID: $LAUNCHER_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f $LAUNCHER_LOG"
echo "  tail -f logs/parallel_training/training_instance*.log"
echo ""
echo "Check status:"
echo "  ps -p $LAUNCHER_PID"
echo "  nvidia-smi"
echo ""
echo "Stop training:"
echo "  kill $LAUNCHER_PID"
echo "  # Or use: ./stop_parallel_training.sh"
echo ""
echo "Training will continue even if you logout."
echo "Log out safely: exit"
