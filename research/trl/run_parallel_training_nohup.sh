#!/bin/bash
# Wrapper script to run parallel training in nohup mode
# This allows training to continue even after logout

set -e

DATASET=${1:-math}
EVAL_EVERY=${2:-40}
MODEL=${3:-Qwen/Qwen2.5-1.5B-Instruct}
NUM_ITERATIONS=${4:-1}
NUM_INSTANCES=${5:-1}  # Default to 1 instance for stability (avoids NCCL conflicts)
BATCH_SIZE=${6:-}  # Optional: batch size per device
GRAD_ACCUM_STEPS=${7:-}  # Optional: gradient accumulation steps
FP32_MASTER_WEIGHTS=${GRAIL_FP32_MASTER_WEIGHTS:-}  # Optional: use FP32 master weights

# W&B configuration (inherited from environment or defaults)
WANDB_PROJECT_VAL=${WANDB_PROJECT:-grail-lium-sweep}
WANDB_TAGS_VAL=${WANDB_TAGS:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Create logs directory
mkdir -p logs/parallel_training

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Use GRAIL_RUN_PREFIX for unique launcher logs and PID files when running multiple experiments
RUN_PREFIX_LABEL=${GRAIL_RUN_PREFIX:-default}
LAUNCHER_LOG="logs/parallel_training/launcher_${RUN_PREFIX_LABEL}_${TIMESTAMP}.log"
PID_FILE="logs/parallel_training/launcher_${RUN_PREFIX_LABEL}.pid"

echo "=================================================="
echo "PARALLEL TRAINING LAUNCHER (NOHUP MODE)"
echo "=================================================="
echo "Dataset: $DATASET"
echo "Eval every: $EVAL_EVERY steps"
echo "Model: $MODEL"
echo "Num Iterations: $NUM_ITERATIONS"
echo "Num Instances: $NUM_INSTANCES"
echo "Batch Size: ${BATCH_SIZE:-default}"
echo "Grad Accum Steps: ${GRAD_ACCUM_STEPS:-default}"
echo "FP32 Master Weights: ${FP32_MASTER_WEIGHTS:-false}"
echo "W&B Project: $WANDB_PROJECT_VAL"
echo "W&B Tags: $WANDB_TAGS_VAL"
echo "Run Prefix: $RUN_PREFIX_LABEL"
echo "Start Instance: ${GRAIL_START_INSTANCE:-0}"
echo "Seed Override: ${GRAIL_SEED:-default}"
echo "Base Ports: vLLM=${GRAIL_BASE_PORT:-8000}, Group=${GRAIL_BASE_GROUP_PORT:-51200}"
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

# Build command with optional parameters
CMD="python -u run_parallel_training.py \
    --dataset $DATASET \
    --eval-every $EVAL_EVERY \
    --model $MODEL \
    --num-iterations $NUM_ITERATIONS \
    --num-instances $NUM_INSTANCES"

# Add W&B project if provided
if [ -n "$WANDB_PROJECT_VAL" ]; then
    CMD="$CMD --wandb-project $WANDB_PROJECT_VAL"
fi

# Add W&B tags if provided
if [ -n "$WANDB_TAGS_VAL" ]; then
    CMD="$CMD --wandb-tags $WANDB_TAGS_VAL"
fi

# Add optional batch size if provided
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

# Add optional grad accum steps if provided
if [ -n "$GRAD_ACCUM_STEPS" ]; then
    CMD="$CMD --grad-accum-steps $GRAD_ACCUM_STEPS"
fi

# Add seed override if provided (GRAIL_SEED env var)
if [ -n "$GRAIL_SEED" ]; then
    CMD="$CMD --seed $GRAIL_SEED"
fi

# Add learning rate override if provided (GRAIL_TRAINER_LR env var)
if [ -n "$GRAIL_TRAINER_LR" ]; then
    CMD="$CMD --lr $GRAIL_TRAINER_LR"
fi

# Add FP32 master weights flag if enabled
if [ -n "$FP32_MASTER_WEIGHTS" ] && [ "$FP32_MASTER_WEIGHTS" = "true" ]; then
    CMD="$CMD --fp32-master-weights"
fi

nohup $CMD > "$LAUNCHER_LOG" 2>&1 &


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
