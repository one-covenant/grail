#!/bin/bash
# TRL GRPO Training Script
#
# This script runs GRPO training with TRL using the self-contained environment.
# Uses 1+1 GPU strategy: 1 GPU for VLLM, 1 GPU for training.
#
# Usage:
#   cd research/trl
#   ./run_algo.sh [dataset]  # dataset: math (default), gsm8k, or mbpp
#
# Environment variables:
#   VLLM_GPU: GPU for VLLM server (default: 0)
#   TRAIN_GPU: GPU for training (default: 1)
#   VLLM_PORT: Port for VLLM server (default: 8000)
#   MODEL_ID: Model to use (default: Qwen/Qwen2.5-1.5B-Instruct)

set -euo pipefail

# Configuration
DATASET="${1:-math}"
VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"

# Get script directory (research/trl)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=============================================="
echo "TRL GRPO Training"
echo "=============================================="
echo "Dataset:    $DATASET"
echo "Model:      $MODEL_ID"
echo "VLLM GPU:   $VLLM_GPU (port $VLLM_PORT)"
echo "Train GPU:  $TRAIN_GPU"
echo "Repo root:  $REPO_ROOT"
echo "=============================================="

# Ensure TRL environment is set up
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Setting up TRL environment..."
    (cd "$SCRIPT_DIR" && uv sync)
fi

# Ensure VLLM environment is set up
VLLM_DIR="$REPO_ROOT/tools/vllm-server"
if [ ! -d "$VLLM_DIR/.venv" ]; then
    echo "Setting up VLLM environment..."
    (cd "$VLLM_DIR" && uv sync)
fi

# Start VLLM server
echo ""
echo "Starting VLLM server on GPU $VLLM_GPU, port $VLLM_PORT..."
VLLM_TRL_BIN="$VLLM_DIR/.venv/bin/trl"
if [ ! -x "$VLLM_TRL_BIN" ]; then
    echo "ERROR: VLLM environment missing trl binary at $VLLM_TRL_BIN"
    echo "Try: (cd \"$VLLM_DIR\" && uv sync)"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$VLLM_GPU nohup "$VLLM_TRL_BIN" vllm-serve \
    --model "$MODEL_ID" \
    --tensor-parallel-size 1 \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.9 \
    > "$SCRIPT_DIR/vllm_server_${DATASET}.log" 2>&1 &

VLLM_PID=$!
echo "VLLM server started (PID: $VLLM_PID)"

# Wait for VLLM to be ready
echo "Waiting for VLLM server to be ready (60s)..."
sleep 60

# Check if VLLM is still running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: VLLM server died during startup"
    echo "Last 50 lines of log:"
    tail -50 "$SCRIPT_DIR/vllm_server_${DATASET}.log"
    exit 1
fi

echo "VLLM server is ready!"

# Start training
echo ""
echo "Starting training on GPU $TRAIN_GPU..."
TRL_PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$TRL_PYTHON_BIN" ]; then
    echo "ERROR: TRL environment missing python at $TRL_PYTHON_BIN"
    echo "Try: (cd \"$SCRIPT_DIR\" && uv sync)"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$TRAIN_GPU "$TRL_PYTHON_BIN" train_trl_grpo.py \
    --dataset "$DATASET" \
    2>&1 | tee "train_${DATASET}.log"

TRAIN_EXIT=${PIPESTATUS[0]}

# Cleanup
echo ""
echo "Cleaning up VLLM server..."
kill $VLLM_PID 2>/dev/null || true

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $TRAIN_EXIT"
fi

exit $TRAIN_EXIT
