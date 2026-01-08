#!/bin/bash
# Test script to verify VLLM + training connection on a single instance
# This helps debug issues before running all 4 parallel instances

set -e

DATASET=${1:-gsm8k}
VLLM_GPU=0
TRAINING_GPU=1
PORT=8000
SEED=42

echo "=================================================="
echo "Testing single instance setup"
echo "=================================================="
echo "VLLM GPU: $VLLM_GPU"
echo "Training GPU: $TRAINING_GPU"
echo "Port: $PORT"
echo "Seed: $SEED"
echo "Dataset: $DATASET"
echo "=================================================="

# Create log directory
mkdir -p logs/test_instance

# Start TRL's vLLM server (exposes /get_world_size/ and /init_communicator/ endpoints)
echo ""
echo "Starting TRL vLLM server on GPU $VLLM_GPU, port $PORT..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VISIBLE_DEVICES=$VLLM_GPU "$SCRIPT_DIR/.venv/bin/trl" vllm-serve \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port $PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    > logs/test_instance/vllm.log 2>&1 &

VLLM_PID=$!
echo "VLLM server started (PID: $VLLM_PID)"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$VLLM_PID" ]; then
        echo "Stopping VLLM server (PID: $VLLM_PID)..."
        kill -TERM $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
    echo "Done!"
}
trap cleanup EXIT

# Wait for VLLM to be ready
echo ""
echo "Waiting for VLLM server to be ready..."
max_attempts=60
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; then
        echo "✓ VLLM server is ready!"
        break
    fi
    attempt=$((attempt + 1))
    if [ $((attempt % 10)) -eq 0 ]; then
        echo "  Still waiting... ($attempt/$max_attempts)"
    fi
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "✗ VLLM server failed to start. Check logs/test_instance/vllm.log"
    exit 1
fi

# Start training
echo ""
echo "Starting training on GPU $TRAINING_GPU..."
CUDA_VISIBLE_DEVICES=$TRAINING_GPU "$SCRIPT_DIR/.venv/bin/python" train_trl_grpo.py \
    --dataset $DATASET \
    --seed $SEED \
    --vllm-port $PORT \
    --run-suffix test_instance \
    --eval-every 5 \
    2>&1 | tee logs/test_instance/training.log

echo ""
echo "✓ Test completed successfully!"
