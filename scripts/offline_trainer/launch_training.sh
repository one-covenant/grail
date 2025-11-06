#!/bin/bash
# Launch script for offline GRPO training on GSM8K
# This script starts a vLLM server and runs the offline trainer

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  GRAIL Offline Trainer - GSM8K Environment${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# Configuration
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
VLLM_PORT=30001
VLLM_HOST="127.0.0.1"
MAX_MODEL_LEN=2048
GPU_MEM_UTIL=0.25  # Further reduced to leave room for training model

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VLLM_PYTHON="$REPO_ROOT/tools/vllm-server/.venv/bin/python"

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Model: $MODEL_ID"
echo -e "  vLLM Server: http://$VLLM_HOST:$VLLM_PORT"
echo -e "  Max Model Length: $MAX_MODEL_LEN"
echo -e "  GPU Memory Utilization: $GPU_MEM_UTIL"
echo -e ""

# Check if vLLM Python exists
if [ ! -f "$VLLM_PYTHON" ]; then
    echo -e "${RED}Error: vLLM Python not found at $VLLM_PYTHON${NC}"
    echo -e "${YELLOW}Please run: scripts/setup_vllm_env.sh${NC}"
    exit 1
fi

# Function to check if vLLM server is ready
wait_for_vllm() {
    local max_wait=180
    local elapsed=0
    echo -e "${YELLOW}Waiting for vLLM server to be ready...${NC}"
    
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://$VLLM_HOST:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ vLLM server is ready!${NC}"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo -e "  Still waiting... (${elapsed}s elapsed)"
        fi
    done
    
    echo -e "${RED}✗ vLLM server did not start within ${max_wait}s${NC}"
    return 1
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo -e "  Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null || true
        echo -e "${GREEN}  ✓ vLLM server stopped${NC}"
    fi
}

trap cleanup EXIT INT TERM

# Start vLLM server in background
echo -e "${BLUE}Starting vLLM server...${NC}"
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 32 \
    --trust-remote-code \
    > "$SCRIPT_DIR/vllm_server.log" 2>&1 &

VLLM_PID=$!
echo -e "  vLLM server started (PID: $VLLM_PID)"
echo -e "  Logs: $SCRIPT_DIR/vllm_server.log"

# Wait for server to be ready
if ! wait_for_vllm; then
    echo -e "${RED}Failed to start vLLM server. Check logs at: $SCRIPT_DIR/vllm_server.log${NC}"
    exit 1
fi

# Run offline trainer
echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting Offline Training${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}\n"

cd "$SCRIPT_DIR"
python run_offline_grpo.py

echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"

