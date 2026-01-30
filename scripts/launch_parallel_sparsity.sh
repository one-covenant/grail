#!/bin/bash
# Launch parallel sparsity computation for all iter1 experiments
# Batch 1: 14 processes (all Qwen 0.5B, Gemma 1B, Qwen 1.5B + 2 Llama 3B)
# Batch 2: 2 remaining Llama 3B processes

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv_linux/bin/python"
COMPUTE_SCRIPT="$SCRIPT_DIR/compute_k_step_sparsity_parallel.py"
OUTPUT_DIR="$PROJECT_DIR/data/sparsity_parallel"
LOG_DIR="/tmp/sparsity_logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Starting parallel sparsity computation..."
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo ""

# Define all experiments
declare -a EXPERIMENTS=(
    # Qwen 0.5B - 4 seeds (~40GB each)
    "Qwen 0.5B|42"
    "Qwen 0.5B|1337"
    "Qwen 0.5B|2024"
    "Qwen 0.5B|9999"
    # Gemma 1B - 4 seeds (~60GB each)
    "Gemma 1B|42"
    "Gemma 1B|1337"
    "Gemma 1B|2024"
    "Gemma 1B|9999"
    # Qwen 1.5B - 4 seeds (~120GB each)
    "Qwen 1.5B|42"
    "Qwen 1.5B|1337"
    "Qwen 1.5B|2024"
    "Qwen 1.5B|9999"
    # Llama 3B - 2 seeds in batch 1 (~200GB each)
    "Llama 3B|42"
    "Llama 3B|1337"
)

declare -a BATCH2_EXPERIMENTS=(
    # Llama 3B - 2 remaining seeds
    "Llama 3B|2024"
    "Llama 3B|9999"
)

# Launch batch 1
echo "=== BATCH 1: Launching 14 processes ==="
PIDS=()

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r model seed <<< "$exp"
    safe_name=$(echo "${model}_${seed}" | tr ' ' '_')
    output_file="$OUTPUT_DIR/${safe_name}.csv"
    log_file="$LOG_DIR/${safe_name}.log"

    echo "  Starting: $model seed=$seed"
    nohup $VENV "$COMPUTE_SCRIPT" --model "$model" --seed "$seed" --output "$output_file" > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} processes. PIDs: ${PIDS[*]}"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/*.log"
echo "  ps aux | grep compute_k_step"
echo ""
echo "Waiting for batch 1 to complete..."

# Wait for all batch 1 processes
for pid in "${PIDS[@]}"; do
    wait $pid || echo "Process $pid exited with error"
done

echo ""
echo "=== BATCH 1 COMPLETE ==="
echo ""

# Launch batch 2 if needed
if [ ${#BATCH2_EXPERIMENTS[@]} -gt 0 ]; then
    echo "=== BATCH 2: Launching ${#BATCH2_EXPERIMENTS[@]} processes ==="
    PIDS=()

    for exp in "${BATCH2_EXPERIMENTS[@]}"; do
        IFS='|' read -r model seed <<< "$exp"
        safe_name=$(echo "${model}_${seed}" | tr ' ' '_')
        output_file="$OUTPUT_DIR/${safe_name}.csv"
        log_file="$LOG_DIR/${safe_name}.log"

        echo "  Starting: $model seed=$seed"
        nohup $VENV "$COMPUTE_SCRIPT" --model "$model" --seed "$seed" --output "$output_file" > "$log_file" 2>&1 &
        PIDS+=($!)
    done

    echo "Waiting for batch 2 to complete..."

    for pid in "${PIDS[@]}"; do
        wait $pid || echo "Process $pid exited with error"
    done

    echo ""
    echo "=== BATCH 2 COMPLETE ==="
fi

echo ""
echo "=== ALL DONE ==="
echo ""

# Merge all CSVs
echo "Merging CSV files..."
MERGED_FILE="$PROJECT_DIR/data/sparsity_k_step.csv"

# Write header
head -1 "$OUTPUT_DIR"/Qwen_0.5B_42.csv > "$MERGED_FILE"

# Append data from all files (skip headers)
for f in "$OUTPUT_DIR"/*.csv; do
    tail -n +2 "$f" >> "$MERGED_FILE"
done

echo "Merged results: $MERGED_FILE"
echo "Total lines: $(wc -l < "$MERGED_FILE")"
