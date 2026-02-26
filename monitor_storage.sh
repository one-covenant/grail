#!/bin/bash
# Real-time storage monitoring for grail-validator

echo "=== GRAIL VALIDATOR STORAGE MONITOR ==="
echo "Start time: $(date)"
echo ""

# Initial snapshot
echo "[$(date '+%H:%M:%S')] === INITIAL SNAPSHOT ==="
CONTAINER_ID=$(docker inspect grail-validator --format='{{.Id}}')
echo "Container ID: ${CONTAINER_ID:0:12}"

# Function to get volume size safely
get_volume_size() {
    local vol_name=$1
    local mount_point=$(docker volume inspect "$vol_name" --format='{{.Mountpoint}}' 2>/dev/null)
    if [ -n "$mount_point" ]; then
        stat -f%z "$mount_point" 2>/dev/null || du -sb "$mount_point" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

# Monitor loop - captures every 60 seconds
for i in {1..30}; do
    TS=$(date '+%H:%M:%S')

    # Container docker logs size
    LOG_SIZE=$(docker logs grail-validator 2>&1 | wc -c)

    # Application grail.log size
    APP_LOG_SIZE=$(docker exec grail-validator stat -c%s /var/log/grail/grail.log 2>/dev/null || echo "0")

    # Cache contents
    CACHE_SIZE=$(docker exec grail-validator du -sb /root/.cache 2>/dev/null | cut -f1 || echo "0")
    HF_SIZE=$(docker exec grail-validator du -sb /root/.cache/huggingface 2>/dev/null | cut -f1 || echo "0")
    WANDB_SIZE=$(docker exec grail-validator du -sb /root/.cache/wandb 2>/dev/null | cut -f1 || echo "0")

    # Memory and CPU
    MEM=$(docker stats grail-validator --no-stream --format='{{.MemUsage}}' 2>/dev/null | cut -d' ' -f1)
    CPU=$(docker stats grail-validator --no-stream --format='{{.CPUPerc}}' 2>/dev/null)

    # Format output
    echo "[$TS] Snapshot $i:"
    echo "  Docker logs: $(numfmt --to=iec-i --suffix=B $LOG_SIZE 2>/dev/null || echo "$LOG_SIZE B")"
    echo "  Grail.log: $(numfmt --to=iec-i --suffix=B $APP_LOG_SIZE 2>/dev/null || echo "$APP_LOG_SIZE B")"
    echo "  Cache total: $(numfmt --to=iec-i --suffix=B $CACHE_SIZE 2>/dev/null || echo "$CACHE_SIZE B")"
    echo "  └─ HuggingFace: $(numfmt --to=iec-i --suffix=B $HF_SIZE 2>/dev/null || echo "$HF_SIZE B")"
    echo "  └─ WandB: $(numfmt --to=iec-i --suffix=B $WANDB_SIZE 2>/dev/null || echo "$WANDB_SIZE B")"
    echo "  Memory: $MEM | CPU: $CPU"
    echo ""

    [ $i -lt 30 ] && sleep 60
done

echo "=== FINAL ANALYSIS ==="
echo "End time: $(date)"
echo "Monitor duration: 29 minutes"
