#!/bin/bash

# GRAIL Docker Cleanup Script
# Manages space in Docker volumes by cleaning up cache bloat
# Run periodically: 0 0 * * * /path/to/cleanup.sh (daily at midnight)

set -e

CONTAINER_NAME="grail-validator"
VOLUME_MOUNT_POINT="/var/lib/docker/volumes/docker_validator-cache/_data"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="/tmp/grail_cleanup_$(date '+%Y%m%d_%H%M%S').log"

echo "[$TIMESTAMP] === GRAIL Docker Cleanup Started ===" | tee "$LOG_FILE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# PHASE 1: Clean WandB Cache (4.9 GB growth - CRITICAL)
# ============================================================================
cleanup_wandb_cache() {
    log_info "Phase 1: Cleaning WandB cache..."

    WANDB_DIR="$VOLUME_MOUNT_POINT/wandb"

    if [ ! -d "$WANDB_DIR" ]; then
        log_warn "WandB directory not found: $WANDB_DIR"
        return
    fi

    BEFORE=$(du -sh "$WANDB_DIR" 2>/dev/null | cut -f1)

    # Keep only recent runs (last 5 runs)
    log_info "Removing old WandB runs (keeping last 5)..."
    find "$WANDB_DIR/logs" -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
    find "$WANDB_DIR/artifacts" -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true

    AFTER=$(du -sh "$WANDB_DIR" 2>/dev/null | cut -f1)
    log_info "WandB cleanup: $BEFORE → $AFTER"
}

# ============================================================================
# PHASE 2: Clean HuggingFace Cache (GSM8K datasets)
# ============================================================================
cleanup_hf_cache() {
    log_info "Phase 2: Cleaning HuggingFace cache..."

    HF_DIR="$VOLUME_MOUNT_POINT/huggingface"

    if [ ! -d "$HF_DIR" ]; then
        log_warn "HF directory not found: $HF_DIR"
        return
    fi

    BEFORE=$(du -sh "$HF_DIR" 2>/dev/null | cut -f1)

    # Remove cache older than 14 days (keeps recent datasets)
    log_info "Removing stale HF cache (older than 14 days)..."
    find "$HF_DIR" -type f -mtime +14 -delete 2>/dev/null || true

    # Clean empty directories
    find "$HF_DIR" -type d -empty -delete 2>/dev/null || true

    AFTER=$(du -sh "$HF_DIR" 2>/dev/null | cut -f1)
    log_info "HF cache cleanup: $BEFORE → $AFTER"
}

# ============================================================================
# PHASE 3: Check and Report Container Logs
# ============================================================================
cleanup_container_logs() {
    log_info "Phase 3: Checking container logs..."

    if ! command -v docker &> /dev/null; then
        log_warn "Docker not found, skipping container log check"
        return
    fi

    # Get container ID
    CONTAINER_ID=$(docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.ID}}" | head -1)

    if [ -z "$CONTAINER_ID" ]; then
        log_warn "Container '$CONTAINER_NAME' not found"
        return
    fi

    CONTAINER_LOG_PATH="/var/lib/docker/containers/$CONTAINER_ID/check-point-dir"
    JSON_LOG="/var/lib/docker/containers/$CONTAINER_ID/$CONTAINER_ID-json.log"

    if [ -f "$JSON_LOG" ]; then
        LOG_SIZE=$(du -h "$JSON_LOG" 2>/dev/null | cut -f1)
        log_info "Container JSON log size: $LOG_SIZE"
    fi
}

# ============================================================================
# PHASE 4: Report Overall Space Usage
# ============================================================================
report_space_usage() {
    log_info "Phase 4: Space usage report..."

    echo "" | tee -a "$LOG_FILE"
    log_info "=== DOCKER SPACE USAGE ==="
    docker system df 2>/dev/null | tee -a "$LOG_FILE" || log_warn "Docker system df failed"

    echo "" | tee -a "$LOG_FILE"
    log_info "=== VOLUME BREAKDOWN ==="
    sudo du -sh "$VOLUME_MOUNT_POINT"/* 2>/dev/null | sort -rh | tee -a "$LOG_FILE" || log_warn "Volume breakdown failed"

    echo "" | tee -a "$LOG_FILE"
    log_info "=== ROOT FILESYSTEM ==="
    df -h / | tee -a "$LOG_FILE"
}

# ============================================================================
# PHASE 5: Optional Aggressive Cleanup (if below threshold)
# ============================================================================
aggressive_cleanup() {
    log_warn "Running aggressive cleanup mode..."

    log_info "Pruning unused Docker resources..."
    docker system prune -f 2>/dev/null || true

    log_info "Cleaning old log files..."
    find /var/log -type f -mtime +30 -delete 2>/dev/null || true
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Check if running with sudo if volume access needed
if [ -d "$VOLUME_MOUNT_POINT" ] && [ ! -w "$VOLUME_MOUNT_POINT" ]; then
    log_error "No write permission to $VOLUME_MOUNT_POINT. Run with sudo."
    exit 1
fi

# Run cleanup phases
cleanup_wandb_cache
cleanup_hf_cache
cleanup_container_logs
report_space_usage

# Optional: Check disk usage and run aggressive cleanup if >80%
DISK_USAGE=$(df / | tail -1 | awk '{print int($5)}')
if [ "$DISK_USAGE" -gt 80 ]; then
    log_warn "Disk usage at ${DISK_USAGE}% - running aggressive cleanup!"
    aggressive_cleanup
fi

echo "" | tee -a "$LOG_FILE"
log_info "=== Cleanup completed ==="
log_info "Full log: $LOG_FILE"
echo "" | tee -a "$LOG_FILE"
