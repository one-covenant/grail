# GRAIL Validator Disk Space Management Guide

## Overview

This guide documents how the GRAIL validator is configured to prevent disk space issues and keep storage manageable over long-running operations.

## Problem Summary

**What was consuming space:**
- Docker images: 13.87GB
- Docker volumes (cache): 53.92GB
- HuggingFace model cache: 5.8GB
- uv package cache: 7.5GB
- WandB local cache: Unknown (can grow large)

**Solution implemented:**
- Ephemeral storage (1.5TB on /dev/vdb) for checkpoints and logs
- Automatic log rotation with reduced sizes
- Aggressive WandB cache cleanup
- Docker cleanup script with periodic maintenance

---

## Current Configuration

### 1. Root Filesystem (/dev/vda1)
- **Total:** 97GB
- **Used:** ~52GB (54%)
- **Available:** ~46GB
- **Status:** ✅ Healthy

### 2. Ephemeral Storage (/dev/vdb)
- **Total:** 1.5TB
- **Used:** 40KB
- **Available:** 1.4TB+
- **Purpose:** Checkpoints, logs, and cache
- **Status:** ✅ Excellent

---

## Docker Configuration

### Environment Variables (.env)

```bash
# Cache directory (1.5TB ephemeral storage)
GRAIL_CACHE_DIR=/ephemeral/grail

# Logs directory (ephemeral - no root FS bloat)
GRAIL_LOG_FILE=/ephemeral/grail/grail.log

# Log rotation (reduced to prevent bloat)
GRAIL_LOG_MAX_SIZE=50MB
GRAIL_LOG_BACKUP_COUNT=3

# WandB (cached in /tmp, not persistent)
WANDB_CACHE_DIR=/tmp/.wandb_cache
WANDB_DATA_DIR=/tmp/.wandb_data
```

### Docker Compose (docker-compose.validator.yml)

**Logging Configuration:**
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "50m"
    max-file: "3"
```

**Volumes Mounted:**
```yaml
volumes:
  - /ephemeral/grail:/ephemeral/grail  # Checkpoints + logs (1.5TB)
  - validator-cache:/root/.cache        # Docker-managed cache
  - validator-data:/app/data            # Temporary data
```

---

## Cleanup & Maintenance

### Manual Cleanup

Run the cleanup script immediately:
```bash
bash /home/shadeform/grail/docker/docker-cleanup.sh
```

Run aggressive cleanup (also clears HuggingFace model cache):
```bash
bash /home/shadeform/grail/docker/docker-cleanup.sh --aggressive
```

### Automated Cleanup (Cron)

To enable periodic cleanup, add to root's crontab:
```bash
sudo crontab -e
```

Add one of these lines:

**Option 1: Daily cleanup at 2 AM**
```bash
0 2 * * * /home/shadeform/grail/docker/docker-cleanup.sh >> /var/log/grail/docker-cleanup.log 2>&1
```

**Option 2: Every 6 hours**
```bash
0 */6 * * * /home/shadeform/grail/docker/docker-cleanup.sh >> /var/log/grail/docker-cleanup.log 2>&1
```

**Option 3: Aggressive weekly cleanup + daily lightweight cleanup**
```bash
# Daily cleanup at 2 AM
0 2 * * * /home/shadeform/grail/docker/docker-cleanup.sh >> /var/log/grail/docker-cleanup.log 2>&1

# Aggressive cleanup every Sunday at 3 AM
0 3 * * 0 /home/shadeform/grail/docker/docker-cleanup.sh --aggressive >> /var/log/grail/docker-cleanup.log 2>&1
```

### What Gets Cleaned

**Every Run:**
- Docker containers stopped >72 hours
- Docker images unused >7 days
- Docker build cache
- Orphaned Docker volumes
- Old validator logs (>30 days)
- Logs from docker-cleanup itself (>60 days)
- uv archive cache (re-downloads on demand)
- Python __pycache__ directories
- WandB local cache
- Temporary Python files

**With `--aggressive` Flag:**
- HuggingFace model files not accessed in 60+ days
- (Will re-download on next validation run)

---

## Checkpoint Management

### Automatic Retention Policy

The validator automatically keeps:
- **Last 3 windows** (most recent checkpoints)
- **Milestone checkpoints** (every 100 windows)
- **Windows 0-9** (always)

Older checkpoints are automatically deleted from both local disk and R2.

**Configuration (.env):**
```bash
GRAIL_CHECKPOINT_RETENTION_LIMIT=3
GRAIL_CHECKPOINT_MILESTONE_INTERVAL=100
```

### Checkpoint Storage

- **Location:** `/ephemeral/grail/checkpoints/`
- **Size:** ~2-5GB per checkpoint (varies by model)
- **Cleanup:** Automatic via `ValidationService.cleanup_local()`
- **Status:** Called after each window to maintain limit

---

## Monitoring

### Disk Space Alerts

The cleanup script alerts if available space drops below **5GB**:
```
WARNING: Low disk space! Available: 4500MB
```

### Logs

- **Cleanup logs:** `/var/log/grail/docker-cleanup.log`
- **Validator logs:** `/ephemeral/grail/grail.log`
- **Rotation:** Automatic (50MB per file, keep 3 backups)

### Check Current Status

```bash
# Overall disk usage
df -h

# Docker resource usage
docker system df

# Ephemeral storage usage
du -sh /ephemeral/grail/

# Validator container logs
docker logs --tail 100 grail-validator
```

---

## Current Status

✅ **Validator Running:**
- Container: `grail-validator` (Up, healthy)
- Promtail: `grail-promtail` (Up, shipping logs to Loki)
- Watchtower: `grail-watchtower` (Up, auto-updating images)

✅ **Storage Optimized:**
- Root FS: 46GB available (54% used)
- Ephemeral: 1.4TB+ available (logs & checkpoints)
- WandB: Writing to /tmp (ephemeral, not persistent)

✅ **Log Management:**
- Max size: 50MB per file
- Max backups: 3 files
- Auto-rotation enabled
- Promtail shipping to Grafana Loki

---

## Best Practices

1. **Run cleanup weekly** or after heavy validation windows
2. **Monitor** `/var/log/grail/docker-cleanup.log` for alerts
3. **Use aggressive cleanup** when < 100GB available on root FS
4. **Archive old logs** if long-term storage needed
5. **Review Watchtower logs** if auto-updates cause issues

---

## Troubleshooting

### Disk Space Issues

```bash
# Quick diagnosis
df -h
docker system df
du -sh /ephemeral/grail/

# Emergency cleanup
docker compose -f docker/docker-compose.validator.yml down
bash docker/docker-cleanup.sh --aggressive
docker compose --env-file .env --profile promtail -f docker/docker-compose.validator.yml up -d
```

### Promtail Not Shipping Logs

```bash
# Check Promtail status
docker logs grail-promtail

# Verify log file exists
docker exec grail-validator ls -lah /ephemeral/grail/grail.log

# Check Loki connectivity
curl http://3.140.247.6:3100/loki/api/v1/push -H "Content-Type: application/json"
```

### Checkpoint Download Issues

```bash
# Clear local checkpoint cache (will re-download)
rm -rf /ephemeral/grail/checkpoint-*

# Restart validator
docker compose -f docker/docker-compose.validator.yml restart grail-validator
```

---

## Reference

- **Docker Compose:** `/home/shadeform/grail/docker/docker-compose.validator.yml`
- **Cleanup Script:** `/home/shadeform/grail/docker/docker-cleanup.sh`
- **Environment Config:** `/home/shadeform/grail/.env`
- **Validator Docs:** `/home/shadeform/grail/docs/validator.md`
- **Checkpoint Manager:** `grail/infrastructure/checkpoints.py`
