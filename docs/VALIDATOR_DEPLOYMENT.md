# GRAIL Validator Deployment with Watchtower

This guide explains how to deploy GRAIL validators with automatic updates using Docker and Watchtower.

## Overview

Watchtower automatically monitors and updates your validator container when new images are published to the GitHub Container Registry (ghcr.io). This ensures your validator always runs the latest stable version without manual intervention.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- Bittensor wallet configured
- At least 16GB RAM recommended

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/tplr-ai/grail.git
cd grail
```

### 2. Configure Environment

Copy the example configuration:

```bash
cp docker/.env.validator.example docker/.env.validator
```

Edit `docker/.env.validator` with your settings:

```bash
# Required settings
WALLET_NAME=your_wallet_name
WALLET_HOTKEY=your_hotkey_name

# Optional: Monitoring (recommended)
WANDB_API_KEY=your_wandb_api_key
```

### 3. Deploy with Docker Compose

Start the validator and Watchtower:

```bash
docker-compose -f docker/docker-compose.validator.yml --env-file docker/.env.validator up -d
```

### 4. Monitor Logs

View validator logs:
```bash
docker logs -f grail-validator
```

View Watchtower logs:
```bash
docker logs -f watchtower
```

## How It Works

### Dockerfile Compatibility

The GRAIL Dockerfile is optimized for Watchtower compatibility:

1. **Stable Base Image**: Uses official PyTorch image with CUDA support
2. **Clean Entrypoint**: Simple `ENTRYPOINT ["uv", "run", "grail"]` allows proper signal handling
3. **Proper Layers**: Dependencies cached separately from application code
4. **Health Checks**: Built-in health monitoring for container status

### Update Process

When Watchtower detects a new image:

1. **Pulls New Image**: Downloads latest `ghcr.io/tplr-ai/grail:latest`
2. **Graceful Shutdown**: Sends SIGTERM to validator (120s timeout)
3. **Container Recreation**: Starts new container with same configuration
4. **Cleanup**: Removes old image to save disk space

### Update Schedule

- **Current Setting**: Checks every 30 seconds for rapid updates
- This aggressive polling ensures validators stay up-to-date with the latest releases
- The image is public on GitHub Container Registry, no authentication needed

## Configuration Options

### Watchtower Settings

The current configuration uses:
- **30-second intervals** for checking updates (via `--interval 30` command)
- **Automatic cleanup** of old images after updates
- **Label-based updates** - only containers with `watchtower.enable=true` label
- **Include restarting** containers in update checks

### Environment Variables

Required variables in `docker/.env.validator`:
- `WALLET_NAME` - Your Bittensor wallet name
- `WALLET_HOTKEY` - Your hotkey name

Optional variables:
- `NETUID` - Network UID (default: 81)
- `SUBTENSOR_NETWORK` - Network to connect to (default: finney)
- `WANDB_API_KEY` - For monitoring with Weights & Biases
- `WANDB_PROJECT` - WandB project name (default: grail-validator)

## Advanced Deployment

### Using Specific Image Tags

To pin a specific version instead of `latest`:

```yaml
# In docker/docker-compose.validator.yml
services:
  validator:
    image: ghcr.io/tplr-ai/grail:v1.2.3  # Use specific version
```

### Multiple Validators

Run multiple validators on the same machine:

1. Create separate compose files with unique container names
2. Use different ports for each validator
3. Ensure sufficient GPU memory for all instances

### Manual Update Control

To disable automatic updates temporarily:

```bash
# Stop Watchtower
docker stop watchtower

# Manual update when ready
docker pull ghcr.io/tplr-ai/grail:latest
docker-compose -f docker/docker-compose.validator.yml restart validator
```

## Monitoring

### Container Health

Check validator health:
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.State}}"
```

### Resource Usage

Monitor GPU and memory:
```bash
# GPU usage
nvidia-smi

# Container stats
docker stats grail-validator
```

### Wandb Integration

With `WANDB_API_KEY` configured, track:
- Validation metrics
- Reward history
- Network performance
- System resources

## Troubleshooting

### Validator Not Starting

1. Check logs: `docker logs grail-validator`
2. Verify wallet path: Ensure `~/.bittensor` is accessible
3. Check GPU: Run `nvidia-smi` to verify CUDA availability

### Watchtower Not Updating

1. Check registry access: `docker pull ghcr.io/tplr-ai/grail:latest`
2. Verify labels: Container must have `com.centurylinklabs.watchtower.enable=true`
3. Review Watchtower logs: `docker logs watchtower`

### Network Issues

1. Ensure ports are accessible if using external axon
2. Check firewall rules for Docker networks
3. Verify subtensor endpoint connectivity

## Security Considerations

1. **Read-Only Wallet Mount**: Wallet directory mounted as read-only (`:ro`)
2. **No Root Privileges**: Container runs without privileged mode
3. **Isolated Network**: Uses dedicated Docker network
4. **Automatic Cleanup**: Old images removed to prevent disk bloat

## Best Practices

1. **Monitor Updates**: Set up notifications to track automatic updates
2. **Regular Backups**: Backup wallet and configuration files
3. **Resource Limits**: Consider setting memory/CPU limits in production
4. **Log Rotation**: Configure Docker log rotation to prevent disk fill
5. **Health Monitoring**: Use external monitoring tools for production deployments

## Support

For issues or questions:
- GitHub Issues: https://github.com/tplr-ai/grail/issues
- Documentation: https://github.com/tplr-ai/grail/blob/main/SPEC.md