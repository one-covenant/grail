# Docker Configuration Files

This directory contains all Docker-related configuration files for the GRAIL project.

## Files

### Core Files
- **`Dockerfile`** - Main Docker image definition for GRAIL miners and validators

### Docker Compose Files
- **`docker-compose.validator.yml`** - Production deployment with Watchtower for automatic updates
- **`docker-compose.local-subnet.yml`** - Local testing environment with Bittensor subnet
- **`docker-compose.integration.yml`** - Integration testing configuration

## Quick Start

### Deploy a Validator

```bash
# Copy and configure environment (project root)
cp .env.example .env
# Edit .env with non-sensitive config (no private keys)
# Provide wallet secrets via Docker secrets or a bind-mounted file read at runtime.
# Start validator with auto-updates
docker compose \
  --env-file .env \
  -f docker/docker-compose.validator.yml up -d
```

### Local Testing

```bash
# Automated setup
./scripts/setup-local-subnet.sh

# Or manual startup
docker-compose -f docker/docker-compose.local-subnet.yml up -d
```

## Building Images

Images are automatically built and published via GitHub Actions on:
- New releases (tagged with version)
- Manual workflow dispatch

To build locally:
```bash
docker build -f docker/Dockerfile -t grail:local .
```

## Documentation

For detailed deployment instructions, see:
- [Validator Deployment Guide](../docs/VALIDATOR_DEPLOYMENT.md)
- [Local Subnet Testing](../docs/local-subnet-testing.md)