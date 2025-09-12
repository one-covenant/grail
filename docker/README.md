# Docker Configuration Files

This directory contains all Docker-related configuration files for the GRAIL project.

## Files

### Core Files
- **`Dockerfile`** - Main Docker image definition for GRAIL miners and validators
- **`.env.validator.example`** - Example environment configuration for validators

### Docker Compose Files
- **`docker-compose.validator.yml`** - Production deployment with Watchtower for automatic updates
- **`docker-compose.local-subnet.yml`** - Local testing environment with Bittensor subnet
- **`docker-compose.integration.yml`** - Integration testing configuration

## Quick Start

### Deploy a Validator

```bash
# Copy and configure environment
cp docker/.env.validator.example docker/.env.validator
# Edit docker/.env.validator with your wallet details

# Start validator with auto-updates
docker-compose -f docker/docker-compose.validator.yml --env-file docker/.env.validator up -d
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