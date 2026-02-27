# Docker Configuration Files

This directory contains all Docker-related configuration files for the GRAIL project.

## Files

### Core Files
- **`Dockerfile`** - Main Docker image definition for GRAIL miners and validators
- **`docker-compose.validator.yml`** - Production validator deployment with Watchtower for automatic updates
- **`compose.grafana.yaml`** - Grafana + Loki deployment for centralized logging
- **`promtail.yml`** - Promtail configuration for log shipping to Loki

### Grafana
- **`grafana/`** - Dashboards and provisioning configs for Grafana

## Quick Start

### Deploy a Validator

See [Validator Setup Guide](../docs/validator.md) for full instructions.

```bash
cp .env.example .env
# Edit .env with your wallet names, network, and R2 credentials

docker compose --env-file .env -f docker/docker-compose.validator.yml up -d
```

## Building Images

Images are automatically built and published via GitHub Actions on:
- New releases (tagged with version)
- Manual workflow dispatch

To build locally:
```bash
docker build -f docker/Dockerfile -t grail:local .
```
