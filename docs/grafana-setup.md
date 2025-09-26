# Observability Setup Guide

Quick setup for Grafana + Loki observability stack with public dashboard access.

## Prerequisites

- **Docker + Docker Compose** installed and running
- AWS/host firewall allows inbound to **3000/tcp** (Grafana) and optionally **3100/tcp** (Loki API)

## Quick Start

### 1. Configure Environment (Optional but Recommended)

Add these variables to your `.env` file for logging and external access:

```bash
# Promtail-based log shipping
PROMTAIL_ENABLE=true
PROMTAIL_LOKI_URL=http://localhost:3100/loki/api/v1/push
PROMTAIL_JOB=grail
GRAIL_ENV=prod
GRAIL_LOG_FILE=/var/log/grail/grail.log

# Log rotation settings (optional)
GRAIL_LOG_MAX_SIZE=100MB
GRAIL_LOG_BACKUP_COUNT=5

# Grafana server root URL for external access
GF_SERVER_ROOT_URL=http://your-public-ip:3000
```

### 2. Start the Stack

```bash
cd /home/ubuntu/grail/docker
docker compose --env-file ../.env -f compose.grafana.yaml up -d
```

### 3. Verify Services

```bash
# Check container status
docker compose --env-file ../.env -f compose.grafana.yaml ps

# Health checks
curl -s http://localhost:3000/api/health
curl -s http://localhost:3100/ready
```

### 4. Access Grafana

- **Local**: `http://localhost:3000`
- **Remote**: `http://<your-server-ip>:3000` (if `GF_SERVER_ROOT_URL` is set)
- **Admin credentials**: `admin` / `admin` (change in production)
- **Anonymous access**: Enabled with Viewer role for public dashboards

### 5. Stop the Stack

```bash
docker compose --env-file ../.env -f compose.grafana.yaml down
```

## What's Included

- **Loki**: Log aggregation system (port 3100)
- **Grafana**: Visualization and dashboards (port 3000)
- **Auto-provisioning**: 
  - Loki datasource pre-configured
  - Public log explorer dashboard as default home
- **Public access**: Anonymous viewing enabled with embedding support

## Configuration Details

### Public Access Settings
- Anonymous authentication enabled
- Viewer role for anonymous users
- Embedding allowed for iframe usage
- Sign-up and org creation disabled

### Default Dashboard
The stack automatically loads `grail-public-log-explorer.json` as the home dashboard for anonymous users.

### External Access
To set a custom public URL, add to your `.env` file:
```bash
GF_SERVER_ROOT_URL=http://your-public-ip:3000
```

## Optional Components

These files exist but aren't required for basic operation:

- **`setup_public_dashboard.sh`**: Convenience wrapper script
- **`scripts/setup_grafana_node.sh`**: Manual datasource setup (superseded by provisioning)
- **`.env.example`**: Contains other GRAIL configuration variables (observability variables are already documented there)

## Troubleshooting

### Grafana won't start
- Check if port 3000 is already in use: `netstat -tlnp | grep 3000`
- Verify Docker daemon is running: `docker info`

### No data in dashboards
- Ensure GRAIL is configured to send logs to Loki
- Check Loki is receiving data: `curl http://localhost:3100/loki/api/v1/label`
- Verify datasource connection in Grafana: Settings → Data Sources → Loki → Test

### Can't access externally
- Check firewall rules for port 3000
- Verify `GF_SERVER_ROOT_URL` matches your public IP/domain
- Ensure Docker containers are bound to `0.0.0.0`, not `127.0.0.1`
