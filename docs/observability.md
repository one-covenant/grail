## Observability (Grafana + Loki) for GRAIL

This guide shows how to run Grafana + Loki locally and on AWS EC2, and how to ship GRAIL logs to one or more Loki endpoints while keeping Weights & Biases monitoring enabled.

### Concepts: Observability vs Monitoring
- Observability (ops): logs/metrics/traces for runtime health. Here we ship logs to Loki and visualize with Grafana.
- Monitoring (ML/experiments): run metrics, artifacts, hyperparameters, etc. This remains in Weights & Biases (WANDB).

Both run in parallel: GRAIL logs are pushed to Loki; experiment metrics go to WANDB.

---

## Architecture

- GRAIL process emits logs via Python `logging`.
- A non-blocking Loki handler sends logs to configured Loki endpoints over HTTP.
- You can specify multiple Loki endpoints (e.g., local and EC2) separated by commas.
- Grafana reads logs from Loki and provides dashboards/queries.

Optional: Promtail for file-based log shipping if you also want to forward legacy log files.

---

## Local setup (Docker)

1) Start Grafana + Loki using the provided compose file:

```bash
docker compose -f docker/compose.observability.yaml up -d
```

2) Access services:
- Grafana: http://localhost:3000 (default admin/admin; change on first login)
- Loki: http://localhost:3100

3) Configure your app to ship logs locally (and optionally to EC2 as well):

```bash
# .env example
GRAIL_OBS_LOKI_URL=http://localhost:3100/loki/api/v1/push
GRAIL_OBS_LOKI_LABELS=env=dev,service=grail

# Optional tuning
GRAIL_OBS_LOKI_TIMEOUT_S=2.5
GRAIL_OBS_LOKI_BATCH_SIZE=1
GRAIL_OBS_LOKI_BATCH_INTERVAL_S=1.0
```

4) Run your CLI as usual; logs will be visible in Grafana’s Explore page after a few seconds.

---

## AWS EC2 setup

### 1) Provision the instance
- AMI: Ubuntu 22.04 LTS
- Instance size: t3.small or larger (adjust for your load)
- Storage: 20GB+ gp3 (adjust as needed)
- Security Groups (tighten to your IPs):
  - 22/tcp (SSH) – optional if using SSM
  - 3000/tcp (Grafana) – ideally expose via HTTPS reverse proxy only
  - 3100/tcp (Loki) – recommend not exposing publicly; use private networking or proxy
  - 80/443 if you terminate TLS via Nginx/Traefik

### 2) Install Docker & Compose

```bash
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

### 3) Deploy Grafana + Loki

```bash
sudo mkdir -p /opt/observability
cd /opt/observability
# Copy compose file from repo to EC2 host
sudo tee compose.observability.yaml >/dev/null <<'YML'
version: "3.8"
services:
  loki:
    image: grafana/loki:2.9.6
    command: -config.file=/etc/loki/local-config.yaml
    ports:
      - "3100:3100"
    volumes:
      - loki-data:/loki
    restart: unless-stopped

  grafana:
    image: grafana/grafana:11.1.4
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped
    depends_on:
      - loki

volumes:
  loki-data:
  grafana-data:
YML

docker compose -f compose.observability.yaml up -d
```

Visit `http://<ec2-public-ip>:3000` to log in (admin/admin). Change the password immediately. For production, place Grafana behind HTTPS with a reverse proxy and restrict access by IP.

### 4) Secure exposure (recommended)
- Don’t expose Loki (3100) publicly. Keep it private or behind a reverse proxy with auth.
- Put Grafana behind HTTPS (Nginx/Traefik + Let’s Encrypt) and restrict by IP or SSO.
- Backup Grafana data volume; configure Loki retention to your needs.

---

## Configure GRAIL to ship logs to multiple Loki endpoints

Set multiple endpoints (comma-separated) to ship both locally and to EC2:

```bash
GRAIL_OBS_LOKI_URL=http://localhost:3100/loki/api/v1/push,https://<ec2-host-or-dns>:3100/loki/api/v1/push
GRAIL_OBS_LOKI_LABELS=env=prod,service=grail,network=test,netuid=1

# Optional auth / tenant
GRAIL_OBS_LOKI_TENANT_ID=team-123
GRAIL_OBS_LOKI_USERNAME=your-user
GRAIL_OBS_LOKI_PASSWORD=your-pass

# Tuning (defaults shown)
GRAIL_OBS_LOKI_TIMEOUT_S=2.5
GRAIL_OBS_LOKI_BATCH_SIZE=1
GRAIL_OBS_LOKI_BATCH_INTERVAL_S=1.0
```

Backward-compatible aliases are supported: `GRAIL_LOKI_URL`, `LOKI_URL`, etc.

Run the CLI (example):

```bash
uv run grail validate -v
```

You should see a log line like “Grafana Loki logging enabled for N endpoints”. Then open Grafana → Explore → Data source: Loki, and query `{app="grail"}` or filter by labels set with `GRAIL_OBS_LOKI_LABELS`.

---

## Optional: Promtail for file-based logs

If you also want to scrape local files (e.g., `/var/log/grail/*.log`), use `docker/promtail-config.yaml` as a starting point and run Promtail:

```bash
docker run -d --name promtail \
  -v /var/log/grail:/var/log/grail:ro \
  -v $(pwd)/docker/promtail-config.yaml:/etc/promtail/config.yml \
  --network host \
  grafana/promtail:2.9.6 \
  -config.file=/etc/promtail/config.yml
```

---

## Best practices for Bittensor nodes
- Restrict exposure of Grafana/Loki; prefer private networking + proxy.
- Include labels for `wallet`, `hotkey`, `network`, `netuid` to filter per-node.
- Keep `BATCH_SIZE` small to reduce log latency; increase if throughput is high.
- Use `-vv` CLI to enable more verbose app logs during troubleshooting.
- Consider Prometheus + alerts later if you need SLOs; keep logging layer simple.

---

## Troubleshooting
- No logs in Grafana:
  - Verify `GRAIL_OBS_LOKI_URL` and that endpoints are reachable from the node.
  - Look for “Grafana Loki logging enabled for N endpoints” at startup.
  - Ensure security groups permit outbound traffic to EC2 and inbound on 3000 (Grafana) if needed.
- High log volume / drops:
  - Increase `GRAIL_OBS_LOKI_BATCH_SIZE` or decrease `GRAIL_OBS_LOKI_BATCH_INTERVAL_S`.
  - Consider Promtail + file-based shipping.
- Shutdown flush:
  - The app registers `logging.shutdown()` at exit; ensure graceful stops (SIGTERM) where possible.


