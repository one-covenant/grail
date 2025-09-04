## Quickstart: Run 2 Miners + 1 Validator with Docker Compose (Linux)

Follow these steps to bring up a fully working local stack using Docker Compose. This runs:
- MinIO (local S3)
- Two miners (`miner-1`, `miner-2`)
- One validator (`validator`)

### 1) Prerequisites
- Docker and Docker Compose installed
  - Ubuntu:
```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```
- Open ports 9000 and 9001 locally (MinIO).
- No GPU required.

### 2) Clone the repo
```bash
git clone https://github.com/tplr-ai/grail.git
cd grail
```

### 3) Start the stack
```bash
docker compose -f docker-compose.integration.yml up -d --build
```
This will:
- Start MinIO and create the `grail` bucket
- Start `miner-1`, `miner-2`, and `validator` on Bittensor public testnet (`BT_NETWORK=test`)
- Use a short window length (3) for faster iteration

### 4) Check status and logs
```bash
docker compose -f docker-compose.integration.yml ps
docker compose -f docker-compose.integration.yml logs -f miner-1
docker compose -f docker-compose.integration.yml logs -f validator
```
Healthy signs:
- Miners: messages like "Using base model for window …" then "Successfully uploaded window …"
- Validator: messages like "Processing window …" and "Setting weights …"

### 5) MinIO console (optional)
Open `http://localhost:9001` (user: `minioadmin`, pass: `minioadmin`).
- Bucket: `grail`
- Expect files under `grail/windows/` for the miners

### 6) Run minimal integration tests (optional)
Install `uv` (if not installed) and test locally from the repo root:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.cargo/bin/uv sync --all-extras
~/.cargo/bin/uv run pytest -m integration -q
```
The tests expect the Compose stack to be running and MinIO at `localhost:9000`.

### 7) Stop and clean up
```bash
docker compose -f docker-compose.integration.yml down -v
```

### Troubleshooting
- Ports 9000/9001 busy: stop other services or change host ports in the compose file.
- Slow or no progress: public testnet can be intermittent. Retry or set a custom `BT_CHAIN_ENDPOINT`.
- MinIO unreachable: ensure Docker network is healthy; check `s3` and `s3-setup` logs.
- Window too slow: reduce window length by editing `GRAIL_WINDOW_LENGTH` in the compose env.


