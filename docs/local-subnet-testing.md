# Local Subnet Testing Environment

This guide explains how to run GRAIL with a local Bittensor subnet for development and testing.

## Overview

The local testing environment includes:
- **2 Subtensor nodes** (Alice and Bob) running a local blockchain
- **MinIO S3** for object storage (replacing Cloudflare R2)
- **2 GRAIL miners** (M1, M2) connected to the local subnet (netuid 2)
- **1 GRAIL validator** (Alice/default as subnet owner, UID 0)

## Prerequisites

1. Docker and Docker Compose installed
2. NVIDIA GPU drivers and Docker GPU support (for miners/validators)
3. Bittensor CLI installed: `pip install bittensor`
4. At least 8GB RAM available for the subtensor nodes

## Quick Start

### Automated Setup

Run the setup script to automatically create wallets, fund them, and start all services:

```bash
./scripts/setup-local-subnet.sh
```

This script will:
1. Start local subtensor nodes (Alice and Bob)
2. Create wallets for miners and validator
3. Fund wallets from Alice's initial balance (not needed with Alice wallet)
4. Create a subnet (netuid 2)
5. Register all neurons on netuid 2 (with automatic retries)
6. Start the subnet's emission schedule
7. Start GRAIL miners and validator

### Manual Setup

If you prefer manual control, follow these steps:

#### 1. Start the local subnet and storage

```bash
docker compose -f docker/docker-compose.local-subnet.yml up -d alice bob s3 s3-setup
```

Wait for the subtensor to be ready (about 15 seconds).

#### 2. Create wallets

```bash
# Import Alice wallet (she has initial funds in local chain)
btcli wallet regen_coldkey --wallet.name Alice \
    --mnemonic "bottom drive obey lake curtain smoke basket hold race lonely fit walk" \
    --no-use-password

# Create hotkeys for miners under Alice
# Note: default hotkey will be the validator (subnet owner, UID 0)
btcli wallet new_hotkey --wallet.name Alice --wallet.hotkey default --n_words 12 --no-use-password
btcli wallet new_hotkey --wallet.name Alice --wallet.hotkey M1 --n_words 12 --no-use-password
btcli wallet new_hotkey --wallet.name Alice --wallet.hotkey M2 --n_words 12 --no-use-password
```

#### 3. Fund wallets

Alice already has initial tokens from the local chain, so no funding is needed.

#### 4. Create subnet

```bash
btcli subnet create --subtensor.network local --subtensor.chain_endpoint ws://localhost:9944 \
    --wallet.name Alice --wallet.hotkey default --no_prompt
```

#### 5. Register neurons

```bash
# Register miners (validator is automatically registered as subnet owner, UID 0)
btcli subnet register --netuid 2 --subtensor.network local --subtensor.chain_endpoint ws://localhost:9944 \
    --wallet.name Alice --wallet.hotkey M1 --no_prompt

btcli subnet register --netuid 2 --subtensor.network local --subtensor.chain_endpoint ws://localhost:9944 \
    --wallet.name Alice --wallet.hotkey M2 --no_prompt
```

#### 6. Start subnet emission

```bash
btcli subnet start --netuid 2 --subtensor.network local --subtensor.chain_endpoint ws://localhost:9944 \
    --wallet.name Alice --wallet.hotkey default --no_prompt
```

#### 7. Start GRAIL services

```bash
docker compose -f docker/docker-compose.local-subnet.yml up -d miner-1 miner-2 validator
```

## Monitoring

### View logs

```bash
# All services
docker compose -f docker-compose.local-subnet.yml logs -f

# Specific service
docker compose -f docker-compose.local-subnet.yml logs -f miner-1
docker compose -f docker-compose.local-subnet.yml logs -f validator
docker compose -f docker-compose.local-subnet.yml logs -f alice
```

### Check subnet status

```bash
# View metagraph
btcli subnet metagraph --netuid 2 --subtensor.network local --subtensor.chain_endpoint ws://localhost:9944

# Check registrations
btcli subnet list --subtensor.network local --subtensor.chain_endpoint ws://localhost:9944
```

### Access services

- **Subtensor RPC (Alice)**: `ws://localhost:9944`
- **Subtensor RPC (Bob)**: `ws://localhost:9945`
- **MinIO Console**: http://localhost:9001 (user: `minioadmin`, pass: `minioadmin`)
- **MinIO S3 API**: http://localhost:9000

## Configuration

### Environment Variables

The docker-compose file uses environment variables from your `.env` file. Key variables:

- `NETUID`: Subnet ID (default: 2 for newly created subnet)
- `BT_WALLET_COLD`: Coldkey name for each service
- `BT_WALLET_HOT`: Hotkey name for each service
- `GRAIL_MODEL_NAME`: Model to use (default: Qwen/Qwen2-0.5B-Instruct)
- `GRAIL_WINDOW_LENGTH`: Window length for testing (default: 3 blocks)
- `GRAIL_ROLLOUTS_PER_PROBLEM`: Keep at 4 (fixed in this release)

### GPU Allocation

The compose file allocates GPUs as follows:
- Miner 1: GPU 0
- Miner 2: GPU 1
- Validator: GPU 2

Adjust the `device_ids` in the compose file if you have different GPU configurations.

## Troubleshooting

### Subtensor won't start
- Check Docker memory limits: subtensor needs at least 4GB
- Ensure ports 9944, 9945, 30334, 30335 are not in use

### Wallets not found
- Ensure wallets are in `~/.bittensor/wallets/`
- Check wallet names match those in docker-compose environment

### Registration fails
- Ensure wallets have sufficient balance
- Check subnet exists (create if needed)
- Verify chain endpoint is accessible

### Miners/validators won't start
- Check GPU availability: `nvidia-smi`
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
- Check logs for specific errors

## Cleanup

Stop all services:
```bash
docker compose -f docker-compose.local-subnet.yml down
```

Remove all data (blockchain, storage):
```bash
docker compose -f docker-compose.local-subnet.yml down -v
```

## Advanced Usage

### Custom Chain Configuration

You can modify the subtensor configuration by editing the command parameters in the docker-compose file:
- `--chain`: Chain specification (default: `/localnet.json`)
- `--rpc-methods`: RPC method exposure (default: `unsafe` for testing)
- `--rpc-cors`: CORS settings (default: `all` for testing)

### Multiple Validators

To add more validators, duplicate the validator service in docker-compose and adjust:
- Service name
- Wallet configuration
- GPU allocation
- Ports if needed

### Production-like Testing

For more realistic testing:
1. Increase `GRAIL_WINDOW_LENGTH` to match production (e.g., 360)
2. Enable drand by removing `--no-drand` flag
3. Set `WANDB_MODE=online` with valid API key
4. Use larger models if GPU memory allows

## Next Steps

- Review the [R2 Credentials Design](./r2_credentials_design.md) for storage configuration
- Check [Docker Compose Integration Setup](./docker_compose_integration_setup.md) for production setup
- Monitor your subnet with the Bittensor CLI tools