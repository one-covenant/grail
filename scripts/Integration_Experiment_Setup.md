# Running Tier 3 Test on lium.io

Quick guide for setting up and running Tier 3 integration tests on a new lium.io node.

## Prerequisites

- Setting up a node with 3+ GPUs on lium.io
- Test TAO tokens in your coldkey wallet for hotkey registration fees (get from faucet)
- Cloudflare R2 account with bucket and API credentials (for validator storage)
- Optional: WandB account for monitoring, Hugging Face account for dataset uploads

## Setup Steps

### 1. Initial Node Setup

Run the setup script to install Docker, NVIDIA Container Toolkit, and uv:

```bash
cd /root/grail
bash scripts/setup_node.sh setup
```

### 2. Install Python Dependencies

Install the project dependencies using uv:

```bash
uv sync --all-extras
```

### 3. Configure Test Hotkeys

Create test hotkeys for your miners and validator, then register them on the chain:

```bash
# Create validator hotkey (required)
uv run btcli wallet new_hotkey --wallet.name <your-test-wallet> --wallet.hotkey grail-hotkey

# Create miner hotkeys hk1 and hk2 for testing
uv run btcli wallet new_hotkey --wallet.name <your-test-wallet> --wallet.hotkey hk1
uv run btcli wallet new_hotkey --wallet.name <your-test-wallet> --wallet.hotkey hk2

# Register all hotkeys on the test subnet (requires TAO for registration fee)
uv run btcli subnet register --wallet.name <your-test-wallet> --wallet.hotkey grail-hotkey --netuid 429 --subtensor.network test
uv run btcli subnet register --wallet.name <your-test-wallet> --wallet.hotkey hk1 --netuid 429 --subtensor.network test
uv run btcli subnet register --wallet.name <your-test-wallet> --wallet.hotkey hk2 --netuid 429 --subtensor.network test
```

### 4. Set Environment Variables

Copy the example environment file and configure all required parameters:

```bash
# Copy the example file
cp .env.example .env

# Edit the file to configure your settings
nano .env  # or use your preferred editor
```

Key parameters to configure in `.env`:

**Network & Wallet (Required):**
- `BT_NETWORK=test` (for test subnet)
- `BT_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443` (test subnet endpoint)
- `NETUID=429` (grail test subnet ID)
- `BT_WALLET_COLD=<your-test-wallet>`

**Storage (Required for validators, optional for miners):**
- `R2_BUCKET_ID` - Your Cloudflare R2 bucket name
- `R2_ACCOUNT_ID` - Your Cloudflare account ID
- `R2_WRITE_ACCESS_KEY_ID` - R2 API access key
- `R2_WRITE_SECRET_ACCESS_KEY` - R2 API secret key

**Optional but Recommended:**
- `HF_TOKEN` - Hugging Face token for dataset uploads
- `HF_USERNAME` - Your Hugging Face username
- `WANDB_API_KEY` - For monitoring in WandB
- `WANDB_PROJECT` - WandB project name

NOTE: Don't set `BT_WALLET_HOT` and `GRAIL_MODEL_NAME` in the `.env` file.

See `.env.example` for detailed instructions on obtaining each value.

### 5. Run the Test

Start the test with 1 validator and 2 miners in the background:

```bash
nohup python scripts/run_tier3_test.py \
    --miners "Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507" \
    --validator "Qwen/Qwen3-4B-Instruct-2507" \
    --validator-delay 15 \
    --hotkeys hk1,hk2 \
    > tier3_test.out 2> tier3_test.err &
```

This configuration:
- **Miner 1**: Uses a different model (Qwen3-0.6B) with hotkey `hk1`
- **Miner 2**: Uses the same model as validator (Qwen3-4B-Instruct-2507) with hotkey `hk2`
- **Validator**: Uses Qwen3-4B-Instruct-2507 with hotkey `grail-hotkey` (hardcoded)
- **Delay**: Waits 15 seconds after starting miners before launching validator
- **Network**: Runs on test subnet (netuid 429) using test TAO tokens

## Monitoring

- **Live logs**: `tail -f tier3_test.out`
- **Error logs**: `tail -f tier3_test.err`
- **Detailed logs**: Check `logs/tier3/<timestamp>/` directory
- **Process status**: `ps aux | grep run_tier3_test.py`

## Stopping the Test

```bash
# Find the process ID
ps aux | grep run_tier3_test.py

# Kill the process (it will gracefully stop all miners/validator)
kill <PID>
```

## Notes

- The script automatically assigns GPUs to each miner and validator
- Logs are organized by timestamp in `logs/tier3/`
- Each service (miner/validator) has its own log file
- The validator runs on test subnet (checks all miners on the network)
