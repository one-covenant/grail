#!/bin/bash
# Setup script for local subtensor testing environment

set -e

echo "🚀 Setting up local subtensor testing environment..."

# Check if btcli is installed
if ! command -v btcli &> /dev/null; then
    echo "❌ btcli is not installed. Please install it first:"
    echo "pip install bittensor"
    exit 1
fi

# Configuration
CHAIN_ENDPOINT="ws://localhost:9944"
NETUID=2  # The newly created subnet will be netuid 2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting local subtensor nodes...${NC}"
docker compose -f docker/docker-compose.local-subnet.yml up -d alice bob

echo "Waiting for subtensor to be ready..."
sleep 15

echo -e "${GREEN}✅ Subtensor nodes started${NC}"

# Create wallets if they don't exist
echo -e "${YELLOW}Creating wallets...${NC}"

# Set wallet path
WALLET_PATH="${HOME}/.bittensor/wallets"

# Create Alice wallet from known seed (for local testing)
# Alice's well-known seed hex for substrate dev chains
if ! [ -f "$WALLET_PATH/Alice/coldkey" ]; then
    echo "Creating Alice wallet from seed..."
    # Create the wallet directory
    mkdir -p "$WALLET_PATH/Alice/hotkeys"
    
    # Import Alice's wallet using the known seed hex
    # This is the well-known private key for //Alice in substrate dev chains
    btcli wallet regen_coldkey --wallet.path "$WALLET_PATH" --wallet.name Alice \
        --seed "0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a" \
        --no-use-password || echo "Alice wallet may already exist"
fi

# Create default hotkey for Alice if it doesn't exist
if ! [ -f "$WALLET_PATH/Alice/hotkeys/default" ]; then
    echo "Creating default hotkey for Alice..."
    btcli wallet new_hotkey --wallet.path "$WALLET_PATH" --wallet.name Alice --wallet.hotkey default --n_words 12 --no-use-password
fi

# Create hotkeys for miners
# Note: The subnet owner (Alice with default hotkey) will act as validator (UID 0)
if ! [ -f "$WALLET_PATH/Alice/hotkeys/M1" ]; then
    echo "Creating miner hotkey M1..."
    btcli wallet new_hotkey --wallet.path "$WALLET_PATH" --wallet.name Alice --wallet.hotkey M1 --n_words 12 --no-use-password
fi

if ! [ -f "$WALLET_PATH/Alice/hotkeys/M2" ]; then
    echo "Creating miner hotkey M2..."
    btcli wallet new_hotkey --wallet.path "$WALLET_PATH" --wallet.name Alice --wallet.hotkey M2 --n_words 12 --no-use-password
fi

echo -e "${GREEN}✅ Wallets created${NC}"

# No need to fund wallets - Alice already has 2funds
echo -e "${GREEN}✅ Using Alice's existing funds${NC}"

# Create subnet
echo -e "${YELLOW}Creating subnet...${NC}"
# Use echo to provide empty inputs for all prompts
echo -e "\n\n\n\n\n\n\n" | btcli subnet create --subtensor.network local --subtensor.chain_endpoint $CHAIN_ENDPOINT \
    --wallet.path "$WALLET_PATH" --wallet.name Alice --wallet.hotkey default \
    --subnet-name "grail-local" \
    --no_prompt || echo "Subnet may already exist"

# Function to register a neuron with retries
register_neuron() {
    local hotkey=$1
    local description=$2
    local max_retries=5
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Registering $description (attempt $((retry_count + 1))/$max_retries)..."
        
        # Try to register
        if btcli subnet register --netuid $NETUID --subtensor.network local --subtensor.chain_endpoint $CHAIN_ENDPOINT \
            --wallet.path "$WALLET_PATH" --wallet.name Alice --wallet.hotkey $hotkey --no_prompt 2>&1 | tee /tmp/register_$hotkey.log | grep -q "✅ Registered"; then
            echo -e "${GREEN}✅ Successfully registered $description${NC}"
            return 0
        fi
        
        # Check if already registered
        if grep -q "already registered" /tmp/register_$hotkey.log; then
            echo "$description may already be registered"
            return 0
        fi
        
        # Check for specific errors and wait accordingly
        if grep -q "ancient birth block" /tmp/register_$hotkey.log; then
            echo "Got 'ancient birth block' error, waiting longer before retry..."
            sleep 10
        else
            echo "Registration failed, retrying in 5 seconds..."
            sleep 5
        fi
        
        retry_count=$((retry_count + 1))
    done
    
    echo -e "${RED}❌ Failed to register $description after $max_retries attempts${NC}"
    return 1
}

# Register neurons
echo -e "${YELLOW}Registering miner neurons on subnet $NETUID...${NC}"
echo "Note: The subnet owner (Alice/default) is automatically registered as UID 0 (validator)"

# Add a small delay after subnet creation to ensure it's fully initialized
sleep 5

# Register miner neurons with retry logic
M1_SUCCESS=0
M2_SUCCESS=0

if register_neuron "M1" "miner M1"; then
    M1_SUCCESS=1
fi

if register_neuron "M2" "miner M2"; then
    M2_SUCCESS=1
fi

# Check if all registrations succeeded
if [ $M1_SUCCESS -eq 1 ] && [ $M2_SUCCESS -eq 1 ]; then
    echo -e "${GREEN}✅ All miner neurons registered successfully${NC}"
    echo "Validator (Alice/default) is already registered as UID 0"
else
    echo -e "${YELLOW}⚠️  Some miners may have failed to register${NC}"
    echo "M1: $([ $M1_SUCCESS -eq 1 ] && echo '✅' || echo '❌')"
    echo "M2: $([ $M2_SUCCESS -eq 1 ] && echo '✅' || echo '❌')"
fi

# Start the subnet's emission schedule
echo -e "${YELLOW}Starting subnet emission schedule...${NC}"
btcli subnet start --netuid $NETUID --subtensor.network local --subtensor.chain_endpoint $CHAIN_ENDPOINT \
    --wallet.path "$WALLET_PATH" --wallet.name Alice --wallet.hotkey default --no_prompt || echo "Subnet emission may already be started"

echo -e "${GREEN}✅ Subnet emission schedule started${NC}"

# Start GRAIL services
echo -e "${YELLOW}Starting GRAIL miners and validator...${NC}"
docker compose -f docker/docker-compose.local-subnet.yml up -d miner-1 miner-2 validator

echo -e "${GREEN}✅ All services started!${NC}"

echo ""
echo "==================================================================="
echo -e "${GREEN}Local subnet is ready!${NC}"
echo "==================================================================="
echo ""
echo "Subtensor endpoints:"
echo "  - Alice node: ws://localhost:9944"
echo "  - Bob node: ws://localhost:9945"
echo ""
echo "MinIO (S3) console: http://localhost:9001"
echo "  - Username: minioadmin"
echo "  - Password: minioadmin"
echo ""
echo "Monitor logs:"
echo "  All services:  docker compose -f docker/docker-compose.local-subnet.yml logs -f"
echo "  Miner 1 only:  docker compose -f docker/docker-compose.local-subnet.yml logs -f miner-1"
echo "  Validator:     docker compose -f docker/docker-compose.local-subnet.yml logs -f validator"
echo ""
echo "Rebuild and restart services (after code changes):"
echo "  All GRAIL:     docker compose -f docker/docker-compose.local-subnet.yml up -d --build miner-1 miner-2 validator"
echo "  Miner 1 only:  docker compose -f docker/docker-compose.local-subnet.yml up -d --build miner-1"
echo "  Validator:     docker compose -f docker/docker-compose.local-subnet.yml up -d --build validator"
echo ""docker compose -nf docker-compose.local-subnet.yml logs -f miner-1
echo "Stop everything:"
echo "  docker compose -f docker/docker-compose.local-subnet.yml down"
echo ""
echo "Clean up volumes:"
echo "  docker compose -f docker/docker-compose.local-subnet.yml down -v"
echo "==================================================================="