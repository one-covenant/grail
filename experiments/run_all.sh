#!/bin/bash

# GRAIL Experiments Runner Script
# This script sets up the environment and runs all experiments

echo "======================================"
echo "GRAIL Security Experiments Runner"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create results directory if it doesn't exist
mkdir -p results

# Check GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run all experiments
echo ""
echo "Starting experiments..."
echo "======================================"
python run.py --all --verbose

# Generate summary
echo ""
echo "Generating summary report..."
python run.py --summary

echo ""
echo "======================================"
echo "All experiments completed!"
echo "Results saved in: ./results/"
echo "======================================"