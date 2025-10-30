#!/bin/bash
# Setup isolated vLLM server environment for GRAIL evaluation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VLLM_DIR="$PROJECT_ROOT/tools/vllm-server"

echo "🔧 Setting up isolated vLLM server environment..."
echo "   Location: $VLLM_DIR"

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create vLLM server directory if needed
mkdir -p "$VLLM_DIR"

# Create and sync environment
cd "$VLLM_DIR"

echo "📦 Creating virtual environment..."
uv venv

echo "📦 Installing vLLM and dependencies..."
uv sync

# Verify installation
echo "✅ Verifying vLLM installation..."
if .venv/bin/python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')" 2>/dev/null; then
    echo "✅ vLLM server environment ready"
    echo ""
    echo "Environment location: $VLLM_DIR/.venv"
    echo "Python executable: $VLLM_DIR/.venv/bin/python"
    echo ""
    echo "The GRAIL trainer will automatically use this environment for vLLM evaluation."
else
    echo "⚠️  Warning: vLLM import failed. Check CUDA/GPU compatibility."
    exit 1
fi

