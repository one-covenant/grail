# vLLM Server Environment

Isolated Python environment for vLLM inference during GRAIL evaluation.

## Why Separate?

`bittensor==9.12.1` conflicts with vLLM's binary dependencies (CUDA kernels, Triton). Separate environments avoid version resolution deadlocks.

## Quick Setup

```bash
bash scripts/setup_vllm_env.sh
```

Creates `tools/vllm-server/.venv/` with vLLM v0.7+ and compatible torch.

## Usage

Automatically used when `backend="vllm"` in `EvalConfig`. Server runs as subprocess via HTTP.

**Override path**:
```bash
export GRAIL_VLLM_PYTHON=/custom/vllm-env/bin/python
grail train ...
```

## Verify Installation

```bash
tools/vllm-server/.venv/bin/python -c "import vllm; print(vllm.__version__)"
```
