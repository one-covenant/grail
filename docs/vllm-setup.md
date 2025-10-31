# vLLM Evaluation Backend

This guide covers using vLLM for inference-accelerated evaluation in GRAIL.

## Overview

- **Main environment** (`.venv/`): GRAIL training with bittensor
- **vLLM environment** (`tools/vllm-server/.venv/`): Isolated inference server

vLLM runs as a subprocess, communicating via HTTP. This avoids dependency conflicts with bittensor.

## Setup

```bash
bash scripts/setup_vllm_env.sh
```

Creates isolated vLLM v0.7+ environment with compatible torch.

## Configuration

```python
backend = "vllm"              # Use vLLM backend
sglang_start_server = True    # Auto-start server
sglang_host = "127.0.0.1"
sglang_port = 30000
sglang_server_timeout_s = 120.0
```

Or override path:
```bash
export GRAIL_VLLM_PYTHON=/custom/vllm-env/bin/python
grail train ...
```

## How It Works

1. Trainer saves model checkpoint (temp directory)
2. Trainer frees GPU memory
3. vLLM server spawns using isolated environment
4. Evaluator sends prompts via HTTP to OpenAI-compatible API
5. Server terminates, training models reload

## Troubleshooting

**Server fails to start**
```bash
tools/vllm-server/.venv/bin/python -c "import vllm; print(vllm.__version__)"
```

**Executable not found**
- Run: `bash scripts/setup_vllm_env.sh`

**CUDA errors**
- Verify GPU: `nvidia-smi`
- Check env: `GRAIL_VLLM_PYTHON=.../bin/python python -m torch.utils.collect_env`

**Module not found: vllm.entrypoints.openai.api_server**
- Check vLLM version in isolated environment
- Update if needed: `cd tools/vllm-server && .venv/bin/pip install --upgrade vllm`

## Performance Comparison

| Backend | Throughput | Setup | Conflict |
|---------|-----------|-------|----------|
| HF | 1x | None | None |
| SGLang | 3–5x | Medium | None (included) |
| vLLM | 5–10x | Complex | High (requires isolation) |

**Default**: SGLang (no extra setup, built-in to main env)

**Use vLLM when**: You need maximum throughput for large evaluation sets.

## Advanced

### Custom vLLM Version

```bash
cd tools/vllm-server
# Edit pyproject.toml: vllm>=0.7.5
uv sync
```

### Long-running Server

```python
backend = "vllm"
sglang_start_server = False    # Don't spawn subprocess
sglang_host = "10.0.1.50"      # Remote vLLM server
sglang_port = 8000
```

### Deployment via Environment Variable

```bash
# Pre-built vLLM environment at /opt/vllm
export GRAIL_VLLM_PYTHON=/opt/vllm/bin/python
grail train ...
```

## File Structure

```
grail/
├── .venv/                          # Main (bittensor + training)
├── tools/vllm-server/
│   ├── pyproject.toml              # vLLM dependencies (tracked)
│   ├── .venv/                      # Isolated env (gitignored)
│   └── README.md
├── scripts/setup_vllm_env.sh       # Auto-setup
├── grail/trainer/
│   ├── config.py                   # EvalConfig (vllm_python_executable)
│   └── evaluator.py                # HTTP client
└── grail/neurons/trainer.py        # Server spawn
```
