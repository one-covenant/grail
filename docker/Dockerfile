FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# devel image provides nvcc/ptxas needed by Triton for kernel compilation.
# System dependencies only (no Python from apt, uv manages it).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git git-lfs build-essential && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Install uv, then let uv install Python 3.12 (Ubuntu 22.04 only ships 3.10).
# This matches the CI workflow and avoids needing the deadsnakes PPA.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv --version
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.12

WORKDIR /app

# Enable NVIDIA runtime and configure HF/Transformers defaults
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock /app/

# Install dependencies without building the project itself.
# The project uses dynamic versioning from grail/__init__.py which isn't
# copied yet, so --no-install-project skips the project build while still
# caching all dependencies in this layer.
RUN uv sync --no-install-project && \
    uv cache clean && \
    rm -rf /root/.cache/pip /root/.cache/uv && \
    find /root/.local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code then install the project (fast, deps already cached)
COPY grail /app/grail
COPY README.md /app/README.md
RUN uv sync --no-editable && \
    uv cache clean

# Flash Attention 2: required for consistent proof verification across
# miner and validator. ATTN_IMPLEMENTATION is pinned to "flash_attention_2"
# in grail/protocol/constants.py and provider.py raises at startup if the
# import fails. Install the prebuilt cu12+torch2.9+cp312+cxx11abiTRUE wheel
# from Dao-AILab releases (~254 MB, ~10 s install) instead of building from
# source (~30-60 min per CI run).
#
# Compute-capability coverage of this single wheel (Dao-AILab cu12 standard):
#   sm_80 (A100, A10G), sm_86 (A40, RTX 30xx), sm_89 (L40, L40S, RTX 40xx),
#   sm_90 (H100, H200), sm_100 (B200 - empirically verified on
#   basilica-grail-trainer at compute capability 10.0).
#   Pre-Ampere GPUs (V100/sm_70, T4/sm_75, RTX 20xx/sm_75) are NOT covered;
#   miners on those cards must build flash-attn from source.
#
# The cxx11abi must match torch's; PyTorch >= 2.5 Linux wheels are
# cxx11abi=TRUE by default. Pinned by SHA256 for supply-chain safety. Bump
# FLASH_ATTN_VERSION (and the wheel filename + SHA256) when upgrading
# torch, since the wheel name encodes the matching torch minor.
ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_WHEEL=flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
ARG FLASH_ATTN_SHA256=4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810
RUN mkdir -p /tmp/wheels && \
    curl -LsSf -o "/tmp/wheels/${FLASH_ATTN_WHEEL}" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${FLASH_ATTN_WHEEL}" && \
    echo "${FLASH_ATTN_SHA256}  /tmp/wheels/${FLASH_ATTN_WHEEL}" | sha256sum -c - && \
    uv pip install "/tmp/wheels/${FLASH_ATTN_WHEEL}" && \
    rm -rf /tmp/wheels && \
    uv cache clean

# Verify grail's import surface in the base image. This catches any
# silent dep downgrade introduced by future flash-attn/torch pins.
RUN /app/.venv/bin/python -c "\
import torch, transformers, grail; \
assert torch.__version__.startswith('2.9.1'), f'torch={torch.__version__}'; \
assert transformers.__version__ == '4.57.1', f'transformers={transformers.__version__}'; \
print('grail base image OK: torch=', torch.__version__, 'transformers=', transformers.__version__)"

# Validator-only base image. The pipelined miner needs SGLang too;
# build the miner image on top of this one via docker/Dockerfile.miner.
#
# Entrypoint: --no-sync so uv does NOT re-resolve the venv on every
# container start (which would silently undo any post-uv-sync installs
# such as the flash-attn wheel and the future SGLang layer).
ENTRYPOINT ["uv", "run", "--no-sync", "grail"]
