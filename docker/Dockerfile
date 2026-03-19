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

# Install uv, then let uv install Python 3.12.
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

# Flash Attention 2: required for consistent proof verification across
# miner and validator. Built from source using nvcc from the devel base image.
# Targets A100 (sm_80), L40S/4090 (sm_89), H100 (sm_90).
RUN TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" MAX_JOBS=4 \
    uv pip install "flash-attn>=2.7.0" --no-build-isolation && \
    uv cache clean

# Copy application code then install the project (fast, deps already cached)
COPY grail /app/grail
COPY README.md /app/README.md
RUN uv sync --no-editable && \
    uv cache clean

# Entrypoint delegates to uv
ENTRYPOINT ["uv", "run", "grail"]
