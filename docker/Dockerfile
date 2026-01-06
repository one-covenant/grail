FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git git-lfs build-essential && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Install uv and add to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv --version
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Enable NVIDIA runtime and configure HF/Transformers defaults
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock /app/

# Sync runtime dependencies only and clean up in same layer to reduce size
RUN uv sync && \
    uv cache clean && \
    rm -rf /root/.cache/pip /root/.cache/uv && \
    find /root/.local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code (separate layer for better caching)
COPY grail /app/grail
COPY README.md /app/README.md

# # Default environment suitable for containers
# ENV WANDB_MODE=online \
#     GRAIL_MONITORING_BACKEND=wandb \
#     HF_HUB_DISABLE_TELEMETRY=1

# Entrypoint delegates to uv
ENTRYPOINT ["uv", "run", "grail"]
