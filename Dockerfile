FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y && \
    /root/.cargo/bin/uv --version

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock /app/
COPY grail /app/grail
COPY README.md /app/README.md

# Sync dependencies (all extras to include dev if needed)
RUN /root/.cargo/bin/uv sync --all-extras

# Default environment suitable for containers
ENV WANDB_MODE=disabled \
    GRAIL_MONITORING_BACKEND=null \
    HF_HUB_DISABLE_TELEMETRY=1

# Entrypoint delegates to uv
ENTRYPOINT ["/root/.cargo/bin/uv", "run", "python", "-m", "grail"]


