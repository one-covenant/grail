#!/usr/bin/env bash

set -euo pipefail

# Minimal setup for running Grail locally and via Docker/Compose on Ubuntu/Debian
# - Installs Docker + Compose plugin
# - Installs & configures NVIDIA Container Toolkit (GPU in Docker; 24.04→22.04 repo fallback)
# - Installs uv (for local runs with pyproject)

info() { echo -e "[INFO] $*"; }
warn() { echo -e "[WARN] $*" >&2; }
err() { echo -e "[ERROR] $*" >&2; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    return 1
  fi
}

is_apt_based() {
  if command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

ensure_sudo() {
  if [[ $(id -u) -eq 0 ]]; then
    SUDO=""
  else
    SUDO="sudo"
  fi
}

has_systemd() {
  if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
    return 0
  fi
  return 1
}

install_docker() {
  if require_cmd docker; then
    info "Docker already installed: $(docker --version)"
    return 0
  fi
  if ! is_apt_based; then
    err "Automatic Docker install only supported on apt-based systems. Install Docker manually."
    return 1
  fi

  ensure_sudo
  ${SUDO} apt-get update -y
  ${SUDO} apt-get install -y ca-certificates curl gnupg lsb-release
  ${SUDO} install -m 0755 -d /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
    curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg | ${SUDO} gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    ${SUDO} chmod a+r /etc/apt/keyrings/docker.gpg
  fi
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | ${SUDO} tee /etc/apt/sources.list.d/docker.list >/dev/null

  ${SUDO} apt-get update -y
  ${SUDO} apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  if has_systemd; then
    ${SUDO} systemctl enable --now docker || true
  else
    warn "Systemd not detected; skipping 'systemctl enable --now docker'. Start Docker manually if needed."
  fi
  if command -v usermod >/dev/null 2>&1 && [[ -n "${SUDO}" ]]; then
    ${SUDO} usermod -aG docker "$USER" || true
    warn "Added $USER to docker group. Log out/in for this to take effect."
  fi
  info "Docker installed successfully."
}

install_and_configure_nvidia() {
  ensure_sudo
  if ! require_cmd nvidia-ctk; then
    ${SUDO} mkdir -p /usr/share/keyrings
    local distribution=$(. /etc/os-release; v="$ID$VERSION_ID"; [[ "$v" == "ubuntu24.04" ]] && echo ubuntu22.04 || echo "$v")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | ${SUDO} gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    ${SUDO} apt-get update -y && ${SUDO} apt-get install -y nvidia-container-toolkit
  fi
  ${SUDO} nvidia-ctk runtime configure --runtime=docker || true
  if has_systemd; then ${SUDO} systemctl restart docker || true; else ${SUDO} rm -f /var/run/docker.pid || true; ${SUDO} pkill dockerd || true; nohup dockerd >/tmp/dockerd.log 2>&1 & sleep 3; fi
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q 'nvidia' && info "NVIDIA runtime ready" || warn "NVIDIA runtime not detected"
}

install_uv() {
  if require_cmd uv; then
    info "uv already installed: $(uv --version)"
    return 0
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  # Persist PATH for future shells
  local bashrc="$HOME/.bashrc"
  local profile="$HOME/.profile"
  local zshrc="$HOME/.zshrc"
  local fish_conf="$HOME/.config/fish/config.fish"

  ensure_line() {
    local file="$1"; shift
    local line="$*"
    if [[ -f "$file" ]]; then
      grep -qxF "$line" "$file" || echo "$line" >> "$file"
    else
      echo "$line" >> "$file"
    fi
  }

  ensure_line "$bashrc" "[ -f \"$HOME/.local/bin/env\" ] && source \"$HOME/.local/bin/env\""
  ensure_line "$profile" "[ -f \"$HOME/.local/bin/env\" ] && source \"$HOME/.local/bin/env\""
  ensure_line "$zshrc" "[ -f \"$HOME/.local/bin/env\" ] && source \"$HOME/.local/bin/env\""
  if [[ -d "$(dirname "$fish_conf")" ]]; then
    ensure_line "$fish_conf" "test -f $HOME/.local/bin/env.fish; and source $HOME/.local/bin/env.fish"
  fi

  info "Installed uv: $(uv --version)"
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") setup

Actions performed:
  - Install Docker Engine + Compose plugin
  - Install & configure NVIDIA Docker runtime
  - Install uv (for local runs)

After setup:
  - Build image:    docker build -t grail:local -f Dockerfile .
  - Compose up:     docker compose -f docker-compose.integration.yml --env-file .env up -d --build
  - Compose down:   docker compose -f docker-compose.integration.yml down
  - Install deps:   uv sync
  - Local mine:     uv run grail -vv mine --no-drand
  - Local validate: uv run grail -vv validate --no-drand
USAGE
}

main() {
  local cmd="${1:-setup}"
  case "${cmd}" in
    setup)
      install_docker
      install_and_configure_nvidia
      install_uv
      info "Setup complete. You may need to re-login for docker group changes."
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage
      ;;
  esac
}

main "$@"


