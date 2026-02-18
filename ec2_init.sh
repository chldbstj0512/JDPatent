#!/usr/bin/env bash
set -euo pipefail

# JDPatent EC2 bootstrap
# Usage:
#   bash ec2_init.sh
#   bash ec2_init.sh --skip-up

SKIP_UP="false"
for arg in "$@"; do
  case "$arg" in
    --skip-up) SKIP_UP="true" ;;
    *)
      echo "[ERROR] Unknown argument: $arg"
      exit 1
      ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[INFO] Project dir: $PROJECT_DIR"

if ! command -v sudo >/dev/null 2>&1; then
  echo "[ERROR] sudo is required."
  exit 1
fi

install_docker_ubuntu() {
  sudo apt-get update -y
  sudo apt-get install -y docker.io docker-compose-plugin
  sudo systemctl enable docker
  sudo systemctl start docker
}

install_docker_amazon() {
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf update -y
    sudo dnf install -y docker
  else
    sudo yum update -y
    sudo yum install -y docker
  fi
  sudo systemctl enable docker
  sudo systemctl start docker

  if ! docker compose version >/dev/null 2>&1; then
    echo "[INFO] docker compose plugin not found. Installing plugin..."
    DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"
    mkdir -p "$DOCKER_CONFIG/cli-plugins"
    ARCH="$(uname -m)"
    case "$ARCH" in
      x86_64) BIN_ARCH="x86_64" ;;
      aarch64|arm64) BIN_ARCH="aarch64" ;;
      *)
        echo "[ERROR] Unsupported architecture: $ARCH"
        exit 1
        ;;
    esac
    curl -SL "https://github.com/docker/compose/releases/download/v2.30.3/docker-compose-linux-${BIN_ARCH}" \
      -o "$DOCKER_CONFIG/cli-plugins/docker-compose"
    chmod +x "$DOCKER_CONFIG/cli-plugins/docker-compose"
  fi
}

if ! command -v docker >/dev/null 2>&1; then
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    case "${ID:-}" in
      ubuntu|debian) install_docker_ubuntu ;;
      amzn|amazon) install_docker_amazon ;;
      *)
        echo "[ERROR] Unsupported OS: ${ID:-unknown}"
        exit 1
        ;;
    esac
  else
    echo "[ERROR] Cannot detect OS."
    exit 1
  fi
else
  echo "[INFO] Docker already installed."
  sudo systemctl enable docker || true
  sudo systemctl start docker || true
fi

if ! groups "$USER" | grep -q docker; then
  echo "[INFO] Adding $USER to docker group."
  sudo usermod -aG docker "$USER"
  echo "[WARN] Re-login may be required to use docker without sudo."
fi

if [ ! -f ".env" ]; then
  if [ -f ".env.api.example" ]; then
    cp .env.api.example .env
    echo "[INFO] Created .env from .env.api.example"
    echo "[WARN] Fill OPENAI/PINECONE keys in .env before running production workload."
  else
    echo "[ERROR] .env not found and .env.api.example missing."
    exit 1
  fi
else
  echo "[INFO] Using existing .env"
fi

echo "[INFO] Docker: $(docker --version)"
echo "[INFO] Compose: $(docker compose version)"

if [ "$SKIP_UP" = "true" ]; then
  echo "[INFO] Skip service startup (--skip-up). Done."
  exit 0
fi

echo "[INFO] Building and starting containers..."
docker compose up -d --build

echo "[INFO] Done."
echo "[INFO] Health check: curl http://localhost:8001/healthz"

