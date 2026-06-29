#!/usr/bin/env bash
set -euo pipefail

SPARK_VLLM_DIR="${SPARK_VLLM_DIR:-../spark-vllm-docker}"
SPARK_VLLM_REPO="${SPARK_VLLM_REPO:-https://github.com/eugr/spark-vllm-docker.git}"
BUILD_ON_INSTALL="${BUILD_ON_INSTALL:-auto}"
FORCE_BUILD="${FORCE_BUILD:-0}"
BUILD_PROFILE="${BUILD_PROFILE:---tf5}"
FRESH_CLONE=0

echo "=== AutoRound baseline setup check ==="
echo "  spark-vllm-docker: $SPARK_VLLM_DIR"
echo

echo "[1/5] Checking Docker..."
docker version --format 'Docker {{.Server.Version}}' >/dev/null
docker version --format '  Server {{.Server.Version}}'

echo "[2/5] Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "WARNING: nvidia-smi is not on PATH."
fi

echo "[3/5] Checking git and uv..."
command -v git >/dev/null 2>&1 || {
  echo "ERROR: git is not installed or not on PATH."
  exit 1
}
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
git --version
uv --version
uvx --version

echo "[4/5] Checking Hugging Face auth..."
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Export it before launch if the model requires auth."
fi
uvx hf auth whoami || {
  echo "WARNING: Hugging Face CLI auth check failed."
  echo "Run: uvx hf auth login"
}

echo "[5/5] Preparing spark-vllm-docker..."
if [[ ! -d "$SPARK_VLLM_DIR/.git" ]]; then
  echo "Cloning $SPARK_VLLM_REPO ..."
  git clone "$SPARK_VLLM_REPO" "$SPARK_VLLM_DIR"
  FRESH_CLONE=1
else
  echo "Found existing checkout: $SPARK_VLLM_DIR"
fi

if [[ "$FORCE_BUILD" == "1" ]]; then
  SHOULD_BUILD=1
elif [[ "$BUILD_ON_INSTALL" == "1" ]]; then
  SHOULD_BUILD=1
elif [[ "$BUILD_ON_INSTALL" == "auto" && "$FRESH_CLONE" == "1" ]]; then
  SHOULD_BUILD=1
else
  SHOULD_BUILD=0
fi

if [[ "$SHOULD_BUILD" == "1" ]]; then
  echo "Building spark-vllm-docker once with profile $BUILD_PROFILE ..."
  (
    cd "$SPARK_VLLM_DIR"
    ./build-and-copy.sh "$BUILD_PROFILE"
  )
else
  echo "Skipping build; existing spark-vllm-docker checkout is being reused."
  echo "Set FORCE_BUILD=1 to rebuild."
fi

echo
echo "AutoRound baseline setup check complete."
echo "Next: bash setup/autoround_download_model.sh"
