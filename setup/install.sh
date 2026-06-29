#!/usr/bin/env bash
set -euo pipefail

ENTRPI_DIR="${ENTRPI_DIR:-$HOME/cogni-brain}"
ENTRPI_REPO="${ENTRPI_REPO:-https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark.git}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
ENTRPI_DEFAULT_CONTAINER="${ENTRPI_DEFAULT_CONTAINER:-qwen-spark}"

echo "=== DGX Spark Cogni-Brain setup ==="
echo "  Entrpi runtime:     $ENTRPI_DIR"
echo "  Entrpi repo:        $ENTRPI_REPO"
echo "  Served model name:  $SERVED_MODEL_NAME"
echo "  Container name:     $CONTAINER_NAME"
echo "  Profile:            dense (hybrid INT4+FP8 + int8 lm-head + DFlash n=12)"
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

echo "[3/5] Checking git and uvx..."
command -v git >/dev/null 2>&1 || {
  echo "ERROR: git is not installed or not on PATH."
  exit 1
}
if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
git --version
uvx --version

echo "[4/5] Checking Hugging Face auth..."
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Export it before launch if model access requires auth."
fi
uvx hf auth whoami || {
  echo "WARNING: Hugging Face CLI auth check failed."
  echo "Run: uvx hf auth login"
}

echo "[5/6] Preparing Entrpi DFlash runtime and downloading models..."
if [[ ! -d "$ENTRPI_DIR/.git" ]]; then
  echo "Cloning $ENTRPI_REPO ..."
  git clone "$ENTRPI_REPO" "$ENTRPI_DIR"
else
  echo "Found existing Entrpi checkout: $ENTRPI_DIR"
fi

# Patch the served model name in runtime/serve.sh if needed.
SERVE_SCRIPT="$ENTRPI_DIR/runtime/serve.sh"
if [[ -f "$SERVE_SCRIPT" ]]; then
  if grep -q -- "--served-model-name qwen" "$SERVE_SCRIPT"; then
    sed -i "s/--served-model-name qwen/--served-model-name \"$SERVED_MODEL_NAME\"/g" "$SERVE_SCRIPT"
    echo "Patched runtime/serve.sh served model name to $SERVED_MODEL_NAME."
  elif grep -q -- "--served-model-name \"$SERVED_MODEL_NAME\"" "$SERVE_SCRIPT" || grep -q -- "--served-model-name $SERVED_MODEL_NAME" "$SERVE_SCRIPT"; then
    echo "runtime/serve.sh already serves $SERVED_MODEL_NAME."
  else
    echo "WARNING: Could not find the expected served-model-name line in runtime/serve.sh."
    echo "Check $SERVE_SCRIPT before starting the runtime."
  fi
else
  echo "WARNING: $SERVE_SCRIPT not found. The Entrpi checkout layout may have changed."
fi

# Verify the Entrpi runtime will use nspec=12 by default.
# docker/start.sh always passes --nspec 12 explicitly, so this is a belt-and-
# suspenders check that warns if upstream changed the default in runtime/serve.sh.
# It does NOT patch the file — the explicit --nspec flag from docker/start.sh wins.
if [[ -f "$SERVE_SCRIPT" ]]; then
  NSPEC_DEFAULT=$(grep 'NSPEC="' "$SERVE_SCRIPT" | grep -oE '[0-9]+' | head -1 || true)
  if [[ -n "$NSPEC_DEFAULT" && "$NSPEC_DEFAULT" != "12" ]]; then
    echo "WARNING: runtime/serve.sh NSPEC default is $NSPEC_DEFAULT, not 12."
    echo "  docker/start.sh will still pass --nspec 12 explicitly."
    echo "  If you are launching the Entrpi runtime by hand, use: --nspec 12"
  else
    echo "runtime/serve.sh nspec default: ${NSPEC_DEFAULT:-ok (--nspec passed explicitly by docker/start.sh)}."
  fi
fi

# [6/6] Patch default container name references from qwen-spark to spark-brain.
echo "[6/6] Patching Entrpi checkout container name and downloading models..."
# Exclude .bak files so repeated runs do not find stale backup copies.
if grep -RIl --exclude="*.bak" "$ENTRPI_DEFAULT_CONTAINER" "$ENTRPI_DIR" >/dev/null 2>&1; then
  grep -RIl --exclude="*.bak" "$ENTRPI_DEFAULT_CONTAINER" "$ENTRPI_DIR" 2>/dev/null | while IFS= read -r file; do
    sed -i "s/$ENTRPI_DEFAULT_CONTAINER/$CONTAINER_NAME/g" "$file"
  done
  echo "Patched Entrpi checkout container references to $CONTAINER_NAME."
fi

# Download the primary model and drafter (idempotent — skips if cached).
bash "$(dirname "$0")/download_model.sh"

echo
echo "Setup complete."
echo "Status: bash docker/status.sh"
echo "Start:  bash docker/start.sh"
