#!/usr/bin/env bash
set -euo pipefail

SPARK_VLLM_DIR="${SPARK_VLLM_DIR:-../spark-vllm-docker}"
MODEL_ID="${MODEL_ID:-Intel/Qwen3.5-122B-A10B-int4-AutoRound}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
MODEL_REPO_DIR="models--${MODEL_ID//\//--}"
HF_HUB_CANDIDATES=()

if [[ -n "${MODEL_DIR:-}" ]]; then
  HF_HUB_CANDIDATES+=("$MODEL_DIR")
fi
if [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
  HF_HUB_CANDIDATES+=("$HUGGINGFACE_HUB_CACHE/$MODEL_REPO_DIR")
fi
if [[ -n "${HF_HOME:-}" ]]; then
  HF_HUB_CANDIDATES+=("$HF_HOME/hub/$MODEL_REPO_DIR")
fi
HF_HUB_CANDIDATES+=("$HOME/.cache/huggingface/hub/$MODEL_REPO_DIR")

model_cache_exists() {
  local candidate
  for candidate in "${HF_HUB_CANDIDATES[@]}"; do
    if [[ -d "$candidate" ]] && find "$candidate" -type f -print -quit | grep -q .; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

echo "=== Downloading Qwen3.5-122B-A10B model ==="
echo "  spark-vllm-docker: $SPARK_VLLM_DIR"
echo "  Model:             $MODEL_ID"
echo

if [[ ! -x "$SPARK_VLLM_DIR/hf-download.sh" ]]; then
  echo "ERROR: $SPARK_VLLM_DIR/hf-download.sh not found or not executable."
  echo "Run: bash setup/install.sh"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Download may fail if the model requires auth."
  echo
fi

if [[ "$FORCE_DOWNLOAD" != "1" ]]; then
  if FOUND_MODEL_DIR="$(model_cache_exists)"; then
    echo "Model cache already exists; skipping download."
    echo "  Found: $FOUND_MODEL_DIR"
    echo "Set FORCE_DOWNLOAD=1 to run hf-download.sh anyway."
    echo
    echo "Next: bash docker/start.sh"
    exit 0
  fi
fi

(
  cd "$SPARK_VLLM_DIR"
  ./hf-download.sh "$MODEL_ID"
)

echo
echo "Download complete."
echo "Next: bash docker/start.sh"
