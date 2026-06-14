#!/usr/bin/env bash
set -euo pipefail

SPARK_VLLM_DIR="${SPARK_VLLM_DIR:-../spark-vllm-docker}"
MODEL_ID="${MODEL_ID:-Intel/Qwen3.5-122B-A10B-int4-AutoRound}"

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

(
  cd "$SPARK_VLLM_DIR"
  ./hf-download.sh "$MODEL_ID"
)

echo
echo "Download complete."
echo "Next: bash docker/start.sh"
