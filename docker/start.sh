#!/usr/bin/env bash
set -euo pipefail

ENTRPI_DIR="${ENTRPI_DIR:-$HOME/cogni-brain}"
IMAGE="${IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix}"
MODEL_ID="${MODEL_ID:-bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid}"
DRAFTER_MODEL_ID="${DRAFTER_MODEL_ID:-z-lab/Qwen3.5-122B-A10B-DFlash}"
CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
# Profile and speculative tokens — must be explicit; Entrpi upstream default
# profile is 'dflash' (not 'dense') and nspec defaults vary by profile. 'dense'
# uses the hybrid INT4+FP8 model with int8 lm-head and DFlash, which is the
# configuration that produces the validated 54.44 tok/s result.
PROFILE="${PROFILE:-dense}"
NSPEC="${NSPEC:-12}"

echo "=== Cogni-Brain DFlash dense launch ==="
echo "  Entrpi runtime:         $ENTRPI_DIR"
echo "  Image:                  $IMAGE"
echo "  Profile:                $PROFILE"
echo "  Model (hybrid):         $MODEL_ID"
echo "  Drafter:                $DRAFTER_MODEL_ID"
echo "  Speculative tokens:     $NSPEC"
echo "  Container:              $CONTAINER_NAME"
echo "  Port:                   $PORT"
echo "  Max model length:       $MAX_MODEL_LEN"
echo "  Max num seqs:           $MAX_NUM_SEQS"
echo "  Max num batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo

if [[ ! -x "$ENTRPI_DIR/install.sh" ]]; then
  echo "ERROR: $ENTRPI_DIR/install.sh not found or not executable."
  echo "Run: bash setup/install.sh"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. The container may fail if model access requires auth."
  echo
fi

EXTRA_ARGS=("$@")

# Idempotent: remove any existing container with the same name before starting.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container $CONTAINER_NAME ..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm   "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

echo "Starting Entrpi runtime..."
(
  cd "$ENTRPI_DIR"
  REPO_DIR="$ENTRPI_DIR" \
  QWEN_IMAGE="$IMAGE" \
  HYBRID_REPO="$MODEL_ID" \
  DRAFT_REPO="$DRAFTER_MODEL_ID" \
  NAME="$CONTAINER_NAME" \
  PORT="$PORT" \
  CTX="$MAX_MODEL_LEN" \
  GPU_MEM="$GPU_MEMORY_UTILIZATION" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  HF_HOME="$HF_CACHE_DIR" \
  ./install.sh --start --no-pull --no-download \
    --profile "$PROFILE" \
    --nspec "$NSPEC" \
    "${EXTRA_ARGS[@]}"
)

echo
echo "Container started."
echo "Logs:        docker logs -f $CONTAINER_NAME"
echo "Health:      curl -sf http://localhost:$PORT/health && echo OK"
echo "Model list:  curl -sf http://localhost:$PORT/v1/models"
