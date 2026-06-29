#!/usr/bin/env bash
set -euo pipefail

SPARK_VLLM_DIR="${SPARK_VLLM_DIR:-../spark-vllm-docker}"
RECIPE="${RECIPE:-qwen3.5-122b-int4-autoround}"
CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
PORT="${PORT:-8000}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}}"

echo "=== AutoRound baseline spark-vllm-docker launch ==="
echo "  spark-vllm-docker: $SPARK_VLLM_DIR"
echo "  Recipe:            $RECIPE"
echo "  Container:         $CONTAINER_NAME"
echo "  Served name:       $SERVED_MODEL_NAME"
echo "  Port:              $PORT"
echo "  Speculative config:$SPECULATIVE_CONFIG"
echo

if [[ ! -x "$SPARK_VLLM_DIR/run-recipe.sh" ]]; then
  echo "ERROR: $SPARK_VLLM_DIR/run-recipe.sh not found or not executable."
  echo "Run: bash setup/autoround_install.sh"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. The container may fail if model access requires auth."
  echo
fi

EXTRA_ARGS=("$@")
if [[ "${EXTRA_ARGS[0]:-}" == "--" ]]; then
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

HF_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ENV=(-e "HF_TOKEN=${HF_TOKEN}")
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Cleaning up existing container..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

echo "Starting recipe..."
(
  cd "$SPARK_VLLM_DIR"
  ./run-recipe.sh "$RECIPE" --solo \
    -d --name "$CONTAINER_NAME" \
    -p "$PORT:8000" \
    "${HF_ENV[@]}" \
    -- \
    --served-model-name "$SERVED_MODEL_NAME" \
    --speculative-config "$SPECULATIVE_CONFIG" \
    "${EXTRA_ARGS[@]}"
)

echo
echo "Container started."
echo "Logs:        docker logs -f $CONTAINER_NAME"
echo "Health:      curl -sf http://localhost:$PORT/health && echo OK"
echo "Model list:  curl -sf http://localhost:$PORT/v1/models"
