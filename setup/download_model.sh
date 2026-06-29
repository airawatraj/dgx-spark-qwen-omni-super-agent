#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid}"
DRAFTER_MODEL_ID="${DRAFTER_MODEL_ID:-z-lab/Qwen3.5-122B-A10B-DFlash}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

repo_cache_dir() {
  local repo_id="$1"
  local repo_dir="models--${repo_id//\//--}"

  if [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
    printf '%s\n' "$HUGGINGFACE_HUB_CACHE/$repo_dir"
    return
  fi
  if [[ -n "${HF_HOME:-}" ]]; then
    printf '%s\n' "$HF_HOME/hub/$repo_dir"
    return
  fi
  printf '%s\n' "$HOME/.cache/huggingface/hub/$repo_dir"
}

repo_cache_exists() {
  local repo_id="$1"
  local candidate
  candidate="$(repo_cache_dir "$repo_id")"
  if [[ -d "$candidate" ]] && find "$candidate" -type f -print -quit | grep -q .; then
    echo "$candidate"
    return 0
  fi
  return 1
}

download_repo() {
  local repo_id="$1"
  local label="$2"

  echo
  echo "=== Downloading $label ==="
  echo "  Repo: $repo_id"

  if [[ "$FORCE_DOWNLOAD" != "1" ]]; then
    if found_dir="$(repo_cache_exists "$repo_id")"; then
      echo "Already present in Hugging Face cache:"
      echo "  $found_dir"
      return
    fi
  fi

  uvx hf download "$repo_id"
}

echo "=== Downloading primary DFlash dense model assets ==="
echo "  Model:   $MODEL_ID"
echo "  Drafter: $DRAFTER_MODEL_ID"
echo

if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Download may fail if the model requires auth."
fi

download_repo "$MODEL_ID" "hybrid INT4+FP8 model"
download_repo "$DRAFTER_MODEL_ID" "DFlash drafter"

echo
echo "Download check complete."
echo "Next: bash docker/start.sh"
