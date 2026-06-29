#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
PORT="${PORT:-8000}"

echo "=== AutoRound baseline vLLM status ==="
echo

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  STARTED_AT=$(docker inspect "$CONTAINER_NAME" --format '{{.State.StartedAt}}')
  echo "  Container: running (started $STARTED_AT)"
else
  echo "  Container: NOT RUNNING"
fi

if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "  API:       healthy (http://localhost:$PORT)"
else
  echo "  API:       not reachable"
fi

echo
echo "=== Models ==="
curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null || echo "  Models endpoint not available"

echo
echo
echo "=== Metrics snapshot ==="
METRICS="$(curl -sf "http://localhost:$PORT/metrics" 2>/dev/null || true)"
if [[ -n "$METRICS" ]]; then
  printf '%s\n' "$METRICS" | awk '
    !/^#/ && /gpu_cache_usage_perc/ {printf "  KV cache used: %.1f%%\n", $2 * 100}
    !/^#/ && /num_requests_running/ {print "  Requests running: " $2}
    !/^#/ && /num_requests_waiting/ {print "  Requests waiting: " $2}
  '
else
  echo "  Metrics endpoint not available"
fi

echo
echo "=== Recent logs ==="
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker logs "$CONTAINER_NAME" --tail 10 2>&1 | sed 's/^/  /'
else
  echo "  No container found"
fi
