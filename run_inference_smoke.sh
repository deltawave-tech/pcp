#!/bin/bash
set -euo pipefail

CONFIG_FILE=${1:-experiments/inference_template.json}
BACKEND=${BACKEND:-cpu}
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-8000}
CONTROL_HOST=${CONTROL_HOST:-127.0.0.1}
CONTROL_PORT=${CONTROL_PORT:-8091}

cleanup() {
  echo ""
  echo "Shutting down inference cluster..."
  if [ -n "${CTRL_PID:-}" ]; then kill $CTRL_PID 2>/dev/null || true; fi
  if [ -n "${WORKER_PID:-}" ]; then kill $WORKER_PID 2>/dev/null || true; fi
}
trap cleanup SIGINT SIGTERM

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file not found at $CONFIG_FILE"
  exit 1
fi

# Build binary if needed
if [ ! -x "./result/bin/pcp" ]; then
  echo "Building PCP via nix..."
  nix build
fi

EXE="./result/bin/pcp"

# Basic config sanity check
if command -v jq >/dev/null 2>&1; then
  WEIGHTS_PATH=$(jq -r '.weights_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
  VMFB_PATH=$(jq -r '.generation_vmfb_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
  if [ -z "$WEIGHTS_PATH" ] || [ -z "$VMFB_PATH" ]; then
    echo "Error: weights_path and generation_vmfb_path must be set in $CONFIG_FILE"
    exit 1
  fi
  if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Error: weights file not found at $WEIGHTS_PATH"
    exit 1
  fi
  if [ ! -f "$VMFB_PATH" ]; then
    echo "Error: VMFB file not found at $VMFB_PATH"
    exit 1
  fi
else
  echo "Warning: jq not found; skipping config path validation"
fi

if [ -z "${PCP_API_TOKEN:-}" ]; then
  echo "Error: PCP_API_TOKEN env var must be set for API auth"
  exit 1
fi

$EXE --inference \
  --inference-config "$CONFIG_FILE" \
  --api-host "$API_HOST" --api-port "$API_PORT" \
  --control-host "$CONTROL_HOST" --control-port "$CONTROL_PORT" \
  > /tmp/inference_controller.log 2>&1 &
CTRL_PID=$!

sleep 2

$EXE --worker \
  --connect "$CONTROL_HOST:$CONTROL_PORT" \
  --backend "$BACKEND" \
  > /tmp/inference_worker.log 2>&1 &
WORKER_PID=$!

echo "Controller PID: $CTRL_PID"
echo "Worker PID: $WORKER_PID"

echo "Checking healthz..."
curl -s "http://$API_HOST:$API_PORT/healthz" || true

echo "Checking readyz..."
curl -s "http://$API_HOST:$API_PORT/readyz" || true

wait $CTRL_PID
cleanup
