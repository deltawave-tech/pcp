#!/bin/bash
set -euo pipefail

CONFIG_FILE=${1:-experiments/inference_template.json}
BACKEND=${BACKEND:-cpu}
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-8000}
CONTROL_HOST=${CONTROL_HOST:-127.0.0.1}
CONTROL_PORT=${CONTROL_PORT:-8091}
GATEWAY_HOST=${GATEWAY_HOST:-127.0.0.1}
GATEWAY_PORT=${GATEWAY_PORT:-8010}
GATEWAY_CONFIG=${GATEWAY_CONFIG:-/tmp/pcp_inference_smoke_gateway.json}
PCP_GATEWAY_API_TOKEN=${PCP_GATEWAY_API_TOKEN:-${PCP_API_TOKEN:-dev}}
PCP_GATEWAY_INTERNAL_TOKEN=${PCP_GATEWAY_INTERNAL_TOKEN:-dev-internal}

cleanup() {
  echo ""
  echo "Shutting down inference gateway cluster..."
  if [ -n "${GATEWAY_PID:-}" ]; then kill $GATEWAY_PID 2>/dev/null || true; fi
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

cat >"$GATEWAY_CONFIG" <<EOF
{
  "gateway_id": "inference-smoke-gateway",
  "lab_id": "local-smoke",
  "graph_backend": "memory",
  "api_token_env": "PCP_GATEWAY_API_TOKEN",
  "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
  "worker_fabric": {
    "host": "${CONTROL_HOST}",
    "port": ${CONTROL_PORT}
  },
  "api": {
    "host": "${GATEWAY_HOST}",
    "port": ${GATEWAY_PORT},
    "token_env": "PCP_GATEWAY_API_TOKEN",
    "internal_token_env": "PCP_GATEWAY_INTERNAL_TOKEN"
  },
  "controllers": {
    "inference": {
      "enabled": true,
      "config_path": "${CONFIG_FILE}",
      "service_id": "inference-main",
      "api": {
        "host": "${API_HOST}",
        "port": ${API_PORT}
      }
    }
  }
}
EOF

env PCP_GATEWAY_API_TOKEN="$PCP_GATEWAY_API_TOKEN" \
  PCP_GATEWAY_INTERNAL_TOKEN="$PCP_GATEWAY_INTERNAL_TOKEN" \
  $EXE --gateway \
  --gateway-config "$GATEWAY_CONFIG" \
  --gateway-host "$GATEWAY_HOST" --gateway-port "$GATEWAY_PORT" \
  > /tmp/inference_gateway.log 2>&1 &
GATEWAY_PID=$!

sleep 2

$EXE --worker \
  --connect "$CONTROL_HOST:$CONTROL_PORT" \
  --backend "$BACKEND" \
  > /tmp/inference_worker.log 2>&1 &
WORKER_PID=$!

echo "Gateway PID: $GATEWAY_PID"
echo "Worker PID: $WORKER_PID"

echo "Checking healthz..."
curl -s "http://$API_HOST:$API_PORT/healthz" || true

echo "Checking readyz..."
curl -s "http://$API_HOST:$API_PORT/readyz" || true

wait $GATEWAY_PID
cleanup
