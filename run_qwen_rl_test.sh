#!/bin/bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-experiments/qwen_rl_test.json}"
WORKER_BACKEND="${WORKER_BACKEND:-cuda}"
CONTROL_HOST="${CONTROL_HOST:-127.0.0.1}"
CONTROL_PORT="${CONTROL_PORT:-8090}"
CONTROLLER_API_HOST="${CONTROLLER_API_HOST:-127.0.0.1}"
CONTROLLER_API_PORT="${CONTROLLER_API_PORT:-8091}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-18010}"
GATEWAY_CONFIG="${GATEWAY_CONFIG:-/tmp/pcp_qwen_rl_gateway.json}"
PCP_GATEWAY_API_TOKEN="${PCP_GATEWAY_API_TOKEN:-dev}"
PCP_GATEWAY_INTERNAL_TOKEN="${PCP_GATEWAY_INTERNAL_TOKEN:-dev-internal}"
STATUS_JSON="${STATUS_JSON:-/tmp/pcp_qwen_rl_status.json}"

cleanup() {
    set +e
    echo ""
    echo "Shutting down gateway RL cluster..."
    if [ -n "${WORKER_PID:-}" ]; then
        kill "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi
    if [ -n "${GATEWAY_PID:-}" ]; then
        kill "$GATEWAY_PID" 2>/dev/null || true
        wait "$GATEWAY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

if [ ! -f "models/qwen_rl_generation.mlir" ]; then
    echo "Error: Generation model not found at models/qwen_rl_generation.mlir"
    echo "Run: python tools/export_qwen_rl_generation.py"
    exit 1
fi

if [ ! -f "models/qwen_grpo_training.mlir" ] && [ ! -f "models/qwen_grpo_training.cuda.vmfb" ]; then
    echo "Error: Training model not found"
    echo "Run: python tools/export_qwen_forward.py"
    exit 1
fi

if [ ! -f "checkpoints/initial_weights/qwen_flat.bin" ]; then
    echo "Error: Weights not found at checkpoints/initial_weights/qwen_flat.bin"
    echo "Run: python tools/convert_weights.py"
    exit 1
fi

if [ ! -x "./result/bin/pcp" ]; then
    echo "Building PCP via nix..."
    nix build
fi

cat >"$GATEWAY_CONFIG" <<EOF
{
  "gateway_id": "qwen-rl-gateway",
  "lab_id": "local-rl",
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
    "rl": {
      "enabled": true,
      "config_path": "${CONFIG_FILE}",
      "service_id": "rl-main",
      "workers": 1,
      "backend": "${WORKER_BACKEND}",
      "api": {
        "host": "${CONTROLLER_API_HOST}",
        "port": ${CONTROLLER_API_PORT}
      }
    }
  }
}
EOF

echo "Using locally built PCP binary..."
EXE="./result/bin/pcp"

echo "Starting PCP Qwen RL Test (Gateway + RL Controller + 1 Worker)"
echo "Config: $CONFIG_FILE"
echo "Gateway: ${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "RL API: ${CONTROLLER_API_HOST}:${CONTROLLER_API_PORT}"
echo "Worker Fabric: ${CONTROL_HOST}:${CONTROL_PORT}"

pkill -f "./result/bin/pcp --gateway --gateway-config ${GATEWAY_CONFIG} --gateway-host ${GATEWAY_HOST} --gateway-port ${GATEWAY_PORT}" 2>/dev/null || true
pkill -f "./result/bin/pcp --worker --connect ${CONTROL_HOST}:${CONTROL_PORT}" 2>/dev/null || true
sleep 1

env PCP_GATEWAY_API_TOKEN="$PCP_GATEWAY_API_TOKEN" \
    PCP_GATEWAY_INTERNAL_TOKEN="$PCP_GATEWAY_INTERNAL_TOKEN" \
    $EXE --gateway \
    --gateway-config "$GATEWAY_CONFIG" \
    --gateway-host "$GATEWAY_HOST" \
    --gateway-port "$GATEWAY_PORT" > /tmp/qwen_rl_gateway.log 2> /tmp/qwen_rl_gateway_debug.log &
GATEWAY_PID=$!

sleep 2

$EXE --worker \
     --connect "${CONTROL_HOST}:${CONTROL_PORT}" \
     --backend "$WORKER_BACKEND" > /tmp/qwen_rl_worker.log 2>&1 &
WORKER_PID=$!

echo "Gateway PID: $GATEWAY_PID"
echo "Worker PID: $WORKER_PID"

for _ in $(seq 1 600); do
    curl -sf "http://${CONTROLLER_API_HOST}:${CONTROLLER_API_PORT}/healthz" >/dev/null 2>&1 || {
        sleep 1
        continue
    }

    curl -sf "http://${CONTROLLER_API_HOST}:${CONTROLLER_API_PORT}/v1/controller" \
        -H "Authorization: Bearer ${PCP_GATEWAY_API_TOKEN}" \
        > "$STATUS_JSON" || {
        sleep 1
        continue
    }

    if grep -q '"status":"completed"' "$STATUS_JSON"; then
        echo "RL training completed."
        cat "$STATUS_JSON"
        exit 0
    fi

    if grep -q '"status":"failed"' "$STATUS_JSON"; then
        echo "RL training failed."
        cat "$STATUS_JSON"
        exit 1
    fi

    sleep 1
done

echo "Timed out waiting for RL completion."
cat "$STATUS_JSON" 2>/dev/null || true
exit 1
