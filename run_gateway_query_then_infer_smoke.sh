#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-18113}"
GLOBAL_PORT="${GLOBAL_PORT:-19111}"
INF_API_PORT="${INF_API_PORT:-18001}"
INF_CTRL_PORT="${INF_CTRL_PORT:-18081}"
LOG_DIR="${LOG_DIR:-/tmp/pcp_gateway_query_then_infer_smoke}"
CONFIG_PATH="$LOG_DIR/gateway_federated_smoke.json"

GATEWAY_TOKEN="${PCP_GATEWAY_API_TOKEN:-dev-gateway}"
INTERNAL_TOKEN="${PCP_GATEWAY_INTERNAL_TOKEN:-dev-internal}"
GLOBAL_TOKEN="${PCP_GLOBAL_CONTROLLER_TOKEN:-dev-global}"

GATEWAY_LOG="$LOG_DIR/gateway.log"
GLOBAL_LOG="$LOG_DIR/global_controller.log"
INFERENCE_LOG="$LOG_DIR/inference.log"
WORKER_LOG="$LOG_DIR/worker.log"

SERVICES_JSON="$LOG_DIR/services.json"
GLOBAL_STATUS_JSON="$LOG_DIR/global_status.json"
RESPONSE_JSON="$LOG_DIR/query_then_infer.json"

GATEWAY_PID=""
GLOBAL_PID=""
INFERENCE_PID=""
WORKER_PID=""

cleanup() {
  local exit_code=$?
  set +e

  for pid in "$WORKER_PID" "$INFERENCE_PID" "$GATEWAY_PID" "$GLOBAL_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" 2>/dev/null || true
    fi
  done

  pkill -f "pcp --gateway --gateway-config $CONFIG_PATH --gateway-host ${HOST} --gateway-port ${GATEWAY_PORT}" >/dev/null 2>&1 || true
  pkill -f "pcp --global-controller --api-host ${HOST} --api-port ${GLOBAL_PORT} --api-token-env PCP_GLOBAL_CONTROLLER_TOKEN" >/dev/null 2>&1 || true
  pkill -f "pcp --inference --inference-config experiments/inference_qwen.json --api-host ${HOST} --api-port ${INF_API_PORT} --control-host ${HOST} --control-port ${INF_CTRL_PORT}" >/dev/null 2>&1 || true
  pkill -f "pcp --worker --connect ${HOST}:${INF_CTRL_PORT} --backend cuda --target sm_80" >/dev/null 2>&1 || true

  if [[ $exit_code -ne 0 ]]; then
    echo
    echo "Gateway log tail:"
    tail -n 120 "$GATEWAY_LOG" 2>/dev/null || true
    echo
    echo "Global controller log tail:"
    tail -n 120 "$GLOBAL_LOG" 2>/dev/null || true
    echo
    echo "Inference log tail:"
    tail -n 120 "$INFERENCE_LOG" 2>/dev/null || true
    echo
    echo "Worker log tail:"
    tail -n 120 "$WORKER_LOG" 2>/dev/null || true
  fi

  exit "$exit_code"
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"

cat >"$CONFIG_PATH" <<EOF
{
  "gateway_id": "lab-alpha-gateway",
  "lab_id": "lab-alpha",
  "graph_backend": "memory",
  "global_controller_endpoint": "http://${HOST}:${GLOBAL_PORT}",
  "federation": {
    "enabled": true,
    "upstream": "http://${HOST}:${GLOBAL_PORT}",
    "token_env": "PCP_GLOBAL_CONTROLLER_TOKEN",
    "heartbeat_interval_ms": 250
  },
  "api_token_env": "PCP_GATEWAY_API_TOKEN",
  "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN"
}
EOF

if [[ ! -x "./result/bin/pcp" ]]; then
  nix build
fi

export PATH="$ROOT_DIR/venv/bin:$PATH"

wait_for_url() {
  local url="$1"
  for _ in $(seq 1 120); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

wait_for_json_match() {
  local url="$1"
  local token="$2"
  local expected="$3"
  local output_path="$4"
  for _ in $(seq 1 120); do
    local response
    response="$(curl -sf -H "Authorization: Bearer ${token}" "$url")" || {
      sleep 0.5
      continue
    }
    printf '%s' "$response" >"$output_path"
    if grep -q "$expected" "$output_path"; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

env PCP_GLOBAL_CONTROLLER_TOKEN="$GLOBAL_TOKEN" \
  ./result/bin/pcp \
    --global-controller \
    --api-host "$HOST" \
    --api-port "$GLOBAL_PORT" \
    --api-token-env PCP_GLOBAL_CONTROLLER_TOKEN \
    >"$GLOBAL_LOG" 2>&1 &
GLOBAL_PID="$!"

env PCP_GATEWAY_API_TOKEN="$GATEWAY_TOKEN" \
  PCP_GATEWAY_INTERNAL_TOKEN="$INTERNAL_TOKEN" \
  PCP_GLOBAL_CONTROLLER_TOKEN="$GLOBAL_TOKEN" \
  ./result/bin/pcp \
    --gateway \
    --gateway-config "$CONFIG_PATH" \
    --gateway-host "$HOST" \
    --gateway-port "$GATEWAY_PORT" \
    >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID="$!"

env PCP_API_TOKEN="$GATEWAY_TOKEN" \
  PCP_GATEWAY_URL="http://${HOST}:${GATEWAY_PORT}" \
  PCP_GATEWAY_TOKEN="$GATEWAY_TOKEN" \
  PCP_GATEWAY_INTERNAL_TOKEN="$INTERNAL_TOKEN" \
  PCP_GATEWAY_SERVICE_ID="inference-main" \
  PCP_GATEWAY_NAMESPACE="lab-alpha/shared" \
  ./result/bin/pcp \
    --inference \
    --inference-config experiments/inference_qwen.json \
    --api-host "$HOST" \
    --api-port "$INF_API_PORT" \
    --control-host "$HOST" \
    --control-port "$INF_CTRL_PORT" \
    >"$INFERENCE_LOG" 2>&1 &
INFERENCE_PID="$!"

./result/bin/pcp \
  --worker \
  --connect "${HOST}:${INF_CTRL_PORT}" \
  --backend cuda \
  --target sm_80 \
  >"$WORKER_LOG" 2>&1 &
WORKER_PID="$!"

wait_for_url "http://${HOST}:${GLOBAL_PORT}/healthz"
wait_for_url "http://${HOST}:${GATEWAY_PORT}/healthz"
wait_for_url "http://${HOST}:${INF_API_PORT}/readyz"
wait_for_json_match "http://${HOST}:${GATEWAY_PORT}/v1/federation/status" "$GATEWAY_TOKEN" '"connected":true' "$SERVICES_JSON"
wait_for_json_match "http://${HOST}:${GATEWAY_PORT}/v1/services" "$GATEWAY_TOKEN" '"service_id":"inference-main"' "$SERVICES_JSON"

curl -sf \
  -X PUT \
  -H "Authorization: Bearer ${GATEWAY_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"default_visibility":"shared","allow_global_replication":true,"allow_raw_payload_export":true}' \
  "http://${HOST}:${GATEWAY_PORT}/v1/graph/policies/lab-alpha/shared" \
  >/dev/null

curl -sf \
  -X POST \
  -H "Authorization: Bearer ${GATEWAY_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"mutations":[{"mutation_type":"upsert_entity","namespace_id":"lab-alpha/shared","target_id":"experiment:shared-1","payload":{"entity_type":"experiment","display_name":"Shared Experiment","properties":{"scope":"shared","owner":"lab-alpha"}},"visibility":"shared","provenance":{"service_id":"eve","actor_id":"eve-agent-1"}},{"mutation_type":"upsert_entity","namespace_id":"lab-alpha/shared","target_id":"experiment:local-1","payload":{"entity_type":"experiment","display_name":"Local Experiment","properties":{"scope":"local","owner":"lab-alpha"}},"visibility":"local","provenance":{"service_id":"eve","actor_id":"eve-agent-1"}}]}' \
  "http://${HOST}:${GATEWAY_PORT}/v1/graph/mutate" \
  >/dev/null

wait_for_json_match "http://${HOST}:${GLOBAL_PORT}/v1/global-graph/status" "$GLOBAL_TOKEN" '"entities":1' "$GLOBAL_STATUS_JSON"

curl -sf \
  -X POST \
  -H "Authorization: Bearer ${GATEWAY_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"graph_query":{"query_mode":"local_plus_global","namespaces":["lab-alpha/shared"],"entity_types":["experiment"],"limit":10},"inference":{"model":"qwen2.5-0.5b-instruct","messages":[{"role":"user","content":"Reply with CONTEXT_OK and nothing else."}],"max_tokens":8}}' \
  "http://${HOST}:${GATEWAY_PORT}/v1/inference/query/chat/completions" \
  >"$RESPONSE_JSON"

grep -q '"graph_query"' "$RESPONSE_JSON"
grep -q '"graph_context_prompt"' "$RESPONSE_JSON"
grep -q '"completion"' "$RESPONSE_JSON"
grep -q '"entity_id":"experiment:shared-1"' "$RESPONSE_JSON"
grep -q '"entity_id":"experiment:local-1"' "$RESPONSE_JSON"
grep -q 'experiment:shared-1' "$RESPONSE_JSON"
grep -q 'experiment:local-1' "$RESPONSE_JSON"
grep -q '"object":"chat.completion"' "$RESPONSE_JSON"
grep -q '"session_id":"' "$RESPONSE_JSON"

printf 'services: %s\n' "$(cat "$SERVICES_JSON")"
printf 'global_status: %s\n' "$(cat "$GLOBAL_STATUS_JSON")"
printf 'query_then_infer: %s\n' "$(cat "$RESPONSE_JSON")"
