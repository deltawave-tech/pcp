#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

CONFIG_FILE=${CONFIG_FILE:-experiments/inference_qwen.json}
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-18000}
CONTROL_HOST=${CONTROL_HOST:-127.0.0.1}
CONTROL_PORT=${CONTROL_PORT:-18080}
GATEWAY_HOST=${GATEWAY_HOST:-127.0.0.1}
GATEWAY_PORT=${GATEWAY_PORT:-18100}
BACKEND=${BACKEND:-cuda}
TARGET=${TARGET:-sm_80}
MODEL_ID=${MODEL_ID:-qwen2.5-0.5b-instruct}
PCP_API_TOKEN=${PCP_API_TOKEN:-dev}
PCP_GATEWAY_INTERNAL_TOKEN=${PCP_GATEWAY_INTERNAL_TOKEN:-dev-internal}
VENV_DIR=${VENV_DIR:-"$ROOT_DIR/venv"}
LOG_DIR=${LOG_DIR:-/tmp/pcp_qwen_inference_smoke}
GATEWAY_LOG="$LOG_DIR/gateway.log"
WORKER_LOG="$LOG_DIR/worker.log"
RESPONSE_JSON="$LOG_DIR/response.json"
CONTROLLER_JSON="$LOG_DIR/controller.json"
JOB_JSON="$LOG_DIR/job.json"
WORKERS_JSON="$LOG_DIR/workers.json"
METRICS_BEFORE_JSON="$LOG_DIR/metrics_before.json"
METRICS_AFTER_JSON="$LOG_DIR/metrics_after.json"
CANCEL_JSON="$LOG_DIR/cancel.json"
GATEWAY_STATUS_JSON="$LOG_DIR/gateway_status.json"
GATEWAY_CONFIG="$LOG_DIR/gateway_qwen_inference.json"

GATEWAY_PID=""
WORKER_PID=""

cleanup() {
  local exit_code=$?
  set +e

  if [ -n "$WORKER_PID" ]; then
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  if [ -n "$GATEWAY_PID" ]; then
    kill "$GATEWAY_PID" 2>/dev/null || true
    wait "$GATEWAY_PID" 2>/dev/null || true
  fi

  pkill -f "./result/bin/pcp --worker --connect ${CONTROL_HOST}:${CONTROL_PORT}" 2>/dev/null || true
  pkill -f "./result/bin/pcp --gateway --gateway-config ${GATEWAY_CONFIG} --gateway-host ${GATEWAY_HOST} --gateway-port ${GATEWAY_PORT}" 2>/dev/null || true

  if [ $exit_code -ne 0 ]; then
    echo
    echo "Gateway log tail:"
    tail -n 80 "$GATEWAY_LOG" 2>/dev/null || true
    echo
    echo "Worker log tail:"
    tail -n 80 "$WORKER_LOG" 2>/dev/null || true
  fi

  exit $exit_code
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"
rm -f "$GATEWAY_LOG" "$WORKER_LOG" "$RESPONSE_JSON" \
  "$CONTROLLER_JSON" "$JOB_JSON" "$WORKERS_JSON" \
  "$METRICS_BEFORE_JSON" "$METRICS_AFTER_JSON" "$CANCEL_JSON" \
  "$GATEWAY_STATUS_JSON" "$GATEWAY_CONFIG"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing virtualenv: $VENV_DIR" >&2
  exit 1
fi

if [ ! -x "./result/bin/pcp" ]; then
  echo "Building PCP..."
  nix build
fi

if ! "$VENV_DIR/bin/python" -c "import transformers" >/dev/null 2>&1; then
  echo "Missing transformers in $VENV_DIR" >&2
  exit 1
fi

export PATH="$VENV_DIR/bin:$PATH"
export PCP_API_TOKEN
export PCP_GATEWAY_INTERNAL_TOKEN

fetch_auth_json() {
  local path=$1
  local output=$2
  curl -sf "http://${API_HOST}:${API_PORT}${path}" \
    -H "Authorization: Bearer ${PCP_API_TOKEN}" \
    >"$output"
}

post_auth_json() {
  local path=$1
  local output=$2
  curl -sf -X POST "http://${API_HOST}:${API_PORT}${path}" \
    -H "Authorization: Bearer ${PCP_API_TOKEN}" \
    -H "Content-Type: application/json" \
    >"$output"
}

echo "Cleaning up stale PCP inference processes..."
pkill -f "./result/bin/pcp --worker --connect ${CONTROL_HOST}:${CONTROL_PORT}" 2>/dev/null || true
pkill -f "./result/bin/pcp --gateway --gateway-config ${GATEWAY_CONFIG} --gateway-host ${GATEWAY_HOST} --gateway-port ${GATEWAY_PORT}" 2>/dev/null || true
sleep 1

cat >"$GATEWAY_CONFIG" <<EOF
{
  "gateway_id": "qwen-inference-smoke",
  "lab_id": "local-smoke",
  "graph_backend": "memory",
  "api_token_env": "PCP_API_TOKEN",
  "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
  "worker_fabric": {
    "host": "${CONTROL_HOST}",
    "port": ${CONTROL_PORT}
  },
  "api": {
    "host": "${GATEWAY_HOST}",
    "port": ${GATEWAY_PORT},
    "token_env": "PCP_API_TOKEN",
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

echo "Starting gateway with embedded inference controller..."
./result/bin/pcp \
  --gateway \
  --gateway-config "$GATEWAY_CONFIG" \
  --gateway-host "$GATEWAY_HOST" \
  --gateway-port "$GATEWAY_PORT" \
  >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

echo "Starting worker..."
./result/bin/pcp \
  --worker \
  --connect "$CONTROL_HOST:$CONTROL_PORT" \
  --backend "$BACKEND" \
  --target "$TARGET" \
  >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!

echo "Gateway PID: $GATEWAY_PID"
echo "Worker PID: $WORKER_PID"

for _ in $(seq 1 60); do
  if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
    echo "Gateway exited during startup" >&2
    exit 1
  fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Worker exited during startup" >&2
    exit 1
  fi
  if curl -sf "http://${API_HOST}:${API_PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Healthz:"
curl -sf "http://${API_HOST}:${API_PORT}/healthz"
echo

for _ in $(seq 1 120); do
  if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
    echo "Gateway exited before readyz succeeded" >&2
    exit 1
  fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Worker exited before readyz succeeded" >&2
    exit 1
  fi
  if curl -sf "http://${API_HOST}:${API_PORT}/readyz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Readyz:"
curl -sf "http://${API_HOST}:${API_PORT}/readyz"
echo

echo "Checking gateway service registry..."
fetch_auth_json_gateway() {
  local path=$1
  local output=$2
  curl -sf "http://${GATEWAY_HOST}:${GATEWAY_PORT}${path}" \
    -H "Authorization: Bearer ${PCP_API_TOKEN}" \
    >"$output"
}

fetch_auth_json_gateway /v1/services "$GATEWAY_STATUS_JSON"
cat "$GATEWAY_STATUS_JSON"
echo

echo "Checking operator controller endpoint..."
fetch_auth_json /v1/controller "$CONTROLLER_JSON"
cat "$CONTROLLER_JSON"
echo

echo "Checking operator job endpoint..."
fetch_auth_json /v1/job "$JOB_JSON"
cat "$JOB_JSON"
echo

echo "Checking operator workers endpoint..."
fetch_auth_json /v1/workers "$WORKERS_JSON"
cat "$WORKERS_JSON"
echo

echo "Checking operator metrics endpoint before request..."
fetch_auth_json /v1/metrics "$METRICS_BEFORE_JSON"
cat "$METRICS_BEFORE_JSON"
echo

"$VENV_DIR/bin/python" - "$CONTROLLER_JSON" "$JOB_JSON" "$WORKERS_JSON" "$METRICS_BEFORE_JSON" "$GATEWAY_STATUS_JSON" <<'PY'
import json
import sys

controller_path, job_path, workers_path, metrics_path, gateway_path = sys.argv[1:]

with open(controller_path, "r", encoding="utf-8") as f:
    controller = json.load(f)
with open(job_path, "r", encoding="utf-8") as f:
    job = json.load(f)
with open(workers_path, "r", encoding="utf-8") as f:
    workers = json.load(f)
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)
with open(gateway_path, "r", encoding="utf-8") as f:
    gateway = json.load(f)

assert controller["mode"] == "inference", controller
assert controller["job_type"] == "inference", controller
assert controller["ready"] is True, controller
assert controller["auth_enabled"] is True, controller
assert controller["workers_connected"] >= 1, controller
assert controller["model_id"] == "qwen2.5-0.5b-instruct", controller

assert job["job_type"] == "inference", job
assert job["workers_connected"] >= 1, job

worker_items = workers.get("workers")
assert isinstance(worker_items, list) and len(worker_items) >= 1, workers
assert worker_items[0]["status"] in {"ready", "busy", "connected"}, workers

assert metrics["mode"] == "inference", metrics
assert metrics["workers_connected"] >= 1, metrics
assert metrics["inference"]["total_requests"] >= 0, metrics

services = gateway.get("services") or []
assert any(service["service_id"] == "inference-main" for service in services), gateway
PY

echo "Requesting completion..."
curl -sf "http://${API_HOST}:${API_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${PCP_API_TOKEN}" \
  -d "$(cat <<'EOF'
{"model":"qwen2.5-0.5b-instruct","messages":[{"role":"system","content":"Reply with only a short answer, no explanation."},{"role":"user","content":"Describe the Moon in one English adjective."}],"max_tokens":8,"temperature":0.0}
EOF
)" >"$RESPONSE_JSON"

cat "$RESPONSE_JSON"
echo

echo "Checking operator metrics endpoint after request..."
fetch_auth_json /v1/metrics "$METRICS_AFTER_JSON"
cat "$METRICS_AFTER_JSON"
echo

echo "Decoded text:"
"$VENV_DIR/bin/python" - "$RESPONSE_JSON" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

choices = data.get("choices") or []
message = ""
if choices:
    message = ((choices[0].get("message") or {}).get("content")) or ""

print(message)

bad_chars = sum(1 for ch in message if ord(ch) < 32 and ch not in "\n\r\t")
forbidden = len(re.findall(r"[\[\]{}\\/0-9_]", message))
words = re.findall(r"[A-Za-z]+", message)
word_count = len(words)
if not message.strip():
    raise SystemExit("empty completion")
if bad_chars:
    raise SystemExit(f"completion contains {bad_chars} control characters")
if forbidden:
    raise SystemExit(f"completion contains {forbidden} forbidden punctuation/digit characters")
if word_count < 1:
    raise SystemExit(f"completion does not contain alphabetic words: {message!r}")
if word_count > 6:
    raise SystemExit(f"completion is unexpectedly long for constrained prompt: {message!r}")
PY

"$VENV_DIR/bin/python" - "$METRICS_BEFORE_JSON" "$METRICS_AFTER_JSON" <<'PY'
import json
import sys

before_path, after_path = sys.argv[1:]
with open(before_path, "r", encoding="utf-8") as f:
    before = json.load(f)
with open(after_path, "r", encoding="utf-8") as f:
    after = json.load(f)

before_inf = before["inference"]
after_inf = after["inference"]

if after_inf["total_requests"] < before_inf["total_requests"] + 1:
    raise SystemExit(f"total_requests did not increase: before={before_inf['total_requests']} after={after_inf['total_requests']}")
if after_inf["completed_requests"] < before_inf["completed_requests"] + 1:
    raise SystemExit(f"completed_requests did not increase: before={before_inf['completed_requests']} after={after_inf['completed_requests']}")
if after_inf["prompt_tokens"] <= before_inf["prompt_tokens"]:
    raise SystemExit(f"prompt_tokens did not increase: before={before_inf['prompt_tokens']} after={after_inf['prompt_tokens']}")
PY

echo "Smoke test completed successfully."
