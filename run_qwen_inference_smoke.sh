#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

CONFIG_FILE=${CONFIG_FILE:-experiments/inference_qwen.json}
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-18000}
CONTROL_HOST=${CONTROL_HOST:-127.0.0.1}
CONTROL_PORT=${CONTROL_PORT:-18080}
BACKEND=${BACKEND:-cuda}
TARGET=${TARGET:-sm_80}
MODEL_ID=${MODEL_ID:-qwen2.5-0.5b-instruct}
PCP_API_TOKEN=${PCP_API_TOKEN:-dev}
VENV_DIR=${VENV_DIR:-"$ROOT_DIR/venv"}
LOG_DIR=${LOG_DIR:-/tmp/pcp_qwen_inference_smoke}
CTRL_LOG="$LOG_DIR/controller.log"
WORKER_LOG="$LOG_DIR/worker.log"
RESPONSE_JSON="$LOG_DIR/response.json"

CTRL_PID=""
WORKER_PID=""

cleanup() {
  local exit_code=$?
  set +e

  if [ -n "$WORKER_PID" ]; then
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  if [ -n "$CTRL_PID" ]; then
    kill "$CTRL_PID" 2>/dev/null || true
    wait "$CTRL_PID" 2>/dev/null || true
  fi

  pkill -f "./result/bin/pcp --worker --connect ${CONTROL_HOST}:${CONTROL_PORT}" 2>/dev/null || true
  pkill -f "./result/bin/pcp --inference --inference-config ${CONFIG_FILE}" 2>/dev/null || true

  if [ $exit_code -ne 0 ]; then
    echo
    echo "Controller log tail:"
    tail -n 80 "$CTRL_LOG" 2>/dev/null || true
    echo
    echo "Worker log tail:"
    tail -n 80 "$WORKER_LOG" 2>/dev/null || true
  fi

  exit $exit_code
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"
rm -f "$CTRL_LOG" "$WORKER_LOG" "$RESPONSE_JSON"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing virtualenv: $VENV_DIR" >&2
  exit 1
fi

if [ ! -x "./result/bin/pcp" ] || ! ./result/bin/pcp --help | rg -q -- "--inference"; then
  echo "Building PCP..."
  nix build
fi

if ! "$VENV_DIR/bin/python" -c "import transformers" >/dev/null 2>&1; then
  echo "Missing transformers in $VENV_DIR" >&2
  exit 1
fi

export PATH="$VENV_DIR/bin:$PATH"
export PCP_API_TOKEN

echo "Cleaning up stale PCP inference processes..."
pkill -f "./result/bin/pcp --worker --connect ${CONTROL_HOST}:${CONTROL_PORT}" 2>/dev/null || true
pkill -f "./result/bin/pcp --inference --inference-config ${CONFIG_FILE}" 2>/dev/null || true
sleep 1

echo "Starting controller..."
./result/bin/pcp \
  --inference \
  --inference-config "$CONFIG_FILE" \
  --api-host "$API_HOST" \
  --api-port "$API_PORT" \
  --control-host "$CONTROL_HOST" \
  --control-port "$CONTROL_PORT" \
  >"$CTRL_LOG" 2>&1 &
CTRL_PID=$!

echo "Starting worker..."
./result/bin/pcp \
  --worker \
  --connect "$CONTROL_HOST:$CONTROL_PORT" \
  --backend "$BACKEND" \
  --target "$TARGET" \
  >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!

echo "Controller PID: $CTRL_PID"
echo "Worker PID: $WORKER_PID"

for _ in $(seq 1 60); do
  if ! kill -0 "$CTRL_PID" 2>/dev/null; then
    echo "Controller exited during startup" >&2
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
  if ! kill -0 "$CTRL_PID" 2>/dev/null; then
    echo "Controller exited before readyz succeeded" >&2
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

echo "Smoke test completed successfully."
