#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PCP_BIN="${PCP_BIN:-"$ROOT_DIR/result/bin/pcp"}"

CONFIG_PATH_DEFAULT="$ROOT_DIR/experiments/nanochat_u16_h100_8x_d20_b32_t2048_fifo_slices.json"
CONFIG_PATH="${1:-$CONFIG_PATH_DEFAULT}"

# Make --config robust to where you invoke this script from (we cd into $ROOT_DIR in tmux).
if [[ "$CONFIG_PATH" != /* ]]; then
  if [[ -f "$CONFIG_PATH" ]]; then
    CONFIG_PATH="$(realpath "$CONFIG_PATH")"
  elif [[ -f "$ROOT_DIR/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$(realpath "$ROOT_DIR/$CONFIG_PATH")"
  fi
fi

PORT="${PCP_PORT:-8080}"
SHEPHERD_HOST="${PCP_SHEPHERD_HOST:-0.0.0.0}"
SHEPHERD_CONNECT_HOST="${PCP_CONNECT_HOST:-127.0.0.1}"
SCALE="${PCP_SCALE:-8}"
BACKEND="${PCP_BACKEND:-cuda}"
TARGET="${PCP_TARGET:-sm_90a}"
NO_DASHBOARD="${PCP_NO_DASHBOARD:-0}"

if [[ ! -x "$PCP_BIN" ]]; then
  echo "Error: PCP binary not found/executable: $PCP_BIN" >&2
  echo "Hint: from $ROOT_DIR run: nix build -L .#pcp" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config not found: $CONFIG_PATH" >&2
  exit 1
fi

shepherd_cmd=("$PCP_BIN" --supervise -- --shepherd --config "$CONFIG_PATH" --host "$SHEPHERD_HOST" --port "$PORT" --workers "$SCALE")
if [[ "$NO_DASHBOARD" == "1" ]]; then
  shepherd_cmd+=("--no-dashboard")
fi

node_cmd=("$PCP_BIN" --node-manager --scale "$SCALE" --host "$SHEPHERD_CONNECT_HOST" --port "$PORT" --backend "$BACKEND" --target "$TARGET")

cmd_to_string() {
  local -a cmd=("$@")
  local out
  printf -v out "%q " "${cmd[@]}"
  out="${out% }"
  printf "%s" "$out"
}

if command -v tmux >/dev/null 2>&1; then
  session="${PCP_TMUX_SESSION:-pcp_h100_8x}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Error: tmux session already exists: $session" >&2
    echo "Hint: tmux kill-session -t $session" >&2
    exit 1
  fi

  tmux new-session -d -s "$session" -c "$ROOT_DIR"
  shepherd_str="$(cmd_to_string "${shepherd_cmd[@]}")"
  tmux send-keys -t "$session:0.0" "cd \"$ROOT_DIR\" && $shepherd_str" C-m
  tmux split-window -h -t "$session:0" -c "$ROOT_DIR"
  node_str="$(cmd_to_string "${node_cmd[@]}")"
  tmux send-keys -t "$session:0.1" "cd \"$ROOT_DIR\" && $node_str" C-m
  tmux select-layout -t "$session:0" even-horizontal
  tmux attach -t "$session"
else
  echo "tmux not found. Run these in two terminals:"
  echo
  echo "Terminal 1 (shepherd):"
  echo "  cd \"$ROOT_DIR\""
  echo -n "  "
  cmd_to_string "${shepherd_cmd[@]}"
  echo
  echo
  echo "Terminal 2 (node-manager):"
  echo "  cd \"$ROOT_DIR\""
  echo -n "  "
  cmd_to_string "${node_cmd[@]}"
  echo
fi
