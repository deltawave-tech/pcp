#!/bin/bash

# Set PATH to include zig
export PATH=$PATH:/opt/zig

# Cleanup function to kill all background processes
cleanup() {
    echo ""
    echo "Shutting down cluster..."
    if [ ! -z "$SHEPHERD_PID" ]; then
        kill $SHEPHERD_PID 2>/dev/null
    fi
    if [ ! -z "$WORKER_PID" ]; then
        kill $WORKER_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# Configuration file
CONFIG_FILE="experiments/qwen_rl_test.json"

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Verify models exist
if [ ! -f "models/qwen_rl_generation.mlir" ]; then
    echo "Error: Generation model not found at models/qwen_rl_generation.mlir"
    echo "Run: python tools/export_qwen_rl_generation.py"
    exit 1
fi

# Check for MLIR or pre-compiled VMFB (backend-specific caching is handled internally)
if [ ! -f "models/qwen_grpo_training.mlir" ] && [ ! -f "models/qwen_grpo_training.cuda.vmfb" ]; then
    echo "Error: Training model not found"
    echo "Run: python tools/export_qwen_forward.py"
    exit 1
fi

# Verify weights exist
if [ ! -f "checkpoints/initial_weights/qwen_flat.bin" ]; then
    echo "Error: Weights not found at checkpoints/initial_weights/qwen_flat.bin"
    echo "Run: python tools/convert_weights.py"
    exit 1
fi

# Use locally built Nix binary
echo "Using locally built PCP binary..."
EXE="./result/bin/pcp"

echo "Starting PCP Qwen RL Test with GRPO (CUDA Shepherd + 1 CUDA Worker)"
echo "Config: $CONFIG_FILE"
echo "Logs: /tmp/rl_shepherd.log, /tmp/rl_worker.log"
echo ""
echo "Starting RL Training..."
sleep 1

# Start RL Shepherd with experiment config
echo "Starting RL Shepherd with debug output to /tmp/rl_shepherd_debug.log"
$EXE --shepherd \
     --rl \
     --no-dashboard \
     --backend cuda \
     --config "$CONFIG_FILE" \
     --host 0.0.0.0 \
     --port 8090 \
     --workers 1 > /tmp/rl_shepherd.log 2> /tmp/rl_shepherd_debug.log &
SHEPHERD_PID=$!
echo "Shepherd PID: $SHEPHERD_PID"

# Give Shepherd time to bind port and start TUI
sleep 2

# Start CUDA Worker for RL rollout generation
echo "Starting CUDA Worker (PID will be printed)..."
$EXE --worker \
     --connect 127.0.0.1:8090 \
     --backend cuda > /tmp/rl_worker.log 2>&1 &
WORKER_PID=$!
echo "Worker PID: $WORKER_PID"

# Wait for Shepherd to finish
wait $SHEPHERD_PID
EXIT_CODE=$?

# Wait for terminal to restore
sleep 1

# Cleanup
if [ $EXIT_CODE -eq 0 ]; then
    echo "RL Training Complete. Cleaning up worker..."
else
    echo "RL Shepherd exited with code $EXIT_CODE. Cleaning up..."
fi
cleanup
