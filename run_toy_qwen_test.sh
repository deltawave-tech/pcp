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
CONFIG_FILE="experiments/toy_qwen_test.json"

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Verify dataset exists
if [ ! -f "data/tiny_shakespeare.txt" ]; then
    echo "Error: Dataset not found at data/tiny_shakespeare.txt"
    exit 1
fi

# Use locally built Nix binary
echo "Using locally built PCP binary..."
EXE="./result/bin/pcp"

echo "Starting PCP Toy Qwen Test (Supervisor + 1 CUDA Worker)"
echo "Config: $CONFIG_FILE"
echo "Logs: /tmp/shepherd.log, /tmp/worker.log"
echo ""
echo "Starting TUI..."
sleep 1

# Start Shepherd with experiment config
$EXE --shepherd \
     --config "$CONFIG_FILE" \
     --host 0.0.0.0 \
     --port 8090 \
     --workers 1 > /tmp/shepherd.log 2>&1 &
SHEPHERD_PID=$!

# Give Shepherd time to bind port and start TUI
sleep 2

# Start CUDA Worker
$EXE --worker \
     --connect 127.0.0.1:8090 \
     --backend cuda > /tmp/worker.log 2>&1 &
WORKER_PID=$!

# Wait for Shepherd to finish
wait $SHEPHERD_PID
EXIT_CODE=$?

# Wait for terminal to restore
sleep 1

# Cleanup
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training Complete. Cleaning up worker..."
else
    echo "Shepherd exited with code $EXIT_CODE. Cleaning up..."
fi
cleanup
