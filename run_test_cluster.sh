#!/bin/bash

# Set PATH to include zig
export PATH=$PATH:/opt/zig

# Cleanup function to kill all background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down cluster..."
    if [ ! -z "$SHEPHERD_PID" ]; then
        kill $SHEPHERD_PID 2>/dev/null
    fi
    if [ ! -z "$WORKER1_PID" ]; then
        kill $WORKER1_PID 2>/dev/null
    fi
    if [ ! -z "$WORKER2_PID" ]; then
        kill $WORKER2_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# 0. Setup
# Verify dataset exists
if [ ! -f "data/tiny_shakespeare.txt" ]; then
    echo "💣 Error: Dataset not found at data/tiny_shakespeare.txt"
    exit 1
fi

# 1. Build the project
echo "⛓️ Building PCP Distributed..."
zig build
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

# Path to the executable
EXE="./zig-out/bin/pcp"
# Use the nano GPT model
MODEL_PATH="models/nano_stablehlo.mlir"

if [ ! -f "$MODEL_PATH" ]; then
    echo "💣 Error: Model file $MODEL_PATH not found."
    echo "Run your generate_model.py script first!"
    exit 1
fi

echo "🌙 Starting PCP Test Cluster"
echo "Model: $MODEL_PATH"

# 2. Start Shepherd (Coordinator)
# Configured for 3 workers
echo "Starting Shepherd (configured for 3 workers)..."
$EXE --shepherd \
     --host 0.0.0.0 \
     --port 8090 \
     --workers 3 \
     --model "$MODEL_PATH" &
SHEPHERD_PID=$!

# Give Shepherd time to bind port and load model
sleep 3

# 3. Start Worker 1 (CPU)
echo "Starting Worker 1 (CPU)..."
$EXE --worker \
     --connect 127.0.0.1:8090 \
     --backend cpu &
WORKER1_PID=$!

# 4. Start Worker 2 (CUDA)
echo "Starting Worker 2 (CUDA)..."
$EXE --worker \
     --connect 127.0.0.1:8090 \
     --backend cuda &
WORKER2_PID=$!

# Note: Shepherd is configured for 3 workers but only 2 are spawned
# This will cause the shepherd to wait for the 3rd worker to connect

# Wait for Shepherd to finish (it exits after training loop completes)
wait $SHEPHERD_PID

# Use cleanup function for consistent shutdown
echo "🌚 Training Complete. Cleaning up workers..."
cleanup
