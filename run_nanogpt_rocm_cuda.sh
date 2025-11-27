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
# Use the GPU-aligned nano GPT model
MODEL_PATH="models/nanogpt_forward_32.mlir"

if [ ! -f "$MODEL_PATH" ]; then
    echo "💣 Error: Model file $MODEL_PATH not found."
    echo "Run your generate_model.py script first!"
    exit 1
fi

echo "🌙 Starting PCP NanoGPT CUDA + ROCm Cluster"
echo "Model: $MODEL_PATH"

# 2. Start Shepherd (Coordinator)
# Configured for 2 workers (1 local CUDA + 1 remote ROCm)
echo "Starting Shepherd (configured for 2 workers: 1 CUDA local + 1 ROCm remote)..."
$EXE --shepherd \
     --host 0.0.0.0 \
     --port 8090 \
     --workers 2 \
     --model "$MODEL_PATH" &
SHEPHERD_PID=$!

# Give Shepherd time to bind port and load model
sleep 3

# 3. Start Worker 1 (CUDA - Local)
echo "Starting Worker 1 (CUDA - Local)..."
$EXE --worker \
     --connect 127.0.0.1:8090 \
     --backend cuda &
WORKER1_PID=$!

echo ""
echo "⏳ Waiting for remote ROCm worker to connect..."
echo "   Run this on the remote machine:"
echo "   ./zig-out/bin/pcp --worker --connect <this_machine_ip>:8090 --backend rocm --amd-target gfx942"
echo ""

# Wait for Shepherd to finish (it exits after training loop completes)
wait $SHEPHERD_PID

# Use cleanup function for consistent shutdown
echo "🌚 Training Complete. Cleaning up workers..."
cleanup
