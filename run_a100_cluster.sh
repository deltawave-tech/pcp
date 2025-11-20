#!/bin/bash

# Set PATH to include zig
export PATH=$PATH:/opt/zig

# 0. Setup
# Ensure data exists
if [ ! -f "tiny_shakespeare.txt" ]; then
    echo "Downloading dataset..."
    curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    mv input.txt tiny_shakespeare.txt
fi

# 1. Build the project
echo "⛓️ Building PCP Distributed..."
zig build
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

# Path to the executable
EXE="./zig-out/bin/main_distributed"
# Use the model file you generated earlier
MODEL_PATH="models/nano_stablehlo.mlir"

if [ ! -f "$MODEL_PATH" ]; then
    echo "💣 Error: Model file $MODEL_PATH not found."
    echo "Run your generate_model.py script first!"
    exit 1
fi

echo "🌙 Starting PCP A100 Cluster Test"
echo "Model: $MODEL_PATH"

# 2. Start Shepherd (Coordinator)
# It will wait for 2 workers.
echo "Starting Shepherd..."
$EXE --shepherd \
     --port 8090 \
     --workers 2 \
     --model "$MODEL_PATH" &
SHEPHERD_PID=$!

# Give Shepherd time to bind port and load model
sleep 3

# 3. Start Worker 1 (CUDA)
echo "Starting Worker 1 (CUDA)..."
$EXE --worker \
     --connect 127.0.0.1:8090 &
WORKER1_PID=$!

# 4. Start Worker 2 (CUDA)
echo "Starting Worker 2 (CUDA)..."
$EXE --worker \
     --connect 127.0.0.1:8090 &
WORKER2_PID=$!

# Wait for Shepherd to finish (it exits after training loop completes)
wait $SHEPHERD_PID

echo "🌚 Training Complete. Cleaning up workers..."
kill $WORKER1_PID 2>/dev/null
kill $WORKER2_PID 2>/dev/null
