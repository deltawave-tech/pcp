#!/bin/bash

export PATH=$PATH:/opt/zig

if [ ! -f "data/tiny_shakespeare.txt" ]; then
    echo "💣 Error: Dataset not found at data/tiny_shakespeare.txt"
    exit 1
fi

echo "⛓️ Building PCP Distributed..."
zig build
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

EXE="./zig-out/bin/main_distributed"
MODEL_PATH="models/nanogpt_forward.mlir"

if [ ! -f "$MODEL_PATH" ]; then
    echo "💣 Error: Model file $MODEL_PATH not found."
    exit 1
fi

echo "🌙 Starting PCP CPU-Only Test"
echo "Model: $MODEL_PATH"

echo "Starting Shepherd..."
$EXE --shepherd \
     --port 8090 \
     --workers 2 \
     --model "$MODEL_PATH" &
SHEPHERD_PID=$!

sleep 3

echo "Starting Worker 1 (CPU)..."
$EXE --worker \
     --connect 127.0.0.1:8090 \
     --backend cpu &
WORKER1_PID=$!

echo "Starting Worker 2 (CPU)..."
$EXE --worker \
     --connect 127.0.0.1:8090 \
     --backend cpu &
WORKER2_PID=$!

wait $SHEPHERD_PID

echo "🌚 Training Complete. Cleaning up workers..."
kill $WORKER1_PID 2>/dev/null
kill $WORKER2_PID 2>/dev/null
