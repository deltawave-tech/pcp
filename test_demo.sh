#!/bin/bash
# Demo Distributed Training Test Script
# Showcases the PCP distributed training system with simulated execution

set -e

echo "🪐 PCP Distributed Training Demo"
echo "================================="
echo ""

# Build the project first
echo "Building project..."
zig build 2>/dev/null || {
    echo "💣 Build failed - please run 'zig build' to see errors"
    exit 1
}

echo "🌙 Build successful"
echo ""

# Check if binary exists
BINARY="./zig-out/bin/pcp_distributed"
if [ ! -f "$BINARY" ]; then
    echo "💣 Binary not found at $BINARY"
    exit 1
fi

echo "Starting Distributed Training Demo"
echo "-------------------------------------"
echo "   Algorithm: DiLoCo (Distributed Low-Communication)"
echo "   Backend: Demo (Simulated Execution)"
echo "   Workers: 2"
echo "   Architecture: Shepherd-Worker Pattern"
echo ""

# Kill any existing processes
pkill -f "pcp_distributed" 2>/dev/null || true
sleep 1

echo "👻 Launching components..."

# Start shepherd in background
echo "   📡 Starting Shepherd coordinator..."
$BINARY --shepherd --workers 2 --demo-execution > shepherd.log 2>&1 &
SHEPHERD_PID=$!

# Give shepherd time to start
sleep 2

# Start first worker
echo "   🤖 Starting Worker 1..."
$BINARY --worker --connect 127.0.0.1:8080 --demo-execution > worker1.log 2>&1 &
WORKER1_PID=$!

# Start second worker  
echo "   🤖 Starting Worker 2..."
$BINARY --worker --connect 127.0.0.1:8080 --demo-execution > worker2.log 2>&1 &
WORKER2_PID=$!

echo ""
echo "Demo training in progress..."
echo "   - DiLoCo algorithm coordinates parameter updates"
echo "   - Workers simulate MLIR compilation and execution"
echo "   - TUI dashboard shows real-time training metrics"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up processes..."
    kill $SHEPHERD_PID $WORKER1_PID $WORKER2_PID 2>/dev/null || true
    wait 2>/dev/null || true
    echo ""
    echo "Training logs available:"
    echo "   - Shepherd: shepherd.log"  
    echo "   - Worker 1: worker1.log"
    echo "   - Worker 2: worker2.log"
    echo ""
    echo "🌚 Demo completed successfully!"
}

trap cleanup EXIT

# Wait for training to complete (or timeout after 30 seconds)
TIMEOUT=30
ELAPSED=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    if ! kill -0 $SHEPHERD_PID 2>/dev/null; then
        echo "🌙 Training completed!"
        break
    fi
    
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    
    # Show progress dots
    if [ $((ELAPSED % 3)) -eq 0 ]; then
        echo -n "."
    fi
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo ""
    echo "Demo timeout reached (${TIMEOUT}s)"
fi

# Let cleanup trap handle the rest