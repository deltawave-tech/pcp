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
    if [ ! -z "$SUPERVISOR1_PID" ]; then
        kill $SUPERVISOR1_PID 2>/dev/null
    fi
    if [ ! -z "$SUPERVISOR2_PID" ]; then
        kill $SUPERVISOR2_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# 0. Setup
# Configuration file (using experiment config)
CONFIG_FILE="experiment_nanogpt.json"

# Verify config exists (optional - will use defaults if not present)
if [ -f "$CONFIG_FILE" ]; then
    echo "📋 Using experiment config: $CONFIG_FILE"
else
    echo "📋 No config file found, using defaults"
fi

# Verify dataset exists
if [ ! -f "data/tiny_shakespeare.txt" ]; then
    echo "💣 Error: Dataset not found at data/tiny_shakespeare.txt"
    exit 1
fi

# 1. Use Nix binary (no build needed)
echo "⛓️ Using PCP from Nix/Cachix..."

# Path to the executable (uses nix-installed binary from PATH)
EXE="pcp"

echo "🌙 Starting PCP NanoGPT 2-Supervisor Cluster (Config-based)"
echo "   This demonstrates the Supervisor Pattern with fault-tolerant workers"

# 2. Start Shepherd (Coordinator)
# Configured for 2 workers (spawned by supervisors)
echo "Starting Shepherd (configured for 2 workers)..."
if [ -f "$CONFIG_FILE" ]; then
    $EXE --shepherd \
         --config "$CONFIG_FILE" \
         --host 0.0.0.0 \
         --port 8090 \
         --workers 2 &
else
    # Fallback to model path if no config
    MODEL_PATH="models/nanogpt_forward_32.mlir"
    $EXE --shepherd \
         --model "$MODEL_PATH" \
         --host 0.0.0.0 \
         --port 8090 \
         --workers 2 &
fi
SHEPHERD_PID=$!

# Give Shepherd time to bind port and load model
sleep 3

# 3. Start Supervisor 1 (CUDA - Local)
# The supervisor will automatically spawn a worker child process
echo "Starting Supervisor 1 (CUDA - Local)..."
$EXE --supervisor \
     --host 127.0.0.1 \
     --port 8090 \
     --backend cuda &
SUPERVISOR1_PID=$!

# Give first supervisor time to register and spawn worker
sleep 2

# 4. Start Supervisor 2 (CUDA - Local)
# The supervisor will automatically spawn a worker child process
echo "Starting Supervisor 2 (CUDA - Local)..."
$EXE --supervisor \
     --host 127.0.0.1 \
     --port 8090 \
     --backend cuda &
SUPERVISOR2_PID=$!

echo ""
echo "✅ Cluster started with Supervisor Pattern:"
echo "   - 1 Shepherd (coordinator)"
echo "   - 2 Supervisors (fault-tolerant parent processes)"
echo "   - 2 Workers (automatically spawned and managed by supervisors)"
echo ""
echo "   Benefits:"
echo "   - Workers automatically restart if they crash"
echo "   - Supervisors maintain control plane connection to Shepherd"
echo "   - Workers handle data plane (training operations)"
echo ""

# Wait for Shepherd to finish (it exits after training loop completes)
wait $SHEPHERD_PID

# Use cleanup function for consistent shutdown
echo "🌚 Training Complete. Cleaning up supervisors and workers..."
cleanup
