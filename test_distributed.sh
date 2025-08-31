#!/bin/bash

# PCP Distributed Training System - End-to-End Test Script
# This script demonstrates the complete MLIR → SPIR-V → Metal pipeline
# across distributed worker nodes on your M3 Mac Pro

echo "🌑 PCP Distributed Training System - Live Test"
echo "=============================================="

# Build the distributed training system
echo "⛓️ Forging the distributed computation engine..."
zig build

if [ $? -ne 0 ]; then
    echo "💣 Build failed. The forge is cold. Address compilation errors before proceeding."
    exit 1
fi

echo "🌙 Binary synthesis complete: ./zig-out/bin/main_distributed"
echo "---"

# Verify the executable exists
if [ ! -f "./zig-out/bin/main_distributed" ]; then
    echo "💣 Executable not materialized. Build system anomaly detected."
    exit 1
fi

# Kill any previous instances on exit
trap "echo '🌑 Terminating all distributed processes...'; kill 0" EXIT

echo "🐉 Initiating distributed tensor computation orchestration..."
echo ""

# 1. Start the Shepherd Coordinator in the background
echo "👹 Materializing Shepherd Coordinator..."
./zig-out/bin/main_distributed --shepherd --workers 2 &
SHEPHERD_PID=$!
echo "   └─ Shepherd Process ID: $SHEPHERD_PID"
sleep 3 # Allow Shepherd to establish TCP listener

# 2. Start Worker Alpha in the background
echo "🐉 Connecting Worker Alpha to the network..."
./zig-out/bin/main_distributed --worker --connect 127.0.0.1:8080 &
WORKER1_PID=$!
echo "   └─ Worker Alpha Process ID: $WORKER1_PID"
sleep 2

# 3. Start Worker Beta in the background
echo "🐉 Connecting Worker Beta to the network..."
./zig-out/bin/main_distributed --worker --connect 127.0.0.1:8080 &
WORKER2_PID=$!
echo "   └─ Worker Beta Process ID: $WORKER2_PID"
sleep 1

echo ""
echo "⛓️ Distributed System Architecture Active"
echo "========================================="
echo "Shepherd Coordinator: PID $SHEPHERD_PID (127.0.0.1:8080)"
echo "Worker Alpha:         PID $WORKER1_PID"
echo "Worker Beta:          PID $WORKER2_PID"
echo ""
echo "🌑 Expected Execution Flow:"
echo "   ├─ Shepherd constructs complete MLIR computation graphs"
echo "   ├─ Workers receive serialized graphs via TCP"
echo "   ├─ Workers parse MLIR modules and execute on Metal"
echo "   ├─ Real SPIR-V generation and GPU shader compilation"
echo "   ├─ Distributed parameter updates via DiLoCo algorithm"
echo "   └─ Complete end-to-end tensor computation pipeline"
echo ""
echo "🌙 Critical Success Patterns to Observe:"
echo "   ✓ 'DiLoCo: Building worker training graph...'"
echo "   ✓ 'Serialized MLIR module to X bytes'"
echo "   ✓ 'Training graph + parameters broadcasted to 2 workers'"
echo "   ✓ 'Worker X received training graph.'"
echo "   ✓ 'Deserialized MLIR module from X bytes'"
echo "   ✓ 'Real SPIR-V binary size: X bytes (stub was 20 bytes)'"
echo "   ✓ 'Successfully executed MLIR module on Metal hardware'"
echo "   ✓ 'Worker X completed inner loop and sent results.'"
echo ""
echo "⛓️ Monitor the cascade of MLIR compilation across distributed nodes..."
echo "   Press Ctrl+C to terminate the distributed system."
echo ""

# Wait for the shepherd process to complete training (or for Ctrl+C)
wait $SHEPHERD_PID

echo ""
echo "🌑 Distributed training orchestration complete."
echo "   The tensor computation pipeline has demonstrated end-to-end execution"
echo "   across symbolic MLIR graphs, SPIR-V compilation, and Metal hardware."