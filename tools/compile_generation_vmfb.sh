#!/bin/bash

# Pre-compile Qwen RL Generation Graph to VMFB
# This only needs to be run ONCE (or when the model changes)
# Compilation can take 15-30 minutes for this 3.7GB model

set -e

IREE_COMPILE="/home/phil/projects/iree-build/tools/iree-compile"
INPUT_MLIR="models/qwen_rl_generation.mlir"
OUTPUT_VMFB="models/qwen_rl_generation.vmfb"

# Check if input exists
if [ ! -f "$INPUT_MLIR" ]; then
    echo "Error: $INPUT_MLIR not found"
    exit 1
fi

# Check if iree-compile exists
if [ ! -f "$IREE_COMPILE" ]; then
    echo "Error: iree-compile not found at $IREE_COMPILE"
    exit 1
fi

echo "=========================================="
echo "Pre-compiling Qwen RL Generation VMFB"
echo "=========================================="
echo "Input:  $INPUT_MLIR"
echo "Output: $OUTPUT_VMFB"
echo "Target: CUDA (Workers will generate on GPU)"
echo ""
echo "⚠️  This will take 15-30 minutes..."
echo "=========================================="
echo ""

# Compile with CUDA target and 64-bit index support
time $IREE_COMPILE "$INPUT_MLIR" \
  -o "$OUTPUT_VMFB" \
  --iree-hal-target-backends=cuda \
  --iree-hal-target-device=cuda \
  --iree-vm-target-index-bits=64 \
  --iree-stream-resource-index-bits=64 \
  --iree-input-demote-i64-to-i32=false

echo ""
echo "=========================================="
echo "✓ Compilation Complete!"
echo "=========================================="
ls -lh "$OUTPUT_VMFB"
echo ""
echo "You can now run: ./run_qwen_rl_test.sh"
echo "Workers will load generation VMFB instantly!"
