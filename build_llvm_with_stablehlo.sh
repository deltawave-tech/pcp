#!/bin/bash

# Build script for LLVM/MLIR with StableHLO integration
# This script rebuilds LLVM with StableHLO as an external project

set -e  # Exit on any error

echo "⚕️ Building LLVM/MLIR with StableHLO integration..."

# Get absolute paths
PCP_PROJECT_ROOT=$(pwd)
LLVM_SOURCE_DIR="$PCP_PROJECT_ROOT/llvm-project"
LLVM_BUILD_DIR="$PCP_PROJECT_ROOT/llvm-build"
STABLEHLO_SOURCE_DIR="$PCP_PROJECT_ROOT/stablehlo"

echo "📁 Project root: $PCP_PROJECT_ROOT"
echo "📁 LLVM source: $LLVM_SOURCE_DIR"
echo "📁 LLVM build: $LLVM_BUILD_DIR"
echo "📁 StableHLO source: $STABLEHLO_SOURCE_DIR"

# Verify directories exist
if [ ! -d "$LLVM_SOURCE_DIR" ]; then
    echo "💣 Error: LLVM source directory not found at $LLVM_SOURCE_DIR"
    echo "   Make sure you cloned with --recursive or run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -d "$STABLEHLO_SOURCE_DIR" ]; then
    echo "💣 Error: StableHLO source directory not found at $STABLEHLO_SOURCE_DIR"
    echo "   Make sure you cloned with --recursive or run: git submodule update --init --recursive"
    exit 1
fi

# Create build directory
echo "📂 Creating build directory..."
mkdir -p "$LLVM_BUILD_DIR"
cd "$LLVM_BUILD_DIR"

# Configure LLVM with StableHLO
echo "⚕️  Configuring LLVM with StableHLO..."
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_EXTERNAL_PROJECTS="stablehlo" \
    -DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR="$STABLEHLO_SOURCE_DIR" \
    "$LLVM_SOURCE_DIR/llvm"

# Build LLVM with StableHLO
echo "⚕️ Building LLVM with StableHLO (this will take a while...)..."
ninja

echo "🪐 LLVM/MLIR with StableHLO build completed successfully!"
echo "📁 Build artifacts are in: $LLVM_BUILD_DIR"
echo "🌙 You can now run: zig build"
