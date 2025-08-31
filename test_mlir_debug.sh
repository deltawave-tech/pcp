#!/bin/bash
set -e

echo "Starting comprehensive MLIR debugging test suite..."
echo ""

# Test 1: Null pointer checks
echo "Building null pointer check test..."
zig build-exe src/minimal_null_check_test.zig -I src -L llvm-build/lib -l c -l MLIR-C -l LLVMSupport -O Debug

echo "🧪 Running null pointer check test..."
echo "================================================="
if ./minimal_null_check_test; then
    echo "🌙 Null pointer check test passed!"
else
    echo "💣 Null pointer check test failed!"
    echo "   This indicates a fundamental MLIR C API issue."
    exit 1
fi

echo ""

# Test 2: Basic MLIR type operations
echo "Building minimal MLIR type test..."
zig build-exe src/minimal_mlir_type_test.zig -I src -L llvm-build/lib -l c -l MLIR-C -l LLVMSupport -O Debug

echo "🧪 Running minimal MLIR type test..."
echo "================================================="
if ./minimal_mlir_type_test; then
    echo "🌙 Basic MLIR type test passed!"
else
    echo "💣 Basic MLIR type test failed!"
    echo "   This indicates an issue with high-level MLIR type operations."
    exit 1
fi

echo ""

# Test 3: Autodiff system (most likely to fail)
echo "Building minimal autodiff test..."
zig build-exe src/minimal_autodiff_test.zig -I src -L llvm-build/lib -l c -l MLIR-C -l LLVMSupport -O Debug

echo "🧪 Running minimal autodiff test..."
echo "================================================="
if ./minimal_autodiff_test; then
    echo "🌙 Autodiff test passed!"
    echo "   The segfault is likely in a more complex DiLoCo-specific interaction."
else
    echo "💣 Autodiff test failed!"
    echo "   This is likely where the segfault occurs in DiLoCo!"
    echo "   Check the error output above for specific failure points."
    exit 1
fi

echo ""
echo "🌚 ALL MINIMAL TESTS COMPLETED SUCCESSFULLY!"
echo ""
echo "SUMMARY:"
echo "   ✓ MLIR C API null pointer safety: PASSED"
echo "   ✓ Basic MLIR type operations: PASSED"  
echo "   ✓ Autodiff graph building: PASSED"
echo ""
echo "NEXT STEPS:"
echo "   Since all minimal tests pass, the segfault is likely occurring in:"
echo "   1. Complex tensor data manipulation in DiLoCo"
echo "   2. Multi-threaded MLIR context usage"
echo "   3. Memory corruption in parameter serialization"
echo "   4. Backend-specific execution paths"
echo ""
echo "   Run the actual DiLoCo demo with detailed logging to narrow down the issue."