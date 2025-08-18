// File: src/mlir/pass_anchors.cpp
// Force-load bridge to ensure pass libraries are included by the linker

// These are the real C++ headers for the pass creation functions.
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h" // For StableHLO passes

// We use extern "C" to make these functions callable from Zig.
extern "C" {

// A single function to anchor all necessary pass libraries.
// This forces the linker to include all required pass libraries,
// ensuring their C++ static initializers run and register the passes.
void mlirForceLoadAllRequiredPasses() {
  // StableHLO pass registration
  mlir::stablehlo::registerPassPipelines();

  // Canonicalization and cleanup - use default parameters for overloaded functions
  (void)mlir::createCanonicalizerPass();
  (void)mlir::createCSEPass;

  // Core Conversions - use default parameters
  (void)mlir::createConvertFuncToLLVMPass();
  (void)mlir::createConvertLinalgToLoopsPass;
  (void)mlir::createSCFToControlFlowPass; // Updated API name

  // GPU and SPIR-V Conversions - use default parameters
  (void)mlir::createGpuToLLVMConversionPass();
  (void)mlir::createConvertGPUToSPIRVPass; // Updated API name

  // Bufferization
  (void)mlir::bufferization::createEmptyTensorToAllocTensorPass;

  // Original anchors
  (void)mlir::createConvertLinalgToAffineLoopsPass; // Anchors Linalg
  (void)mlir::createGpuAsyncRegionPass;             // Anchors GPU
  (void)mlir::createSCFToControlFlowPass;           // Anchors SCF
  (void)mlir::createLowerAffinePass;                // Anchors Affine
  (void)mlir::memref::createExpandOpsPass;          // Anchors MemRef
}

} // extern "C"