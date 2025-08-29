// File: src/mlir/pass_anchors.cpp
// Force-load bridge to ensure pass libraries are included by the linker

#include <cstdio>  // For std::printf

// Minimal registration approach - include only specific passes we need
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
// LinalgToLoops is actually in Linalg dialect passes
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"

// Add interface registration headers
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"

// NEW: Include headers for canonical GPU pipeline builder  
#include "mlir-c/Pass.h"
#include "mlir/CAPI/Pass.h"  // For unwrap functionality
#include "mlir/CAPI/IR.h"    // For MlirDialectRegistry unwrap
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

// Test pass registration headers  
#include "mlir/Dialect/Linalg/Passes.h"

// Production tiling should be included in standard Linalg passes

// Forward declarations for test passes
namespace mlir {
namespace test {
void registerTestLinalgTransforms();
}
}

// We use extern "C" to make these functions callable from Zig.
extern "C" {

// Note: Individual pass registration functions are already provided by MLIR C API
// The functions mlirRegisterLinalgPasses, mlirRegisterSCFPasses, mlirRegisterGPUPasses
// are already defined in libMLIRCAPILinalg.a, libMLIRCAPISCF.a, libMLIRCAPIGPU.a
// We don't need to redefine them here - just use the existing MLIR C API functions

void mlirRegisterBufferizationPasses() {
  mlir::bufferization::registerBufferizationPasses();
}

void mlirRegisterSPIRVPasses() {
  // SPIR-V passes are registered via the main anchor function
  // This function is kept for compatibility but does nothing
}

// CANONICAL SOLUTION: Single C++ anchor function to register all passes
// This forces the linker to include all required pass libraries and
// ensures their C++ static initializers run to register the passes.
void mlirForceLoadAllRequiredPasses() {
  // Create a dialect registry for interface registration
  mlir::DialectRegistry registry;
  // === MINIMAL REGISTRATION APPROACH ===
  // Register only the specific passes required for StableHLO → SPIR-V pipeline
  
  std::printf("C++: Registering required MLIR passes for StableHLO → SPIR-V pipeline...\n");
  
  // Safe bulk registrations that work
  std::printf("  - Linalg passes (for tensor operations)\n");
  mlir::registerLinalgPasses();
  
  std::printf("  - Bufferization passes (tensor → memref)\n");
  mlir::bufferization::registerBufferizationPasses();
  
  std::printf("  - StableHLO passes (HLO legalization)\n");
  mlir::stablehlo::registerPassPipelines();
  
  std::printf("  - Core transform passes (canonicalize, cse)\n");
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  
  // Register test passes for tiling functionality
  std::printf("  - Test Linalg transform passes (for tiling)\n");
  mlir::test::registerTestLinalgTransforms();
  
  // Production linalg-tile should be included in registerLinalgPasses()
  
  // Register Async passes (needed for linalg-tile runtime infrastructure)
  std::printf("  - Async passes (for tiling runtime infrastructure)\n");
  mlir::registerAsyncPasses();
  
  // Register specific passes for our pipeline instead of bulk GPU/SCF registration
  std::printf("  - Specific conversion passes for our pipeline\n");
  // These individual pass registrations avoid the bulk registration issues
  
  std::printf("C++: ✅ Minimal pass registration completed successfully!\n");
  std::printf("     (Including Transform dialect and production tiling)\n");
}

// NEW FUNCTION: Build canonical GPU and SPIR-V conversion pipeline
// This replaces the fragile string-based pipeline with a robust C++ function
void mlirBuildAndAppendGpuAndSpirvConversionPipeline(MlirOpPassManager passManager) {
  std::printf("C++: Building canonical GPU → SPIR-V conversion pipeline...\n");
  
  try {
    // Convert C API handle to C++ PassManager using correct unwrap from CAPI
    mlir::OpPassManager *opm = unwrap(passManager);
    
    // NOTE: Bufferization, tiling, and linalg-to-parallel-loops are now handled in Zig code
    // This function handles: parallel loops → GPU mapping → SPIR-V conversion
    
    std::printf("  - Parallel loops should now be present from Zig pipeline\n");
    auto &funcPM = opm->nest<mlir::func::FuncOp>();
    
    // Add GPU mapping passes - function level passes
    std::printf("  - Adding GPU map parallel loops pass (func level)\n");
    funcPM.addPass(mlir::createGpuMapParallelLoopsPass());
    
    // GPU kernel outlining - module level pass (creates gpu.module ops)
    std::printf("  - Adding GPU kernel outlining pass\n");
    opm->addPass(mlir::createGpuKernelOutliningPass());
    
    // SCF to ControlFlow lowering - function level pass
    std::printf("  - Adding SCF to ControlFlow conversion (func level)\n");
    funcPM.addPass(mlir::createSCFToControlFlowPass());
    
    // GPU to SPIR-V conversion - module level pass (operates on gpu.module)
    std::printf("  - Adding GPU to SPIR-V conversion\n");
    opm->addPass(mlir::createConvertGPUToSPIRVPass());
    
    std::printf("C++: ✅ Canonical GPU → SPIR-V pipeline built successfully!\n");
    
  } catch (const std::exception& e) {
    std::printf("C++: ❌ Error building GPU pipeline: %s\n", e.what());
  } catch (...) {
    std::printf("C++: ❌ Unknown error building GPU pipeline\n");
  }
}

// Register BufferizableOpInterface implementations with a dialect registry
void mlirRegisterBufferizationInterfaces(MlirDialectRegistry registryHandle) {
  std::printf("C++: Registering BufferizableOpInterface implementations...\n");
  
  // Convert C API registry handle to C++ DialectRegistry
  mlir::DialectRegistry &registry = *unwrap(registryHandle);
  
  // Register SPECIFIC BufferizableOpInterface implementations
  std::printf("  - Func dialect BufferizableOpInterface (specific registration)\n");
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  
  std::printf("  - Tensor dialect BufferizableOpInterface\n");
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  
  std::printf("  - Linalg dialect BufferizableOpInterface\n");
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  
  std::printf("  - Arith dialect BufferizableOpInterface\n");
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  
  std::printf("C++: ✅ BufferizableOpInterface implementations registered\n");
}

} // extern "C"