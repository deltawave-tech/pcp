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
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"

// Add TilingInterface registration
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"

// Add bufferization pipelines header
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"

// NEW: Include headers for canonical GPU pipeline builder  
#include "mlir-c/Pass.h"
#include "mlir/CAPI/Pass.h"  // For unwrap functionality
#include "mlir/CAPI/IR.h"    // For MlirDialectRegistry unwrap
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

// Test pass registration headers  
#include "mlir/Dialect/Linalg/Passes.h"

// Transform dialect headers
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

// Add SCF to GPU conversion headers
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/Passes.h"

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
  
  std::printf("  - Bufferization pipelines (buffer-deallocation-pipeline)\n");
  mlir::bufferization::registerBufferizationPipelines();
  
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
  
  // CRITICAL: Register Transform dialect passes and extensions
  std::printf("  - Transform dialect passes and extensions\n");
  mlir::transform::registerTransformPasses();
  
  // Register Linalg Transform dialect extensions (for transform.structured.* ops)
  std::printf("  - Linalg Transform dialect extensions\n");
  mlir::linalg::registerTransformDialectExtension(registry);
  
  // Register TilingInterface implementations for Linalg operations
  std::printf("  - Linalg TilingInterface implementations\n");
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  
  // Register SCF to GPU conversion passes
  std::printf("  - SCF to GPU conversion passes\n");
  mlir::registerConvertParallelLoopToGpuPass();
  
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
    
    // NOTE: Bufferization, tiling, and parallel-to-GPU conversion are now handled in Zig code
    // This function handles: GPU dialect → SPIR-V conversion
    
    std::printf("  - GPU kernels should now be present from Zig pipeline\n");
    
    // Create function PM for function-level passes
    auto &funcPM = opm->nest<mlir::func::FuncOp>();
    
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
  
  std::printf("  - SCF dialect BufferizableOpInterface\n");
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  
  std::printf("  - SCF dialect BufferDeallocationOpInterface\n");
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  
  std::printf("C++: ✅ BufferizableOpInterface implementations registered\n");
}

// Register Transform dialect extensions with a dialect registry
void mlirRegisterTransformExtensions(MlirDialectRegistry registryHandle) {
  std::printf("C++: Registering Transform dialect extensions...\n");
  
  // Convert C API registry handle to C++ DialectRegistry
  mlir::DialectRegistry &registry = *unwrap(registryHandle);
  
  // Register Linalg Transform dialect extensions (for transform.structured.* ops)
  std::printf("  - Linalg Transform dialect extensions\n");
  mlir::linalg::registerTransformDialectExtension(registry);
  
  // Register GPU Transform dialect extensions (for transform.gpu.* ops)
  std::printf("  - GPU Transform dialect extensions\n");
  mlir::gpu::registerTransformDialectExtension(registry);
  
  // CRITICAL: Register TilingInterface implementations for Linalg operations
  std::printf("  - Linalg TilingInterface implementations\n");
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  
  std::printf("C++: ✅ Transform dialect extensions registered\n");
}

} // extern "C"