#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

// Real MLIR C++ headers for SPIR-V serialization
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

extern "C" {

typedef void (*MlirStringCallback)(MlirStringRef, void*);

// Real C API wrapper for SPIR-V serialization using MLIR's actual serializer
MlirLogicalResult mlirTranslateModuleToSPIRV(
    MlirModule module,
    MlirStringCallback callback,
    void* userData) {
    
    // Unwrap the C MlirModule handle to a C++ mlir::ModuleOp
    mlir::ModuleOp moduleOp = unwrap(module);
    if (!moduleOp) {
        return mlirLogicalResultFailure();
    }

    // Find the SPIR-V module inside the regular module
    mlir::spirv::ModuleOp spirvModule;
    moduleOp.walk([&](mlir::spirv::ModuleOp spvMod) {
        spirvModule = spvMod;
        return mlir::WalkResult::interrupt();
    });

    if (!spirvModule) {
        // No SPIR-V module found
        return mlirLogicalResultFailure();
    }

    // SmallVector to hold the generated SPIR-V binary words
    llvm::SmallVector<uint32_t> spirv_binary;

    // Call the actual MLIR SPIR-V serialization function
    // This populates the spirv_binary vector with real SPIR-V bytecode
    mlir::LogicalResult result = mlir::spirv::serialize(spirvModule, spirv_binary);

    if (mlir::failed(result)) {
        // Serialization failed - return failure
        return mlirLogicalResultFailure();
    }

    // Serialization succeeded - pass the real SPIR-V binary back to Zig
    MlirStringRef resultRef = {
        .data = reinterpret_cast<const char*>(spirv_binary.data()),
        .length = spirv_binary.size() * sizeof(uint32_t)
    };
    
    callback(resultRef, userData);
    
    return mlirLogicalResultSuccess();
}

}