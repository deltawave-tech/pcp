// In src/mlir/pass_anchors.cpp

#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir-c/IR.h"
#include <cstdio>

// This file provides stable C symbols for the C++ dialect registration functions.
// We only need to define the dialects we explicitly load in mlir_ctx.zig.

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Func, func, mlir::func::FuncDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arith, arith, mlir::arith::ArithDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Stablehlo, stablehlo, mlir::stablehlo::StablehloDialect)

// Packed argument struct to ensure stable ABI (all passed via pointer)
struct PcpOpArgs {
    intptr_t nResults;
    MlirType *results;
    intptr_t nOperands;
    MlirValue *operands;
    intptr_t nAttributes;
    MlirNamedAttribute *attributes;
    intptr_t nRegions;
    MlirRegion *regions;
};

MlirOperation pcpCreateOperation(
    const MlirStringRef *name,
    const MlirLocation *location,
    const PcpOpArgs *args // Passed by pointer, guarantees register passing (RDI, RSI, RDX)
) {
    MlirOperationState state = mlirOperationStateGet(*name, *location);

    if (args->nResults > 0 && args->results != nullptr) {
        mlirOperationStateAddResults(&state, args->nResults, args->results);
    }
    if (args->nOperands > 0 && args->operands != nullptr) {
        mlirOperationStateAddOperands(&state, args->nOperands, args->operands);
    }
    if (args->nAttributes > 0 && args->attributes != nullptr) {
        mlirOperationStateAddAttributes(&state, args->nAttributes, args->attributes);
    }
    if (args->nRegions > 0 && args->regions != nullptr) {
        mlirOperationStateAddOwnedRegions(&state, args->nRegions, args->regions);
    }

    return mlirOperationCreate(&state);
}

} // extern "C"