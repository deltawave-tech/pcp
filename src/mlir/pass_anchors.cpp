// In src/mlir/pass_anchors.cpp

#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "stablehlo/dialect/StablehloOps.h"

// This file provides stable C symbols for the C++ dialect registration functions.
// We only need to define the dialects we explicitly load in mlir_ctx.zig.

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Func, func, mlir::func::FuncDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arith, arith, mlir::arith::ArithDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Stablehlo, stablehlo, mlir::stablehlo::StablehloDialect)

} // extern "C"