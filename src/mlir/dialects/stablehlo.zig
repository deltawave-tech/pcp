// File: src/mlir/dialects/stablehlo.zig
// StableHLO dialect wrapper providing safe, idiomatic Zig interface for StableHLO operations

const std = @import("std");
const c = @import("../c.zig");
const mlir = @import("../../mlir.zig");

// --- Operation Builders for the StableHLO Dialect ---

/// Creates a stablehlo.add operation for element-wise addition
pub fn add(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.add", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()}, // Result type is same as broadcasted inputs
        .location = loc,
    });
}

/// Creates a stablehlo.subtract operation for element-wise subtraction
pub fn subtract(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.subtract", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.multiply operation for element-wise multiplication
pub fn multiply(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.multiply", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.divide operation for element-wise division
pub fn divide(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.divide", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.negate operation for element-wise negation
pub fn negate(ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.negate", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.constant operation
pub fn constant(ctx: mlir.Context, args: struct {
    value: mlir.Attribute,
    result_type: mlir.Type,
}) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.constant", .{
        .attributes = &.{.{ "value", args.value }},
        .results = &.{args.result_type},
        .location = mlir.Location.unknown(ctx), // Constants often have an unknown location
    });
}

/// Creates a stablehlo.maximum operation (used for ReLU)
pub fn maximum(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.maximum", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.minimum operation
pub fn minimum(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.minimum", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.compare operation
pub fn compare(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, direction: CompareDirection, loc: mlir.Location) mlir.Operation {
    const direction_attr = mlir.Attribute.integerAttr(ctx, @intFromEnum(direction), mlir.Type.f32Type(ctx));
    return mlir.Operation.create(ctx, "stablehlo.compare", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()}, // Should be tensor of i1, but simplified
        .attributes = &.{.{ "comparison_direction", direction_attr }},
        .location = loc,
    });
}

pub const CompareDirection = enum {
    EQ, // equal
    NE, // not equal
    LT, // less than
    LE, // less than or equal
    GT, // greater than
    GE, // greater than or equal
};

/// Creates a stablehlo.select operation
pub fn select(ctx: mlir.Context, pred: mlir.Value, on_true: mlir.Value, on_false: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.select", .{
        .operands = &.{ pred, on_true, on_false },
        .results = &.{on_true.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.transpose operation
pub fn transpose(ctx: mlir.Context, operand: mlir.Value, permutation: []const i64, loc: mlir.Location) mlir.Operation {
    const perm_attr = mlir.Attribute.denseI64ArrayAttr(ctx, permutation);
    
    // Calculate result type with transposed dimensions
    const input_type = operand.getType().as(mlir.RankedTensorType).?;
    
    var result_dims = std.ArrayList(i64).init(std.heap.page_allocator);
    defer result_dims.deinit();
    
    for (permutation) |perm_idx| {
        result_dims.append(input_type.getDimension(@intCast(perm_idx))) catch @panic("OOM");
    }
    
    const result_type = mlir.Type.tensor(result_dims.items, input_type.getElementType());
    
    return mlir.Operation.create(ctx, "stablehlo.transpose", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{.{ "permutation", perm_attr }},
        .location = loc,
    });
}

/// Creates a stablehlo.dot_general operation for matrix multiplication
/// This is the most complex operation due to its attributes
pub fn dot_general(
    ctx: mlir.Context,
    lhs: mlir.Value,
    rhs: mlir.Value,
    args: struct {
        dot_dimension_numbers: mlir.DotDimensionNumbersAttribute,
    },
) mlir.Operation {
    const lhs_type = lhs.getType().as(mlir.RankedTensorType).?;
    const rhs_type = rhs.getType().as(mlir.RankedTensorType).?;

    // Infer the output shape based on the dot dimension numbers
    // This is a simplified example. A full implementation would be more robust.
    var result_dims = std.ArrayList(i64).init(std.heap.page_allocator);
    defer result_dims.deinit();
    
    // Add batching dimensions
    for (args.dot_dimension_numbers.getLhsBatchingDimensions()) |dim_idx| {
        result_dims.append(lhs_type.getDimension(@intCast(dim_idx))) catch @panic("OOM");
    }
    
    // Add non-contracting, non-batching LHS dimensions
    for (0..lhs_type.getRank()) |i| {
        const i_i64 = @as(i64, @intCast(i));
        var is_batching = false;
        for (args.dot_dimension_numbers.getLhsBatchingDimensions()) |batch_dim| {
            if (i_i64 == batch_dim) {
                is_batching = true;
                break;
            }
        }
        var is_contracting = false;
        for (args.dot_dimension_numbers.getLhsContractingDimensions()) |contract_dim| {
            if (i_i64 == contract_dim) {
                is_contracting = true;
                break;
            }
        }
        if (!is_batching and !is_contracting) {
            result_dims.append(lhs_type.getDimension(i)) catch @panic("OOM");
        }
    }
    
    // Add non-contracting, non-batching RHS dimensions
    for (0..rhs_type.getRank()) |i| {
        const i_i64 = @as(i64, @intCast(i));
        var is_batching = false;
        for (args.dot_dimension_numbers.getRhsBatchingDimensions()) |batch_dim| {
            if (i_i64 == batch_dim) {
                is_batching = true;
                break;
            }
        }
        var is_contracting = false;
        for (args.dot_dimension_numbers.getRhsContractingDimensions()) |contract_dim| {
            if (i_i64 == contract_dim) {
                is_contracting = true;
                break;
            }
        }
        if (!is_batching and !is_contracting) {
            result_dims.append(rhs_type.getDimension(i)) catch @panic("OOM");
        }
    }
    
    const result_type = mlir.Type.tensor(result_dims.items, lhs_type.getElementType());

    return mlir.Operation.create(ctx, "stablehlo.dot_general", .{
        .operands = &.{ lhs, rhs },
        .results = &.{result_type},
        .attributes = &.{.{ "dot_dimension_numbers", args.dot_dimension_numbers.asAttr() }},
        .location = mlir.Location.unknown(ctx),
    });
}

/// Creates a stablehlo.reshape operation
pub fn reshape(ctx: mlir.Context, operand: mlir.Value, result_shape: []const i64, loc: mlir.Location) mlir.Operation {
    const input_type = operand.getType().as(mlir.RankedTensorType).?;
    const result_type = mlir.Type.tensor(result_shape, input_type.getElementType());
    
    return mlir.Operation.create(ctx, "stablehlo.reshape", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .location = loc,
    });
}

/// Creates a stablehlo.broadcast_in_dim operation
pub fn broadcast_in_dim(ctx: mlir.Context, operand: mlir.Value, result_shape: []const i64, broadcast_dimensions: []const i64, loc: mlir.Location) mlir.Operation {
    const input_type = operand.getType().as(mlir.RankedTensorType).?;
    const result_type = mlir.Type.tensor(result_shape, input_type.getElementType());
    const broadcast_dims_attr = mlir.Attribute.denseI64ArrayAttr(ctx, broadcast_dimensions);
    
    return mlir.Operation.create(ctx, "stablehlo.broadcast_in_dim", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{.{ "broadcast_dimensions", broadcast_dims_attr }},
        .location = loc,
    });
}

/// Creates a stablehlo.concatenate operation
pub fn concatenate(ctx: mlir.Context, operands: []const mlir.Value, dimension: i64, loc: mlir.Location) mlir.Operation {
    // All operands should have the same type except for the concatenation dimension
    const first_type = operands[0].getType().as(mlir.RankedTensorType).?;
    
    // Calculate result shape (sum along concatenation dimension)
    var result_dims = std.ArrayList(i64).init(std.heap.page_allocator);
    defer result_dims.deinit();
    
    for (0..first_type.getRank()) |i| {
        if (@as(i64, @intCast(i)) == dimension) {
            var total_dim: i64 = 0;
            for (operands) |operand| {
                const operand_type = operand.getType().as(mlir.RankedTensorType).?;
                total_dim += operand_type.getDimension(i);
            }
            result_dims.append(total_dim) catch @panic("OOM");
        } else {
            result_dims.append(first_type.getDimension(i)) catch @panic("OOM");
        }
    }
    
    const result_type = mlir.Type.tensor(result_dims.items, first_type.getElementType());
    const dimension_attr = mlir.Attribute.integerAttr(ctx, dimension, mlir.Type.f32Type(ctx)); // Should be i64 type
    
    // Convert []const mlir.Value to [*]*c.c.MlirValue for C API
    var c_operands = std.ArrayList(*c.c.MlirValue).init(std.heap.page_allocator);
    defer c_operands.deinit();
    
    for (operands) |operand| {
        c_operands.append(operand.handle) catch @panic("OOM");
    }
    
    return mlir.Operation.create(ctx, "stablehlo.concatenate", .{
        .operands = operands,
        .results = &.{result_type},
        .attributes = &.{.{ "dimension", dimension_attr }},
        .location = loc,
    });
}

/// Creates a stablehlo.reduce operation (simplified version)
pub fn reduce(ctx: mlir.Context, operand: mlir.Value, init_value: mlir.Value, dimensions: []const i64, loc: mlir.Location) mlir.Operation {
    const input_type = operand.getType().as(mlir.RankedTensorType).?;
    
    // Calculate result shape (remove reduced dimensions)
    var result_dims = std.ArrayList(i64).init(std.heap.page_allocator);
    defer result_dims.deinit();
    
    for (0..input_type.getRank()) |i| {
        const i_i64 = @as(i64, @intCast(i));
        var is_reduced = false;
        for (dimensions) |dim| {
            if (i_i64 == dim) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            result_dims.append(input_type.getDimension(i)) catch @panic("OOM");
        }
    }
    
    const result_type = mlir.Type.tensor(result_dims.items, input_type.getElementType());
    const dimensions_attr = mlir.Attribute.denseI64ArrayAttr(ctx, dimensions);
    
    return mlir.Operation.create(ctx, "stablehlo.reduce", .{
        .operands = &.{ operand, init_value },
        .results = &.{result_type},
        .attributes = &.{.{ "dimensions", dimensions_attr }},
        .location = loc,
    });
}

/// Utility functions for creating common constants

/// Create a scalar constant
pub fn scalarConstant(ctx: mlir.Context, value: f64, element_type: mlir.Type) mlir.Operation {
    const attr = mlir.Attribute.floatAttr(ctx, value, element_type);
    const scalar_type = mlir.Type.tensor(&.{}, element_type); // Scalar tensor (rank 0)
    
    return constant(ctx, .{
        .value = attr,
        .result_type = scalar_type,
    });
}

/// Create a zero constant of given shape and type
pub fn zeroConstant(ctx: mlir.Context, shape: []const i64, element_type: mlir.Type) mlir.Operation {
    const attr = mlir.Attribute.floatAttr(ctx, 0.0, element_type);
    const tensor_type = mlir.Type.tensor(shape, element_type);
    
    return constant(ctx, .{
        .value = attr,
        .result_type = tensor_type,
    });
}

/// Create a ones constant of given shape and type
pub fn onesConstant(ctx: mlir.Context, shape: []const i64, element_type: mlir.Type) mlir.Operation {
    const attr = mlir.Attribute.floatAttr(ctx, 1.0, element_type);
    const tensor_type = mlir.Type.tensor(shape, element_type);
    
    return constant(ctx, .{
        .value = attr,
        .result_type = tensor_type,
    });
}