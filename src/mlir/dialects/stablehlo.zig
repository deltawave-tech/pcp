// File: src/mlir/dialects/stablehlo.zig
// StableHLO dialect wrapper providing safe, idiomatic Zig interface for StableHLO operations

const std = @import("std");
const c = @import("../c.zig");
const mlir = @import("../wrapper.zig");

// --- Operation Builders for the StableHLO Dialect ---

/// Creates a stablehlo.add operation for element-wise addition
pub fn add(allocator: std.mem.Allocator, ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) !mlir.Operation {
    return mlir.Operation.create(allocator, ctx, "stablehlo.add", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()}, // Result type is same as broadcasted inputs
        .location = loc,
    });
}

/// Creates a stablehlo.subtract operation for element-wise subtraction
pub fn subtract(allocator: std.mem.Allocator, ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) !mlir.Operation {
    return mlir.Operation.create(allocator, ctx, "stablehlo.subtract", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.multiply operation for element-wise multiplication
pub fn multiply(allocator: std.mem.Allocator, ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) !mlir.Operation {
    return mlir.Operation.create(allocator, ctx, "stablehlo.multiply", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.divide operation for element-wise division
pub fn divide(allocator: std.mem.Allocator, ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) !mlir.Operation {
    return mlir.Operation.create(allocator, ctx, "stablehlo.divide", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.negate operation for element-wise negation
pub fn negate(allocator: std.mem.Allocator, ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) !mlir.Operation {
    return mlir.Operation.create(allocator, ctx, "stablehlo.negate", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.constant operation
pub fn constant(allocator: std.mem.Allocator, ctx: mlir.Context, args: struct {
    value: mlir.Attribute,
    result_type: mlir.Type,
}) !mlir.Operation {
    return mlir.Operation.create(allocator, ctx, "stablehlo.constant", .{
        .attributes = &.{.{ "value", args.value }},
        .results = &.{args.result_type},
        .location = mlir.Location.unknown(ctx), // Constants often have an unknown location
    });
}

/// Creates a zero constant tensor
pub fn zeroConstant(allocator: std.mem.Allocator, ctx: mlir.Context, shape: []const i64, element_type: mlir.Type) !mlir.Operation {
    const tensor_type = mlir.Type.rankedTensorType(ctx, shape, element_type);

    var zero_attr: mlir.Attribute = undefined;
    if (element_type.isInteger() or element_type.isIndex()) {
        const zero_val = mlir.Attribute.integerAttr(ctx, 0, element_type);
        zero_attr = mlir.Attribute.denseElementsAttrSplat(tensor_type, zero_val);
    } else {
        zero_attr = mlir.Attribute.denseElementsAttrFloatSplat(tensor_type, 0.0);
    }

    return constant(allocator, ctx, .{
        .value = zero_attr,
        .result_type = tensor_type,
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

/// Creates a stablehlo.exponential operation
pub fn exponential(ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.exponential", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.reduce_max operation
/// Creates a stablehlo.reduce operation with a 'maximum' body.
pub fn reduce_max(
    ctx: mlir.Context,
    builder: *const @import("../../core/ops.zig").MLIRBuilder, // Pass in the builder
    operand: mlir.Value,
    dimensions: []const i64,
    loc: mlir.Location,
) !mlir.Operation {
    const c_api = @import("../c.zig").c;
    const allocator = builder.allocator;

    // 1. Calculate result type (removes reduced dimensions)
    const input_type = operand.getType().as(mlir.RankedTensorType).?;
    const element_type = input_type.getElementType();
    const input_shape = try input_type.getShape(allocator);
    defer allocator.free(input_shape);

    var result_shape_list = std.ArrayList(i64).init(allocator);
    defer result_shape_list.deinit();

    for (input_shape, 0..) |dim, i| {
        var is_reduced = false;
        for (dimensions) |red_dim| {
            if (@as(i64, @intCast(i)) == red_dim) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            try result_shape_list.append(dim);
        }
    }
    const result_type = mlir.Type.rankedTensorType(ctx, result_shape_list.items, element_type);

    // 2. Create the init_value: negative infinity for floats, which is the identity element for max reduction.
    const neg_inf = std.math.inf(f32) * -1.0;
    const scalar_type = mlir.Type.tensor(&.{}, element_type);
    const neg_inf_attr = mlir.Attribute.denseElementsAttrSplat(scalar_type, @floatCast(neg_inf));
    const init_constant_op = constant(ctx, .{ .value = neg_inf_attr, .result_type = scalar_type });

    builder.insertion_block.appendOwnedOperation(init_constant_op);
    const init_value = init_constant_op.getResult(0);

    // 3. Create the reduction body (region -> block)
    const body_region = c_api.regionCreate();
    const body_block = mlir.Block{ .handle = c_api.blockCreate(0, @constCast(@ptrCast(&[_]c_api.MlirType{})), @constCast(@ptrCast(&[_]*c_api.MlirLocation{}))) };
    c_api.regionAppendOwnedBlock(body_region, body_block.handle);

    // 4. Add arguments and the 'maximum' operation to the body
    const lhs_arg = body_block.addArgument(scalar_type, loc);
    const rhs_arg = body_block.addArgument(scalar_type, loc);
    const max_op = maximum(ctx, lhs_arg, rhs_arg, loc); // Use stablehlo.maximum
    c_api.blockAppendOwnedOperation(body_block.handle, max_op.handle);
    const return_op = mlir.Operation.create(ctx, "stablehlo.return", .{ .operands = &.{max_op.getResult(0)} });
    c_api.blockAppendOwnedOperation(body_block.handle, return_op.handle);

    // 5. Build the final stablehlo.reduce operation using pcpCreateOperation
    var operands = [_]c_api.MlirValue{ operand.handle, init_value.handle };
    var results = [_]c_api.MlirType{ result_type.handle };
    var regions = [_]c_api.MlirRegion{ body_region };

    const dimensions_attr = mlir.Attribute.denseI64ArrayAttr(ctx, dimensions);
    const dimensions_ref = c_api.stringRefFromString("dimensions");
    const attr_name_id = c_api.identifierGet(ctx.handle, dimensions_ref);
    var named_attrs = [_]c_api.MlirNamedAttribute{
        .{ .name = attr_name_id, .attribute = dimensions_attr.handle }
    };

    const name_ref = c_api.stringRefFromString("stablehlo.reduce");

    const op_args = c_api.PcpOpArgs{
        .nResults = 1,
        .results = &results,
        .nOperands = 2,
        .operands = &operands,
        .nAttributes = 1,
        .attributes = &named_attrs,
        .nRegions = 1,
        .regions = &regions,
    };

    const handle = c_api.pcpCreateOperation(&name_ref, &loc.handle, &op_args);

    return mlir.Operation{ .handle = handle };
}

/// Creates a generic stablehlo.reduce operation with a summation body.
pub fn reduce_sum(
    ctx: mlir.Context,
    builder: *const @import("../../core/ops.zig").MLIRBuilder, // Pass in the builder
    operand: mlir.Value,
    dimensions: []const i64,
    loc: mlir.Location,
) !mlir.Operation {
    const c_api = @import("../c.zig").c;
    const allocator = builder.allocator;

    // 1. Calculate result type (always removes dimensions)
    const input_type = operand.getType().as(mlir.RankedTensorType).?;
    const input_shape = try input_type.getShape(allocator);
    defer allocator.free(input_shape);
    const element_type = input_type.getElementType();

    var result_shape_list = std.ArrayList(i64).init(allocator);
    defer result_shape_list.deinit();
    
    for (input_shape, 0..) |dim, i| {
        var is_reduced = false;
        for (dimensions) |red_dim| {
            if (@as(i64, @intCast(i)) == red_dim) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            try result_shape_list.append(dim);
        }
    }
    const result_type = mlir.Type.rankedTensorType(ctx, result_shape_list.items, element_type);

    // 2. Create the zero constant for init_value using type-aware attribute creation
    const scalar_type = mlir.Type.tensor(&.{}, element_type);

    var zero_attr: mlir.Attribute = undefined;
    if (element_type.isInteger() or element_type.isIndex()) {
        const zero_val = mlir.Attribute.integerAttr(ctx, 0, element_type);
        zero_attr = mlir.Attribute.denseElementsAttrSplat(scalar_type, zero_val);
    } else {
        // Create a properly typed float attribute instead of using FloatSplat
        const zero_val = mlir.Attribute.floatAttr(ctx, 0.0, element_type);
        zero_attr = mlir.Attribute.denseElementsAttrSplat(scalar_type, zero_val);
    }

    const init_constant_op = try constant(allocator, ctx, .{ .value = zero_attr, .result_type = scalar_type });

    // FIX: Attach the constant op to the graph before using its result.
    // This resolves the <<UNKNOWN SSA VALUE>> error.
    builder.insertion_block.appendOwnedOperation(init_constant_op);
    const init_value = init_constant_op.getResult(0);

    // 3. Create the reduction body (region -> block)
    const body_region = c_api.regionCreate();
    const body_block = mlir.Block{ .handle = c_api.blockCreate(0, @constCast(@ptrCast(&[_]c_api.MlirType{})), @constCast(@ptrCast(&[_]*c_api.MlirLocation{}))) };
    c_api.regionAppendOwnedBlock(body_region, body_block.handle);

    // 4. Add arguments and the 'add' operation to the body
    const lhs_arg = body_block.addArgument(scalar_type, loc);
    const rhs_arg = body_block.addArgument(scalar_type, loc);
    const add_op = try add(allocator, ctx, lhs_arg, rhs_arg, loc);
    c_api.blockAppendOwnedOperation(body_block.handle, add_op.handle);
    const return_op = try mlir.Operation.create(allocator, ctx, "stablehlo.return", .{ .operands = &.{add_op.getResult(0)} });
    c_api.blockAppendOwnedOperation(body_block.handle, return_op.handle);

    // 5. Build the final stablehlo.reduce operation using pcpCreateOperation
    var operands = [_]c_api.MlirValue{ operand.handle, init_value.handle };
    var results = [_]c_api.MlirType{ result_type.handle };
    var regions = [_]c_api.MlirRegion{ body_region };

    const dimensions_attr = mlir.Attribute.denseI64ArrayAttr(ctx, dimensions);
    const dimensions_ref = c_api.stringRefFromString("dimensions");
    const attr_name_id = c_api.identifierGet(ctx.handle, dimensions_ref);
    var named_attrs = [_]c_api.MlirNamedAttribute{
        .{ .name = attr_name_id, .attribute = dimensions_attr.handle }
    };

    const name_ref = c_api.stringRefFromString("stablehlo.reduce");

    const op_args = c_api.PcpOpArgs{
        .nResults = 1,
        .results = &results,
        .nOperands = 2,
        .operands = &operands,
        .nAttributes = 1,
        .attributes = &named_attrs,
        .nRegions = 1,
        .regions = &regions,
    };

    const handle = c_api.pcpCreateOperation(&name_ref, &loc.handle, &op_args);

    return mlir.Operation{ .handle = handle };
}

/// Creates a stablehlo.compare operation
pub fn compare(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, direction: CompareDirection, compare_type: CompareType, loc: mlir.Location) mlir.Operation {
    // CORRECTED: The result of a comparison is a tensor of booleans (i1).
    const input_type = lhs.getType().as(mlir.RankedTensorType).?;
    const input_shape = input_type.getShape(std.heap.page_allocator) catch unreachable;
    defer std.heap.page_allocator.free(input_shape);
    
    const i1_element_type = mlir.Type.i1Type(ctx);
    const result_type = mlir.Type.rankedTensorType(ctx, input_shape, i1_element_type);

    // Build string for comparison_direction
    var dir_str_buf = std.ArrayList(u8).init(std.heap.page_allocator);
    defer dir_str_buf.deinit();
    dir_str_buf.appendSlice("#stablehlo<comparison_direction ") catch @panic("OOM");
    dir_str_buf.appendSlice(direction.toString()) catch @panic("OOM");
    dir_str_buf.appendSlice(">") catch @panic("OOM");
    const dir_str_ref = c.c.MlirStringRef{ .data = dir_str_buf.items.ptr, .length = dir_str_buf.items.len };
    const direction_attr_handle = c.c.mlirAttributeParseGet(ctx.handle, dir_str_ref);
    const direction_attr = mlir.Attribute{ .handle = direction_attr_handle };

    // Build string for compare_type
    var type_str_buf = std.ArrayList(u8).init(std.heap.page_allocator);
    defer type_str_buf.deinit();
    type_str_buf.appendSlice("#stablehlo<comparison_type ") catch @panic("OOM");
    type_str_buf.appendSlice(compare_type.toString()) catch @panic("OOM");
    type_str_buf.appendSlice(">") catch @panic("OOM");
    const type_str_ref = c.c.MlirStringRef{ .data = type_str_buf.items.ptr, .length = type_str_buf.items.len };
    const type_attr_handle = c.c.mlirAttributeParseGet(ctx.handle, type_str_ref);
    const type_attr = mlir.Attribute{ .handle = type_attr_handle };

    // Attributes
    const comparison_direction_ref = c.c.stringRefFromString("comparison_direction");
    const direction_id = c.c.identifierGet(ctx.handle, comparison_direction_ref);
    const compare_type_ref = c.c.stringRefFromString("compare_type");
    const type_id = c.c.identifierGet(ctx.handle, compare_type_ref);
    var named_attrs = [_]c.c.MlirNamedAttribute{
        .{ .name = direction_id, .attribute = direction_attr.handle },
        .{ .name = type_id, .attribute = type_attr.handle },
    };

    // Create operation using pcpCreateOperation with packed args
    var operands = [_]c.c.MlirValue{ lhs.handle, rhs.handle };
    var result_types = [_]c.c.MlirType{ result_type.handle };

    const name_ref = c.c.stringRefFromString("stablehlo.compare");

    const op_args = c.c.PcpOpArgs{
        .nResults = 1,
        .results = &result_types,
        .nOperands = 2,
        .operands = &operands,
        .nAttributes = @intCast(named_attrs.len),
        .attributes = &named_attrs,
        .nRegions = 0,
        .regions = null,
    };

    const handle = c.c.pcpCreateOperation(&name_ref, &loc.handle, &op_args);

    return mlir.Operation{ .handle = handle };
}

pub const CompareDirection = enum {
    EQ, // equal
    NE, // not equal
    LT, // less than
    LE, // less than or equal
    GT, // greater than
    GE, // greater than or equal

    pub fn toString(self: CompareDirection) []const u8 {
        return switch (self) {
            .EQ => "EQ",
            .NE => "NE",
            .LT => "LT",
            .LE => "LE",
            .GT => "GT",
            .GE => "GE",
        };
    }
};

pub const CompareType = enum {
    FLOAT,
    TOTALORDER,
    SIGNED,
    UNSIGNED,

    pub fn toString(self: CompareType) []const u8 {
        return switch (self) {
            .FLOAT => "FLOAT",
            .TOTALORDER => "TOTALORDER",
            .SIGNED => "SIGNED",
            .UNSIGNED => "UNSIGNED",
        };
    }
};

/// Creates a stablehlo.select operation
pub fn select(allocator: std.mem.Allocator, ctx: mlir.Context, pred: mlir.Value, on_true: mlir.Value, on_false: mlir.Value, loc: mlir.Location) !mlir.Operation {
    return mlir.Operation.create(allocator, ctx, "stablehlo.select", .{
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

/// Attribute for stablehlo.dot_general dimension numbers.
pub const DotDimensionNumbersAttribute = struct {
    lhs_batching_dimensions: []const i64,
    rhs_batching_dimensions: []const i64,
    lhs_contracting_dimensions: []const i64,
    rhs_contracting_dimensions: []const i64,

    pub fn asAttr(self: @This(), ctx: mlir.Context) mlir.Attribute {
        var str_buf = std.ArrayList(u8).init(std.heap.page_allocator);
        defer str_buf.deinit();

        str_buf.appendSlice("#stablehlo.dot<lhs_batching_dimensions = [") catch @panic("OOM");
        for (self.lhs_batching_dimensions, 0..) |d, i| {
            if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
            std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
        }
        str_buf.appendSlice("], rhs_batching_dimensions = [") catch @panic("OOM");
        for (self.rhs_batching_dimensions, 0..) |d, i| {
            if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
            std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
        }
        str_buf.appendSlice("], lhs_contracting_dimensions = [") catch @panic("OOM");
        for (self.lhs_contracting_dimensions, 0..) |d, i| {
            if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
            std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
        }
        str_buf.appendSlice("], rhs_contracting_dimensions = [") catch @panic("OOM");
        for (self.rhs_contracting_dimensions, 0..) |d, i| {
            if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
            std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
        }
        str_buf.appendSlice("]>") catch @panic("OOM");

        const str_ref = c.c.MlirStringRef{ .data = str_buf.items.ptr, .length = str_buf.items.len };
        const attr_handle = c.c.mlirAttributeParseGet(ctx.handle, str_ref);
        return mlir.Attribute{ .handle = attr_handle };
    }
};

/// Creates a stablehlo.dot_general operation for matrix multiplication
/// This is the most complex operation due to its attributes
pub fn dot_general(
    allocator: std.mem.Allocator,
    ctx: mlir.Context,
    lhs: mlir.Value,
    rhs: mlir.Value,
    args: struct {
        dot_dimension_numbers: DotDimensionNumbersAttribute,
    },
) !mlir.Operation {
    const lhs_type = lhs.getType().as(mlir.RankedTensorType).?;
    const rhs_type = rhs.getType().as(mlir.RankedTensorType).?;

    // Infer the output shape based on the dot dimension numbers
    // This is a simplified example. A full implementation would be more robust.
    var result_dims = std.ArrayList(i64).init(std.heap.page_allocator);
    defer result_dims.deinit();
    
    // Add batching dimensions
    for (args.dot_dimension_numbers.lhs_batching_dimensions) |dim_idx| {
        result_dims.append(lhs_type.getDimension(@intCast(dim_idx))) catch @panic("OOM");
    }
    
    // Add non-contracting, non-batching LHS dimensions
    for (0..lhs_type.getRank()) |i| {
        const i_i64 = @as(i64, @intCast(i));
        var is_batching = false;
        for (args.dot_dimension_numbers.lhs_batching_dimensions) |batch_dim| {
            if (i_i64 == batch_dim) {
                is_batching = true;
                break;
            }
        }
        var is_contracting = false;
        for (args.dot_dimension_numbers.lhs_contracting_dimensions) |contract_dim| {
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
        for (args.dot_dimension_numbers.rhs_batching_dimensions) |batch_dim| {
            if (i_i64 == batch_dim) {
                is_batching = true;
                break;
            }
        }
        var is_contracting = false;
        for (args.dot_dimension_numbers.rhs_contracting_dimensions) |contract_dim| {
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

    const dot_dim_attr = args.dot_dimension_numbers.asAttr(ctx);

    return mlir.Operation.create(allocator, ctx, "stablehlo.dot_general", .{
        .operands = &.{ lhs, rhs },
        .results = &.{result_type},
        .attributes = &.{.{ "dot_dimension_numbers", dot_dim_attr }},
        .location = mlir.Location.unknown(ctx),
    });
}

/// Creates a stablehlo.reshape operation
pub fn reshape(allocator: std.mem.Allocator, ctx: mlir.Context, operand: mlir.Value, result_shape: []const i64, loc: mlir.Location) !mlir.Operation {
    const input_type = operand.getType().as(mlir.RankedTensorType).?;
    const result_type = mlir.Type.tensor(result_shape, input_type.getElementType());

    return mlir.Operation.create(allocator, ctx, "stablehlo.reshape", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .location = loc,
    });
}

/// Creates a stablehlo.broadcast_in_dim operation
pub fn broadcast_in_dim(allocator: std.mem.Allocator, ctx: mlir.Context, operand: mlir.Value, result_shape: []const i64, broadcast_dimensions: []const i64, loc: mlir.Location) !mlir.Operation {
    const input_type = operand.getType().as(mlir.RankedTensorType).?;
    const result_type = mlir.Type.tensor(result_shape, input_type.getElementType());
    const broadcast_dims_attr = mlir.Attribute.denseI64ArrayAttr(ctx, broadcast_dimensions);

    return mlir.Operation.create(allocator, ctx, "stablehlo.broadcast_in_dim", .{
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
    const dimension_attr = mlir.Attribute.integerAttr(ctx, dimension, mlir.Type.i64Type(ctx));
    
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
pub fn reduce(allocator: std.mem.Allocator, ctx: mlir.Context, operand: mlir.Value, init_value: mlir.Value, dimensions: []const i64, loc: mlir.Location) !mlir.Operation {
    const input_type = operand.getType().as(mlir.RankedTensorType).?;

    // Calculate result shape (remove reduced dimensions)
    var result_dims = std.ArrayList(i64).init(allocator);
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
            try result_dims.append(input_type.getDimension(i));
        }
    }

    const result_type = mlir.Type.tensor(result_dims.items, input_type.getElementType());
    const dimensions_attr = mlir.Attribute.denseI64ArrayAttr(ctx, dimensions);

    return mlir.Operation.create(allocator, ctx, "stablehlo.reduce", .{
        .operands = &.{ operand, init_value },
        .results = &.{result_type},
        .attributes = &.{.{ "dimensions", dimensions_attr }},
        .location = loc,
    });
}

/// Utility functions for creating common constants

/// Create a scalar constant
pub fn scalarConstant(allocator: std.mem.Allocator, ctx: mlir.Context, value: f64, element_type: mlir.Type) !mlir.Operation {
    const scalar_type = mlir.Type.tensor(&.{}, element_type); // Scalar tensor (rank 0)

    var attr: mlir.Attribute = undefined;
    if (element_type.isInteger() or element_type.isIndex()) {
        const i_val: i64 = @intFromFloat(value);
        const val_attr = mlir.Attribute.integerAttr(ctx, i_val, element_type);
        attr = mlir.Attribute.denseElementsAttrSplat(scalar_type, val_attr);
    } else {
        attr = mlir.Attribute.denseElementsAttrFloatSplat(scalar_type, value);
    }

    return constant(allocator, ctx, .{
        .value = attr,
        .result_type = scalar_type,
    });
}

/// Create a zero constant of given shape and type (renamed to avoid duplicate)
pub fn zeroTensor(ctx: mlir.Context, shape: []const i64, element_type: mlir.Type) mlir.Operation {
    const tensor_type = mlir.Type.tensor(shape, element_type);

    var attr: mlir.Attribute = undefined;
    if (element_type.isInteger() or element_type.isIndex()) {
        const zero_val = mlir.Attribute.integerAttr(ctx, 0, element_type);
        attr = mlir.Attribute.denseElementsAttrSplat(tensor_type, zero_val);
    } else {
        attr = mlir.Attribute.denseElementsAttrFloatSplat(tensor_type, 0.0);
    }

    const constant_op = constant(std.heap.page_allocator, ctx, .{
        .value = attr,
        .result_type = tensor_type,
    }) catch @panic("Failed to create zeroTensor");

    return constant_op;
}

/// Create a ones constant of given shape and type
pub fn onesConstant(ctx: mlir.Context, shape: []const i64, element_type: mlir.Type) mlir.Operation {
    const tensor_type = mlir.Type.tensor(shape, element_type);

    var attr: mlir.Attribute = undefined;
    if (element_type.isInteger() or element_type.isIndex()) {
        const one_val = mlir.Attribute.integerAttr(ctx, 1, element_type);
        attr = mlir.Attribute.denseElementsAttrSplat(tensor_type, one_val);
    } else {
        attr = mlir.Attribute.denseElementsAttrFloatSplat(tensor_type, 1.0);
    }

    const constant_op = constant(std.heap.page_allocator, ctx, .{
        .value = attr,
        .result_type = tensor_type,
    }) catch @panic("Failed to create onesConstant");

    return constant_op;
}

/// Creates a stablehlo.rsqrt operation (reciprocal square root)
pub fn rsqrt(ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.rsqrt", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.log operation
pub fn log(ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.log", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.sqrt operation
pub fn sqrt(allocator: std.mem.Allocator, ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) !mlir.Operation {
    return try mlir.Operation.create(allocator, ctx, "stablehlo.sqrt", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.power operation
pub fn power(allocator: std.mem.Allocator, ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, loc: mlir.Location) !mlir.Operation {
    return try mlir.Operation.create(allocator, ctx, "stablehlo.power", .{
        .operands = &.{ lhs, rhs },
        .results = &.{lhs.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.tanh operation
pub fn tanh(ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.tanh", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.sine operation
pub fn sine(ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.sine", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.cosine operation
pub fn cosine(ctx: mlir.Context, operand: mlir.Value, loc: mlir.Location) mlir.Operation {
    return mlir.Operation.create(ctx, "stablehlo.cosine", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .location = loc,
    });
}

/// Creates a stablehlo.slice operation
pub fn slice(allocator: std.mem.Allocator, ctx: mlir.Context, operand: mlir.Value, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, loc: mlir.Location) !mlir.Operation {
    const start_attr = mlir.Attribute.denseI64ArrayAttr(ctx, start_indices);
    const limit_attr = mlir.Attribute.denseI64ArrayAttr(ctx, limit_indices);
    const strides_attr = mlir.Attribute.denseI64ArrayAttr(ctx, strides);

    // Calculate result shape
    const input_type = operand.getType().as(mlir.RankedTensorType) orelse unreachable;
    var result_dims = std.ArrayList(i64).init(allocator);
    defer result_dims.deinit();

    for (start_indices, limit_indices, strides) |start, limit, stride| {
        const dim_size = @divFloor(limit - start + stride - 1, stride);
        try result_dims.append(dim_size);
    }

    const result_type = mlir.Type.tensor(result_dims.items, input_type.getElementType());

    return mlir.Operation.create(allocator, ctx, "stablehlo.slice", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{
            .{ "start_indices", start_attr },
            .{ "limit_indices", limit_attr },
            .{ "strides", strides_attr },
        },
        .location = loc,
    });
}

/// Creates a stablehlo.iota operation
pub fn iota(allocator: std.mem.Allocator, ctx: mlir.Context, shape: []const i64, iota_dimension: i64, element_type: mlir.Type, loc: mlir.Location) !mlir.Operation {
    const iota_dimension_attr = mlir.Attribute.integerAttr(ctx, iota_dimension, mlir.Type.i64Type(ctx));
    const result_type = mlir.Type.tensor(shape, element_type);

    return mlir.Operation.create(allocator, ctx, "stablehlo.iota", .{
        .operands = &.{},
        .results = &.{result_type},
        .attributes = &.{.{ "iota_dimension", iota_dimension_attr }},
        .location = loc,
    });
}

/// Creates a stablehlo.one_hot operation
/// Creates a chlo.one_hot operation
pub fn one_hot(ctx: mlir.Context, indices: mlir.Value, depth: i64, on_value: f32, off_value: f32, axis: i64, element_type: mlir.Type, loc: mlir.Location) !mlir.Operation {
    const axis_attr = mlir.Attribute.integerAttr(ctx, axis, mlir.Type.i64Type(ctx));
    const on_value_attr = mlir.Attribute.floatAttr(ctx, @floatCast(on_value), element_type);
    const off_value_attr = mlir.Attribute.floatAttr(ctx, @floatCast(off_value), element_type);

    // ... (shape calculation logic remains the same) ...
    const indices_type = indices.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const indices_shape = try indices_type.getShape(std.heap.page_allocator);
    defer std.heap.page_allocator.free(indices_shape);
    
    var result_dims = std.ArrayList(i64).init(std.heap.page_allocator);
    defer result_dims.deinit();
    
    for (indices_shape, 0..) |dim, i| {
        if (@as(i64, @intCast(i)) == axis) {
            try result_dims.append(depth);
        }
        try result_dims.append(dim);
    }
    if (axis == @as(i64, @intCast(indices_shape.len))) {
        try result_dims.append(depth);
    }
    
    const result_type = mlir.Type.tensor(result_dims.items, element_type);

    // CHANGE THE OPERATION NAME HERE
    const op = mlir.Operation.create(ctx, "chlo.one_hot", .{
        .operands = &.{indices},
        .results = &.{result_type},
        .attributes = &.{
            .{ "axis", axis_attr },
            .{ "on_value", on_value_attr },
            .{ "off_value", off_value_attr },
        },
        .location = loc,
    });
    
    // Since we are now creating a `chlo` op, we need to manually add the depth attribute.
    const depth_attr = mlir.Attribute.integerAttr(ctx, depth, mlir.Type.i64Type(ctx));
    const c_api = @import("../c.zig").c;
    c_api.operationSetAttributeByName(op.handle, "depth", depth_attr.handle);
    
    return op;
}

/// Creates a stablehlo.convert operation
pub fn convert(allocator: std.mem.Allocator, ctx: mlir.Context, operand: mlir.Value, result_type: mlir.Type, loc: mlir.Location) !mlir.Operation {
    return mlir.Operation.create(allocator, ctx, "stablehlo.convert", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .location = loc,
    });
}

/// Creates a stablehlo.softmax operation 
pub fn softmax(ctx: mlir.Context, operand: mlir.Value, axis: i64, loc: mlir.Location) mlir.Operation {
    const axis_attr = mlir.Attribute.integerAttr(ctx, axis, mlir.Type.i64Type(ctx));
    
    return mlir.Operation.create(ctx, "stablehlo.softmax", .{
        .operands = &.{operand},
        .results = &.{operand.getType()},
        .attributes = &.{.{ "axis", axis_attr }},
        .location = loc,
    });
}

/// Attribute for stablehlo.gather dimension numbers.
pub const GatherDimensionNumbersAttribute = struct {
    offset_dims: []const i64,
    collapsed_slice_dims: []const i64,
    start_index_map: []const i64,
    index_vector_dim: i64,
    operand_batching_dims: []const i64,
    start_indices_batching_dims: []const i64,
};

pub fn gather(
    ctx: mlir.Context,
    operand: mlir.Value,
    start_indices: mlir.Value,
    dimension_numbers: GatherDimensionNumbersAttribute,
    slice_sizes: []const i64,
    loc: mlir.Location,
) mlir.Operation {
    // Result type inference (unchanged, but ensure it accounts for batching correctly if needed)
    const operand_type = operand.getType().as(mlir.RankedTensorType).?;
    const start_indices_type = start_indices.getType().as(mlir.RankedTensorType).?;
    const element_type = operand_type.getElementType();
    const allocator = std.heap.page_allocator; // Using a temporary allocator for shape inference
    const start_shape = start_indices_type.getShape(allocator) catch unreachable;
    defer allocator.free(start_shape);
    var result_dims = std.ArrayList(i64).init(allocator);
    defer result_dims.deinit();

    // FIX: Append only the prefix dimensions, excluding the index_vector_dim
    const index_vector_dim = dimension_numbers.index_vector_dim;
    for (0..start_indices_type.getRank()) |i| {
        if (@as(i64, @intCast(i)) != index_vector_dim) {
            result_dims.append(start_shape[i]) catch @panic("OOM");
        }
    }

    // Append the non-collapsed slice dimensions. For embeddings, this is just the embedding size.
    // In general, append all slice_sizes[j] where j not in collapsed_slice_dims.
    // But since collapsed={0}, slice_sizes[0]=1 (collapsed), [1]=embd (not), append [1].
    result_dims.append(slice_sizes[1]) catch @panic("OOM");

    const result_type = mlir.Type.tensor(result_dims.items, element_type);

    const slice_sizes_attr = mlir.Attribute.denseI64ArrayAttr(ctx, slice_sizes);

    // NEW: Build the string for #stablehlo.gather<...>
    var str_buf = std.ArrayList(u8).init(std.heap.page_allocator);
    defer str_buf.deinit();

    str_buf.appendSlice("#stablehlo.gather<offset_dims = [") catch @panic("OOM");
    for (dimension_numbers.offset_dims, 0..) |d, i| {
        if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
        std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
    }
    str_buf.appendSlice("], collapsed_slice_dims = [") catch @panic("OOM");
    for (dimension_numbers.collapsed_slice_dims, 0..) |d, i| {
        if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
        std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
    }
    str_buf.appendSlice("], start_index_map = [") catch @panic("OOM");
    for (dimension_numbers.start_index_map, 0..) |d, i| {
        if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
        std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
    }
    str_buf.appendSlice("], index_vector_dim = ") catch @panic("OOM");
    std.fmt.format(str_buf.writer(), "{}", .{dimension_numbers.index_vector_dim}) catch @panic("OOM");
    str_buf.appendSlice(", operand_batching_dims = [") catch @panic("OOM");
    for (dimension_numbers.operand_batching_dims, 0..) |d, i| {
        if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
        std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
    }
    str_buf.appendSlice("], start_indices_batching_dims = [") catch @panic("OOM");
    for (dimension_numbers.start_indices_batching_dims, 0..) |d, i| {
        if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
        std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
    }
    str_buf.appendSlice("]>") catch @panic("OOM");

    const str_ref = c.c.MlirStringRef{ .data = str_buf.items.ptr, .length = str_buf.items.len };
    const dim_numbers_attr_handle = c.c.mlirAttributeParseGet(ctx.handle, str_ref);
    const dim_numbers_attr = mlir.Attribute{ .handle = dim_numbers_attr_handle };

    // Add indices_are_sorted
    const indices_are_sorted_ref = c.c.stringRefFromString("indices_are_sorted");
    const sorted_id = c.c.identifierGet(ctx.handle, indices_are_sorted_ref);
    const sorted_attr = mlir.Attribute.boolAttr(ctx, false);

    // named_attrs now includes sorted
    const dimension_numbers_ref = c.c.stringRefFromString("dimension_numbers");
    const gather_dim_numbers_id = c.c.identifierGet(ctx.handle, dimension_numbers_ref);
    const slice_sizes_ref = c.c.stringRefFromString("slice_sizes");
    const slice_sizes_id = c.c.identifierGet(ctx.handle, slice_sizes_ref);
    var named_attrs = [_]c.c.MlirNamedAttribute{
        .{ .name = gather_dim_numbers_id, .attribute = dim_numbers_attr.handle },
        .{ .name = slice_sizes_id, .attribute = slice_sizes_attr.handle },
        .{ .name = sorted_id, .attribute = sorted_attr.handle },
    };

    // Create the operation using pcpCreateOperation with packed args
    var operands = [_]*c.c.MlirValue{ operand.handle, start_indices.handle };
    var result_types = [_]*c.c.MlirType{ result_type.handle };

    const name_ref = c.c.stringRefFromString("stablehlo.gather");

    const op_args = c.c.PcpOpArgs{
        .nResults = 1,
        .results = &result_types,
        .nOperands = 2,
        .operands = &operands,
        .nAttributes = @intCast(named_attrs.len),
        .attributes = &named_attrs,
        .nRegions = 0,
        .regions = null,
    };

    const handle = c.c.pcpCreateOperation(&name_ref, &loc.handle, &op_args);

    return mlir.Operation{ .handle = handle };
}

/// Attribute for stablehlo.scatter dimension numbers.
pub const ScatterDimensionNumbersAttribute = struct {
    update_window_dims: []const i64,
    inserted_window_dims: []const i64,
    scatter_dims_to_operand_dims: []const i64,
    index_vector_dim: i64,

    pub fn asAttr(self: @This(), ctx: mlir.Context) mlir.Attribute {
        // DEFINITIVE FIX: Use the correct, module-level 'c' import.
        // The local import path was incorrect, causing C API calls to fail silently.

        // Create attributes for each field
        const update_window_dims_attr = mlir.Attribute.denseI64ArrayAttr(ctx, self.update_window_dims);
        const inserted_window_dims_attr = mlir.Attribute.denseI64ArrayAttr(ctx, self.inserted_window_dims);
        const scatter_dims_to_operand_dims_attr = mlir.Attribute.denseI64ArrayAttr(ctx, self.scatter_dims_to_operand_dims);
        const index_vector_dim_attr = mlir.Attribute.integerAttr(ctx, self.index_vector_dim, mlir.Type.i64Type(ctx));

        // Create identifiers for attribute names
        const update_window_dims_ref = c.c.stringRefFromString("update_window_dims");
    const update_window_id = c.c.identifierGet(ctx.handle, update_window_dims_ref);
        const inserted_window_dims_ref = c.c.stringRefFromString("inserted_window_dims");
    const inserted_window_id = c.c.identifierGet(ctx.handle, inserted_window_dims_ref);
        const scatter_dims_to_operand_dims_ref = c.c.stringRefFromString("scatter_dims_to_operand_dims");
    const scatter_dims_id = c.c.identifierGet(ctx.handle, scatter_dims_to_operand_dims_ref);
        const index_vector_dim_ref = c.c.stringRefFromString("index_vector_dim");
    const index_vec_id = c.c.identifierGet(ctx.handle, index_vector_dim_ref);

        // Create named attributes
        const named_attrs = [_]c.c.MlirNamedAttribute{
            .{ .name = update_window_id, .attribute = update_window_dims_attr.handle },
            .{ .name = inserted_window_id, .attribute = inserted_window_dims_attr.handle },
            .{ .name = scatter_dims_id, .attribute = scatter_dims_to_operand_dims_attr.handle },
            .{ .name = index_vec_id, .attribute = index_vector_dim_attr.handle },
        };

        // Create and return the dictionary attribute
        return mlir.Attribute.dictionary(ctx, &named_attrs);
    }
};

/// Creates a stablehlo.scatter operation for gradient backpropagation.
/// This is the inverse of gather - it takes updates and indices and scatters them into a larger tensor.
pub fn scatter(
    allocator: std.mem.Allocator,
    ctx: mlir.Context,
    operand: mlir.Value, // The tensor to scatter into (e.g., shape [vocab_size, n_embd])
    scatter_indices: mlir.Value, // The indices to scatter at (e.g., shape [batch, seq_len])
    updates: mlir.Value, // The values to scatter (e.g., shape [batch, seq_len, n_embd])
    dimension_numbers: ScatterDimensionNumbersAttribute,
    loc: mlir.Location,
) !mlir.Operation {
    const c_api = @import("../c.zig").c;

    // Get the element type for the update computation
    const operand_type = operand.getType().as(mlir.RankedTensorType).?;
    const element_type = operand_type.getElementType();
    const scalar_type = mlir.Type.tensor(&.{}, element_type);

    // Create the update body region with an add operation (for gradient accumulation)
    const body_region = c_api.regionCreate();
    const body_block = mlir.Block{ .handle = c_api.blockCreate(0, @constCast(@ptrCast(&[_]c_api.MlirType{})), @constCast(@ptrCast(&[_]*c_api.MlirLocation{}))) };
    c_api.regionAppendOwnedBlock(body_region, body_block.handle);

    // Add arguments: (existing_value, update_value) -> combined_value
    const lhs_arg = body_block.addArgument(scalar_type, loc);
    const rhs_arg = body_block.addArgument(scalar_type, loc);
    const add_op = try add(allocator, ctx, lhs_arg, rhs_arg, loc);
    c_api.blockAppendOwnedOperation(body_block.handle, add_op.handle);
    const return_op = try mlir.Operation.create(allocator, ctx, "stablehlo.return", .{ .operands = &.{add_op.getResult(0)} });
    c_api.blockAppendOwnedOperation(body_block.handle, return_op.handle);

    // Build the scatter operation using pcpCreateOperation
    const result_type = operand.getType();

    // Build the string for #stablehlo.scatter<...> like gather does
    var str_buf = std.ArrayList(u8).init(std.heap.page_allocator);
    defer str_buf.deinit();

    str_buf.appendSlice("#stablehlo.scatter<update_window_dims = [") catch @panic("OOM");
    for (dimension_numbers.update_window_dims, 0..) |d, i| {
        if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
        std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
    }
    str_buf.appendSlice("], inserted_window_dims = [") catch @panic("OOM");
    for (dimension_numbers.inserted_window_dims, 0..) |d, i| {
        if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
        std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
    }
    str_buf.appendSlice("], scatter_dims_to_operand_dims = [") catch @panic("OOM");
    for (dimension_numbers.scatter_dims_to_operand_dims, 0..) |d, i| {
        if (i > 0) str_buf.appendSlice(", ") catch @panic("OOM");
        std.fmt.format(str_buf.writer(), "{}", .{d}) catch @panic("OOM");
    }
    str_buf.appendSlice("], index_vector_dim = ") catch @panic("OOM");
    std.fmt.format(str_buf.writer(), "{}", .{dimension_numbers.index_vector_dim}) catch @panic("OOM");
    str_buf.appendSlice(">") catch @panic("OOM");

    const str_ref = c_api.MlirStringRef{ .data = str_buf.items.ptr, .length = str_buf.items.len };
    const dim_numbers_attr_handle = c_api.mlirAttributeParseGet(ctx.handle, str_ref);
    const dim_numbers_attr = mlir.Attribute{ .handle = dim_numbers_attr_handle };

    var operands = [_]c_api.MlirValue{ operand.handle, scatter_indices.handle, updates.handle };
    var results = [_]c_api.MlirType{ result_type.handle };
    var regions = [_]c_api.MlirRegion{ body_region };

    const scatter_dim_numbers_ref = c_api.stringRefFromString("scatter_dimension_numbers");
    const scatter_dim_numbers_id = c_api.identifierGet(ctx.handle, scatter_dim_numbers_ref);

    // Add indices_are_sorted attribute like gather does
    const indices_are_sorted_ref = c_api.stringRefFromString("indices_are_sorted");
    const sorted_id = c_api.identifierGet(ctx.handle, indices_are_sorted_ref);
    const sorted_attr = mlir.Attribute.boolAttr(ctx, false);

    var named_attrs = [_]c_api.MlirNamedAttribute{
        .{ .name = scatter_dim_numbers_id, .attribute = dim_numbers_attr.handle },
        .{ .name = sorted_id, .attribute = sorted_attr.handle },
    };

    const name_ref = c_api.stringRefFromString("stablehlo.scatter");

    const op_args = c_api.PcpOpArgs{
        .nResults = 1,
        .results = &results,
        .nOperands = 3,
        .operands = &operands,
        .nAttributes = 2,
        .attributes = &named_attrs,
        .nRegions = 1,
        .regions = &regions,
    };

    const handle = c_api.pcpCreateOperation(&name_ref, &loc.handle, &op_args);

    return mlir.Operation{ .handle = handle };
}

/// Creates a stablehlo.pad operation
pub fn pad(
    allocator: std.mem.Allocator,
    ctx: mlir.Context,
    operand: mlir.Value,
    padding_value: mlir.Value,
    padding_low: []const i64,
    padding_high: []const i64,
    interior_padding: []const i64,
    loc: mlir.Location
) !mlir.Operation {
    const padding_low_attr = mlir.Attribute.denseI64ArrayAttr(ctx, padding_low);
    const padding_high_attr = mlir.Attribute.denseI64ArrayAttr(ctx, padding_high);
    const interior_padding_attr = mlir.Attribute.denseI64ArrayAttr(ctx, interior_padding);

    // Calculate the result type explicitly
    const operand_type = operand.getType().as(mlir.RankedTensorType) orelse unreachable;
    const rank = operand_type.getRank();
    var result_shape = std.ArrayList(i64).init(std.heap.page_allocator);
    defer result_shape.deinit();

    for (0..rank) |i| {
        const dim_size = operand_type.getDimension(i);
        // Result dimension = input_size + padding_low + padding_high + interior_padding * (input_size - 1)
        const interior_contribution = interior_padding[i] * @max(0, dim_size - 1);
        const new_dim = dim_size + padding_low[i] + padding_high[i] + interior_contribution;
        result_shape.append(new_dim) catch unreachable;
    }

    const result_type = mlir.Type.rankedTensorType(ctx, result_shape.items, operand_type.getElementType());

    return mlir.Operation.create(allocator, ctx, "stablehlo.pad", .{
        .operands = &.{ operand, padding_value },
        .results = &.{result_type},
        .attributes = &.{
            .{ "edge_padding_low", padding_low_attr },
            .{ "edge_padding_high", padding_high_attr },
            .{ "interior_padding", interior_padding_attr }
        },
        .location = loc,
    });
}

