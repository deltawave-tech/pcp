const std = @import("std");
const mlir = @import("../mlir/wrapper.zig");
const ops = @import("../core/ops.zig");
const tensor = @import("../core/tensor.zig");
const c = @import("../mlir/c.zig").c;
const hlo = @import("../mlir/dialects/stablehlo.zig");

const Allocator = std.mem.Allocator;
const MLIRBuilder = ops.MLIRBuilder;

/// Vector-Jacobian Product (VJP) function signature
/// Takes the forward operation and output gradients, returns input gradients
pub const VJPFn = *const fn(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) anyerror![]mlir.Value;

/// Check if an mlir.Value comes from a constant operation at compile time
fn isValueFromConstantOp(value: mlir.Value) bool {
    if (!c.mlirValueIsAOpResult(value.handle)) return false;
    const owner = mlir.Operation{ .handle = c.mlirOpResultGetOwner(value.handle) };
    const name = owner.getName();
    return std.mem.eql(u8, name, "stablehlo.constant") or std.mem.eql(u8, name, "arith.constant");
}

/// VJP rule for addition: both inputs get the same gradient
pub fn addVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const a = primals[0];
    const b = primals[1];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const grad_a = try ops.reduceGradient(builder, grad_out, a);
    try result.append(grad_a);

    const grad_b = try ops.reduceGradient(builder, grad_out, b);
    try result.append(grad_b);

    return result.toOwnedSlice();
}

/// VJP rule for subtraction: da = grad_out, db = -grad_out
pub fn subtractVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const a = primals[0];
    const b = primals[1];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const grad_a = try ops.reduceGradient(builder, grad_out, a);
    try result.append(grad_a);

    const neg_grad = try builder.createAndAttach("stablehlo.negate", &.{grad_out}, &.{grad_out.getType()}, .{});
    const grad_b = try ops.reduceGradient(builder, neg_grad.getResult(0), b);
    try result.append(grad_b);

    return result.toOwnedSlice();
}

/// VJP rule for multiplication: da = grad_out * b, db = grad_out * a
pub fn multiplyVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const a = primals[0];
    const b = primals[1];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const grad_a_broadcast = try ops.broadcastOperands(builder, grad_out, b);
    defer builder.allocator.free(grad_a_broadcast.shape);
    const grad_a_type = grad_a_broadcast.lhs.getType();
    const grad_a_raw = try builder.createAndAttach("stablehlo.multiply", &.{ grad_a_broadcast.lhs, grad_a_broadcast.rhs }, &.{grad_a_type}, .{});

    const grad_a = try ops.reduceGradient(builder, grad_a_raw.getResult(0), a);
    try result.append(grad_a);

    const grad_b_broadcast = try ops.broadcastOperands(builder, grad_out, a);
    defer builder.allocator.free(grad_b_broadcast.shape);
    const grad_b_type = grad_b_broadcast.lhs.getType();
    const grad_b_raw = try builder.createAndAttach("stablehlo.multiply", &.{ grad_b_broadcast.lhs, grad_b_broadcast.rhs }, &.{grad_b_type}, .{});

    const grad_b = try ops.reduceGradient(builder, grad_b_raw.getResult(0), b);
    try result.append(grad_b);

    return result.toOwnedSlice();
}

/// VJP rule for division: da = grad_out / b, db = -grad_out * a / (b * b)
pub fn divideVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.print("DEBUG: Entering divideVJP\n", .{});

    if (adjoints.len < 1) {
        std.debug.print("FATAL: divideVJP adjoints empty\n", .{});
        return error.InvalidAdjoints;
    }
    if (primals.len < 2) {
        std.debug.print("FATAL: divideVJP primals < 2\n", .{});
        return error.InvalidPrimals;
    }

    const grad_out_raw = adjoints[0];
    const a = primals[0];
    const b = primals[1];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    errdefer result.deinit();

    std.debug.print("DEBUG: divideVJP: computing grad_a\n", .{});
    // da = grad_out / b with explicit broadcasting
    const grad_a_broadcast = try ops.broadcastOperands(builder, grad_out, b);
    defer builder.allocator.free(grad_a_broadcast.shape);
    const grad_a_type = grad_a_broadcast.lhs.getType();
    const grad_a_raw = try builder.createAndAttach("stablehlo.divide", &.{ grad_a_broadcast.lhs, grad_a_broadcast.rhs }, &.{grad_a_type}, .{});

    // Reduce to match a's shape if broadcasting occurred
    const grad_a = try ops.reduceGradient(builder, grad_a_raw.getResult(0), a);
    try result.append(grad_a);

    std.debug.print("DEBUG: divideVJP: computing grad_b\n", .{});
    // db = -grad_out * a / (b * b)

    // a * grad_out with explicit broadcasting
    const a_times_grad_broadcast = try ops.broadcastOperands(builder, a, grad_out);
    defer builder.allocator.free(a_times_grad_broadcast.shape);
    const a_times_grad_type = a_times_grad_broadcast.lhs.getType();
    const a_times_grad = try builder.createAndAttach("stablehlo.multiply", &.{ a_times_grad_broadcast.lhs, a_times_grad_broadcast.rhs }, &.{a_times_grad_type}, .{});

    // b * b (no broadcasting needed, both operands are b)
    const b_squared = try builder.createAndAttach("stablehlo.multiply", &.{ b, b }, &.{b.getType()}, .{});

    // (a * grad_out) / (b * b) with explicit broadcasting
    const div_operands_broadcast = try ops.broadcastOperands(builder, a_times_grad.getResult(0), b_squared.getResult(0));
    defer builder.allocator.free(div_operands_broadcast.shape);
    const div_operands_type = div_operands_broadcast.lhs.getType();
    const positive_grad_b_raw = try builder.createAndAttach("stablehlo.divide", &.{ div_operands_broadcast.lhs, div_operands_broadcast.rhs }, &.{div_operands_type}, .{});

    // Negate
    const positive_grad_b_neg = try builder.createAndAttach("stablehlo.negate", &.{ positive_grad_b_raw.getResult(0) }, &.{div_operands_type}, .{});

    // Reduce to match b's shape if broadcasting occurred
    const grad_b = try ops.reduceGradient(builder, positive_grad_b_neg.getResult(0), b);
    try result.append(grad_b);

    std.debug.print("DEBUG: divideVJP: converting to owned slice\n", .{});
    const slice = try result.toOwnedSlice();

    std.debug.print("DEBUG: divideVJP allocated slice len={} ptr={*}\n", .{slice.len, slice.ptr});
    return slice;
}

/// VJP rule for power: d(a^b)/da = grad_out * b * a^(b-1)
pub fn powerVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const base = primals[0];
    const exp = primals[1];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);
    const result_elem_type = result_ranked_type.getElementType();

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const y_op = try builder.createAndAttach("stablehlo.power", &.{ base, exp }, &.{result_type}, .{});
    const y = y_op.getResult(0);

    const y_div_base_broadcast = try ops.broadcastOperands(builder, y, base);
    defer builder.allocator.free(y_div_base_broadcast.shape);

    const zero_tensor = try ops.constant(builder, 0.0, y_div_base_broadcast.shape, result_elem_type);
    const zero = zero_tensor.value;

    const compare_type = result_elem_type.getStableHLOCompareType();
    const base_is_zero_op = hlo.compare(builder.ctx, y_div_base_broadcast.rhs, zero, .EQ, compare_type, builder.loc);
    builder.insertion_block.appendOwnedOperation(base_is_zero_op);
    const base_is_zero = base_is_zero_op.getResult(0);

    const y_div_base_type = y_div_base_broadcast.lhs.getType();
    const y_div_base_op = try builder.createAndAttach(
        "stablehlo.divide",
        &.{ y_div_base_broadcast.lhs, y_div_base_broadcast.rhs },
        &.{y_div_base_type},
        .{},
    );

    const safe_y_div_base_op = try builder.createAndAttach(
        "stablehlo.select",
        &.{ base_is_zero, zero, y_div_base_op.getResult(0) },
        &.{y_div_base_type},
        .{},
    );

    const exp_times_pow_broadcast = try ops.broadcastOperands(builder, exp, safe_y_div_base_op.getResult(0));
    defer builder.allocator.free(exp_times_pow_broadcast.shape);
    const exp_times_pow_type = exp_times_pow_broadcast.lhs.getType();
    const exp_times_pow = try builder.createAndAttach(
        "stablehlo.multiply",
        &.{ exp_times_pow_broadcast.lhs, exp_times_pow_broadcast.rhs },
        &.{exp_times_pow_type},
        .{},
    );

    const grad_base_broadcast = try ops.broadcastOperands(builder, grad_out, exp_times_pow.getResult(0));
    defer builder.allocator.free(grad_base_broadcast.shape);
    const grad_base_type = grad_base_broadcast.lhs.getType();
    const grad_base_raw = try builder.createAndAttach(
        "stablehlo.multiply",
        &.{ grad_base_broadcast.lhs, grad_base_broadcast.rhs },
        &.{grad_base_type},
        .{},
    );
    const grad_base = try ops.reduceGradient(builder, grad_base_raw.getResult(0), base);
    try result.append(grad_base);

    const exp_zero_tensor = try ops.constant(builder, 0.0, result_shape, result_elem_type);
    try result.append(exp_zero_tensor.value);

    return result.toOwnedSlice();
}

/// VJP rule for negation: da = -grad_out
pub fn negateVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = primals;
    const grad_out_raw = adjoints[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const grad_a = try builder.createAndAttach("stablehlo.negate", &.{grad_out}, &.{grad_out.getType()}, .{});
    try result.append(grad_a.getResult(0));

    return result.toOwnedSlice();
}

/// VJP rule for transpose: da = transpose(grad_out) with inverse permutation
pub fn transposeVJP(builder: *MLIRBuilder, original_op: mlir.Operation, primals: []const mlir.Value, adjoints: []const mlir.Value) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const a = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    const permutation_ref = c.stringRefFromString("permutation");
    const permutation_attr = c.operationGetAttributeByName(original_op.handle, permutation_ref);
    if (@intFromPtr(permutation_attr.ptr) == 0) return error.MissingPermutationAttribute;

    const input_type = a.getType().as(mlir.RankedTensorType).?;
    const rank = input_type.getRank();
    if (!c.attributeIsADenseI64Array(permutation_attr)) return error.InvalidPermutationAttributeType;

    var permutation = try builder.allocator.alloc(i64, rank);
    defer builder.allocator.free(permutation);
    for (0..rank) |i| permutation[i] = c.denseI64ArrayGetElement(permutation_attr, @intCast(i));

    var inv_permutation = try builder.allocator.alloc(i64, rank);
    defer builder.allocator.free(inv_permutation);
    for (permutation, 0..) |target_dim, source_dim| inv_permutation[@intCast(target_dim)] = @intCast(source_dim);

    const inv_perm_attr = mlir.Attribute.denseI64ArrayAttr(builder.ctx, inv_permutation);

    const transpose_op = try builder.createAndAttach("stablehlo.transpose",
        &.{grad_out},
        &.{a.getType()},
        .{ .attributes = &.{.{ "permutation", inv_perm_attr }} }
    );

    try result.append(transpose_op.getResult(0));
    return result.toOwnedSlice();
}

/// VJP rule for matrix multiplication
/// For Y = A @ B, the gradients are:
/// - dA = dY @ B^T
/// - dB = A^T @ dY
pub fn matmulVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const a = primals[0];
    const b = primals[1];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const dot_dim_attr = c.mlirOperationGetAttributeByName(original_op.handle, c.stringRefCreateFromCString("dot_dimension_numbers"));
    if (@intFromPtr(dot_dim_attr.ptr) == 0) {
        return error.MissingDotDimensionNumbersAttribute;
    }

    const a_type = a.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const b_type = b.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const grad_out_ranked_type = grad_out.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;

    const a_rank = a_type.getRank();
    const b_rank = b_type.getRank();
    const grad_rank = grad_out_ranked_type.getRank();

    var batch_dims = std.ArrayList(i64).init(builder.allocator);
    defer batch_dims.deinit();

    const num_batch_dims = @min(@min(a_rank, b_rank), grad_rank) - 2;
    for (0..num_batch_dims) |i| {
        try batch_dims.append(@intCast(i));
    }

    const grad_a_dot_dims = hlo.DotDimensionNumbersAttribute{
        .lhs_batching_dimensions = batch_dims.items,
        .rhs_batching_dimensions = batch_dims.items,
        .lhs_contracting_dimensions = &.{@intCast(grad_rank - 1)},
        .rhs_contracting_dimensions = &.{@intCast(b_rank - 1)},
    };
    const grad_a_op = try hlo.dot_general(builder.allocator, builder.ctx, grad_out, b, .{ .dot_dimension_numbers = grad_a_dot_dims });
    builder.insertion_block.appendOwnedOperation(grad_a_op);
    try result.append(grad_a_op.getResult(0));

    const grad_b_dot_dims = hlo.DotDimensionNumbersAttribute{
        .lhs_batching_dimensions = batch_dims.items,
        .rhs_batching_dimensions = batch_dims.items,
        .lhs_contracting_dimensions = &.{@intCast(a_rank - 2)},
        .rhs_contracting_dimensions = &.{@intCast(grad_rank - 2)},
    };
    const grad_b_op = try hlo.dot_general(builder.allocator, builder.ctx, a, grad_out, .{ .dot_dimension_numbers = grad_b_dot_dims });
    builder.insertion_block.appendOwnedOperation(grad_b_op);
    try result.append(grad_b_op.getResult(0));

    return result.toOwnedSlice();
}

/// VJP rule for ReLU (max(x, 0)): gradient flows through only where x > 0
pub fn reluVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const x = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const x_type = x.getType().as(mlir.RankedTensorType).?;
    const elem_type = x_type.getElementType();
    const x_shape = try x_type.getShape(builder.allocator);
    defer builder.allocator.free(x_shape);

    const zero_scalar = try builder.scalarConstant(0.0, elem_type);
    const zero = try ops.broadcastToShape(builder, zero_scalar.value, x_shape);

    const compare_type = elem_type.getStableHLOCompareType();
    const mask_op = hlo.compare(builder.ctx, x, zero, .GT, compare_type, builder.loc);
    builder.insertion_block.appendOwnedOperation(mask_op);
    const mask = mask_op.getResult(0);

    const zero_grad = try ops.broadcastToShape(builder, zero_scalar.value, x_shape);
    const grad_x = try builder.createAndAttach("stablehlo.select", &.{ mask, grad_out, zero_grad }, &.{grad_out.getType()}, .{});
    try result.append(grad_x.getResult(0));

    return result.toOwnedSlice();
}

/// VJP rule for constants: no gradient (constants don't have inputs)
pub fn constantVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    _ = primals;
    _ = adjoints;

    // CRITICAL FIX: Allocate an empty slice using the builder's allocator.
    // Returning &[_]mlir.Value{} returns a pointer to static/stack memory,
    // which causes allocator.free() to crash with "invalid pointer".
    const empty = try builder.allocator.alloc(mlir.Value, 0);
    return empty;
}

/// VJP rule for reshape: da = reshape(grad_out, original_shape)
pub fn reshapeVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const input = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const original_shape_type = input.getType();
    const grad_input = try builder.createAndAttach("stablehlo.reshape", &.{grad_out}, &.{original_shape_type}, .{});
    try result.append(grad_input.getResult(0));

    return result.toOwnedSlice();
}

pub fn reduceSumVJP(builder: *MLIRBuilder, original_op: mlir.Operation, primals: []const mlir.Value, adjoints: []const mlir.Value) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const input = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    if (!isValueFromConstantOp(input)) {
        const input_type = input.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
        const input_rank = input_type.getRank();
        const original_shape_type = input.getType();

        const dimensions_ref = c.stringRefFromString("dimensions");
        const dimensions_attr = c.operationGetAttributeByName(original_op.handle, dimensions_ref);

        if (@intFromPtr(dimensions_attr.ptr) != 0) {
            var broadcast_dims = std.ArrayList(i64).init(builder.allocator);
            defer broadcast_dims.deinit();

            const num_reduced_dims = c.mlirDenseArrayGetNumElements(dimensions_attr);
            var reduced_dims_list = std.ArrayList(i64).init(builder.allocator);
            defer reduced_dims_list.deinit();

            for (0..@intCast(num_reduced_dims)) |i| {
                const d = c.denseI64ArrayGetElement(dimensions_attr, @intCast(i));
                try reduced_dims_list.append(d);
            }

            for (0..input_rank) |i| {
                const dim_idx = @as(i64, @intCast(i));
                var is_reduced = false;
                for (reduced_dims_list.items) |r| {
                    if (r == dim_idx) {
                        is_reduced = true;
                        break;
                    }
                }

                if (!is_reduced) {
                    try broadcast_dims.append(dim_idx);
                }
            }

            const broadcast_dims_attr = mlir.Attribute.denseI64ArrayAttr(builder.ctx, broadcast_dims.items);
            const broadcast_op = try builder.createAndAttach("stablehlo.broadcast_in_dim",
                &.{grad_out},
                &.{original_shape_type},
                .{ .attributes = &.{.{ "broadcast_dimensions", broadcast_dims_attr }} }
            );
            try result.append(broadcast_op.getResult(0));

        } else {
            const grad_input = try builder.createAndAttach("stablehlo.broadcast", &.{grad_out}, &.{original_shape_type}, .{});
            try result.append(grad_input.getResult(0));
        }
    }
    return result.toOwnedSlice();
}

/// VJP rule for reduce_max: gradient flows only to the maximum elements
pub fn reduceMaxVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out = adjoints[0];
    const input = primals[0];

    // DEBUG: Print operation info
    std.debug.print("\n=== DEBUG reduceMaxVJP ===\n", .{});
    std.debug.print("Operation name: {s}\n", .{original_op.getName()});

    // 1. Extract Dimensions
    const dimensions_ref = c.stringRefFromString("dimensions");
    const dimensions_attr = c.operationGetAttributeByName(original_op.handle, dimensions_ref);
    std.debug.print("dimensions_attr ptr: {}\n", .{@intFromPtr(dimensions_attr.ptr)});
    if (@intFromPtr(dimensions_attr.ptr) == 0) return error.MissingDimensionsAttribute;

    // Check if it's a DenseI64Array
    const is_dense_i64_array = c.attributeIsADenseI64Array(dimensions_attr);
    std.debug.print("DEBUG reduceMaxVJP: Op {s}, is_dense_i64_array: {}\n", .{ original_op.getName(), is_dense_i64_array });

    const num_dims = c.mlirDenseArrayGetNumElements(dimensions_attr);
    std.debug.print("DEBUG reduceMaxVJP: Op {s}, Num Dims: {}\n", .{ original_op.getName(), num_dims });
    var reduce_dims = try builder.allocator.alloc(i64, @intCast(num_dims));
    defer builder.allocator.free(reduce_dims);

    for (0..@intCast(num_dims)) |i| {
        reduce_dims[i] = c.denseI64ArrayGetElement(dimensions_attr, @intCast(i));
    }
    std.debug.print("DEBUG reduceMaxVJP: Extracted reduce_dims: {any}\n", .{reduce_dims});

    // 2. Recompute Max (Raw Reduce)
    const input_tensor = try builder.newTensor(input);
    // Use raw hlo.reduce_max to get the reduced shape tensor
    const max_op = try hlo.reduce_max(builder.ctx, builder, input, reduce_dims, builder.loc);
    builder.insertion_block.appendOwnedOperation(max_op);
    const max_val_raw = max_op.getResult(0);

    // 3. Calculate "Kept Dims" Shape (e.g., [2048, 65] -> reduce dim 1 -> [2048, 1])
    const input_type = input.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const input_shape = try input_type.getShape(builder.allocator);
    defer builder.allocator.free(input_shape);
    const input_rank = input_shape.len;

    var kept_dims_shape = try builder.allocator.alloc(i64, input_rank);
    defer builder.allocator.free(kept_dims_shape);

    // Create simple 0..N identity mapping for broadcast dimensions
    var broadcast_dims = try builder.allocator.alloc(i64, input_rank);
    defer builder.allocator.free(broadcast_dims);

    for (0..input_rank) |i| {
        var is_reduced = false;
        for (reduce_dims) |d| {
            if (d == @as(i64, @intCast(i))) {
                is_reduced = true;
                break;
            }
        }
        if (is_reduced) {
            kept_dims_shape[i] = 1; // Collapsed dim becomes 1
        } else {
            kept_dims_shape[i] = input_shape[i]; // Maintained dim
        }
        broadcast_dims[i] = @intCast(i);
    }

    // 4. Reshape Max and Grad to [Dimensions... 1 ... Dimensions]
    // This explicitly aligns ranks so broadcasting works correctly
    const max_tensor_wrapper = try builder.newTensor(max_val_raw);
    const max_reshaped = try ops.reshape(builder, max_tensor_wrapper, kept_dims_shape);

    const grad_tensor_wrapper = try builder.newTensor(grad_out);
    const grad_reshaped = try ops.reshape(builder, grad_tensor_wrapper, kept_dims_shape);

    // 5. Broadcast back to full Input Shape
    // Now we broadcast from e.g. [2048, 1] -> [2048, 65] safely
    const max_broadcast = try ops.broadcastInDim(builder, max_reshaped, input_shape, broadcast_dims);
    const grad_broadcast = try ops.broadcastInDim(builder, grad_reshaped, input_shape, broadcast_dims);

    // 6. Create Mask (input == max) and Select Gradients
    const mask = try ops.compare(builder, input_tensor, max_broadcast, .EQ);

    const zero = try ops.constant(builder, 0.0, input_shape, input_type.getElementType());
    const grad_input = try ops.select(builder, mask, grad_broadcast, zero);

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    try result.append(grad_input.value);

    // Handle init_value operand if present (needs zero gradient)
    if (original_op.getNumOperands() > 1) {
        const init_val = original_op.getOperand(1);
        const init_type = init_val.getType().as(mlir.RankedTensorType) orelse return error.InvalidInitType;
        const init_shape = try init_type.getShape(builder.allocator);
        defer builder.allocator.free(init_shape);
        const zero_init = try ops.constant(builder, 0.0, init_shape, init_type.getElementType());
        try result.append(zero_init.value);
    }

    return result.toOwnedSlice();
}

/// Helper for gatherVJP: Attempts to handle vector index gather via linearization
/// Returns the gradient value if successful, null otherwise
pub fn tryVectorIndexLinearization(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    indices_i64: mlir.Value,
    grad_out: mlir.Value,
    operand_type: mlir.RankedTensorType,
    indices_type: mlir.RankedTensorType,
    grad_out_type: mlir.RankedTensorType,
) !?mlir.Value {
    const dim_numbers_attr = c.mlirOperationGetAttributeByName(original_op.handle, c.stringRefFromString("dimension_numbers"));
    const slice_sizes_attr = c.mlirOperationGetAttributeByName(original_op.handle, c.stringRefFromString("slice_sizes"));

    if (@intFromPtr(dim_numbers_attr.ptr) == 0 or @intFromPtr(slice_sizes_attr.ptr) == 0) {
        return null;
    }

    const operand_rank = operand_type.getRank();
    var slice_volume: i64 = 1;
    var total_elements: i64 = 1;
    for (0..operand_rank) |i| {
        slice_volume *= c.denseI64ArrayGetElement(slice_sizes_attr, @intCast(i));
        total_elements *= operand_type.getDimension(i);
    }

    const index_vector_dim = c.stablehloGatherDimensionNumbersGetIndexVectorDim(dim_numbers_attr);
    const indices_rank = indices_type.getRank();

    // Check for vector index gather that produces scalars (collapses all dims)
    if (slice_volume != 1 or index_vector_dim >= @as(i64, @intCast(indices_rank)) or indices_type.getDimension(@intCast(index_vector_dim)) <= 1) {
        return null;
    }

    std.debug.print("DEBUG gatherVJP: Detected Scalar Gather with Vector Indices. Using Linearization.\n", .{});

    // 1. Linearize Indices
    const map_size = c.stablehloGatherDimensionNumbersGetStartIndexMapSize(dim_numbers_attr);

    // Calculate strides
    var strides = try builder.allocator.alloc(i64, operand_rank);
    defer builder.allocator.free(strides);
    var current_stride: i64 = 1;
    var i: usize = operand_rank;
    while (i > 0) {
        i -= 1;
        strides[i] = current_stride;
        current_stride *= operand_type.getDimension(i);
    }

    // Shape of indices after removing vector dim
    var linear_index_shape = try builder.allocator.alloc(i64, indices_rank - 1);
    defer builder.allocator.free(linear_index_shape);
    var dim_idx: usize = 0;
    for (0..indices_rank) |d| {
        if (d != @as(usize, @intCast(index_vector_dim))) {
            linear_index_shape[dim_idx] = indices_type.getDimension(d);
            dim_idx += 1;
        }
    }

    // Accumulate linear index
    var linear_indices: ?mlir.Value = null;

    for (0..@as(usize, @intCast(map_size))) |k| {
        // Slice indices at index_vector_dim = k
        var start = try builder.allocator.alloc(i64, indices_rank);
        defer builder.allocator.free(start);
        var limit = try builder.allocator.alloc(i64, indices_rank);
        defer builder.allocator.free(limit);
        var slice_strides = try builder.allocator.alloc(i64, indices_rank);
        defer builder.allocator.free(slice_strides);

        for (0..indices_rank) |d| {
            start[d] = 0;
            limit[d] = indices_type.getDimension(d);
            slice_strides[d] = 1;
        }
        start[@intCast(index_vector_dim)] = @intCast(k);
        limit[@intCast(index_vector_dim)] = @intCast(k + 1);

        const slice_op = try hlo.slice(builder.allocator, builder.ctx, indices_i64, start, limit, slice_strides, builder.loc);
        builder.insertion_block.appendOwnedOperation(slice_op);

        const reshape_op = try hlo.reshape(builder.allocator, builder.ctx, slice_op.getResult(0), linear_index_shape, builder.loc);
        builder.insertion_block.appendOwnedOperation(reshape_op);
        const component = reshape_op.getResult(0);

        const operand_dim_idx = c.stablehloGatherDimensionNumbersGetStartIndexMapElem(dim_numbers_attr, @intCast(k));
        const dim_stride = strides[@intCast(operand_dim_idx)];

        // Create constant stride tensor
        const stride_tensor = try ops.constant(builder, @floatFromInt(dim_stride), linear_index_shape, mlir.Type.i64Type(builder.ctx));

        const scaled_comp_op = try hlo.multiply(builder.allocator, builder.ctx, component, stride_tensor.value, builder.loc);
        builder.insertion_block.appendOwnedOperation(scaled_comp_op);
        const scaled_comp = scaled_comp_op.getResult(0);

        if (linear_indices) |acc| {
            const add_op = try hlo.add(builder.allocator, builder.ctx, acc, scaled_comp, builder.loc);
            builder.insertion_block.appendOwnedOperation(add_op);
            linear_indices = add_op.getResult(0);
        } else {
            linear_indices = scaled_comp;
        }
    }

    // 2. OneHot & MatMul
    // Flatten linear indices to [N]
    var n_elements: i64 = 1;
    for (linear_index_shape) |d| n_elements *= d;

    const flat_indices_shape = [_]i64{n_elements};
    const flat_reshape_op = try hlo.reshape(builder.allocator, builder.ctx, linear_indices.?, &flat_indices_shape, builder.loc);
    builder.insertion_block.appendOwnedOperation(flat_reshape_op);

    const one_hot_flat = try ops.oneHot(
        builder,
        try builder.newTensor(flat_reshape_op.getResult(0)),
        total_elements,
        1.0, 0.0, -1,
        grad_out_type.getElementType()
    );

    // Flatten GradOut to [N, 1]
    const grad_flat_shape = [_]i64{n_elements, 1};
    const grad_flat_op = try hlo.reshape(builder.allocator, builder.ctx, grad_out, &grad_flat_shape, builder.loc);
    builder.insertion_block.appendOwnedOperation(grad_flat_op);

    // MatMul: [N, Total]^T @ [N, 1] -> [Total, 1] (Contract dim 0 of LHS with dim 0 of RHS)
    // OneHot is [N, Total]. Contract N.
    const dot_dims = hlo.DotDimensionNumbersAttribute{
        .lhs_batching_dimensions = &.{},
        .rhs_batching_dimensions = &.{},
        .lhs_contracting_dimensions = &.{0},
        .rhs_contracting_dimensions = &.{0},
    };

    const grad_table_op = try hlo.dot_general(
        builder.allocator,
        builder.ctx,
        one_hot_flat.value,
        grad_flat_op.getResult(0),
        .{ .dot_dimension_numbers = dot_dims }
    );
    builder.insertion_block.appendOwnedOperation(grad_table_op);

    // Reshape to Operand Shape
    const operand_shape = try operand_type.getShape(builder.allocator);
    defer builder.allocator.free(operand_shape);
    const final_op = try hlo.reshape(builder.allocator, builder.ctx, grad_table_op.getResult(0), operand_shape, builder.loc);
    builder.insertion_block.appendOwnedOperation(final_op);

    return final_op.getResult(0);
}

/// VJP rule for sine: d(sin(x)) = cos(x) * dx
pub fn sinVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const x = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const x_tensor = try builder.newTensor(x);
    const cos_x = try ops.cos(builder, x_tensor);

    const grad_tensor = try builder.newTensor(grad_out);
    const grad_x_raw = try ops.multiply(builder, grad_tensor, cos_x);

    const grad_x = try ops.reduceGradient(builder, grad_x_raw.value, x);
    try result.append(grad_x);

    return result.toOwnedSlice();
}

/// VJP rule for cosine: d(cos(x)) = -sin(x) * dx
pub fn cosVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const x = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const x_tensor = try builder.newTensor(x);
    const sin_x = try ops.sin(builder, x_tensor);

    const neg_sin_x = try ops.negate(builder, sin_x);

    const grad_tensor = try builder.newTensor(grad_out);
    const grad_x_raw = try ops.multiply(builder, grad_tensor, neg_sin_x);

    const grad_x = try ops.reduceGradient(builder, grad_x_raw.value, x);
    try result.append(grad_x);

    return result.toOwnedSlice();
}

/// VJP rule for logistic (sigmoid): d(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x)) * dx
pub fn logisticVJP(
    builder: *MLIRBuilder,
    _: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const x = primals[0];

    // Recompute sigmoid(x) in the gradient function instead of reusing forward result
    const sigmoid_x_op = try builder.createAndAttach("stablehlo.logistic", &.{x}, &.{x.getType()}, .{});
    const primal_out = sigmoid_x_op.getResult(0);

    const result_type = primal_out.getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const out_tensor = try builder.newTensor(primal_out);
    const grad_tensor = try builder.newTensor(grad_out);

    // 1 - sigmoid(x)
    const element_type = result_ranked_type.getElementType();
    const one = try ops.constant(builder, 1.0, result_shape, element_type);
    const one_minus_sig = try ops.subtract(builder, one, out_tensor);

    // sigmoid(x) * (1 - sigmoid(x))
    const sigmoid_grad = try ops.multiply(builder, out_tensor, one_minus_sig);

    // grad_out * sigmoid_grad
    const grad_x_raw = try ops.multiply(builder, grad_tensor, sigmoid_grad);

    const grad_x = try ops.reduceGradient(builder, grad_x_raw.value, x);
    try result.append(grad_x);

    return result.toOwnedSlice();
}

/// VJP rule for concatenate: gradient flows back via slicing
pub fn concatenateVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out = adjoints[0];

    const dimension_attr_name = c.stringRefFromString("dimension");
    const dimension_attr = c.operationGetAttributeByName(original_op.handle, dimension_attr_name);
    const dim = c.integerAttrGetValueInt(dimension_attr);

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    var offset: i64 = 0;

    for (primals) |primal| {
        const p_type = primal.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
        const p_shape = try p_type.getShape(builder.allocator);
        defer builder.allocator.free(p_shape);
        const size = p_shape[@intCast(dim)];

        const start_indices = try builder.allocator.alloc(i64, p_shape.len);
        defer builder.allocator.free(start_indices);
        @memset(start_indices, 0);
        start_indices[@intCast(dim)] = offset;

        const limit_indices = try builder.allocator.dupe(i64, p_shape);
        defer builder.allocator.free(limit_indices);
        limit_indices[@intCast(dim)] = offset + size;

        const strides = try builder.allocator.alloc(i64, p_shape.len);
        defer builder.allocator.free(strides);
        @memset(strides, 1);

        const grad_slice = try ops.slice(builder, try builder.newTensor(grad_out), start_indices, limit_indices, strides);
        try result.append(grad_slice.value);

        offset += size;
    }

    return result.toOwnedSlice();
}

/// VJP rule for gather: Replaces scatter with One-Hot + MatMul to avoid GPU atomics
pub fn gatherVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const operand = primals[0];      // Embedding table [Vocab, Embed]
    const start_indices = primals[1]; // Indices [Batch, Seq]

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    if (isValueFromConstantOp(operand)) {
        return result.toOwnedSlice();
    }

    // Get dimensions
    const operand_type = operand.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const indices_type = start_indices.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;

    // Convert indices to i64 if needed
    const indices_elem_type = indices_type.getElementType();
    const indices_i64 = if (indices_elem_type.isInteger()) start_indices else blk: {
        const indices_shape = try indices_type.getShape(builder.allocator);
        defer builder.allocator.free(indices_shape);
        const indices_i64_type = mlir.Type.rankedTensorType(
            builder.ctx,
            indices_shape,
            mlir.Type.i64Type(builder.ctx)
        );
        const convert_op = try builder.createAndAttach("stablehlo.convert", &.{start_indices}, &.{indices_i64_type}, .{});
        break :blk convert_op.getResult(0);
    };

    // Try vector index linearization first
    if (try tryVectorIndexLinearization(builder, original_op, indices_i64, grad_out, operand_type, indices_type, grad_out_type)) |linear_grad| {
        try result.append(linear_grad);
        return result.toOwnedSlice();
    }

    // Fall back to OneHot + MatMul approach
    const vocab_size = operand_type.getDimension(0);
    std.debug.print("DEBUG gatherVJP: vocab_size={}, operand_rank={}, indices_rank={}, grad_out_rank={}\n", .{
        vocab_size, operand_type.getRank(), indices_type.getRank(), grad_out_type.getRank()
    });

    // Create one-hot: [Batch, Seq] -> [Batch, Seq, Vocab]
    std.debug.print("DEBUG gatherVJP: About to create one-hot tensor...\n", .{});
    const one_hot = try ops.oneHot(
        builder,
        try builder.newTensor(indices_i64),
        vocab_size,
        1.0, 0.0, -1,
        grad_out_type.getElementType()
    );
    std.debug.print("DEBUG gatherVJP: One-hot created, rank={}\n", .{one_hot.shape.rank()});

    // Get one-hot shape - this determines N (number of index lookups)
    const one_hot_type = one_hot.value.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const one_hot_shape = try one_hot_type.getShape(builder.allocator);
    defer builder.allocator.free(one_hot_shape);

    // Flatten one-hot to [N, Vocab] where N is product of all dims except last (vocab dim)
    var n_dims: i64 = 1;
    for (0..(one_hot_shape.len - 1)) |i| {
        n_dims *= one_hot_shape[i];
    }
    const vocab_dim = one_hot_shape[one_hot_shape.len - 1];

    std.debug.print("DEBUG gatherVJP: One-hot shape: {any}, n_dims={}\n", .{one_hot_shape, n_dims});

    // Get embed_dim from grad_out shape (already extracted at top)
    const embed_dim = grad_out_shape[grad_out_shape.len - 1];

    // Flatten one_hot to [N, Vocab]
    const one_hot_flat_shape = [_]i64{ n_dims, vocab_dim };
    const one_hot_flat_type = mlir.Type.rankedTensorType(
        builder.ctx,
        &one_hot_flat_shape,
        one_hot_type.getElementType()
    );
    const one_hot_flat_op = try builder.createAndAttach("stablehlo.reshape", &.{one_hot.value}, &.{one_hot_flat_type}, .{});
    const one_hot_flat = one_hot_flat_op.getResult(0);

    // Calculate total elements in grad_out
    var grad_out_elements: i64 = 1;
    for (grad_out_shape) |dim| {
        grad_out_elements *= dim;
    }
    const expected_elements = n_dims * embed_dim;

    std.debug.print("DEBUG gatherVJP: grad_out has {} elements, need {} for [N={}, Embed={}]\n", .{grad_out_elements, expected_elements, n_dims, embed_dim});

    // If grad_out has fewer elements than expected, broadcast/reshape appropriately
    const grad_out_flat: mlir.Value = if (grad_out_elements == expected_elements) blk: {
        // Can directly reshape
        const grad_out_flat_shape = [_]i64{ n_dims, embed_dim };
        const grad_out_flat_type = mlir.Type.rankedTensorType(
            builder.ctx,
            &grad_out_flat_shape,
            grad_out_type.getElementType()
        );
        const grad_out_flat_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out}, &.{grad_out_flat_type}, .{});
        break :blk grad_out_flat_op.getResult(0);
    } else blk2: {
        // Need to broadcast grad_out to match n_dims
        // First reshape grad_out to flatten all but last dim
        var grad_n: i64 = 1;
        for (0..(grad_out_shape.len - 1)) |i| {
            grad_n *= grad_out_shape[i];
        }

        std.debug.print("DEBUG gatherVJP: Broadcasting grad_out from [{}, {}] to [{}, {}]\n", .{grad_n, embed_dim, n_dims, embed_dim});

        // Broadcast to match n_dims
        const broadcast_shape = [_]i64{ n_dims, embed_dim };

        // Calculate broadcast dimensions
        const broadcast_dims = [_]i64{0, 1}; // Broadcast along first two dimensions
        const broadcast_op = try hlo.broadcast_in_dim(
            builder.allocator,
            builder.ctx,
            grad_out,
            &broadcast_shape,
            &broadcast_dims,
            builder.loc
        );
        builder.insertion_block.appendOwnedOperation(broadcast_op);
        break :blk2 broadcast_op.getResult(0);
    };

    std.debug.print("DEBUG gatherVJP: Flattened shapes - one_hot: [{}, {}], grad_out: [{}, {}]\n", .{n_dims, vocab_dim, n_dims, embed_dim});

    // Now do dot_general: [N, Vocab]^T @ [N, Embed] = [Vocab, Embed]
    const dot_dims = hlo.DotDimensionNumbersAttribute{
        .lhs_batching_dimensions = &.{},
        .rhs_batching_dimensions = &.{},
        .lhs_contracting_dimensions = &.{0}, // Contract on N dimension
        .rhs_contracting_dimensions = &.{0}, // Contract on N dimension
    };

    // Grad_table = OneHot^T @ Grad_out
    const grad_table_op = try hlo.dot_general(
        builder.allocator,
        builder.ctx,
        one_hot_flat,
        grad_out_flat,
        .{ .dot_dimension_numbers = dot_dims }
    );

    builder.insertion_block.appendOwnedOperation(grad_table_op);
    try result.append(grad_table_op.getResult(0));

    return result.toOwnedSlice();
}

/// VJP rule for function return: just pass through the gradient
pub fn returnVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    _ = primals;

    // dupe allocates using allocator, so it returns safe pointer (even if empty? alloc(0) is safe)
    // But to be ultra-safe with our defensive freeing:
    if (adjoints.len == 0) {
        return builder.allocator.alloc(mlir.Value, 0);
    }
    return builder.allocator.dupe(mlir.Value, adjoints);
}

/// VJP rule for slice: scatter gradient back to original tensor positions
/// For slice(input, start_indices, limit_indices, strides):
/// - grad_input = scatter zeros tensor with grad_out at sliced positions
pub fn sliceVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const input = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    if (!isValueFromConstantOp(input)) {
        const input_type = input.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
        const input_rank = input_type.getRank();

        // Parse attributes from the original slice operation
        const start_indices_attr = c.mlirOperationGetAttributeByName(original_op.handle, c.stringRefCreateFromCString("start_indices"));
        const limit_indices_attr = c.mlirOperationGetAttributeByName(original_op.handle, c.stringRefCreateFromCString("limit_indices"));

        if (@intFromPtr(start_indices_attr.ptr) == 0 or @intFromPtr(limit_indices_attr.ptr) == 0) {
            return error.MissingSliceAttribute;
        }

        // Extract start indices for padding_low
        var start_indices = try builder.allocator.alloc(i64, input_rank);
        defer builder.allocator.free(start_indices);
        for (0..input_rank) |i| {
            start_indices[i] = c.denseI64ArrayGetElement(start_indices_attr, @intCast(i));
        }

        // Calculate padding_high = input_shape - limit_indices
        var padding_high = try builder.allocator.alloc(i64, input_rank);
        defer builder.allocator.free(padding_high);
        for (0..input_rank) |i| {
            const limit_index = c.denseI64ArrayGetElement(limit_indices_attr, @intCast(i));
            padding_high[i] = input_type.getDimension(i) - limit_index;
        }

        // Interior padding is all zeros for the inverse of slice
        const interior_padding = try builder.allocator.alloc(i64, input_rank);
        defer builder.allocator.free(interior_padding);
        @memset(interior_padding, 0);

        // Create a scalar zero for the padding value
        const zero_scalar_op = try hlo.scalarConstant(builder.allocator, builder.ctx, 0.0, input_type.getElementType());
        builder.insertion_block.appendOwnedOperation(zero_scalar_op);
        const zero_scalar = zero_scalar_op.getResult(0);

        // Create the hlo.pad operation to expand grad_out back to input shape
        const pad_op = try hlo.pad(builder.allocator, builder.ctx, grad_out, zero_scalar, start_indices, padding_high, interior_padding, builder.loc);
        builder.insertion_block.appendOwnedOperation(pad_op);

        try result.append(pad_op.getResult(0));

        std.debug.print("✓ sliceVJP: Created gradient flow using pad operation\n", .{});
    }

    return result.toOwnedSlice();
}

/// VJP rule for broadcast_in_dim: The inverse of broadcasting is to sum the gradients
/// back into the original, smaller shape.
pub fn broadcastInDimVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const input = primals[0]; // The original, smaller tensor

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // Only compute gradients for non-constant operands
    if (!isValueFromConstantOp(input)) {
        // The gradient of the input is the sum of the output gradients, reduced
        // along the dimensions that were added by the broadcast.

        const input_type = input.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;

        const input_shape = try input_type.getShape(builder.allocator);
        defer builder.allocator.free(input_shape);

        const input_rank = input_shape.len;
        const grad_out_rank = grad_out_shape.len;

        // Find the dimensions that need to be reduced. These are all dimensions in the
        // output shape that are not part of the input shape's broadcast mapping.
        var dims_to_reduce = std.ArrayList(i64).init(builder.allocator);
        defer dims_to_reduce.deinit();

        // Handle case where grad_out_rank >= input_rank (normal broadcast case)
        if (grad_out_rank >= input_rank) {
            // Simple logic: reduce the leading dimensions that were added by broadcast
            // This works for cases where broadcast adds dimensions at the front
            for (0..(grad_out_rank - input_rank)) |i| {
                try dims_to_reduce.append(@intCast(i));
            }

            // Also reduce dimensions where the input dimension is 1 but output dimension is larger
            for (0..input_rank) |i| {
                const input_dim_idx = (grad_out_rank - input_rank) + i;
                if (input_shape[i] == 1 and grad_out_shape[input_dim_idx] > 1) {
                    try dims_to_reduce.append(@intCast(input_dim_idx));
                }
            }
        }
        // If grad_out_rank < input_rank, no dimensions to reduce (gradient passes through)

        // Create a tensor wrapper for grad_out
        const grad_out_tensor = try builder.newTensor(grad_out);

        // Create a reduce_sum op to get the gradient of the input
        // Use keepdims=true to preserve the shape, then reshape if needed
        var grad_input_tensor: tensor.Tensor(void) = undefined;
        if (dims_to_reduce.items.len > 0) {
            // Use keepdims=true to preserve dimension count
            grad_input_tensor = try ops.reduceSum(builder, grad_out_tensor, dims_to_reduce.items, true);

            // After reduction with keepdims=true, we have the right rank but may need to verify shape
            // The reduced dimensions will be size 1, which matches the original input shape
            // However, if shapes still don't match exactly, reshape to the original input type
            const grad_result_type = grad_input_tensor.value.getType();
            const input_as_type = mlir.Type{ .handle = input_type.handle };
            if (!grad_result_type.isEqual(input_as_type)) {
                const reshape_op = try builder.createAndAttach(
                    "stablehlo.reshape",
                    &.{grad_input_tensor.value},
                    &.{input_as_type},
                    .{},
                );
                grad_input_tensor = try builder.newTensor(reshape_op.getResult(0));
            }
        } else {
            // If no dimensions to reduce, the gradient passes through unchanged
            grad_input_tensor = grad_out_tensor;
        }

        try result.append(grad_input_tensor.value);

        std.debug.print("✓ broadcastInDimVJP: Created gradient flow for broadcast_in_dim operation\n", .{});
    }

    return result.toOwnedSlice();
}

pub fn expVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const a = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const exp_a = try builder.createAndAttach("stablehlo.exponential", &.{a}, &.{a.getType()}, .{});

    const grad_in_broadcast = try ops.broadcastOperands(builder, grad_out, exp_a.getResult(0));
    defer builder.allocator.free(grad_in_broadcast.shape);
    const grad_in_type = grad_in_broadcast.lhs.getType();
    const grad_in_raw = try builder.createAndAttach("stablehlo.multiply", &.{grad_in_broadcast.lhs, grad_in_broadcast.rhs}, &.{grad_in_type}, .{});

    const grad_in = try ops.reduceGradient(builder, grad_in_raw.getResult(0), a);
    try result.append(grad_in);
    return result.toOwnedSlice();
}

/// VJP rule for log: da = grad_out / a
pub fn logVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const a = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const grad_in_broadcast = try ops.broadcastOperands(builder, grad_out, a);
    defer builder.allocator.free(grad_in_broadcast.shape);
    const grad_in_type = grad_in_broadcast.lhs.getType();
    const grad_in_raw = try builder.createAndAttach("stablehlo.divide", &.{grad_in_broadcast.lhs, grad_in_broadcast.rhs}, &.{grad_in_type}, .{});

    const grad_in = try ops.reduceGradient(builder, grad_in_raw.getResult(0), a);
    try result.append(grad_in);
    return result.toOwnedSlice();
}

/// VJP for rsqrt: da = -0.5 * grad_out * (a ^ -1.5)
pub fn rsqrtVJP(builder: *MLIRBuilder, original_op: mlir.Operation, primals: []const mlir.Value, adjoints: []const mlir.Value) ![]mlir.Value {
    _ = original_op;
    const grad_out_raw = adjoints[0];
    const a = primals[0];
    const tensor_type = a.getType();
    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    const rsqrt_op = try builder.createAndAttach("stablehlo.rsqrt", &.{a}, &.{tensor_type}, .{});
    const rsqrt_val = rsqrt_op.getResult(0);
    const sq = try builder.createAndAttach("stablehlo.multiply", &.{rsqrt_val, rsqrt_val}, &.{tensor_type}, .{});
    const cub = try builder.createAndAttach("stablehlo.multiply", &.{sq.getResult(0), rsqrt_val}, &.{tensor_type}, .{});

    const ranked_type = tensor_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const a_shape = try ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(a_shape);
    const elem_type = ranked_type.getElementType();

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, a_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{tensor_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    const neg_half_tensor = try ops.constant(builder, -0.5, a_shape, elem_type);
    const neg_half = neg_half_tensor.value;

    const term1_broadcast = try ops.broadcastOperands(builder, grad_out, neg_half);
    defer builder.allocator.free(term1_broadcast.shape);
    const term1_type = term1_broadcast.lhs.getType();
    const term1 = try builder.createAndAttach("stablehlo.multiply", &.{term1_broadcast.lhs, term1_broadcast.rhs}, &.{term1_type}, .{});

    const final_grad_broadcast = try ops.broadcastOperands(builder, term1.getResult(0), cub.getResult(0));
    defer builder.allocator.free(final_grad_broadcast.shape);
    const final_grad_type = final_grad_broadcast.lhs.getType();
    const final_grad_raw = try builder.createAndAttach("stablehlo.multiply", &.{final_grad_broadcast.lhs, final_grad_broadcast.rhs}, &.{final_grad_type}, .{});

    const final_grad = try ops.reduceGradient(builder, final_grad_raw.getResult(0), a);
    try result.append(final_grad);
    return result.toOwnedSlice();
}

/// VJP rule for convert: cast gradient back to input type
/// NOTE: The gradient shape might differ from primal shape due to operations after the convert.
/// We need to first match shapes via broadcast/reduce, then convert element type.
pub fn convertVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const input = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const grad_reshaped = try ops.reduceGradient(builder, grad_out, input);

    const grad_reshaped_type = grad_reshaped.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const grad_reshaped_shape = try grad_reshaped_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_reshaped_shape);

    const input_type_ranked = input.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const input_shape = try input_type_ranked.getShape(builder.allocator);
    defer builder.allocator.free(input_shape);

    var grad_shape_matched = grad_reshaped;
    if (grad_reshaped_shape.len != input_shape.len or !std.mem.eql(i64, grad_reshaped_shape, input_shape)) {
        const grad_reshaped_elem_type = grad_reshaped_type.getElementType();
        const ctx = mlir.Context{ .handle = c.mlirTypeGetContext(grad_reshaped.getType().handle) };
        const reshape_type = mlir.Type.rankedTensorType(ctx, input_shape, grad_reshaped_elem_type);
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_reshaped}, &.{reshape_type}, .{});
        grad_shape_matched = reshape_op.getResult(0);
    }

    const input_elem_type = input_type_ranked.getElementType();
    const ctx = mlir.Context{ .handle = c.mlirTypeGetContext(input.getType().handle) };
    const target_type = mlir.Type.rankedTensorType(ctx, input_shape, input_elem_type);

    const convert_op = try builder.createAndAttach("stablehlo.convert", &.{grad_shape_matched}, &.{target_type}, .{});

    try result.append(convert_op.getResult(0));
    return result.toOwnedSlice();
}

pub fn selectVJP(builder: *MLIRBuilder, original_op: mlir.Operation, primals: []const mlir.Value, adjoints: []const mlir.Value) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const pred = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    const shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(shape);
    const elem_type = result_ranked_type.getElementType();

    const zero_tensor = try ops.constant(builder, 0.0, shape, elem_type);
    const zero = zero_tensor.value;

    const grad_true = try builder.createAndAttach("stablehlo.select", &.{pred, grad_out, zero}, &.{result_type}, .{});
    const grad_false = try builder.createAndAttach("stablehlo.select", &.{pred, zero, grad_out}, &.{result_type}, .{});

    const pred_ranked_type = pred.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const pred_shape = try pred_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(pred_shape);
    const pred_elem_type = pred_ranked_type.getElementType();

    const pred_zero_tensor = try ops.constant(builder, 0.0, pred_shape, pred_elem_type);
    const pred_grad = pred_zero_tensor.value;

    try result.append(pred_grad);
    try result.append(grad_true.getResult(0));
    try result.append(grad_false.getResult(0));
    return result.toOwnedSlice();
}

/// VJP rule for tanh: d(tanh(x)) = (1 - tanh(x)^2) * dx
pub fn tanhVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out_raw = adjoints[0];
    const x = primals[0];

    const result_type = original_op.getResult(0).getType();
    const result_ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const result_shape = try result_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(result_shape);

    const grad_out_type = grad_out_raw.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const grad_out_shape = try grad_out_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_out_shape);

    var grad_out = grad_out_raw;
    if (!std.mem.eql(i64, result_shape, grad_out_shape)) {
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_out_raw}, &.{result_type}, .{});
        grad_out = reshape_op.getResult(0);
    }

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // Recompute tanh(x) in gradient graph
    const tanh_op = try builder.createAndAttach("stablehlo.tanh", &.{x}, &.{x.getType()}, .{});
    const y = try builder.newTensor(tanh_op.getResult(0));

    // Calculate 1.0 - y^2
    const y_sq = try ops.multiply(builder, y, y);
    const one = try ops.constant(builder, 1.0, result_shape, result_ranked_type.getElementType());
    const one_minus_sq = try ops.subtract(builder, one, y_sq);

    // grad_in = grad_out * (1 - y^2)
    const grad_tensor = try builder.newTensor(grad_out);
    const grad_x_raw = try ops.multiply(builder, grad_tensor, one_minus_sq);

    const grad_x = try ops.reduceGradient(builder, grad_x_raw.value, x);
    try result.append(grad_x);

    return result.toOwnedSlice();
}
