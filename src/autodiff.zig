const std = @import("std");
const mlir = @import("mlir.zig");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const c = @import("mlir/c.zig").c;
const hlo = @import("mlir/dialects/stablehlo.zig");

const Allocator = std.mem.Allocator;
const MLIRBuilder = ops.MLIRBuilder;

/// MLIR-based Automatic Differentiation using Graph-to-Graph Transformation
/// This implements reverse-mode AD on MLIR computation graphs using VJP rules

/// Vector-Jacobian Product (VJP) function signature
/// Takes the forward operation and output gradients, returns input gradients
/// NEW: Also takes primals (operands of the forward op) that are already mapped to the gradient scope.
/// And the original operation for accessing attributes when needed.
const VJPFn = *const fn(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) anyerror![]mlir.Value;

/// Runtime dispatch from operation name to VJP rule
fn getVjpFn(op_name: []const u8) ?VJPFn {
    if (std.mem.eql(u8, op_name, "stablehlo.add")) {
        return addVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.subtract")) {
        return subtractVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.multiply")) {
        return multiplyVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.divide")) {
        return divideVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.negate")) {
        return negateVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.transpose")) {
        return transposeVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.dot_general")) {
        return matmulVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.maximum")) {
        return reluVJP; // ReLU implemented as max(x, 0)
    } else if (std.mem.eql(u8, op_name, "stablehlo.constant")) {
        return constantVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.reshape")) {
        return reshapeVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.reduce_sum")) {
        return reduceSumVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.reduce")) {
        return reduceSumVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.gather")) {
        return gatherVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.slice")) {
        return sliceVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.broadcast_in_dim")) {
        return broadcastInDimVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.exponential")) {
        return expVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.log")) {
        return logVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.rsqrt")) {
        return rsqrtVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.convert")) {
        return convertVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.select")) {
        return selectVJP;
    } else if (std.mem.eql(u8, op_name, "func.return")) {
        return returnVJP;
    } else {
        return null;
    }
}

/// Check if an mlir.Value comes from a constant operation at compile time
fn isValueFromConstantOp(value: mlir.Value) bool {
    // In a full implementation, this would check if the defining op is stablehlo.constant
    // For now, we'll assume all values are non-constant to be conservative
    _ = value;
    return false;
}

/// Broadcast a value to a target shape using broadcast_in_dim
fn broadcastToShape(builder: *MLIRBuilder, value: mlir.Value, target_shape: []const i64) !mlir.Value {
    const value_type = value.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const value_shape = try value_type.getShape(builder.allocator);
    defer builder.allocator.free(value_shape);

    // Check if broadcasting is needed
    var needs_broadcast = false;
    if (value_shape.len != target_shape.len) {
        needs_broadcast = true;
    } else {
        for (value_shape, target_shape) |v, t| {
            if (v != t) {
                needs_broadcast = true;
                break;
            }
        }
    }

    if (!needs_broadcast) {
        return value;
    }

    // Create target type
    const element_type = value_type.getElementType();
    const ctx = mlir.Context{ .handle = c.mlirTypeGetContext(value.getType().handle) };
    const target_type = mlir.Type.rankedTensorType(ctx, target_shape, element_type);

    // Compute broadcast_dimensions attribute
    // Maps dimensions of the value to dimensions of the target
    var broadcast_dims = std.ArrayList(i64).init(builder.allocator);
    defer broadcast_dims.deinit();

    // Align from the right
    const rank_diff = target_shape.len - value_shape.len;
    for (0..value_shape.len) |i| {
        try broadcast_dims.append(@intCast(rank_diff + i));
    }

    const broadcast_dims_attr = mlir.Attribute.denseI64ArrayAttr(builder.ctx, broadcast_dims.items);
    const broadcast_op = try builder.createAndAttach("stablehlo.broadcast_in_dim",
        &.{value},
        &.{target_type},
        .{ .attributes = &.{.{ "broadcast_dimensions", broadcast_dims_attr }} }
    );

    return broadcast_op.getResult(0);
}

/// Compute broadcasted shape and broadcast operands to that shape for StableHLO operations
/// StableHLO requires explicit broadcasting - operands must have identical shapes AND types
fn broadcastOperands(builder: *MLIRBuilder, lhs: mlir.Value, rhs: mlir.Value) !struct { lhs: mlir.Value, rhs: mlir.Value, shape: []const i64 } {
    const lhs_type = lhs.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const rhs_type = rhs.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;

    const lhs_shape = try lhs_type.getShape(builder.allocator);
    defer builder.allocator.free(lhs_shape);
    const rhs_shape = try rhs_type.getShape(builder.allocator);
    defer builder.allocator.free(rhs_shape);

    const lhs_elem_type = lhs_type.getElementType();
    const rhs_elem_type = rhs_type.getElementType();

    const lhs_rank = lhs_shape.len;
    const rhs_rank = rhs_shape.len;
    const result_rank = @max(lhs_rank, rhs_rank);

    // Compute broadcasted shape
    const result_shape = try builder.allocator.alloc(i64, result_rank);

    var i: usize = 0;
    while (i < result_rank) : (i += 1) {
        const lhs_idx = if (i < lhs_rank) lhs_rank - 1 - i else null;
        const rhs_idx = if (i < rhs_rank) rhs_rank - 1 - i else null;

        const lhs_dim = if (lhs_idx) |idx| lhs_shape[idx] else 1;
        const rhs_dim = if (rhs_idx) |idx| rhs_shape[idx] else 1;

        const result_dim = if (lhs_dim == rhs_dim) lhs_dim
                          else if (lhs_dim == 1) rhs_dim
                          else if (rhs_dim == 1) lhs_dim
                          else {
            builder.allocator.free(result_shape);
            return error.IncompatibleShapesForBroadcast;
        };

        result_shape[result_rank - 1 - i] = result_dim;
    }

    // Handle element type conversion if needed (promote to common type)
    // For simplicity, if types differ, use rhs's type (usually the gradient type)
    var lhs_converted = lhs;
    const rhs_converted = rhs;

    // Check if element types match using proper MLIR type equality
    const types_match = lhs_elem_type.isEqual(rhs_elem_type);

    if (!types_match) {
        std.debug.print("DEBUG broadcastOperands: Element types differ, converting lhs\n", .{});
        std.debug.print("  lhs_shape: {any}, rhs_shape: {any}, result_shape: {any}\n", .{lhs_shape, rhs_shape, result_shape});
        // Convert lhs to rhs's element type (keeping lhs's original shape)
        const ctx = mlir.Context{ .handle = c.mlirTypeGetContext(lhs.getType().handle) };
        const lhs_converted_type = mlir.Type.rankedTensorType(ctx, lhs_shape, rhs_elem_type);
        std.debug.print("  Creating convert [broadcastOperands]: input_shape={any} output_shape={any}\n", .{lhs_shape, lhs_shape});
        const convert_op = try builder.createAndAttach("stablehlo.convert", &.{lhs}, &.{lhs_converted_type}, .{});
        lhs_converted = convert_op.getResult(0);
        std.debug.print("  Convert created successfully, now broadcasting\n", .{});
    }

    // Broadcast operands to the common shape
    std.debug.print("DEBUG broadcastOperands: Broadcasting lhs to {any}\n", .{result_shape});
    const broadcasted_lhs = try broadcastToShape(builder, lhs_converted, result_shape);
    std.debug.print("DEBUG broadcastOperands: Broadcasting rhs to {any}\n", .{result_shape});
    const broadcasted_rhs = try broadcastToShape(builder, rhs_converted, result_shape);
    std.debug.print("DEBUG broadcastOperands: Done\n", .{});

    return .{ .lhs = broadcasted_lhs, .rhs = broadcasted_rhs, .shape = result_shape };
}

/// Main automatic differentiation function - transforms forward graph to gradient graph
pub fn buildGradientGraph(
    allocator: Allocator,
    builder: *MLIRBuilder,
    forward_fn: mlir.Operation,
) !mlir.Operation {
    std.debug.print("Building gradient graph from forward function...\n", .{});
    
    // Derive the gradient function name from the forward function name
    const sym_name_ref = c.stringRefFromString("sym_name");
    const forward_fn_name_attr = c.operationGetAttributeByName(forward_fn.handle, sym_name_ref);
    if (@intFromPtr(forward_fn_name_attr.ptr) == 0 or !c.attributeIsAString(forward_fn_name_attr)) {
        return error.MissingOrInvalidSymName;
    }
    const forward_fn_name_ref = c.stringAttributeGetValue(forward_fn_name_attr);
    const forward_fn_name = c.fromStringRef(forward_fn_name_ref);
    const grad_fn_name = try std.fmt.allocPrint(allocator, "{s}_grad", .{forward_fn_name});
    defer allocator.free(grad_fn_name);
    
    // FIXED: Use the existing builder to maintain single context
    // This ensures gradient graph is built in the same context as forward graph
    
    // 1. Create the gradient function within the existing module/context
    const gradient_fn = try createGradientFunction(builder, forward_fn, grad_fn_name);

    // 2. Get the existing entry block (created automatically by createFunction)
    // CRITICAL FIX: Do NOT create a new block here. Use the one builder.createFunction made.
    const grad_fn_region = gradient_fn.getRegion(0);
    const grad_fn_block = grad_fn_region.getBlock(0);

    // CRITICAL FIX: Set the insertion point to the new gradient function's block.
    const original_insertion_block = builder.getInsertionBlock();
    builder.setInsertionBlock(grad_fn_block);
    defer builder.setInsertionBlock(original_insertion_block); // Restore on exit

    std.debug.print("Using gradient function entry block\n", .{});

    // Map from forward-pass values (primals) to their gradients (adjoints)
    var adjoint_map = std.AutoHashMap(mlir.Value, mlir.Value).init(allocator);
    defer adjoint_map.deinit();

    // Map from forward-pass values to their corresponding values in the gradient function's scope
    var value_map = std.AutoHashMap(mlir.Value, mlir.Value).init(allocator);
    defer value_map.deinit();

    // Get the operations in reverse topological order
    std.debug.print("Getting operations in reverse order...\n", .{});
    const ops_reversed = try getOperationsInReverseOrder(allocator, forward_fn);
    defer allocator.free(ops_reversed);

    // --- NEW INITIALIZATION LOGIC ---
    const forward_block = forward_fn.getRegion(0).getBlock(0);
    const num_forward_args = forward_block.getNumArguments();

    // 1. Map the arguments of the forward function to the arguments of the gradient function
    for (0..num_forward_args) |i| {
        const forward_arg = forward_block.getArgument(i);
        // CRITICAL FIX: Retrieve existing argument from the block instead of adding a new one
        const grad_arg = grad_fn_block.getArgument(i);
        try value_map.put(forward_arg, grad_arg);
    }

    // 2. The *last* argument of the gradient function is the incoming gradient for the loss
    const loss_value = getReturnValue(forward_fn) orelse return error.NoReturnOperation;
    // The loss gradient is the argument at index `num_forward_args` (last one)
    const loss_grad_arg = grad_fn_block.getArgument(num_forward_args);
    try adjoint_map.put(loss_value, loss_grad_arg);
    // --- END INITIALIZATION LOGIC ---
    
    // CRITICAL FIX: First pass - build value_map by processing operations in FORWARD order
    // This ensures all primal values are available in gradient scope before we compute gradients
    std.debug.print("First pass: Building primal value mappings in forward order...\n", .{});
    const ops_forward = try getOperationsInForwardOrder(allocator, forward_fn);
    defer allocator.free(ops_forward);
    
    for (ops_forward) |op| {
        const op_name = op.getName();
        
        if (std.mem.eql(u8, op_name, "stablehlo.constant")) {
            // Clone constants into gradient scope
            const cloned_op = mlir.Operation{ .handle = c.operationClone(op.handle) };
            builder.insertion_block.appendOwnedOperation(cloned_op);
            try value_map.put(op.getResult(0), cloned_op.getResult(0));
        } else if (!std.mem.eql(u8, op_name, "func.return")) {
            // Clone regular operations into gradient scope with mapped operands
            const cloned_op = mlir.Operation{ .handle = c.operationClone(op.handle) };
            
            // Replace operands with their mapped versions
            for (0..cloned_op.getNumOperands()) |i| {
                const primal_operand = op.getOperand(i);
                if (value_map.get(primal_operand)) |mapped_operand| {
                    c.mlirOperationSetOperand(cloned_op.handle, @intCast(i), mapped_operand.handle);
                }
            }
            
            builder.insertion_block.appendOwnedOperation(cloned_op);
            
            // Map the results
            for (0..op.getNumResults()) |i| {
                try value_map.put(op.getResult(i), cloned_op.getResult(i));
            }
        }
    }
    
    std.debug.print("Second pass: Computing gradients in reverse order...\n", .{});
    
    // Walk the forward graph backwards, applying VJP rules using the SHARED builder
    for (ops_reversed) |op| {
        std.debug.print("Processing operation VJP for: {s}\n", .{op.getName()});
        try processOperationVJP(allocator, builder, op, &adjoint_map, &value_map);
    }
    
    // Collect gradients and create return statement using the SHARED builder
    try finalizeGradientFunction(allocator, builder, gradient_fn, forward_fn, &adjoint_map);
    
    // Gradient function built successfully
    
    // Return the gradient function (not the whole module)
    return gradient_fn;
}

/// Process a single operation's VJP rule
fn processOperationVJP(
    allocator: Allocator,
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value),
    value_map: *std.AutoHashMap(mlir.Value, mlir.Value),
) !void {
    const op_name = op.getName();
    // Get gradients (now safely handles empty case)
    const output_gradients = try getOutputGradients(allocator, op, adjoint_map);

    // Always free the slice. Since getOutputGradients now guarantees a safe slice (even if len 0),
    // this allocator.free call is safe.
    defer allocator.free(output_gradients);

    if (output_gradients.len == 0) return;
    if (std.mem.eql(u8, op_name, "stablehlo.constant")) return;

    if (getVjpFn(op_name)) |vjp_rule| {
        var mapped_primals = std.ArrayList(mlir.Value).init(allocator);
        defer mapped_primals.deinit();

        for (0..op.getNumOperands()) |i| {
            const primal_operand = op.getOperand(i);
            const mapped_operand = value_map.get(primal_operand) orelse return error.PrimalValueNotFound;
            try mapped_primals.append(mapped_operand);
        }

        const input_gradients = try vjp_rule(builder, op, mapped_primals.items, output_gradients);
        defer builder.allocator.free(input_gradients); // Ensure VJP result slice is freed

        try addInputGradients(builder, op, input_gradients, adjoint_map);
    }
}

/// VJP rule for addition: both inputs get the same gradient
fn addVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const a = primals[0];
    const b = primals[1];

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // Gradient for both inputs is the output gradient, but reduced to match their shapes
    const grad_a = try reduceGradient(builder, grad_out, a);
    try result.append(grad_a);

    const grad_b = try reduceGradient(builder, grad_out, b);
    try result.append(grad_b);

    return result.toOwnedSlice();
}

/// VJP rule for subtraction: da = grad_out, db = -grad_out
fn subtractVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const a = primals[0];
    const b = primals[1];

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // da = grad_out, reduced to match a's shape
    const grad_a = try reduceGradient(builder, grad_out, a);
    try result.append(grad_a);

    // db = -grad_out, reduced to match b's shape
    const neg_grad = try builder.createAndAttach("stablehlo.negate", &.{grad_out}, &.{grad_out.getType()}, .{});
    const grad_b = try reduceGradient(builder, neg_grad.getResult(0), b);
    try result.append(grad_b);

    return result.toOwnedSlice();
}

/// VJP rule for multiplication: da = grad_out * b, db = grad_out * a
fn multiplyVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const a = primals[0]; // Primal 'a' - NOW VALID IN THIS SCOPE
    const b = primals[1]; // Primal 'b' - NOW VALID IN THIS SCOPE

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // da = grad_out * b with explicit broadcasting
    const grad_a_broadcast = try broadcastOperands(builder, grad_out, b);
    defer builder.allocator.free(grad_a_broadcast.shape);
    const grad_a_type = grad_a_broadcast.lhs.getType();
    const grad_a_raw = try builder.createAndAttach("stablehlo.multiply", &.{ grad_a_broadcast.lhs, grad_a_broadcast.rhs }, &.{grad_a_type}, .{});

    // Reduce to match a's shape if broadcasting occurred
    const grad_a = try reduceGradient(builder, grad_a_raw.getResult(0), a);
    try result.append(grad_a);

    // db = grad_out * a with explicit broadcasting
    const grad_b_broadcast = try broadcastOperands(builder, grad_out, a);
    defer builder.allocator.free(grad_b_broadcast.shape);
    const grad_b_type = grad_b_broadcast.lhs.getType();
    const grad_b_raw = try builder.createAndAttach("stablehlo.multiply", &.{ grad_b_broadcast.lhs, grad_b_broadcast.rhs }, &.{grad_b_type}, .{});

    // Reduce to match b's shape if broadcasting occurred
    const grad_b = try reduceGradient(builder, grad_b_raw.getResult(0), b);
    try result.append(grad_b);

    return result.toOwnedSlice();
}

/// VJP rule for division: da = grad_out / b, db = -grad_out * a / (b * b)
/// Helper to reduce gradient to match primal shape (handles broadcasting)
fn reduceGradient(builder: *MLIRBuilder, grad: mlir.Value, target_shape: mlir.Value) !mlir.Value {
    const grad_type = grad.getType().as(mlir.RankedTensorType) orelse return grad;
    const target_type = target_shape.getType().as(mlir.RankedTensorType) orelse return grad;

    const grad_shape = try grad_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_shape);
    const target_shape_arr = try target_type.getShape(builder.allocator);
    defer builder.allocator.free(target_shape_arr);

    // If shapes match, no reduction needed
    if (grad_shape.len == target_shape_arr.len) {
        var match = true;
        for (grad_shape, target_shape_arr) |g, t| {
            if (g != t) {
                match = false;
                break;
            }
        }
        if (match) return grad;
    }

    // Find dimensions to reduce
    var reduce_dims = std.ArrayList(i64).init(builder.allocator);
    defer reduce_dims.deinit();

    // Handle rank mismatch - reduce leading dimensions
    const rank_diff = @as(i64, @intCast(grad_shape.len)) - @as(i64, @intCast(target_shape_arr.len));
    if (rank_diff > 0) {
        for (0..@intCast(rank_diff)) |i| {
            try reduce_dims.append(@intCast(i));
        }
    }

    // Handle size-1 dimensions in target (broadcasted dimensions)
    if (rank_diff >= 0) {
        for (target_shape_arr, 0..) |t_dim, i| {
            const grad_idx = @as(usize, @intCast(@as(i64, @intCast(i)) + rank_diff));
            if (grad_idx < grad_shape.len and t_dim == 1 and grad_shape[grad_idx] != 1) {
                try reduce_dims.append(@as(i64, @intCast(grad_idx)));
            }
        }
    }

    if (reduce_dims.items.len == 0) return grad;

    // Use ops.reduceSum which properly creates the reduction region
    const grad_tensor = try builder.newTensor(grad);
    const reduced_tensor = try ops.reduceSum(builder, grad_tensor, reduce_dims.items, false);

    return reduced_tensor.value;
}

fn divideVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.print("DEBUG: Entering divideVJP\n", .{});
    _ = original_op;

    if (adjoints.len < 1) {
        std.debug.print("FATAL: divideVJP adjoints empty\n", .{});
        return error.InvalidAdjoints;
    }
    if (primals.len < 2) {
        std.debug.print("FATAL: divideVJP primals < 2\n", .{});
        return error.InvalidPrimals;
    }

    const grad_out = adjoints[0];
    const a = primals[0];
    const b = primals[1];

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    errdefer result.deinit();

    std.debug.print("DEBUG: divideVJP: computing grad_a\n", .{});
    // da = grad_out / b with explicit broadcasting
    const grad_a_broadcast = try broadcastOperands(builder, grad_out, b);
    defer builder.allocator.free(grad_a_broadcast.shape);
    const grad_a_type = grad_a_broadcast.lhs.getType();
    const grad_a_raw = try builder.createAndAttach("stablehlo.divide", &.{ grad_a_broadcast.lhs, grad_a_broadcast.rhs }, &.{grad_a_type}, .{});

    // Reduce to match a's shape if broadcasting occurred
    const grad_a = try reduceGradient(builder, grad_a_raw.getResult(0), a);
    try result.append(grad_a);

    std.debug.print("DEBUG: divideVJP: computing grad_b\n", .{});
    // db = -grad_out * a / (b * b)

    // a * grad_out with explicit broadcasting
    const a_times_grad_broadcast = try broadcastOperands(builder, a, grad_out);
    defer builder.allocator.free(a_times_grad_broadcast.shape);
    const a_times_grad_type = a_times_grad_broadcast.lhs.getType();
    const a_times_grad = try builder.createAndAttach("stablehlo.multiply", &.{ a_times_grad_broadcast.lhs, a_times_grad_broadcast.rhs }, &.{a_times_grad_type}, .{});

    // b * b (no broadcasting needed, both operands are b)
    const b_squared = try builder.createAndAttach("stablehlo.multiply", &.{ b, b }, &.{b.getType()}, .{});

    // (a * grad_out) / (b * b) with explicit broadcasting
    const div_operands_broadcast = try broadcastOperands(builder, a_times_grad.getResult(0), b_squared.getResult(0));
    defer builder.allocator.free(div_operands_broadcast.shape);
    const div_operands_type = div_operands_broadcast.lhs.getType();
    const positive_grad_b_raw = try builder.createAndAttach("stablehlo.divide", &.{ div_operands_broadcast.lhs, div_operands_broadcast.rhs }, &.{div_operands_type}, .{});

    // Negate
    const positive_grad_b_neg = try builder.createAndAttach("stablehlo.negate", &.{ positive_grad_b_raw.getResult(0) }, &.{div_operands_type}, .{});

    // Reduce to match b's shape if broadcasting occurred
    const grad_b = try reduceGradient(builder, positive_grad_b_neg.getResult(0), b);
    try result.append(grad_b);

    std.debug.print("DEBUG: divideVJP: converting to owned slice\n", .{});
    const slice = try result.toOwnedSlice();

    std.debug.print("DEBUG: divideVJP allocated slice len={} ptr={*}\n", .{slice.len, slice.ptr});
    return slice;
}

/// VJP rule for negation: da = -grad_out
fn negateVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    _ = primals;
    const grad_out = adjoints[0];
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();
    
    const grad_a = try builder.createAndAttach("stablehlo.negate", &.{grad_out}, &.{grad_out.getType()}, .{});
    try result.append(grad_a.getResult(0));
    
    return result.toOwnedSlice();
}

/// VJP rule for transpose: da = transpose(grad_out) with inverse permutation
fn transposeVJP(builder: *MLIRBuilder, original_op: mlir.Operation, primals: []const mlir.Value, adjoints: []const mlir.Value) ![]mlir.Value {
    const grad_out = adjoints[0];
    const a = primals[0];
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

    // Use safe wrapper instead of raw C API
    const inv_perm_attr = mlir.Attribute.denseI64ArrayAttr(builder.ctx, inv_permutation);

    // Safe operation creation via builder
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
fn matmulVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out = adjoints[0];
    const a = primals[0]; // Primal 'a', now valid in grad scope
    const b = primals[1]; // Primal 'b', now valid in grad scope
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();
    
    // Parse the original operation's dot_dimension_numbers attribute
    const dot_dim_attr = c.mlirOperationGetAttributeByName(original_op.handle, c.stringRefCreateFromCString("dot_dimension_numbers"));
    if (@intFromPtr(dot_dim_attr.ptr) == 0) {
        return error.MissingDotDimensionNumbersAttribute;
    }
    
    // For this implementation, we'll handle the common case of batched matrix multiplication
    // A @ B where A is [batch..., M, K] and B is [batch..., K, N] -> Y is [batch..., M, N]
    // The original operation contracts the last dim of A with the second-to-last dim of B
    
    // For a more general implementation, we would parse the actual attribute here
    // For now, we'll assume standard batched matmul: contracting dims are [-1, -2]
    
    // Get tensor shapes to determine dimensions
    const a_type = a.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const b_type = b.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const grad_out_type = grad_out.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    
    const a_rank = a_type.getRank();
    const b_rank = b_type.getRank();
    const grad_rank = grad_out_type.getRank();
    
    // For batched matmul, batch dimensions are all but the last 2
    var batch_dims = std.ArrayList(i64).init(builder.allocator);
    defer batch_dims.deinit();
    
    const num_batch_dims = @min(@min(a_rank, b_rank), grad_rank) - 2;
    for (0..num_batch_dims) |i| {
        try batch_dims.append(@intCast(i));
    }
    
    // VJP for 'a' (dA = dY @ B^T):
    // Contract the last dimension of grad_out with the last dimension of B
    const grad_a_dot_dims = hlo.DotDimensionNumbersAttribute{
        .lhs_batching_dimensions = batch_dims.items,
        .rhs_batching_dimensions = batch_dims.items,
        .lhs_contracting_dimensions = &.{@intCast(grad_rank - 1)}, // Last dim of grad_out (N)
        .rhs_contracting_dimensions = &.{@intCast(b_rank - 1)},    // Last dim of B (N)
    };
    const grad_a_op = try hlo.dot_general(builder.allocator, builder.ctx, grad_out, b, .{ .dot_dimension_numbers = grad_a_dot_dims });
    builder.insertion_block.appendOwnedOperation(grad_a_op);
    try result.append(grad_a_op.getResult(0));
    
    // VJP for 'b' (dB = A^T @ dY):
    // Contract the second-to-last dimension of A with the second-to-last dimension of grad_out
    const grad_b_dot_dims = hlo.DotDimensionNumbersAttribute{
        .lhs_batching_dimensions = batch_dims.items,
        .rhs_batching_dimensions = batch_dims.items,
        .lhs_contracting_dimensions = &.{@intCast(a_rank - 2)},    // Second-to-last dim of A (M)
        .rhs_contracting_dimensions = &.{@intCast(grad_rank - 2)}, // Second-to-last dim of grad_out (M)
    };
    const grad_b_op = try hlo.dot_general(builder.allocator, builder.ctx, a, grad_out, .{ .dot_dimension_numbers = grad_b_dot_dims });
    builder.insertion_block.appendOwnedOperation(grad_b_op);
    try result.append(grad_b_op.getResult(0));
    
    return result.toOwnedSlice();
}

/// VJP rule for ReLU (max(x, 0)): gradient flows through only where x > 0
fn reluVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const x = primals[0]; // Input to ReLU
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();
    
    // Create mask: x > 0
    const zero = try builder.scalarConstant(0.0);
    const mask = try builder.createAndAttach("stablehlo.compare", &.{ x, zero.value }, &.{x.getType()}, .{}); // x > 0
    
    // Apply mask: grad_out * mask
    const grad_x = try builder.createAndAttach("stablehlo.select", &.{ mask.getResult(0), grad_out, zero.value }, &.{grad_out.getType()}, .{});
    try result.append(grad_x.getResult(0));
    
    return result.toOwnedSlice();
}

/// VJP rule for constants: no gradient (constants don't have inputs)
fn constantVJP(
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
fn reshapeVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const input = primals[0]; // Original input to reshape
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();
    
    // Get the original input shape and reshape the gradient back to it
    const original_shape_type = input.getType();
    
    // Create reshape operation to restore original shape
    const grad_input = try builder.createAndAttach("stablehlo.reshape", &.{grad_out}, &.{original_shape_type}, .{});
    try result.append(grad_input.getResult(0));
    
    return result.toOwnedSlice();
}

fn reduceSumVJP(builder: *MLIRBuilder, original_op: mlir.Operation, primals: []const mlir.Value, adjoints: []const mlir.Value) ![]mlir.Value {
    const grad_out = adjoints[0];
    const input = primals[0];
    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    if (!isValueFromConstantOp(input)) {
        const original_shape_type = input.getType();
        const dimensions_ref = c.stringRefFromString("dimensions");
        const dimensions_attr = c.operationGetAttributeByName(original_op.handle, dimensions_ref);

        if (@intFromPtr(dimensions_attr.ptr) != 0) {
            var broadcast_dims = std.ArrayList(i64).init(builder.allocator);
            defer broadcast_dims.deinit();

            const grad_out_type = grad_out.getType().as(mlir.RankedTensorType).?;
            const grad_rank = grad_out_type.getRank();

            if (grad_rank == 0) {
                // Scalar broadcast (empty dims)
                // SAFE: Use high-level wrapper
                const empty_dims: [0]i64 = .{};
                const empty_broadcast_dims_attr = mlir.Attribute.denseI64ArrayAttr(builder.ctx, &empty_dims);

                // SAFE PATTERN: Use builder.createAndAttach
                const broadcast_op = try builder.createAndAttach("stablehlo.broadcast_in_dim",
                    &.{grad_out},
                    &.{original_shape_type},
                    .{ .attributes = &.{.{ "broadcast_dimensions", empty_broadcast_dims_attr }} }
                );
                try result.append(broadcast_op.getResult(0));
            } else {
                // Partial broadcast
                for (0..@intCast(grad_rank)) |i| try broadcast_dims.append(@intCast(i));

                // SAFE: Use high-level wrapper instead of raw C API
                const broadcast_dims_attr = mlir.Attribute.denseI64ArrayAttr(builder.ctx, broadcast_dims.items);

                // SAFE PATTERN: Use builder.createAndAttach
                const broadcast_op = try builder.createAndAttach("stablehlo.broadcast_in_dim",
                    &.{grad_out},
                    &.{original_shape_type},
                    .{ .attributes = &.{.{ "broadcast_dimensions", broadcast_dims_attr }} }
                );
                try result.append(broadcast_op.getResult(0));
            }
        } else {
            const grad_input = try builder.createAndAttach("stablehlo.broadcast", &.{grad_out}, &.{original_shape_type}, .{});
            try result.append(grad_input.getResult(0));
        }
    }
    return result.toOwnedSlice();
}

/// VJP rule for gather: da = scatter(grad_out, start_indices, original_shape)
fn gatherVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out = adjoints[0];
    const operand = primals[0];      // Original operand (e.g., embedding table)
    const start_indices = primals[1]; // Indices used for gathering

    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    // Gradient for operand: scatter grad_out back to the original positions
    if (!isValueFromConstantOp(operand)) {
        const original_shape_type = operand.getType();

        // Create zero tensor with original shape using proper constant creation
        const ranked_type = original_shape_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
        const shape = try ranked_type.getShape(builder.allocator);
        defer builder.allocator.free(shape);
        const element_type = ranked_type.getElementType();

        const zero_constant_op = try hlo.zeroConstant(builder.allocator, builder.ctx, shape, element_type);
        builder.insertion_block.appendOwnedOperation(zero_constant_op);
        const zero_tensor = zero_constant_op;

        // Extract gather dimension numbers from the original gather operation
        const dim_numbers_attr_ref = c.stringRefFromString("dimension_numbers");
        const dim_numbers_attr = c.operationGetAttributeByName(original_op.handle, dim_numbers_attr_ref);

        if (@intFromPtr(dim_numbers_attr.ptr) == 0) {
            return error.MissingDimensionNumbersAttribute;
        }

        // Extract gather dimension numbers components
        const offset_dims_size = c.stablehloGatherDimensionNumbersGetOffsetDimsSize(dim_numbers_attr);
        const collapsed_slice_dims_size = c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(dim_numbers_attr);
        const start_index_map_size = c.stablehloGatherDimensionNumbersGetStartIndexMapSize(dim_numbers_attr);
        const index_vector_dim = c.stablehloGatherDimensionNumbersGetIndexVectorDim(dim_numbers_attr);

        // Extract offset_dims (becomes update_window_dims for scatter)
        var update_window_dims = try builder.allocator.alloc(i64, @intCast(offset_dims_size));
        defer builder.allocator.free(update_window_dims);
        for (0..@intCast(offset_dims_size)) |i| {
            update_window_dims[i] = c.stablehloGatherDimensionNumbersGetOffsetDimsElem(dim_numbers_attr, @intCast(i));
        }

        // Extract collapsed_slice_dims (becomes inserted_window_dims for scatter)
        var inserted_window_dims = try builder.allocator.alloc(i64, @intCast(collapsed_slice_dims_size));
        defer builder.allocator.free(inserted_window_dims);
        for (0..@intCast(collapsed_slice_dims_size)) |i| {
            inserted_window_dims[i] = c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(dim_numbers_attr, @intCast(i));
        }

        // Extract start_index_map (becomes scatter_dims_to_operand_dims for scatter)
        var scatter_dims_to_operand_dims = try builder.allocator.alloc(i64, @intCast(start_index_map_size));
        defer builder.allocator.free(scatter_dims_to_operand_dims);
        for (0..@intCast(start_index_map_size)) |i| {
            scatter_dims_to_operand_dims[i] = c.stablehloGatherDimensionNumbersGetStartIndexMapElem(dim_numbers_attr, @intCast(i));
        }

        const scatter_dim_numbers = hlo.ScatterDimensionNumbersAttribute{
            .update_window_dims = update_window_dims,
            .inserted_window_dims = inserted_window_dims,
            .scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
            .index_vector_dim = index_vector_dim,
        };

        const scatter_op = try hlo.scatter(builder.allocator, builder.ctx, zero_tensor.getResult(0), start_indices, grad_out, scatter_dim_numbers, builder.loc);
        builder.insertion_block.appendOwnedOperation(scatter_op);
        try result.append(scatter_op.getResult(0));
    }

    // Gradient for start_indices: typically zero (indices don't have gradients)
    // We don't add anything for start_indices as they're typically integers

    return result.toOwnedSlice();
}

/// VJP rule for function return: just pass through the gradient
fn returnVJP(
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
fn sliceVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out = adjoints[0];
    const input = primals[0];

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
fn broadcastInDimVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const input = primals[0]; // The original, smaller tensor

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // Only compute gradients for non-constant operands
    if (!isValueFromConstantOp(input)) {
        // The gradient of the input is the sum of the output gradients, reduced
        // along the dimensions that were added by the broadcast.

        const input_type = input.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
        const grad_out_type = grad_out.getType().as(mlir.RankedTensorType) orelse return error.InvalidTensorType;

        const input_shape = try input_type.getShape(builder.allocator);
        defer builder.allocator.free(input_shape);
        const grad_out_shape = try grad_out_type.getShape(builder.allocator);
        defer builder.allocator.free(grad_out_shape);

        const input_rank = input_shape.len;
        const grad_out_rank = grad_out_shape.len;

        // Find the dimensions that need to be reduced. These are all dimensions in the
        // output shape that are not part of the input shape's broadcast mapping.
        var dims_to_reduce = std.ArrayList(i64).init(builder.allocator);
        defer dims_to_reduce.deinit();

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

        // Create a tensor wrapper for grad_out
        const grad_out_tensor = try builder.newTensor(grad_out);
        
        // Create a reduce_sum op to get the gradient of the input
        var grad_input_tensor: tensor.Tensor(void) = undefined;
        if (dims_to_reduce.items.len > 0) {
            grad_input_tensor = try ops.reduceSum(builder, grad_out_tensor, dims_to_reduce.items, false);
        } else {
            // If no dimensions to reduce, the gradient passes through unchanged
            grad_input_tensor = grad_out_tensor;
        }
        
        try result.append(grad_input_tensor.value);
        
        std.debug.print("✓ broadcastInDimVJP: Created gradient flow for broadcast_in_dim operation\n", .{});
    }

    return result.toOwnedSlice();
}

// Helper functions for graph traversal and manipulation


fn createGradientFunction(builder: *MLIRBuilder, forward_fn: mlir.Operation, name: []const u8) !mlir.Operation {
    const forward_region = forward_fn.getRegion(0);
    const forward_block = forward_region.getBlock(0);
    const num_args = forward_block.getNumArguments();

    var input_types = std.ArrayList(mlir.Type).init(builder.allocator);
    defer input_types.deinit();
    for (0..num_args) |i| {
        const arg = forward_block.getArgument(i);
        try input_types.append(arg.getType());
    }

    const return_value = getReturnValue(forward_fn) orelse {
        return error.NoReturnOperation;
    };
    try input_types.append(return_value.getType());

    var output_types = std.ArrayList(mlir.Type).init(builder.allocator);
    defer output_types.deinit();
    for (0..num_args) |i| {
        const arg = forward_block.getArgument(i);
        try output_types.append(arg.getType());
    }

    // UPDATE: Pass builder.allocator to functionType
    const function_type = try mlir.Type.functionType(builder.allocator, builder.ctx, input_types.items, output_types.items);

    // SAFE REFACTOR: Delegate to builder.createFunction
    // This uses the already-fixed, robust manual initialization logic in ops.zig
    // instead of trying to replicate C-API calls here.
    const result = try builder.createFunction(name, function_type);

    // The builder automatically appends the entry block, but createFunction returns it.
    // We don't need to do anything else as createFunction handles the region/block setup.

    std.debug.print("Created gradient function '{s}' via builder\n", .{name});
    return result.func_op;
}

fn getOperationsInReverseOrder(allocator: Allocator, fn_op: mlir.Operation) ![]mlir.Operation {
    std.debug.print("  Building reverse operation order...\n", .{});

    var sorted_ops = std.ArrayList(mlir.Operation).init(allocator);
    errdefer sorted_ops.deinit();

    var worklist = std.ArrayList(mlir.Operation).init(allocator);
    defer worklist.deinit();

    var visited = std.AutoHashMap(mlir.Operation, void).init(allocator);
    defer visited.deinit();

    // CRITICAL FIX: The fn_op is the func.func operation itself. We must start
    // the reverse traversal from ITS terminator (the func.return op).
    const func_body_region = fn_op.getRegion(0);
    const func_body_block = func_body_region.getBlock(0);
    
    // FIXED: Use getLastOpGeneric instead of getLastOp (which relies on blockGetTerminator)
    const terminator = func_body_block.getLastOpGeneric();
    
    if (terminator) |term_op| {
        // Verify it's actually a func.return
        const op_name = term_op.getName();
        if (std.mem.eql(u8, op_name, "func.return")) {
            try worklist.append(term_op);
        } else {
            std.debug.print("  ERROR: Last operation is not func.return, it's: {s}\\n", .{op_name});
            return error.InvalidTerminator;
        }
    } else {
        std.debug.print("  ERROR: No last operation found in the forward function!\\n", .{});
        return error.NoTerminatorFound;
    }

    // Process the worklist using the direct C-API call
    while (worklist.items.len > 0) {
        const op = worklist.orderedRemove(worklist.items.len - 1);
        if (visited.contains(op)) continue;
        try visited.put(op, {});
        try sorted_ops.append(op);

        for (0..op.getNumOperands()) |i| {
            const operand = op.getOperand(i);

            // Check if this value is an operation result (not a block argument)
            if (c.mlirValueIsAOpResult(operand.handle)) {
                // Get the operation that produced this result
                const def_op_handle = c.mlirOpResultGetOwner(operand.handle);
                const def_op = mlir.Operation{ .handle = def_op_handle };
                if (!visited.contains(def_op)) {
                    try worklist.append(def_op);
                }
            }
            // If it's a block argument, the traversal correctly stops here
        }
    }
    
    std.debug.print("  Found {} operations in reverse order\n", .{sorted_ops.items.len});
    return sorted_ops.toOwnedSlice();
}

fn getOperationsInForwardOrder(allocator: Allocator, fn_op: mlir.Operation) ![]mlir.Operation {
    var operations = std.ArrayList(mlir.Operation).init(allocator);
    errdefer operations.deinit();
    
    const func_body_region = fn_op.getRegion(0);
    const func_body_block = func_body_region.getBlock(0);
    
    // Simply iterate through operations in forward order
    var maybe_op = func_body_block.getFirstOp();
    while (maybe_op) |op| {
        try operations.append(op);
        maybe_op = op.getNext();
    }
    
    return operations.toOwnedSlice();
}

fn getReturnValue(fn_op: mlir.Operation) ?mlir.Value {
    const region = fn_op.getRegion(0);
    const block = region.getBlock(0);

    // --- RECOMMENDED FIX ---

    // 1. First, try the more direct `getLastOpGeneric()`, which we've confirmed works.
    if (block.getLastOpGeneric()) |last_op| {
        // 2. Verify that the last operation is, in fact, a `func.return`.
        if (std.mem.eql(u8, last_op.getName(), "func.return")) {
            // This is the expected, successful path.
            if (last_op.getNumOperands() > 0) {
                return last_op.getOperand(0);
            }
        }
    }

    // 3. Fallback to the existing iteration method for full robustness, just in case.
    std.debug.print("getReturnValue: Falling back to full iteration to find terminator.\n", .{});
    var maybe_op = block.getFirstOp();
    while (maybe_op) |op| {
        if (std.mem.eql(u8, op.getName(), "func.return")) {
            if (op.getNumOperands() > 0) {
                return op.getOperand(0);
            }
        }
        maybe_op = op.getNext();
    }
    
    // If no return value is found by either method.
    return null;
}

fn getOutputGradients(allocator: Allocator, op: mlir.Operation, adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value)) ![]mlir.Value {
    var gradients = std.ArrayList(mlir.Value).init(allocator);

    const num_results = op.getNumResults();
    for (0..num_results) |i| {
        const result_value = op.getResult(i);
        if (adjoint_map.get(result_value)) |gradient| {
            try gradients.append(gradient);
        }
    }

    // CRITICAL SAFETY FIX: Handle empty lists explicitly.
    // std.ArrayList.toOwnedSlice() behavior on empty lists can vary.
    // Explicitly allocating size 0 ensures we get a pointer the allocator recognizes (or is safe to free).
    if (gradients.items.len == 0) {
        gradients.deinit();
        return allocator.alloc(mlir.Value, 0);
    }

    return gradients.toOwnedSlice();
}

fn addInputGradients(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    input_gradients: []const mlir.Value,
    adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value),
) !void {
    // Add gradients for each input operand to the adjoint map
    // If a gradient already exists, sum them
    const num_operands = op.getNumOperands();
    
    for (0..num_operands) |i| {
        if (i >= input_gradients.len) break;
        
        const operand = op.getOperand(i);
        const grad = input_gradients[i];
        
        // If a gradient for this operand already exists, add the new one to it.
        // Must broadcast to handle shape mismatches
        if (adjoint_map.get(operand)) |existing_grad| {
            const add_broadcast = try broadcastOperands(builder, existing_grad, grad);
            defer builder.allocator.free(add_broadcast.shape);
            const add_type = add_broadcast.lhs.getType();
            const sum_op = try builder.createAndAttach("stablehlo.add", &.{ add_broadcast.lhs, add_broadcast.rhs }, &.{add_type}, .{});
            try adjoint_map.put(operand, sum_op.getResult(0));
        } else {
            try adjoint_map.put(operand, grad);
        }
    }
}

fn finalizeGradientFunction(allocator: Allocator, builder: *MLIRBuilder, gradient_fn: mlir.Operation, forward_fn: mlir.Operation, adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value)) !void {
    _ = gradient_fn;
    const forward_region = forward_fn.getRegion(0);
    const forward_block = forward_region.getBlock(0);
    const num_args = forward_block.getNumArguments();
    var input_gradients = std.ArrayList(mlir.Value).init(allocator);
    defer input_gradients.deinit();

    std.debug.print("Finalizing gradient function with {} forward function arguments\n", .{num_args});

    for (0..num_args) |i| {
        const arg = forward_block.getArgument(i);
        if (adjoint_map.get(arg)) |gradient| {
            try input_gradients.append(gradient);
            std.debug.print("  Found gradient for argument {}\n", .{i});
        } else {
            // Create zero gradient using safe builder constant helper
            const arg_type = arg.getType();
            const ranked_type = arg_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
            const shape = try ranked_type.getShape(allocator);
            defer allocator.free(shape);

            const elem_type = ranked_type.getElementType();

            // Use builder to create constant 0.0
            const zero_tensor = try ops.constant(builder, 0.0, shape, elem_type);
            try input_gradients.append(zero_tensor.value);

            std.debug.print("  Created zero gradient for argument {}\n", .{i});
        }
    }

    // Create return operation using safe builder
    // func.return takes the input_gradients as operands
    _ = try builder.createAndAttach("func.return", input_gradients.items, &.{}, .{});

    std.debug.print("Added func.return to gradient function with {} gradients\n", .{input_gradients.items.len});
}

/// High-level API for automatic differentiation
pub const AutoDiff = struct {
    allocator: Allocator,
    builder: *MLIRBuilder,
    
    pub fn init(allocator: Allocator, builder: *MLIRBuilder) AutoDiff {
        return AutoDiff{
            .allocator = allocator,
            .builder = builder,
        };
    }
    
    /// Create gradient function from forward function
    pub fn grad(self: *AutoDiff, forward_fn: mlir.Operation) !mlir.Operation {
        return buildGradientGraph(self.allocator, self.builder, forward_fn);
    }
    
    /// Compose forward and backward pass into a single training function
    pub fn valueAndGrad(self: *AutoDiff, forward_fn: mlir.Operation) !mlir.Operation {
        // Create function that returns both value and gradients
        const grad_fn = try self.grad(forward_fn);
        
        // In a real implementation, this would create a new function that calls both
        // forward_fn and grad_fn and returns both results
        _ = grad_fn;
        return forward_fn; // Placeholder
    }
};

/// VJP rule for exponential: da = grad_out * exp(a)
fn expVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const a = primals[0];

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // recompute exp(a)
    // In an optimized system we would cache this from forward pass,
    // but recomputing is safe and correct.
    const exp_a = try builder.createAndAttach("stablehlo.exponential", &.{a}, &.{a.getType()}, .{});

    // grad_in = grad_out * exp(a) with explicit broadcasting
    const grad_in_broadcast = try broadcastOperands(builder, grad_out, exp_a.getResult(0));
    defer builder.allocator.free(grad_in_broadcast.shape);
    const grad_in_type = grad_in_broadcast.lhs.getType();
    const grad_in_raw = try builder.createAndAttach("stablehlo.multiply", &.{grad_in_broadcast.lhs, grad_in_broadcast.rhs}, &.{grad_in_type}, .{});

    // Reduce to match a's shape if broadcasting occurred
    const grad_in = try reduceGradient(builder, grad_in_raw.getResult(0), a);
    try result.append(grad_in);
    return result.toOwnedSlice();
}

/// VJP rule for log: da = grad_out / a
fn logVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const a = primals[0];

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // grad_in = grad_out / a with explicit broadcasting
    const grad_in_broadcast = try broadcastOperands(builder, grad_out, a);
    defer builder.allocator.free(grad_in_broadcast.shape);
    const grad_in_type = grad_in_broadcast.lhs.getType();
    const grad_in_raw = try builder.createAndAttach("stablehlo.divide", &.{grad_in_broadcast.lhs, grad_in_broadcast.rhs}, &.{grad_in_type}, .{});

    // Reduce to match a's shape if broadcasting occurred
    const grad_in = try reduceGradient(builder, grad_in_raw.getResult(0), a);
    try result.append(grad_in);
    return result.toOwnedSlice();
}

/// VJP for rsqrt: da = -0.5 * grad_out * (a ^ -1.5)
fn rsqrtVJP(builder: *MLIRBuilder, original_op: mlir.Operation, primals: []const mlir.Value, adjoints: []const mlir.Value) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const a = primals[0];
    const tensor_type = a.getType();
    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    const rsqrt_op = try builder.createAndAttach("stablehlo.rsqrt", &.{a}, &.{tensor_type}, .{});
    const rsqrt_val = rsqrt_op.getResult(0);
    const sq = try builder.createAndAttach("stablehlo.multiply", &.{rsqrt_val, rsqrt_val}, &.{tensor_type}, .{});
    const cub = try builder.createAndAttach("stablehlo.multiply", &.{sq.getResult(0), rsqrt_val}, &.{tensor_type}, .{});

    // Use safe ops.constant helper
    const ranked_type = tensor_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const shape = try ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(shape);
    const elem_type = ranked_type.getElementType();

    const neg_half_tensor = try ops.constant(builder, -0.5, shape, elem_type);
    const neg_half = neg_half_tensor.value;

    // term1 = grad_out * neg_half with explicit broadcasting
    const term1_broadcast = try broadcastOperands(builder, grad_out, neg_half);
    defer builder.allocator.free(term1_broadcast.shape);
    const term1_type = term1_broadcast.lhs.getType();
    const term1 = try builder.createAndAttach("stablehlo.multiply", &.{term1_broadcast.lhs, term1_broadcast.rhs}, &.{term1_type}, .{});

    // final_grad = term1 * cub with explicit broadcasting
    const final_grad_broadcast = try broadcastOperands(builder, term1.getResult(0), cub.getResult(0));
    defer builder.allocator.free(final_grad_broadcast.shape);
    const final_grad_type = final_grad_broadcast.lhs.getType();
    const final_grad_raw = try builder.createAndAttach("stablehlo.multiply", &.{final_grad_broadcast.lhs, final_grad_broadcast.rhs}, &.{final_grad_type}, .{});

    // Reduce to match a's shape if broadcasting occurred
    const final_grad = try reduceGradient(builder, final_grad_raw.getResult(0), a);
    try result.append(final_grad);
    return result.toOwnedSlice();
}

/// VJP rule for convert: cast gradient back to input type
/// NOTE: The gradient shape might differ from primal shape due to operations after the convert.
/// We need to first match shapes via broadcast/reduce, then convert element type.
fn convertVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const input = primals[0];

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // First, reduce/broadcast gradient to match input shape
    const grad_reshaped = try reduceGradient(builder, grad_out, input);

    // Get shapes to verify they match
    const grad_reshaped_type = grad_reshaped.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const grad_reshaped_shape = try grad_reshaped_type.getShape(builder.allocator);
    defer builder.allocator.free(grad_reshaped_shape);

    const input_type_ranked = input.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
    const input_shape = try input_type_ranked.getShape(builder.allocator);
    defer builder.allocator.free(input_shape);

    // If shapes still don't match, reshape to match input shape
    // Note: At this point shapes should only differ by size-1 dimensions
    var grad_shape_matched = grad_reshaped;
    if (grad_reshaped_shape.len != input_shape.len or !std.mem.eql(i64, grad_reshaped_shape, input_shape)) {
        std.debug.print("DEBUG convertVJP: Shapes differ after reduce, reshaping {any} -> {any}\n", .{grad_reshaped_shape, input_shape});
        // Use reshape for shape adjustments (adding/removing size-1 dimensions)
        const grad_reshaped_elem_type = grad_reshaped_type.getElementType();
        const ctx = mlir.Context{ .handle = c.mlirTypeGetContext(grad_reshaped.getType().handle) };
        const reshape_type = mlir.Type.rankedTensorType(ctx, input_shape, grad_reshaped_elem_type);
        const reshape_op = try builder.createAndAttach("stablehlo.reshape", &.{grad_reshaped}, &.{reshape_type}, .{});
        grad_shape_matched = reshape_op.getResult(0);
    }

    // Then convert element type to match input (shapes must match at this point)
    const input_elem_type = input_type_ranked.getElementType();
    const ctx = mlir.Context{ .handle = c.mlirTypeGetContext(input.getType().handle) };
    const target_type = mlir.Type.rankedTensorType(ctx, input_shape, input_elem_type);

    const convert_op = try builder.createAndAttach("stablehlo.convert", &.{grad_shape_matched}, &.{target_type}, .{});

    try result.append(convert_op.getResult(0));
    return result.toOwnedSlice();
}

fn selectVJP(builder: *MLIRBuilder, original_op: mlir.Operation, primals: []const mlir.Value, adjoints: []const mlir.Value) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const pred = primals[0];
    var result = std.ArrayList(mlir.Value).init(builder.allocator);

    const result_type = grad_out.getType();
    const ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const shape = try ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(shape);
    const elem_type = ranked_type.getElementType();

    // Use safe ops.constant helper instead of manual state manipulation
    const zero_tensor = try ops.constant(builder, 0.0, shape, elem_type);
    const zero = zero_tensor.value;

    // Build the select operations for gradients
    // grad_true = select(pred, grad_out, zero)
    const grad_true = try builder.createAndAttach("stablehlo.select", &.{pred, grad_out, zero}, &.{result_type}, .{});

    // grad_false = select(pred, zero, grad_out)
    const grad_false = try builder.createAndAttach("stablehlo.select", &.{pred, zero, grad_out}, &.{result_type}, .{});

    // Create zero gradient for predicate
    const pred_type = pred.getType();
    const pred_ranked_type = pred_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
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

/// Test function for MLIR autodiff
pub fn testMLIRAutoDiff(allocator: Allocator) !void {
    std.debug.print("\n=== Testing MLIR Automatic Differentiation ===\n", .{});
    
    // Create MLIR context for this test
    var ctx = try mlir.Context.init();
    defer ctx.deinit();
    c.mlirContextSetAllowUnregisteredDialects(ctx.handle, true);
    
    // Create MLIR builder
    var builder = try MLIRBuilder.init(allocator, ctx);
    defer builder.deinit();
    
    // Create autodiff system
    _ = AutoDiff.init(allocator, &builder);
    
    std.debug.print("✓ AutoDiff system initialized\n", .{});
    
    // In a real test, we would:
    // 1. Build a forward function with StableHLO ops
    // 2. Call autodiff.grad() to get gradient function
    // 3. Verify the gradient graph is correct
    
    std.debug.print("✓ MLIR autodiff test completed\n", .{});
}