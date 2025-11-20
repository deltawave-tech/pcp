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

/// Main automatic differentiation function - transforms forward graph to gradient graph
pub fn buildGradientGraph(
    allocator: Allocator,
    builder: *MLIRBuilder,
    forward_fn: mlir.Operation,
) !mlir.Operation {
    std.debug.print("Building gradient graph from forward function...\n", .{});
    
    // Derive the gradient function name from the forward function name
    const forward_fn_name_attr = c.operationGetAttributeByName(forward_fn.handle, "sym_name");
    if (@intFromPtr(forward_fn_name_attr) == 0 or !c.attributeIsAString(forward_fn_name_attr)) {
        return error.MissingOrInvalidSymName;
    }
    const string_attr = @as(*c.MlirStringAttribute, @ptrCast(forward_fn_name_attr));
    const forward_fn_name_ref = c.stringAttributeGetValue(string_attr);
    const forward_fn_name = c.fromStringRef(forward_fn_name_ref);
    const grad_fn_name = try std.fmt.allocPrint(allocator, "{s}_grad", .{forward_fn_name});
    defer allocator.free(grad_fn_name);
    
    // FIXED: Use the existing builder to maintain single context
    // This ensures gradient graph is built in the same context as forward graph
    
    // 1. Create the gradient function within the existing module/context  
    const gradient_fn = try createGradientFunction(builder, forward_fn, grad_fn_name);

    // 2. Create and append the entry block to the gradient function
    const grad_fn_region = gradient_fn.getRegion(0);
    const grad_fn_block = try ops.MLIRBuilder.createBlock();
    c.regionAppendOwnedBlock(grad_fn_region.handle, grad_fn_block.handle);
    
    // CRITICAL FIX: Set the insertion point to the new gradient function's block.
    const original_insertion_block = builder.getInsertionBlock();
    builder.setInsertionBlock(grad_fn_block);
    defer builder.setInsertionBlock(original_insertion_block); // Restore on exit
    
    std.debug.print("Created gradient function block and appended to region\n", .{});
    
    // Map from forward-pass values (primals) to their gradients (adjoints)
    var adjoint_map = std.AutoHashMap(mlir.Value, mlir.Value).init(allocator);
    defer adjoint_map.deinit();

    // NEW: Map from forward-pass values to their corresponding values in the gradient function's scope
    var value_map = std.AutoHashMap(mlir.Value, mlir.Value).init(allocator);
    defer value_map.deinit();
    
    // Get the operations in reverse topological order
    std.debug.print("Getting operations in reverse order...\n", .{});
    const ops_reversed = try getOperationsInReverseOrder(allocator, forward_fn);
    defer allocator.free(ops_reversed);
    std.debug.print("Got {} operations in reverse order\n", .{ops_reversed.len});
    
    // --- START: NEW INITIALIZATION LOGIC ---
    // 1. Map the arguments of the forward function to the arguments of the gradient function
    const forward_block = forward_fn.getRegion(0).getBlock(0);
    const num_forward_args = forward_block.getNumArguments();
    for (0..num_forward_args) |i| {
        const forward_arg = forward_block.getArgument(i);
        const grad_arg = grad_fn_block.addArgument(forward_arg.getType(), builder.loc);
        try value_map.put(forward_arg, grad_arg);
    }

    // 2. The *last* argument of the gradient function is the incoming gradient for the forward function's result
    const loss_value = getReturnValue(forward_fn) orelse return error.NoReturnOperation;
    const loss_grad_arg = grad_fn_block.addArgument(loss_value.getType(), builder.loc);
    try adjoint_map.put(loss_value, loss_grad_arg);
    // --- END: NEW INITIALIZATION LOGIC ---
    
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
    
    // Get output gradients for this operation
    const output_gradients = try getOutputGradients(allocator, op, adjoint_map);
    defer allocator.free(output_gradients);
    
    if (output_gradients.len == 0) {
        // No gradient flows through this operation
        return;
    }

    // Constants are now handled in the first pass, so we can skip them here
    if (std.mem.eql(u8, op_name, "stablehlo.constant")) {
        return;
    }
    
    // Runtime dispatch to the correct VJP rule
    if (getVjpFn(op_name)) |vjp_rule| {
        // NEW: Look up primals in the value_map to get handles valid in the grad scope.
        var mapped_primals = std.ArrayList(mlir.Value).init(allocator);
        defer mapped_primals.deinit();

        for (0..op.getNumOperands()) |i| {
            const primal_operand = op.getOperand(i);
            const mapped_operand = value_map.get(primal_operand) orelse {
                // This is the error we were seeing. It means a value wasn't mapped.
                // This should now be resolved by handling constants correctly.
                std.debug.print("FATAL: Could not find primal value in value_map. This indicates a bug in the reverse walk or constant handling.\n", .{});
                op.dump();
                return error.PrimalValueNotFound;
            };
            try mapped_primals.append(mapped_operand);
        }

        const input_gradients = try vjp_rule(builder, op, mapped_primals.items, output_gradients);
        defer allocator.free(input_gradients);
        
        // Add input gradients to the adjoint map
        try addInputGradients(builder, op, input_gradients, adjoint_map);
        
        // NOTE: Forward operations are now pre-built in the first pass, so we don't need to recreate them here
    } else {
        std.debug.print("Warning: No VJP rule for operation: {s}\n", .{op_name});
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
    _ = primals;
    const grad_out = adjoints[0];

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // Gradient for both inputs is just the output gradient.
    try result.append(grad_out);
    try result.append(grad_out);

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
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();
    
    // da = grad_out
    if (primals.len > 0) {
        try result.append(grad_out);
    }
    
    // db = -grad_out
    if (primals.len > 1) {
        const neg_grad = try builder.createAndAttach("stablehlo.negate", &.{grad_out}, &.{grad_out.getType()}, .{});
        try result.append(neg_grad.getResult(0));
    }
    
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
    
    // da = grad_out * b
    const grad_a = try builder.createAndAttach("stablehlo.multiply", &.{ grad_out, b }, &.{grad_out.getType()}, .{});
    try result.append(grad_a.getResult(0));
    
    // db = grad_out * a  
    const grad_b = try builder.createAndAttach("stablehlo.multiply", &.{ grad_out, a }, &.{grad_out.getType()}, .{});
    try result.append(grad_b.getResult(0));
    
    return result.toOwnedSlice();
}

/// VJP rule for division: da = grad_out / b, db = -grad_out * a / (b * b)
fn divideVJP(
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
    
    // da = grad_out / b
    const grad_a = try builder.createAndAttach("stablehlo.divide", &.{ grad_out, b }, &.{grad_out.getType()}, .{});
    try result.append(grad_a.getResult(0));
    
    // db = -grad_out * a / (b * b)
    // Compute a * grad_out
    const a_times_grad = try builder.createAndAttach("stablehlo.multiply", &.{ a, grad_out }, &.{grad_out.getType()}, .{});
    // Compute b * b
    const b_squared = try builder.createAndAttach("stablehlo.multiply", &.{ b, b }, &.{b.getType()}, .{});
    // Compute (a * grad_out) / (b * b)
    const positive_grad_b = try builder.createAndAttach("stablehlo.divide", &.{ a_times_grad.getResult(0), b_squared.getResult(0) }, &.{grad_out.getType()}, .{});
    // Negate to get -grad_out * a / (b * b)
    const grad_b = try builder.createAndAttach("stablehlo.negate", &.{ positive_grad_b.getResult(0) }, &.{grad_out.getType()}, .{});
    try result.append(grad_b.getResult(0));
    
    return result.toOwnedSlice();
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
fn transposeVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out = adjoints[0];
    const a = primals[0]; // Primal 'a'
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();
    
    // Get the permutation attribute from the original transpose operation
    const permutation_attr = c.operationGetAttributeByName(original_op.handle, "permutation");
        
    if (@intFromPtr(permutation_attr) == 0) {
        return error.MissingPermutationAttribute;
    }
    
    // Get the rank to know how many elements in permutation
    const input_type = a.getType().as(mlir.RankedTensorType).?;
    const rank = input_type.getRank();
    
    // Parse the actual permutation from the DenseI64ArrayAttr
    // The permutation_attr is a DenseI64ArrayAttr containing the permutation indices
    if (!c.attributeIsADenseI64Array(permutation_attr)) {
        return error.InvalidPermutationAttributeType;
    }
    
    const num_elements = c.denseArrayGetNumElements(permutation_attr);
    
    if (num_elements != rank) {
        return error.PermutationRankMismatch;
    }
    
    // Extract the actual permutation from the attribute
    var permutation = try builder.allocator.alloc(i64, rank);
    defer builder.allocator.free(permutation);
    
    for (0..rank) |i| {
        permutation[i] = c.denseI64ArrayGetElement(permutation_attr, @intCast(i));
    }
    
    // Compute inverse permutation: if perm[i] = j, then inv_perm[j] = i
    var inv_permutation = try builder.allocator.alloc(i64, rank);
    defer builder.allocator.free(inv_permutation);
    
    for (permutation, 0..) |target_dim, source_dim| {
        inv_permutation[@intCast(target_dim)] = @intCast(source_dim);
    }
    
    // Create the inverse permutation attribute
    const inv_perm_attr = c.mlirDenseI64ArrayGet(builder.ctx.handle, @intCast(rank), inv_permutation.ptr);
    
    // Create transpose operation with inverse permutation
    var state = c.operationStateGet("stablehlo.transpose", builder.loc.handle);
    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(@constCast(&grad_out.handle)));
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(@constCast(&a.getType().handle)));
    
    // Add the permutation attribute
    const attr_name = c.identifierGet(builder.ctx.handle, "permutation");
    const named_attr = c.MlirNamedAttribute{ .name = attr_name, .attribute = inv_perm_attr };
    c.mlirOperationStateAddAttributes(&state, 1, @ptrCast(@constCast(&named_attr)));
    
    const transpose_op = c.operationCreate(&state);
    builder.insertion_block.appendOwnedOperation(mlir.Operation{ .handle = transpose_op });
    
    try result.append(mlir.Value{ .handle = c.operationGetResult(transpose_op, 0) });
    
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
    if (@intFromPtr(dot_dim_attr) == 0) {
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
    const grad_a_op = hlo.dot_general(builder.ctx, grad_out, b, .{ .dot_dimension_numbers = grad_a_dot_dims });
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
    const grad_b_op = hlo.dot_general(builder.ctx, a, grad_out, .{ .dot_dimension_numbers = grad_b_dot_dims });
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
    _ = builder;
    _ = primals;
    _ = adjoints;
    
    // Constants have no inputs, so no gradients to return
    return &[_]mlir.Value{};
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

/// VJP rule for reduce_sum: da = broadcast(grad_out, original_shape)
fn reduceSumVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    const grad_out = adjoints[0];
    const input = primals[0]; // Original input to reduce_sum
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    if (!isValueFromConstantOp(input)) {
        // Get the original input shape and broadcast the gradient back to it
        const original_shape_type = input.getType();
        const input_tensor_type = original_shape_type.as(mlir.RankedTensorType).?;
        _ = input_tensor_type; // Suppress unused variable warning
        
        // Get the dimensions attribute from the reduce_sum operation
        const dimensions_attr = c.operationGetAttributeByName(original_op.handle, "dimensions");
        
        if (@intFromPtr(dimensions_attr) != 0) {
            // Parse the dimensions that were reduced
            // For DenseI64ArrayAttr, we need manual parsing
            // For now, create broadcast_dimensions manually
            
            // The broadcast_dimensions should contain all dimensions from the original shape
            // that were NOT in the reduction dimensions
            var broadcast_dims = std.ArrayList(i64).init(builder.allocator);
            defer broadcast_dims.deinit();
            
            // Simple heuristic: if grad_out is scalar (rank 0), all dims were reduced
            const grad_out_type = grad_out.getType().as(mlir.RankedTensorType).?;
            const grad_rank = grad_out_type.getRank();
            
            if (grad_rank == 0) {
                // Full reduction to scalar - use broadcast_in_dim with empty broadcast_dimensions
                // For empty array, we need to pass a valid pointer to empty data
                const empty_dims: [0]i64 = .{};
                const empty_broadcast_dims_attr = c.mlirDenseI64ArrayGet(builder.ctx.handle, 0, &empty_dims);
                
                var state = c.operationStateGet("stablehlo.broadcast_in_dim", builder.loc.handle);
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(@constCast(&grad_out.handle)));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(@constCast(&original_shape_type.handle)));
                
                // Add the broadcast_dimensions attribute (empty for scalar broadcast)
                const attr_name = c.identifierGet(builder.ctx.handle, "broadcast_dimensions");
                const named_attr = c.MlirNamedAttribute{ .name = attr_name, .attribute = empty_broadcast_dims_attr };
                c.mlirOperationStateAddAttributes(&state, 1, @ptrCast(@constCast(&named_attr)));
                
                const broadcast_op = c.operationCreate(&state);
                builder.insertion_block.appendOwnedOperation(mlir.Operation{ .handle = broadcast_op });
                
                try result.append(mlir.Value{ .handle = c.operationGetResult(broadcast_op, 0) });
            } else {
                // Partial reduction - compute which dimensions remain
                // For now, assume reduction kept the first grad_rank dimensions
                for (0..@intCast(grad_rank)) |i| {
                    try broadcast_dims.append(@intCast(i));
                }
                
                // Create broadcast_in_dim with proper dimensions
                const broadcast_dims_attr = c.mlirDenseI64ArrayGet(builder.ctx.handle, @intCast(broadcast_dims.items.len), broadcast_dims.items.ptr);
                
                var state = c.operationStateGet("stablehlo.broadcast_in_dim", builder.loc.handle);
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(@constCast(&grad_out.handle)));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(@constCast(&original_shape_type.handle)));
                
                // Add the broadcast_dimensions attribute
                const attr_name = c.identifierGet(builder.ctx.handle, "broadcast_dimensions");
                const named_attr = c.MlirNamedAttribute{ .name = attr_name, .attribute = broadcast_dims_attr };
                c.mlirOperationStateAddAttributes(&state, 1, @ptrCast(@constCast(&named_attr)));
                
                const broadcast_op = c.operationCreate(&state);
                // CRITICAL FIX: Use the builder's pre-established insertion block instead of accessing module regions directly
                builder.insertion_block.appendOwnedOperation(mlir.Operation{ .handle = broadcast_op });
                
                try result.append(mlir.Value{ .handle = c.operationGetResult(broadcast_op, 0) });
            }
        } else {
            // Fallback: assume full reduction and use simple broadcast
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
    _ = original_op;
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
        
        const zero_constant_op = hlo.zeroConstant(builder.ctx, shape, element_type);
        builder.insertion_block.appendOwnedOperation(zero_constant_op);
        const zero_tensor = zero_constant_op;
        
        // Use proper stablehlo.scatter operation with dimension numbers
        // For embedding lookup, we typically scatter back to dimension 0
        const scatter_dim_numbers = hlo.ScatterDimensionNumbersAttribute{
            .update_window_dims = &[_]i64{2}, // n_embd dimension
            .inserted_window_dims = &[_]i64{0}, // vocab_size dimension
            .scatter_dims_to_operand_dims = &[_]i64{0}, // scatter to dimension 0
            .index_vector_dim = 1, // index vector is along dimension 1
        };
        
        const scatter_op = hlo.scatter(builder.ctx, zero_tensor.getResult(0), start_indices, grad_out, scatter_dim_numbers, builder.loc);
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
    
    // Return operation just passes gradients through to its operand
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

        if (@intFromPtr(start_indices_attr) == 0 or @intFromPtr(limit_indices_attr) == 0) {
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
        const zero_scalar_op = hlo.scalarConstant(builder.ctx, 0.0, input_type.getElementType());
        builder.insertion_block.appendOwnedOperation(zero_scalar_op);
        const zero_scalar = zero_scalar_op.getResult(0);

        // Create the hlo.pad operation to expand grad_out back to input shape
        const pad_op = hlo.pad(builder.ctx, grad_out, zero_scalar, start_indices, padding_high, interior_padding, builder.loc);
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
    // Get the forward function's block to inspect its arguments
    const forward_region = forward_fn.getRegion(0);
    const forward_block = forward_region.getBlock(0);
    
    // Get the number of arguments in the forward function
    const num_args = forward_block.getNumArguments();
    
    // Collect input types: forward function args + output gradient
    var input_types = std.ArrayList(mlir.Type).init(builder.allocator);
    defer input_types.deinit();
    
    // Add types for all forward function arguments
    for (0..num_args) |i| {
        const arg = forward_block.getArgument(i);
        try input_types.append(arg.getType());
    }
    
    // Add type for output gradient (same as forward function's return type)
    const return_value = getReturnValue(forward_fn) orelse {
        return error.NoReturnOperation;
    };
    const output_grad_type = return_value.getType();
    try input_types.append(output_grad_type);
    
    // Collect output types: gradients for each forward function argument
    var output_types = std.ArrayList(mlir.Type).init(builder.allocator);
    defer output_types.deinit();
    
    for (0..num_args) |i| {
        const arg = forward_block.getArgument(i);
        try output_types.append(arg.getType());
    }
    
    // Create function type
    const function_type = mlir.Type.functionType(builder.ctx, input_types.items, output_types.items);
    
    // Create the gradient function using the C API directly for proper region setup
    const location = builder.loc;
    var op_state = c.operationStateGet("func.func", location.handle);
    
    // Add attributes
    const sym_name_attr = mlir.Attribute.stringAttr(builder.ctx, name);
    const function_type_attr = mlir.Attribute.typeAttr(function_type);
    
    const sym_name_id = c.identifierGet(builder.ctx.handle, "sym_name");
    const function_type_id = c.identifierGet(builder.ctx.handle, "function_type");
    
    var attributes = [_]c.MlirNamedAttribute{
        c.MlirNamedAttribute{ .name = sym_name_id, .attribute = sym_name_attr.handle },
        c.MlirNamedAttribute{ .name = function_type_id, .attribute = function_type_attr.handle },
    };
    
    c.mlirOperationStateAddAttributes(&op_state, 2, @ptrCast(&attributes));
    
    // Create and add a region for the function body
    const region = c.regionCreate();
    c.operationStateAddOwnedRegions(&op_state, 1, @ptrCast(@constCast(&region)));
    
    // Create the operation
    const grad_fn = mlir.Operation{ .handle = c.operationCreate(&op_state) };
    
    // Add the gradient function to the module
    const module_body = c.moduleGetBody(builder.module.handle);
    c.blockAppendOwnedOperation(module_body, grad_fn.handle);
    
    std.debug.print("Created gradient function with {} inputs and {} outputs\n", .{input_types.items.len, output_types.items.len});
    
    return grad_fn;
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
    // Get gradients for all outputs of this operation
    var gradients = std.ArrayList(mlir.Value).init(allocator);
    
    // Iterate over all results of this operation
    const num_results = op.getNumResults();
    for (0..num_results) |i| {
        const result_value = op.getResult(i);
        
        // Look up this result's gradient in the adjoint_map
        if (adjoint_map.get(result_value)) |gradient| {
            try gradients.append(gradient);
        }
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
        if (adjoint_map.get(operand)) |existing_grad| {
            const sum_op = try builder.createAndAttach("stablehlo.add", &.{ existing_grad, grad }, &.{grad.getType()}, .{});
            try adjoint_map.put(operand, sum_op.getResult(0));
        } else {
            try adjoint_map.put(operand, grad);
        }
    }
}

fn finalizeGradientFunction(
    allocator: Allocator,
    builder: *MLIRBuilder,
    gradient_fn: mlir.Operation,
    forward_fn: mlir.Operation,
    adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value),
) !void {
    // Get the arguments of the original forward function
    const forward_region = forward_fn.getRegion(0);
    const forward_block = forward_region.getBlock(0);
    
    // Get number of arguments in the forward function
    const num_args = forward_block.getNumArguments();
    
    // Collect gradients for function arguments
    var input_gradients = std.ArrayList(mlir.Value).init(allocator);
    defer input_gradients.deinit();
    
    std.debug.print("Finalizing gradient function with {} forward function arguments\n", .{num_args});
    
    // For each argument of the forward function, look up its gradient
    for (0..num_args) |i| {
        const arg = forward_block.getArgument(i);
        
        if (adjoint_map.get(arg)) |gradient| {
            try input_gradients.append(gradient);
            std.debug.print("  Found gradient for argument {}\n", .{i});
        } else {
            // Create a zero-filled tensor for missing gradients
            const arg_type = arg.getType();
            const zero_data = [_]f32{0.0};
            const zero_bytes = std.mem.sliceAsBytes(zero_data[0..]);
            
            // Create a shape for the zero tensor (assuming scalar for now)
            var zero_shape = try tensor.Shape.initWithDims(builder.ctx, &[_]i64{1}, .f32);
            defer zero_shape.deinit();
            
            const zero_gradient = try builder.createConstant(zero_bytes, arg_type, zero_shape);
            try input_gradients.append(zero_gradient);
            std.debug.print("  Created zero gradient for argument {}\n", .{i});
        }
    }
    
    // Get the block of the new gradient function
    // This assumes the new grad_fn has one region and one block
    const grad_fn_block = gradient_fn.getRegion(0).getBlock(0);
    
    // Create the final 'func.return' operation with the collected gradients
    const return_op = mlir.Operation.create(builder.ctx, "func.return", .{
        .operands = input_gradients.items,
        .location = builder.loc,
    });
    
    // Add the return op to the block, making the gradient function complete and valid
    c.blockAppendOwnedOperation(grad_fn_block.handle, return_op.handle);
    
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

    // grad_in = grad_out * exp(a)
    const grad_in = try builder.createAndAttach("stablehlo.multiply", &.{grad_out, exp_a.getResult(0)}, &.{grad_out.getType()}, .{});

    try result.append(grad_in.getResult(0));
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

    // grad_in = grad_out / a
    const grad_in = try builder.createAndAttach("stablehlo.divide", &.{grad_out, a}, &.{grad_out.getType()}, .{});

    try result.append(grad_in.getResult(0));
    return result.toOwnedSlice();
}

/// VJP for rsqrt: da = -0.5 * grad_out * (a ^ -1.5)
fn rsqrtVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const a = primals[0];
    const tensor_type = a.getType();

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    // 1. rsqrt(a)
    const rsqrt_op = try builder.createAndAttach("stablehlo.rsqrt", &.{a}, &.{tensor_type}, .{});
    const rsqrt_val = rsqrt_op.getResult(0);

    // 2. rsqrt(a)^3
    const sq = try builder.createAndAttach("stablehlo.multiply", &.{rsqrt_val, rsqrt_val}, &.{tensor_type}, .{});
    const cub = try builder.createAndAttach("stablehlo.multiply", &.{sq.getResult(0), rsqrt_val}, &.{tensor_type}, .{});

    // 3. Constant -0.5 (Robust Method: Explicit Byte Buffer)
    const ranked_type = tensor_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const shape = try ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(shape);

    var elem_count: usize = 1;
    for (shape) |d| elem_count *= @intCast(d);

    const neg_half_data = try builder.allocator.alloc(f32, elem_count);
    defer builder.allocator.free(neg_half_data);
    @memset(neg_half_data, -0.5);

    const neg_half_bytes = std.mem.sliceAsBytes(neg_half_data);
    const neg_half_attr = mlir.Attribute.denseElementsAttr(builder.ctx, tensor_type, neg_half_bytes);

    const neg_half_op = mlir.Operation.create(builder.ctx, "stablehlo.constant", .{
        .attributes = &.{.{ "value", neg_half_attr }},
        .results = &.{tensor_type},
        .location = builder.loc,
    });
    builder.insertion_block.appendOwnedOperation(neg_half_op);

    // 4. Multiply: grad_out * -0.5 * rsqrt^3
    const term1 = try builder.createAndAttach("stablehlo.multiply", &.{grad_out, neg_half_op.getResult(0)}, &.{tensor_type}, .{});
    const final_grad = try builder.createAndAttach("stablehlo.multiply", &.{term1.getResult(0), cub.getResult(0)}, &.{tensor_type}, .{});

    try result.append(final_grad.getResult(0));
    return result.toOwnedSlice();
}

/// VJP rule for convert: cast gradient back to input type
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

    // Convert gradient back to input type
    const input_type = input.getType();
    const convert_op = try builder.createAndAttach("stablehlo.convert", &.{grad_out}, &.{input_type}, .{});

    try result.append(convert_op.getResult(0));
    return result.toOwnedSlice();
}

fn selectVJP(
    builder: *MLIRBuilder,
    original_op: mlir.Operation,
    primals: []const mlir.Value,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = original_op;
    const grad_out = adjoints[0];
    const pred = primals[0];
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    defer result.deinit();

    const result_type = grad_out.getType();

    // 1. Create zero f32 tensor for the select branches (Standard logic)
    const ranked_type = result_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const shape = try ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(shape);

    var elem_count: usize = 1;
    for (shape) |d| elem_count *= @intCast(d);

    const zero_data = try builder.allocator.alloc(f32, elem_count);
    defer builder.allocator.free(zero_data);
    @memset(zero_data, 0.0);

    const zero_attr = mlir.Attribute.denseElementsAttr(builder.ctx, result_type, std.mem.sliceAsBytes(zero_data));
    const zero_op = mlir.Operation.create(builder.ctx, "stablehlo.constant", .{
        .attributes = &.{.{ "value", zero_attr }},
        .results = &.{result_type},
        .location = builder.loc,
    });
    builder.insertion_block.appendOwnedOperation(zero_op);
    const zero = zero_op.getResult(0);

    // 2. Create zero predicate gradient (i1 type) safely via i8
    // We do this to avoid issues with packing i1 values in raw buffers
    const pred_type = pred.getType();
    const pred_ranked_type = pred_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
    const pred_shape = try pred_ranked_type.getShape(builder.allocator);
    defer builder.allocator.free(pred_shape);

    var pred_elem_count: usize = 1;
    for (pred_shape) |d| pred_elem_count *= @intCast(d);

    // Create i8 type and tensor type
    const i8_mlir_type = c.mlirIntegerTypeGet(builder.ctx.handle, 8);
    const i8_type = mlir.Type{ .handle = i8_mlir_type };
    const i8_tensor_type = mlir.Type.rankedTensorType(builder.ctx, pred_shape, i8_type);

    // Allocate zeroed bytes
    const pred_zero_data = try builder.allocator.alloc(u8, pred_elem_count);
    defer builder.allocator.free(pred_zero_data);
    @memset(pred_zero_data, 0);

    // Create i8 constant
    const pred_zero_attr = mlir.Attribute.denseElementsAttr(builder.ctx, i8_tensor_type, pred_zero_data);
    const pred_zero_op = mlir.Operation.create(builder.ctx, "stablehlo.constant", .{
        .attributes = &.{.{ "value", pred_zero_attr }},
        .results = &.{i8_tensor_type},
        .location = builder.loc,
    });
    builder.insertion_block.appendOwnedOperation(pred_zero_op);

    // Convert i8 zero -> i1 zero
    const pred_zero_cvt = try builder.createAndAttach("stablehlo.convert", &.{pred_zero_op.getResult(0)}, &.{pred_type}, .{});

    // 3. Compute gradients for select
    // grad_true = select(pred, grad_out, 0.0)
    const grad_true = try builder.createAndAttach("stablehlo.select", &.{pred, grad_out, zero}, &.{result_type}, .{});

    // grad_false = select(pred, 0.0, grad_out)
    const grad_false = try builder.createAndAttach("stablehlo.select", &.{pred, zero, grad_out}, &.{result_type}, .{});

    // Gradients: [pred (dummy i1 zero), on_true, on_false]
    try result.append(pred_zero_cvt.getResult(0));
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