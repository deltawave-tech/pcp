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
const VJPFn = *const fn(
    builder: *MLIRBuilder,
    op: mlir.Operation,
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
    } else if (std.mem.eql(u8, op_name, "stablehlo.gather")) {
        return gatherVJP;
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
    
    // FIXED: Use the existing builder to maintain single context
    // This ensures gradient graph is built in the same context as forward graph
    
    // 1. Create the gradient function within the existing module/context  
    const gradient_fn = try createGradientFunction(builder, forward_fn);

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
    
    // Get the operations in reverse topological order
    std.debug.print("Getting operations in reverse order...\n", .{});
    const ops_reversed = try getOperationsInReverseOrder(allocator, forward_fn);
    defer allocator.free(ops_reversed);
    std.debug.print("Got {} operations in reverse order\n", .{ops_reversed.len});
    
    // Initialize gradient of loss (output) to 1.0
    std.debug.print("Getting return value from forward function...\n", .{});
    const loss_value = getReturnValue(forward_fn) orelse {
        std.debug.print("Error: No return operation found in forward function\n", .{});
        return error.NoReturnOperation;
    };
    std.debug.print("Creating ones data...\n", .{});
    const ones_data = [_]f32{1.0};
    const ones_bytes = std.mem.sliceAsBytes(ones_data[0..]);
    std.debug.print("Creating ones shape...\n", .{});
    var ones_shape = try tensor.Shape.initWithDims(builder.ctx, &[_]i64{1}, .f32);
    defer ones_shape.deinit();
    std.debug.print("Creating ones constant...\n", .{});
    const loss_type = loss_value.getType();
    const ones_constant = try builder.createConstant(ones_bytes, loss_type, ones_shape);
    std.debug.print("Putting loss value in adjoint map...\n", .{});
    try adjoint_map.put(loss_value, ones_constant);
    
    std.debug.print("Starting reverse-mode AD through {} operations...\n", .{ops_reversed.len});
    
    // Debug: Print operation names
    for (ops_reversed, 0..) |op, i| {
        const op_name = op.getName();
        std.debug.print("  Operation {}: {s}\n", .{i, op_name});
    }
    
    std.debug.print("Starting VJP processing loop...\n", .{});
    
    // Walk the forward graph backwards, applying VJP rules using the SHARED builder
    for (ops_reversed) |op| {
        std.debug.print("Processing operation VJP for: {s}\n", .{op.getName()});
        try processOperationVJP(allocator, builder, op, &adjoint_map);
    }
    
    // Collect gradients and create return statement using the SHARED builder
    try finalizeGradientFunction(allocator, builder, gradient_fn, forward_fn, &adjoint_map);
    
    std.debug.print("✓ Successfully built gradient graph\n", .{});
    std.debug.print("--- Gradient Graph (now safe to dump!) ---\n", .{});
    gradient_fn.dump();
    
    // Return the gradient function (not the whole module)
    return gradient_fn;
}

/// Process a single operation's VJP rule
fn processOperationVJP(
    allocator: Allocator,
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value),
) !void {
    const op_name = op.getName();
    
    // Get output gradients for this operation
    const output_gradients = try getOutputGradients(allocator, op, adjoint_map);
    defer allocator.free(output_gradients);
    
    if (output_gradients.len == 0) {
        // No gradient flows through this operation
        return;
    }
    
    // Runtime dispatch to the correct VJP rule
    if (getVjpFn(op_name)) |vjp_rule| {
        // Apply the VJP rule to get input gradients
        const input_gradients = try vjp_rule(builder, op, output_gradients);
        defer allocator.free(input_gradients);
        
        // Add input gradients to the adjoint map
        try addInputGradients(builder, op, input_gradients, adjoint_map);
    } else {
        std.debug.print("Warning: No VJP rule for operation: {s}\n", .{op_name});
    }
}

/// VJP rule for addition: both inputs get the same gradient
fn addVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = op; // For now, we don't use the op parameter
    std.debug.assert(adjoints.len == 1);
    const grad_out = adjoints[0];

    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    errdefer result.deinit();

    // Gradient for both inputs is just the output gradient.
    try result.append(grad_out);
    try result.append(grad_out);

    return result.toOwnedSlice();
}

/// VJP rule for subtraction: da = grad_out, db = -grad_out
fn subtractVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0);
    const b = op.getOperand(1);
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    // da = grad_out
    if (!isValueFromConstantOp(a)) {
        try result.append(grad_out);
    }
    
    // db = -grad_out
    if (!isValueFromConstantOp(b)) {
        const neg_grad = try builder.createAndAttach("stablehlo.negate", &.{grad_out}, &.{grad_out.getType()});
        try result.append(neg_grad.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for multiplication: da = grad_out * b, db = grad_out * a
fn multiplyVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0); // Primal 'a'
    const b = op.getOperand(1); // Primal 'b'
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    // da = grad_out * b
    if (!isValueFromConstantOp(a)) {
        const grad_a = try builder.createAndAttach("stablehlo.multiply", &.{ grad_out, b }, &.{grad_out.getType()});
        try result.append(grad_a.getResult(0));
    }
    
    // db = grad_out * a  
    if (!isValueFromConstantOp(b)) {
        const grad_b = try builder.createAndAttach("stablehlo.multiply", &.{ grad_out, a }, &.{grad_out.getType()});
        try result.append(grad_b.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for division: da = grad_out / b, db = -grad_out * a / (b * b)
fn divideVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0); // Primal 'a'
    const b = op.getOperand(1); // Primal 'b'
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    // da = grad_out / b
    if (!isValueFromConstantOp(a)) {
        const grad_a = try builder.createAndAttach("stablehlo.divide", &.{ grad_out, b }, &.{grad_out.getType()});
        try result.append(grad_a.getResult(0));
    }
    
    // db = -grad_out * a / (b * b)
    if (!isValueFromConstantOp(b)) {
        // Compute a * grad_out
        const a_times_grad = try builder.createAndAttach("stablehlo.multiply", &.{ a, grad_out }, &.{grad_out.getType()});
        // Compute b * b
        const b_squared = try builder.createAndAttach("stablehlo.multiply", &.{ b, b }, &.{b.getType()});
        // Compute (a * grad_out) / (b * b)
        const positive_grad_b = try builder.createAndAttach("stablehlo.divide", &.{ a_times_grad.getResult(0), b_squared.getResult(0) }, &.{grad_out.getType()});
        // Negate to get -grad_out * a / (b * b)
        const grad_b = try builder.createAndAttach("stablehlo.negate", &.{ positive_grad_b.getResult(0) }, &.{grad_out.getType()});
        try result.append(grad_b.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for negation: da = -grad_out
fn negateVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0); // Primal 'a'
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    // da = -grad_out
    if (!isValueFromConstantOp(a)) {
        const grad_a = try builder.createAndAttach("stablehlo.negate", &.{grad_out}, &.{grad_out.getType()});
        try result.append(grad_a.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for transpose: da = transpose(grad_out) with inverse permutation
fn transposeVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0); // Primal 'a'
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    if (!isValueFromConstantOp(a)) {
        // Get the permutation attribute from the original transpose operation
        const permutation_attr = c.operationGetAttributeByName(op.handle, "permutation");
        
        if (@intFromPtr(permutation_attr) != 0) {
            // Get the rank to know how many elements in permutation
            const input_type = a.getType().as(mlir.RankedTensorType).?;
            const rank = input_type.getRank();
            
            // For DenseI64ArrayAttr, we need to parse it manually
            // For now, use a simplified approach - assume common cases
            var permutation = try builder.allocator.alloc(i64, rank);
            defer builder.allocator.free(permutation);
            
            // Common case: 2D transpose [1, 0]
            if (rank == 2) {
                permutation[0] = 1;
                permutation[1] = 0;
            } else {
                // General case: reverse all dimensions  
                for (0..rank) |i| {
                    permutation[i] = @intCast(rank - 1 - i);
                }
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
            // CRITICAL FIX: Use the builder's pre-established insertion block instead of accessing module regions directly
            builder.insertion_block.appendOwnedOperation(mlir.Operation{ .handle = transpose_op });
            
            try result.append(mlir.Value{ .handle = c.operationGetResult(transpose_op, 0) });
        } else {
            // Fallback: assume simple 2D transpose [1, 0] -> [1, 0] (self-inverse)
            const grad_a_op = try builder.createAndAttach("stablehlo.transpose", &.{grad_out}, &.{a.getType()});
            try result.append(grad_a_op.getResult(0));
        }
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for matrix multiplication
/// For Y = A @ B, the gradients are:
/// - dA = dY @ B^T
/// - dB = A^T @ dY
fn matmulVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0); // Primal 'a'
    const b = op.getOperand(1); // Primal 'b'
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    _ = a.getType().as(mlir.RankedTensorType).?;
    _ = b.getType().as(mlir.RankedTensorType).?;
    
    // VJP for 'a': grad_out @ transpose(b)
    if (!isValueFromConstantOp(a)) {
        // Create transpose operation for b with permutation [1, 0]
        const b_transposed_op = try builder.createAndAttach("stablehlo.transpose", &.{b}, &.{b.getType()});
        
        // Matmul for dL/dA using dot_general
        const grad_a_op = try builder.createAndAttach("stablehlo.dot_general", &.{ grad_out, b_transposed_op.getResult(0) }, &.{grad_out.getType()});
        try result.append(grad_a_op.getResult(0));
    }
    
    // VJP for 'b': transpose(a) @ grad_out
    if (!isValueFromConstantOp(b)) {
        // Create transpose operation for a with permutation [1, 0]
        const a_transposed_op = try builder.createAndAttach("stablehlo.transpose", &.{a}, &.{a.getType()});
        
        // Matmul for dL/dB using dot_general
        const grad_b_op = try builder.createAndAttach("stablehlo.dot_general", &.{ a_transposed_op.getResult(0), grad_out }, &.{grad_out.getType()});
        try result.append(grad_b_op.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for ReLU (max(x, 0)): gradient flows through only where x > 0
fn reluVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const x = op.getOperand(0); // Input to ReLU
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    if (!isValueFromConstantOp(x)) {
        // Create mask: x > 0
        const zero = try builder.scalarConstant(0.0);
        const mask = try builder.createAndAttach("stablehlo.compare", &.{ x, zero }, &.{x.getType()}); // x > 0
        
        // Apply mask: grad_out * mask
        const grad_x = try builder.createAndAttach("stablehlo.select", &.{ mask.getResult(0), grad_out, zero }, &.{grad_out.getType()});
        try result.append(grad_x.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for constants: no gradient (constants don't have inputs)
fn constantVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = builder;
    _ = op;
    _ = adjoints;
    
    // Constants have no inputs, so no gradients to return
    return &[_]mlir.Value{};
}

/// VJP rule for reshape: da = reshape(grad_out, original_shape)
fn reshapeVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const input = op.getOperand(0); // Original input to reshape
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    if (!isValueFromConstantOp(input)) {
        // Get the original input shape and reshape the gradient back to it
        const original_shape_type = input.getType();
        
        // Create reshape operation to restore original shape
        const grad_input = try builder.createAndAttach("stablehlo.reshape", &.{grad_out}, &.{original_shape_type});
        try result.append(grad_input.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for reduce_sum: da = broadcast(grad_out, original_shape)
fn reduceSumVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const input = op.getOperand(0); // Original input to reduce_sum
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    if (!isValueFromConstantOp(input)) {
        // Get the original input shape and broadcast the gradient back to it
        const original_shape_type = input.getType();
        const input_tensor_type = original_shape_type.as(mlir.RankedTensorType).?;
        _ = input_tensor_type; // Suppress unused variable warning
        
        // Get the dimensions attribute from the reduce_sum operation
        const dimensions_attr = c.operationGetAttributeByName(op.handle, "dimensions");
        
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
                // Full reduction to scalar - broadcast_dimensions is empty for scalar broadcast
                const grad_input = try builder.createAndAttach("stablehlo.broadcast", &.{grad_out}, &.{original_shape_type});
                try result.append(grad_input.getResult(0));
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
            const grad_input = try builder.createAndAttach("stablehlo.broadcast", &.{grad_out}, &.{original_shape_type});
            try result.append(grad_input.getResult(0));
        }
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for gather: da = scatter(grad_out, start_indices, original_shape)
fn gatherVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const operand = op.getOperand(0);      // Original operand (e.g., embedding table)
    const start_indices = op.getOperand(1); // Indices used for gathering
    
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
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = op;
    
    // Return operation just passes gradients through to its operand
    return builder.allocator.dupe(mlir.Value, adjoints);
}

// Helper functions for graph traversal and manipulation


fn createGradientFunction(builder: *MLIRBuilder, forward_fn: mlir.Operation) !mlir.Operation {
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
    const sym_name_attr = mlir.Attribute.stringAttr(builder.ctx, "gradient_function");
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
            const sum_op = try builder.createAndAttach("stablehlo.add", &.{ existing_grad, grad }, &.{grad.getType()});
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