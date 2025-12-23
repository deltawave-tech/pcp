const std = @import("std");
const mlir = @import("../mlir/wrapper.zig");
const ops = @import("../core/ops.zig");
const tensor = @import("../core/tensor.zig");
const c = @import("../mlir/c.zig").c;
const hlo = @import("../mlir/dialects/stablehlo.zig");
const vjp_rules = @import("vjp_rules.zig");

const Allocator = std.mem.Allocator;
const MLIRBuilder = ops.MLIRBuilder;
const VJPFn = vjp_rules.VJPFn;

/// MLIR-based Automatic Differentiation using Graph-to-Graph Transformation
/// This implements reverse-mode AD on MLIR computation graphs using VJP rules

/// Runtime dispatch from operation name to VJP rule
fn getVjpFn(op_name: []const u8) ?VJPFn {
    if (std.mem.eql(u8, op_name, "stablehlo.add")) {
        return vjp_rules.addVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.subtract")) {
        return vjp_rules.subtractVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.multiply")) {
        return vjp_rules.multiplyVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.divide")) {
        return vjp_rules.divideVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.negate")) {
        return vjp_rules.negateVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.transpose")) {
        return vjp_rules.transposeVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.dot_general")) {
        return vjp_rules.matmulVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.maximum")) {
        return vjp_rules.reluVJP; // ReLU implemented as max(x, 0)
    } else if (std.mem.eql(u8, op_name, "stablehlo.constant")) {
        return vjp_rules.constantVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.reshape")) {
        return vjp_rules.reshapeVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.reduce_sum")) {
        return vjp_rules.reduceSumVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.reduce")) {
        return vjp_rules.reduceSumVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.gather")) {
        return vjp_rules.gatherVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.slice")) {
        return vjp_rules.sliceVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.broadcast_in_dim")) {
        return vjp_rules.broadcastInDimVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.exponential")) {
        return vjp_rules.expVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.log")) {
        return vjp_rules.logVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.rsqrt")) {
        return vjp_rules.rsqrtVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.convert")) {
        return vjp_rules.convertVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.select")) {
        return vjp_rules.selectVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.logistic")) {
        return vjp_rules.logisticVJP;
    } else if (std.mem.eql(u8, op_name, "func.return")) {
        return vjp_rules.returnVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.sine")) {
        return vjp_rules.sinVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.cosine")) {
        return vjp_rules.cosVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.concatenate")) {
        return vjp_rules.concatenateVJP;
    } else {
        return null;
    }
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

        // Ensure gradient matches operand shape using reduceGradient
        const grad_matched = try ops.reduceGradient(builder, grad, operand);

        // If a gradient for this operand already exists, add the new one to it.
        if (adjoint_map.get(operand)) |existing_grad| {
            const op_name = op.getName();
            std.debug.print("DEBUG addInputGradients: Adding gradients for operand {} of operation: {s}\n", .{i, op_name});

            // Debug: print shapes
            const operand_type = operand.getType().as(mlir.RankedTensorType);
            if (operand_type) |ot| {
                const operand_shape = try ot.getShape(builder.allocator);
                defer builder.allocator.free(operand_shape);
                std.debug.print("  Operand shape: {any}\n", .{operand_shape});
            }
            const existing_grad_type = existing_grad.getType().as(mlir.RankedTensorType);
            if (existing_grad_type) |egt| {
                const existing_grad_shape = try egt.getShape(builder.allocator);
                defer builder.allocator.free(existing_grad_shape);
                std.debug.print("  Existing grad shape: {any}\n", .{existing_grad_shape});
            }
            const grad_matched_type = grad_matched.getType().as(mlir.RankedTensorType);
            if (grad_matched_type) |gmt| {
                const grad_matched_shape = try gmt.getShape(builder.allocator);
                defer builder.allocator.free(grad_matched_shape);
                std.debug.print("  New grad shape (after reduceGradient): {any}\n", .{grad_matched_shape});
            }

            // Ensure existing gradient matches operand shape
            // Get operand shape
            const operand_ranked_type = operand.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
            const operand_shape_arr = try operand_ranked_type.getShape(builder.allocator);
            defer builder.allocator.free(operand_shape_arr);

            // Broadcast existing_grad to operand shape if needed
            const existing_grad_matched = try ops.broadcastToShape(builder, existing_grad, operand_shape_arr);

            // Both gradients should now match operand shape, so broadcast should work
            const add_broadcast = try ops.broadcastOperands(builder, existing_grad_matched, grad_matched);
            defer builder.allocator.free(add_broadcast.shape);
            const add_type = add_broadcast.lhs.getType();
            const sum_op = try builder.createAndAttach("stablehlo.add", &.{ add_broadcast.lhs, add_broadcast.rhs }, &.{add_type}, .{});
            try adjoint_map.put(operand, sum_op.getResult(0));
        } else {
            try adjoint_map.put(operand, grad_matched);
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
