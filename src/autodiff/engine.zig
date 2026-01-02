const std = @import("std");
const mlir = @import("../mlir/wrapper.zig");
const ops = @import("../core/ops.zig");
const tensor = @import("../core/tensor.zig");
const c = @import("../mlir/c.zig").c;
const hlo = @import("../mlir/dialects/stablehlo.zig");
const vjp_rules = @import("vjp_rules.zig");
const build_options = @import("build_options");

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
    } else if (std.mem.eql(u8, op_name, "stablehlo.power")) {
        return vjp_rules.powerVJP;
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
        // Return null to trigger special handling in processOperationVJP
        return null;
    } else if (std.mem.eql(u8, op_name, "stablehlo.gather")) {
        return vjp_rules.gatherVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.slice")) {
        return vjp_rules.sliceVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.concatenate")) {
        return vjp_rules.concatenateVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.broadcast_in_dim")) {
        return vjp_rules.broadcastInDimVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.exponential")) {
        return vjp_rules.expVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.log")) {
        return vjp_rules.logVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.tanh")) {
        return vjp_rules.tanhVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.rsqrt")) {
        return vjp_rules.rsqrtVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.power")) {
        return vjp_rules.powerVJP;
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
    } else if (std.mem.eql(u8, op_name, "stablehlo.tanh")) {
        return vjp_rules.tanhVJP;
    } else {
        return null;
    }
}

/// Main automatic differentiation function - transforms forward graph to gradient graph
pub fn buildGradientGraph(
    allocator: Allocator,
    builder: *MLIRBuilder,
    forward_fn: mlir.Operation,
    gradient_clip_min: f64,
    gradient_clip_max: f64,
) !mlir.Operation {
    if (build_options.pcp_verbose_logs) {
        std.debug.print("Building gradient graph from forward function...\n", .{});
    }

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

    if (build_options.pcp_verbose_logs) {
        std.debug.print("Using gradient function entry block\n", .{});
    }

    // Map from forward-pass values (primals) to their gradients (adjoints)
    var adjoint_map = std.AutoHashMap(mlir.Value, mlir.Value).init(allocator);
    defer adjoint_map.deinit();

    // Map from forward-pass values to their corresponding values in the gradient function's scope
    var value_map = std.AutoHashMap(mlir.Value, mlir.Value).init(allocator);
    defer value_map.deinit();

    const forward_block = forward_fn.getRegion(0).getBlock(0);
    const num_forward_args = forward_block.getNumArguments();

    for (0..num_forward_args) |i| {
        const forward_arg = forward_block.getArgument(i);
        const grad_arg = grad_fn_block.getArgument(i);
        try value_map.put(forward_arg, grad_arg);
    }

    const loss_value = getReturnValue(forward_fn) orelse return error.NoReturnOperation;
    const loss_grad_arg = grad_fn_block.getArgument(num_forward_args);
    try adjoint_map.put(loss_value, loss_grad_arg);

    // First pass: Clone forward operations into gradient function
    if (build_options.pcp_verbose_logs) {
        std.debug.print("First pass: Building primal value mappings in forward order...\n", .{});
    }
    const ops_forward = try getOperationsInForwardOrder(allocator, forward_fn);
    defer allocator.free(ops_forward);

    for (ops_forward) |op| {
        const op_name = op.getName();

        if (std.mem.eql(u8, op_name, "stablehlo.constant")) {
            const cloned_op = mlir.Operation{ .handle = c.operationClone(op.handle) };
            builder.insertion_block.appendOwnedOperation(cloned_op);
            try value_map.put(op.getResult(0), cloned_op.getResult(0));
        } else if (!std.mem.eql(u8, op_name, "func.return")) {
            const cloned_op = mlir.Operation{ .handle = c.operationClone(op.handle) };

            for (0..cloned_op.getNumOperands()) |i| {
                const primal_operand = op.getOperand(i);
                if (value_map.get(primal_operand)) |mapped_operand| {
                    c.mlirOperationSetOperand(cloned_op.handle, @intCast(i), mapped_operand.handle);
                }
            }

            builder.insertion_block.appendOwnedOperation(cloned_op);

            for (0..op.getNumResults()) |i| {
                try value_map.put(op.getResult(i), cloned_op.getResult(i));
            }
        }
    }

    // Second pass: Compute gradients using use-counting (reverse Kahn's algorithm)
    if (build_options.pcp_verbose_logs) {
        std.debug.print("Second pass: Computing gradients using use-counting...\n", .{});
    }
    try processGradientsWithUseCounting(allocator, builder, forward_fn, &adjoint_map, &value_map, gradient_clip_min, gradient_clip_max);

    // Collect gradients and create return statement using the SHARED builder
    try finalizeGradientFunction(allocator, builder, gradient_fn, forward_fn, &adjoint_map, &value_map);

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
    gradient_clip_min: f64,
    gradient_clip_max: f64,
) !void {
    const op_name = op.getName();
    // Get gradients (now safely handles empty case)
    const output_gradients = try getOutputGradients(allocator, op, adjoint_map);

    // Always free the slice. Since getOutputGradients now guarantees a safe slice (even if len 0),
    // this allocator.free call is safe.
    defer allocator.free(output_gradients);

    if (output_gradients.len == 0) return;
    if (std.mem.eql(u8, op_name, "stablehlo.constant")) return;

    // Special handling for stablehlo.reduce - inspect the body to determine VJP rule
    var vjp_rule: ?VJPFn = getVjpFn(op_name);
    if (vjp_rule == null and std.mem.eql(u8, op_name, "stablehlo.reduce")) {
        // Inspect the reduction region to decide VJP rule
        const region = op.getRegion(0);
        const block = region.getBlock(0);

        var is_max = false;
        var maybe_op = block.getFirstOp();

        // Iterate through ops in the reduce body (usually just one math op and a return)
        while (maybe_op) |body_op| {
            const name = body_op.getName();
            if (std.mem.eql(u8, name, "stablehlo.maximum")) {
                is_max = true;
                break;
            }
            maybe_op = body_op.getNext();
        }

        if (is_max) {
            vjp_rule = vjp_rules.reduceMaxVJP;
        } else {
            // Default to Sum for stablehlo.add or complex reductions (safe fallback for now)
            vjp_rule = vjp_rules.reduceSumVJP;
        }
    }

    if (vjp_rule) |rule| {
        var mapped_primals = std.ArrayList(mlir.Value).init(allocator);
        defer mapped_primals.deinit();

        for (0..op.getNumOperands()) |i| {
            const primal_operand = op.getOperand(i);
            const mapped_operand = value_map.get(primal_operand) orelse return error.PrimalValueNotFound;
            try mapped_primals.append(mapped_operand);
        }

        const input_gradients = try rule(builder, op, mapped_primals.items, output_gradients);
        defer builder.allocator.free(input_gradients); // Ensure VJP result slice is freed

        try addInputGradients(builder, op, input_gradients, adjoint_map, gradient_clip_min, gradient_clip_max);
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
    if (build_options.pcp_verbose_logs) {
        std.debug.print("Created gradient function '{s}' via builder\n", .{name});
    }
    return result.func_op;
}

/// Process gradients using use-counting (reverse Kahn's algorithm)
/// Ensures operations are processed only after ALL consumers have contributed gradients
fn processGradientsWithUseCounting(
    allocator: Allocator,
    builder: *MLIRBuilder,
    forward_fn: mlir.Operation,
    adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value),
    value_map: *std.AutoHashMap(mlir.Value, mlir.Value),
    gradient_clip_min: f64,
    gradient_clip_max: f64,
) !void {
    const forward_block = forward_fn.getRegion(0).getBlock(0);

    // Step 1: Count how many times each operation is used as a producer
    var use_counts = std.AutoHashMap(mlir.Operation, usize).init(allocator);
    defer use_counts.deinit();

    var maybe_op = forward_block.getFirstOp();
    while (maybe_op) |op| {
        for (0..op.getNumOperands()) |i| {
            const operand = op.getOperand(i);
            if (c.mlirValueIsAOpResult(operand.handle)) {
                const producer_op = mlir.Operation{ .handle = c.mlirOpResultGetOwner(operand.handle) };
                const entry = try use_counts.getOrPut(producer_op);
                if (!entry.found_existing) entry.value_ptr.* = 0;
                entry.value_ptr.* += 1;
            }
        }
        maybe_op = op.getNext();
    }

    // Step 2: Initialize ready queue with the terminator
    const terminator = forward_block.getLastOpGeneric() orelse return error.NoTerminator;
    const op_name = terminator.getName();
    if (!std.mem.eql(u8, op_name, "func.return")) return error.InvalidTerminator;

    var ready_queue = std.ArrayList(mlir.Operation).init(allocator);
    defer ready_queue.deinit();
    try ready_queue.append(terminator);

    // Step 3: Process operations in reverse order using use-counting
    while (ready_queue.items.len > 0) {
        const current_op = ready_queue.orderedRemove(0);

        if (build_options.pcp_verbose_logs) {
            std.debug.print("Processing operation VJP for: {s}\n", .{current_op.getName()});
        }
        try processOperationVJP(allocator, builder, current_op, adjoint_map, value_map, gradient_clip_min, gradient_clip_max);

        // Decrement use counts for producers and add to queue when ready
        for (0..current_op.getNumOperands()) |i| {
            const operand = current_op.getOperand(i);
            if (c.mlirValueIsAOpResult(operand.handle)) {
                const producer_op = mlir.Operation{ .handle = c.mlirOpResultGetOwner(operand.handle) };

                if (use_counts.getPtr(producer_op)) |count| {
                    count.* -= 1;
                    if (count.* == 0) {
                        try ready_queue.append(producer_op);
                    }
                }
            }
        }
    }
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
    if (build_options.pcp_verbose_logs) {
        std.debug.print("getReturnValue: Falling back to full iteration to find terminator.\n", .{});
    }
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
    gradient_clip_min: f64,
    gradient_clip_max: f64,
) !void {
    const num_operands = op.getNumOperands();

    for (0..num_operands) |i| {
        if (i >= input_gradients.len) break;

        const operand = op.getOperand(i);
        const grad = input_gradients[i];

        const grad_matched = try ops.reduceGradient(builder, grad, operand);

        const grad_type = grad_matched.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
        const elem_type = grad_type.getElementType();

        var safe_grad = grad_matched;

        if (!elem_type.isInteger() and !elem_type.isIndex()) {
            const grad_shape = try grad_type.getShape(builder.allocator);
            defer builder.allocator.free(grad_shape);

            const grad_tensor = try builder.newTensor(grad_matched);
            const f32_type = mlir.Type.f32Type(builder.ctx);

            const grad_f32 = if (!elem_type.isEqual(f32_type)) blk: {
                const target_type = mlir.Type.rankedTensorType(builder.ctx, grad_shape, f32_type);
                const convert_op = try builder.createAndAttach("stablehlo.convert", &.{grad_matched}, &.{target_type}, .{});
                break :blk try builder.newTensor(convert_op.getResult(0));
            } else grad_tensor;

            const min_val_f32 = try ops.constant(builder, gradient_clip_min, grad_shape, f32_type);
            const max_val_f32 = try ops.constant(builder, gradient_clip_max, grad_shape, f32_type);

            const clamped_1 = try ops.maximum(builder, grad_f32, min_val_f32);
            const clamped_final = try ops.minimum(builder, clamped_1, max_val_f32);

            safe_grad = if (!elem_type.isEqual(f32_type)) blk: {
                const target_type = mlir.Type.rankedTensorType(builder.ctx, grad_shape, elem_type);
                const convert_op = try builder.createAndAttach("stablehlo.convert", &.{clamped_final.value}, &.{target_type}, .{});
                break :blk convert_op.getResult(0);
            } else clamped_final.value;
        }

        if (adjoint_map.get(operand)) |existing_grad| {
            const operand_ranked_type = operand.getType().as(mlir.RankedTensorType) orelse return error.NotRankedTensor;
            const operand_shape_arr = try operand_ranked_type.getShape(builder.allocator);
            defer builder.allocator.free(operand_shape_arr);

            const existing_grad_matched = try ops.broadcastToShape(builder, existing_grad, operand_shape_arr);

            const add_broadcast = try ops.broadcastOperands(builder, existing_grad_matched, safe_grad);
            defer builder.allocator.free(add_broadcast.shape);

            const add_type = add_broadcast.lhs.getType();
            const sum_op = try builder.createAndAttach("stablehlo.add", &.{ add_broadcast.lhs, add_broadcast.rhs }, &.{add_type}, .{});

            try adjoint_map.put(operand, sum_op.getResult(0));
        } else {
            try adjoint_map.put(operand, safe_grad);
        }
    }
}

fn finalizeGradientFunction(allocator: Allocator, builder: *MLIRBuilder, gradient_fn: mlir.Operation, forward_fn: mlir.Operation, adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value), value_map: *std.AutoHashMap(mlir.Value, mlir.Value)) !void {
    _ = gradient_fn;
    const forward_region = forward_fn.getRegion(0);
    const forward_block = forward_region.getBlock(0);
    const num_args = forward_block.getNumArguments();
    var input_gradients = std.ArrayList(mlir.Value).init(allocator);
    defer input_gradients.deinit();

    if (build_options.pcp_verbose_logs) {
        std.debug.print("\n=== FINALIZING GRADIENT FUNCTION ===\n", .{});
        std.debug.print("Finalizing gradient function with {} forward function arguments\n", .{num_args});
        std.debug.print("Adjoint map has {} entries\n", .{adjoint_map.count()});

        var adj_iter = adjoint_map.iterator();
        var adj_count: usize = 0;
        while (adj_iter.next()) |entry| : (adj_count += 1) {
            if (adj_count < 5) {
                const val_ptr = @intFromPtr(entry.key_ptr.*.handle.ptr);
                std.debug.print("  Adjoint map entry {}: value ptr={}\n", .{ adj_count, val_ptr });
            }
        }
        if (adjoint_map.count() > 5) {
            std.debug.print("  ... and {} more entries\n", .{adjoint_map.count() - 5});
        }

        std.debug.print("\nLooking for gradients for forward function arguments:\n", .{});
    }

    for (0..num_args) |i| {
        const arg = forward_block.getArgument(i);
        if (build_options.pcp_verbose_logs and i < 5) {
            const arg_ptr = @intFromPtr(arg.handle.ptr);
            std.debug.print("  Forward arg {}: value ptr={}\n", .{ i, arg_ptr });
        }
        if (build_options.pcp_verbose_logs) {
            if (value_map.get(arg)) |mapped_arg| {
                if (i < 5) {
                    const mapped_arg_ptr = @intFromPtr(mapped_arg.handle.ptr);
                    std.debug.print("  Mapped arg {}: value ptr={}\n", .{ i, mapped_arg_ptr });
                }
            } else {
                std.debug.print("  WARNING: No mapped value for argument {}\n", .{i});
            }
        }

        if (value_map.get(arg) == null) {
            const arg_type = arg.getType();
            const ranked_type = arg_type.as(mlir.RankedTensorType) orelse return error.InvalidTensorType;
            const shape = try ranked_type.getShape(allocator);
            defer allocator.free(shape);
            const elem_type = ranked_type.getElementType();
            const zero_tensor = try ops.constant(builder, 0.0, shape, elem_type);
            try input_gradients.append(zero_tensor.value);
            if (build_options.pcp_verbose_logs) {
                std.debug.print("  Created ZERO gradient for argument {}\n", .{i});
            }
            continue;
        }

        if (adjoint_map.get(arg)) |gradient| {
            try input_gradients.append(gradient);
            if (build_options.pcp_verbose_logs and i < 5) {
                const grad_type = gradient.getType().as(mlir.RankedTensorType) orelse {
                    std.debug.print("  Arg {} gradient: NOT A RANKED TENSOR\n", .{i});
                    continue;
                };
                const grad_shape = try grad_type.getShape(allocator);
                defer allocator.free(grad_shape);
                std.debug.print("  Arg {} gradient shape: {any}\n", .{ i, grad_shape });
            }
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
            if (build_options.pcp_verbose_logs) {
                std.debug.print("  Created ZERO gradient for argument {}\n", .{i});
            }
        }
    }

    // Create return operation using safe builder
    // func.return takes the input_gradients as operands
    _ = try builder.createAndAttach("func.return", input_gradients.items, &.{}, .{});
    if (build_options.pcp_verbose_logs) {
        std.debug.print("Added func.return to gradient function with {} gradients\n", .{input_gradients.items.len});
    }
}
