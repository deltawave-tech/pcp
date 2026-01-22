const std = @import("std");
const Allocator = std.mem.Allocator;

// Imports from your project structure
const mlir = @import("../mlir/wrapper.zig");
const ops = @import("../core/ops.zig");
const autodiff = @import("../autodiff/engine.zig");
const adam_mlir = @import("../optimizers/adam_mlir.zig");
const mlir_ctx = @import("../mlir/context.zig");

// Type aliases
const MLIRBuilder = ops.MLIRBuilder;
const AdamMLIR = adam_mlir.AdamMLIR(f32); // Assuming f32 for DiLoCo

pub const GraphBuilder = struct {
    /// Builds the training graph with two entry points for gradient accumulation:
    /// - @compute_gradients: Forward -> Gradient (run N times for micro-batches)
    /// - @apply_optimizer: Apply accumulated gradients to update weights (run once)
    /// Returns the serialized MLIR module bytes.
    pub fn buildTrainingGraph(
        allocator: Allocator,
        builder: *MLIRBuilder,
        forward_mlir_source: []const u8,
        optimizer: *AdamMLIR,
        num_params: usize,
    ) ![]u8 {
        std.debug.print("Compiling training graph via GraphBuilder (gradient accumulation mode)...\n", .{});

        // === PHASE 1: LOAD AND PARSE FORWARD PASS ===
        const temp_module = try mlir.Module.parse(builder.ctx, forward_mlir_source);
        defer temp_module.deinit();

        const forward_fn_to_clone = try temp_module.findFunction("main");

        // === PHASE 2: CLONE INTO MAIN MODULE ===
        const c_api = @import("../mlir/c.zig").c;
        const cloned_forward_fn = mlir.Operation{ .handle = c_api.operationClone(forward_fn_to_clone.handle) };

        const new_fn_name = "model_forward_pass";
        const new_name_attr = mlir.Attribute.stringAttr(builder.ctx, new_fn_name);
        const sym_name_ref = c_api.stringRefFromString("sym_name");
        c_api.operationSetAttributeByName(cloned_forward_fn.handle, sym_name_ref, new_name_attr.handle);

        builder.module_body.appendOwnedOperation(cloned_forward_fn);

        // === PHASE 3: AUTODIFF ===
        std.debug.print("GraphBuilder: Running Autodiff...\n", .{});
        const gradient_clip_min = @as(f64, @floatCast(optimizer.conf.gradient_clip_min));
        const gradient_clip_max = @as(f64, @floatCast(optimizer.conf.gradient_clip_max));
        _ = try autodiff.buildGradientGraph(allocator, builder, cloned_forward_fn, gradient_clip_min, gradient_clip_max);
        const grad_fn_name = "model_forward_pass_grad";

        const forward_fn_type = cloned_forward_fn.getType().as(mlir.FunctionType) orelse return error.NotAFunctionType;
        const f32_type = mlir.Type.f32Type(builder.ctx);

        // === PHASE 4A: BUILD @compute_gradients ===
        // Inputs: [Params(F32)..., Data...]
        // Outputs: [Gradients(F32)..., Loss]
        std.debug.print("GraphBuilder: Building @compute_gradients...\n", .{});
        {
            var grad_input_types = std.ArrayList(mlir.Type).init(allocator);
            defer grad_input_types.deinit();

            // Parameters as F32 master weights
            for (0..num_params) |i| {
                const param_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
                const shape = try param_type.getShape(allocator);
                defer allocator.free(shape);
                const f32_param_type = mlir.Type.rankedTensorType(builder.ctx, shape, f32_type);
                try grad_input_types.append(f32_param_type);
            }

            // Data inputs (remaining inputs from forward function)
            const num_forward_inputs = forward_fn_type.getNumInputs();
            for (num_params..num_forward_inputs) |i| {
                try grad_input_types.append(forward_fn_type.getInput(i));
            }

            // Output types: Gradients (F32) + Loss
            var grad_output_types = std.ArrayList(mlir.Type).init(allocator);
            defer grad_output_types.deinit();

            // Gradients as F32
            for (0..num_params) |i| {
                try grad_output_types.append(grad_input_types.items[i]);
            }

            // Loss in model's original dtype
            try grad_output_types.append(forward_fn_type.getResult(0));

            const grad_func_type = try mlir.Type.functionType(allocator, builder.ctx, grad_input_types.items, grad_output_types.items);
            const grad_func = try builder.createFunction("compute_gradients", grad_func_type);
            builder.setInsertionBlock(grad_func.entry_block);

            const args = try grad_func.entry_block.getArguments(allocator);
            defer allocator.free(args);

            const params_in = args[0..num_params];
            const data_in = args[num_params..];

            // Cast F32 params to model dtype for gradient function
            var grad_call_operands = std.ArrayList(mlir.Value).init(allocator);
            defer grad_call_operands.deinit();

            for (0..num_params) |i| {
                const target_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
                const target_elem_type = target_type.getElementType();

                const p_f32_tensor = try builder.newTensor(params_in[i]);
                const p_model_dtype = try ops.convert(builder, p_f32_tensor, target_elem_type);
                try grad_call_operands.append(p_model_dtype.value);
            }
            try grad_call_operands.appendSlice(data_in);

            // Add loss gradient (1.0)
            const loss_result_type = forward_fn_type.getResult(0);
            const loss_elem_type = loss_result_type.as(mlir.RankedTensorType).?.getElementType();
            const one_tensor = try ops.constant(builder, 1.0, &.{}, loss_elem_type);
            try grad_call_operands.append(one_tensor.value);

            // Result types for grad call
            var grad_call_result_types = std.ArrayList(mlir.Type).init(allocator);
            defer grad_call_result_types.deinit();
            for (0..num_forward_inputs) |i| {
                try grad_call_result_types.append(forward_fn_type.getInput(i));
            }

            const grad_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, grad_fn_name);
            const grad_call_op = try builder.createAndAttach("func.call", grad_call_operands.items, grad_call_result_types.items, .{
                .attributes = &.{.{ "callee", grad_callee_attr }},
            });

            // Call forward pass to get loss
            var fwd_operands = std.ArrayList(mlir.Value).init(allocator);
            defer fwd_operands.deinit();

            for (0..num_params) |i| {
                const target_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
                const target_elem_type = target_type.getElementType();

                const p_f32_tensor = try builder.newTensor(params_in[i]);
                const p_model_dtype = try ops.convert(builder, p_f32_tensor, target_elem_type);
                try fwd_operands.append(p_model_dtype.value);
            }
            try fwd_operands.appendSlice(data_in);

            const fwd_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, new_fn_name);
            const fwd_call_op = try builder.createAndAttach("func.call", fwd_operands.items, &.{forward_fn_type.getResult(0)}, .{
                .attributes = &.{.{ "callee", fwd_callee_attr }},
            });

            // Build return values: [Gradients(F32)..., Loss]
            var returns = std.ArrayList(mlir.Value).init(allocator);
            defer returns.deinit();

            // Convert gradients to F32 for accumulation
            for (0..num_params) |i| {
                const grad_model_dtype = try builder.newTensor(grad_call_op.getResult(i));
                const grad_f32 = try ops.convert(builder, grad_model_dtype, f32_type);
                try returns.append(grad_f32.value);
            }
            try returns.append(fwd_call_op.getResult(0));

            _ = try builder.createAndAttach("func.return", returns.items, &.{}, .{});
        }

        // === PHASE 4B: BUILD @apply_optimizer ===
        // Inputs: [Params(F32)..., Grads(F32)..., M(F32)..., V(F32)..., Timestep(F32)]
        // Outputs: [NewParams(F32)..., NewM(F32)..., NewV(F32)...]
        std.debug.print("GraphBuilder: Building @apply_optimizer...\n", .{});
        {
            var opt_input_types = std.ArrayList(mlir.Type).init(allocator);
            defer opt_input_types.deinit();

            // Params, Grads, M, V (all F32)
            for (0..4) |_| {
                for (0..num_params) |i| {
                    const param_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
                    const shape = try param_type.getShape(allocator);
                    defer allocator.free(shape);
                    try opt_input_types.append(mlir.Type.rankedTensorType(builder.ctx, shape, f32_type));
                }
            }

            // Timestep (scalar F32)
            try opt_input_types.append(mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type));

            // Output types: NewParams, NewM, NewV (all F32)
            var opt_output_types = std.ArrayList(mlir.Type).init(allocator);
            defer opt_output_types.deinit();

            for (0..3) |_| {
                for (0..num_params) |i| {
                    try opt_output_types.append(opt_input_types.items[i]);
                }
            }

            const opt_func_type = try mlir.Type.functionType(allocator, builder.ctx, opt_input_types.items, opt_output_types.items);
            const opt_func = try builder.createFunction("apply_optimizer", opt_func_type);
            builder.setInsertionBlock(opt_func.entry_block);

            const args = try opt_func.entry_block.getArguments(allocator);
            defer allocator.free(args);

            const params_in = args[0..num_params];
            const grads_in = args[num_params .. num_params * 2];
            const m_in = args[num_params * 2 .. num_params * 3];
            const v_in = args[num_params * 3 .. num_params * 4];
            const t_in = args[num_params * 4];

            var returns = std.ArrayList(mlir.Value).init(allocator);
            defer returns.deinit();
            var new_m_list = std.ArrayList(mlir.Value).init(allocator);
            defer new_m_list.deinit();
            var new_v_list = std.ArrayList(mlir.Value).init(allocator);
            defer new_v_list.deinit();

            const t_tensor = try builder.newTensor(t_in);

            for (0..num_params) |i| {
                const p = try builder.newTensor(params_in[i]);
                const g = try builder.newTensor(grads_in[i]);
                const m = try builder.newTensor(m_in[i]);
                const v = try builder.newTensor(v_in[i]);

                const res = try optimizer.update(p, g, m, v, t_tensor);

                try returns.append(res.new_params.value);
                try new_m_list.append(res.new_m.value);
                try new_v_list.append(res.new_v.value);
            }

            try returns.appendSlice(new_m_list.items);
            try returns.appendSlice(new_v_list.items);

            _ = try builder.createAndAttach("func.return", returns.items, &.{}, .{});
        }

        // === PHASE 4C: BUILD @accumulate_gradients ===
        std.debug.print("GraphBuilder: Building @accumulate_gradients...\n", .{});
        {
            var acc_input_types = std.ArrayList(mlir.Type).init(allocator);
            defer acc_input_types.deinit();
            var acc_output_types = std.ArrayList(mlir.Type).init(allocator);
            defer acc_output_types.deinit();

            for (0..num_params) |i| {
                const param_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
                const shape = try param_type.getShape(allocator);
                defer allocator.free(shape);
                const grad_type = mlir.Type.rankedTensorType(builder.ctx, shape, f32_type);
                try acc_input_types.append(grad_type);
                try acc_output_types.append(grad_type);
            }

            for (0..num_params) |i| {
                try acc_input_types.append(acc_output_types.items[i]);
            }

            const acc_func_type = try mlir.Type.functionType(allocator, builder.ctx, acc_input_types.items, acc_output_types.items);
            const acc_func = try builder.createFunction("accumulate_gradients", acc_func_type);
            builder.setInsertionBlock(acc_func.entry_block);

            const args = try acc_func.entry_block.getArguments(allocator);
            defer allocator.free(args);

            var results = std.ArrayList(mlir.Value).init(allocator);
            defer results.deinit();

            for (0..num_params) |i| {
                const acc = try builder.newTensor(args[i]);
                const new_grad = try builder.newTensor(args[num_params + i]);
                const sum = try ops.add(builder, acc, new_grad);
                try results.append(sum.value);
            }

            _ = try builder.createAndAttach("func.return", results.items, &.{}, .{});
        }

        // === PHASE 5: SERIALIZE ===
        if (!builder.module.op().verify()) {
            builder.module.op().dump();
            return error.ModuleVerificationFailed;
        }

        return mlir_ctx.serializeMLIRModule(allocator, builder.module);
    }

    /// Builds a gradient-computation graph for GRPO/RL.
    /// Input: Forward pass MLIR (Params..., Buffers..., Data...) -> Loss
    /// Output: Backward pass MLIR (Params..., Buffers..., Data...) -> (Grads_Params...)
    ///
    /// Unlike buildTrainingGraph (which includes the optimizer in MLIR), this one only
    /// calculates gradients. The optimizer step happens on the Shepherd CPU for GRPO/RL.
    ///
    /// @param allocator: Memory allocator
    /// @param builder: MLIRBuilder with initialized context and module
    /// @param forward_mlir_source: MLIR source text of the forward pass
    /// @param num_params: Number of trainable parameters (gradients returned for these)
    /// @param num_buffers: Number of non-trainable buffers (e.g., rope constants, masks)
    /// @return Serialized MLIR module bytes for the backward pass
    pub fn buildGrpoBackwardPass(
        allocator: Allocator,
        builder: *MLIRBuilder,
        forward_mlir_source: []const u8,
        num_params: usize,
        num_buffers: usize,
    ) ![]u8 {
        const c_api = @import("../mlir/c.zig").c;

        std.debug.print("Compiling GRPO Backward Pass via GraphBuilder...\n", .{});
        std.debug.print("  num_params={}, num_buffers={}\n", .{ num_params, num_buffers });

        // === PHASE 1: LOAD AND PARSE FORWARD PASS ===
        const temp_module = try mlir.Module.parse(builder.ctx, forward_mlir_source);
        defer temp_module.deinit();

        // Find "main" in the temp module
        const forward_fn_to_clone = try temp_module.findFunction("main");

        // === PHASE 2: CLONE INTO MAIN MODULE ===
        const cloned_forward_fn = mlir.Operation{ .handle = c_api.operationClone(forward_fn_to_clone.handle) };

        // Rename to avoid conflict with the new 'main' we will create
        const new_fn_name = "qwen_forward";
        const new_name_attr = mlir.Attribute.stringAttr(builder.ctx, new_fn_name);
        const sym_name_ref = c_api.stringRefFromString("sym_name");
        c_api.operationSetAttributeByName(cloned_forward_fn.handle, sym_name_ref, new_name_attr.handle);

        builder.module_body.appendOwnedOperation(cloned_forward_fn);

        // === PHASE 3: AUTODIFF ===
        std.debug.print("GraphBuilder: Running AutoDiff on Qwen forward pass...\n", .{});
        // GRPO gradients can be large, use reasonable clipping bounds
        const gradient_clip_min: f64 = -100.0;
        const gradient_clip_max: f64 = 100.0;
        _ = try autodiff.buildGradientGraph(allocator, builder, cloned_forward_fn, gradient_clip_min, gradient_clip_max);
        const grad_fn_name = "qwen_forward_grad";

        // === PHASE 4: BUILD ORCHESTRATOR 'main' ===
        std.debug.print("GraphBuilder: Structuring 'main' for GRPO gradient computation...\n", .{});

        const forward_fn_type = cloned_forward_fn.getType().as(mlir.FunctionType) orelse return error.NotAFunctionType;
        const total_inputs = forward_fn_type.getNumInputs();

        // Validate input counts
        const num_data_inputs = total_inputs - num_params - num_buffers;
        std.debug.print("  total_inputs={}, num_data_inputs={}\n", .{ total_inputs, num_data_inputs });

        // 4.1 Define Input Types - same as forward pass
        var main_input_types = std.ArrayList(mlir.Type).init(allocator);
        defer main_input_types.deinit();
        for (0..total_inputs) |i| {
            try main_input_types.append(forward_fn_type.getInput(i));
        }

        // 4.2 Define Output Types - only gradients for trainable parameters
        var main_output_types = std.ArrayList(mlir.Type).init(allocator);
        defer main_output_types.deinit();
        for (0..num_params) |i| {
            try main_output_types.append(forward_fn_type.getInput(i));
        }

        const main_func_type = try mlir.Type.functionType(allocator, builder.ctx, main_input_types.items, main_output_types.items);

        // 4.3 Create main function
        const main_result = try builder.createFunction("main", main_func_type);
        builder.setInsertionBlock(main_result.entry_block);

        // 4.4 Get arguments from main function
        const main_args = try main_result.entry_block.getArguments(allocator);
        defer allocator.free(main_args);

        // 4.5 Prepare Call to Gradient Function
        // Grad function takes: (original inputs..., loss_gradient)
        var grad_operands = std.ArrayList(mlir.Value).init(allocator);
        defer grad_operands.deinit();

        // Add all original inputs (Params + Buffers + Data)
        try grad_operands.appendSlice(main_args);

        // Add incoming gradient of Loss (Scalar 1.0)
        // The forward pass returns a scalar loss
        const loss_result_type = forward_fn_type.getResult(0);
        const loss_ranked_type = loss_result_type.as(mlir.RankedTensorType) orelse return error.LossNotRankedTensor;
        const loss_elem_type = loss_ranked_type.getElementType();
        const loss_shape = try loss_ranked_type.getShape(allocator);
        defer allocator.free(loss_shape);

        const one_tensor = try ops.constant(builder, 1.0, loss_shape, loss_elem_type);
        try grad_operands.append(one_tensor.value);

        // 4.6 Determine result types for gradient call (one gradient per input)
        var grad_call_result_types = std.ArrayList(mlir.Type).init(allocator);
        defer grad_call_result_types.deinit();
        for (main_args) |arg| {
            try grad_call_result_types.append(arg.getType());
        }

        // 4.7 Create the CallOp to gradient function
        const grad_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, grad_fn_name);
        const grad_call_op = try builder.createAndAttach("func.call", grad_operands.items, grad_call_result_types.items, .{
            .attributes = &.{.{ "callee", grad_callee_attr }},
        });

        // 4.8 Return ONLY Parameter Gradients
        // Discard gradients for Buffers and Data inputs
        var return_operands = std.ArrayList(mlir.Value).init(allocator);
        defer return_operands.deinit();

        for (0..num_params) |i| {
            try return_operands.append(grad_call_op.getResult(i));
        }

        _ = try builder.createAndAttach("func.return", return_operands.items, &.{}, .{});

        // === PHASE 5: VERIFY AND SERIALIZE ===
        if (!builder.module.op().verify()) {
            std.log.err("GRPO backward pass module verification failed!", .{});
            builder.module.op().dump();
            return error.ModuleVerificationFailed;
        }

        std.debug.print("✓ GRPO Backward Pass graph built successfully\n", .{});
        return mlir_ctx.serializeMLIRModule(allocator, builder.module);
    }
};
