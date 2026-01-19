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
    /// Builds the complete training graph: Forward -> Gradient -> Optimizer Update
    /// Returns the serialized MLIR module bytes.
    pub fn buildTrainingGraph(
        allocator: Allocator,
        builder: *MLIRBuilder,
        forward_mlir_source: []const u8,
        optimizer: *AdamMLIR,
        num_params: usize, // Must be passed in, derived from introspection in Init
    ) ![]u8 {
        std.debug.print("Compiling training graph via GraphBuilder...\n", .{});

        // === PHASE 1: LOAD AND PARSE FORWARD PASS ===
        // Parse the text file into a temporary module to extract the function
        const temp_module = try mlir.Module.parse(builder.ctx, forward_mlir_source);
        defer temp_module.deinit();

        // Find "main" in the temp module using the helper from Step 2C
        const forward_fn_to_clone = try temp_module.findFunction("main");

        // === PHASE 2: CLONE INTO MAIN MODULE ===
        const c_api = @import("../mlir/c.zig").c;
        const cloned_forward_fn = mlir.Operation{ .handle = c_api.operationClone(forward_fn_to_clone.handle) };

        // Rename to avoid conflict with the new 'main' we will create
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

        // === PHASE 4: BUILD ORCHESTRATOR 'main' ===
        std.debug.print("GraphBuilder: Structuring 'main' with Optimizer...\n", .{});

        const forward_fn_type = cloned_forward_fn.getType().as(mlir.FunctionType) orelse return error.NotAFunctionType;
        const num_forward_inputs = forward_fn_type.getNumInputs();

        // 4.1 Define Types (Mixed Precision with F32 Master Weights)
        // CRITICAL: Parameters are now F32 "master weights" to prevent precision loss.
        // The gradient function still expects the model's original dtype (bf16), so we cast internally.
        var main_input_types = std.ArrayList(mlir.Type).init(allocator);
        defer main_input_types.deinit();

        const f32_type = mlir.Type.f32Type(builder.ctx);

        // A. Parameters (N) - FORCE F32 for master weights (prevents small update truncation)
        for (0..num_params) |i| {
            const param_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
            const shape = try param_type.getShape(allocator);
            defer allocator.free(shape);
            const f32_param_type = mlir.Type.rankedTensorType(builder.ctx, shape, f32_type);
            try main_input_types.append(f32_param_type);
        }

        // B. M States (N) - FORCE F32 for mixed precision
        for (0..num_params) |i| {
            const param_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
            const shape = try param_type.getShape(allocator);
            defer allocator.free(shape);
            const state_type = mlir.Type.rankedTensorType(builder.ctx, shape, f32_type);
            try main_input_types.append(state_type);
        }

        // C. V States (N) - FORCE F32 for mixed precision
        for (0..num_params) |i| {
            const param_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
            const shape = try param_type.getShape(allocator);
            defer allocator.free(shape);
            const state_type = mlir.Type.rankedTensorType(builder.ctx, shape, f32_type);
            try main_input_types.append(state_type);
        }

        // D. Timestep (Scalar f32)
        const scalar_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type);
        try main_input_types.append(scalar_type);

        // E. Data Inputs (Remaining)
        for (num_params..num_forward_inputs) |i| try main_input_types.append(forward_fn_type.getInput(i));

        // Define Outputs: [NewParams, NewM, NewV, Loss]
        var main_output_types = std.ArrayList(mlir.Type).init(allocator);
        defer main_output_types.deinit();

        // New Params - FORCE F32 for master weights (use types from step A)
        for (0..num_params) |i| try main_output_types.append(main_input_types.items[i]);

        // New M - FORCE F32 (use types from step B)
        for (num_params..num_params * 2) |i| try main_output_types.append(main_input_types.items[i]);

        // New V - FORCE F32 (use types from step C)
        for (num_params * 2..num_params * 3) |i| try main_output_types.append(main_input_types.items[i]);

        // Loss
        try main_output_types.append(forward_fn_type.getResult(0));

        const main_func_type = try mlir.Type.functionType(allocator, builder.ctx, main_input_types.items, main_output_types.items);

        // 4.2 Create Function
        const main_result = try builder.createFunction("main", main_func_type);
        const main_block = main_result.entry_block;
        builder.setInsertionBlock(main_block);

        // 4.3 Slice Arguments
        const main_func_args = try main_block.getArguments(allocator);
        defer allocator.free(main_func_args);

        const params_in = main_func_args[0..num_params];
        const m_in = main_func_args[num_params .. num_params * 2];
        const v_in = main_func_args[num_params * 2 .. num_params * 3];
        const timestep_val = main_func_args[num_params * 3];
        const data_in = main_func_args[num_params * 3 + 1 ..];

        // 4.4 Calculate Gradients
        // CRITICAL: Gradient function expects model's original dtype (bf16), but params_in are F32.
        // Cast F32 params -> model dtype before calling gradient function.
        var grad_operands = std.ArrayList(mlir.Value).init(allocator);
        defer grad_operands.deinit();

        // Cast F32 params to model's original dtype for gradient function
        for (0..num_params) |i| {
            const p_f32 = params_in[i];
            const target_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
            const target_elem_type = target_type.getElementType();

            const p_f32_tensor = try builder.newTensor(p_f32);
            const p_model_dtype = try ops.convert(builder, p_f32_tensor, target_elem_type);
            try grad_operands.append(p_model_dtype.value);
        }
        try grad_operands.appendSlice(data_in);

        // Helper to create 1.0 constant for loss gradient
        const loss_result_type = forward_fn_type.getResult(0); // Loss tensor type
        const loss_elem_type = loss_result_type.as(mlir.RankedTensorType).?.getElementType();
        const one_tensor = try ops.constant(builder, 1.0, &.{}, loss_elem_type);
        try grad_operands.append(one_tensor.value);

        // Prepare result types for grad call (uses model's original dtype, NOT F32)
        var grad_result_types = std.ArrayList(mlir.Type).init(allocator);
        defer grad_result_types.deinit();
        for (0..num_params) |i| try grad_result_types.append(forward_fn_type.getInput(i));
        for (data_in) |d| try grad_result_types.append(d.getType());

        const grad_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, grad_fn_name);
        const grad_call_op = try builder.createAndAttach("func.call", grad_operands.items, grad_result_types.items, .{
            .attributes = &.{.{ "callee", grad_callee_attr }},
        });

        // 4.5 Optimizer Step
        var final_returns = std.ArrayList(mlir.Value).init(allocator);
        defer final_returns.deinit();

        std.debug.print("DEBUG: Creating timestep_tensor...\n", .{});
        const timestep_tensor = try builder.newTensor(timestep_val);
        std.debug.print("DEBUG: timestep_tensor created OK\n", .{});
        var new_params = std.ArrayList(mlir.Value).init(allocator);
        var new_ms = std.ArrayList(mlir.Value).init(allocator);
        var new_vs = std.ArrayList(mlir.Value).init(allocator);
        defer new_params.deinit();
        defer new_ms.deinit();
        defer new_vs.deinit();

        for (0..num_params) |i| {
            std.debug.print("DEBUG: Optimizer loop iteration {}/{}\n", .{ i, num_params });
            std.debug.print("DEBUG: Creating param_t...\n", .{});
            const param_t = try builder.newTensor(params_in[i]);
            std.debug.print("DEBUG: Creating grad_t...\n", .{});
            const grad_t = try builder.newTensor(grad_call_op.getResult(i));
            std.debug.print("DEBUG: Creating m_t...\n", .{});
            const m_t = try builder.newTensor(m_in[i]);
            std.log.info("Creating v_t... v_in.len={}, i={}", .{ v_in.len, i });
            // Debug: check if the value's type is a RankedTensorType
            const v_type = v_in[i].getType();
            const v_is_ranked = v_type.as(mlir.RankedTensorType) != null;
            std.log.info("v_in[{}] type is ranked tensor: {}", .{ i, v_is_ranked });
            const v_t = try builder.newTensor(v_in[i]);
            std.log.info("v_t created successfully, calling optimizer.update", .{});

            const update_res = try optimizer.update(param_t, grad_t, m_t, v_t, timestep_tensor);
            std.log.info("optimizer.update completed for param {}", .{i});

            try new_params.append(update_res.new_params.value);
            try new_ms.append(update_res.new_m.value);
            try new_vs.append(update_res.new_v.value);
        }

        // 4.6 Calculate Forward Loss (for reporting)
        // We re-call forward just to get the loss value cleanly, or we could have returned it from grad
        // To save compute, usually grad returns the loss too, but standard VJP here doesn't.
        // We will call the forward pass function.
        // NOTE: Forward function expects model's original dtype, so we must cast F32 params.
        var fwd_operands = std.ArrayList(mlir.Value).init(allocator);
        defer fwd_operands.deinit();

        // Cast updated F32 params to model dtype for forward call
        for (0..num_params) |i| {
            const p_f32 = new_params.items[i];
            const target_type = forward_fn_type.getInput(i).as(mlir.RankedTensorType).?;
            const target_elem_type = target_type.getElementType();

            const p_f32_tensor = try builder.newTensor(p_f32);
            const p_model_dtype = try ops.convert(builder, p_f32_tensor, target_elem_type);
            try fwd_operands.append(p_model_dtype.value);
        }
        try fwd_operands.appendSlice(data_in);

        const fwd_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, new_fn_name);
        const fwd_call_op = try builder.createAndAttach("func.call", fwd_operands.items, &.{forward_fn_type.getResult(0)}, .{
            .attributes = &.{.{ "callee", fwd_callee_attr }},
        });

        // 4.7 Return
        try final_returns.appendSlice(new_params.items);
        try final_returns.appendSlice(new_ms.items);
        try final_returns.appendSlice(new_vs.items);
        try final_returns.append(fwd_call_op.getResult(0)); // The loss

        _ = try builder.createAndAttach("func.return", final_returns.items, &.{}, .{});

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
