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

        // 4.1 Define Types
        var main_input_types = std.ArrayList(mlir.Type).init(allocator);
        defer main_input_types.deinit();

        // A. Parameters (N)
        for (0..num_params) |i| try main_input_types.append(forward_fn_type.getInput(i));
        // B. M States (N) - Same shape as params
        for (0..num_params) |i| try main_input_types.append(forward_fn_type.getInput(i));
        // C. V States (N) - Same shape as params
        for (0..num_params) |i| try main_input_types.append(forward_fn_type.getInput(i));
        // D. Timestep (Scalar)
        const element_type = mlir.Type.f32Type(builder.ctx); // Assuming f32
        const scalar_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, element_type);
        try main_input_types.append(scalar_type);
        // E. Data Inputs (Remaining)
        for (num_params..num_forward_inputs) |i| try main_input_types.append(forward_fn_type.getInput(i));

        // Define Outputs: [NewParams, NewM, NewV, Loss]
        var main_output_types = std.ArrayList(mlir.Type).init(allocator);
        defer main_output_types.deinit();
        for (0..num_params) |i| try main_output_types.append(forward_fn_type.getInput(i)); // New Params
        for (0..num_params) |i| try main_output_types.append(forward_fn_type.getInput(i)); // New M
        for (0..num_params) |i| try main_output_types.append(forward_fn_type.getInput(i)); // New V
        try main_output_types.append(forward_fn_type.getResult(0)); // Loss

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
        // Prepare inputs for Grad Call: Params + Data + LossGrad(1.0)
        var grad_operands = std.ArrayList(mlir.Value).init(allocator);
        defer grad_operands.deinit();
        try grad_operands.appendSlice(params_in);
        try grad_operands.appendSlice(data_in);

        // Helper to create 1.0 constant
        const loss_result_type = forward_fn_type.getResult(0); // Loss tensor type
        const loss_elem_type = loss_result_type.as(mlir.RankedTensorType).?.getElementType();
        const one_tensor = try ops.constant(builder, 1.0, &.{}, loss_elem_type);
        try grad_operands.append(one_tensor.value);

        // Prepare result types for grad call
        var grad_result_types = std.ArrayList(mlir.Type).init(allocator);
        defer grad_result_types.deinit();
        for (params_in) |p| try grad_result_types.append(p.getType());
        for (data_in) |d| try grad_result_types.append(d.getType());

        const grad_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, grad_fn_name);
        const grad_call_op = try builder.createAndAttach("func.call", grad_operands.items, grad_result_types.items, .{
            .attributes = &.{.{ "callee", grad_callee_attr }},
        });

        // 4.5 Optimizer Step
        var final_returns = std.ArrayList(mlir.Value).init(allocator);
        defer final_returns.deinit();

        const timestep_tensor = try builder.newTensor(timestep_val);
        var new_params = std.ArrayList(mlir.Value).init(allocator);
        var new_ms = std.ArrayList(mlir.Value).init(allocator);
        var new_vs = std.ArrayList(mlir.Value).init(allocator);
        defer new_params.deinit();
        defer new_ms.deinit();
        defer new_vs.deinit();

        for (0..num_params) |i| {
            const param_t = try builder.newTensor(params_in[i]);
            const grad_t = try builder.newTensor(grad_call_op.getResult(i));
            const m_t = try builder.newTensor(m_in[i]);
            const v_t = try builder.newTensor(v_in[i]);

            const update_res = try optimizer.update(param_t, grad_t, m_t, v_t, timestep_tensor);

            try new_params.append(update_res.new_params.value);
            try new_ms.append(update_res.new_m.value);
            try new_vs.append(update_res.new_v.value);
        }

        // 4.6 Calculate Forward Loss (for reporting)
        // We re-call forward just to get the loss value cleanly, or we could have returned it from grad
        // To save compute, usually grad returns the loss too, but standard VJP here doesn't.
        // We will call the forward pass function.
        var fwd_operands = std.ArrayList(mlir.Value).init(allocator);
        defer fwd_operands.deinit();
        try fwd_operands.appendSlice(params_in);
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
};
