// In src/examples/gpt2_model_test.zig
const std = @import("std");
const pcp = @import("pcp");

const MLIRContext = pcp.mlir_ctx.MLIRContext;
const MLIRBuilder = pcp.ops.MLIRBuilder;
const GPT2Config = pcp.models.gpt2.GPT2Config;
const getParameterShapes = pcp.models.gpt2.getParameterShapes;
const mlir = pcp.mlir;

pub fn main() !void {
    std.debug.print("=== GPT-2 Model Graph Construction Test ===\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 1. Use a "nano" config for a small, manageable graph
    const config = GPT2Config.nano();
    const batch_size = 4;
    const block_size = 8;

    // 2. Initialize the MLIR context and builder
    var mlir_ctx = try MLIRContext.init(allocator);
    defer mlir_ctx.deinit();

    var builder = try MLIRBuilder.init(allocator, mlir_ctx.getContext());
    defer builder.deinit();

    // 3. Define the `func.func` signature for the training step
    var func_input_types = std.ArrayList(mlir.Type).init(allocator);
    defer func_input_types.deinit();

    // Add types for all trainable parameters
    const param_shapes = try getParameterShapes(allocator, config);
    defer {
        for (param_shapes) |s| allocator.free(s);
        allocator.free(param_shapes);
    }
    const f32_type = mlir.Type.f32Type(builder.ctx);
    for (param_shapes) |shape| {
        try func_input_types.append(mlir.Type.rankedTensorType(builder.ctx, shape, f32_type));
    }

    // Add types for input data (x) and targets (y)
    const data_shape = &[_]i64{batch_size, block_size};
    const i32_type = mlir.Type.i32Type(builder.ctx); // Inputs are token IDs
    try func_input_types.append(mlir.Type.rankedTensorType(builder.ctx, data_shape, i32_type));
    try func_input_types.append(mlir.Type.rankedTensorType(builder.ctx, data_shape, i32_type));

    // Define the function's output types (just the scalar loss for now)
    const scalar_f32_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type);
    const func_output_types = &[_]mlir.Type{scalar_f32_type};

    const func_type = mlir.Type.functionType(builder.ctx, func_input_types.items, func_output_types);

    // 4. Create the function and get its entry block and arguments
    const result = try builder.createFunction("gpt2_training_step", func_type);
    _ = result.func_op; // Attach to module automatically
    const entry_block = result.entry_block;

    // Set the builder's insertion point to inside our new function
    builder.setInsertionBlock(entry_block);

    // 5. Instantiate the GPT-2 model using the function arguments as parameters
    var all_func_args = try entry_block.getArguments(allocator);
    defer allocator.free(all_func_args);

    const GPT2Model = pcp.models.gpt2.GPT2(f32);
    var params_slice = all_func_args[0..param_shapes.len];
    var model = try GPT2Model.init(allocator, config, &builder, &params_slice);
    defer model.deinit();

    // Get handles to the input data tensors
    const x = try builder.newTensor(all_func_args[param_shapes.len]);
    const y = try builder.newTensor(all_func_args[param_shapes.len + 1]);

    // 6. Build the forward pass and loss calculation graph
    const outputs = try model.forwardWithLoss(x, y, &builder);
    
    // 7. Create the `func.return` operation
    _ = try builder.createAndAttach("func.return", &.{outputs.loss.value}, &.{});

    // 8. Dump the generated MLIR module for inspection
    std.debug.print("\n--- Generated MLIR for GPT-2 Training Step ---\n", .{});
    builder.module.op().dump();
    std.debug.print("\n--- End of MLIR Dump ---\n", .{});

    // 9. (Crucial) Verify the module. This will catch any MLIR errors.
    const verification_result = builder.module.op().verify();
    if (verification_result.isFailure()) {
        std.debug.print("💣 MLIR verification failed!\n", .{});
        return error.MLIRVerificationFailed;
    }

    std.debug.print("🌙 MLIR module verified successfully!\n", .{});
    std.debug.print("🌙 GPT-2 model graph construction test passed.\n", .{});
}