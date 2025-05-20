const std = @import("std");
const pcp = @import("pcp");
const tensorModule = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;

/// Helper function for pointer casting
fn ptrCastHelper(comptime T: type, ptr: anytype) T {
    // We still need to use alignCast for pointers that require higher alignment
    return @ptrCast(@alignCast(ptr));
}

const Allocator = std.mem.Allocator;
const Tensor = tensorModule.Tensor;
const DType = tensorModule.DType;
// Using Plan-based approach instead of Node-based autodiff

// Helper function to print tensor contents
fn printTensor(t: Tensor) void {
    const buf = ptrCastHelper([*]f32, t.buffer.data.ptr)[0..t.shape.elemCount()];

    std.debug.print("Shape: [", .{});
    for (t.shape.dims) |dim| {
        std.debug.print("{}, ", .{dim});
    }
    std.debug.print("]\n", .{});

    if (t.shape.rank() == 2) {
        const rows = t.shape.dims[0];
        const cols = t.shape.dims[1];

        for (0..rows) |i| {
            std.debug.print("[ ", .{});
            for (0..cols) |j| {
                std.debug.print("{d:.4} ", .{buf[i * cols + j]});
            }
            std.debug.print("]\n", .{});
        }
    } else {
        // For other ranks, just print all values
        std.debug.print("[ ", .{});
        for (buf) |val| {
            std.debug.print("{d:.4} ", .{val});
        }
        std.debug.print("]\n", .{});
    }
}

// Test a simple model with Plan-based approach
fn testSimpleModel(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Simple Model with Plan-Based Approach ===\n", .{});

    // Create simple tensors
    var dims = [_]usize{ 2, 2 };

    // Create tensors for parameters
    var w1 = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer w1.deinit();
    try w1.setScalar(&[_]usize{ 0, 0 }, 0.1);
    try w1.setScalar(&[_]usize{ 0, 1 }, 0.2);
    try w1.setScalar(&[_]usize{ 1, 0 }, 0.3);
    try w1.setScalar(&[_]usize{ 1, 1 }, 0.4);

    var w2 = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer w2.deinit();
    try w2.setScalar(&[_]usize{ 0, 0 }, 0.5);
    try w2.setScalar(&[_]usize{ 0, 1 }, 0.6);
    try w2.setScalar(&[_]usize{ 1, 0 }, 0.7);
    try w2.setScalar(&[_]usize{ 1, 1 }, 0.8);

    // Create input
    var x = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer x.deinit();
    try x.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try x.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try x.setScalar(&[_]usize{ 1, 0 }, 3.0);
    try x.setScalar(&[_]usize{ 1, 1 }, 4.0);

    // Target values
    var y_target = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer y_target.deinit();
    try y_target.setScalar(&[_]usize{ 0, 0 }, 0.5);
    try y_target.setScalar(&[_]usize{ 0, 1 }, 0.5);
    try y_target.setScalar(&[_]usize{ 1, 0 }, 0.5);
    try y_target.setScalar(&[_]usize{ 1, 1 }, 0.5);

    // Create the autodiff plans
    std.debug.print("Creating Plan-based operations\n", .{});

    // Create MatMul plans
    const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend, f32, null, null, null);
    var matmul_plan1 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan1.deinit();

    var matmul_plan2 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan2.deinit();

    // Create ReLU plan
    const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend, f32, null);
    var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
    defer relu_plan.deinit();

    // Create Subtract plan
    const SubtractPlanType = autodiff.SubtractPlanWithGrad(ops.CpuBackend, f32, null);
    var subtract_plan = autodiff.AutoDiffPlan(SubtractPlanType).init(allocator);
    defer subtract_plan.deinit();

    // Create Multiply plan
    const MultiplyPlanType = autodiff.MultiplyPlanWithGrad(ops.CpuBackend, f32, null);
    var multiply_plan = autodiff.AutoDiffPlan(MultiplyPlanType).init(allocator);
    defer multiply_plan.deinit();

    // Forward pass using plans
    std.debug.print("Running forward pass with plans...\n", .{});

    // h_pre = x @ w1
    const h_pre = try matmul_plan1.forward(.{ .a = x, .b = w1 });
    defer h_pre.deinit();

    // h = relu(h_pre)
    const h = try relu_plan.forward(h_pre);
    defer h.deinit();

    // y_pred = h @ w2
    const y_pred = try matmul_plan2.forward(.{ .a = h, .b = w2 });
    defer y_pred.deinit();

    // Print prediction
    std.debug.print("\nPrediction (Plan-based):\n", .{});
    printTensor(y_pred);

    // Compute MSE loss components
    // diff = y_pred - y_target
    const diff = try subtract_plan.forward(.{ .a = y_pred, .b = y_target });
    defer diff.deinit();

    // diff_squared = diff * diff
    const diff_squared = try multiply_plan.forward(.{ .a = diff, .b = diff });
    defer diff_squared.deinit();

    // Calculate mean manually
    const diff_squared_buf = ptrCastHelper([*]f32, diff_squared.buffer.data.ptr)[0..diff_squared.shape.elemCount()];
    var sum: f32 = 0.0;
    for (diff_squared_buf) |val| {
        sum += val;
    }
    const loss_value = sum / @as(f32, @floatFromInt(diff_squared.shape.elemCount()));

    std.debug.print("\nMSE Loss: {d:.6}\n", .{loss_value});

    // Backward pass
    std.debug.print("\nRunning backward pass with plans...\n", .{});

    // Create gradient with ones
    var grad_ones = try Tensor.filled(allocator, y_pred.shape.dims, y_pred.dtype, 1.0, y_pred.backend);
    defer grad_ones.deinit();

    // Backprop through second matmul
    const grads2 = try matmul_plan2.backward(grad_ones);
    defer grads2.da.deinit();
    defer grads2.db.deinit();

    // Backprop through relu
    const relu_grad = try relu_plan.backward(grads2.da);
    defer relu_grad.deinit();

    // Backprop through first matmul
    const grads1 = try matmul_plan1.backward(relu_grad);
    defer grads1.da.deinit();
    defer grads1.db.deinit();

    // Print gradients
    std.debug.print("\nGradients for w1 (Plan-based):\n", .{});
    printTensor(grads1.db);

    std.debug.print("\nGradients for w2 (Plan-based):\n", .{});
    printTensor(grads2.db);

    std.debug.print("\nPlan-based model test completed successfully\n", .{});
}

// Test embedding functionality with Plan-based approach
fn testEmbeddings(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Embedding Lookup with Plan-Based Approach ===\n", .{});

    // Create a small embedding table
    const vocab_size: usize = 5;
    const embed_dim: usize = 3;
    var embed_dims = [_]usize{ vocab_size, embed_dim }; // 5 words, 3-dimensional embeddings
    var embeddings = try Tensor.zeros(allocator, &embed_dims, .f32, .cpu);
    defer embeddings.deinit();

    // Fill with test values
    var embed_buf = ptrCastHelper([*]f32, embeddings.buffer.data.ptr)[0..embeddings.shape.elemCount()];
    // Word 0: [0.1, 0.2, 0.3]
    embed_buf[0] = 0.1;
    embed_buf[1] = 0.2;
    embed_buf[2] = 0.3;
    // Word 1: [0.4, 0.5, 0.6]
    embed_buf[3] = 0.4;
    embed_buf[4] = 0.5;
    embed_buf[5] = 0.6;
    // Word 2: [0.7, 0.8, 0.9]
    embed_buf[6] = 0.7;
    embed_buf[7] = 0.8;
    embed_buf[8] = 0.9;
    // Word 3: [1.0, 1.1, 1.2]
    embed_buf[9] = 1.0;
    embed_buf[10] = 1.1;
    embed_buf[11] = 1.2;
    // Word 4: [1.3, 1.4, 1.5]
    embed_buf[12] = 1.3;
    embed_buf[13] = 1.4;
    embed_buf[14] = 1.5;

    std.debug.print("Created embedding table with shape [{}, {}]\n", .{ embed_dims[0], embed_dims[1] });

    // Create indices tensor for lookup
    var indices_dims = [_]usize{ 2, 2 }; // Batch size 2, sequence length 2
    var indices = try Tensor.zeros(allocator, &indices_dims, .f32, .cpu);
    defer indices.deinit();

    // Set indices: [[1, 2], [3, 0]]
    try indices.setScalar(&[_]usize{ 0, 0 }, 1.0); // first word of first sequence: word ID 1
    try indices.setScalar(&[_]usize{ 0, 1 }, 2.0); // second word of first sequence: word ID 2
    try indices.setScalar(&[_]usize{ 1, 0 }, 3.0); // first word of second sequence: word ID 3
    try indices.setScalar(&[_]usize{ 1, 1 }, 0.0); // second word of second sequence: word ID 0

    std.debug.print("Created indices tensor with shape [{}, {}]\n", .{ indices_dims[0], indices_dims[1] });

    // Create embedding lookup plan
    const EmbedPlanType = autodiff.EmbeddingLookupPlanWithGrad(ops.CpuBackend, f32, vocab_size, embed_dim);
    var embed_plan = autodiff.AutoDiffPlan(EmbedPlanType).init(allocator);
    defer embed_plan.deinit();

    // Create subtract plan for loss computation
    const SubtractPlanType = autodiff.SubtractPlanWithGrad(ops.CpuBackend, f32, null);
    var subtract_plan = autodiff.AutoDiffPlan(SubtractPlanType).init(allocator);
    defer subtract_plan.deinit();

    // Create multiply plan for squaring the difference
    const MultiplyPlanType = autodiff.MultiplyPlanWithGrad(ops.CpuBackend, f32, null);
    var multiply_plan = autodiff.AutoDiffPlan(MultiplyPlanType).init(allocator);
    defer multiply_plan.deinit();

    // Perform embedding lookup using plan
    std.debug.print("Performing embedding lookup with plan...\n", .{});
    const lookup_result = try embed_plan.forward(.{ .params = embeddings, .indices = indices });
    defer lookup_result.deinit();

    // Print results
    std.debug.print("\nEmbedding lookup result (Plan-based):\n", .{});
    printTensor(lookup_result);

    // Create target tensor with the same shape as lookup_result
    const target = try Tensor.filled(allocator, lookup_result.shape.dims, lookup_result.dtype, 0.5, lookup_result.backend);
    defer target.deinit();

    std.debug.print("\nTarget tensor:\n", .{});
    printTensor(target);

    // Compute difference: diff = lookup_result - target
    std.debug.print("\nComputing loss with plans...\n", .{});
    const diff = try subtract_plan.forward(.{ .a = lookup_result, .b = target });
    defer diff.deinit();

    // Square the difference: diff_squared = diff * diff
    const diff_squared = try multiply_plan.forward(.{ .a = diff, .b = diff });
    defer diff_squared.deinit();

    // Calculate mean manually
    const diff_squared_buf = ptrCastHelper([*]f32, diff_squared.buffer.data.ptr)[0..diff_squared.shape.elemCount()];
    var sum: f32 = 0.0;
    for (diff_squared_buf) |val| {
        sum += val;
    }
    const mse = sum / @as(f32, @floatFromInt(diff_squared.shape.elemCount()));

    std.debug.print("\nMSE Loss: {d:.6}\n", .{mse});

    // Create gradient with ones
    var grad_ones = try Tensor.filled(allocator, diff_squared.shape.dims, diff_squared.dtype, 1.0, diff_squared.backend);
    defer grad_ones.deinit();

    // Backward pass
    std.debug.print("\nRunning backward pass with plans...\n", .{});

    // Backprop through multiply
    const diff_grads = try multiply_plan.backward(grad_ones);
    defer diff_grads.da.deinit();
    defer diff_grads.db.deinit();

    // Backprop through subtract
    const lookup_target_grads = try subtract_plan.backward(diff_grads.da);
    defer lookup_target_grads.da.deinit();
    defer lookup_target_grads.db.deinit();

    // Backprop through embedding lookup
    const embed_grads = try embed_plan.backward(lookup_target_grads.da);
    defer embed_grads.deinit();

    // Print embedding gradients
    std.debug.print("\nEmbedding gradients (Plan-based):\n", .{});
    printTensor(embed_grads);

    std.debug.print("\nPlan-based embedding test completed successfully\n", .{});
}

// Test a more complex model with Plan-based approach
fn testComplexModel(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Complex Model with Plan-Based Approach ===\n", .{});

    // Create emulation of GPT-2 tokens with a small vocab (10) and small embedding dim (4)
    const vocab_size: usize = 10;
    const embed_dim: usize = 4;

    var wte_dims = [_]usize{ vocab_size, embed_dim };
    var wte = try Tensor.random(allocator, &wte_dims, .f32, .cpu);
    defer wte.deinit();

    // Scale down random values
    const wte_ptr = ptrCastHelper([*]f32, wte.buffer.data.ptr)[0..wte.shape.elemCount()];
    for (wte_ptr) |*val| {
        val.* *= 0.1;
    }

    // Position embeddings: 5 positions, embedding dim 4
    const num_positions: usize = 5;
    var wpe_dims = [_]usize{ num_positions, embed_dim };
    var wpe = try Tensor.random(allocator, &wpe_dims, .f32, .cpu);
    defer wpe.deinit();

    // Scale down random values
    const wpe_ptr = ptrCastHelper([*]f32, wpe.buffer.data.ptr)[0..wpe.shape.elemCount()];
    for (wpe_ptr) |*val| {
        val.* *= 0.1;
    }

    // Create input: batch size 2, sequence length 3, with token IDs
    const batch_size: usize = 2;
    const seq_len: usize = 3;
    var input_dims = [_]usize{ batch_size, seq_len };
    var input_ids = try Tensor.zeros(allocator, &input_dims, .f32, .cpu);
    defer input_ids.deinit();

    // Set token IDs: [[1, 2, 3], [4, 5, 6]]
    try input_ids.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try input_ids.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try input_ids.setScalar(&[_]usize{ 0, 2 }, 3.0);
    try input_ids.setScalar(&[_]usize{ 1, 0 }, 4.0);
    try input_ids.setScalar(&[_]usize{ 1, 1 }, 5.0);
    try input_ids.setScalar(&[_]usize{ 1, 2 }, 6.0);

    // Create position indices
    var pos_indices = try Tensor.zeros(allocator, &input_dims, .f32, .cpu);
    defer pos_indices.deinit();

    // Fill with position IDs
    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            try pos_indices.setScalar(&[_]usize{ b, s }, @as(f32, @floatFromInt(s)));
        }
    }

    // Create projection layer weights
    var proj_dims = [_]usize{ embed_dim, vocab_size }; // Project back to vocab size
    var proj = try Tensor.random(allocator, &proj_dims, .f32, .cpu);
    defer proj.deinit();

    // Scale down
    const proj_ptr = ptrCastHelper([*]f32, proj.buffer.data.ptr)[0..proj.shape.elemCount()];
    for (proj_ptr) |*val| {
        val.* *= 0.1;
    }

    // Create plans for all operations
    std.debug.print("Creating Plan-based operations\n", .{});

    // Token embedding lookup plan
    const TokenEmbedPlanType = autodiff.EmbeddingLookupPlanWithGrad(ops.CpuBackend, f32, vocab_size, embed_dim);
    var token_embed_plan = autodiff.AutoDiffPlan(TokenEmbedPlanType).init(allocator);
    defer token_embed_plan.deinit();

    // Position embedding lookup plan
    const PosEmbedPlanType = autodiff.EmbeddingLookupPlanWithGrad(ops.CpuBackend, f32, num_positions, embed_dim);
    var pos_embed_plan = autodiff.AutoDiffPlan(PosEmbedPlanType).init(allocator);
    defer pos_embed_plan.deinit();

    // Add plan
    const AddPlanType = autodiff.AddPlanWithGrad(ops.CpuBackend, f32, null);
    var add_plan = autodiff.AutoDiffPlan(AddPlanType).init(allocator);
    defer add_plan.deinit();

    // MatMul plan
    const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend, f32, null, null, null);
    var matmul_plan = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan.deinit();

    // Softmax plan
    const SoftmaxPlanType = autodiff.SoftmaxPlanWithGrad(ops.CpuBackend, f32, null, null);
    var softmax_plan = autodiff.AutoDiffPlan(SoftmaxPlanType).init(allocator);
    defer softmax_plan.deinit();

    // Subtract plan
    const SubtractPlanType = autodiff.SubtractPlanWithGrad(ops.CpuBackend, f32, null);
    var subtract_plan = autodiff.AutoDiffPlan(SubtractPlanType).init(allocator);
    defer subtract_plan.deinit();

    // Multiply plan
    const MultiplyPlanType = autodiff.MultiplyPlanWithGrad(ops.CpuBackend, f32, null);
    var multiply_plan = autodiff.AutoDiffPlan(MultiplyPlanType).init(allocator);
    defer multiply_plan.deinit();

    // Forward pass using plans
    std.debug.print("Running forward pass with plans...\n", .{});

    // Token embeddings
    const token_embeds = try token_embed_plan.forward(.{ .params = wte, .indices = input_ids });
    defer token_embeds.deinit();

    // Position embeddings
    const pos_embeds = try pos_embed_plan.forward(.{ .params = wpe, .indices = pos_indices });
    defer pos_embeds.deinit();

    // Reshape embeddings to match dimensions for adding
    const flat_size = batch_size * seq_len * embed_dim;
    var token_embeds_flat = try Tensor.zeros(allocator, &[_]usize{flat_size}, .f32, token_embeds.backend);
    defer token_embeds_flat.deinit();

    var pos_embeds_flat = try Tensor.zeros(allocator, &[_]usize{flat_size}, .f32, pos_embeds.backend);
    defer pos_embeds_flat.deinit();

    // Copy data - make sure we're using the correct sizes
    const token_data_size = @min(token_embeds_flat.buffer.data.len, token_embeds.buffer.data.len);
    const pos_data_size = @min(pos_embeds_flat.buffer.data.len, pos_embeds.buffer.data.len);

    @memcpy(token_embeds_flat.buffer.data[0..token_data_size], token_embeds.buffer.data[0..token_data_size]);
    @memcpy(pos_embeds_flat.buffer.data[0..pos_data_size], pos_embeds.buffer.data[0..pos_data_size]);

    // Reshape to 2D for next operations
    var token_embeds_reshaped = try Tensor.zeros(allocator, &[_]usize{ batch_size * seq_len, embed_dim }, .f32, token_embeds.backend);
    defer token_embeds_reshaped.deinit();

    var pos_embeds_reshaped = try Tensor.zeros(allocator, &[_]usize{ batch_size * seq_len, embed_dim }, .f32, pos_embeds.backend);
    defer pos_embeds_reshaped.deinit();

    // Copy data to reshaped tensors - make sure we're using the correct sizes
    const token_reshape_size = @min(token_embeds_reshaped.buffer.data.len, token_embeds_flat.buffer.data.len);
    const pos_reshape_size = @min(pos_embeds_reshaped.buffer.data.len, pos_embeds_flat.buffer.data.len);

    @memcpy(token_embeds_reshaped.buffer.data[0..token_reshape_size], token_embeds_flat.buffer.data[0..token_reshape_size]);
    @memcpy(pos_embeds_reshaped.buffer.data[0..pos_reshape_size], pos_embeds_flat.buffer.data[0..pos_reshape_size]);

    // Add token and position embeddings
    const combined = try add_plan.forward(.{ .a = token_embeds_reshaped, .b = pos_embeds_reshaped });
    defer combined.deinit();

    // Apply projection: logits = combined @ proj
    const logits = try matmul_plan.forward(.{ .a = combined, .b = proj });
    defer logits.deinit();

    // Apply softmax to get probabilities
    const probs = try softmax_plan.forward(logits);
    defer probs.deinit();

    // Create target tensor: one-hot
    var targets = try Tensor.zeros(allocator, &[_]usize{ batch_size * seq_len, vocab_size }, .f32, .cpu);
    defer targets.deinit();

    // Set target values (just set position 0 to 1.0 for each example)
    const targets_buf = ptrCastHelper([*]f32, targets.buffer.data.ptr)[0..targets.shape.elemCount()];
    for (0..batch_size * seq_len) |i| {
        targets_buf[i * vocab_size] = 1.0;
    }

    // Compute loss: diff = probs - targets
    const diff = try subtract_plan.forward(.{ .a = probs, .b = targets });
    defer diff.deinit();

    // Square the difference: diff_squared = diff * diff
    const diff_squared = try multiply_plan.forward(.{ .a = diff, .b = diff });
    defer diff_squared.deinit();

    // Calculate sum and mean manually
    const diff_squared_buf = ptrCastHelper([*]f32, diff_squared.buffer.data.ptr)[0..diff_squared.shape.elemCount()];
    var sum: f32 = 0.0;
    for (diff_squared_buf) |val| {
        sum += val;
    }
    const mse = sum / @as(f32, @floatFromInt(diff_squared.shape.elemCount()));

    std.debug.print("\nMSE Loss: {d:.6}\n", .{mse});

    // Create gradient with ones
    var grad_ones = try Tensor.filled(allocator, diff_squared.shape.dims, diff_squared.dtype, 1.0, diff_squared.backend);
    defer grad_ones.deinit();

    // Backward pass
    std.debug.print("\nRunning backward pass with plans...\n", .{});

    // Backprop through square
    const diff_grads = try multiply_plan.backward(grad_ones);
    defer diff_grads.da.deinit();
    defer diff_grads.db.deinit();

    // Backprop through subtract
    const prob_target_grads = try subtract_plan.backward(diff_grads.da);
    defer prob_target_grads.da.deinit();
    defer prob_target_grads.db.deinit();

    // Backprop through softmax
    const logits_grad = try softmax_plan.backward(prob_target_grads.da);
    defer logits_grad.deinit();

    // Backprop through projection
    // Note: We need to provide the same format as in the forward pass
    const proj_grads = try matmul_plan.backward(logits_grad);
    defer proj_grads.da.deinit();
    defer proj_grads.db.deinit();

    // Backprop through add
    const embed_grads = try add_plan.backward(proj_grads.da);
    defer embed_grads.da.deinit();
    defer embed_grads.db.deinit();

    // Print gradients
    std.debug.print("\nGradients for token embeddings (Plan-based):\n", .{});
    const token_grads = try token_embed_plan.backward(embed_grads.da);
    defer token_grads.deinit();
    printTensor(token_grads);

    std.debug.print("\nGradients for position embeddings (Plan-based):\n", .{});
    const pos_grads = try pos_embed_plan.backward(embed_grads.db);
    defer pos_grads.deinit();
    printTensor(pos_grads);

    std.debug.print("\nGradients for projection matrix (Plan-based):\n", .{});
    printTensor(proj_grads.db);

    std.debug.print("\nPlan-based complex model test completed successfully\n", .{});
}

// Test a simple model with Plan-based approach
fn testPlanBasedModel(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Plan-Based Model ===\n", .{});

    // Create simple tensors
    var dims = [_]usize{ 2, 2 };

    // Create tensors for parameters
    var w1_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer w1_tensor.deinit();

    try w1_tensor.setScalar(&[_]usize{ 0, 0 }, 0.1);
    try w1_tensor.setScalar(&[_]usize{ 0, 1 }, 0.2);
    try w1_tensor.setScalar(&[_]usize{ 1, 0 }, 0.3);
    try w1_tensor.setScalar(&[_]usize{ 1, 1 }, 0.4);

    var w2_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer w2_tensor.deinit();

    try w2_tensor.setScalar(&[_]usize{ 0, 0 }, 0.5);
    try w2_tensor.setScalar(&[_]usize{ 0, 1 }, 0.6);
    try w2_tensor.setScalar(&[_]usize{ 1, 0 }, 0.7);
    try w2_tensor.setScalar(&[_]usize{ 1, 1 }, 0.8);

    // Create input
    var x_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer x_tensor.deinit();

    try x_tensor.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try x_tensor.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try x_tensor.setScalar(&[_]usize{ 1, 0 }, 3.0);
    try x_tensor.setScalar(&[_]usize{ 1, 1 }, 4.0);

    // Target values
    var y_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer y_tensor.deinit();

    try y_tensor.setScalar(&[_]usize{ 0, 0 }, 0.5);
    try y_tensor.setScalar(&[_]usize{ 0, 1 }, 0.5);
    try y_tensor.setScalar(&[_]usize{ 1, 0 }, 0.5);
    try y_tensor.setScalar(&[_]usize{ 1, 1 }, 0.5);

    // Create the autodiff plans
    const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend, f32, null, null, null);
    var matmul_plan1 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan1.deinit();

    var matmul_plan2 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan2.deinit();

    const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend, f32, null);
    var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
    defer relu_plan.deinit();

    // Forward pass using plans
    std.debug.print("Running forward pass with plans...\n", .{});

    // h_pre = x @ w1
    const h_pre = try matmul_plan1.forward(.{ .a = x_tensor, .b = w1_tensor });
    defer h_pre.deinit();

    // h = relu(h_pre)
    const h = try relu_plan.forward(h_pre);
    defer h.deinit();

    // y_pred = h @ w2
    const y_pred = try matmul_plan2.forward(.{ .a = h, .b = w2_tensor });
    defer y_pred.deinit();

    // Print prediction
    std.debug.print("\nPrediction (Plan-based):\n", .{});
    printTensor(y_pred);

    // Compute MSE loss manually
    var diff = try ops.subtract(allocator, y_pred, y_tensor);
    defer diff.deinit();

    var diff_squared = try ops.multiply(allocator, diff, diff);
    defer diff_squared.deinit();

    // Calculate mean manually
    const diff_squared_buf = ptrCastHelper([*]f32, diff_squared.buffer.data.ptr)[0..diff_squared.shape.elemCount()];
    var sum: f32 = 0.0;
    for (diff_squared_buf) |val| {
        sum += val;
    }
    const mse = sum / @as(f32, @floatFromInt(diff_squared.shape.elemCount()));

    std.debug.print("\nMSE Loss: {d:.6}\n", .{mse});

    // Create gradient with ones
    var grad_ones = try Tensor.filled(allocator, y_pred.shape.dims, y_pred.dtype, 1.0, y_pred.backend);
    defer grad_ones.deinit();

    // Backward pass
    std.debug.print("\nRunning backward pass with plans...\n", .{});

    // Backprop through second matmul
    const grads2 = try matmul_plan2.backward(grad_ones);
    defer grads2.da.deinit();
    defer grads2.db.deinit();

    // Backprop through relu
    const relu_grad = try relu_plan.backward(grads2.da);
    defer relu_grad.deinit();

    // Backprop through first matmul
    const grads1 = try matmul_plan1.backward(relu_grad);
    defer grads1.da.deinit();
    defer grads1.db.deinit();

    // Print gradients
    std.debug.print("\nGradients for w1 (Plan-based):\n", .{});
    printTensor(grads1.db);

    std.debug.print("\nGradients for w2 (Plan-based):\n", .{});
    printTensor(grads2.db);

    std.debug.print("\nPlan-based model test completed successfully\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Starting autodiff with Plan-based approach...\n", .{});

    // Run all Plan-based tests
    try testSimpleModel(allocator);
    try testEmbeddings(allocator);
    try testComplexModel(allocator);
    try testPlanBasedModel(allocator);

    std.debug.print("\nAll Plan-based tests completed successfully!\n", .{});
}
