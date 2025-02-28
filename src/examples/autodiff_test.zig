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
const Node = autodiff.Node;

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

// Test a simple model with backward pass
fn testSimpleModel(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Simple Model ===\n", .{});
    
    // Create simple tensors
    var dims = [_]usize{ 2, 2 };
    
    // Create tensors for parameters
    var w1_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try w1_tensor.setScalar(&[_]usize{0, 0}, 0.1);
    try w1_tensor.setScalar(&[_]usize{0, 1}, 0.2);
    try w1_tensor.setScalar(&[_]usize{1, 0}, 0.3);
    try w1_tensor.setScalar(&[_]usize{1, 1}, 0.4);
    
    var w2_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try w2_tensor.setScalar(&[_]usize{0, 0}, 0.5);
    try w2_tensor.setScalar(&[_]usize{0, 1}, 0.6);
    try w2_tensor.setScalar(&[_]usize{1, 0}, 0.7);
    try w2_tensor.setScalar(&[_]usize{1, 1}, 0.8);
    
    // Create input
    var x_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try x_tensor.setScalar(&[_]usize{0, 0}, 1.0);
    try x_tensor.setScalar(&[_]usize{0, 1}, 2.0);
    try x_tensor.setScalar(&[_]usize{1, 0}, 3.0);
    try x_tensor.setScalar(&[_]usize{1, 1}, 4.0);
    
    // Target values
    var y_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try y_tensor.setScalar(&[_]usize{0, 0}, 0.5);
    try y_tensor.setScalar(&[_]usize{0, 1}, 0.5);
    try y_tensor.setScalar(&[_]usize{1, 0}, 0.5);
    try y_tensor.setScalar(&[_]usize{1, 1}, 0.5);
    
    // Convert to nodes (requires_grad = true for parameters)
    std.debug.print("Creating parameter nodes\n", .{});
    
    // For parameters, make copies to keep the originals
    const w1_copy = try Tensor.init(allocator, w1_tensor.shape.dims, w1_tensor.dtype, w1_tensor.backend);
    @memcpy(w1_copy.buffer.data, w1_tensor.buffer.data[0..w1_tensor.buffer.data.len]);
    
    const w2_copy = try Tensor.init(allocator, w2_tensor.shape.dims, w2_tensor.dtype, w2_tensor.backend);
    @memcpy(w2_copy.buffer.data, w2_tensor.buffer.data[0..w2_tensor.buffer.data.len]);
    
    var w1 = try autodiff.variable(allocator, w1_copy, true);
    var w2 = try autodiff.variable(allocator, w2_copy, true);
    
    // For input, make a copy
    const x_copy = try Tensor.init(allocator, x_tensor.shape.dims, x_tensor.dtype, x_tensor.backend);
    @memcpy(x_copy.buffer.data, x_tensor.buffer.data[0..x_tensor.buffer.data.len]);
    
    var x = try autodiff.variable(allocator, x_copy, false); // Doesn't need gradients
    
    // Forward pass: h = relu(x @ w1), y_pred = h @ w2
    std.debug.print("Building computational graph\n", .{});
    var h_pre = try autodiff.matmul(allocator, x, w1); // Linear layer
    var h = try autodiff.relu(allocator, h_pre); // Activation
    var y_pred = try autodiff.matmul(allocator, h, w2); // Second linear layer
    
    // Print prediction
    std.debug.print("\nPrediction:\n", .{});
    printTensor(y_pred.tensor);
    
    // Compute loss: MSE = mean((y_pred - y_target)^2)
    // Convert y_target to node
    const y_copy = try Tensor.init(allocator, y_tensor.shape.dims, y_tensor.dtype, y_tensor.backend);
    @memcpy(y_copy.buffer.data, y_tensor.buffer.data[0..y_tensor.buffer.data.len]);
    
    var y_target = try autodiff.variable(allocator, y_copy, false);
    
    // Compute difference
    var diff = try autodiff.subtract(allocator, y_pred, y_target);
    
    // Square the difference
    var diff_squared = try autodiff.multiply(allocator, diff, diff);
    
    // Create a tensor of ones to use for reducing
    const ones = try Tensor.filled(allocator, diff_squared.tensor.shape.dims, diff_squared.tensor.dtype, 1.0, diff_squared.tensor.backend);
    var ones_node = try autodiff.variable(allocator, ones, false);
    
    // Create a small constant factor node for mean calculation
    const ones_for_reduction = try Tensor.filled(allocator, &[_]usize{diff_squared.tensor.shape.dims[1], 1}, diff_squared.tensor.dtype, 1.0, diff_squared.tensor.backend);
    var ones_for_reduction_node = try autodiff.variable(allocator, ones_for_reduction, false);
    
    // Sum through matrix multiplication with ones tensor
    // sum(diff_squared) = diff_squared @ ones
    var sum_squared_diff = try autodiff.matmul(allocator, diff_squared, ones_for_reduction_node);
    
    // Create scale factor - matching the shape of sum_squared_diff
    const scale_factor = try Tensor.filled(allocator, sum_squared_diff.tensor.shape.dims, sum_squared_diff.tensor.dtype, 0.25, sum_squared_diff.tensor.backend);
    var scale_node = try autodiff.variable(allocator, scale_factor, false);
    
    // Multiply by 0.25 to get mean (element-wise multiplication)
    var loss = try autodiff.multiply(allocator, sum_squared_diff, scale_node);
    
    // Print loss
    std.debug.print("\nLoss:\n", .{});
    printTensor(loss.tensor);
    
    // Run backward pass
    std.debug.print("\nRunning backward pass...\n", .{});
    try autodiff.backward(allocator, loss);
    
    // Print gradients
    std.debug.print("\nGradients for w1:\n", .{});
    if (w1.grad) |grad| {
        printTensor(grad);
    } else {
        std.debug.print("No gradient\n", .{});
    }
    
    std.debug.print("\nGradients for w2:\n", .{});
    if (w2.grad) |grad| {
        printTensor(grad);
    } else {
        std.debug.print("No gradient\n", .{});
    }
    
    // Clean up intermediate nodes
    std.debug.print("\nCleaning up nodes\n", .{});
    loss.deinit();
    sum_squared_diff.deinit();
    scale_node.deinit();
    ones_for_reduction_node.deinit();
    ones_node.deinit();
    diff_squared.deinit();
    diff.deinit();
    y_target.deinit();
    y_pred.deinit();
    h.deinit();
    h_pre.deinit();
    x.deinit();
    w2.deinit();
    w1.deinit();
    
    // Clean up original tensors
    w1_tensor.deinit();
    w2_tensor.deinit();
    x_tensor.deinit();
    y_tensor.deinit();
    
    std.debug.print("\nModel test completed successfully\n", .{});
}

// Test embedding functionality
fn testEmbeddings(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Embedding Lookup ===\n", .{});
    
    // Create a small embedding table
    var embed_dims = [_]usize{ 5, 3 }; // 5 words, 3-dimensional embeddings
    var embeddings = try Tensor.zeros(allocator, &embed_dims, .f32, .cpu);
    
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
    
    std.debug.print("Created embedding table with shape [{}, {}]\n", .{embed_dims[0], embed_dims[1]});
    
    // Create embedding parameters node
    var embed_node = try autodiff.variable(allocator, embeddings, true);
    
    // Create indices tensor for lookup
    var indices_dims = [_]usize{ 2, 2 }; // Batch size 2, sequence length 2
    var indices = try Tensor.zeros(allocator, &indices_dims, .f32, .cpu);
    
    // Set indices: [[1, 2], [3, 0]]
    try indices.setScalar(&[_]usize{0, 0}, 1.0); // first word of first sequence: word ID 1
    try indices.setScalar(&[_]usize{0, 1}, 2.0); // second word of first sequence: word ID 2
    try indices.setScalar(&[_]usize{1, 0}, 3.0); // first word of second sequence: word ID 3
    try indices.setScalar(&[_]usize{1, 1}, 0.0); // second word of second sequence: word ID 0
    
    std.debug.print("Created indices tensor with shape [{}, {}]\n", .{indices_dims[0], indices_dims[1]});
    
    // Store the reference count before lookup
    const embed_ref_before = embeddings.getRefCount();
    const indices_ref_before = indices.getRefCount();
    std.debug.print("Before lookup: embed_ref_count={}, indices_ref_count={}\n", 
        .{embed_ref_before, indices_ref_before});
    
    // Perform embedding lookup
    std.debug.print("Performing embedding lookup\n", .{});
    var lookup_result = try autodiff.embedding_lookup(allocator, embed_node, indices);
    
    // Check reference counts after lookup
    const embed_ref_after = embeddings.getRefCount();
    const indices_ref_after = indices.getRefCount();
    std.debug.print("After lookup: embed_ref_count={}, indices_ref_count={}\n", 
        .{embed_ref_after, indices_ref_after});
    
    // Print results
    std.debug.print("\nEmbedding lookup result:\n", .{});
    printTensor(lookup_result.tensor);
    
    // Create a simpler loss function
    // Just use mean squared error directly on the embedding output
    // Create target tensor with the same shape as lookup_result.tensor
    const target = try Tensor.filled(allocator, lookup_result.tensor.shape.dims, 
                                 lookup_result.tensor.dtype, 0.5, lookup_result.tensor.backend);
    
    std.debug.print("\nTarget tensor:\n", .{});
    printTensor(target);
    
    var target_node = try autodiff.variable(allocator, target, false);
    
    // Compute difference
    std.debug.print("\nComputing loss directly on 3D tensors\n", .{});
    var diff = try autodiff.subtract(allocator, lookup_result, target_node);
    
    // Square the difference
    var diff_squared = try autodiff.multiply(allocator, diff, diff);
    
    // We need to sum up the gradients
    // Create a vector of ones with shape matching diff_squared's last two dimensions [seq_len, embed_dim]
    const ones_dims = [_]usize{lookup_result.tensor.shape.dims[1] * lookup_result.tensor.shape.dims[2], 1};
    const ones = try Tensor.filled(allocator, &ones_dims, .f32, 1.0, .cpu);
    var ones_node = try autodiff.variable(allocator, ones, false);
    
    // Sum over last dimension first
    var batch_sums = std.ArrayList(*autodiff.Node).init(allocator);
    defer batch_sums.deinit();
    
    // Loop through each batch element
    for (0..lookup_result.tensor.shape.dims[0]) |b| {
        // Extract this batch element (shape [seq_len, embed_dim])
        var batch_slice_dims = [_]usize{1, lookup_result.tensor.shape.dims[1], lookup_result.tensor.shape.dims[2]};
        var batch_slice = try Tensor.zeros(allocator, &batch_slice_dims, .f32, .cpu);
        
        const diff_squared_buf = ptrCastHelper([*]f32, diff_squared.tensor.buffer.data.ptr)[0..diff_squared.tensor.shape.elemCount()];
        const batch_slice_buf = ptrCastHelper([*]f32, batch_slice.buffer.data.ptr)[0..batch_slice.shape.elemCount()];
        
        // Copy data for this batch
        const start_idx = b * lookup_result.tensor.shape.dims[1] * lookup_result.tensor.shape.dims[2];
        const copy_len = lookup_result.tensor.shape.dims[1] * lookup_result.tensor.shape.dims[2];
        @memcpy(batch_slice_buf, diff_squared_buf[start_idx..start_idx+copy_len]);
        
        const batch_node = try autodiff.variable(allocator, batch_slice, true);
        try batch_sums.append(batch_node);
    }
    
    // Create a loss node for the mean
    // For simplicity, just use the first batch element's sum as the loss
    var loss = batch_sums.items[0];
    
    // Print loss
    std.debug.print("\nLoss:\n", .{});
    printTensor(loss.tensor);
    
    // Run backward pass
    std.debug.print("\nRunning backward pass...\n", .{});
    try autodiff.backward(allocator, loss);
    
    // Check embedding gradients
    std.debug.print("\nEmbedding gradients:\n", .{});
    if (embed_node.grad) |grad| {
        printTensor(grad);
    } else {
        std.debug.print("No gradient\n", .{});
    }
    
    // Cleanup 
    std.debug.print("\nCleaning up\n", .{});
    
    // Clean up batch nodes except loss which we'll clean up separately
    for (batch_sums.items) |batch_node| {
        if (batch_node != loss) {
            batch_node.deinit();
        }
    }
    
    loss.deinit();
    ones_node.deinit();
    diff_squared.deinit();
    diff.deinit();
    target_node.deinit();
    lookup_result.deinit();
    embed_node.deinit();
    
    // Since the indices were retained by the embedding_lookup, but lookup_result's deinit 
    // will now clean up any indices nodes, we do NOT need to release our original reference
    std.debug.print("Final indices ref_count before exiting: {}\n", .{indices.getRefCount()});
    // DO NOT CALL indices.deinit() here - that would cause a double-free
    
    std.debug.print("\nEmbedding test completed successfully\n", .{});
}

// Test a more complex model with connected nodes
fn testComplexModel(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Complex Model with Embedding Lookups ===\n", .{});
    
    // Create emulation of GPT-2 tokens with a small vocab (10) and small embedding dim (4)
    var wte_dims = [_]usize{ 10, 4 }; // Vocab size 10, embedding dim 4
    var wte = try Tensor.random(allocator, &wte_dims, .f32, .cpu);
    
    // Scale down random values
    const wte_ptr = ptrCastHelper([*]f32, wte.buffer.data.ptr)[0..wte.shape.elemCount()];
    for (wte_ptr) |*val| {
        val.* *= 0.1;
    }
    
    // Position embeddings: 5 positions, embedding dim 4
    var wpe_dims = [_]usize{ 5, 4 };
    var wpe = try Tensor.random(allocator, &wpe_dims, .f32, .cpu);
    
    // Scale down random values
    const wpe_ptr = ptrCastHelper([*]f32, wpe.buffer.data.ptr)[0..wpe.shape.elemCount()];
    for (wpe_ptr) |*val| {
        val.* *= 0.1;
    }
    
    // Create input: batch size 2, sequence length 3, with token IDs
    var input_dims = [_]usize{ 2, 3 };
    var input_ids = try Tensor.zeros(allocator, &input_dims, .f32, .cpu);
    
    // Set token IDs: [[1, 2, 3], [4, 5, 6]]
    try input_ids.setScalar(&[_]usize{0, 0}, 1.0);
    try input_ids.setScalar(&[_]usize{0, 1}, 2.0);
    try input_ids.setScalar(&[_]usize{0, 2}, 3.0);
    try input_ids.setScalar(&[_]usize{1, 0}, 4.0);
    try input_ids.setScalar(&[_]usize{1, 1}, 5.0);
    try input_ids.setScalar(&[_]usize{1, 2}, 6.0);
    
    // Create parameter nodes
    var wte_node = try autodiff.variable(allocator, wte, true);
    var wpe_node = try autodiff.variable(allocator, wpe, true);
    
    // Store parameter nodes for gradient inspection
    var model_params = std.ArrayList(*Node).init(allocator);
    try model_params.append(wte_node);
    try model_params.append(wpe_node);
    
    std.debug.print("Building forward pass with autodiff.embedding_lookup\n", .{});
    
    // Use embedding lookup for token embeddings
    var token_embeds = try autodiff.embedding_lookup(allocator, wte_node, input_ids);
    
    // Create position indices
    var pos_indices_dims = [_]usize{ 2, 3 };
    var pos_indices = try Tensor.zeros(allocator, &pos_indices_dims, .f32, .cpu);
    
    // Fill with position IDs
    for (0..2) |b| {
        for (0..3) |s| {
            try pos_indices.setScalar(&[_]usize{b, s}, @as(f32, @floatFromInt(s)));
        }
    }
    
    // Use embedding lookup for position embeddings
    var pos_embeds = try autodiff.embedding_lookup(allocator, wpe_node, pos_indices);
    
    // Reshape embeddings to [batch_size*seq_len, embed_dim]
    const batch_size = input_ids.shape.dims[0];
    const seq_len = input_ids.shape.dims[1];
    const embed_dim = wte_node.tensor.shape.dims[1];
    
    var flat_dims = [_]usize{ batch_size * seq_len, embed_dim };
    
    // Create flattened tensors
    var token_embeds_flat = try Tensor.zeros(allocator, &flat_dims, .f32, .cpu);
    var pos_embeds_flat = try Tensor.zeros(allocator, &flat_dims, .f32, .cpu);
    
    // Copy data
    const token_embeds_buf = ptrCastHelper([*]f32, token_embeds.tensor.buffer.data.ptr)[0..token_embeds.tensor.shape.elemCount()];
    const pos_embeds_buf = ptrCastHelper([*]f32, pos_embeds.tensor.buffer.data.ptr)[0..pos_embeds.tensor.shape.elemCount()];
    
    const token_flat_buf = ptrCastHelper([*]f32, token_embeds_flat.buffer.data.ptr)[0..token_embeds_flat.shape.elemCount()];
    const pos_flat_buf = ptrCastHelper([*]f32, pos_embeds_flat.buffer.data.ptr)[0..pos_embeds_flat.shape.elemCount()];
    
    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            for (0..embed_dim) |e| {
                const src_idx = (b * seq_len + s) * embed_dim + e;
                const dst_idx = (b * seq_len + s) * embed_dim + e;
                token_flat_buf[dst_idx] = token_embeds_buf[src_idx];
                pos_flat_buf[dst_idx] = pos_embeds_buf[src_idx];
            }
        }
    }
    
    // Create nodes for flattened tensors
    var token_embeds_flat_node = try autodiff.variable(allocator, token_embeds_flat, true);
    var pos_embeds_flat_node = try autodiff.variable(allocator, pos_embeds_flat, true);
    
    // Add token and position embeddings
    var combined = try autodiff.add(allocator, token_embeds_flat_node, pos_embeds_flat_node);
    
    // Create a small projection layer
    var proj_dims = [_]usize{ 4, 10 }; // Project back to vocab size
    var proj = try Tensor.random(allocator, &proj_dims, .f32, .cpu);
    
    // Scale down
    const proj_ptr = ptrCastHelper([*]f32, proj.buffer.data.ptr)[0..proj.shape.elemCount()];
    for (proj_ptr) |*val| {
        val.* *= 0.1;
    }
    
    var proj_node = try autodiff.variable(allocator, proj, true);
    
    // Store with other parameters
    try model_params.append(proj_node);
    
    // Apply projection: logits = combined @ proj
    var logits = try autodiff.matmul(allocator, combined, proj_node);
    
    // Apply softmax to get probabilities
    var probs = try autodiff.softmax(allocator, logits);
    
    // Create target tensor: one-hot
    var targets_dims = [_]usize{ batch_size * seq_len, 10 };
    var targets = try Tensor.zeros(allocator, &targets_dims, .f32, .cpu);
    
    // Set target values (just set position 0 to 1.0 for each example)
    const targets_buf = ptrCastHelper([*]f32, targets.buffer.data.ptr)[0..targets.shape.elemCount()];
    for (0..batch_size * seq_len) |i| {
        targets_buf[i * 10] = 1.0;
    }
    
    var targets_node = try autodiff.variable(allocator, targets, false);
    
    // Compute cross-entropy loss (simplified)
    // Just subtract targets from probs, square and sum
    var diff = try autodiff.subtract(allocator, probs, targets_node);
    var diff_squared = try autodiff.multiply(allocator, diff, diff);
    
    // Create a tensor of ones to use for reducing
    const ones_dims = [_]usize{ 10, 1 };
    const ones = try Tensor.filled(allocator, &ones_dims, .f32, 1.0, .cpu);
    var ones_node = try autodiff.variable(allocator, ones, false);
    
    // Sum through matrix multiplication with ones tensor
    var loss_per_example = try autodiff.matmul(allocator, diff_squared, ones_node);
    
    // Mean across examples - create with same shape as loss_per_example
    const mean_factor = try Tensor.filled(allocator, loss_per_example.tensor.shape.dims, .f32, 1.0 / @as(f32, @floatFromInt(batch_size * seq_len)), .cpu);
    var mean_node = try autodiff.variable(allocator, mean_factor, false);
    
    // Get scalar loss by element-wise multiplication
    var loss = try autodiff.multiply(allocator, loss_per_example, mean_node);
    
    // Print loss
    std.debug.print("\nLoss:\n", .{});
    printTensor(loss.tensor);
    
    // Run backward pass
    std.debug.print("\nRunning backward pass...\n", .{});
    try autodiff.backward(allocator, loss);
    
    // Check all parameter gradients
    for (model_params.items, 0..) |param, i| {
        std.debug.print("\nGradients for parameter {d}:\n", .{i});
        if (param.grad) |grad| {
            printTensor(grad);
        } else {
            std.debug.print("No gradient\n", .{});
        }
    }
    
    // Clean up
    std.debug.print("\nCleaning up\n", .{});
    
    // Check reference counts before cleanup
    std.debug.print("Before cleanup: input_ids ref_count={}, pos_indices ref_count={}\n", 
        .{input_ids.getRefCount(), pos_indices.getRefCount()});
    
    // Clean up parameters list
    model_params.deinit();
    
    // Clean up nodes (in reverse creation order)
    loss.deinit();
    mean_node.deinit();
    loss_per_example.deinit();
    ones_node.deinit();
    diff_squared.deinit();
    diff.deinit();
    targets_node.deinit();
    probs.deinit();
    logits.deinit();
    proj_node.deinit();
    combined.deinit();
    pos_embeds_flat_node.deinit();
    token_embeds_flat_node.deinit();
    pos_embeds.deinit(); 
    token_embeds.deinit();
    wpe_node.deinit();
    wte_node.deinit();
    
    // We should NOT explicitly free input_ids and pos_indices here
    // They're released through the embedding_lookup node's deinit
    // Just print the reference counts to verify they're still valid
    std.debug.print("Final ref counts: input_ids={}, pos_indices={}\n", 
        .{input_ids.getRefCount(), pos_indices.getRefCount()});
    
    std.debug.print("Tensors cleaned up successfully\n", .{});
    
    std.debug.print("\nComplex model test completed successfully\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("Starting autodiff debug tests\n", .{});
    
    // Test simple operations
    try testSimpleModel(allocator);
    
    // Test embedding functionality
    try testEmbeddings(allocator);
    
    // Test complex model with embedding lookups
    try testComplexModel(allocator);
    
    std.debug.print("\nAll tests completed successfully!\n", .{});
}