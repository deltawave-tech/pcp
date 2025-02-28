const std = @import("std");
const pcp = @import("pcp");
const tensor = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;

/// Helper function for pointer casting
fn ptrCastHelper(comptime T: type, ptr: anytype) T {
    // We still need to use alignCast for pointers that require higher alignment
    return @ptrCast(@alignCast(ptr));
}

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const DType = tensor.DType;
const Shape = tensor.Shape;
const BackendType = tensor.BackendType;
const Node = autodiff.Node;

pub const GPT2Config = struct {
    vocab_size: usize = 50257,
    n_positions: usize = 1024,
    n_embd: usize = 768,
    n_layer: usize = 12,
    n_head: usize = 12,
    layer_norm_epsilon: f32 = 1e-5,
    initializer_range: f32 = 0.02,
};

// Attention layer implementation
pub const Attention = struct {
    // Weights and biases for queries, keys, values, and output projections
    c_attn_weight: Tensor,
    c_attn_bias: Tensor,
    c_proj_weight: Tensor,
    c_proj_bias: Tensor,
    
    // Configuration
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !Attention {
        const n_embd = config.n_embd;
        const n_head = config.n_head;
        const head_dim = n_embd / n_head;
        
        // QKV combined weight (n_embd x 3*n_embd)
        const c_attn_weight_dims = [_]usize{ n_embd, 3 * n_embd };
        const c_attn_weight = try Tensor.random(allocator, &c_attn_weight_dims, .f32, backend);
        
        // Scale the random initialization
        const c_attn_weight_ptr = ptrCastHelper([*]f32, c_attn_weight.buffer.data.ptr)[0..c_attn_weight.shape.elemCount()];
        for (c_attn_weight_ptr) |*val| {
            val.* *= config.initializer_range;
        }
        
        // QKV combined bias (3*n_embd)
        const c_attn_bias_dims = [_]usize{3 * n_embd};
        const c_attn_bias = try Tensor.zeros(allocator, &c_attn_bias_dims, .f32, backend);
        
        // Output projection weight (n_embd x n_embd)
        const c_proj_weight_dims = [_]usize{ n_embd, n_embd };
        const c_proj_weight = try Tensor.random(allocator, &c_proj_weight_dims, .f32, backend);
        
        // Scale the random initialization
        const c_proj_weight_ptr = ptrCastHelper([*]f32, c_proj_weight.buffer.data.ptr)[0..c_proj_weight.shape.elemCount()];
        for (c_proj_weight_ptr) |*val| {
            val.* *= config.initializer_range;
        }
        
        // Output projection bias (n_embd)
        const c_proj_bias_dims = [_]usize{n_embd};
        const c_proj_bias = try Tensor.zeros(allocator, &c_proj_bias_dims, .f32, backend);
        
        return Attention{
            .c_attn_weight = c_attn_weight,
            .c_attn_bias = c_attn_bias,
            .c_proj_weight = c_proj_weight,
            .c_proj_bias = c_proj_bias,
            .n_embd = n_embd,
            .n_head = n_head,
            .head_dim = head_dim,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Attention) void {
        self.c_attn_weight.deinit();
        self.c_attn_bias.deinit();
        self.c_proj_weight.deinit();
        self.c_proj_bias.deinit();
    }
    
    // Forward pass through the attention layer
    pub fn forward(self: *Attention, x_node: *Node) !*Node {
        // Make copies of the tensors before converting to nodes, since variable() takes ownership
        // of the tensor and will free it when the node is freed
        const c_attn_weight_copy = try Tensor.init(
            self.allocator, 
            self.c_attn_weight.shape.dims, 
            self.c_attn_weight.dtype, 
            self.c_attn_weight.backend
        );
        
        // Copy data
        @memcpy(
            c_attn_weight_copy.buffer.data,
            self.c_attn_weight.buffer.data[0..self.c_attn_weight.buffer.data.len]
        );
        
        const c_attn_bias_copy = try Tensor.init(
            self.allocator, 
            self.c_attn_bias.shape.dims, 
            self.c_attn_bias.dtype, 
            self.c_attn_bias.backend
        );
        @memcpy(
            c_attn_bias_copy.buffer.data,
            self.c_attn_bias.buffer.data[0..self.c_attn_bias.buffer.data.len]
        );
        
        const c_proj_weight_copy = try Tensor.init(
            self.allocator, 
            self.c_proj_weight.shape.dims, 
            self.c_proj_weight.dtype, 
            self.c_proj_weight.backend
        );
        @memcpy(
            c_proj_weight_copy.buffer.data,
            self.c_proj_weight.buffer.data[0..self.c_proj_weight.buffer.data.len]
        );
        
        const c_proj_bias_copy = try Tensor.init(
            self.allocator, 
            self.c_proj_bias.shape.dims, 
            self.c_proj_bias.dtype, 
            self.c_proj_bias.backend
        );
        @memcpy(
            c_proj_bias_copy.buffer.data,
            self.c_proj_bias.buffer.data[0..self.c_proj_bias.buffer.data.len]
        );
        
        // Convert tensors to nodes
        const c_attn_weight_node = try autodiff.variable(self.allocator, c_attn_weight_copy, true);
        defer c_attn_weight_node.deinit();
        
        const c_attn_bias_node = try autodiff.variable(self.allocator, c_attn_bias_copy, true);
        defer c_attn_bias_node.deinit();
        
        const c_proj_weight_node = try autodiff.variable(self.allocator, c_proj_weight_copy, true);
        defer c_proj_weight_node.deinit();
        
        const c_proj_bias_node = try autodiff.variable(self.allocator, c_proj_bias_copy, true);
        defer c_proj_bias_node.deinit();
        
        // Get batch size from input shape
        const batch_size = x_node.tensor.shape.dims[0];
        
        // Step 1: Compute QKV projections: x @ c_attn_weight + c_attn_bias
        const qkv_proj = try autodiff.matmul(self.allocator, x_node, c_attn_weight_node);
        defer qkv_proj.deinit();
        
        // Add bias
        var qkv_bias_expanded = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, 3 * self.n_embd}, 
            x_node.tensor.dtype, 
            x_node.tensor.backend
        );
        
        // Fill bias across batch dimension
        const qkv_bias_expanded_buf = ptrCastHelper([*]f32, qkv_bias_expanded.buffer.data.ptr)[0..qkv_bias_expanded.shape.elemCount()];
        const c_attn_bias_buf = ptrCastHelper([*]f32, self.c_attn_bias.buffer.data.ptr)[0..self.c_attn_bias.shape.elemCount()];
        
        for (0..batch_size) |b| {
            for (0..3 * self.n_embd) |i| {
                qkv_bias_expanded_buf[b * (3 * self.n_embd) + i] = c_attn_bias_buf[i];
            }
        }
        
        const qkv_bias_node = try autodiff.variable(self.allocator, qkv_bias_expanded, false);
        defer qkv_bias_node.deinit();
        
        const qkv = try autodiff.add(self.allocator, qkv_proj, qkv_bias_node);
        defer qkv.deinit();
        
        // Step 2: Split QKV into separate query, key, value tensors
        var q_tensor = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, self.n_embd}, 
            qkv.tensor.dtype, 
            qkv.tensor.backend
        );
        
        var k_tensor = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, self.n_embd}, 
            qkv.tensor.dtype, 
            qkv.tensor.backend
        );
        
        var v_tensor = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, self.n_embd}, 
            qkv.tensor.dtype, 
            qkv.tensor.backend
        );
        
        // Split the QKV tensor
        const qkv_buf = ptrCastHelper([*]f32, qkv.tensor.buffer.data.ptr)[0..qkv.tensor.shape.elemCount()];
        const q_buf = ptrCastHelper([*]f32, q_tensor.buffer.data.ptr)[0..q_tensor.shape.elemCount()];
        const k_buf = ptrCastHelper([*]f32, k_tensor.buffer.data.ptr)[0..k_tensor.shape.elemCount()];
        const v_buf = ptrCastHelper([*]f32, v_tensor.buffer.data.ptr)[0..v_tensor.shape.elemCount()];
        
        for (0..batch_size) |b| {
            for (0..self.n_embd) |i| {
                q_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + i];
                k_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + self.n_embd + i];
                v_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + 2 * self.n_embd + i];
            }
        }
        
        // Convert to nodes
        const q_node = try autodiff.variable(self.allocator, q_tensor, true);
        defer q_node.deinit();
        
        const k_node = try autodiff.variable(self.allocator, k_tensor, true);
        defer k_node.deinit();
        
        const v_node = try autodiff.variable(self.allocator, v_tensor, true);
        defer v_node.deinit();
        
        // Step 3: Reshape for multi-head attention: [batch_size, n_head, seq_len, head_dim]
        // For simplicity, we'll treat batch_size as batch_size*seq_len and reshape later
        
        // Step 4: Compute attention scores: Q @ K^T / sqrt(head_dim)
        // First transpose K: [batch_size, n_embd] -> [n_embd, batch_size]
        const k_transpose = try ops.transpose(self.allocator, k_node.tensor);
        
        const k_transpose_node = try autodiff.variable(self.allocator, k_transpose, true);
        defer k_transpose_node.deinit();
        
        // Q @ K^T: [batch_size, n_embd] @ [n_embd, batch_size] -> [batch_size, batch_size]
        const attention_scores_raw = try autodiff.matmul(self.allocator, q_node, k_transpose_node);
        defer attention_scores_raw.deinit();
        
        // Scale by sqrt(head_dim)
        const scaling_factor = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));
        
        // Apply scaling manually instead of creating additional nodes
        var attention_scores_tensor = try Tensor.zeros(
            self.allocator,
            attention_scores_raw.tensor.shape.dims,
            attention_scores_raw.tensor.dtype,
            attention_scores_raw.tensor.backend
        );
        
        const scores_raw_buf = ptrCastHelper([*]f32, attention_scores_raw.tensor.buffer.data.ptr)[0..attention_scores_raw.tensor.shape.elemCount()];
        const scores_buf = ptrCastHelper([*]f32, attention_scores_tensor.buffer.data.ptr)[0..attention_scores_tensor.shape.elemCount()];
        
        for (0..attention_scores_tensor.shape.elemCount()) |i| {
            scores_buf[i] = scores_raw_buf[i] * scaling_factor;
        }
        
        const attention_scores = try autodiff.variable(self.allocator, attention_scores_tensor, true);
        defer attention_scores.deinit();
        
        // Step 5: Apply softmax to get attention weights
        const attention_weights = try autodiff.softmax(self.allocator, attention_scores);
        defer attention_weights.deinit();
        
        // Step 6: Apply attention weights to values: [batch_size, batch_size] @ [batch_size, n_embd]
        const context = try autodiff.matmul(self.allocator, attention_weights, v_node);
        defer context.deinit();
        
        // Step 7: Apply output projection
        const output_raw = try autodiff.matmul(self.allocator, context, c_proj_weight_node);
        defer output_raw.deinit();
        
        // Expand bias for addition
        var proj_bias_expanded = try Tensor.zeros(
            self.allocator,
            &[_]usize{batch_size, self.n_embd},
            output_raw.tensor.dtype,
            output_raw.tensor.backend
        );
        
        // Fill bias across batch dimension
        const proj_bias_expanded_buf = ptrCastHelper([*]f32, proj_bias_expanded.buffer.data.ptr)[0..proj_bias_expanded.shape.elemCount()];
        const c_proj_bias_buf = ptrCastHelper([*]f32, self.c_proj_bias.buffer.data.ptr)[0..self.c_proj_bias.shape.elemCount()];
        
        for (0..batch_size) |b| {
            for (0..self.n_embd) |i| {
                proj_bias_expanded_buf[b * self.n_embd + i] = c_proj_bias_buf[i];
            }
        }
        
        const proj_bias_node = try autodiff.variable(self.allocator, proj_bias_expanded, false);
        defer proj_bias_node.deinit();
        
        // Add bias to get final output
        const output = try autodiff.add(self.allocator, output_raw, proj_bias_node);
        
        return output;
    }
};

// MLP (Feed-Forward) layer implementation
pub const MLP = struct {
    c_fc_weight: Tensor,
    c_fc_bias: Tensor,
    c_proj_weight: Tensor,
    c_proj_bias: Tensor,
    
    n_embd: usize,
    n_inner: usize,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !MLP {
        const n_embd = config.n_embd;
        const n_inner = 4 * n_embd; // Typically 4x the embedding dimension
        
        // FC weights and biases
        const c_fc_weight_dims = [_]usize{ n_embd, n_inner };
        const c_fc_weight = try Tensor.random(allocator, &c_fc_weight_dims, .f32, backend);
        
        // Scale the random initialization
        const c_fc_weight_ptr = ptrCastHelper([*]f32, c_fc_weight.buffer.data.ptr)[0..c_fc_weight.shape.elemCount()];
        for (c_fc_weight_ptr) |*val| {
            val.* *= config.initializer_range;
        }
        
        const c_fc_bias_dims = [_]usize{n_inner};
        const c_fc_bias = try Tensor.zeros(allocator, &c_fc_bias_dims, .f32, backend);
        
        // Projection weights and biases
        const c_proj_weight_dims = [_]usize{ n_inner, n_embd };
        const c_proj_weight = try Tensor.random(allocator, &c_proj_weight_dims, .f32, backend);
        
        // Scale the random initialization
        const c_proj_weight_ptr = ptrCastHelper([*]f32, c_proj_weight.buffer.data.ptr)[0..c_proj_weight.shape.elemCount()];
        for (c_proj_weight_ptr) |*val| {
            val.* *= config.initializer_range;
        }
        
        const c_proj_bias_dims = [_]usize{n_embd};
        const c_proj_bias = try Tensor.zeros(allocator, &c_proj_bias_dims, .f32, backend);
        
        return MLP{
            .c_fc_weight = c_fc_weight,
            .c_fc_bias = c_fc_bias,
            .c_proj_weight = c_proj_weight,
            .c_proj_bias = c_proj_bias,
            .n_embd = n_embd,
            .n_inner = n_inner,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MLP) void {
        self.c_fc_weight.deinit();
        self.c_fc_bias.deinit();
        self.c_proj_weight.deinit();
        self.c_proj_bias.deinit();
    }
    
    pub fn forward(self: *MLP, x_node: *Node) !*Node {
        // Make copies of the tensors before converting to nodes
        const c_fc_weight_copy = try Tensor.init(
            self.allocator, 
            self.c_fc_weight.shape.dims, 
            self.c_fc_weight.dtype, 
            self.c_fc_weight.backend
        );
        @memcpy(
            c_fc_weight_copy.buffer.data,
            self.c_fc_weight.buffer.data[0..self.c_fc_weight.buffer.data.len]
        );
        
        const c_fc_bias_copy = try Tensor.init(
            self.allocator, 
            self.c_fc_bias.shape.dims, 
            self.c_fc_bias.dtype, 
            self.c_fc_bias.backend
        );
        @memcpy(
            c_fc_bias_copy.buffer.data,
            self.c_fc_bias.buffer.data[0..self.c_fc_bias.buffer.data.len]
        );
        
        const c_proj_weight_copy = try Tensor.init(
            self.allocator, 
            self.c_proj_weight.shape.dims, 
            self.c_proj_weight.dtype, 
            self.c_proj_weight.backend
        );
        @memcpy(
            c_proj_weight_copy.buffer.data,
            self.c_proj_weight.buffer.data[0..self.c_proj_weight.buffer.data.len]
        );
        
        const c_proj_bias_copy = try Tensor.init(
            self.allocator, 
            self.c_proj_bias.shape.dims, 
            self.c_proj_bias.dtype, 
            self.c_proj_bias.backend
        );
        @memcpy(
            c_proj_bias_copy.buffer.data,
            self.c_proj_bias.buffer.data[0..self.c_proj_bias.buffer.data.len]
        );
        
        // Convert tensors to nodes
        const c_fc_weight_node = try autodiff.variable(self.allocator, c_fc_weight_copy, true);
        defer c_fc_weight_node.deinit();
        
        const c_fc_bias_node = try autodiff.variable(self.allocator, c_fc_bias_copy, true);
        defer c_fc_bias_node.deinit();
        
        const c_proj_weight_node = try autodiff.variable(self.allocator, c_proj_weight_copy, true);
        defer c_proj_weight_node.deinit();
        
        const c_proj_bias_node = try autodiff.variable(self.allocator, c_proj_bias_copy, true);
        defer c_proj_bias_node.deinit();
        
        // MLP computation:
        // 1. Linear(x, c_fc_weight, c_fc_bias)
        // 2. GELU activation
        // 3. Linear(result, c_proj_weight, c_proj_bias)
        
        // Step 1: First linear layer: x @ c_fc_weight + c_fc_bias
        const fc_out = try autodiff.matmul(self.allocator, x_node, c_fc_weight_node);
        defer fc_out.deinit();
        
        // Add bias
        const batch_size = x_node.tensor.shape.dims[0];
        
        // Expand bias for batch dimension
        var fc_bias_expanded = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, self.n_inner}, 
            x_node.tensor.dtype, 
            x_node.tensor.backend
        );
        
        // Fill bias across batch dimension
        const fc_bias_expanded_buf = ptrCastHelper([*]f32, fc_bias_expanded.buffer.data.ptr)[0..fc_bias_expanded.shape.elemCount()];
        const c_fc_bias_buf = ptrCastHelper([*]f32, self.c_fc_bias.buffer.data.ptr)[0..self.c_fc_bias.shape.elemCount()];
        
        for (0..batch_size) |b| {
            for (0..self.n_inner) |i| {
                fc_bias_expanded_buf[b * self.n_inner + i] = c_fc_bias_buf[i];
            }
        }
        
        const fc_bias_node = try autodiff.variable(self.allocator, fc_bias_expanded, false);
        defer fc_bias_node.deinit();
        
        const fc_biased = try autodiff.add(self.allocator, fc_out, fc_bias_node);
        defer fc_biased.deinit();
        
        // Step 2: Apply activation (using ReLU for simplicity, could be GELU)
        const fc_activated = try autodiff.relu(self.allocator, fc_biased);
        defer fc_activated.deinit();
        
        // Step 3: Second linear layer: activated @ c_proj_weight + c_proj_bias
        const proj_out = try autodiff.matmul(self.allocator, fc_activated, c_proj_weight_node);
        defer proj_out.deinit();
        
        // Add bias
        var proj_bias_expanded = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, self.n_embd}, 
            x_node.tensor.dtype, 
            x_node.tensor.backend
        );
        
        // Fill bias across batch dimension
        const proj_bias_expanded_buf = ptrCastHelper([*]f32, proj_bias_expanded.buffer.data.ptr)[0..proj_bias_expanded.shape.elemCount()];
        const c_proj_bias_buf = ptrCastHelper([*]f32, self.c_proj_bias.buffer.data.ptr)[0..self.c_proj_bias.shape.elemCount()];
        
        for (0..batch_size) |b| {
            for (0..self.n_embd) |i| {
                proj_bias_expanded_buf[b * self.n_embd + i] = c_proj_bias_buf[i];
            }
        }
        
        const proj_bias_node = try autodiff.variable(self.allocator, proj_bias_expanded, false);
        defer proj_bias_node.deinit();
        
        const output = try autodiff.add(self.allocator, proj_out, proj_bias_node);
        
        return output;
    }
};

// Layer normalization implementation
pub const LayerNorm = struct {
    weight: Tensor,
    bias: Tensor,
    epsilon: f32,
    n_embd: usize,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !LayerNorm {
        const n_embd = config.n_embd;
        const epsilon = config.layer_norm_epsilon;
        
        const weight_dims = [_]usize{n_embd};
        const weight = try Tensor.filled(allocator, &weight_dims, .f32, 1.0, backend);
        
        const bias_dims = [_]usize{n_embd};
        const bias = try Tensor.zeros(allocator, &bias_dims, .f32, backend);
        
        return LayerNorm{
            .weight = weight,
            .bias = bias,
            .epsilon = epsilon,
            .n_embd = n_embd,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *LayerNorm) void {
        self.weight.deinit();
        self.bias.deinit();
    }
    
    // Note: In a real implementation, layer norm would be more complex
    // This is a simplified placeholder
    pub fn forward(self: *LayerNorm, x_node: *Node) !*Node {
        // Real layer norm would compute mean and variance
        // For now, we just pass through the input (simplified)
        
        // We don't actually use weight and bias in this simplified implementation,
        // and we're just returning the input node directly, so no need to make copies
        
        _ = self; // Use self parameter to avoid unused parameter warning
        return x_node;
    }
};

// Transformer layer (attention + MLP)
pub const Block = struct {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: MLP,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !Block {
        const ln_1 = try LayerNorm.init(allocator, config, backend);
        const attn = try Attention.init(allocator, config, backend);
        const ln_2 = try LayerNorm.init(allocator, config, backend);
        const mlp = try MLP.init(allocator, config, backend);
        
        return Block{
            .ln_1 = ln_1,
            .attn = attn,
            .ln_2 = ln_2,
            .mlp = mlp,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Block) void {
        self.ln_1.deinit();
        self.attn.deinit();
        self.ln_2.deinit();
        self.mlp.deinit();
    }
    
    pub fn forward(self: *Block, x_node: *Node) !*Node {
        // With reference counting, we don't need to worry as much about manual memory management
        // Each tensor's reference count will be incremented when nodes are created
        // and decremented when nodes are freed
        std.debug.print("Block.forward: Starting with input shape [{}, {}]\n", 
            .{x_node.tensor.shape.dims[0], x_node.tensor.shape.dims[1]});
        
        // Step 1: Layer norm -> attention
        // In our simplified implementation, LayerNorm just passes through the input
        const norm1 = try self.ln_1.forward(x_node);
        std.debug.print("Block.forward: After ln_1.forward\n", .{});
        
        // Step 2: Apply attention to normalized input
        const attn = try self.attn.forward(norm1);
        std.debug.print("Block.forward: After attn.forward\n", .{});
        
        // Step 3: Add attention output to input (residual connection)
        // With reference counting, we don't need to worry about double-freeing
        var res1 = try autodiff.add(self.allocator, x_node, attn);
        std.debug.print("Block.forward: After first residual connection\n", .{});
        
        // Step 4: We can still clean up intermediate nodes when we're done with them
        // The attn node will be freed, and its tensor's reference count decremented
        attn.deinit();
        std.debug.print("Block.forward: Cleaned up attention output node\n", .{});
        
        // Step 5: Layer norm on residual output
        const norm2 = try self.ln_2.forward(res1);
        std.debug.print("Block.forward: After ln_2.forward\n", .{});
        
        // Step 6: Apply MLP to normalized residual
        const mlp = try self.mlp.forward(norm2);
        std.debug.print("Block.forward: After mlp.forward\n", .{});
        
        // Step 7: Add MLP output to residual (second residual connection)
        const res2 = try autodiff.add(self.allocator, res1, mlp);
        std.debug.print("Block.forward: After second residual connection\n", .{});
        
        // Step 8: Clean up MLP output after it's been used in res2
        // This decrements the ref count of the tensor in the mlp node
        mlp.deinit();
        std.debug.print("Block.forward: Cleaned up MLP output node\n", .{});
        
        // Step 9: Since we're done with res1, clean it up
        // This decrements the ref count of the tensor in the res1 node
        res1.deinit();
        std.debug.print("Block.forward: Cleaned up res1 node\n", .{});
        
        // With reference counting, our layer norm implementation still passes through
        // the input, but we no longer need to worry about who "owns" the tensor
        // All tensors are freed when their ref count drops to zero
        
        std.debug.print("Block.forward: Returning res2 with shape [{}, {}]\n", 
            .{res2.tensor.shape.dims[0], res2.tensor.shape.dims[1]});
        return res2;
    }
};

// Main GPT-2 model
pub const GPT2 = struct {
    wte: Tensor, // Token embeddings
    wpe: Tensor, // Position embeddings
    blocks: []Block, // Transformer blocks
    ln_f: LayerNorm, // Final layer norm
    
    config: GPT2Config,
    allocator: Allocator,
    
    // Parameter nodes for gradient tracking
    wte_node: ?*Node, 
    wpe_node: ?*Node,
    
    pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !GPT2 {
        // Token embeddings (vocab_size x n_embd)
        const wte_dims = [_]usize{ config.vocab_size, config.n_embd };
        const wte = try Tensor.random(allocator, &wte_dims, .f32, backend);
        
        // Scale the random initialization
        const wte_ptr = ptrCastHelper([*]f32, wte.buffer.data.ptr)[0..wte.shape.elemCount()];
        for (wte_ptr) |*val| {
            val.* *= config.initializer_range;
        }
        
        // Position embeddings (n_positions x n_embd)
        const wpe_dims = [_]usize{ config.n_positions, config.n_embd };
        const wpe = try Tensor.random(allocator, &wpe_dims, .f32, backend);
        
        // Scale the random initialization
        const wpe_ptr = ptrCastHelper([*]f32, wpe.buffer.data.ptr)[0..wpe.shape.elemCount()];
        for (wpe_ptr) |*val| {
            val.* *= config.initializer_range;
        }
        
        // Create transformer blocks
        const blocks = try allocator.alloc(Block, config.n_layer);
        for (blocks) |*block| {
            block.* = try Block.init(allocator, config, backend);
        }
        
        // Final layer norm
        const ln_f = try LayerNorm.init(allocator, config, backend);
        
        return GPT2{
            .wte = wte,
            .wpe = wpe,
            .blocks = blocks,
            .ln_f = ln_f,
            .config = config,
            .allocator = allocator,
            .wte_node = null,
            .wpe_node = null,
        };
    }
    
    pub fn deinit(self: *GPT2) void {
        // Clean up parameter nodes if they exist
        if (self.wte_node) |node| {
            node.deinit();
        }
        
        if (self.wpe_node) |node| {
            node.deinit();
        }
        
        // Clean up tensors
        self.wte.deinit();
        self.wpe.deinit();
        
        for (self.blocks) |*block| {
            block.deinit();
        }
        self.allocator.free(self.blocks);
        
        self.ln_f.deinit();
    }
    
    // Forward pass with the model
    // Note: This is greatly simplified - in a real implementation 
    // we would need to handle positions, attention masks, etc.
    // With reference counting, we don't need to worry as much about explicitly freeing tensors
    pub fn forward(self: *GPT2, input_ids: Tensor) !*Node {
        // Safety check for absurdly large dimensions
        if (input_ids.shape.dims[0] > 10000 or input_ids.shape.dims[1] > 10000) {
            return error.InputDimensionsTooLarge;
        }
        
        // Assume input_ids is a batch of token IDs [batch, seq_len]
        const batch_size = input_ids.shape.dims[0];
        const seq_len = input_ids.shape.dims[1];
        
        // Step 0: Create nodes for model parameters that need gradients
        // Create embedding nodes only if they don't exist yet
        if (self.wte_node == null) {
            // Important: These nodes use the actual model parameters (not copies)
            // so gradients will accumulate in the real model parameters
            self.wte_node = try autodiff.variable(self.allocator, self.wte, true);
            std.debug.print("Created wte_node for gradient tracking\n", .{});
        }
        
        if (self.wpe_node == null) {
            self.wpe_node = try autodiff.variable(self.allocator, self.wpe, true);
            std.debug.print("Created wpe_node for gradient tracking\n", .{});
        }
        
        // Extract token IDs from the input tensor
        const input_buf = ptrCastHelper([*]f32, input_ids.buffer.data.ptr)[0..input_ids.shape.elemCount()];
        
        // Create a tensor to hold token embeddings: [batch_size, seq_len, embedding_dim]
        var token_embed_dims = [_]usize{ batch_size, seq_len, self.config.n_embd };
        var token_embeddings = try Tensor.zeros(self.allocator, &token_embed_dims, input_ids.dtype, input_ids.backend);
        defer token_embeddings.deinit();
        
        // Get the token embedding buffer
        const token_embed_buf = ptrCastHelper([*]f32, token_embeddings.buffer.data.ptr)[0..token_embeddings.shape.elemCount()];
        
        // Get embeddings directly from the node's tensor 
        const wte_buf = ptrCastHelper([*]f32, self.wte_node.?.tensor.buffer.data.ptr);
        
        // Lookup embeddings for each token ID
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                const token_id_f = input_buf[b * seq_len + s];
                const token_id = @min(@as(usize, @intFromFloat(token_id_f)), self.config.vocab_size - 1);
                
                // Calculate offsets safely without creating full slices
                const token_embed_offset = (b * seq_len + s) * self.config.n_embd;
                const wte_offset = token_id * self.config.n_embd;
                
                // Copy embedding for this token into the token embeddings tensor
                for (0..self.config.n_embd) |e| {
                    token_embed_buf[token_embed_offset + e] = wte_buf[wte_offset + e];
                }
            }
        }
        
        // Step 2: Add position embeddings
        // Create position indices tensor: [seq_len]
        var pos_indices = try Tensor.zeros(self.allocator, &[_]usize{seq_len}, .f32, input_ids.backend);
        defer pos_indices.deinit();
        
        // Fill position indices
        const pos_indices_buf = ptrCastHelper([*]f32, pos_indices.buffer.data.ptr)[0..pos_indices.shape.elemCount()];
        for (0..seq_len) |i| {
            pos_indices_buf[i] = @floatFromInt(i);
        }
        
        // Create combined embeddings tensor: [batch_size, seq_len, embedding_dim]
        var embeddings = try Tensor.zeros(self.allocator, &token_embed_dims, input_ids.dtype, input_ids.backend);
        defer embeddings.deinit();
        
        // Get position embeddings buffer directly from the node
        const wpe_buf = ptrCastHelper([*]f32, self.wpe_node.?.tensor.buffer.data.ptr);
        const embeddings_buf = ptrCastHelper([*]f32, embeddings.buffer.data.ptr)[0..embeddings.shape.elemCount()];
        
        // Add token and position embeddings
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                const pos = @min(s, self.config.n_positions - 1);
                // Calculate offsets safely
                const embed_offset = (b * seq_len + s) * self.config.n_embd;
                const wpe_offset = pos * self.config.n_embd;
                
                for (0..self.config.n_embd) |e| {
                    // Token embedding + Position embedding
                    embeddings_buf[embed_offset + e] = 
                        token_embed_buf[embed_offset + e] + 
                        wpe_buf[wpe_offset + e];
                }
            }
        }
        
        // Reshape to [batch_size*seq_len, embedding_dim] for model processing
        var reshaped_dims = [_]usize{ batch_size * seq_len, self.config.n_embd };
        var reshaped_embeddings = try Tensor.zeros(self.allocator, &reshaped_dims, input_ids.dtype, input_ids.backend);
        
        // Copy data to reshaped tensor
        const reshaped_buf = ptrCastHelper([*]f32, reshaped_embeddings.buffer.data.ptr)[0..reshaped_embeddings.shape.elemCount()];
        for (0..batch_size * seq_len) |i| {
            for (0..self.config.n_embd) |e| {
                reshaped_buf[i * self.config.n_embd + e] = embeddings_buf[i * self.config.n_embd + e];
            }
        }
        
        // Convert embedded tensor to node
        const input_node = try autodiff.variable(self.allocator, reshaped_embeddings, true);
        
        // Keep track of our node for cleanup
        var current = input_node;
        
        // Process through transformer blocks
        for (self.blocks) |*block| {
            const new_output = try block.forward(current);
            
            // Update current and clean up old one if needed
            if (current != input_node) {
                current.deinit();
            }
            current = new_output;
        }
        
        // Final layer norm
        const ln_output = try self.ln_f.forward(current);
        
        // Apply language modeling head to get logits
        // We use the actual word embedding parameters (transposed) as the output layer
        // This ensures parameter sharing between input embeddings and output layer
        
        // Create a tensor for the transposed embedding matrix
        var final_wte_dims = [_]usize{ self.config.n_embd, self.config.vocab_size };
        var final_wte = try Tensor.zeros(self.allocator, &final_wte_dims, input_ids.dtype, input_ids.backend);
        // Don't defer final_wte.deinit() here, as ownership will transfer to the variable node
        
        // Copy data from the full embeddings (transposed)
        const final_wte_buf = ptrCastHelper([*]f32, final_wte.buffer.data.ptr)[0..final_wte.shape.elemCount()];
        const orig_wte_buf = ptrCastHelper([*]f32, self.wte_node.?.tensor.buffer.data.ptr);
        
        // Copy transposed version of the weight matrix
        for (0..self.config.n_embd) |e| {
            for (0..self.config.vocab_size) |v| {
                // Embed dimension first, vocab second (transposed from original)
                final_wte_buf[e * self.config.vocab_size + v] = orig_wte_buf[v * self.config.n_embd + e];
            }
        }
        
        // Create a node for this final projection matrix (shares weights with embedding)
        // autodiff.variable() takes ownership of the tensor
        const wte_t_node = try autodiff.variable(self.allocator, final_wte, true);
        defer wte_t_node.deinit();
        
        // Project hidden states to vocabulary logits
        const logits = try autodiff.matmul(self.allocator, ln_output, wte_t_node);
        
        // Cleanup intermediate nodes
        if (ln_output != logits) {
            ln_output.deinit();
        }
        
        if (current != input_node and current != ln_output) {
            current.deinit();
        }
        
        if (input_node != current) {
            input_node.deinit();
        }
        
        // Clean up the input_ids tensor - with ref counting this simply decrements the ref count
        input_ids.deinit();
        
        return logits;
    }
};