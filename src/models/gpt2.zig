const std = @import("std");
const pcp = @import("pcp");
const tensor = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;

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
        const c_attn_weight_ptr = @as([*]f32, @ptrCast(@alignCast(c_attn_weight.buffer.data.ptr)))[0..c_attn_weight.shape.elemCount()];
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
        const c_proj_weight_ptr = @as([*]f32, @ptrCast(@alignCast(c_proj_weight.buffer.data.ptr)))[0..c_proj_weight.shape.elemCount()];
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
        // Convert tensors to nodes
        const c_attn_weight_node = try autodiff.variable(self.allocator, self.c_attn_weight, true);
        defer c_attn_weight_node.deinit();
        
        const c_attn_bias_node = try autodiff.variable(self.allocator, self.c_attn_bias, true);
        defer c_attn_bias_node.deinit();
        
        const c_proj_weight_node = try autodiff.variable(self.allocator, self.c_proj_weight, true);
        defer c_proj_weight_node.deinit();
        
        const c_proj_bias_node = try autodiff.variable(self.allocator, self.c_proj_bias, true);
        defer c_proj_bias_node.deinit();
        
        // Get batch size from input shape
        const batch_size = x_node.tensor.shape.dims[0];
        
        // Step 1: Compute QKV projections: x @ c_attn_weight + c_attn_bias
        const qkv_proj = try autodiff.matmul(self.allocator, x_node, c_attn_weight_node);
        defer qkv_proj.deinit();
        
        // Add bias
        const qkv_bias_expanded = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, 3 * self.n_embd}, 
            x_node.tensor.dtype, 
            x_node.tensor.backend
        );
        defer qkv_bias_expanded.deinit();
        
        // Fill bias across batch dimension
        const qkv_bias_expanded_buf = @as([*]f32, @ptrCast(@alignCast(qkv_bias_expanded.buffer.data.ptr)))[0..qkv_bias_expanded.shape.elemCount()];
        const c_attn_bias_buf = @as([*]f32, @ptrCast(@alignCast(self.c_attn_bias.buffer.data.ptr)))[0..self.c_attn_bias.shape.elemCount()];
        
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
        const q_tensor = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, self.n_embd}, 
            qkv.tensor.dtype, 
            qkv.tensor.backend
        );
        defer q_tensor.deinit();
        
        const k_tensor = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, self.n_embd}, 
            qkv.tensor.dtype, 
            qkv.tensor.backend
        );
        defer k_tensor.deinit();
        
        const v_tensor = try Tensor.zeros(self.allocator, 
            &[_]usize{batch_size, self.n_embd}, 
            qkv.tensor.dtype, 
            qkv.tensor.backend
        );
        defer v_tensor.deinit();
        
        // Split the QKV tensor
        const qkv_buf = @as([*]f32, @ptrCast(@alignCast(qkv.tensor.buffer.data.ptr)))[0..qkv.tensor.shape.elemCount()];
        const q_buf = @as([*]f32, @ptrCast(@alignCast(q_tensor.buffer.data.ptr)))[0..q_tensor.shape.elemCount()];
        const k_buf = @as([*]f32, @ptrCast(@alignCast(k_tensor.buffer.data.ptr)))[0..k_tensor.shape.elemCount()];
        const v_buf = @as([*]f32, @ptrCast(@alignCast(v_tensor.buffer.data.ptr)))[0..v_tensor.shape.elemCount()];
        
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
        defer k_transpose.deinit();
        
        const k_transpose_node = try autodiff.variable(self.allocator, k_transpose, true);
        defer k_transpose_node.deinit();
        
        // Q @ K^T: [batch_size, n_embd] @ [n_embd, batch_size] -> [batch_size, batch_size]
        const attention_scores_raw = try autodiff.matmul(self.allocator, q_node, k_transpose_node);
        defer attention_scores_raw.deinit();
        
        // Scale by sqrt(head_dim)
        const scaling_factor = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));
        
        // Create a scaling tensor
        const scaling_tensor = try Tensor.filled(
            self.allocator, 
            &[_]usize{1, 1}, 
            attention_scores_raw.tensor.dtype, 
            scaling_factor,
            attention_scores_raw.tensor.backend
        );
        defer scaling_tensor.deinit();
        
        const scaling_node = try autodiff.variable(self.allocator, scaling_tensor, false);
        defer scaling_node.deinit();
        
        // Apply scaling via broadcast multiplication
        // For simplicity, we'll scale manually:
        const attention_scores_tensor = try Tensor.zeros(
            self.allocator,
            attention_scores_raw.tensor.shape.dims,
            attention_scores_raw.tensor.dtype,
            attention_scores_raw.tensor.backend
        );
        
        const scores_raw_buf = @as([*]f32, @ptrCast(@alignCast(attention_scores_raw.tensor.buffer.data.ptr)))[0..attention_scores_raw.tensor.shape.elemCount()];
        const scores_buf = @as([*]f32, @ptrCast(@alignCast(attention_scores_tensor.buffer.data.ptr)))[0..attention_scores_tensor.shape.elemCount()];
        
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
        const proj_bias_expanded = try Tensor.zeros(
            self.allocator,
            &[_]usize{batch_size, self.n_embd},
            output_raw.tensor.dtype,
            output_raw.tensor.backend
        );
        defer proj_bias_expanded.deinit();
        
        // Fill bias across batch dimension
        const proj_bias_expanded_buf = @as([*]f32, @ptrCast(@alignCast(proj_bias_expanded.buffer.data.ptr)))[0..proj_bias_expanded.shape.elemCount()];
        const c_proj_bias_buf = @as([*]f32, @ptrCast(@alignCast(self.c_proj_bias.buffer.data.ptr)))[0..self.c_proj_bias.shape.elemCount()];
        
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
        const c_fc_weight_ptr = @as([*]f32, @ptrCast(@alignCast(c_fc_weight.buffer.data.ptr)))[0..c_fc_weight.shape.elemCount()];
        for (c_fc_weight_ptr) |*val| {
            val.* *= config.initializer_range;
        }
        
        const c_fc_bias_dims = [_]usize{n_inner};
        const c_fc_bias = try Tensor.zeros(allocator, &c_fc_bias_dims, .f32, backend);
        
        // Projection weights and biases
        const c_proj_weight_dims = [_]usize{ n_inner, n_embd };
        const c_proj_weight = try Tensor.random(allocator, &c_proj_weight_dims, .f32, backend);
        
        // Scale the random initialization
        const c_proj_weight_ptr = @as([*]f32, @ptrCast(@alignCast(c_proj_weight.buffer.data.ptr)))[0..c_proj_weight.shape.elemCount()];
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
        // Convert tensors to nodes
        const c_fc_weight_node = try autodiff.variable(self.allocator, self.c_fc_weight, true);
        const c_fc_bias_node = try autodiff.variable(self.allocator, self.c_fc_bias, true);
        const c_proj_weight_node = try autodiff.variable(self.allocator, self.c_proj_weight, true);
        const c_proj_bias_node = try autodiff.variable(self.allocator, self.c_proj_bias, true);
        
        // MLP computation
        
        // For our simplified implementation, we'll directly create an output tensor with the same shape as the input
        var output_dims = [_]usize{ x_node.tensor.shape.dims[0], x_node.tensor.shape.dims[1] };
        var output_tensor = try Tensor.zeros(self.allocator, &output_dims, x_node.tensor.dtype, x_node.tensor.backend);
        
        // Fill with random small values (simplified for demo)
        const rand_seed = @as(u64, @bitCast(@as(i64, std.time.milliTimestamp())));
        var prng = std.rand.DefaultPrng.init(rand_seed);
        const rand = prng.random();
        
        const output_buf = @as([*]f32, @ptrCast(@alignCast(output_tensor.buffer.data.ptr)))[0..output_tensor.shape.elemCount()];
        for (output_buf) |*value| {
            value.* = rand.float(f32) * 0.1;
        }
        
        // Output tensor has the same shape as input
        
        // Create a node for the output tensor
        const output_node = try autodiff.variable(self.allocator, output_tensor, true);
        
        // Clean up intermediate nodes
        c_fc_weight_node.deinit();
        c_fc_bias_node.deinit(); 
        c_proj_weight_node.deinit();
        c_proj_bias_node.deinit();
        
        return output_node;
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
        // For now, we just scale by the weight (simplified)
        const weight_node = try autodiff.variable(self.allocator, self.weight, true);
        const bias_node = try autodiff.variable(self.allocator, self.bias, true);
        
        // Dummy: just pass through for now
        // In real implementation, this would normalize
        
        weight_node.deinit();
        bias_node.deinit();
        
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
        // Layer norm -> attention -> residual
        const norm1 = try self.ln_1.forward(x_node);
        const attn = try self.attn.forward(norm1);
        const res1 = try autodiff.add(self.allocator, x_node, attn);
        
        // Layer norm -> MLP -> residual
        const norm2 = try self.ln_2.forward(res1);
        const mlp = try self.mlp.forward(norm2);
        const res2 = try autodiff.add(self.allocator, res1, mlp);
        
        // Clean up intermediate nodes
        norm1.deinit();
        attn.deinit();
        res1.deinit();
        norm2.deinit();
        mlp.deinit();
        
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
    
    pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !GPT2 {
        // Token embeddings (vocab_size x n_embd)
        const wte_dims = [_]usize{ config.vocab_size, config.n_embd };
        const wte = try Tensor.random(allocator, &wte_dims, .f32, backend);
        
        // Scale the random initialization
        const wte_ptr = @as([*]f32, @ptrCast(@alignCast(wte.buffer.data.ptr)))[0..wte.shape.elemCount()];
        for (wte_ptr) |*val| {
            val.* *= config.initializer_range;
        }
        
        // Position embeddings (n_positions x n_embd)
        const wpe_dims = [_]usize{ config.n_positions, config.n_embd };
        const wpe = try Tensor.random(allocator, &wpe_dims, .f32, backend);
        
        // Scale the random initialization
        const wpe_ptr = @as([*]f32, @ptrCast(@alignCast(wpe.buffer.data.ptr)))[0..wpe.shape.elemCount()];
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
        };
    }
    
    pub fn deinit(self: *GPT2) void {
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
    pub fn forward(self: *GPT2, input_ids: Tensor) !*Node {
        // Assume input_ids is a batch of token IDs [batch, seq_len]
        const batch_size = input_ids.shape.dims[0];
        const seq_len = input_ids.shape.dims[1];
        
        // Step 1: Implement proper embedding lookup
        // Extract token IDs from the input tensor
        const input_buf = @as([*]f32, @ptrCast(@alignCast(input_ids.buffer.data.ptr)))[0..input_ids.shape.elemCount()];
        
        // Create a tensor to hold token embeddings: [batch_size, seq_len, embedding_dim]
        var token_embed_dims = [_]usize{ batch_size, seq_len, self.config.n_embd };
        var token_embeddings = try Tensor.zeros(self.allocator, &token_embed_dims, input_ids.dtype, input_ids.backend);
        defer token_embeddings.deinit();
        
        // Get the token embedding buffer
        const token_embed_buf = @as([*]f32, @ptrCast(@alignCast(token_embeddings.buffer.data.ptr)))[0..token_embeddings.shape.elemCount()];
        
        // Get the word embedding matrix (wte) buffer
        const wte_buf = @as([*]f32, @ptrCast(@alignCast(self.wte.buffer.data.ptr)))[0..self.wte.shape.elemCount()];
        
        // Lookup embeddings for each token ID
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                const token_id_f = input_buf[b * seq_len + s];
                const token_id = @min(@as(usize, @intFromFloat(token_id_f)), self.config.vocab_size - 1);
                
                // Copy embedding for this token into the token embeddings tensor
                for (0..self.config.n_embd) |e| {
                    token_embed_buf[(b * seq_len + s) * self.config.n_embd + e] = 
                        wte_buf[token_id * self.config.n_embd + e];
                }
            }
        }
        
        // Step 2: Add position embeddings
        // Create a position indices tensor: [seq_len]
        var pos_indices = try Tensor.zeros(self.allocator, &[_]usize{seq_len}, .f32, input_ids.backend);
        defer pos_indices.deinit();
        
        // Fill position indices
        const pos_indices_buf = @as([*]f32, @ptrCast(@alignCast(pos_indices.buffer.data.ptr)))[0..pos_indices.shape.elemCount()];
        for (0..seq_len) |i| {
            pos_indices_buf[i] = @floatFromInt(i);
        }
        
        // Create combined embeddings tensor: [batch_size, seq_len, embedding_dim]
        var embeddings = try Tensor.zeros(self.allocator, &token_embed_dims, input_ids.dtype, input_ids.backend);
        
        // Get position embeddings buffer
        const wpe_buf = @as([*]f32, @ptrCast(@alignCast(self.wpe.buffer.data.ptr)))[0..self.wpe.shape.elemCount()];
        const embeddings_buf = @as([*]f32, @ptrCast(@alignCast(embeddings.buffer.data.ptr)))[0..embeddings.shape.elemCount()];
        
        // Add token and position embeddings
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                const pos = @min(s, self.config.n_positions - 1);
                
                for (0..self.config.n_embd) |e| {
                    // Token embedding + Position embedding
                    embeddings_buf[(b * seq_len + s) * self.config.n_embd + e] = 
                        token_embed_buf[(b * seq_len + s) * self.config.n_embd + e] + 
                        wpe_buf[pos * self.config.n_embd + e];
                }
            }
        }
        
        // Reshape to [batch_size*seq_len, embedding_dim] for model processing
        var reshaped_dims = [_]usize{ batch_size * seq_len, self.config.n_embd };
        var reshaped_embeddings = try Tensor.zeros(self.allocator, &reshaped_dims, input_ids.dtype, input_ids.backend);
        
        // Copy data to reshaped tensor
        const reshaped_buf = @as([*]f32, @ptrCast(@alignCast(reshaped_embeddings.buffer.data.ptr)))[0..reshaped_embeddings.shape.elemCount()];
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
        const output = try self.ln_f.forward(current);
        
        // Clean up
        if (current != input_node) {
            current.deinit();
        }
        
        // Clean up (embeddings is consumed by reshaped_embeddings)
        embeddings.deinit();
        
        return output;
    }
};