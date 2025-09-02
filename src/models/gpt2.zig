const std = @import("std");
const hlo = @import("../mlir/dialects/stablehlo.zig");
const mlir = @import("../mlir.zig");
const ops = @import("../ops.zig");

const Allocator = std.mem.Allocator;
const Shape = @import("../tensor.zig").Shape;
const DType = @import("../tensor.zig").DType;
const MLIRBuilder = @import("../ops.zig").MLIRBuilder;
const Tensor = @import("../tensor.zig").Tensor(void);

pub const GPT2Config = struct {
    vocab_size: usize = 50257,
    n_positions: usize = 1024,
    n_embd: usize = 768,
    n_layer: usize = 12,
    n_head: usize = 12,
    layer_norm_epsilon: f32 = 1e-5,
    initializer_range: f32 = 0.02,
    
    pub fn nano() GPT2Config {
        return GPT2Config{
            .vocab_size = 50257,
            .n_positions = 128,
            .n_embd = 32,
            .n_layer = 2,
            .n_head = 2,
            .layer_norm_epsilon = 1e-5,
            .initializer_range = 0.02,
        };
    }
};

/// Count total number of parameters needed for the entire GPT-2 model
pub fn countTotalParameters(config: GPT2Config) usize {
    const n_embd = config.n_embd;
    const n_layer = config.n_layer;
    const vocab_size = config.vocab_size;
    const n_positions = config.n_positions;
    
    var total: usize = 0;
    
    // Token embeddings (vocab_size, n_embd)
    total += vocab_size * n_embd;
    
    // Position embeddings (n_positions, n_embd)
    total += n_positions * n_embd;
    
    // Per-layer parameters
    for (0..n_layer) |_| {
        // LayerNorm 1: weight (n_embd) + bias (n_embd)
        total += n_embd + n_embd;
        
        // Attention: c_attn_weight (n_embd, 3*n_embd) + c_attn_bias (3*n_embd) + 
        //            c_proj_weight (n_embd, n_embd) + c_proj_bias (n_embd)
        total += n_embd * 3 * n_embd + 3 * n_embd + n_embd * n_embd + n_embd;
        
        // LayerNorm 2: weight (n_embd) + bias (n_embd)
        total += n_embd + n_embd;
        
        // MLP: c_fc_weight (n_embd, 4*n_embd) + c_fc_bias (4*n_embd) +
        //      c_proj_weight (4*n_embd, n_embd) + c_proj_bias (n_embd)
        total += n_embd * 4 * n_embd + 4 * n_embd + 4 * n_embd * n_embd + n_embd;
    }
    
    // Final LayerNorm: weight (n_embd) + bias (n_embd)
    total += n_embd + n_embd;
    
    // Language model head (n_embd, vocab_size)
    total += n_embd * vocab_size;
    
    return total;
}

/// Create a list of all parameter shapes for the GPT-2 model in initialization order
pub fn getParameterShapes(allocator: Allocator, config: GPT2Config) ![][]i64 {
    const n_embd = config.n_embd;
    const n_layer = config.n_layer;
    const vocab_size = config.vocab_size;
    const n_positions = config.n_positions;
    
    var shapes = std.ArrayList([]i64).init(allocator);
    
    // Token embeddings (vocab_size, n_embd)
    try shapes.append(try allocator.dupe(i64, &[_]i64{ @intCast(vocab_size), @intCast(n_embd) }));
    
    // Position embeddings (n_positions, n_embd)
    try shapes.append(try allocator.dupe(i64, &[_]i64{ @intCast(n_positions), @intCast(n_embd) }));
    
    // Per-layer parameters
    for (0..n_layer) |_| {
        // LayerNorm 1: weight (n_embd) + bias (n_embd)
        try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(n_embd)}));
        try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(n_embd)}));
        
        // Attention: c_attn_weight (n_embd, 3*n_embd) + c_attn_bias (3*n_embd) + 
        //            c_proj_weight (n_embd, n_embd) + c_proj_bias (n_embd)
        try shapes.append(try allocator.dupe(i64, &[_]i64{ @intCast(n_embd), @intCast(3 * n_embd) }));
        try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(3 * n_embd)}));
        try shapes.append(try allocator.dupe(i64, &[_]i64{ @intCast(n_embd), @intCast(n_embd) }));
        try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(n_embd)}));
        
        // LayerNorm 2: weight (n_embd) + bias (n_embd)
        try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(n_embd)}));
        try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(n_embd)}));
        
        // MLP: c_fc_weight (n_embd, 4*n_embd) + c_fc_bias (4*n_embd) +
        //      c_proj_weight (4*n_embd, n_embd) + c_proj_bias (n_embd)
        try shapes.append(try allocator.dupe(i64, &[_]i64{ @intCast(n_embd), @intCast(4 * n_embd) }));
        try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(4 * n_embd)}));
        try shapes.append(try allocator.dupe(i64, &[_]i64{ @intCast(4 * n_embd), @intCast(n_embd) }));
        try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(n_embd)}));
    }
    
    // Final LayerNorm: weight (n_embd) + bias (n_embd)
    try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(n_embd)}));
    try shapes.append(try allocator.dupe(i64, &[_]i64{@intCast(n_embd)}));
    
    // Language model head (n_embd, vocab_size)
    try shapes.append(try allocator.dupe(i64, &[_]i64{ @intCast(n_embd), @intCast(vocab_size) }));
    
    return shapes.toOwnedSlice();
}

// Layer normalization implementation with correct variance calculation
pub fn LayerNorm(comptime DataType: type) type {
    _ = DataType;
    
    return struct {
        weight: Tensor,
        bias: Tensor,
        epsilon: f32,
        n_embd: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, config: GPT2Config, builder: *MLIRBuilder, params: *[]mlir.Value) !@This() {
            const n_embd = config.n_embd;
            const epsilon = config.layer_norm_epsilon;

            // Consume parameters from the slice
            if (params.len < 2) return error.InsufficientParameters;
            
            const weight = try builder.newTensor(params.*[0]);
            const bias = try builder.newTensor(params.*[1]);
            
            // Advance the parameter slice
            params.* = params.*[2..];

            return @This(){
                .weight = weight,
                .bias = bias,
                .epsilon = epsilon,
                .n_embd = n_embd,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.weight.deinit();
            self.bias.deinit();
        }

        /// Mathematically correct layer normalization implementation
        pub fn forward(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            const ctx = builder.ctx;
            
            // Input x has shape [batch, seq_len, n_embd]
            // We normalize along the last dimension (n_embd)
            const last_dim = x.shape.rank() - 1;
            const n_embd_size = x.shape.getDimension(last_dim);
            
            // Step 1: Calculate mean along last dimension
            // sum = stablehlo.reduce_sum(x, dimension=last_dim, keep_dims=true)
            const sum_tensor = try ops.reduceSum(builder, x, &[_]i64{@intCast(last_dim)}, true); // keep_dims=true for broadcasting
            
            // Create size constant and divide to get mean
            const size_const = try ops.constant(builder, @floatFromInt(n_embd_size), &.{}, mlir.Type.f32Type(ctx));
            
            const mean = try ops.divide(builder, sum_tensor, size_const);
            
            // Step 2: Calculate variance = mean(x^2) - mean(x)^2
            // x_squared = stablehlo.multiply(x, x)
            const x_squared = try ops.multiply(builder, x, x);
            
            // mean_x_squared = reduce_sum(x_squared) / size
            const sum_x_squared = try ops.reduceSum(builder, x_squared, &[_]i64{@intCast(last_dim)}, false);
            const mean_x_squared = try ops.divide(builder, sum_x_squared, size_const);
            
            // mean_squared = mean * mean
            const mean_squared = try ops.multiply(builder, mean, mean);
            
            // variance = mean_x_squared - mean_squared
            const variance = try ops.subtract(builder, mean_x_squared, mean_squared);
            
            // Step 3: Apply normalization
            // Add epsilon: variance_eps = variance + epsilon
            const epsilon_const = try ops.constant(builder, self.epsilon, &.{}, mlir.Type.f32Type(ctx));
            const variance_eps = try ops.add(builder, variance, epsilon_const);
            
            // rsqrt_var = rsqrt(variance_eps)
            const rsqrt_var = try ops.rsqrt(builder, variance_eps);
            
            // Center x: x_centered = x - mean
            const x_centered = try ops.subtract(builder, x, mean);
            
            // Normalize: x_norm = x_centered * rsqrt_var
            const x_norm = try ops.multiply(builder, x_centered, rsqrt_var);
            
            // Step 4: Apply scale and shift
            // Scale: x_scaled = x_norm * self.weight
            const x_scaled = try ops.multiply(builder, x_norm, self.weight);
            
            // Shift: output = x_scaled + self.bias
            const output = try ops.add(builder, x_scaled, self.bias);
            return output;
        }
    };
}

// Multi-head causal self-attention implementation
pub fn Attention(comptime DataType: type) type {
    _ = DataType;
    
    return struct {
        c_attn_weight: Tensor,
        c_attn_bias: Tensor,
        c_proj_weight: Tensor,
        c_proj_bias: Tensor,
        n_embd: usize,
        n_head: usize,
        head_dim: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, config: GPT2Config, builder: *MLIRBuilder, params: *[]mlir.Value) !@This() {
            const n_embd = config.n_embd;
            const n_head = config.n_head;
            const head_dim = n_embd / n_head;

            // Consume parameters from the slice
            if (params.len < 4) return error.InsufficientParameters;
            
            const c_attn_weight = try builder.newTensor(params.*[0]);
            const c_attn_bias = try builder.newTensor(params.*[1]);
            const c_proj_weight = try builder.newTensor(params.*[2]);
            const c_proj_bias = try builder.newTensor(params.*[3]);
            
            // Advance the parameter slice
            params.* = params.*[4..];

            return @This(){
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

        pub fn deinit(self: *@This()) void {
            self.c_attn_weight.deinit();
            self.c_attn_bias.deinit();
            self.c_proj_weight.deinit();
            self.c_proj_bias.deinit();
        }

        /// Complete multi-head causal self-attention implementation
        pub fn forward(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            const ctx = builder.ctx;
            
            // Input x has shape [B, T, C] where B=batch, T=sequence, C=n_embd
            const batch_size = x.shape.getDimension(0);
            const seq_len = x.shape.getDimension(1);
            const n_embd = @as(i64, @intCast(self.n_embd));
            const n_head = @as(i64, @intCast(self.n_head));
            const head_dim = @as(i64, @intCast(self.head_dim));
            
            // Step 1: QKV projection
            // qkv = (x @ c_attn_weight) + c_attn_bias  [B, T, 3*C]
            const qkv_proj = try ops.matmul(builder, x, self.c_attn_weight);
            
            const qkv = try ops.add(builder, qkv_proj, self.c_attn_bias);

            
            // Step 2: Split QKV into separate Q, K, V tensors
            // Each has shape [B, T, C]
            const slice_start_q = [_]i64{ 0, 0, 0 };
            const slice_limit_q = [_]i64{ batch_size, seq_len, n_embd };
            const slice_strides = [_]i64{ 1, 1, 1 };
            
            const q = try ops.slice(builder, qkv, &slice_start_q, &slice_limit_q, &slice_strides);

            
            const slice_start_k = [_]i64{ 0, 0, n_embd };
            const slice_limit_k = [_]i64{ batch_size, seq_len, 2 * n_embd };
            
            const k = try ops.slice(builder, qkv, &slice_start_k, &slice_limit_k, &slice_strides);

            
            const slice_start_v = [_]i64{ 0, 0, 2 * n_embd };
            const slice_limit_v = [_]i64{ batch_size, seq_len, 3 * n_embd };
            
            const v = try ops.slice(builder, qkv, &slice_start_v, &slice_limit_v, &slice_strides);

            
            // Step 3: Reshape for multi-head attention
            // Reshape from [B, T, C] to [B, T, n_head, head_dim]
            const head_shape = [_]i64{ batch_size, seq_len, n_head, head_dim };
            
            const q_reshaped = try ops.reshape(builder, q, &head_shape);

            
            const k_reshaped = try ops.reshape(builder, k, &head_shape);

            
            const v_reshaped = try ops.reshape(builder, v, &head_shape);

            
            // Transpose to [B, n_head, T, head_dim] for efficient batched matmul
            const head_transpose_perm = [_]i64{ 0, 2, 1, 3 };
            
            const q_heads = try ops.transpose(builder, q_reshaped, &head_transpose_perm);

            
            const k_heads = try ops.transpose(builder, k_reshaped, &head_transpose_perm);

            
            const v_heads = try ops.transpose(builder, v_reshaped, &head_transpose_perm);

            
            // Step 4: Calculate attention scores
            // k_transposed: [B, n_head, head_dim, T] for matmul
            const k_transpose_perm = [_]i64{ 0, 1, 3, 2 };
            const k_t = try ops.transpose(builder, k_heads, &k_transpose_perm);

            
            // scores = q @ k_t  [B, n_head, T, T]
            const scores = try ops.matmul(builder, q_heads, k_t);
            
            // Scale by 1/sqrt(head_dim)
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
            const scale_const = try builder.scalarConstant(scale);
            
            const scaled_scores = try ops.multiply(builder, scores, scale_const);

            
            // Step 5: Apply causal mask
            // Create triangular mask where mask[i, j] = i >= j (allow current and past)
            const iota_0 = try ops.iota(builder, &[_]i64{ seq_len, seq_len }, 0, mlir.Type.i32Type(ctx));

            
            const iota_1 = try ops.iota(builder, &[_]i64{ seq_len, seq_len }, 1, mlir.Type.i32Type(ctx));

            
            // mask = iota_0 >= iota_1 (true for lower triangular + diagonal)
            const mask_2d = try ops.compare(builder, iota_0, iota_1, hlo.CompareDirection.GE);

            
            // Broadcast mask to [B, n_head, T, T]
            const mask_broadcast_dims = [_]i64{ 2, 3 };
            const mask_target_shape = [_]i64{ batch_size, n_head, seq_len, seq_len };
            const mask = try ops.broadcastInDim(builder, mask_2d, &mask_target_shape, &mask_broadcast_dims);

            
            // Apply mask: select(mask, scores, -inf)
            const neg_inf = try builder.scalarConstant(-1e9);
            
            const masked_scores = try ops.select(builder, mask, scaled_scores, neg_inf);

            
            // Step 6: Apply softmax
            const attn_weights = try ops.softmax(builder, masked_scores);

            
            // Step 7: Apply attention to values
            // output = attn_weights @ v_heads  [B, n_head, T, head_dim]
            const attn_output = try ops.matmul(builder, attn_weights, v_heads);
            
            // Step 8: Transpose back and reshape
            // Transpose from [B, n_head, T, head_dim] to [B, T, n_head, head_dim]
            const output_transpose_perm = [_]i64{ 0, 2, 1, 3 };
            const output_transposed = try ops.transpose(builder, attn_output, &output_transpose_perm);

            
            // Reshape to [B, T, C]
            const output_shape = [_]i64{ batch_size, seq_len, n_embd };
            const output_reshaped = try ops.reshape(builder, output_transposed, &output_shape);

            
            // Step 9: Apply output projection
            // final_output = (output_reshaped @ c_proj_weight) + c_proj_bias
            const proj_output = try ops.matmul(builder, output_reshaped, self.c_proj_weight);
            
            const final_output = try ops.add(builder, proj_output, self.c_proj_bias);
            return final_output;
        }
    };
}

// MLP implementation with GELU activation
pub fn MLP(comptime DataType: type) type {
    _ = DataType;
    
    return struct {
        c_fc_weight: Tensor,
        c_fc_bias: Tensor,
        c_proj_weight: Tensor,
        c_proj_bias: Tensor,
        n_embd: usize,
        n_inner: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, config: GPT2Config, builder: *MLIRBuilder, params: *[]mlir.Value) !@This() {
            const n_embd = config.n_embd;
            const n_inner = 4 * n_embd;

            // Consume parameters from the slice
            if (params.len < 4) return error.InsufficientParameters;
            
            const c_fc_weight = try builder.newTensor(params.*[0]);
            const c_fc_bias = try builder.newTensor(params.*[1]);
            const c_proj_weight = try builder.newTensor(params.*[2]);
            const c_proj_bias = try builder.newTensor(params.*[3]);
            
            // Advance the parameter slice
            params.* = params.*[4..];

            return @This(){
                .c_fc_weight = c_fc_weight,
                .c_fc_bias = c_fc_bias,
                .c_proj_weight = c_proj_weight,
                .c_proj_bias = c_proj_bias,
                .n_embd = n_embd,
                .n_inner = n_inner,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.c_fc_weight.deinit();
            self.c_fc_bias.deinit();
            self.c_proj_weight.deinit();
            self.c_proj_bias.deinit();
        }

        /// GELU activation function: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        fn gelu(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            _ = self;
            
            // Constants for GELU approximation
            const sqrt_2_over_pi = try builder.scalarConstant(@sqrt(2.0 / std.math.pi));
            const coeff = try builder.scalarConstant(0.044715);
            const half = try builder.scalarConstant(0.5);
            const one = try builder.scalarConstant(1.0);
            
            // x^3
            const x_squared = try ops.multiply(builder, x, x);
            const x_cubed = try ops.multiply(builder, x_squared, x);
            
            // 0.044715 * x^3
            const coeff_x_cubed = try ops.multiply(builder, coeff, x_cubed);
            
            // x + 0.044715 * x^3
            const inner = try ops.add(builder, x, coeff_x_cubed);
            
            // sqrt(2/π) * (x + 0.044715 * x^3)
            const scaled_inner = try ops.multiply(builder, sqrt_2_over_pi, inner);
            
            // tanh(sqrt(2/π) * (x + 0.044715 * x^3))
            const tanh_result = try ops.tanh(builder, scaled_inner);
            
            // 1 + tanh(...)
            const one_plus_tanh = try ops.add(builder, one, tanh_result);
            
            // x * (1 + tanh(...))
            const x_times = try ops.multiply(builder, x, one_plus_tanh);
            
            // 0.5 * x * (1 + tanh(...))
            return ops.multiply(builder, half, x_times);
        }

        pub fn forward(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            // Step 1: First linear layer: x @ c_fc_weight + c_fc_bias
            const fc_proj = try ops.matmul(builder, x, self.c_fc_weight);
            const fc_biased = try ops.add(builder, fc_proj, self.c_fc_bias);

            
            // Step 2: Apply GELU activation
            const fc_activated = try self.gelu(fc_biased, builder);

            
            // Step 3: Second linear layer: activated @ c_proj_weight + c_proj_bias
            const proj_output = try ops.matmul(builder, fc_activated, self.c_proj_weight);
            const final_output = try ops.add(builder, proj_output, self.c_proj_bias);
            return final_output;
        }
    };
}

// Transformer block
pub fn Block(comptime DataType: type) type {
    return struct {
        ln_1: LayerNorm(DataType),
        attn: Attention(DataType),
        ln_2: LayerNorm(DataType),
        mlp: MLP(DataType),
        allocator: Allocator,

        pub fn init(allocator: Allocator, config: GPT2Config, builder: *MLIRBuilder, params: *[]mlir.Value) !@This() {
            const ln_1 = try LayerNorm(DataType).init(allocator, config, builder, params);
            const attn = try Attention(DataType).init(allocator, config, builder, params);
            const ln_2 = try LayerNorm(DataType).init(allocator, config, builder, params);
            const mlp = try MLP(DataType).init(allocator, config, builder, params);

            return @This(){
                .ln_1 = ln_1,
                .attn = attn,
                .ln_2 = ln_2,
                .mlp = mlp,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.ln_1.deinit();
            self.attn.deinit();
            self.ln_2.deinit();
            self.mlp.deinit();
        }

        pub fn forward(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            // Pre-norm architecture: LayerNorm -> Attention -> Add residual
            const norm1 = try self.ln_1.forward(x, builder);
            const attn_output = try self.attn.forward(norm1, builder);

            // Residual connection 1
            const res1 = try ops.add(builder, x, attn_output);

            // Pre-norm architecture: LayerNorm -> MLP -> Add residual
            const norm2 = try self.ln_2.forward(res1, builder);
            const mlp_output = try self.mlp.forward(norm2, builder);

            // Residual connection 2
            const res2 = try ops.add(builder, res1, mlp_output);
            return res2;
        }
    };
}

/// Correct cross-entropy loss implementation
pub fn crossEntropyLoss(comptime DataType: type) type {
    _ = DataType;
    
    return struct {
        pub fn run(allocator: Allocator, logits: Tensor, targets: Tensor, builder: *MLIRBuilder) !Tensor {
            _ = allocator;
            const ctx = builder.ctx;
            
            // logits: [B, T, V] where V = vocab_size
            // targets: [B, T] with integer token IDs
            
            const batch_size = logits.shape.getDimension(0);
            const seq_len = logits.shape.getDimension(1);
            const vocab_size = logits.shape.getDimension(2);
            
            // Step 1: Convert targets to one-hot encoding [B, T, V]
            // First convert targets to i32 type as required by one_hot
            const i32_type = mlir.Type.i32Type(ctx);
            const targets_shape = [_]i64{ batch_size, seq_len };
            const targets_i32_type = mlir.Type.rankedTensorType(ctx, &targets_shape, i32_type);
            const targets_i32 = try ops.convert(builder, targets, targets_i32_type);
            
            const targets_one_hot = try ops.oneHot(builder, targets_i32, vocab_size, 1.0, 0.0, 2, mlir.Type.f32Type(ctx));
            
            // Step 2: Compute log softmax
            // log_softmax(x) = x - log(sum(exp(x), axis=-1))
            const exp_logits = try ops.exp(builder, logits);
            const sum_exp = try ops.reduceSum(builder, exp_logits, &[_]i64{2}, true); // keep_dims for broadcasting
            const log_sum_exp = try ops.log(builder, sum_exp);
            const log_probs = try ops.subtract(builder, logits, log_sum_exp);

            // Step 3: Select correct log probabilities
            // Multiply log_probs by one-hot targets to zero out incorrect classes
            const selected_log_probs = try ops.multiply(builder, log_probs, targets_one_hot);

            // Sum over vocabulary dimension to get negative log likelihood per token
            const nll_per_token = try ops.reduceSum(builder, selected_log_probs, &[_]i64{2}, false);

            
            // Step 4: Compute mean loss
            // Negate to get positive loss values
            const neg_one = try ops.constant(builder, -1.0, &.{}, mlir.Type.f32Type(ctx));
            const loss_per_token = try ops.multiply(builder, nll_per_token, neg_one);

            // Sum all losses
            const total_loss = try ops.reduceSum(builder, loss_per_token, &[_]i64{0, 1}, false);
            
            // Divide by number of tokens to get mean loss
            const num_tokens = batch_size * seq_len;
            const num_tokens_const = try ops.constant(builder, @floatFromInt(num_tokens), &.{}, mlir.Type.f32Type(ctx));
            const mean_loss = try ops.divide(builder, total_loss, num_tokens_const);
            return mean_loss;
        }
    };
}

// Main GPT-2 model with trainable parameters
pub fn GPT2(comptime DataType: type) type {
    return struct {
        wte: Tensor, // Token embeddings
        wpe: Tensor, // Position embeddings
        blocks: []Block(DataType), // Transformer blocks
        ln_f: LayerNorm(DataType), // Final layer norm
        lm_head: Tensor, // Language modeling head

        config: GPT2Config,
        allocator: Allocator,

        pub fn init(allocator: Allocator, config: GPT2Config, builder: *MLIRBuilder, params: *[]mlir.Value) !@This() {
            // Consume token embeddings
            if (params.len < 2) return error.InsufficientParameters;
            const wte = try builder.newTensor(params.*[0]);
            const wpe = try builder.newTensor(params.*[1]);
            params.* = params.*[2..];

            // Create transformer blocks
            const blocks = try allocator.alloc(Block(DataType), config.n_layer);
            for (blocks) |*block| {
                block.* = try Block(DataType).init(allocator, config, builder, params);
            }

            // Final layer norm
            const ln_f = try LayerNorm(DataType).init(allocator, config, builder, params);

            // Language model head
            if (params.len < 1) return error.InsufficientParameters;
            const lm_head = try builder.newTensor(params.*[0]);
            params.* = params.*[1..];

            return @This(){
                .wte = wte,
                .wpe = wpe,
                .blocks = blocks,
                .ln_f = ln_f,
                .lm_head = lm_head,
                .config = config,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.wte.deinit();
            self.wpe.deinit();
            
            for (self.blocks) |*block| {
                block.deinit();
            }
            self.allocator.free(self.blocks);
            
            self.ln_f.deinit();
            self.lm_head.deinit();
        }

        pub fn forward(self: *@This(), input_ids: Tensor, builder: *MLIRBuilder) !Tensor {
            const ctx = builder.ctx;
            
            // Input validation
            const batch_size = input_ids.shape.getDimension(0);
            const seq_len = input_ids.shape.getDimension(1);
            
            if (seq_len > self.config.n_positions) {
                return error.SequenceTooLong;
            }
            
            // Step 1: Token embedding lookup [B, T] -> [B, T, C]
            const token_embeddings = try ops.gather(builder, self.wte, input_ids, 
                hlo.GatherDimensionNumbersAttribute{
                    .offset_dims = &.{2}, // The output dimensions that are not from the indices
                    .collapsed_slice_dims = &.{0}, // We are taking a slice from dimension 0 of the embedding table
                    .start_index_map = &.{0}, // Map dimension 0 of indices to dimension 0 of the table
                    .index_vector_dim = 1, // The indices are vectors along the last dimension
                }, &.{ 1, @intCast(self.config.n_embd) }); // Slice size: 1 token, full embedding dimension
            
            // Step 2: Position embeddings
            // Create position indices [0, 1, 2, ..., seq_len-1]
            const pos_indices = try ops.iota(builder, &[_]i64{seq_len}, 0, mlir.Type.i32Type(ctx));

            
            // Position embedding lookup [T] -> [T, C]
            const pos_emb_2d = try ops.gather(builder, self.wpe, pos_indices,
                hlo.GatherDimensionNumbersAttribute{
                    .offset_dims = &.{1}, // The output dimensions that are not from the indices
                    .collapsed_slice_dims = &.{0}, // We are taking a slice from dimension 0 of the position embedding table
                    .start_index_map = &.{0}, // Map dimension 0 of indices to dimension 0 of the table
                    .index_vector_dim = 0, // The indices are scalars (rank 1 tensor)
                }, &.{ 1, @intCast(self.config.n_embd) }); // Slice size: 1 position, full embedding dimension
            
            // Broadcast to [B, T, C]
            const pos_target_shape = [_]i64{ batch_size, seq_len, @intCast(self.config.n_embd) };
            const pos_broadcast_dims = [_]i64{ 1, 2 };
            const pos_embeddings = try ops.broadcastInDim(builder, pos_emb_2d, &pos_target_shape, &pos_broadcast_dims);

            
            // Add token and position embeddings
            const current = try ops.add(builder, token_embeddings, pos_embeddings);

            // Step 3: Process through transformer blocks
            var current_output = current;
            for (self.blocks) |*block| {
                const new_output = try block.forward(current_output, builder);
                current_output.deinit();
                current_output = new_output;
            }

            // Step 4: Final layer norm
            const final_hidden = try self.ln_f.forward(current_output, builder);
            current_output.deinit();

            // Step 5: Language model head projection
            return ops.matmul(builder, final_hidden, self.lm_head);
        }

        pub fn forwardWithLoss(self: *@This(), input_ids: Tensor, targets: Tensor, builder: *MLIRBuilder) !struct { logits: Tensor, loss: Tensor } {
            const logits = try self.forward(input_ids, builder);
            const CrossEntropyLoss = crossEntropyLoss(DataType);
            const loss = try CrossEntropyLoss.run(self.allocator, logits, targets, builder);
            
            return .{ .logits = logits, .loss = loss };
        }
    };
}

