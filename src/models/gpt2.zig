const std = @import("std");
const pcp = @import("pcp");
const hlo = pcp.mlir.dialects.stablehlo;
const mlir = @import("../mlir.zig");

const Allocator = std.mem.Allocator;
const Shape = pcp.tensor.Shape;
const DType = pcp.tensor.DType;
const MLIRBuilder = pcp.ops.MLIRBuilder;
const Tensor = pcp.tensor.Tensor(void);

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
            const loc = builder.loc;
            
            // Input x has shape [batch, seq_len, n_embd]
            // We normalize along the last dimension (n_embd)
            const last_dim = x.shape.rank() - 1;
            const n_embd_size = x.shape.getDimension(last_dim);
            
            // Step 1: Calculate mean along last dimension
            // sum = stablehlo.reduce_sum(x, dimension=last_dim, keep_dims=true)
            var sum_tensor = try builder.createAndAppendOp(hlo.reduce_sum(ctx, x.value, &[_]i64{last_dim}, true, loc));
            
            // Create size constant and divide to get mean
            const size_const = try builder.scalarConstant(@floatFromInt(n_embd_size));
            
            var mean = try builder.createAndAppendOp(hlo.divide(ctx, sum_tensor.value, size_const.value, loc));
            
            // Step 2: Calculate variance = mean(x^2) - mean(x)^2
            // x_squared = stablehlo.multiply(x, x)
            const x_squared_op = hlo.multiply(ctx, x.value, x.value, loc);
            var x_squared = try builder.newTensor(x_squared_op.getResult(0));

            
            // mean_x_squared = reduce_sum(x_squared) / size
            const sum_x_squared_op = hlo.reduce_sum(ctx, x_squared.value, &[_]i64{last_dim}, true, loc);
            var sum_x_squared = try builder.newTensor(sum_x_squared_op.getResult(0));

            
            const mean_x_squared_op = hlo.divide(ctx, sum_x_squared.value, size_const.value, loc);
            var mean_x_squared = try builder.newTensor(mean_x_squared_op.getResult(0));

            
            // mean_squared = mean * mean
            const mean_squared_op = hlo.multiply(ctx, mean.value, mean.value, loc);
            var mean_squared = try builder.newTensor(mean_squared_op.getResult(0));

            
            // variance = mean_x_squared - mean_squared
            const variance_op = hlo.subtract(ctx, mean_x_squared.value, mean_squared.value, loc);
            var variance = try builder.newTensor(variance_op.getResult(0));

            
            // Step 3: Apply normalization
            // Add epsilon: variance_eps = variance + epsilon
            const epsilon_const = try builder.scalarConstant(self.epsilon);
            
            const variance_eps_op = hlo.add(ctx, variance.value, epsilon_const.value, loc);
            var variance_eps = try builder.newTensor(variance_eps_op.getResult(0));

            
            // rsqrt_var = rsqrt(variance_eps)
            const rsqrt_var_op = hlo.rsqrt(ctx, variance_eps.value, loc);
            var rsqrt_var = try builder.newTensor(rsqrt_var_op.getResult(0));

            
            // Center x: x_centered = x - mean
            const x_centered_op = hlo.subtract(ctx, x.value, mean.value, loc);
            var x_centered = try builder.newTensor(x_centered_op.getResult(0));

            
            // Normalize: x_norm = x_centered * rsqrt_var
            const x_norm_op = hlo.multiply(ctx, x_centered.value, rsqrt_var.value, loc);
            var x_norm = try builder.newTensor(x_norm_op.getResult(0));

            
            // Step 4: Apply scale and shift
            // Scale: x_scaled = x_norm * self.weight
            const x_scaled_op = hlo.multiply(ctx, x_norm.value, self.weight.value, loc);
            var x_scaled = try builder.newTensor(x_scaled_op.getResult(0));

            
            // Shift: output = x_scaled + self.bias
            const output_op = hlo.add(ctx, x_scaled.value, self.bias.value, loc);
            return builder.newTensor(output_op.getResult(0));
        }

        pub fn forwardWithGrad(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            return self.forward(x, builder);
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
            const loc = builder.loc;
            
            // Input x has shape [B, T, C] where B=batch, T=sequence, C=n_embd
            const batch_size = x.shape.getDimension(0);
            const seq_len = x.shape.getDimension(1);
            const n_embd = @as(i64, @intCast(self.n_embd));
            const n_head = @as(i64, @intCast(self.n_head));
            const head_dim = @as(i64, @intCast(self.head_dim));
            
            // Step 1: QKV projection
            // qkv = (x @ c_attn_weight) + c_attn_bias  [B, T, 3*C]
            var qkv_proj = try builder.createAndAppendOp(hlo.dot_general(ctx, x.value, self.c_attn_weight.value, .{
                .dot_dimension_numbers = hlo.DotDimensionNumbersAttribute{
                    .lhs_batching_dimensions = &.{},
                    .rhs_batching_dimensions = &.{},
                    .lhs_contracting_dimensions = &.{2}, // Contract over C (last dim of x)
                    .rhs_contracting_dimensions = &.{0}, // Contract over first dim of weight
                },
            }));
            
            var qkv = try builder.createAndAppendOp(hlo.add(ctx, qkv_proj.value, self.c_attn_bias.value, loc));

            
            // Step 2: Split QKV into separate Q, K, V tensors
            // Each has shape [B, T, C]
            const slice_start_q = [_]i64{ 0, 0, 0 };
            const slice_limit_q = [_]i64{ batch_size, seq_len, n_embd };
            const slice_strides = [_]i64{ 1, 1, 1 };
            
            const q_op = hlo.slice(ctx, qkv.value, &slice_start_q, &slice_limit_q, &slice_strides, loc);
            var q = try builder.newTensor(q_op.getResult(0));

            
            const slice_start_k = [_]i64{ 0, 0, n_embd };
            const slice_limit_k = [_]i64{ batch_size, seq_len, 2 * n_embd };
            
            const k_op = hlo.slice(ctx, qkv.value, &slice_start_k, &slice_limit_k, &slice_strides, loc);
            var k = try builder.newTensor(k_op.getResult(0));

            
            const slice_start_v = [_]i64{ 0, 0, 2 * n_embd };
            const slice_limit_v = [_]i64{ batch_size, seq_len, 3 * n_embd };
            
            const v_op = hlo.slice(ctx, qkv.value, &slice_start_v, &slice_limit_v, &slice_strides, loc);
            var v = try builder.newTensor(v_op.getResult(0));

            
            // Step 3: Reshape for multi-head attention
            // Reshape from [B, T, C] to [B, T, n_head, head_dim]
            const head_shape = [_]i64{ batch_size, seq_len, n_head, head_dim };
            
            const q_reshaped_op = hlo.reshape(ctx, q.value, &head_shape, loc);
            var q_reshaped = try builder.newTensor(q_reshaped_op.getResult(0));

            
            const k_reshaped_op = hlo.reshape(ctx, k.value, &head_shape, loc);
            var k_reshaped = try builder.newTensor(k_reshaped_op.getResult(0));

            
            const v_reshaped_op = hlo.reshape(ctx, v.value, &head_shape, loc);
            var v_reshaped = try builder.newTensor(v_reshaped_op.getResult(0));

            
            // Transpose to [B, n_head, T, head_dim] for efficient batched matmul
            const head_transpose_perm = [_]i64{ 0, 2, 1, 3 };
            
            const q_heads_op = hlo.transpose(ctx, q_reshaped.value, &head_transpose_perm, loc);
            var q_heads = try builder.newTensor(q_heads_op.getResult(0));

            
            const k_heads_op = hlo.transpose(ctx, k_reshaped.value, &head_transpose_perm, loc);
            var k_heads = try builder.newTensor(k_heads_op.getResult(0));

            
            const v_heads_op = hlo.transpose(ctx, v_reshaped.value, &head_transpose_perm, loc);
            var v_heads = try builder.newTensor(v_heads_op.getResult(0));

            
            // Step 4: Calculate attention scores
            // k_transposed: [B, n_head, head_dim, T] for matmul
            const k_transpose_perm = [_]i64{ 0, 1, 3, 2 };
            const k_t_op = hlo.transpose(ctx, k_heads.value, &k_transpose_perm, loc);
            var k_t = try builder.newTensor(k_t_op.getResult(0));

            
            // scores = q @ k_t  [B, n_head, T, T]
            var scores = try builder.createAndAppendOp(hlo.dot_general(ctx, q_heads.value, k_t.value, .{
                .dot_dimension_numbers = hlo.DotDimensionNumbersAttribute{
                    .lhs_batching_dimensions = &.{0, 1}, // Batch over B and n_head
                    .rhs_batching_dimensions = &.{0, 1},
                    .lhs_contracting_dimensions = &.{3}, // Contract over head_dim
                    .rhs_contracting_dimensions = &.{2},
                },
            }));
            
            // Scale by 1/sqrt(head_dim)
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
            const scale_const = try builder.scalarConstant(scale);
            
            const scaled_scores_op = hlo.multiply(ctx, scores.value, scale_const.value, loc);
            var scaled_scores = try builder.newTensor(scaled_scores_op.getResult(0));

            
            // Step 5: Apply causal mask
            // Create triangular mask where mask[i, j] = i >= j (allow current and past)
            const iota_0_op = hlo.iota(ctx, &[_]i64{ seq_len, seq_len }, 0, mlir.Type.i32Type(ctx), loc);
            var iota_0 = try builder.newTensor(iota_0_op.getResult(0));

            
            const iota_1_op = hlo.iota(ctx, &[_]i64{ seq_len, seq_len }, 1, mlir.Type.i32Type(ctx), loc);
            var iota_1 = try builder.newTensor(iota_1_op.getResult(0));

            
            // mask = iota_0 >= iota_1 (true for lower triangular + diagonal)
            const mask_2d_op = hlo.compare(ctx, iota_0.value, iota_1.value, hlo.ComparisonDirection.GE, loc);
            var mask_2d = try builder.newTensor(mask_2d_op.getResult(0));

            
            // Broadcast mask to [B, n_head, T, T]
            const mask_broadcast_dims = [_]i64{ 2, 3 };
            const mask_target_shape = [_]i64{ batch_size, n_head, seq_len, seq_len };
            const mask_op = hlo.broadcast_in_dim(ctx, mask_2d.value, &mask_target_shape, &mask_broadcast_dims, loc);
            var mask = try builder.newTensor(mask_op.getResult(0));

            
            // Apply mask: select(mask, scores, -inf)
            const neg_inf = try builder.scalarConstant(-1e9);
            
            const masked_scores_op = hlo.select(ctx, mask.value, scaled_scores.value, neg_inf.value, loc);
            var masked_scores = try builder.newTensor(masked_scores_op.getResult(0));

            
            // Step 6: Apply softmax
            const attn_weights_op = hlo.softmax(ctx, masked_scores.value, -1, loc);
            var attn_weights = try builder.newTensor(attn_weights_op.getResult(0));

            
            // Step 7: Apply attention to values
            // output = attn_weights @ v_heads  [B, n_head, T, head_dim]
            const attn_output_op = hlo.dot_general(ctx, attn_weights.value, v_heads.value, .{
                .dot_dimension_numbers = hlo.DotDimensionNumbersAttribute{
                    .lhs_batching_dimensions = &.{0, 1},
                    .rhs_batching_dimensions = &.{0, 1},
                    .lhs_contracting_dimensions = &.{3}, // Contract over last T dim
                    .rhs_contracting_dimensions = &.{2},
                },
            });
            var attn_output = try builder.newTensor(attn_output_op.getResult(0));
            
            // Step 8: Transpose back and reshape
            // Transpose from [B, n_head, T, head_dim] to [B, T, n_head, head_dim]
            const output_transpose_perm = [_]i64{ 0, 2, 1, 3 };
            const output_transposed_op = hlo.transpose(ctx, attn_output.value, &output_transpose_perm, loc);
            var output_transposed = try builder.newTensor(output_transposed_op.getResult(0));

            
            // Reshape to [B, T, C]
            const output_shape = [_]i64{ batch_size, seq_len, n_embd };
            const output_reshaped_op = hlo.reshape(ctx, output_transposed.value, &output_shape, loc);
            var output_reshaped = try builder.newTensor(output_reshaped_op.getResult(0));

            
            // Step 9: Apply output projection
            // final_output = (output_reshaped @ c_proj_weight) + c_proj_bias
            const proj_op = hlo.dot_general(ctx, output_reshaped.value, self.c_proj_weight.value, .{
                .dot_dimension_numbers = hlo.DotDimensionNumbersAttribute{
                    .lhs_batching_dimensions = &.{},
                    .rhs_batching_dimensions = &.{},
                    .lhs_contracting_dimensions = &.{2}, // Contract over C
                    .rhs_contracting_dimensions = &.{0},
                },
            });
            var proj_output = try builder.newTensor(proj_op.getResult(0));
            
            const final_op = hlo.add(ctx, proj_output.value, self.c_proj_bias.value, loc);
            return builder.newTensor(final_op.getResult(0));
        }

        pub fn forwardWithGrad(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            return self.forward(x, builder);
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
            const ctx = builder.ctx;
            const loc = builder.loc;
            
            // Constants for GELU approximation
            const sqrt_2_over_pi = try builder.scalarConstant(@sqrt(2.0 / std.math.pi));
            const coeff = try builder.scalarConstant(0.044715);
            const half = try builder.scalarConstant(0.5);
            const one = try builder.scalarConstant(1.0);
            
            // x^3
            const x_squared_op = hlo.multiply(ctx, x.value, x.value, loc);
            var x_squared = try builder.newTensor(x_squared_op.getResult(0));

            
            const x_cubed_op = hlo.multiply(ctx, x_squared.value, x.value, loc);
            var x_cubed = try builder.newTensor(x_cubed_op.getResult(0));

            
            // 0.044715 * x^3
            const coeff_x_cubed_op = hlo.multiply(ctx, coeff.value, x_cubed.value, loc);
            var coeff_x_cubed = try builder.newTensor(coeff_x_cubed_op.getResult(0));

            
            // x + 0.044715 * x^3
            const inner_op = hlo.add(ctx, x.value, coeff_x_cubed.value, loc);
            var inner = try builder.newTensor(inner_op.getResult(0));

            
            // sqrt(2/π) * (x + 0.044715 * x^3)
            const scaled_inner_op = hlo.multiply(ctx, sqrt_2_over_pi.value, inner.value, loc);
            var scaled_inner = try builder.newTensor(scaled_inner_op.getResult(0));

            
            // tanh(sqrt(2/π) * (x + 0.044715 * x^3))
            const tanh_op = hlo.tanh(ctx, scaled_inner.value, loc);
            var tanh_result = try builder.newTensor(tanh_op.getResult(0));

            
            // 1 + tanh(...)
            const one_plus_tanh_op = hlo.add(ctx, one.value, tanh_result.value, loc);
            var one_plus_tanh = try builder.newTensor(one_plus_tanh_op.getResult(0));

            
            // x * (1 + tanh(...))
            const x_times_op = hlo.multiply(ctx, x.value, one_plus_tanh.value, loc);
            var x_times = try builder.newTensor(x_times_op.getResult(0));

            
            // 0.5 * x * (1 + tanh(...))
            const result_op = hlo.multiply(ctx, half.value, x_times.value, loc);
            return builder.newTensor(result_op.getResult(0));
        }

        pub fn forward(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            const ctx = builder.ctx;
            const loc = builder.loc;
            
            // Step 1: First linear layer: x @ c_fc_weight + c_fc_bias
            var fc_proj = try builder.createAndAppendOp(hlo.dot_general(ctx, x.value, self.c_fc_weight.value, .{
                .dot_dimension_numbers = hlo.DotDimensionNumbersAttribute{
                    .lhs_batching_dimensions = &.{},
                    .rhs_batching_dimensions = &.{},
                    .lhs_contracting_dimensions = &.{2}, // Contract over n_embd
                    .rhs_contracting_dimensions = &.{0},
                },
            }));
            
            var fc_biased = try builder.createAndAppendOp(hlo.add(ctx, fc_proj.value, self.c_fc_bias.value, loc));

            
            // Step 2: Apply GELU activation
            var fc_activated = try self.gelu(fc_biased, builder);

            
            // Step 3: Second linear layer: activated @ c_proj_weight + c_proj_bias
            const proj_op = hlo.dot_general(ctx, fc_activated.value, self.c_proj_weight.value, .{
                .dot_dimension_numbers = hlo.DotDimensionNumbersAttribute{
                    .lhs_batching_dimensions = &.{},
                    .rhs_batching_dimensions = &.{},
                    .lhs_contracting_dimensions = &.{2}, // Contract over n_inner
                    .rhs_contracting_dimensions = &.{0},
                },
            });
            var proj_output = try builder.newTensor(proj_op.getResult(0));
            
            const final_op = hlo.add(ctx, proj_output.value, self.c_proj_bias.value, loc);
            return builder.newTensor(final_op.getResult(0));
        }

        pub fn forwardWithGrad(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            return self.forward(x, builder);
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
            const ctx = builder.ctx;
            const loc = builder.loc;
            
            // Pre-norm architecture: LayerNorm -> Attention -> Add residual
            var norm1 = try self.ln_1.forward(x, builder);


            var attn_output = try self.attn.forward(norm1, builder);


            // Residual connection 1
            const res1_op = hlo.add(ctx, x.value, attn_output.value, loc);
            var res1 = try builder.newTensor(res1_op.getResult(0));


            // Pre-norm architecture: LayerNorm -> MLP -> Add residual
            var norm2 = try self.ln_2.forward(res1, builder);


            var mlp_output = try self.mlp.forward(norm2, builder);


            // Residual connection 2
            const res2_op = hlo.add(ctx, res1.value, mlp_output.value, loc);
            return builder.newTensor(res2_op.getResult(0));
        }

        pub fn forwardWithGrad(self: *@This(), x: Tensor, builder: *MLIRBuilder) !Tensor {
            return self.forward(x, builder);
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
            const loc = builder.loc;
            
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
            const targets_i32_op = hlo.convert(ctx, targets.value, targets_i32_type, loc);
            var targets_i32 = try builder.newTensor(targets_i32_op.getResult(0));
            
            const one_hot_op = hlo.one_hot(ctx, targets_i32.value, vocab_size, 1.0, 0.0, 2, mlir.Type.f32Type(ctx), loc);
            var targets_one_hot = try builder.newTensor(one_hot_op.getResult(0));
            
            // Step 2: Compute log softmax
            // log_softmax(x) = x - log(sum(exp(x), axis=-1))
            const exp_logits_op = hlo.exponential(ctx, logits.value, loc);
            var exp_logits = try builder.newTensor(exp_logits_op.getResult(0));

            
            const sum_exp_op = hlo.reduce_sum(ctx, exp_logits.value, &[_]i64{2}, true, loc);
            var sum_exp = try builder.newTensor(sum_exp_op.getResult(0));

            
            const log_sum_exp_op = hlo.log(ctx, sum_exp.value, loc);
            var log_sum_exp = try builder.newTensor(log_sum_exp_op.getResult(0));

            
            const log_probs_op = hlo.subtract(ctx, logits.value, log_sum_exp.value, loc);
            var log_probs = try builder.newTensor(log_probs_op.getResult(0));

            
            // Step 3: Select correct log probabilities
            // Multiply log_probs by one-hot targets to zero out incorrect classes
            const selected_log_probs_op = hlo.multiply(ctx, log_probs.value, targets_one_hot.value, loc);
            var selected_log_probs = try builder.newTensor(selected_log_probs_op.getResult(0));

            
            // Sum over vocabulary dimension to get negative log likelihood per token
            const nll_per_token_op = hlo.reduce_sum(ctx, selected_log_probs.value, &[_]i64{2}, false, loc);
            var nll_per_token = try builder.newTensor(nll_per_token_op.getResult(0));

            
            // Step 4: Compute mean loss
            // Negate to get positive loss values
            const neg_one = try builder.scalarConstant(-1.0);
            
            const loss_per_token_op = hlo.multiply(ctx, nll_per_token.value, neg_one.value, loc);
            var loss_per_token = try builder.newTensor(loss_per_token_op.getResult(0));

            
            // Sum all losses
            const total_loss_op = hlo.reduce_sum(ctx, loss_per_token.value, &[_]i64{0, 1}, false, loc);
            var total_loss = try builder.newTensor(total_loss_op.getResult(0));

            
            // Divide by number of tokens to get mean loss
            const num_tokens = batch_size * seq_len;
            const num_tokens_const = try builder.scalarConstant(@floatFromInt(num_tokens));
            
            const mean_loss_op = hlo.divide(ctx, total_loss.value, num_tokens_const.value, loc);
            return builder.newTensor(mean_loss_op.getResult(0));
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
            const loc = builder.loc;
            
            // Input validation
            const batch_size = input_ids.shape.getDimension(0);
            const seq_len = input_ids.shape.getDimension(1);
            
            if (seq_len > self.config.n_positions) {
                return error.SequenceTooLong;
            }
            
            // Step 1: Token embedding lookup [B, T] -> [B, T, C]
            const token_emb_op = hlo.gather(ctx, self.wte.value, input_ids.value, 
                hlo.GatherDimensionNumbersAttribute{
                    .offset_dims = &.{2}, // The output dimensions that are not from the indices
                    .collapsed_slice_dims = &.{0}, // We are taking a slice from dimension 0 of the embedding table
                    .start_index_map = &.{0}, // Map dimension 0 of indices to dimension 0 of the table
                    .index_vector_dim = 1, // The indices are vectors along the last dimension
                }, &.{ 1, @intCast(self.config.n_embd) }, loc); // Slice size: 1 token, full embedding dimension
            var token_embeddings = try builder.newTensor(token_emb_op.getResult(0));
            
            // Step 2: Position embeddings
            // Create position indices [0, 1, 2, ..., seq_len-1]
            const pos_indices_op = hlo.iota(ctx, &[_]i64{seq_len}, 0, mlir.Type.i32Type(ctx), loc);
            var pos_indices = try builder.newTensor(pos_indices_op.getResult(0));

            
            // Position embedding lookup [T] -> [T, C]
            const pos_emb_op = hlo.gather(ctx, self.wpe.value, pos_indices.value,
                hlo.GatherDimensionNumbersAttribute{
                    .offset_dims = &.{1}, // The output dimensions that are not from the indices
                    .collapsed_slice_dims = &.{0}, // We are taking a slice from dimension 0 of the position embedding table
                    .start_index_map = &.{0}, // Map dimension 0 of indices to dimension 0 of the table
                    .index_vector_dim = 0, // The indices are scalars (rank 1 tensor)
                }, &.{ 1, @intCast(self.config.n_embd) }, loc); // Slice size: 1 position, full embedding dimension
            var pos_emb_2d = try builder.newTensor(pos_emb_op.getResult(0));
            
            // Broadcast to [B, T, C]
            const pos_target_shape = [_]i64{ batch_size, seq_len, @intCast(self.config.n_embd) };
            const pos_broadcast_dims = [_]i64{ 1, 2 };
            const pos_emb_op = hlo.broadcast_in_dim(ctx, pos_emb_2d.value, &pos_target_shape, &pos_broadcast_dims, loc);
            var pos_embeddings = try builder.newTensor(pos_emb_op.getResult(0));

            
            // Add token and position embeddings
            const embeddings_op = hlo.add(ctx, token_embeddings.value, pos_embeddings.value, loc);
            var current = try builder.newTensor(embeddings_op.getResult(0));

            // Step 3: Process through transformer blocks
            for (self.blocks) |*block| {
                const new_output = try block.forward(current, builder);
                current.deinit();
                current = new_output;
            }

            // Step 4: Final layer norm
            var final_hidden = try self.ln_f.forward(current, builder);
            current.deinit();

            // Step 5: Language model head projection
            const logits_op = hlo.dot_general(ctx, final_hidden.value, self.lm_head.value, .{
                .dot_dimension_numbers = hlo.DotDimensionNumbersAttribute{
                    .lhs_batching_dimensions = &.{},
                    .rhs_batching_dimensions = &.{},
                    .lhs_contracting_dimensions = &.{2}, // Contract over n_embd
                    .rhs_contracting_dimensions = &.{0},
                },
            });
            
            return builder.newTensor(logits_op.getResult(0));
        }

        pub fn forwardWithGrad(self: *@This(), input_ids: Tensor, builder: *MLIRBuilder) !Tensor {
            return self.forward(input_ids, builder);
        }

        pub fn forwardWithLoss(self: *@This(), input_ids: Tensor, targets: Tensor, builder: *MLIRBuilder) !struct { logits: Tensor, loss: Tensor } {
            const logits = try self.forward(input_ids, builder);
            const loss = try crossEntropyLoss(DataType).run(self.allocator, logits, targets, builder);
            
            return .{ .logits = logits, .loss = loss };
        }
    };
}

/// Compute perplexity from loss
pub fn computePerplexity(comptime DataType: type) type {
    _ = DataType;

    return struct {
        pub fn run(allocator: Allocator, logits: Tensor, targets: Tensor, builder: *MLIRBuilder) !f32 {
            _ = allocator;
            const loss = try crossEntropyLoss(DataType).run(allocator, logits, targets, builder);

            // In practice, perplexity = exp(loss)
            // For MLIR tensors, we return a placeholder
            return 1.0;
        }
    };
}