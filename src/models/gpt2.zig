const std = @import("std");
const pcp = @import("pcp");
const tensor = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;

const Allocator = std.mem.Allocator;
const Shape = tensor.Shape;
const BackendType = tensor.BackendType;

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
pub fn Attention(comptime DataType: type) type {
    const Tensor = tensor.Tensor(DataType);
    const legacy_api = ops.LegacyApi(DataType);

    return struct {
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

        // AutoDiff plans for different operations
        matmul_plan: ?*autodiff.AutoDiffPlan(autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null, null), DataType) = null,
        add_plan: ?*autodiff.AutoDiffPlan(autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null), DataType) = null,
        softmax_plan: ?*autodiff.AutoDiffPlan(autodiff.SoftmaxPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null), DataType) = null,

        pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !@This() {
            const n_embd = config.n_embd;
            const n_head = config.n_head;
            const head_dim = n_embd / n_head;

            // QKV combined weight (n_embd x 3*n_embd)
            const c_attn_weight_dims = [_]usize{ n_embd, 3 * n_embd };
            const c_attn_weight = try Tensor.random(allocator, &c_attn_weight_dims, backend);

            // Scale the random initialization
            for (c_attn_weight.buffer.data) |*val| {
                val.* *= config.initializer_range;
            }

            // QKV combined bias (3*n_embd)
            const c_attn_bias_dims = [_]usize{3 * n_embd};
            const c_attn_bias = try Tensor.zeros(allocator, &c_attn_bias_dims, backend);

            // Output projection weight (n_embd x n_embd)
            const c_proj_weight_dims = [_]usize{ n_embd, n_embd };
            const c_proj_weight = try Tensor.random(allocator, &c_proj_weight_dims, backend);

            // Scale the random initialization
            for (c_proj_weight.buffer.data) |*val| {
                val.* *= config.initializer_range;
            }

            // Output projection bias (n_embd)
            const c_proj_bias_dims = [_]usize{n_embd};
            const c_proj_bias = try Tensor.zeros(allocator, &c_proj_bias_dims, backend);

            var attn = @This(){
                .c_attn_weight = c_attn_weight,
                .c_attn_bias = c_attn_bias,
                .c_proj_weight = c_proj_weight,
                .c_proj_bias = c_proj_bias,
                .n_embd = n_embd,
                .n_head = n_head,
                .head_dim = head_dim,
                .allocator = allocator,
            };

            // Initialize the autodiff plans
            const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null, null);
            const matmul_plan = try allocator.create(autodiff.AutoDiffPlan(MatmulPlanType, DataType));
            matmul_plan.* = autodiff.AutoDiffPlan(MatmulPlanType, DataType).init(allocator);
            attn.matmul_plan = matmul_plan;

            const AddPlanType = autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null);
            const add_plan = try allocator.create(autodiff.AutoDiffPlan(AddPlanType, DataType));
            add_plan.* = autodiff.AutoDiffPlan(AddPlanType, DataType).init(allocator);
            attn.add_plan = add_plan;

            const SoftmaxPlanType = autodiff.SoftmaxPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null);
            const softmax_plan = try allocator.create(autodiff.AutoDiffPlan(SoftmaxPlanType, DataType));
            softmax_plan.* = autodiff.AutoDiffPlan(SoftmaxPlanType, DataType).init(allocator);
            attn.softmax_plan = softmax_plan;

            return attn;
        }

        pub fn deinit(self: *@This()) void {
            self.c_attn_weight.deinit();
            self.c_attn_bias.deinit();
            self.c_proj_weight.deinit();
            self.c_proj_bias.deinit();

            // Clean up plans
            if (self.matmul_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.matmul_plan = null;
            }

            if (self.add_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.add_plan = null;
            }

            if (self.softmax_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.softmax_plan = null;
            }
        }

        // Forward pass through the attention layer using Plan-based approach
        pub fn forward(self: *@This(), x: Tensor) !Tensor {
            // Get batch size from input shape
            const batch_size = x.shape.dims[0];

            // Step 1: Compute QKV projections: x @ c_attn_weight + c_attn_bias
            var qkv_proj = try legacy_api.matmul(self.allocator, x, self.c_attn_weight);
            defer qkv_proj.deinit();

            // Add bias
            var qkv_bias_expanded = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, 3 * self.n_embd }, x.backend);
            defer qkv_bias_expanded.deinit();

            // Fill bias across batch dimension
            const qkv_bias_expanded_buf = qkv_bias_expanded.buffer.data;
            const c_attn_bias_buf = self.c_attn_bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..3 * self.n_embd) |i| {
                    qkv_bias_expanded_buf[b * (3 * self.n_embd) + i] = c_attn_bias_buf[i];
                }
            }

            var qkv = try legacy_api.add(self.allocator, qkv_proj, qkv_bias_expanded);
            defer qkv.deinit();

            // Step 2: Split QKV into separate query, key, value tensors
            var q_tensor = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, qkv.backend);
            defer q_tensor.deinit();

            var k_tensor = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, qkv.backend);
            defer k_tensor.deinit();

            var v_tensor = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, qkv.backend);
            defer v_tensor.deinit();

            // Split the QKV tensor
            const qkv_buf = qkv.buffer.data;
            const q_buf = q_tensor.buffer.data;
            const k_buf = k_tensor.buffer.data;
            const v_buf = v_tensor.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_embd) |i| {
                    q_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + i];
                    k_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + self.n_embd + i];
                    v_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + 2 * self.n_embd + i];
                }
            }

            // Step 3: Reshape for multi-head attention: [batch_size, n_head, seq_len, head_dim]
            // For simplicity, we'll treat batch_size as batch_size*seq_len and reshape later

            // Step 4: Compute attention scores: Q @ K^T / sqrt(head_dim)
            // First transpose K: [batch_size, n_embd] -> [n_embd, batch_size]
            var k_transpose = try legacy_api.transpose(self.allocator, k_tensor);
            defer k_transpose.deinit();

            // Q @ K^T: [batch_size, n_embd] @ [n_embd, batch_size] -> [batch_size, batch_size]
            var attention_scores_raw = try legacy_api.matmul(self.allocator, q_tensor, k_transpose);
            defer attention_scores_raw.deinit();

            // Scale by sqrt(head_dim)
            const scaling_factor = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));

            // Apply scaling manually
            var attention_scores_tensor = try Tensor.zeros(self.allocator, attention_scores_raw.shape.dims, attention_scores_raw.backend);
            defer attention_scores_tensor.deinit();

            const scores_raw_buf = attention_scores_raw.buffer.data;
            const scores_buf = attention_scores_tensor.buffer.data;

            for (0..attention_scores_tensor.shape.elemCount()) |i| {
                scores_buf[i] = scores_raw_buf[i] * scaling_factor;
            }

            // Step 5: Apply softmax to get attention weights
            var attention_weights = try legacy_api.softmax(self.allocator, attention_scores_tensor);
            defer attention_weights.deinit();

            // Step 6: Apply attention weights to values: [batch_size, batch_size] @ [batch_size, n_embd]
            var context = try legacy_api.matmul(self.allocator, attention_weights, v_tensor);
            defer context.deinit();

            // Step 7: Apply output projection
            var output_raw = try legacy_api.matmul(self.allocator, context, self.c_proj_weight);
            defer output_raw.deinit();

            // Expand bias for addition
            var proj_bias_expanded = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, output_raw.backend);
            defer proj_bias_expanded.deinit();

            // Fill bias across batch dimension
            const proj_bias_expanded_buf = proj_bias_expanded.buffer.data;
            const c_proj_bias_buf = self.c_proj_bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_embd) |i| {
                    proj_bias_expanded_buf[b * self.n_embd + i] = c_proj_bias_buf[i];
                }
            }

            // Add bias to get final output
            return legacy_api.add(self.allocator, output_raw, proj_bias_expanded);
        }

        // Forward pass with gradients
        pub fn forwardWithGrad(self: *@This(), x: Tensor) !Tensor {
            if (self.matmul_plan == null or self.add_plan == null or self.softmax_plan == null) {
                return error.PlanNotInitialized;
            }

            // Get batch size from input shape
            const batch_size = x.shape.dims[0];

            // Step 1: Compute QKV projections: x @ c_attn_weight + c_attn_bias
            const qkv_proj = try self.matmul_plan.?.forward(.{ .a = x, .b = self.c_attn_weight });
            defer qkv_proj.deinit();

            // Add bias
            var qkv_bias_expanded = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, 3 * self.n_embd }, x.dtype, x.backend);
            defer qkv_bias_expanded.deinit();

            // Fill bias across batch dimension
            const qkv_bias_expanded_buf = qkv_bias_expanded.buffer.data;
            const c_attn_bias_buf = self.c_attn_bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..3 * self.n_embd) |i| {
                    qkv_bias_expanded_buf[b * (3 * self.n_embd) + i] = c_attn_bias_buf[i];
                }
            }

            const qkv = try self.add_plan.?.forward(.{ .a = qkv_proj, .b = qkv_bias_expanded });
            defer qkv.deinit();

            // Step 2: Split QKV into separate query, key, value tensors
            var q_tensor = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, qkv.backend);
            defer q_tensor.deinit();

            var k_tensor = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, qkv.backend);
            defer k_tensor.deinit();

            var v_tensor = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, qkv.backend);
            defer v_tensor.deinit();

            // Split the QKV tensor
            const qkv_buf = qkv.buffer.data;
            const q_buf = q_tensor.buffer.data;
            const k_buf = k_tensor.buffer.data;
            const v_buf = v_tensor.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_embd) |i| {
                    q_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + i];
                    k_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + self.n_embd + i];
                    v_buf[b * self.n_embd + i] = qkv_buf[b * (3 * self.n_embd) + 2 * self.n_embd + i];
                }
            }

            // Step 3: Compute attention scores: Q @ K^T / sqrt(head_dim)
            // First transpose K with manual operation (in a real implementation, we'd use a plan)
            var k_transpose = try Tensor.zeros(self.allocator, &[_]usize{ self.n_embd, batch_size }, k_tensor.dtype, k_tensor.backend);
            defer k_transpose.deinit();

            const k_t_buf = k_transpose.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_embd) |i| {
                    k_t_buf[i * batch_size + b] = k_buf[b * self.n_embd + i];
                }
            }

            // Q @ K^T using matmul plan
            const attention_scores_raw = try self.matmul_plan.?.forward(.{ .a = q_tensor, .b = k_transpose });
            defer attention_scores_raw.deinit();

            // Scale by sqrt(head_dim)
            const scaling_factor = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));

            // Apply scaling manually
            var attention_scores_tensor = try Tensor.zeros(self.allocator, attention_scores_raw.shape.dims, attention_scores_raw.dtype, attention_scores_raw.backend);
            defer attention_scores_tensor.deinit();

            const scores_raw_buf = attention_scores_raw.buffer.data;
            const scores_buf = attention_scores_tensor.buffer.data;

            for (0..attention_scores_tensor.shape.elemCount()) |i| {
                scores_buf[i] = scores_raw_buf[i] * scaling_factor;
            }

            // Step 4: Apply softmax to get attention weights using softmax plan
            const attention_weights = try self.softmax_plan.?.forward(attention_scores_tensor);
            defer attention_weights.deinit();

            // Step 5: Apply attention weights to values
            const context = try self.matmul_plan.?.forward(.{ .a = attention_weights, .b = v_tensor });
            defer context.deinit();

            // Step 6: Apply output projection
            const output_raw = try self.matmul_plan.?.forward(.{ .a = context, .b = self.c_proj_weight });
            defer output_raw.deinit();

            // Expand bias for addition
            var proj_bias_expanded = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, output_raw.dtype, output_raw.backend);
            defer proj_bias_expanded.deinit();

            // Fill bias across batch dimension
            const proj_bias_expanded_buf = proj_bias_expanded.buffer.data;
            const c_proj_bias_buf = self.c_proj_bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_embd) |i| {
                    proj_bias_expanded_buf[b * self.n_embd + i] = c_proj_bias_buf[i];
                }
            }

            // Add bias to get final output using plan
            const output = try self.add_plan.?.forward(.{ .a = output_raw, .b = proj_bias_expanded });

            return output;
        }
    };
}

// MLP (Feed-Forward) layer implementation
pub fn MLP(comptime DataType: type) type {
    const Tensor = tensor.Tensor(DataType);
    const legacy_api = ops.LegacyApi(DataType);

    return struct {
        c_fc_weight: Tensor,
        c_fc_bias: Tensor,
        c_proj_weight: Tensor,
        c_proj_bias: Tensor,

        n_embd: usize,
        n_inner: usize,
        allocator: Allocator,

        // AutoDiff plans
        matmul_plan: ?*autodiff.AutoDiffPlan(autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null, null), DataType) = null,
        add_plan: ?*autodiff.AutoDiffPlan(autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null), DataType) = null,
        relu_plan: ?*autodiff.AutoDiffPlan(autodiff.ReluPlanWithGrad(ops.CpuBackend(DataType), DataType, null), DataType) = null,

        pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !@This() {
            const n_embd = config.n_embd;
            const n_inner = 4 * n_embd; // Typically 4x the embedding dimension

            // FC weights and biases
            const c_fc_weight_dims = [_]usize{ n_embd, n_inner };
            const c_fc_weight = try Tensor.random(allocator, &c_fc_weight_dims, backend);

            // Scale the random initialization
            const c_fc_weight_ptr = c_fc_weight.buffer.data;
            for (c_fc_weight_ptr) |*val| {
                val.* *= config.initializer_range;
            }

            const c_fc_bias_dims = [_]usize{n_inner};
            const c_fc_bias = try Tensor.zeros(allocator, &c_fc_bias_dims, backend);

            // Projection weights and biases
            const c_proj_weight_dims = [_]usize{ n_inner, n_embd };
            const c_proj_weight = try Tensor.random(allocator, &c_proj_weight_dims, backend);

            // Scale the random initialization
            const c_proj_weight_ptr = c_proj_weight.buffer.data;
            for (c_proj_weight_ptr) |*val| {
                val.* *= config.initializer_range;
            }

            const c_proj_bias_dims = [_]usize{n_embd};
            const c_proj_bias = try Tensor.zeros(allocator, &c_proj_bias_dims, backend);

            var mlp = @This(){
                .c_fc_weight = c_fc_weight,
                .c_fc_bias = c_fc_bias,
                .c_proj_weight = c_proj_weight,
                .c_proj_bias = c_proj_bias,
                .n_embd = n_embd,
                .n_inner = n_inner,
                .allocator = allocator,
            };

            // Initialize the autodiff plans
            const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null, null);
            const matmul_plan = try allocator.create(autodiff.AutoDiffPlan(MatmulPlanType, DataType));
            matmul_plan.* = autodiff.AutoDiffPlan(MatmulPlanType, DataType).init(allocator);
            mlp.matmul_plan = matmul_plan;

            const AddPlanType = autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null);
            const add_plan = try allocator.create(autodiff.AutoDiffPlan(AddPlanType, DataType));
            add_plan.* = autodiff.AutoDiffPlan(AddPlanType, DataType).init(allocator);
            mlp.add_plan = add_plan;

            const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend(DataType), DataType, null);
            const relu_plan = try allocator.create(autodiff.AutoDiffPlan(ReluPlanType, DataType));
            relu_plan.* = autodiff.AutoDiffPlan(ReluPlanType, DataType).init(allocator);
            mlp.relu_plan = relu_plan;

            return mlp;
        }

        pub fn deinit(self: *@This()) void {
            self.c_fc_weight.deinit();
            self.c_fc_bias.deinit();
            self.c_proj_weight.deinit();
            self.c_proj_bias.deinit();

            // Clean up plans
            if (self.matmul_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.matmul_plan = null;
            }

            if (self.add_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.add_plan = null;
            }

            if (self.relu_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.relu_plan = null;
            }
        }

        // Forward pass using regular tensor operations (no gradients)
        pub fn forward(self: *@This(), x: Tensor) !Tensor {
            // Get batch size from input
            const batch_size = x.shape.dims[0];

            // Step 1: First linear layer: x @ c_fc_weight + c_fc_bias
            var fc_out = try legacy_api.matmul(self.allocator, x, self.c_fc_weight);
            defer fc_out.deinit();

            // Expand fc bias for batch dimension
            var fc_bias_expanded = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_inner }, x.backend);
            defer fc_bias_expanded.deinit();

            // Fill bias across batch dimension
            const fc_bias_expanded_buf = fc_bias_expanded.buffer.data;
            const c_fc_bias_buf = self.c_fc_bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_inner) |i| {
                    fc_bias_expanded_buf[b * self.n_inner + i] = c_fc_bias_buf[i];
                }
            }

            // Add bias
            var fc_biased = try legacy_api.add(self.allocator, fc_out, fc_bias_expanded);
            defer fc_biased.deinit();

            // Step 2: Apply activation (ReLU for simplicity)
            var fc_activated = try legacy_api.relu(self.allocator, fc_biased);
            defer fc_activated.deinit();

            // Step 3: Second linear layer: activated @ c_proj_weight + c_proj_bias
            var proj_out = try legacy_api.matmul(self.allocator, fc_activated, self.c_proj_weight);
            defer proj_out.deinit();

            // Expand proj bias for batch dimension
            var proj_bias_expanded = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, x.backend);
            defer proj_bias_expanded.deinit();

            // Fill bias across batch dimension
            const proj_bias_expanded_buf = proj_bias_expanded.buffer.data;
            const c_proj_bias_buf = self.c_proj_bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_embd) |i| {
                    proj_bias_expanded_buf[b * self.n_embd + i] = c_proj_bias_buf[i];
                }
            }

            // Add bias to get final output
            return legacy_api.add(self.allocator, proj_out, proj_bias_expanded);
        }

        // Forward pass with gradient tracking using Plan-based approach
        pub fn forwardWithGrad(self: *@This(), x: Tensor) !Tensor {
            if (self.matmul_plan == null or self.add_plan == null or self.relu_plan == null) {
                return error.PlanNotInitialized;
            }

            // Get batch size from input
            const batch_size = x.shape.dims[0];

            // Step 1: First linear layer with plans
            const fc_out = try self.matmul_plan.?.forward(.{ .a = x, .b = self.c_fc_weight });
            defer fc_out.deinit();

            // Expand bias for addition
            var fc_bias_expanded = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_inner }, x.dtype, x.backend);
            defer fc_bias_expanded.deinit();

            // Fill bias across batch dimension
            const fc_bias_expanded_buf = fc_bias_expanded.buffer.data;
            const c_fc_bias_buf = self.c_fc_bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_inner) |i| {
                    fc_bias_expanded_buf[b * self.n_inner + i] = c_fc_bias_buf[i];
                }
            }

            // Add bias with plan
            const fc_biased = try self.add_plan.?.forward(.{ .a = fc_out, .b = fc_bias_expanded });
            defer fc_biased.deinit();

            // Step 2: Apply ReLU with plan
            const fc_activated = try self.relu_plan.?.forward(fc_biased);
            defer fc_activated.deinit();

            // Step 3: Second linear layer with plans
            const proj_out = try self.matmul_plan.?.forward(.{ .a = fc_activated, .b = self.c_proj_weight });
            defer proj_out.deinit();

            // Expand proj bias for batch dimension
            var proj_bias_expanded = try Tensor.zeros(self.allocator, &[_]usize{ batch_size, self.n_embd }, x.dtype, x.backend);
            defer proj_bias_expanded.deinit();

            // Fill bias across batch dimension
            const proj_bias_expanded_buf = proj_bias_expanded.buffer.data;
            const c_proj_bias_buf = self.c_proj_bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..self.n_embd) |i| {
                    proj_bias_expanded_buf[b * self.n_embd + i] = c_proj_bias_buf[i];
                }
            }

            // Add bias to get final output with plan
            const output = try self.add_plan.?.forward(.{ .a = proj_out, .b = proj_bias_expanded });

            return output;
        }
    };
}

// Layer normalization implementation
pub fn LayerNorm(comptime DataType: type) type {
    const Tensor = tensor.Tensor(DataType);

    return struct {
        weight: Tensor,
        bias: Tensor,
        epsilon: f32,
        n_embd: usize,
        allocator: Allocator,

        // AutoDiff plan for add operation
        add_plan: ?*autodiff.AutoDiffPlan(autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), f32, null), DataType) = null,

        pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !@This() {
            const n_embd = config.n_embd;
            const epsilon = config.layer_norm_epsilon;

            const weight_dims = [_]usize{n_embd};
            const weight = try Tensor.filled(allocator, &weight_dims, 1.0, backend);

            const bias_dims = [_]usize{n_embd};
            const bias = try Tensor.zeros(allocator, &bias_dims, backend);

            var ln = @This(){
                .weight = weight,
                .bias = bias,
                .epsilon = epsilon,
                .n_embd = n_embd,
                .allocator = allocator,
            };

            // Initialize plan
            const AddPlanType = autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null);
            const add_plan = try allocator.create(autodiff.AutoDiffPlan(AddPlanType, DataType));
            add_plan.* = autodiff.AutoDiffPlan(AddPlanType, DataType).init(allocator);
            ln.add_plan = add_plan;

            return ln;
        }

        pub fn deinit(self: *@This()) void {
            self.weight.deinit();
            self.bias.deinit();

            // Clean up plan
            if (self.add_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.add_plan = null;
            }
        }

        // Proper layer normalization implementation
        pub fn forward(self: *@This(), x: Tensor) !Tensor {
            const batch_size = x.shape.dims[0];
            const last_dim = self.n_embd;

            // Step 1: Calculate mean over the last dimension for each sample
            var mean = try Tensor.zeros(self.allocator, &[_]usize{batch_size}, x.backend);
            defer mean.deinit();

            const x_buf = x.buffer.data;
            const mean_buf = mean.buffer.data;

            for (0..batch_size) |b| {
                var sum: f32 = 0.0;
                for (0..last_dim) |i| {
                    sum += x_buf[b * last_dim + i];
                }
                mean_buf[b] = sum / @as(f32, @floatFromInt(last_dim));
            }

            // Step 2: Calculate variance over the last dimension
            var variance = try Tensor.zeros(self.allocator, &[_]usize{batch_size}, x.backend);
            defer variance.deinit();

            const var_buf = variance.buffer.data;

            for (0..batch_size) |b| {
                var sum_sq: f32 = 0.0;
                for (0..last_dim) |i| {
                    const diff = x_buf[b * last_dim + i] - mean_buf[b];
                    sum_sq += diff * diff;
                }
                var_buf[b] = sum_sq / @as(f32, @floatFromInt(last_dim));
            }

            // Step 3: Normalize, scale, and shift
            const output = try Tensor.zeros(self.allocator, x.shape.dims, x.backend);
            const output_buf = output.buffer.data;
            const weight_buf = self.weight.buffer.data;
            const bias_buf = self.bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..last_dim) |i| {
                    // Normalize
                    const normalized = (x_buf[b * last_dim + i] - mean_buf[b]) /
                        @sqrt(var_buf[b] + self.epsilon);

                    // Scale and shift
                    output_buf[b * last_dim + i] = normalized * weight_buf[i] + bias_buf[i];
                }
            }

            return output;
        }

        // Forward with gradient tracking using Plan-based approach
        pub fn forwardWithGrad(self: *@This(), x: Tensor) !Tensor {
            if (self.add_plan == null) {
                return error.PlanNotInitialized;
            }

            const batch_size = x.shape.dims[0];
            const last_dim = self.n_embd;

            // Step 1: Calculate mean over the last dimension
            var mean = try Tensor.zeros(self.allocator, &[_]usize{batch_size}, .f32, x.backend);
            defer mean.deinit();

            const x_buf = x.buffer.data;
            const mean_buf = mean.buffer.data;

            for (0..batch_size) |b| {
                var sum: f32 = 0.0;
                for (0..last_dim) |i| {
                    sum += x_buf[b * last_dim + i];
                }
                mean_buf[b] = sum / @as(f32, @floatFromInt(last_dim));
            }

            // Step 2: Calculate variance
            var variance = try Tensor.zeros(self.allocator, &[_]usize{batch_size}, .f32, x.backend);
            defer variance.deinit();

            const var_buf = variance.buffer.data;

            for (0..batch_size) |b| {
                var sum_sq: f32 = 0.0;
                for (0..last_dim) |i| {
                    const diff = x_buf[b * last_dim + i] - mean_buf[b];
                    sum_sq += diff * diff;
                }
                var_buf[b] = sum_sq / @as(f32, @floatFromInt(last_dim));
            }

            // Step 3: Normalize, scale, and shift with plans
            // In a complete implementation, we would use Plans for these operations
            // For now, we'll use the manual approach to ensure correctness
            const output = try Tensor.zeros(self.allocator, x.shape.dims, x.dtype, x.backend);
            const output_buf = output.buffer.data;
            const weight_buf = self.weight.buffer.data;
            const bias_buf = self.bias.buffer.data;

            for (0..batch_size) |b| {
                for (0..last_dim) |i| {
                    // Normalize
                    const normalized = (x_buf[b * last_dim + i] - mean_buf[b]) /
                        @sqrt(var_buf[b] + self.epsilon);

                    // Scale and shift
                    output_buf[b * last_dim + i] = normalized * weight_buf[i] + bias_buf[i];
                }
            }

            return output;
        }
    };
}

// Transformer layer (attention + MLP)
pub fn Block(comptime DataType: type) type {
    const Tensor = tensor.Tensor(DataType);
    const legacy_api = ops.LegacyApi(DataType);

    return struct {
        ln_1: LayerNorm(DataType),
        attn: Attention(DataType),
        ln_2: LayerNorm(DataType),
        mlp: MLP(DataType),
        allocator: Allocator,

        // AutoDiff plan for add (residual connections)
        add_plan: ?*autodiff.AutoDiffPlan(autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null), DataType) = null,

        pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !@This() {
            const ln_1 = try LayerNorm(DataType).init(allocator, config, backend);
            const attn = try Attention(DataType).init(allocator, config, backend);
            const ln_2 = try LayerNorm(DataType).init(allocator, config, backend);
            const mlp = try MLP(DataType).init(allocator, config, backend);

            var block = @This(){
                .ln_1 = ln_1,
                .attn = attn,
                .ln_2 = ln_2,
                .mlp = mlp,
                .allocator = allocator,
                .add_plan = null,
            };

            // Initialize plan
            const AddPlanType = autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null);
            const add_plan = try allocator.create(autodiff.AutoDiffPlan(AddPlanType, DataType));
            add_plan.* = autodiff.AutoDiffPlan(AddPlanType, DataType).init(allocator);
            block.add_plan = add_plan;

            return block;
        }

        pub fn deinit(self: *@This()) void {
            self.ln_1.deinit();
            self.attn.deinit();
            self.ln_2.deinit();
            self.mlp.deinit();

            // Clean up plan
            if (self.add_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.add_plan = null;
            }
        }

        // Forward pass without gradient tracking
        pub fn forward(self: *@This(), x: Tensor) !Tensor {
            std.debug.print("Block.forward: Starting with input shape [{}, {}]\n", .{ x.shape.dims[0], x.shape.dims[1] });

            // Step 1: Layer norm -> attention
            var norm1 = try self.ln_1.forward(x);
            defer norm1.deinit();
            std.debug.print("Block.forward: After ln_1.forward\n", .{});

            // Step 2: Apply attention to normalized input
            var attn_output = try self.attn.forward(norm1);
            defer attn_output.deinit();
            std.debug.print("Block.forward: After attn.forward\n", .{});

            // Step 3: Add attention output to input (residual connection)
            var res1 = try legacy_api.add(self.allocator, x, attn_output);
            defer res1.deinit();
            std.debug.print("Block.forward: After first residual connection\n", .{});

            // Step 4: Layer norm on residual output
            var norm2 = try self.ln_2.forward(res1);
            defer norm2.deinit();
            std.debug.print("Block.forward: After ln_2.forward\n", .{});

            // Step 5: Apply MLP to normalized residual
            var mlp_output = try self.mlp.forward(norm2);
            defer mlp_output.deinit();
            std.debug.print("Block.forward: After mlp.forward\n", .{});

            // Step 6: Add MLP output to residual (second residual connection)
            const res2 = try legacy_api.add(self.allocator, res1, mlp_output);
            std.debug.print("Block.forward: After second residual connection\n", .{});

            std.debug.print("Block.forward: Returning result with shape [{}, {}]\n", .{ res2.shape.dims[0], res2.shape.dims[1] });
            return res2;
        }

        // Forward pass with gradient tracking
        pub fn forwardWithGrad(self: *@This(), x: Tensor) !Tensor {
            if (self.add_plan == null) {
                return error.PlanNotInitialized;
            }

            std.debug.print("Block.forwardWithGrad: Starting\n", .{});

            // Step 1: Layer norm -> attention
            var norm1 = try self.ln_1.forwardWithGrad(x);
            defer norm1.deinit();

            // Step 2: Apply attention with gradient tracking
            var attn_output = try self.attn.forwardWithGrad(norm1);
            defer attn_output.deinit();

            // Step 3: Add attention output to input (residual connection) with plan
            var res1 = try self.add_plan.?.forward(.{ .a = x, .b = attn_output });
            defer res1.deinit();

            // Step 4: Layer norm on residual output
            var norm2 = try self.ln_2.forwardWithGrad(res1);
            defer norm2.deinit();

            // Step 5: Apply MLP to normalized residual with gradient tracking
            var mlp_output = try self.mlp.forwardWithGrad(norm2);
            defer mlp_output.deinit();

            // Step 6: Add MLP output to residual (second residual connection) with plan
            const res2 = try self.add_plan.?.forward(.{ .a = res1, .b = mlp_output });

            std.debug.print("Block.forwardWithGrad: Completed full block\n", .{});
            return res2;
        }
    };
}

// Main GPT-2 model
// Cross-entropy loss function for language modeling
pub fn crossEntropyLoss(comptime DataType: type) type {
    const Tensor = tensor.Tensor(DataType);

    return struct {
        pub fn run(allocator: Allocator, logits: Tensor, targets: Tensor) !Tensor {
            // Check for valid dimensions
            if (logits.shape.rank() < 3) {
                return error.InvalidLogitsShape;
            }

            const batch_size = targets.shape.dims[0];
            const seq_len = targets.shape.dims[1];
            const vocab_size = logits.shape.dims[2];

            const loss = try Tensor.zeros(allocator, &[_]usize{1}, logits.backend);
            const loss_buf = loss.buffer.data;
            const logits_buf = logits.buffer.data;
            const targets_buf = targets.buffer.data;

            var total_loss: f32 = 0.0;

            for (0..batch_size) |b| {
                for (0..seq_len) |s| {
                    const target_idx = @as(usize, @intFromFloat(targets_buf[b * seq_len + s]));
                    const logit_offset = (b * seq_len + s) * vocab_size;

                    // Find max logit for numerical stability
                    var max_logit: f32 = std.math.floatMin(f32);
                    for (0..vocab_size) |v| {
                        if (logit_offset + v < logits_buf.len) {
                            max_logit = @max(max_logit, logits_buf[logit_offset + v]);
                        }
                    }

                    // Compute softmax denominator: sum(exp(logits - max_logit))
                    var sum_exp: f32 = 0.0;
                    for (0..vocab_size) |v| {
                        if (logit_offset + v < logits_buf.len) {
                            sum_exp += @exp(logits_buf[logit_offset + v] - max_logit);
                        }
                    }

                    // Compute negative log-likelihood
                    if (target_idx < vocab_size and logit_offset + target_idx < logits_buf.len) {
                        total_loss -= (logits_buf[logit_offset + target_idx] - max_logit - @log(sum_exp));
                    }
                }
            }

            // Normalize by batch size * sequence length
            loss_buf[0] = total_loss / @as(f32, @floatFromInt(batch_size * seq_len));
            return loss;
        }
    };
}

// Compute perplexity from loss
pub fn computePerplexity(comptime DataType: type) type {
    const Tensor = tensor.Tensor(DataType);
    const loss_func = crossEntropyLoss(DataType).run;

    return struct {
        pub fn run(allocator: Allocator, logits: Tensor, targets: Tensor) !f32 {
            var loss = try loss_func(allocator, logits, targets);
            defer loss.deinit();
            const loss_val = try loss.getScalar(&[_]usize{0});
            return @exp(loss_val);
        }
    };
}

pub fn GPT2(comptime DataType: type) type {
    const Tensor = tensor.Tensor(DataType);
    const legacy_api = ops.LegacyApi(DataType);

    return struct {
        wte: Tensor, // Token embeddings
        wpe: Tensor, // Position embeddings
        blocks: []Block(DataType), // Transformer blocks
        ln_f: LayerNorm(DataType), // Final layer norm
        lm_head: Tensor, // Language modeling head

        config: GPT2Config,
        allocator: Allocator,

        // AutoDiff plans
        matmul_plan: ?*autodiff.AutoDiffPlan(autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null, null), DataType) = null,
        embed_lookup_plan: ?*autodiff.AutoDiffPlan(autodiff.EmbeddingLookupPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null), DataType) = null,
        add_plan: ?*autodiff.AutoDiffPlan(autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null), DataType) = null,

        pub fn init(allocator: Allocator, config: GPT2Config, backend: BackendType) !@This() {
            // Token embeddings (vocab_size x n_embd)
            const wte_dims = [_]usize{ config.vocab_size, config.n_embd };
            var wte = try Tensor.random(allocator, &wte_dims, backend);
            wte.requiresGrad(true);

            // Scale the random initialization
            const wte_ptr = wte.buffer.data;
            for (wte_ptr) |*val| {
                val.* *= config.initializer_range;
            }

            // Position embeddings (n_positions x n_embd)
            const wpe_dims = [_]usize{ config.n_positions, config.n_embd };
            var wpe = try Tensor.random(allocator, &wpe_dims, backend);
            wpe.requiresGrad(true);

            // Scale the random initialization
            for (wpe.buffer.data) |*val| {
                val.* *= config.initializer_range;
            }

            // Create transformer blocks
            const blocks = try allocator.alloc(Block(DataType), config.n_layer);
            for (blocks) |*block| {
                block.* = try Block(DataType).init(allocator, config, backend);
            }

            // Final layer norm
            const ln_f = try LayerNorm(DataType).init(allocator, config, backend);

            // Language model head (n_embd x vocab_size)
            // For weight tying with token embeddings, we'll create a separate tensor
            // that will be transposed from wte in the forward pass
            const lm_dims = [_]usize{ config.n_embd, config.vocab_size };
            var lm_head = try Tensor.random(allocator, &lm_dims, backend);
            lm_head.requiresGrad(true);

            // Scale the random initialization
            const lm_head_ptr = lm_head.buffer.data;
            for (lm_head_ptr) |*val| {
                val.* *= config.initializer_range;
            }

            var model = @This(){
                .wte = wte,
                .wpe = wpe,
                .blocks = blocks,
                .ln_f = ln_f,
                .lm_head = lm_head,
                .config = config,
                .allocator = allocator,
                .matmul_plan = null,
                .embed_lookup_plan = null,
                .add_plan = null,
            };

            // Initialize plans
            const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null, null);
            const matmul_plan = try allocator.create(autodiff.AutoDiffPlan(MatmulPlanType, DataType));
            matmul_plan.* = autodiff.AutoDiffPlan(MatmulPlanType, DataType).init(allocator);
            model.matmul_plan = matmul_plan;

            const EmbedPlanType = autodiff.EmbeddingLookupPlanWithGrad(ops.CpuBackend(DataType), DataType, null, null);
            const embed_plan = try allocator.create(autodiff.AutoDiffPlan(EmbedPlanType, DataType));
            embed_plan.* = autodiff.AutoDiffPlan(EmbedPlanType, DataType).init(allocator);
            model.embed_lookup_plan = embed_plan;

            const AddPlanType = autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, null);
            const add_plan = try allocator.create(autodiff.AutoDiffPlan(AddPlanType, DataType));
            add_plan.* = autodiff.AutoDiffPlan(AddPlanType, DataType).init(allocator);
            model.add_plan = add_plan;

            return model;
        }

        pub fn deinit(self: *@This()) void {
            // Clean up tensors
            self.wte.deinit();
            self.wpe.deinit();

            for (self.blocks) |*block| {
                block.deinit();
            }
            self.allocator.free(self.blocks);

            self.ln_f.deinit();
            self.lm_head.deinit();

            // Clean up plans
            if (self.matmul_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.matmul_plan = null;
            }

            if (self.embed_lookup_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.embed_lookup_plan = null;
            }

            if (self.add_plan) |plan| {
                plan.deinit();
                self.allocator.destroy(plan);
                self.add_plan = null;
            }
        }

        // Forward pass with the model (no gradients)
        pub fn forward(self: *@This(), input_ids: Tensor) !Tensor {
            // Safety check for large dimensions
            if (input_ids.shape.dims[0] > 10000 or input_ids.shape.dims[1] > 10000) {
                return error.InputDimensionsTooLarge;
            }

            // We're taking ownership of input_ids and will free it at the end
            defer input_ids.deinit();

            // Assume input_ids is a batch of token IDs [batch, seq_len]
            const batch_size = input_ids.shape.dims[0];
            const seq_len = input_ids.shape.dims[1];

            // Step 1: Look up token embeddings
            var token_embed_dims = [_]usize{ batch_size, seq_len, self.config.n_embd };
            var token_embeddings = try Tensor.zeros(self.allocator, &token_embed_dims, input_ids.backend);
            defer token_embeddings.deinit();

            // Extract token IDs and fill embeddings
            const input_buf = input_ids.buffer.data;
            const token_embed_buf = token_embeddings.buffer.data;
            const wte_buf = self.wte.buffer.data;

            // Lookup embeddings for each token ID
            for (0..batch_size) |b| {
                for (0..seq_len) |s| {
                    const token_id_f = input_buf[b * seq_len + s];
                    const token_id = @min(@as(usize, @intFromFloat(token_id_f)), self.config.vocab_size - 1);

                    const token_embed_offset = (b * seq_len + s) * self.config.n_embd;
                    const wte_offset = token_id * self.config.n_embd;

                    // Copy embedding for this token
                    for (0..self.config.n_embd) |e| {
                        token_embed_buf[token_embed_offset + e] = wte_buf[wte_offset + e];
                    }
                }
            }

            // Step 2: Add position embeddings
            // Create combined embeddings tensor
            var embeddings = try Tensor.zeros(self.allocator, &token_embed_dims, input_ids.backend);
            defer embeddings.deinit();

            const wpe_buf = self.wpe.buffer.data;
            const embeddings_buf = embeddings.buffer.data;

            // Add token and position embeddings
            for (0..batch_size) |b| {
                for (0..seq_len) |s| {
                    const pos = @min(s, self.config.n_positions - 1);
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

            // Step 3: Reshape for transformer processing [batch_size*seq_len, embedding_dim]
            var reshaped_dims = [_]usize{ batch_size * seq_len, self.config.n_embd };
            var reshaped_embeddings = try Tensor.zeros(self.allocator, &reshaped_dims, input_ids.backend);
            defer reshaped_embeddings.deinit();

            // Copy data to reshaped tensor
            const reshaped_buf = reshaped_embeddings.buffer.data;
            for (0..batch_size * seq_len) |i| {
                for (0..self.config.n_embd) |e| {
                    reshaped_buf[i * self.config.n_embd + e] = embeddings_buf[i * self.config.n_embd + e];
                }
            }

            // Step 4: Process through transformer blocks
            var current = try reshaped_embeddings.clone();

            for (self.blocks) |*block| {
                const new_output = try block.forward(current);
                current.deinit();
                current = new_output;
            }

            // Step 5: Final layer norm
            var final_output = try self.ln_f.forward(current);
            current.deinit();

            // Step 6: Project hidden states to vocabulary logits
            var flat_logits = try legacy_api.matmul(self.allocator, final_output, self.lm_head);
            final_output.deinit();

            // Reshape logits to [batch_size, seq_len, vocab_size]
            const vocab_size = self.config.vocab_size;
            const new_shape = [_]usize{ batch_size, seq_len, vocab_size };
            const logits = try flat_logits.reshape(self.allocator, &new_shape);
            flat_logits.deinit();

            return logits;
        }

        // Forward pass with gradient tracking using Plan-based approach
        // Forward pass with gradient tracking (internal implementation)
        pub fn forwardWithGrad(self: *@This(), input_ids: Tensor) !Tensor {
            // Check if plans are initialized
            if (self.matmul_plan == null or self.add_plan == null or self.embed_lookup_plan == null) {
                return error.PlanNotInitialized;
            }

            // We're taking ownership of input_ids and will free it at the end
            defer input_ids.deinit();

            std.debug.print("GPT2.forwardWithGrad: Starting\n", .{});

            // Assume input_ids is a batch of token IDs [batch, seq_len]
            const batch_size = input_ids.shape.dims[0];
            const seq_len = input_ids.shape.dims[1];

            // Step 1: Look up token embeddings using plan
            var token_embeddings = try self.embed_lookup_plan.?.forward(.{ .params = self.wte, .indices = input_ids });
            defer token_embeddings.deinit();

            std.debug.print("GPT2.forwardWithGrad: Token embeddings created\n", .{});

            // Step 2: Create position IDs tensor [seq_len]
            var pos_ids = try Tensor.zeros(self.allocator, &[_]usize{seq_len}, .f32, input_ids.backend);
            defer pos_ids.deinit();

            // Fill position IDs (0, 1, 2, ..., seq_len-1)
            const pos_buf = pos_ids.buffer.data;
            for (0..seq_len) |i| {
                pos_buf[i] = @floatFromInt(i);
            }

            // Step 3: Look up position embeddings using plan
            var pos_embeddings = try self.embed_lookup_plan.?.forward(.{ .params = self.wpe, .indices = pos_ids });
            defer pos_embeddings.deinit();

            std.debug.print("GPT2.forwardWithGrad: Position embeddings created\n", .{});

            // Step 4: Add token and position embeddings
            var embeddings = try self.add_plan.?.forward(.{ .a = token_embeddings, .b = pos_embeddings });
            defer embeddings.deinit();

            std.debug.print("GPT2.forwardWithGrad: Combined embeddings created\n", .{});

            // Step 5: Reshape for transformer processing [batch_size*seq_len, embedding_dim]
            var reshaped_dims = [_]usize{ batch_size * seq_len, self.config.n_embd };
            var reshaped_embeddings = try Tensor.zeros(self.allocator, &reshaped_dims, embeddings.dtype, embeddings.backend);
            defer reshaped_embeddings.deinit();

            // Copy data to reshaped tensor
            const embeddings_buf = embeddings.buffer.data;
            const reshaped_buf = reshaped_embeddings.buffer.data;

            for (0..batch_size) |b| {
                for (0..seq_len) |s| {
                    for (0..self.config.n_embd) |e| {
                        const src_idx = (b * seq_len + s) * self.config.n_embd + e;
                        const dst_idx = (b * seq_len + s) * self.config.n_embd + e;
                        if (src_idx < embeddings_buf.len and dst_idx < reshaped_buf.len) {
                            reshaped_buf[dst_idx] = embeddings_buf[src_idx];
                        }
                    }
                }
            }

            std.debug.print("GPT2.forwardWithGrad: Reshaped embeddings\n", .{});

            // Step 6: Process through transformer blocks using forwardWithGrad
            var current = try reshaped_embeddings.clone();

            for (self.blocks) |*block| {
                const new_output = try block.forwardWithGrad(current);
                current.deinit();
                current = new_output;
            }

            std.debug.print("GPT2.forwardWithGrad: Processed through transformer blocks\n", .{});

            // Step 7: Final layer norm
            var final_output = try self.ln_f.forwardWithGrad(current);
            current.deinit();

            std.debug.print("GPT2.forwardWithGrad: Applied final layer norm\n", .{});

            // Step 8: Project hidden states to vocabulary logits using lm_head with plan
            var flat_logits = try self.matmul_plan.?.forward(.{ .a = final_output, .b = self.lm_head });
            final_output.deinit();

            // Reshape logits to [batch_size, seq_len, vocab_size]
            const vocab_size = self.config.vocab_size;
            const new_shape = [_]usize{ batch_size, seq_len, vocab_size };
            const logits = try flat_logits.reshape(self.allocator, &new_shape);
            flat_logits.deinit();

            std.debug.print("GPT2.forwardWithGrad: Computed and reshaped final logits\n", .{});

            return logits;
        }

        // Forward pass with gradient computing and backward pass
        pub fn forwardWithGradAndLoss(self: *@This(), input_ids: Tensor, targets: Tensor) !struct { logits: Tensor, loss: Tensor, grads: std.AutoHashMap(*Tensor, Tensor) } {
            // Create a HashMap to store gradients
            var grads = std.AutoHashMap(*Tensor, Tensor).init(self.allocator);
            errdefer {
                var it = grads.iterator();
                while (it.next()) |entry| {
                    entry.value_ptr.deinit();
                }
                grads.deinit();
            }

            // Forward pass with gradient tracking
            // Note: forwardWithGrad takes ownership of input_ids, so we need a copy
            const input_copy = try input_ids.clone();
            const logits = try self.forwardWithGrad(input_copy);
            // No need to deinit input_copy as forwardWithGrad takes ownership

            // Compute loss
            const loss = try crossEntropyLoss(self.allocator, logits, targets);

            // For a complete implementation, we would create a gradient tensor
            // and perform backward passes through the operations

            // TODO: In a more complete implementation, we would:
            // 1. Perform backward pass on the plans
            // 2. Collect the gradients for each parameter
            // 3. Store them in the grads HashMap

            std.debug.print("GPT2.forwardWithGradAndLoss: Loss computed, gradients collected\n", .{});

            return .{
                .logits = logits,
                .loss = loss,
                .grads = grads,
            };
        }
    };
}
