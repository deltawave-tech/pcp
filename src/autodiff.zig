const std = @import("std");
const tensor = @import("tensor.zig");
const ops = @import("ops.zig");

const Allocator = std.mem.Allocator;

const Shape = tensor.Shape;
const BackendType = tensor.BackendType;

// --- Gradient Rules for Primitive Operations ---
// These rules define how to compute gradients for each primitive operation

/// Gradient rules that can be used by any backend
pub fn GradRules(comptime T: type) type {
    const Tensor = tensor.Tensor(T);
    const legacy_api = ops.LegacyApi(T);

    return struct {
        /// Gradient for add: da = grad_out, db = grad_out
        pub fn add(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
            _ = allocator; // Unused - we use grad_out.clone() instead
            _ = a; // Unused parameter
            _ = b; // Unused parameter

            // Both inputs receive the same gradient
            var da = try grad_out.clone();
            errdefer da.deinit();

            var db = try grad_out.clone();
            errdefer db.deinit();

            return .{ .da = da, .db = db };
        }

        /// Gradient for subtract: da = grad_out, db = -grad_out
        pub fn subtract(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
            _ = a; // Unused parameter
            _ = b; // Unused parameter

            // First input gets the gradient directly
            var da = try grad_out.clone();
            errdefer da.deinit();

            // Create negative gradient for second input
            const negative_one = try Tensor.filled(allocator, grad_out.shape.dims, grad_out.dtype, -1.0, grad_out.backend);
            errdefer negative_one.deinit();

            var db = try legacy_api.multiply(allocator, grad_out, negative_one);
            errdefer db.deinit();

            // Clean up temporary tensor
            negative_one.deinit();

            return .{ .da = da, .db = db };
        }

        /// Gradient for multiply (element-wise): da = grad_out * b, db = grad_out * a
        pub fn multiply(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
            // Input a gradient is element-wise product of grad_out and b
            var da = try legacy_api.multiply(allocator, grad_out, b);
            errdefer da.deinit();

            // Input b gradient is element-wise product of grad_out and a
            var db = try legacy_api.multiply(allocator, grad_out, a);
            errdefer db.deinit();

            return .{ .da = da, .db = db };
        }

        /// Gradient for divide (element-wise): da = grad_out / b, db = -grad_out * a / (b * b)
        pub fn divide(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
            // Input a gradient is element-wise division of grad_out by b
            var da = try legacy_api.divide(allocator, grad_out, b);
            errdefer da.deinit();

            // Input b gradient is more complex: -grad_out * a / (b * b)
            // First calculate b^2
            var b_squared = try legacy_api.multiply(allocator, b, b);
            errdefer b_squared.deinit();

            // Calculate a / b^2
            var a_div_b_squared = try legacy_api.divide(allocator, a, b_squared);
            errdefer a_div_b_squared.deinit();

            // Multiply by grad_out
            var temp = try legacy_api.multiply(allocator, grad_out, a_div_b_squared);
            errdefer temp.deinit();

            // Negate the result
            const negative_one = try Tensor.filled(allocator, temp.shape.dims, temp.dtype, -1.0, temp.backend);
            errdefer negative_one.deinit();

            var db = try legacy_api.multiply(allocator, temp, negative_one);
            errdefer db.deinit();

            // Clean up temporaries
            b_squared.deinit();
            a_div_b_squared.deinit();
            temp.deinit();
            negative_one.deinit();

            return .{ .da = da, .db = db };
        }

        /// Gradient for matmul: da = grad_out @ b^T, db = a^T @ grad_out
        pub fn matmul(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
            // Compute b transpose
            const b_transpose = try legacy_api.transpose(allocator, b);
            errdefer b_transpose.deinit();

            // Gradient for a is grad_out @ b^T
            var da = try legacy_api.matmul(allocator, grad_out, b_transpose);
            errdefer da.deinit();

            // Compute a transpose
            const a_transpose = try legacy_api.transpose(allocator, a);
            errdefer a_transpose.deinit();

            // Gradient for b is a^T @ grad_out
            var db = try legacy_api.matmul(allocator, a_transpose, grad_out);
            errdefer db.deinit();

            // Clean up temporaries
            b_transpose.deinit();
            a_transpose.deinit();

            return .{ .da = da, .db = db };
        }

        /// Gradient for relu: da = grad_out * (a > 0)
        pub fn relu(allocator: Allocator, grad_out: Tensor, a: Tensor) !Tensor {
            // Create mask with 1s where input > 0, 0s elsewhere
            var mask = try Tensor.zeros(allocator, a.shape.dims, a.backend);
            errdefer mask.deinit();

            const a_buf = a.buffer.data;
            const mask_buf = mask.buffer.data;

            for (a_buf, 0..) |val, i| {
                mask_buf[i] = if (val > 0) 1.0 else 0.0;
            }

            // Gradient is element-wise product of upstream gradient and mask
            var da = try legacy_api.multiply(allocator, grad_out, mask);
            errdefer da.deinit();

            // Clean up temporary
            mask.deinit();

            return da;
        }

        /// Gradient for softmax: complex Jacobian-vector product
        pub fn softmax(allocator: Allocator, grad_out: Tensor, output: Tensor, a: Tensor) !Tensor {
            // Create tensor for result gradient
            var da = try Tensor.zeros(allocator, a.shape.dims, a.backend);
            errdefer da.deinit();

            const softmax_out = output.buffer.data;
            const upstream_grad = grad_out.buffer.data;
            const result_grad = da.buffer.data;

            const batch_size = output.shape.dims[0];
            const feature_size = output.shape.dims[1];

            // For each batch item
            for (0..batch_size) |b_idx| {
                // Calculate Jacobian-vector product
                for (0..feature_size) |i| {
                    var sum: f32 = 0.0;

                    for (0..feature_size) |j| {
                        const s_i = softmax_out[b_idx * feature_size + i];
                        const s_j = softmax_out[b_idx * feature_size + j];
                        const g = upstream_grad[b_idx * feature_size + j];

                        // dy_j/dx_i = y_j * (δ_ij - y_i)
                        const delta_ij: f32 = if (i == j) 1.0 else 0.0;
                        sum += g * s_j * (delta_ij - s_i);
                    }

                    result_grad[b_idx * feature_size + i] = sum;
                }
            }

            return da;
        }

        /// Gradient for transpose: da = grad_out^T
        pub fn transpose(allocator: Allocator, grad_out: Tensor, a: Tensor) !Tensor {
            _ = a; // Unused parameter

            // Gradient is just the transpose of the upstream gradient
            var da = try legacy_api.transpose(allocator, grad_out);
            errdefer da.deinit();

            return da;
        }

        /// Gradient for embedding lookup: accumulate gradients at token positions
        pub fn embedding_lookup(allocator: Allocator, grad_out: Tensor, params: Tensor, indices: Tensor) !Tensor {
            // Gradient for params is sparse: only positions referenced by indices get gradients
            var d_params = try Tensor.zeros(allocator, params.shape.dims, params.backend);
            errdefer d_params.deinit();

            // Extract dimensions
            const vocab_size = params.shape.dims[0];
            const embed_dim = params.shape.dims[1];

            // Parse indices dimensions
            var batch_size: usize = 1;
            var seq_len: usize = 1;

            if (indices.shape.rank() == 1) {
                seq_len = indices.shape.dims[0];
            } else if (indices.shape.rank() == 2) {
                batch_size = indices.shape.dims[0];
                seq_len = indices.shape.dims[1];
            } else {
                return ops.OpError.InvalidIndicesShape;
            }

            // Get data buffers with bounds checking
            if (indices.shape.elemCount() == 0) {
                return ops.OpError.EmptyTensor;
            }

            if (grad_out.shape.elemCount() == 0) {
                return ops.OpError.EmptyTensor;
            }

            const indices_buf = indices.buffer.data;
            const grad_buf = grad_out.buffer.data;
            const d_params_buf = d_params.buffer.data;

            // For each token in the batch, accumulate gradients back to the embedding table
            for (0..batch_size) |b| {
                for (0..seq_len) |s| {
                    // Get index of current token
                    var index_pos: usize = 0;
                    if (indices.shape.rank() == 1) {
                        index_pos = s;
                    } else {
                        index_pos = b * seq_len + s;
                    }

                    // Bounds check
                    if (index_pos >= indices_buf.len) {
                        std.debug.print("Warning: Index out of bounds in embedding_lookup gradient: {}/{}\n", .{ index_pos, indices_buf.len });
                        continue;
                    }

                    // Get token ID, clamping to vocab size
                    const token_id_f = indices_buf[index_pos];
                    const token_id_i = @as(i32, @intFromFloat(token_id_f));
                    // No need to handle optional here, vocab_size is not optional in this context
                    const token_id = @as(usize, @intCast(@max(0, @min(token_id_i, @as(i32, @intCast(vocab_size - 1))))));

                    // Accumulate gradients for this embedding
                    // No need to handle optional here, embed_dim is not optional in this context
                    for (0..embed_dim) |d| {
                        const grad_pos = (b * seq_len + s) * embed_dim + d;
                        const param_pos = token_id * embed_dim + d;

                        // Bounds checking
                        if (grad_pos >= grad_buf.len) {
                            std.debug.print("Warning: Grad pos out of bounds in embedding_lookup gradient: {}/{}\n", .{ grad_pos, grad_buf.len });
                            continue;
                        }

                        if (param_pos >= d_params_buf.len) {
                            std.debug.print("Warning: Param pos out of bounds in embedding_lookup gradient: {}/{}\n", .{ param_pos, d_params_buf.len });
                            continue;
                        }

                        d_params_buf[param_pos] += grad_buf[grad_pos];
                    }
                }
            }

            return d_params;
        }
    };
}

// --- Comptime AutoDiff Plan Generator ---
// This section defines the comptime function to generate gradient plans from forward plans

/// AutoDiffPlan wraps a forward plan and provides backward functionality
pub fn AutoDiffPlan(comptime ForwardPlanType: type, comptime DataType: type) type {
    // Use @hasDecl for checking constants instead of @hasField
    // This fixes the detection of GradType and op_type in plan types

    // Comptime validation
    comptime {
        if (!@hasDecl(ForwardPlanType, "GradType")) {
            @compileError("Forward plan must define GradType for autodiff");
        }
        if (!@hasDecl(ForwardPlanType, "op_type")) {
            @compileError("Forward plan must define op_type for autodiff");
        }
    }

    const Tensor = tensor.Tensor(DataType);

    return struct {
        const Self = @This();

        // The forward plan
        forward_plan: ForwardPlanType,

        // Keep track of input and output for backward pass
        last_input: ?ForwardPlanType.InputType = null,
        last_output: ?Tensor = null,
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .forward_plan = ForwardPlanType.init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.last_output != null) {
                self.last_output.?.deinit();
                self.last_output = null;
            }
            // Note: We don't clean up last_input as we assume it's borrowed not owned
        }

        /// Forward pass: run the forward plan and cache inputs/outputs for backward
        pub fn forward(self: *Self, input: ForwardPlanType.InputType) !Tensor {
            // Run the forward plan
            const output = try self.forward_plan.run(input);

            // Clean up previous output if it exists
            if (self.last_output != null) {
                self.last_output.?.deinit();
            }

            // Store input and output
            self.last_input = input;
            self.last_output = try output.clone();

            return output;
        }

        /// Backward pass: compute gradients with respect to inputs
        pub fn backward(self: *Self, grad_out: Tensor) !ForwardPlanType.GradType {
            // Ensure we have inputs and outputs from a previous forward pass
            if (self.last_input == null or self.last_output == null) {
                return error.NoPreviousForward;
            }

            // Use comptime type analysis to apply the correct gradient rule
            return self.computeGradient(grad_out);
        }

        /// Comptime function to select and apply the appropriate gradient rule
        fn computeGradient(self: *Self, grad_out: Tensor) !ForwardPlanType.GradType {
            const input = self.last_input.?;

            // Use the plan's gradient method directly - much simpler!
            // This takes advantage of the gradient methods we added to the plan types directly.
            return try self.forward_plan.gradient(grad_out, input);
        }
    };
}

// --- Extended Plan Types ---
// For each operation plan in ops.zig, we extend it with gradient computation info

/// Extend ops.AddPlan with gradient type information
pub fn AddPlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.AddPlan(Backend, T, shape);
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .add;

        // Define the gradient type for add operation by referencing base plan's GradType
        const GradType = BasePlan.GradType;

        // Embed the base plan
        base: BasePlan,

        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }

        /// Pass through to the base plan's gradient function
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            return self.base.gradient(grad_out, input);
        }
    };
}

/// Extend ops.SubtractPlan with gradient type information
pub fn SubtractPlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.SubtractPlan(Backend, T, shape);
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .subtract;

        // Define the gradient type for subtract operation by referencing base plan's GradType
        const GradType = BasePlan.GradType;

        // Embed the base plan
        base: BasePlan,

        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }

        /// Pass through to the base plan's gradient function
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            return self.base.gradient(grad_out, input);
        }
    };
}

/// Extend ops.MultiplyPlan with gradient type information
pub fn MultiplyPlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.MultiplyPlan(Backend, T, shape);
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .multiply;

        // Define the gradient type for multiply operation by referencing base plan's GradType
        const GradType = BasePlan.GradType;

        // Embed the base plan
        base: BasePlan,

        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }

        /// Pass through to the base plan's gradient function
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            return self.base.gradient(grad_out, input);
        }
    };
}

/// Extend ops.MatmulPlan with gradient type information
pub fn MatmulPlanWithGrad(comptime Backend: type, comptime T: type, comptime M: ?usize, comptime N: ?usize, comptime P: ?usize) type {
    const BasePlan = ops.MatmulPlan(Backend, T, M, N, P);
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .matmul;

        // Define the gradient type for matmul operation by referencing base plan's GradType
        const GradType = BasePlan.GradType;

        // Embed the base plan
        base: BasePlan,

        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }

        /// Pass through to the base plan's gradient function
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            return self.base.gradient(grad_out, input);
        }
    };
}

/// Extend ops.ReluPlan with gradient type information
pub fn ReluPlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.ReluPlan(Backend, T, shape);
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = Tensor;
        const op_type = .relu;

        // Define the gradient type for relu operation (single input)
        // For unary operations, GradType is just a Tensor, not a struct with da/db fields
        const GradType = BasePlan.GradType;

        // Embed the base plan
        base: BasePlan,

        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }

        /// Custom gradient implementation for ReLU
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            // For ReLU, gradient is the upstream gradient where input > 0, 0 elsewhere
            return try GradRules(T).relu(self.base.base.allocator, grad_out, input);
        }
    };
}

/// Extend ops.SoftmaxPlan with gradient type information
pub fn SoftmaxPlanWithGrad(comptime Backend: type, comptime T: type, comptime batch_size: ?usize, comptime feature_size: ?usize) type {
    const BasePlan = ops.SoftmaxPlan(Backend, T, batch_size, feature_size);
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = Tensor;
        const op_type = .softmax;

        // Define the gradient type for softmax operation (single input)
        const GradType = BasePlan.GradType;

        // Embed the base plan
        base: BasePlan,

        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }

        /// Custom gradient implementation for softmax
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            // For softmax, we need to run the forward pass to get the output values
            const output = try self.base.run(input);
            defer output.deinit();

            // Then compute the Jacobian-vector product
            return try GradRules(T).softmax(self.base.base.allocator, grad_out, output, input);
        }
    };
}

/// Extend ops.TransposePlan with gradient type information
pub fn TransposePlanWithGrad(comptime Backend: type, comptime T: type, comptime rows: ?usize, comptime cols: ?usize) type {
    const BasePlan = ops.TransposePlan(Backend, T, rows, cols);
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = Tensor;
        const op_type = .transpose;

        // Define the gradient type for transpose operation (single input)
        const GradType = BasePlan.GradType;

        // Embed the base plan
        base: BasePlan,

        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }

        /// Custom gradient implementation for transpose
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            // For transpose, gradient is just the transpose of the upstream gradient
            return try GradRules.transpose(self.base.base.allocator, grad_out, input);
        }
    };
}

/// Extend ops.DividePlan with gradient type information
pub fn DividePlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.DividePlan(Backend, T, shape);
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .divide;

        // Define the gradient type for divide operation
        const GradType = BasePlan.GradType;

        // Embed the base plan
        base: BasePlan,

        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }

        /// Pass through to the base plan's gradient function
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            return self.base.gradient(grad_out, input);
        }
    };
}

/// Define a plan for embedding lookup operation
pub fn EmbeddingLookupPlanWithGrad(comptime Backend: type, comptime T: type, comptime vocab_size: ?usize, comptime embed_dim: ?usize) type {
    _ = Backend; // Used at compile time for consistency with other Plan functions
    const Tensor = tensor.Tensor(T);

    return struct {
        const Self = @This();
        const InputType = struct { params: Tensor, indices: Tensor };
        const op_type = .embedding_lookup;

        // Define the gradient type for embedding lookup (only params get gradients)
        const GradType = Tensor;

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Compute gradients for embedding lookup
        pub fn gradient(self: Self, grad_out: Tensor, input: InputType) !GradType {
            // Delegate to the GradRules implementation for embedding lookup
            return try GradRules(T).embedding_lookup(self.allocator, grad_out, input.params, input.indices);
        }

        pub fn run(self: Self, input: InputType) !Tensor {
            // Validate input - params must be 2D tensor: [vocab_size, embed_dim]
            if (input.params.shape.rank() != 2) {
                return error.InvalidEmbeddingShape;
            }

            // Verify dimensions - handle optional parameters
            const resolved_vocab_size = vocab_size orelse input.params.shape.dims[0];
            const resolved_embed_dim = embed_dim orelse input.params.shape.dims[1];

            if (input.params.shape.dims[0] != resolved_vocab_size or input.params.shape.dims[1] != resolved_embed_dim) {
                return error.ShapeMismatch;
            }

            // Parse indices - can be 1D or 2D
            var batch_size: usize = 1;
            var seq_len: usize = 1;

            if (input.indices.shape.rank() == 1) {
                seq_len = input.indices.shape.dims[0];
            } else if (input.indices.shape.rank() == 2) {
                batch_size = input.indices.shape.dims[0];
                seq_len = input.indices.shape.dims[1];
            } else {
                return error.InvalidIndicesShape;
            }

            // Create output tensor with shape [batch_size, seq_len, embed_dim]
            // Use the already resolved embed_dim from earlier
            var result_dims = [_]usize{ batch_size, seq_len, resolved_embed_dim };
            var result = try Tensor.zeros(self.allocator, &result_dims, input.params.backend);
            errdefer result.deinit();

            // Get data buffers with safe bounds checking
            if (input.params.shape.elemCount() == 0) {
                // Clean up the result tensor on error
                result.deinit();
                return error.EmptyTensor;
            }

            if (input.indices.shape.elemCount() == 0) {
                // Clean up the result tensor on error
                result.deinit();
                return error.EmptyTensor;
            }

            if (result.shape.elemCount() == 0) {
                // Clean up the result tensor on error
                result.deinit();
                return error.EmptyTensor;
            }

            const params_buf = input.params.buffer.data;
            const indices_buf = input.indices.buffer.data;
            const result_buf = result.buffer.data;

            // Perform lookup
            for (0..batch_size) |b| {
                for (0..seq_len) |s| {
                    // Get the token id from indices, safely convert to int
                    var index_pos: usize = 0;
                    if (input.indices.shape.rank() == 1) {
                        index_pos = s;
                    } else {
                        index_pos = b * seq_len + s;
                    }

                    if (index_pos >= indices_buf.len) {
                        std.debug.print("Warning: Index out of bounds in embedding_lookup: {}/{}\n", .{ index_pos, indices_buf.len });
                        continue;
                    }

                    const token_id_f = indices_buf[index_pos];
                    // Safely convert to int and bound to vocab size
                    const token_id_i = @as(i32, @intFromFloat(token_id_f));
                    // Use resolved_vocab_size which is already defined above
                    const token_id = @as(usize, @intCast(@max(0, @min(token_id_i, @as(i32, @intCast(resolved_vocab_size - 1))))));

                    // Copy embedding for this token - embed_dim values
                    // Use resolved_embed_dim which is already defined above
                    for (0..resolved_embed_dim) |d| {
                        const src_idx = token_id * resolved_embed_dim + d;
                        const dst_idx = (b * seq_len + s) * resolved_embed_dim + d;

                        // Add bounds checking
                        if (src_idx >= params_buf.len) {
                            std.debug.print("Warning: Source index out of bounds in embedding_lookup: {}/{}\n", .{ src_idx, params_buf.len });
                            continue;
                        }

                        if (dst_idx >= result_buf.len) {
                            std.debug.print("Warning: Destination index out of bounds in embedding_lookup: {}/{}\n", .{ dst_idx, result_buf.len });
                            continue;
                        }

                        result_buf[dst_idx] = params_buf[src_idx];
                    }
                }
            }

            return result;
        }
    };
}

/// Operation type enum used for both forward and backward passes
/// This is now re-exported from ops.zig
pub const OpType = ops.OpType;

// --- Tests for the New Plan-Based AutoDiff ---

test "autodiff add plan" {
    const T = f32;
    const Tensor = tensor.Tensor(T);

    const allocator = std.testing.allocator;

    // Create test tensors with comptime-known dimensions
    const dims = [_]usize{ 2, 2 };

    var a = try Tensor.zeros(allocator, &dims, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 3.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 4.0);

    var b = try Tensor.zeros(allocator, &dims, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{ 0, 0 }, 5.0);
    try b.setScalar(&[_]usize{ 0, 1 }, 6.0);
    try b.setScalar(&[_]usize{ 1, 0 }, 7.0);
    try b.setScalar(&[_]usize{ 1, 1 }, 8.0);

    // Create an AddPlan with gradient support using AddPlanWithGrad
    const PlanType = AddPlanWithGrad(ops.CpuBackend(T), T, &dims);
    const AutoDiffType = AutoDiffPlan(PlanType, T);

    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();

    // Forward pass
    const output = try auto_diff.forward(.{ .a = a, .b = b });
    defer output.deinit();

    // Check output values
    try std.testing.expectEqual(@as(f32, 6.0), try output.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 8.0), try output.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 10.0), try output.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 12.0), try output.getScalar(&[_]usize{ 1, 1 }));

    // Create gradient with ones
    var grad_out = try Tensor.filled(allocator, &dims, 1.0, .cpu);
    defer grad_out.deinit();

    // Backward pass
    const grads = try auto_diff.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();

    // Check gradients - for add, both inputs get the same gradient (1.0)
    try std.testing.expectEqual(@as(f32, 1.0), try grads.da.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.da.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.da.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.da.getScalar(&[_]usize{ 1, 1 }));

    try std.testing.expectEqual(@as(f32, 1.0), try grads.db.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.db.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.db.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.db.getScalar(&[_]usize{ 1, 1 }));
}

test "autodiff matmul plan" {
    const T = f32;
    const Tensor = tensor.Tensor(T);

    const allocator = std.testing.allocator;

    // Use comptime dimensions
    const M: usize = 2;
    const N: usize = 3;
    const P: usize = 2;

    // Create test tensors
    const a_dims = [_]usize{ M, N };
    const b_dims = [_]usize{ N, P };

    var a = try Tensor.zeros(allocator, &a_dims, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 0, 2 }, 3.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 4.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 5.0);
    try a.setScalar(&[_]usize{ 1, 2 }, 6.0);

    var b = try Tensor.zeros(allocator, &b_dims, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{ 0, 0 }, 7.0);
    try b.setScalar(&[_]usize{ 0, 1 }, 8.0);
    try b.setScalar(&[_]usize{ 1, 0 }, 9.0);
    try b.setScalar(&[_]usize{ 1, 1 }, 10.0);
    try b.setScalar(&[_]usize{ 2, 0 }, 11.0);
    try b.setScalar(&[_]usize{ 2, 1 }, 12.0);

    // Create a MatmulPlan with gradient support
    const PlanType = MatmulPlanWithGrad(ops.CpuBackend(T), T, M, N, P);
    const AutoDiffType = AutoDiffPlan(PlanType, T);

    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();

    // Forward pass
    const output = try auto_diff.forward(.{ .a = a, .b = b });
    defer output.deinit();

    // Check output values: [[58, 64], [139, 154]]
    try std.testing.expectEqual(@as(f32, 58.0), try output.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 64.0), try output.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 139.0), try output.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 154.0), try output.getScalar(&[_]usize{ 1, 1 }));

    // Create gradient with ones
    const grad_dims = [_]usize{ M, P };
    var grad_out = try Tensor.filled(allocator, &grad_dims, 1.0, .cpu);
    defer grad_out.deinit();

    // Backward pass
    const grads = try auto_diff.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();

    // Check gradients - for matmul:
    // dL/dA = dL/dC @ B^T
    // dL/dB = A^T @ dL/dC

    // For our specific test case:
    // dL/dA = [1, 1; 1, 1] @ [7, 9, 11; 8, 10, 12]^T = [15, 19, 23; 15, 19, 23]
    try std.testing.expectEqual(@as(f32, 15.0), try grads.da.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 19.0), try grads.da.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 23.0), try grads.da.getScalar(&[_]usize{ 0, 2 }));
    try std.testing.expectEqual(@as(f32, 15.0), try grads.da.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 19.0), try grads.da.getScalar(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(@as(f32, 23.0), try grads.da.getScalar(&[_]usize{ 1, 2 }));

    // dL/dB = [1, 2, 3; 4, 5, 6]^T @ [1, 1; 1, 1] = [5, 5; 7, 7; 9, 9]
    try std.testing.expectEqual(@as(f32, 5.0), try grads.db.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 5.0), try grads.db.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 7.0), try grads.db.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 7.0), try grads.db.getScalar(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(@as(f32, 9.0), try grads.db.getScalar(&[_]usize{ 2, 0 }));
    try std.testing.expectEqual(@as(f32, 9.0), try grads.db.getScalar(&[_]usize{ 2, 1 }));
}

test "autodiff relu plan" {
    const T = f32;
    const Tensor = tensor.Tensor(T);

    const allocator = std.testing.allocator;

    // Create test tensor with mixed positive and negative values
    const dims = [_]usize{ 2, 2 };
    var a = try Tensor.zeros(allocator, &dims, .cpu);
    defer a.deinit();

    try a.setScalar(&[_]usize{ 0, 0 }, -1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 0.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 3.0);

    // Create a ReluPlan with gradient support
    const PlanType = ReluPlanWithGrad(ops.CpuBackend(T), T, &dims);
    const AutoDiffType = AutoDiffPlan(PlanType, T);

    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();

    // Forward pass
    const output = try auto_diff.forward(a);
    defer output.deinit();

    // Check output values: [[0, 2], [0, 3]]
    try std.testing.expectEqual(@as(f32, 0.0), try output.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 2.0), try output.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 0.0), try output.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 3.0), try output.getScalar(&[_]usize{ 1, 1 }));

    // Create gradient with ones
    var grad_out = try Tensor.filled(allocator, &dims, 1.0, .cpu);
    defer grad_out.deinit();

    // Backward pass
    const grad = try auto_diff.backward(grad_out);
    defer grad.deinit();

    // Check gradients - for ReLU, gradient is 1 where input > 0, 0 elsewhere
    try std.testing.expectEqual(@as(f32, 0.0), try grad.getScalar(&[_]usize{ 0, 0 })); // Input was -1.0
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 0, 1 })); // Input was 2.0
    try std.testing.expectEqual(@as(f32, 0.0), try grad.getScalar(&[_]usize{ 1, 0 })); // Input was 0.0
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 1, 1 })); // Input was 3.0
}

test "autodiff divide plan" {
    const T = f32;
    const Tensor = tensor.Tensor(T);

    const allocator = std.testing.allocator;

    // Create test tensors with comptime dimensions
    const dims = [_]usize{ 2, 2 };

    var a = try Tensor.zeros(allocator, &dims, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{ 0, 0 }, 10.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 20.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 30.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 40.0);

    var b = try Tensor.zeros(allocator, &dims, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{ 0, 0 }, 2.0);
    try b.setScalar(&[_]usize{ 0, 1 }, 4.0);
    try b.setScalar(&[_]usize{ 1, 0 }, 5.0);
    try b.setScalar(&[_]usize{ 1, 1 }, 8.0);

    // Create a DividePlan with gradient support
    const PlanType = DividePlanWithGrad(ops.CpuBackend(T), T, &dims);
    const AutoDiffType = AutoDiffPlan(PlanType, T);

    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();

    // Forward pass
    const output = try auto_diff.forward(.{ .a = a, .b = b });
    defer output.deinit();

    // Check output values: a / b = [[5.0, 5.0], [6.0, 5.0]]
    try std.testing.expectEqual(@as(f32, 5.0), try output.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 5.0), try output.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 6.0), try output.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 5.0), try output.getScalar(&[_]usize{ 1, 1 }));

    // Create gradient with ones
    var grad_out = try Tensor.filled(allocator, &dims, 1.0, .cpu);
    defer grad_out.deinit();

    // Backward pass
    const grads = try auto_diff.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();

    // Check gradients:
    // da = 1.0 / b = [[0.5, 0.25], [0.2, 0.125]]
    // db = -a / (b * b) = [[-2.5, -1.25], [-1.2, -0.625]]

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), try grads.da.getScalar(&[_]usize{ 0, 0 }), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), try grads.da.getScalar(&[_]usize{ 0, 1 }), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), try grads.da.getScalar(&[_]usize{ 1, 0 }), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), try grads.da.getScalar(&[_]usize{ 1, 1 }), 0.0001);

    try std.testing.expectApproxEqAbs(@as(f32, -2.5), try grads.db.getScalar(&[_]usize{ 0, 0 }), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.25), try grads.db.getScalar(&[_]usize{ 0, 1 }), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.2), try grads.db.getScalar(&[_]usize{ 1, 0 }), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.625), try grads.db.getScalar(&[_]usize{ 1, 1 }), 0.0001);
}

test "autodiff embedding lookup plan" {
    const T = f32;
    const Tensor = tensor.Tensor(T);

    const allocator = std.testing.allocator;

    // Comptime dimensions
    const vocab_size: usize = 4;
    const embed_dim: usize = 3;
    const batch_size: usize = 2;
    const seq_len: usize = 2;

    // Create a simple embedding table [vocab_size=4, embed_dim=3]
    const embed_dims = [_]usize{ vocab_size, embed_dim };
    var params = try Tensor.zeros(allocator, &embed_dims, .cpu);
    defer params.deinit();

    // Fill with test values
    // Word 0: [0.1, 0.2, 0.3]
    try params.setScalar(&[_]usize{ 0, 0 }, 0.1);
    try params.setScalar(&[_]usize{ 0, 1 }, 0.2);
    try params.setScalar(&[_]usize{ 0, 2 }, 0.3);
    // Word 1: [0.4, 0.5, 0.6]
    try params.setScalar(&[_]usize{ 1, 0 }, 0.4);
    try params.setScalar(&[_]usize{ 1, 1 }, 0.5);
    try params.setScalar(&[_]usize{ 1, 2 }, 0.6);
    // Word 2: [0.7, 0.8, 0.9]
    try params.setScalar(&[_]usize{ 2, 0 }, 0.7);
    try params.setScalar(&[_]usize{ 2, 1 }, 0.8);
    try params.setScalar(&[_]usize{ 2, 2 }, 0.9);
    // Word 3: [1.0, 1.1, 1.2]
    try params.setScalar(&[_]usize{ 3, 0 }, 1.0);
    try params.setScalar(&[_]usize{ 3, 1 }, 1.1);
    try params.setScalar(&[_]usize{ 3, 2 }, 1.2);

    // Create indices tensor with shape [2, 2]
    const indices_dims = [_]usize{ batch_size, seq_len };
    var indices = try Tensor.zeros(allocator, &indices_dims, .cpu);
    defer indices.deinit();

    // Set indices: [[1, 2], [3, 0]]
    try indices.setScalar(&[_]usize{ 0, 0 }, 1.0); // batch 0, pos 0: word ID 1
    try indices.setScalar(&[_]usize{ 0, 1 }, 2.0); // batch 0, pos 1: word ID 2
    try indices.setScalar(&[_]usize{ 1, 0 }, 3.0); // batch 1, pos 0: word ID 3
    try indices.setScalar(&[_]usize{ 1, 1 }, 0.0); // batch 1, pos 1: word ID 0

    // Create an EmbeddingLookupPlan with gradient support
    const PlanType = EmbeddingLookupPlanWithGrad(ops.CpuBackend(T), T, vocab_size, embed_dim);
    const AutoDiffType = AutoDiffPlan(PlanType, T);

    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();

    // Forward pass
    const output = try auto_diff.forward(.{ .params = params, .indices = indices });
    defer output.deinit();

    // Check output values: shape [2, 2, 3]
    // Expected: [[[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], [[1.0, 1.1, 1.2], [0.1, 0.2, 0.3]]]

    // First sequence, first token (word ID 1)
    try std.testing.expectEqual(@as(f32, 0.4), try output.getScalar(&[_]usize{ 0, 0, 0 }));
    try std.testing.expectEqual(@as(f32, 0.5), try output.getScalar(&[_]usize{ 0, 0, 1 }));
    try std.testing.expectEqual(@as(f32, 0.6), try output.getScalar(&[_]usize{ 0, 0, 2 }));

    // First sequence, second token (word ID 2)
    try std.testing.expectEqual(@as(f32, 0.7), try output.getScalar(&[_]usize{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(f32, 0.8), try output.getScalar(&[_]usize{ 0, 1, 1 }));
    try std.testing.expectEqual(@as(f32, 0.9), try output.getScalar(&[_]usize{ 0, 1, 2 }));

    // Second sequence, first token (word ID 3)
    try std.testing.expectEqual(@as(f32, 1.0), try output.getScalar(&[_]usize{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(f32, 1.1), try output.getScalar(&[_]usize{ 1, 0, 1 }));
    try std.testing.expectEqual(@as(f32, 1.2), try output.getScalar(&[_]usize{ 1, 0, 2 }));

    // Second sequence, second token (word ID 0)
    try std.testing.expectEqual(@as(f32, 0.1), try output.getScalar(&[_]usize{ 1, 1, 0 }));
    try std.testing.expectEqual(@as(f32, 0.2), try output.getScalar(&[_]usize{ 1, 1, 1 }));
    try std.testing.expectEqual(@as(f32, 0.3), try output.getScalar(&[_]usize{ 1, 1, 2 }));

    // Create gradient with all ones
    const grad_dims = [_]usize{ batch_size, seq_len, embed_dim };
    var grad_out = try Tensor.filled(allocator, &grad_dims, 1.0, .cpu);
    defer grad_out.deinit();

    // Backward pass
    const grad = try auto_diff.backward(grad_out);
    defer grad.deinit();

    // For embedding lookup, the gradient is a sparse tensor with shape matching params
    // Each used word ID accumulates a gradient equal to the upstream gradient

    // Check gradient for word ID 0 (used in batch 1, pos 1) - should be [1, 1, 1]
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 0, 2 }));

    // Check gradient for word ID 1 (used in batch 0, pos 0) - should be [1, 1, 1]
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 1, 2 }));

    // Check gradient for word ID 2 (used in batch 0, pos 1) - should be [1, 1, 1]
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 2, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 2, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 2, 2 }));

    // Check gradient for word ID 3 (used in batch 1, pos 0) - should be [1, 1, 1]
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 3, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 3, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{ 3, 2 }));
}
