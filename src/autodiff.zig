const std = @import("std");
const tensor = @import("tensor.zig");
const ops = @import("ops.zig");

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

// --- Gradient Rules for Primitive Operations ---
// These rules define how to compute gradients for each primitive operation

/// Gradient rules that can be used by any backend
pub const GradRules = struct {
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
        
        var db = try ops.multiply(allocator, grad_out, negative_one);
        errdefer db.deinit();
        
        // Clean up temporary tensor
        negative_one.deinit();
        
        return .{ .da = da, .db = db };
    }

    /// Gradient for multiply (element-wise): da = grad_out * b, db = grad_out * a
    pub fn multiply(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
        // Input a gradient is element-wise product of grad_out and b
        var da = try ops.multiply(allocator, grad_out, b);
        errdefer da.deinit();
        
        // Input b gradient is element-wise product of grad_out and a
        var db = try ops.multiply(allocator, grad_out, a);
        errdefer db.deinit();
        
        return .{ .da = da, .db = db };
    }

    /// Gradient for divide (element-wise): da = grad_out / b, db = -grad_out * a / (b * b)
    pub fn divide(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
        // Input a gradient is element-wise division of grad_out by b
        var da = try ops.divide(allocator, grad_out, b);
        errdefer da.deinit();
        
        // Input b gradient is more complex: -grad_out * a / (b * b)
        // First calculate b^2
        var b_squared = try ops.multiply(allocator, b, b);
        errdefer b_squared.deinit();
        
        // Calculate a / b^2
        var a_div_b_squared = try ops.divide(allocator, a, b_squared);
        errdefer a_div_b_squared.deinit();
        
        // Multiply by grad_out
        var temp = try ops.multiply(allocator, grad_out, a_div_b_squared);
        errdefer temp.deinit();
        
        // Negate the result
        const negative_one = try Tensor.filled(allocator, temp.shape.dims, temp.dtype, -1.0, temp.backend);
        errdefer negative_one.deinit();
        
        var db = try ops.multiply(allocator, temp, negative_one);
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
        const b_transpose = try ops.transpose(allocator, b);
        errdefer b_transpose.deinit();
        
        // Gradient for a is grad_out @ b^T
        var da = try ops.matmul(allocator, grad_out, b_transpose);
        errdefer da.deinit();
        
        // Compute a transpose
        const a_transpose = try ops.transpose(allocator, a);
        errdefer a_transpose.deinit();
        
        // Gradient for b is a^T @ grad_out
        var db = try ops.matmul(allocator, a_transpose, grad_out);
        errdefer db.deinit();
        
        // Clean up temporaries
        b_transpose.deinit();
        a_transpose.deinit();
        
        return .{ .da = da, .db = db };
    }

    /// Gradient for relu: da = grad_out * (a > 0)
    pub fn relu(allocator: Allocator, grad_out: Tensor, a: Tensor) !Tensor {
        // Create mask with 1s where input > 0, 0s elsewhere
        var mask = try Tensor.zeros(allocator, a.shape.dims, a.dtype, a.backend);
        errdefer mask.deinit();
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const mask_buf = ptrCastHelper([*]f32, mask.buffer.data.ptr)[0..mask.shape.elemCount()];
        
        for (a_buf, 0..) |val, i| {
            mask_buf[i] = if (val > 0) 1.0 else 0.0;
        }
        
        // Gradient is element-wise product of upstream gradient and mask
        var da = try ops.multiply(allocator, grad_out, mask);
        errdefer da.deinit();
        
        // Clean up temporary
        mask.deinit();
        
        return da;
    }

    /// Gradient for softmax: complex Jacobian-vector product
    pub fn softmax(allocator: Allocator, grad_out: Tensor, output: Tensor, a: Tensor) !Tensor {
        // Create tensor for result gradient
        var da = try Tensor.zeros(allocator, a.shape.dims, a.dtype, a.backend);
        errdefer da.deinit();
        
        const softmax_out = ptrCastHelper([*]f32, output.buffer.data.ptr)[0..output.shape.elemCount()];
        const upstream_grad = ptrCastHelper([*]f32, grad_out.buffer.data.ptr)[0..grad_out.shape.elemCount()];
        const result_grad = ptrCastHelper([*]f32, da.buffer.data.ptr)[0..da.shape.elemCount()];
        
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
        var da = try ops.transpose(allocator, grad_out);
        errdefer da.deinit();
        
        return da;
    }
    
    /// Gradient for embedding lookup: accumulate gradients at token positions
    pub fn embedding_lookup(allocator: Allocator, grad_out: Tensor, 
                           params: Tensor, indices: Tensor) !Tensor {
        // Gradient for params is sparse: only positions referenced by indices get gradients
        var d_params = try Tensor.zeros(allocator, params.shape.dims, params.dtype, params.backend);
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
        
        const indices_buf = ptrCastHelper([*]f32, indices.buffer.data.ptr)[0..indices.shape.elemCount()];
        const grad_buf = ptrCastHelper([*]f32, grad_out.buffer.data.ptr)[0..grad_out.shape.elemCount()];
        const d_params_buf = ptrCastHelper([*]f32, d_params.buffer.data.ptr)[0..d_params.shape.elemCount()];
        
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
                    std.debug.print("Warning: Index out of bounds in embedding_lookup gradient: {}/{}\n", 
                        .{index_pos, indices_buf.len});
                    continue;
                }
                
                // Get token ID, clamping to vocab size
                const token_id_f = indices_buf[index_pos];
                const token_id_i = @as(i32, @intFromFloat(token_id_f));
                const token_id = @as(usize, @intCast(@max(0, @min(token_id_i, @as(i32, @intCast(vocab_size - 1))))));
                
                // Accumulate gradients for this embedding
                for (0..embed_dim) |d| {
                    const grad_pos = (b * seq_len + s) * embed_dim + d;
                    const param_pos = token_id * embed_dim + d;
                    
                    // Bounds checking
                    if (grad_pos >= grad_buf.len) {
                        std.debug.print("Warning: Grad pos out of bounds in embedding_lookup gradient: {}/{}\n", 
                            .{grad_pos, grad_buf.len});
                        continue;
                    }
                    
                    if (param_pos >= d_params_buf.len) {
                        std.debug.print("Warning: Param pos out of bounds in embedding_lookup gradient: {}/{}\n", 
                            .{param_pos, d_params_buf.len});
                        continue;
                    }
                    
                    d_params_buf[param_pos] += grad_buf[grad_pos];
                }
            }
        }
        
        return d_params;
    }
};

// --- Comptime AutoDiff Plan Generator ---
// This section defines the comptime function to generate gradient plans from forward plans

/// AutoDiffPlan wraps a forward plan and provides backward functionality
pub fn AutoDiffPlan(comptime ForwardPlanType: type) type {
    // Comptime validation
    comptime {
        if (!@hasField(ForwardPlanType, "GradType")) {
            @compileError("Forward plan must define GradType for autodiff");
        }
        if (!@hasField(ForwardPlanType, "op_type")) {
            @compileError("Forward plan must define op_type for autodiff");
        }
    }
    
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
            const output = self.last_output.?;
            
            // Use comptime type information to select the right gradient function
            switch (ForwardPlanType.op_type) {
                // Binary operations that need both inputs
                .add => {
                    return try GradRules.add(self.allocator, grad_out, input.a, input.b);
                },
                .subtract => {
                    return try GradRules.subtract(self.allocator, grad_out, input.a, input.b);
                },
                .multiply => {
                    return try GradRules.multiply(self.allocator, grad_out, input.a, input.b);
                },
                .divide => {
                    return try GradRules.divide(self.allocator, grad_out, input.a, input.b);
                },
                .matmul => {
                    return try GradRules.matmul(self.allocator, grad_out, input.a, input.b);
                },
                
                // Unary operations with single input tensor
                .relu => {
                    return try GradRules.relu(self.allocator, grad_out, input);
                },
                .softmax => {
                    return try GradRules.softmax(self.allocator, grad_out, output, input);
                },
                .transpose => {
                    return try GradRules.transpose(self.allocator, grad_out, input);
                },
                
                // Special case for embedding lookup which has structured input
                .embedding_lookup => {
                    return try GradRules.embedding_lookup(self.allocator, grad_out, input.params, input.indices);
                },
                
                // Default error case
                else => {
                    @compileError("Unsupported operation type for gradient: " ++ @tagName(ForwardPlanType.op_type));
                },
            }
        }
    };
}

// --- Extended Plan Types ---
// For each operation plan in ops.zig, we extend it with gradient computation info

/// Extend ops.AddPlan with gradient type information
pub fn AddPlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.AddPlan(Backend, T, shape);
    
    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .add;
        
        // Define the gradient type for add operation
        const GradType = struct { da: Tensor, db: Tensor };
        
        // Embed the base plan
        base: BasePlan,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
}

/// Extend ops.SubtractPlan with gradient type information
pub fn SubtractPlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.SubtractPlan(Backend, T, shape);
    
    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .subtract;
        
        // Define the gradient type for subtract operation
        const GradType = struct { da: Tensor, db: Tensor };
        
        // Embed the base plan
        base: BasePlan,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
}

/// Extend ops.MultiplyPlan with gradient type information
pub fn MultiplyPlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.MultiplyPlan(Backend, T, shape);
    
    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .multiply;
        
        // Define the gradient type for multiply operation
        const GradType = struct { da: Tensor, db: Tensor };
        
        // Embed the base plan
        base: BasePlan,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
}

/// Extend ops.MatmulPlan with gradient type information
pub fn MatmulPlanWithGrad(comptime Backend: type, comptime T: type, comptime M: ?usize, comptime N: ?usize, comptime P: ?usize) type {
    const BasePlan = ops.MatmulPlan(Backend, T, M, N, P);
    
    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .matmul;
        
        // Define the gradient type for matmul operation
        const GradType = struct { da: Tensor, db: Tensor };
        
        // Embed the base plan
        base: BasePlan,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
}

/// Extend ops.ReluPlan with gradient type information
pub fn ReluPlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.ReluPlan(Backend, T, shape);
    
    return struct {
        const Self = @This();
        const InputType = Tensor;
        const op_type = .relu;
        
        // Define the gradient type for relu operation (single input)
        const GradType = Tensor;
        
        // Embed the base plan
        base: BasePlan,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
}

/// Extend ops.SoftmaxPlan with gradient type information
pub fn SoftmaxPlanWithGrad(comptime Backend: type, comptime T: type, comptime batch_size: ?usize, comptime feature_size: ?usize) type {
    const BasePlan = ops.SoftmaxPlan(Backend, T, batch_size, feature_size);
    
    return struct {
        const Self = @This();
        const InputType = Tensor;
        const op_type = .softmax;
        
        // Define the gradient type for softmax operation (single input)
        const GradType = Tensor;
        
        // Embed the base plan
        base: BasePlan,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
}

/// Extend ops.TransposePlan with gradient type information
pub fn TransposePlanWithGrad(comptime Backend: type, comptime T: type, comptime rows: ?usize, comptime cols: ?usize) type {
    const BasePlan = ops.TransposePlan(Backend, T, rows, cols);
    
    return struct {
        const Self = @This();
        const InputType = Tensor;
        const op_type = .transpose;
        
        // Define the gradient type for transpose operation (single input)
        const GradType = Tensor;
        
        // Embed the base plan
        base: BasePlan,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
}

/// Extend ops.DividePlan with gradient type information
pub fn DividePlanWithGrad(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    const BasePlan = ops.DividePlan(Backend, T, shape);
    
    return struct {
        const Self = @This();
        const InputType = BasePlan.InputType;
        const op_type = .divide;
        
        // Define the gradient type for divide operation
        const GradType = struct { da: Tensor, db: Tensor };
        
        // Embed the base plan
        base: BasePlan,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .base = BasePlan.init(allocator) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
}

/// Define a plan for embedding lookup operation
pub fn EmbeddingLookupPlanWithGrad(comptime Backend: type, comptime T: type, comptime vocab_size: ?usize, comptime embed_dim: ?usize) type {
    _ = Backend; // Used at compile time for consistency with other Plan functions
    _ = T; // Used at compile time for consistency with other Plan functions
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
        
        pub fn run(self: Self, input: InputType) !Tensor {
            // Validate input - params must be 2D tensor: [vocab_size, embed_dim]
            if (input.params.shape.rank() != 2) {
                return error.InvalidEmbeddingShape;
            }
            
            // Verify dimensions
            if (input.params.shape.dims[0] != vocab_size or input.params.shape.dims[1] != embed_dim) {
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
            var result_dims = [_]usize{ batch_size, seq_len, embed_dim };
            var result = try Tensor.zeros(self.allocator, &result_dims, input.params.dtype, input.params.backend);
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
            
            const params_buf = ptrCastHelper([*]f32, input.params.buffer.data.ptr)[0..input.params.shape.elemCount()];
            const indices_buf = ptrCastHelper([*]f32, input.indices.buffer.data.ptr)[0..input.indices.shape.elemCount()];
            const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
            
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
                        std.debug.print("Warning: Index out of bounds in embedding_lookup: {}/{}\n", 
                            .{index_pos, indices_buf.len});
                        continue;
                    }
                    
                    const token_id_f = indices_buf[index_pos];
                    // Safely convert to int and bound to vocab size
                    const token_id_i = @as(i32, @intFromFloat(token_id_f));
                    const token_id = @as(usize, @intCast(@max(0, @min(token_id_i, @as(i32, @intCast(vocab_size - 1))))));
                    
                    // Copy embedding for this token - embed_dim values
                    for (0..embed_dim) |d| {
                        const src_idx = token_id * embed_dim + d;
                        const dst_idx = (b * seq_len + s) * embed_dim + d;
                        
                        // Add bounds checking
                        if (src_idx >= params_buf.len) {
                            std.debug.print("Warning: Source index out of bounds in embedding_lookup: {}/{}\n", 
                                .{src_idx, params_buf.len});
                            continue;
                        }
                        
                        if (dst_idx >= result_buf.len) {
                            std.debug.print("Warning: Destination index out of bounds in embedding_lookup: {}/{}\n", 
                                .{dst_idx, result_buf.len});
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

// --- DEPRECATED: Legacy Interface ---
// This Node-based interface is DEPRECATED and will be removed in the future.
// New code should use the comptime-based Plan approach instead.
// 
// TODO: Remove this legacy interface once all examples are migrated to the new system

/// Operation type for the computational graph
pub const OpType = enum {
    add,
    subtract,
    multiply,
    divide,
    matmul,
    relu,
    softmax,
    transpose,
    embedding_lookup,
    // More operations will be added
};

/// A node in the computational graph
pub const Node = struct {
    // Output of this node
    tensor: Tensor,
    
    // Operation that created this node
    op_type: ?OpType = null,
    
    // Inputs to this operation
    inputs: std.ArrayList(*Node),
    
    // Gradient of the loss with respect to this node's output
    grad: ?Tensor = null,
    
    // Whether this node requires gradient computation
    requires_grad: bool,
    
    // Allocator for managing memory
    allocator: Allocator,
    
    // Track if tensor has already been released (for debugging)
    tensor_released: bool = false,
    
    pub fn init(allocator: Allocator, tensor_val: Tensor, requires_grad: bool) !*Node {
        // Create the node
        const node = try allocator.create(Node);
        node.* = Node{
            .tensor = tensor_val,
            .inputs = std.ArrayList(*Node).init(allocator),
            .requires_grad = requires_grad,
            .allocator = allocator,
            .tensor_released = false,
        };
        return node;
    }
    
    pub fn deinit(self: *Node) void {
        // In embedding_lookup, we store the indices node as a direct input,
        // so we need to properly clean up input nodes first
        if (self.op_type == .embedding_lookup and self.inputs.items.len == 2) {
            // The second input is the indices_node
            if (self.inputs.items.len > 1) {
                // Clean up the indices node
                const indices_node = self.inputs.items[1];
                indices_node.deinit();
            }
        }
        
        // Clean up inputs array without cleaning up the nodes themselves
        self.inputs.deinit();
        
        // Clean up gradient if it exists
        if (self.grad) |*g| {
            g.deinit();
        }
        
        // Check if tensor has already been released
        if (!self.tensor_released) {
            // Release the tensor (decrement ref count)
            self.tensor.deinit();
            self.tensor_released = true;
        } else {
            std.debug.print("Warning: Node tensor already released\n", .{});
        }
        
        // Finally, deallocate the Node itself
        self.allocator.destroy(self);
    }
    
    /// Create a gradient tensor with the same shape as this node's tensor
    pub fn initGrad(self: *Node) !void {
        if (self.grad == null and self.requires_grad) {
            self.grad = try Tensor.zeros(
                self.allocator, 
                self.tensor.shape.dims, 
                self.tensor.dtype, 
                self.tensor.backend
            );
        }
    }
    
    /// Initialize gradient with ones (for loss nodes)
    pub fn initGradOnes(self: *Node) !void {
        if (self.grad == null and self.requires_grad) {
            self.grad = try Tensor.filled(
                self.allocator, 
                self.tensor.shape.dims, 
                self.tensor.dtype, 
                1.0,
                self.tensor.backend
            );
        }
    }
};

/// Create a computational graph node from a tensor
/// DEPRECATED: Use the Plan-based approach instead
@deprecated("Use the comptime Plan-based approach instead")
pub fn variable(allocator: Allocator, t: Tensor, requires_grad: bool) !*Node {
    return Node.init(allocator, t, requires_grad);
}

/// Add two nodes element-wise
/// DEPRECATED: Use the Plan-based approach with AutoDiffPlan instead
@deprecated("Use the comptime Plan-based approach with AutoDiffPlan instead")
pub fn add(allocator: Allocator, a: *Node, b: *Node) !*Node {
    // Perform the operation using the legacy interface
    var result_tensor = try ops.add(allocator, a.tensor, b.tensor);
    result_tensor.requires_grad = a.requires_grad or b.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .add;
    
    // Save input nodes for backward pass
    try result.inputs.append(a);
    try result.inputs.append(b);
    
    return result;
}

/// Subtract two nodes element-wise
pub fn subtract(allocator: Allocator, a: *Node, b: *Node) !*Node {
    // Perform the operation using the legacy interface
    var result_tensor = try ops.subtract(allocator, a.tensor, b.tensor);
    result_tensor.requires_grad = a.requires_grad or b.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .subtract;
    
    // Save input nodes for backward pass
    try result.inputs.append(a);
    try result.inputs.append(b);
    
    return result;
}

/// Multiply two nodes element-wise
pub fn multiply(allocator: Allocator, a: *Node, b: *Node) !*Node {
    // Perform the operation using the legacy interface
    var result_tensor = try ops.multiply(allocator, a.tensor, b.tensor);
    result_tensor.requires_grad = a.requires_grad or b.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .multiply;
    
    // Save input nodes for backward pass
    try result.inputs.append(a);
    try result.inputs.append(b);
    
    return result;
}

/// Matrix multiplication of two nodes
pub fn matmul(allocator: Allocator, a: *Node, b: *Node) !*Node {
    // Perform the operation using the legacy interface
    var result_tensor = try ops.matmul(allocator, a.tensor, b.tensor);
    result_tensor.requires_grad = a.requires_grad or b.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .matmul;
    
    // Save input nodes for backward pass
    try result.inputs.append(a);
    try result.inputs.append(b);
    
    return result;
}

/// Apply ReLU activation to a node
pub fn relu(allocator: Allocator, a: *Node) !*Node {
    // Perform the operation using the legacy interface
    var result_tensor = try ops.relu(allocator, a.tensor);
    result_tensor.requires_grad = a.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .relu;
    
    // Save input node for backward pass
    try result.inputs.append(a);
    
    return result;
}

/// Apply softmax activation to a node
pub fn softmax(allocator: Allocator, a: *Node) !*Node {
    // Perform the operation using the legacy interface
    var result_tensor = try ops.softmax(allocator, a.tensor);
    result_tensor.requires_grad = a.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .softmax;
    
    // Save input node for backward pass
    try result.inputs.append(a);
    
    return result;
}

/// Transpose a node
pub fn transpose(allocator: Allocator, a: *Node) !*Node {
    // Perform the operation using the legacy interface
    var result_tensor = try ops.transpose(allocator, a.tensor);
    result_tensor.requires_grad = a.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .transpose;
    
    // Save input node for backward pass
    try result.inputs.append(a);
    
    return result;
}

/// Embedding lookup operation
pub fn embedding_lookup(allocator: Allocator, params: *Node, indices: Tensor) !*Node {
    // This function is kept from the original implementation
    // because it's complex and has a lot of special handling logic
    
    // Validate input - params must be 2D tensor: [vocab_size, embedding_dim]
    if (params.tensor.shape.rank() != 2) {
        return error.InvalidEmbeddingShape;
    }
    
    // Get vocab size and embedding dimensions
    const vocab_size = params.tensor.shape.dims[0];
    const embed_dim = params.tensor.shape.dims[1];
    
    // Parse indices - can be 1D or 2D
    var batch_size: usize = 1;
    var seq_len: usize = 1;
    
    if (indices.shape.rank() == 1) {
        seq_len = indices.shape.dims[0];
    } else if (indices.shape.rank() == 2) {
        batch_size = indices.shape.dims[0];
        seq_len = indices.shape.dims[1];
    } else {
        return error.InvalidIndicesShape;
    }
    
    // Create output tensor with shape [batch_size, seq_len, embed_dim]
    var result_dims = [_]usize{ batch_size, seq_len, embed_dim };
    var result_tensor = try Tensor.zeros(allocator, &result_dims, params.tensor.dtype, params.tensor.backend);
    
    // Get data buffers with safe bounds checking
    if (params.tensor.shape.elemCount() == 0) {
        // Clean up the result tensor on error
        result_tensor.deinit();
        return error.EmptyTensor;
    }
    
    if (indices.shape.elemCount() == 0) {
        // Clean up the result tensor on error
        result_tensor.deinit();
        return error.EmptyTensor;
    }
    
    if (result_tensor.shape.elemCount() == 0) {
        // Clean up the result tensor on error
        result_tensor.deinit();
        return error.EmptyTensor;
    }
    
    const params_buf = ptrCastHelper([*]f32, params.tensor.buffer.data.ptr)[0..params.tensor.shape.elemCount()];
    const indices_buf = ptrCastHelper([*]f32, indices.buffer.data.ptr)[0..indices.shape.elemCount()];
    const result_buf = ptrCastHelper([*]f32, result_tensor.buffer.data.ptr)[0..result_tensor.shape.elemCount()];
    
    // Perform lookup
    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            // Get the token id from indices, safely convert to int
            var index_pos: usize = 0;
            if (indices.shape.rank() == 1) {
                index_pos = s;
            } else {
                index_pos = b * seq_len + s;
            }
            
            if (index_pos >= indices_buf.len) {
                std.debug.print("Warning: Index out of bounds in embedding_lookup: {}/{}\n", 
                    .{index_pos, indices_buf.len});
                continue;
            }
            
            const token_id_f = indices_buf[index_pos];
            // Safely convert to int and bound to vocab size
            const token_id_i = @as(i32, @intFromFloat(token_id_f));
            const token_id = @as(usize, @intCast(@max(0, @min(token_id_i, @as(i32, @intCast(vocab_size - 1))))));
            
            // Copy embedding for this token - embed_dim values
            for (0..embed_dim) |d| {
                const src_idx = token_id * embed_dim + d;
                const dst_idx = (b * seq_len + s) * embed_dim + d;
                
                // Add bounds checking
                if (src_idx >= params_buf.len) {
                    std.debug.print("Warning: Source index out of bounds in embedding_lookup: {}/{}\n", 
                        .{src_idx, params_buf.len});
                    continue;
                }
                
                if (dst_idx >= result_buf.len) {
                    std.debug.print("Warning: Destination index out of bounds in embedding_lookup: {}/{}\n", 
                        .{dst_idx, result_buf.len});
                    continue;
                }
                
                result_buf[dst_idx] = params_buf[src_idx];
            }
        }
    }
    
    // Set requires_grad based on the parameters
    result_tensor.requires_grad = params.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .embedding_lookup;
    
    // Save input nodes for backward pass
    try result.inputs.append(params);
    
    // Create a copy of the indices tensor that we'll own
    // First, increase the reference count of the original indices tensor
    const indices_for_copy = indices.retain();
    
    // Store indices in a new node (without requiring gradients)
    // Pass ownership of the retained tensor to this node
    const indices_node = try Node.init(allocator, indices_for_copy, false);
    try result.inputs.append(indices_node);
    
    return result;
}

/// Compute gradients through the computational graph
/// DEPRECATED: Use the Plan-based approach with AutoDiffPlan instead
@deprecated("Use the comptime Plan-based approach with AutoDiffPlan instead")
pub fn backward(allocator: Allocator, node: *Node) !void {
    // Initialize gradient of the output node to ones
    try node.initGradOnes();
    
    // Build a topological sort of the graph
    var visited = std.AutoHashMap(*Node, void).init(allocator);
    defer visited.deinit();
    
    var topo_order = std.ArrayList(*Node).init(allocator);
    defer topo_order.deinit();
    
    // DFS to build topological sort
    try buildTopoSort(node, &visited, &topo_order);
    
    // Print the number of nodes in the computation graph for debugging
    std.debug.print("Backward pass: found {} nodes in computation graph\n", .{topo_order.items.len});
    
    // Backward pass in reverse topological order
    var i: usize = topo_order.items.len;
    while (i > 0) {
        i -= 1;
        const current = topo_order.items[i];
        
        // Skip nodes that don't require gradients
        if (!current.requires_grad) continue;
        
        // Debug print the current node
        std.debug.print("Processing node {} with op_type {any}\n", 
            .{i, current.op_type});
        
        // Process according to operation type
        if (current.op_type) |op| {
            switch (op) {
                .add => try backwardAdd(allocator, current),
                .subtract => try backwardSubtract(allocator, current),
                .multiply => try backwardMultiply(allocator, current),
                .matmul => try backwardMatmul(allocator, current),
                .relu => try backwardRelu(allocator, current),
                .softmax => try backwardSoftmax(allocator, current),
                .transpose => try backwardTranspose(allocator, current),
                .embedding_lookup => try backwardEmbeddingLookup(allocator, current),
                // Add more operation types as they are implemented
                else => return error.UnsupportedOperationBackward,
            }
        } else {
            // If this is a variable node with no operation, we might need to handle 
            // special cases like parameter nodes
            std.debug.print("Skipping variable node with no operation\n", .{});
        }
    }
}

/// Helper function to build a topological sort
fn buildTopoSort(
    node: *Node, 
    visited: *std.AutoHashMap(*Node, void), 
    topo_order: *std.ArrayList(*Node)
) !void {
    // Check if node was already visited
    if (visited.contains(node)) return;
    
    // Mark as visited
    try visited.put(node, {});
    
    // Visit all inputs first
    if (node.inputs.items.len > 0) {
        for (node.inputs.items) |input| {
            try buildTopoSort(input, visited, topo_order);
        }
    }
    
    // Add current node to the order
    try topo_order.append(node);
}

/// Backward pass for addition
fn backwardAdd(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    const b = node.inputs.items[1];
    
    if (node.grad) |grad| {
        // dL/da = dL/dc * dc/da = dL/dc * 1
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // If we already have gradients, accumulate
                const temp = try ops.add(allocator, a_grad.*, grad);
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
        
        // dL/db = dL/dc * dc/db = dL/dc * 1
        if (b.requires_grad) {
            try b.initGrad();
            if (b.grad) |*b_grad| {
                // If we already have gradients, accumulate
                const temp = try ops.add(allocator, b_grad.*, grad);
                b_grad.deinit();
                b_grad.* = temp;
            }
        }
    }
}

/// Backward pass for subtraction
fn backwardSubtract(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    const b = node.inputs.items[1];
    
    if (node.grad) |grad| {
        // dL/da = dL/dc * dc/da = dL/dc * 1
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // If we already have gradients, accumulate
                const temp = try ops.add(allocator, a_grad.*, grad);
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
        
        // dL/db = dL/dc * dc/db = dL/dc * (-1)
        if (b.requires_grad) {
            try b.initGrad();
            if (b.grad) |*b_grad| {
                // For subtraction, the gradient for b is negated
                // Create the constant tensor for -1.0
                const negative_one = try Tensor.filled(
                    allocator,
                    grad.shape.dims,
                    grad.dtype,
                    -1.0,
                    grad.backend
                );
                
                // Negate the gradient
                const neg_grad = try ops.multiply(allocator, grad, negative_one);
                // Clean up the temporary tensor
                negative_one.deinit();
                
                // Accumulate gradients
                const temp = try ops.add(allocator, b_grad.*, neg_grad);
                // Clean up the negated gradient tensor
                neg_grad.deinit();
                
                b_grad.deinit();
                b_grad.* = temp;
            }
        }
    }
}

/// Backward pass for element-wise multiplication
fn backwardMultiply(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    const b = node.inputs.items[1];
    
    if (node.grad) |grad| {
        // dL/da = dL/dc * dc/da = dL/dc * b
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                const temp_grad = try ops.multiply(allocator, grad, b.tensor);
                const temp = try ops.add(allocator, a_grad.*, temp_grad);
                // Clean up the temporary gradient
                temp_grad.deinit();
                
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
        
        // dL/db = dL/dc * dc/db = dL/dc * a
        if (b.requires_grad) {
            try b.initGrad();
            if (b.grad) |*b_grad| {
                const temp_grad = try ops.multiply(allocator, grad, a.tensor);
                const temp = try ops.add(allocator, b_grad.*, temp_grad);
                // Clean up the temporary gradient
                temp_grad.deinit();
                
                b_grad.deinit();
                b_grad.* = temp;
            }
        }
    }
}

/// Backward pass for matrix multiplication
fn backwardMatmul(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    const b = node.inputs.items[1];
    
    if (node.grad) |grad| {
        // dL/da = dL/dc * dc/da = dL/dc * b^T
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // Compute b transpose
                const b_transpose = try ops.transpose(allocator, b.tensor);
                
                // Compute gradient contribution
                const temp_grad = try ops.matmul(allocator, grad, b_transpose);
                
                // Release the transpose tensor
                b_transpose.deinit();
                
                // Accumulate gradient
                const temp = try ops.add(allocator, a_grad.*, temp_grad);
                
                // Release the temporary gradient
                temp_grad.deinit();
                
                // Replace the old gradient with the new one
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
        
        // dL/db = dL/dc * dc/db = a^T * dL/dc
        if (b.requires_grad) {
            try b.initGrad();
            if (b.grad) |*b_grad| {
                // Compute a transpose
                const a_transpose = try ops.transpose(allocator, a.tensor);
                
                // Compute gradient contribution
                const temp_grad = try ops.matmul(allocator, a_transpose, grad);
                
                // Release the transpose tensor
                a_transpose.deinit();
                
                // Accumulate gradient
                const temp = try ops.add(allocator, b_grad.*, temp_grad);
                
                // Release the temporary gradient
                temp_grad.deinit();
                
                // Replace the old gradient with the new one
                b_grad.deinit();
                b_grad.* = temp;
            }
        }
    }
}

/// Backward pass for ReLU
fn backwardRelu(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 1) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    
    if (node.grad) |grad| {
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // Create ReLU derivative mask: 1 where input > 0, 0 elsewhere
                const mask = try Tensor.zeros(allocator, a.tensor.shape.dims, a.tensor.dtype, a.tensor.backend);
                
                const input_buf = ptrCastHelper([*]f32, a.tensor.buffer.data.ptr)[0..a.tensor.shape.elemCount()];
                const mask_buf = ptrCastHelper([*]f32, mask.buffer.data.ptr)[0..mask.shape.elemCount()];
                
                for (input_buf, 0..) |val, i| {
                    mask_buf[i] = if (val > 0) 1.0 else 0.0;
                }
                
                // Gradient is only passed where input was positive
                const temp_grad = try ops.multiply(allocator, grad, mask);
                // Clean up the mask tensor
                mask.deinit();
                
                const temp = try ops.add(allocator, a_grad.*, temp_grad);
                // Clean up the temporary gradient
                temp_grad.deinit();
                
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
    }
}

/// Backward pass for softmax
fn backwardSoftmax(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 1) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    
    if (node.grad) |grad| {
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // Use the modern GradRule implementation
                const da = try GradRules.softmax(allocator, grad, node.tensor, a.tensor);
                
                // Accumulate the result
                const temp = try ops.add(allocator, a_grad.*, da);
                // Clean up the temporary gradient
                da.deinit();
                
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
    }
}

/// Backward pass for transpose
fn backwardTranspose(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 1) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    
    if (node.grad) |grad| {
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // For transpose, we need to transpose the gradient
                const transposed_grad = try ops.transpose(allocator, grad);
                
                // Accumulate gradients
                const temp = try ops.add(allocator, a_grad.*, transposed_grad);
                // Clean up temporary
                transposed_grad.deinit();
                
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
    }
}

/// Backward pass for embedding lookup
fn backwardEmbeddingLookup(allocator: Allocator, node: *Node) !void {
    _ = allocator; // Allocator is used indirectly in lower-level functions
    // This function is kept from the original implementation 
    // because embedding_lookup has complex gradient logic
    
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const params = node.inputs.items[0]; // Embedding parameters
    const indices_node = node.inputs.items[1]; // Indices tensor node
    
    if (node.grad) |grad| {
        if (params.requires_grad) {
            try params.initGrad();
            if (params.grad) |*params_grad| {
                // Extract dimensions
                const vocab_size = params.tensor.shape.dims[0];
                const embed_dim = params.tensor.shape.dims[1];
                
                // Get the indices tensor
                const indices = indices_node.tensor;
                
                // Parse indices dimensions
                var batch_size: usize = 1;
                var seq_len: usize = 1;
                
                if (indices.shape.rank() == 1) {
                    seq_len = indices.shape.dims[0];
                } else if (indices.shape.rank() == 2) {
                    batch_size = indices.shape.dims[0];
                    seq_len = indices.shape.dims[1];
                }
                
                // Safety checks for empty tensors
                if (indices.shape.elemCount() == 0) {
                    std.debug.print("Warning: Empty indices tensor in backwardEmbeddingLookup\n", .{});
                    return;
                }
                
                if (grad.shape.elemCount() == 0) {
                    std.debug.print("Warning: Empty gradient tensor in backwardEmbeddingLookup\n", .{});
                    return;
                }
                
                if (params_grad.*.shape.elemCount() == 0) {
                    std.debug.print("Warning: Empty params_grad tensor in backwardEmbeddingLookup\n", .{});
                    return;
                }
                
                // Get data buffers with bounds checking
                const indices_buf = ptrCastHelper([*]f32, indices.buffer.data.ptr)[0..indices.shape.elemCount()];
                const grad_buf = ptrCastHelper([*]f32, grad.buffer.data.ptr)[0..grad.shape.elemCount()];
                const params_grad_buf = ptrCastHelper([*]f32, params_grad.*.buffer.data.ptr)[0..params_grad.*.shape.elemCount()];
                
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
                            std.debug.print("Warning: Index out of bounds in backwardEmbeddingLookup: {}/{}\n", 
                                .{index_pos, indices_buf.len});
                            continue;
                        }
                        
                        // Get token ID, clamping to vocab size
                        const token_id_f = indices_buf[index_pos];
                        const token_id_i = @as(i32, @intFromFloat(token_id_f));
                        const token_id = @as(usize, @intCast(@max(0, @min(token_id_i, @as(i32, @intCast(vocab_size - 1))))));
                        
                        // Accumulate gradients for this embedding
                        for (0..embed_dim) |d| {
                            const grad_pos = (b * seq_len + s) * embed_dim + d;
                            const param_pos = token_id * embed_dim + d;
                            
                            // Bounds checking
                            if (grad_pos >= grad_buf.len) {
                                std.debug.print("Warning: Grad pos out of bounds in backwardEmbeddingLookup: {}/{}\n", 
                                    .{grad_pos, grad_buf.len});
                                continue;
                            }
                            
                            if (param_pos >= params_grad_buf.len) {
                                std.debug.print("Warning: Param pos out of bounds in backwardEmbeddingLookup: {}/{}\n", 
                                    .{param_pos, params_grad_buf.len});
                                continue;
                            }
                            
                            params_grad_buf[param_pos] += grad_buf[grad_pos];
                        }
                    }
                }
            }
        }
    }
}

// --- Tests for the New Plan-Based AutoDiff ---

test "autodiff add plan" {
    const allocator = std.testing.allocator;
    
    // Create test tensors with comptime-known dimensions
    const dims = [_]usize{ 2, 2 };
    
    var a = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{0, 0}, 1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{1, 0}, 3.0);
    try a.setScalar(&[_]usize{1, 1}, 4.0);
    
    var b = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{0, 0}, 5.0);
    try b.setScalar(&[_]usize{0, 1}, 6.0);
    try b.setScalar(&[_]usize{1, 0}, 7.0);
    try b.setScalar(&[_]usize{1, 1}, 8.0);
    
    // Create an AddPlan with gradient support using AddPlanWithGrad
    const PlanType = AddPlanWithGrad(ops.CpuBackend, f32, &dims);
    const AutoDiffType = AutoDiffPlan(PlanType);
    
    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();
    
    // Forward pass
    const output = try auto_diff.forward(.{ .a = a, .b = b });
    defer output.deinit();
    
    // Check output values
    try std.testing.expectEqual(@as(f32, 6.0), try output.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 8.0), try output.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 10.0), try output.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 12.0), try output.getScalar(&[_]usize{1, 1}));
    
    // Create gradient with ones
    var grad_out = try Tensor.filled(allocator, &dims, .f32, 1.0, .cpu);
    defer grad_out.deinit();
    
    // Backward pass
    const grads = try auto_diff.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();
    
    // Check gradients - for add, both inputs get the same gradient (1.0)
    try std.testing.expectEqual(@as(f32, 1.0), try grads.da.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.da.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.da.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.da.getScalar(&[_]usize{1, 1}));
    
    try std.testing.expectEqual(@as(f32, 1.0), try grads.db.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.db.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.db.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try grads.db.getScalar(&[_]usize{1, 1}));
}

test "autodiff matmul plan" {
    const allocator = std.testing.allocator;
    
    // Use comptime dimensions
    const M: usize = 2;
    const N: usize = 3;
    const P: usize = 2;
    
    // Create test tensors
    const a_dims = [_]usize{ M, N };
    const b_dims = [_]usize{ N, P };
    
    var a = try Tensor.zeros(allocator, &a_dims, .f32, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{0, 0}, 1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{0, 2}, 3.0);
    try a.setScalar(&[_]usize{1, 0}, 4.0);
    try a.setScalar(&[_]usize{1, 1}, 5.0);
    try a.setScalar(&[_]usize{1, 2}, 6.0);
    
    var b = try Tensor.zeros(allocator, &b_dims, .f32, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{0, 0}, 7.0);
    try b.setScalar(&[_]usize{0, 1}, 8.0);
    try b.setScalar(&[_]usize{1, 0}, 9.0);
    try b.setScalar(&[_]usize{1, 1}, 10.0);
    try b.setScalar(&[_]usize{2, 0}, 11.0);
    try b.setScalar(&[_]usize{2, 1}, 12.0);
    
    // Create a MatmulPlan with gradient support
    const PlanType = MatmulPlanWithGrad(ops.CpuBackend, f32, M, N, P);
    const AutoDiffType = AutoDiffPlan(PlanType);
    
    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();
    
    // Forward pass
    const output = try auto_diff.forward(.{ .a = a, .b = b });
    defer output.deinit();
    
    // Check output values: [[58, 64], [139, 154]]
    try std.testing.expectEqual(@as(f32, 58.0), try output.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 64.0), try output.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 139.0), try output.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 154.0), try output.getScalar(&[_]usize{1, 1}));
    
    // Create gradient with ones
    const grad_dims = [_]usize{ M, P };
    var grad_out = try Tensor.filled(allocator, &grad_dims, .f32, 1.0, .cpu);
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
    try std.testing.expectEqual(@as(f32, 15.0), try grads.da.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 19.0), try grads.da.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 23.0), try grads.da.getScalar(&[_]usize{0, 2}));
    try std.testing.expectEqual(@as(f32, 15.0), try grads.da.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 19.0), try grads.da.getScalar(&[_]usize{1, 1}));
    try std.testing.expectEqual(@as(f32, 23.0), try grads.da.getScalar(&[_]usize{1, 2}));
    
    // dL/dB = [1, 2, 3; 4, 5, 6]^T @ [1, 1; 1, 1] = [5, 5; 7, 7; 9, 9]
    try std.testing.expectEqual(@as(f32, 5.0), try grads.db.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), try grads.db.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 7.0), try grads.db.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 7.0), try grads.db.getScalar(&[_]usize{1, 1}));
    try std.testing.expectEqual(@as(f32, 9.0), try grads.db.getScalar(&[_]usize{2, 0}));
    try std.testing.expectEqual(@as(f32, 9.0), try grads.db.getScalar(&[_]usize{2, 1}));
}

test "autodiff relu plan" {
    const allocator = std.testing.allocator;
    
    // Create test tensor with mixed positive and negative values
    const dims = [_]usize{ 2, 2 };
    var a = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a.deinit();
    
    try a.setScalar(&[_]usize{0, 0}, -1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{1, 0}, 0.0);
    try a.setScalar(&[_]usize{1, 1}, 3.0);
    
    // Create a ReluPlan with gradient support
    const PlanType = ReluPlanWithGrad(ops.CpuBackend, f32, &dims);
    const AutoDiffType = AutoDiffPlan(PlanType);
    
    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();
    
    // Forward pass
    const output = try auto_diff.forward(a);
    defer output.deinit();
    
    // Check output values: [[0, 2], [0, 3]]
    try std.testing.expectEqual(@as(f32, 0.0), try output.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 2.0), try output.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 0.0), try output.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 3.0), try output.getScalar(&[_]usize{1, 1}));
    
    // Create gradient with ones
    var grad_out = try Tensor.filled(allocator, &dims, .f32, 1.0, .cpu);
    defer grad_out.deinit();
    
    // Backward pass
    const grad = try auto_diff.backward(grad_out);
    defer grad.deinit();
    
    // Check gradients - for ReLU, gradient is 1 where input > 0, 0 elsewhere
    try std.testing.expectEqual(@as(f32, 0.0), try grad.getScalar(&[_]usize{0, 0})); // Input was -1.0
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{0, 1})); // Input was 2.0
    try std.testing.expectEqual(@as(f32, 0.0), try grad.getScalar(&[_]usize{1, 0})); // Input was 0.0
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{1, 1})); // Input was 3.0
}

test "autodiff divide plan" {
    const allocator = std.testing.allocator;
    
    // Create test tensors with comptime dimensions
    const dims = [_]usize{ 2, 2 };
    
    var a = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{0, 0}, 10.0);
    try a.setScalar(&[_]usize{0, 1}, 20.0);
    try a.setScalar(&[_]usize{1, 0}, 30.0);
    try a.setScalar(&[_]usize{1, 1}, 40.0);
    
    var b = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{0, 0}, 2.0);
    try b.setScalar(&[_]usize{0, 1}, 4.0);
    try b.setScalar(&[_]usize{1, 0}, 5.0);
    try b.setScalar(&[_]usize{1, 1}, 8.0);
    
    // Create a DividePlan with gradient support
    const PlanType = DividePlanWithGrad(ops.CpuBackend, f32, &dims);
    const AutoDiffType = AutoDiffPlan(PlanType);
    
    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();
    
    // Forward pass
    const output = try auto_diff.forward(.{ .a = a, .b = b });
    defer output.deinit();
    
    // Check output values: a / b = [[5.0, 5.0], [6.0, 5.0]]
    try std.testing.expectEqual(@as(f32, 5.0), try output.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), try output.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 6.0), try output.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), try output.getScalar(&[_]usize{1, 1}));
    
    // Create gradient with ones
    var grad_out = try Tensor.filled(allocator, &dims, .f32, 1.0, .cpu);
    defer grad_out.deinit();
    
    // Backward pass
    const grads = try auto_diff.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();
    
    // Check gradients:
    // da = 1.0 / b = [[0.5, 0.25], [0.2, 0.125]]
    // db = -a / (b * b) = [[-2.5, -1.25], [-1.2, -0.625]]
    
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), try grads.da.getScalar(&[_]usize{0, 0}), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), try grads.da.getScalar(&[_]usize{0, 1}), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), try grads.da.getScalar(&[_]usize{1, 0}), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), try grads.da.getScalar(&[_]usize{1, 1}), 0.0001);
    
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), try grads.db.getScalar(&[_]usize{0, 0}), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.25), try grads.db.getScalar(&[_]usize{0, 1}), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.2), try grads.db.getScalar(&[_]usize{1, 0}), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.625), try grads.db.getScalar(&[_]usize{1, 1}), 0.0001);
}

test "autodiff embedding lookup plan" {
    const allocator = std.testing.allocator;
    
    // Comptime dimensions
    const vocab_size: usize = 4;
    const embed_dim: usize = 3;
    const batch_size: usize = 2;
    const seq_len: usize = 2;
    
    // Create a simple embedding table [vocab_size=4, embed_dim=3]
    const embed_dims = [_]usize{ vocab_size, embed_dim };
    var params = try Tensor.zeros(allocator, &embed_dims, .f32, .cpu);
    defer params.deinit();
    
    // Fill with test values
    // Word 0: [0.1, 0.2, 0.3]
    try params.setScalar(&[_]usize{0, 0}, 0.1);
    try params.setScalar(&[_]usize{0, 1}, 0.2);
    try params.setScalar(&[_]usize{0, 2}, 0.3);
    // Word 1: [0.4, 0.5, 0.6]
    try params.setScalar(&[_]usize{1, 0}, 0.4);
    try params.setScalar(&[_]usize{1, 1}, 0.5);
    try params.setScalar(&[_]usize{1, 2}, 0.6);
    // Word 2: [0.7, 0.8, 0.9]
    try params.setScalar(&[_]usize{2, 0}, 0.7);
    try params.setScalar(&[_]usize{2, 1}, 0.8);
    try params.setScalar(&[_]usize{2, 2}, 0.9);
    // Word 3: [1.0, 1.1, 1.2]
    try params.setScalar(&[_]usize{3, 0}, 1.0);
    try params.setScalar(&[_]usize{3, 1}, 1.1);
    try params.setScalar(&[_]usize{3, 2}, 1.2);
    
    // Create indices tensor with shape [2, 2]
    const indices_dims = [_]usize{ batch_size, seq_len };
    var indices = try Tensor.zeros(allocator, &indices_dims, .f32, .cpu);
    defer indices.deinit();
    
    // Set indices: [[1, 2], [3, 0]]
    try indices.setScalar(&[_]usize{0, 0}, 1.0);  // batch 0, pos 0: word ID 1
    try indices.setScalar(&[_]usize{0, 1}, 2.0);  // batch 0, pos 1: word ID 2
    try indices.setScalar(&[_]usize{1, 0}, 3.0);  // batch 1, pos 0: word ID 3
    try indices.setScalar(&[_]usize{1, 1}, 0.0);  // batch 1, pos 1: word ID 0
    
    // Create an EmbeddingLookupPlan with gradient support
    const PlanType = EmbeddingLookupPlanWithGrad(ops.CpuBackend, f32, vocab_size, embed_dim);
    const AutoDiffType = AutoDiffPlan(PlanType);
    
    var auto_diff = AutoDiffType.init(allocator);
    defer auto_diff.deinit();
    
    // Forward pass
    const output = try auto_diff.forward(.{ .params = params, .indices = indices });
    defer output.deinit();
    
    // Check output values: shape [2, 2, 3]
    // Expected: [[[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], [[1.0, 1.1, 1.2], [0.1, 0.2, 0.3]]]
    
    // First sequence, first token (word ID 1)
    try std.testing.expectEqual(@as(f32, 0.4), try output.getScalar(&[_]usize{0, 0, 0}));
    try std.testing.expectEqual(@as(f32, 0.5), try output.getScalar(&[_]usize{0, 0, 1}));
    try std.testing.expectEqual(@as(f32, 0.6), try output.getScalar(&[_]usize{0, 0, 2}));
    
    // First sequence, second token (word ID 2)
    try std.testing.expectEqual(@as(f32, 0.7), try output.getScalar(&[_]usize{0, 1, 0}));
    try std.testing.expectEqual(@as(f32, 0.8), try output.getScalar(&[_]usize{0, 1, 1}));
    try std.testing.expectEqual(@as(f32, 0.9), try output.getScalar(&[_]usize{0, 1, 2}));
    
    // Second sequence, first token (word ID 3)
    try std.testing.expectEqual(@as(f32, 1.0), try output.getScalar(&[_]usize{1, 0, 0}));
    try std.testing.expectEqual(@as(f32, 1.1), try output.getScalar(&[_]usize{1, 0, 1}));
    try std.testing.expectEqual(@as(f32, 1.2), try output.getScalar(&[_]usize{1, 0, 2}));
    
    // Second sequence, second token (word ID 0)
    try std.testing.expectEqual(@as(f32, 0.1), try output.getScalar(&[_]usize{1, 1, 0}));
    try std.testing.expectEqual(@as(f32, 0.2), try output.getScalar(&[_]usize{1, 1, 1}));
    try std.testing.expectEqual(@as(f32, 0.3), try output.getScalar(&[_]usize{1, 1, 2}));
    
    // Create gradient with all ones
    const grad_dims = [_]usize{ batch_size, seq_len, embed_dim };
    var grad_out = try Tensor.filled(allocator, &grad_dims, .f32, 1.0, .cpu);
    defer grad_out.deinit();
    
    // Backward pass
    const grad = try auto_diff.backward(grad_out);
    defer grad.deinit();
    
    // For embedding lookup, the gradient is a sparse tensor with shape matching params
    // Each used word ID accumulates a gradient equal to the upstream gradient
    
    // Check gradient for word ID 0 (used in batch 1, pos 1) - should be [1, 1, 1]
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{0, 2}));
    
    // Check gradient for word ID 1 (used in batch 0, pos 0) - should be [1, 1, 1]
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{1, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{1, 2}));
    
    // Check gradient for word ID 2 (used in batch 0, pos 1) - should be [1, 1, 1]
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{2, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{2, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{2, 2}));
    
    // Check gradient for word ID 3 (used in batch 1, pos 0) - should be [1, 1, 1]
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{3, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{3, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try grad.getScalar(&[_]usize{3, 2}));
}

// Tests from the original implementation are still supported with the legacy interface

test "variable creation" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create a tensor that we'll copy for the variable
    var t = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    
    try t.setScalar(&[_]usize{0, 0}, 1.0);
    try t.setScalar(&[_]usize{0, 1}, 2.0);
    try t.setScalar(&[_]usize{1, 0}, 3.0);
    try t.setScalar(&[_]usize{1, 1}, 4.0);
    
    // Create a copy for the variable since variable() takes ownership
    const t_copy = try Tensor.init(allocator, t.shape.dims, t.dtype, t.backend);
    
    // Copy the data
    @memcpy(t_copy.buffer.data, t.buffer.data[0..t.buffer.data.len]);
    
    // Now we can create the variable with our copy and keep the original
    var node = try variable(allocator, t_copy, true);
    defer node.deinit();
    
    // Now we can safely use t
    defer t.deinit();
    
    try std.testing.expect(node.requires_grad);
    try std.testing.expectEqual(@as(usize, 2), node.tensor.shape.rank());
    try std.testing.expectEqual(@as(f32, 1.0), try node.tensor.getScalar(&[_]usize{0, 0}));
}

test "simple addition and backward" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create tensors
    var a_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a_tensor.setScalar(&[_]usize{0, 0}, 1.0);
    try a_tensor.setScalar(&[_]usize{0, 1}, 2.0);
    try a_tensor.setScalar(&[_]usize{1, 0}, 3.0);
    try a_tensor.setScalar(&[_]usize{1, 1}, 4.0);
    
    var b_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try b_tensor.setScalar(&[_]usize{0, 0}, 5.0);
    try b_tensor.setScalar(&[_]usize{0, 1}, 6.0);
    try b_tensor.setScalar(&[_]usize{1, 0}, 7.0);
    try b_tensor.setScalar(&[_]usize{1, 1}, 8.0);
    
    // Create copies of tensors for the variables
    const a_tensor_copy = try Tensor.init(allocator, a_tensor.shape.dims, a_tensor.dtype, a_tensor.backend);
    @memcpy(a_tensor_copy.buffer.data, a_tensor.buffer.data[0..a_tensor.buffer.data.len]);
    
    const b_tensor_copy = try Tensor.init(allocator, b_tensor.shape.dims, b_tensor.dtype, b_tensor.backend);
    @memcpy(b_tensor_copy.buffer.data, b_tensor.buffer.data[0..b_tensor.buffer.data.len]);
    
    // Create nodes with the copies
    var a = try variable(allocator, a_tensor_copy, true);
    var b = try variable(allocator, b_tensor_copy, true);
    
    // Now we can safely use the original tensors
    defer a_tensor.deinit();
    defer b_tensor.deinit();
    
    // Forward pass: c = a + b
    var c = try add(allocator, a, b);
    
    // Check result: c = [[6, 8], [10, 12]]
    try std.testing.expectEqual(@as(f32, 6.0), try c.tensor.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 8.0), try c.tensor.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 10.0), try c.tensor.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 12.0), try c.tensor.getScalar(&[_]usize{1, 1}));
    
    // Backward pass
    try backward(allocator, c);
    
    // For addition, gradients are 1s
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{1, 1}));
    
    try std.testing.expectEqual(@as(f32, 1.0), try b.grad.?.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try b.grad.?.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try b.grad.?.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try b.grad.?.getScalar(&[_]usize{1, 1}));
    
    // Cleanup
    a.deinit();
    b.deinit();
    c.deinit();
}

test "multiplication and backward" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create tensors
    var a_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a_tensor.setScalar(&[_]usize{0, 0}, 2.0);
    try a_tensor.setScalar(&[_]usize{0, 1}, 3.0);
    try a_tensor.setScalar(&[_]usize{1, 0}, 4.0);
    try a_tensor.setScalar(&[_]usize{1, 1}, 5.0);
    
    var b_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try b_tensor.setScalar(&[_]usize{0, 0}, 0.5);
    try b_tensor.setScalar(&[_]usize{0, 1}, 1.0);
    try b_tensor.setScalar(&[_]usize{1, 0}, 1.5);
    try b_tensor.setScalar(&[_]usize{1, 1}, 2.0);
    
    // Create copies of tensors for the variables
    const a_tensor_copy = try Tensor.init(allocator, a_tensor.shape.dims, a_tensor.dtype, a_tensor.backend);
    @memcpy(a_tensor_copy.buffer.data, a_tensor.buffer.data[0..a_tensor.buffer.data.len]);
    
    const b_tensor_copy = try Tensor.init(allocator, b_tensor.shape.dims, b_tensor.dtype, b_tensor.backend);
    @memcpy(b_tensor_copy.buffer.data, b_tensor.buffer.data[0..b_tensor.buffer.data.len]);
    
    // Create nodes with the copies
    var a = try variable(allocator, a_tensor_copy, true);
    var b = try variable(allocator, b_tensor_copy, true);
    
    // Now we can safely use the original tensors
    defer a_tensor.deinit();
    defer b_tensor.deinit();
    
    // Forward pass: c = a * b (element-wise)
    var c = try multiply(allocator, a, b);
    
    // Check result: c = [[1, 3], [6, 10]]
    try std.testing.expectEqual(@as(f32, 1.0), try c.tensor.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 3.0), try c.tensor.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 6.0), try c.tensor.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 10.0), try c.tensor.getScalar(&[_]usize{1, 1}));
    
    // Backward pass
    try backward(allocator, c);
    
    // For element-wise multiplication, dL/da = dL/dc * b
    try std.testing.expectEqual(@as(f32, 0.5), try a.grad.?.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.5), try a.grad.?.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 2.0), try a.grad.?.getScalar(&[_]usize{1, 1}));
    
    // And dL/db = dL/dc * a
    try std.testing.expectEqual(@as(f32, 2.0), try b.grad.?.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 3.0), try b.grad.?.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 4.0), try b.grad.?.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), try b.grad.?.getScalar(&[_]usize{1, 1}));
    
    // Cleanup nodes (which will also clean up their tensors)
    a.deinit();
    b.deinit();
    c.deinit();
}