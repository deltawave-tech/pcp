const std = @import("std");
const tensor = @import("tensor.zig");

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

/// Error types for tensor operations
pub const OpError = error{
    ShapeMismatch,
    UnsupportedDataType,
    UnsupportedBackend,
    DimensionMismatch,
    NotImplemented,
    InvalidEmbeddingShape,
    InvalidIndicesShape,
    EmptyTensor,
    InvalidAxes,
};

/// Operation type enum used for both forward and backward passes
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
    sum_reduce,
    pow,
    expand,
    reshape,
    layer_norm,
    cross_entropy,
    batched_matmul,
};

/// --- Primitive Operations ---
/// These are the building blocks for comptime plans, implemented by backends
pub const Primitives = struct {
    /// Add two tensors element-wise
    pub fn add(comptime T: type, a: Tensor, b: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const b_buf = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..b.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        for (a_buf, b_buf, 0..) |a_val, b_val, i| {
            result_buf[i] = a_val + b_val;
        }
    }
    
    /// Batched matrix multiplication
    /// Input shapes: a [batch, M, K], b [batch, K, N]
    /// Output shape: [batch, M, N]
    pub fn batched_matmul(comptime T: type, a: Tensor, b: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr);
        const b_buf = ptrCastHelper([*]f32, b.buffer.data.ptr);
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr);
        
        // Extract dimensions
        const batch_size = a.shape.dims[0];
        const m = a.shape.dims[1];
        const k = a.shape.dims[2]; // also b.shape.dims[1]
        const n = b.shape.dims[2];
        
        for (0..batch_size) |batch_idx| {
            for (0..m) |i| {
                for (0..n) |j| {
                    var sum: f32 = 0.0;
                    
                    for (0..k) |l| {
                        const a_idx = batch_idx * (m * k) + i * k + l;
                        const b_idx = batch_idx * (k * n) + l * n + j;
                        
                        sum += a_buf[a_idx] * b_buf[b_idx];
                    }
                    
                    const result_idx = batch_idx * (m * n) + i * n + j;
                    result_buf[result_idx] = sum;
                }
            }
        }
    }
    
    /// Sum elements of a tensor along specified axes
    pub fn sum_reduce(comptime T: type, a: Tensor, axes: []const usize, keep_dims: bool, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        // First, zero-initialize the result buffer
        for (result_buf) |*val| {
            val.* = 0.0;
        }
        
        // Create helper arrays for stride calculation
        var a_strides: [8]usize = undefined;  // Support up to 8 dimensions
        for (0..a.shape.rank()) |i| {
            a_strides[i] = a.shape.strides[i];
        }
        
        var result_strides: [8]usize = undefined;  // Support up to 8 dimensions
        for (0..result.shape.rank()) |i| {
            result_strides[i] = result.shape.strides[i];
        }
        
        // Create mapping from input dim to output dim
        var dim_map: [8]isize = undefined;
        var next_out_dim: usize = 0;
        
        // For each input dimension, map it to the corresponding output dimension or -1 if it's reduced
        for (0..a.shape.rank()) |in_dim| {
            // Check if this dimension is being reduced
            var is_reduced = false;
            for (axes) |axis| {
                if (in_dim == axis) {
                    is_reduced = true;
                    break;
                }
            }
            
            if (is_reduced) {
                dim_map[in_dim] = -1;  // Mark as reduced
            } else {
                dim_map[in_dim] = @intCast(next_out_dim);
                next_out_dim += 1;
            }
        }
        
        // For each element in the input tensor
        var input_coords: [8]usize = [_]usize{0} ** 8;
        var remaining_elements = a.shape.elemCount();
        var in_idx: usize = 0;
        
        while (remaining_elements > 0) {
            // Compute the output coordinates
            var out_coords: [8]usize = [_]usize{0} ** 8;
            var valid_output = true;
            
            for (0..a.shape.rank()) |in_dim| {
                if (dim_map[in_dim] >= 0) {
                    const out_dim = @as(usize, @intCast(dim_map[in_dim]));
                    out_coords[out_dim] = input_coords[in_dim];
                } else if (keep_dims) {
                    // If we're keeping dimensions, the reduced dims will be at 0
                    const out_dim = in_dim;
                    out_coords[out_dim] = 0;
                }
            }
            
            // Compute output index
            var out_idx: usize = 0;
            for (0..result.shape.rank()) |i| {
                out_idx += out_coords[i] * result_strides[i];
            }
            
            // Accumulate the value
            if (valid_output) {
                result_buf[out_idx] += a_buf[in_idx];
            }
            
            // Move to the next input element
            in_idx += 1;
            remaining_elements -= 1;
            
            // Update the input coordinates
            var carry: usize = 1;
            for (0..a.shape.rank()) |i| {
                if (carry == 0) break;
                input_coords[i] += carry;
                if (input_coords[i] >= a.shape.dims[i]) {
                    input_coords[i] = 0;
                    carry = 1;
                } else {
                    carry = 0;
                }
            }
        }
    }
    
    /// Element-wise power operation
    pub fn pow(comptime T: type, base: Tensor, exponent: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const base_buf = ptrCastHelper([*]f32, base.buffer.data.ptr)[0..base.shape.elemCount()];
        const exponent_buf = ptrCastHelper([*]f32, exponent.buffer.data.ptr)[0..exponent.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        // Determine if exponent is a scalar
        const is_scalar_exponent = exponent.shape.elemCount() == 1;
        
        if (is_scalar_exponent) {
            const exp_val = exponent_buf[0];
            for (base_buf, 0..) |base_val, i| {
                result_buf[i] = std.math.pow(T, base_val, exp_val);
            }
        } else {
            // Element-wise power
            for (base_buf, exponent_buf, 0..) |base_val, exp_val, i| {
                result_buf[i] = std.math.pow(T, base_val, exp_val);
            }
        }
    }
    
    /// Expand a tensor to a target shape (broadcasting)
    pub fn expand(comptime T: type, a: Tensor, target_shape: []const usize, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        // Create helper arrays for stride calculation
        var a_strides: [8]usize = [_]usize{0} ** 8;  // Support up to 8 dimensions
        var a_dims: [8]usize = [_]usize{1} ** 8;     // Initialize with size 1 for all dims
        
        // Fill in the actual dimensions, starting from the rightmost (lowest) dimensions
        const rank_diff = target_shape.len - a.shape.rank();
        for (0..a.shape.rank()) |i| {
            a_dims[i + rank_diff] = a.shape.dims[i];
        }
        
        // Calculate strides
        var stride: usize = 1;
        for (0..8) |i_rev| {
            const i = 7 - i_rev;  // Go in reverse to calculate strides
            a_strides[i] = stride;
            stride *= a_dims[i];
        }
        
        // For each element in the output tensor
        var coords: [8]usize = [_]usize{0} ** 8;
        var target_strides: [8]usize = [_]usize{0} ** 8;
        
        // Calculate target strides
        stride = 1;
        for (0..target_shape.len) |i_rev| {
            const i = target_shape.len - 1 - i_rev;
            target_strides[i] = stride;
            stride *= target_shape[i];
        }
        
        // For each element in the result tensor
        var remaining_elements = result.shape.elemCount();
        var out_idx: usize = 0;
        
        while (remaining_elements > 0) {
            // Calculate the current coordinates in the target shape
            var tmp_idx = out_idx;
            for (0..target_shape.len) |i| {
                coords[i] = tmp_idx / target_strides[i];
                tmp_idx %= target_strides[i];
            }
            
            // Map to coordinates in the source tensor - for each dimension,
            // if the source dim is 1, use 0; otherwise use the target coordinate
            var in_idx: usize = 0;
            for (0..target_shape.len) |i| {
                // Only consider dimensions that also exist in the input
                if (i >= rank_diff) {
                    const a_dim = i - rank_diff;
                    if (a_dim < a.shape.rank()) {
                        // If dim size is 1, broadcast by using coord=0
                        const coord = if (a.shape.dims[a_dim] == 1) 0 else coords[i];
                        in_idx += coord * a.shape.strides[a_dim];
                    }
                }
            }
            
            // Copy the value
            result_buf[out_idx] = a_buf[in_idx];
            
            // Move to the next element
            out_idx += 1;
            remaining_elements -= 1;
        }
    }
    
    /// Reshape a tensor to a new shape
    pub fn reshape(comptime T: type, a: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        // For reshape, we just copy the data since it's the same number of elements
        std.mem.copy(f32, result_buf, a_buf);
    }

    /// Subtract two tensors element-wise
    pub fn subtract(comptime T: type, a: Tensor, b: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const b_buf = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..b.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        for (a_buf, b_buf, 0..) |a_val, b_val, i| {
            result_buf[i] = a_val - b_val;
        }
    }

    /// Multiply two tensors element-wise
    pub fn multiply(comptime T: type, a: Tensor, b: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const b_buf = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..b.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        for (a_buf, b_buf, 0..) |a_val, b_val, i| {
            result_buf[i] = a_val * b_val;
        }
    }

    /// Divide two tensors element-wise
    pub fn divide(comptime T: type, a: Tensor, b: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const b_buf = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..b.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        for (a_buf, b_buf, 0..) |a_val, b_val, i| {
            result_buf[i] = a_val / b_val;
        }
    }

    /// Matrix multiplication for 2D tensors
    pub fn matmul(comptime T: type, a: Tensor, b: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr);
        const b_buf = ptrCastHelper([*]f32, b.buffer.data.ptr);
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr);
        
        const m = a.shape.dims[0];
        const n = a.shape.dims[1]; // also b.shape.dims[0]
        const p = b.shape.dims[1];
        
        // Simple 3-loop matrix multiplication
        for (0..m) |i| {
            for (0..p) |j| {
                var sum: f32 = 0.0;
                for (0..n) |k| {
                    sum += a_buf[i * n + k] * b_buf[k * p + j];
                }
                result_buf[i * p + j] = sum;
            }
        }
    }

    /// Apply ReLU activation function element-wise
    pub fn relu(comptime T: type, a: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        for (a_buf, 0..) |a_val, i| {
            result_buf[i] = if (a_val > 0) a_val else 0;
        }
    }

    /// Apply softmax to 2D tensor (on last dimension)
    pub fn softmax(comptime T: type, a: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr);
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr);
        
        const batch_size = a.shape.dims[0];
        const feature_size = a.shape.dims[1];
        
        // Apply softmax for each row
        for (0..batch_size) |i| {
            // Find max value for numerical stability
            var max_val: f32 = std.math.floatMin(f32);
            for (0..feature_size) |j| {
                const val = a_buf[i * feature_size + j];
                max_val = if (val > max_val) val else max_val;
            }
            
            // Compute exp(x - max) and sum
            var sum: f32 = 0.0;
            for (0..feature_size) |j| {
                const val = std.math.exp(a_buf[i * feature_size + j] - max_val);
                result_buf[i * feature_size + j] = val;
                sum += val;
            }
            
            // Normalize
            for (0..feature_size) |j| {
                result_buf[i * feature_size + j] /= sum;
            }
        }
    }

    /// Transpose a 2D tensor
    pub fn transpose(comptime T: type, a: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const a_buf = ptrCastHelper([*]f32, a.buffer.data.ptr);
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr);
        
        const rows = a.shape.dims[0];
        const cols = a.shape.dims[1];
        
        for (0..rows) |i| {
            for (0..cols) |j| {
                result_buf[j * rows + i] = a_buf[i * cols + j];
            }
        }
    }
    
    /// Embedding lookup - fetch embeddings based on indices
    pub fn embedding_lookup(comptime T: type, params: Tensor, indices: Tensor, result: *Tensor) void {
        // Move type checking to comptime
        comptime {
            if (T != f32) @compileError("Only f32 supported for now");
        }
        
        const vocab_size = params.shape.dims[0];
        const embed_dim = params.shape.dims[1];
        
        // Extract batch size and sequence length from indices shape
        var batch_size: usize = 1;
        var seq_len: usize = 1;
        
        if (indices.shape.rank() == 1) {
            seq_len = indices.shape.dims[0];
        } else if (indices.shape.rank() == 2) {
            batch_size = indices.shape.dims[0];
            seq_len = indices.shape.dims[1];
        }
        
        const params_buf = ptrCastHelper([*]f32, params.buffer.data.ptr)[0..params.shape.elemCount()];
        const indices_buf = ptrCastHelper([*]f32, indices.buffer.data.ptr)[0..indices.shape.elemCount()];
        const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
        
        // Perform lookup
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                // Get the token id from indices
                var index_pos: usize = 0;
                if (indices.shape.rank() == 1) {
                    index_pos = s;
                } else {
                    index_pos = b * seq_len + s;
                }
                
                // Bounds check
                if (index_pos >= indices_buf.len) {
                    continue;
                }
                
                const token_id_f = indices_buf[index_pos];
                // Safely convert to int and bound to vocab size
                const token_id_i = @as(i32, @intFromFloat(token_id_f));
                const token_id = @as(usize, @intCast(@max(0, @min(token_id_i, @as(i32, @intCast(vocab_size - 1))))));
                
                // Copy embedding for this token
                for (0..embed_dim) |d| {
                    const src_idx = token_id * embed_dim + d;
                    const dst_idx = (b * seq_len + s) * embed_dim + d;
                    
                    // Bounds checking
                    if (src_idx >= params_buf.len) {
                        continue;
                    }
                    if (dst_idx >= result_buf.len) {
                        continue;
                    }
                    
                    result_buf[dst_idx] = params_buf[src_idx];
                }
            }
        }
    }
};

/// --- Plan Definition ---
/// Generic type for comptime-generated operation plans
pub fn PlanType(comptime Backend: type, comptime InputType: type, comptime OutputType: type) type {
    _ = Backend; // Used to match interface, actual backend implementation comes from specific plan
    return struct {
        const Self = @This();
        allocator: Allocator,  // Make allocator public

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        pub fn run(self: Self, input: InputType) !OutputType {
            // To be overridden by specific plans
            _ = self;
            _ = input;
            return OpError.NotImplemented;
        }
        
        // Default gradient type for operations without gradients
        pub const GradType = void;
    };
}

/// --- Backend Definition ---
/// CPU Backend with default primitive implementations
pub const CpuBackend = struct {
    pub fn hasPrimitive(comptime name: []const u8) bool {
        return std.mem.eql(u8, name, "add") or
               std.mem.eql(u8, name, "subtract") or
               std.mem.eql(u8, name, "multiply") or
               std.mem.eql(u8, name, "divide") or
               std.mem.eql(u8, name, "matmul") or
               std.mem.eql(u8, name, "relu") or
               std.mem.eql(u8, name, "softmax") or
               std.mem.eql(u8, name, "transpose") or
               std.mem.eql(u8, name, "embedding_lookup") or
               std.mem.eql(u8, name, "sum_reduce") or
               std.mem.eql(u8, name, "pow") or
               std.mem.eql(u8, name, "expand") or
               std.mem.eql(u8, name, "reshape") or
               std.mem.eql(u8, name, "batched_matmul");
    }

    // Include default primitive implementations
    pub usingnamespace Primitives;
};

/// --- Operation Plans ---

/// Add two tensors element-wise
pub fn AddPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("add")) @compileError("Backend must implement add primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor }; // Make InputType public
        pub const op_type = OpType.add;
        pub const GradType = struct { da: Tensor, db: Tensor }; // Add GradType for autodiff
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        /// Compute gradients for add operation: da = grad_out, db = grad_out
        pub fn gradient(self: @This(), grad_out: Tensor, _: InputType) !GradType {
            _ = self; // Not used
            
            // Both inputs receive the same gradient
            var da = try grad_out.clone();
            errdefer da.deinit();
            
            var db = try grad_out.clone();
            errdefer db.deinit();
            
            return .{ .da = da, .db = db };
        }

        pub fn run(self: @This(), input: InputType) !Tensor {
            // Runtime shape check for inputs
            if (!input.a.shape.eql(input.b.shape)) {
                return OpError.ShapeMismatch;
            }

            // Verify input shapes match the comptime shape if provided
            if (comptime shape != null) {
                var shape_matches = true;
                if (input.a.shape.dims.len != shape.?.len) {
                    shape_matches = false;
                } else {
                    for (input.a.shape.dims, 0..) |dim, i| {
                        if (dim != shape.?[i]) {
                            shape_matches = false;
                            break;
                        }
                    }
                }
                if (!shape_matches) {
                    return OpError.ShapeMismatch;
                }
            }

            // Check dtypes
            if (input.a.dtype != DType.f32 or input.b.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }

            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.a.shape.dims, DType.f32, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.add(T, input.a, input.b, &result);
            return result;
        }
    };
}

/// Subtract two tensors element-wise
pub fn SubtractPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("subtract")) @compileError("Backend must implement subtract primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.subtract;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        /// Compute gradients for subtract operation: da = grad_out, db = -grad_out
        pub fn gradient(self: @This(), grad_out: Tensor, _: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // First input gets the gradient directly
            var da = try grad_out.clone();
            errdefer da.deinit();
            
            // Create negative gradient for second input
            const negative_one = try Tensor.filled(allocator, grad_out.shape.dims, grad_out.dtype, -1.0, grad_out.backend);
            errdefer negative_one.deinit();
            
            var db = try multiply(allocator, grad_out, negative_one);
            errdefer db.deinit();
            
            // Clean up temporary tensor
            negative_one.deinit();
            
            return .{ .da = da, .db = db };
        }

        pub fn run(self: @This(), input: InputType) !Tensor {
            // Runtime shape check
            if (!input.a.shape.eql(input.b.shape)) {
                return OpError.ShapeMismatch;
            }

            // Verify input shapes match the comptime shape if provided
            if (comptime shape != null) {
                var shape_matches = true;
                if (input.a.shape.dims.len != shape.?.len) {
                    shape_matches = false;
                } else {
                    for (input.a.shape.dims, 0..) |dim, i| {
                        if (dim != shape.?[i]) {
                            shape_matches = false;
                            break;
                        }
                    }
                }
                if (!shape_matches) {
                    return OpError.ShapeMismatch;
                }
            }

            // Check dtypes
            if (input.a.dtype != DType.f32 or input.b.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }

            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.a.shape.dims, DType.f32, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.subtract(T, input.a, input.b, &result);
            return result;
        }
    };
}

/// Multiply two tensors element-wise
pub fn MultiplyPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("multiply")) @compileError("Backend must implement multiply primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.multiply;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        /// Compute gradients for multiply operation: da = grad_out * b, db = grad_out * a
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // Input a gradient is element-wise product of grad_out and b
            var da = try multiply(allocator, grad_out, input.b);
            errdefer da.deinit();
            
            // Input b gradient is element-wise product of grad_out and a
            var db = try multiply(allocator, grad_out, input.a);
            errdefer db.deinit();
            
            return .{ .da = da, .db = db };
        }

        pub fn run(self: @This(), input: InputType) !Tensor {
            // Runtime shape check
            if (!input.a.shape.eql(input.b.shape)) {
                return OpError.ShapeMismatch;
            }

            // Verify input shapes match the comptime shape if provided
            if (comptime shape != null) {
                var shape_matches = true;
                if (input.a.shape.dims.len != shape.?.len) {
                    shape_matches = false;
                } else {
                    for (input.a.shape.dims, 0..) |dim, i| {
                        if (dim != shape.?[i]) {
                            shape_matches = false;
                            break;
                        }
                    }
                }
                if (!shape_matches) {
                    return OpError.ShapeMismatch;
                }
            }

            // Check dtypes
            if (input.a.dtype != DType.f32 or input.b.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }

            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.a.shape.dims, DType.f32, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.multiply(T, input.a, input.b, &result);
            return result;
        }
    };
}

/// Divide two tensors element-wise 
pub fn DividePlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("divide")) @compileError("Backend must implement divide primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.divide;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }

        /// Compute gradients for divide: da = grad_out / b, db = -grad_out * a / (b * b)
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // Input a gradient is element-wise division of grad_out by b
            var da = try divide(allocator, grad_out, input.b);
            errdefer da.deinit();
            
            // Input b gradient is more complex: -grad_out * a / (b * b)
            // First calculate b^2
            var b_squared = try multiply(allocator, input.b, input.b);
            errdefer b_squared.deinit();
            
            // Calculate a / b^2
            var a_div_b_squared = try divide(allocator, input.a, b_squared);
            errdefer a_div_b_squared.deinit();
            
            // Multiply by grad_out
            var temp = try multiply(allocator, grad_out, a_div_b_squared);
            errdefer temp.deinit();
            
            // Negate the result
            const negative_one = try Tensor.filled(allocator, temp.shape.dims, temp.dtype, -1.0, temp.backend);
            errdefer negative_one.deinit();
            
            var db = try multiply(allocator, temp, negative_one);
            errdefer db.deinit();
            
            // Clean up temporaries
            b_squared.deinit();
            a_div_b_squared.deinit();
            temp.deinit();
            negative_one.deinit();
            
            return .{ .da = da, .db = db };
        }

        pub fn run(self: @This(), input: InputType) !Tensor {
            // Runtime shape check
            if (!input.a.shape.eql(input.b.shape)) {
                return OpError.ShapeMismatch;
            }

            // Verify input shapes match the comptime shape if provided
            if (comptime shape != null) {
                var shape_matches = true;
                if (input.a.shape.dims.len != shape.?.len) {
                    shape_matches = false;
                } else {
                    for (input.a.shape.dims, 0..) |dim, i| {
                        if (dim != shape.?[i]) {
                            shape_matches = false;
                            break;
                        }
                    }
                }
                if (!shape_matches) {
                    return OpError.ShapeMismatch;
                }
            }

            // Check dtypes
            if (input.a.dtype != DType.f32 or input.b.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }

            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.a.shape.dims, DType.f32, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.divide(T, input.a, input.b, &result);
            return result;
        }
    };
}

/// Matrix multiplication for 2D tensors
pub fn MatmulPlan(comptime Backend: type, comptime T: type, comptime M: ?usize, comptime N: ?usize, comptime P: ?usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("matmul")) @compileError("Backend must implement matmul primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.matmul;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        /// Compute gradients for matmul: da = grad_out @ b^T, db = a^T @ grad_out
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // Compute b transpose
            const b_transpose = try transpose(allocator, input.b);
            errdefer b_transpose.deinit();
            
            // Gradient for a is grad_out @ b^T
            var da = try matmul(allocator, grad_out, b_transpose);
            errdefer da.deinit();
            
            // Compute a transpose
            const a_transpose = try transpose(allocator, input.a);
            errdefer a_transpose.deinit();
            
            // Gradient for b is a^T @ grad_out
            var db = try matmul(allocator, a_transpose, grad_out);
            errdefer db.deinit();
            
            // Clean up temporaries
            b_transpose.deinit();
            a_transpose.deinit();
            
            return .{ .da = da, .db = db };
        }

        pub fn run(self: @This(), input: InputType) !Tensor {
            // Runtime validation for 2D shapes
            if (input.a.shape.rank() != 2 or input.b.shape.rank() != 2) {
                return OpError.DimensionMismatch;
            }

            // Check dimension compatibility for matmul
            if (input.a.shape.dims[1] != input.b.shape.dims[0]) {
                return OpError.ShapeMismatch;
            }

            // Verify input shapes match comptime dimensions if provided
            const m = input.a.shape.dims[0];
            const n = input.a.shape.dims[1];
            const p = input.b.shape.dims[1];
            
            if (M != null and m != M.?) return OpError.ShapeMismatch;
            if (N != null and n != N.?) return OpError.ShapeMismatch;
            if (P != null and p != P.?) return OpError.ShapeMismatch;

            // Check dtypes
            if (input.a.dtype != DType.f32 or input.b.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }

            // Allocate result
            const result_dims = [_]usize{ m, p };
            var result = try Tensor.zeros(self.base.allocator, &result_dims, DType.f32, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.matmul(T, input.a, input.b, &result);
            return result;
        }
    };
}

/// ReLU activation element-wise
pub fn ReluPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("relu")) @compileError("Backend must implement relu primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = Tensor;
        pub const op_type = OpType.relu;
        pub const GradType = Tensor;
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        /// Compute gradient for relu: da = grad_out * (input > 0)
        pub fn gradient(self: @This(), grad_out: Tensor, input: Tensor) !GradType {
            const allocator = self.base.allocator;
            
            // Create mask with 1s where input > 0, 0s elsewhere
            var mask = try Tensor.zeros(allocator, input.shape.dims, input.dtype, input.backend);
            errdefer mask.deinit();
            
            const input_buf = ptrCastHelper([*]f32, input.buffer.data.ptr)[0..input.shape.elemCount()];
            const mask_buf = ptrCastHelper([*]f32, mask.buffer.data.ptr)[0..mask.shape.elemCount()];
            
            for (input_buf, 0..) |val, i| {
                mask_buf[i] = if (val > 0) 1.0 else 0.0;
            }
            
            // Gradient is element-wise product of upstream gradient and mask
            var result = try multiply(allocator, grad_out, mask);
            errdefer result.deinit();
            
            // Clean up temporary
            mask.deinit();
            
            return result;
        }

        pub fn run(self: @This(), input: Tensor) !Tensor {
            // Verify input shape matches the comptime shape if provided
            if (comptime shape != null) {
                var shape_matches = true;
                if (input.shape.dims.len != shape.?.len) {
                    shape_matches = false;
                } else {
                    for (input.shape.dims, 0..) |dim, i| {
                        if (dim != shape.?[i]) {
                            shape_matches = false;
                            break;
                        }
                    }
                }
                if (!shape_matches) {
                    return OpError.ShapeMismatch;
                }
            }

            if (input.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }

            var result = try Tensor.zeros(self.base.allocator, input.shape.dims, DType.f32, input.backend);
            errdefer result.deinit();

            Backend.relu(T, input, &result);
            return result;
        }
    };
}

/// Softmax activation for 2D tensors (on last dimension)
pub fn SoftmaxPlan(comptime Backend: type, comptime T: type, comptime batch_size: ?usize, comptime feature_size: ?usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("softmax")) @compileError("Backend must implement softmax primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = Tensor;
        pub const op_type = OpType.softmax;
        pub const GradType = Tensor;
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }

        pub fn run(self: @This(), input: Tensor) !Tensor {
            // Runtime validation
            if (input.shape.rank() != 2) {
                return OpError.DimensionMismatch;
            }

            // Verify dimensions if comptime values provided
            if (comptime batch_size != null and input.shape.dims[0] != batch_size.?) {
                return OpError.ShapeMismatch;
            }
            
            if (comptime feature_size != null and input.shape.dims[1] != feature_size.?) {
                return OpError.ShapeMismatch;
            }

            if (input.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }

            var result = try Tensor.zeros(self.base.allocator, input.shape.dims, DType.f32, input.backend);
            errdefer result.deinit();

            Backend.softmax(T, input, &result);
            return result;
        }
    };
}

/// Transpose a 2D tensor
pub fn TransposePlan(comptime Backend: type, comptime T: type, comptime rows: ?usize, comptime cols: ?usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("transpose")) @compileError("Backend must implement transpose primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = Tensor;
        pub const op_type = OpType.transpose;
        pub const GradType = Tensor;
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }

        pub fn run(self: @This(), input: Tensor) !Tensor {
            // Runtime validation
            if (input.shape.rank() != 2) {
                return OpError.DimensionMismatch;
            }

            // Verify dimensions if comptime values provided
            if (comptime rows != null and input.shape.dims[0] != rows.?) {
                return OpError.ShapeMismatch;
            }
            
            if (comptime cols != null and input.shape.dims[1] != cols.?) {
                return OpError.ShapeMismatch;
            }

            if (input.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }

            // Result dimensions are swapped
            const result_dims = [_]usize{ input.shape.dims[1], input.shape.dims[0] };
            var result = try Tensor.zeros(self.base.allocator, &result_dims, DType.f32, input.backend);
            errdefer result.deinit();

            Backend.transpose(T, input, &result);
            return result;
        }
    };
}

/// Embedding lookup - fetch embeddings based on indices
pub fn EmbeddingLookupPlan(comptime Backend: type, comptime T: type, comptime vocab_size: ?usize, comptime embed_dim: ?usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("embedding_lookup")) @compileError("Backend must implement embedding_lookup primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { params: Tensor, indices: Tensor };
        pub const op_type = OpType.embedding_lookup;
        pub const GradType = Tensor; // Only params get gradients
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }

        pub fn run(self: @This(), input: InputType) !Tensor {
            // Validate params tensor dimensions
            if (input.params.shape.rank() != 2) {
                return OpError.InvalidEmbeddingShape;
            }
            
            // Verify dimensions if comptime values provided
            if (comptime vocab_size != null and input.params.shape.dims[0] != vocab_size.?) {
                return OpError.ShapeMismatch;
            }
            
            if (comptime embed_dim != null and input.params.shape.dims[1] != embed_dim.?) {
                return OpError.ShapeMismatch;
            }
            
            // Parse indices dimensions
            var batch_size: usize = 1;
            var seq_len: usize = 1;
            
            if (input.indices.shape.rank() == 1) {
                seq_len = input.indices.shape.dims[0];
            } else if (input.indices.shape.rank() == 2) {
                batch_size = input.indices.shape.dims[0];
                seq_len = input.indices.shape.dims[1];
            } else {
                return OpError.InvalidIndicesShape;
            }
            
            // Check dtypes
            if (input.params.dtype != DType.f32 or input.indices.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }
            
            // Create output tensor
            const result_dims = [_]usize{ batch_size, seq_len, input.params.shape.dims[1] };
            var result = try Tensor.zeros(self.base.allocator, &result_dims, DType.f32, input.params.backend);
            errdefer result.deinit();
            
            // Execute primitive
            Backend.embedding_lookup(T, input.params, input.indices, &result);
            return result;
        }
    };
}

/// Sum elements of a tensor along specified axes
pub fn SumReducePlan(comptime Backend: type, comptime T: type, comptime axes: ?[]const usize, comptime keep_dims: bool) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("sum_reduce")) @compileError("Backend must implement sum_reduce primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = Tensor;
        pub const op_type = OpType.sum_reduce;
        pub const GradType = Tensor;
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,
        
        // Store axes and keep_dims for backward pass
        axes_buffer: []usize,

        pub fn init(allocator: Allocator) @This() {
            // Allocate storage for axes, or use empty array if axes is null
            var axes_buffer = allocator.alloc(usize, if (comptime axes != null) axes.?.len else 0) catch @panic("Failed to allocate axes buffer");
            
            // If axes is provided at comptime, copy the values
            if (comptime axes != null) {
                for (axes.?, 0..) |axis, i| {
                    axes_buffer[i] = axis;
                }
            }
            
            return .{
                .base = Base.init(allocator),
                .axes_buffer = axes_buffer,
            };
        }
        
        pub fn deinit(self: *@This()) void {
            self.base.allocator.free(self.axes_buffer);
        }
        
        /// Compute gradient for sum_reduce: broadcast grad_out back to input shape
        pub fn gradient(self: @This(), grad_out: Tensor, input: Tensor) !GradType {
            const allocator = self.base.allocator;
            
            // The gradient is gradient_out broadcasted back to the original input shape
            // We need to expand/broadcast grad_out to match input shape
            
            // Create result tensor with input shape
            var result = try Tensor.zeros(allocator, input.shape.dims, input.dtype, input.backend);
            errdefer result.deinit();
            
            // Determine if we need to reshape first (if keep_dims was false)
            if (!keep_dims) {
                // Need to reshape grad_out to have singleton dimensions where reduction happened
                var expanded_shape = try allocator.alloc(usize, input.shape.rank());
                defer allocator.free(expanded_shape);
                
                // Start with input shape
                std.mem.copy(usize, expanded_shape, input.shape.dims);
                
                // For each reduced axis, set dimension to 1
                for (self.axes_buffer) |axis| {
                    expanded_shape[axis] = 1;
                }
                
                // Reshape grad_out to have singleton dimensions
                var reshaped_grad = try Tensor.zeros(allocator, expanded_shape, grad_out.dtype, grad_out.backend);
                errdefer reshaped_grad.deinit();
                
                // Copy data into reshaped tensor
                const grad_buf = ptrCastHelper([*]f32, grad_out.buffer.data.ptr)[0..grad_out.shape.elemCount()];
                const reshaped_buf = ptrCastHelper([*]f32, reshaped_grad.buffer.data.ptr)[0..reshaped_grad.shape.elemCount()];
                
                std.mem.copy(f32, reshaped_buf, grad_buf);
                
                // Now broadcast from the reshaped tensor to input shape
                Backend.expand(T, reshaped_grad, input.shape.dims, &result);
                
                // Clean up temporary
                reshaped_grad.deinit();
            } else {
                // keep_dims was true, so we can broadcast directly
                Backend.expand(T, grad_out, input.shape.dims, &result);
            }
            
            return result;
        }

        pub fn run(self: @This(), input: Tensor) !Tensor {
            // Check dtypes
            if (input.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }
            
            // Compute output shape based on input shape and reduction axes
            var output_dims = try self.base.allocator.alloc(usize, input.shape.rank());
            defer self.base.allocator.free(output_dims);
            
            // Copy input dimensions as starting point
            std.mem.copy(usize, output_dims, input.shape.dims);
            
            // Set reduced dimensions to 1 if keep_dims, or remove them if !keep_dims
            for (self.axes_buffer) |axis| {
                if (axis >= input.shape.rank()) {
                    return OpError.InvalidAxes;
                }
                
                if (keep_dims) {
                    output_dims[axis] = 1;
                }
            }
            
            // If not keep_dims, we need to create a new shape without the reduced dimensions
            var result_dims: []usize = undefined;
            
            if (!keep_dims) {
                // Count non-reduced dimensions
                var non_reduced_count: usize = input.shape.rank();
                for (self.axes_buffer) |_| {
                    non_reduced_count -= 1;
                }
                
                // Allocate and fill result dimensions
                result_dims = try self.base.allocator.alloc(usize, non_reduced_count);
                defer self.base.allocator.free(result_dims);
                
                var result_idx: usize = 0;
                for (0..input.shape.rank()) |i| {
                    // Check if this dimension is being reduced
                    var is_reduced = false;
                    for (self.axes_buffer) |axis| {
                        if (i == axis) {
                            is_reduced = true;
                            break;
                        }
                    }
                    
                    if (!is_reduced) {
                        result_dims[result_idx] = output_dims[i];
                        result_idx += 1;
                    }
                }
            } else {
                // Use the modified input dimensions
                result_dims = output_dims;
            }
            
            // Create output tensor
            var result = try Tensor.zeros(self.base.allocator, result_dims, input.dtype, input.backend);
            errdefer result.deinit();
            
            // Execute primitive
            Backend.sum_reduce(T, input, self.axes_buffer, keep_dims, &result);
            return result;
        }
    };
}

/// Element-wise power operation
pub fn PowPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("pow")) @compileError("Backend must implement pow primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { base: Tensor, exponent: Tensor };
        pub const op_type = OpType.pow;
        pub const GradType = struct { dbase: Tensor, dexponent: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        /// Compute gradients for power operation
        /// dbase = exponent * base^(exponent-1) * grad_out
        /// dexponent = base^exponent * ln(base) * grad_out
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // Check if exponent is a scalar
            const is_scalar_exponent = input.exponent.shape.elemCount() == 1;
            
            // Compute exponent - 1
            var exponent_minus_one: Tensor = undefined;
            
            if (is_scalar_exponent) {
                // If scalar, just create a new scalar with value-1
                exponent_minus_one = try Tensor.zeros(allocator, input.exponent.shape.dims, input.exponent.dtype, input.exponent.backend);
                errdefer exponent_minus_one.deinit();
                
                const exp_val = try input.exponent.getScalar(&[_]usize{0});
                try exponent_minus_one.setScalar(&[_]usize{0}, exp_val - 1.0);
            } else {
                // Element-wise subtraction
                const one = try Tensor.filled(allocator, input.exponent.shape.dims, input.exponent.dtype, 1.0, input.exponent.backend);
                defer one.deinit();
                
                exponent_minus_one = try subtract(allocator, input.exponent, one);
                errdefer exponent_minus_one.deinit();
            }
            
            // Compute base^(exponent-1)
            var base_pow_exp_minus_one = try Tensor.zeros(allocator, input.base.shape.dims, input.base.dtype, input.base.backend);
            errdefer base_pow_exp_minus_one.deinit();
            
            Backend.pow(T, input.base, exponent_minus_one, &base_pow_exp_minus_one);
            
            // Compute exponent * base^(exponent-1)
            var temp1: Tensor = undefined;
            
            if (is_scalar_exponent) {
                // If scalar exponent, just multiply by the scalar value
                const exp_val = try input.exponent.getScalar(&[_]usize{0});
                const scalar = try Tensor.filled(allocator, input.base.shape.dims, input.base.dtype, exp_val, input.base.backend);
                defer scalar.deinit();
                
                temp1 = try multiply(allocator, base_pow_exp_minus_one, scalar);
                errdefer temp1.deinit();
            } else {
                temp1 = try multiply(allocator, input.exponent, base_pow_exp_minus_one);
                errdefer temp1.deinit();
            }
            
            // Compute dbase = exponent * base^(exponent-1) * grad_out
            var dbase = try multiply(allocator, temp1, grad_out);
            errdefer dbase.deinit();
            
            // Clean up
            exponent_minus_one.deinit();
            base_pow_exp_minus_one.deinit();
            temp1.deinit();
            
            // For simple cases like x^2 or x^(-0.5), we could optimize further
            
            // For dexponent, we need log(base) which gets complex if base is negative or zero
            // For common cases like x^2, we just need base^exponent * ln(base) * grad_out
            
            // Compute base^exponent (we already have the result of the forward pass)
            var result = try self.run(input);
            defer result.deinit();
            
            // Compute log(base)
            var log_base = try Tensor.zeros(allocator, input.base.shape.dims, input.base.dtype, input.base.backend);
            errdefer log_base.deinit();
            
            // Calculate log safely
            const base_buf = ptrCastHelper([*]f32, input.base.buffer.data.ptr)[0..input.base.shape.elemCount()];
            const log_buf = ptrCastHelper([*]f32, log_base.buffer.data.ptr)[0..log_base.shape.elemCount()];
            
            for (base_buf, 0..) |val, i| {
                // Handle edge cases - if base is negative or zero, set log to 0
                log_buf[i] = if (val > 0) std.math.log(T, val) else 0;
            }
            
            // Compute result * log(base)
            var temp2 = try multiply(allocator, result, log_base);
            errdefer temp2.deinit();
            
            // Compute dexponent = base^exponent * ln(base) * grad_out
            var dexponent = try multiply(allocator, temp2, grad_out);
            errdefer dexponent.deinit();
            
            // Clean up
            log_base.deinit();
            temp2.deinit();
            
            return .{ .dbase = dbase, .dexponent = dexponent };
        }

        pub fn run(self: @This(), input: InputType) !Tensor {
            // Verify shapes: either same shape or exponent is scalar
            const is_scalar_exponent = input.exponent.shape.elemCount() == 1;
            
            if (!is_scalar_exponent and !input.base.shape.eql(input.exponent.shape)) {
                return OpError.ShapeMismatch;
            }
            
            // Verify input shapes match the comptime shape if provided
            if (comptime shape != null) {
                var shape_matches = true;
                if (input.base.shape.dims.len != shape.?.len) {
                    shape_matches = false;
                } else {
                    for (input.base.shape.dims, 0..) |dim, i| {
                        if (dim != shape.?[i]) {
                            shape_matches = false;
                            break;
                        }
                    }
                }
                if (!shape_matches) {
                    return OpError.ShapeMismatch;
                }
            }
            
            // Check dtypes
            if (input.base.dtype != DType.f32 or input.exponent.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }
            
            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.base.shape.dims, DType.f32, input.base.backend);
            errdefer result.deinit();
            
            // Execute primitive
            Backend.pow(T, input.base, input.exponent, &result);
            return result;
        }
    };
}

/// Expand a tensor to a target shape (broadcasting)
pub fn ExpandPlan(comptime Backend: type, comptime T: type, comptime target_shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("expand")) @compileError("Backend must implement expand primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { input: Tensor, target_shape_runtime: ?[]const usize = null };
        pub const op_type = OpType.expand;
        pub const GradType = Tensor;
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,
        
        // Store the resolved target shape for backward pass if using runtime shape
        target_shape_storage: ?[]usize = null,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        pub fn deinit(self: *@This()) void {
            if (self.target_shape_storage != null) {
                self.base.allocator.free(self.target_shape_storage.?);
                self.target_shape_storage = null;
            }
        }
        
        /// Compute gradient for expand operation: sum over broadcast dimensions
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // Get the output shape - either from comptime or from runtime storage
            var shape: []const usize = undefined;
            if (comptime target_shape != null) {
                shape = target_shape.?;
            } else if (self.target_shape_storage != null) {
                shape = self.target_shape_storage.?;
            } else {
                // This shouldn't happen if run() was called before gradient()
                return error.MissingTargetShape;
            }
            
            // To compute the gradient, we need to sum the gradient along the dimensions
            // that were broadcasted (expanded)
            
            // Identify the dimensions that were broadcasted
            var axes_buffer = std.ArrayList(usize).init(allocator);
            defer axes_buffer.deinit();
            
            // Size difference - need to account for expanded rank and expanded dimensions
            const rank_diff = shape.len - input.input.shape.rank();
            
            // First, handle dimensions added by rank expansion (input dims were prepended with 1s)
            for (0..rank_diff) |i| {
                try axes_buffer.append(i);
            }
            
            // Then handle dimensions that were broadcasted (input dim was 1, output is greater)
            for (0..input.input.shape.rank()) |i| {
                const in_dim = i;
                const out_dim = i + rank_diff;
                
                if (in_dim < input.input.shape.rank() and 
                    input.input.shape.dims[in_dim] == 1 and 
                    out_dim < shape.len and 
                    shape[out_dim] > 1) {
                    try axes_buffer.append(out_dim);
                }
            }
            
            // If no dimensions were broadcasted, just clone the gradient
            if (axes_buffer.items.len == 0) {
                return try grad_out.clone();
            }
            
            // Sum along the broadcasted dimensions
            const keep_dims = true; // Need to keep dims for reshaping back
            
            // Use SumReducePlan to handle the reduction
            var sum_plan = SumReducePlan(Backend, T, null).init(allocator);
            defer sum_plan.deinit();
            
            // Set the runtime axes
            sum_plan.axes_buffer = try allocator.dupe(usize, axes_buffer.items);
            defer allocator.free(sum_plan.axes_buffer);
            
            // Perform the sum reduction
            var reduced = try sum_plan.run(grad_out);
            errdefer reduced.deinit();
            
            // If there was a rank expansion, we need to reshape to remove leading dims
            if (rank_diff > 0) {
                // Create a shape matching the input tensor
                var result = try Tensor.zeros(allocator, input.input.shape.dims, input.input.dtype, input.input.backend);
                errdefer result.deinit();
                
                // Copy the data - since reshape just copies the data
                const reduced_buf = ptrCastHelper([*]f32, reduced.buffer.data.ptr)[0..reduced.shape.elemCount()];
                const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
                
                // Need to handle potential reshaping carefully
                if (reduced.shape.elemCount() == result.shape.elemCount()) {
                    std.mem.copy(f32, result_buf, reduced_buf);
                } else {
                    // This is an error case - the reduced shape should match the input shape
                    reduced.deinit();
                    result.deinit();
                    return error.ShapeMismatch;
                }
                
                // Clean up and return
                reduced.deinit();
                return result;
            }
            
            return reduced;
        }

        pub fn run(self: *@This(), input: InputType) !Tensor {
            // Determine the target shape - either from comptime or runtime
            var shape: []const usize = undefined;
            
            if (comptime target_shape != null) {
                // Use comptime target shape
                shape = target_shape.?;
            } else if (input.target_shape_runtime != null) {
                // Use runtime target shape
                shape = input.target_shape_runtime.?;
                
                // Store the shape for backward pass
                if (self.target_shape_storage != null) {
                    self.base.allocator.free(self.target_shape_storage.?);
                }
                self.target_shape_storage = try self.base.allocator.dupe(usize, shape);
            } else {
                return error.MissingTargetShape;
            }
            
            // Verify that input tensor can be broadcast to target shape
            // 1. Input rank must be <= target rank
            if (input.input.shape.rank() > shape.len) {
                return OpError.ShapeMismatch;
            }
            
            // 2. For each dimension, either input dim is 1 or input dim == target dim
            const rank_diff = shape.len - input.input.shape.rank();
            for (0..input.input.shape.rank()) |i| {
                const target_idx = i + rank_diff;
                if (input.input.shape.dims[i] != 1 and input.input.shape.dims[i] != shape[target_idx]) {
                    return OpError.ShapeMismatch;
                }
            }
            
            // Check dtypes
            if (input.input.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }
            
            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, shape, input.input.dtype, input.input.backend);
            errdefer result.deinit();
            
            // Execute primitive
            Backend.expand(T, input.input, shape, &result);
            return result;
        }
    };
}

/// Reshape a tensor to a new shape
pub fn ReshapePlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("reshape")) @compileError("Backend must implement reshape primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { input: Tensor, shape_runtime: ?[]const usize = null };
        pub const op_type = OpType.reshape;
        pub const GradType = Tensor;
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,
        
        // Store shapes for backward pass
        input_shape_storage: ?[]usize = null,
        output_shape_storage: ?[]usize = null,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        pub fn deinit(self: *@This()) void {
            if (self.input_shape_storage != null) {
                self.base.allocator.free(self.input_shape_storage.?);
                self.input_shape_storage = null;
            }
            if (self.output_shape_storage != null) {
                self.base.allocator.free(self.output_shape_storage.?);
                self.output_shape_storage = null;
            }
        }
        
        /// Compute gradient for reshape operation: reshape gradient to input shape
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // For reshape, the gradient is just the gradient reshaped back to the input shape
            // We need the original input shape
            var input_shape: []const usize = undefined;
            
            if (self.input_shape_storage != null) {
                input_shape = self.input_shape_storage.?;
            } else {
                input_shape = input.input.shape.dims;
            }
            
            // Create tensor with input shape
            var result = try Tensor.zeros(allocator, input_shape, grad_out.dtype, grad_out.backend);
            errdefer result.deinit();
            
            // Copy the data - since reshape just reinterprets the underlying data
            const grad_buf = ptrCastHelper([*]f32, grad_out.buffer.data.ptr)[0..grad_out.shape.elemCount()];
            const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
            
            std.mem.copy(f32, result_buf, grad_buf);
            
            return result;
        }

        pub fn run(self: *@This(), input: InputType) !Tensor {
            // Determine the target shape - either from comptime or runtime
            var shape: []const usize = undefined;
            
            if (comptime shape != null) {
                // Use comptime shape
                shape = shape.?;
            } else if (input.shape_runtime != null) {
                // Use runtime shape
                shape = input.shape_runtime.?;
            } else {
                return error.MissingTargetShape;
            }
            
            // Store shapes for backward pass
            if (self.input_shape_storage != null) {
                self.base.allocator.free(self.input_shape_storage.?);
            }
            self.input_shape_storage = try self.base.allocator.dupe(usize, input.input.shape.dims);
            
            if (self.output_shape_storage != null) {
                self.base.allocator.free(self.output_shape_storage.?);
            }
            self.output_shape_storage = try self.base.allocator.dupe(usize, shape);
            
            // Check if reshape is valid (same number of elements)
            var new_size: usize = 1;
            for (shape) |dim| {
                new_size *= dim;
            }
            
            if (new_size != input.input.shape.elemCount()) {
                return OpError.ShapeMismatch;
            }
            
            // Check dtypes
            if (input.input.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }
            
            // Allocate result tensor with new shape
            var result = try Tensor.zeros(self.base.allocator, shape, input.input.dtype, input.input.backend);
            errdefer result.deinit();
            
            // Execute primitive - just copy the data
            Backend.reshape(T, input.input, &result);
            return result;
        }
    };
}

/// Batched matrix multiplication for 3D tensors
pub fn BatchedMatmulPlan(comptime Backend: type, comptime T: type, comptime BatchSize: ?usize, comptime M: ?usize, comptime K: ?usize, comptime N: ?usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("batched_matmul")) @compileError("Backend must implement batched_matmul primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.batched_matmul;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }
        
        /// Compute gradients for batched matmul:
        /// da[b,i,k] = sum_j(grad_out[b,i,j] * b[b,k,j])
        /// db[b,k,j] = sum_i(a[b,i,k] * grad_out[b,i,j])
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // We need to transpose the last two dimensions of tensors
            // For batched tensors, we need to handle each batch separately
            
            // Extract dimensions
            const batch_size = input.a.shape.dims[0];
            const m = input.a.shape.dims[1];
            const k = input.a.shape.dims[2]; // also input.b.shape.dims[1]
            const n = input.b.shape.dims[2];
            
            // Create da with shape [batch_size, m, k]
            const da_dims = [_]usize{ batch_size, m, k };
            var da = try Tensor.zeros(allocator, &da_dims, grad_out.dtype, grad_out.backend);
            errdefer da.deinit();
            
            // Create db with shape [batch_size, k, n]
            const db_dims = [_]usize{ batch_size, k, n };
            var db = try Tensor.zeros(allocator, &db_dims, grad_out.dtype, grad_out.backend);
            errdefer db.deinit();
            
            // 1. Compute db = a^T @ grad_out
            // We need to transpose a from [batch, m, k] to [batch, k, m]
            const a_transpose_dims = [_]usize{ batch_size, k, m };
            var a_transpose = try Tensor.zeros(allocator, &a_transpose_dims, input.a.dtype, input.a.backend);
            errdefer a_transpose.deinit();
            
            // Manually transpose the last two dimensions of a
            const a_buf = ptrCastHelper([*]f32, input.a.buffer.data.ptr)[0..input.a.shape.elemCount()];
            const a_t_buf = ptrCastHelper([*]f32, a_transpose.buffer.data.ptr)[0..a_transpose.shape.elemCount()];
            
            for (0..batch_size) |b| {
                for (0..m) |i| {
                    for (0..k) |j| {
                        const src_idx = b * (m * k) + i * k + j;
                        const dst_idx = b * (k * m) + j * m + i;
                        
                        if (src_idx < a_buf.len && dst_idx < a_t_buf.len) {
                            a_t_buf[dst_idx] = a_buf[src_idx];
                        }
                    }
                }
            }
            
            // Now do batched matmul: db = a_transpose @ grad_out
            Backend.batched_matmul(T, a_transpose, grad_out, &db);
            
            // 2. Compute da = grad_out @ b^T
            // We need to transpose b from [batch, k, n] to [batch, n, k]
            const b_transpose_dims = [_]usize{ batch_size, n, k };
            var b_transpose = try Tensor.zeros(allocator, &b_transpose_dims, input.b.dtype, input.b.backend);
            errdefer b_transpose.deinit();
            
            // Manually transpose the last two dimensions of b
            const b_buf = ptrCastHelper([*]f32, input.b.buffer.data.ptr)[0..input.b.shape.elemCount()];
            const b_t_buf = ptrCastHelper([*]f32, b_transpose.buffer.data.ptr)[0..b_transpose.shape.elemCount()];
            
            for (0..batch_size) |b| {
                for (0..k) |i| {
                    for (0..n) |j| {
                        const src_idx = b * (k * n) + i * n + j;
                        const dst_idx = b * (n * k) + j * k + i;
                        
                        if (src_idx < b_buf.len && dst_idx < b_t_buf.len) {
                            b_t_buf[dst_idx] = b_buf[src_idx];
                        }
                    }
                }
            }
            
            // Now do batched matmul: da = grad_out @ b_transpose
            Backend.batched_matmul(T, grad_out, b_transpose, &da);
            
            // Clean up temporaries
            a_transpose.deinit();
            b_transpose.deinit();
            
            return .{ .da = da, .db = db };
        }

        pub fn run(self: @This(), input: InputType) !Tensor {
            // Runtime validation for 3D shapes
            if (input.a.shape.rank() != 3 or input.b.shape.rank() != 3) {
                return OpError.DimensionMismatch;
            }
            
            // Check that batch dimensions match
            if (input.a.shape.dims[0] != input.b.shape.dims[0]) {
                return OpError.ShapeMismatch;
            }
            
            // Check dimension compatibility for batched matmul
            if (input.a.shape.dims[2] != input.b.shape.dims[1]) {
                return OpError.ShapeMismatch;
            }
            
            // Verify input shapes match comptime dimensions if provided
            const batch_size = input.a.shape.dims[0];
            const m = input.a.shape.dims[1];
            const k = input.a.shape.dims[2];
            const n = input.b.shape.dims[2];
            
            if (BatchSize != null and batch_size != BatchSize.?) return OpError.ShapeMismatch;
            if (M != null and m != M.?) return OpError.ShapeMismatch;
            if (K != null and k != K.?) return OpError.ShapeMismatch;
            if (N != null and n != N.?) return OpError.ShapeMismatch;
            
            // Check dtypes
            if (input.a.dtype != DType.f32 or input.b.dtype != DType.f32) {
                return OpError.UnsupportedDataType;
            }
            
            // Allocate result
            const result_dims = [_]usize{ batch_size, m, n };
            var result = try Tensor.zeros(self.base.allocator, &result_dims, DType.f32, input.a.backend);
            errdefer result.deinit();
            
            // Execute primitive
            Backend.batched_matmul(T, input.a, input.b, &result);
            return result;
        }
    };
}

/// Cross-entropy loss plan - computes loss between logits and target indices
pub fn CrossEntropyLossPlan(comptime Backend: type, comptime T: type, comptime shape_logits: ?[]const usize, comptime shape_targets: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (T != f32) @compileError("Only f32 supported for now");
    }
    
    return struct {
        pub const InputType = struct {
            logits: Tensor,  // shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            targets: Tensor, // shape [batch_size, seq_len] or [batch_size] containing class indices
        };
        
        pub const op_type = OpType.cross_entropy;
        pub const GradType = Tensor; // Only logits gets gradients: shape matches logits
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,
        
        // Subplans for the computation
        softmax_plan: SoftmaxPlan(Backend, T, null, null),
        sum_reduce_plan: SumReducePlan(Backend, T, null, false),
        subtract_plan: SubtractPlan(Backend, T, null),
        divide_plan: DividePlan(Backend, T, null),
        multiply_plan: MultiplyPlan(Backend, T, null),
        
        // For tracking intermediate results
        softmax_output: ?Tensor = null,
        target_indices: ?[]usize = null,
        batch_size: ?usize = null,
        
        pub fn init(allocator: Allocator) @This() {
            return .{
                .base = Base.init(allocator),
                .softmax_plan = SoftmaxPlan(Backend, T, null, null).init(allocator),
                .sum_reduce_plan = SumReducePlan(Backend, T, null, false).init(allocator),
                .subtract_plan = SubtractPlan(Backend, T, null).init(allocator),
                .divide_plan = DividePlan(Backend, T, null).init(allocator),
                .multiply_plan = MultiplyPlan(Backend, T, null).init(allocator),
            };
        }
        
        pub fn deinit(self: *@This()) void {
            self.sum_reduce_plan.deinit();
            
            if (self.softmax_output != null) {
                self.softmax_output.?.deinit();
                self.softmax_output = null;
            }
            
            if (self.target_indices != null) {
                self.base.allocator.free(self.target_indices.?);
                self.target_indices = null;
            }
        }
        
        /// Compute gradient for cross entropy loss
        /// dlogits = (softmax(logits) - one_hot(targets)) / batch_size
        pub fn gradient(self: *@This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            if (self.softmax_output == null || self.target_indices == null || self.batch_size == null) {
                return error.MissingIntermediates;
            }
            
            // Get batch size
            const batch_size = @as(f32, @floatFromInt(self.batch_size.?));
            
            // 1. Create a copy of softmax outputs (which will be our gradients)
            var dlogits = try self.softmax_output.?.clone();
            errdefer dlogits.deinit();
            
            // 2. Subtract 1.0 from the positions of target classes
            // This effectively implements softmax - one_hot
            const dlogits_buf = ptrCastHelper([*]f32, dlogits.buffer.data.ptr)[0..dlogits.shape.elemCount()];
            
            // For each target index, subtract 1.0 from the corresponding softmax output
            for (self.target_indices.?, 0..) |idx, i| {
                if (idx < dlogits_buf.len) {
                    dlogits_buf[idx] -= 1.0;
                }
            }
            
            // 3. Divide by batch size to normalize the gradient
            const batch_size_tensor = try Tensor.filled(allocator, &[_]usize{1}, dlogits.dtype, batch_size, dlogits.backend);
            defer batch_size_tensor.deinit();
            
            var result = try self.divide_plan.run(.{ .a = dlogits, .b = batch_size_tensor });
            defer dlogits.deinit();
            
            // 4. Multiply by the incoming gradient (which is a scalar for loss)
            const grad_out_value = try grad_out.getScalar(&[_]usize{0});
            
            if (grad_out_value != 1.0) {
                const grad_out_tensor = try Tensor.filled(allocator, &[_]usize{1}, result.dtype, grad_out_value, result.backend);
                defer grad_out_tensor.deinit();
                
                var scaled_result = try self.multiply_plan.run(.{ .a = result, .b = grad_out_tensor });
                result.deinit();
                result = scaled_result;
            }
            
            return result;
        }
        
        pub fn run(self: *@This(), input: InputType) !Tensor {
            const allocator = self.base.allocator;
            
            // Clean up previous intermediate results
            if (self.softmax_output != null) {
                self.softmax_output.?.deinit();
                self.softmax_output = null;
            }
            
            if (self.target_indices != null) {
                allocator.free(self.target_indices.?);
                self.target_indices = null;
            }
            
            // Determine shapes and dimensions
            // Logits shape can be either [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            const logits_rank = input.logits.shape.rank();
            const target_rank = input.targets.shape.rank();
            
            if (logits_rank < 2 || logits_rank > 3) {
                return OpError.DimensionMismatch;
            }
            
            if (target_rank != logits_rank - 1) {
                return OpError.DimensionMismatch;
            }
            
            // Extract dimensions
            var batch_size: usize = 0;
            var seq_len: usize = 1;
            var vocab_size: usize = 0;
            
            if (logits_rank == 3) {
                // Shape [batch_size, seq_len, vocab_size]
                batch_size = input.logits.shape.dims[0];
                seq_len = input.logits.shape.dims[1];
                vocab_size = input.logits.shape.dims[2];
                
                // Targets should be [batch_size, seq_len]
                if (input.targets.shape.rank() != 2 || 
                    input.targets.shape.dims[0] != batch_size || 
                    input.targets.shape.dims[1] != seq_len) {
                    return OpError.ShapeMismatch;
                }
            } else {
                // Shape [batch_size, vocab_size]
                batch_size = input.logits.shape.dims[0];
                vocab_size = input.logits.shape.dims[1];
                
                // Targets should be [batch_size]
                if (input.targets.shape.rank() != 1 || input.targets.shape.dims[0] != batch_size) {
                    return OpError.ShapeMismatch;
                }
            }
            
            // For storing target indices (positions in the flattened softmax outputs)
            var total_samples = batch_size * seq_len;
            self.target_indices = try allocator.alloc(usize, total_samples);
            errdefer allocator.free(self.target_indices.?);
            
            // Store batch size for gradient calculation
            self.batch_size = batch_size;
            
            // Apply softmax to logits
            // We need to reshape logits to [batch_size*seq_len, vocab_size] for softmax
            var reshaped_logits: Tensor = undefined;
            
            if (logits_rank == 3) {
                // Need to reshape from [batch_size, seq_len, vocab_size] to [batch_size*seq_len, vocab_size]
                const reshape_dims = [_]usize{ batch_size * seq_len, vocab_size };
                reshaped_logits = try Tensor.zeros(allocator, &reshape_dims, input.logits.dtype, input.logits.backend);
                errdefer reshaped_logits.deinit();
                
                // Copy data - we need to reorder
                const logits_buf = ptrCastHelper([*]f32, input.logits.buffer.data.ptr)[0..input.logits.shape.elemCount()];
                const reshaped_buf = ptrCastHelper([*]f32, reshaped_logits.buffer.data.ptr)[0..reshaped_logits.shape.elemCount()];
                
                for (0..batch_size) |b| {
                    for (0..seq_len) |s| {
                        for (0..vocab_size) |v| {
                            const src_idx = (b * seq_len + s) * vocab_size + v;
                            const dst_idx = (b * seq_len + s) * vocab_size + v;
                            
                            if (src_idx < logits_buf.len && dst_idx < reshaped_buf.len) {
                                reshaped_buf[dst_idx] = logits_buf[src_idx];
                            }
                        }
                    }
                }
            } else {
                // Already in the right shape [batch_size, vocab_size]
                reshaped_logits = try input.logits.clone();
                errdefer reshaped_logits.deinit();
            }
            
            // Apply softmax to each row (each sample)
            var softmax_result = try self.softmax_plan.run(reshaped_logits);
            errdefer softmax_result.deinit();
            
            // Store softmax result for gradient computation
            self.softmax_output = try softmax_result.clone();
            
            // Get target indices
            var targets_buf = ptrCastHelper([*]f32, input.targets.buffer.data.ptr)[0..input.targets.shape.elemCount()];
            
            // Compute loss by gathering the softmax probabilities of the target classes
            // And calculating the negative log of these probabilities
            var total_loss: f32 = 0.0;
            var valid_samples: usize = 0;
            
            // Compute indices into the softmax_result for each target
            for (0..total_samples) |i| {
                if (i < targets_buf.len) {
                    const class_idx = @as(usize, @intFromFloat(targets_buf[i]));
                    
                    if (class_idx < vocab_size) {
                        // Compute the index into the flattened softmax result
                        const idx = i * vocab_size + class_idx;
                        self.target_indices.?[i] = idx;
                        
                        // Get the softmax probability and compute negative log
                        const softmax_buf = ptrCastHelper([*]f32, softmax_result.buffer.data.ptr)[0..softmax_result.shape.elemCount()];
                        
                        if (idx < softmax_buf.len) {
                            const prob = softmax_buf[idx];
                            // Add small epsilon to avoid log(0)
                            const epsilon = 1e-10;
                            total_loss -= std.math.log(prob + epsilon);
                            valid_samples += 1;
                        }
                    }
                }
            }
            
            // Normalize by number of samples
            if (valid_samples > 0) {
                total_loss /= @as(f32, @floatFromInt(valid_samples));
            }
            
            // Create a scalar tensor with the loss value
            var result = try Tensor.zeros(allocator, &[_]usize{1}, input.logits.dtype, input.logits.backend);
            errdefer result.deinit();
            
            try result.setScalar(&[_]usize{0}, total_loss);
            
            // Clean up
            reshaped_logits.deinit();
            softmax_result.deinit();
            
            return result;
        }
    };
}

/// Layer normalization plan - normalizes inputs along the last axis
pub fn LayerNormPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize, comptime epsilon: f32) type {
    // Comptime validation
    comptime {
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        pub const InputType = struct { 
            input: Tensor, 
            gamma: Tensor, // scale parameter
            beta: Tensor,  // shift parameter
        };
        
        pub const op_type = OpType.layer_norm;
        pub const GradType = struct { dinput: Tensor, dgamma: Tensor, dbeta: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,
        
        // Subplans for the computation
        sum_reduce_plan: SumReducePlan(Backend, T, null, true),
        pow_plan: PowPlan(Backend, T, null),
        divide_plan: DividePlan(Backend, T, null),
        subtract_plan: SubtractPlan(Backend, T, null),
        multiply_plan: MultiplyPlan(Backend, T, null),
        add_plan: AddPlan(Backend, T, null),
        
        // For tracking intermediate results in backward
        mean: ?Tensor = null,
        var_eps: ?Tensor = null,
        normalized: ?Tensor = null,

        pub fn init(allocator: Allocator) @This() {
            return .{
                .base = Base.init(allocator),
                .sum_reduce_plan = SumReducePlan(Backend, T, null).init(allocator),
                .pow_plan = PowPlan(Backend, T, null).init(allocator),
                .divide_plan = DividePlan(Backend, T, null).init(allocator),
                .subtract_plan = SubtractPlan(Backend, T, null).init(allocator),
                .multiply_plan = MultiplyPlan(Backend, T, null).init(allocator),
                .add_plan = AddPlan(Backend, T, null).init(allocator),
            };
        }
        
        pub fn deinit(self: *@This()) void {
            self.sum_reduce_plan.deinit();
            
            if (self.mean != null) {
                self.mean.?.deinit();
                self.mean = null;
            }
            
            if (self.var_eps != null) {
                self.var_eps.?.deinit();
                self.var_eps = null;
            }
            
            if (self.normalized != null) {
                self.normalized.?.deinit();
                self.normalized = null;
            }
        }
        
        /// Compute gradient for layer normalization
        pub fn gradient(self: *@This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // Ensure we have intermediate values from the forward pass
            if (self.mean == null or self.var_eps == null or self.normalized == null) {
                return error.MissingIntermediates;
            }
            
            // Compute gradient for beta (shift): sum over all except last dim
            const last_dim = input.input.shape.rank() - 1;
            var axes = try allocator.alloc(usize, last_dim);
            defer allocator.free(axes);
            
            for (0..last_dim) |i| {
                axes[i] = i;
            }
            
            // Set the runtime axes for sum reduction
            self.sum_reduce_plan.axes_buffer = try allocator.dupe(usize, axes);
            defer allocator.free(self.sum_reduce_plan.axes_buffer);
            
            // dbeta = sum(grad_out, axes=[0...n-2])
            const dbeta = try self.sum_reduce_plan.run(grad_out);
            errdefer dbeta.deinit();
            
            // dgamma = sum(grad_out * normalized, axes=[0...n-2])
            const grad_norm = try self.multiply_plan.run(.{ .a = grad_out, .b = self.normalized.? });
            errdefer grad_norm.deinit();
            
            const dgamma = try self.sum_reduce_plan.run(grad_norm);
            errdefer dgamma.deinit();
            
            // Clean up temp
            grad_norm.deinit();
            
            // Compute gradient for input - more complex
            // dinput = (gamma / sqrt(var_eps)) * (grad_out - 1/N * sum(grad_out) - normalized * 1/N * sum(grad_out * normalized))
            
            // First, compute scale = gamma / sqrt(var_eps)
            const scale = try self.divide_plan.run(.{ .a = input.gamma, .b = self.var_eps.? });
            errdefer scale.deinit();
            
            // We need to broadcast scale to match input dimensions
            var scale_expanded = try scale.clone(); // Start with scale shape
            errdefer scale_expanded.deinit();
            
            // Now we need to expand scale to the full input shape
            // We'll do this by constructing a tensor with ones in all dimensions except the last
            var expand_shape = try allocator.dupe(usize, input.input.shape.dims);
            defer allocator.free(expand_shape);
            
            // Compute a mask for input mean: sum(grad_out) / N
            const mean_grad = try self.sum_reduce_plan.run(grad_out);
            errdefer mean_grad.deinit();
            
            // N is the product of all dimensions except the last
            var n: usize = 1;
            for (0..last_dim) |i| {
                n *= input.input.shape.dims[i];
            }
            
            // Create 1/N tensor
            const recip_n = try Tensor.filled(allocator, &[_]usize{1}, grad_out.dtype, 1.0 / @as(f32, @floatFromInt(n)), grad_out.backend);
            defer recip_n.deinit();
            
            // mean_grad = sum(grad_out) / N
            const norm_mean_grad = try self.multiply_plan.run(.{ .a = mean_grad, .b = recip_n });
            errdefer norm_mean_grad.deinit();
            
            // Expand norm_mean_grad to match input shape for broadcasting
            var expanded_mean_grad = try norm_mean_grad.clone();
            errdefer expanded_mean_grad.deinit();
            
            // Compute weighted_mean = sum(grad_out * normalized) / N
            const weighted_grad = try self.multiply_plan.run(.{ .a = grad_out, .b = self.normalized.? });
            errdefer weighted_grad.deinit();
            
            const sum_weighted_grad = try self.sum_reduce_plan.run(weighted_grad);
            errdefer sum_weighted_grad.deinit();
            
            const norm_weighted_grad = try self.multiply_plan.run(.{ .a = sum_weighted_grad, .b = recip_n });
            errdefer norm_weighted_grad.deinit();
            
            // Expand norm_weighted_grad for broadcasting
            var expanded_weighted_grad = try norm_weighted_grad.clone();
            errdefer expanded_weighted_grad.deinit();
            
            // Compute norm_term = normalized * norm_weighted_grad (expanded)
            const norm_term = try self.multiply_plan.run(.{ .a = self.normalized.?, .b = expanded_weighted_grad });
            errdefer norm_term.deinit();
            
            // Compute adjustment = grad_out - norm_mean_grad (expanded) - norm_term
            const temp1 = try self.subtract_plan.run(.{ .a = grad_out, .b = expanded_mean_grad });
            errdefer temp1.deinit();
            
            const adjustment = try self.subtract_plan.run(.{ .a = temp1, .b = norm_term });
            errdefer adjustment.deinit();
            
            // Compute dinput = scale (expanded) * adjustment
            var dinput = try self.multiply_plan.run(.{ .a = scale_expanded, .b = adjustment });
            errdefer dinput.deinit();
            
            // Clean up temps
            scale.deinit();
            scale_expanded.deinit();
            mean_grad.deinit();
            norm_mean_grad.deinit();
            expanded_mean_grad.deinit();
            weighted_grad.deinit();
            sum_weighted_grad.deinit();
            norm_weighted_grad.deinit();
            expanded_weighted_grad.deinit();
            norm_term.deinit();
            temp1.deinit();
            adjustment.deinit();
            
            return .{ .dinput = dinput, .dgamma = dgamma, .dbeta = dbeta };
        }

        pub fn run(self: *@This(), input: InputType) !Tensor {
            const allocator = self.base.allocator;
            
            // Verify input shapes
            const last_dim = input.input.shape.dims[input.input.shape.rank() - 1];
            
            if (input.gamma.shape.rank() != 1 or input.beta.shape.rank() != 1) {
                return OpError.ShapeMismatch;
            }
            
            if (input.gamma.shape.dims[0] != last_dim or input.beta.shape.dims[0] != last_dim) {
                return OpError.ShapeMismatch;
            }
            
            // Clean up any previous intermediate results
            if (self.mean != null) {
                self.mean.?.deinit();
                self.mean = null;
            }
            
            if (self.var_eps != null) {
                self.var_eps.?.deinit();
                self.var_eps = null;
            }
            
            if (self.normalized != null) {
                self.normalized.?.deinit();
                self.normalized = null;
            }
            
            // LayerNorm implementation:
            // 1. Compute mean along the last dimension
            const last_dim_index = input.input.shape.rank() - 1;
            var axes = try allocator.alloc(usize, 1);
            defer allocator.free(axes);
            axes[0] = last_dim_index;
            
            // Set the runtime axes for sum reduction
            self.sum_reduce_plan.axes_buffer = try allocator.dupe(usize, axes);
            defer allocator.free(self.sum_reduce_plan.axes_buffer);
            
            // Compute mean along last dim: mean = input.mean(dim=-1, keepdim=True)
            var mean = try self.sum_reduce_plan.run(input.input);
            errdefer mean.deinit();
            
            // Divide by dimension size to get mean
            const dim_size = try Tensor.filled(allocator, mean.shape.dims, mean.dtype, 1.0 / @as(f32, @floatFromInt(last_dim)), mean.backend);
            defer dim_size.deinit();
            
            mean = try self.multiply_plan.run(.{ .a = mean, .b = dim_size });
            errdefer mean.deinit();
            
            // 2. Compute variance: var = ((input - mean)^2).mean(dim=-1, keepdim=True)
            const centered = try self.subtract_plan.run(.{ .a = input.input, .b = mean });
            errdefer centered.deinit();
            
            // Square the centered values: centered^2
            const two = try Tensor.filled(allocator, &[_]usize{1}, input.input.dtype, 2.0, input.input.backend);
            defer two.deinit();
            
            const squared = try self.pow_plan.run(.{ .base = centered, .exponent = two });
            errdefer squared.deinit();
            
            // Compute mean of squared values
            const var = try self.sum_reduce_plan.run(squared);
            errdefer var.deinit();
            
            const var_mean = try self.multiply_plan.run(.{ .a = var, .b = dim_size });
            errdefer var_mean.deinit();
            
            // 3. Add epsilon for numerical stability: var_eps = var + epsilon
            const eps_tensor = try Tensor.filled(allocator, var_mean.shape.dims, var_mean.dtype, epsilon, var_mean.backend);
            defer eps_tensor.deinit();
            
            const var_eps = try self.add_plan.run(.{ .a = var_mean, .b = eps_tensor });
            errdefer var_eps.deinit();
            
            // 4. Compute sqrt(var_eps)
            const half = try Tensor.filled(allocator, &[_]usize{1}, var_eps.dtype, 0.5, var_eps.backend);
            defer half.deinit();
            
            const std_dev = try self.pow_plan.run(.{ .base = var_eps, .exponent = half });
            errdefer std_dev.deinit();
            
            // 5. Normalize: (input - mean) / sqrt(var_eps)
            const normalized = try self.divide_plan.run(.{ .a = centered, .b = std_dev });
            errdefer normalized.deinit();
            
            // 6. Scale and shift: gamma * normalized + beta
            const scaled = try self.multiply_plan.run(.{ .a = normalized, .b = input.gamma });
            errdefer scaled.deinit();
            
            const result = try self.add_plan.run(.{ .a = scaled, .b = input.beta });
            errdefer result.deinit();
            
            // Store intermediates for backward pass
            self.mean = mean;
            self.var_eps = var_eps;
            self.normalized = try normalized.clone();
            
            // Clean up temporary tensors
            centered.deinit();
            squared.deinit();
            var.deinit();
            var_mean.deinit();
            std_dev.deinit();
            scaled.deinit();
            
            return result;
        }
    };
}

// --- Tests ---

test "element-wise operations" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create two 2x2 tensors
    var a = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a.deinit();
    
    var b = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer b.deinit();
    
    // Set values in tensor a: [[1, 2], [3, 4]]
    try a.setScalar(&[_]usize{0, 0}, 1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{1, 0}, 3.0);
    try a.setScalar(&[_]usize{1, 1}, 4.0);
    
    // Set values in tensor b: [[5, 6], [7, 8]]
    try b.setScalar(&[_]usize{0, 0}, 5.0);
    try b.setScalar(&[_]usize{0, 1}, 6.0);
    try b.setScalar(&[_]usize{1, 0}, 7.0);
    try b.setScalar(&[_]usize{1, 1}, 8.0);
    
    // Test addition using the legacy interface
    {
        var result = try add(allocator, a, b);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 8.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 10.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 12.0), try result.getScalar(&[_]usize{1, 1}));
    }
    
    // Test addition using legacy interface
    {
        var result = try add(allocator, a, b);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 8.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 10.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 12.0), try result.getScalar(&[_]usize{1, 1}));
    }
    
    // Test subtraction
    {
        var result = try subtract(allocator, a, b);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, -4.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, -4.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, -4.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, -4.0), try result.getScalar(&[_]usize{1, 1}));
    }
    
    // Test element-wise multiplication
    {
        var result = try multiply(allocator, a, b);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 5.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 12.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 21.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 32.0), try result.getScalar(&[_]usize{1, 1}));
    }
}

test "matrix multiplication" {
    const allocator = std.testing.allocator;
    var a_dims = [_]usize{ 2, 3 };
    var b_dims = [_]usize{ 3, 2 };
    
    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    var a = try Tensor.zeros(allocator, &a_dims, .f32, .cpu);
    defer a.deinit();
    
    try a.setScalar(&[_]usize{0, 0}, 1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{0, 2}, 3.0);
    try a.setScalar(&[_]usize{1, 0}, 4.0);
    try a.setScalar(&[_]usize{1, 1}, 5.0);
    try a.setScalar(&[_]usize{1, 2}, 6.0);
    
    // Create a 3x2 matrix: [[7, 8], [9, 10], [11, 12]]
    var b = try Tensor.zeros(allocator, &b_dims, .f32, .cpu);
    defer b.deinit();
    
    try b.setScalar(&[_]usize{0, 0}, 7.0);
    try b.setScalar(&[_]usize{0, 1}, 8.0);
    try b.setScalar(&[_]usize{1, 0}, 9.0);
    try b.setScalar(&[_]usize{1, 1}, 10.0);
    try b.setScalar(&[_]usize{2, 0}, 11.0);
    try b.setScalar(&[_]usize{2, 1}, 12.0);
    
    // Test matrix multiplication using the legacy interface
    {
        var result = try matmul(allocator, a, b);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 58.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 64.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 139.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 154.0), try result.getScalar(&[_]usize{1, 1}));
    }
    
    // Test using legacy interface
    {
        var result = try matmul(allocator, a, b);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 58.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 64.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 139.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 154.0), try result.getScalar(&[_]usize{1, 1}));
    }
}

test "activation functions" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create a 2x2 tensor: [[-1, 2], [0, 3]]
    var a = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a.deinit();
    
    try a.setScalar(&[_]usize{0, 0}, -1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{1, 0}, 0.0);
    try a.setScalar(&[_]usize{1, 1}, 3.0);
    
    // Test ReLU using the legacy interface
    {
        var result = try relu(allocator, a);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{1, 1}));
    }
    
    // Test ReLU using legacy interface
    {
        var result = try relu(allocator, a);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{1, 1}));
    }
    
    // Create a 2x2 tensor: [[1, 2], [3, 4]]
    var b = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer b.deinit();
    
    try b.setScalar(&[_]usize{0, 0}, 1.0);
    try b.setScalar(&[_]usize{0, 1}, 2.0);
    try b.setScalar(&[_]usize{1, 0}, 3.0);
    try b.setScalar(&[_]usize{1, 1}, 4.0);
    
    // Test Softmax using legacy interface
    {
        var result = try softmax(allocator, b);
        defer result.deinit();
        
        // Check that softmax outputs sum to 1 for each row
        const first_row_0 = try result.getScalar(&[_]usize{0, 0});
        const first_row_1 = try result.getScalar(&[_]usize{0, 1});
        const second_row_0 = try result.getScalar(&[_]usize{1, 0});
        const second_row_1 = try result.getScalar(&[_]usize{1, 1});
        
        try std.testing.expectApproxEqRel(@as(f32, 1.0), first_row_0 + first_row_1, 0.0001);
        try std.testing.expectApproxEqRel(@as(f32, 1.0), second_row_0 + second_row_1, 0.0001);
        
        // Check that second element is larger than first in each row
        try std.testing.expect(first_row_1 > first_row_0);
        try std.testing.expect(second_row_1 > second_row_0);
    }
}

test "transpose" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 3 };
    
    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    var a = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a.deinit();
    
    try a.setScalar(&[_]usize{0, 0}, 1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{0, 2}, 3.0);
    try a.setScalar(&[_]usize{1, 0}, 4.0);
    try a.setScalar(&[_]usize{1, 1}, 5.0);
    try a.setScalar(&[_]usize{1, 2}, 6.0);
    
    // Test Transpose using the legacy interface
    {
        var result = try transpose(allocator, a);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 1.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 4.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 5.0), try result.getScalar(&[_]usize{1, 1}));
        try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{2, 0}));
        try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{2, 1}));
    }
    
    // Test using legacy interface
    {
        var result = try transpose(allocator, a);
        defer result.deinit();
        
        try std.testing.expectEqual(@as(f32, 1.0), try result.getScalar(&[_]usize{0, 0}));
        try std.testing.expectEqual(@as(f32, 4.0), try result.getScalar(&[_]usize{0, 1}));
        try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{1, 0}));
        try std.testing.expectEqual(@as(f32, 5.0), try result.getScalar(&[_]usize{1, 1}));
        try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{2, 0}));
        try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{2, 1}));
    }
}