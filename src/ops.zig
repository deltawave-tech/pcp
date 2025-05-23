const std = @import("std");
const Allocator = std.mem.Allocator;

const tensor = @import("tensor.zig");
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
};

/// --- Primitive Operations ---
/// These are the building blocks for comptime plans, implemented by backends
pub fn Primitives(comptime T: type) type {
    const Tensor = tensor.Tensor(T);

    return struct {
        /// Add two tensors element-wise
        pub fn add(a: Tensor, b: Tensor, result: *Tensor) void {
            // Move type checking to comptime
            comptime {
                if (T != f32) @compileError("Only f32 supported for now");
            }

            const a_buf = a.buffer.data;
            const b_buf = b.buffer.data;
            const result_buf = result.buffer.data;

            for (a_buf, b_buf, 0..) |a_val, b_val, i| {
                result_buf[i] = a_val + b_val;
            }
        }

        /// Subtract two tensors element-wise
        pub fn subtract(a: Tensor, b: Tensor, result: *Tensor) void {
            // Move type checking to comptime
            comptime {
                if (T != f32) @compileError("Only f32 supported for now");
            }

            const a_buf = a.buffer.data;
            const b_buf = b.buffer.data;
            const result_buf = result.buffer.data;

            for (a_buf, b_buf, 0..) |a_val, b_val, i| {
                result_buf[i] = a_val - b_val;
            }
        }

        /// Multiply two tensors element-wise
        pub fn multiply(a: Tensor, b: Tensor, result: *Tensor) void {
            // Move type checking to comptime
            comptime {
                if (T != f32) @compileError("Only f32 supported for now");
            }

            const a_buf = a.buffer.data;
            const b_buf = b.buffer.data;
            const result_buf = result.buffer.data;

            for (a_buf, b_buf, 0..) |a_val, b_val, i| {
                result_buf[i] = a_val * b_val;
            }
        }

        /// Divide two tensors element-wise
        pub fn divide(a: Tensor, b: Tensor, result: *Tensor) void {
            // Move type checking to comptime
            comptime {
                if (T != f32) @compileError("Only f32 supported for now");
            }

            const a_buf = a.buffer.data;
            const b_buf = b.buffer.data;
            const result_buf = result.buffer.data;

            for (a_buf, b_buf, 0..) |a_val, b_val, i| {
                result_buf[i] = a_val / b_val;
            }
        }

        /// Matrix multiplication for 2D tensors
        pub fn matmul(a: Tensor, b: Tensor, result: *Tensor) void {
            // Move type checking to comptime
            comptime {
                if (T != f32) @compileError("Only f32 supported for now");
            }

            const a_buf = a.buffer.data;
            const b_buf = b.buffer.data;
            const result_buf = result.buffer.data;

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
        pub fn relu(a: Tensor, result: *Tensor) void {
            // Move type checking to comptime
            comptime {
                if (T != f32) @compileError("Only f32 supported for now");
            }

            const a_buf = a.buffer.data;
            const result_buf = result.buffer.data;

            for (a_buf, 0..) |a_val, i| {
                result_buf[i] = if (a_val > 0) a_val else 0;
            }
        }

        /// Apply softmax to 2D tensor (on last dimension)
        pub fn softmax(a: Tensor, result: *Tensor) void {
            // Move type checking to comptime
            comptime {
                if (T != f32) @compileError("Only f32 supported for now");
            }

            const a_buf = a.buffer.data;
            const result_buf = result.buffer.data;

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
        pub fn transpose(a: Tensor, result: *Tensor) void {
            // Move type checking to comptime
            comptime {
                if (T != f32) @compileError("Only f32 supported for now");
            }

            const a_buf = a.buffer.data;
            const result_buf = result.buffer.data;

            const rows = a.shape.dims[0];
            const cols = a.shape.dims[1];

            for (0..rows) |i| {
                for (0..cols) |j| {
                    result_buf[j * rows + i] = a_buf[i * cols + j];
                }
            }
        }

        /// Embedding lookup - fetch embeddings based on indices
        pub fn embedding_lookup(params: Tensor, indices: Tensor, result: *Tensor) void {
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

            const params_buf = params.buffer.data;
            const indices_buf = indices.buffer.data;
            const result_buf = result.buffer.data;

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
}

/// --- Plan Definition ---
/// Generic type for comptime-generated operation plans
pub fn PlanType(comptime Backend: type, comptime InputType: type, comptime OutputType: type) type {
    _ = Backend; // Used to match interface, actual backend implementation comes from specific plan
    return struct {
        const Self = @This();
        allocator: Allocator, // Make allocator public

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
pub fn CpuBackend(comptime T: type) type {
    return struct {
        pub fn hasPrimitive(comptime name: []const u8) bool {
            return std.mem.eql(u8, name, "add") or
                std.mem.eql(u8, name, "subtract") or
                std.mem.eql(u8, name, "multiply") or
                std.mem.eql(u8, name, "divide") or
                std.mem.eql(u8, name, "matmul") or
                std.mem.eql(u8, name, "relu") or
                std.mem.eql(u8, name, "softmax") or
                std.mem.eql(u8, name, "transpose") or
                std.mem.eql(u8, name, "embedding_lookup");
        }

        // Include default primitive implementations
        pub usingnamespace Primitives(T);
    };
}

/// --- Operation Plans ---
/// Add two tensors element-wise
pub fn AddPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("add")) @compileError("Backend must implement add primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    const Tensor = tensor.Tensor(T);

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

            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.a.shape.dims, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.add(input.a, input.b, &result);
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

    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

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
            const negative_one = try Tensor.filled(allocator, grad_out.shape.dims, -1.0, grad_out.backend);
            errdefer negative_one.deinit();

            var db = try legacy_api.multiply(allocator, grad_out, negative_one);
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

            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.a.shape.dims, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.subtract(input.a, input.b, &result);
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

    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

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
            var da = try legacy_api.multiply(allocator, grad_out, input.b);
            errdefer da.deinit();

            // Input b gradient is element-wise product of grad_out and a
            var db = try legacy_api.multiply(allocator, grad_out, input.a);
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

            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.a.shape.dims, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.multiply(input.a, input.b, &result);
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

    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

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
            var da = try legacy_api.divide(allocator, grad_out, input.b);
            errdefer da.deinit();

            // Input b gradient is more complex: -grad_out * a / (b * b)
            // First calculate b^2
            var b_squared = try legacy_api.multiply(allocator, input.b, input.b);
            errdefer b_squared.deinit();

            // Calculate a / b^2
            var a_div_b_squared = try legacy_api.divide(allocator, input.a, b_squared);
            errdefer a_div_b_squared.deinit();

            // Multiply by grad_out
            var temp = try legacy_api.multiply(allocator, grad_out, a_div_b_squared);
            errdefer temp.deinit();

            // Negate the result
            const negative_one = try Tensor.filled(allocator, temp.shape.dims, -1.0, temp.backend);
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

            // Allocate result tensor
            var result = try Tensor.zeros(self.base.allocator, input.a.shape.dims, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.divide(input.a, input.b, &result);
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

    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

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
            const b_transpose = try legacy_api.transpose(allocator, input.b);
            errdefer b_transpose.deinit();

            // Gradient for a is grad_out @ b^T
            var da = try legacy_api.matmul(allocator, grad_out, b_transpose);
            errdefer da.deinit();

            // Compute a transpose
            const a_transpose = try legacy_api.transpose(allocator, input.a);
            errdefer a_transpose.deinit();

            // Gradient for b is a^T @ grad_out
            var db = try legacy_api.matmul(allocator, a_transpose, grad_out);
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

            // Allocate result
            const result_dims = [_]usize{ m, p };
            var result = try Tensor.zeros(self.base.allocator, &result_dims, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.matmul(input.a, input.b, &result);
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

    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

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

            const input_buf = input.buffer.data;
            const mask_buf = mask.buffer.data;

            for (input_buf, 0..) |val, i| {
                mask_buf[i] = if (val > 0) 1.0 else 0.0;
            }

            // Gradient is element-wise product of upstream gradient and mask
            var result = try legacy_api.multiply(allocator, grad_out, mask);
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

            var result = try Tensor.zeros(self.base.allocator, input.shape.dims, input.backend);
            errdefer result.deinit();

            Backend.relu(input, &result);
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

    const Tensor = tensor.Tensor(T);

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

            var result = try Tensor.zeros(self.base.allocator, input.shape.dims, input.backend);
            errdefer result.deinit();

            Backend.softmax(input, &result);
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

    const Tensor = tensor.Tensor(T);

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

            // Result dimensions are swapped
            const result_dims = [_]usize{ input.shape.dims[1], input.shape.dims[0] };
            var result = try Tensor.zeros(self.base.allocator, &result_dims, input.backend);
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

    const Tensor = tensor.Tensor(T);

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

            // Create output tensor
            const result_dims = [_]usize{ batch_size, seq_len, input.params.shape.dims[1] };
            var result = try Tensor.zeros(self.base.allocator, &result_dims, input.params.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.embedding_lookup(T, input.params, input.indices, &result);
            return result;
        }
    };
}

pub fn LegacyApi(comptime T: type) type {
    const Tensor = tensor.Tensor(T);

    return struct {
        /// --- Legacy Interface for Backward Compatibility ---
        /// These functions provide backward compatibility with the existing API
        /// Add two tensors element-wise (legacy interface)
        pub fn add(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
            // For the legacy interface, we need to perform the checks at runtime
            // since we can't use runtime values for comptime parameters

            // Check if shapes match
            if (!a.shape.eql(b.shape)) {
                return OpError.ShapeMismatch;
            }

            // Create result tensor
            var result = try Tensor.zeros(allocator, a.shape.dims, a.backend);
            errdefer result.deinit();

            // Perform the operation using the primitive function
            CpuBackend(T).add(a, b, &result);

            return result;
        }

        /// Subtract two tensors element-wise (legacy interface)
        pub fn subtract(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
            // For the legacy interface, we need to perform the checks at runtime

            // Check if shapes match
            if (!a.shape.eql(b.shape)) {
                return OpError.ShapeMismatch;
            }

            // Create result tensor
            var result = try Tensor.zeros(allocator, a.shape.dims, a.backend);
            errdefer result.deinit();

            CpuBackend(T).subtract(a, b, &result);
            return result;
        }

        /// Multiply two tensors element-wise (legacy interface)
        pub fn multiply(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
            // For the legacy interface, we need to perform the checks at runtime

            // Check if shapes match
            if (!a.shape.eql(b.shape)) {
                return OpError.ShapeMismatch;
            }

            // Create result tensor
            var result = try Tensor.zeros(allocator, a.shape.dims, a.backend);
            errdefer result.deinit();

            // Perform the operation using the primitive function
            CpuBackend(T).multiply(a, b, &result);
            return result;
        }

        /// Divide two tensors element-wise (legacy interface)
        pub fn divide(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
            // For the legacy interface, we need to perform the checks at runtime

            // Check if shapes match
            if (!a.shape.eql(b.shape)) {
                return OpError.ShapeMismatch;
            }

            // Create result tensor
            var result = try Tensor.zeros(allocator, a.shape.dims, a.backend);
            errdefer result.deinit();

            CpuBackend(T).divide(a, b, &result);

            return result;
        }

        /// Matrix multiplication (legacy interface)
        pub fn matmul(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
            // For the legacy interface, we need to perform the checks at runtime

            // Check dimensions for matrix multiplication
            if (a.shape.rank() != 2 or b.shape.rank() != 2) {
                return OpError.DimensionMismatch;
            }

            if (a.shape.dims[1] != b.shape.dims[0]) {
                return OpError.ShapeMismatch;
            }

            // Result dimensions: [a.dims[0], b.dims[1]]
            const result_dims = [_]usize{ a.shape.dims[0], b.shape.dims[1] };
            var result = try Tensor.zeros(allocator, &result_dims, a.backend);
            errdefer result.deinit();

            CpuBackend(T).matmul(a, b, &result);

            return result;
        }

        /// Apply ReLU activation function element-wise (legacy interface)
        pub fn relu(allocator: Allocator, a: Tensor) !Tensor {
            // For the legacy interface, we need to perform the checks at runtime

            // Create result tensor
            var result = try Tensor.zeros(allocator, a.shape.dims, a.backend);
            errdefer result.deinit();

            // Perform the operation using the primitive function
            CpuBackend(T).relu(a, &result);
            return result;
        }

        /// Softmax activation (legacy interface)
        pub fn softmax(allocator: Allocator, a: Tensor) !Tensor {
            // For the legacy interface, we need to perform the checks at runtime

            if (a.shape.rank() != 2) {
                return OpError.DimensionMismatch;
            }

            // Create result tensor
            var result = try Tensor.zeros(allocator, a.shape.dims, a.backend);
            errdefer result.deinit();

            // Perform the operation using the primitive function
            CpuBackend(T).softmax(a, &result);
            return result;
        }

        /// Transpose a 2D tensor (legacy interface)
        pub fn transpose(allocator: Allocator, a: Tensor) !Tensor {
            // For the legacy interface, we need to perform the checks at runtime

            if (a.shape.rank() != 2) {
                return OpError.DimensionMismatch;
            }

            // Result dimensions are swapped
            const result_dims = [_]usize{ a.shape.dims[1], a.shape.dims[0] };
            var result = try Tensor.zeros(allocator, &result_dims, a.backend);
            errdefer result.deinit();

            // Perform the operation using the primitive function
            CpuBackend(T).transpose(a, &result);
            return result;
        }
    };
}

// --- Tests ---

test "element-wise operations" {
    const T = f32;
    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };

    // Create two 2x2 tensors
    var a = try Tensor.zeros(allocator, &dims, .cpu);
    defer a.deinit();

    var b = try Tensor.zeros(allocator, &dims, .cpu);
    defer b.deinit();

    // Set values in tensor a: [[1, 2], [3, 4]]
    try a.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 3.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 4.0);

    // Set values in tensor b: [[5, 6], [7, 8]]
    try b.setScalar(&[_]usize{ 0, 0 }, 5.0);
    try b.setScalar(&[_]usize{ 0, 1 }, 6.0);
    try b.setScalar(&[_]usize{ 1, 0 }, 7.0);
    try b.setScalar(&[_]usize{ 1, 1 }, 8.0);

    // Test addition using the legacy interface
    {
        var result = try legacy_api.add(allocator, a, b);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 8.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 10.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 12.0), try result.getScalar(&[_]usize{ 1, 1 }));
    }

    // Test addition using legacy interface
    {
        var result = try legacy_api.add(allocator, a, b);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 8.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 10.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 12.0), try result.getScalar(&[_]usize{ 1, 1 }));
    }

    // Test subtraction
    {
        var result = try legacy_api.subtract(allocator, a, b);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, -4.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, -4.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, -4.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, -4.0), try result.getScalar(&[_]usize{ 1, 1 }));
    }

    // Test element-wise multiplication
    {
        var result = try legacy_api.multiply(allocator, a, b);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 5.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 12.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 21.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 32.0), try result.getScalar(&[_]usize{ 1, 1 }));
    }
}

test "matrix multiplication" {
    const T = f32;
    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

    const allocator = std.testing.allocator;
    var a_dims = [_]usize{ 2, 3 };
    var b_dims = [_]usize{ 3, 2 };

    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    var a = try Tensor.zeros(allocator, &a_dims, .cpu);
    defer a.deinit();

    try a.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 0, 2 }, 3.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 4.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 5.0);
    try a.setScalar(&[_]usize{ 1, 2 }, 6.0);

    // Create a 3x2 matrix: [[7, 8], [9, 10], [11, 12]]
    var b = try Tensor.zeros(allocator, &b_dims, .cpu);
    defer b.deinit();

    try b.setScalar(&[_]usize{ 0, 0 }, 7.0);
    try b.setScalar(&[_]usize{ 0, 1 }, 8.0);
    try b.setScalar(&[_]usize{ 1, 0 }, 9.0);
    try b.setScalar(&[_]usize{ 1, 1 }, 10.0);
    try b.setScalar(&[_]usize{ 2, 0 }, 11.0);
    try b.setScalar(&[_]usize{ 2, 1 }, 12.0);

    // Test matrix multiplication using the legacy interface
    {
        var result = try legacy_api.matmul(allocator, a, b);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 58.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 64.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 139.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 154.0), try result.getScalar(&[_]usize{ 1, 1 }));
    }

    // Test using legacy interface
    {
        var result = try legacy_api.matmul(allocator, a, b);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 58.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 64.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 139.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 154.0), try result.getScalar(&[_]usize{ 1, 1 }));
    }
}

test "activation functions" {
    const T = f32;
    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };

    // Create a 2x2 tensor: [[-1, 2], [0, 3]]
    var a = try Tensor.zeros(allocator, &dims, .cpu);
    defer a.deinit();

    try a.setScalar(&[_]usize{ 0, 0 }, -1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 0.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 3.0);

    // Test ReLU using the legacy interface
    {
        var result = try legacy_api.relu(allocator, a);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{ 1, 1 }));
    }

    // Test ReLU using legacy interface
    {
        var result = try legacy_api.relu(allocator, a);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{ 1, 1 }));
    }

    // Create a 2x2 tensor: [[1, 2], [3, 4]]
    var b = try Tensor.zeros(allocator, &dims, .cpu);
    defer b.deinit();

    try b.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try b.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try b.setScalar(&[_]usize{ 1, 0 }, 3.0);
    try b.setScalar(&[_]usize{ 1, 1 }, 4.0);

    // Test Softmax using legacy interface
    {
        var result = try legacy_api.softmax(allocator, b);
        defer result.deinit();

        // Check that softmax outputs sum to 1 for each row
        const first_row_0 = try result.getScalar(&[_]usize{ 0, 0 });
        const first_row_1 = try result.getScalar(&[_]usize{ 0, 1 });
        const second_row_0 = try result.getScalar(&[_]usize{ 1, 0 });
        const second_row_1 = try result.getScalar(&[_]usize{ 1, 1 });

        try std.testing.expectApproxEqRel(@as(f32, 1.0), first_row_0 + first_row_1, 0.0001);
        try std.testing.expectApproxEqRel(@as(f32, 1.0), second_row_0 + second_row_1, 0.0001);

        // Check that second element is larger than first in each row
        try std.testing.expect(first_row_1 > first_row_0);
        try std.testing.expect(second_row_1 > second_row_0);
    }
}

test "transpose" {
    const T = f32;
    const Tensor = tensor.Tensor(T);
    const legacy_api = LegacyApi(T);

    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 3 };

    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    var a = try Tensor.zeros(allocator, &dims, .cpu);
    defer a.deinit();

    try a.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 0, 2 }, 3.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 4.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 5.0);
    try a.setScalar(&[_]usize{ 1, 2 }, 6.0);

    // Test Transpose using the legacy interface
    {
        var result = try legacy_api.transpose(allocator, a);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 1.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 4.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 5.0), try result.getScalar(&[_]usize{ 1, 1 }));
        try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{ 2, 0 }));
        try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{ 2, 1 }));
    }

    // Test using legacy interface
    {
        var result = try legacy_api.transpose(allocator, a);
        defer result.deinit();

        try std.testing.expectEqual(@as(f32, 1.0), try result.getScalar(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f32, 4.0), try result.getScalar(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f32, 5.0), try result.getScalar(&[_]usize{ 1, 1 }));
        try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{ 2, 0 }));
        try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{ 2, 1 }));
    }
}
