const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("tensor.zig");
const ops = @import("ops.zig");
const autodiff = @import("autodiff.zig");

const Tensor = tensor.Tensor;
const DType = tensor.DType;
const BackendType = tensor.BackendType;
const Shape = tensor.Shape;

/// Configuration for property testing
pub const PropertyTestConfig = struct {
    /// Number of test iterations to run
    iterations: usize = 50,
    
    /// Seed for random number generation (null for time-based seed)
    seed: ?u64 = null,
    
    /// Enable verbose output
    verbose: bool = false,
    
    /// Default epsilon (absolute error tolerance) for floating point comparisons
    epsilon: f32 = 1e-5,
    
    /// Epsilon for matrix multiplication tests (needs to be larger due to accumulated error)
    matmul_epsilon: f32 = 1e-4,
    
    /// Epsilon for finite difference tests (needs to be larger due to numerical approximation)
    finite_diff_epsilon: f32 = 1e-2,
    
    /// Enable performance testing
    measure_performance: bool = false,
    
    /// Enable empty tensor testing (may cause alignment issues in some operations)
    enable_empty_tensors: bool = false,
    
    /// Maximum tensor rank to test (number of dimensions)
    max_rank: usize = 4,
    
    /// Maximum size for tensor dimensions
    max_dim: usize = 5,
    
    /// Control the percentage of edge cases in generated tensors
    edge_case_probability: f32 = 0.1,
};

/// Default configuration for property testing
pub const default_config = PropertyTestConfig{};

/// Helper function for pointer casting (same as in tensor.zig)
fn ptrCastHelper(comptime T: type, ptr: anytype) T {
    return @ptrCast(@alignCast(ptr));
}

/// Performance measurement struct for tensor operations
pub const PerformanceMeasurement = struct {
    operation_name: []const u8,
    elapsed_ns: u64,
    tensor_sizes: []const usize,
    
    pub fn print(self: PerformanceMeasurement) void {
        std.debug.print("Performance: {s} - ", .{self.operation_name});
        
        // Print tensor dimensions
        std.debug.print("Tensor dims: [", .{});
        for (self.tensor_sizes, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{}", .{dim});
        }
        std.debug.print("] - ", .{});
        
        // Print elapsed time in appropriate units
        if (self.elapsed_ns < 1000) {
            std.debug.print("{d} ns\n", .{self.elapsed_ns});
        } else if (self.elapsed_ns < 1000_000) {
            std.debug.print("{d:.3} µs\n", .{@as(f64, @floatFromInt(self.elapsed_ns)) / 1000.0});
        } else if (self.elapsed_ns < 1000_000_000) {
            std.debug.print("{d:.3} ms\n", .{@as(f64, @floatFromInt(self.elapsed_ns)) / 1000_000.0});
        } else {
            std.debug.print("{d:.3} s\n", .{@as(f64, @floatFromInt(self.elapsed_ns)) / 1000_000_000.0});
        }
    }
};

/// Generates a random tensor with the specified shape constraints and configuration options
/// Returns a tensor with random shape and values according to the parameters
pub fn randomTensor(
    allocator: Allocator, 
    rnd: *std.Random, 
    config: PropertyTestConfig,
    dtype: ?DType
) !Tensor {
    const rank = rnd.intRangeAtMost(usize, 1, config.max_rank);
    const data_type = dtype orelse blk: {
        // Choose a random data type from available options
        const types = [_]DType{ .f32, .f16, .f64, .i32, .bool };
        const type_idx = rnd.intRangeAtMost(usize, 0, types.len - 1);
        break :blk types[type_idx];
    };
    
    // Prepare a buffer for dimensions
    var shape_buffer: [4]usize = undefined;
    
    // Generate random dimensions for each axis
    var has_zero_dim = false;
    for (0..rank) |i| {
        // Empty dimensions support
        if (config.enable_empty_tensors and rnd.float(f32) < 0.05) {
            // 5% chance of empty dimension (0) if enabled
            shape_buffer[i] = 0;
            has_zero_dim = true;
        } else if (rnd.float(f32) < config.edge_case_probability) {
            // Chance of single-element dimension (1)
            shape_buffer[i] = 1;
        } else {
            // Otherwise, random dimension between 1 and max_dim
            shape_buffer[i] = rnd.intRangeAtMost(usize, 1, config.max_dim);
        }
    }
    
    // Create tensor with generated shape
    var tensor_obj = try Tensor.zeros(allocator, shape_buffer[0..rank], data_type, .cpu);
    errdefer tensor_obj.deinit();
    
    // If tensor has zero dimension, no need to fill with values
    if (has_zero_dim) {
        return tensor_obj;
    }
    
    // Fill with random values based on dtype
    switch (data_type) {
        .f32 => {
            const buf = ptrCastHelper([*]f32, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
            for (buf) |*x| {
                x.* = rnd.float(f32) * 2.0 - 1.0; // Random value between -1.0 and 1.0
            }
        },
        .f16 => {
            // Since f16 is not natively supported in Zig, we need a conversion from f32
            const buf = ptrCastHelper([*]u16, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
            for (buf) |*x| {
                const f32_val = rnd.float(f32) * 2.0 - 1.0;
                x.* = f32ToF16Bits(f32_val); // Convert from f32 to f16 bits
            }
        },
        .f64 => {
            const buf = ptrCastHelper([*]f64, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
            for (buf) |*x| {
                x.* = rnd.float(f64) * 2.0 - 1.0; // Random value between -1.0 and 1.0
            }
        },
        .i32 => {
            const buf = ptrCastHelper([*]i32, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
            for (buf) |*x| {
                x.* = rnd.intRangeLessThan(i32, -100, 100); // Random integers between -100 and 99
            }
        },
        .i64 => {
            const buf = ptrCastHelper([*]i64, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
            for (buf) |*x| {
                x.* = rnd.intRangeLessThan(i64, -100, 100); // Random integers between -100 and 99
            }
        },
        .bool => {
            const buf = ptrCastHelper([*]u8, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
            for (buf) |*x| {
                x.* = if (rnd.float(f32) < 0.5) 0 else 1; // Random boolean
            }
        },
    }
    
    return tensor_obj;
}


/// Helper function to convert f32 to f16 bit representation
fn f32ToF16Bits(value: f32) u16 {
    // Simple implementation of float conversion based on the IEEE-754 standard
    // This is a basic implementation and doesn't handle all edge cases perfectly
    
    const f32_bits = @bitCast(u32, value);
    const sign = @as(u16, @truncate((f32_bits >> 31) & 0x1));
    var exponent = @as(i32, @intCast((f32_bits >> 23) & 0xFF)) - 127;
    var mantissa = f32_bits & 0x7FFFFF;
    
    // Handle special cases
    if (exponent == 128) {
        // Infinity or NaN
        if (mantissa == 0) {
            // Infinity
            return (sign << 15) | 0x7C00;
        } else {
            // NaN
            return (sign << 15) | 0x7E00;
        }
    }
    
    // Adjust exponent for f16 bias
    exponent += 15;
    
    // Handle very small values
    if (exponent <= 0) {
        if (exponent < -10) {
            // Too small, return zero
            return sign << 15;
        }
        // Denormalized value
        mantissa = (mantissa | 0x800000) >> (14 - exponent);
        return (sign << 15) | @as(u16, @truncate(mantissa >> 13));
    }
    
    // Handle very large values
    if (exponent > 30) {
        // Too large, return infinity
        return (sign << 15) | 0x7C00;
    }
    
    // Normal case
    return (sign << 15) | (@as(u16, @truncate(exponent)) << 10) | @as(u16, @truncate(mantissa >> 13));
}

/// Generates a pair of tensors with the same shape using specified configuration
/// Returns a pair of tensors with identical shapes but different random values
pub fn randomTensorPair(
    allocator: Allocator, 
    rnd: *std.Random, 
    config: PropertyTestConfig,
    dtype: ?DType
) !struct { a: Tensor, b: Tensor } {
    // Use the same data type for both tensors
    const data_type = dtype orelse blk: {
        const types = [_]DType{ .f32, .f16, .f64, .i32, .bool };
        const type_idx = rnd.intRangeAtMost(usize, 0, types.len - 1);
        break :blk types[type_idx];
    };
    
    // Create the first tensor
    var a = try randomTensor(allocator, rnd, config, data_type);
    errdefer a.deinit();
    
    // Create the second tensor with the same shape
    var b = try Tensor.zeros(allocator, a.shape.dims, data_type, .cpu);
    errdefer b.deinit();
    
    // If tensor has zero dimension, no need to fill with values
    if (a.shape.elemCount() == 0) {
        return .{ .a = a, .b = b };
    }
    
    // Fill tensor b with different random values based on dtype
    switch (data_type) {
        .f32 => {
            const buf_b = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..b.shape.elemCount()];
            for (buf_b) |*x| {
                x.* = rnd.float(f32) * 2.0 - 1.0;
            }
        },
        .f16 => {
            const buf_b = ptrCastHelper([*]u16, b.buffer.data.ptr)[0..b.shape.elemCount()];
            for (buf_b) |*x| {
                const f32_val = rnd.float(f32) * 2.0 - 1.0;
                x.* = f32ToF16Bits(f32_val);
            }
        },
        .f64 => {
            const buf_b = ptrCastHelper([*]f64, b.buffer.data.ptr)[0..b.shape.elemCount()];
            for (buf_b) |*x| {
                x.* = rnd.float(f64) * 2.0 - 1.0;
            }
        },
        .i32 => {
            const buf_b = ptrCastHelper([*]i32, b.buffer.data.ptr)[0..b.shape.elemCount()];
            for (buf_b) |*x| {
                x.* = rnd.intRangeLessThan(i32, -100, 100);
            }
        },
        .i64 => {
            const buf_b = ptrCastHelper([*]i64, b.buffer.data.ptr)[0..b.shape.elemCount()];
            for (buf_b) |*x| {
                x.* = rnd.intRangeLessThan(i64, -100, 100);
            }
        },
        .bool => {
            const buf_b = ptrCastHelper([*]u8, b.buffer.data.ptr)[0..b.shape.elemCount()];
            for (buf_b) |*x| {
                x.* = if (rnd.float(f32) < 0.5) 0 else 1;
            }
        },
    }
    
    return .{ .a = a, .b = b };
}


/// Generates a trio of tensors with compatible shapes for matrix multiplication using configuration
/// Returns tensors with shapes (m x n), (n x p), and (p x q) for testing matrix multiplication chains
pub fn randomMatmulTrio(
    allocator: Allocator, 
    rnd: *std.Random, 
    config: PropertyTestConfig,
    dtype: ?DType
) !struct { a: Tensor, b: Tensor, c: Tensor } {
    // Matrix multiplication usually requires floating point, but integer matmul is also possible
    // For property testing of matrix multiplication, we prioritize floating point types
    const data_type = dtype orelse blk: {
        const types = [_]DType{ .f32, .f16, .f64 };
        const type_idx = rnd.intRangeAtMost(usize, 0, types.len - 1);
        break :blk types[type_idx];
    };
    
    // For matrix multiplication, we need (m x n) * (n x p) * (p x q)
    var m: usize = undefined;
    var n: usize = undefined;
    var p: usize = undefined;
    var q: usize = undefined;
    
    // Generate dimensions including edge cases
    if (rnd.float(f32) < config.edge_case_probability) {
        // Edge case: Single-element dimensions (1x1 matrices)
        m = 1;
        n = 1;
        p = 1;
        q = 1;
    } else if (rnd.float(f32) < config.edge_case_probability * 2) {
        // Edge case: Large dimension mismatch for numerical stability testing
        m = rnd.intRangeAtMost(usize, 1, 2);
        n = rnd.intRangeAtMost(usize, config.max_dim / 2, config.max_dim);
        p = rnd.intRangeAtMost(usize, config.max_dim / 2, config.max_dim);
        q = rnd.intRangeAtMost(usize, 1, 2);
    } else {
        // Normal case: Random dimensions
        m = rnd.intRangeAtMost(usize, 1, config.max_dim);
        n = rnd.intRangeAtMost(usize, 1, config.max_dim);
        p = rnd.intRangeAtMost(usize, 1, config.max_dim);
        q = rnd.intRangeAtMost(usize, 1, config.max_dim);
    }
    
    // Handle empty tensor case if enabled
    if (config.enable_empty_tensors and rnd.float(f32) < 0.05) {
        const dim_to_zero = rnd.intRangeAtMost(usize, 0, 3);
        switch (dim_to_zero) {
            0 => m = 0,
            1 => n = 0,
            2 => p = 0,
            3 => q = 0,
            else => {},
        }
    }

    // Create tensors with compatible shapes for matmul: (m x n) @ (n x p) @ (p x q)
    var a = try Tensor.zeros(allocator, &[_]usize{ m, n }, data_type, .cpu);
    errdefer a.deinit();
    
    var b = try Tensor.zeros(allocator, &[_]usize{ n, p }, data_type, .cpu);
    errdefer b.deinit();
    
    var c = try Tensor.zeros(allocator, &[_]usize{ p, q }, data_type, .cpu);
    errdefer c.deinit();

    // If any tensor has zero elements, no need to fill with values
    if (a.shape.elemCount() == 0 or b.shape.elemCount() == 0 or c.shape.elemCount() == 0) {
        return .{ .a = a, .b = b, .c = c };
    }

    // Fill with random values based on dtype
    switch (data_type) {
        .f32 => {
            // Fill tensor a
            {
                const buf_a = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
                for (buf_a) |*x| {
                    x.* = rnd.float(f32) * 2.0 - 1.0;
                }
            }
            
            // Fill tensor b
            {
                const buf_b = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..b.shape.elemCount()];
                for (buf_b) |*x| {
                    x.* = rnd.float(f32) * 2.0 - 1.0;
                }
            }
            
            // Fill tensor c
            {
                const buf_c = ptrCastHelper([*]f32, c.buffer.data.ptr)[0..c.shape.elemCount()];
                for (buf_c) |*x| {
                    x.* = rnd.float(f32) * 2.0 - 1.0;
                }
            }
        },
        .f16 => {
            // Fill tensors similar to above but with f16 values
            {
                const buf_a = ptrCastHelper([*]u16, a.buffer.data.ptr)[0..a.shape.elemCount()];
                for (buf_a) |*x| {
                    x.* = f32ToF16Bits(rnd.float(f32) * 2.0 - 1.0);
                }
            }
            
            {
                const buf_b = ptrCastHelper([*]u16, b.buffer.data.ptr)[0..b.shape.elemCount()];
                for (buf_b) |*x| {
                    x.* = f32ToF16Bits(rnd.float(f32) * 2.0 - 1.0);
                }
            }
            
            {
                const buf_c = ptrCastHelper([*]u16, c.buffer.data.ptr)[0..c.shape.elemCount()];
                for (buf_c) |*x| {
                    x.* = f32ToF16Bits(rnd.float(f32) * 2.0 - 1.0);
                }
            }
        },
        .f64 => {
            // Fill tensors with f64 values
            {
                const buf_a = ptrCastHelper([*]f64, a.buffer.data.ptr)[0..a.shape.elemCount()];
                for (buf_a) |*x| {
                    x.* = rnd.float(f64) * 2.0 - 1.0;
                }
            }
            
            {
                const buf_b = ptrCastHelper([*]f64, b.buffer.data.ptr)[0..b.shape.elemCount()];
                for (buf_b) |*x| {
                    x.* = rnd.float(f64) * 2.0 - 1.0;
                }
            }
            
            {
                const buf_c = ptrCastHelper([*]f64, c.buffer.data.ptr)[0..c.shape.elemCount()];
                for (buf_c) |*x| {
                    x.* = rnd.float(f64) * 2.0 - 1.0;
                }
            }
        },
        // Support integer matmul as well
        .i32 => {
            // Fill tensors with integer values
            {
                const buf_a = ptrCastHelper([*]i32, a.buffer.data.ptr)[0..a.shape.elemCount()];
                for (buf_a) |*x| {
                    // Smaller range for integers to avoid overflow in matmul
                    x.* = rnd.intRangeLessThan(i32, -10, 10);
                }
            }
            
            {
                const buf_b = ptrCastHelper([*]i32, b.buffer.data.ptr)[0..b.shape.elemCount()];
                for (buf_b) |*x| {
                    x.* = rnd.intRangeLessThan(i32, -10, 10);
                }
            }
            
            {
                const buf_c = ptrCastHelper([*]i32, c.buffer.data.ptr)[0..c.shape.elemCount()];
                for (buf_c) |*x| {
                    x.* = rnd.intRangeLessThan(i32, -10, 10);
                }
            }
        },
        else => {
            // For now other types are not well supported for matmul
            // Fill with zeros
        },
    }

    return .{ .a = a, .b = b, .c = c };
}


/// Error information returned when tensors are not approximately equal
pub const TensorComparisonError = struct {
    max_abs_diff: f32,      // Maximum absolute difference found
    max_rel_diff: f32,      // Maximum relative difference found
    failure_index: usize,   // Index where the largest difference was found
    a_value: f64,           // Value in tensor A at failure point (f64 to support all dtypes)
    b_value: f64,           // Value in tensor B at failure point (f64 to support all dtypes)
    shape_mismatch: bool,   // Whether the tensors had different shapes
    dtype_mismatch: bool,   // Whether the tensors had different data types
    dtype_a: DType,         // Data type of tensor A
    dtype_b: DType,         // Data type of tensor B
};

/// Helper function to check if two tensors are approximately equal
/// Returns a result indicating if tensors are equal within tolerance, or detailed error information
pub fn tensorsApproxEqual(
    a: Tensor, 
    b: Tensor, 
    epsilon: f32, 
    use_relative_error: ?bool
) !union(enum) { 
    equal: bool, 
    err: TensorComparisonError 
} {
    const use_rel = use_relative_error orelse false;
    
    // Check shape equality
    if (!a.shape.eql(b.shape)) {
        return .{ 
            .err = .{
                .max_abs_diff = std.math.floatMax(f32),
                .max_rel_diff = std.math.floatMax(f32),
                .failure_index = 0,
                .a_value = 0,
                .b_value = 0,
                .shape_mismatch = true,
                .dtype_mismatch = a.dtype != b.dtype,
                .dtype_a = a.dtype,
                .dtype_b = b.dtype,
            }
        };
    }
    
    // Check dtype equality - allow comparison between different numeric types
    // but return error for incompatible types (e.g., bool vs numeric)
    const is_numeric_a = (a.dtype == .f32 or a.dtype == .f16 or a.dtype == .f64 or 
                         a.dtype == .i32 or a.dtype == .i64);
    const is_numeric_b = (b.dtype == .f32 or b.dtype == .f16 or b.dtype == .f64 or 
                         b.dtype == .i32 or b.dtype == .i64);
                         
    if ((a.dtype == .bool or b.dtype == .bool) and a.dtype != b.dtype) {
        return error.IncompatibleDataTypes;
    }
    
    if (!is_numeric_a or !is_numeric_b) {
        // Currently we only properly support numeric type comparisons
        return error.UnsupportedDataType;
    }
    
    // Since empty tensors (element count 0) have no values to compare,
    // they are equal if their shapes match (which we already checked)
    if (a.shape.elemCount() == 0 and b.shape.elemCount() == 0) {
        return .{ .equal = true };
    }
    
    var max_abs_diff: f32 = 0;
    var max_rel_diff: f32 = 0;
    var failure_index: usize = 0;
    var a_failure_value: f64 = 0;
    var b_failure_value: f64 = 0;
    
    // We'll compare values by converting them to f64 for maximum precision
    // First, create buffers to hold the converted values
    const elem_count = a.shape.elemCount();
    var a_values: []f64 = undefined;
    var b_values: []f64 = undefined;
    
    // Extract values based on dtype
    switch (a.dtype) {
        .f32 => {
            const buf = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..elem_count];
            a_values = try a.allocator.alloc(f64, elem_count);
            defer a.allocator.free(a_values);
            
            for (buf, 0..) |val, i| {
                a_values[i] = val;
            }
        },
        .f16 => {
            const buf = ptrCastHelper([*]u16, a.buffer.data.ptr)[0..elem_count];
            a_values = try a.allocator.alloc(f64, elem_count);
            defer a.allocator.free(a_values);
            
            for (buf, 0..) |val, i| {
                a_values[i] = f16ToF64(val);
            }
        },
        .f64 => {
            const buf = ptrCastHelper([*]f64, a.buffer.data.ptr)[0..elem_count];
            a_values = buf;
        },
        .i32 => {
            const buf = ptrCastHelper([*]i32, a.buffer.data.ptr)[0..elem_count];
            a_values = try a.allocator.alloc(f64, elem_count);
            defer a.allocator.free(a_values);
            
            for (buf, 0..) |val, i| {
                a_values[i] = @as(f64, @floatFromInt(val));
            }
        },
        .i64 => {
            const buf = ptrCastHelper([*]i64, a.buffer.data.ptr)[0..elem_count];
            a_values = try a.allocator.alloc(f64, elem_count);
            defer a.allocator.free(a_values);
            
            for (buf, 0..) |val, i| {
                a_values[i] = @as(f64, @floatFromInt(val));
            }
        },
        .bool => {
            const buf = ptrCastHelper([*]u8, a.buffer.data.ptr)[0..elem_count];
            a_values = try a.allocator.alloc(f64, elem_count);
            defer a.allocator.free(a_values);
            
            for (buf, 0..) |val, i| {
                a_values[i] = if (val != 0) 1.0 else 0.0;
            }
        },
    }
    
    // Do the same for tensor B
    switch (b.dtype) {
        .f32 => {
            const buf = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..elem_count];
            b_values = try b.allocator.alloc(f64, elem_count);
            defer b.allocator.free(b_values);
            
            for (buf, 0..) |val, i| {
                b_values[i] = val;
            }
        },
        .f16 => {
            const buf = ptrCastHelper([*]u16, b.buffer.data.ptr)[0..elem_count];
            b_values = try b.allocator.alloc(f64, elem_count);
            defer b.allocator.free(b_values);
            
            for (buf, 0..) |val, i| {
                b_values[i] = f16ToF64(val);
            }
        },
        .f64 => {
            const buf = ptrCastHelper([*]f64, b.buffer.data.ptr)[0..elem_count];
            b_values = buf;
        },
        .i32 => {
            const buf = ptrCastHelper([*]i32, b.buffer.data.ptr)[0..elem_count];
            b_values = try b.allocator.alloc(f64, elem_count);
            defer b.allocator.free(b_values);
            
            for (buf, 0..) |val, i| {
                b_values[i] = @as(f64, @floatFromInt(val));
            }
        },
        .i64 => {
            const buf = ptrCastHelper([*]i64, b.buffer.data.ptr)[0..elem_count];
            b_values = try b.allocator.alloc(f64, elem_count);
            defer b.allocator.free(b_values);
            
            for (buf, 0..) |val, i| {
                b_values[i] = @as(f64, @floatFromInt(val));
            }
        },
        .bool => {
            const buf = ptrCastHelper([*]u8, b.buffer.data.ptr)[0..elem_count];
            b_values = try b.allocator.alloc(f64, elem_count);
            defer b.allocator.free(b_values);
            
            for (buf, 0..) |val, i| {
                b_values[i] = if (val != 0) 1.0 else 0.0;
            }
        },
    }
    
    // Now compare the values
    for (a_values, b_values, 0..) |val_a, val_b, i| {
        const abs_diff = @abs(val_a - val_b);
        
        // Check max absolute diff for reporting
        if (abs_diff > max_abs_diff) {
            max_abs_diff = @floatCast(abs_diff);
            failure_index = i;
            a_failure_value = val_a;
            b_failure_value = val_b;
        }
        
        if (use_rel) {
            // Relative error: |a-b|/(|a|+|b|+tiny)
            // Adding tiny prevents division by zero when both values are very small
            const denominator = @abs(val_a) + @abs(val_b) + 1e-10;
            const rel_diff = abs_diff / denominator;
            
            if (rel_diff > max_rel_diff) {
                max_rel_diff = @floatCast(rel_diff);
            }
            
            if (rel_diff > epsilon) {
                return .{
                    .err = .{
                        .max_abs_diff = max_abs_diff,
                        .max_rel_diff = max_rel_diff,
                        .failure_index = failure_index,
                        .a_value = a_failure_value,
                        .b_value = b_failure_value,
                        .shape_mismatch = false,
                        .dtype_mismatch = a.dtype != b.dtype,
                        .dtype_a = a.dtype,
                        .dtype_b = b.dtype,
                    }
                };
            }
        } else {
            // Use absolute error comparison
            if (abs_diff > epsilon) {
                return .{
                    .err = .{
                        .max_abs_diff = max_abs_diff,
                        .max_rel_diff = 0,
                        .failure_index = failure_index,
                        .a_value = a_failure_value,
                        .b_value = b_failure_value,
                        .shape_mismatch = false,
                        .dtype_mismatch = a.dtype != b.dtype,
                        .dtype_a = a.dtype,
                        .dtype_b = b.dtype,
                    }
                };
            }
        }
    }
    
    return .{ .equal = true };
}

/// Convert an f16 bit pattern to an f64 value
fn f16ToF64(f16_bits: u16) f64 {
    const sign = (f16_bits >> 15) & 0x1;
    const exponent = (f16_bits >> 10) & 0x1F;
    const mantissa = f16_bits & 0x3FF;
    
    if (exponent == 0) {
        // Denormalized number or zero
        if (mantissa == 0) {
            return if (sign == 0) 0.0 else -0.0;
        } else {
            const val = @as(f64, @floatFromInt(mantissa)) * std.math.pow(f64, 2.0, -24.0);
            return if (sign == 0) val else -val;
        }
    } else if (exponent == 0x1F) {
        // Infinity or NaN
        if (mantissa == 0) {
            return if (sign == 0) std.math.inf(f64) else -std.math.inf(f64);
        } else {
            return std.math.nan(f64);
        }
    }
    
    // Normalized number
    const exponent_value = @as(i32, @intCast(exponent)) - 15 + 1023; // Convert from f16 bias to f64 bias
    const mantissa_value = @as(f64, @floatFromInt(mantissa)) / 1024.0;
    const val = std.math.ldexp(1.0 + mantissa_value, exponent_value - 1023);
    
    return if (sign == 0) val else -val;
}

/// Helper to print a tensor error report with details on where the comparison failed
pub fn printTensorErrorReport(error_info: TensorComparisonError) void {
    if (error_info.shape_mismatch) {
        std.debug.print("Tensor shape mismatch\n", .{});
        std.debug.print("Tensor data types: A={}, B={}\n", .{error_info.dtype_a, error_info.dtype_b});
        return;
    }
    
    if (error_info.dtype_mismatch) {
        std.debug.print("Tensor data type mismatch: A={}, B={}\n", .{error_info.dtype_a, error_info.dtype_b});
    }
    
    std.debug.print("Tensor comparison failed at index {}\n", .{error_info.failure_index});
    std.debug.print("Values: A={d:.8}, B={d:.8}\n", .{error_info.a_value, error_info.b_value});
    std.debug.print("Absolute difference: {d:.8}\n", .{error_info.max_abs_diff});
    
    if (error_info.max_rel_diff > 0) {
        std.debug.print("Relative difference: {d:.8}\n", .{error_info.max_rel_diff});
    }
    
    // Print additional info about the types
    std.debug.print("Data types: A={}, B={}\n", .{error_info.dtype_a, error_info.dtype_b});
}

//
// Generic property test runner
//

/// Configuration for a specific shrinking strategy
pub const ShrinkConfig = struct {
    /// The dimension to use for this shrinking attempt (if shrinking dimensions)
    max_dim: usize = 1,
    
    /// Whether to simplify tensor values (zeros, small values)
    simplify_values: bool = false,
    
    /// Level of value simplification (0-3, where 3 is most aggressive)
    simplification_level: u8 = 0,
    
    /// Track the number of shrinking attempts for this configuration
    attempts: usize = 0,
    
    /// Maximum attempts before giving up on this strategy
    max_attempts: usize = 5,
};

/// Thread-local state for shrinking process
pub const ShrinkingState = struct {
    /// Current configuration for shrinking process
    config: PropertyTestConfig,
    
    /// The random seed to use for reproducing the failure
    seed: u64,
    
    /// The current shrinking strategy being attempted
    strategy: enum {
        none,
        reduce_dimensions,
        simplify_values,
        simplify_single_value,
    } = .none,
    
    /// Configuration for the current shrinking attempt
    shrink_config: ShrinkConfig = .{},
    
    /// Whether shrinking is in progress
    is_shrinking: bool = false,
    
    /// The index of the value being simplified (for single value strategy)
    target_index: usize = 0,
    
    /// Function to update a tensor according to shrinking strategy
    /// Returns whether the tensor was modified
    pub fn shrinkTensor(self: *ShrinkingState, tensor: *Tensor) bool {
        if (!self.is_shrinking) return false;
        
        switch (self.strategy) {
            .none => return false,
            
            .reduce_dimensions => {
                // This is handled at generation time via config.max_dim
                return false;
            },
            
            .simplify_values => {
                // Replace some or all values with simpler values
                if (tensor.dtype != .f32) return false; // Only handle f32 for now
                
                const buf = ptrCastHelper([*]f32, tensor.buffer.data.ptr)[0..tensor.shape.elemCount()];
                if (buf.len == 0) return false;
                
                var modified = false;
                
                switch (self.shrink_config.simplification_level) {
                    0 => {
                        // Replace extreme values with more moderate ones
                        for (buf) |*val| {
                            if (@abs(val.*) > 0.9) {
                                val.* = if (val.* > 0) 0.5 else -0.5;
                                modified = true;
                            }
                        }
                    },
                    1 => {
                        // Replace negative values with small positive ones
                        for (buf) |*val| {
                            if (val.* < 0) {
                                val.* = 0.1;
                                modified = true;
                            }
                        }
                    },
                    2 => {
                        // Replace all values with very small values
                        for (buf) |*val| {
                            if (@abs(val.*) > 0.01) {
                                val.* = if (val.* > 0) 0.01 else -0.01;
                                modified = true;
                            }
                        }
                    },
                    3 => {
                        // Replace almost everything with zeros
                        for (buf) |*val| {
                            if (@abs(val.*) > 0.0001) {
                                val.* = 0;
                                modified = true;
                            }
                        }
                    },
                    else => {},
                }
                
                return modified;
            },
            
            .simplify_single_value => {
                // Replace a single value with a simpler one
                if (tensor.dtype != .f32) return false; // Only handle f32 for now
                
                const buf = ptrCastHelper([*]f32, tensor.buffer.data.ptr)[0..tensor.shape.elemCount()];
                if (buf.len == 0 or self.target_index >= buf.len) return false;
                
                // Get the current value
                const current = buf[self.target_index];
                
                // Try a simpler value
                const new_value: f32 = switch (self.shrink_config.simplification_level) {
                    0 => 0,
                    1 => if (current > 0) 0.5 else -0.5,
                    2 => if (current > 0) 0.1 else -0.1,
                    3 => if (current > 0) 1 else -1,
                    else => current,
                };
                
                // Only modify if the value would change
                if (current != new_value) {
                    buf[self.target_index] = new_value;
                    return true;
                }
                
                return false;
            },
        }
    }
};

/// Global state for the current shrinking process
var shrinking_state = ShrinkingState{
    .config = PropertyTestConfig{},
    .seed = 0,
};

/// Run a property test function multiple times with different random inputs
/// Uses advanced shrinking strategies to find minimal failing inputs
pub fn runPropertyTest(
    allocator: Allocator,
    comptime testFn: fn (Allocator, *std.Random) anyerror!void,
    config: PropertyTestConfig
) !void {
    // Reset shrinking state
    shrinking_state = ShrinkingState{
        .config = config,
        .seed = config.seed orelse @as(u64, @intCast(@abs(std.time.milliTimestamp()))),
    };
    
    // Log seed if verbose
    if (config.verbose) {
        std.debug.print("Using seed: {}\n", .{shrinking_state.seed});
    }
    
    // Create PRNG with specified seed
    var prng = std.Random.DefaultPrng.init(shrinking_state.seed);
    var rand = prng.random();
    const rnd = &rand;
    
    // Measure overall test runtime if performance testing is enabled
    const start_time = if (config.measure_performance) std.time.nanoTimestamp() else 0;
    
    // Run the test multiple times with different random inputs
    for (0..config.iterations) |i| {
        if (config.verbose and i % 10 == 0) {
            std.debug.print("Running iteration {}/{}\n", .{i + 1, config.iterations});
        }
        
        // Measure individual test runtime if performance testing is enabled
        const test_start_time = if (config.measure_performance) std.time.nanoTimestamp() else 0;
        
        // Try to run the test
        testFn(allocator, rnd) catch |err| {
            // If the test fails, try to shrink the input
            std.debug.print("Test failed on iteration {} with error: {!}\n", .{i + 1, err});
            std.debug.print("Failure reproduces with seed: {}\n", .{shrinking_state.seed});
            
            // Try to find a minimal failing case by shrinking
            try shrinkFailingTest(allocator, testFn);
            
            return err;
        };
        
        // Log performance information if requested
        if (config.measure_performance) {
            const test_elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - test_start_time));
            std.debug.print("Test iteration {} took: ", .{i + 1});
            
            if (test_elapsed < 1000) {
                std.debug.print("{} ns\n", .{test_elapsed});
            } else if (test_elapsed < 1000_000) {
                std.debug.print("{d:.3} µs\n", .{@as(f64, @floatFromInt(test_elapsed)) / 1000.0});
            } else if (test_elapsed < 1000_000_000) {
                std.debug.print("{d:.3} ms\n", .{@as(f64, @floatFromInt(test_elapsed)) / 1000_000.0});
            } else {
                std.debug.print("{d:.3} s\n", .{@as(f64, @floatFromInt(test_elapsed)) / 1000_000_000.0});
            }
        }
    }
    
    // Log total test time if performance testing is enabled
    if (config.measure_performance) {
        const total_elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
        std.debug.print("Total test time: ");
        
        if (total_elapsed < 1000_000) {
            std.debug.print("{d:.3} µs\n", .{@as(f64, @floatFromInt(total_elapsed)) / 1000.0});
        } else if (total_elapsed < 1000_000_000) {
            std.debug.print("{d:.3} ms\n", .{@as(f64, @floatFromInt(total_elapsed)) / 1000_000.0});
        } else {
            std.debug.print("{d:.3} s\n", .{@as(f64, @floatFromInt(total_elapsed)) / 1000_000_000.0});
        }
    }
}

/// Try to shrink failing test inputs to find a minimal failing case
fn shrinkFailingTest(
    allocator: Allocator,
    comptime testFn: fn (Allocator, *std.Random) anyerror!void
) !void {
    std.debug.print("Attempting to find minimal failing case...\n", .{});
    
    // Enable shrinking mode
    shrinking_state.is_shrinking = true;
    
    // First strategy: reduce dimensions
    shrinking_state.strategy = .reduce_dimensions;
    const dim_steps = [_]usize{ 4, 3, 2, 1 };
    var reduced_dims = false;
    
    for (dim_steps) |dim| {
        // Set a temporary config with reduced dimensions
        var reduced_config = shrinking_state.config;
        reduced_config.max_dim = dim;
        shrinking_state.config = reduced_config;
        
        std.debug.print("Strategy 1: Trying with max_dim={}\n", .{dim});
        var shrink_prng = std.Random.DefaultPrng.init(shrinking_state.seed);
        var shrink_rand = shrink_prng.random();
        
        // Try to run with reduced dimensions
        testFn(allocator, &shrink_rand) catch {
            // If it still fails, we found a smaller failing case
            reduced_dims = true;
            std.debug.print("Test still fails with max_dim={}\n", .{dim});
            break;
        };
    }
    
    // If reducing dimensions didn't help, restore original config
    if (!reduced_dims) {
        std.debug.print("Dimension reduction didn't produce a smaller failing case\n", .{});
        shrinking_state.config = shrinking_state.config;
    }
    
    // Second strategy: simplify values
    shrinking_state.strategy = .simplify_values;
    const simplification_levels = [_]u8{ 0, 1, 2, 3 };
    var simplified_values = false;
    
    for (simplification_levels) |level| {
        shrinking_state.shrink_config.simplification_level = level;
        
        std.debug.print("Strategy 2: Trying value simplification level {}\n", .{level});
        var shrink_prng = std.Random.DefaultPrng.init(shrinking_state.seed);
        var shrink_rand = shrink_prng.random();
        
        // Try to run with simplified values
        testFn(allocator, &shrink_rand) catch {
            // If it still fails, we found a smaller failing case
            simplified_values = true;
            std.debug.print("Test still fails with value simplification level {}\n", .{level});
            break;
        };
    }
    
    // Third strategy: binary search for critical value
    if (!simplified_values and !reduced_dims) {
        std.debug.print("Value simplification didn't produce a smaller failing case\n", .{});
        shrinking_state.strategy = .simplify_single_value;
        
        // Try to simplify single values
        shrinking_state.shrink_config.simplification_level = 0; // Start with zeros
        var simplified_single = false;
        
        // Try a fixed number of random indices
        for (0..5) |_| {
            // Create new PRNG to try different random values for the test
            var shrink_prng = std.Random.DefaultPrng.init(shrinking_state.seed + 100);
            var shrink_rand = shrink_prng.random();
            
            // Create a dummy tensor to get a reasonable target index
            const fake_tensor = try randomTensor(allocator, &shrink_rand, default_config, null);
            defer fake_tensor.deinit();
            
            const max_elem = fake_tensor.shape.elemCount();
            if (max_elem == 0) continue;
            
            shrinking_state.target_index = shrink_rand.uintLessThan(usize, max_elem);
            
            std.debug.print("Strategy 3: Trying to simplify value at index {}\n", .{shrinking_state.target_index});
            
            testFn(allocator, &shrink_rand) catch {
                // If it still fails, we found a critical value
                simplified_single = true;
                std.debug.print("Test fails when simplifying value at index {}\n", .{shrinking_state.target_index});
                break;
            };
        }
        
        if (!simplified_single) {
            std.debug.print("Single value simplification didn't produce a smaller failing case\n", .{});
        }
    }
    
    // Reset shrinking state
    shrinking_state.is_shrinking = false;
    shrinking_state.strategy = .none;
    
    std.debug.print("Shrinking complete\n", .{});
}


/// Hook functions that can be called from tensor generators to apply shrinking
pub fn applyShrinking(tensor: *Tensor) void {
    if (shrinking_state.is_shrinking) {
        _ = shrinking_state.shrinkTensor(tensor);
    }
}

//
// Property checks for basic tensor operations
//

/// Check property: A + B = B + A (commutativity of addition)
pub fn checkAdditionCommutativity(allocator: Allocator, rnd: *std.Random) !void {
    // Get config from shrinking state or use default
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    
    // Generate random tensors with options for different dtypes and dimensions
    var pair = try randomTensorPair(
        allocator, 
        rnd, 
        config,
        .f32          // dtype
    );
    defer pair.a.deinit();
    defer pair.b.deinit();

    var a_plus_b = try ops.add(allocator, pair.a, pair.b);
    defer a_plus_b.deinit();
    
    var b_plus_a = try ops.add(allocator, pair.b, pair.a);
    defer b_plus_a.deinit();

    // Check equality with relative error and improved reporting
    const comparison = try tensorsApproxEqual(a_plus_b, b_plus_a, 1e-5, true);
    
    if (comparison == .err) {
        std.debug.print("Addition commutativity test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        
        std.debug.print("Tensor shapes: ", .{});
        for (pair.a.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        return error.TensorComparisonFailed;
    }
}

/// Check property: A - B + B ≈ A (addition and subtraction cancellation)
pub fn checkAddSubtractCancellation(allocator: Allocator, rnd: *std.Random) !void {
    // Get config from shrinking state or use default
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    
    var pair = try randomTensorPair(
        allocator, 
        rnd, 
        config,
        .f32          // dtype
    );
    defer pair.a.deinit();
    defer pair.b.deinit();

    var a_minus_b = try ops.subtract(allocator, pair.a, pair.b);
    defer a_minus_b.deinit();
    
    var result = try ops.add(allocator, a_minus_b, pair.b);
    defer result.deinit();

    // Use a slightly larger epsilon due to potential floating-point errors in the subtraction
    const comparison = try tensorsApproxEqual(result, pair.a, 1e-5, true);
    
    if (comparison == .err) {
        std.debug.print("Addition/subtraction cancellation test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        
        std.debug.print("Tensor shapes: ", .{});
        for (pair.a.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        return error.TensorComparisonFailed;
    }
}

/// Check property: (A * B) * C = A * (B * C) (associativity of matrix multiplication)
pub fn checkMatmulAssociativity(allocator: Allocator, rnd: *std.Random) !void {
    // Get config from shrinking state or use default
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    
    var trio = try randomMatmulTrio(
        allocator, 
        rnd, 
        config,
        .f32          // dtype
    );
    defer trio.a.deinit();
    defer trio.b.deinit();
    defer trio.c.deinit();

    // Compute (A * B) * C
    var ab = try ops.matmul(allocator, trio.a, trio.b);
    defer ab.deinit();
    
    var abc = try ops.matmul(allocator, ab, trio.c);
    defer abc.deinit();

    // Compute A * (B * C)
    var bc = try ops.matmul(allocator, trio.b, trio.c);
    defer bc.deinit();
    
    var a_bc = try ops.matmul(allocator, trio.a, bc);
    defer a_bc.deinit();

    // Use relative error comparison - matrix multiplication accumulates errors
    // so relative error is more appropriate than absolute error
    const comparison = try tensorsApproxEqual(abc, a_bc, 1e-4, true);
    
    if (comparison == .err) {
        std.debug.print("Matrix multiplication associativity test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        
        std.debug.print("Tensor A shape: ", .{});
        for (trio.a.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        std.debug.print("Tensor B shape: ", .{});
        for (trio.b.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        std.debug.print("Tensor C shape: ", .{});
        for (trio.c.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        return error.TensorComparisonFailed;
    }
}

/// Check property: ReLU(x) ≥ 0 for all elements
pub fn checkReLUNonNegativity(allocator: Allocator, rnd: *std.Random) !void {
    // Get config from shrinking state or use default
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    
    var tensor_obj = try randomTensor(
        allocator, 
        rnd, 
        config,
        .f32          // dtype
    );
    defer tensor_obj.deinit();

    var result = try ops.relu(allocator, tensor_obj);
    defer result.deinit();

    const buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..result.shape.elemCount()];
    for (buf, 0..) |val, i| {
        if (val < 0) {
            std.debug.print("ReLU non-negativity test failed at index {}:\n", .{i});
            std.debug.print("Value after ReLU: {d:.6}\n", .{val});
            return error.ReLUProducedNegativeValue;
        }
    }
}

/// Check property: A⋅(B+C)=A⋅B+A⋅C (distributivity of matmul over addition)
pub fn checkMatmulDistributivity(allocator: Allocator, rnd: *std.Random) !void {
    // Generate compatible tensors for matmul distributivity test
    const m = rnd.intRangeAtMost(usize, 1, 5);
    const n = rnd.intRangeAtMost(usize, 1, 5);
    const p = rnd.intRangeAtMost(usize, 1, 5);
    
    // A is m x n, B and C are n x p
    var a = try Tensor.zeros(allocator, &[_]usize{ m, n }, .f32, .cpu);
    defer a.deinit();
    
    var b = try Tensor.zeros(allocator, &[_]usize{ n, p }, .f32, .cpu);
    defer b.deinit();
    
    var c = try Tensor.zeros(allocator, &[_]usize{ n, p }, .f32, .cpu);
    defer c.deinit();
    
    // Fill with random values
    {
        const buf_a = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        for (buf_a) |*x| {
            x.* = rnd.float(f32) * 2.0 - 1.0;
        }
    }
    
    {
        const buf_b = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..b.shape.elemCount()];
        for (buf_b) |*x| {
            x.* = rnd.float(f32) * 2.0 - 1.0;
        }
    }
    
    {
        const buf_c = ptrCastHelper([*]f32, c.buffer.data.ptr)[0..c.shape.elemCount()];
        for (buf_c) |*x| {
            x.* = rnd.float(f32) * 2.0 - 1.0;
        }
    }
    
    // Calculate A⋅(B+C)
    var b_plus_c = try ops.add(allocator, b, c);
    defer b_plus_c.deinit();
    
    var left = try ops.matmul(allocator, a, b_plus_c);
    defer left.deinit();
    
    // Calculate A⋅B + A⋅C
    var ab = try ops.matmul(allocator, a, b);
    defer ab.deinit();
    
    var ac = try ops.matmul(allocator, a, c);
    defer ac.deinit();
    
    var right = try ops.add(allocator, ab, ac);
    defer right.deinit();
    
    // Compare the results using relative error
    const comparison = try tensorsApproxEqual(left, right, 1e-4, true);
    
    if (comparison == .err) {
        std.debug.print("Matrix multiplication distributivity test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        
        std.debug.print("Tensor A shape: [{}, {}]\n", .{ m, n });
        std.debug.print("Tensor B shape: [{}, {}]\n", .{ n, p });
        
        return error.TensorComparisonFailed;
    }
}

/// Check property: A * I = A (identity property of matrix multiplication)
pub fn checkMatmulIdentity(allocator: Allocator, rnd: *std.Random) !void {
    // Generate a random tensor
    var tensor_obj = try randomTensor(
        allocator, 
        rnd, 
        5,            // max_dim
        2,            // max_rank (2D only for matmul)
        true,         // include edge cases
        .f32          // dtype
    );
    defer tensor_obj.deinit();
    
    // Handle the case of empty tensors
    if (tensor_obj.shape.elemCount() == 0) {
        return; // Skip test for empty tensors
    }
    
    // Ensure tensor is at least 2D for matmul
    if (tensor_obj.shape.rank() != 2) {
        return; // Skip test for non-2D tensors
    }
    
    // Create identity matrix of appropriate size
    const dim = tensor_obj.shape.dims[1];
    var identity = try Tensor.zeros(allocator, &[_]usize{ dim, dim }, .f32, .cpu);
    defer identity.deinit();
    
    // Set the diagonal elements to 1.0
    const identity_buf = ptrCastHelper([*]f32, identity.buffer.data.ptr);
    for (0..dim) |i| {
        identity_buf[i * dim + i] = 1.0;
    }
    
    // A * I
    var result = try ops.matmul(allocator, tensor_obj, identity);
    defer result.deinit();
    
    // Result should be approximately equal to A
    const comparison = try tensorsApproxEqual(result, tensor_obj, 1e-5, true);
    
    if (comparison == .err) {
        std.debug.print("Matrix multiplication identity test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        
        std.debug.print("Tensor shape: ", .{});
        for (tensor_obj.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        return error.TensorComparisonFailed;
    }
}

/// Helper function to calculate a numerical approximation of gradients using finite differences
/// This is useful for verifying that our autodiff gradients are correct
pub fn finiteDifference(
    allocator: Allocator, 
    op: anytype, 
    input: Tensor, 
    epsilon: f32,
    reduction_method: enum { sum, mean, none } 
) !Tensor {
    // Compute the forward result of the operation
    const forward = try op.forward(input);
    defer forward.deinit();
    
    // Create a result tensor to store the gradients
    var grad = try Tensor.zeros(allocator, input.shape.dims, input.dtype, .cpu);
    errdefer grad.deinit();
    
    // Handle different data types
    switch (input.dtype) {
        .f32 => try finiteDifferenceSpecific(f32, allocator, op, input, forward, epsilon, reduction_method, &grad),
        .f16 => {
            // For f16, we compute in f32 precision for stability
            var f32_input = try convertTensorToF32(allocator, input);
            defer f32_input.deinit();
            
            var f32_grad = try Tensor.zeros(allocator, input.shape.dims, .f32, .cpu);
            defer f32_grad.deinit();
            
            var f32_op_adapter = F32OperationAdapter{ .op = op };
            try finiteDifferenceSpecific(f32, allocator, &f32_op_adapter, f32_input, forward, epsilon, reduction_method, &f32_grad);
            
            // Convert back to f16
            try convertF32ToTensor(f32_grad, &grad);
        },
        .f64 => try finiteDifferenceSpecific(f64, allocator, op, input, forward, epsilon, reduction_method, &grad),
        .i32 => {
            // For integer types, convert to float, compute, and convert back
            var f32_input = try convertTensorToF32(allocator, input);
            defer f32_input.deinit();
            
            var f32_grad = try Tensor.zeros(allocator, input.shape.dims, .f32, .cpu);
            defer f32_grad.deinit();
            
            var f32_op_adapter = F32OperationAdapter{ .op = op };
            try finiteDifferenceSpecific(f32, allocator, &f32_op_adapter, f32_input, forward, epsilon, reduction_method, &f32_grad);
            
            // Convert back to i32
            try convertF32ToTensor(f32_grad, &grad);
        },
        else => return error.UnsupportedDataTypeForFiniteDifference,
    }
    
    return grad;
}

/// Converts a tensor of any supported type to an f32 tensor
fn convertTensorToF32(allocator: Allocator, input: Tensor) !Tensor {
    var result = try Tensor.zeros(allocator, input.shape.dims, .f32, .cpu);
    errdefer result.deinit();
    
    const elem_count = input.shape.elemCount();
    const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..elem_count];
    
    switch (input.dtype) {
        .f32 => {
            const input_buf = ptrCastHelper([*]f32, input.buffer.data.ptr)[0..elem_count];
            @memcpy(result_buf, input_buf);
        },
        .f16 => {
            const input_buf = ptrCastHelper([*]u16, input.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = @floatCast(f16ToF64(val));
            }
        },
        .f64 => {
            const input_buf = ptrCastHelper([*]f64, input.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = @floatCast(val);
            }
        },
        .i32 => {
            const input_buf = ptrCastHelper([*]i32, input.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = @floatFromInt(val);
            }
        },
        .i64 => {
            const input_buf = ptrCastHelper([*]i64, input.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = @floatFromInt(val);
            }
        },
        .bool => {
            const input_buf = ptrCastHelper([*]u8, input.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = if (val != 0) 1.0 else 0.0;
            }
        },
    }
    
    return result;
}

/// Converts an f32 tensor to the target tensor's data type
fn convertF32ToTensor(input: Tensor, result: *Tensor) !void {
    const elem_count = input.shape.elemCount();
    const input_buf = ptrCastHelper([*]f32, input.buffer.data.ptr)[0..elem_count];
    
    switch (result.dtype) {
        .f32 => {
            const result_buf = ptrCastHelper([*]f32, result.buffer.data.ptr)[0..elem_count];
            @memcpy(result_buf, input_buf);
        },
        .f16 => {
            const result_buf = ptrCastHelper([*]u16, result.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = f32ToF16Bits(val);
            }
        },
        .f64 => {
            const result_buf = ptrCastHelper([*]f64, result.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = val;
            }
        },
        .i32 => {
            const result_buf = ptrCastHelper([*]i32, result.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = @intFromFloat(val);
            }
        },
        .i64 => {
            const result_buf = ptrCastHelper([*]i64, result.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = @intFromFloat(val);
            }
        },
        .bool => {
            const result_buf = ptrCastHelper([*]u8, result.buffer.data.ptr)[0..elem_count];
            for (input_buf, 0..) |val, i| {
                result_buf[i] = if (val > 0.5) 1 else 0;
            }
        },
    }
}

/// Adapter that wraps an operation to always work with f32 inputs/outputs
const F32OperationAdapter = struct {
    op: anytype,
    
    pub fn forward(self: *F32OperationAdapter, input: Tensor) !Tensor {
        return self.op.forward(input);
    }
};

/// Type-specific implementation of finite difference
fn finiteDifferenceSpecific(
    comptime T: type,
    allocator: Allocator, 
    op: anytype, 
    input: Tensor, 
    forward: Tensor,
    epsilon: f32,
    reduction_method: enum { sum, mean, none },
    grad: *Tensor
) !void {
    // Access to input buffer
    const input_buf = ptrCastHelper([*]T, input.buffer.data.ptr)[0..input.shape.elemCount()];
    const grad_buf = ptrCastHelper([*]T, grad.buffer.data.ptr)[0..grad.shape.elemCount()];
    
    // Get output shape for reduction, if needed
    const needs_reduction = reduction_method != .none;
    const output_size = forward.shape.elemCount();
    
    // For each element in the input tensor
    for (0..input.shape.elemCount()) |i| {
        // Save the original value
        const orig_val = input_buf[i];
        
        // Use a smaller epsilon for larger values to avoid numerical issues
        const scaled_epsilon: T = if (@abs(orig_val) > 1.0) 
            @as(T, @floatCast(epsilon * @abs(orig_val))) 
        else 
            @floatCast(epsilon);
        
        // Perturb the input by epsilon
        input_buf[i] = orig_val + scaled_epsilon;
        
        // Compute the perturbed output
        const perturbed_forward = try op.forward(input);
        defer perturbed_forward.deinit();
        
        // Compute the finite difference gradient
        if (needs_reduction) {
            // For operations that reduce dimensionality (e.g., sum)
            // We need to accumulate gradient over all output elements
            const forward_buf = ptrCastHelper([*]T, forward.buffer.data.ptr)[0..forward.shape.elemCount()];
            const perturbed_buf = ptrCastHelper([*]T, perturbed_forward.buffer.data.ptr)[0..perturbed_forward.shape.elemCount()];
            
            var diff_sum: T = 0.0;
            for (forward_buf, perturbed_buf) |orig, pert| {
                diff_sum += pert - orig;
            }
            
            // Apply reduction method
            switch (reduction_method) {
                .sum => {
                    // Sum is already computed
                    grad_buf[i] = diff_sum / scaled_epsilon;
                },
                .mean => {
                    // Divide by output size to get mean
                    grad_buf[i] = diff_sum / (@as(T, @floatFromInt(output_size)) * scaled_epsilon);
                },
                .none => unreachable, // Should never happen
            }
        } else {
            // For element-wise operations, each output element corresponds to one input element
            const forward_buf = ptrCastHelper([*]T, forward.buffer.data.ptr)[0..forward.shape.elemCount()];
            const perturbed_buf = ptrCastHelper([*]T, perturbed_forward.buffer.data.ptr)[0..perturbed_forward.shape.elemCount()];
            
            // Compute the difference at this specific element
            if (i < forward.shape.elemCount()) {
                grad_buf[i] = (perturbed_buf[i] - forward_buf[i]) / scaled_epsilon;
            }
        }
        
        // Restore the original value
        input_buf[i] = orig_val;
    }
}

/// Enhanced finite difference function for binary operations
pub fn finiteDifferenceBinary(
    allocator: Allocator, 
    op: anytype, 
    a: Tensor, 
    b: Tensor, 
    input_index: u8, // 0 for a, 1 for b
    epsilon: f32,
    reduction_method: enum { sum, mean, none } 
) !Tensor {
    // Ensure valid input_index
    if (input_index != 0 and input_index != 1) {
        return error.InvalidInputIndex;
    }
    
    // The tensor we're computing gradients for
    const input = if (input_index == 0) a else b;
    const other = if (input_index == 0) b else a;
    
    // Forward pass
    const forward = try op.forward(.{ .a = a, .b = b });
    defer forward.deinit();
    
    // Create gradient tensor
    var grad = try Tensor.zeros(allocator, input.shape.dims, input.dtype, .cpu);
    errdefer grad.deinit();
    
    // Handle different data types (simplified to f32 only for now)
    if (input.dtype != .f32) {
        return error.UnsupportedDataTypeForFiniteDifference;
    }
    
    // Access buffers
    const input_buf = ptrCastHelper([*]f32, input.buffer.data.ptr)[0..input.shape.elemCount()];
    const grad_buf = ptrCastHelper([*]f32, grad.buffer.data.ptr)[0..grad.shape.elemCount()];
    
    // Get output shape for reduction, if needed
    const needs_reduction = reduction_method != .none;
    const output_size = forward.shape.elemCount();
    
    // For each element in the input tensor
    for (0..input.shape.elemCount()) |i| {
        // Save the original value
        const orig_val = input_buf[i];
        
        // Use a smaller epsilon for larger values
        const scaled_epsilon = if (@abs(orig_val) > 1.0) 
            epsilon * @abs(orig_val) 
        else 
            epsilon;
        
        // Perturb the input by epsilon
        input_buf[i] = orig_val + scaled_epsilon;
        
        // Compute the perturbed output
        const perturbed_forward = if (input_index == 0) 
            try op.forward(.{ .a = input, .b = other })
        else
            try op.forward(.{ .a = other, .b = input });
            
        defer perturbed_forward.deinit();
        
        // Compute the finite difference gradient
        if (needs_reduction) {
            // Accumulate over all output elements
            const forward_buf = ptrCastHelper([*]f32, forward.buffer.data.ptr)[0..forward.shape.elemCount()];
            const perturbed_buf = ptrCastHelper([*]f32, perturbed_forward.buffer.data.ptr)[0..perturbed_forward.shape.elemCount()];
            
            var diff_sum: f32 = 0.0;
            for (forward_buf, perturbed_buf) |orig, pert| {
                diff_sum += pert - orig;
            }
            
            // Apply reduction method
            switch (reduction_method) {
                .sum => {
                    grad_buf[i] = diff_sum / scaled_epsilon;
                },
                .mean => {
                    grad_buf[i] = diff_sum / (@as(f32, @floatFromInt(output_size)) * scaled_epsilon);
                },
                .none => unreachable,
            }
        } else {
            // Element-wise gradient
            const forward_buf = ptrCastHelper([*]f32, forward.buffer.data.ptr)[0..forward.shape.elemCount()];
            const perturbed_buf = ptrCastHelper([*]f32, perturbed_forward.buffer.data.ptr)[0..perturbed_forward.shape.elemCount()];
            
            // For operations where output shape doesn't match input shape
            if (i < forward.shape.elemCount()) {
                grad_buf[i] = (perturbed_buf[i] - forward_buf[i]) / scaled_epsilon;
            }
        }
        
        // Restore the original value
        input_buf[i] = orig_val;
    }
    
    return grad;
}

/// Check property: A divided by 1 equals A
pub fn checkDivideByOne(allocator: Allocator, rnd: *std.Random) !void {
    // Get config from shrinking state or use default
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    
    var tensor_obj = try randomTensor(
        allocator, 
        rnd, 
        config,
        .f32          // dtype
    );
    defer tensor_obj.deinit();
    
    // Skip test for empty tensors
    if (tensor_obj.shape.elemCount() == 0) {
        return;
    }
    
    // Create a tensor of ones with the same shape
    var ones = try Tensor.filled(allocator, tensor_obj.shape.dims, .f32, 1.0, .cpu);
    defer ones.deinit();
    
    // A / 1
    var result = try ops.divide(allocator, tensor_obj, ones);
    defer result.deinit();
    
    // Result should be approximately equal to A
    const comparison = try tensorsApproxEqual(result, tensor_obj, 1e-5, true);
    
    if (comparison == .err) {
        std.debug.print("Division by one test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        
        std.debug.print("Tensor shape: ", .{});
        for (tensor_obj.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        return error.TensorComparisonFailed;
    }
}

/// Check property: A/A = 1 for all non-zero A
pub fn checkDivisionByItself(allocator: Allocator, rnd: *std.Random) !void {
    // Get config from shrinking state or use default
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    // Disable edge cases for this test specifically
    config.edge_case_probability = 0;
    
    var tensor_obj = try randomTensor(
        allocator, 
        rnd, 
        config,
        .f32          // dtype
    );
    defer tensor_obj.deinit();
    
    // Make sure no values are too close to zero to avoid division issues
    const buf = ptrCastHelper([*]f32, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
    for (buf) |*x| {
        if (@abs(x.*) < 0.5) {
            x.* = if (x.* < 0) -0.5 else 0.5;
        }
    }
    
    // A / A
    var result = try ops.divide(allocator, tensor_obj, tensor_obj);
    defer result.deinit();
    
    // Create a tensor of ones with the same shape for comparison
    var ones = try Tensor.filled(allocator, tensor_obj.shape.dims, .f32, 1.0, .cpu);
    defer ones.deinit();
    
    // Result should be all ones
    const comparison = try tensorsApproxEqual(result, ones, 1e-5, true);
    
    if (comparison == .err) {
        std.debug.print("Division by itself test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        
        std.debug.print("Tensor shape: ", .{});
        for (tensor_obj.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        return error.TensorComparisonFailed;
    }
}

/// Check property: Gradient of addition (both inputs get the same gradient as the output)
pub fn checkAdditionGradient(allocator: Allocator, rnd: *std.Random) !void {
    const CpuBackend = ops.CpuBackend;
    
    // Get config from shrinking state or use default
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    
    // Create tensors with compatible shape for addition
    var pair = try randomTensorPair(
        allocator, 
        rnd, 
        config,
        null          // default dtype
    );
    defer pair.a.deinit();
    defer pair.b.deinit();
    
    // Skip test for empty tensors
    if (pair.a.shape.elemCount() == 0) {
        return;
    }

    // Set up the operation plan with autodiff
    const AddPlanType = autodiff.AddPlanWithGrad(CpuBackend, f32, null);
    var add_plan = autodiff.AutoDiffPlan(AddPlanType).init(allocator);
    defer add_plan.deinit(); // Make sure to clean up the plan
    
    // Forward pass
    const result = try add_plan.forward(.{ .a = pair.a, .b = pair.b });
    defer result.deinit();
    
    // Create gradient (ones for simplicity)
    var grad_out = try Tensor.filled(allocator, result.shape.dims, .f32, 1.0, .cpu);
    defer grad_out.deinit();
    
    // Backward pass
    const grads = try add_plan.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();
    
    // Check that both gradients are ones (gradient of add distributes equally)
    const ones = try Tensor.filled(allocator, pair.a.shape.dims, .f32, 1.0, .cpu);
    defer ones.deinit();
    
    const grad_a_comparison = try tensorsApproxEqual(grads.da, ones, 1e-5, true);
    const grad_b_comparison = try tensorsApproxEqual(grads.db, ones, 1e-5, true);
    
    if (grad_a_comparison == .err or grad_b_comparison == .err) {
        std.debug.print("Addition gradient test failed:\n", .{});
        if (grad_a_comparison == .err) {
            std.debug.print("Gradient for input A failed:\n", .{});
            printTensorErrorReport(grad_a_comparison.err);
        }
        if (grad_b_comparison == .err) {
            std.debug.print("Gradient for input B failed:\n", .{});
            printTensorErrorReport(grad_b_comparison.err);
        }
        
        std.debug.print("Tensor shapes: ", .{});
        for (pair.a.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        return error.GradientVerificationFailed;
    }
}

/// Check property: Gradient of matrix multiplication
pub fn checkMatmulGradient(allocator: Allocator, rnd: *std.Random) !void {
    const CpuBackend = ops.CpuBackend;
    
    // Create a and b with compatible shapes for matmul
    const m = rnd.intRangeAtMost(usize, 2, 5);
    const n = rnd.intRangeAtMost(usize, 2, 5);
    const p = rnd.intRangeAtMost(usize, 2, 5);
    
    var a = try Tensor.zeros(allocator, &[_]usize{ m, n }, .f32, .cpu);
    defer a.deinit();
    
    var b = try Tensor.zeros(allocator, &[_]usize{ n, p }, .f32, .cpu);
    defer b.deinit();
    
    // Fill with random values
    {
        const buf_a = ptrCastHelper([*]f32, a.buffer.data.ptr)[0..a.shape.elemCount()];
        for (buf_a) |*x| {
            x.* = rnd.float(f32) * 2.0 - 1.0;
        }
    }
    
    {
        const buf_b = ptrCastHelper([*]f32, b.buffer.data.ptr)[0..b.shape.elemCount()];
        for (buf_b) |*x| {
            x.* = rnd.float(f32) * 2.0 - 1.0;
        }
    }
    
    // Set up the operation plan with autodiff
    const MatmulPlanType = autodiff.MatmulPlanWithGrad(CpuBackend, f32, null, null, null);
    var matmul_plan = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan.deinit(); // Make sure to clean up the plan
    
    // Forward pass
    const result = try matmul_plan.forward(.{ .a = a, .b = b });
    defer result.deinit();
    
    // Create gradient (ones for simplicity)
    var grad_out = try Tensor.filled(allocator, result.shape.dims, .f32, 1.0, .cpu);
    defer grad_out.deinit();
    
    // Backward pass
    const grads = try matmul_plan.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();
    
    // Check gradient shapes match input shapes
    try std.testing.expectEqualSlices(usize, a.shape.dims, grads.da.shape.dims);
    try std.testing.expectEqualSlices(usize, b.shape.dims, grads.db.shape.dims);
    
    // Compute gradients manually to check against
    var b_transposed = try ops.transpose(allocator, b);
    defer b_transposed.deinit();
    
    var expected_grad_a = try ops.matmul(allocator, grad_out, b_transposed);
    defer expected_grad_a.deinit();
    
    var a_transposed = try ops.transpose(allocator, a);
    defer a_transposed.deinit();
    
    var expected_grad_b = try ops.matmul(allocator, a_transposed, grad_out);
    defer expected_grad_b.deinit();
    
    // Check gradients are correct
    const grad_a_comparison = try tensorsApproxEqual(grads.da, expected_grad_a, 1e-5, true);
    const grad_b_comparison = try tensorsApproxEqual(grads.db, expected_grad_b, 1e-5, true);
    
    if (grad_a_comparison == .err or grad_b_comparison == .err) {
        std.debug.print("Matrix multiplication gradient test failed:\n", .{});
        if (grad_a_comparison == .err) {
            std.debug.print("Gradient for input A failed:\n", .{});
            printTensorErrorReport(grad_a_comparison.err);
        }
        if (grad_b_comparison == .err) {
            std.debug.print("Gradient for input B failed:\n", .{});
            printTensorErrorReport(grad_b_comparison.err);
        }
        
        return error.GradientVerificationFailed;
    }
}

/// Check property: ReLU gradient is 0 for negative inputs, 1 for positive inputs
pub fn checkReLUGradient(allocator: Allocator, rnd: *std.Random) !void {
    const CpuBackend = ops.CpuBackend;
    
    // Get config from shrinking state or use default
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    
    // Create tensor with mix of positive and negative values
    var tensor_obj = try randomTensor(
        allocator, 
        rnd, 
        config,
        .f32          // dtype
    );
    defer tensor_obj.deinit();
    
    // Skip test for empty tensors
    if (tensor_obj.shape.elemCount() == 0) {
        return;
    }
    
    // Set up the operation plan with autodiff
    const ReluPlanType = autodiff.ReluPlanWithGrad(CpuBackend, f32, null);
    var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
    defer relu_plan.deinit(); // Make sure to clean up the plan
    
    // Forward pass
    const result = try relu_plan.forward(tensor_obj);
    defer result.deinit();
    
    // Create gradient (ones for simplicity)
    var grad_out = try Tensor.filled(allocator, result.shape.dims, .f32, 1.0, .cpu);
    defer grad_out.deinit();
    
    // Backward pass
    const grad = try relu_plan.backward(grad_out);
    defer grad.deinit();
    
    // Check gradient values are 0 for negative input values, 1 for positive inputs
    const input_buf = ptrCastHelper([*]f32, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
    const grad_buf = ptrCastHelper([*]f32, grad.buffer.data.ptr)[0..grad.shape.elemCount()];
    
    for (input_buf, grad_buf, 0..) |input, gradient, i| {
        if (input <= 0) {
            if (!std.math.approxEqAbs(f32, gradient, 0.0, 1e-5)) {
                std.debug.print("ReLU gradient test failed at index {}:\n", .{i});
                std.debug.print("Input: {d:.6}, Expected gradient: 0.0, Actual: {d:.6}\n", .{input, gradient});
                return error.GradientVerificationFailed;
            }
        } else {
            if (!std.math.approxEqAbs(f32, gradient, 1.0, 1e-5)) {
                std.debug.print("ReLU gradient test failed at index {}:\n", .{i});
                std.debug.print("Input: {d:.6}, Expected gradient: 1.0, Actual: {d:.6}\n", .{input, gradient});
                return error.GradientVerificationFailed;
            }
        }
    }
}

/// Check property: Gradient of divide
pub fn checkDivideGradient(allocator: Allocator, rnd: *std.Random) !void {
    const CpuBackend = ops.CpuBackend;
    
    // Create tensors with compatible shape for division
    var pair = try randomTensorPair(allocator, rnd, 5, null, null, null);
    defer pair.a.deinit();
    defer pair.b.deinit();
    
    // Make sure b doesn't have values close to zero to avoid division issues
    const b_buf = ptrCastHelper([*]f32, pair.b.buffer.data.ptr)[0..pair.b.shape.elemCount()];
    for (b_buf) |*x| {
        if (@abs(x.*) < 0.5) {
            x.* = if (x.* < 0) -0.5 else 0.5;
        }
    }
    
    // Set up the operation plan with autodiff
    const DividePlanType = autodiff.DividePlanWithGrad(CpuBackend, f32, null);
    var divide_plan = autodiff.AutoDiffPlan(DividePlanType).init(allocator);
    defer divide_plan.deinit(); // Make sure to clean up the plan
    
    // Forward pass
    const result = try divide_plan.forward(.{ .a = pair.a, .b = pair.b });
    defer result.deinit();
    
    // Create gradient (ones for simplicity)
    var grad_out = try Tensor.filled(allocator, result.shape.dims, .f32, 1.0, .cpu);
    defer grad_out.deinit();
    
    // Backward pass
    const grads = try divide_plan.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();
    
    // Check gradient shapes match input shapes
    try std.testing.expectEqualSlices(usize, pair.a.shape.dims, grads.da.shape.dims);
    try std.testing.expectEqualSlices(usize, pair.b.shape.dims, grads.db.shape.dims);
    
    // Compute gradients manually to check against
    // da = grad_out / b
    var expected_grad_a = try ops.divide(allocator, grad_out, pair.b);
    defer expected_grad_a.deinit();
    
    // db = -grad_out * a / (b * b)
    var b_squared = try ops.multiply(allocator, pair.b, pair.b);
    defer b_squared.deinit();
    
    var a_div_b_squared = try ops.divide(allocator, pair.a, b_squared);
    defer a_div_b_squared.deinit();
    
    var temp = try ops.multiply(allocator, grad_out, a_div_b_squared);
    defer temp.deinit();
    
    var negative_one = try Tensor.filled(allocator, temp.shape.dims, .f32, -1.0, .cpu);
    defer negative_one.deinit();
    
    var expected_grad_b = try ops.multiply(allocator, temp, negative_one);
    defer expected_grad_b.deinit();
    
    // Check gradients are correct with a slightly larger epsilon due to division
    const grad_a_comparison = try tensorsApproxEqual(grads.da, expected_grad_a, 1e-4, true);
    const grad_b_comparison = try tensorsApproxEqual(grads.db, expected_grad_b, 1e-4, true);
    
    if (grad_a_comparison == .err or grad_b_comparison == .err) {
        std.debug.print("Division gradient test failed:\n", .{});
        if (grad_a_comparison == .err) {
            std.debug.print("Gradient for input A failed:\n", .{});
            printTensorErrorReport(grad_a_comparison.err);
        }
        if (grad_b_comparison == .err) {
            std.debug.print("Gradient for input B failed:\n", .{});
            printTensorErrorReport(grad_b_comparison.err);
        }
        
        return error.GradientVerificationFailed;
    }
}

//
// Finite difference gradient test
//

/// Check gradient correctness using finite differences
pub fn checkGradientWithFiniteDifferences(allocator: Allocator, rnd: *std.Random) !void {
    const CpuBackend = ops.CpuBackend;
    
    // Use configuration from default or global state if we're shrinking
    var config = if (shrinking_state.is_shrinking) shrinking_state.config else default_config;
    
    // Create a tensor for testing gradients with specified data type
    // Try multiple data types to ensure broad coverage
    const test_dtype = blk: {
        const dtypes = [_]DType{ .f32, .f16, .f64 };
        const idx = rnd.uintLessThan(usize, dtypes.len);
        break :blk dtypes[idx];
    };
    
    var test_tensor = try randomTensor(
        allocator, 
        rnd,
        config,
        test_dtype
    );
    defer test_tensor.deinit();
    
    // Skip test for empty tensors
    if (test_tensor.shape.elemCount() == 0) {
        return;
    }
    
    // Apply shrinking if active
    if (shrinking_state.is_shrinking) {
        applyShrinking(&test_tensor);
    }
    
    // Measure performance if enabled
    const start_time = if (config.measure_performance) std.time.nanoTimestamp() else 0;
    
    // Test ReLU operation (simple element-wise operation)
    try testOperationGradient(
        allocator,
        "ReLU",
        test_tensor,
        rnd,
        config.finite_diff_epsilon
    );
    
    // If we have a 2D tensor, test matmul too
    if (test_tensor.shape.rank() == 2 and test_tensor.shape.elemCount() > 0) {
        // Create a compatible second tensor for matrix multiplication
        const n = test_tensor.shape.dims[1];
        const p = rnd.intRangeAtMost(usize, 1, config.max_dim);
        
        var second_tensor = try Tensor.zeros(allocator, &[_]usize{ n, p }, test_tensor.dtype, .cpu);
        defer second_tensor.deinit();
        
        // Fill with random values
        switch (test_tensor.dtype) {
            .f32 => {
                const buf = ptrCastHelper([*]f32, second_tensor.buffer.data.ptr)[0..second_tensor.shape.elemCount()];
                for (buf) |*x| {
                    x.* = rnd.float(f32) * 2.0 - 1.0;
                }
            },
            .f16 => {
                const buf = ptrCastHelper([*]u16, second_tensor.buffer.data.ptr)[0..second_tensor.shape.elemCount()];
                for (buf) |*x| {
                    x.* = f32ToF16Bits(rnd.float(f32) * 2.0 - 1.0);
                }
            },
            .f64 => {
                const buf = ptrCastHelper([*]f64, second_tensor.buffer.data.ptr)[0..second_tensor.shape.elemCount()];
                for (buf) |*x| {
                    x.* = rnd.float(f64) * 2.0 - 1.0;
                }
            },
            else => {
                // Skip test for non-float types
                return;
            },
        }
        
        // Test matrix multiplication gradients
        try testBinaryOperationGradient(
            allocator,
            "MatMul",
            test_tensor,
            second_tensor,
            rnd,
            config.matmul_epsilon // Use larger epsilon for matmul
        );
    }
    
    // Log performance information if enabled
    if (config.measure_performance) {
        const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
        var performance = PerformanceMeasurement{
            .operation_name = "Gradient Verification",
            .elapsed_ns = elapsed,
            .tensor_sizes = test_tensor.shape.dims,
        };
        performance.print();
    }
}

/// Test gradient verification for a unary operation
fn testOperationGradient(
    allocator: Allocator,
    op_name: []const u8,
    test_tensor: Tensor,
    rnd: *std.Random,
    epsilon: f32
) !void {
    const CpuBackend = ops.CpuBackend;
    
    // Skip test for empty tensors
    if (test_tensor.shape.elemCount() == 0) {
        return;
    }
    
    // Select operation based on name
    if (std.mem.eql(u8, op_name, "ReLU")) {
        // Create a simple relu operation plan
        const ReluPlanType = autodiff.ReluPlanWithGrad(CpuBackend, f32, null);
        var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
        defer relu_plan.deinit();
        
        // Create a copy of test_tensor that we'll use for autodiff and for finite differences
        var test_copy_autodiff = try test_tensor.clone();
        defer test_copy_autodiff.deinit();
        
        var test_copy_finite = try test_tensor.clone();
        defer test_copy_finite.deinit();
        
        // Forward pass
        const result = try relu_plan.forward(test_copy_autodiff);
        defer result.deinit();
        
        // Create gradient (ones for simplicity)
        var grad_out = try Tensor.filled(allocator, result.shape.dims, result.dtype, 1.0, .cpu);
        defer grad_out.deinit();
        
        // Backward pass to get autodiff gradients
        const autodiff_grad = try relu_plan.backward(grad_out);
        defer autodiff_grad.deinit();
        
        // Calculate numerical gradient using finite differences
        const reduction_method = .none; // ReLU is element-wise, no reduction needed
        var numerical_grad = try finiteDifference(allocator, &relu_plan, test_copy_finite, epsilon, reduction_method);
        defer numerical_grad.deinit();
        
        // Compare the autodiff gradient with the numerical gradient
        const comparison = try tensorsApproxEqual(autodiff_grad, numerical_grad, epsilon, true);
        
        if (comparison == .err) {
            std.debug.print("ReLU finite difference gradient test failed:\n", .{});
            printTensorErrorReport(comparison.err);
            
            // Get the input tensor value at the failure point
            const input_buf = if (test_tensor.dtype == .f32)
                ptrCastHelper([*]f32, test_tensor.buffer.data.ptr)
            else
                return error.UnsupportedDataTypeForErrorReporting;
                
            const input_val = input_buf[comparison.err.failure_index];
            
            std.debug.print("Input value at failure: {d:.6}\n", .{input_val});
            std.debug.print("This failure is expected near activation boundaries (e.g., x ≈ 0 for ReLU)\n", .{});
            
            // ReLU has discontinuous gradient at x=0, so we should tolerate errors near 0
            if (@abs(input_val) < 1e-3) {
                std.debug.print("Ignoring error near ReLU activation boundary (x ≈ 0)\n", .{});
                return;
            }
            
            std.debug.print("Tensor shape: ", .{});
            for (test_tensor.shape.dims) |d| {
                std.debug.print("{} ", .{d});
            }
            std.debug.print("\n", .{});
            
            return error.GradientVerificationFailed;
        }
    } else {
        std.debug.print("Operation {s} not implemented for gradient testing\n", .{op_name});
    }
}

/// Test gradient verification for a binary operation
fn testBinaryOperationGradient(
    allocator: Allocator,
    op_name: []const u8,
    a: Tensor,
    b: Tensor,
    rnd: *std.Random,
    epsilon: f32
) !void {
    const CpuBackend = ops.CpuBackend;
    
    // Skip test for empty tensors
    if (a.shape.elemCount() == 0 or b.shape.elemCount() == 0) {
        return;
    }
    
    // Select operation based on name
    if (std.mem.eql(u8, op_name, "MatMul")) {
        // Create matmul operation plan
        const MatmulPlanType = autodiff.MatmulPlanWithGrad(CpuBackend, f32, null, null, null);
        var matmul_plan = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
        defer matmul_plan.deinit();
        
        // Create copies for testing
        var a_copy = try a.clone();
        defer a_copy.deinit();
        
        var b_copy = try b.clone();
        defer b_copy.deinit();
        
        // Forward pass
        const result = try matmul_plan.forward(.{ .a = a_copy, .b = b_copy });
        defer result.deinit();
        
        // Create gradient (ones for simplicity)
        var grad_out = try Tensor.filled(allocator, result.shape.dims, result.dtype, 1.0, .cpu);
        defer grad_out.deinit();
        
        // Backward pass to get autodiff gradients
        const grads = try matmul_plan.backward(grad_out);
        defer grads.da.deinit();
        defer grads.db.deinit();
        
        // Calculate numerical gradients using finite differences
        // Input a gradient
        var a_copy_fd = try a.clone();
        defer a_copy_fd.deinit();
        
        var b_copy_fd = try b.clone();
        defer b_copy_fd.deinit();
        
        var numerical_grad_a = try finiteDifferenceBinary(
            allocator,
            &matmul_plan,
            a_copy_fd,
            b_copy_fd,
            0, // Input index for a
            epsilon,
            .none // MatMul preserves dimensions in a special way
        );
        defer numerical_grad_a.deinit();
        
        // Calculate numerical gradient for b
        var a_copy_fd2 = try a.clone();
        defer a_copy_fd2.deinit();
        
        var b_copy_fd2 = try b.clone();
        defer b_copy_fd2.deinit();
        
        var numerical_grad_b = try finiteDifferenceBinary(
            allocator,
            &matmul_plan,
            a_copy_fd2,
            b_copy_fd2,
            1, // Input index for b
            epsilon,
            .none // MatMul preserves dimensions in a special way
        );
        defer numerical_grad_b.deinit();
        
        // Compare gradients
        // A gradient comparison
        const a_comparison = try tensorsApproxEqual(grads.da, numerical_grad_a, epsilon * 10, true);
        if (a_comparison == .err) {
            std.debug.print("MatMul gradient test for A failed:\n", .{});
            printTensorErrorReport(a_comparison.err);
            
            // Special exemption for very small values
            const a_val = a_comparison.err.a_value;
            const b_val = a_comparison.err.b_value;
            
            if (@abs(a_val) < 1e-6 and @abs(b_val) < 1e-6) {
                std.debug.print("Ignoring error for very small values\n", .{});
                return;
            }
            
            return error.GradientVerificationFailedForA;
        }
        
        // B gradient comparison
        const b_comparison = try tensorsApproxEqual(grads.db, numerical_grad_b, epsilon * 10, true);
        if (b_comparison == .err) {
            std.debug.print("MatMul gradient test for B failed:\n", .{});
            printTensorErrorReport(b_comparison.err);
            
            // Special exemption for very small values
            const a_val = b_comparison.err.a_value;
            const b_val = b_comparison.err.b_value;
            
            if (@abs(a_val) < 1e-6 and @abs(b_val) < 1e-6) {
                std.debug.print("Ignoring error for very small values\n", .{});
                return;
            }
            
            return error.GradientVerificationFailedForB;
        }
    } else {
        std.debug.print("Binary operation {s} not implemented for gradient testing\n", .{op_name});
    }
}

//
// Test runners
//

/// Configuration for standard tests
fn standardTestConfig() PropertyTestConfig {
    return PropertyTestConfig{
        .iterations = 50,
        .seed = 42,
        .verbose = false,
        .epsilon = 1e-5,
        .matmul_epsilon = 1e-4,
        .finite_diff_epsilon = 1e-2,
        .measure_performance = false,
        .enable_empty_tensors = true, // Enable empty tensors for thorough testing
        .max_rank = 4,
        .max_dim = 5,
        .edge_case_probability = 0.1,
    };
}

/// Configuration for performance testing
fn perfTestConfig() PropertyTestConfig {
    return PropertyTestConfig{
        .iterations = 3,
        .seed = 42,
        .verbose = true,
        .epsilon = 1e-5,
        .matmul_epsilon = 1e-4,
        .finite_diff_epsilon = 1e-2,
        .measure_performance = true,
        .enable_empty_tensors = false, // Disable for performance testing
        .max_rank = 3,
        .max_dim = 16, // Larger tensors to measure performance
        .edge_case_probability = 0.0, // No edge cases for performance testing
    };
}

/// Configuration for gradient tests
fn gradientTestConfig() PropertyTestConfig {
    return PropertyTestConfig{
        .iterations = 10, // Fewer iterations since gradient tests are more expensive
        .seed = 42,
        .verbose = false,
        .epsilon = 1e-5,
        .matmul_epsilon = 1e-4,
        .finite_diff_epsilon = 1e-2,
        .measure_performance = false,
        .enable_empty_tensors = false, // Disable for gradient tests
        .max_rank = 3,
        .max_dim = 4, // Smaller tensors for gradient tests
        .edge_case_probability = 0.1,
    };
}

test "property-based addition commutativity" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkAdditionCommutativity,
        standardTestConfig()
    );
}

test "property-based addition-subtraction cancellation" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkAddSubtractCancellation,
        standardTestConfig()
    );
}

test "property-based matmul associativity" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkMatmulAssociativity,
        standardTestConfig()
    );
}

test "property-based matmul distributivity" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkMatmulDistributivity,
        standardTestConfig()
    );
}

test "property-based relu non-negativity" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkReLUNonNegativity,
        standardTestConfig()
    );
}

test "property-based matmul identity" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkMatmulIdentity,
        standardTestConfig()
    );
}

test "property-based divide by one" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkDivideByOne,
        standardTestConfig()
    );
}

test "property-based division by itself" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkDivisionByItself,
        standardTestConfig()
    );
}

test "property-based finite difference gradient verification" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkGradientWithFiniteDifferences,
        gradientTestConfig()
    );
}

// Autodiff property tests

test "property-based addition gradient" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkAdditionGradient,
        gradientTestConfig()
    );
}

test "property-based matmul gradient" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkMatmulGradient,
        gradientTestConfig()
    );
}

test "property-based relu gradient" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkReLUGradient,
        gradientTestConfig()
    );
}

test "property-based divide gradient" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkDivideGradient,
        gradientTestConfig()
    );
}

// Performance tests - these are disabled by default but can be enabled with the "perf" build flag
test "performance-test-matmul" {
    if (!@import("builtin").is_test) return error.SkipZigTest;
    
    const allocator = std.testing.allocator;
    if (@hasDecl(@import("root"), "perf_test") and @import("root").perf_test) {
        try runPropertyTest(
            allocator,
            checkMatmulAssociativity,
            perfTestConfig()
        );
    }
}

test "performance-test-gradients" {
    if (!@import("builtin").is_test) return error.SkipZigTest;
    
    const allocator = std.testing.allocator;
    if (@hasDecl(@import("root"), "perf_test") and @import("root").perf_test) {
        var config = perfTestConfig();
        config.max_dim = 8; // Smaller for gradient tests which are more expensive
        
        try runPropertyTest(
            allocator,
            checkGradientWithFiniteDifferences,
            config
        );
    }
}