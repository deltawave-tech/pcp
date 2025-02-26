const std = @import("std");
const tensor = @import("tensor.zig");

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
};

/// Add two tensors element-wise
pub fn add(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
    // Check if shapes match
    if (!a.shape.eql(b.shape)) {
        return OpError.ShapeMismatch;
    }
    
    // Check if dtypes match
    if (a.dtype != b.dtype) {
        return OpError.UnsupportedDataType;
    }
    
    // Create result tensor
    var result = try Tensor.zeros(allocator, a.shape.dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @as([*]f32, @ptrCast(@alignCast(a.buffer.data.ptr)))[0..a.shape.elemCount()];
            const b_buf = @as([*]f32, @ptrCast(@alignCast(b.buffer.data.ptr)))[0..b.shape.elemCount()];
            const result_buf = @as([*]f32, @ptrCast(@alignCast(result.buffer.data.ptr)))[0..result.shape.elemCount()];
            
            for (a_buf, 0..) |a_val, i| {
                result_buf[i] = a_val + b_buf[i];
            }
        },
        // Implement other dtype cases as needed
        else => {
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}

/// Subtract two tensors element-wise
pub fn subtract(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
    // Check if shapes match
    if (!a.shape.eql(b.shape)) {
        return OpError.ShapeMismatch;
    }
    
    // Check if dtypes match
    if (a.dtype != b.dtype) {
        return OpError.UnsupportedDataType;
    }
    
    // Create result tensor
    var result = try Tensor.zeros(allocator, a.shape.dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @as([*]f32, @ptrCast(@alignCast(a.buffer.data.ptr)))[0..a.shape.elemCount()];
            const b_buf = @as([*]f32, @ptrCast(@alignCast(b.buffer.data.ptr)))[0..b.shape.elemCount()];
            const result_buf = @as([*]f32, @ptrCast(@alignCast(result.buffer.data.ptr)))[0..result.shape.elemCount()];
            
            for (a_buf, 0..) |a_val, i| {
                result_buf[i] = a_val - b_buf[i];
            }
        },
        // Implement other dtype cases as needed
        else => {
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}

/// Multiply two tensors element-wise (Hadamard product)
pub fn multiply(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
    // Check if shapes match
    if (!a.shape.eql(b.shape)) {
        return OpError.ShapeMismatch;
    }
    
    // Check if dtypes match
    if (a.dtype != b.dtype) {
        return OpError.UnsupportedDataType;
    }
    
    // Create result tensor
    var result = try Tensor.zeros(allocator, a.shape.dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @as([*]f32, @ptrCast(@alignCast(a.buffer.data.ptr)))[0..a.shape.elemCount()];
            const b_buf = @as([*]f32, @ptrCast(@alignCast(b.buffer.data.ptr)))[0..b.shape.elemCount()];
            const result_buf = @as([*]f32, @ptrCast(@alignCast(result.buffer.data.ptr)))[0..result.shape.elemCount()];
            
            for (a_buf, 0..) |a_val, i| {
                result_buf[i] = a_val * b_buf[i];
            }
        },
        // Implement other dtype cases as needed
        else => {
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}

/// Divide two tensors element-wise
pub fn divide(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
    // Check if shapes match
    if (!a.shape.eql(b.shape)) {
        return OpError.ShapeMismatch;
    }
    
    // Check if dtypes match
    if (a.dtype != b.dtype) {
        return OpError.UnsupportedDataType;
    }
    
    // Create result tensor
    var result = try Tensor.zeros(allocator, a.shape.dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @as([*]f32, @ptrCast(@alignCast(a.buffer.data.ptr)))[0..a.shape.elemCount()];
            const b_buf = @as([*]f32, @ptrCast(@alignCast(b.buffer.data.ptr)))[0..b.shape.elemCount()];
            const result_buf = @as([*]f32, @ptrCast(@alignCast(result.buffer.data.ptr)))[0..result.shape.elemCount()];
            
            for (a_buf, 0..) |a_val, i| {
                // Division by zero check could be added here
                result_buf[i] = a_val / b_buf[i];
            }
        },
        // Implement other dtype cases as needed
        else => {
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}

/// Matrix multiplication (for 2D tensors)
pub fn matmul(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
    // Check dimensions for matrix multiplication: (m, n) x (n, p) -> (m, p)
    if (a.shape.rank() != 2 or b.shape.rank() != 2) {
        return OpError.DimensionMismatch;
    }
    
    if (a.shape.dims[1] != b.shape.dims[0]) {
        return OpError.ShapeMismatch;
    }
    
    // Check if dtypes match
    if (a.dtype != b.dtype) {
        return OpError.UnsupportedDataType;
    }
    
    // Result dimensions: [a.dims[0], b.dims[1]]
    const result_dims = [_]usize{ a.shape.dims[0], b.shape.dims[1] };
    const result = try Tensor.zeros(allocator, &result_dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @as([*]f32, @ptrCast(@alignCast(a.buffer.data.ptr)));
            const b_buf = @as([*]f32, @ptrCast(@alignCast(b.buffer.data.ptr)));
            const result_buf = @as([*]f32, @ptrCast(@alignCast(result.buffer.data.ptr)));
            
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
        },
        // Implement other dtype cases as needed
        else => {
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}

/// Apply ReLU activation function element-wise
pub fn relu(allocator: Allocator, a: Tensor) !Tensor {
    var result = try Tensor.zeros(allocator, a.shape.dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @as([*]f32, @ptrCast(@alignCast(a.buffer.data.ptr)))[0..a.shape.elemCount()];
            const result_buf = @as([*]f32, @ptrCast(@alignCast(result.buffer.data.ptr)))[0..result.shape.elemCount()];
            
            for (a_buf, 0..) |a_val, i| {
                result_buf[i] = if (a_val > 0) a_val else 0;
            }
        },
        // Implement other dtype cases as needed
        else => {
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}

/// Softmax activation (for 2D tensors, operating on last dimension)
pub fn softmax(allocator: Allocator, a: Tensor) !Tensor {
    if (a.shape.rank() != 2) {
        return OpError.DimensionMismatch;
    }
    
    const result = try Tensor.zeros(allocator, a.shape.dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @as([*]f32, @ptrCast(@alignCast(a.buffer.data.ptr)));
            const result_buf = @as([*]f32, @ptrCast(@alignCast(result.buffer.data.ptr)));
            
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
        },
        // Implement other dtype cases as needed
        else => {
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}

/// Transpose a 2D tensor
pub fn transpose(allocator: Allocator, a: Tensor) !Tensor {
    if (a.shape.rank() != 2) {
        return OpError.DimensionMismatch;
    }
    
    // Result dimensions are swapped
    const result_dims = [_]usize{ a.shape.dims[1], a.shape.dims[0] };
    const result = try Tensor.zeros(allocator, &result_dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @as([*]f32, @ptrCast(@alignCast(a.buffer.data.ptr)));
            const result_buf = @as([*]f32, @ptrCast(@alignCast(result.buffer.data.ptr)));
            
            const rows = a.shape.dims[0];
            const cols = a.shape.dims[1];
            
            // Note: changing to const here would mean we can't use the variable name in the nested loop
            // Zig requires loop variables to be mutable
            for (0..rows) |i| {
                for (0..cols) |j| {
                    result_buf[j * rows + i] = a_buf[i * cols + j];
                }
            }
        },
        // Implement other dtype cases as needed
        else => {
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}

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
    
    // Test addition: a + b = [[6, 8], [10, 12]]
    var result = try add(allocator, a, b);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 8.0), try result.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 10.0), try result.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 12.0), try result.getScalar(&[_]usize{1, 1}));
    
    // Test subtraction: a - b = [[-4, -4], [-4, -4]]
    var result2 = try subtract(allocator, a, b);
    defer result2.deinit();
    
    try std.testing.expectEqual(@as(f32, -4.0), try result2.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, -4.0), try result2.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, -4.0), try result2.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, -4.0), try result2.getScalar(&[_]usize{1, 1}));
    
    // Test element-wise multiplication: a * b = [[5, 12], [21, 32]]
    var result3 = try multiply(allocator, a, b);
    defer result3.deinit();
    
    try std.testing.expectEqual(@as(f32, 5.0), try result3.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 12.0), try result3.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 21.0), try result3.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 32.0), try result3.getScalar(&[_]usize{1, 1}));
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
    
    // Matrix multiplication: a * b = [[58, 64], [139, 154]]
    var result = try matmul(allocator, a, b);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(f32, 58.0), try result.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 64.0), try result.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 139.0), try result.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 154.0), try result.getScalar(&[_]usize{1, 1}));
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
    
    // ReLU: max(0, x) = [[0, 2], [0, 3]]
    var result = try relu(allocator, a);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 0.0), try result.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{1, 1}));
    
    // Create a 2x2 tensor: [[1, 2], [3, 4]]
    var b = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer b.deinit();
    
    try b.setScalar(&[_]usize{0, 0}, 1.0);
    try b.setScalar(&[_]usize{0, 1}, 2.0);
    try b.setScalar(&[_]usize{1, 0}, 3.0);
    try b.setScalar(&[_]usize{1, 1}, 4.0);
    
    // Softmax on each row
    var result2 = try softmax(allocator, b);
    defer result2.deinit();
    
    // First row: softmax([1, 2]) = [0.269, 0.731]
    // Second row: softmax([3, 4]) = [0.269, 0.731]
    const first_row_0 = try result2.getScalar(&[_]usize{0, 0});
    const first_row_1 = try result2.getScalar(&[_]usize{0, 1});
    const second_row_0 = try result2.getScalar(&[_]usize{1, 0});
    const second_row_1 = try result2.getScalar(&[_]usize{1, 1});
    
    // Check that softmax outputs sum to 1 for each row
    try std.testing.expectApproxEqRel(@as(f32, 1.0), first_row_0 + first_row_1, 0.0001);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), second_row_0 + second_row_1, 0.0001);
    
    // Check that second element is larger than first in each row
    try std.testing.expect(first_row_1 > first_row_0);
    try std.testing.expect(second_row_1 > second_row_0);
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
    
    // Transpose: [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
    var result = try transpose(allocator, a);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(f32, 1.0), try result.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 4.0), try result.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 2.0), try result.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), try result.getScalar(&[_]usize{1, 1}));
    try std.testing.expectEqual(@as(f32, 3.0), try result.getScalar(&[_]usize{2, 0}));
    try std.testing.expectEqual(@as(f32, 6.0), try result.getScalar(&[_]usize{2, 1}));
}