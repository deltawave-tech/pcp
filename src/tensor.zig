const std = @import("std");
const Allocator = std.mem.Allocator;

/// Helper function for pointer casting
fn ptrCastHelper(comptime T: type, ptr: anytype) T {
    // We still need to use alignCast for pointers that require higher alignment
    return @ptrCast(@alignCast(ptr));
}

/// Supported data types for tensors
pub const DType = enum {
    f16,
    f32,
    f64,
    i32,
    i64,
    bool,
    
    pub fn sizeInBytes(self: DType) usize {
        return switch (self) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
            .i32 => 4,
            .i64 => 8,
            .bool => 1,
        };
    }
};

/// Tensor shape information
pub const Shape = struct {
    dims: []const usize,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, dimensions: []const usize) !Shape {
        const dims_copy = try allocator.dupe(usize, dimensions);
        return Shape{
            .dims = dims_copy,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Shape) void {
        self.allocator.free(self.dims);
    }
    
    pub fn rank(self: Shape) usize {
        return self.dims.len;
    }
    
    pub fn elemCount(self: Shape) usize {
        if (self.dims.len == 0) return 0;
        
        var result: usize = 1;
        for (self.dims) |dim| {
            // Check for potential overflow before multiplying
            if (dim > 0 and result > std.math.maxInt(usize) / dim) {
                std.debug.print("Warning: Potential integer overflow in elemCount: {} * {} would overflow\n", 
                    .{result, dim});
                return std.math.maxInt(usize);
            }
            result *= dim;
        }
        return result;
    }
    
    pub fn bytesRequired(self: Shape, dtype: DType) usize {
        const count = self.elemCount();
        const size = dtype.sizeInBytes();
        
        // Check for potential overflow before multiplying
        if (count > 0 and size > 0 and count > std.math.maxInt(usize) / size) {
            std.debug.print("Warning: Potential integer overflow in bytesRequired: {} * {} would overflow\n", 
                .{count, size});
            return std.math.maxInt(usize);
        }
        
        return count * size;
    }
    
    pub fn eql(self: Shape, other: Shape) bool {
        if (self.dims.len != other.dims.len) return false;
        
        for (self.dims, 0..) |dim, i| {
            if (dim != other.dims[i]) return false;
        }
        
        return true;
    }
};

/// Memory buffer for storing tensor data
pub const Buffer = struct {
    data: []u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, size: usize) !Buffer {
        const data = try allocator.alloc(u8, size);
        return Buffer{
            .data = data,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Buffer) void {
        self.allocator.free(self.data);
    }
};

/// The backend type that will process tensor operations
pub const BackendType = enum {
    cpu,
    metal,
    // Will add more backends later (cuda, rocm, etc.)
};

/// Core tensor struct that represents an n-dimensional array
pub const Tensor = struct {
    shape: Shape,
    dtype: DType,
    buffer: Buffer,
    requires_grad: bool = false,
    backend: BackendType = .cpu,
    ref_count: *usize, // Reference count
    allocator: Allocator, // Store allocator for reference count management
    
    /// Initialize a tensor with a shape and data type
    pub fn init(allocator: Allocator, dims: []const usize, dtype: DType, backend: BackendType) !Tensor {
        const shape = try Shape.init(allocator, dims);
        const buffer_size = shape.bytesRequired(dtype);
        
        // Check if we had an overflow
        if (buffer_size == std.math.maxInt(usize)) {
            // Clean up shape to avoid memory leak
            var shape_copy = shape;
            shape_copy.deinit();
            return error.TensorTooLarge;
        }
        
        const buffer = try Buffer.init(allocator, buffer_size);
        
        // Create and initialize a reference count
        const ref_count = try allocator.create(usize);
        ref_count.* = 1;
        
        return Tensor{
            .shape = shape,
            .dtype = dtype,
            .buffer = buffer,
            .backend = backend,
            .ref_count = ref_count,
            .allocator = allocator,
        };
    }
    
    /// Create a copy of a tensor with shared data but independent reference count
    pub fn initShared(self: Tensor) !Tensor {
        // Increment the reference count
        self.ref_count.* += 1;
        
        return self;
    }
    
    /// Create a deep copy of a tensor with completely independent memory
    pub fn clone(self: Tensor) !Tensor {
        var new_tensor = try init(self.allocator, self.shape.dims, self.dtype, self.backend);
        
        // Copy the data buffer contents
        @memcpy(new_tensor.buffer.data, self.buffer.data[0..self.buffer.data.len]);
        
        // Copy requires_grad flag
        new_tensor.requires_grad = self.requires_grad;
        
        return new_tensor;
    }
    
    /// Reshape a tensor to a new shape, preserving data
    pub fn reshape(self: Tensor, allocator: Allocator, new_dims: []const usize) !Tensor {
        var elem_count: usize = 1;
        for (new_dims) |dim| {
            elem_count *= dim;
        }
        
        if (elem_count != self.shape.elemCount()) {
            return error.ShapeMismatch;
        }
        
        var new_tensor = try Tensor.init(allocator, new_dims, self.dtype, self.backend);
        errdefer new_tensor.deinit();
        
        @memcpy(new_tensor.buffer.data, self.buffer.data[0..self.buffer.data.len]);
        new_tensor.requires_grad = self.requires_grad;
        
        return new_tensor;
    }
    
    /// Initialize a tensor with a specific value
    pub fn filled(allocator: Allocator, dims: []const usize, dtype: DType, value: anytype, backend: BackendType) !Tensor {
        var tensor = try init(allocator, dims, dtype, backend);
        
        // Fill the buffer with the value based on dtype
        switch (dtype) {
            .f32 => {
                const val: f32 = @floatCast(value);
                const f32_buf = ptrCastHelper([*]f32, tensor.buffer.data.ptr)[0..tensor.shape.elemCount()];
                for (f32_buf) |*ptr| {
                    ptr.* = val;
                }
            },
            // Implement other dtype cases as needed
            else => {
                // Clean up resources on error
                tensor.deinit();
                return error.UnsupportedDataTypeForFill;
            }
        }
        
        return tensor;
    }
    
    /// Initialize a tensor with zeros
    pub fn zeros(allocator: Allocator, dims: []const usize, dtype: DType, backend: BackendType) !Tensor {
        const tensor = try init(allocator, dims, dtype, backend);
        
        // Fill with zeros
        @memset(tensor.buffer.data, 0);
        
        return tensor;
    }
    
    /// Create a tensor with random values
    pub fn random(allocator: Allocator, dims: []const usize, dtype: DType, backend: BackendType) !Tensor {
        const tensor = try init(allocator, dims, dtype, backend);
        // Use a timestamp cast to unsigned value for seed
        const seed = @as(u64, @bitCast(@as(i64, std.time.milliTimestamp())));
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        
        switch (dtype) {
            .f32 => {
                const f32_buf = ptrCastHelper([*]f32, tensor.buffer.data.ptr)[0..tensor.shape.elemCount()];
                for (f32_buf) |*ptr| {
                    ptr.* = rand.float(f32);
                }
            },
            // Implement other dtype cases as needed
            else => {
                // Clean up resources on error
                tensor.deinit();
                return error.UnsupportedDataTypeForRandom;
            }
        }
        
        return tensor;
    }
    
    /// Increment reference count
    pub fn retain(self: Tensor) Tensor {
        self.ref_count.* += 1;
        // Debug info
        std.debug.print("Tensor retained: ref_count = {}, shape = [", .{self.ref_count.*});
        for (self.shape.dims) |dim| {
            std.debug.print("{}, ", .{dim});
        }
        std.debug.print("]\n", .{});
        return self;
    }
    
    /// Decrement reference count and free resources if it reaches zero
    pub fn release(self: *const Tensor) void {
        // Safety check: don't decrement if already zero
        if (self.ref_count.* == 0) {
            // Add more debug info to help track down the issue
            std.debug.print("Warning: Trying to release tensor with zero reference count\n", .{});
            std.debug.print("  Shape: [", .{});
            for (self.shape.dims) |dim| {
                std.debug.print("{}, ", .{dim});
            }
            std.debug.print("]\n", .{});
            return;
        }
        
        self.ref_count.* -= 1;
        
        if (self.ref_count.* == 0) {
            // Clean up all resources when reference count reaches zero
            var shape_copy = self.shape;
            var buffer_copy = self.buffer;
            const allocator = self.allocator;
            
            // Free shape and buffer
            shape_copy.deinit();
            buffer_copy.deinit();
            
            // Free the reference count itself
            allocator.destroy(self.ref_count);
        }
    }
    
    /// Clean up resources
    pub fn deinit(self: *const Tensor) void {
        // Call release to properly handle reference counting
        self.release();
    }
    
    /// Set requires_grad flag for autograd
    pub fn requiresGrad(self: *Tensor, requires: bool) void {
        self.requires_grad = requires;
    }
    
    /// Get a scalar value at a specific index
    pub fn getScalar(self: Tensor, indices: []const usize) !f32 {
        if (indices.len != self.shape.rank()) {
            return error.InvalidIndices;
        }
        
        // Calculate linear index
        var linear_idx: usize = 0;
        var stride: usize = 1;
        
        var i: usize = self.shape.rank();
        while (i > 0) {
            i -= 1;
            if (indices[i] >= self.shape.dims[i]) {
                return error.IndexOutOfBounds;
            }
            linear_idx += indices[i] * stride;
            stride *= self.shape.dims[i];
        }
        
        switch (self.dtype) {
            .f32 => {
                const f32_buf = ptrCastHelper([*]f32, self.buffer.data.ptr);
                return f32_buf[linear_idx];
            },
            // Implement other dtype cases as needed
            else => {
                return error.UnsupportedDataTypeForGetScalar;
            }
        }
    }
    
    /// Set a scalar value at a specific index
    pub fn setScalar(self: *Tensor, indices: []const usize, value: anytype) !void {
        if (indices.len != self.shape.rank()) {
            return error.InvalidIndices;
        }
        
        // Calculate linear index
        var linear_idx: usize = 0;
        var stride: usize = 1;
        
        var i: usize = self.shape.rank();
        while (i > 0) {
            i -= 1;
            if (indices[i] >= self.shape.dims[i]) {
                return error.IndexOutOfBounds;
            }
            linear_idx += indices[i] * stride;
            stride *= self.shape.dims[i];
        }
        
        switch (self.dtype) {
            .f32 => {
                const f32_buf = ptrCastHelper([*]f32, self.buffer.data.ptr);
                f32_buf[linear_idx] = @floatCast(value);
            },
            // Implement other dtype cases as needed
            else => {
                return error.UnsupportedDataTypeForSetScalar;
            }
        }
    }
    
    /// Get the current reference count (for testing and debugging)
    pub fn getRefCount(self: Tensor) usize {
        return self.ref_count.*;
    }
};

test "tensor creation and access" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 3 };
    
    var tensor = try Tensor.init(allocator, &dims, .f32, .cpu);
    defer tensor.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), tensor.shape.rank());
    try std.testing.expectEqual(@as(usize, 6), tensor.shape.elemCount());
    try std.testing.expectEqual(@as(usize, 1), tensor.getRefCount());
    
    var indices = [_]usize{ 1, 2 };
    try tensor.setScalar(&indices, 42.0);
    
    const value = try tensor.getScalar(&indices);
    try std.testing.expectEqual(@as(f32, 42.0), value);
}

test "tensor zeros initialization" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 3 };
    
    var tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer tensor.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), tensor.getRefCount());
    
    var indices = [_]usize{ 1, 2 };
    const value = try tensor.getScalar(&indices);
    try std.testing.expectEqual(@as(f32, 0.0), value);
}

test "tensor filled initialization" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 3 };
    
    var tensor = try Tensor.filled(allocator, &dims, .f32, 3.14, .cpu);
    defer tensor.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), tensor.getRefCount());
    
    var indices = [_]usize{ 0, 1 };
    const value = try tensor.getScalar(&indices);
    try std.testing.expectEqual(@as(f32, 3.14), value);
}

test "tensor shape equality" {
    const allocator = std.testing.allocator;
    
    var dims1 = [_]usize{ 2, 3 };
    var shape1 = try Shape.init(allocator, &dims1);
    defer shape1.deinit();
    
    var dims2 = [_]usize{ 2, 3 };
    var shape2 = try Shape.init(allocator, &dims2);
    defer shape2.deinit();
    
    var dims3 = [_]usize{ 3, 2 };
    var shape3 = try Shape.init(allocator, &dims3);
    defer shape3.deinit();
    
    try std.testing.expect(shape1.eql(shape2));
    try std.testing.expect(!shape1.eql(shape3));
}

test "tensor reference counting" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create a tensor and verify initial ref count is 1
    var tensor1 = try Tensor.init(allocator, &dims, .f32, .cpu);
    try std.testing.expectEqual(@as(usize, 1), tensor1.getRefCount());
    
    // Fill with some values
    try tensor1.setScalar(&[_]usize{0, 0}, 1.0);
    try tensor1.setScalar(&[_]usize{0, 1}, 2.0);
    try tensor1.setScalar(&[_]usize{1, 0}, 3.0);
    try tensor1.setScalar(&[_]usize{1, 1}, 4.0);
    
    // Create a shared reference and verify ref count is now 2
    var tensor2 = try tensor1.initShared();
    try std.testing.expectEqual(@as(usize, 2), tensor1.getRefCount());
    try std.testing.expectEqual(@as(usize, 2), tensor2.getRefCount());
    
    // Verify both tensors point to the same data
    try std.testing.expectEqual(@as(f32, 1.0), try tensor2.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 4.0), try tensor2.getScalar(&[_]usize{1, 1}));
    
    // Modify through one tensor and verify both see the change
    try tensor1.setScalar(&[_]usize{0, 0}, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), try tensor2.getScalar(&[_]usize{0, 0}));
    
    // Create an independent clone
    var tensor3 = try tensor1.clone();
    // Ref count of original tensor should be unchanged
    try std.testing.expectEqual(@as(usize, 2), tensor1.getRefCount());
    // New tensor should have its own ref count of 1
    try std.testing.expectEqual(@as(usize, 1), tensor3.getRefCount());
    
    // Verify clone has the same values initially
    try std.testing.expectEqual(@as(f32, 42.0), try tensor3.getScalar(&[_]usize{0, 0}));
    
    // Modify clone and verify it doesn't affect original
    try tensor3.setScalar(&[_]usize{0, 0}, 99.0);
    try std.testing.expectEqual(@as(f32, 42.0), try tensor1.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 99.0), try tensor3.getScalar(&[_]usize{0, 0}));
    
    // Release one shared reference, verify ref count decrements
    tensor2.deinit();
    try std.testing.expectEqual(@as(usize, 1), tensor1.getRefCount());
    
    // Clean up remaining tensors
    tensor1.deinit();
    tensor3.deinit();
}