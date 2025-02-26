const std = @import("std");
const Allocator = std.mem.Allocator;

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
            result *= dim;
        }
        return result;
    }
    
    pub fn bytesRequired(self: Shape, dtype: DType) usize {
        return self.elemCount() * dtype.sizeInBytes();
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
    
    /// Initialize a tensor with a shape and data type
    pub fn init(allocator: Allocator, dims: []const usize, dtype: DType, backend: BackendType) !Tensor {
        const shape = try Shape.init(allocator, dims);
        const buffer_size = shape.bytesRequired(dtype);
        const buffer = try Buffer.init(allocator, buffer_size);
        
        return Tensor{
            .shape = shape,
            .dtype = dtype,
            .buffer = buffer,
            .backend = backend,
        };
    }
    
    /// Initialize a tensor with a specific value
    pub fn filled(allocator: Allocator, dims: []const usize, dtype: DType, value: anytype, backend: BackendType) !Tensor {
        var tensor = try init(allocator, dims, dtype, backend);
        
        // Fill the buffer with the value based on dtype
        switch (dtype) {
            .f32 => {
                const val: f32 = @floatCast(value);
                const f32_buf = @as([*]f32, @ptrCast(@alignCast(tensor.buffer.data.ptr)))[0..tensor.shape.elemCount()];
                for (f32_buf) |*ptr| {
                    ptr.* = val;
                }
            },
            // Implement other dtype cases as needed
            else => {
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
        var prng = std.rand.DefaultPrng.init(seed);
        const rand = prng.random();
        
        switch (dtype) {
            .f32 => {
                const f32_buf = @as([*]f32, @ptrCast(@alignCast(tensor.buffer.data.ptr)))[0..tensor.shape.elemCount()];
                for (f32_buf) |*ptr| {
                    ptr.* = rand.float(f32);
                }
            },
            // Implement other dtype cases as needed
            else => {
                return error.UnsupportedDataTypeForRandom;
            }
        }
        
        return tensor;
    }
    
    /// Clean up resources
    pub fn deinit(self: *Tensor) void {
        self.shape.deinit();
        self.buffer.deinit();
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
                const f32_buf = @as([*]f32, @ptrCast(@alignCast(self.buffer.data.ptr)));
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
                const f32_buf = @as([*]f32, @ptrCast(@alignCast(self.buffer.data.ptr)));
                f32_buf[linear_idx] = @floatCast(value);
            },
            // Implement other dtype cases as needed
            else => {
                return error.UnsupportedDataTypeForSetScalar;
            }
        }
    }
};

test "tensor creation and access" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 3 };
    
    var tensor = try Tensor.init(allocator, &dims, .f32, .cpu);
    defer tensor.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), tensor.shape.rank());
    try std.testing.expectEqual(@as(usize, 6), tensor.shape.elemCount());
    
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
    
    var indices = [_]usize{ 1, 2 };
    const value = try tensor.getScalar(&indices);
    try std.testing.expectEqual(@as(f32, 0.0), value);
}

test "tensor filled initialization" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 3 };
    
    var tensor = try Tensor.filled(allocator, &dims, .f32, 3.14, .cpu);
    defer tensor.deinit();
    
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