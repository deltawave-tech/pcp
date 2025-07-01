const std = @import("std");
const mlir = @import("mlir.zig");
const mlir_ctx = @import("mlir_ctx.zig"); 

const Allocator = std.mem.Allocator;

/// Supported data types for tensors (maps to MLIR types)
pub const DType = enum {
    f16,
    f32,
    f64,
    i32,
    i64,
    bool,

    pub fn toMLIRType(self: DType, ctx: mlir.Context) mlir.Type {
        return switch (self) {
            .f32 => mlir.Type.f32Type(ctx),
            .f64 => mlir.Type.f64Type(ctx),
            // TODO: Add other types as we implement them in MLIR wrappers
            else => @panic("DType not yet implemented in MLIR wrappers"),
        };
    }

    /// Convert from MLIR type to DType (following expert pattern)
    pub fn fromMlirType(mlir_type: mlir.Type) DType {
        // Simple implementation - could be enhanced with proper type introspection
        _ = mlir_type;
        return .f32; // For now, assume f32 - TODO: Add proper type detection
    }

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

/// Tensor shape information - uses i64 to match MLIR dimension type
pub const Shape = struct {
    dims: []const i64, // Changed from usize to i64 for MLIR compatibility
    dtype: DType,
    allocator: Allocator,

    pub fn init(allocator: Allocator, dimensions: []const i64, dtype: DType) !Shape {
        const dims_copy = try allocator.dupe(i64, dimensions);
        return Shape{
            .dims = dims_copy,
            .dtype = dtype,
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
            if (dim <= 0) return 0; // Handle dynamic or invalid dimensions
            result *= @intCast(dim);
        }
        return result;
    }

    pub fn eql(self: Shape, other: Shape) bool {
        if (self.dtype != other.dtype) return false;
        return std.mem.eql(i64, self.dims, other.dims);
    }

    /// Convert shape to MLIR tensor type
    pub fn toMLIRType(self: Shape, ctx: mlir.Context) mlir.Type {
        const element_type = self.dtype.toMLIRType(ctx);
        return mlir.Type.rankedTensorType(ctx, self.dims, element_type);
    }

    /// Create Shape from MLIR RankedTensorType (following expert pattern)
    pub fn fromMLIR(shaped_type: mlir.RankedTensorType, allocator: Allocator) !Shape {
        const tensor_rank = shaped_type.getRank();
        var dims = try allocator.alloc(i64, tensor_rank);
        for (0..tensor_rank) |i| {
            dims[i] = shaped_type.getDimension(i);
        }
        const dtype = DType.fromMlirType(shaped_type.getElementType());
        return Shape{
            .dims = dims,
            .dtype = dtype,
            .allocator = allocator,
        };
    }
};

/// Forward declaration for MLIRBuilder (will be defined in ops.zig)
pub const MLIRBuilder = @import("ops.zig").MLIRBuilder;

/// The new MLIR-based Tensor struct - symbolic handle to MLIR values
/// The DataType parameter is now phantom (just for Zig type safety)
pub fn Tensor(comptime T: type) type {
    _ = T; // Phantom type parameter

    return struct {
        const Self = @This();

        shape: Shape,
        value: mlir.Value, // Core change: MLIR SSA value handle
        builder: *MLIRBuilder, // Reference to graph builder

        /// Creates a tensor from a block argument (function input)
        pub fn newArgument(builder: *MLIRBuilder, shape: Shape) !Self {
            const mlir_type = shape.toMLIRType(builder.ctx);
            const arg_value = try builder.addBlockArgument(mlir_type);

            return Self{
                .shape = shape,
                .value = arg_value,
                .builder = builder,
            };
        }

        /// Creates a tensor from a constant value
        pub fn newConstant(builder: *MLIRBuilder, host_data: []const u8, shape: Shape) !Self {
            const mlir_type = shape.toMLIRType(builder.ctx);
            const const_value = try builder.createConstant(host_data, mlir_type, shape);

            return Self{
                .shape = shape,
                .value = const_value,
                .builder = builder,
            };
        }

        /// Create a tensor filled with zeros
        pub fn zeros(builder: *MLIRBuilder, dims: []const i64, dtype: DType) !Self {
            var shape = try Shape.init(builder.allocator, dims, dtype);
            errdefer shape.deinit();

            // Create zero data
            const elem_count = shape.elemCount();
            const data_size = elem_count * dtype.sizeInBytes();
            const zero_data = try builder.allocator.alloc(u8, data_size);
            defer builder.allocator.free(zero_data);
            @memset(zero_data, 0);

            return try Self.newConstant(builder, zero_data, shape);
        }

        /// Create a tensor filled with a specific value
        pub fn filled(builder: *MLIRBuilder, dims: []const i64, dtype: DType, value: f32) !Self {
            var shape = try Shape.init(builder.allocator, dims, dtype);
            errdefer shape.deinit();

            // Create filled data (simplified for f32 only for now)
            if (dtype != .f32) return error.UnsupportedDTypeForFilled;
            
            const elem_count = shape.elemCount();
            const data = try builder.allocator.alloc(f32, elem_count);
            defer builder.allocator.free(data);
            
            for (data) |*item| {
                item.* = value;
            }

            const bytes = std.mem.sliceAsBytes(data);
            return try Self.newConstant(builder, bytes, shape);
        }

        /// No more deinit needed - MLIR context owns all values
        pub fn deinit(self: Self) void {
            _ = self;
            // No-op: MLIR context manages memory
        }

        // --- MLIR-based operations (following expert pattern) ---
        // These are wrappers around the standalone functions in ops_mlir.zig

        pub fn add(self: Self, other: Self) !Self {
            const ops = @import("ops.zig");
            return try ops.add(self.builder, self, other);
        }

        pub fn subtract(self: Self, other: Self) !Self {
            const ops = @import("ops.zig");
            return try ops.subtract(self.builder, self, other);
        }

        pub fn multiply(self: Self, other: Self) !Self {
            const ops = @import("ops.zig");
            return try ops.multiply(self.builder, self, other);
        }

        pub fn matmul(self: Self, other: Self) !Self {
            const ops = @import("ops.zig");
            return try ops.matmul(self.builder, self, other);
        }

        pub fn relu(self: Self) !Self {
            const ops = @import("ops.zig");
            return try ops.relu(self.builder, self);
        }

        // Note: No more getScalar/setScalar - tensors are symbolic!
        // Data only exists at graph boundaries (inputs/outputs)
    };
}

// Common tensor types
pub const TensorF32 = Tensor(f32);
pub const TensorF64 = Tensor(f64);
pub const TensorI32 = Tensor(i32);