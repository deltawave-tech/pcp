const std = @import("std");
const mlir = @import("../mlir/wrapper.zig");
const mlir_ctx = @import("../mlir/context.zig");
const c = @import("../mlir/c.zig").c;

const Allocator = std.mem.Allocator;

/// Supported data types for tensors (maps to MLIR types)
pub const DType = enum {
    f16,
    bf16,
    f32,
    f64,
    i32,
    i64,
    bool,

    pub fn toMLIRType(self: DType, ctx: mlir.Context) mlir.Type {
        return switch (self) {
            .f32 => mlir.Type.f32Type(ctx),
            .f64 => mlir.Type.f64Type(ctx),
            .bf16 => mlir.Type.bf16Type(ctx),
            .i32 => mlir.Type.i32Type(ctx),
            .i64 => mlir.Type.i64Type(ctx),
            else => @panic("DType not yet implemented in MLIR wrappers"),
        };
    }

    /// Convert from MLIR type to DType
    pub fn fromMlirType(mlir_type: mlir.Type) DType {
        const ctx = mlir_type.getContext();

        if (mlir_type.isInteger()) {
            return .i32;
        } else if (mlir_type.isBF16(ctx)) {
            return .bf16;
        } else if (mlir_type.isF64(ctx)) {
            return .f64;
        } else if (mlir_type.isF32(ctx)) {
            return .f32;
        } else if (mlir_type.isIndex()) {
            return .i64;
        }

        return .f32;
    }

    pub fn sizeInBytes(self: DType) usize {
        return switch (self) {
            .f16 => 2,
            .bf16 => 2,
            .f32 => 4,
            .f64 => 8,
            .i32 => 4,
            .i64 => 8,
            .bool => 1,
        };
    }
};

/// Tensor shape information - queries MLIR types on-demand (no allocations)
pub const Shape = struct {
    // Store the MLIR type as the source of truth. No Zig allocations needed.
    mlir_type: mlir.RankedTensorType,
    dtype: DType,

    /// Create Shape from MLIR RankedTensorType - no allocation needed
    pub fn init(mlir_type: mlir.RankedTensorType) !Shape {
        const dtype = DType.fromMlirType(mlir_type.getElementType());
        return Shape{
            .mlir_type = mlir_type,
            .dtype = dtype,
        };
    }

    /// Create Shape with explicit dimensions (for non-MLIR usage)
    pub fn initWithDims(ctx: mlir.Context, dimensions: []const i64, dtype: DType) !Shape {
        const element_type = dtype.toMLIRType(ctx);
        const mlir_tensor_type = mlir.Type.rankedTensorType(ctx, dimensions, element_type);
        const ranked_type = mlir.RankedTensorType{ .handle = mlir_tensor_type.handle };
        return Shape{
            .mlir_type = ranked_type,
            .dtype = dtype,
        };
    }

    /// deinit becomes a no-op as there is nothing to free
    pub fn deinit(self: *Shape) void {
        _ = self;
    }

    /// rank() queries the MLIR type directly
    pub fn rank(self: Shape) usize {
        return self.mlir_type.getRank();
    }

    /// Get a specific dimension by index
    pub fn getDimension(self: Shape, index: usize) i64 {
        return self.mlir_type.getDimension(index);
    }

    /// elemCount() queries the MLIR type directly
    pub fn elemCount(self: Shape) usize {
        const rank_val = self.rank();
        if (rank_val == 0) return 0;

        var result: usize = 1;
        for (0..rank_val) |i| {
            const dim = self.getDimension(i);
            if (dim <= 0) return 0; // Handle dynamic or invalid dimensions
            result *= @intCast(dim);
        }
        return result;
    }

    pub fn eql(self: Shape, other: Shape) bool {
        if (self.dtype != other.dtype) return false;
        const self_rank = self.rank();
        const other_rank = other.rank();
        if (self_rank != other_rank) return false;
        
        for (0..self_rank) |i| {
            if (self.getDimension(i) != other.getDimension(i)) return false;
        }
        return true;
    }

    /// Compare shape with a slice of dimensions
    pub fn eqlDims(self: Shape, dims: []const i64) bool {
        const self_rank = self.rank();
        if (self_rank != dims.len) return false;
        
        for (0..self_rank) |i| {
            if (self.getDimension(i) != dims[i]) return false;
        }
        return true;
    }

    /// Convert shape to MLIR tensor type - can reuse existing MLIR type
    pub fn toMLIRType(self: Shape, ctx: mlir.Context) mlir.Type {
        _ = ctx; // For now, just return the existing MLIR type
        return mlir.Type{ .handle = self.mlir_type.handle };
    }

    /// Provide a way to get dimensions for iteration when a slice is needed
    /// Caller must free the returned slice
    pub fn getDims(self: Shape, allocator: Allocator) ![]i64 {
        const rank_val = self.rank();
        const dims_slice = try allocator.alloc(i64, rank_val);
        for (0..rank_val) |i| {
            dims_slice[i] = self.getDimension(i);
        }
        return dims_slice;
    }

    /// Create Shape from MLIR RankedTensorType - no allocation!
    pub fn fromMLIR(shaped_type: mlir.RankedTensorType, allocator: Allocator) !Shape {
        _ = allocator; // No longer needed
        return Shape.init(shaped_type);
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
            var shape = try Shape.initWithDims(builder.ctx, dims, dtype);
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
            var shape = try Shape.initWithDims(builder.ctx, dims, dtype);
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

        /// Creates a tensor from a host slice of data.
        pub fn fromSlice(comptime D: type, builder: *MLIRBuilder, host_slice: []const D, dims: []const i64) !Self {
            if (D != f32) {
                // For now, only support f32 for simplicity. Extend as needed.
                return error.UnsupportedDTypeForFromSlice;
            }
            const dtype = DType.f32;
            const shape = try Shape.initWithDims(builder.ctx, dims, dtype);

            const host_bytes = std.mem.sliceAsBytes(host_slice);
            return Self.newConstant(builder, host_bytes, shape);
        }

        // --- MLIR-based operations ---
        // These are wrappers around the standalone functions in ops.zig

        pub fn add(self: Self, other: Self) !Self {
            const ops_mod = @import("ops.zig");
            return try ops_mod.add(self.builder, self, other);
        }

        pub fn subtract(self: Self, other: Self) !Self {
            const ops_mod = @import("ops.zig");
            return try ops_mod.subtract(self.builder, self, other);
        }

        pub fn multiply(self: Self, other: Self) !Self {
            const ops_mod = @import("ops.zig");
            return try ops_mod.multiply(self.builder, self, other);
        }

        pub fn matmul(self: Self, other: Self) !Self {
            const ops_mod = @import("ops.zig");
            return try ops_mod.matmul(self.builder, self, other);
        }

        pub fn relu(self: Self) !Self {
            const ops_mod = @import("ops.zig");
            return try ops_mod.relu(self.builder, self);
        }

        /// Create a generic Tensor from a raw byte slice and shape.
        /// Effectively wraps a stablehlo.constant.
        pub fn fromBytes(builder: *MLIRBuilder, bytes: []const u8, shape_dims: []const i64, dtype: DType) !Self {
            const tensor_type = mlir.Type.rankedTensorType(builder.ctx, shape_dims, dtype.toMLIRType(builder.ctx));

            // Create shape object
            const ts_shape = try Shape.initWithDims(builder.ctx, shape_dims, dtype);

            const value = try builder.createConstant(bytes, tensor_type, ts_shape);
            return try builder.newTensor(value);
        }

        /// Extract raw bytes from a stablehlo.constant tensor.
        /// This allocates new memory for the result which caller must free.
        pub fn toBytes(self: Self, allocator: std.mem.Allocator) ![]u8 {
            // 1. Get Defining Op
            if (!c.mlirValueIsAOpResult(self.value.handle)) return error.NotAnOperationResult;
            const op_handle = c.mlirOpResultGetOwner(self.value.handle);

            // 2. Check Op Name
            const name_id = c.mlirOperationGetName(op_handle);
            const name_ref = c.mlirIdentifierStr(name_id);
            const name = c.fromStringRef(name_ref);
            if (!std.mem.eql(u8, name, "stablehlo.constant")) return error.NotAConstantOp;

            // 3. Get 'value' Attribute
            const val_ref = c.stringRefFromString("value");
            const attr = c.operationGetAttributeByName(op_handle, val_ref);
            if (@intFromPtr(attr.ptr) == 0 or !c.mlirAttributeIsADenseElements(attr)) {
                return error.InvalidConstantAttribute;
            }

            // 4. Extract Data
            const raw_ptr = c.mlirDenseElementsAttrGetRawData(attr);
            if (@intFromPtr(raw_ptr) == 0) return error.InvalidRawDataPointer;

            const num_bytes = self.shape.elemCount() * self.shape.dtype.sizeInBytes();

            // 5. Dupe
            const data_slice: [*]const u8 = @ptrCast(raw_ptr);
            return try allocator.dupe(u8, data_slice[0..num_bytes]);
        }

        // Note: No more getScalar/setScalar - tensors are symbolic!
        // Data only exists at graph boundaries (inputs/outputs)
    };
}

// Common tensor types
pub const TensorF32 = Tensor(f32);
pub const TensorF64 = Tensor(f64);
pub const TensorI32 = Tensor(i32);