const std = @import("std");
const c = @import("mlir/c.zig");

/// Test function for MLIR operations
pub fn testMLIROperations(allocator: std.mem.Allocator) !void {
    std.debug.print("Testing basic MLIR operations...\n", .{});
    
    var ctx = try Context.init();
    defer ctx.deinit();
    
    std.debug.print("✓ MLIR Context created successfully\n", .{});
    
    const module = try Module.createEmpty(ctx);
    defer module.deinit();
    
    std.debug.print("✓ Empty MLIR module created\n", .{});
    
    const f32_type = Type.f32Type(ctx);
    _ = f32_type; // Use the type to avoid unused variable warning
    
    std.debug.print("✓ F32 type created\n", .{});
    
    std.debug.print("✓ MLIR operations test completed\n", .{});
    _ = allocator; // Suppress unused parameter warning
}

const Allocator = std.mem.Allocator;

/// MLIR Context - owns all IR objects and manages their lifecycle
pub const Context = struct {
    handle: *c.c.MlirContext,
    
    const Self = @This();
    
    pub fn init() !Self {
        const handle = c.c.contextCreate();
        return Self{ .handle = handle };
    }
    
    pub fn initWithRegistry(registry: DialectRegistry, enable_threading: bool) !Self {
        const handle = c.c.contextCreate();
        c.c.contextSetAllowUnregisteredDialects(handle, true);
        c.c.dialectRegistryLoadAll(registry.handle, handle);
        _ = enable_threading; // TODO: Implement threading support
        return Self{ .handle = handle };
    }
    
    pub fn deinit(self: Self) void {
        c.c.contextDestroy(self.handle);
    }
    
    pub fn numRegisteredDialects(self: Self) usize {
        return @intCast(c.c.contextGetNumRegisteredDialects(self.handle));
    }
    
    pub fn isRegisteredOperation(self: Self, name: []const u8) bool {
        return c.c.contextIsRegisteredOperation(self.handle, name);
    }
};
    
    /// Dialect Registry - manages available dialects
    pub const DialectRegistry = struct {
        handle: *c.c.MlirDialectRegistry,
        
        const Self = @This();
        
        pub fn init() !Self {
            const handle = c.c.dialectRegistryCreate();
            return Self{ .handle = handle };
        }
        
        pub fn deinit(self: Self) void {
            c.c.dialectRegistryDestroy(self.handle);
        }
    };
    
    /// Dialect Handle - for registering specific dialects
    pub const DialectHandle = struct {
        handle: *c.c.MlirDialectHandle,
        
        const Self = @This();
        
        pub fn fromString(name: []const u8) Self {
            const handle = if (std.mem.eql(u8, name, "func"))
                c.c.dialectHandleFuncGet()
            else if (std.mem.eql(u8, name, "arith"))
                c.c.dialectHandleArithGet()
            else if (std.mem.eql(u8, name, "scf"))
                c.c.dialectHandleScfGet()
            else
                @panic("Unsupported dialect");
            
            return Self{ .handle = handle };
        }
        
        pub fn insertDialect(self: Self, registry: DialectRegistry) void {
            c.c.dialectHandleInsertDialect(self.handle, registry.handle);
        }
    };
    
    /// MLIR Module - represents a compilation unit
    pub const Module = struct {
        handle: *c.c.MlirModule,
        
        const Self = @This();
        
        pub fn createEmpty(context: Context) !Self {
            const location = c.c.locationUnknownGet(context.handle);
            const handle = c.c.moduleCreateEmpty(location);
            return Self{ .handle = handle };
        }
        
        pub fn parse(context: Context, module_str: []const u8) !Self {
            const handle = c.c.moduleParseString(context.handle, module_str);
            return Self{ .handle = handle };
        }
        
        pub fn deinit(self: Self) void {
            c.c.moduleDestroy(self.handle);
        }
        
        pub fn op(self: Self) Operation {
            return Operation{ .handle = c.c.moduleGetOperation(self.handle) };
        }
    };
    
    /// MLIR Operation - represents a single operation in the IR
    pub const Operation = struct {
        handle: *c.c.MlirOperation,
        
        const Self = @This();
        
        /// Create an operation from an operation state
        pub fn create(ctx: Context, op_name: []const u8, args: struct {
            operands: []const Value = &.{},
            results: []const Type = &.{},
            attributes: []const struct { []const u8, Attribute } = &.{},
            location: ?Location = null,
        }) Self {
            const loc = args.location orelse Location.unknown(ctx);
            
            var state = c.c.operationStateGet(op_name, loc.handle);
            
            // Add operands
            for (args.operands) |operand| {
                c.c.mlirOperationStateAddOperands(&state, 1, &operand.handle);
            }
            
            // Add results  
            for (args.results) |result_type| {
                c.c.mlirOperationStateAddResults(&state, 1, &result_type.handle);
            }
            
            // Add attributes
            for (args.attributes) |attr_pair| {
                const name_ref = c.c.stringRefFromString(attr_pair[0]);
                c.c.mlirOperationStateAddAttributes(&state, 1, &c.c.mlirNamedAttributeGet(name_ref, attr_pair[1].handle));
            }
            
            const handle = c.c.operationCreate(&state);
            return Self{ .handle = handle };
        }
        
        pub fn deinit(self: Self) void {
            c.c.operationDestroy(self.handle);
        }
        
        pub fn dump(self: Self) void {
            c.c.operationDump(self.handle);
        }
        
        pub fn print(self: Self, writer: anytype) !void {
            // Simple implementation - in practice you'd want a proper callback
            _ = writer;
            self.dump(); // For now, just dump to stdout
        }
        
        pub fn getResult(self: Self, index: usize) Value {
            return Value{ .handle = c.c.operationGetResult(self.handle, @intCast(index)) };
        }
        
        pub fn getOperand(self: Self, index: usize) Value {
            return Value{ .handle = c.c.mlirOperationGetOperand(self.handle, @intCast(index)) };
        }
        
        pub fn getName(self: Self) []const u8 {
            const name_id = c.c.operationGetName(self.handle);
            const name_ref = c.c.identifierStr(name_id);
            return c.c.stringRefToSlice(name_ref);
        }
    };
    
    /// Pass Manager - manages compilation passes
    pub const PassManager = struct {
        handle: *c.c.MlirPassManager,
        
        const Self = @This();
        
        pub fn init(context: Context) !Self {
            const handle = c.c.passManagerCreate(context.handle);
            return Self{ .handle = handle };
        }
        
        pub fn deinit(self: Self) void {
            c.c.passManagerDestroy(self.handle);
        }
        
        pub fn asOpPassManager(self: Self) OpPassManager {
            return OpPassManager{ .handle = c.c.passManagerGetAsOpPassManager(self.handle) };
        }
        
        pub fn runOnOp(self: Self, operation: Operation) !void {
            const result = c.c.passManagerRunOnOp(self.handle, operation.handle);
            if (result.isFailure()) {
                return error.PassManagerFailed;
            }
        }
    };
    
    /// Operation Pass Manager - manages passes for operations
    pub const OpPassManager = struct {
        handle: *c.c.MlirOpPassManager,
        
        const Self = @This();
        
        pub fn addPipeline(self: Self, pipeline: []const u8) !void {
            const result = c.c.opPassManagerAddPipeline(self.handle, pipeline);
            if (result.isFailure()) {
                return error.PipelineAddFailed;
            }
        }
    };
    
    /// MLIR Type system
    pub const Type = struct {
        handle: *c.c.MlirType,
        
        const Self = @This();
        
        pub fn f32Type(context: Context) Self {
            return Self{ .handle = c.c.floatTypeGetF32(context.handle) };
        }
        
        pub fn f64Type(context: Context) Self {
            return Self{ .handle = c.c.floatTypeGetF64(context.handle) };
        }
        
        pub fn rankedTensorType(context: Context, shape: []const i64, element_type: Self) Self {
            _ = context;
            const handle = c.c.rankedTensorTypeGet(@intCast(shape.len), shape, element_type.handle, null);
            return Self{ .handle = handle };
        }
        
        /// Create tensor type with shape and element type
        pub fn tensor(shape: []const i64, element_type: Self) Self {
            const handle = c.c.rankedTensorTypeGet(@intCast(shape.len), shape, element_type.handle, null);
            return Self{ .handle = handle };
        }
        
        /// Cast to a specific type if possible
        pub fn as(self: Self, comptime T: type) ?T {
            if (T == RankedTensorType) {
                if (c.c.typeIsARankedTensor(self.handle)) {
                    return RankedTensorType{ .handle = self.handle };
                }
            }
            return null;
        }
    };

    /// MLIR Value - represents SSA values in the IR
    pub const Value = struct {
        handle: *c.c.MlirValue,
        
        const Self = @This();
        
        /// Get the type of this value
        pub fn getType(self: Self) Type {
            return Type{ .handle = c.c.valueGetType(self.handle) };
        }
    };
    
    /// MLIR Attribute - represents compile-time constants and metadata
    pub const Attribute = struct {
        handle: *c.c.MlirAttribute,
        
        const Self = @This();
        
        /// Create a float attribute
        pub fn floatAttr(context: Context, value: f64, attr_type: Type) Self {
            return Self{ .handle = c.c.mlirFloatAttrDoubleGet(context.handle, attr_type.handle, value) };
        }
        
        /// Create an integer attribute
        pub fn integerAttr(context: Context, value: i64, attr_type: Type) Self {
            _ = context;
            return Self{ .handle = c.c.mlirIntegerAttrGet(attr_type.handle, value) };
        }
        
        /// Create a dense array attribute for i64 values
        pub fn denseI64ArrayAttr(context: Context, values: []const i64) Self {
            return Self{ .handle = c.c.mlirDenseI64ArrayGet(context.handle, @intCast(values.len), values.ptr) };
        }
    };
    
    /// Dot dimension numbers attribute for matmul operations
    pub const DotDimensionNumbersAttribute = struct {
        lhs_batching_dimensions: []const i64,
        rhs_batching_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
        
        const Self = @This();
        
        pub fn getLhsBatchingDimensions(self: Self) []const i64 {
            return self.lhs_batching_dimensions;
        }
        
        pub fn getRhsBatchingDimensions(self: Self) []const i64 {
            return self.rhs_batching_dimensions;
        }
        
        pub fn getLhsContractingDimensions(self: Self) []const i64 {
            return self.lhs_contracting_dimensions;
        }
        
        pub fn getRhsContractingDimensions(self: Self) []const i64 {
            return self.rhs_contracting_dimensions;
        }
        
        pub fn asAttr(self: Self) Attribute {
            // This would create the actual MLIR attribute
            // For now, return a placeholder
            _ = self;
            return Attribute{ .handle = undefined }; // Placeholder
        }
    };

    /// MLIR OpBuilder - for constructing operations (placeholder for now)
    pub const OpBuilder = struct {
        // Placeholder - real OpBuilder integration comes later
        const Self = @This();
    };

    /// MLIR Block - represents a basic block
    pub const Block = struct {
        handle: *c.c.MlirBlock,
        
        const Self = @This();
        
        pub fn addArgument(self: Self, arg_type: Type, loc: Location) Value {
            return Value{ .handle = c.c.blockAddArgument(self.handle, arg_type.handle, loc.handle) };
        }
    };

    /// MLIR Location - source location information
    pub const Location = struct {
        handle: *c.c.MlirLocation,
        
        const Self = @This();
        
        pub fn unknown(context: Context) Self {
            return Self{ .handle = c.c.locationUnknownGet(context.handle) };
        }
    };

    /// MLIR RankedTensorType - for tensor type introspection
    pub const RankedTensorType = struct {
        handle: *c.c.MlirType,
        
        const Self = @This();
        
        /// Get the rank (number of dimensions) 
        pub fn getRank(self: Self) usize {
            return @intCast(c.c.shapedTypeGetRank(self.handle));
        }
        
        /// Get a specific dimension
        pub fn getDimension(self: Self, pos: usize) i64 {
            return c.c.shapedTypeGetDimSize(self.handle, @intCast(pos));
        }
        
        /// Get the element type
        pub fn getElementType(self: Self) Type {
            return Type{ .handle = c.c.shapedTypeGetElementType(self.handle) };
        }
    };
    
    /// Helper functions
    pub fn isFailure(result: c.c.MlirLogicalResult) bool {
        return result.isFailure();
    }
    
    pub fn isSuccess(result: c.c.MlirLogicalResult) bool {
        return result.isSuccess();
    }
    
    pub fn registerAllPasses() void {
        c.c.mlirRegisterAllPasses();
    }
    
    /// String utilities for MLIR C API
    pub fn stringRefToSlice(ref: c.c.MlirStringRef) []const u8 {
        return ref.data[0..ref.length];
    }