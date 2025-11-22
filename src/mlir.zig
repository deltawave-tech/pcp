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

// initOperationState has been removed - we now use pcpCreateOperation instead

/// MLIR Context - owns all IR objects and manages their lifecycle
pub const Context = struct {
    handle: c.c.MlirContext,

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
        const name_ref = c.c.stringRefFromString(name);
        return c.c.contextIsRegisteredOperation(self.handle, name_ref);
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
        handle: c.c.MlirModule,

        const Self = @This();

        pub fn createEmpty(context: Context) !Self {
            const location = c.c.locationUnknownGet(context.handle);
            const handle = c.c.moduleCreateEmpty(location);
            return Self{ .handle = handle };
        }

        pub fn parse(context: Context, module_str: []const u8) !Self {
            std.debug.print("Module.parse: About to call moduleParseString with {} bytes\n", .{module_str.len});

            const module_ref = c.c.stringRefFromString(module_str);
            const handle = c.c.moduleParseString(context.handle, module_ref);

            std.debug.print("Module.parse: moduleParseString returned handle: 0x{x}\n", .{@intFromPtr(handle.ptr)});

            if (@intFromPtr(handle.ptr) == 0) {
                std.debug.print("Module.parse: Handle is null - returning MLIRParseError\n", .{});
                return error.MLIRParseError;
            }

            std.debug.print("Module.parse: Handle is valid - returning module\n", .{});
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
        handle: c.c.MlirOperation,
        
        const Self = @This();
        
        /// Create an operation safely using the C++ helper with packed args
        pub fn create(allocator: std.mem.Allocator, ctx: Context, op_name: []const u8, args: struct {
            operands: []const Value = &.{},
            results: []const Type = &.{},
            attributes: []const struct { []const u8, Attribute } = &.{},
            location: ?Location = null,
        }) !Self {
            const loc = args.location orelse Location.unknown(ctx);
            const name_ref = c.c.stringRefFromString(op_name);

            // 1. Prepare Operands
            var operand_handles: ?[]c.c.MlirValue = null;
            if (args.operands.len > 0) {
                const handles = try allocator.alloc(c.c.MlirValue, args.operands.len);
                for (args.operands, 0..) |opd, i| handles[i] = opd.handle;
                operand_handles = handles;
            }
            defer if (operand_handles) |h| allocator.free(h);

            // 2. Prepare Results
            var result_handles: ?[]c.c.MlirType = null;
            if (args.results.len > 0) {
                const handles = try allocator.alloc(c.c.MlirType, args.results.len);
                for (args.results, 0..) |res, i| handles[i] = res.handle;
                result_handles = handles;
            }
            defer if (result_handles) |h| allocator.free(h);

            // 3. Prepare Attributes
            var attr_handles: ?[]c.c.MlirNamedAttribute = null;
            if (args.attributes.len > 0) {
                const handles = try allocator.alloc(c.c.MlirNamedAttribute, args.attributes.len);
                for (args.attributes, 0..) |attr_pair, i| {
                    const name_str = c.c.stringRefFromString(attr_pair[0]);
                    const name_id = c.c.mlirIdentifierGet(ctx.handle, name_str);
                    handles[i] = .{ .name = name_id, .attribute = attr_pair[1].handle };
                }
                attr_handles = handles;
            }
            defer if (attr_handles) |h| allocator.free(h);

            // 4. Pack Arguments safely
            const op_args = c.c.PcpOpArgs{
                .nResults = @intCast(args.results.len),
                .results = if (result_handles) |h| h.ptr else null,
                .nOperands = @intCast(args.operands.len),
                .operands = if (operand_handles) |h| h.ptr else null,
                .nAttributes = @intCast(args.attributes.len),
                .attributes = if (attr_handles) |h| h.ptr else null,
                .nRegions = 0,
                .regions = null,
            };

            // 5. Call C++ Helper
            const handle = c.c.pcpCreateOperation(&name_ref, &loc.handle, &op_args);

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
            return c.c.fromStringRef(name_ref);
        }
        
        pub fn getRegion(self: Self, index: usize) Region {
            return Region{ .handle = c.c.operationGetRegion(self.handle, @intCast(index)) };
        }
        
        pub fn getNext(self: Self) ?Self {
            const next_op = c.c.operationGetNextInBlock(self.handle);
            if (@intFromPtr(next_op.ptr) == 0) return null;
            return Self{ .handle = next_op };
        }
        
        pub fn getNumOperands(self: Self) usize {
            return @intCast(c.c.operationGetNumOperands(self.handle));
        }
        
        pub fn getNumResults(self: Self) usize {
            return @intCast(c.c.operationGetNumResults(self.handle));
        }
        
        pub fn getLocation(self: Self) Location {
            return Location{ .handle = c.c.operationGetLocation(self.handle) };
        }
        
        pub fn getNumAttributes(self: Self) usize {
            return @intCast(c.c.operationGetNumAttributes(self.handle));
        }
        
        pub fn getAttribute(self: Self, index: usize) c.c.MlirNamedAttribute {
            return c.c.operationGetAttribute(self.handle, @intCast(index));
        }
        
        pub fn verify(self: Self) bool {
            return c.c.mlirOperationVerify(self.handle);
        }

        // ADD THIS METHOD
        pub fn getType(self: Self) Type {
            const type_name_ref = c.c.stringRefFromString("function_type");
            const type_attr = c.c.mlirOperationGetAttributeByName(self.handle, type_name_ref);
            // An MlirTypeAttr is a type of MlirAttribute that wraps an MlirType
            return Type{ .handle = c.c.mlirTypeAttrGetValue(type_attr) };
        }
    };
    
    /// MLIR Region - represents a CFG region
    pub const Region = struct {
        handle: c.c.MlirRegion,
        
        const Self = @This();
        
        pub fn getBlock(self: Self, index: usize) Block {
            _ = index; // For now, just get the first block
            return Block{ .handle = c.c.regionGetFirstBlock(self.handle) };
        }
        
        pub fn appendOwnedBlock(self: Self, block: Block) void {
            c.c.regionAppendOwnedBlock(self.handle, block.handle);
        }
    };
    
    /// MLIR Block - represents a basic block
    pub const Block = struct {
        handle: c.c.MlirBlock,
        
        const Self = @This();
        
        pub fn getFirstOp(self: Self) ?Operation {
            const first_op = c.c.blockGetFirstOperation(self.handle);
            if (@intFromPtr(first_op.ptr) == 0) return null;
            return Operation{ .handle = first_op };
        }
        
        pub fn addArgument(self: Self, arg_type: Type, loc: Location) Value {
            return Value{ .handle = c.c.mlirBlockAddArgument(self.handle, arg_type.handle, loc.handle) };
        }

        pub fn getArgument(self: Self, pos: usize) Value {
            return Value{ .handle = c.c.blockGetArgument(self.handle, @intCast(pos)) };
        }
        
        pub fn getLastOp(self: Self) ?Operation {
            const terminator = c.c.blockGetTerminator(self.handle);
            if (@intFromPtr(terminator) == 0) return null;
            return Operation{ .handle = terminator };
        }
        
        /// Get the actual last operation in the block (not necessarily a terminator)
        pub fn getLastOpGeneric(self: Self) ?Operation {
            // Walk through all operations to find the last one
            var maybe_op = self.getFirstOp();
            var last_op: ?Operation = null;
            while (maybe_op) |op| {
                last_op = op;
                maybe_op = op.getNext();
            }
            return last_op;
        }
        
        pub fn getNumArguments(self: Self) usize {
            return @intCast(c.c.blockGetNumArguments(self.handle));
        }
        
        pub fn getArguments(self: Self, allocator: std.mem.Allocator) ![]Value {
            const num_args = self.getNumArguments();
            var args = try allocator.alloc(Value, num_args);
            for (0..num_args) |i| {
                args[i] = self.getArgument(i);
            }
            return args;
        }
        
        pub fn appendOwnedOperation(self: Self, operation: Operation) void {
            c.c.blockAppendOwnedOperation(self.handle, operation.handle);
        }
    };
    
    /// Pass Manager - manages compilation passes
    pub const PassManager = struct {
        handle: *c.c.MlirPassManager,
        
        const Self = @This();
        
        pub fn init(context: Context) !Self {
            std.debug.print("PassManager.init: Creating pass manager with context handle = {}\n", .{@intFromPtr(context.handle)});
            const handle = c.c.passManagerCreate(context.handle);
            std.debug.print("PassManager.init: Created pass manager with handle = {}\n", .{@intFromPtr(handle)});
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
        handle: c.c.MlirType,
        
        const Self = @This();
        
        pub fn f32Type(context: Context) Self {
            return Self{ .handle = c.c.floatTypeGetF32(context.handle) };
        }
        
        pub fn f64Type(context: Context) Self {
            return Self{ .handle = c.c.floatTypeGetF64(context.handle) };
        }
        
        pub fn i32Type(context: Context) Self {
            return Self{ .handle = c.c.mlirIntegerTypeGet(context.handle, 32) };
        }
        
        pub fn i64Type(context: Context) Self {
            return Self{ .handle = c.c.mlirIntegerTypeGet(context.handle, 64) };
        }
        
        pub fn i1Type(context: Context) Self {
            return Self{ .handle = c.c.mlirIntegerTypeGet(context.handle, 1) };
        }
        
        pub fn rankedTensorType(context: Context, shape: []const i64, element_type: Self) Self {
            _ = context;
            const null_attr = c.c.MlirAttribute{ .ptr = null };
            const handle = c.c.rankedTensorTypeGet(@intCast(shape.len), shape.ptr, element_type.handle, null_attr);
            return Self{ .handle = handle };
        }

        /// Create tensor type with shape and element type
        pub fn tensor(shape: []const i64, element_type: Self) Self {
            const null_attr = c.c.MlirAttribute{ .ptr = null };
            const handle = c.c.rankedTensorTypeGet(@intCast(shape.len), shape.ptr, element_type.handle, null_attr);
            return Self{ .handle = handle };
        }
        
        /// Cast to a specific type if possible
        pub fn as(self: Self, comptime T: type) ?T {
            if (T == RankedTensorType) {
                if (c.c.typeIsARankedTensor(self.handle)) {
                    return RankedTensorType{ .handle = self.handle };
                }
            } else if (T == FunctionType) {
                if (c.c.typeIsAFunction(self.handle)) {
                    return FunctionType{ .handle = self.handle };
                }
            }
            return null;
        }
        
        /// Create a function type with given inputs and results
        /// UPDATED: Now accepts an allocator for temporary C-API array construction
        pub fn functionType(allocator: std.mem.Allocator, context: Context, inputs: []const Type, results: []const Type) !Self {
            // Use the provided allocator (safe Zig GPA) instead of c_allocator
            var input_handles = try allocator.alloc(c.c.MlirType, inputs.len);
            defer allocator.free(input_handles);

            for (inputs, 0..) |input, i| {
                input_handles[i] = input.handle;
            }

            var result_handles = try allocator.alloc(c.c.MlirType, results.len);
            defer allocator.free(result_handles);

            for (results, 0..) |result, i| {
                result_handles[i] = result.handle;
            }

            return Self{ .handle = c.c.functionTypeGet(context.handle, @intCast(inputs.len), input_handles.ptr, @intCast(results.len), result_handles.ptr) };
        }
    };

    /// MLIR Value - represents SSA values in the IR
    pub const Value = struct {
        handle: c.c.MlirValue,
        
        const Self = @This();
        
        /// Get the type of this value
        pub fn getType(self: Self) Type {
            return Type{ .handle = c.c.valueGetType(self.handle) };
        }
        
        /// Dump this value to stdout for debugging
        pub fn dump(self: Self) void {
            c.c.mlirValueDump(self.handle);
        }
        
    };
    
    /// MLIR Attribute - represents compile-time constants and metadata
    pub const Attribute = struct {
        handle: c.c.MlirAttribute,
        
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
        
        /// Create a dense elements attribute from host data
        pub fn denseElementsAttr(context: Context, shaped_type: Type, host_data: []const u8) Self {
            _ = context;
            return Self{ .handle = c.c.mlirDenseElementsAttrRawBufferGet(shaped_type.handle, host_data.len, host_data.ptr) };
        }
        
        /// Create a dense elements attribute by "splatting" a single value across all elements
        pub fn denseElementsAttrSplat(shaped_type: Type, value: f64) Self {
            return Self{ .handle = c.c.mlirDenseElementsAttrFloatSplatGet(shaped_type.handle, @floatCast(value)) };
        }
        
        /// Create a dictionary attribute from named attributes.
        pub fn dictionary(context: Context, named_attrs: []const c.c.MlirNamedAttribute) Self {
            return Self{ .handle = c.c.dictionaryAttrGet(context.handle, @intCast(named_attrs.len), named_attrs.ptr) };
        }
        
        /// Create a type attribute from a type
        pub fn typeAttr(type_val: Type) Self {
            return Self{ .handle = c.c.typeAttr(type_val.handle) };
        }
        
        /// Create a string attribute
        pub fn stringAttr(context: Context, value: []const u8) Self {
            const value_ref = c.c.stringRefFromString(value);
            return Self{ .handle = c.c.stringAttrGet(context.handle, value_ref) };
        }
        
        /// Create a symbol reference attribute
        /// SAFE FIX: Use stringRefFromString which respects length, not null terminator
        pub fn symbolRefAttr(context: Context, value: []const u8) Self {
            const str_ref = c.c.stringRefFromString(value);
            const handle = c.c.mlirSymbolRefAttrGet(context.handle, str_ref, 0, null);
            return Self{ .handle = handle };
        }
        
        /// Create a boolean attribute
        pub fn boolAttr(context: Context, value: bool) Self {
            return Self{ .handle = c.c.mlirBoolAttrGet(context.handle, if (value) 1 else 0) };
        }
        
        /// Creates an attribute by parsing its textual representation.
        pub fn fromParseString(ctx: Context, attr_string: []const u8) !Self {
            const str_ref = c.c.stringRefFromString(attr_string);
            const handle = c.c.mlirAttributeParseGet(ctx.handle, str_ref);
            if (@intFromPtr(handle) == 0) {
                std.log.err("Failed to parse MLIR attribute from string: {s}", .{attr_string});
                return error.AttributeParseFailed;
            }
            return Self{ .handle = handle };
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


    /// MLIR Location - source location information
    /// FIXED: Now holds the struct directly (value type) instead of pointer
    pub const Location = struct {
        handle: c.c.MlirLocation,

        const Self = @This();

        pub fn unknown(context: Context) Self {
            // locationUnknownGet returns struct by value
            return Self{ .handle = c.c.locationUnknownGet(context.handle) };
        }
    };

    /// MLIR RankedTensorType - for tensor type introspection
    pub const RankedTensorType = struct {
        handle: c.c.MlirType,
        
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
        
        /// Get the shape as a slice of dimensions
        /// Caller is responsible for freeing the returned slice
        pub fn getShape(self: Self, allocator: std.mem.Allocator) ![]i64 {
            const rank = self.getRank();
            const shape = try allocator.alloc(i64, rank);
            for (0..rank) |i| {
                shape[i] = self.getDimension(i);
            }
            return shape;
        }
    };

    /// MLIR FunctionType - for function type introspection
    pub const FunctionType = struct {
        handle: c.c.MlirType,
        
        const Self = @This();
        
        /// Get the number of input types
        pub fn getNumInputs(self: Self) usize {
            return @intCast(c.c.functionTypeGetNumInputs(self.handle));
        }
        
        /// Get the number of result types
        pub fn getNumResults(self: Self) usize {
            return @intCast(c.c.functionTypeGetNumResults(self.handle));
        }
        
        /// Get a specific input type
        pub fn getInput(self: Self, pos: usize) Type {
            return Type{ .handle = c.c.functionTypeGetInput(self.handle, @intCast(pos)) };
        }
        
        /// Get a specific result type
        pub fn getResult(self: Self, pos: usize) Type {
            return Type{ .handle = c.c.functionTypeGetResult(self.handle, @intCast(pos)) };
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
        c.c.registerAllPasses();
    }
    
    /// String utilities for MLIR C API
    pub fn stringRefToSlice(ref: c.c.MlirStringRef) []const u8 {
        return ref.data[0..ref.length];
    }

/// MLIR Dialects
pub const dialects = struct {
    pub const stablehlo = @import("mlir/dialects/stablehlo.zig");
};