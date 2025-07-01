const std = @import("std");

/// Raw MLIR C API bindings - always available since we require MLIR
pub const c = struct {
    // Basic MLIR C types
    pub const MlirContext = opaque {};
    pub const MlirType = opaque {};
    pub const MlirValue = opaque {};
    pub const MlirOperation = opaque {};
    pub const MlirBlock = opaque {};
    pub const MlirRegion = opaque {};
    pub const MlirLocation = opaque {};
    pub const MlirModule = opaque {};
    pub const MlirAttribute = opaque {};
    pub const MlirDialectRegistry = opaque {};
    pub const MlirPassManager = opaque {};
    pub const MlirOpPassManager = opaque {};
    pub const MlirOperationState = extern struct {
        name: MlirStringRef,
        location: *MlirLocation,
        nResults: isize,
        results: [*]*MlirType,
        nOperands: isize,
        operands: [*]*MlirValue,
        nRegions: isize,
        regions: [*]*MlirRegion,
        nSuccessors: isize,
        successors: [*]*MlirBlock,
        nAttributes: isize,
        attributes: [*]*MlirAttribute,
        enableResultTypeInference: bool,
    };
    
    // String reference
    pub const MlirStringRef = extern struct {
        data: [*]const u8,
        length: usize,
    };
    
    // Logical result for MLIR operations
    pub const MlirLogicalResult = extern struct {
        value: i8,
        
        pub fn isSuccess(self: MlirLogicalResult) bool {
            return self.value != 0;
        }
        
        pub fn isFailure(self: MlirLogicalResult) bool {
            return self.value == 0;
        }
    };
    
    // Context operations
    extern fn mlirContextCreate() *MlirContext;
    extern fn mlirContextDestroy(ctx: *MlirContext) void;
    extern fn mlirContextSetAllowUnregisteredDialects(ctx: *MlirContext, allow: bool) void;
    extern fn mlirContextGetNumRegisteredDialects(ctx: *MlirContext) isize;
    extern fn mlirContextIsRegisteredOperation(ctx: *MlirContext, name: MlirStringRef) bool;
    
    // Dialect registry operations
    extern fn mlirDialectRegistryCreate() *MlirDialectRegistry;
    extern fn mlirDialectRegistryDestroy(registry: *MlirDialectRegistry) void;
    
    // Location operations
    extern fn mlirLocationUnknownGet(ctx: *MlirContext) *MlirLocation;
    
    // Module operations
    extern fn mlirModuleCreateEmpty(location: *MlirLocation) *MlirModule;
    extern fn mlirModuleDestroy(module: *MlirModule) void;
    extern fn mlirModuleGetOperation(module: *MlirModule) *MlirOperation;
    extern fn mlirModuleCreateParse(ctx: *MlirContext, module_str: MlirStringRef) *MlirModule;
    
    // Operation operations
    pub const MlirIdentifier = opaque {};
    extern fn mlirOperationCreate(state: *MlirOperationState) *MlirOperation;
    extern fn mlirOperationDestroy(op: *MlirOperation) void;
    extern fn mlirOperationPrint(op: *MlirOperation, callback: *const fn([*]const u8, usize, ?*anyopaque) callconv(.C) void, userData: ?*anyopaque) void;
    extern fn mlirOperationDump(op: *MlirOperation) void;
    extern fn mlirOperationGetResult(op: *MlirOperation, pos: isize) *MlirValue;
    extern fn mlirOperationGetOperand(op: *MlirOperation, pos: isize) *MlirValue;
    extern fn mlirOperationGetName(op: *MlirOperation) *MlirIdentifier;
    extern fn mlirIdentifierStr(identifier: *MlirIdentifier) MlirStringRef;
    extern fn mlirOperationGetAttributeByName(operation: *MlirOperation, name: MlirStringRef) *MlirAttribute;
    
    // Operation state operations
    extern fn mlirOperationStateGet(name: MlirStringRef, location: *MlirLocation) MlirOperationState;
    extern fn mlirOperationStateAddOperands(state: *MlirOperationState, n: isize, operands: [*]*MlirValue) void;
    extern fn mlirOperationStateAddResults(state: *MlirOperationState, n: isize, results: [*]*MlirType) void;
    extern fn mlirOperationStateAddAttributes(state: *MlirOperationState, n: isize, attributes: [*]*MlirNamedAttribute) void;
    
    // Named attribute
    pub const MlirNamedAttribute = extern struct {
        name: MlirIdentifier,
        attribute: *MlirAttribute,
    };
    extern fn mlirNamedAttributeGet(name: MlirStringRef, attr: *MlirAttribute) MlirNamedAttribute;
    
    // Type operations
    extern fn mlirF32TypeGet(ctx: *MlirContext) *MlirType;
    extern fn mlirF64TypeGet(ctx: *MlirContext) *MlirType;
    extern fn mlirRankedTensorTypeGet(rank: isize, shape: [*]const i64, elementType: *MlirType, encoding: *MlirAttribute) *MlirType;
    // Correct MLIR C API functions from BuiltinTypes.h
    extern fn mlirShapedTypeGetRank(type: *MlirType) i64;
    extern fn mlirShapedTypeGetDimSize(type: *MlirType, dim: isize) i64;
    extern fn mlirShapedTypeGetElementType(type: *MlirType) *MlirType;
    extern fn mlirTypeIsARankedTensor(type: *MlirType) bool;
    
    // Value operations
    extern fn mlirValueGetType(value: *MlirValue) *MlirType;
    extern fn mlirBlockAddArgument(block: *MlirBlock, type: *MlirType, loc: *MlirLocation) *MlirValue;
    
    // OpBuilder operations (TODO: Find correct MLIR C API functions)
    // extern fn mlirOpBuilderCreate(ctx: *MlirContext) *MlirOpBuilder;
    // extern fn mlirOpBuilderDestroy(builder: *MlirOpBuilder) void;
    
    // Block operations
    extern fn mlirBlockCreate(nArgs: isize, args: [*]*MlirType, locs: [*]*MlirLocation) *MlirBlock;
    extern fn mlirBlockDestroy(block: *MlirBlock) void;
    extern fn mlirModuleGetBody(module: *MlirModule) *MlirBlock;
    extern fn mlirBlockAppendOwnedOperation(block: *MlirBlock, operation: *MlirOperation) void;
    extern fn mlirBlockInsertArgument(block: *MlirBlock, pos: isize, type: *MlirType, loc: *MlirLocation) *MlirValue;
    
    // Attribute operations for constants
    extern fn mlirAttributeGetNull() *MlirAttribute;
    extern fn mlirDenseElementsAttrRawBufferGet(shapedType: *MlirType, rawBufferSize: usize, rawBuffer: *const anyopaque) *MlirAttribute;
    extern fn mlirFloatAttrDoubleGet(ctx: *MlirContext, type: *MlirType, value: f64) *MlirAttribute;
    extern fn mlirIntegerAttrGet(type: *MlirType, value: i64) *MlirAttribute;
    extern fn mlirDenseI64ArrayGet(ctx: *MlirContext, size: isize, values: [*]const i64) *MlirAttribute;
    extern fn mlirAttributeIsAString(attr: *MlirAttribute) bool;
    
    // String attribute functions
    pub const MlirStringAttribute = opaque {};
    extern fn mlirStringAttributeGetValue(attr: *MlirStringAttribute) MlirStringRef;
    
    // Pass manager operations
    extern fn mlirPassManagerCreate(ctx: *MlirContext) *MlirPassManager;
    extern fn mlirPassManagerDestroy(pm: *MlirPassManager) void;
    extern fn mlirPassManagerGetAsOpPassManager(pm: *MlirPassManager) *MlirOpPassManager;
    extern fn mlirPassManagerRunOnOp(pm: *MlirPassManager, op: *MlirOperation) MlirLogicalResult;
    extern fn mlirOpPassManagerAddPipeline(opm: *MlirOpPassManager, pipelineElements: MlirStringRef, callback: ?*const fn(MlirLogicalResult, ?*anyopaque) callconv(.C) void, userData: ?*anyopaque) MlirLogicalResult;
    
    // Pass and dialect registration
    extern fn mlirRegisterAllDialects(registry: *MlirDialectRegistry) void;
    pub extern fn mlirRegisterAllPasses() void;
    extern fn mlirContextLoadAllAvailableDialects(ctx: *MlirContext) void;
    
    // NEW: Specific dialect registration functions
    extern fn mlirRegisterStableHLODialect(registry: *MlirDialectRegistry) void;
    extern fn mlirRegisterGPUDialect(registry: *MlirDialectRegistry) void;
    extern fn mlirRegisterLinalgDialect(registry: *MlirDialectRegistry) void;
    extern fn mlirRegisterSPIRVDialect(registry: *MlirDialectRegistry) void;
    extern fn mlirRegisterVectorDialect(registry: *MlirDialectRegistry) void;
    extern fn mlirRegisterAsyncDialect(registry: *MlirDialectRegistry) void;
    
    // Dialect handle operations (for registering specific dialects)
    pub const MlirDialectHandle = opaque {};
    extern fn mlirGetDialectHandle__func__() *MlirDialectHandle;
    extern fn mlirGetDialectHandle__arith__() *MlirDialectHandle;
    extern fn mlirGetDialectHandle__scf__() *MlirDialectHandle;
    extern fn mlirGetDialectHandle__gpu__() *MlirDialectHandle;
    extern fn mlirGetDialectHandle__linalg__() *MlirDialectHandle;
    extern fn mlirGetDialectHandle__spirv__() *MlirDialectHandle;
    extern fn mlirDialectHandleInsertDialect(handle: *MlirDialectHandle, registry: *MlirDialectRegistry) void;
    
    // NEW: Translation to Target Formats (SPIR-V)
    pub const MlirStringCallback = *const fn(MlirStringRef, ?*anyopaque) callconv(.C) void;
    extern fn mlirTranslateModuleToSPIRV(
        module: *MlirModule,
        writer: MlirStringCallback,
        userData: ?*anyopaque,
    ) MlirLogicalResult;
    
    // NEW: Module Walking Functions
    pub const MlirWalkResult = enum(c_int) {
        Advance,
        Interrupt,
        Skip,
    };
    
    pub const MlirOperationWalkCallback = *const fn(
        op: *MlirOperation,
        userData: ?*anyopaque,
    ) callconv(.C) MlirWalkResult;
    
    extern fn mlirOperationWalk(
        op: *MlirOperation,
        callback: MlirOperationWalkCallback,
        userData: ?*anyopaque,
    ) void;
    
    // NEW: Operation introspection functions (declarations moved to earlier section)
    
    // NEW: String attribute functions (declarations moved to earlier section)
    
    // Helper functions
    pub fn stringRefFromString(str: []const u8) MlirStringRef {
        return MlirStringRef{
            .data = str.ptr,
            .length = str.len,
        };
    }
    
    pub fn stringRefToString(sref: MlirStringRef, allocator: std.mem.Allocator) ![]u8 {
        const result = try allocator.alloc(u8, sref.length);
        @memcpy(result, sref.data[0..sref.length]);
        return result;
    }
    
    // Context wrapper functions
    pub fn contextCreate() *MlirContext {
        return mlirContextCreate();
    }
    
    pub fn contextDestroy(ctx: *MlirContext) void {
        mlirContextDestroy(ctx);
    }
    
    pub fn contextSetAllowUnregisteredDialects(ctx: *MlirContext, allow: bool) void {
        mlirContextSetAllowUnregisteredDialects(ctx, allow);
    }
    
    pub fn contextGetNumRegisteredDialects(ctx: *MlirContext) isize {
        return mlirContextGetNumRegisteredDialects(ctx);
    }
    
    pub fn contextIsRegisteredOperation(ctx: *MlirContext, name: []const u8) bool {
        return mlirContextIsRegisteredOperation(ctx, stringRefFromString(name));
    }
    
    // Registry wrapper functions
    pub fn dialectRegistryCreate() *MlirDialectRegistry {
        return mlirDialectRegistryCreate();
    }
    
    pub fn dialectRegistryDestroy(registry: *MlirDialectRegistry) void {
        mlirDialectRegistryDestroy(registry);
    }
    
    pub fn registerAllDialects(registry: *MlirDialectRegistry) void {
        mlirRegisterAllDialects(registry);
    }
    
    pub fn dialectRegistryLoadAll(registry: *MlirDialectRegistry, ctx: *MlirContext) void {
        _ = registry; // Unused in new API
        mlirContextLoadAllAvailableDialects(ctx);
    }
    
    // Pass manager wrapper functions
    pub fn passManagerCreate(ctx: *MlirContext) *MlirPassManager {
        return mlirPassManagerCreate(ctx);
    }
    
    pub fn passManagerDestroy(pm: *MlirPassManager) void {
        mlirPassManagerDestroy(pm);
    }
    
    pub fn passManagerGetAsOpPassManager(pm: *MlirPassManager) *MlirOpPassManager {
        return mlirPassManagerGetAsOpPassManager(pm);
    }
    
    pub fn passManagerRunOnOp(pm: *MlirPassManager, op: *MlirOperation) MlirLogicalResult {
        return mlirPassManagerRunOnOp(pm, op);
    }
    
    pub fn opPassManagerAddPipeline(opm: *MlirOpPassManager, pipeline: []const u8) MlirLogicalResult {
        return mlirOpPassManagerAddPipeline(opm, stringRefFromString(pipeline), null, null);
    }
    
    // Module wrapper functions
    pub fn moduleCreateEmpty(location: *MlirLocation) *MlirModule {
        return mlirModuleCreateEmpty(location);
    }
    
    pub fn moduleDestroy(module: *MlirModule) void {
        mlirModuleDestroy(module);
    }
    
    pub fn moduleGetOperation(module: *MlirModule) *MlirOperation {
        return mlirModuleGetOperation(module);
    }
    
    pub fn moduleParseString(ctx: *MlirContext, module_str: []const u8) *MlirModule {
        return mlirModuleCreateParse(ctx, stringRefFromString(module_str));
    }
    
    // Location wrapper functions
    pub fn locationUnknownGet(ctx: *MlirContext) *MlirLocation {
        return mlirLocationUnknownGet(ctx);
    }
    
    // Operation wrapper functions
    pub fn operationCreate(state: *MlirOperationState) *MlirOperation {
        return mlirOperationCreate(state);
    }
    
    pub fn operationStateGet(name: []const u8, location: *MlirLocation) MlirOperationState {
        return mlirOperationStateGet(stringRefFromString(name), location);
    }
    
    pub fn operationGetResult(op: *MlirOperation, pos: isize) *MlirValue {
        return mlirOperationGetResult(op, pos);
    }
    
    pub fn operationDestroy(op: *MlirOperation) void {
        mlirOperationDestroy(op);
    }
    
    pub fn operationDump(op: *MlirOperation) void {
        mlirOperationDump(op);
    }
    
    // Type wrapper functions
    pub fn floatTypeGetF32(ctx: *MlirContext) *MlirType {
        return mlirF32TypeGet(ctx);
    }
    
    pub fn floatTypeGetF64(ctx: *MlirContext) *MlirType {
        return mlirF64TypeGet(ctx);
    }
    
    pub fn rankedTensorTypeGet(rank: isize, shape: []const i64, elementType: *MlirType, encoding: ?*MlirAttribute) *MlirType {
        return mlirRankedTensorTypeGet(rank, shape.ptr, elementType, encoding orelse attributeGetNull());
    }
    
    // Dialect handle helpers
    pub fn dialectHandleFuncGet() *MlirDialectHandle {
        return mlirGetDialectHandle__func__();
    }
    
    pub fn dialectHandleArithGet() *MlirDialectHandle {
        return mlirGetDialectHandle__arith__();
    }
    
    pub fn dialectHandleScfGet() *MlirDialectHandle {
        return mlirGetDialectHandle__scf__();
    }
    
    pub fn dialectHandleGPUGet() *MlirDialectHandle {
        return mlirGetDialectHandle__gpu__();
    }
    
    pub fn dialectHandleLinalgGet() *MlirDialectHandle {
        return mlirGetDialectHandle__linalg__();
    }
    
    pub fn dialectHandleSPIRVGet() *MlirDialectHandle {
        return mlirGetDialectHandle__spirv__();
    }
    
    pub fn dialectHandleInsertDialect(handle: *MlirDialectHandle, registry: *MlirDialectRegistry) void {
        mlirDialectHandleInsertDialect(handle, registry);
    }
    
    // NEW: Dialect registration wrapper functions
    pub fn registerStableHLODialect(registry: *MlirDialectRegistry) void {
        mlirRegisterStableHLODialect(registry);
    }
    
    pub fn registerGPUDialect(registry: *MlirDialectRegistry) void {
        mlirRegisterGPUDialect(registry);
    }
    
    pub fn registerLinalgDialect(registry: *MlirDialectRegistry) void {
        mlirRegisterLinalgDialect(registry);
    }
    
    pub fn registerSPIRVDialect(registry: *MlirDialectRegistry) void {
        mlirRegisterSPIRVDialect(registry);
    }
    
    pub fn registerVectorDialect(registry: *MlirDialectRegistry) void {
        mlirRegisterVectorDialect(registry);
    }
    
    pub fn registerAsyncDialect(registry: *MlirDialectRegistry) void {
        mlirRegisterAsyncDialect(registry);
    }
    
    // NEW: Translation wrapper functions
    pub fn translateModuleToSPIRV(
        module: *MlirModule,
        writer: MlirStringCallback,
        userData: ?*anyopaque,
    ) MlirLogicalResult {
        return mlirTranslateModuleToSPIRV(module, writer, userData);
    }
    
    // NEW: Module walking wrapper functions
    pub fn operationWalk(
        op: *MlirOperation,
        callback: MlirOperationWalkCallback,
        userData: ?*anyopaque,
    ) void {
        mlirOperationWalk(op, callback, userData);
    }
    
    // NEW: Operation introspection wrapper functions
    pub fn operationGetName(operation: *MlirOperation) *MlirIdentifier {
        return mlirOperationGetName(operation);
    }
    
    pub fn operationGetAttributeByName(operation: *MlirOperation, name: []const u8) *MlirAttribute {
        return mlirOperationGetAttributeByName(operation, stringRefFromString(name));
    }
    
    pub fn identifierStr(identifier: *MlirIdentifier) MlirStringRef {
        return mlirIdentifierStr(identifier);
    }
    
    // NEW: String attribute wrapper functions
    pub fn attributeIsAString(attr: *MlirAttribute) bool {
        return mlirAttributeIsAString(attr);
    }
    
    pub fn stringAttributeGetValue(attr: *MlirStringAttribute) MlirStringRef {
        return mlirStringAttributeGetValue(attr);
    }
    
    // Helper function to convert MlirStringRef to Zig string
    pub fn fromStringRef(ref: MlirStringRef) []const u8 {
        return ref.data[0..ref.length];
    }
    
    // Value wrapper functions
    pub fn valueGetType(value: *MlirValue) *MlirType {
        return mlirValueGetType(value);
    }
    
    // Type introspection wrapper functions
    pub fn typeIsARankedTensor(tensor_type: *MlirType) bool {
        return mlirTypeIsARankedTensor(tensor_type);
    }
    
    pub fn shapedTypeGetRank(tensor_type: *MlirType) i64 {
        return mlirShapedTypeGetRank(tensor_type);
    }
    
    pub fn shapedTypeGetDimSize(tensor_type: *MlirType, dim: isize) i64 {
        return mlirShapedTypeGetDimSize(tensor_type, dim);
    }
    
    pub fn shapedTypeGetElementType(tensor_type: *MlirType) *MlirType {
        return mlirShapedTypeGetElementType(tensor_type);
    }
    
    // OpBuilder wrapper functions (TODO: implement when we have correct API)
    // pub fn opBuilderCreate(ctx: *MlirContext) *MlirOpBuilder {
    //     return mlirOpBuilderCreate(ctx);
    // }
    
    // Block wrapper functions
    pub fn blockCreate(nArgs: isize, args: [*]*MlirType, locs: [*]*MlirLocation) *MlirBlock {
        return mlirBlockCreate(nArgs, args, locs);
    }
    
    pub fn blockDestroy(block: *MlirBlock) void {
        mlirBlockDestroy(block);
    }
    
    pub fn blockAddArgument(block: *MlirBlock, pos: isize, arg_type: *MlirType, loc: *MlirLocation) *MlirValue {
        return mlirBlockInsertArgument(block, pos, arg_type, loc);
    }
    
    pub fn blockAppendOwnedOperation(block: *MlirBlock, operation: *MlirOperation) void {
        mlirBlockAppendOwnedOperation(block, operation);
    }
    
    pub fn moduleGetBody(module: *MlirModule) *MlirBlock {
        return mlirModuleGetBody(module);
    }
    
    // Attribute wrapper functions
    pub fn attributeGetNull() *MlirAttribute {
        return mlirAttributeGetNull();
    }
    
    pub fn denseElementsAttrRawBufferGet(shaped_type: *MlirType, raw_buffer_size: usize, raw_buffer: *const anyopaque) *MlirAttribute {
        return mlirDenseElementsAttrRawBufferGet(shaped_type, raw_buffer_size, raw_buffer);
    }
    
};