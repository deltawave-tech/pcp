const std = @import("std");

/// Raw MLIR C API bindings
pub const c = struct {
    // Basic MLIR C types
    pub const MlirContext = opaque {};
    pub const MlirType = opaque {};
    pub const MlirValue = opaque {};
    pub const MlirOperation = opaque {};
    pub const MlirBlock = opaque {};
    pub const MlirRegion = opaque {};
    // FIXED: MlirLocation must be a struct to match C ABI pass-by-value
    pub const MlirLocation = extern struct {
        ptr: ?*const anyopaque,
    };
    pub const MlirModule = opaque {};
    pub const MlirAttribute = opaque {};
    pub const MlirDialect = opaque {};
    pub const MlirDialectRegistry = opaque {};
    pub const MlirPassManager = opaque {};
    pub const MlirOpPassManager = opaque {};
    // INCREASED SIZE: The C struct appears to be larger than 128 bytes in this build.
    // We increase this to 1024 to safely cover the struct size and avoid stack smashing/garbage reads.
    pub const StateSize: usize = 1024;
    pub const StateAlign: usize = 8; // Standard pointer alignment on 64-bit

    pub const OpaqueState = extern struct {
        data: [StateSize]u8 align(StateAlign),
    };

    // Phantom definition for casting (layout overlay)
    pub const MlirOperationState = extern struct {
        name: MlirStringRef,
        location: MlirLocation,
        // Pad the rest to the full size
        _pad: [StateSize - @sizeOf(MlirStringRef) - @sizeOf(MlirLocation)]u8,
    };

    // String reference
    pub const MlirStringRef = extern struct {
        data: [*]const u8,
        length: usize,
    };

    // String utilities
    pub extern fn mlirStringRefCreateFromCString(str: [*:0]const u8) MlirStringRef;

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
    extern fn mlirLocationUnknownGet(ctx: *MlirContext) MlirLocation;

    // Module operations
    extern fn mlirModuleCreateEmpty(location: MlirLocation) *MlirModule;
    extern fn mlirModuleDestroy(module: *MlirModule) void;
    extern fn mlirModuleGetOperation(module: *MlirModule) *MlirOperation;
    extern fn mlirModuleCreateParse(ctx: *MlirContext, module_str: MlirStringRef) *MlirModule;

    // Operation operations
    pub const MlirIdentifier = opaque {};
    pub extern fn mlirOperationCreate(state: *MlirOperationState) *MlirOperation;
    // ENSURE THIS EXTERN IS PRESENT AND PUBLIC
    pub extern fn mlirOperationDestroy(op: *MlirOperation) void;
    extern fn mlirOperationClone(op: *MlirOperation) *MlirOperation;
    pub extern fn mlirOperationPrint(op: *MlirOperation, callback: *const fn ([*]const u8, usize, ?*anyopaque) callconv(.C) void, userData: ?*anyopaque) void;
    extern fn mlirOperationDump(op: *MlirOperation) void;
    pub extern fn mlirOperationGetResult(op: *MlirOperation, pos: isize) *MlirValue;
    // ADD THIS NEW EXTERN
    pub extern fn mlirOperationVerify(op: *MlirOperation) MlirLogicalResult;
    pub extern fn mlirOperationGetOperand(op: *MlirOperation, pos: isize) *MlirValue;
    pub extern fn mlirOperationSetOperand(op: *MlirOperation, pos: isize, newValue: *MlirValue) void;
    pub extern fn mlirOperationGetName(op: *MlirOperation) *MlirIdentifier;
    pub extern fn mlirIdentifierStr(identifier: *MlirIdentifier) MlirStringRef;
    pub extern fn mlirOperationGetAttributeByName(operation: *MlirOperation, name: MlirStringRef) *MlirAttribute;
    pub extern fn mlirOperationRemoveAttributeByName(op: *MlirOperation, name: MlirStringRef) bool;
    pub extern fn mlirAttributeParseGet(ctx: *MlirContext, str: MlirStringRef) *MlirAttribute;
    pub extern fn mlirTypeAttrGetValue(attr: *MlirAttribute) *MlirType;
    
    // ADD THIS NEW EXTERN FOR SymbolRefAttr
    pub extern fn mlirSymbolRefAttrGet(ctx: *MlirContext, value: MlirStringRef) *MlirAttribute;
    pub extern fn mlirOperationGetLocation(op: *MlirOperation) MlirLocation;
    pub extern fn mlirOperationGetNumAttributes(op: *MlirOperation) isize;
    pub extern fn mlirOperationGetAttribute(op: *MlirOperation, pos: isize) MlirNamedAttribute;

    // Add this extern
    pub extern fn mlirBoolAttrGet(ctx: *MlirContext, value: c_int) *MlirAttribute;

    // Graph traversal operations
    extern fn mlirOperationGetRegion(op: *MlirOperation, pos: isize) *MlirRegion;
    extern fn mlirRegionGetFirstBlock(region: *MlirRegion) *MlirBlock;
    extern fn mlirBlockGetFirstOperation(block: *MlirBlock) *MlirOperation;
    extern fn mlirBlockGetTerminator(block: *MlirBlock) *MlirOperation;
    pub extern fn mlirRegionCreate() *MlirRegion;
    extern fn mlirRegionAppendOwnedBlock(region: *MlirRegion, block: *MlirBlock) void;
    extern fn mlirOperationGetNextInBlock(op: *MlirOperation) *MlirOperation;
    extern fn mlirOperationGetNumOperands(op: *MlirOperation) isize;
    extern fn mlirOperationGetNumResults(op: *MlirOperation) isize;

    // Operation state operations
    pub extern fn mlirOperationStateGet(name: MlirStringRef, location: MlirLocation) MlirOperationState;
    pub extern fn mlirOperationStateAddOperands(state: *MlirOperationState, n: isize, operands: [*]*MlirValue) void;
    pub extern fn mlirOperationStateAddResults(state: *MlirOperationState, n: isize, results: [*]*MlirType) void;
    // DEFINITIVE FIX: The C API expects a pointer to the first element of the array (*),
    // not a pointer-to-a-pointer (**). The binding is now corrected to [*]const.
    pub extern fn mlirOperationStateAddAttributes(state: *MlirOperationState, n: isize, attributes: [*]const MlirNamedAttribute) void;
    pub extern fn mlirOperationStateAddOwnedRegions(state: *MlirOperationState, n: isize, regions: [*]*MlirRegion) void;

    // Named attribute
    pub const MlirNamedAttribute = extern struct {
        name: *MlirIdentifier,
        attribute: *MlirAttribute,
    };
    pub extern fn mlirNamedAttributeGet(name: MlirStringRef, attr: *MlirAttribute) MlirNamedAttribute;

    // Type operations
    extern fn mlirF32TypeGet(ctx: *MlirContext) *MlirType;
    extern fn mlirF64TypeGet(ctx: *MlirContext) *MlirType;
    pub extern fn mlirIntegerTypeGet(ctx: *MlirContext, bitwidth: c_uint) *MlirType;
    extern fn mlirRankedTensorTypeGet(rank: isize, shape: [*]const i64, elementType: *MlirType, encoding: *MlirAttribute) *MlirType;
    // Correct MLIR C API functions from BuiltinTypes.h
    extern fn mlirShapedTypeGetRank(type: *MlirType) i64;
    extern fn mlirShapedTypeGetDimSize(type: *MlirType, dim: isize) i64;
    extern fn mlirShapedTypeGetElementType(type: *MlirType) *MlirType;
    extern fn mlirTypeIsARankedTensor(type: *MlirType) bool;

    // Value operations
    extern fn mlirValueGetType(value: *MlirValue) *MlirType;
    pub extern fn mlirBlockAddArgument(block: *MlirBlock, type: *MlirType, loc: MlirLocation) *MlirValue;
    pub extern fn mlirValueIsAOpResult(value: *MlirValue) bool;
    pub extern fn mlirOpResultGetOwner(value: *MlirValue) *MlirOperation;
    pub extern fn mlirValueDump(value: *MlirValue) void;
    extern fn mlirStringRefFromString(str: [*:0]const u8) MlirStringRef;
    extern fn mlirBlockGetArgument(block: *MlirBlock, pos: isize) *MlirValue;
    
    // Dense array attribute operations
    pub extern fn mlirAttributeIsADenseI64Array(attr: *MlirAttribute) bool;
    pub extern fn mlirDenseI64ArrayGet(ctx: *MlirContext, num_elements: isize, elements: [*]const i64) *MlirAttribute;
    pub extern fn mlirDenseArrayGetNumElements(attr: *MlirAttribute) isize;
    pub extern fn mlirDenseI64ArrayGetElement(attr: *MlirAttribute, pos: isize) i64;
    extern fn mlirBlockGetNumArguments(block: *MlirBlock) isize;

    // Function type operations
    extern fn mlirFunctionTypeGet(ctx: *MlirContext, numInputs: isize, inputs: [*]const *MlirType, numResults: isize, results: [*]const *MlirType) *MlirType;
    extern fn mlirFunctionTypeGetNumInputs(type: *MlirType) isize;
    extern fn mlirFunctionTypeGetNumResults(type: *MlirType) isize;
    extern fn mlirFunctionTypeGetInput(type: *MlirType, pos: isize) *MlirType;
    extern fn mlirFunctionTypeGetResult(type: *MlirType, pos: isize) *MlirType;
    extern fn mlirTypeIsAFunction(type: *MlirType) bool;
    extern fn mlirTypeAttrGet(type: *MlirType) *MlirAttribute;

    // OpBuilder operations (TODO: Find correct MLIR C API functions)
    // extern fn mlirOpBuilderCreate(ctx: *MlirContext) *MlirOpBuilder;
    // extern fn mlirOpBuilderDestroy(builder: *MlirOpBuilder) void;

    // Block operations
    extern fn mlirBlockCreate(nArgs: isize, args: [*]*MlirType, locs: [*]const MlirLocation) *MlirBlock;
    extern fn mlirBlockDestroy(block: *MlirBlock) void;
    pub extern fn mlirModuleGetBody(module: *MlirModule) *MlirBlock;
    pub extern fn mlirBlockAppendOwnedOperation(block: *MlirBlock, operation: *MlirOperation) void;
    extern fn mlirBlockInsertArgument(block: *MlirBlock, pos: isize, type: *MlirType, loc: MlirLocation) *MlirValue;

    // Attribute operations for constants
    extern fn mlirAttributeGetNull() *MlirAttribute;
    pub extern fn mlirDenseElementsAttrRawBufferGet(shapedType: *MlirType, rawBufferSize: usize, rawBuffer: *const anyopaque) *MlirAttribute;
    pub extern fn mlirDenseElementsAttrFloatSplatGet(shapedType: *MlirType, value: f64) *MlirAttribute;
    pub extern fn mlirFloatAttrDoubleGet(ctx: *MlirContext, type: *MlirType, value: f64) *MlirAttribute;
    pub extern fn mlirIntegerAttrGet(type: *MlirType, value: i64) *MlirAttribute;
    extern fn mlirAttributeIsAString(attr: *MlirAttribute) bool;
    extern fn mlirStringAttrGet(ctx: *MlirContext, str: MlirStringRef) *MlirAttribute;
    
    // --- ADD THESE TWO NEW EXTERNS ---
    pub extern fn mlirUnitAttrGet(ctx: *MlirContext) *MlirAttribute;
    pub extern fn mlirOperationSetAttributeByName(op: *MlirOperation, name: MlirStringRef, attr: *MlirAttribute) void;
    // Functions to extract data from dense elements attributes
    pub extern fn mlirDenseElementsAttrGetRawData(attr: *MlirAttribute) *const anyopaque;
    // pub extern fn mlirDenseElementsAttrGetNumElements(attr: *MlirAttribute) isize; // Not available in current MLIR build
    pub extern fn mlirAttributeIsADenseElements(attr: *MlirAttribute) bool;

    // Dictionary attribute operations
    pub extern fn mlirDictionaryAttrGet(ctx: *MlirContext, numElements: isize, elements: [*]const MlirNamedAttribute) *MlirAttribute;

    pub extern fn mlirIdentifierGet(ctx: *MlirContext, str: MlirStringRef) *MlirIdentifier;

    // String attribute functions
    pub const MlirStringAttribute = opaque {};
    extern fn mlirStringAttrGetValue(attr: *MlirStringAttribute) MlirStringRef;

    // Pass manager operations
    extern fn mlirPassManagerCreate(ctx: *MlirContext) *MlirPassManager;
    extern fn mlirPassManagerDestroy(pm: *MlirPassManager) void;
    extern fn mlirPassManagerGetAsOpPassManager(pm: *MlirPassManager) *MlirOpPassManager;
    extern fn mlirPassManagerRunOnOp(pm: *MlirPassManager, op: *MlirOperation) MlirLogicalResult;
    extern fn mlirOpPassManagerAddPipeline(opm: *MlirOpPassManager, pipelineElements: MlirStringRef, callback: ?*const fn (MlirLogicalResult, ?*anyopaque) callconv(.C) void, userData: ?*anyopaque) MlirLogicalResult;
    
    // NEW: Canonical GPU pipeline builder from C++ bridge
    extern fn mlirBuildAndAppendGpuAndSpirvConversionPipeline(pm: *MlirOpPassManager) void;
    
    // Register BufferizableOpInterface implementations with dialect registry
    extern fn mlirRegisterBufferizationInterfaces(registry: *MlirDialectRegistry) void;
    
    // Register Transform dialect extensions with dialect registry
    extern fn mlirRegisterTransformExtensions(registry: *MlirDialectRegistry) void;

    // Pass and dialect registration - dialect handle pattern
    pub const MlirDialectHandle = extern struct {
        ptr: ?*anyopaque,
    };
    pub extern fn mlirGetDialectHandle__func__() MlirDialectHandle;
    pub extern fn mlirGetDialectHandle__arith__() MlirDialectHandle;
    extern fn mlirGetDialectHandle__linalg__() MlirDialectHandle;
    extern fn mlirGetDialectHandle__gpu__() MlirDialectHandle;
    extern fn mlirGetDialectHandle__spirv__() MlirDialectHandle;
    extern fn mlirGetDialectHandle__scf__() MlirDialectHandle;
    // ADD THIS LINE for the Async dialect
    extern fn mlirGetDialectHandle__async__() MlirDialectHandle;
    // StableHLO dialect registration - NOW ENABLED!
    pub extern fn mlirGetDialectHandle__stablehlo__() MlirDialectHandle;
    extern fn mlirGetDialectHandle__chlo__() MlirDialectHandle;
    extern fn mlirGetDialectHandle__tensor__() MlirDialectHandle;
    extern fn mlirGetDialectHandle__transform__() MlirDialectHandle;

    pub extern fn mlirDialectHandleInsertDialect(handle: MlirDialectHandle, registry: *MlirDialectRegistry) void;
    pub extern fn mlirContextAppendDialectRegistry(ctx: *MlirContext, registry: *MlirDialectRegistry) void;
    pub extern fn mlirDialectHandleLoadDialect(handle: MlirDialectHandle, ctx: *MlirContext) *MlirDialect;
    extern fn mlirContextLoadAllAvailableDialects(ctx: *MlirContext) void;

    // Pass registration functions
    // Hybrid approach: Granular for core, bulk for StableHLO
    extern fn mlirRegisterAllStablehloPasses() void; // StableHLO C-API only provides bulk registration
    extern fn mlirRegisterTransformsCanonicalizer() void;
    extern fn mlirRegisterTransformsCSE() void;
    
    // Pass registration via C++ anchor function
    // Individual MLIR C API registration functions are not available
    
    // NEW: Single pass anchor function to force linker to load all required pass libraries
    extern fn mlirForceLoadAllRequiredPasses() void;

    // Pass types and management
    pub const MlirPass = opaque {};
    extern fn mlirOpPassManagerAddOwnedPass(pm: *MlirOpPassManager, pass: *MlirPass) void;

    // NEW: Specific dialect registration functions (commented out - not available in current MLIR build)
    // extern fn mlirRegisterStableHLODialect(registry: *MlirDialectRegistry) void;
    // extern fn mlirRegisterGPUDialect(registry: *MlirDialectRegistry) void;
    // extern fn mlirRegisterLinalgDialect(registry: *MlirDialectRegistry) void;
    // extern fn mlirRegisterSPIRVDialect(registry: *MlirDialectRegistry) void;
    // extern fn mlirRegisterVectorDialect(registry: *MlirDialectRegistry) void;
    // extern fn mlirRegisterAsyncDialect(registry: *MlirDialectRegistry) void;

    // Dialect handle operations (for registering specific dialects) - REMOVED
    // Using mlirContextLoadAllAvailableDialects instead

    // NEW: Translation to Target Formats (SPIR-V)
    pub const MlirStringCallback = *const fn (MlirStringRef, ?*anyopaque) callconv(.C) void;
    // SPIR-V translation functions - implemented in spirv_bridge.cpp
    extern fn mlirTranslateModuleToSPIRV(
        module: *MlirModule,
        callback: MlirStringCallback,
        userData: ?*anyopaque,
    ) MlirLogicalResult;
    
    // NEW: SPIRV-Cross translation functions - implemented in spirv_cross_bridge.cpp
    extern fn mlirTranslateSPIRVToMSL(
        spirv_data: *const anyopaque,
        spirv_size: usize,
        callback: MlirStringCallback,
        userData: ?*anyopaque,
    ) MlirLogicalResult;
    
    // GPU kernel metadata extraction
    pub const GPUKernelMetadata = extern struct {
        name: [*:0]const u8,
        grid_size: [3]u32,
        block_size: [3]u32,
    };
    
    extern fn mlirExtractGPUKernelMetadata(
        module: *MlirModule,
        out_kernels: *[*]GPUKernelMetadata,
    ) usize;
    
    extern fn mlirFreeGPUKernelMetadata(
        kernels: [*]GPUKernelMetadata,
        count: usize,
    ) void;

    // NEW EXTERNS FOR KERNEL NAME EXTRACTION FROM SPIR-V
    extern fn mlirExtractKernelNamesFromSPIRV(
        spirv_data: [*]const u8,
        spirv_size: usize,
        out_names: *[*][*:0]const u8,
    ) usize;

    extern fn mlirFreeKernelNames(
        names: [*][*:0]const u8,
        count: usize,
    ) void;

    // NEW: Module Walking Functions
    pub const MlirWalkResult = enum(c_int) {
        Advance,
        Interrupt,
        Skip,
    };

    pub const MlirOperationWalkCallback = *const fn (
        op: *MlirOperation,
        userData: ?*anyopaque,
    ) callconv(.C) MlirWalkResult;

    extern fn mlirOperationWalk(
        op: *MlirOperation,
        callback: MlirOperationWalkCallback,
        userData: ?*anyopaque,
    ) void;

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

    pub fn stringRefCreateFromCString(str: [*:0]const u8) MlirStringRef {
        return mlirStringRefCreateFromCString(str);
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

    // Dialect handle functions
    pub fn getDialectHandleFunc() MlirDialectHandle {
        return mlirGetDialectHandle__func__();
    }

    pub fn getDialectHandleArith() MlirDialectHandle {
        return mlirGetDialectHandle__arith__();
    }

    pub fn getDialectHandleLinalg() MlirDialectHandle {
        return mlirGetDialectHandle__linalg__();
    }

    pub fn getDialectHandleGPU() MlirDialectHandle {
        return mlirGetDialectHandle__gpu__();
    }

    pub fn getDialectHandleSPIRV() MlirDialectHandle {
        return mlirGetDialectHandle__spirv__();
    }

    pub fn getDialectHandleSCF() MlirDialectHandle {
        return mlirGetDialectHandle__scf__();
    }

    // ADD THIS WRAPPER for the Async dialect
    pub fn getDialectHandleAsync() MlirDialectHandle {
        return mlirGetDialectHandle__async__();
    }

    pub fn getDialectHandleStableHLO() MlirDialectHandle {
        return mlirGetDialectHandle__stablehlo__();
    }

    pub fn getDialectHandleCHLO() MlirDialectHandle {
        return mlirGetDialectHandle__chlo__();
    }

    pub fn getDialectHandleTensor() MlirDialectHandle {
        return mlirGetDialectHandle__tensor__();
    }

    pub fn getDialectHandleTransform() MlirDialectHandle {
        return mlirGetDialectHandle__transform__();
    }

    pub fn dialectHandleInsertDialect(handle: MlirDialectHandle, registry: *MlirDialectRegistry) void {
        mlirDialectHandleInsertDialect(handle, registry);
    }

    pub fn contextAppendDialectRegistry(ctx: *MlirContext, registry: *MlirDialectRegistry) void {
        mlirContextAppendDialectRegistry(ctx, registry);
    }

    pub fn dialectHandleLoadDialect(handle: MlirDialectHandle, ctx: *MlirContext) *MlirDialect {
        return mlirDialectHandleLoadDialect(handle, ctx);
    }

    pub fn contextLoadAllAvailableDialects(ctx: *MlirContext) void {
        mlirContextLoadAllAvailableDialects(ctx);
    }

    // Pass registration wrapper functions - COMPLETE
    pub fn registerAllStablehloPasses() void {
        mlirRegisterAllStablehloPasses();
    }

    pub fn registerCanonicalizerPass() void {
        mlirRegisterTransformsCanonicalizer();
    }

    pub fn registerCSEPass() void {
        mlirRegisterTransformsCSE();
    }
    
    // Pass registration uses single comprehensive anchor function
    
    // NEW: Zig wrapper for the consolidated pass anchor function
    pub fn forceLoadAllRequiredPasses() void {
        mlirForceLoadAllRequiredPasses();
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

    // NEW: Zig wrapper for canonical GPU pipeline builder
    pub fn buildAndAppendGpuAndSpirvConversionPipeline(pm: *MlirOpPassManager) void {
        mlirBuildAndAppendGpuAndSpirvConversionPipeline(pm);
    }
    
    // Register BufferizableOpInterface implementations with dialect registry
    pub fn registerBufferizationInterfaces(registry: *MlirDialectRegistry) void {
        mlirRegisterBufferizationInterfaces(registry);
    }
    
    // Register Transform dialect extensions with dialect registry
    pub fn registerTransformExtensions(registry: *MlirDialectRegistry) void {
        mlirRegisterTransformExtensions(registry);
    }

    // Module wrapper functions
    pub fn moduleCreateEmpty(location: MlirLocation) *MlirModule {
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
    pub fn locationUnknownGet(ctx: *MlirContext) MlirLocation {
        return mlirLocationUnknownGet(ctx);
    }

    // Operation wrapper functions
    pub fn operationCreate(state: *MlirOperationState) *MlirOperation {
        return mlirOperationCreate(state);
    }

    pub fn operationStateGet(name: []const u8, location: MlirLocation) MlirOperationState {
        return mlirOperationStateGet(stringRefFromString(name), location);
    }

    pub fn operationStateAddOwnedRegions(state: *MlirOperationState, n: isize, regions: [*]*MlirRegion) void {
        mlirOperationStateAddOwnedRegions(state, n, regions);
    }

    pub fn operationGetResult(op: *MlirOperation, pos: isize) *MlirValue {
        return mlirOperationGetResult(op, pos);
    }

    // ENSURE THIS WRAPPER IS PRESENT
    pub fn operationDestroy(op: *MlirOperation) void {
        mlirOperationDestroy(op);
    }
    
    // ADD THIS NEW WRAPPER
    pub fn operationVerify(op: *MlirOperation) MlirLogicalResult {
        return mlirOperationVerify(op);
    }

    pub fn operationClone(op: *MlirOperation) *MlirOperation {
        return mlirOperationClone(op);
    }

    pub fn operationDump(op: *MlirOperation) void {
        mlirOperationDump(op);
    }

    // Graph traversal wrapper functions
    pub fn operationGetRegion(op: *MlirOperation, pos: isize) *MlirRegion {
        return mlirOperationGetRegion(op, pos);
    }

    pub fn regionGetFirstBlock(region: *MlirRegion) *MlirBlock {
        return mlirRegionGetFirstBlock(region);
    }

    pub fn regionCreate() *MlirRegion {
        return mlirRegionCreate();
    }

    pub fn blockGetFirstOperation(block: *MlirBlock) *MlirOperation {
        return mlirBlockGetFirstOperation(block);
    }

    pub fn operationGetNextInBlock(op: *MlirOperation) *MlirOperation {
        return mlirOperationGetNextInBlock(op);
    }

    pub fn operationGetNumOperands(op: *MlirOperation) isize {
        return mlirOperationGetNumOperands(op);
    }

    pub fn operationGetNumResults(op: *MlirOperation) isize {
        return mlirOperationGetNumResults(op);
    }

    pub fn blockGetTerminator(block: *MlirBlock) *MlirOperation {
        return mlirBlockGetTerminator(block);
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

    // Translation wrapper functions
    pub fn translateModuleToSPIRV(
        module: *MlirModule,
        writer: MlirStringCallback,
        userData: ?*anyopaque,
    ) MlirLogicalResult {
        return mlirTranslateModuleToSPIRV(module, writer, userData);
    }
    
    pub fn translateSPIRVToMSL(
        spirv_data: *const anyopaque,
        spirv_size: usize,
        writer: MlirStringCallback,
        userData: ?*anyopaque,
    ) MlirLogicalResult {
        return mlirTranslateSPIRVToMSL(spirv_data, spirv_size, writer, userData);
    }
    
    pub fn extractGPUKernelMetadata(
        module: *MlirModule,
        out_kernels: *[*]GPUKernelMetadata,
    ) usize {
        return mlirExtractGPUKernelMetadata(module, out_kernels);
    }
    
    pub fn freeGPUKernelMetadata(
        kernels: [*]GPUKernelMetadata,
        count: usize,
    ) void {
        mlirFreeGPUKernelMetadata(kernels, count);
    }

    // NEW WRAPPERS FOR SPIRV KERNEL EXTRACTION
    pub fn extractKernelNamesFromSPIRV(
        spirv_data: []const u8,
        out_names: *[*][*:0]const u8,
    ) usize {
        return mlirExtractKernelNamesFromSPIRV(spirv_data.ptr, spirv_data.len, out_names);
    }
    
    pub fn freeKernelNames(
        names: [*][*:0]const u8,
        count: usize,
    ) void {
        mlirFreeKernelNames(names, count);
    }

    // Module walking wrapper functions
    pub fn operationWalk(
        op: *MlirOperation,
        callback: MlirOperationWalkCallback,
        userData: ?*anyopaque,
    ) void {
        mlirOperationWalk(op, callback, userData);
    }

    // Operation introspection wrapper functions
    pub fn operationGetName(operation: *MlirOperation) *MlirIdentifier {
        return mlirOperationGetName(operation);
    }

    pub fn operationGetAttributeByName(operation: *MlirOperation, name: []const u8) *MlirAttribute {
        return mlirOperationGetAttributeByName(operation, stringRefFromString(name));
    }

    pub fn operationGetNumAttributes(op: *MlirOperation) isize {
        return mlirOperationGetNumAttributes(op);
    }

    pub fn operationGetAttribute(op: *MlirOperation, pos: isize) MlirNamedAttribute {
        return mlirOperationGetAttribute(op, pos);
    }

    pub fn operationGetLocation(op: *MlirOperation) MlirLocation {
        return mlirOperationGetLocation(op);
    }

    pub fn regionAppendOwnedBlock(region: *MlirRegion, block: *MlirBlock) void {
        mlirRegionAppendOwnedBlock(region, block);
    }

    pub fn identifierStr(identifier: *MlirIdentifier) MlirStringRef {
        return mlirIdentifierStr(identifier);
    }

    // String attribute wrapper functions
    pub fn attributeIsAString(attr: *MlirAttribute) bool {
        return mlirAttributeIsAString(attr);
    }

    pub fn stringAttributeGetValue(attr: *MlirStringAttribute) MlirStringRef {
        return mlirStringAttrGetValue(attr);
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
    pub fn blockCreate(nArgs: isize, args: [*]*MlirType, locs: [*]const MlirLocation) *MlirBlock {
        return mlirBlockCreate(nArgs, args, locs);
    }

    pub fn blockDestroy(block: *MlirBlock) void {
        mlirBlockDestroy(block);
    }

    pub fn blockAddArgument(block: *MlirBlock, pos: isize, arg_type: *MlirType, loc: MlirLocation) *MlirValue {
        return mlirBlockInsertArgument(block, pos, arg_type, loc);
    }

    pub fn blockGetArgument(block: *MlirBlock, pos: isize) *MlirValue {
        return mlirBlockGetArgument(block, pos);
    }

    pub fn blockGetNumArguments(block: *MlirBlock) isize {
        return mlirBlockGetNumArguments(block);
    }

    pub fn functionTypeGet(ctx: *MlirContext, inputs: []const *MlirType, results: []const *MlirType) *MlirType {
        return mlirFunctionTypeGet(ctx, @intCast(inputs.len), inputs.ptr, @intCast(results.len), results.ptr);
    }

    pub fn functionTypeGetNumInputs(func_type: *MlirType) isize {
        return mlirFunctionTypeGetNumInputs(func_type);
    }

    pub fn functionTypeGetNumResults(func_type: *MlirType) isize {
        return mlirFunctionTypeGetNumResults(func_type);
    }

    pub fn functionTypeGetInput(func_type: *MlirType, pos: isize) *MlirType {
        return mlirFunctionTypeGetInput(func_type, pos);
    }

    pub fn functionTypeGetResult(func_type: *MlirType, pos: isize) *MlirType {
        return mlirFunctionTypeGetResult(func_type, pos);
    }

    pub fn typeIsAFunction(mlir_type: *MlirType) bool {
        return mlirTypeIsAFunction(mlir_type);
    }

    pub fn typeAttr(type_val: *MlirType) *MlirAttribute {
        return mlirTypeAttrGet(type_val);
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

    pub fn identifierGet(ctx: *MlirContext, str: []const u8) *MlirIdentifier {
        return mlirIdentifierGet(ctx, stringRefFromString(str));
    }

    pub fn dictionaryAttrGet(ctx: *MlirContext, elements: []const MlirNamedAttribute) *MlirAttribute {
        return mlirDictionaryAttrGet(ctx, @intCast(elements.len), elements.ptr);
    }

    pub fn stringAttrGet(ctx: *MlirContext, str: []const u8) *MlirAttribute {
        return mlirStringAttrGet(ctx, stringRefFromString(str));
    }

    // --- ADD THESE TWO NEW WRAPPERS ---
    pub fn unitAttrGet(ctx: *MlirContext) *MlirAttribute {
        return mlirUnitAttrGet(ctx);
    }
    
    pub fn operationSetAttributeByName(op: *MlirOperation, name: []const u8, attr: *MlirAttribute) void {
        mlirOperationSetAttributeByName(op, stringRefFromString(name), attr);
    }

    pub fn operationRemoveAttributeByName(op: *MlirOperation, name: []const u8) bool {
        return mlirOperationRemoveAttributeByName(op, stringRefFromString(name));
    }

    pub fn attributeParseGet(ctx: *MlirContext, str: []const u8) *MlirAttribute {
        return mlirAttributeParseGet(ctx, stringRefFromString(str));
    }

    // Dense array attribute wrapper functions
    pub fn attributeIsADenseI64Array(attr: *MlirAttribute) bool {
        return mlirAttributeIsADenseI64Array(attr);
    }

    pub fn denseArrayGetNumElements(attr: *MlirAttribute) isize {
        return mlirDenseArrayGetNumElements(attr);
    }

    pub fn denseI64ArrayGet(ctx: *MlirContext, num_elements: isize, elements: []const i64) *MlirAttribute {
        return mlirDenseI64ArrayGet(ctx, num_elements, elements.ptr);
    }

    pub fn denseI64ArrayGetElement(attr: *MlirAttribute, pos: isize) i64 {
        return mlirDenseI64ArrayGetElement(attr, pos);
    }
};
