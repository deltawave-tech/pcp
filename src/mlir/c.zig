const std = @import("std");

/// Automatically generated C bindings for MLIR C API
const mlir_c = @cImport({
    @cInclude("api.h");
});

/// Compatibility wrapper to maintain existing API
/// This allows gradual migration from manual bindings to auto-generated ones
pub const c = struct {
    // Re-export all types from the @cImport
    pub const MlirContext = mlir_c.MlirContext;
    pub const MlirType = mlir_c.MlirType;
    pub const MlirValue = mlir_c.MlirValue;
    pub const MlirOperation = mlir_c.MlirOperation;
    pub const MlirBlock = mlir_c.MlirBlock;
    pub const MlirRegion = mlir_c.MlirRegion;
    pub const MlirLocation = mlir_c.MlirLocation;
    pub const MlirModule = mlir_c.MlirModule;
    pub const MlirAttribute = mlir_c.MlirAttribute;
    pub const MlirDialect = mlir_c.MlirDialect;
    pub const MlirDialectRegistry = mlir_c.MlirDialectRegistry;
    pub const MlirPassManager = mlir_c.MlirPassManager;
    pub const MlirOpPassManager = mlir_c.MlirOpPassManager;
    pub const MlirStringRef = mlir_c.MlirStringRef;
    pub const MlirLogicalResult = mlir_c.MlirLogicalResult;
    pub const MlirIdentifier = mlir_c.MlirIdentifier;
    pub const MlirNamedAttribute = mlir_c.MlirNamedAttribute;
    pub const MlirOperationState = mlir_c.MlirOperationState;

    // Re-export all functions with both full and shortened names for compatibility
    pub const mlirContextCreate = mlir_c.mlirContextCreate;
    pub const contextCreate = mlir_c.mlirContextCreate;
    pub const mlirContextDestroy = mlir_c.mlirContextDestroy;
    pub const contextDestroy = mlir_c.mlirContextDestroy;
    pub const mlirContextSetAllowUnregisteredDialects = mlir_c.mlirContextSetAllowUnregisteredDialects;
    pub const contextSetAllowUnregisteredDialects = mlir_c.mlirContextSetAllowUnregisteredDialects;
    pub const mlirContextGetNumRegisteredDialects = mlir_c.mlirContextGetNumRegisteredDialects;
    pub const contextGetNumRegisteredDialects = mlir_c.mlirContextGetNumRegisteredDialects;
    pub const mlirContextIsRegisteredOperation = mlir_c.mlirContextIsRegisteredOperation;
    pub const contextIsRegisteredOperation = mlir_c.mlirContextIsRegisteredOperation;

    pub const mlirDialectRegistryCreate = mlir_c.mlirDialectRegistryCreate;
    pub const dialectRegistryCreate = mlir_c.mlirDialectRegistryCreate;
    pub const mlirDialectRegistryDestroy = mlir_c.mlirDialectRegistryDestroy;
    pub const dialectRegistryDestroy = mlir_c.mlirDialectRegistryDestroy;
    pub const mlirContextAppendDialectRegistry = mlir_c.mlirContextAppendDialectRegistry;
    pub const contextAppendDialectRegistry = mlir_c.mlirContextAppendDialectRegistry;
    pub const mlirContextLoadAllAvailableDialects = mlir_c.mlirContextLoadAllAvailableDialects;
    pub const contextLoadAllAvailableDialects = mlir_c.mlirContextLoadAllAvailableDialects;

    // Dialect handle functions
    pub const mlirDialectHandleInsertDialect = mlir_c.mlirDialectHandleInsertDialect;
    pub const mlirGetDialectHandle__func__ = mlir_c.mlirGetDialectHandle__func__;
    pub const mlirGetDialectHandle__arith__ = mlir_c.mlirGetDialectHandle__arith__;
    pub const mlirGetDialectHandle__scf__ = mlir_c.mlirGetDialectHandle__scf__;

    // StableHLO dialect handle (defined in pass_anchors.cpp via MLIR_DEFINE_CAPI_DIALECT_REGISTRATION)
    pub extern fn mlirGetDialectHandle__stablehlo__() mlir_c.MlirDialectHandle;

    pub const mlirLocationUnknownGet = mlir_c.mlirLocationUnknownGet;
    pub const locationUnknownGet = mlir_c.mlirLocationUnknownGet;
    pub const mlirLocationFileLineColGet = mlir_c.mlirLocationFileLineColGet;
    pub const locationFileLineColGet = mlir_c.mlirLocationFileLineColGet;

    pub const mlirModuleCreateEmpty = mlir_c.mlirModuleCreateEmpty;
    pub const moduleCreateEmpty = mlir_c.mlirModuleCreateEmpty;
    pub const mlirModuleDestroy = mlir_c.mlirModuleDestroy;
    pub const moduleDestroy = mlir_c.mlirModuleDestroy;
    pub const mlirModuleGetOperation = mlir_c.mlirModuleGetOperation;
    pub const moduleGetOperation = mlir_c.mlirModuleGetOperation;
    pub const mlirModuleGetBody = mlir_c.mlirModuleGetBody;
    pub const moduleGetBody = mlir_c.mlirModuleGetBody;
    pub const mlirModuleCreateParse = mlir_c.mlirModuleCreateParse;
    pub const moduleParseString = mlir_c.mlirModuleCreateParse;

    pub const mlirOperationCreate = mlir_c.mlirOperationCreate;
    pub const mlirOperationDestroy = mlir_c.mlirOperationDestroy;
    pub const operationDestroy = mlir_c.mlirOperationDestroy;
    pub const mlirOperationDump = mlir_c.mlirOperationDump;
    pub const operationDump = mlir_c.mlirOperationDump;
    pub const mlirOperationGetResult = mlir_c.mlirOperationGetResult;
    pub const operationGetResult = mlir_c.mlirOperationGetResult;
    pub const mlirOperationGetOperand = mlir_c.mlirOperationGetOperand;
    pub const operationGetOperand = mlir_c.mlirOperationGetOperand;
    pub const mlirOperationGetName = mlir_c.mlirOperationGetName;
    pub const operationGetName = mlir_c.mlirOperationGetName;
    pub const mlirOperationGetNumRegions = mlir_c.mlirOperationGetNumRegions;
    pub const operationGetNumRegions = mlir_c.mlirOperationGetNumRegions;
    pub const mlirOperationGetRegion = mlir_c.mlirOperationGetRegion;
    pub const operationGetRegion = mlir_c.mlirOperationGetRegion;
    pub const mlirOperationVerify = mlir_c.mlirOperationVerify;
    pub const operationVerify = mlir_c.mlirOperationVerify;
    pub const mlirOperationSetOperand = mlir_c.mlirOperationSetOperand;
    pub const operationSetOperand = mlir_c.mlirOperationSetOperand;
    pub const mlirOperationGetAttributeByName = mlir_c.mlirOperationGetAttributeByName;
    pub const operationGetAttributeByName = mlir_c.mlirOperationGetAttributeByName;
    pub const mlirOperationRemoveAttributeByName = mlir_c.mlirOperationRemoveAttributeByName;
    pub const operationRemoveAttributeByName = mlir_c.mlirOperationRemoveAttributeByName;
    pub const mlirOperationSetAttributeByName = mlir_c.mlirOperationSetAttributeByName;
    pub const operationSetAttributeByName = mlir_c.mlirOperationSetAttributeByName;
    pub const mlirOperationGetNumOperands = mlir_c.mlirOperationGetNumOperands;
    pub const operationGetNumOperands = mlir_c.mlirOperationGetNumOperands;
    pub const mlirOperationGetNumResults = mlir_c.mlirOperationGetNumResults;
    pub const operationGetNumResults = mlir_c.mlirOperationGetNumResults;
    pub const mlirOperationGetNextInBlock = mlir_c.mlirOperationGetNextInBlock;
    pub const operationGetNextInBlock = mlir_c.mlirOperationGetNextInBlock;
    pub const mlirOperationPrint = mlir_c.mlirOperationPrint;
    pub const operationPrint = mlir_c.mlirOperationPrint;
    pub const mlirOperationClone = mlir_c.mlirOperationClone;
    pub const operationClone = mlir_c.mlirOperationClone;

    pub const mlirOperationStateGet = mlir_c.mlirOperationStateGet;
    pub const operationStateGet = mlir_c.mlirOperationStateGet;
    pub const mlirOperationStateAddOperands = mlir_c.mlirOperationStateAddOperands;
    pub const operationStateAddOperands = mlir_c.mlirOperationStateAddOperands;
    pub const mlirOperationStateAddResults = mlir_c.mlirOperationStateAddResults;
    pub const operationStateAddResults = mlir_c.mlirOperationStateAddResults;
    pub const mlirOperationStateAddAttributes = mlir_c.mlirOperationStateAddAttributes;
    pub const operationStateAddAttributes = mlir_c.mlirOperationStateAddAttributes;
    pub const mlirOperationStateAddOwnedRegions = mlir_c.mlirOperationStateAddOwnedRegions;
    pub const operationStateAddOwnedRegions = mlir_c.mlirOperationStateAddOwnedRegions;

    // Packed arguments struct for safe ABI passing
    pub const PcpOpArgs = extern struct {
        nResults: isize,
        results: ?[*]mlir_c.MlirType,
        nOperands: isize,
        operands: ?[*]mlir_c.MlirValue,
        nAttributes: isize,
        attributes: ?[*]const mlir_c.MlirNamedAttribute,
        nRegions: isize,
        regions: ?[*]mlir_c.MlirRegion,
    };

    // C++ helper for packed operation creation
    pub extern fn pcpCreateOperation(
        name: *const MlirStringRef,
        location: *const MlirLocation,
        args: *const PcpOpArgs
    ) mlir_c.MlirOperation;

    pub const mlirBlockCreate = mlir_c.mlirBlockCreate;
    pub const blockCreate = mlir_c.mlirBlockCreate;
    pub const mlirBlockAddArgument = mlir_c.mlirBlockAddArgument;
    pub const blockAddArgument = mlir_c.mlirBlockAddArgument;
    pub const mlirBlockAppendOwnedOperation = mlir_c.mlirBlockAppendOwnedOperation;
    pub const blockAppendOwnedOperation = mlir_c.mlirBlockAppendOwnedOperation;
    pub const mlirBlockGetFirstOperation = mlir_c.mlirBlockGetFirstOperation;
    pub const blockGetFirstOperation = mlir_c.mlirBlockGetFirstOperation;
    pub const mlirBlockIsNull = mlir_c.mlirBlockIsNull;
    pub const blockIsNull = mlir_c.mlirBlockIsNull;
    pub const mlirBlockGetArgument = mlir_c.mlirBlockGetArgument;
    pub const blockGetArgument = mlir_c.mlirBlockGetArgument;
    pub const mlirBlockGetNumArguments = mlir_c.mlirBlockGetNumArguments;
    pub const blockGetNumArguments = mlir_c.mlirBlockGetNumArguments;

    pub const mlirRegionCreate = mlir_c.mlirRegionCreate;
    pub const regionCreate = mlir_c.mlirRegionCreate;
    pub const mlirRegionAppendOwnedBlock = mlir_c.mlirRegionAppendOwnedBlock;
    pub const regionAppendOwnedBlock = mlir_c.mlirRegionAppendOwnedBlock;
    pub const mlirRegionGetFirstBlock = mlir_c.mlirRegionGetFirstBlock;
    pub const regionGetFirstBlock = mlir_c.mlirRegionGetFirstBlock;

    pub const mlirIdentifierGet = mlir_c.mlirIdentifierGet;
    pub const identifierGet = mlir_c.mlirIdentifierGet;
    pub const mlirIdentifierStr = mlir_c.mlirIdentifierStr;
    pub const identifierStr = mlir_c.mlirIdentifierStr;

    pub const mlirAttributeParseGet = mlir_c.mlirAttributeParseGet;
    pub const mlirTypeAttrGetValue = mlir_c.mlirTypeAttrGetValue;
    pub const mlirSymbolRefAttrGet = mlir_c.mlirSymbolRefAttrGet;
    pub const mlirStringAttrGet = mlir_c.mlirStringAttrGet;
    pub const stringAttrGet = mlir_c.mlirStringAttrGet;
    pub const mlirAttributeIsAString = mlir_c.mlirAttributeIsAString;
    pub const attributeIsAString = mlir_c.mlirAttributeIsAString;
    pub const mlirStringAttrGetValue = mlir_c.mlirStringAttrGetValue;
    pub const stringAttributeGetValue = mlir_c.mlirStringAttrGetValue;
    pub const mlirAttributeIsADenseElements = mlir_c.mlirAttributeIsADenseElements;
    pub const mlirAttributeIsADenseI64Array = mlir_c.mlirAttributeIsADenseI64Array;
    pub const attributeIsADenseI64Array = mlir_c.mlirAttributeIsADenseI64Array;
    pub const mlirIntegerAttrGet = mlir_c.mlirIntegerAttrGet;
    pub const mlirBoolAttrGet = mlir_c.mlirBoolAttrGet;
    pub const mlirTypeAttrGet = mlir_c.mlirTypeAttrGet;
    pub const typeAttr = mlir_c.mlirTypeAttrGet;
    pub const mlirFloatAttrDoubleGet = mlir_c.mlirFloatAttrDoubleGet;
    pub const mlirDenseElementsAttrGet = mlir_c.mlirDenseElementsAttrGet;
    pub const mlirDenseElementsAttrRawBufferGet = mlir_c.mlirDenseElementsAttrRawBufferGet;
    pub const mlirDenseElementsAttrGetRawData = mlir_c.mlirDenseElementsAttrGetRawData;
    pub const mlirDenseElementsAttrGetNumElements = mlir_c.mlirDenseElementsAttrGetNumElements;
    pub const mlirDenseElementsAttrSplatGet = mlir_c.mlirDenseElementsAttrSplatGet;
    pub const mlirDenseElementsAttrFloatSplatGet = mlir_c.mlirDenseElementsAttrFloatSplatGet;
    pub const mlirDenseI64ArrayGet = mlir_c.mlirDenseI64ArrayGet;
    pub const mlirDenseI64ArrayGetElement = mlir_c.mlirDenseI64ArrayGetElement;
    pub const denseI64ArrayGetElement = mlir_c.mlirDenseI64ArrayGetElement;
    pub const mlirUnitAttrGet = mlir_c.mlirUnitAttrGet;
    pub const mlirDictionaryAttrGet = mlir_c.mlirDictionaryAttrGet;
    pub const dictionaryAttrGet = mlir_c.mlirDictionaryAttrGet;

    pub const mlirTypeParse = mlir_c.mlirTypeParse;
    pub const typeParse = mlir_c.mlirTypeParse;
    pub const mlirTypeGetContext = mlir_c.mlirTypeGetContext;
    pub const typeGetContext = mlir_c.mlirTypeGetContext;
    pub const mlirTypeDump = mlir_c.mlirTypeDump;
    pub const typeDump = mlir_c.mlirTypeDump;
    pub const mlirIntegerTypeGet = mlir_c.mlirIntegerTypeGet;
    pub const mlirF32TypeGet = mlir_c.mlirF32TypeGet;
    pub const floatTypeGetF32 = mlir_c.mlirF32TypeGet;
    pub const mlirF64TypeGet = mlir_c.mlirF64TypeGet;
    pub const floatTypeGetF64 = mlir_c.mlirF64TypeGet;
    pub const mlirIntegerTypeGetWidth = mlir_c.mlirIntegerTypeGetWidth;
    pub const mlirRankedTensorTypeGet = mlir_c.mlirRankedTensorTypeGet;
    pub const rankedTensorTypeGet = mlir_c.mlirRankedTensorTypeGet;
    pub const mlirRankedTensorTypeGetRank = mlir_c.mlirRankedTensorTypeGetRank;
    pub const mlirRankedTensorTypeGetDimSize = mlir_c.mlirRankedTensorTypeGetDimSize;
    pub const mlirShapedTypeGetElementType = mlir_c.mlirShapedTypeGetElementType;
    pub const shapedTypeGetElementType = mlir_c.mlirShapedTypeGetElementType;
    pub const mlirShapedTypeGetRank = mlir_c.mlirShapedTypeGetRank;
    pub const shapedTypeGetRank = mlir_c.mlirShapedTypeGetRank;
    pub const mlirShapedTypeGetDimSize = mlir_c.mlirShapedTypeGetDimSize;
    pub const shapedTypeGetDimSize = mlir_c.mlirShapedTypeGetDimSize;
    pub const mlirFunctionTypeGet = mlir_c.mlirFunctionTypeGet;
    pub const functionTypeGet = mlir_c.mlirFunctionTypeGet;
    pub const mlirFunctionTypeGetNumInputs = mlir_c.mlirFunctionTypeGetNumInputs;
    pub const functionTypeGetNumInputs = mlir_c.mlirFunctionTypeGetNumInputs;
    pub const mlirFunctionTypeGetInput = mlir_c.mlirFunctionTypeGetInput;
    pub const functionTypeGetInput = mlir_c.mlirFunctionTypeGetInput;
    pub const mlirFunctionTypeGetNumResults = mlir_c.mlirFunctionTypeGetNumResults;
    pub const functionTypeGetNumResults = mlir_c.mlirFunctionTypeGetNumResults;
    pub const mlirFunctionTypeGetResult = mlir_c.mlirFunctionTypeGetResult;
    pub const functionTypeGetResult = mlir_c.mlirFunctionTypeGetResult;
    pub const mlirTypeIsARankedTensor = mlir_c.mlirTypeIsARankedTensor;
    pub const typeIsARankedTensor = mlir_c.mlirTypeIsARankedTensor;
    pub const mlirTypeIsAFunction = mlir_c.mlirTypeIsAFunction;
    pub const typeIsAFunction = mlir_c.mlirTypeIsAFunction;
    pub const mlirTypeIsAInteger = mlir_c.mlirTypeIsAInteger;
    pub const typeIsAInteger = mlir_c.mlirTypeIsAInteger;
    pub const mlirTypeIsAIndex = mlir_c.mlirTypeIsAIndex;
    pub const typeIsAIndex = mlir_c.mlirTypeIsAIndex;

    pub const mlirValueGetType = mlir_c.mlirValueGetType;
    pub const valueGetType = mlir_c.mlirValueGetType;
    pub const mlirValueDump = mlir_c.mlirValueDump;
    pub const valueDump = mlir_c.mlirValueDump;
    pub const mlirValueIsAOpResult = mlir_c.mlirValueIsAOpResult;
    pub const mlirOpResultGetOwner = mlir_c.mlirOpResultGetOwner;

    // Helper functions
    pub fn stringRefFromString(str: []const u8) MlirStringRef {
        return MlirStringRef{
            .data = str.ptr,
            .length = str.len,
        };
    }

    // Alias for compatibility
    pub const stringRefCreateFromCString = mlir_c.mlirStringRefCreateFromCString;

    pub fn fromStringRef(ref: MlirStringRef) []const u8 {
        return ref.data[0..ref.length];
    }
};
