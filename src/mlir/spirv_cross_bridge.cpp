#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "spirv_cross_c.h"
#include <cstring>
#include <vector>
#include <cstdlib>

extern "C" {

typedef void (*MlirStringCallback)(MlirStringRef, void*);

// C API wrapper for SPIR-V to MSL translation using SPIRV-Cross
MlirLogicalResult mlirTranslateSPIRVToMSL(
    const void* spirv_data,
    size_t spirv_size,
    MlirStringCallback callback,
    void* userData) {
    
    // Create SPIRV-Cross context
    spvc_context context = nullptr;
    spvc_result result = spvc_context_create(&context);
    if (result != SPVC_SUCCESS) {
        return mlirLogicalResultFailure();
    }
    
    // Parse SPIR-V binary
    spvc_parsed_ir ir = nullptr;
    result = spvc_context_parse_spirv(context, 
                                     reinterpret_cast<const SpvId*>(spirv_data), 
                                     spirv_size / sizeof(SpvId), 
                                     &ir);
    if (result != SPVC_SUCCESS) {
        spvc_context_destroy(context);
        return mlirLogicalResultFailure();
    }
    
    // Create MSL compiler
    spvc_compiler compiler = nullptr;
    result = spvc_context_create_compiler(context, SPVC_BACKEND_MSL, ir, 
                                         SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, 
                                         &compiler);
    if (result != SPVC_SUCCESS) {
        spvc_context_destroy(context);
        return mlirLogicalResultFailure();
    }
    
    // Set MSL-specific options
    spvc_compiler_options options = nullptr;
    result = spvc_compiler_create_compiler_options(compiler, &options);
    if (result == SPVC_SUCCESS) {
        // Set MSL version (Metal 2.0)
        spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, 
                                      0x00020000); // MSL 2.0
        
        // Enable MSL features
        spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_ENABLE_POINT_SIZE_BUILTIN, 
                                      SPVC_FALSE);
        spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_DISABLE_RASTERIZATION, 
                                      SPVC_FALSE);
        
        spvc_compiler_install_compiler_options(compiler, options);
    }
    
    // Compile to MSL
    const char* msl_source = nullptr;
    result = spvc_compiler_compile(compiler, &msl_source);
    if (result != SPVC_SUCCESS) {
        spvc_context_destroy(context);
        return mlirLogicalResultFailure();
    }
    
    // Create result string
    MlirStringRef msl_result = {
        .data = msl_source,
        .length = strlen(msl_source)
    };
    
    // Call callback with result
    callback(msl_result, userData);
    
    // Cleanup
    spvc_context_destroy(context);
    
    return mlirLogicalResultSuccess();
}

// Extract GPU kernel metadata from MLIR module
typedef struct {
    const char* name;
    uint32_t grid_size[3];
    uint32_t block_size[3];
} GPUKernelMetadata;

typedef struct {
    GPUKernelMetadata* kernels;
    size_t count;
    size_t capacity;
} GPUKernelList;

// Walker callback to extract GPU kernel information
MlirWalkResult extractGPUKernelCallback(MlirOperation op, void* userData) {
    GPUKernelList* kernels = static_cast<GPUKernelList*>(userData);
    
    // Get operation name
    MlirIdentifier name_id = mlirOperationGetName(op);
    MlirStringRef name_ref = mlirIdentifierStr(name_id);
    
    // Convert to C string for comparison
    std::string op_name(name_ref.data, name_ref.length);
    
    // Look for gpu.launch_func operations
    if (op_name == "gpu.launch_func") {
        // Resize array if needed
        if (kernels->count >= kernels->capacity) {
            kernels->capacity = kernels->capacity == 0 ? 4 : kernels->capacity * 2;
            kernels->kernels = static_cast<GPUKernelMetadata*>(
                realloc(kernels->kernels, kernels->capacity * sizeof(GPUKernelMetadata)));
        }
        
        GPUKernelMetadata* kernel = &kernels->kernels[kernels->count];
        
        // Extract kernel name from 'kernel' attribute
        MlirStringRef kernel_attr_name = {.data = "kernel", .length = 6};
        MlirAttribute kernel_attr = mlirOperationGetAttributeByName(op, kernel_attr_name);
        
        if (!mlirAttributeIsNull(kernel_attr)) {
            // For now, use a placeholder name - proper attribute extraction needs more MLIR C API
            kernel->name = strdup("gpu_kernel");
        } else {
            kernel->name = strdup("unknown_kernel");
        }
        
        // Extract grid size from 'gridSizeX', 'gridSizeY', 'gridSizeZ' operands
        // For now, use default values - proper operand extraction needs more MLIR C API
        kernel->grid_size[0] = 1;
        kernel->grid_size[1] = 1;
        kernel->grid_size[2] = 1;
        
        // Extract block size from 'blockSizeX', 'blockSizeY', 'blockSizeZ' operands
        kernel->block_size[0] = 256;
        kernel->block_size[1] = 1;
        kernel->block_size[2] = 1;
        
        kernels->count++;
    }
    
    return (MlirWalkResult)0; // MLIR_WALK_RESULT_ADVANCE
}

// C API wrapper for extracting GPU kernel metadata
size_t mlirExtractGPUKernelMetadata(
    MlirModule module,
    GPUKernelMetadata** out_kernels) {
    
    GPUKernelList kernels = {nullptr, 0, 0};
    
    // Walk the module to find gpu.launch_func operations
    MlirOperation module_op = mlirModuleGetOperation(module);
    mlirOperationWalk(module_op, extractGPUKernelCallback, &kernels, MlirWalkPreOrder);
    
    *out_kernels = kernels.kernels;
    return kernels.count;
}

// Cleanup function for kernel metadata
void mlirFreeGPUKernelMetadata(GPUKernelMetadata* kernels, size_t count) {
    for (size_t i = 0; i < count; i++) {
        free(const_cast<char*>(kernels[i].name));
    }
    free(kernels);
}

// NEW FUNCTION to extract kernel names using SPIRV-Cross
size_t mlirExtractKernelNamesFromSPIRV(
    const uint8_t* spirv_data,
    size_t spirv_size,
    const char*** out_names) {
    
    // Create SPIRV-Cross context
    spvc_context context = nullptr;
    spvc_result result = spvc_context_create(&context);
    if (result != SPVC_SUCCESS) {
        *out_names = nullptr;
        return 0;
    }
    
    // Parse SPIR-V binary
    spvc_parsed_ir ir = nullptr;
    result = spvc_context_parse_spirv(context, 
                                     reinterpret_cast<const SpvId*>(spirv_data), 
                                     spirv_size / sizeof(SpvId), 
                                     &ir);
    if (result != SPVC_SUCCESS) {
        spvc_context_destroy(context);
        *out_names = nullptr;
        return 0;
    }
    
    // Create compiler to access entry points
    spvc_compiler compiler = nullptr;
    result = spvc_context_create_compiler(context, SPVC_BACKEND_MSL, ir, 
                                         SPVC_CAPTURE_MODE_COPY, 
                                         &compiler);
    if (result != SPVC_SUCCESS) {
        spvc_context_destroy(context);
        *out_names = nullptr;
        return 0;
    }
    
    // Get entry points
    const spvc_entry_point* entry_points = nullptr;
    size_t num_entry_points = 0;
    result = spvc_compiler_get_entry_points(compiler, &entry_points, &num_entry_points);
    if (result != SPVC_SUCCESS || num_entry_points == 0) {
        spvc_context_destroy(context);
        *out_names = nullptr;
        return 0;
    }
    
    // Allocate memory for the C-style string array
    *out_names = static_cast<const char**>(malloc(num_entry_points * sizeof(char*)));
    if (!*out_names) {
        spvc_context_destroy(context);
        return 0;
    }
    
    // Copy entry point names
    for (size_t i = 0; i < num_entry_points; ++i) {
        (*out_names)[i] = strdup(entry_points[i].name);
        if (!(*out_names)[i]) {
            // Cleanup on failure
            for (size_t j = 0; j < i; ++j) {
                free(const_cast<char*>((*out_names)[j]));
            }
            free(*out_names);
            *out_names = nullptr;
            spvc_context_destroy(context);
            return 0;
        }
    }
    
    spvc_context_destroy(context);
    return num_entry_points;
}

// NEW FUNCTION to free the memory allocated by the above function
void mlirFreeKernelNames(const char** names, size_t count) {
    if (!names) return;
    for (size_t i = 0; i < count; ++i) {
        free(const_cast<char*>(names[i]));
    }
    free(names);
}

}