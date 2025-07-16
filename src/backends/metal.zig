const std = @import("std");
const tensor = @import("../tensor.zig");
const mlir = @import("../mlir.zig");
const ops = @import("../ops.zig");
const mlir_ctx = @import("../mlir_ctx.zig");
const builtin = @import("builtin");
const c = @import("../mlir/c.zig").c;

const Allocator = std.mem.Allocator;
const DataType = f32;
const Tensor = tensor.Tensor(DataType);
const DType = tensor.DType;
const Shape = tensor.Shape;
const MLIRBuilder = ops.MLIRBuilder;
const MLIRContext = mlir_ctx.MLIRContext;

pub const MetalError = error{
    MetalInitFailed,
    DeviceNotFound,
    CommandQueueCreationFailed,
    LibraryCreationFailed,
    FunctionCreationFailed,
    PipelineCreationFailed,
    BufferCreationFailed,
    CommandBufferCreationFailed,
    EncodingFailed,
    UnsupportedPlatform,
    ThreadgroupSizeComputationFailed,
    MLIRCompilationFailed,
    MLIRExecutionFailed,
};

// C interface to Metal Framework
const MTLDevice = opaque {};
const MTLCommandQueue = opaque {};
const MTLLibrary = opaque {};
const MTLFunction = opaque {};
const MTLComputePipelineState = opaque {};
const MTLBuffer = opaque {};
const MTLCommandBuffer = opaque {};
const MTLComputeCommandEncoder = opaque {};
const NSError = opaque {};
const NSString = opaque {};

// Metal C API functions - Complete implementation
// Use our bridge function to create device safely
extern fn MTL_CreateSystemDefaultDevice() ?*MTLDevice;
extern fn MTLDevice_newCommandQueue(device: *MTLDevice) ?*MTLCommandQueue;
extern fn MTLDevice_newLibraryWithSource(device: *MTLDevice, source: *NSString, options: ?*anyopaque, err: *?*NSError) ?*MTLLibrary;
extern fn MTLDevice_newBufferWithBytes(device: *MTLDevice, bytes: *const anyopaque, length: usize, options: u64) ?*MTLBuffer;
extern fn MTLDevice_newBufferWithLength(device: *MTLDevice, length: usize, options: u64) ?*MTLBuffer;
extern fn MTLBuffer_contents(buffer: *MTLBuffer) *anyopaque;
extern fn MTLBuffer_length(buffer: *MTLBuffer) usize;
extern fn MTLLibrary_newFunctionWithName(library: *MTLLibrary, name: *NSString) ?*MTLFunction;
extern fn MTLDevice_newComputePipelineStateWithFunction(device: *MTLDevice, function: *MTLFunction, err: *?*NSError) ?*MTLComputePipelineState;
extern fn MTLCommandQueue_commandBuffer(queue: *MTLCommandQueue) ?*MTLCommandBuffer;
extern fn MTLCommandBuffer_computeCommandEncoder(buffer: *MTLCommandBuffer) ?*MTLComputeCommandEncoder;
extern fn MTLComputeCommandEncoder_setComputePipelineState(encoder: *MTLComputeCommandEncoder, state: *MTLComputePipelineState) void;
extern fn MTLComputeCommandEncoder_setBuffer(encoder: *MTLComputeCommandEncoder, buffer: *MTLBuffer, offset: usize, index: usize) void;
extern fn MTLComputePipelineState_maxTotalThreadsPerThreadgroup(state: *MTLComputePipelineState) usize;
extern fn MTLComputePipelineState_threadExecutionWidth(state: *MTLComputePipelineState) usize;

// MTLSize struct that matches the Objective-C struct
const MTLSize = extern struct {
    width: c_ulong,
    height: c_ulong,
    depth: c_ulong,
};

extern fn MTLComputeCommandEncoder_dispatchThreads(encoder: *MTLComputeCommandEncoder, threads: MTLSize, threadsPerThreadgroup: MTLSize) void;
extern fn MTLComputeCommandEncoder_endEncoding(encoder: *MTLComputeCommandEncoder) void;
extern fn MTLCommandBuffer_commit(buffer: *MTLCommandBuffer) void;
extern fn MTLCommandBuffer_waitUntilCompleted(buffer: *MTLCommandBuffer) void;
extern fn NSString_alloc() *NSString;
extern fn NSString_initWithUTF8String(self: *NSString, cString: [*:0]const u8) *NSString;
extern fn NSString_release(self: *NSString) void;

// Helper function to dispatch threads using our array format to MTLSize
fn dispatchThreads(encoder: *MTLComputeCommandEncoder, threads: [3]usize, threadsPerThreadgroup: [3]usize) void {
    const threads_mtl = MTLSize{
        .width = threads[0],
        .height = threads[1],
        .depth = threads[2],
    };

    const threads_per_group_mtl = MTLSize{
        .width = threadsPerThreadgroup[0],
        .height = threadsPerThreadgroup[1],
        .depth = threadsPerThreadgroup[2],
    };

    MTLComputeCommandEncoder_dispatchThreads(encoder, threads_mtl, threads_per_group_mtl);
}

// Metal resource management constants
const MTLResourceOptions = struct {
    pub const StorageModeShared: u64 = 0 << 4;
    pub const StorageModeManaged: u64 = 1 << 4;
    pub const StorageModePrivate: u64 = 2 << 4;
    pub const StorageModeMemoryless: u64 = 3 << 4;
};


// MLIR-Metal execution engine using proper MLIR dialect lowering
pub const MLIRMetalExecutionEngine = struct {
    device: ?*MTLDevice = null,
    command_queue: ?*MTLCommandQueue = null,
    library: ?*MTLLibrary = null,
    allocator: Allocator,

    // MLIR execution capabilities
    mlir_context: ?*MLIRContext = null,
    
    // Generated Metal library from MLIR compilation
    compiled_library: ?*MTLLibrary = null,
    compiled_functions: std.HashMap([]const u8, *MTLFunction, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    compiled_pipelines: std.HashMap([]const u8, *MTLComputePipelineState, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

    pub fn init(allocator: Allocator) !MLIRMetalExecutionEngine {
        // Check if running on Apple platform
        if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
            return MetalError.UnsupportedPlatform;
        }

        // Get the default Metal device
        const device = MTL_CreateSystemDefaultDevice() orelse {
            return MetalError.DeviceNotFound;
        };

        // Create a command queue
        const command_queue = MTLDevice_newCommandQueue(device) orelse {
            return MetalError.CommandQueueCreationFailed;
        };

        // Initialize MLIR context with StableHLO → GPU → SPIR-V pipeline
        const mlir_context = try allocator.create(MLIRContext);
        mlir_context.* = try MLIRContext.init(allocator);

        // Create execution engine
        const engine = MLIRMetalExecutionEngine{
            .device = device,
            .command_queue = command_queue,
            .allocator = allocator,
            .mlir_context = mlir_context,
            .compiled_functions = std.HashMap([]const u8, *MTLFunction, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .compiled_pipelines = std.HashMap([]const u8, *MTLComputePipelineState, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
        };

        return engine;
    }

    pub fn deinit(self: *MLIRMetalExecutionEngine) void {
        // Clean up MLIR resources
        if (self.mlir_context) |ctx| {
            ctx.deinit();
            self.allocator.destroy(ctx);
        }
        
        // Clean up compiled resources
        self.compiled_functions.deinit();
        self.compiled_pipelines.deinit();
        self.device = null;
        self.command_queue = null;
        self.library = null;
        self.compiled_library = null;
        self.mlir_context = null;
    }
    
    /// Execute an MLIR module on Metal hardware
    /// Execute a specific function operation with tensor inputs and outputs
    pub fn executeFunction(self: *MLIRMetalExecutionEngine, func_op: mlir.Operation, inputs: []const Tensor, outputs: []Tensor) ![][]f32 {
        // 1. Create a new module to hold the function for compilation
        const temp_module = try mlir.Module.createEmpty(mlir.Context{ .handle = self.mlir_context.?.context });
        defer temp_module.deinit();
        
        // 2. CLONE the gradient function into the new module using the new utility
        const cloned_func_op = try self.cloneFunctionToModule(func_op, temp_module);
        _ = cloned_func_op; // cloned_func_op is now inside temp_module
        
        // 3. Extract REAL data from the input Tensors using extractDataFromConstantOp
        var input_arrays = std.ArrayList([]const f32).init(self.allocator);
        defer {
            for (input_arrays.items) |arr| {
                self.allocator.free(arr);
            }
            input_arrays.deinit();
        }
        
        // Extract data from each input tensor
        for (inputs) |input_tensor| {
            const data = try self.extractDataFromConstantOp(input_tensor.value);
            try input_arrays.append(data);
        }
        
        // 4. Allocate and prepare output buffers based on actual output tensor shapes
        var output_arrays = std.ArrayList([]f32).init(self.allocator);
        defer output_arrays.deinit();
        
        for (outputs) |output_tensor| {
            // For now, assume scalar outputs (size 1)
            // TODO: Extract actual shape from tensor type
            _ = output_tensor;
            try output_arrays.append(try self.allocator.alloc(f32, 1));
        }
        
        // 5. Execute the module containing ONLY the cloned function
        try self.executeMLIRModule(temp_module, input_arrays.items, output_arrays.items);
        
        // 6. Copy results back to output tensors (if provided) and prepare return data
        var result_arrays = std.ArrayList([]f32).init(self.allocator);
        defer result_arrays.deinit();
        
        for (output_arrays.items, 0..) |output_data, i| {
            // Make a copy for the return value
            const result_copy = try self.allocator.dupe(f32, output_data);
            try result_arrays.append(result_copy);
            
            // Copy to output tensor if provided
            if (i < outputs.len) {
                // TODO: Copy data back to outputs[i] when we have proper tensor data access
                _ = outputs[i];
            }
        }
        
        // 7. Print verification information
        std.debug.print("Execution results: ", .{});
        for (result_arrays.items, 0..) |result, i| {
            std.debug.print("output[{}] = {d} ", .{i, result[0]});
        }
        std.debug.print("\n", .{});
        
        // Free the temporary output buffers
        for (output_arrays.items) |arr| {
            self.allocator.free(arr);
        }
        
        return result_arrays.toOwnedSlice();
    }
    
    /// Clone a function operation to a destination module using ValueMap approach
    fn cloneFunctionToModule(
        self: *MLIRMetalExecutionEngine,
        source_fn_op: mlir.Operation,
        dest_module: mlir.Module,
    ) !mlir.Operation {
        // A map from old MlirValue handles to new MlirValue handles
        var value_map = std.AutoHashMap(mlir.Value, mlir.Value).init(self.allocator);
        defer value_map.deinit();
        
        // --- 1. Get signature from the source function ---
        _ = c.operationGetAttributeByName(source_fn_op.handle, "function_type");
        // Note: This function doesn't return null - it returns a valid attribute or the operation fails
        
        // --- 2. Create the new function shell in the destination module ---
        const location = mlir.Location.unknown(mlir.Context{ .handle = self.mlir_context.?.context });
        const op_name_str = "func.func";
        
        var new_op_state = c.operationStateGet(op_name_str, location.handle);
        
        // Copy attributes from source function
        var attributes = std.ArrayList(c.MlirNamedAttribute).init(self.allocator);
        defer attributes.deinit();
        
        const num_attrs = source_fn_op.getNumAttributes();
        for (0..num_attrs) |i| {
            const attr = source_fn_op.getAttribute(i);
            try attributes.append(attr);
        }
        
        if (attributes.items.len > 0) {
            c.mlirOperationStateAddAttributes(&new_op_state, @intCast(attributes.items.len), @ptrCast(attributes.items.ptr));
        }
        
        // Create a region for the function body
        const region = c.regionCreate();
        c.operationStateAddOwnedRegions(&new_op_state, 1, @ptrCast(@constCast(&region)));
        
        // Create the new function operation
        const dest_fn_op = mlir.Operation{ .handle = c.operationCreate(&new_op_state) };
        
        // Add the function to the destination module
        const dest_module_body = c.moduleGetBody(dest_module.handle);
        c.blockAppendOwnedOperation(dest_module_body, dest_fn_op.handle);
        
        // Create an entry block for the new function
        const dest_region = dest_fn_op.getRegion(0);
        // Create empty arrays for block arguments
        const args: [*]*c.MlirType = undefined;
        const locs: [*]*c.MlirLocation = undefined;
        const dest_block = mlir.Block{ .handle = c.blockCreate(0, args, locs) };
        dest_region.appendOwnedBlock(dest_block);
        
        // --- 3. Map block arguments ---
        const source_block = source_fn_op.getRegion(0).getBlock(0);
        const num_args = source_block.getNumArguments();
        
        for (0..num_args) |i| {
            const old_arg = source_block.getArgument(i);
            const new_arg = dest_block.addArgument(old_arg.getType(), location);
            try value_map.put(old_arg, new_arg);
        }
        
        // --- 4. Iterate and rebuild operations ---
        var maybe_op = source_block.getFirstOp();
        while (maybe_op) |old_op| {
            // Prepare state for the new operation
            const old_op_name = old_op.getName();
            var op_state = c.operationStateGet(old_op_name, old_op.getLocation().handle);
            
            // Remap operands
            var new_operands = std.ArrayList(*c.MlirValue).init(self.allocator);
            defer new_operands.deinit();
            
            for (0..old_op.getNumOperands()) |i| {
                const old_operand = old_op.getOperand(i);
                const new_operand = value_map.get(old_operand) orelse {
                    return error.UnmappedOperand;
                };
                try new_operands.append(new_operand.handle);
            }
            
            if (new_operands.items.len > 0) {
                c.mlirOperationStateAddOperands(&op_state, @intCast(new_operands.items.len), new_operands.items.ptr);
            }
            
            // Copy result types
            var result_types = std.ArrayList(*c.MlirType).init(self.allocator);
            defer result_types.deinit();
            
            for (0..old_op.getNumResults()) |i| {
                const old_result = old_op.getResult(i);
                try result_types.append(old_result.getType().handle);
            }
            
            if (result_types.items.len > 0) {
                c.mlirOperationStateAddResults(&op_state, @intCast(result_types.items.len), result_types.items.ptr);
            }
            
            // Copy attributes
            var op_attributes = std.ArrayList(c.MlirNamedAttribute).init(self.allocator);
            defer op_attributes.deinit();
            
            const num_op_attrs = old_op.getNumAttributes();
            for (0..num_op_attrs) |i| {
                const attr = old_op.getAttribute(i);
                try op_attributes.append(attr);
            }
            
            if (op_attributes.items.len > 0) {
                c.mlirOperationStateAddAttributes(&op_state, @intCast(op_attributes.items.len), @ptrCast(op_attributes.items.ptr));
            }
            
            // Create the new operation
            const new_op = mlir.Operation{ .handle = c.operationCreate(&op_state) };
            dest_block.appendOwnedOperation(new_op);
            
            // Update the value map with the new results
            for (0..old_op.getNumResults()) |i| {
                try value_map.put(old_op.getResult(i), new_op.getResult(i));
            }
            
            maybe_op = old_op.getNext();
        }
        
        return dest_fn_op;
    }
    
    /// Extract data from a stablehlo.constant operation
    fn extractDataFromConstantOp(self: *MLIRMetalExecutionEngine, value: mlir.Value) ![]const f32 {
        // Find the defining operation for this value using the same pattern as autodiff
        if (!c.mlirValueIsAOpResult(value.handle)) {
            return error.NoDefiningOp;
        }
        
        // Get the operation that produced this result
        const def_op_handle = c.mlirOpResultGetOwner(value.handle);
        const defining_op = def_op_handle;
        
        // Get the operation name to verify it's a constant
        const op_name = c.mlirOperationGetName(defining_op);
        const name_str = c.mlirIdentifierStr(op_name);
        const name_slice = c.fromStringRef(name_str);
        
        if (!std.mem.eql(u8, name_slice, "stablehlo.constant")) {
            return error.NotAConstantOp;
        }
        
        // Get the "value" attribute from the constant operation
        const value_attr = c.operationGetAttributeByName(defining_op, "value");
        if (@intFromPtr(value_attr) == 0) {
            return error.NoValueAttribute;
        }
        
        // Check if it's a dense elements attribute
        if (!c.mlirAttributeIsADenseElements(value_attr)) {
            return error.NotDenseElementsAttribute;
        }
        
        // Extract the raw data
        const raw_data = c.mlirDenseElementsAttrGetRawData(value_attr);
        
        // Calculate number of elements from the tensor type
        const tensor_type = value.getType().as(mlir.RankedTensorType).?;
        const shape = tensor_type.getShape();
        var num_elements: usize = 1;
        for (shape) |dim| {
            num_elements *= @intCast(dim);
        }
        
        // Cast to f32 array
        const data_ptr: [*]const f32 = @ptrCast(@alignCast(raw_data));
        const data_slice = data_ptr[0..num_elements];
        
        // Make a copy for the caller
        const result = try self.allocator.dupe(f32, data_slice);
        return result;
    }

    pub fn executeMLIRModule(self: *MLIRMetalExecutionEngine, module: mlir.Module, inputs: [][]const f32, outputs: [][]f32) !void {
        const device = self.device orelse return MetalError.DeviceNotFound;
        _ = self.command_queue orelse return MetalError.CommandQueueCreationFailed;
        const mlir_context = self.mlir_context orelse return MetalError.MLIRCompilationFailed;
        
        // Step 1: Lower MLIR module through the StableHLO → GPU → SPIR-V pipeline
        std.debug.print("Lowering MLIR module through StableHLO → GPU → SPIR-V pipeline...\n", .{});
        try mlir_context.lowerToSPIRV(module);
        
        // Step 2: Extract GPU kernel names from the lowered module
        const kernel_names = try mlir_context.getGpuKernelNames(module);
        defer {
            for (kernel_names) |name| {
                self.allocator.free(name);
            }
            self.allocator.free(kernel_names);
        }
        
        // Step 3: Translate SPIR-V to Metal Shading Language (MSL)
        const spirv_binary = try mlir_context.translateToSPIRV(module);
        defer self.allocator.free(spirv_binary);
        
        const msl_code = try mlir_ctx.translateSpirvToMsl(self.allocator, spirv_binary);
        defer self.allocator.free(msl_code);
        
        std.debug.print("Generated MSL code:\n{s}\n", .{msl_code});
        
        // Step 4: Compile MSL code to Metal library
        const metal_library = try self.compileMSLToMetal(msl_code);
        
        // Step 5: Extract GPU kernel metadata
        const gpu_kernels = try mlir_ctx.extractGPUKernelInfo(self.allocator, module);
        defer {
            for (gpu_kernels) |*kernel| {
                kernel.deinit(self.allocator);
            }
            self.allocator.free(gpu_kernels);
        }
        
        // Step 6: Create Metal buffers for inputs and outputs
        var input_buffers = std.ArrayList(*MTLBuffer).init(self.allocator);
        defer input_buffers.deinit();
        
        var output_buffers = std.ArrayList(*MTLBuffer).init(self.allocator);
        defer output_buffers.deinit();
        
        // Create input buffers
        for (inputs) |input_data| {
            const buffer = MTLDevice_newBufferWithBytes(
                device,
                input_data.ptr,
                input_data.len * @sizeOf(f32),
                MTLResourceOptions.StorageModeShared
            ) orelse return MetalError.BufferCreationFailed;
            
            try input_buffers.append(buffer);
        }
        
        // Create output buffers
        for (outputs) |output_data| {
            const buffer = MTLDevice_newBufferWithLength(
                device,
                output_data.len * @sizeOf(f32),
                MTLResourceOptions.StorageModeShared
            ) orelse return MetalError.BufferCreationFailed;
            
            try output_buffers.append(buffer);
        }
        
        // Step 7: Execute GPU kernels on Metal
        try self.executeGPUKernelsFromInfo(gpu_kernels, metal_library, input_buffers.items, output_buffers.items);
        
        // Step 8: Copy results back to output arrays
        for (outputs, 0..) |output_data, i| {
            const buffer = output_buffers.items[i];
            const buffer_contents = MTLBuffer_contents(buffer);
            const buffer_data: [*]f32 = @ptrCast(@alignCast(buffer_contents));
            
            for (output_data, 0..) |*out_val, j| {
                out_val.* = buffer_data[j];
            }
        }
        
        std.debug.print("✓ Successfully executed MLIR module on Metal hardware\n", .{});
    }
    
    
    /// Compile MSL code to Metal library
    fn compileMSLToMetal(self: *MLIRMetalExecutionEngine, msl_code: []const u8) !*MTLLibrary {
        const device = self.device orelse return MetalError.DeviceNotFound;
        
        // Create NSString from MSL code
        const source_str = NSString_alloc();
        const source = NSString_initWithUTF8String(source_str, @ptrCast(msl_code));
        defer NSString_release(source);
        
        // Compile the Metal code
        var compile_error: ?*NSError = null;
        const library = MTLDevice_newLibraryWithSource(device, source, null, &compile_error) orelse {
            std.debug.print("Failed to compile MSL code\n", .{});
            return MetalError.LibraryCreationFailed;
        };
        
        self.compiled_library = library;
        return library;
    }
    
    
    /// Execute GPU kernels on Metal hardware using kernel info from MLIR
    fn executeGPUKernelsFromInfo(self: *MLIRMetalExecutionEngine, kernels: []mlir_ctx.GPUKernelInfo, library: *MTLLibrary, input_buffers: []*MTLBuffer, output_buffers: []*MTLBuffer) !void {
        const device = self.device orelse return MetalError.DeviceNotFound;
        const command_queue = self.command_queue orelse return MetalError.CommandQueueCreationFailed;
        
        // Create command buffer
        const command_buffer = MTLCommandQueue_commandBuffer(command_queue) orelse {
            return MetalError.CommandBufferCreationFailed;
        };
        
        const compute_encoder = MTLCommandBuffer_computeCommandEncoder(command_buffer) orelse {
            return MetalError.EncodingFailed;
        };
        
        // Execute each kernel
        for (kernels) |kernel| {
            try self.executeGPUKernelFromInfo(kernel, library, device, compute_encoder, input_buffers, output_buffers);
        }
        
        MTLComputeCommandEncoder_endEncoding(compute_encoder);
        MTLCommandBuffer_commit(command_buffer);
        MTLCommandBuffer_waitUntilCompleted(command_buffer);
    }
    
    /// Execute a single GPU kernel using MLIR-extracted kernel info
    fn executeGPUKernelFromInfo(self: *MLIRMetalExecutionEngine, kernel: mlir_ctx.GPUKernelInfo, library: *MTLLibrary, device: *MTLDevice, encoder: *MTLComputeCommandEncoder, input_buffers: []*MTLBuffer, output_buffers: []*MTLBuffer) !void {
        // Get or create compute pipeline for this kernel
        const pipeline = try self.getOrCreatePipeline(kernel.name, library, device);
        
        MTLComputeCommandEncoder_setComputePipelineState(encoder, pipeline);
        
        // Bind input buffers
        for (input_buffers, 0..) |buffer, i| {
            MTLComputeCommandEncoder_setBuffer(encoder, buffer, 0, i);
        }
        
        // Bind output buffers
        for (output_buffers, 0..) |buffer, i| {
            MTLComputeCommandEncoder_setBuffer(encoder, buffer, 0, input_buffers.len + i);
        }
        
        // Calculate total threads needed
        const total_threads = if (input_buffers.len > 0) 
            MTLBuffer_length(input_buffers[0]) / @sizeOf(f32)
        else
            1;
        
        // Dispatch threads using MLIR-extracted block size
        dispatchThreads(
            encoder,
            [3]usize{ total_threads, 1, 1 },
            kernel.block_size
        );
    }
    
    /// Get or create a compute pipeline for a kernel
    fn getOrCreatePipeline(self: *MLIRMetalExecutionEngine, kernel_name: []const u8, library: *MTLLibrary, device: *MTLDevice) !*MTLComputePipelineState {
        // Check if pipeline already exists
        if (self.compiled_pipelines.get(kernel_name)) |pipeline| {
            return pipeline;
        }
        
        // Create new pipeline
        const kernel_name_str = NSString_alloc();
        const name = NSString_initWithUTF8String(kernel_name_str, @ptrCast(kernel_name));
        defer NSString_release(name);
        
        const function = MTLLibrary_newFunctionWithName(library, name) orelse {
            return MetalError.FunctionCreationFailed;
        };
        
        var pipeline_error: ?*NSError = null;
        const pipeline = MTLDevice_newComputePipelineStateWithFunction(device, function, &pipeline_error) orelse {
            return MetalError.PipelineCreationFailed;
        };
        
        // Cache the pipeline
        const owned_name = try self.allocator.dupe(u8, kernel_name);
        try self.compiled_pipelines.put(owned_name, pipeline);
        
        return pipeline;
    }
};

// Global MLIR-Metal execution engine
var global_metal_engine: ?MLIRMetalExecutionEngine = null;

// Initialize the Metal backend with MLIR support
pub fn init(allocator: Allocator) !void {
    if (global_metal_engine != null) return;

    std.debug.print("MLIR-Metal backend: Starting initialization...\n", .{});

    // Check if we're on an Apple platform
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
        std.debug.print("MLIR-Metal backend: Unsupported platform! Requires macOS or iOS.\n", .{});
        return MetalError.UnsupportedPlatform;
    }

    std.debug.print("MLIR-Metal backend: Initializing execution engine...\n", .{});
    global_metal_engine = try MLIRMetalExecutionEngine.init(allocator);
    
    std.debug.print("MLIR-Metal backend: Initialization completed successfully.\n", .{});
    std.debug.print("MLIR-Metal backend: Ready to execute MLIR graphs on Metal.\n", .{});
}

// Clean up the Metal backend
pub fn deinit() void {
    if (global_metal_engine) |*engine| {
        engine.deinit();
        global_metal_engine = null;
    }
}

// Get the global MLIR-Metal execution engine
pub fn getExecutionEngine() !*MLIRMetalExecutionEngine {
    if (global_metal_engine) |*engine| {
        return engine;
    }

    return MetalError.MetalInitFailed;
}



// MLIR-Metal Backend API using proper dialect lowering
pub const MLIRMetalBackend = struct {
    // Check if this backend supports StableHLO operations through dialect lowering
    pub fn supportsStableHLOOp(comptime op_name: []const u8) bool {
        // With proper MLIR dialect lowering, we can support all StableHLO operations
        // that can be lowered to GPU operations
        return std.mem.startsWith(u8, op_name, "stablehlo.");
    }

    /// Execute an MLIR module containing StableHLO operations using MLIR dialect lowering
    pub fn executeMLIRModule(module: mlir.Module, inputs: [][]const f32, outputs: [][]f32) !void {
        const engine = try getExecutionEngine();
        return try engine.executeMLIRModule(module, inputs, outputs);
    }
    
    /// Execute a single MLIR builder's module using dialect lowering
    pub fn executeMLIRBuilder(builder: *MLIRBuilder, inputs: [][]const f32, outputs: [][]f32) !void {
        return try executeMLIRModule(builder.module, inputs, outputs);
    }
    
    /// Check if the MLIR module can be lowered to Metal
    pub fn canLowerToMetal(module: mlir.Module) bool {
        // In a full implementation, this would analyze the module to ensure
        // all operations can be lowered through the StableHLO → GPU → Metal pipeline
        _ = module;
        return true; // Placeholder
    }
};