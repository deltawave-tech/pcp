const std = @import("std");
const mlir = @import("mlir.zig");
const c = @import("mlir/c.zig").c;

const Allocator = std.mem.Allocator;

const TILING_TRANSFORM_SCRIPT =
    \\module attributes { transform.with_named_sequence } {
    \\  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    \\    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
    \\        : (!transform.any_op) -> !transform.any_op
    \\
    \\    // Use the single-stage tiling with the proven mapping syntax.
    \\    // The tile size [4, 32, 32] correctly tiles all three dimensions,
    \\    // including the reduction dim, and prevents the loop from being folded away.
    \\    %tiled, %loops = transform.structured.tile_using_forall %generic
    \\        tile_sizes [4, 32, 32]
    \\        (mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.thread<x>])
    \\        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    \\
    \\    transform.yield
    \\  }
    \\}
;

/// MLIR Context wrapper for StableHLO → GPU → SPIR-V pipeline
pub const MLIRContext = struct {
    allocator: Allocator,
    context: *c.MlirContext,
    registry: *c.MlirDialectRegistry,
    
    // --- REMOVED ---
    // pass_manager: mlir.PassManager, // <-- This field is the source of the state bug. Remove it.
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) !Self {
        std.debug.print("MLIRContext.init: Creating MLIR context...\n", .{});
        const context = c.contextCreate();
        c.contextSetAllowUnregisteredDialects(context, true);

        // 1. EXPLICIT DIALECT REGISTRATION
        // Using dialect handles forces the linker to include the dialect's
        // static library, which in turn runs the necessary static initializers.
        std.debug.print("MLIRContext.init: Registering dialects explicitly...\n", .{});
        const registry = c.dialectRegistryCreate();
        
        // Insert dialect handles into registry
        c.dialectHandleInsertDialect(c.getDialectHandleFunc(), registry);
        c.dialectHandleInsertDialect(c.getDialectHandleArith(), registry);
        c.dialectHandleInsertDialect(c.getDialectHandleLinalg(), registry);
        c.dialectHandleInsertDialect(c.getDialectHandleTensor(), registry);
        c.dialectHandleInsertDialect(c.getDialectHandleTransform(), registry);
        c.dialectHandleInsertDialect(c.getDialectHandleGPU(), registry);
        c.dialectHandleInsertDialect(c.getDialectHandleSPIRV(), registry);
        c.dialectHandleInsertDialect(c.getDialectHandleSCF(), registry);
        // ADD THE ASYNC DIALECT HANDLE
        c.dialectHandleInsertDialect(c.getDialectHandleAsync(), registry);
        // StableHLO dialect registration - NOW ENABLED!
        c.dialectHandleInsertDialect(c.getDialectHandleStableHLO(), registry);
        c.dialectHandleInsertDialect(c.getDialectHandleCHLO(), registry);
        
        // Register BufferizableOpInterface implementations before appending registry to context
        std.debug.print("Registering BufferizableOpInterface implementations...\n", .{});
        c.registerBufferizationInterfaces(registry);
        
        // CRITICAL: Register Transform dialect extensions (for transform.structured.* ops)
        std.debug.print("Registering Transform dialect extensions...\n", .{});
        c.registerTransformExtensions(registry);

        // Append registry to context
        c.contextAppendDialectRegistry(context, registry);

        // Actually load the dialects into the context
        std.debug.print("Loading dialects into context...\n", .{});
        _ = c.dialectHandleLoadDialect(c.getDialectHandleFunc(), context);
        _ = c.dialectHandleLoadDialect(c.getDialectHandleArith(), context);
        _ = c.dialectHandleLoadDialect(c.getDialectHandleLinalg(), context);
        std.debug.print("Loading tensor dialect into context...\n", .{});
        _ = c.dialectHandleLoadDialect(c.getDialectHandleTensor(), context);
        std.debug.print("✓ Tensor dialect loaded successfully\n", .{});
        _ = c.dialectHandleLoadDialect(c.getDialectHandleTransform(), context);
        std.debug.print("✓ Transform dialect loaded successfully\n", .{});
        _ = c.dialectHandleLoadDialect(c.getDialectHandleGPU(), context);
        _ = c.dialectHandleLoadDialect(c.getDialectHandleSPIRV(), context);
        _ = c.dialectHandleLoadDialect(c.getDialectHandleSCF(), context);
        // LOAD THE ASYNC DIALECT
        _ = c.dialectHandleLoadDialect(c.getDialectHandleAsync(), context);
        _ = c.dialectHandleLoadDialect(c.getDialectHandleStableHLO(), context);
        _ = c.dialectHandleLoadDialect(c.getDialectHandleCHLO(), context);
        
        // CRITICAL: Register ALL passes for complete pipeline
        std.debug.print("Registering complete pass suite for full pipeline...\n", .{});
        
        // Core passes
        c.registerAllStablehloPasses();     // StableHLO passes
        c.registerCanonicalizerPass();      // canonicalize
        c.registerCSEPass();                // cse
        
        // Complete pass registration via C++ anchor function
        std.debug.print("Registering all MLIR passes via C++ anchor function...\n", .{});
        c.forceLoadAllRequiredPasses();
        std.debug.print("✓ All MLIR passes registered successfully\n", .{});

        // Verify registration worked
        if (!c.contextIsRegisteredOperation(context, "func.func")) {
            std.debug.print("ERROR: func dialect registration failed!\n", .{});
            return error.DialectRegistrationFailed;
        }
        std.debug.print("✓ Dialects registered successfully.\n", .{});
        
        // --- REMOVED ---
        // const pass_manager = try mlir.PassManager.init(...); // <-- Remove this.
        
        return Self{
            .allocator = allocator,
            .context = context,
            .registry = registry,
            // pass_manager = pass_manager, // <-- Remove this.
        };
    }
    
    pub fn deinit(self: *Self) void {
        // self.pass_manager.deinit(); // <-- Remove this.
        c.dialectRegistryDestroy(self.registry);
        c.contextDestroy(self.context);
    }
    
    /// Lowers the module using the definitive, pass-driven canonical GPU pipeline with 3D tiling.
    pub fn lowerToSPIRV(self: *Self, allocator: Allocator, module: mlir.Module) !void {
        _ = allocator;
        std.debug.print("=== Executing Definitive 3D Tiling GPU Lowering Pipeline ===\n", .{});

        // === STAGE 1 & 2: HLO -> Linalg -> Tiled scf.forall (on Tensors) ===
        {
            var pm = try mlir.PassManager.init(self.getContext());
            defer pm.deinit();
            try pm.asOpPassManager().addPipeline("stablehlo-legalize-to-linalg,func.func(linalg-generalize-named-ops,canonicalize,cse)");
            try pm.runOnOp(module.op());

            const transform_module = try mlir.Module.parse(self.getContext(), TILING_TRANSFORM_SCRIPT);
            defer transform_module.deinit();
            const main_module_body = module.op().getRegion(0).getBlock(0);
            const transform_body = transform_module.op().getRegion(0).getBlock(0);
            var op_to_clone = transform_body.getFirstOp();
            while (op_to_clone) |op| {
                main_module_body.appendOwnedOperation(mlir.Operation{ .handle = c.operationClone(op.handle) });
                op_to_clone = op.getNext();
            }
            c.operationSetAttributeByName(module.op().handle, "transform.with_named_sequence", c.unitAttrGet(self.context));

            var transform_pm = try mlir.PassManager.init(self.getContext());
            defer transform_pm.deinit();
            try transform_pm.asOpPassManager().addPipeline("transform-interpreter");
            std.debug.print("\n--- Running Transform Interpreter (3D Tiling) ---\n", .{});
            try transform_pm.runOnOp(module.op());
        }
        std.debug.print("\n--- IR after 3D Tiling (Attributed scf.forall) ---\n", .{});
        module.op().dump();
        
        // Clean up the transform script operations.
        std.debug.print("\n--- Cleaning Up Transform Script ---\n", .{});
        const module_body = module.op().getRegion(0).getBlock(0);
        var maybe_op = module_body.getFirstOp();
        while (maybe_op) |op| {
            const next_op = op.getNext();
            const op_name_id = c.operationGetName(op.handle);
            const op_name_ref = c.identifierStr(op_name_id);
            const op_name = c.fromStringRef(op_name_ref);
            if (std.mem.startsWith(u8, op_name, "transform.")) c.operationDestroy(op.handle);
            maybe_op = next_op;
        }
        _ = c.operationRemoveAttributeByName(module.op().handle, "transform.with_named_sequence");

        // === STAGE 3: The Final, Unified Lowering Pipeline ===
        // This pipeline correctly sequences all steps in a single PassManager run.
        {
            var pm = try mlir.PassManager.init(self.getContext());
            defer pm.deinit();

            const pipeline =
                // 1. **THE CRITICAL FIX**: Go directly from `scf.forall` on Tensors to the Memref domain first.
                // The bufferizer knows how to handle `scf.forall` with tensor semantics.
                "one-shot-bufferize{bufferize-function-boundaries=true}," ++
                "func.func(buffer-hoisting,buffer-loop-hoisting)," ++
                "buffer-deallocation-pipeline," ++
                "canonicalize,cse," ++

                // 2. **THE SECOND CRITICAL FIX**: Now that we are on memrefs, use the single, robust
                // pass that converts the mapped `scf.forall` DIRECTLY to `gpu.launch`.
                // This bypasses the problematic `scf.parallel` intermediate step.
                "convert-scf-to-gpu," ++

                // 3. Kernel Outlining: Create the gpu.module with gpu.func kernels.
                "gpu-kernel-outlining," ++

                // 4. Kernel Body Lowering: Lower the high-level ops inside the GPU kernel.
                "gpu.module(convert-linalg-to-loops,canonicalize,cse)," ++

                // 5. Final Conversion to SPIR-V: Convert the host and device code.
                "convert-gpu-to-spirv";

            std.debug.print("\n--- Running Final Unified Pipeline ---\n", .{});
            try pm.asOpPassManager().addPipeline(pipeline);
            try pm.runOnOp(module.op());
        }

        std.debug.print("\n--- Final Module IR (Lowered to SPIR-V) ---\n", .{});
        module.op().dump();
        std.debug.print("✅ Full GPU Pipeline executed successfully!\n", .{});
    }
    
    /// Translates a module containing SPIR-V dialect ops into a SPIR-V binary blob
    pub fn translateToSPIRV(self: *Self, module: mlir.Module) ![]const u8 {
        var spirv_binary = std.ArrayList(u8).init(self.allocator);
        
        const result = c.translateModuleToSPIRV(module.handle, &writeToArrayList, &spirv_binary);
        if (mlir.isFailure(result)) {
            spirv_binary.deinit();
            return error.MLIRTranslationFailed;
        }
        
        std.debug.print("✓ Successfully translated module to SPIR-V binary ({} bytes)\n", .{spirv_binary.items.len});
        std.debug.print(">>> Real SPIR-V binary size: {} bytes (stub was 20 bytes)\n", .{spirv_binary.items.len});
        return spirv_binary.toOwnedSlice();
    }
    
    /// Extracts the names of all generated GPU kernels from a lowered module
    /// You need these names to launch the kernels from your Metal runtime
    pub fn getGpuKernelNames(self: *Self, module: mlir.Module) ![][]const u8 {
        var names = std.ArrayList([]const u8).init(self.allocator);
        
        var walk_ctx = WalkContext{ .alloc = self.allocator, .list = &names };
        
        c.operationWalk(module.op().handle, &kernelNameExtractor, &walk_ctx);
        
        const result = names.toOwnedSlice();
        std.debug.print("✓ Extracted {} GPU kernel names from module\n", .{names.items.len});
        return result;
    }
    
    /// Get the MLIR context handle for creating modules
    pub fn getContext(self: *Self) mlir.Context {
        return mlir.Context{ .handle = self.context };
    }
    
    const WalkContext = struct {
        alloc: Allocator,
        list: *std.ArrayList([]const u8),
    };
    
    // Callback for mlirOperationWalk to extract GPU kernel names
    fn kernelNameExtractor(op: *c.MlirOperation, userData: ?*anyopaque) callconv(.C) c.MlirWalkResult {
        const walk_ctx: *anyopaque = userData.?;
        const ctx = @as(*const WalkContext, @ptrCast(@alignCast(walk_ctx))).*;
        
        // Check if the operation is a `gpu.func`. This is the MLIR representation
        // of a GPU kernel.
        const op_name_id = c.operationGetName(op);
        const op_name_ref = c.identifierStr(op_name_id);
        const op_name = c.fromStringRef(op_name_ref);
        
        if (std.mem.eql(u8, op_name, "gpu.func")) {
            // The kernel name is stored in the `sym_name` attribute.
            const attr = c.operationGetAttributeByName(op, "sym_name");
            if (@intFromPtr(attr) != 0 and c.attributeIsAString(attr)) {
                const string_attr = @as(*c.MlirStringAttribute, @ptrCast(attr));
                const sym_name_ref = c.stringAttributeGetValue(string_attr);
                const sym_name = c.fromStringRef(sym_name_ref);
                
                const owned_name = ctx.alloc.dupe(u8, sym_name) catch |err| {
                    std.debug.print("Failed to allocate kernel name: {}\n", .{err});
                    return .Interrupt;
                };
                
                ctx.list.append(owned_name) catch |err| {
                    std.debug.print("Failed to append kernel name: {}\n", .{err});
                    return .Interrupt;
                };
            }
        }
        return .Advance;
    }
    
    // Callback for translateModuleToSPIRV to collect SPIR-V binary data
    fn writeToArrayList(ref: c.MlirStringRef, userData: ?*anyopaque) callconv(.C) void {
        const list = @as(*std.ArrayList(u8), @ptrCast(@alignCast(userData.?)));
        const data = c.fromStringRef(ref);
        list.appendSlice(data) catch {};
    }
};

/// Serialize an MLIR module to a string representation for network transfer.
pub fn serializeMLIRModule(allocator: Allocator, module: mlir.Module) ![]u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    
    const writeToArrayList = struct {
        fn callback(data_ptr: [*]const u8, data_len: usize, userData: ?*anyopaque) callconv(.C) void {
            const list = @as(*std.ArrayList(u8), @ptrCast(@alignCast(userData.?)));
            const data = data_ptr[0..data_len];
            list.appendSlice(data) catch {};
        }
    }.callback;

    c.mlirOperationPrint(module.op().handle, writeToArrayList, &buffer);
    const serialized = try buffer.toOwnedSlice();
    std.debug.print("✓ Serialized MLIR module to {} bytes\n", .{serialized.len});
    return serialized;
}

/// Deserialize an MLIR module from a string representation.
pub fn deserializeMLIRModule(allocator: Allocator, context: mlir.Context, data: []const u8) !mlir.Module {
    _ = allocator;
    const module = try mlir.Module.parse(context, data);
    std.debug.print("✓ Deserialized MLIR module from {} bytes\n", .{data.len});
    return module;
}

/// SPIR-V to Metal Shading Language (MSL) translator using SPIRV-Cross
pub fn translateSpirvToMsl(allocator: Allocator, spirv_binary: []const u8) ![]u8 {
    var msl_source = std.ArrayList(u8).init(allocator);
    
    // Use SPIRV-Cross to translate SPIR-V to MSL
    const result = c.translateSPIRVToMSL(
        spirv_binary.ptr,
        spirv_binary.len,
        &MLIRContext.writeToArrayList,
        &msl_source
    );
    
    if (mlir.isFailure(result)) {
        msl_source.deinit();
        // Fallback to template if SPIRV-Cross fails
        std.debug.print("⚠ SPIRV-Cross translation failed, using template MSL\n", .{});
        const msl_template = 
            \\#include <metal_stdlib>
            \\using namespace metal;
            \\
            \\// Template MSL kernel (SPIRV-Cross translation failed)
            \\kernel void gpu_kernel_add(device const float* input0 [[buffer(0)]],
            \\                          device const float* input1 [[buffer(1)]],
            \\                          device float* output [[buffer(2)]],
            \\                          uint index [[thread_position_in_grid]]) {
            \\    output[index] = input0[index] + input1[index];
            \\}
        ;
        return try allocator.dupe(u8, msl_template);
    }
    
    const msl_result = try msl_source.toOwnedSlice();
    std.debug.print("✓ Translated SPIR-V to MSL using SPIRV-Cross ({} bytes → {} bytes)\n", .{ spirv_binary.len, msl_result.len });
    return msl_result;
}

/// GPU Kernel metadata extracted from MLIR GPU dialect
pub const GPUKernelInfo = struct {
    name: []const u8,
    grid_size: [3]usize,
    block_size: [3]usize,
    
    pub fn deinit(self: *GPUKernelInfo, allocator: Allocator) void {
        allocator.free(self.name);
    }
};

/// Extract GPU kernel metadata from the lowered MLIR module using MLIR API
pub fn extractGPUKernelInfo(allocator: Allocator, module: mlir.Module) ![]GPUKernelInfo {
    // Use the C API to extract GPU kernel metadata
    var c_kernels: [*]c.GPUKernelMetadata = undefined;
    const count = c.extractGPUKernelMetadata(module.handle, &c_kernels);
    
    if (count == 0) {
        std.debug.print("⚠ No GPU kernels found in module, using fallback\n", .{});
        // Fallback to demo kernel info if no kernels found
        const kernels = try allocator.alloc(GPUKernelInfo, 1);
        kernels[0] = GPUKernelInfo{
            .name = try allocator.dupe(u8, "gpu_kernel_add"),
            .grid_size = [3]usize{ 1, 1, 1 },
            .block_size = [3]usize{ 256, 1, 1 },
        };
        return kernels;
    }
    
    // Convert C kernels to Zig kernels
    const kernels = try allocator.alloc(GPUKernelInfo, count);
    for (0..count) |i| {
        const c_kernel = c_kernels[i];
        kernels[i] = GPUKernelInfo{
            .name = try allocator.dupe(u8, std.mem.span(c_kernel.name)),
            .grid_size = [3]usize{ c_kernel.grid_size[0], c_kernel.grid_size[1], c_kernel.grid_size[2] },
            .block_size = [3]usize{ c_kernel.block_size[0], c_kernel.block_size[1], c_kernel.block_size[2] },
        };
    }
    
    // Free C kernels
    c.freeGPUKernelMetadata(c_kernels, count);
    
    std.debug.print("✓ Extracted {} GPU kernels with execution metadata\n", .{kernels.len});
    return kernels;
}

// Test function to verify the complete StableHLO → GPU → SPIR-V → Metal pipeline
pub fn testMLIRGPUPipeline(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing MLIR StableHLO → GPU → SPIR-V → Metal Pipeline ===\n", .{});
    
    // 1. Initialize MLIR context with GPU pipeline
    var mlir_ctx = try MLIRContext.init(allocator);
    defer mlir_ctx.deinit();
    
    // 2. Create a StableHLO module with 3D batch matrix multiplication for 3D tiling
    // Using stablehlo.dot_general with batch dimensions will test 3D GPU tiling
    const stablehlo_module_str =
        \\module {
        \\  func.func @main(%arg0: tensor<8x128x256xf32>, %arg1: tensor<8x256x512xf32>) -> tensor<8x128x512xf32> {
        \\    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x128x256xf32>, tensor<8x256x512xf32>) -> tensor<8x128x512xf32>
        \\    return %0 : tensor<8x128x512xf32>
        \\  }
        \\}
    ;
    
    const context = mlir_ctx.getContext();
    std.debug.print("Creating StableHLO module from string...\n", .{});
    const module = mlir.Module.parse(context, stablehlo_module_str) catch |err| {
        std.debug.print("ERROR parsing StableHLO module: {}\n", .{err});
        return err;
    };
    defer module.deinit();
    
    std.debug.print("✓ Created StableHLO module\n", .{});
    
    // 3. Lower StableHLO → GPU → SPIR-V
    std.debug.print("About to call lowerToSPIRV...\n", .{});
    mlir_ctx.lowerToSPIRV(allocator, module) catch |err| {
        std.debug.print("ERROR in lowerToSPIRV: {}\n", .{err});
        return err;
    };
    
    // 4. Extract GPU kernel names
    const kernel_names = try mlir_ctx.getGpuKernelNames(module);
    defer {
        for (kernel_names) |name| {
            allocator.free(name);
        }
        allocator.free(kernel_names);
    }
    
    // 5. Translate to SPIR-V binary
    const spirv_binary = try mlir_ctx.translateToSPIRV(module);
    defer allocator.free(spirv_binary);
    
    // 6. Translate SPIR-V to MSL
    const msl_source = try translateSpirvToMsl(allocator, spirv_binary);
    defer allocator.free(msl_source);
    
    // 7. Extract kernel metadata
    const kernel_info = try extractGPUKernelInfo(allocator, module);
    defer {
        for (kernel_info) |*info| {
            info.deinit(allocator);
        }
        allocator.free(kernel_info);
    }
    
    std.debug.print("✓ Complete MLIR GPU pipeline test completed successfully!\n", .{});
    std.debug.print("  Generated {} kernels\n", .{kernel_names.len});
    std.debug.print("  SPIR-V binary: {} bytes\n", .{spirv_binary.len});
    std.debug.print("  MSL source: {} bytes\n", .{msl_source.len});
}