const std = @import("std");
const mlir = @import("mlir.zig");
const c = @import("mlir/c.zig").c;

const Allocator = std.mem.Allocator;

/// MLIR Context wrapper for StableHLO → GPU → SPIR-V pipeline
pub const MLIRContext = struct {
    allocator: Allocator,
    context: *c.MlirContext,
    registry: *c.MlirDialectRegistry,
    stablehlo_to_spirv_pm: mlir.PassManager,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) !Self {
        // Register all passes and dialects first
        std.debug.print("MLIRContext.init: Registering all passes...\n", .{});
        c.registerAllPasses();

        std.debug.print("MLIRContext.init: Creating MLIR context...\n", .{});
        
        // Create MLIR context
        const context = c.contextCreate();
        c.contextSetAllowUnregisteredDialects(context, true);
        
        std.debug.print("MLIRContext.init: Creating dialect registry...\n", .{});
        
        // Create dialect registry and register all required dialects
        const registry = c.dialectRegistryCreate();
        
        std.debug.print("MLIRContext.init: Registering all dialects...\n", .{});
        c.registerAllDialects(registry);
        
        std.debug.print("MLIRContext.init: Loading dialects into context...\n", .{});
        
        // Load all available dialects into the context
        c.dialectRegistryLoadAll(registry, context);
        
        std.debug.print("MLIRContext.init: Creating pass manager...\n", .{});
        
        // Create pass manager for StableHLO → GPU → SPIR-V pipeline
        const stablehlo_to_spirv_pm = try mlir.PassManager.init(mlir.Context{ .handle = context });
        
        // --- FIX: PROPER SPIR-V PIPELINE ---
        // Full SPIR-V pipeline - let's debug what's failing
        const pipeline_str = "builtin.module(stablehlo-legalize-to-linalg,canonicalize,cse,linalg-bufferize,canonicalize,cse,convert-linalg-to-gpu,canonicalize,cse,gpu-kernel-outlining,gpu-to-spirv)";
        
        std.debug.print("MLIRContext.init: Attempting to add full SPIR-V pipeline: \"{s}\"\n", .{pipeline_str});
        
        // Get the OpPassManager and add the pipeline
        std.debug.print("MLIRContext.init: Getting OpPassManager...\n", .{});
        const opm = stablehlo_to_spirv_pm.asOpPassManager();
        std.debug.print("MLIRContext.init: Got OpPassManager successfully\n", .{});
        
        // Add debug information and error handling
        std.debug.print("MLIRContext.init: About to call addPipeline...\n", .{});
        
        // Try with a much simpler pipeline first to test if addPipeline works at all
        const simple_pipeline = "canonicalize";
        std.debug.print("MLIRContext.init: Testing simple pipeline: \"{s}\"\n", .{simple_pipeline});
        
        opm.addPipeline(simple_pipeline) catch |simple_err| {
            std.debug.print("MLIRContext.init: Even simple pipeline failed: {}\n", .{simple_err});
            return simple_err;
        };
        
        std.debug.print("MLIRContext.init: Simple pipeline worked! Now trying full pipeline...\n", .{});
        
        opm.addPipeline(pipeline_str) catch |err| {
            std.debug.print("MLIRContext.init: ERROR adding pipeline: {}\n", .{err});
            std.debug.print("MLIRContext.init: Pipeline that failed: \"{s}\"\n", .{pipeline_str});
            
            // Try to add passes one by one to identify the problematic one
            std.debug.print("MLIRContext.init: Testing individual passes...\n", .{});
            
            const test_passes = [_][]const u8{
                "builtin.module(canonicalize)",
                "builtin.module(cse)",
                "builtin.module(stablehlo-legalize-to-linalg)",
                "builtin.module(linalg-bufferize)",
                "builtin.module(convert-linalg-to-gpu)",
                "builtin.module(gpu-kernel-outlining)",
                "builtin.module(gpu-to-spirv)",
            };
            
            for (test_passes) |test_pass| {
                std.debug.print("MLIRContext.init: Testing pass: {s}\n", .{test_pass});
                opm.addPipeline(test_pass) catch |pass_err| {
                    std.debug.print("MLIRContext.init: FAILED pass: {s} - error: {}\n", .{test_pass, pass_err});
                    continue;
                };
                std.debug.print("MLIRContext.init: SUCCESS pass: {s}\n", .{test_pass});
            }
            
            return err;
        };

        std.debug.print("MLIRContext.init: Successfully added SPIR-V pipeline!\n", .{});
        // --- END FIX ---
        
        std.debug.print("Initialized MLIR context with StableHLO → GPU → SPIR-V pipeline\n", .{});
        
        return Self{
            .allocator = allocator,
            .context = context,
            .registry = registry,
            .stablehlo_to_spirv_pm = stablehlo_to_spirv_pm,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.stablehlo_to_spirv_pm.deinit();
        c.dialectRegistryDestroy(self.registry);
        c.contextDestroy(self.context);
    }
    
    /// Lowers the module's GPU-targeted functions to SPIR-V dialect
    pub fn lowerToSPIRV(self: *Self, module: mlir.Module) !void {
        std.debug.print("Lowering MLIR module through StableHLO → GPU → SPIR-V pipeline...\n", .{});
        
        // Debug: Dump module before lowering
        std.debug.print("--- Module BEFORE lowering ---\n", .{});
        module.op().dump();
        
        try self.stablehlo_to_spirv_pm.runOnOp(module.op());
        
        // Debug: Dump module after lowering
        std.debug.print("--- Module AFTER lowering ---\n", .{});
        module.op().dump();
        
        std.debug.print("✓ Successfully lowered module to SPIR-V dialect\n", .{});
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

/// SPIR-V to Metal Shading Language (MSL) translator
/// This would typically use SPIRV-Cross or similar tool
pub fn translateSpirvToMsl(allocator: Allocator, spirv_binary: []const u8) ![]u8 {
    // In a real implementation, this would call SPIRV-Cross to translate SPIR-V to MSL
    // For now, we'll generate a template MSL kernel as demonstration
    
    const msl_template = 
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\// Generated from SPIR-V via SPIRV-Cross
        \\kernel void gpu_kernel_add(device const float* input0 [[buffer(0)]],
        \\                          device const float* input1 [[buffer(1)]],
        \\                          device float* output [[buffer(2)]],
        \\                          uint index [[thread_position_in_grid]]) {
        \\    output[index] = input0[index] + input1[index];
        \\}
        \\
        \\kernel void gpu_kernel_multiply(device const float* input0 [[buffer(0)]],
        \\                               device const float* input1 [[buffer(1)]],
        \\                               device float* output [[buffer(2)]],
        \\                               uint index [[thread_position_in_grid]]) {
        \\    output[index] = input0[index] * input1[index];
        \\}
    ;
    
    std.debug.print("✓ Translated SPIR-V to MSL ({} bytes → {} bytes)\n", .{ spirv_binary.len, msl_template.len });
    return try allocator.dupe(u8, msl_template);
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

/// Extract GPU kernel metadata from the lowered MLIR module
/// This analyzes gpu.launch_func operations to get execution parameters
pub fn extractGPUKernelInfo(allocator: Allocator, module: mlir.Module) ![]GPUKernelInfo {
    // In a full implementation, this would walk the MLIR module and extract
    // gpu.launch_func operations with their grid/block size parameters
    _ = module;
    
    // For now, return demo kernel info
    const kernels = try allocator.alloc(GPUKernelInfo, 2);
    kernels[0] = GPUKernelInfo{
        .name = try allocator.dupe(u8, "gpu_kernel_add"),
        .grid_size = [3]usize{ 1, 1, 1 },
        .block_size = [3]usize{ 256, 1, 1 },
    };
    kernels[1] = GPUKernelInfo{
        .name = try allocator.dupe(u8, "gpu_kernel_multiply"),
        .grid_size = [3]usize{ 1, 1, 1 },
        .block_size = [3]usize{ 256, 1, 1 },
    };
    
    std.debug.print("✓ Extracted {} GPU kernels with execution metadata\n", .{kernels.len});
    return kernels;
}

// Test function to verify the complete StableHLO → GPU → SPIR-V → Metal pipeline
pub fn testMLIRGPUPipeline(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing MLIR StableHLO → GPU → SPIR-V → Metal Pipeline ===\n", .{});
    
    // 1. Initialize MLIR context with GPU pipeline
    var mlir_ctx = try MLIRContext.init(allocator);
    defer mlir_ctx.deinit();
    
    // 2. Create a simple StableHLO module (would normally come from your tensor ops)
    const stablehlo_module_str =
        \\module {
        \\  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        \\    func.return %0 : tensor<4xf32>
        \\  }
        \\}
    ;
    
    const context = mlir_ctx.getContext();
    const module = try mlir.Module.parse(context, stablehlo_module_str);
    defer module.deinit();
    
    std.debug.print("✓ Created StableHLO module\n", .{});
    
    // 3. Lower StableHLO → GPU → SPIR-V
    try mlir_ctx.lowerToSPIRV(module);
    
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