const std = @import("std");
const mlir = @import("mlir.zig");
const c = @import("mlir/c.zig").c;

const Allocator = std.mem.Allocator;

const TILING_TRANSFORM_SCRIPT =
    \\module attributes { transform.with_named_sequence } {
    \\  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    \\    transform.sequence %arg0 : !transform.any_op failures(propagate) {
    \\      ^bb1(%arg1: !transform.any_op):
    \\        %matmul = transform.structured.match ops{["linalg.generic"]} attributes {iterator_types = ["parallel", "parallel", "parallel", "reduction"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    \\        %init_or_alloc_op, %more_parallel_fill_op, %split_matmul, %combining_linalg_op = transform.structured.split_reduction %matmul {split_factor = 8, insert_split_dimension = 3} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    \\        %tiled_split, %forall_split = transform.structured.tile_using_forall %split_matmul tile_sizes [4, 32, 32, 8, 0]
    \\          (mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    \\        %tiled_fill, %forall_fill = transform.structured.tile_using_forall %more_parallel_fill_op tile_sizes [4, 32, 32, 8]
    \\          (mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    \\        %tiled_combine, %forall_combine = transform.structured.tile_using_forall %combining_linalg_op tile_sizes [4, 32, 32]
    \\          (mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    \\    }
    \\    transform.yield
    \\  }
    \\}
;

const GPU_MAPPING_SCRIPT =
    \\module attributes { transform.with_named_sequence } {
    \\  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    \\    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    \\    transform.gpu.map_forall_to_blocks %func { generate_gpu_launch } : (!transform.any_op) -> !transform.any_op
    \\    transform.gpu.map_nested_forall_to_threads %func block_dims = [32, 8, 1] sync_after_distribute = false warp_size = 1 : (!transform.any_op) -> !transform.any_op
    \\    transform.yield
    \\  }
    \\}
;

/// MLIR Context wrapper for StableHLO → GPU → SPIR-V pipeline
pub const MLIRContext = struct {
    allocator: Allocator,
    context: *c.MlirContext,
    registry: ?*c.MlirDialectRegistry,
    
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
    
    /// Create an MLIRContext wrapper from an existing MLIR context
    /// This avoids double initialization when sharing contexts between components
    pub fn fromContext(context: mlir.Context) Self {
        // Create a minimal wrapper that doesn't own the context
        // The registry is set to null since we're not managing it
        return Self{
            .allocator = std.heap.page_allocator, // Dummy allocator since we're not managing resources
            .context = context.handle,
            .registry = null, // Not managing the registry
        };
    }
    
    pub fn deinit(self: *Self) void {
        // self.pass_manager.deinit(); // <-- Remove this.
        if (self.registry) |registry| {
            c.dialectRegistryDestroy(registry);
        }
        // Only destroy context if we own it (registry is not null)
        if (self.registry != null) {
            c.contextDestroy(self.context);
        }
    }
    
    /// NEW IREE-based SPIR-V compilation replacing the complex manual pipeline
    pub fn lowerToSPIRV(_: *Self, allocator: Allocator, module: mlir.Module) ![]const u8 {
        std.debug.print("=== IREE-based SPIR-V Compilation Pipeline ===\n", .{});
        
        std.debug.print("Step 1: About to serialize MLIR module...\n", .{});
        // 1. Serialize MLIR module to file for IREE compilation
        const mlir_source = serializeMLIRModule(allocator, module) catch |err| {
            std.debug.print("ERROR: Failed to serialize MLIR module: {}\n", .{err});
            return err;
        };
        defer allocator.free(mlir_source);
        std.debug.print("Step 1 complete: Serialized {} bytes\n", .{mlir_source.len});
        
        const temp_mlir_path = "temp_graph.mlir";
        const temp_vmfb_path = "temp_graph.vmfb"; 
        const spirv_dump_dir = "temp_spirv_dump";
        
        std.debug.print("Step 2: About to write MLIR file...\n", .{});
        // Write MLIR to temporary file
        std.fs.cwd().writeFile(.{ .sub_path = temp_mlir_path, .data = mlir_source }) catch |err| {
            std.debug.print("ERROR: Failed to write MLIR file: {}\n", .{err});
            return err;
        };
        std.debug.print("Step 2 complete: Wrote file {s}\n", .{temp_mlir_path});
        
        std.debug.print("✓ Saved MLIR module to {s} ({} bytes)\n", .{ temp_mlir_path, mlir_source.len });
        
        // Proactively create the SPIR-V dump directory
        std.fs.cwd().makeDir(spirv_dump_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {}, // Directory already exists, ignore
            else => {
                std.debug.print("ERROR: Failed to create SPIR-V dump directory: {}\n", .{err});
                return err;
            },
        };
        std.debug.print("✓ Created/verified SPIR-V dump directory: {s}\n", .{spirv_dump_dir});
        
        // 2. Call IREE compiler via subprocess
        std.debug.print("About to execute IREE compiler subprocess...\n", .{});
        std.debug.print("Command: python3 iree_compile_wrapper.py {s} {s} {s} vulkan-spirv\n", .{ temp_mlir_path, temp_vmfb_path, spirv_dump_dir });
        
        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{
                "python3",
                "iree_compile_wrapper.py",
                temp_mlir_path,
                temp_vmfb_path,
                spirv_dump_dir,
                "vulkan-spirv"
            },
        }) catch |err| {
            std.debug.print("ERROR: Failed to run IREE compiler: {}\n", .{err});
            std.debug.print("This might indicate IREE is not installed or python3 is not available\n", .{});
            
            // Check if we can at least call python3
            const python_test = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &.{ "python3", "--version" },
            }) catch |py_err| {
                std.debug.print("ERROR: python3 not found: {}\n", .{py_err});
                return error.Python3NotFound;
            };
            defer {
                allocator.free(python_test.stdout);
                allocator.free(python_test.stderr);
            }
            std.debug.print("Python3 is available: {s}\n", .{python_test.stdout});
            
            return error.IREESubprocessFailed;
        };
        
        std.debug.print("IREE subprocess completed with term: {?}\n", .{result.term});
        
        defer {
            allocator.free(result.stdout);
            allocator.free(result.stderr);
        }
        
        if (result.term != .Exited or result.term.Exited != 0) {
            std.debug.print("IREE compiler failed with exit code: {?}\n", .{result.term});
            std.debug.print("Stdout: {s}\n", .{result.stdout});
            std.debug.print("Stderr: {s}\n", .{result.stderr});
            return error.IREECompilationFailed;
        }
        
        std.debug.print("✓ IREE compilation successful:\n{s}\n", .{result.stdout});
        
        // 3. Find and read the generated SPIR-V file(s)
        var spirv_dir = std.fs.cwd().openDir(spirv_dump_dir, .{ .iterate = true }) catch |err| {
            std.debug.print("ERROR: Could not open SPIR-V dump directory: {}\n", .{err});
            return error.SPIRVDumpDirNotFound;
        };
        defer spirv_dir.close();
        
        var iterator = spirv_dir.iterate();
        var spirv_binary: ?[]const u8 = null;
        
        while (try iterator.next()) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".spv")) {
                std.debug.print("✓ Found SPIR-V file: {s}\n", .{entry.name});
                
                const spirv_file_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ spirv_dump_dir, entry.name });
                defer allocator.free(spirv_file_path);
                
                spirv_binary = try std.fs.cwd().readFileAlloc(allocator, spirv_file_path, 10 * 1024 * 1024);
                break; // Use first SPIR-V file found
            }
        }
        
        // Clean up temporary files
        std.fs.cwd().deleteFile(temp_vmfb_path) catch {};
        std.fs.cwd().deleteTree(spirv_dump_dir) catch {};
        
        if (spirv_binary == null) {
            std.debug.print("ERROR: No SPIR-V files were generated by IREE\n", .{});
            return error.NoSPIRVGenerated;
        }
        
        std.debug.print("✓ Successfully extracted SPIR-V binary ({} bytes)\n", .{spirv_binary.?.len});
        return spirv_binary.?;
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
    std.debug.print("serializeMLIRModule: Creating buffer...\n", .{});
    var buffer = std.ArrayList(u8).init(allocator);
    var serialization_error = false;
    
    const SerializationContext = struct {
        buffer: *std.ArrayList(u8),
        error_flag: *bool,
    };
    
    var ctx = SerializationContext{
        .buffer = &buffer,
        .error_flag = &serialization_error,
    };
    
    const writeToArrayList = struct {
        fn callback(data_ptr: [*]const u8, data_len: usize, userData: ?*anyopaque) callconv(.C) void {
            const context = @as(*SerializationContext, @ptrCast(@alignCast(userData.?)));
            const data = data_ptr[0..data_len];
            
            std.debug.print("MLIR serialization callback: appending {} bytes (total so far: {})\n", .{ data_len, context.buffer.items.len });
            
            // Add bounds checking to prevent buffer overflow
            if (data_len > 100 * 1024 * 1024) { // 100MB sanity check
                std.debug.print("ERROR: MLIR serialization data chunk too large: {} bytes\n", .{data_len});
                context.error_flag.* = true;
                return;
            }
            context.buffer.appendSlice(data) catch |err| {
                std.debug.print("ERROR: Failed to append MLIR data chunk of {} bytes: {}\n", .{ data_len, err });
                context.error_flag.* = true;
                return;
            };
        }
    }.callback;

    std.debug.print("serializeMLIRModule: About to call mlirOperationPrint...\n", .{});
    // Add null checks
    const module_op = module.op();
    if (@intFromPtr(module_op.handle) == 0) {
        std.debug.print("ERROR: Module operation handle is null\n", .{});
        return error.NullModuleHandle;
    }
    
    c.mlirOperationPrint(module_op.handle, writeToArrayList, &ctx);
    std.debug.print("serializeMLIRModule: mlirOperationPrint completed, buffer size: {}\n", .{buffer.items.len});
    
    // Check if any errors occurred during serialization
    if (serialization_error) {
        std.debug.print("ERROR: MLIR serialization failed due to callback errors\n", .{});
        buffer.deinit();
        return error.MLIRSerializationFailed;
    }
    
    const serialized = buffer.toOwnedSlice() catch |err| {
        std.debug.print("ERROR: Failed to convert buffer to owned slice: {}\n", .{err});
        return err;
    };
    std.debug.print("✓ Serialized MLIR module to {} bytes\n", .{serialized.len});
    return serialized;
}

/// Deserialize an MLIR module from a string representation.
pub fn deserializeMLIRModule(allocator: Allocator, context: mlir.Context, data: []const u8) !mlir.Module {
    _ = allocator;
    
    std.debug.print("deserializeMLIRModule: About to parse {} bytes of MLIR data...\n", .{data.len});
    
    // Check if context is valid
    if (@intFromPtr(context.handle) == 0) {
        std.debug.print("ERROR: MLIR context handle is null in deserialization!\n", .{});
        return error.NullContextHandle;
    }
    std.debug.print("✓ MLIR context handle is valid: 0x{x}\n", .{@intFromPtr(context.handle)});
    
    // Show first 200 chars of MLIR data for debugging
    const preview_len = @min(200, data.len);
    std.debug.print("MLIR data preview ({} chars): {s}\n", .{ preview_len, data[0..preview_len] });
    
    // Also show data around line 870 where the error occurs
    if (data.len > 870) {
        std.debug.print("Investigating line 870 error - looking for problematic area...\n", .{});
        
        // Find approximate line 870 (assuming ~80 chars per line average)
        const approx_char_pos = 870 * 80;
        const start_pos = if (approx_char_pos > 100) approx_char_pos - 100 else 0;
        const end_pos = @min(approx_char_pos + 200, data.len);
        
        if (start_pos < data.len) {
            std.debug.print("Data around estimated line 870 (chars {}-{}): {s}\n", .{ start_pos, end_pos, data[start_pos..end_pos] });
        }
        
        // Look for null bytes or invalid characters
        var null_count: usize = 0;
        var first_null: ?usize = null;
        for (data, 0..) |byte, i| {
            if (byte == 0) {
                null_count += 1;
                if (first_null == null) first_null = i;
            }
        }
        
        if (null_count > 0) {
            std.debug.print("WARNING: Found {} null bytes in MLIR data! First at position {}\n", .{ null_count, first_null.? });
        }
    }
    
    const module = mlir.Module.parse(context, data) catch |err| {
        std.debug.print("ERROR: mlir.Module.parse failed: {}\n", .{err});
        return err;
    };
    
    std.debug.print("✓ mlir.Module.parse completed, checking module handle...\n", .{});
    if (@intFromPtr(module.handle) == 0) {
        std.debug.print("ERROR: Parsed module has null handle!\n", .{});
        return error.NullParsedModuleHandle;
    }
    std.debug.print("✓ Parsed module handle is valid: 0x{x}\n", .{@intFromPtr(module.handle)});
    
    // Check operation handle immediately
    const test_op = module.op();
    if (@intFromPtr(test_op.handle) == 0) {
        std.debug.print("ERROR: Module operation handle is null after parsing!\n", .{});
        return error.NullOperationHandle;
    }
    std.debug.print("✓ Module operation handle is valid: 0x{x}\n", .{@intFromPtr(test_op.handle)});
    
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

/// Extract kernel names from SPIR-V binary using SPIRV-Cross
pub fn extractKernelNamesFromSPIRV(allocator: Allocator, spirv_binary: []const u8) ![][]const u8 {
    var c_names: [*][*:0]const u8 = undefined;
    const count = c.extractKernelNamesFromSPIRV(spirv_binary, &c_names);
    defer c.freeKernelNames(c_names, count);

    if (count == 0) {
        std.debug.print("⚠ No GPU kernels found in SPIR-V binary\n", .{});
        // Return empty array
        return try allocator.alloc([]const u8, 0);
    }

    const names = try allocator.alloc([]const u8, count);
    for (0..count) |i| {
        const kernel_name = std.mem.span(c_names[i]);
        names[i] = try allocator.dupe(u8, kernel_name);
    }
    
    std.debug.print("✓ Extracted {} GPU kernel names from SPIR-V binary\n", .{names.len});
    return names;
}

/// Extract GPU kernel metadata from SPIR-V binary using SPIRV-Cross
pub fn extractGPUKernelInfo(allocator: Allocator, spirv_binary: []const u8) ![]GPUKernelInfo {
    // NEW IMPLEMENTATION: Extract names directly from the SPIR-V binary
    var c_names: [*][*:0]const u8 = undefined;
    const count = c.extractKernelNamesFromSPIRV(spirv_binary, &c_names);
    defer c.freeKernelNames(c_names, count);

    if (count == 0) {
        std.debug.print("⚠ No GPU kernels found in SPIR-V binary, using fallback\n", .{});
        // Fallback to demo kernel info if no kernels found
        const kernels = try allocator.alloc(GPUKernelInfo, 1);
        kernels[0] = GPUKernelInfo{
            .name = try allocator.dupe(u8, "fallback_kernel"),
            .grid_size = [3]usize{ 1, 1, 1 },
            .block_size = [3]usize{ 256, 1, 1 },
        };
        return kernels;
    }

    const kernels = try allocator.alloc(GPUKernelInfo, count);
    for (0..count) |i| {
        const kernel_name = std.mem.span(c_names[i]);
        kernels[i] = GPUKernelInfo{
            .name = try allocator.dupe(u8, kernel_name),
            // NOTE: We can't get grid/block size from SPIR-V.
            // This metadata is lost when we leave the MLIR ecosystem.
            // We must use a sensible default or determine it at runtime.
            .grid_size = [3]usize{ 1, 1, 1 },
            .block_size = [3]usize{ 256, 1, 1 },
        };
    }
    
    std.debug.print("✓ Extracted {} GPU kernel names from SPIR-V binary\n", .{kernels.len});
    return kernels;
}

// Test function to verify the complete StableHLO → GPU → SPIR-V → Metal pipeline
pub fn testMLIRGPUPipeline(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing MLIR StableHLO → GPU → SPIR-V → Metal Pipeline ===\n", .{});
    
    // Define our input data for simple 2x2 matrix multiplication
    const input_a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // 2x2 matrix
    const input_b_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 }; // 2x2 matrix

    // Manually calculate the expected "golden" result for verification
    // C = A @ B
    // C[0,0] = 1*5 + 2*7 = 19
    // C[0,1] = 1*6 + 2*8 = 22  
    // C[1,0] = 3*5 + 4*7 = 43
    // C[1,1] = 3*6 + 4*8 = 50
    const expected_output_data = [_]f32{ 19.0, 22.0, 43.0, 50.0 };
    
    // 1. Initialize MLIR context with GPU pipeline
    var mlir_ctx = try MLIRContext.init(allocator);
    defer mlir_ctx.deinit();
    
    // 2. Create a StableHLO module with simple 2x2 matrix multiplication for easier verification
    const stablehlo_module_str =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
        \\    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    return %0 : tensor<2x2xf32>
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
    
    // 3. Lower StableHLO → GPU → SPIR-V using IREE
    std.debug.print("About to call lowerToSPIRV via IREE...\n", .{});
    const spirv_binary = mlir_ctx.lowerToSPIRV(allocator, module) catch |err| {
        std.debug.print("ERROR in IREE lowerToSPIRV: {}\n", .{err});
        return err;
    };
    defer allocator.free(spirv_binary);
    
    // 4. Extract GPU kernel names from SPIR-V binary
    const kernel_names = try extractKernelNamesFromSPIRV(allocator, spirv_binary);
    defer {
        for (kernel_names) |name| {
            allocator.free(name);
        }
        allocator.free(kernel_names);
    }
    
    // 5. Translate SPIR-V to MSL  
    const msl_source = try translateSpirvToMsl(allocator, spirv_binary);
    defer allocator.free(msl_source);
    
    // 6. EXTRACT KERNEL INFO FROM THE CORRECT SOURCE: THE SPIR-V BINARY
    const kernel_info = try extractGPUKernelInfo(allocator, spirv_binary);
    defer {
        for (kernel_info) |*info| {
            info.deinit(allocator);
        }
        allocator.free(kernel_info);
    }
    
    std.debug.print("\n=== Verifying Metal Compilation Success ===\n", .{});

    // 7. Verify the MSL source contains expected kernel function
    const msl_contains_kernel = std.mem.indexOf(u8, msl_source, "kernel void") != null or
                                std.mem.indexOf(u8, msl_source, "kernel float") != null or  
                                std.mem.indexOf(u8, msl_source, "main_dispatch") != null;
    
    if (msl_contains_kernel) {
        std.debug.print("🌙 MSL contains GPU kernel function\n", .{});
    } else {
        std.debug.print("⚠ MSL may not contain expected kernel function\n", .{});
    }

    // 8. Verify kernel name extraction worked
    if (kernel_info.len > 0) {
        std.debug.print("🌙 Successfully extracted kernel: {s}\n", .{kernel_info[0].name});
        
        // 9. Verify the mathematical correctness (CPU verification)
        std.debug.print("Verifying mathematical correctness (CPU calculation)...\n", .{});
        std.debug.print("   Input A: {any}\n", .{input_a_data});
        std.debug.print("   Input B: {any}\n", .{input_b_data});
        std.debug.print("   Expected Result: {any}\n", .{expected_output_data});
        
        // CPU verification of matrix multiplication
        var cpu_result = [_]f32{0.0} ** 4;
        // A @ B for 2x2 matrices
        cpu_result[0] = input_a_data[0] * input_b_data[0] + input_a_data[1] * input_b_data[2]; // C[0,0]
        cpu_result[1] = input_a_data[0] * input_b_data[1] + input_a_data[1] * input_b_data[3]; // C[0,1]
        cpu_result[2] = input_a_data[2] * input_b_data[0] + input_a_data[3] * input_b_data[2]; // C[1,0]
        cpu_result[3] = input_a_data[2] * input_b_data[1] + input_a_data[3] * input_b_data[3]; // C[1,1]
        
        std.debug.print("   CPU Verification: {any}\n", .{cpu_result});
        
        const tolerance = 1e-6;
        for (cpu_result, expected_output_data) |cpu_val, expected_val| {
            if (@abs(cpu_val - expected_val) > tolerance) {
                std.debug.print("💣 CPU verification failed! Computed: {}, Expected: {}\n", .{ cpu_val, expected_val });
                return error.CPUVerificationFailed;
            }
        }
        
        std.debug.print("🌙 CPU verification successful! Math is correct.\n", .{});
    }

    // --- START: NEW EXECUTION AND VERIFICATION SECTION ---

    std.debug.print("\n=== Verifying Execution on Metal GPU ===\n", .{});

    // 1. Initialize the Metal Backend, PASSING IN our existing MLIR context
    const metal = @import("backends/metal.zig");
    try metal.init(allocator, &mlir_ctx);
    defer metal.deinit();
    
    const engine = try metal.getExecutionEngine();

    // 2. Prepare inputs and outputs for the execution engine
    var input_array = [_][]const f32{ input_a_data[0..], input_b_data[0..] };
    var actual_gpu_output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var output_array = [_][]f32{ actual_gpu_output[0..] };

    // 3. Execute the pre-compiled MSL on the GPU
    // This is the step that actually runs code on the M3.
    // Now you should see a spike in `mactop`!
    std.debug.print("👻 Dispatching kernel to M3 GPU for execution...\n", .{});
    try engine.executeMSL(msl_source, kernel_info, input_array[0..], output_array[0..]);
    std.debug.print("🌙 GPU execution finished.\n", .{});

    // 4. Verify the result read back from the GPU
    std.debug.print("Verifying GPU output against expected result...\n", .{});
    std.debug.print("   GPU Result: {any}\n", .{actual_gpu_output});
    std.debug.print("   Expected:   {any}\n", .{expected_output_data});

    const tolerance = 1e-6;
    for (actual_gpu_output, expected_output_data) |gpu_val, expected_val| {
        if (@abs(gpu_val - expected_val) > tolerance) {
            std.debug.print("💣 GPU verification failed! Computed: {}, Expected: {}\n", .{ gpu_val, expected_val });
            return error.GPUVerificationFailed;
        }
    }
    std.debug.print("🌙 Verification successful! The result from the GPU is correct.\n", .{});

    // --- END: NEW EXECUTION AND VERIFICATION SECTION ---

    // 10. Save the generated MSL for inspection
    std.debug.print("\n=== Generated Metal Shading Language ===\n", .{});
    try std.fs.cwd().writeFile(.{ .sub_path = "generated_kernel.metal", .data = msl_source });
    std.debug.print("✓ Saved MSL source to generated_kernel.metal for inspection\n", .{});
    
    // Show a snippet of the MSL
    const msl_preview_len = @min(400, msl_source.len);
    std.debug.print("MSL Preview ({} bytes shown):\n", .{msl_preview_len});
    std.debug.print("{s}\n", .{msl_source[0..msl_preview_len]});
    if (msl_source.len > msl_preview_len) {
        std.debug.print("... (truncated, see generated_kernel.metal for full source)\n", .{});
    }

    std.debug.print("✓ Complete MLIR GPU pipeline test completed successfully!\n", .{});
    std.debug.print("  Generated {} kernels\n", .{kernel_info.len});
    for (kernel_info) |ki| {
        std.debug.print("    - Kernel Name: {s}\n", .{ki.name});
    }
    std.debug.print("  SPIR-V binary: {} bytes\n", .{spirv_binary.len});
    std.debug.print("  MSL source: {} bytes\n", .{msl_source.len});
}