const std = @import("std");
const mlir = @import("wrapper.zig");
const c = @import("c.zig").c;

const Allocator = std.mem.Allocator;

const TILING_TRANSFORM_SCRIPT =
    \\module attributes { transform.with_named_sequence } {
    \\  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    \\    transform.sequence %arg0 : !transform.any_op failures(propagate) {
    \\      ^bb1(%arg1: !transform.any_op):
    \\        %matmul = transform.structured.match ops{["linalg.generic"]} attributes {iterator_types = ["parallel", "parallel", "parallel", "reduction"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    \\        %init_or_alloc_op, %more_parallel_fill_op, %split_matmul, %combining_linalg_op = transform.structured.split_reduction %matmul {split_factor = 8, insert_split_dimension = 3} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    \\        %tiled_split, %forall_split = transform.structured.tile_using_forall %split_matmul tile_sizes [8, 64, 64, 16, 0]
    \\          (mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    \\        %tiled_fill, %forall_fill = transform.structured.tile_using_forall %more_parallel_fill_op tile_sizes [8, 64, 64, 16]
    \\          (mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.thread<x>, #gpu.thread<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    \\        %tiled_combine, %forall_combine = transform.structured.tile_using_forall %combining_linalg_op tile_sizes [8, 64, 64]
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
    \\    transform.gpu.map_nested_forall_to_threads %func block_dims = [64, 4, 1] sync_after_distribute = false warp_size = 1 : (!transform.any_op) -> !transform.any_op
    \\    transform.yield
    \\  }
    \\}
;

/// MLIR Context wrapper for StableHLO → GPU → SPIR-V pipeline
pub const MLIRContext = struct {
    allocator: Allocator,
    context: c.MlirContext,
    registry: ?c.MlirDialectRegistry,

    // --- REMOVED ---
    // pass_manager: mlir.PassManager, // <-- This field is the source of the state bug. Remove it.

    const Self = @This();

    pub fn init(allocator: Allocator) !Self {
        std.debug.print("MLIRContext.init: Creating IREE-backed MLIR context...\n", .{});
        const context = c.contextCreate();
        c.contextSetAllowUnregisteredDialects(context, true);

        // Register StableHLO dialect explicitly (external dialect, needs CAPI registration)
        const registry = c.dialectRegistryCreate();
        defer c.dialectRegistryDestroy(registry);
        c.mlirDialectHandleInsertDialect(c.mlirGetDialectHandle__stablehlo__(), registry);
        c.contextAppendDialectRegistry(context, registry);

        // Load all available dialects (core MLIR dialects from libIREECompiler)
        c.contextLoadAllAvailableDialects(context);

        std.debug.print("Checking if stablehlo.constant operation is registered...\n", .{});
        const constant_name = c.stringRefFromString("stablehlo.constant");
        if (!c.contextIsRegisteredOperation(context, constant_name)) {
            // Try alternative operation names
            std.debug.print("stablehlo.constant not found, trying stablehlo.add...\n", .{});
            const add_name = c.stringRefFromString("stablehlo.add");
            if (!c.contextIsRegisteredOperation(context, add_name)) {
                std.log.err("FATAL: Stablehlo dialect not registered. Check IREE build and linking.", .{});
                return error.DialectRegistrationFailed;
            }
        }
        std.debug.print("StableHLO dialect successfully registered!\n", .{});

        return Self{
            .allocator = allocator,
            .context = context,
            .registry = null, // We are not managing the registry after appending it.
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

    /// Compiles an MLIR module to a VMFB artifact using the IREE compiler.
    /// target_arch: Optional GPU target architecture (e.g., "gfx942" for MI300X, "sm_80" for A100)
    pub fn compileToVMFB(self: *Self, allocator: Allocator, mlir_source: []const u8, iree_target: []const u8, target_arch: ?[]const u8) ![]u8 {
        _ = self;

        // 2. Create unique temporary file paths
        const timestamp = std.time.timestamp();
        const temp_mlir_path = try std.fmt.allocPrint(allocator, "/tmp/pcp_module_{d}.mlir", .{timestamp});
        defer allocator.free(temp_mlir_path);
        const temp_vmfb_path = try std.fmt.allocPrint(allocator, "/tmp/pcp_module_{d}.vmfb", .{timestamp});
        defer allocator.free(temp_vmfb_path);

        // Write transform dialect scripts to temp files
        const temp_tiling_path = try std.fmt.allocPrint(allocator, "/tmp/pcp_tiling_{d}.mlir", .{timestamp});
        defer allocator.free(temp_tiling_path);
        const temp_mapping_path = try std.fmt.allocPrint(allocator, "/tmp/pcp_mapping_{d}.mlir", .{timestamp});
        defer allocator.free(temp_mapping_path);

        try std.fs.cwd().writeFile(.{ .sub_path = temp_tiling_path, .data = TILING_TRANSFORM_SCRIPT });
        try std.fs.cwd().writeFile(.{ .sub_path = temp_mapping_path, .data = GPU_MAPPING_SCRIPT });

        // 3. Write MLIR to temporary file
        try std.fs.cwd().writeFile(.{ .sub_path = temp_mlir_path, .data = mlir_source });

        // 4. Call the IREE compiler as a subprocess
        // Use IREE_COMPILE_PATH env var if set, otherwise look for locally built compiler, then fall back to PATH
        const iree_compile_path = std.process.getEnvVarOwned(allocator, "IREE_COMPILE_PATH") catch |err| blk: {
            if (err == error.EnvironmentVariableNotFound) {
                // Try locally built compiler in IREE_BUILD_DIR
                const build_dir = std.process.getEnvVarOwned(allocator, "IREE_BUILD_DIR") catch {
                    break :blk @as([]const u8, "iree-compile");
                };
                defer allocator.free(build_dir);
                const local_path = std.fs.path.join(allocator, &[_][]const u8{ build_dir, "tools", "iree-compile" }) catch {
                    break :blk @as([]const u8, "iree-compile");
                };
                // Check if it exists
                std.fs.accessAbsolute(local_path, .{}) catch {
                    allocator.free(local_path);
                    break :blk @as([]const u8, "iree-compile");
                };
                std.log.info("Using locally built iree-compile: {s}", .{local_path});
                break :blk local_path;
            }
            break :blk @as([]const u8, "iree-compile");
        };
        defer if (!std.mem.eql(u8, iree_compile_path, "iree-compile")) allocator.free(iree_compile_path);

        const target_arg = try std.fmt.allocPrint(allocator, "--iree-hal-target-backends={s}", .{iree_target});
        defer allocator.free(target_arg);

        const is_rocm = std.mem.eql(u8, iree_target, "rocm") or std.mem.eql(u8, iree_target, "hip");
        const is_cuda = std.mem.eql(u8, iree_target, "cuda");
        const is_metal = std.mem.eql(u8, iree_target, "metal-spirv");

        var argv = std.ArrayList([]const u8).init(allocator);
        defer argv.deinit();

        var hip_target_arg: ?[]u8 = null;
        var cuda_arch_arg: ?[]u8 = null;
        // var tiling_flag: ?[]u8 = null;
        // var dump_dir: ?[]u8 = null;
        defer if (hip_target_arg) |arg| allocator.free(arg);
        defer if (cuda_arch_arg) |arg| allocator.free(arg);
        // defer if (tiling_flag) |arg| allocator.free(arg);
        // defer if (dump_dir) |arg| allocator.free(arg);

        try argv.append(iree_compile_path);
        try argv.append(temp_mlir_path);
        try argv.append(target_arg);

        // Handle architecture targets
        if (is_rocm) {
            // Default to gfx942 (MI300X) if not specified for ROCm
            const arch = target_arch orelse "gfx942";
            std.log.info("Compiling for HIP target: {s}", .{arch});
            hip_target_arg = try std.fmt.allocPrint(allocator, "--iree-hip-target={s}", .{arch});
            try argv.append(hip_target_arg.?);
            try argv.append("--iree-codegen-llvmgpu-use-vector-distribution=false");
            try argv.append("--iree-codegen-llvmgpu-use-reduction-vector-distribution=false");

            // RDNA3 (gfx11xx) Specific Stability Fixes
            // RDNA3 is natively Wave32. We must disable prefetching (pipelining) to prevent
            // register spilling/hangs (Status 10).
            if (std.mem.indexOf(u8, arch, "gfx11") != null) {
                // 1. Disable software pipelining (Verified in IREE help output)
                try argv.append("--iree-llvmgpu-enable-prefetch=false");

                // 2. Aggressively limit waves per EU to 1
                // This forces minimal register usage to prevent spills on Wave32 arch.
                try argv.append("--iree-hip-waves-per-eu=1");
            }
        } else if (is_cuda) {
            // Default to sm_80 (A100) if not specified for CUDA
            const arch = target_arch orelse "sm_80";
            std.log.info("Compiling for CUDA target: {s}", .{arch});
            cuda_arch_arg = try std.fmt.allocPrint(allocator, "--iree-hal-target-device=cuda://{s}", .{arch});
            try argv.append(cuda_arch_arg.?);

            // Disable all advanced CUDA codegen features to avoid LLVM lowering issues
            try argv.append("--iree-codegen-llvmgpu-use-vector-distribution=false");
            try argv.append("--iree-codegen-llvmgpu-use-reduction-vector-distribution=false");
        } else if (is_metal) {
            // Use MSL source (runtime will compile to Metal)
            try argv.append("--iree-metal-compile-to-metallib=false");
        }

        // Apply transform dialect scripts for GPU targets
        // COMMENTED OUT: Transform scripts causing CUDA compilation to hang
        // if (is_cuda or is_rocm) {
        //     tiling_flag = try std.fmt.allocPrint(allocator, "--iree-codegen-transform-dialect-library={s}", .{temp_tiling_path});
        //     try argv.append(tiling_flag.?);

        //     // Enable debugging: dump intermediates to /tmp/dumps
        //     dump_dir = try std.fmt.allocPrint(allocator, "--iree-hal-dump-executable-intermediates-to=/tmp/dumps_{d}", .{timestamp});
        //     try argv.append(dump_dir.?);

        //     std.log.info("Using transform scripts: tiling={s}", .{temp_tiling_path});
        // }

        // --- FIXES FOR NanoGPT COMPILATION ---

        // 1. Enable 64-bit indexing (Required for large models)
        try argv.append("--iree-vm-target-index-bits=64");
        try argv.append("--iree-stream-resource-index-bits=64");
        try argv.append("--iree-input-demote-i64-to-i32=false");

        // 2. Force aggressive fusion to merge func.call into SCF loop body
        // NOTE: These flags cause MLIR domination errors when fusing into scf.for loops
        // try argv.append("--iree-dispatch-creation-enable-aggressive-fusion");
        // try argv.append("--iree-dispatch-creation-fuse-multi-use");
        try argv.append("--iree-global-opt-propagate-transposes");

        // 3. Disable prefetching
        try argv.append("--iree-llvmgpu-enable-prefetch=false");

        // 4. Optimize for minimal peak memory (critical for large backward graphs)
        try argv.append("--iree-stream-partitioning-favor=min-peak-memory");

        // 5. Cap individual allocs to 4GB to force tiling
        try argv.append("--iree-stream-resource-max-allocation-size=4294967296");

        try argv.append("-o");
        try argv.append(temp_vmfb_path);

        std.log.info("Compiling with flags: {s}", .{argv.items});

        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = argv.items,
            .max_output_bytes = 100 * 1024 * 1024, // 100MB limit for stderr/stdout
        }) catch |err| {
            std.log.err("Failed to execute `iree-compile`: {s}", .{@errorName(err)});
            return err;
        };
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);

        if (result.term != .Exited or result.term.Exited != 0) {
            std.log.err("IREE compilation failed with stdout:\n{s}", .{result.stdout});
            std.log.err("IREE compilation failed with stderr:\n{s}", .{result.stderr});
            std.log.err("MLIR Source saved to: {s}", .{temp_mlir_path});
            return error.IREECompilationFailed;
        }

        const vmfb_binary = try std.fs.cwd().readFileAlloc(allocator, temp_vmfb_path, 2 * 1024 * 1024 * 1024); // 2GB limit

        // Cleanup temp files
        std.fs.deleteFileAbsolute(temp_mlir_path) catch {};
        std.fs.deleteFileAbsolute(temp_vmfb_path) catch {};
        std.fs.deleteFileAbsolute(temp_tiling_path) catch {};
        std.fs.deleteFileAbsolute(temp_mapping_path) catch {};

        return vmfb_binary;
    }

    /// Get the MLIR context handle for creating modules
    pub fn getContext(self: *Self) mlir.Context {
        return mlir.Context{ .handle = self.context };
    }
};

/// Serialize an MLIR module to a string representation for network transfer.
pub fn serializeMLIRModule(allocator: Allocator, module: mlir.Module) ![]u8 {
    // Debug: std.debug.print("serializeMLIRModule: Creating buffer...\n", .{});
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
        fn callback(string_ref: c.MlirStringRef, userData: ?*anyopaque) callconv(.C) void {
            const context = @as(*SerializationContext, @ptrCast(@alignCast(userData.?)));
            const data = c.fromStringRef(string_ref);

            // Debug: std.debug.print("MLIR serialization callback: appending {} bytes (total so far: {})\n", .{ data.len, context.buffer.items.len });

            // Add bounds checking to prevent buffer overflow
            // Large backward pass models with many gradients can exceed 1GB of MLIR text
            if (data.len > 2 * 1024 * 1024 * 1024) { // 2GB sanity check
                std.debug.print("ERROR: MLIR serialization data chunk too large: {} bytes\n", .{data.len});
                context.error_flag.* = true;
                return;
            }
            context.buffer.appendSlice(data) catch |err| {
                std.debug.print("ERROR: Failed to append MLIR data chunk of {} bytes: {}\n", .{ data.len, err });
                context.error_flag.* = true;
                return;
            };
        }
    }.callback;

    // Debug: std.debug.print("serializeMLIRModule: About to call mlirOperationPrint...\n", .{});
    // Add null checks
    const module_op = module.op();
    if (@intFromPtr(module_op.handle.ptr) == 0) {
        std.debug.print("ERROR: Module operation handle is null\n", .{});
        return error.NullModuleHandle;
    }

    c.mlirOperationPrint(module_op.handle, writeToArrayList, &ctx);
    // Debug: std.debug.print("serializeMLIRModule: mlirOperationPrint completed, buffer size: {}\n", .{buffer.items.len});

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
