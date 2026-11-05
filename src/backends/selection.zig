/// Compile-time backend selection for the PCP distributed training system
/// This module provides clean separation between algorithms, orchestration, and execution
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

// Core interfaces
const execution = @import("executor.zig");
const worker_backend = @import("worker_backend.zig");

// Backend implementations
const iree_backend = @import("iree.zig");

// NEW: Necessary imports for HostExecutor
const mlir_ctx = @import("../mlir/context.zig");
const mlir = @import("../mlir/wrapper.zig");
const tensor = @import("../core/tensor.zig");

const Executor = execution.Executor;
const WorkerBackend = worker_backend.WorkerBackend;

/// Compile-time backend selection based on target platform and build options
pub const Backend = enum {
    metal,
    cuda,
    vulkan,
    rocm,
    cpu,

    /// Automatically select the best backend for the current platform
    pub fn selectDefault() Backend {
        return switch (builtin.os.tag) {
            .macos, .ios => .metal,
            .linux => .cpu, // Testing CPU backend to isolate CUDA runtime issue
            else => .cpu,
        };
    }

    /// Convert backend to string representation
    pub fn toString(self: Backend) []const u8 {
        return switch (self) {
            .metal => "metal",
            .cuda => "cuda",
            .vulkan => "vulkan",
            .rocm => "rocm",
            .cpu => "cpu",
        };
    }

    /// Convert backend to IREE driver name (for runtime)
    pub fn toIreeDriverName(self: Backend) []const u8 {
        return switch (self) {
            .metal => "metal", // IREE's Metal runtime driver name
            .cuda => "cuda",
            .vulkan => "vulkan",
            .rocm => "hip", // IREE uses 'hip' as the runtime driver name for ROCm
            .cpu => "local-sync", // IREE's CPU runtime driver name
        };
    }

    /// Convert backend to IREE compilation target name (for iree-compile)
    pub fn toIreeCompilationTarget(self: Backend) []const u8 {
        return switch (self) {
            .metal => "metal-spirv",
            .cuda => "cuda",
            .vulkan => "vulkan-spirv",
            .rocm => "rocm",
            .cpu => "llvm-cpu", // IREE's CPU compilation target
        };
    }
};

/// Host executor used by gateway controllers for local orchestration work.
/// Heavy model execution is delegated to workers; optimizer/coordinator updates run here.
const HostExecutor = struct {
    allocator: Allocator,
    context: mlir_ctx.MLIRContext,
    iree_backend: *iree_backend.IreeBackend,

    pub fn init(allocator: Allocator) !*HostExecutor {
        const self = try allocator.create(HostExecutor);
        self.allocator = allocator;
        // Initialize a real MLIR context for graph building
        self.context = try mlir_ctx.MLIRContext.init(allocator);

        // Initialize IREE with CPU (local-sync) backend for gateway-local
        // optimizer and coordinator graphs.
        // CPU backend always uses device 0
        self.iree_backend = try iree_backend.IreeBackend.init(allocator, .cpu, 0);

        return self;
    }

    pub fn deinit(self: *HostExecutor) void {
        self.iree_backend.deinit();
        self.context.deinit();
        self.allocator.destroy(self);
    }

    pub fn asExecutor(self: *HostExecutor) Executor {
        return Executor{
            .ptr = self,
            .vtable = &.{
                .materialize = materialize,
                .materialize_module = materialize_module,
                .getContext = getContext,
                .deinit = deinitInterface,
            },
        };
    }

    /// Materialize tensors using CPU backend (for optimizer updates)
    /// This compiles the current module to VMFB and runs it on the CPU.
    fn materialize(ptr: *anyopaque, t: tensor.Tensor(void)) anyerror![]u8 {
        const self: *HostExecutor = @ptrCast(@alignCast(ptr));

        // 1. Get the module from the tensor's builder
        const module = t.builder.module;

        // 2. Serialize MLIR to string
        // We use the helper from mlir_ctx to serialize the current state of the graph
        const mlir_source = try mlir_ctx.serializeMLIRModule(self.allocator, module);
        defer self.allocator.free(mlir_source);

        // 3. Compile to VMFB for CPU (llvm-cpu)
        // Gateway-local coordinator work runs on CPU, so target llvm-cpu.
        const vmfb = try self.context.compileToVMFB(self.allocator, mlir_source, "llvm-cpu", null);
        defer self.allocator.free(vmfb);

        // 4. Execute using the internal IREE backend
        // We assume the entry point is "main".
        // For optimizer updates, inputs are often constants embedded in the graph,
        // so we pass empty inputs here.
        // Note: If inputs were needed, the Tensor struct/Builder would need to track them.
        const inputs: [][]const u8 = &.{};
        const shapes: [][]const i64 = &.{};

        const outputs = try self.iree_backend.execute(vmfb, "main", inputs, shapes, null);

        if (outputs.len == 0) return error.NoOutput;

        // Return the first output (result of the computation).
        // Caller takes ownership of the memory.
        const result = outputs[0];

        // Clean up other outputs if any (unexpected for single tensor materialize)
        for (outputs[1..]) |o| {
            self.allocator.free(o);
        }
        self.allocator.free(outputs); // Free the array container

        return result;
    }

    /// Serialize the MLIR module so it can be distributed (or inspected).
    fn materialize_module(ptr: *anyopaque, module: mlir.Module) anyerror![]u8 {
        const self: *HostExecutor = @ptrCast(@alignCast(ptr));
        return mlir_ctx.serializeMLIRModule(self.allocator, module);
    }

    /// Provide access to the MLIR context for graph building.
    fn getContext(ptr: *anyopaque) mlir.Context {
        const self: *HostExecutor = @ptrCast(@alignCast(ptr));
        return self.context.getContext();
    }

    fn deinitInterface(ptr: *anyopaque) void {
        const self: *HostExecutor = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
};

/// Create the gateway-local executor used for orchestration and optimizer steps.
pub fn createExecutor(allocator: Allocator, backend: Backend) !Executor {
    _ = backend;
    std.log.info("Initializing gateway-local executor for backend orchestration", .{});
    const host_exec = try HostExecutor.init(allocator);
    return host_exec.asExecutor();
}

/// Create a WorkerBackend for the specified backend
pub fn createWorkerBackend(allocator: Allocator, backend: Backend, device_id: usize) !WorkerBackend {
    // Use IREE backend for all hardware accelerators
    std.log.info("Using IREE backend for {s} execution on device {}", .{ backend.toString(), device_id });
    const iree = try iree_backend.IreeBackend.init(allocator, backend, device_id);
    return iree.asWorkerBackend();
}

/// Test the backend selection system
pub fn testBackendSelection(_: Allocator) !void {
    std.log.info("Testing backend selection system...");

    const default_backend = Backend.selectDefault();
    std.log.info("Default backend for this platform: {}", .{default_backend});

    std.log.info("✓ Backend selection test completed");
}
