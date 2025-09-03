/// Compile-time backend selection for the PCP distributed training system
/// This module provides clean separation between algorithms, orchestration, and execution

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

// Core interfaces
const execution = @import("execution.zig");
const worker_backend = @import("backends/worker_backend.zig");

// Algorithm implementations
const diloco = @import("algorithms/diloco.zig");
const shepherd = @import("controllers/shepherd.zig");
const worker = @import("worker.zig");

// Backend implementations
const metal_backend = @import("backends/metal.zig");
const demo_backend = @import("backends/demo.zig");
// const cuda_backend = @import("backends/cuda.zig");  // Future
// const cpu_backend = @import("backends/cpu.zig");    // Future

const Executor = execution.Executor;
const WorkerBackend = worker_backend.WorkerBackend;
const DiLoCo = diloco.DiLoCo;
const Shepherd = shepherd.Shepherd;
const Worker = worker.Worker;

/// Compile-time backend selection based on target platform and build options
pub const Backend = enum {
    metal,
    cuda,
    cpu,
    demo,
    
    /// Automatically select the best backend for the current platform
    pub fn selectDefault() Backend {
        return switch (builtin.os.tag) {
            .macos, .ios => .metal,
            .linux => .cuda, // Assume CUDA availability on Linux
            else => .cpu,
        };
    }
};

/// Create an Executor for the specified backend
pub fn createExecutor(allocator: Allocator, backend: Backend) !Executor {
    return switch (backend) {
        .metal => blk: {
            // Initialize Metal backend and return its Executor interface
            // Note: We need to create an MLIR context for Metal backend
            const mlir_ctx = @import("mlir_ctx.zig");
            var context = try mlir_ctx.MLIRContext.init(allocator);
            try metal_backend.init(allocator, &context);
            const engine = try metal_backend.getExecutionEngine();
            break :blk engine.asExecutor();
        },
        .demo => blk: {
            // Demo backend uses the real Metal MLIR context but simulated execution
            const mlir_ctx = @import("mlir_ctx.zig");
            var context = try mlir_ctx.MLIRContext.init(allocator);
            try metal_backend.init(allocator, &context);
            const engine = try metal_backend.getExecutionEngine();
            
            // Wrap the real executor with demo behavior
            const demo = try demo_backend.DemoBackend.init(allocator);
            demo.setRealExecutor(engine.asExecutor());
            break :blk demo.asExecutor();
        },
        .cuda => {
            // TODO: Implement CUDA backend
            std.log.err("CUDA backend not implemented yet", .{});
            return error.BackendNotImplemented;
        },
        .cpu => {
            // TODO: Implement CPU backend  
            std.log.err("CPU backend not implemented yet", .{});
            return error.BackendNotImplemented;
        },
    };
}

/// Create a WorkerBackend for the specified backend
pub fn createWorkerBackend(allocator: Allocator, backend: Backend) !WorkerBackend {
    return switch (backend) {
        .metal => blk: {
            // Initialize Metal backend and return its WorkerBackend interface
            const mlir_ctx = @import("mlir_ctx.zig");
            var context = try mlir_ctx.MLIRContext.init(allocator);
            try metal_backend.init(allocator, &context);
            const engine = try metal_backend.getExecutionEngine();
            break :blk engine.asWorkerBackend();
        },
        .demo => blk: {
            // Demo backend for demonstration purposes
            std.log.info("Using demo worker backend for simulated execution", .{});
            const demo = try demo_backend.DemoBackend.init(allocator);
            break :blk demo.asWorkerBackend();
        },
        .cuda => {
            // TODO: Implement CUDA backend
            std.log.err("CUDA WorkerBackend not implemented yet", .{});
            return error.BackendNotImplemented;
        },
        .cpu => {
            // TODO: Implement CPU backend
            std.log.err("CPU WorkerBackend not implemented yet", .{});
            return error.BackendNotImplemented;
        },
    };
}

/// Initialize a complete distributed training system with the specified backend
pub const DistributedTrainingSystem = struct {
    allocator: Allocator,
    backend: Backend,
    shepherd: Shepherd,
    executor: Executor,
    
    const Self = @This();
    
    /// Initialize the distributed training system
    pub fn init(allocator: Allocator, backend: Backend) !Self {
        return initWithDemo(allocator, backend, false);
    }
    
    /// Initialize the distributed training system with optional demo mode
    pub fn initWithDemo(allocator: Allocator, preferred_backend: Backend, demo_execution: bool) !Self {
        const backend = if (demo_execution) Backend.demo else preferred_backend;
        
        std.log.info("Initializing distributed training system with {} backend{s}", .{ 
            backend, 
            if (demo_execution) " (demo mode)" else ""
        });
        
        // Create the executor for algorithms
        const executor = try createExecutor(allocator, backend);
        
        // Initialize the shepherd coordinator
        var shepherd_instance = Shepherd.init(allocator);
        shepherd_instance.setExecutor(executor);
        
        return Self{
            .allocator = allocator,
            .backend = backend,
            .shepherd = shepherd_instance,
            .executor = executor,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.shepherd.deinit();
        // Executor cleanup is handled by shepherd
    }
    
    /// Create and configure a DiLoCo algorithm instance
    /// NOTE: This method is now deprecated. Use DiLoCo.init directly with a shared MLIRBuilder.
    pub fn createDiLoCoAlgorithm(self: *Self, config: diloco.DiLoCoConfig) !DiLoCo {
        _ = self;
        _ = config;
        @panic("createDiLoCoAlgorithm is deprecated - use DiLoCo.init directly with shared MLIRBuilder");
    }
    
    /// Create a worker with the appropriate backend
    pub fn createWorker(self: *Self) !Worker {
        const worker_backend_instance = try createWorkerBackend(self.allocator, self.backend);
        return Worker.init(self.allocator, worker_backend_instance);
    }
    
    /// Start the system and run training
    pub fn runTraining(self: *Self, config: diloco.DiLoCoConfig, required_workers: usize) !void {
        std.log.info("Starting distributed training with {} workers", .{required_workers});
        
        // Create and set the DiLoCo algorithm
        var diloco_algorithm = try self.createDiLoCoAlgorithm(config);
        defer diloco_algorithm.deinit();
        
        const algorithm_interface = diloco_algorithm.asTrainingAlgorithm();
        self.shepherd.setAlgorithm(@constCast(&algorithm_interface));
        
        // Start training
        try self.shepherd.startTraining(required_workers);
        
        std.log.info("Distributed training completed successfully");
    }
};

/// Example usage function showing the complete system integration
pub fn exampleUsage(allocator: Allocator) !void {
    std.log.info("=== PCP Distributed Training System Example ===");
    
    // 1. Select backend (can be overridden via build options)
    const backend = comptime Backend.selectDefault();
    std.log.info("Selected backend: {}", .{backend});
    
    // 2. Initialize the distributed training system
    var system = try DistributedTrainingSystem.init(allocator, backend);
    defer system.deinit();
    
    // 3. Configure DiLoCo algorithm
    const config = diloco.DiLoCoConfig{
        .base_config = .{
            .learning_rate = 0.001,
            .outer_loop_steps = 5,
            .max_epochs = 100,
        },
        .tau = 10, // Inner loop steps
        .nesterov_momentum = 0.9,
        .parameter_averaging = true,
        .param_count = 1000,
    };
    
    // 4. Start training with 3 workers
    try system.runTraining(config, 3);
    
    std.log.info("✓ Example completed successfully");
}


/// Test the backend selection system
pub fn testBackendSelection(allocator: Allocator) !void {
    std.log.info("Testing backend selection system...");
    
    // Test backend selection
    const default_backend = Backend.selectDefault();
    std.log.info("Default backend for this platform: {}", .{default_backend});
    
    // Test system initialization with demo backend 
    var system = try DistributedTrainingSystem.init(allocator, .demo);
    defer system.deinit();
    
    std.log.info("✓ Backend selection test completed");
}