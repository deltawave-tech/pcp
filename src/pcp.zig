/// Main entry point for distributed training
/// This file handles command-line arguments and launches either Shepherd or Worker

const std = @import("std");
const print = std.debug.print;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

// Import our distributed training components
const shepherd = @import("nodes/controllers/shepherd.zig");
const worker = @import("nodes/workers/worker.zig");
const diloco = @import("algorithms/diloco.zig");
const training_algorithm = @import("algorithms/training_algorithm.zig");
const backend_selection = @import("backends/selection.zig");
const ops = @import("core/ops.zig");
const mlir = @import("mlir/wrapper.zig");
const autodiff = @import("autodiff/engine.zig");
const dashboard = @import("ui/dashboard.zig");

const Shepherd = shepherd.Shepherd;
const Worker = worker.Worker;
const DiLoCo = diloco.DiLoCo;
const DiLoCoConfig = diloco.DiLoCoConfig;
const MLIRBuilder = ops.MLIRBuilder;

/// Command line arguments
const Args = struct {
    mode: Mode,
    host: []const u8,
    port: u16,
    workers: usize,
    model_path: ?[]const u8,
    backend: ?backend_selection.Backend,
    amd_target: ?[]const u8,

    const Mode = enum {
        shepherd,
        worker,
    };

    pub fn parse(_: Allocator, args: [][:0]u8) !Args {
        if (args.len < 2) {
            return Args{
                .mode = .shepherd,
                .host = "127.0.0.1",
                .port = 8080,
                .workers = 2,
                .model_path = null,
                .backend = null,
                .amd_target = null,
            };
        }

        var mode: Mode = .shepherd;
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 8080;
        var workers: usize = 2;
        var model_path: ?[]const u8 = null;
        var backend: ?backend_selection.Backend = null;
        var amd_target: ?[]const u8 = null;

        var i: usize = 1;
        while (i < args.len) {
            if (std.mem.eql(u8, args[i], "--worker")) {
                mode = .worker;
            } else if (std.mem.eql(u8, args[i], "--shepherd")) {
                mode = .shepherd;
            } else if (std.mem.eql(u8, args[i], "--host")) {
                i += 1;
                if (i < args.len) {
                    host = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--port")) {
                i += 1;
                if (i < args.len) {
                    port = std.fmt.parseInt(u16, args[i], 10) catch 8080;
                }
            } else if (std.mem.eql(u8, args[i], "--workers")) {
                i += 1;
                if (i < args.len) {
                    workers = std.fmt.parseInt(usize, args[i], 10) catch 2;
                }
            } else if (std.mem.eql(u8, args[i], "--connect")) {
                mode = .worker;
                i += 1;
                if (i < args.len) {
                    // Parse host:port format
                    const connect_str = args[i];
                    if (std.mem.indexOf(u8, connect_str, ":")) |colon_idx| {
                        host = connect_str[0..colon_idx];
                        port = std.fmt.parseInt(u16, connect_str[colon_idx + 1..], 10) catch 8080;
                    } else {
                        host = connect_str;
                    }
                }
            } else if (std.mem.eql(u8, args[i], "--model")) {
                i += 1;
                if (i < args.len) {
                    model_path = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--backend")) {
                i += 1;
                if (i < args.len) {
                    const backend_str = args[i];
                    if (std.mem.eql(u8, backend_str, "cpu")) {
                        backend = .cpu;
                    } else if (std.mem.eql(u8, backend_str, "cuda")) {
                        backend = .cuda;
                    } else if (std.mem.eql(u8, backend_str, "metal")) {
                        backend = .metal;
                    } else if (std.mem.eql(u8, backend_str, "vulkan")) {
                        backend = .vulkan;
                    } else if (std.mem.eql(u8, backend_str, "rocm")) {
                        backend = .rocm;
                    } else {
                        print("Unknown backend: {s}\n", .{backend_str});
                        return error.InvalidBackend;
                    }
                }
            } else if (std.mem.eql(u8, args[i], "--amd-target")) {
                i += 1;
                if (i < args.len) {
                    amd_target = args[i];
                }
            }
            i += 1;
        }

        return Args{
            .mode = mode,
            .host = host,
            .port = port,
            .workers = workers,
            .model_path = model_path,
            .backend = backend,
            .amd_target = amd_target,
        };
    }

    pub fn printUsage() void {
        print("Usage: pcp_distributed [options]\n", .{});
        print("Options:\n", .{});
        print("  --shepherd           Run as Shepherd coordinator (default)\n", .{});
        print("  --worker             Run as Worker\n", .{});
        print("  --connect <host:port> Connect to Shepherd at host:port\n", .{});
        print("  --host <host>        Host to bind/connect to (default: 127.0.0.1)\n", .{});
        print("  --port <port>        Port to bind/connect to (default: 8080)\n", .{});
        print("  --workers <count>    Number of workers to wait for (default: 2)\n", .{});
        print("  --model <path>       Path to MLIR model file (Shepherd only)\n", .{});
        print("  --backend <type>     Backend to use: cpu, cuda, metal, vulkan, rocm (default: auto)\n", .{});
        print("  --amd-target <arch>  AMD GPU target architecture (e.g., gfx942 for MI300X)\n", .{});
        print("  --help               Show this help message\n", .{});
    }
};


/// Run as Shepherd coordinator
fn runShepherd(allocator: Allocator, args: Args) !void {
    print("👹 Starting Shepherd coordinator...\n", .{});
    print("   Host: {s}\n", .{args.host});
    print("   Port: {}\n", .{args.port});
    print("   Waiting for {} workers\n", .{args.workers});

    // Determine backend based on platform (A100 = Linux = CUDA)
    const backend = backend_selection.Backend.selectDefault();
    print("   Backend: {s}\n", .{backend.toString()});

    // Initialize system
    var system = try backend_selection.DistributedTrainingSystem.init(
        allocator,
        backend
    );
    defer system.deinit();

    // Get the shared MLIR context from the executor
    const shared_mlir_context = system.executor.getContext();

    // Create MLIRBuilder with the shared context
    var mlir_builder = try MLIRBuilder.init(allocator, shared_mlir_context);
    defer mlir_builder.deinit();

    // Now use system.shepherd instead of a local variable
    const shepherd_controller = &system.shepherd;

    // Initialize the DataLoader for real training data
    print("Loading Tiny Shakespeare dataset...\n", .{});
    // Dataset expected at data/tiny_shakespeare.txt
    var data_loader = try @import("data/loader.zig").DataLoader.init(allocator, "data/tiny_shakespeare.txt");
    defer data_loader.deinit();
    print("🌙 Dataset loaded with {} tokens\n", .{data_loader.tokens.len});

    var diloco_config = DiLoCoConfig.default();
    diloco_config.base_config.batch_size = 64;
    diloco_config.base_config.learning_rate = 0.0006;
    diloco_config.tau = 50;
    diloco_config.base_config.outer_loop_steps = 100;

    if (args.model_path) |path| {
        diloco_config.model_mlir_path = path;
        print("   Model: {s}\n", .{path});
    } else {
        print("   Model: Default (src/models/nanogpt_forward.mlir)\n", .{});
    }

    print("Initializing DiLoCo algorithm...\n", .{});
    // This will trigger Autodiff -> Compilation
    var diloco_algo = try DiLoCo.init(allocator, shepherd_controller, diloco_config, system.executor, &mlir_builder, &data_loader);
    defer diloco_algo.deinit();
    print("🌙 DiLoCo algorithm initialized successfully\n", .{});

    var training_algo = diloco_algo.asTrainingAlgorithm();
    shepherd_controller.setAlgorithm(&training_algo);

    // Start listening for workers in a separate thread
    const listen_thread = try std.Thread.spawn(.{}, shepherdListenThread, .{ shepherd_controller, args.host, args.port });
    listen_thread.detach();

    // Start the TUI dashboard in its own thread
    const dashboard_thread = try std.Thread.spawn(.{}, dashboard.runDashboard, .{});
    defer dashboard_thread.join();

    // Wait for workers and start training
    print("Waiting for workers to connect...\n", .{});
    shepherd_controller.startTraining(args.workers) catch |err| {
        print("💣 Training failed with error: {}\n", .{err});
        return err;
    };

    print("🌑 Training completed successfully!\n", .{});
}

/// Shepherd listening thread
fn shepherdListenThread(shepherd_controller: *Shepherd, host: []const u8, port: u16) !void {
    try shepherd_controller.listen(host, port);
}

/// Run as Worker
fn runWorker(allocator: Allocator, args: Args) !void {
    print("🐉 Starting Worker...\n", .{});
    print("   Connecting to: {s}:{}\n", .{ args.host, args.port });

    const backend = args.backend orelse backend_selection.Backend.selectDefault();
    print("   Backend: {s}\n", .{backend.toString()});

    if (args.amd_target) |amd_target| {
        print("   AMD Target: {s}\n", .{amd_target});
    }

    const worker_backend_instance = try backend_selection.createWorkerBackend(allocator, backend);

    var worker_instance = try Worker.init(allocator, worker_backend_instance);
    defer worker_instance.deinit();

    // Connect to shepherd with AMD target info
    try worker_instance.connect(args.host, args.port, args.amd_target);

    print("🌙 Connected to Shepherd with ID: {}\n", .{worker_instance.getNodeId().?});

    // Start worker main loop
    try worker_instance.run();

    print("💀 Worker shutting down\n", .{});
}

/// Main function
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);

    // Check for help
    for (process_args) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            Args.printUsage();
            return;
        }
    }

    const args = try Args.parse(allocator, process_args);

    print("🪐 PCP Distributed Training System\n", .{});
    print("=====================================\n", .{});

    // Debug: Print parsed arguments
    print("Mode: {s}\n", .{@tagName(args.mode)});
    print("Host: {s}\n", .{args.host});
    print("Port: {}\n", .{args.port});
    print("Workers: {}\n", .{args.workers});
    print("=====================================\n", .{});

    switch (args.mode) {
        .shepherd => try runShepherd(allocator, args),
        .worker => try runWorker(allocator, args),
    }
}

/// Test function for the distributed system
pub fn testDistributedSystem(allocator: Allocator) !void {
    std.log.info("Testing distributed system components...");

    // Test Shepherd
    try shepherd.testShepherd(allocator);

    // Test Worker
    try worker.testWorker(allocator);

    // Test DiLoCo
    try diloco.testDiLoCo(allocator);

    std.log.info("🌚 All distributed system tests passed");
}

/// Integration test - run a mini distributed training session
pub fn testIntegration(_: Allocator) !void {
    std.log.info("Running integration test...");

    // This would spawn a shepherd and multiple workers in separate threads
    // and run a mini training session to verify the full pipeline works

    // For now, just log that the structure is in place
    std.log.info("🌙 Integration test structure ready");
}
