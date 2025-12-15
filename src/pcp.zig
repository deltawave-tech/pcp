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

/// JSON Config File Structure
const ExperimentConfig = struct {
    model_path: []const u8 = "src/models/nanogpt_forward.mlir",
    data_path: []const u8 = "data/tiny_shakespeare.txt",
    learning_rate: f32 = 0.001,
    batch_size: usize = 64,
    block_size: usize = 64,
    tau: usize = 10,
    outer_loop_steps: usize = 100,
    nesterov_momentum: f32 = 0.9,
    max_epochs: usize = 10,

    // WandB configuration
    wandb_project: []const u8 = "pcp-distributed",
    wandb_entity: ?[]const u8 = null,
    wandb_run_name: ?[]const u8 = null,
    wandb_api_key: ?[]const u8 = null,
};

/// Config result that owns the parsed JSON data
const ConfigResult = struct {
    config: ExperimentConfig,
    parsed: ?std.json.Parsed(ExperimentConfig),
    json_data: ?[]u8,  // Keep the raw JSON buffer alive
    allocator: Allocator,

    pub fn deinit(self: *@This()) void {
        if (self.parsed) |*p| {
            p.deinit();
        }
        if (self.json_data) |data| {
            self.allocator.free(data);
        }
    }
};

/// Command line arguments
const Args = struct {
    mode: Mode,
    host: []const u8,
    port: u16,
    workers: usize,
    config_path: ?[]const u8,
    model_path: ?[]const u8,
    backend: ?backend_selection.Backend,
    target_arch: ?[]const u8,
    supervisor_id: ?i64,
    supervise: bool,
    child_args: std.ArrayList([]const u8),

    const Mode = enum {
        shepherd,
        worker,
        supervisor,
    };

    pub fn parse(allocator: Allocator, args: [][:0]u8) !Args {
        var child_args_list = std.ArrayList([]const u8).init(allocator);

        if (args.len < 2) {
            return Args{
                .mode = .shepherd,
                .host = "127.0.0.1",
                .port = 8080,
                .workers = 2,
                .config_path = null,
                .model_path = null,
                .backend = null,
                .target_arch = null,
                .supervisor_id = null,
                .supervise = false,
                .child_args = child_args_list,
            };
        }

        var mode: Mode = .shepherd;
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 8080;
        var workers: usize = 2;
        var config_path: ?[]const u8 = null;
        var model_path: ?[]const u8 = null;
        var backend: ?backend_selection.Backend = null;
        var target_arch: ?[]const u8 = null;
        var supervisor_id: ?i64 = null;
        var supervise: bool = false;

        var i: usize = 1;
        while (i < args.len) {
            if (std.mem.eql(u8, args[i], "--worker")) {
                mode = .worker;
            } else if (std.mem.eql(u8, args[i], "--shepherd")) {
                mode = .shepherd;
            } else if (std.mem.eql(u8, args[i], "--supervisor")) {
                mode = .supervisor;
            } else if (std.mem.eql(u8, args[i], "--config")) {
                i += 1;
                if (i < args.len) {
                    config_path = args[i];
                }
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
            } else if (std.mem.eql(u8, args[i], "--target")) {
                i += 1;
                if (i < args.len) {
                    target_arch = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--supervisor-id")) {
                i += 1;
                if (i < args.len) {
                    supervisor_id = std.fmt.parseInt(i64, args[i], 10) catch null;
                }
            } else if (std.mem.eql(u8, args[i], "--supervise")) {
                // All subsequent arguments belong to the child process
                supervise = true;
                // Collect all remaining args for the child
                i += 1;
                while (i < args.len) : (i += 1) {
                    try child_args_list.append(args[i]);
                }
                break; // Stop parsing parent args
            }
            i += 1;
        }

        return Args{
            .mode = mode,
            .host = host,
            .port = port,
            .workers = workers,
            .config_path = config_path,
            .model_path = model_path,
            .backend = backend,
            .target_arch = target_arch,
            .supervisor_id = supervisor_id,
            .supervise = supervise,
            .child_args = child_args_list,
        };
    }

    pub fn printUsage() void {
        print("Usage: pcp_distributed [options]\n", .{});
        print("Options:\n", .{});
        print("  --shepherd           Run as Shepherd coordinator (default)\n", .{});
        print("  --worker             Run as Worker\n", .{});
        print("  --supervisor         Run as Supervisor (manages a Worker child process)\n", .{});
        print("  --supervise -- <child_args>  Run with supervision (spawns child with args after --)\n", .{});
        print("  --config <path>      Path to experiment JSON config file (Shepherd only)\n", .{});
        print("  --connect <host:port> Connect to Shepherd at host:port\n", .{});
        print("  --host <host>        Host to bind/connect to (default: 127.0.0.1)\n", .{});
        print("  --port <port>        Port to bind/connect to (default: 8080)\n", .{});
        print("  --workers <count>    Number of workers to wait for (default: 2)\n", .{});
        print("  --model <path>       Path to MLIR model file (Shepherd only, overrides config)\n", .{});
        print("  --backend <type>     Backend to use: cpu, cuda, metal, vulkan, rocm (default: auto)\n", .{});
        print("  --target <arch>      GPU target architecture (e.g., gfx942 for MI300X, sm_80 for A100)\n", .{});
        print("  --supervisor-id <id> Internal: Supervisor ID (used by spawned workers)\n", .{});
        print("  --help               Show this help message\n", .{});
        print("\nExamples:\n", .{});
        print("  # Run resilient shepherd:\n", .{});
        print("  ./pcp --supervise -- --shepherd --config experiment.json\n", .{});
        print("\n  # Run resilient worker:\n", .{});
        print("  ./pcp --supervise -- --worker --host 127.0.0.1 --port 8080\n", .{});
    }
};

/// Load experiment configuration from JSON file
fn loadConfig(allocator: Allocator, path: ?[]const u8) !ConfigResult {
    if (path) |p| {
        std.log.info("Loading config from: {s}", .{p});
        const data = try std.fs.cwd().readFileAlloc(allocator, p, 1024 * 1024);
        // Don't free data - the parsed strings point into it!
        const parsed = try std.json.parseFromSlice(ExperimentConfig, allocator, data, .{ .ignore_unknown_fields = true });
        // Keep both the parsed struct and raw JSON data alive so strings remain valid
        return ConfigResult{
            .config = parsed.value,
            .parsed = parsed,
            .json_data = data,
            .allocator = allocator,
        };
    }
    std.log.info("No config file specified, using defaults", .{});
    return ConfigResult{
        .config = ExperimentConfig{},
        .parsed = null,
        .json_data = null,
        .allocator = allocator,
    };
}

/// Run as Shepherd coordinator
fn runShepherd(allocator: Allocator, args: Args) !void {
    // 1. Load Experiment Config from JSON file (or use defaults)
    var config_result = try loadConfig(allocator, args.config_path);
    defer config_result.deinit();
    const exp_config = config_result.config;

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

    // 2. Initialize DataManager for chunk-based data partitioning
    print("Initializing DataManager for dataset: {s}...\n", .{exp_config.data_path});
    const file = try std.fs.cwd().openFile(exp_config.data_path, .{});
    defer file.close();
    const stat = try file.stat();
    const total_size = stat.size;
    const chunk_size = 100 * 1024; // 100KB chunks
    try shepherd_controller.initDataManager(total_size, chunk_size, exp_config.max_epochs);
    print("🌙 DataManager initialized: {} total size, {} byte chunks, {} max epochs\n", .{ total_size, chunk_size, exp_config.max_epochs });

    // 3. Populate DiLoCo Config with experiment configuration
    var diloco_config = DiLoCoConfig.default();
    diloco_config.model_mlir_path = exp_config.model_path;
    diloco_config.data_path = exp_config.data_path;
    diloco_config.batch_size = exp_config.batch_size;
    diloco_config.block_size = exp_config.block_size;
    diloco_config.tau = exp_config.tau;
    diloco_config.base_config.learning_rate = exp_config.learning_rate;
    diloco_config.base_config.outer_loop_steps = exp_config.outer_loop_steps;
    diloco_config.nesterov_momentum = exp_config.nesterov_momentum;
    diloco_config.wandb_project = exp_config.wandb_project;
    diloco_config.wandb_entity = exp_config.wandb_entity;
    diloco_config.wandb_run_name = exp_config.wandb_run_name;
    diloco_config.wandb_api_key = exp_config.wandb_api_key;

    // CLI flag overrides config file
    if (args.model_path) |path| {
        diloco_config.model_mlir_path = path;
        print("   Model: {s} (overridden by CLI)\n", .{path});
    } else {
        print("   Model: {s}\n", .{diloco_config.model_mlir_path});
    }

    print("Initializing DiLoCo algorithm...\n", .{});
    // This will trigger Autodiff -> Compilation
    var diloco_algo = try DiLoCo.init(allocator, shepherd_controller, diloco_config, system.executor, &mlir_builder);
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

    if (args.target_arch) |target| {
        print("   Target Architecture: {s}\n", .{target});
    }

    const worker_backend_instance = try backend_selection.createWorkerBackend(allocator, backend);

    var worker_instance = try Worker.init(allocator, worker_backend_instance, args.supervisor_id);
    defer worker_instance.deinit();

    // Start worker main loop with automatic reconnection
    try worker_instance.runRobust(args.host, args.port, args.target_arch);

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

    // Check if supervision is enabled
    if (args.supervise) {
        print("🛡️  Running in supervision mode (resilient)\n", .{});
        print("   Child args: ", .{});
        for (args.child_args.items) |arg| {
            print("{s} ", .{arg});
        }
        print("\n", .{});

        // This process becomes the Supervisor
        // It will spawn a child process with args.child_args
        var s = try @import("nodes/supervisor.zig").Supervisor.init(
            allocator,
            args.host,
            args.port,
            args.child_args.items,
        );
        defer s.deinit();
        try s.run();
        return;
    }

    switch (args.mode) {
        .shepherd => try runShepherd(allocator, args),
        .worker => try runWorker(allocator, args),
        .supervisor => {
            // Legacy supervisor mode for workers only
            var s = try @import("nodes/supervisor.zig").Supervisor.init(
                allocator,
                args.host,
                args.port,
                &[_][]const u8{ "--worker" },
            );
            defer s.deinit();
            try s.run();
        },
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
