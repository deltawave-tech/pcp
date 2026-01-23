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
const grpo = @import("algorithms/grpo.zig");
const rl_shepherd = @import("nodes/controllers/rl_shepherd.zig");
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

const GRPOJsonConfig = struct {
    num_iterations: usize,
    group_size: usize,
    learning_rate: f32,
    beta: f32,
    prompt_file: []const u8,
    num_prompts: usize,
    weights_path: []const u8,
    generation_vmfb_path: []const u8,
    generation_mlir_path: []const u8,
    training_mlir_path: []const u8,
    num_gen_data_inputs: usize,
};

/// JSON Config File Structure - all required fields must be explicitly set
const ExperimentConfig = struct {
    model_path: []const u8,
    data_path: []const u8,
    tokenizer: []const u8,
    sampling: []const u8,
    learning_rate: f32,
    tau: usize,
    outer_loop_steps: usize,
    nesterov_momentum: f32,
    max_epochs: usize,

    // Required: Precision control ("f32", "bf16", or "f16")
    dtype: []const u8,

    // Optional: Effective batch size for gradient accumulation
    effective_batch_size: ?usize = null,

    // Optional: Use in-graph SCF loop for gradient accumulation (default: false = use in-code accumulation)
    // In-code accumulation calls compute_gradients multiple times from worker code
    // In-graph accumulation uses a single compute_gradients_accumulated call with internal SCF loop
    use_in_graph_accumulation: bool = false,

    // Optional: GRPO/RL configuration
    grpo_config: ?GRPOJsonConfig = null,

    // Optional: WandB configuration
    wandb_project: ?[]const u8 = null,
    wandb_entity: ?[]const u8 = null,
    wandb_run_name: ?[]const u8 = null,
    wandb_api_key: ?[]const u8 = null,

    // Optional: Checkpoint configuration
    checkpoint_dir: ?[]const u8 = null,
    should_resume: bool = false,
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
    device_id: usize,
    scale: usize,
    should_resume: bool,
    rl_mode: bool,
    no_dashboard: bool,
    terminate: bool,
    child_args: std.ArrayList([]const u8),

    const Mode = enum {
        shepherd,
        worker,
        node_manager,
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
                .device_id = 0,
                .scale = 1,
                .should_resume = false,
                .rl_mode = false,
                .no_dashboard = false,
                .terminate = false,
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
        var device_id: usize = 0;
        var scale: usize = 1;
        var should_resume: bool = false;
        var rl_mode: bool = false;
        var no_dashboard: bool = false;
        var terminate: bool = false;

        var i: usize = 1;
        while (i < args.len) {
            if (std.mem.eql(u8, args[i], "--worker")) {
                mode = .worker;
            } else if (std.mem.eql(u8, args[i], "--shepherd")) {
                mode = .shepherd;
            } else if (std.mem.eql(u8, args[i], "--node-manager")) {
                mode = .node_manager;
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
            } else if (std.mem.eql(u8, args[i], "--device-id")) {
                i += 1;
                if (i < args.len) {
                    device_id = std.fmt.parseInt(usize, args[i], 10) catch 0;
                }
            } else if (std.mem.eql(u8, args[i], "--scale")) {
                i += 1;
                if (i < args.len) {
                    scale = std.fmt.parseInt(usize, args[i], 10) catch 1;
                }
            } else if (std.mem.eql(u8, args[i], "--resume")) {
                should_resume = true;
            } else if (std.mem.eql(u8, args[i], "--rl")) {
                rl_mode = true;
            } else if (std.mem.eql(u8, args[i], "--no-dashboard")) {
                no_dashboard = true;
            } else if (std.mem.eql(u8, args[i], "--terminate")) {
                terminate = true;
            } else if (std.mem.eql(u8, args[i], "--supervise")) {
                supervise = true;
                i += 1;
                while (i < args.len) : (i += 1) {
                    try child_args_list.append(args[i]);
                }
                break;
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
            .device_id = device_id,
            .scale = scale,
            .should_resume = should_resume,
            .rl_mode = rl_mode,
            .no_dashboard = no_dashboard,
            .terminate = terminate,
            .child_args = child_args_list,
        };
    }

    pub fn printUsage() void {
        print("Usage: pcp_distributed [options]\n", .{});
        print("Options:\n", .{});
        print("  --shepherd           Run as Shepherd coordinator (default)\n", .{});
        print("  --worker             Run as Worker\n", .{});
        print("  --node-manager       Run as Node Manager (spawns multiple supervised workers)\n", .{});
        print("  --supervise -- <child_args>  Run with supervision (spawns child with args after --)\n", .{});
        print("  --config <path>      Path to experiment JSON config file (Shepherd only)\n", .{});
        print("  --connect <host:port> Connect to Shepherd at host:port\n", .{});
        print("  --host <host>        Host to bind/connect to (default: 127.0.0.1)\n", .{});
        print("  --port <port>        Port to bind/connect to (default: 8080)\n", .{});
        print("  --workers <count>    Number of workers to wait for (default: 2)\n", .{});
        print("  --model <path>       Path to MLIR model file (Shepherd only, overrides config)\n", .{});
        print("  --resume             Resume from previous training state\n", .{});
        print("  --rl                 Enable RL mode with GRPO algorithm (Shepherd only)\n", .{});
        print("  --no-dashboard       Disable TUI dashboard for clean log output (Shepherd only)\n", .{});
        print("  --terminate          Auto-terminate shepherd when training completes\n", .{});
        print("  --backend <type>     Backend to use: cpu, cuda, metal, vulkan, rocm (default: auto)\n", .{});
        print("  --target <arch>      GPU target architecture (e.g., gfx942 for MI300X, sm_80 for A100)\n", .{});
        print("  --device-id <id>     GPU device ID to use (default: 0, for multi-GPU nodes)\n", .{});
        print("  --scale <N>          Number of supervised workers to spawn (NodeManager only, default: 1)\n", .{});
        print("  --supervisor-id <id> Internal: Supervisor ID (used by spawned workers)\n", .{});
        print("  --help               Show this help message\n", .{});
        print("\nExamples:\n", .{});
        print("  # Run resilient shepherd:\n", .{});
        print("  ./pcp --supervise -- --shepherd --config experiment.json\n", .{});
        print("\n  # Run resilient worker on GPU 0:\n", .{});
        print("  ./pcp --supervise -- --worker --host 127.0.0.1 --port 8080 --device-id 0\n", .{});
        print("\n  # Run 8 workers on an 8xH100 node (one per GPU):\n", .{});
        print("  for i in {{0..7}}; do ./pcp --worker --device-id $i & done\n", .{});
        print("\n  # Run node manager with 8 supervised workers:\n", .{});
        print("  ./pcp --node-manager --scale 8 --backend cuda\n", .{});
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
    std.log.err("No config file specified. Use --config <path> to provide an experiment configuration.", .{});
    return error.ConfigFileRequired;
}

fn runRLShepherd(allocator: Allocator, args: Args) !void {
    print("Starting RL Shepherd coordinator with GRPO...\n", .{});

    const training_backend = args.backend orelse {
        print("Error: --backend flag is required for RL Shepherd mode\n", .{});
        return error.BackendRequired;
    };

    print("   Training Backend: {s}\n", .{training_backend.toString()});

    var config_result = try loadConfig(allocator, args.config_path);
    defer config_result.deinit();
    const exp_config = config_result.config;

    const json_grpo = exp_config.grpo_config orelse {
        print("Error: grpo_config is required in config file for RL mode\n", .{});
        return error.GRPOConfigRequired;
    };

    var rl_shepherd_controller = rl_shepherd.RLShepherd.init(allocator);
    defer rl_shepherd_controller.deinit();

    rl_shepherd_controller.training_backend_type = training_backend;

    const listen_thread = try std.Thread.spawn(.{}, rlShepherdListenThread, .{ &rl_shepherd_controller, args.host, args.port });
    // Ensure shepherd stops so listen thread can exit, preventing deadlock on error
    // Note: Zig executes defers in reverse order, so stop() runs before join()
    defer rl_shepherd_controller.base.stop();
    defer listen_thread.join();

    std.time.sleep(500 * std.time.ns_per_ms);

    const grpo_config = grpo.GRPOConfig{
        .num_iterations = json_grpo.num_iterations,
        .group_size = json_grpo.group_size,
        .learning_rate = json_grpo.learning_rate,
        .beta = json_grpo.beta,
        .prompt_file = json_grpo.prompt_file,
        .num_prompts = json_grpo.num_prompts,
        .weights_path = json_grpo.weights_path,
        .generation_vmfb_path = json_grpo.generation_vmfb_path,
        .generation_mlir_path = json_grpo.generation_mlir_path,
        .training_mlir_path = json_grpo.training_mlir_path,
        .num_gen_data_inputs = json_grpo.num_gen_data_inputs,
    };

    var grpo_algo = grpo.GRPO.init(allocator, &rl_shepherd_controller, grpo_config);
    var training_algo = grpo_algo.asTrainingAlgorithm();

    print("Starting GRPO training...\n", .{});
    try training_algo.run();

    print("GRPO training completed!\n", .{});

    if (args.terminate) {
        print("--terminate flag set, exiting...\n", .{});
        std.process.exit(0);
    }
}

/// RL Shepherd listening thread
fn rlShepherdListenThread(rl_shepherd_controller: *rl_shepherd.RLShepherd, host: []const u8, port: u16) !void {
    try rl_shepherd_controller.listen(host, port);
}

/// Run as Shepherd coordinator
fn runShepherd(allocator: Allocator, args: Args) !void {
    // Check if RL mode is enabled
    if (args.rl_mode) {
        return runRLShepherd(allocator, args);
    }

    const LATEST_RUN_FILE = ".pcp_latest_run";
    var run_id_buf: [64]u8 = undefined;
    var run_id: []const u8 = undefined;

    // 1. Determine Run ID (Resume vs Fresh Start)
    if (args.should_resume) {
        // Attempt to read the ID of the run we are resuming
        const f = std.fs.cwd().openFile(LATEST_RUN_FILE, .{}) catch |err| {
            print("Cannot resume: failed to open {s} to retrieve Run ID: {}\n", .{LATEST_RUN_FILE, err});
            return err;
        };
        defer f.close();
        const read_len = try f.readAll(&run_id_buf);
        if (read_len == 0) return error.EmptyRunIdFile;
        run_id = run_id_buf[0..read_len];
        print("Resuming previous Run ID: {s}\n", .{run_id});
    } else {
        // Fresh start: Generate new ID based on timestamp
        const timestamp = std.time.timestamp();
        run_id = try std.fmt.bufPrint(&run_id_buf, "run_{d}", .{timestamp});

        // Persist this ID immediately so we can recover it if we crash
        const f = try std.fs.cwd().createFile(LATEST_RUN_FILE, .{});
        defer f.close();
        try f.writeAll(run_id);
    }

    // 2. Setup paths using the determined run_id
    try std.fs.cwd().makePath("checkpoints");
    const run_dir_path = try std.fs.path.join(allocator, &[_][]const u8{ "checkpoints", run_id });
    defer allocator.free(run_dir_path);
    // Ensure directory exists (it might not if we resumed a run that crashed before creating it, though unlikely)
    try std.fs.cwd().makePath(run_dir_path);

    const recovery_dir_path = try std.fs.path.join(allocator, &[_][]const u8{ run_dir_path, "recovery" });
    defer allocator.free(recovery_dir_path);
    try std.fs.cwd().makePath(recovery_dir_path);

    var config_result = try loadConfig(allocator, args.config_path);
    defer config_result.deinit();
    const exp_config = config_result.config;

    print("👹 Starting Shepherd coordinator...\n", .{});
    print("   Run ID: {s}\n", .{run_id});
    print("   Output Dir: {s}\n", .{run_dir_path});
    print("   Resume: {}\n", .{args.should_resume});
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
    // 64MB chunks - must be large enough for tau inner loop steps
    // Calculation: B6 * T2048 * Tau100 * 2bytes = ~2.4MB per round minimum
    const chunk_size = 64 * 1024 * 1024;
    try shepherd_controller.initDataManager(total_size, chunk_size, exp_config.max_epochs);
    print("🌙 DataManager initialized: {} total size, {} byte chunks, {} max epochs\n", .{ total_size, chunk_size, exp_config.max_epochs });

    // 3. Populate DiLoCo Config with experiment configuration
    var diloco_config = DiLoCoConfig.default();
    diloco_config.model_mlir_path = exp_config.model_path;
    diloco_config.data_path = exp_config.data_path;
    diloco_config.tokenizer_type = exp_config.tokenizer;
    diloco_config.sampling_type = exp_config.sampling;
    diloco_config.tau = exp_config.tau;
    diloco_config.base_config.learning_rate = exp_config.learning_rate;
    diloco_config.base_config.outer_loop_steps = exp_config.outer_loop_steps;
    diloco_config.nesterov_momentum = exp_config.nesterov_momentum;
    diloco_config.wandb_project = exp_config.wandb_project orelse "pcp-distributed";
    diloco_config.wandb_entity = exp_config.wandb_entity;
    diloco_config.wandb_run_name = exp_config.wandb_run_name;
    diloco_config.wandb_api_key = exp_config.wandb_api_key;
    diloco_config.checkpoint_dir = run_dir_path;
    diloco_config.resume_training = args.should_resume;

    // Map dtype string to enum
    if (std.mem.eql(u8, exp_config.dtype, "bf16")) {
        diloco_config.dtype = .bf16;
    } else if (std.mem.eql(u8, exp_config.dtype, "f16")) {
        diloco_config.dtype = .f16;
    } else if (std.mem.eql(u8, exp_config.dtype, "f32")) {
        diloco_config.dtype = .f32;
    } else {
        std.log.err("Invalid dtype '{s}'. Must be 'f32', 'bf16', or 'f16'", .{exp_config.dtype});
        return error.InvalidDType;
    }
    print("   Precision: {s}\n", .{exp_config.dtype});

    if (exp_config.effective_batch_size) |ebs| {
        diloco_config.effective_batch_size = ebs;
        print("   Effective Batch Size: {}\n", .{ebs});
    }

    diloco_config.use_in_graph_accumulation = exp_config.use_in_graph_accumulation;
    if (exp_config.use_in_graph_accumulation) {
        print("   Gradient Accumulation: in-graph (SCF loop)\n", .{});
    } else {
        print("   Gradient Accumulation: in-code (multiple kernel calls)\n", .{});
    }

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

    // Start the TUI dashboard in its own thread (unless --no-dashboard)
    var dashboard_thread: ?std.Thread = null;
    if (!args.no_dashboard) {
        dashboard_thread = try std.Thread.spawn(.{}, dashboard.runDashboard, .{});
    }
    defer if (dashboard_thread) |t| t.join();

    // Wait for workers and start training
    print("Waiting for workers to connect...\n", .{});
    shepherd_controller.startTraining(args.workers) catch |err| {
        print("💣 Training failed with error: {}\n", .{err});
        return err;
    };

    print("🌑 Training completed successfully!\n", .{});

    if (args.terminate) {
        print("--terminate flag set, exiting...\n", .{});
        std.process.exit(0);
    }
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
    print("   Device ID: {}\n", .{args.device_id});

    if (args.target_arch) |target| {
        print("   Target Architecture: {s}\n", .{target});
    }

    const worker_backend_instance = try backend_selection.createWorkerBackend(allocator, backend, args.device_id);

    var worker_instance = try Worker.init(allocator, worker_backend_instance, args.supervisor_id);
    defer worker_instance.deinit();

    // Start worker main loop with automatic reconnection
    try worker_instance.runRobust(args.host, args.port, args.target_arch);

    print("💀 Worker shutting down\n", .{});
}

/// Run as Node Manager
fn runNodeManager(allocator: Allocator, args: Args) !void {
    const backend = args.backend orelse backend_selection.Backend.selectDefault();
    const backend_str = backend.toString();

    print("Starting Node Manager...\n", .{});
    print("   Target Backend: {s}\n", .{backend_str});
    print("   Spawning {} Supervisors\n", .{args.scale});

    const NodeManager = @import("nodes/node_manager.zig").NodeManager;
    var manager = try NodeManager.init(
        allocator,
        args.host,
        args.port,
        backend_str,
        args.target_arch,
    );
    defer manager.deinit();

    try manager.spawnSupervisors(args.scale);

    // Keep running and monitor
    print("Node Manager monitoring {} supervisors...\n", .{args.scale});
    try manager.wait();

    print("Node Manager shutting down\n", .{});
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
        .node_manager => try runNodeManager(allocator, args),
    }
}

/// Test function for the distributed system
pub fn testDistributedSystem(allocator: Allocator) !void {
    std.log.info("Testing distributed system components...");

    // Test Shepherd
    try shepherd.testShepherd(allocator);

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
