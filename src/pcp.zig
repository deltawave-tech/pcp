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
const inference_shepherd = @import("nodes/controllers/inference_shepherd.zig");
const training_algorithm = @import("algorithms/training_algorithm.zig");
const backend_selection = @import("backends/selection.zig");
const ops = @import("core/ops.zig");
const mlir = @import("mlir/wrapper.zig");
const autodiff = @import("autodiff/engine.zig");
const dashboard = @import("ui/dashboard.zig");
const inference_config = @import("inference/config.zig");
const control_api = @import("control_plane/api.zig");
const control_state = @import("control_plane/state.zig");
const gateway = @import("gateway/gateway.zig");
const gateway_api = @import("gateway/api.zig");
const gateway_config = @import("gateway/config.zig");
const gateway_federation_client = @import("gateway/federation_client.zig");
const gateway_service_client = @import("gateway/service_client.zig");
const global_controller = @import("global_controller/controller.zig");
const global_controller_api = @import("global_controller/api.zig");

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

    // Optional: Static rematerialization planner controls
    remat_policy: autodiff.RematPolicy = .legacy_threshold,
    activation_memory_budget_bytes: ?u64 = null,
    remat_cost_model: autodiff.RematCostModel = .static_heuristic,
    remat_allow_expensive_ops: bool = true,
    checkmate_backend: autodiff.CheckmateBackend = .milp_glpk,
    checkmate_milp_solver_command: ?[]const u8 = null,
    checkmate_milp_lp_path: ?[]const u8 = null,
    checkmate_milp_solution_path: ?[]const u8 = null,
    checkmate_milp_keep_artifacts: bool = false,

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
    json_data: ?[]u8, // Keep the raw JSON buffer alive
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
    inference_config_path: ?[]const u8,
    gateway_config_path: ?[]const u8,
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
    streaming_mode: bool,
    api_host: []const u8,
    api_port: u16,
    control_host: []const u8,
    control_port: u16,
    api_token_env: ?[]const u8,
    enable_api: bool,
    child_args: std.ArrayList([]const u8),

    const Mode = enum {
        shepherd,
        worker,
        node_manager,
        inference,
        gateway,
        global_controller,
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
                .inference_config_path = null,
                .gateway_config_path = null,
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
                .streaming_mode = false,
                .api_host = "127.0.0.1",
                .api_port = 8000,
                .control_host = "127.0.0.1",
                .control_port = 8080,
                .api_token_env = null,
                .enable_api = false,
                .child_args = child_args_list,
            };
        }

        var mode: Mode = .shepherd;
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 8080;
        var workers: usize = 2;
        var config_path: ?[]const u8 = null;
        var inference_config_path: ?[]const u8 = null;
        var gateway_config_path: ?[]const u8 = null;
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
        var streaming_mode: bool = false;
        var api_host: []const u8 = "127.0.0.1";
        var api_port: u16 = 8000;
        var control_host: []const u8 = "127.0.0.1";
        var control_port: u16 = 8080;
        var api_token_env: ?[]const u8 = null;
        var enable_api: bool = false;

        var i: usize = 1;
        while (i < args.len) {
            if (std.mem.eql(u8, args[i], "--worker")) {
                mode = .worker;
            } else if (std.mem.eql(u8, args[i], "--shepherd")) {
                mode = .shepherd;
            } else if (std.mem.eql(u8, args[i], "--node-manager")) {
                mode = .node_manager;
            } else if (std.mem.eql(u8, args[i], "--inference")) {
                mode = .inference;
                enable_api = true;
            } else if (std.mem.eql(u8, args[i], "--gateway")) {
                mode = .gateway;
                enable_api = true;
            } else if (std.mem.eql(u8, args[i], "--global-controller")) {
                mode = .global_controller;
                enable_api = true;
            } else if (std.mem.eql(u8, args[i], "--config")) {
                i += 1;
                if (i < args.len) {
                    config_path = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--inference-config")) {
                i += 1;
                if (i < args.len) {
                    inference_config_path = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--gateway-config")) {
                i += 1;
                if (i < args.len) {
                    gateway_config_path = args[i];
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
            } else if (std.mem.eql(u8, args[i], "--api-host")) {
                enable_api = true;
                i += 1;
                if (i < args.len) {
                    api_host = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--api-port")) {
                enable_api = true;
                i += 1;
                if (i < args.len) {
                    api_port = std.fmt.parseInt(u16, args[i], 10) catch 8000;
                }
            } else if (std.mem.eql(u8, args[i], "--gateway-host")) {
                enable_api = true;
                i += 1;
                if (i < args.len) {
                    api_host = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--gateway-port")) {
                enable_api = true;
                i += 1;
                if (i < args.len) {
                    api_port = std.fmt.parseInt(u16, args[i], 10) catch 18010;
                }
            } else if (std.mem.eql(u8, args[i], "--control-host")) {
                i += 1;
                if (i < args.len) {
                    control_host = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--control-port")) {
                i += 1;
                if (i < args.len) {
                    control_port = std.fmt.parseInt(u16, args[i], 10) catch 8080;
                }
            } else if (std.mem.eql(u8, args[i], "--api-token-env")) {
                enable_api = true;
                i += 1;
                if (i < args.len) {
                    api_token_env = args[i];
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
                        port = std.fmt.parseInt(u16, connect_str[colon_idx + 1 ..], 10) catch 8080;
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
            } else if (std.mem.eql(u8, args[i], "--streaming")) {
                streaming_mode = true;
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
            .inference_config_path = inference_config_path,
            .gateway_config_path = gateway_config_path,
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
            .streaming_mode = streaming_mode,
            .api_host = api_host,
            .api_port = api_port,
            .control_host = control_host,
            .control_port = control_port,
            .api_token_env = api_token_env,
            .enable_api = enable_api,
            .child_args = child_args_list,
        };
    }

    pub fn printUsage() void {
        print("Usage: pcp_distributed [options]\n", .{});
        print("Options:\n", .{});
        print("  --shepherd           Run as Shepherd coordinator (default)\n", .{});
        print("  --worker             Run as Worker\n", .{});
        print("  --node-manager       Run as Node Manager (spawns multiple supervised workers)\n", .{});
        print("  --inference          Run inference controller (OpenAI-compatible API)\n", .{});
        print("  --gateway            Run gateway controller (local gateway API)\n", .{});
        print("  --global-controller  Run global controller (gateway federation registry)\n", .{});
        print("  --supervise -- <child_args>  Run with supervision (spawns child with args after --)\n", .{});
        print("  --config <path>      Path to experiment JSON config file (Shepherd only)\n", .{});
        print("  --inference-config <path>  Path to inference JSON config file (Inference only)\n", .{});
        print("  --gateway-config <path>  Path to gateway JSON config file (Gateway only)\n", .{});
        print("  --connect <host:port> Connect to Shepherd at host:port\n", .{});
        print("  --host <host>        Host to bind/connect to (default: 127.0.0.1)\n", .{});
        print("  --port <port>        Port to bind/connect to (default: 8080)\n", .{});
        print("  --api-host <host>    API host to bind for controller/operator APIs (default: 127.0.0.1)\n", .{});
        print("  --api-port <port>    API port to bind for controller/operator APIs (default: 8000)\n", .{});
        print("  --gateway-host <host> Gateway API host to bind (Gateway only, default: 127.0.0.1)\n", .{});
        print("  --gateway-port <port> Gateway API port to bind (Gateway only, default: 18010)\n", .{});
        print("  --control-host <host> Control host to bind (Inference only, default: 127.0.0.1)\n", .{});
        print("  --control-port <port> Control port to bind (Inference only, default: 8080)\n", .{});
        print("  --api-token-env <ENV> API token env var name for controller/operator auth\n", .{});
        print("  --workers <count>    Number of workers to wait for (default: 2)\n", .{});
        print("  --model <path>       Path to MLIR model file (Shepherd only, overrides config)\n", .{});
        print("  --resume             Resume from previous training state\n", .{});
        print("  --rl                 Enable RL mode with GRPO algorithm (Shepherd only)\n", .{});
        print("  --no-dashboard       Disable TUI dashboard for clean log output (Shepherd only)\n", .{});
        print("  --terminate          Auto-terminate shepherd when training completes\n", .{});
        print("  --streaming          Enable Streaming DiLoCo mode for asynchronous fragment updates\n", .{});
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
        print("\n  # Run inference controller:\n", .{});
        print("  ./pcp --inference --inference-config inference.json --api-host 0.0.0.0 --api-port 8000\n", .{});
        print("\n  # Run gateway:\n", .{});
        print("  ./pcp --gateway --gateway-config experiments/gateway_local.json --gateway-host 127.0.0.1 --gateway-port 18010\n", .{});
        print("\n  # Run global controller:\n", .{});
        print("  ./pcp --global-controller --api-host 127.0.0.1 --api-port 19010\n", .{});
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

    var operator_state = try control_state.ControllerState.init(
        allocator,
        .rl,
        .rl,
        args.config_path,
        null,
        exp_config.model_path,
        false,
        args.workers,
    );
    defer operator_state.deinit();
    operator_state.setStatus(.starting);

    var rl_shepherd_controller = rl_shepherd.RLShepherd.init(allocator);
    defer rl_shepherd_controller.deinit();

    rl_shepherd_controller.training_backend_type = training_backend;

    const listen_thread = try std.Thread.spawn(.{}, rlShepherdListenThread, .{ &rl_shepherd_controller, args.host, args.port });
    // Ensure shepherd stops so listen thread can exit, preventing deadlock on error
    // Note: Zig executes defers in reverse order, so stop() runs before join()
    defer rl_shepherd_controller.base.stop();
    defer listen_thread.join();

    var owned_api_token: ?[]u8 = null;
    defer if (owned_api_token) |token| allocator.free(token);

    var api_server = control_api.ControlApiServer.init(
        allocator,
        &operator_state,
        &rl_shepherd_controller.base,
        null,
        null,
        null,
        .{
            .ctx = &rl_shepherd_controller.base,
            .cancel = cancelShepherd,
        },
    );
    var api_thread: ?std.Thread = null;
    if (args.enable_api) {
        owned_api_token = try maybeLoadApiToken(allocator, args);
        api_server.api_token = owned_api_token;
        api_thread = try std.Thread.spawn(.{}, controlApiThread, .{ &api_server, args.api_host, args.api_port });
    }
    defer {
        if (api_thread) |thread| {
            api_server.stop();
            thread.join();
        }
    }

    var gateway_client = try gateway_service_client.GatewayClient.initFromEnv(allocator, "rl-main");
    defer if (gateway_client) |*client| client.deinit();
    registerGatewayServiceIfConfigured(
        if (gateway_client) |*client| client else null,
        "rl",
        args.api_host,
        args.api_port,
        &[_][]const u8{ "controller.status", "job.current", "job.cancel" },
        0,
        0,
        "starting",
        "ok",
    );

    std.time.sleep(500 * std.time.ns_per_ms);

    const grpo_config = grpo.GRPOConfig{
        .num_iterations = json_grpo.num_iterations,
        .group_size = json_grpo.group_size,
        .learning_rate = json_grpo.learning_rate,
        .beta = json_grpo.beta,
        .required_workers = args.workers,
        .prompt_file = json_grpo.prompt_file,
        .num_prompts = json_grpo.num_prompts,
        .weights_path = json_grpo.weights_path,
        .generation_vmfb_path = json_grpo.generation_vmfb_path,
        .generation_mlir_path = json_grpo.generation_mlir_path,
        .training_mlir_path = json_grpo.training_mlir_path,
        .num_gen_data_inputs = json_grpo.num_gen_data_inputs,
    };

    var grpo_algo = grpo.GRPO.init(allocator, &rl_shepherd_controller, grpo_config);
    grpo_algo.setControlState(&operator_state);
    grpo_algo.setGatewayClient(if (gateway_client) |*client| client else null);
    var training_algo = grpo_algo.asTrainingAlgorithm();

    print("Starting GRPO training...\n", .{});
    training_algo.run() catch |err| {
        if (err == error.Cancelled) {
            operator_state.setCancelled();
            print("GRPO training cancelled.\n", .{});
            return;
        }
        try operator_state.setFailed(@errorName(err));
        return err;
    };

    print("GRPO training completed!\n", .{});
    if (operator_state.isCancellationRequested()) {
        operator_state.setCancelled();
    } else {
        operator_state.setStatus(.completed);
    }

    if (args.terminate) {
        print("--terminate flag set, exiting...\n", .{});
        std.process.exit(0);
    }
}

fn runInferenceController(allocator: Allocator, args: Args) !void {
    const config_path = args.inference_config_path orelse {
        print("Error: --inference-config is required for inference mode\n", .{});
        return error.ConfigFileRequired;
    };

    var config_result = try inference_config.loadInferenceConfig(allocator, config_path);
    defer config_result.deinit();
    var config = config_result.config;

    if (args.api_token_env) |override_env| {
        config.api_token_env = override_env;
    }

    const api_token = std.process.getEnvVarOwned(allocator, config.api_token_env) catch |err| {
        print("Error: missing required API token env var {s}: {}\n", .{ config.api_token_env, err });
        return error.MissingApiToken;
    };
    defer allocator.free(api_token);

    var operator_state = try control_state.ControllerState.init(
        allocator,
        .inference,
        .inference,
        config_path,
        null,
        config.model_id,
        false,
        1,
    );
    defer operator_state.deinit();
    operator_state.setStatus(.running);

    var gateway_client = try gateway_service_client.GatewayClient.initFromEnv(allocator, "inference-main");
    defer if (gateway_client) |*client| client.deinit();

    var controller = try inference_shepherd.InferenceShepherd.init(
        allocator,
        config,
        api_token,
        &operator_state,
        if (gateway_client) |*client| client else null,
    );
    controller.attachHooks();
    defer controller.deinit();

    const listen_thread = try std.Thread.spawn(.{}, inferenceListenThread, .{ &controller, args.control_host, args.control_port });
    listen_thread.detach();
    defer controller.requestShutdown();

    try controller.startMaintenance();

    const api_thread = try std.Thread.spawn(.{}, inferenceApiThread, .{ &controller, args.api_host, args.api_port });
    defer api_thread.join();

    print("🚀 Starting Inference Controller (phase 1 bootstrap)\n", .{});
    print("   Model ID: {s}\n", .{config.model_id});
    print("   Pool: {s}\n", .{config.pool_name});
    print("   API: {s}:{d}\n", .{ args.api_host, args.api_port });
    print("   Control: {s}:{d}\n", .{ args.control_host, args.control_port });
    print("   Max Context Tokens: {d}\n", .{config.max_context_tokens});
    print("   Default Max Output Tokens: {d}\n", .{config.default_max_output_tokens});
    print("   Session TTL Seconds: {d}\n", .{config.session_ttl_seconds});
    print("   Request Timeout Seconds: {d}\n", .{config.request_timeout_seconds});
    print("   Worker Backend: {s}\n", .{config.worker_backend});
    print("   Worker Target Arch: {s}\n", .{config.worker_target_arch});
    print("   Tokenizer Source: {s}\n", .{config.tokenizer_source});
    if (config.tokenizer_path) |path| {
        print("   Tokenizer Path: {s}\n", .{path});
    }
    print("   Stop Token: EOS only\n", .{});
    controller.registerGatewayService(args.api_host, args.api_port) catch |err| {
        std.log.warn("Failed to register inference service with gateway: {}", .{err});
    };

    while (!controller.isShutdownRequested()) {
        std.time.sleep(1 * std.time.ns_per_s);
    }

    if (operator_state.isCancellationRequested()) {
        operator_state.setCancelled();
    }
}

fn inferenceListenThread(controller: *inference_shepherd.InferenceShepherd, host: []const u8, port: u16) !void {
    try controller.listen(host, port);
}

fn inferenceApiThread(controller: *inference_shepherd.InferenceShepherd, host: []const u8, port: u16) !void {
    try controller.startApi(host, port);
}

fn gatewayApiThread(server: *gateway_api.GatewayApiServer, host: []const u8, port: u16) !void {
    try server.start(host, port);
}

fn globalControllerApiThread(server: *global_controller_api.GlobalControllerApiServer, host: []const u8, port: u16) !void {
    try server.start(host, port);
}

fn controlApiThread(server: *control_api.ControlApiServer, host: []const u8, port: u16) !void {
    try server.start(host, port);
}

fn noOpCancel(_: *anyopaque) void {}

fn cancelShepherd(ctx: *anyopaque) void {
    const shepherd_controller: *Shepherd = @ptrCast(@alignCast(ctx));
    shepherd_controller.stop();
}

fn maybeLoadApiToken(allocator: Allocator, args: Args) !?[]u8 {
    const env_name = args.api_token_env orelse return null;
    return std.process.getEnvVarOwned(allocator, env_name) catch |err| {
        print("Error: missing required API token env var {s}: {}\n", .{ env_name, err });
        return error.MissingApiToken;
    };
}

fn maybeLoadApiTokenByEnv(allocator: Allocator, env_name: ?[]const u8) !?[]u8 {
    const name = env_name orelse return null;
    return std.process.getEnvVarOwned(allocator, name) catch |err| {
        print("Error: missing required API token env var {s}: {}\n", .{ name, err });
        return error.MissingApiToken;
    };
}

fn registerGatewayServiceIfConfigured(
    client: ?*gateway_service_client.GatewayClient,
    service_type: []const u8,
    host: []const u8,
    port: u16,
    capabilities: []const []const u8,
    worker_count: usize,
    ready_worker_count: usize,
    job_status: []const u8,
    health_status: []const u8,
) void {
    const gateway_client = client orelse return;
    const base_url = std.fmt.allocPrint(gateway_client.allocator, "http://{s}:{d}", .{ host, port }) catch |err| {
        std.log.warn("Failed to build gateway registration URL: {}", .{err});
        return;
    };
    defer gateway_client.allocator.free(base_url);

    gateway_client.registerService(
        service_type,
        base_url,
        capabilities,
        worker_count,
        ready_worker_count,
        job_status,
        health_status,
    ) catch |err| {
        std.log.warn("Gateway service registration failed: {}", .{err});
    };
}

fn runGateway(allocator: Allocator, args: Args) !void {
    const config_path = args.gateway_config_path orelse {
        print("Error: --gateway-config is required for gateway mode\n", .{});
        return error.ConfigFileRequired;
    };

    var config_result = try gateway_config.loadGatewayConfig(allocator, config_path);
    defer config_result.deinit();
    var config = config_result.config;

    if (args.api_token_env) |override_env| {
        if (config.api) |*api| {
            api.token_env = override_env;
        } else {
            config.api_token_env = override_env;
        }
    }

    const api_token = try maybeLoadApiTokenByEnv(allocator, config.resolvedApiTokenEnv());
    defer if (api_token) |token| allocator.free(token);
    const internal_api_token = try maybeLoadApiTokenByEnv(allocator, config.resolvedInternalApiTokenEnv());
    defer if (internal_api_token) |token| allocator.free(token);
    const federation_token = if (config.resolvedGlobalControllerEndpoint() != null)
        try maybeLoadApiTokenByEnv(allocator, config.resolvedFederationTokenEnv())
    else
        null;
    defer if (federation_token) |token| allocator.free(token);

    var gateway_instance = gateway.Gateway.init(allocator, config);
    defer gateway_instance.deinit();

    var api_server = gateway_api.GatewayApiServer.init(
        allocator,
        &gateway_instance,
        api_token,
        internal_api_token,
        federation_token,
    );

    var federation_client: ?gateway_federation_client.FederationClient = null;
    var federation_thread: ?std.Thread = null;
    if (config.resolvedGlobalControllerEndpoint()) |endpoint| {
        const gateway_base_url = try std.fmt.allocPrint(allocator, "http://{s}:{d}", .{ args.api_host, args.api_port });
        defer allocator.free(gateway_base_url);
        federation_client = try gateway_federation_client.FederationClient.init(
            allocator,
            &gateway_instance,
            endpoint,
            federation_token,
            gateway_base_url,
            config.resolvedHeartbeatIntervalMs(),
        );
        federation_thread = try std.Thread.spawn(.{}, federationClientThread, .{&federation_client.?});
    }
    defer {
        if (federation_thread) |thread| {
            federation_client.?.stop();
            thread.join();
            federation_client.?.deinit();
        }
    }

    print("🌉 Starting Gateway\n", .{});
    print("   Gateway ID: {s}\n", .{config.gateway_id});
    print("   Lab ID: {s}\n", .{config.lab_id});
    print("   Graph Backend: {s}\n", .{config.graph_backend});
    if (config.neo4j) |neo4j| {
        print("   Neo4j URI: {s}\n", .{neo4j.uri});
        print("   Neo4j User: {s}\n", .{neo4j.user});
        if (neo4j.database) |database| {
            print("   Neo4j Database: {s}\n", .{database});
        }
    }
    print("   API: {s}:{d}\n", .{ args.api_host, args.api_port });
    if (config.resolvedInternalApiTokenEnv()) |env_name| {
        print("   Internal Event Token Env: {s}\n", .{env_name});
    }
    if (config.resolvedGlobalControllerEndpoint()) |endpoint| {
        print("   Global Controller: {s}\n", .{endpoint});
    } else {
        print("   Global Controller: not configured\n", .{});
    }
    print("   Federation: handshake enabled\n", .{});

    try gatewayApiThread(&api_server, args.api_host, args.api_port);
}

fn federationClientThread(client: *gateway_federation_client.FederationClient) !void {
    try client.run();
}

fn runGlobalController(allocator: Allocator, args: Args) !void {
    var owned_api_token: ?[]u8 = null;
    defer if (owned_api_token) |token| allocator.free(token);
    owned_api_token = try maybeLoadApiToken(allocator, args);

    var controller = global_controller.GlobalController.init(allocator);
    defer controller.deinit();

    var api_server = global_controller_api.GlobalControllerApiServer.init(allocator, &controller, owned_api_token);

    print("🌐 Starting Global Controller\n", .{});
    print("   API: {s}:{d}\n", .{ args.api_host, args.api_port });

    try globalControllerApiThread(&api_server, args.api_host, args.api_port);
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
            print("Cannot resume: failed to open {s} to retrieve Run ID: {}\n", .{ LATEST_RUN_FILE, err });
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

    var operator_state = try control_state.ControllerState.init(
        allocator,
        .training,
        .training,
        args.config_path,
        run_id,
        exp_config.model_path,
        args.should_resume,
        args.workers,
    );
    defer operator_state.deinit();
    operator_state.setTrainingProgress(0, 0, exp_config.outer_loop_steps, 0.0, exp_config.learning_rate);
    operator_state.setStatus(.starting);

    var gateway_client = try gateway_service_client.GatewayClient.initFromEnv(allocator, "training-main");
    defer if (gateway_client) |*client| client.deinit();

    // Initialize system
    var system = try backend_selection.DistributedTrainingSystem.init(allocator, backend);
    defer system.deinit();

    // Get the shared MLIR context from the executor
    const shared_mlir_context = system.executor.getContext();

    // Create MLIRBuilder with the shared context
    var mlir_builder = try MLIRBuilder.init(allocator, shared_mlir_context);
    defer mlir_builder.deinit();

    // Now use system.shepherd instead of a local variable
    const shepherd_controller = &system.shepherd;

    var owned_api_token: ?[]u8 = null;
    defer if (owned_api_token) |token| allocator.free(token);

    var api_server = control_api.ControlApiServer.init(
        allocator,
        &operator_state,
        shepherd_controller,
        null,
        null,
        null,
        .{
            .ctx = shepherd_controller,
            .cancel = cancelShepherd,
        },
    );
    var api_thread: ?std.Thread = null;
    if (args.enable_api) {
        owned_api_token = try maybeLoadApiToken(allocator, args);
        api_server.api_token = owned_api_token;
        api_thread = try std.Thread.spawn(.{}, controlApiThread, .{ &api_server, args.api_host, args.api_port });
    }
    defer {
        if (api_thread) |thread| {
            api_server.stop();
            thread.join();
        }
    }

    registerGatewayServiceIfConfigured(
        if (gateway_client) |*client| client else null,
        "training",
        args.api_host,
        args.api_port,
        &[_][]const u8{ "controller.status", "job.current", "job.cancel" },
        0,
        0,
        "starting",
        "ok",
    );

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

    diloco_config.remat_config = .{
        .policy = exp_config.remat_policy,
        .activation_memory_budget_bytes = exp_config.activation_memory_budget_bytes,
        .cost_model = exp_config.remat_cost_model,
        .remat_allow_expensive_ops = exp_config.remat_allow_expensive_ops,
        .checkmate_backend = exp_config.checkmate_backend,
        .checkmate_milp_solver_command = exp_config.checkmate_milp_solver_command,
        .checkmate_milp_lp_path = exp_config.checkmate_milp_lp_path,
        .checkmate_milp_solution_path = exp_config.checkmate_milp_solution_path,
        .checkmate_milp_keep_artifacts = exp_config.checkmate_milp_keep_artifacts,
    };
    print("   Rematerialization Policy: {s}\n", .{@tagName(exp_config.remat_policy)});
    if (exp_config.activation_memory_budget_bytes) |budget| {
        print("   Activation Memory Budget: {} bytes\n", .{budget});
    }
    if (exp_config.remat_policy == .checkmate_optimal) {
        print("   Checkmate Backend: {s}\n", .{@tagName(exp_config.checkmate_backend)});
        if (exp_config.checkmate_milp_solver_command) |cmd| {
            print("   Checkmate MILP Solver Command: {s}\n", .{cmd});
        }
        if (exp_config.checkmate_milp_lp_path) |path| {
            print("   Checkmate LP Path: {s}\n", .{path});
        }
        if (exp_config.checkmate_milp_solution_path) |path| {
            print("   Checkmate Solution Path: {s}\n", .{path});
        }
    }

    // CLI flag overrides config file
    if (args.model_path) |path| {
        diloco_config.model_mlir_path = path;
        print("   Model: {s} (overridden by CLI)\n", .{path});
    } else {
        print("   Model: {s}\n", .{diloco_config.model_mlir_path});
    }

    var diloco_algo: DiLoCo = undefined;
    var streaming_algo: @import("algorithms/streaming_diloco.zig").StreamingDiLoCo = undefined;
    var training_algo: training_algorithm.TrainingAlgorithm = undefined;

    if (args.streaming_mode) {
        print("Initializing StreamingDiLoCo algorithm...\n", .{});
        const stream_config = @import("algorithms/streaming_diloco.zig").StreamingDiLoCoConfig{
            .num_fragments = 4,
            .inner_steps = exp_config.tau,
            .overlap_tau = 1,
            .alpha = 0.5,
            .learning_rate = exp_config.learning_rate,
            .nesterov_momentum = exp_config.nesterov_momentum,
            .max_epochs = exp_config.max_epochs,
            .model_mlir_path = exp_config.model_path,
            .data_path = exp_config.data_path,
            .tokenizer_type = exp_config.tokenizer,
            .sampling_type = exp_config.sampling,
            .dtype = diloco_config.dtype,
        };

        streaming_algo = try @import("algorithms/streaming_diloco.zig").StreamingDiLoCo.init(allocator, shepherd_controller, stream_config, &mlir_builder);
        training_algo = streaming_algo.asTrainingAlgorithm();
        print("🌊 StreamingDiLoCo initialized successfully\n", .{});
    } else {
        print("Initializing Standard DiLoCo algorithm...\n", .{});
        diloco_algo = try DiLoCo.init(allocator, shepherd_controller, diloco_config, system.executor, &mlir_builder);
        diloco_algo.setControlState(&operator_state);
        diloco_algo.setGatewayClient(if (gateway_client) |*client| client else null);
        training_algo = diloco_algo.asTrainingAlgorithm();
        print("🌙 DiLoCo algorithm initialized successfully\n", .{});
    }

    shepherd_controller.setAlgorithm(&training_algo);

    defer {
        if (args.streaming_mode) {
            streaming_algo.deinit();
        } else {
            diloco_algo.deinit();
        }
    }

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
    operator_state.setStatus(.waiting_for_workers);
    shepherd_controller.startTraining(args.workers) catch |err| {
        if (err == error.Cancelled) {
            operator_state.setCancelled();
            print("🛑 Training cancelled.\n", .{});
            return;
        }
        operator_state.setFailed(@errorName(err)) catch {};
        print("💣 Training failed with error: {}\n", .{err});
        return err;
    };

    print("🌑 Training completed successfully!\n", .{});
    if (operator_state.isCancellationRequested()) {
        operator_state.setCancelled();
    } else {
        operator_state.setStatus(.completed);
    }

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
        .inference => try runInferenceController(allocator, args),
        .gateway => try runGateway(allocator, args),
        .global_controller => try runGlobalController(allocator, args),
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
