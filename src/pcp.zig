/// Main entry point for distributed training
/// This file handles command-line arguments and launches either Shepherd or Worker
const std = @import("std");
const print = std.debug.print;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

// Import our distributed training components
const shepherd = @import("nodes/controllers/shepherd.zig");
const data_manager = @import("nodes/controllers/data_manager.zig");
const worker = @import("nodes/workers/worker.zig");
const diloco = @import("algorithms/diloco.zig");
const training_algorithm = @import("algorithms/training_algorithm.zig");
const backend_selection = @import("backends/selection.zig");
const ops = @import("core/ops.zig");
const mlir = @import("mlir/wrapper.zig");
const autodiff = @import("autodiff/engine.zig");
const dashboard = @import("ui/dashboard.zig");
const monitoring = @import("ui/monitoring.zig");
const adam_mlir = @import("optimizers/adam_mlir.zig");
const graph_builder = @import("compiler/graph_builder.zig");
const model_introspection = @import("mlir/model_introspection.zig");
const mlir_ctx = @import("mlir/context.zig");
const sanitizer = @import("compiler/sanitizer.zig");

const Shepherd = shepherd.Shepherd;
const Worker = worker.Worker;
const DiLoCo = diloco.DiLoCo;
const DiLoCoConfig = diloco.DiLoCoConfig;
const MLIRBuilder = ops.MLIRBuilder;

/// JSON Config File Structure
const ExperimentConfig = struct {
    model_path: []const u8 = "src/models/nanogpt_forward.mlir",
    data_path: []const u8 = "data/tiny_shakespeare.txt",
    tokenizer: []const u8 = "char",
    sampling: []const u8 = "random", // "random" (default) or "fifo" (cursor-based)
    chunk_manifest_path: ?[]const u8 = null,
    chunk_shuffle: bool = true,
    seed: u64 = 0,
    chunk_size_mode: []const u8 = "fixed_bytes", // "fixed_bytes" (default) or "diloco_slices"
    chunk_size_bytes: usize = 100 * 1024,
    learning_rate: f32 = 0.001,
    tau: usize = 10,
    outer_loop_steps: usize = 100,
    nesterov_momentum: f32 = 0.9,
    max_epochs: usize = 10,

    // WandB configuration
    wandb_project: []const u8 = "pcp-distributed",
    wandb_entity: ?[]const u8 = null,
    wandb_run_name: ?[]const u8 = null,
    wandb_api_key: ?[]const u8 = null,

    checkpoint_dir: []const u8 = "checkpoints",
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
    model_path: ?[]const u8,
    export_training_dir: ?[]const u8,
    backend: ?backend_selection.Backend,
    target_arch: ?[]const u8,
    supervisor_id: ?i64,
    supervise: bool,
    device_id: usize,
    scale: usize,
    should_resume: bool,
    no_dashboard: bool,
    child_args: std.ArrayList([]const u8),

    const Mode = enum {
        shepherd,
        worker,
        node_manager,
    };

    pub fn parse(allocator: Allocator, args: [][:0]u8) !Args {
        var child_args_list = std.ArrayList([]const u8).init(allocator);
        errdefer child_args_list.deinit();

        if (args.len < 2) {
            return Args{
                .mode = .shepherd,
                .host = "127.0.0.1",
                .port = 8080,
                .workers = 2,
                .config_path = null,
                .model_path = null,
                .export_training_dir = null,
                .backend = null,
                .target_arch = null,
                .supervisor_id = null,
                .supervise = false,
                .device_id = 0,
                .scale = 1,
                .should_resume = false,
                .no_dashboard = false,
                .child_args = child_args_list,
            };
        }

        var mode: Mode = .shepherd;
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 8080;
        var workers: usize = 2;
        var config_path: ?[]const u8 = null;
        var model_path: ?[]const u8 = null;
        var export_training_dir: ?[]const u8 = null;
        var backend: ?backend_selection.Backend = null;
        var target_arch: ?[]const u8 = null;
        var supervisor_id: ?i64 = null;
        var supervise: bool = false;
        var device_id: usize = 0;
        var scale: usize = 1;
        var should_resume: bool = false;
        var no_dashboard: bool = false;

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
            } else if (std.mem.eql(u8, args[i], "--export-training-artifacts")) {
                i += 1;
                if (i < args.len) {
                    export_training_dir = args[i];
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
            } else if (std.mem.eql(u8, args[i], "--no-dashboard")) {
                no_dashboard = true;
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
            .export_training_dir = export_training_dir,
            .backend = backend,
            .target_arch = target_arch,
            .supervisor_id = supervisor_id,
            .supervise = supervise,
            .device_id = device_id,
            .scale = scale,
            .should_resume = should_resume,
            .no_dashboard = no_dashboard,
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
        print("  --export-training-artifacts <dir>  Export training MLIR/VMFB + metadata and exit\n", .{});
        print("  --resume             Resume from previous training state\n", .{});
        print("  --no-dashboard       Disable the TUI dashboard (useful for non-interactive runs)\n", .{});
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
        errdefer allocator.free(data);
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

/// Export training MLIR/VMFB + metadata without running a training session.
fn exportTrainingArtifacts(allocator: Allocator, args: Args) !void {
    const out_dir = args.export_training_dir orelse return error.MissingExportPath;

    var config_result = try loadConfig(allocator, args.config_path);
    defer config_result.deinit();
    const exp_config = config_result.config;

    var diloco_config = DiLoCoConfig.default();
    diloco_config.model_mlir_path = exp_config.model_path;
    diloco_config.data_path = exp_config.data_path;
    diloco_config.tokenizer_type = exp_config.tokenizer;
    diloco_config.sampling_type = exp_config.sampling;
    diloco_config.tau = exp_config.tau;
    diloco_config.base_config.learning_rate = exp_config.learning_rate;
    diloco_config.base_config.outer_loop_steps = exp_config.outer_loop_steps;
    diloco_config.nesterov_momentum = exp_config.nesterov_momentum;

    if (args.model_path) |path| {
        diloco_config.model_mlir_path = path;
    }

    const backend = args.backend orelse backend_selection.Backend.selectDefault();
    const target_arch = args.target_arch;

    std.log.info("Exporting training artifacts to: {s}", .{out_dir});
    std.log.info("   Model: {s}", .{diloco_config.model_mlir_path});
    std.log.info("   Backend: {s}", .{backend.toString()});
    if (target_arch) |t| {
        std.log.info("   Target: {s}", .{t});
    }

    // Initialize system to get a shared MLIR context for graph construction.
    var system = try backend_selection.DistributedTrainingSystem.init(allocator, backend);
    defer system.deinit();

    const shared_mlir_context = system.executor.getContext();
    var mlir_builder = try MLIRBuilder.init(allocator, shared_mlir_context);
    defer mlir_builder.deinit();

    const raw_mlir = try std.fs.cwd().readFileAlloc(allocator, diloco_config.model_mlir_path, 10 * 1024 * 1024);
    defer allocator.free(raw_mlir);

    const sanitized_mlir = try sanitizer.ModelSanitizer.applyStabilityPatches(allocator, raw_mlir);
    defer allocator.free(sanitized_mlir);

    const metadata = try model_introspection.ModelInspector.inspect(
        allocator,
        mlir_builder.ctx,
        sanitized_mlir,
        2, // num_data_inputs (input_ids, targets)
    );
    const parameter_shapes = metadata.parameter_shapes;
    const data_input_shapes = metadata.data_input_shapes;
    defer {
        for (parameter_shapes) |s| allocator.free(s);
        allocator.free(parameter_shapes);
        for (data_input_shapes) |s| allocator.free(s);
        allocator.free(data_input_shapes);
    }

    const AdamMLIR = adam_mlir.AdamMLIR(f32);
    const adam_config = adam_mlir.AdamMLIRConfiguration(f32){
        .learning_rate = diloco_config.base_config.learning_rate,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .weight_decay = 0.01,
        .max_grad_norm = 1.0,
        .gradient_clip_min = -100.0,
        .gradient_clip_max = 100.0,
    };

    const element_type = diloco_config.dtype.toMLIRType(mlir_builder.ctx);
    const adam_optimizer = try allocator.create(AdamMLIR);
    defer {
        adam_optimizer.deinit();
        allocator.destroy(adam_optimizer);
    }
    adam_optimizer.* = try AdamMLIR.init(allocator, &mlir_builder, adam_config, element_type);

    const training_mlir = try graph_builder.GraphBuilder.buildTrainingGraph(
        allocator,
        &mlir_builder,
        sanitized_mlir,
        adam_optimizer,
        parameter_shapes.len,
    );
    defer allocator.free(training_mlir);

    try std.fs.cwd().makePath(out_dir);

    const training_mlir_path = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "training.mlir" });
    defer allocator.free(training_mlir_path);
    try std.fs.cwd().writeFile(.{ .sub_path = training_mlir_path, .data = training_mlir });

    var compile_ctx = try mlir_ctx.MLIRContext.init(allocator);
    defer compile_ctx.deinit();

    const vmfb = try compile_ctx.compileToVMFB(
        allocator,
        training_mlir,
        backend.toIreeCompilationTarget(),
        target_arch,
    );
    defer allocator.free(vmfb);

    const vmfb_path = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "training.vmfb" });
    defer allocator.free(vmfb_path);
    try std.fs.cwd().writeFile(.{ .sub_path = vmfb_path, .data = vmfb });

    const metadata_path = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "metadata.json" });
    defer allocator.free(metadata_path);
    const metadata_file = try std.fs.cwd().createFile(metadata_path, .{});
    defer metadata_file.close();

    var param_shape_array = std.json.Array.init(allocator);
    defer {
        for (param_shape_array.items) |item| {
            switch (item) {
                .array => |arr| {
                    var arr_mut = arr;
                    arr_mut.deinit();
                },
                else => {},
            }
        }
        param_shape_array.deinit();
    }
    for (parameter_shapes) |shape| {
        var dim_array = std.json.Array.init(allocator);
        errdefer dim_array.deinit();
        for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
        try param_shape_array.append(std.json.Value{ .array = dim_array });
    }

    var data_shape_array = std.json.Array.init(allocator);
    defer {
        for (data_shape_array.items) |item| {
            switch (item) {
                .array => |arr| {
                    var arr_mut = arr;
                    arr_mut.deinit();
                },
                else => {},
            }
        }
        data_shape_array.deinit();
    }
    for (data_input_shapes) |shape| {
        var dim_array = std.json.Array.init(allocator);
        errdefer dim_array.deinit();
        for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
        try data_shape_array.append(std.json.Value{ .array = dim_array });
    }

    var adam_obj = std.json.ObjectMap.init(allocator);
    defer adam_obj.deinit();
    try adam_obj.put("learning_rate", std.json.Value{ .float = @floatCast(adam_config.learning_rate) });
    try adam_obj.put("beta1", std.json.Value{ .float = @floatCast(adam_config.beta1) });
    try adam_obj.put("beta2", std.json.Value{ .float = @floatCast(adam_config.beta2) });
    try adam_obj.put("epsilon", std.json.Value{ .float = @floatCast(adam_config.epsilon) });
    try adam_obj.put("weight_decay", std.json.Value{ .float = @floatCast(adam_config.weight_decay) });
    try adam_obj.put("max_grad_norm", std.json.Value{ .float = @floatCast(adam_config.max_grad_norm) });
    try adam_obj.put("gradient_clip_min", std.json.Value{ .float = @floatCast(adam_config.gradient_clip_min) });
    try adam_obj.put("gradient_clip_max", std.json.Value{ .float = @floatCast(adam_config.gradient_clip_max) });

    var nesterov_obj = std.json.ObjectMap.init(allocator);
    defer nesterov_obj.deinit();
    try nesterov_obj.put("learning_rate", std.json.Value{ .float = @floatCast(diloco_config.base_config.learning_rate) });
    try nesterov_obj.put("momentum", std.json.Value{ .float = @floatCast(diloco_config.nesterov_momentum) });

    var root = std.json.ObjectMap.init(allocator);
    defer root.deinit();
    try root.put("num_params", std.json.Value{ .integer = @intCast(parameter_shapes.len) });
    try root.put("parameter_shapes", std.json.Value{ .array = param_shape_array });
    try root.put("data_input_shapes", std.json.Value{ .array = data_shape_array });
    try root.put("adam", std.json.Value{ .object = adam_obj });
    try root.put("nesterov", std.json.Value{ .object = nesterov_obj });
    try root.put("dtype", std.json.Value{ .string = "f32" });
    try root.put("main_function", std.json.Value{ .string = "main" });
    try root.put("grad_function", std.json.Value{ .string = "model_forward_pass_grad" });
    try root.put("timestep_start", std.json.Value{ .float = 1.0 });

    const writer = metadata_file.writer();
    try std.json.stringify(std.json.Value{ .object = root }, .{ .whitespace = .indent_2 }, writer);

    std.log.info("Export complete: {s}", .{out_dir});
}

fn loadChunkSpecsFromManifest(allocator: Allocator, manifest_path: []const u8, total_size: usize) ![]data_manager.ChunkSpec {
    const Manifest = struct {
        shards: []Shard,
        const Shard = struct {
            shard_index: ?usize = null,
            start_offset_bytes: ?usize = null,
            byte_offset: ?usize = null,
            bytes_written: ?usize = null,
            byte_length: ?usize = null,
        };
    };

    const manifest_bytes = try std.fs.cwd().readFileAlloc(allocator, manifest_path, 50 * 1024 * 1024);
    defer allocator.free(manifest_bytes);

    const parsed = try std.json.parseFromSlice(Manifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    if (parsed.value.shards.len == 0) return error.EmptyManifest;

    const specs = try allocator.alloc(data_manager.ChunkSpec, parsed.value.shards.len);
    errdefer allocator.free(specs);

    for (parsed.value.shards, 0..) |shard, i| {
        const id = shard.shard_index orelse i;
        const offset = shard.start_offset_bytes orelse shard.byte_offset orelse return error.InvalidManifest;
        const length = shard.bytes_written orelse shard.byte_length orelse return error.InvalidManifest;
        if (length == 0) return error.InvalidManifest;
        if (offset + length > total_size) return error.InvalidManifest;
        specs[i] = .{ .id = id, .offset = offset, .length = length };
    }

    return specs;
}

const BatchAndBlockSize = struct {
    batch_size: usize,
    block_size: usize,
};

fn inspectModelBatchAndBlockSize(allocator: Allocator, ctx: mlir.Context, model_path: []const u8) !BatchAndBlockSize {
    const raw_mlir = try std.fs.cwd().readFileAlloc(allocator, model_path, 50 * 1024 * 1024);
    defer allocator.free(raw_mlir);

    const sanitized_mlir = try sanitizer.ModelSanitizer.applyStabilityPatches(allocator, raw_mlir);
    defer allocator.free(sanitized_mlir);

    var metadata = try model_introspection.ModelInspector.inspect(
        allocator,
        ctx,
        sanitized_mlir,
        2, // num_data_inputs (input_ids, targets)
    );
    defer metadata.deinit();

    if (metadata.data_input_shapes.len == 0) return error.NotEnoughInputsInModel;
    const shape0 = metadata.data_input_shapes[0];
    if (shape0.len != 2) return error.UnsupportedDataInputShape;

    const batch_dim = shape0[0];
    const block_dim = shape0[1];
    if (batch_dim <= 0 or block_dim <= 0) return error.InvalidDataInputShape;

    const batch_size: usize = @intCast(batch_dim);
    const block_size: usize = @intCast(block_dim);

    // Ensure all data inputs agree (e.g., input_ids and targets are both [B,T]).
    for (metadata.data_input_shapes[1..]) |shape| {
        if (shape.len != shape0.len) return error.UnsupportedDataInputShape;
        if (shape[0] != batch_dim or shape[1] != block_dim) return error.MismatchedDataInputShapes;
    }

    return .{ .batch_size = batch_size, .block_size = block_size };
}

/// Run as Shepherd coordinator
fn runShepherd(allocator: Allocator, args: Args) !void {
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

    const backend = args.backend orelse backend_selection.Backend.selectDefault();
    print("   Backend: {s}\n", .{backend.toString()});

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
    defer shepherd_controller.stop();

    // 2. Initialize DataManager for chunk-based data partitioning
    print("Initializing DataManager for dataset: {s}...\n", .{exp_config.data_path});
    const file = try std.fs.cwd().openFile(exp_config.data_path, .{});
    defer file.close();
    const stat = try file.stat();
    const total_size: usize = @intCast(stat.size);
    const seed = if (exp_config.seed != 0) exp_config.seed else @as(u64, @intCast(std.time.timestamp()));

    if (std.mem.eql(u8, exp_config.sampling, "fifo") and !std.mem.eql(u8, exp_config.chunk_size_mode, "diloco_slices")) {
        std.log.warn(
            "sampling=\"fifo\" is designed to be used with chunk_size_mode=\"diloco_slices\" (or a manifest that already splits into tau*(B*T+1) token slices); otherwise tokens may be skipped within each chunk.",
            .{},
        );
    }

    if (std.mem.eql(u8, exp_config.chunk_size_mode, "diloco_slices")) {
        if (exp_config.chunk_manifest_path != null) {
            print("Error: chunk_manifest_path is not supported with chunk_size_mode=\"diloco_slices\".\n", .{});
            return error.InvalidConfig;
        }
        if (!std.mem.eql(u8, exp_config.tokenizer, "u16")) {
            print("Error: chunk_size_mode=\"diloco_slices\" requires tokenizer=\"u16\".\n", .{});
            return error.InvalidConfig;
        }
        if (!std.mem.eql(u8, exp_config.sampling, "fifo")) {
            print("Error: chunk_size_mode=\"diloco_slices\" requires sampling=\"fifo\".\n", .{});
            return error.InvalidConfig;
        }

        // Auto-split the token stream into fixed slices sized to exactly the per-worker inner-loop
        // consumption for one outer round (nanochat-style FIFO): slice_tokens = tau * (B*T + 1).
        const batch_block = try inspectModelBatchAndBlockSize(allocator, mlir_builder.ctx, exp_config.model_path);
        const B_u64: u64 = @intCast(batch_block.batch_size);
        const T_u64: u64 = @intCast(batch_block.block_size);
        const tau_u64: u64 = @intCast(exp_config.tau);

        const bt_mul = @mulWithOverflow(B_u64, T_u64);
        if (bt_mul[1] != 0) return error.InvalidConfig;
        const bt = bt_mul[0];

        const needed_add = @addWithOverflow(bt, 1);
        if (needed_add[1] != 0) return error.InvalidConfig;
        const needed_tokens_per_step = needed_add[0];

        const slice_mul = @mulWithOverflow(tau_u64, needed_tokens_per_step);
        if (slice_mul[1] != 0) return error.InvalidConfig;
        const slice_tokens = slice_mul[0];

        const bytes_mul = @mulWithOverflow(slice_tokens, 2); // u16 = 2 bytes/token
        if (bytes_mul[1] != 0) return error.InvalidConfig;
        const slice_bytes_u64 = bytes_mul[0];

        if (slice_bytes_u64 == 0 or slice_bytes_u64 > std.math.maxInt(usize)) {
            print("Error: computed slice_bytes is invalid: {}\n", .{slice_bytes_u64});
            return error.InvalidConfig;
        }
        const slice_bytes: usize = @intCast(slice_bytes_u64);

        if (total_size < slice_bytes) {
            print("Error: dataset too small ({}) for one slice ({} bytes). Reduce B/T/tau or use a larger dataset.\n", .{ total_size, slice_bytes });
            return error.DatasetTooSmall;
        }

        const full_chunks_u64: u64 = @intCast(@as(u64, total_size) / slice_bytes_u64);
        if (full_chunks_u64 == 0) return error.DatasetTooSmall;
        if (full_chunks_u64 > std.math.maxInt(usize)) return error.DatasetTooLarge;
        const full_chunks: usize = @intCast(full_chunks_u64);

        const remainder: usize = @intCast(@as(u64, total_size) % slice_bytes_u64);

        const chunk_specs = try allocator.alloc(data_manager.ChunkSpec, full_chunks);
        defer allocator.free(chunk_specs);
        for (0..full_chunks) |i| {
            chunk_specs[i] = .{
                .id = i,
                .offset = i * slice_bytes,
                .length = slice_bytes,
            };
        }

        try shepherd_controller.initDataManagerFromChunkSpecs(total_size, chunk_specs, exp_config.max_epochs, exp_config.chunk_shuffle, seed);
        print("🌙 DataManager initialized: {} total size, {} slice chunks ({} bytes each), remainder {} bytes\n", .{
            total_size,
            full_chunks,
            slice_bytes,
            remainder,
        });
    } else if (exp_config.chunk_manifest_path) |manifest_path| {
        const chunk_specs = try loadChunkSpecsFromManifest(allocator, manifest_path, total_size);
        defer allocator.free(chunk_specs);

        try shepherd_controller.initDataManagerFromChunkSpecs(total_size, chunk_specs, exp_config.max_epochs, exp_config.chunk_shuffle, seed);
        print("🌙 DataManager initialized: {} total size, {} manifest chunks, {} max epochs\n", .{ total_size, chunk_specs.len, exp_config.max_epochs });
    } else {
        const chunk_size = exp_config.chunk_size_bytes;
        try shepherd_controller.initDataManagerFixed(total_size, chunk_size, exp_config.max_epochs, exp_config.chunk_shuffle, seed);
        print("🌙 DataManager initialized: {} total size, {} byte chunks, {} max epochs\n", .{ total_size, chunk_size, exp_config.max_epochs });
    }

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
    diloco_config.wandb_project = exp_config.wandb_project;
    diloco_config.wandb_entity = exp_config.wandb_entity;
    diloco_config.wandb_run_name = exp_config.wandb_run_name;
    diloco_config.wandb_api_key = exp_config.wandb_api_key;
    diloco_config.checkpoint_dir = run_dir_path;
    diloco_config.resume_training = args.should_resume;

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
    var dashboard_thread: ?std.Thread = null;
    const want_dashboard = !args.no_dashboard and std.io.getStdOut().isTty() and std.io.getStdErr().isTty();
    if (want_dashboard) {
        dashboard_thread = try std.Thread.spawn(.{}, dashboard.runDashboard, .{});
    }
    defer if (dashboard_thread) |t| t.join();

    // Wait for workers and start training
    print("Waiting for workers to connect...\n", .{});
    shepherd_controller.startTraining(args.workers) catch |err| {
        monitoring.setStatus(.error_state);
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

    var args = try Args.parse(allocator, process_args);
    defer args.child_args.deinit();

    print("🪐 PCP Distributed Training System\n", .{});
    print("=====================================\n", .{});

    // Debug: Print parsed arguments
    print("Mode: {s}\n", .{@tagName(args.mode)});
    print("Host: {s}\n", .{args.host});
    print("Port: {}\n", .{args.port});
    print("Workers: {}\n", .{args.workers});
    print("=====================================\n", .{});

    if (args.export_training_dir != null) {
        try exportTrainingArtifacts(allocator, args);
        return;
    }

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
