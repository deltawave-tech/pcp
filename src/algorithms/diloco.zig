/// DiLoCo (Distributed Low-Communication) Algorithm Implementation
/// Implements the DiLoCo training algorithm for distributed learning with real MLIR optimizers
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const training_algorithm = @import("training_algorithm.zig");
const shepherd = @import("../nodes/controllers/shepherd.zig");
const message = @import("../network/message.zig");
const binary_protocol = @import("../network/capnp_zig_wrapper.zig");
const nesterov_mlir = @import("../optimizers/nesterov_mlir.zig");
const adam_mlir = @import("../optimizers/adam_mlir.zig");
const nesterov_host = @import("../optimizers/nesterov.zig");
const NesterovHost = nesterov_host.Nesterov;
const AdamMLIR = adam_mlir.AdamMLIR(f32);
const autodiff = @import("../autodiff/engine.zig");
const ops = @import("../core/ops.zig");
const mlir = @import("../mlir/wrapper.zig");
const model_introspection = @import("../mlir/model_introspection.zig");
const GraphBuilder = @import("../compiler/graph_builder.zig").GraphBuilder;
const tensor = @import("../core/tensor.zig");
const execution = @import("../execution.zig");
const monitoring = @import("../ui/monitoring.zig");
const data_loader = @import("../data/loader.zig");
const backend_selection = @import("../backends/selection.zig");
const wandb = @import("../ui/wandb.zig");

const TrainingAlgorithm = training_algorithm.TrainingAlgorithm;
const TrainingStatus = training_algorithm.TrainingStatus;
const TrainingConfig = training_algorithm.TrainingConfig;
const TrainingMetrics = training_algorithm.TrainingMetrics;
const Shepherd = shepherd.Shepherd;
const WorkerConfig = shepherd.WorkerConfig;
const MessageType = message.MessageType;
const NesterovMLIR = nesterov_mlir.NesterovMLIR(f32);
const MLIRBuilder = ops.MLIRBuilder;
const Tensor = tensor.Tensor(void);
const Executor = execution.Executor; // NEW: Generic executor interface
const DataLoader = data_loader.DataLoader;
const ModelSanitizer = @import("../compiler/sanitizer.zig").ModelSanitizer;

const RECOVERY_FILENAME = "shepherd_state.bin";
const DEFAULT_INLINE_BLOB_MAX_BYTES: usize = 512 * 1024 * 1024; // 512MiB

fn resolveInlineBlobMaxBytes() usize {
    if (std.posix.getenv("PCP_BLOB_INLINE_MAX_BYTES")) |raw_z| {
        const raw: []const u8 = std.mem.sliceTo(raw_z, 0);
        if (raw.len > 0) {
            const parsed = std.fmt.parseInt(usize, raw, 10) catch |err| {
                std.log.warn(
                    "Invalid PCP_BLOB_INLINE_MAX_BYTES={s} ({s}); using default {}",
                    .{ raw, @errorName(err), DEFAULT_INLINE_BLOB_MAX_BYTES },
                );
                return DEFAULT_INLINE_BLOB_MAX_BYTES;
            };
            if (parsed > 0) return parsed;
            std.log.warn(
                "Invalid PCP_BLOB_INLINE_MAX_BYTES={s} (must be > 0); using default {}",
                .{ raw, DEFAULT_INLINE_BLOB_MAX_BYTES },
            );
        }
    }
    return DEFAULT_INLINE_BLOB_MAX_BYTES;
}

fn shouldUseFileBlobs(total_param_bytes: usize) bool {
    if (std.posix.getenv("PCP_BLOB_TRANSPORT")) |mode_z| {
        const mode: []const u8 = std.mem.sliceTo(mode_z, 0);
        if (std.mem.eql(u8, mode, "file")) return true;
        if (std.mem.eql(u8, mode, "inline")) return false;
        if (mode.len != 0) {
            std.log.warn("Unknown PCP_BLOB_TRANSPORT={s}; expected \"file\" or \"inline\"", .{mode});
        }
    }
    return total_param_bytes > resolveInlineBlobMaxBytes();
}

fn totalParamBytes(parameter_shapes: [][]i64) !usize {
    var total: usize = 0;
    for (parameter_shapes) |shape| {
        var size: usize = @sizeOf(f32);
        for (shape) |dim| {
            if (dim <= 0) return error.InvalidParameterShape;
            const mul = @mulWithOverflow(size, @as(usize, @intCast(dim)));
            if (mul[1] != 0) return error.ParameterSizeOverflow;
            size = mul[0];
        }
        const add = @addWithOverflow(total, size);
        if (add[1] != 0) return error.ParameterSizeOverflow;
        total = add[0];
    }
    return total;
}

/// DiLoCo algorithm configuration
pub const DiLoCoConfig = struct {
    base_config: TrainingConfig,
    tau: usize,
    nesterov_momentum: f32,
    parameter_averaging: bool,
    model_mlir_path: []const u8,
    data_path: []const u8,
    tokenizer_type: []const u8,
    sampling_type: []const u8,

    wandb_project: []const u8,
    wandb_entity: ?[]const u8,
    wandb_run_name: ?[]const u8,
    wandb_api_key: ?[]const u8,

    checkpoint_dir: []const u8,
    resume_training: bool,
    dtype: tensor.DType = .f32,

    pub fn default() DiLoCoConfig {
        return DiLoCoConfig{
            .base_config = TrainingConfig.default(),
            .tau = 10,
            .nesterov_momentum = 0.9,
            .parameter_averaging = true,
            .model_mlir_path = "src/models/nanogpt_forward.mlir",
            .data_path = "data/tiny_shakespeare.txt",
            .tokenizer_type = "char",
            .sampling_type = "random",
            .wandb_project = "pcp-distributed",
            .wandb_entity = null,
            .wandb_run_name = null,
            .wandb_api_key = null,
            .checkpoint_dir = "checkpoints",
            .resume_training = false,
        };
    }
};

/// DiLoCo algorithm implementation with real MLIR optimizers
/// Now backend-agnostic using the generic Executor interface
pub const DiLoCo = struct {
    allocator: Allocator,
    coordinator: *Shepherd,
    config: DiLoCoConfig,
    status: TrainingStatus,
    metrics: TrainingMetrics,
    current_epoch: usize,

    // MLIR infrastructure
    mlir_builder: *MLIRBuilder, // Now a pointer, as it's owned externally
    adam_optimizer: *AdamMLIR, // Inner optimizer for worker training graph (AdamW)
    host_optimizer: NesterovHost, // Outer optimizer for master parameter updates (Nesterov)
    element_type: mlir.Type,

    // GPT-2 master parameters as multiple MLIR tensors
    master_parameters: ?[]Tensor,
    parameter_shapes: [][]i64, // Multiple parameter shapes for GPT-2
    
    // NEW: Data input shapes for worker buffer creation
    data_input_shapes: [][]i64, // Shapes for input_ids, targets, etc.

    // NEW: Generic executor - replaces direct backend knowledge
    executor: Executor,

    // NEW: MLIR source for on-demand compilation
    worker_graph_mlir_source: []u8,

    // Data loading for real training batches
    data_loader: *DataLoader,

    // WandB logging
    wandb_logger: wandb.WandBLogger,

    const Self = @This();

    // Change the init signature to accept the generic executor (dataset now loaded by workers)
    pub fn init(allocator: Allocator, coordinator: *Shepherd, config: DiLoCoConfig, executor: Executor, mlir_builder: *MLIRBuilder) !Self {
        const element_type = config.dtype.toMLIRType(mlir_builder.ctx);

        const recovery_path = try std.fs.path.join(allocator, &[_][]const u8{ config.checkpoint_dir, "recovery", RECOVERY_FILENAME });
        defer allocator.free(recovery_path);

        if (!config.resume_training) {
            std.fs.cwd().deleteFile(recovery_path) catch |err| {
                if (err != error.FileNotFound) {
                    std.log.warn("Could not delete old recovery file: {}", .{err});
                }
            };
            std.log.info("Fresh start: Cleared previous recovery state.", .{});
        } else {
            std.log.info("Resume mode: Will attempt to load recovery state from {s}", .{recovery_path});
        }

        std.debug.print("Loading model from: {s}\n", .{config.model_mlir_path});
        const raw_mlir = try std.fs.cwd().readFileAlloc(allocator, config.model_mlir_path, 10 * 1024 * 1024);
        defer allocator.free(raw_mlir);

        std.debug.print("Applying ModelSanitizer patches...\n", .{});
        const mlir_source = try ModelSanitizer.applyStabilityPatches(allocator, raw_mlir);
        defer allocator.free(mlir_source);

        // Introspect using the new ModelInspector
        const metadata = try model_introspection.ModelInspector.inspect(
            allocator,
            mlir_builder.ctx,
            mlir_source,
            2, // num_data_inputs (input_ids, targets)
        );
        // Note: DiLoCo struct takes ownership of these slices
        const parameter_shapes = metadata.parameter_shapes;
        const data_input_shapes = metadata.data_input_shapes;
        // Do NOT call metadata.deinit() here because we are stealing the pointers.

        std.debug.print("✓ Introspection complete. Found {} trainable parameter tensors.\n", .{parameter_shapes.len});
        std.debug.print("✓ Found {} data input shapes.\n", .{data_input_shapes.len});

        // Add errdefer to cleanup if subsequent allocations fail
        errdefer {
            for (parameter_shapes) |s| allocator.free(s);
            allocator.free(parameter_shapes);
            for (data_input_shapes) |s| allocator.free(s);
            allocator.free(data_input_shapes);
        }

        // Configure and create AdamW optimizer for INNER loop (workers)
        // Per DiLoCo paper: "the inner optimizer is AdamW"
        const adam_config = adam_mlir.AdamMLIRConfiguration(f32){
            .learning_rate = config.base_config.learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .weight_decay = 0.01, // AdamW weight decay
        };

        const adam_optimizer = try allocator.create(AdamMLIR);
        adam_optimizer.* = try AdamMLIR.init(allocator, mlir_builder, adam_config, element_type);

        // Initialize Host Optimizer for OUTER loop (master updates on CPU)
        // Per DiLoCo paper: "the outer optimizer is Nesterov momentum"
        var host_opt = NesterovHost.init(allocator, .{
            .learning_rate = config.base_config.learning_rate,
            .momentum = config.nesterov_momentum,
        });

        // Pre-allocate velocity buffers based on introspection
        try host_opt.initParameters(parameter_shapes);

        // Initialize WandB logger
        const wb_config = wandb.WandBConfig{
            .project = config.wandb_project,
            .entity = config.wandb_entity,
            .run_name = config.wandb_run_name,
            .api_key = config.wandb_api_key orelse std.posix.getenv("WANDB_API_KEY"),
        };

        const logger = try wandb.WandBLogger.init(allocator, wb_config, .{
            .learning_rate = config.base_config.learning_rate,
            .tau = config.tau,
            .nesterov_momentum = config.nesterov_momentum,
            .optimizer = "Nesterov-AdamW",
        });

        // Build the worker graph ONCE during initialization using the sanitized source
        std.debug.print("Building worker training graph...\n", .{});
        const graph = try GraphBuilder.buildTrainingGraph(
            allocator,
            mlir_builder,
            mlir_source,
            adam_optimizer,
            parameter_shapes.len,
        );

        return Self{
            .allocator = allocator,
            .coordinator = coordinator,
            .config = config,
            .status = .not_started,
            .metrics = TrainingMetrics.init(),
            .current_epoch = 0,
            .mlir_builder = mlir_builder,
            .adam_optimizer = adam_optimizer,
            .host_optimizer = host_opt,
            .element_type = element_type,
            .master_parameters = null,
            .parameter_shapes = parameter_shapes,
            .data_input_shapes = data_input_shapes,
            .executor = executor, // Store the generic executor
            .worker_graph_mlir_source = graph, // Store the MLIR source
            .data_loader = undefined, // No longer used - workers load data locally
            .wandb_logger = logger,
        };
    }

    pub fn deinit(self: *Self) void {
        // Deinit WandB logger first
        self.wandb_logger.deinit();

        if (self.master_parameters) |params| {
            for (params) |param| {
                param.deinit();
            }
            self.allocator.free(params);
        }

        self.adam_optimizer.deinit();
        self.allocator.destroy(self.adam_optimizer);

        // Deinit host optimizer
        self.host_optimizer.deinit();

        // We no longer deinit the builder, as we don't own it.

        // Free the stored MLIR source
        self.allocator.free(self.worker_graph_mlir_source);

        // Free parameter shapes
        for (self.parameter_shapes) |shape| {
            self.allocator.free(shape);
        }
        self.allocator.free(self.parameter_shapes);

        // Free data input shapes
        for (self.data_input_shapes) |shape| {
            self.allocator.free(shape);
        }
        self.allocator.free(self.data_input_shapes);
    }

    /// Get TrainingAlgorithm interface
    pub fn asTrainingAlgorithm(self: *Self) TrainingAlgorithm {
        return TrainingAlgorithm{
            .ptr = self,
            .vtable = &.{
                .run = run,
                .deinit = deinitInterface,
                .getName = getName,
                .getStatus = getStatus,
            },
        };
    }

    /// Main DiLoCo training loop
    pub fn run(ptr: *anyopaque) !void {
        const self: *Self = @ptrCast(@alignCast(ptr));

        self.status = .initializing;
        monitoring.setStatus(.initializing);

        // Calculate total parameter count from introspected shapes
        var total_param_count: usize = 0;
        for (self.parameter_shapes) |shape| {
            var element_count: usize = 1;
            for (shape) |dim| {
                element_count *= @intCast(dim);
            }
            total_param_count += element_count;
        }
        monitoring.setModelInfo(total_param_count, self.config.base_config.learning_rate);

        // Try to recover state before starting
        const recovered = try self.tryLoadRecoveryState();
        if (!recovered) {
            // Cold start - initialize master parameters
            try self.initializeMasterParameters();
        }

        // --- NEW LOGIC: Delegate compilation/distribution to Shepherd ---
        // Ensure initial workers are ready before starting
        try self.coordinator.ensureWorkersCompiled(
            self.worker_graph_mlir_source,
            self.parameter_shapes,
            self.data_input_shapes,
        );
        // --- END NEW LOGIC ---

        self.status = .running;
        monitoring.setStatus(.running);

        // Main outer loop with snapshot-based dynamic topology
        for (0..self.config.base_config.outer_loop_steps) |step| {
            const start_time = std.time.milliTimestamp();

            // Ensure any NEW workers that joined mid-training are compiled & initialized
            try self.coordinator.ensureWorkersCompiled(
                self.worker_graph_mlir_source,
                self.parameter_shapes,
                self.data_input_shapes,
            );

            // Take snapshot of currently connected workers
            const participants = try self.coordinator.snapshotWorkers();
            defer participants.deinit();

            if (participants.items.len == 0) {
                std.log.warn("No workers connected. Waiting...", .{});
                std.time.sleep(1 * std.time.ns_per_s);
                continue;
            }

            // Broadcast to snapshot participants only (workers will load data locally)
            const workers_assigned = try self.broadcastToSnapshot(participants.items);

            if (workers_assigned == 0) {
                std.log.warn("Round {} failed: No workers assigned. Skipping update.", .{step});
                continue;
            }

            // Streaming Accumulation: Collect and apply gradients on the fly
            self.accumulateAndApplyGradients(workers_assigned) catch |err| {
                if (err == error.NonFiniteLoss) {
                    self.status = .failed;
                    monitoring.setStatus(.error_state);
                    std.log.err("Non-finite loss detected; aborting training", .{});
                    break;
                }
                return err;
            };

            // Save recovery state immediately after update
            try self.saveRecoveryState();

            // Save checkpoint every 10 steps or at the end
            if ((step + 1) % 10 == 0 or (step + 1) == self.config.base_config.outer_loop_steps) {
                try self.saveCheckpoint(step + 1);
            }

            // Update metrics
            self.metrics.outer_loop_count += 1;

            // Get true epoch from DataManager
            if (self.coordinator.data_manager) |*dm| {
                self.current_epoch = dm.current_epoch;
            } else {
                // Fallback if no DM
                self.current_epoch = 1;
            }

            // Calculate epoch time and update monitoring
            const end_time = std.time.milliTimestamp();
            const epoch_time_ms: u64 = @intCast(end_time - start_time);
            monitoring.setEpochTime(epoch_time_ms);
            monitoring.setMetrics(self.current_epoch, self.metrics.loss, self.coordinator.getWorkerCount());

            // Log to WandB
            self.wandb_logger.log(.{
                .outer_step = step,
                .epoch = self.current_epoch,
                .loss = self.metrics.loss,
                .min_worker_loss = self.metrics.loss, // Note: Individual worker losses not tracked in streaming mode
                .max_worker_loss = self.metrics.loss, // Note: Individual worker losses not tracked in streaming mode
                .active_workers = workers_assigned,
                .learning_rate = self.host_optimizer.config.learning_rate,
                .epoch_time_ms = epoch_time_ms,
            });

            // Check convergence or stopping conditions
            if (self.shouldStop()) {
                break;
            }
        }

        if (self.status != .failed) {
            self.status = .completed;
            monitoring.setStatus(.completed);
        }
    }

    /// Initialize master parameters using MLIR for the loaded model
    fn initializeMasterParameters(self: *Self) !void {

        // Allocate array for master parameter tensors
        const param_tensors = try self.allocator.alloc(Tensor, self.parameter_shapes.len);
        var rng = std.Random.DefaultPrng.init(12345);

        // Initialize each parameter tensor with appropriate shape
        for (param_tensors, 0..) |*param_tensor, i| {
            const shape = self.parameter_shapes[i];

            // Calculate element count for this tensor
            var element_count: usize = 1;
            for (shape) |dim| {
                element_count *= @intCast(dim);
            }

            // Create parameter data with Xavier initialization
            const param_data = try self.allocator.alloc(f32, element_count);
            defer self.allocator.free(param_data);

            const scale = std.math.sqrt(2.0 / @as(f32, @floatFromInt(element_count)));
            for (param_data) |*param| {
                param.* = rng.random().floatNorm(f32) * scale;
            }

            const byte_data = std.mem.sliceAsBytes(param_data);
            param_tensor.* = try Tensor.fromBytes(self.mlir_builder, byte_data, shape, self.config.dtype);
        }

        self.master_parameters = param_tensors;
    }

    /// Extract raw byte data from a tensor defined by a stablehlo.constant operation.

    /// Save current master parameters to a binary file
    fn saveCheckpoint(self: *Self, step: usize) !void {
        var filename_buf: [64]u8 = undefined;
        const filename = try std.fmt.bufPrint(&filename_buf, "checkpoint_{d}.bin", .{step});

        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ self.config.checkpoint_dir, filename });
        defer self.allocator.free(full_path);

        std.log.info("Saving checkpoint to {s}...", .{full_path});
        const file = try std.fs.cwd().createFile(full_path, .{});
        defer file.close();
        var writer = file.writer();

        // 1. Header
        try writer.writeAll("PCPCHECK");
        try writer.writeInt(u32, 1, .little); // Version

        if (self.master_parameters) |params| {
            // 2. Number of tensors
            try writer.writeInt(u32, @intCast(params.len), .little);

            for (params) |t| {
                // 3a. Write Shape info
                const rank = t.shape.rank();
                try writer.writeInt(u8, @intCast(rank), .little);

                const dims = try t.shape.getDims(self.allocator);
                defer self.allocator.free(dims);
                for (dims) |d| {
                    try writer.writeInt(i64, d, .little);
                }

                // 3b. Write Data
                const raw_bytes = try t.toBytes(self.allocator);
                defer self.allocator.free(raw_bytes);

                // Write size of data block followed by data
                try writer.writeInt(u64, @intCast(raw_bytes.len), .little);
                try writer.writeAll(raw_bytes);
            }
        }
        std.log.info("✓ Checkpoint saved.", .{});
    }


    /// Broadcast to snapshot participants only
    /// Returns the number of workers that were actually assigned chunks
    fn broadcastToSnapshot(self: *Self, workers: []const shepherd.WorkerConnection) !usize {
        if (self.master_parameters == null) {
            return error.ParametersNotInitialized;
        }

        const param_tensors = self.master_parameters.?;

        const total_param_bytes = try totalParamBytes(self.parameter_shapes);
        const use_file_blobs = shouldUseFileBlobs(total_param_bytes);

        var blob_dir: ?[]const u8 = null;
        defer if (blob_dir) |p| self.allocator.free(p);
        var params_path: ?[]const u8 = null;
        defer if (params_path) |p| self.allocator.free(p);

        var b64_encoded_params: ?[]u8 = null;
        defer if (b64_encoded_params) |buf| self.allocator.free(buf);
        var encoded_len: usize = 0;

        if (use_file_blobs) {
            const dir = try std.fs.path.join(self.allocator, &[_][]const u8{ self.config.checkpoint_dir, "blobs" });
            blob_dir = dir;
            std.fs.cwd().makePath(dir) catch {};

            const p = try std.fs.path.join(self.allocator, &[_][]const u8{ dir, "params_f32.bin" });
            params_path = p;

            const file = try std.fs.cwd().createFile(p, .{ .truncate = true });
            defer file.close();
            var writer = file.writer();
            for (param_tensors) |param_tensor| {
                const tensor_data = try param_tensor.toBytes(self.allocator);
                defer self.allocator.free(tensor_data);
                try writer.writeAll(tensor_data);
            }
            file.sync() catch {};
            std.log.info("Params blob written: {s} ({} bytes)", .{ p, total_param_bytes });
        } else {
            // Inline transport: serialize params via Cap'n Proto and Base64 for JSON transport.
            var total_param_bytes_list = ArrayList(u8).init(self.allocator);
            defer total_param_bytes_list.deinit();
            try total_param_bytes_list.ensureTotalCapacity(total_param_bytes);

            for (param_tensors) |param_tensor| {
                const tensor_data = try param_tensor.toBytes(self.allocator);
                defer self.allocator.free(tensor_data);
                try total_param_bytes_list.appendSlice(tensor_data);
            }

            const empty_slice: []const u8 = &[_]u8{};
            const worker_payload = binary_protocol.WorkerPayload{
                .params = total_param_bytes_list.items,
                .input_ids = empty_slice,
                .targets = empty_slice,
            };
            const capnp_bytes = try worker_payload.serialize(self.allocator);
            defer self.allocator.free(capnp_bytes);

            const b64_len = std.base64.standard.Encoder.calcSize(capnp_bytes.len);
            const buf = try self.allocator.alloc(u8, b64_len);
            b64_encoded_params = buf;
            encoded_len = std.base64.standard.Encoder.encode(buf, capnp_bytes).len;
        }

        // Send to each worker with their assigned data chunk
        var workers_assigned: usize = 0;
        for (workers) |worker| {
            // Assign data chunk for this worker
            const chunk = if (self.coordinator.data_manager) |*dm|
                dm.assignNextChunk(worker.node_id)
            else
                null;

            if (chunk == null) {
                std.log.warn("No data chunk available for worker {}", .{worker.node_id});
                continue;
            }

            // Build JSON payload with params and chunk info
            var payload_map = std.json.ObjectMap.init(self.allocator);
            defer payload_map.deinit();

            if (use_file_blobs) {
                try payload_map.put("params_path", std.json.Value{ .string = params_path.? });
                try payload_map.put("params_bytes", std.json.Value{ .integer = @intCast(total_param_bytes) });
                try payload_map.put("blob_dir", std.json.Value{ .string = blob_dir.? });
            } else {
                try payload_map.put("params", std.json.Value{ .string = b64_encoded_params.?[0..encoded_len] });
            }
            try payload_map.put("offset", std.json.Value{ .integer = @intCast(chunk.?.offset) });
            try payload_map.put("length", std.json.Value{ .integer = @intCast(chunk.?.length) });
            try payload_map.put("chunk_id", std.json.Value{ .integer = @intCast(chunk.?.id) });
            try payload_map.put("data_path", std.json.Value{ .string = self.config.data_path });
            try payload_map.put("tau", std.json.Value{ .integer = @intCast(self.config.tau) });
            try payload_map.put("tokenizer", std.json.Value{ .string = self.config.tokenizer_type });
            try payload_map.put("sampling", std.json.Value{ .string = self.config.sampling_type });

            const json_payload = std.json.Value{ .object = payload_map };

            self.coordinator.sendToWorker(worker.node_id, MessageType.START_INNER_LOOP, json_payload) catch |err| {
                std.log.err("Failed to send task to worker {}: {}", .{worker.node_id, err});
            };

            std.log.info("Assigned chunk {} (offset: {}, len: {}) to worker {}", .{
                chunk.?.id,
                chunk.?.offset,
                chunk.?.length,
                worker.node_id,
            });

            workers_assigned += 1;
        }

        return workers_assigned;
    }

    /// Collect results from workers, accumulate gradients on the fly, and apply update.
    /// This keeps memory usage constant O(ModelSize) regardless of worker count.
    fn accumulateAndApplyGradients(self: *Self, expected_count: usize) !void {
        if (expected_count == 0) return;
        if (self.master_parameters == null) return error.ParametersNotInitialized;

        // 1. Initialize Accumulator (Zeroed buffer matching total parameter size)
        const total_param_bytes = try totalParamBytes(self.parameter_shapes);

        const accumulator = try self.allocator.alloc(u8, total_param_bytes);
        defer self.allocator.free(accumulator);
        @memset(accumulator, 0); // Important: Init to zero

        var collected_count: usize = 0;
        var total_loss: f32 = 0.0;

        std.log.info("Streaming results from {} workers...", .{expected_count});

        // 2. Collection Loop
        while (collected_count < expected_count) {
            // Fetch one message at a time
            const responses = try self.coordinator.collectFromWorkers(MessageType.INNER_LOOP_COMPLETE, 1);
            defer {
                for (responses.items) |*msg| msg.deinitClone(self.allocator);
                responses.deinit();
            }

            if (responses.items.len == 0) continue; // Safety check

            const response = responses.items[0];

            // Handle chunk completion tracking
            if (response.data == .object) {
                if (response.data.object.get("chunk_id")) |chunk_val| {
                    if (chunk_val == .integer) {
                        const chunk_id: usize = @intCast(chunk_val.integer);
                        if (self.coordinator.data_manager) |*dm| {
                            dm.markComplete(chunk_id);
                            std.log.info("Worker {} completed chunk {}", .{ response.sender_node, chunk_id });
                        }
                    }
                }
            }

            // Decode Payload
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            // Large payload path: worker writes delta to a shared blob file and sends its path.
            if (response.data == .object) {
                const obj = response.data.object;
                if (obj.get("delta_path")) |delta_val| {
                    const delta_path = switch (delta_val) {
                        .string => |s| s,
                        else => return error.InvalidMessageFormat,
                    };
                    const loss_val_f32: f32 = blk_loss: {
                        if (obj.get("loss")) |loss_val| {
                            break :blk_loss switch (loss_val) {
                                .float => |f| @floatCast(f),
                                .integer => |i| @floatFromInt(i),
                                else => return error.InvalidMessageFormat,
                            };
                        }
                        return error.InvalidMessageFormat;
                    };

                    const file = try std.fs.cwd().openFile(delta_path, .{});
                    defer file.close();

                    const stat = try file.stat();
                    if (stat.size > @as(u64, std.math.maxInt(usize))) return error.DimensionMismatch;
                    const file_size: usize = @intCast(stat.size);
                    if (file_size != total_param_bytes) {
                        std.log.err("Worker delta size mismatch! Expected {}, got {}", .{ total_param_bytes, file_size });
                        return error.DimensionMismatch;
                    }

                    const acc_f32: [*]f32 = @ptrCast(@alignCast(accumulator.ptr));
                    const total_floats: usize = total_param_bytes / @sizeOf(f32);

                    const chunk_floats: usize = 1 << 20; // 1,048,576 f32 ~= 4MiB
                    const buf_f32 = try arena.allocator().alloc(f32, @min(chunk_floats, total_floats));
                    const buf_bytes = std.mem.sliceAsBytes(buf_f32);

                    var reader = file.reader();
                    var offset_bytes: usize = 0;
                    while (offset_bytes < total_param_bytes) {
                        const remaining = total_param_bytes - offset_bytes;
                        const to_read = @min(buf_bytes.len, remaining);
                        try reader.readNoEof(buf_bytes[0..to_read]);

                        const floats_read: usize = to_read / @sizeOf(f32);
                        const acc_slice = acc_f32[offset_bytes / @sizeOf(f32) .. offset_bytes / @sizeOf(f32) + floats_read];
                        for (buf_f32[0..floats_read], 0..) |d, j| {
                            acc_slice[j] += d;
                        }
                        offset_bytes += to_read;
                    }

                    if (std.posix.getenv("PCP_KEEP_BLOB_FILES") == null) {
                        std.fs.cwd().deleteFile(delta_path) catch {};
                    }

                    total_loss += loss_val_f32;
                    collected_count += 1;
                    continue;
                }
            }

            // Extract base64 payload
            const b64_payload = switch (response.data) {
                .string => |s| s,
                .object => |obj| blk: {
                    if (obj.get("params")) |p| {
                        break :blk switch (p) {
                            .string => |s| s,
                            else => return error.InvalidMessageFormat,
                        };
                    }
                    return error.InvalidMessageFormat;
                },
                else => return error.InvalidMessageFormat,
            };

            const capnp_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_payload);
            const capnp_bytes = try arena.allocator().alloc(u8, capnp_len);
            try std.base64.standard.Decoder.decode(capnp_bytes, b64_payload);

            const reader = try binary_protocol.ShepherdPayload.Reader.init(capnp_bytes);
            defer reader.deinit();

            const delta_bytes = try reader.getUpdatedParams(); // This is now the Delta
            const loss_val = try reader.getLoss();

            // 3. Accumulate: Accumulator += Delta
            if (delta_bytes.len != total_param_bytes) {
                std.log.err("Worker delta size mismatch! Expected {}, got {}", .{ total_param_bytes, delta_bytes.len });
                return error.DimensionMismatch;
            }

            const acc_f32: [*]f32 = @ptrCast(@alignCast(accumulator.ptr));
            const delta_f32: [*]const f32 = @ptrCast(@alignCast(delta_bytes.ptr));
            const num_floats = total_param_bytes / @sizeOf(f32);

            // Vectorize addition
            for (0..num_floats) |i| {
                acc_f32[i] += delta_f32[i];
            }

            total_loss += loss_val;
            collected_count += 1;

            // Message is freed here by the defer/arena logic, keeping memory low
        }

        // 4. Average the Accumulator
        const acc_f32: [*]f32 = @ptrCast(@alignCast(accumulator.ptr));
        const num_floats = total_param_bytes / @sizeOf(f32);
        const divisor = @as(f32, @floatFromInt(collected_count));

        for (0..num_floats) |i| {
            acc_f32[i] /= divisor;
        }

        // 5. Apply to Master Parameters (Per Tensor)
        var byte_offset: usize = 0;
        const param_tensors = self.master_parameters.?;

        for (param_tensors, 0..) |*param_tensor_ptr, i| {
            const shape = self.parameter_shapes[i];
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);
            const tensor_size = elem_count * @sizeOf(f32);

            // Get current master data
            const master_bytes = try param_tensor_ptr.toBytes(self.allocator);
            defer self.allocator.free(master_bytes);
            const master_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, master_bytes));

            // Get corresponding averaged delta
            const grad_slice = acc_f32[byte_offset / 4 .. (byte_offset + tensor_size) / 4];

            // Update using Nesterov (Host Optimizer)
            // Note: master_f32 is modified in-place
            try self.host_optimizer.update(i, master_f32, grad_slice);

            // Wrap back to MLIR Tensor
            param_tensor_ptr.deinit();
            const updated_bytes = std.mem.sliceAsBytes(master_f32);
            param_tensor_ptr.* = try Tensor.fromBytes(self.mlir_builder, updated_bytes, shape, .f32);

            byte_offset += tensor_size;
        }

        // Update metrics
        const avg_loss = total_loss / divisor;
        self.metrics.loss = avg_loss;

        if (!std.math.isFinite(avg_loss)) {
            std.log.err("Round complete with non-finite loss: {d}", .{avg_loss});
            monitoring.setMetrics(self.current_epoch, avg_loss, self.coordinator.getWorkerCount());
            return error.NonFiniteLoss;
        }

        std.log.info("Round complete. Avg Loss: {d:.4}", .{avg_loss});
    }



    /// Save recovery state to disk for shepherd resilience
    fn saveRecoveryState(self: *Self) !void {
        const recovery_dir = try std.fs.path.join(self.allocator, &[_][]const u8{ self.config.checkpoint_dir, "recovery" });
        defer self.allocator.free(recovery_dir);

        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ self.config.checkpoint_dir, "recovery", RECOVERY_FILENAME });
        defer self.allocator.free(full_path);

        std.fs.cwd().makePath(recovery_dir) catch {};

        const file = try std.fs.cwd().createFile(full_path, .{});
        defer file.close();
        var writer = file.writer();

        // 1. Magic Header
        try writer.writeAll("PCP_RES");

        // 2. Save Model Parameters
        if (self.master_parameters) |params| {
            try writer.writeInt(u32, @intCast(params.len), .little);
            for (params) |t| {
                const bytes = try t.toBytes(self.allocator);
                defer self.allocator.free(bytes);
                try writer.writeInt(u64, @intCast(bytes.len), .little);
                try writer.writeAll(bytes);
            }
        } else {
            try writer.writeInt(u32, 0, .little);
        }

        // 3. Save Optimizer State
        const opt_bytes = try self.host_optimizer.serialize();
        defer self.allocator.free(opt_bytes);
        try writer.writeInt(u64, @intCast(opt_bytes.len), .little);
        try writer.writeAll(opt_bytes);

        // 4. Save DataManager State
        if (self.coordinator.data_manager) |*dm| {
            const dm_bytes = try dm.serialize();
            defer self.allocator.free(dm_bytes);
            try writer.writeInt(u64, @intCast(dm_bytes.len), .little);
            try writer.writeAll(dm_bytes);
        } else {
            try writer.writeInt(u64, 0, .little);
        }

        std.log.info("State checkpoint saved for recovery.", .{});
    }

    /// Try to load recovery state from disk
    fn tryLoadRecoveryState(self: *Self) !bool {
        if (!self.config.resume_training) return false;

        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ self.config.checkpoint_dir, "recovery", RECOVERY_FILENAME });
        defer self.allocator.free(full_path);

        std.log.info("Attempting to load recovery state from: {s}", .{full_path});
        const file = std.fs.cwd().openFile(full_path, .{}) catch |err| {
            std.log.warn("Could not open recovery file: {}", .{err});
            return false;
        };
        defer file.close();
        var reader = file.reader();

        // 1. Check Header
        var header: [7]u8 = undefined;
        _ = try reader.readAll(&header);
        if (!std.mem.eql(u8, &header, "PCP_RES")) {
            std.log.warn("Invalid recovery file header", .{});
            return false;
        }
        std.log.info("Recovery file header valid, loading state...", .{});

        // 2. Load Parameters
        const num_tensors = try reader.readInt(u32, .little);
        if (num_tensors > 0) {
            // Ensure master_parameters is allocated
            if (self.master_parameters == null) {
                self.master_parameters = try self.allocator.alloc(Tensor, num_tensors);
            }

            for (self.master_parameters.?, 0..) |*t, i| {
                const len = try reader.readInt(u64, .little);
                const bytes = try self.allocator.alloc(u8, len);
                defer self.allocator.free(bytes);
                _ = try reader.readAll(bytes);

                // If tensor already exists, deinit it first
                if (i < self.parameter_shapes.len) {
                    // Recreate tensor from bytes
                    t.* = try Tensor.fromBytes(self.mlir_builder, bytes, self.parameter_shapes[i], .f32);
                }
            }
        }

        // 3. Load Optimizer
        const opt_len = try reader.readInt(u64, .little);
        if (opt_len > 0) {
            const opt_bytes = try self.allocator.alloc(u8, opt_len);
            defer self.allocator.free(opt_bytes);
            _ = try reader.readAll(opt_bytes);
            try self.host_optimizer.deserialize(opt_bytes);
        }

        // 4. Load DataManager
        const dm_len = try reader.readInt(u64, .little);
        if (dm_len > 0) {
            const dm_bytes = try self.allocator.alloc(u8, dm_len);
            defer self.allocator.free(dm_bytes);
            _ = try reader.readAll(dm_bytes);
            if (self.coordinator.data_manager) |*dm| {
                try dm.loadState(dm_bytes);
            }
        }

        std.log.info("Successfully resumed training from checkpoint!", .{});
        return true;
    }

    /// Check if training should stop
    fn shouldStop(self: *Self) bool {
        // Stop if we've reached the configured outer loop steps limit
        // (Note: We keep this to allow "train for X steps", but if user wants
        // pure epoch-based training, they set outer_loop_steps very high)
        if (self.metrics.outer_loop_count >= self.config.base_config.outer_loop_steps) {
            std.log.info("Stopping: Reached maximum outer loop steps ({})", .{self.config.base_config.outer_loop_steps});
            return true;
        }

        // Stop if DataManager says we are done (Max epochs reached)
        if (self.coordinator.data_manager) |*dm| {
            if (dm.current_epoch > dm.max_epochs) {
                std.log.info("Stopping: Reached maximum epochs ({})", .{dm.max_epochs});
                return true;
            }
        }

        if (self.metrics.loss < 0.01) {
            std.log.info("Stopping: Converged (loss < 0.01)", .{});
            return true;
        }

        return false;
    }

    /// Interface implementations
    fn deinitInterface(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }

    fn getName(ptr: *anyopaque) []const u8 {
        _ = ptr;
        return "DiLoCo-MLIR";
    }

    fn getStatus(ptr: *anyopaque) TrainingStatus {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.status;
    }
};
