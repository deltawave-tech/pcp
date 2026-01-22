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
    effective_batch_size: ?usize = null,

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
    master_param_raw_data: ?[][]u8, // Authoritative raw byte storage (MLIR may not copy data)
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
        // 1. Apply Logic/Stability Patches (Log/Div guards)
        var current_source = try ModelSanitizer.applyStabilityPatches(allocator, raw_mlir);

        // 2. Apply BF16 Sanitization (model is now natively BF16 from Python export)
        if (config.dtype == .bf16) {
            std.log.info("Sanitizing BF16 model...", .{});

            // A. Sanitize Hex Constants (fixes -inf/NaN constants)
            std.log.info("  Step A: Sanitizing hex float constants...", .{});
            const decimal_source = try ModelSanitizer.sanitizeHexFloats(allocator, current_source);
            allocator.free(current_source);

            // B. DISABLED: Model is natively BF16 from Python with F32 Softmax preserved
            // Running convertF32ToBF16 would destroy the explicit F32 softmax operations

            current_source = decimal_source;
        }

        std.log.info("Sanitizing large constants...", .{});
        const mlir_source = try ModelSanitizer.sanitizeLargeConstants(allocator, current_source);
        allocator.free(current_source);
        defer allocator.free(mlir_source);

        // Introspect using the new ModelInspector (it will now see bf16 types if converted)
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
            .learning_rate = 0.7,
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
            .master_param_raw_data = null,
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

        // Free the raw parameter data
        if (self.master_param_raw_data) |data| {
            for (data) |bytes| self.allocator.free(bytes);
            self.allocator.free(data);
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
            try self.accumulateAndApplyGradients(workers_assigned);

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

        self.status = .completed;
        monitoring.setStatus(.completed);
    }

    /// Initialize master parameters using MLIR for the loaded model
    fn initializeMasterParameters(self: *Self) !void {

        // Allocate array for master parameter tensors
        const param_tensors = try self.allocator.alloc(Tensor, self.parameter_shapes.len);
        // Allocate array for raw buffers (authoritative storage)
        self.master_param_raw_data = try self.allocator.alloc([]u8, self.parameter_shapes.len);
        var rng = std.Random.DefaultPrng.init(12345);

        // Initialize each parameter tensor with appropriate shape
        for (param_tensors, 0..) |*param_tensor, i| {
            const shape = self.parameter_shapes[i];

            // Calculate element count for this tensor
            var element_count: usize = 1;
            for (shape) |dim| {
                element_count *= @intCast(dim);
            }

            // Calculate byte size based on target dtype
            const bytes_per_element = self.config.dtype.sizeInBytes();
            const byte_data = try self.allocator.alloc(u8, element_count * bytes_per_element);
            // DO NOT defer free - we store it in master_param_raw_data

            // GPT-2 Standard Initialization: Fixed 0.02 stddev
            // NOTE: The original Xavier initialization used total element count instead of fan-in,
            // resulting in ~30x smaller weights (0.003 vs 0.088). This caused:
            // - F32: Gradients ~1e-8 -> Adam v ~1e-16 -> explosion due to epsilon dominance
            // - BF16: Immediate overflow to Inf -> finite guard zeros grads -> model freezes at ~0.25 loss
            const scale: f32 = 0.02;
            for (0..element_count) |j| {
                const value = rng.random().floatNorm(f32) * scale;
                const offset = j * bytes_per_element;

                switch (self.config.dtype) {
                    .bf16 => {
                        // Convert f32 to bf16: just take the upper 16 bits
                        const f32_bits: u32 = @bitCast(value);
                        const bf16_bits: u16 = @truncate(f32_bits >> 16);
                        @memcpy(byte_data[offset .. offset + 2], std.mem.asBytes(&bf16_bits));
                    },
                    .f16 => {
                        // For f16, use a simple conversion (may lose precision)
                        const f32_bits: u32 = @bitCast(value);
                        const sign: u16 = @truncate((f32_bits >> 16) & 0x8000);
                        const exp = @as(i32, @intCast((f32_bits >> 23) & 0xFF)) - 127;
                        const mant: u16 = @truncate((f32_bits >> 13) & 0x3FF);
                        const f16_bits: u16 = if (exp < -14) sign // Underflow to 0
                        else if (exp > 15) sign | 0x7C00 // Overflow to inf
                        else sign | @as(u16, @intCast((exp + 15) & 0x1F)) << 10 | mant;
                        @memcpy(byte_data[offset .. offset + 2], std.mem.asBytes(&f16_bits));
                    },
                    .f32 => {
                        @memcpy(byte_data[offset .. offset + 4], std.mem.asBytes(&value));
                    },
                    .f64 => {
                        const f64_val: f64 = @floatCast(value);
                        @memcpy(byte_data[offset .. offset + 8], std.mem.asBytes(&f64_val));
                    },
                    else => {
                        // For integer types, convert to int
                        const int_val: i32 = @intFromFloat(value * 1000.0);
                        @memcpy(byte_data[offset .. offset + bytes_per_element], std.mem.asBytes(&int_val)[0..bytes_per_element]);
                    },
                }
            }

            // Store raw data in authoritative buffer
            self.master_param_raw_data.?[i] = byte_data;

            param_tensor.* = try Tensor.fromBytes(self.mlir_builder, byte_data, shape, self.config.dtype);
        }

        self.master_parameters = param_tensors;
    }

    /// Extract raw byte data from a tensor defined by a stablehlo.constant operation.

    /// Save current master parameters to a binary file
    fn saveCheckpoint(self: *Self, step: usize) !void {
        if (self.master_param_raw_data == null) return;

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

        const raw_data = self.master_param_raw_data.?;
        // 2. Number of tensors
        try writer.writeInt(u32, @intCast(raw_data.len), .little);

        for (raw_data, 0..) |bytes, i| {
            // 3a. Write Shape info
            const shape = self.parameter_shapes[i];
            try writer.writeInt(u8, @intCast(shape.len), .little);
            for (shape) |d| {
                try writer.writeInt(i64, d, .little);
            }

            // 3b. Write Data directly from raw buffer
            try writer.writeInt(u64, @intCast(bytes.len), .little);
            try writer.writeAll(bytes);
        }
        std.log.info("✓ Checkpoint saved.", .{});
    }


    /// Broadcast to snapshot participants only
    /// Returns the number of workers that were actually assigned chunks
    fn broadcastToSnapshot(self: *Self, workers: []const shepherd.WorkerConnection) !usize {
        if (self.master_param_raw_data == null) {
            return error.ParametersNotInitialized;
        }

        // Concatenate raw parameter data from authoritative storage
        var total_param_bytes = ArrayList(u8).init(self.allocator);
        defer total_param_bytes.deinit();

        for (self.master_param_raw_data.?) |bytes| {
            try total_param_bytes.appendSlice(bytes);
        }

        // Validate Outgoing Parameters (DType Aware)
        const raw_bytes = total_param_bytes.items;
        var sum: f64 = 0.0;
        var non_zero_count: usize = 0;
        var nan_count: usize = 0;
        var inf_count: usize = 0;

        if (self.config.dtype == .f32) {
            const f32_view = std.mem.bytesAsSlice(f32, raw_bytes);
            for (f32_view) |val| {
                if (std.math.isNan(val)) {
                    nan_count += 1;
                } else if (std.math.isInf(val)) {
                    inf_count += 1;
                } else {
                    if (val != 0.0) non_zero_count += 1;
                    sum += @abs(val);
                }
            }
        } else if (self.config.dtype == .bf16) {
            // Correctly interpret bf16 bytes
            var i: usize = 0;
            while (i < raw_bytes.len) : (i += 2) {
                const u16_val = std.mem.readInt(u16, raw_bytes[i..][0..2], .little);
                const u32_val = @as(u32, u16_val) << 16;
                const val: f32 = @bitCast(u32_val);

                if (std.math.isNan(val)) {
                    nan_count += 1;
                } else if (std.math.isInf(val)) {
                    inf_count += 1;
                } else {
                    if (val != 0.0) non_zero_count += 1;
                    sum += @as(f64, @abs(val));
                }
            }
        }

        const num_elements = raw_bytes.len / self.config.dtype.sizeInBytes();

        std.log.warn("SHEPHERD SEND CHECK:", .{});
        std.log.warn("   Total Bytes: {}", .{raw_bytes.len});
        std.log.warn("   Elements: {}", .{num_elements});
        std.log.warn("   Non-Zero: {}", .{non_zero_count});
        std.log.warn("   Abs Sum: {d:.4}", .{sum});

        if (nan_count > 0 or inf_count > 0) {
            std.log.err("CRITICAL: Parameter corruption detected! NaNs: {}, Infs: {}", .{ nan_count, inf_count });
        }

        // Use chunked transfer for large models (> 50MB raw bytes), otherwise Cap'n Proto + base64
        const CHUNK_THRESHOLD = 50 * 1024 * 1024; // 50MB
        const use_chunked_transfer = raw_bytes.len > CHUNK_THRESHOLD;

        var capnp_bytes: ?[]u8 = null;
        var b64_encoded_params: ?[]u8 = null;
        var encoded_len: usize = 0;

        if (use_chunked_transfer) {
            // For large models: broadcast raw parameter bytes via chunked transfer
            // This bypasses Cap'n Proto which has size limits
            std.log.info("Using chunked transfer for {} byte payload (raw bytes)", .{raw_bytes.len});
            try self.coordinator.broadcastLargeData(raw_bytes);
        } else {
            // For small models: use Cap'n Proto serialization + base64
            const empty_slice: []const u8 = &[_]u8{};
            const worker_payload = binary_protocol.WorkerPayload{
                .params = total_param_bytes.items,
                .input_ids = empty_slice,
                .targets = empty_slice,
            };
            capnp_bytes = try worker_payload.serialize(self.allocator);

            const b64_len = std.base64.standard.Encoder.calcSize(capnp_bytes.?.len);
            b64_encoded_params = try self.allocator.alloc(u8, b64_len);
            encoded_len = std.base64.standard.Encoder.encode(b64_encoded_params.?, capnp_bytes.?).len;
        }
        defer if (capnp_bytes) |c| self.allocator.free(c);
        defer if (b64_encoded_params) |p| self.allocator.free(p);

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

            // Build JSON payload with chunk info (params omitted if using chunked transfer)
            var payload_map = std.json.ObjectMap.init(self.allocator);
            defer payload_map.deinit();

            // Only include params if NOT using chunked transfer
            if (!use_chunked_transfer) {
                try payload_map.put("params", std.json.Value{ .string = b64_encoded_params.?[0..encoded_len] });
            } else {
                // Tell worker that weight_blob contains raw bytes (not Cap'n Proto)
                try payload_map.put("raw_params", std.json.Value{ .bool = true });
            }
            try payload_map.put("offset", std.json.Value{ .integer = @intCast(chunk.?.offset) });
            try payload_map.put("length", std.json.Value{ .integer = @intCast(chunk.?.length) });
            try payload_map.put("chunk_id", std.json.Value{ .integer = @intCast(chunk.?.id) });
            try payload_map.put("data_path", std.json.Value{ .string = self.config.data_path });
            try payload_map.put("tau", std.json.Value{ .integer = @intCast(self.config.tau) });
            try payload_map.put("tokenizer", std.json.Value{ .string = self.config.tokenizer_type });
            try payload_map.put("sampling", std.json.Value{ .string = self.config.sampling_type });
            if (self.config.effective_batch_size) |ebs| {
                try payload_map.put("effective_batch_size", std.json.Value{ .integer = @intCast(ebs) });
            }

            // Send dtype to worker for correct precision handling
            const dtype_str: []const u8 = switch (self.config.dtype) {
                .bf16 => "bf16",
                .f16 => "f16",
                else => "f32",
            };
            try payload_map.put("dtype", std.json.Value{ .string = dtype_str });

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

        // 1. Calculate sizes
        var transmission_bytes: usize = 0; // Size coming from network (e.g. bf16)
        var accumulation_bytes: usize = 0; // Size for math (always f32)
        const dtype_size = self.config.dtype.sizeInBytes();

        for (self.parameter_shapes) |shape| {
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);
            transmission_bytes += elem_count * dtype_size;
            accumulation_bytes += elem_count * @sizeOf(f32);
        }

        // Initialize Accumulator (Always f32 for precision during averaging)
        const accumulator = try self.allocator.alloc(u8, accumulation_bytes);
        defer self.allocator.free(accumulator);
        @memset(accumulator, 0);

        var collected_count: usize = 0;
        var total_loss: f32 = 0.0;

        std.log.info("Streaming results from {} workers (Exp: {} bytes/worker)...", .{ expected_count, transmission_bytes });

        // 2. Collection Loop
        while (collected_count < expected_count) {
            // Fetch one message at a time
            const responses = try self.coordinator.collectFromWorkers(MessageType.INNER_LOOP_COMPLETE, 1);
            defer {
                for (responses.items) |*msg| msg.deinitClone(self.allocator);
                responses.deinit();
            }

            if (responses.items.len == 0) continue;

            const response = responses.items[0];

            // Handle chunk completion tracking
            if (response.data == .object) {
                if (response.data.object.get("chunk_id")) |chunk_val| {
                    if (chunk_val == .integer) {
                        const chunk_id: usize = @intCast(chunk_val.integer);
                        if (self.coordinator.data_manager) |*dm| {
                            dm.markComplete(chunk_id);
                        }
                    }
                }
            }

            // Decode Payload
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            var delta_bytes: []const u8 = undefined;
            var loss_val: f32 = 0.0;

            const is_chunked = if (response.data == .object)
                if (response.data.object.get("chunked")) |v| v == .bool and v.bool else false
            else
                false;

            if (is_chunked) {
                const update_state = self.coordinator.getReassembledUpdate(response.sender_node) orelse {
                    std.log.err("Worker {} sent chunked completion but no reassembled data found!", .{response.sender_node});
                    return error.MissingReassembledData;
                };
                delta_bytes = update_state.buffer.items;
                loss_val = update_state.loss;
            } else {
                // Legacy path (Cap'n Proto)
                const b64_payload = switch (response.data) {
                    .string => |s| s,
                    .object => |obj| obj.get("params").?.string,
                    else => return error.InvalidMessageFormat,
                };
                const capnp_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_payload);
                const capnp_bytes = try arena.allocator().alloc(u8, capnp_len);
                try std.base64.standard.Decoder.decode(capnp_bytes, b64_payload);
                const reader = try binary_protocol.ShepherdPayload.Reader.init(capnp_bytes);
                defer reader.deinit();
                delta_bytes = try reader.getUpdatedParams();
                loss_val = try reader.getLoss();
            }

            // 3. Accumulate: Accumulator += Delta
            if (delta_bytes.len != transmission_bytes) {
                std.log.err("Worker delta size mismatch! Expected {}, got {}", .{ transmission_bytes, delta_bytes.len });
                return error.DimensionMismatch;
            }

            const acc_f32: [*]f32 = @ptrCast(@alignCast(accumulator.ptr));
            const num_floats = accumulation_bytes / @sizeOf(f32);

            // Handle type conversion if needed
            switch (self.config.dtype) {
                .f32 => {
                    const delta_f32 = @as([*]const f32, @ptrCast(@alignCast(delta_bytes.ptr)));
                    for (0..num_floats) |i| acc_f32[i] += delta_f32[i];
                },
                .bf16 => {
                    // bf16 is u16 (2 bytes).
                    // Conversion: shift left 16 bits to get f32 (assuming IEEE754 structure overlap)
                    var i: usize = 0;
                    var offset: usize = 0;
                    while (i < num_floats) : (i += 1) {
                        const bf16_val = std.mem.readInt(u16, delta_bytes[offset..][0..2], .little);
                        const as_u32 = @as(u32, bf16_val) << 16;
                        const val_f32: f32 = @bitCast(as_u32);
                        acc_f32[i] += val_f32;
                        offset += 2;
                    }
                },
                else => {
                    std.log.err("Unsupported accumulation dtype: {}", .{self.config.dtype});
                    return error.UnsupportedDType;
                },
            }

            total_loss += loss_val;
            collected_count += 1;

            if (is_chunked) {
                self.coordinator.clearReassembledUpdate(response.sender_node);
            }
        }

        // 4. Average the Accumulator
        const acc_f32: [*]f32 = @ptrCast(@alignCast(accumulator.ptr));
        const num_floats = accumulation_bytes / @sizeOf(f32);
        const divisor = @as(f32, @floatFromInt(collected_count));

        var nan_grad_count: usize = 0;
        for (0..num_floats) |i| {
            acc_f32[i] /= divisor;

            // NaN Guard: If gradients exploded/diverged, zero them out to save the model
            if (std.math.isNan(acc_f32[i]) or std.math.isInf(acc_f32[i])) {
                acc_f32[i] = 0.0;
                nan_grad_count += 1;
            }
        }

        if (nan_grad_count > 0) {
            std.log.err("WARNING: Detected {} NaN/Inf values in aggregated gradients! Zeroed them out.", .{nan_grad_count});
        }

        // 5. Apply to Master Parameters (Per Tensor)
        // Note: Master params might be bf16, Optimizer expects f32, Result must be bf16
        var acc_offset: usize = 0; // Offset in the f32 accumulator
        const param_tensors = self.master_parameters.?;

        for (param_tensors, 0..) |*param_tensor_ptr, i| {
            const shape = self.parameter_shapes[i];
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);

            // Use authoritative raw data instead of querying MLIR
            const master_raw_bytes = self.master_param_raw_data.?[i];

            // Convert Master Bytes -> F32 for Optimizer
            const master_f32_buf = try self.allocator.alloc(f32, elem_count);
            defer self.allocator.free(master_f32_buf);

            switch (self.config.dtype) {
                .f32 => {
                    @memcpy(master_f32_buf, std.mem.bytesAsSlice(f32, master_raw_bytes));
                },
                .bf16 => {
                    var off: usize = 0;
                    for (0..elem_count) |k| {
                        const bf16_val = std.mem.readInt(u16, master_raw_bytes[off..][0..2], .little);
                        const val_f32: f32 = @bitCast(@as(u32, bf16_val) << 16);
                        master_f32_buf[k] = val_f32;
                        off += 2;
                    }
                },
                else => return error.UnsupportedDType,
            }

            // Get corresponding averaged gradient from accumulator
            const grad_slice = acc_f32[acc_offset .. acc_offset + elem_count];

            // Update using Nesterov (Host Optimizer) - In-place on master_f32_buf
            try self.host_optimizer.update(i, master_f32_buf, grad_slice);

            // Convert Updated F32 -> Target DType Bytes
            const updated_bytes = try self.allocator.alloc(u8, elem_count * self.config.dtype.sizeInBytes());

            switch (self.config.dtype) {
                .f32 => {
                    @memcpy(updated_bytes, std.mem.sliceAsBytes(master_f32_buf));
                },
                .bf16 => {
                    var off: usize = 0;
                    for (master_f32_buf) |val| {
                        const f32_bits: u32 = @bitCast(val);
                        // Truncate to bf16
                        const bf16_val: u16 = @truncate(f32_bits >> 16);
                        @memcpy(updated_bytes[off..][0..2], std.mem.asBytes(&bf16_val));
                        off += 2;
                    }
                },
                else => return error.UnsupportedDType,
            }

            // Free old raw buffer and swap pointer
            self.allocator.free(self.master_param_raw_data.?[i]);
            self.master_param_raw_data.?[i] = updated_bytes;

            // Re-create MLIR tensor (points to new buffer)
            param_tensor_ptr.deinit();
            param_tensor_ptr.* = try Tensor.fromBytes(self.mlir_builder, updated_bytes, shape, self.config.dtype);

            acc_offset += elem_count;
        }

        // Update metrics
        self.metrics.loss = total_loss / divisor;
        std.log.info("Round complete. Avg Loss: {d:.4}", .{self.metrics.loss});
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

        // 2. Save Model Parameters (use raw data, bypass MLIR)
        if (self.master_param_raw_data) |raw_data| {
            try writer.writeInt(u32, @intCast(raw_data.len), .little);
            for (raw_data) |bytes| {
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
            // Ensure master_parameters and raw data are allocated
            if (self.master_parameters == null) {
                self.master_parameters = try self.allocator.alloc(Tensor, num_tensors);
            }
            if (self.master_param_raw_data == null) {
                self.master_param_raw_data = try self.allocator.alloc([]u8, num_tensors);
            }

            for (self.master_parameters.?, 0..) |*t, i| {
                const len = try reader.readInt(u64, .little);
                const bytes = try self.allocator.alloc(u8, len);
                // DO NOT defer free - store in master_param_raw_data
                _ = try reader.readAll(bytes);

                // Store raw data
                self.master_param_raw_data.?[i] = bytes;

                // If tensor already exists, deinit it first
                if (i < self.parameter_shapes.len) {
                    // Recreate tensor from bytes with correct dtype
                    t.* = try Tensor.fromBytes(self.mlir_builder, bytes, self.parameter_shapes[i], self.config.dtype);
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
