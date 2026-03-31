const std = @import("std");
const Allocator = std.mem.Allocator;
const training_algorithm = @import("training_algorithm.zig");
const shepherd = @import("../nodes/controllers/shepherd.zig");
const message = @import("../network/message.zig");
const tensor = @import("../core/tensor.zig");
const execution = @import("../execution.zig");
const nesterov_host = @import("../optimizers/nesterov.zig");
const model_introspection = @import("../mlir/model_introspection.zig");
const ops = @import("../core/ops.zig");

const WorkerConnection = shepherd.WorkerConnection;
const ModelSanitizer = @import("../compiler/sanitizer.zig").ModelSanitizer;
const ModelInspector = model_introspection.ModelInspector;
const GraphBuilder = @import("../compiler/graph_builder.zig").GraphBuilder;

const TrainingAlgorithm = training_algorithm.TrainingAlgorithm;
const TrainingStatus = training_algorithm.TrainingStatus;
const Shepherd = shepherd.Shepherd;
const NesterovHost = nesterov_host.Nesterov;
const Tensor = tensor.Tensor(void);

pub const StreamingDiLoCoConfig = struct {
    num_fragments: usize = 4,
    inner_steps: usize = 100,
    overlap_tau: usize = 1,
    alpha: f32 = 0.5,
    learning_rate: f32 = 0.0006,
    nesterov_momentum: f32 = 0.9,
    max_epochs: usize = 10,
    model_mlir_path: []const u8,
    data_path: []const u8,
    tokenizer_type: []const u8 = "char",
    sampling_type: []const u8 = "random",
    dtype: tensor.DType = .f32,
};

/// Tracks the aggregation state of a single fragment across workers
const FragmentState = struct {
    id: usize,
    current_round: usize,
    received_from: std.AutoHashMap(message.NodeId, void),
    pending_messages: std.ArrayList(message.MessageEnvelope),
    /// Accumulators for each tensor belonging to this fragment.
    /// Mapped by global tensor index -> f32 buffer
    accumulators: std.AutoHashMap(usize, []f32),

    pub fn init(allocator: Allocator, id: usize) FragmentState {
        return .{
            .id = id,
            .current_round = 0,
            .received_from = std.AutoHashMap(message.NodeId, void).init(allocator),
            .pending_messages = std.ArrayList(message.MessageEnvelope).init(allocator),
            .accumulators = std.AutoHashMap(usize, []f32).init(allocator),
        };
    }

    pub fn deinit(self: *FragmentState, allocator: Allocator) void {
        self.received_from.deinit();
        for (self.pending_messages.items) |*pending| {
            pending.deinitClone(allocator);
        }
        self.pending_messages.deinit();
        var it = self.accumulators.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.value_ptr.*);
        }
        self.accumulators.deinit();
    }
};

pub const StreamingDiLoCo = struct {
    allocator: Allocator,
    coordinator: *Shepherd,
    config: StreamingDiLoCoConfig,
    status: TrainingStatus,

    host_optimizer: NesterovHost,
    master_param_raw_data: [][]u8,

    parameter_shapes: [][]i64,
    data_input_shapes: [][]i64,
    mlir_source: []u8,

    /// Array of fragment states[0 .. num_fragments - 1]
    fragments: []FragmentState,

    const Self = @This();

    pub fn init(allocator: Allocator, coordinator: *Shepherd, config: StreamingDiLoCoConfig, mlir_builder: *ops.MLIRBuilder) !Self {
        std.log.info("StreamingDiLoCo: Loading MLIR from {s}...", .{config.model_mlir_path});
        const raw_mlir = try std.fs.cwd().readFileAlloc(allocator, config.model_mlir_path, 10 * 1024 * 1024);
        defer allocator.free(raw_mlir);

        const sanitized_mlir = try ModelSanitizer.applyStabilityPatches(allocator, raw_mlir);
        defer allocator.free(sanitized_mlir);

        const metadata = try ModelInspector.inspect(
            allocator,
            mlir_builder.ctx,
            sanitized_mlir,
            2,
        );
        const parameter_shapes = metadata.parameter_shapes;
        const data_input_shapes = metadata.data_input_shapes;

        // Create AdamMLIR optimizer for the graph builder (workers will use AdamW)
        const adam_mlir = @import("../optimizers/adam_mlir.zig");
        const AdamMLIR = adam_mlir.AdamMLIR(f32);
        const adam_config = adam_mlir.AdamMLIRConfiguration(f32){
            .learning_rate = config.learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .weight_decay = 0.01, // AdamW weight decay
        };
        const element_type = config.dtype.toMLIRType(mlir_builder.ctx);
        const adam_optimizer = try allocator.create(AdamMLIR);
        adam_optimizer.* = try AdamMLIR.init(allocator, mlir_builder, adam_config, element_type);

        // Build training graph with compute_gradients function
        // For streaming, we use single-step execution (no accumulation)
        std.log.info("Building training graph for streaming (single-step mode)...", .{});
        const mlir_source = try GraphBuilder.buildTrainingGraph(
            allocator,
            mlir_builder,
            sanitized_mlir,
            adam_optimizer,
            parameter_shapes.len,
        );
        errdefer allocator.free(mlir_source);
        var host_opt = NesterovHost.init(allocator, .{
            .learning_rate = 0.7, // Outer learning rate
            .momentum = config.nesterov_momentum,
        });
        try host_opt.initParameters(parameter_shapes);

        // Allocate master parameters (always f32, matching the MLIR pipeline)
        var master_raw = try allocator.alloc([]u8, parameter_shapes.len);
        var rng = std.Random.DefaultPrng.init(12345);

        for (parameter_shapes, 0..) |shape, i| {
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);

            master_raw[i] = try allocator.alloc(u8, elem_count * 4);
            const dest = std.mem.bytesAsSlice(f32, master_raw[i]);

            // GPT-2 Standard Initialization: Fixed 0.02 stddev
            const scale: f32 = 0.02;
            for (0..elem_count) |j| {
                dest[j] = rng.random().floatNorm(f32) * scale;
            }
        }

        // Initialize Fragment State tracking
        var fragments = try allocator.alloc(FragmentState, config.num_fragments);
        for (0..config.num_fragments) |p| {
            fragments[p] = FragmentState.init(allocator, p);

            // Assign tensors to this fragment using the "Strided" method
            // e.g. If num_fragments = 4, Fragment 0 gets tensors 0, 4, 8, 12...
            for (parameter_shapes, 0..) |shape, i| {
                if (i % config.num_fragments == p) {
                    var elem_count: usize = 1;
                    for (shape) |dim| elem_count *= @intCast(dim);
                    const acc_buffer = try allocator.alloc(f32, elem_count);
                    @memset(acc_buffer, 0.0);
                    try fragments[p].accumulators.put(i, acc_buffer);
                }
            }
        }

        return Self{
            .allocator = allocator,
            .coordinator = coordinator,
            .config = config,
            .status = .not_started,
            .host_optimizer = host_opt,
            .master_param_raw_data = master_raw,
            .parameter_shapes = parameter_shapes,
            .data_input_shapes = data_input_shapes,
            .mlir_source = mlir_source,
            .fragments = fragments,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.master_param_raw_data) |buf| self.allocator.free(buf);
        self.allocator.free(self.master_param_raw_data);

        for (self.fragments) |*f| f.deinit(self.allocator);
        self.allocator.free(self.fragments);

        self.host_optimizer.deinit();

        self.allocator.free(self.mlir_source);
        for (self.parameter_shapes) |s| self.allocator.free(s);
        self.allocator.free(self.parameter_shapes);
        for (self.data_input_shapes) |s| self.allocator.free(s);
        self.allocator.free(self.data_input_shapes);
    }

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

    // --- Interface Methods ---
    fn deinitInterface(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }

    fn getName(ptr: *anyopaque) []const u8 {
        _ = ptr;
        return "StreamingDiLoCo";
    }

    fn getStatus(ptr: *anyopaque) TrainingStatus {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.status;
    }

    fn parseFragmentRound(payload: std.json.ObjectMap) !usize {
        const round_value = payload.get("fragment_round") orelse return 0;
        return switch (round_value) {
            .integer => |value| @intCast(value),
            else => error.InvalidFragmentRound,
        };
    }

    fn isTrainingComplete(self: *Self) bool {
        for (self.fragments) |fragment| {
            if (fragment.current_round < self.config.max_epochs) {
                return false;
            }
        }
        return true;
    }

    /// Main Event Loop for Streaming DiLoCo
    pub fn run(ptr: *anyopaque) !void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.status = .running;

        try self.coordinator.ensureWorkersCompiled(
            self.mlir_source,
            self.parameter_shapes,
            self.data_input_shapes
        );

        const initial_participants = try self.coordinator.snapshotWorkers();
        defer initial_participants.deinit();

        if (initial_participants.items.len == 0) return error.NoWorkersConnected;

        try self.broadcastStartStreaming(initial_participants.items);

        std.log.info("Starting Streaming DiLoCo Event Loop with {} fragments...", .{self.config.num_fragments});

        while (self.status == .running) {
            const participants = try self.coordinator.snapshotWorkers();
            defer participants.deinit();

            const expected_count = participants.items.len;
            if (expected_count == 0) {
                std.time.sleep(1 * std.time.ns_per_s);
                continue;
            }

            // 2. Poll for ANY incoming fragment update (Non-blocking or short timeout)
            // Note: We'll modify Shepherd to expose this queue shortly.
            const msgs = try self.coordinator.collectFromWorkers(message.MessageType.FRAGMENT_UPDATE, 1);
            defer {
                for (msgs.items) |*msg| msg.deinitClone(self.allocator);
                msgs.deinit();
            }

            if (msgs.items.len == 0) continue;

            const msg = msgs.items[0];
            try self.processFragmentMessage(msg, expected_count);

            if (self.isTrainingComplete()) {
                self.status = .completed;
                break;
            }
        }

        if (self.status == .completed) {
            std.log.info("Streaming DiLoCo completed {} rounds per fragment", .{self.config.max_epochs});
            self.coordinator.stop();
        }
    }

    /// Process a single incoming FragmentUpdate from a worker
    fn processFragmentMessage(self: *Self, msg: message.MessageEnvelope, expected_workers: usize) !void {
        const payload = msg.data.object;
        const fragment_id = @as(usize, @intCast(payload.get("fragment_id").?.integer));
        const fragment_round = try parseFragmentRound(payload);

        if (fragment_id >= self.config.num_fragments) return error.InvalidFragmentId;
        if (fragment_round >= self.config.max_epochs) {
            std.log.debug("Ignoring fragment {} round {} beyond configured max {}", .{
                fragment_id, fragment_round, self.config.max_epochs,
            });
            return;
        }

        var frag_state = &self.fragments[fragment_id];
        if (fragment_round < frag_state.current_round) {
            std.log.debug("Ignoring stale fragment {} round {} from worker {}, current round is {}", .{
                fragment_id, fragment_round, msg.sender_node, frag_state.current_round,
            });
            return;
        }
        if (fragment_round > frag_state.current_round) {
            try self.queuePendingMessage(frag_state, msg, fragment_round);
            return;
        }

        try self.processCurrentRoundMessage(frag_state, msg, expected_workers);
    }

    fn queuePendingMessage(
        self: *Self,
        frag_state: *FragmentState,
        msg: message.MessageEnvelope,
        fragment_round: usize,
    ) !void {
        for (frag_state.pending_messages.items) |pending| {
            const pending_payload = pending.data.object;
            const pending_round = parseFragmentRound(pending_payload) catch continue;
            if (pending.sender_node == msg.sender_node and pending_round == fragment_round) {
                return;
            }
        }

        try frag_state.pending_messages.append(try msg.clone(self.allocator));
        std.log.debug("Queued future fragment {} round {} from worker {}", .{
            frag_state.id, fragment_round, msg.sender_node,
        });
    }

    fn processCurrentRoundMessage(
        self: *Self,
        frag_state: *FragmentState,
        msg: message.MessageEnvelope,
        expected_workers: usize,
    ) !void {
        // Skip if this worker already submitted for this round (prevent double counting)
        if (frag_state.received_from.contains(msg.sender_node)) return;

        // Extract binary gradients and accumulate them into frag_state.accumulators
        try self.accumulateFragmentGradients(frag_state, msg.data.object);

        try frag_state.received_from.put(msg.sender_node, {});

        std.log.info("Fragment {} round {} received from worker {} ({}/{})", .{
            frag_state.id, frag_state.current_round, msg.sender_node, frag_state.received_from.count(), expected_workers,
        });

        // Check if we reached consensus for this fragment
        if (frag_state.received_from.count() >= expected_workers) {
            const completed_round = frag_state.current_round;
            try self.applyFragmentUpdate(frag_state, expected_workers, completed_round);
            frag_state.current_round += 1;
            try self.processPendingMessages(frag_state, expected_workers);
        }
    }

    fn processPendingMessages(self: *Self, frag_state: *FragmentState, expected_workers: usize) !void {
        while (frag_state.current_round < self.config.max_epochs) {
            var progressed = false;
            var i: usize = 0;

            while (i < frag_state.pending_messages.items.len) {
                const pending = frag_state.pending_messages.items[i];
                const pending_round = parseFragmentRound(pending.data.object) catch {
                    var bad_pending = frag_state.pending_messages.orderedRemove(i);
                    bad_pending.deinitClone(self.allocator);
                    progressed = true;
                    continue;
                };

                if (pending_round < frag_state.current_round) {
                    var stale = frag_state.pending_messages.orderedRemove(i);
                    stale.deinitClone(self.allocator);
                    progressed = true;
                    continue;
                }

                if (pending_round != frag_state.current_round) {
                    i += 1;
                    continue;
                }

                var pending_msg = frag_state.pending_messages.orderedRemove(i);
                defer pending_msg.deinitClone(self.allocator);

                try self.processCurrentRoundMessage(frag_state, pending_msg, expected_workers);
                progressed = true;

                if (self.isTrainingComplete()) {
                    return;
                }
            }

            if (!progressed) break;
        }
    }

    /// Apply Host Optimizer and Broadcast updated Fragment
    fn applyFragmentUpdate(
        self: *Self,
        frag_state: *FragmentState,
        worker_count: usize,
        fragment_round: usize,
    ) !void {
        std.log.info("Applying Nesterov update to Fragment {} round {}...", .{ frag_state.id, fragment_round });

        const divisor = @as(f32, @floatFromInt(worker_count));

        var it = frag_state.accumulators.iterator();
        while (it.next()) |entry| {
            const tensor_idx = entry.key_ptr.*;
            const acc_buffer = entry.value_ptr.*;

            // 1. Average the accumulated gradients
            for (acc_buffer) |*val| {
                val.* /= divisor;
            }

            // 2. Fetch master parameters (already f32)
            const master_raw = self.master_param_raw_data[tensor_idx];
            const master_f32 = try self.allocator.alloc(f32, acc_buffer.len);
            defer self.allocator.free(master_f32);
            @memcpy(master_f32, std.mem.bytesAsSlice(f32, master_raw));

            // 3. Apply Nesterov Update (in-place modifies master_f32)
            try self.host_optimizer.update(tensor_idx, master_f32, acc_buffer);

            // 4. Save back to raw data
            @memcpy(master_raw, std.mem.sliceAsBytes(master_f32));

            // 5. Reset accumulator for next round
            @memset(acc_buffer, 0.0);
        }

        // Reset tracking
        frag_state.received_from.clearRetainingCapacity();

        // 6. Broadcast the updated fragment back to all workers
        try self.broadcastFragment(frag_state.id, fragment_round);
    }

    fn accumulateFragmentGradients(self: *Self, frag_state: *FragmentState, payload: std.json.ObjectMap) !void {
        // Get the tensor updates from the payload
        const updates = payload.get("updates") orelse return error.MissingUpdatesField;
        const updates_array = switch (updates) {
            .array => |arr| arr,
            else => return error.InvalidUpdatesFormat,
        };

        // Process each tensor update in the fragment
        for (updates_array.items) |update_value| {
            const update = switch (update_value) {
                .object => |obj| obj,
                else => continue,
            };

            const tensor_idx_val = update.get("tensor_idx") orelse continue;
            const tensor_idx = @as(usize, @intCast(switch (tensor_idx_val) {
                .integer => |i| i,
                else => continue,
            }));

            // Check if this tensor belongs to this fragment
            const acc_buffer = frag_state.accumulators.get(tensor_idx) orelse continue;

            // Get the gradient data (base64 encoded)
            const data_val = update.get("data") orelse continue;
            const data_str = switch (data_val) {
                .string => |s| s,
                else => continue,
            };

            // Decode base64 gradient data
            const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(data_str);
            const decoded = try self.allocator.alloc(u8, decoded_len);
            defer self.allocator.free(decoded);
            try std.base64.standard.Decoder.decode(decoded, data_str);

            // Convert bytes to f32 and accumulate
            const gradient_f32 = std.mem.bytesAsSlice(f32, decoded);
            if (gradient_f32.len != acc_buffer.len) {
                std.log.warn("Gradient size mismatch for tensor {}: expected {}, got {}", .{
                    tensor_idx, acc_buffer.len, gradient_f32.len
                });
                continue;
            }

            // Accumulate gradients
            for (acc_buffer, gradient_f32) |*acc, grad| {
                acc.* += grad;
            }
        }
    }

    fn broadcastFragment(self: *Self, fragment_id: usize, fragment_round: usize) !void {
        std.log.info("Broadcasting updated Fragment {} round {} to workers...", .{ fragment_id, fragment_round });

        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();

        try payload.put("fragment_id", std.json.Value{ .integer = @intCast(fragment_id) });
        try payload.put("fragment_round", std.json.Value{ .integer = @intCast(fragment_round) });

        // Collect all tensors belonging to this fragment
        var updates = std.json.Array.init(self.allocator);
        defer updates.deinit();

        var it = self.fragments[fragment_id].accumulators.iterator();
        while (it.next()) |entry| {
            const tensor_idx = entry.key_ptr.*;
            const master_raw = self.master_param_raw_data[tensor_idx];

            // Base64 encode the tensor data
            const b64_len = std.base64.standard.Encoder.calcSize(master_raw.len);
            const b64_encoded = try self.allocator.alloc(u8, b64_len);
            defer self.allocator.free(b64_encoded);

            const encoded_len = std.base64.standard.Encoder.encode(b64_encoded, master_raw).len;
            const final_encoded = try self.allocator.dupe(u8, b64_encoded[0..encoded_len]);

            var update = std.json.ObjectMap.init(self.allocator);
            try update.put("tensor_idx", std.json.Value{ .integer = @intCast(tensor_idx) });
            try update.put("data", std.json.Value{ .string = final_encoded });

            try updates.append(std.json.Value{ .object = update });
        }

        try payload.put("updates", std.json.Value{ .array = updates });

        try self.coordinator.broadcastToWorkers(message.MessageType.FRAGMENT_READY, .{ .object = payload });
    }

    fn broadcastStartStreaming(self: *Self, workers: []const WorkerConnection) !void {
        std.log.info("Broadcasting START_STREAMING_LOOP to {} workers...", .{workers.len});

        var total_param_bytes = std.ArrayList(u8).init(self.allocator);
        defer total_param_bytes.deinit();
        var total_size: usize = 0;
        for (self.master_param_raw_data) |bytes| {
            try total_param_bytes.appendSlice(bytes);
            total_size += bytes.len;
        }
        std.log.info("StreamingDiLoCo: Total parameter size: {} bytes", .{total_size});

        // Use chunked transfer for large models (> 50MB), otherwise base64
        const CHUNK_THRESHOLD = 50 * 1024 * 1024; // 50MB
        const use_chunked_transfer = total_param_bytes.items.len > CHUNK_THRESHOLD;

        var b64_encoded: ?[]u8 = null;
        var actual_encoded_len: usize = 0;

        if (use_chunked_transfer) {
            // For large models: broadcast raw parameter bytes via chunked transfer
            std.log.info("Using chunked transfer for {} byte payload (raw bytes)", .{total_param_bytes.items.len});
            try self.coordinator.broadcastLargeData(total_param_bytes.items);
        } else {
            // For small models: use base64 encoding directly
            const b64_len = std.base64.standard.Encoder.calcSize(total_param_bytes.items.len);
            b64_encoded = try self.allocator.alloc(u8, b64_len);
            actual_encoded_len = std.base64.standard.Encoder.encode(b64_encoded.?, total_param_bytes.items).len;
            std.log.info("Using base64 encoding: {} bytes", .{actual_encoded_len});
        }
        defer if (b64_encoded) |b| self.allocator.free(b);

        // Use the actual batch size from config, not the chunk length!
        const micro_batch: usize = 4; // Default batch size for streaming

        for (workers) |worker| {
            const chunk = if (self.coordinator.data_manager) |*dm|
                dm.assignNextChunk(worker.node_id)
            else null;

            if (chunk == null) {
                std.log.warn("No data chunk available for worker {}", .{worker.node_id});
                continue;
            }

            var payload = std.json.ObjectMap.init(self.allocator);
            defer payload.deinit();

            try payload.put("num_fragments", std.json.Value{ .integer = @intCast(self.config.num_fragments) });
            try payload.put("inner_steps", std.json.Value{ .integer = @intCast(self.config.inner_steps) });
            try payload.put("overlap_tau", std.json.Value{ .integer = @intCast(self.config.overlap_tau) });
            try payload.put("alpha", std.json.Value{ .float = @floatCast(self.config.alpha) });
            try payload.put("max_rounds", std.json.Value{ .integer = @intCast(self.config.max_epochs) });

            try payload.put("data_path", std.json.Value{ .string = self.config.data_path });
            try payload.put("tokenizer", std.json.Value{ .string = self.config.tokenizer_type });
            try payload.put("sampling", std.json.Value{ .string = self.config.sampling_type });
            try payload.put("offset", std.json.Value{ .integer = @intCast(chunk.?.offset) });
            try payload.put("length", std.json.Value{ .integer = @intCast(chunk.?.length) });
            try payload.put("chunk_id", std.json.Value{ .integer = @intCast(chunk.?.id) });

            // Send dtype to worker for correct precision handling
            const dtype_str: []const u8 = switch (self.config.dtype) {
                .bf16 => "bf16",
                .f16 => "f16",
                else => "f32",
            };
            try payload.put("dtype", std.json.Value{ .string = dtype_str });

            // Include params based on transfer method
            if (use_chunked_transfer) {
                // Tell worker that weight_blob contains raw bytes
                try payload.put("raw_params", std.json.Value{ .bool = true });
            } else {
                // Small model: include base64 encoded params
                try payload.put("initial_params", std.json.Value{ .string = b64_encoded.?[0..actual_encoded_len] });
            }
            try payload.put("micro_batch", std.json.Value{ .integer = @intCast(micro_batch) });

            try self.coordinator.sendToWorker(worker.node_id, message.MessageType.START_STREAMING_LOOP, .{ .object = payload });
        }
    }
};
