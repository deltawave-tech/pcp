/// Worker node that connects to the gateway worker fabric and runs local tasks.
const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const tcp_stream = @import("../../network/tcp_stream.zig");
const message = @import("../../network/message.zig");
const binary_protocol = @import("../../network/capnp_zig_wrapper.zig");
const worker_backend = @import("../../backends/worker_backend.zig");
const dataset_mod = @import("../../data/dataset.zig");
const data_assignment = @import("../../data/assignment.zig");
const loader = @import("../../data/loader.zig");
const training_window = @import("../../workloads/training/window.zig");
const decoupled_learner = @import("../../algorithms/decoupled_learner.zig");
const GraphBuilder = @import("../../compiler/graph_builder.zig").GraphBuilder;
const math = @import("../../core/math.zig");
const tensor = @import("../../core/tensor.zig");
const file_util = @import("../../protocol/file_util.zig");
const protocol_invariants = @import("../../protocol/invariants.zig");
const protocol_limits = @import("../../protocol/limits.zig");
const IreeBackend = @import("../../backends/iree.zig").IreeBackend;
const generation_engine = @import("generation_engine.zig");
const streamed_weights = @import("streamed_weights.zig");
const task_handlers = @import("task_handlers.zig");
const custom_extensions = @import("custom_extensions.zig");

const ExtensionWorkerState = custom_extensions.ExtensionWorkerState;

const TcpClient = tcp_stream.TcpClient;
const TcpStreamManager = tcp_stream.TcpStreamManager;
const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
const NodeId = message.NodeId;
const WorkerBackend = worker_backend.WorkerBackend;
const Dataset = dataset_mod.Dataset;
const readFileAllocAtPath = file_util.readFileAllocAtPath;

const MAX_INFERENCE_VMFB_BYTES: usize = 10 * 1024 * 1024 * 1024;
const MAX_INFERENCE_WEIGHTS_BYTES: usize = 64 * 1024 * 1024 * 1024;

const FragmentKey = struct {
    request_id: message.RequestId,
    fragment_id: usize,
    fragment_round: usize,
};

const WeightTransferState = struct {
    buffer: std.ArrayList(u8),
    expected_chunks: usize,
    received_chunks: usize,
    total_bytes: usize,

    pub fn init(allocator: Allocator, total_bytes: usize, expected_chunks: usize) !WeightTransferState {
        return .{
            .buffer = try std.ArrayList(u8).initCapacity(allocator, total_bytes),
            .expected_chunks = expected_chunks,
            .received_chunks = 0,
            .total_bytes = total_bytes,
        };
    }

    pub fn deinit(self: *WeightTransferState) void {
        self.buffer.deinit();
    }
};

const DecoupledLoopConfig = struct {
    num_fragments: usize,
    sync_interval_h: usize,
    overlap_tau: usize,
    learner_alpha: f32,
    max_syncer_steps: usize,
    max_local_steps: usize,

    fn validate(self: @This()) !void {
        try protocol_invariants.validateFragmentCount(self.num_fragments);
        protocol_invariants.assertFragmentCount(self.num_fragments);
        if (self.sync_interval_h == 0) return error.InvalidSyncInterval;
        if (self.sync_interval_h > protocol_limits.syncer_steps_max) return error.SyncIntervalTooLarge;
        if (self.overlap_tau > protocol_limits.syncer_steps_max) return error.OverlapTauTooLarge;
        if (self.max_syncer_steps > protocol_limits.syncer_steps_max) return error.SyncerStepsTooLarge;
        if (self.max_local_steps > protocol_limits.local_steps_max) return error.LocalStepsTooLarge;
    }
};

const InnerLoopConfig = struct {
    tau: usize,
    tokenizer_type: []const u8,
    sampling_type: []const u8,
    dtype_name: []const u8,
    param_dtype: tensor.DType,
    micro_batch: usize,
    effective_batch: usize,
    accumulation_steps: usize,
    use_in_graph_accumulation: bool,
    block_size: usize,
};

const InnerLoopWorkspace = struct {
    current_params: [][]u8 = &.{},
    initial_params_f32: [][]u8 = &.{},
    accum_grads: [][]u8 = &.{},
    device_params: []IreeBackend.DeviceBuffer = &.{},
    current_initialized: usize = 0,
    initial_initialized: usize = 0,
    grad_initialized: usize = 0,
    device_initialized: usize = 0,

    fn deinit(self: *@This(), allocator: Allocator) void {
        for (self.device_params[0..self.device_initialized]) |buffer| buffer.release();
        if (self.device_params.len > 0) allocator.free(self.device_params);
        for (self.accum_grads[0..self.grad_initialized]) |buffer| allocator.free(buffer);
        if (self.accum_grads.len > 0) allocator.free(self.accum_grads);
        for (self.initial_params_f32[0..self.initial_initialized]) |buffer| allocator.free(buffer);
        if (self.initial_params_f32.len > 0) allocator.free(self.initial_params_f32);
        for (self.current_params[0..self.current_initialized]) |buffer| allocator.free(buffer);
        if (self.current_params.len > 0) allocator.free(self.current_params);
    }
};

const MaybeOwnedBytes = struct {
    bytes: []const u8,
    owned: ?[]u8 = null,

    fn deinit(self: @This(), allocator: Allocator) void {
        if (self.owned) |owned| allocator.free(owned);
    }
};

const RegularDecoupledRuntime = struct {
    loop_config: DecoupledLoopConfig,
    tokenizer_type: []const u8,
    sampling_type: []const u8,
    micro_batch: usize,
    block_size: usize,
    data_shape: [2]i64,

    fn dataShape(self: *const @This()) *const [2]i64 {
        return &self.data_shape;
    }
};

const DecoupledIoThreads = struct {
    tx_thread: std.Thread,
    rx_thread: std.Thread,

    fn spawn(worker: *Worker) !@This() {
        return .{
            .tx_thread = try std.Thread.spawn(.{}, Worker.txLoop, .{worker}),
            .rx_thread = try std.Thread.spawn(.{}, Worker.rxLoop, .{worker}),
        };
    }

    pub fn stopAndJoin(self: @This(), worker: *Worker) void {
        worker.tx_mutex.lock();
        worker.decoupled_mode = false;
        worker.tx_cond.broadcast();
        worker.tx_mutex.unlock();
        self.tx_thread.join();
        self.rx_thread.join();
    }
};

fn contextFromMessage(msg: MessageEnvelope) message.MessageContext {
    return .{
        .request_id = msg.request_id,
        .round_id = msg.round_id,
        .task_id = msg.task_id,
    };
}

/// Worker state
pub const WorkerState = enum {
    disconnected,
    connecting,
    connected,
    training,
    shutting_down,
};

fn blobTransferKey(allocator: Allocator, request_id: message.RequestId, blob_name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{d}:{s}", .{ request_id, blob_name });
}

fn verifyNamedBlobRef(bytes: []const u8, blob: training_window.BlobRef) !void {
    std.debug.assert(blob.total_bytes > 0);
    if (bytes.len != blob.total_bytes) return error.WeightBlobSizeMismatch;
    std.debug.assert(bytes.len == blob.total_bytes);
    if (std.hash.Wyhash.hash(0, bytes) != blob.byte_hash) return error.ProgramArtifactHashMismatch;
}

fn parseDtypeName(text: []const u8) ?tensor.DType {
    if (std.mem.eql(u8, text, "i64")) return .i64;
    if (std.mem.eql(u8, text, "i32")) return .i32;
    if (std.mem.eql(u8, text, "f64")) return .f64;
    if (std.mem.eql(u8, text, "f32")) return .f32;
    if (std.mem.eql(u8, text, "f16")) return .f16;
    if (std.mem.eql(u8, text, "bf16")) return .bf16;
    if (std.mem.eql(u8, text, "bool")) return .bool;
    return null;
}

fn f32ShapeByteSize(shape: []const i64) !usize {
    var size: usize = @sizeOf(f32);
    for (shape) |dim| {
        if (dim <= 0) return error.InvalidShapeDimension;
        size = std.math.mul(usize, size, @intCast(dim)) catch return error.ShapeByteSizeOverflow;
    }
    return size;
}

fn shapeElementCount(shape: []const i64) !usize {
    var elem_count: usize = 1;
    for (shape) |dim| {
        if (dim <= 0) return error.InvalidShapeDimension;
        elem_count = std.math.mul(usize, elem_count, @intCast(dim)) catch return error.ShapeElementCountOverflow;
    }
    return elem_count;
}

fn f16BitsToF32(bits: u16) f32 {
    const sign: u32 = (@as(u32, bits) & 0x8000) << 16;
    const exp: u32 = (@as(u32, bits) >> 10) & 0x1F;
    const mant: u32 = @as(u32, bits) & 0x3FF;
    const f32_bits: u32 = if (exp == 0)
        sign
    else if (exp == 31)
        sign | 0x7F800000 | (mant << 13)
    else
        sign | ((exp + 112) << 23) | (mant << 13);
    return @bitCast(f32_bits);
}

fn f32ToF16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    const sign: u16 = @truncate((bits >> 16) & 0x8000);
    const exp = @as(i32, @intCast((bits >> 23) & 0xFF)) - 127;
    const mant: u16 = @truncate((bits >> 13) & 0x3FF);
    if (exp < -14) return sign;
    if (exp > 15) return sign | 0x7C00;
    return sign | (@as(u16, @intCast((exp + 15) & 0x1F)) << 10) | mant;
}

fn readScalarLoss(bytes: []const u8) !f32 {
    if (bytes.len == @sizeOf(f32)) {
        const bits = std.mem.readInt(u32, bytes[0..4], .little);
        return @bitCast(bits);
    }
    if (bytes.len == @sizeOf(u16)) {
        const bf16_bits = std.mem.readInt(u16, bytes[0..2], .little);
        return @bitCast(@as(u32, bf16_bits) << 16);
    }
    return error.InvalidLossSize;
}

fn freeOutputSlices(allocator: Allocator, outputs: [][]u8) void {
    for (outputs) |output| allocator.free(output);
    allocator.free(outputs);
}

/// Worker process that connects to the gateway worker fabric.
/// Backend-specific execution goes through the WorkerBackend interface.
pub const Worker = struct {
    allocator: Allocator,
    client: TcpClient,
    node_id: ?NodeId,
    stable_worker_id: []u8,
    state: WorkerState,
    is_running: bool,

    // Backend abstraction - handles all MLIR compilation and execution
    backend: WorkerBackend,
    target_arch: ?[]u8,

    // NEW: Cached VMFB binary and shapes from graph initialization
    cached_vmfb: ?[]u8,
    cached_parameter_shapes: ?[][]i64,
    cached_parameter_dtypes: ?[]tensor.DType,
    cached_data_input_shapes: ?[][]i64,
    cached_data_input_dtypes: ?[]tensor.DType,
    cached_param_dtype: tensor.DType, // dtype for parameters (f32, bf16, etc.)

    // NEW: AdamW optimizer state buffers (M and V) and timestep
    m_states: ?[][]u8, // One buffer per parameter
    v_states: ?[][]u8, // One buffer per parameter
    timestep: f32,

    // NEW: Dataset for chunk-based data loading
    dataset: ?Dataset,
    current_chunk_id: ?usize,

    // Supervisor Pattern: ID of the supervisor managing this worker
    supervisor_id: ?i64,

    // RL / Generation State
    kv_cache: ?[][]u8, // Persistent KV Cache buffers on Host (legacy, to be sent to device)
    sampler: math.Sampler, // Token sampler
    weight_blob: ?[]u8, // Flat weight buffer for RL generation updates.
    generation_weights_path: ?[]u8,
    generation_weights_format: ?[]u8,

    // Generation Engine: shared decoding helpers for RL + inference
    gen_engine: generation_engine.GenerationEngine,
    loaded_models: std.StringHashMap(generation_engine.LoadedInferenceModel),
    program_artifact_cache: std.AutoHashMap(u64, []u8),
    session_kv: std.StringHashMap(generation_engine.GenerationSession),
    active_generations: std.AutoHashMap(message.RequestId, generation_engine.ActiveGenerationState),
    extension_state: ExtensionWorkerState,

    // Serialize all outbound controller traffic on the shared worker socket.
    inference_mutex: std.Thread.Mutex,
    send_mutex: std.Thread.Mutex,
    decoupled_session_mutex: std.Thread.Mutex,

    // Chunked Transfer: Request-scoped state for reassembling incoming weight chunks
    incoming_weight_transfers: std.AutoHashMap(message.RequestId, WeightTransferState),
    completed_weight_blobs: std.AutoHashMap(message.RequestId, []u8),
    incoming_named_weight_transfers: std.StringHashMap(WeightTransferState),
    completed_named_weight_blobs: std.StringHashMap([]u8),
    active_weight_blob_request_id: ?message.RequestId,

    // Streaming DiLoCo State
    streaming_mode: bool,
    streaming_request_id: message.RequestId,
    decoupled_mode: bool,
    decoupled_request_id: message.RequestId,
    decoupled_regular_device_params: ?[]IreeBackend.DeviceBuffer,
    decoupled_regular_last_loss: f32,
    decoupled_learner_state: ?*decoupled_learner.DecoupledLearnerState,
    current_step: usize,
    num_fragments: usize,
    inner_steps: usize, // H
    overlap_tau: usize, // tau
    alpha: f32,

    // Snapshots of parameters to compute Delta = Snapshot - Current
    // We store a snapshot for every tensor. We only update the ones for fragment P when P syncs.
    tensor_snapshots: ?[][]u8,

    // Async I/O State for Streaming DiLoCo
    tx_queue: std.ArrayList(message.MessageEnvelope),
    tx_mutex: std.Thread.Mutex,
    tx_cond: std.Thread.Condition,

    // Stores out-of-order fragment responses until the compute loop needs them.
    rx_map: std.AutoHashMap(FragmentKey, message.MessageEnvelope),
    rx_mutex: std.Thread.Mutex,
    rx_cond: std.Thread.Condition,

    decoupled_control_queue: std.ArrayList(message.MessageEnvelope),
    decoupled_control_mutex: std.Thread.Mutex,

    const Self = @This();

    pub fn init(allocator: Allocator, backend: WorkerBackend, supervisor_id: ?i64, stable_worker_id: []const u8) !Self {
        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .node_id = null,
            .stable_worker_id = try allocator.dupe(u8, stable_worker_id),
            .state = .disconnected,
            .is_running = false,
            .backend = backend,
            .target_arch = null,
            .cached_vmfb = null,
            .cached_parameter_shapes = null,
            .cached_parameter_dtypes = null,
            .cached_data_input_shapes = null,
            .cached_data_input_dtypes = null,
            .cached_param_dtype = .f32, // Default, will be updated from TRAIN message
            .m_states = null,
            .v_states = null,
            .timestep = 1.0,
            .dataset = null,
            .current_chunk_id = null,
            .supervisor_id = supervisor_id,
            .kv_cache = null,
            .sampler = math.Sampler.init(@intCast(std.time.timestamp())),
            .weight_blob = null,
            .generation_weights_path = null,
            .generation_weights_format = null,
            .gen_engine = generation_engine.GenerationEngine.init(allocator),
            .loaded_models = std.StringHashMap(generation_engine.LoadedInferenceModel).init(allocator),
            .program_artifact_cache = std.AutoHashMap(u64, []u8).init(allocator),
            .session_kv = std.StringHashMap(generation_engine.GenerationSession).init(allocator),
            .active_generations = std.AutoHashMap(message.RequestId, generation_engine.ActiveGenerationState).init(allocator),
            .extension_state = ExtensionWorkerState.init(allocator),
            .inference_mutex = std.Thread.Mutex{},
            .send_mutex = std.Thread.Mutex{},
            .decoupled_session_mutex = std.Thread.Mutex{},
            .incoming_weight_transfers = std.AutoHashMap(message.RequestId, WeightTransferState).init(allocator),
            .completed_weight_blobs = std.AutoHashMap(message.RequestId, []u8).init(allocator),
            .incoming_named_weight_transfers = std.StringHashMap(WeightTransferState).init(allocator),
            .completed_named_weight_blobs = std.StringHashMap([]u8).init(allocator),
            .active_weight_blob_request_id = null,
            .streaming_mode = false,
            .streaming_request_id = 0,
            .decoupled_mode = false,
            .decoupled_request_id = 0,
            .decoupled_regular_device_params = null,
            .decoupled_regular_last_loss = 0.0,
            .decoupled_learner_state = null,
            .current_step = 0,
            .num_fragments = 4,
            .inner_steps = 100,
            .overlap_tau = 1,
            .alpha = 0.5,
            .tensor_snapshots = null,
            // Async I/O State for Streaming DiLoCo
            .tx_queue = std.ArrayList(message.MessageEnvelope).init(allocator),
            .tx_mutex = std.Thread.Mutex{},
            .tx_cond = std.Thread.Condition{},
            .rx_map = std.AutoHashMap(FragmentKey, message.MessageEnvelope).init(allocator),
            .rx_mutex = std.Thread.Mutex{},
            .rx_cond = std.Thread.Condition{},
            .decoupled_control_queue = std.ArrayList(message.MessageEnvelope).init(allocator),
            .decoupled_control_mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.stable_worker_id);

        // Cleanup dataset if it exists
        if (self.dataset) |ds| {
            ds.deinit();
        }

        self.freeCachedGraphState();

        // Cleanup optimizer state buffers
        if (self.m_states) |states| {
            for (states) |s| self.allocator.free(s);
            self.allocator.free(states);
        }
        if (self.v_states) |states| {
            for (states) |s| self.allocator.free(s);
            self.allocator.free(states);
        }

        // Cleanup KV cache buffers
        if (self.kv_cache) |cache_buffers| {
            for (cache_buffers) |buf| self.allocator.free(buf);
            self.allocator.free(cache_buffers);
        }

        // Cleanup weight blob
        if (self.weight_blob) |blob| {
            self.allocator.free(blob);
        }
        if (self.generation_weights_path) |path| {
            self.allocator.free(path);
        }

        // Cleanup completed request-scoped weight blobs
        var completed_it = self.completed_weight_blobs.valueIterator();
        while (completed_it.next()) |blob| {
            self.allocator.free(blob.*);
        }
        self.completed_weight_blobs.deinit();

        var completed_named_it = self.completed_named_weight_blobs.iterator();
        while (completed_named_it.next()) |entry| {
            self.allocator.free(@constCast(entry.key_ptr.*));
            self.allocator.free(entry.value_ptr.*);
        }
        self.completed_named_weight_blobs.deinit();

        // Cleanup in-flight request-scoped weight transfers
        var transfer_it = self.incoming_weight_transfers.valueIterator();
        while (transfer_it.next()) |transfer| {
            transfer.deinit();
        }
        self.incoming_weight_transfers.deinit();

        var named_transfer_it = self.incoming_named_weight_transfers.iterator();
        while (named_transfer_it.next()) |entry| {
            self.allocator.free(@constCast(entry.key_ptr.*));
            entry.value_ptr.deinit();
        }
        self.incoming_named_weight_transfers.deinit();

        // Cleanup tensor snapshots (streaming mode)
        if (self.tensor_snapshots) |snapshots| {
            for (snapshots) |s| self.allocator.free(s);
            self.allocator.free(snapshots);
        }

        self.gen_engine.deinit();
        var model_it = self.loaded_models.iterator();
        while (model_it.next()) |entry| {
            self.freeLoadedModel(entry.value_ptr.*);
        }
        self.loaded_models.deinit();

        self.extension_state.deinit(self.allocator);

        var artifact_it = self.program_artifact_cache.valueIterator();
        while (artifact_it.next()) |bytes| {
            self.allocator.free(bytes.*);
        }
        self.program_artifact_cache.deinit();

        var session_it = self.session_kv.iterator();
        while (session_it.next()) |entry| {
            self.freeGenerationSession(entry.value_ptr.*);
        }
        self.session_kv.deinit();
        self.active_generations.deinit();

        // Cleanup async I/O queues
        for (self.tx_queue.items) |*msg| msg.deinitClone(self.allocator);
        self.tx_queue.deinit();

        var rx_it = self.rx_map.iterator();
        while (rx_it.next()) |entry| {
            entry.value_ptr.*.deinitClone(self.allocator);
        }
        self.rx_map.deinit();

        for (self.decoupled_control_queue.items) |*msg| msg.deinitClone(self.allocator);
        self.decoupled_control_queue.deinit();

        self.client.deinit();
        if (self.target_arch) |target| self.allocator.free(target);
        self.backend.deinit();
    }

    pub fn activateWeightBlobForRequest(self: *Self, request_id: message.RequestId) ![]const u8 {
        if (self.active_weight_blob_request_id) |active_request_id| {
            if (active_request_id == request_id) {
                return self.weight_blob orelse error.WeightBlobNotReceived;
            }
        }

        if (self.completed_weight_blobs.fetchRemove(request_id)) |kv| {
            if (self.weight_blob) |old_blob| {
                self.allocator.free(old_blob);
            }
            self.weight_blob = kv.value;
            self.active_weight_blob_request_id = request_id;
            return kv.value;
        }

        return error.WeightBlobNotReceived;
    }

    fn copyRequestScopedBlob(self: *Self, request_id: message.RequestId) ![]u8 {
        const blob = try self.activateWeightBlobForRequest(request_id);
        return try self.allocator.dupe(u8, blob);
    }

    fn freeCachedGraphState(self: *Self) void {
        if (self.cached_vmfb) |vmfb| {
            self.allocator.free(vmfb);
            self.cached_vmfb = null;
        }
        if (self.cached_parameter_shapes) |shapes| {
            for (shapes) |s| self.allocator.free(s);
            self.allocator.free(shapes);
            self.cached_parameter_shapes = null;
        }
        if (self.cached_parameter_dtypes) |dtypes| {
            self.allocator.free(dtypes);
            self.cached_parameter_dtypes = null;
        }
        if (self.cached_data_input_shapes) |shapes| {
            for (shapes) |s| self.allocator.free(s);
            self.allocator.free(shapes);
            self.cached_data_input_shapes = null;
        }
        if (self.cached_data_input_dtypes) |dtypes| {
            self.allocator.free(dtypes);
            self.cached_data_input_dtypes = null;
        }
        if (self.generation_weights_path) |path| {
            self.allocator.free(path);
            self.generation_weights_path = null;
        }
        if (self.generation_weights_format) |format| {
            self.allocator.free(format);
            self.generation_weights_format = null;
        }
    }

    fn cacheProgramArtifact(self: *Self, artifact_key: u64, bytes: []const u8) !void {
        if (artifact_key == 0) return;
        if (self.program_artifact_cache.fetchRemove(artifact_key)) |existing| {
            self.allocator.free(existing.value);
        }
        const owned = try self.allocator.dupe(u8, bytes);
        errdefer self.allocator.free(owned);
        try self.program_artifact_cache.put(artifact_key, owned);
    }

    fn copyCachedProgramArtifact(self: *Self, artifact_key: u64) ![]u8 {
        const bytes = self.program_artifact_cache.get(artifact_key) orelse return error.ProgramArtifactNotMaterialized;
        return try self.allocator.dupe(u8, bytes);
    }

    fn verifyProgramArtifactHash(bytes: []const u8, expected_hash: u64) !void {
        if (bytes.len == 0) return error.EmptyProgramArtifact;
        if (expected_hash == 0) return;
        const actual_hash = std.hash.Wyhash.hash(0, bytes);
        if (actual_hash != expected_hash) return error.ProgramArtifactHashMismatch;
        std.debug.assert(actual_hash == expected_hash);
    }

    fn storeCompletedWeightBlob(self: *Self, request_id: message.RequestId, blob: []u8) !void {
        if (self.completed_weight_blobs.fetchRemove(request_id)) |existing| {
            self.allocator.free(existing.value);
        }
        errdefer self.allocator.free(blob);
        try self.completed_weight_blobs.put(request_id, blob);
    }

    pub fn takeCompletedNamedWeightBlob(self: *Self, request_id: message.RequestId, blob_name: []const u8) ![]u8 {
        const key = try blobTransferKey(self.allocator, request_id, blob_name);
        defer self.allocator.free(key);
        if (self.completed_named_weight_blobs.fetchRemove(key)) |kv| {
            self.allocator.free(@constCast(kv.key));
            return kv.value;
        }
        return error.WeightBlobNotReceived;
    }

    fn resolveDataAssignmentFromPayload(
        self: *Self,
        payload: std.json.ObjectMap,
        parsed_assignment: *?std.json.Parsed(data_assignment.DataAssignment),
    ) !data_assignment.DataAssignment {
        if (payload.get(data_assignment.field_name)) |assignment_value| {
            const assignment_json = try std.json.stringifyAlloc(self.allocator, assignment_value, .{});
            defer self.allocator.free(assignment_json);
            parsed_assignment.* = try std.json.parseFromSlice(
                data_assignment.DataAssignment,
                self.allocator,
                assignment_json,
                .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
            );
            return parsed_assignment.*.?.value;
        }

        const offset = switch (payload.get("offset") orelse return error.MissingDataAssignment) {
            .integer => |i| if (i >= 0) @as(usize, @intCast(i)) else return error.InvalidOffsetFormat,
            else => return error.InvalidOffsetFormat,
        };
        const length = switch (payload.get("length") orelse return error.MissingLength) {
            .integer => |i| if (i >= 0) @as(usize, @intCast(i)) else return error.InvalidLengthFormat,
            else => return error.InvalidLengthFormat,
        };
        const chunk_id = switch (payload.get("chunk_id") orelse return error.MissingChunkId) {
            .integer => |i| if (i >= 0) @as(usize, @intCast(i)) else return error.InvalidChunkIdFormat,
            else => return error.InvalidChunkIdFormat,
        };
        const data_path = switch (payload.get("data_path") orelse return error.MissingDataPath) {
            .string => |s| s,
            else => return error.InvalidDataPathFormat,
        };

        return .{
            .kind = data_assignment.kind_byte_range,
            .provider = data_assignment.provider_local_path,
            .path = data_path,
            .offset = offset,
            .length = length,
            .chunk_id = chunk_id,
        };
    }

    fn dataAssignmentPath(assignment: data_assignment.DataAssignment) ![]const u8 {
        if (std.mem.eql(u8, assignment.provider, data_assignment.provider_local_path)) {
            return assignment.path orelse error.MissingDataPath;
        }
        if (std.mem.eql(u8, assignment.provider, data_assignment.provider_uri)) {
            const uri = assignment.uri orelse return error.MissingDataUri;
            const file_prefix = "file://";
            if (std.mem.startsWith(u8, uri, file_prefix)) return uri[file_prefix.len..];
            return error.UnsupportedDataAssignmentProvider;
        }
        return error.UnsupportedDataAssignmentProvider;
    }

    fn initializeDatasetFromAssignment(
        self: *Self,
        assignment: data_assignment.DataAssignment,
        tokenizer_type: []const u8,
        sampling_type: []const u8,
    ) !void {
        if (!std.mem.eql(u8, assignment.kind, data_assignment.kind_byte_range)) return error.UnsupportedDataAssignmentKind;
        const data_path = try dataAssignmentPath(assignment);
        const offset = assignment.offset;
        const length = assignment.length;
        const chunk_id = assignment.chunk_id;

        self.current_chunk_id = chunk_id;
        std.log.info("Worker {}: Assigned data range chunk {} ({s}, offset={}, length={})", .{ self.node_id.?, chunk_id, assignment.provider, offset, length });

        if (self.dataset) |ds| ds.deinit();

        if (std.mem.eql(u8, tokenizer_type, "byte")) {
            if (!std.mem.eql(u8, sampling_type, "random")) return error.UnsupportedSamplingMode;
            const byte_ds = try loader.ByteTextDataset.initChunk(self.allocator, data_path, offset, length, @intCast(self.node_id.?));
            self.dataset = byte_ds.asDataset();
        } else if (std.mem.eql(u8, tokenizer_type, "char")) {
            if (!std.mem.eql(u8, sampling_type, "random")) return error.UnsupportedSamplingMode;
            const char_ds = try loader.TextDataset.initChunk(self.allocator, data_path, offset, length, @intCast(self.node_id.?));
            self.dataset = char_ds.asDataset();
        } else if (std.mem.eql(u8, tokenizer_type, "u16")) {
            if (std.mem.eql(u8, sampling_type, "fifo")) {
                const u16_ds = try loader.U16TokenDatasetFifo.initChunk(self.allocator, data_path, offset, length, @intCast(self.node_id.?));
                self.dataset = u16_ds.asDataset();
            } else if (std.mem.eql(u8, sampling_type, "random")) {
                const u16_ds = try loader.U16TokenDataset.initChunk(self.allocator, data_path, offset, length, @intCast(self.node_id.?));
                self.dataset = u16_ds.asDataset();
            } else {
                return error.UnsupportedSamplingMode;
            }
        } else {
            std.log.err("Unknown tokenizer type: {s}", .{tokenizer_type});
            return error.UnknownTokenizer;
        }
    }

    fn freeLoadedModel(self: *Self, model: generation_engine.LoadedInferenceModel) void {
        self.allocator.free(model.model_id);
        self.allocator.free(model.cached_vmfb);
        self.allocator.free(model.weights_blob);
        if (model.weights_path) |path| self.allocator.free(path);
        self.allocator.free(model.weights_format);
        if (model.weight_dtypes) |dtypes| self.allocator.free(dtypes);
        for (model.param_shapes) |shape| self.allocator.free(shape);
        self.allocator.free(model.param_shapes);
        for (model.data_shapes) |shape| self.allocator.free(shape);
        self.allocator.free(model.data_shapes);
        if (model.data_dtypes) |dtypes| self.allocator.free(dtypes);
    }

    fn freeGenerationSession(self: *Self, session: generation_engine.GenerationSession) void {
        self.allocator.free(session.session_id);
        self.allocator.free(session.model_id);
        if (session.kv_cache_host) |cache| {
            for (cache) |buf| self.allocator.free(buf);
            self.allocator.free(cache);
        }
    }

    fn sendControllerMessage(self: *Self, msg: MessageEnvelope) !void {
        self.send_mutex.lock();
        defer self.send_mutex.unlock();
        try self.client.send(msg);
    }

    fn sendInferenceMessage(self: *Self, msg: MessageEnvelope) !void {
        try self.sendControllerMessage(msg);
    }

    /// Connect to the gateway worker-fabric controller.
    pub fn connect(self: *Self, master_host: []const u8, master_port: u16, target_arch: ?[]const u8) !void {
        self.state = .connecting;
        try self.setTargetArch(target_arch);

        // Connect to the master
        try self.client.connect(master_host, master_port);
        std.log.info("Connected to controller at {s}:{}", .{ master_host, master_port });

        // Get our backend type from the backend instance
        const my_backend = self.backend.getBackendType();

        // Create a JSON object for the payload
        var payload_map = std.json.ObjectMap.init(self.allocator);
        try payload_map.put("backend", std.json.Value{ .string = my_backend.toString() });
        try payload_map.put("worker_id", std.json.Value{ .string = self.stable_worker_id });
        std.log.info("Reporting worker ID: {s}", .{self.stable_worker_id});

        // Add target architecture if specified
        if (self.target_arch) |target| {
            try payload_map.put("target_arch", std.json.Value{ .string = target });
            std.log.info("Reporting target architecture: {s}", .{target});
        }

        // Add supervisor ID if this worker is managed by a supervisor
        if (self.supervisor_id) |sid| {
            try payload_map.put("supervisor_id", std.json.Value{ .integer = @intCast(sid) });
            std.log.info("Reporting supervisor ID: {}", .{sid});
        }

        const payload = std.json.Value{ .object = payload_map };

        // Send JoinRequest with the backend information in the payload
        const join_request = tcp_stream.createMessage(
            0, // temporary node_id, assigned by the worker fabric
            "worker", // our service name
            0, // worker-fabric node_id
            "worker_fabric",
            MessageType.JOIN_REQUEST,
            1, // message id
            payload, // Use the new payload
        );

        try self.sendControllerMessage(join_request);

        // Wait for JoinAccept
        std.log.debug("Worker waiting for JoinAccept response from worker fabric...", .{});
        const join_accept_result = self.client.receive() catch |err| {
            std.log.err("Failed to receive JoinAccept from worker fabric: {}", .{err});
            return err;
        };
        defer join_accept_result.parsed.deinit();
        defer self.allocator.free(join_accept_result.buffer); // Free the buffer
        const join_accept = join_accept_result.parsed.value;

        if (!std.mem.eql(u8, join_accept.msg_type, MessageType.JOIN_ACCEPT)) {
            return error.UnexpectedMessage;
        }

        // Extract assigned node_id from response
        self.node_id = join_accept.recipient_node;
        self.state = .connected;
        self.is_running = true;
    }

    /// Main worker loop - listen for commands from the worker fabric.
    pub fn run(self: *Self) !void {
        if (self.state != .connected) {
            return error.NotConnected;
        }

        std.log.info("Worker {} entering main loop", .{self.node_id.?});

        while (self.is_running) {
            const receive_result = self.client.receive() catch |err| {
                std.log.err("Failed to receive worker-fabric message: {}", .{err});
                self.state = .disconnected;
                return;
            };
            defer receive_result.parsed.deinit();
            defer self.allocator.free(receive_result.buffer); // Free the buffer
            const msg = receive_result.parsed.value;

            try self.dispatchTaskMessage(msg);
        }

        std.log.info("Worker {} exiting main loop", .{self.node_id.?});
    }

    fn dispatchTaskMessage(self: *Self, msg: MessageEnvelope) !void {
        const handled = switch (task_handlers.familyForMessageType(msg.msg_type)) {
            .graph => try task_handlers.graph.dispatch(self, msg),
            .transfer => try task_handlers.transfer.dispatch(self, msg),
            .regular_training => try task_handlers.regular_training.dispatch(self, msg),
            .decoupled_training => try task_handlers.decoupled_training.dispatch(self, msg),
            .streaming_diloco => try task_handlers.streaming_diloco.dispatch(self, msg),
            .rl => try task_handlers.rl.dispatch(self, msg),
            .inference => try task_handlers.inference.dispatch(self, msg),
            .custom_extension => try custom_extensions.dispatchTask(self, msg),
            .lifecycle => try task_handlers.lifecycle.dispatch(self, msg),
            .unknown => false,
        };
        if (!handled) std.log.warn("Unknown message type: {s}", .{msg.msg_type});
    }

    /// Robust entry point with automatic reconnection on failure
    pub fn runRobust(self: *Self, host: []const u8, port: u16, target_arch: ?[]const u8) !void {
        var backoff_ms: u64 = 100;
        const max_backoff_ms: u64 = 5000; // Cap at 5 seconds

        while (self.state != .shutting_down) {
            // Reset state on reconnect attempts
            self.state = .connecting;

            self.connect(host, port, target_arch) catch |err| {
                // Only log warning if backoff is significant to reduce noise during quick restarts
                if (backoff_ms > 500) {
                    std.log.warn("Worker failed to connect to worker fabric: {}. Retrying in {}ms...", .{ err, backoff_ms });
                }
                std.time.sleep(backoff_ms * std.time.ns_per_ms);
                backoff_ms = @min(backoff_ms * 2, max_backoff_ms);
                continue;
            };

            // Connected! Reset backoff
            backoff_ms = 100;

            self.run() catch |err| {
                std.log.err("Worker connection dropped: {}", .{err});
            };

            if (self.state == .shutting_down) break;

            // Important: Explicitly disconnect client to close socket fd
            self.client.disconnect();
            self.state = .disconnected;

            // Wait a moment before immediate reconnect to allow worker fabric to recover.
            std.time.sleep(500 * std.time.ns_per_ms);
        }
    }

    /// Handle incoming weight chunks from chunked transfer protocol
    pub fn handleWeightChunk(self: *Self, msg: MessageEnvelope) !void {
        const payload = switch (msg.data) {
            .object => |o| o,
            else => return error.InvalidFormat,
        };

        const request_id = msg.request_id;
        const chunk_index: usize = @intCast(payload.get("chunk_index").?.integer);
        const total_chunks: usize = @intCast(payload.get("total_chunks").?.integer);
        const total_bytes: usize = @intCast(payload.get("total_bytes").?.integer);
        const b64_data = payload.get("data").?.string;
        if (payload.get("blob_name")) |name_value| {
            const blob_name = switch (name_value) {
                .string => |value| value,
                else => return error.InvalidBlobName,
            };
            return try self.handleNamedWeightChunk(request_id, blob_name, chunk_index, total_chunks, total_bytes, b64_data);
        }

        // Initialize buffer on first chunk
        if (chunk_index == 0) {
            if (self.incoming_weight_transfers.fetchRemove(request_id)) |kv| {
                var stale_transfer = kv.value;
                stale_transfer.deinit();
            }
            if (self.completed_weight_blobs.fetchRemove(request_id)) |kv| {
                self.allocator.free(kv.value);
            }

            const transfer = try WeightTransferState.init(self.allocator, total_bytes, total_chunks);
            try self.incoming_weight_transfers.put(request_id, transfer);
            std.log.info("Worker {}: Started receiving request-scoped weights for request {} ({} bytes in {} chunks)", .{
                self.node_id.?,
                request_id,
                total_bytes,
                total_chunks,
            });
        }

        const transfer = self.incoming_weight_transfers.getPtr(request_id) orelse return error.MissingChunkState;

        // Decode and append chunk
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_data);
        const decoded = try self.allocator.alloc(u8, decoded_len);
        defer self.allocator.free(decoded);
        try std.base64.standard.Decoder.decode(decoded, b64_data);

        try transfer.buffer.appendSlice(decoded);
        transfer.received_chunks += 1;

        if (transfer.received_chunks % 10 == 0 or transfer.received_chunks == transfer.expected_chunks) {
            std.log.info("Worker {}: Received chunk {}/{} for request {}", .{
                self.node_id.?,
                transfer.received_chunks,
                transfer.expected_chunks,
                request_id,
            });
        }

        // Finalize when all chunks received
        if (transfer.received_chunks == transfer.expected_chunks) {
            if (self.incoming_weight_transfers.fetchRemove(request_id)) |kv| {
                var completed_transfer = kv.value;
                errdefer completed_transfer.deinit();
                const blob = try completed_transfer.buffer.toOwnedSlice();
                try self.storeCompletedWeightBlob(request_id, blob);
                std.log.info("Worker {}: Reassembly complete for request {}. {} bytes received.", .{
                    self.node_id.?,
                    request_id,
                    blob.len,
                });
            } else {
                return error.MissingChunkState;
            }
        }
    }

    fn handleNamedWeightChunk(
        self: *Self,
        request_id: message.RequestId,
        blob_name: []const u8,
        chunk_index: usize,
        total_chunks: usize,
        total_bytes: usize,
        b64_data: []const u8,
    ) !void {
        const key = try blobTransferKey(self.allocator, request_id, blob_name);
        defer self.allocator.free(key);

        if (chunk_index == 0) {
            if (self.incoming_named_weight_transfers.fetchRemove(key)) |kv| {
                self.allocator.free(@constCast(kv.key));
                var stale_transfer = kv.value;
                stale_transfer.deinit();
            }
            if (self.completed_named_weight_blobs.fetchRemove(key)) |kv| {
                self.allocator.free(@constCast(kv.key));
                self.allocator.free(kv.value);
            }

            {
                const owned_key = try self.allocator.dupe(u8, key);
                errdefer self.allocator.free(owned_key);
                var transfer = try WeightTransferState.init(self.allocator, total_bytes, total_chunks);
                errdefer transfer.deinit();
                try self.incoming_named_weight_transfers.put(owned_key, transfer);
            }
            std.log.info("Worker {}: Started receiving request-scoped blob {s} for request {} ({} bytes in {} chunks)", .{
                self.node_id.?,
                blob_name,
                request_id,
                total_bytes,
                total_chunks,
            });
        }

        const transfer = self.incoming_named_weight_transfers.getPtr(key) orelse return error.MissingChunkState;

        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_data);
        const decoded = try self.allocator.alloc(u8, decoded_len);
        defer self.allocator.free(decoded);
        try std.base64.standard.Decoder.decode(decoded, b64_data);

        try transfer.buffer.appendSlice(decoded);
        transfer.received_chunks += 1;

        if (transfer.received_chunks == transfer.expected_chunks) {
            if (self.incoming_named_weight_transfers.fetchRemove(key)) |kv| {
                var completed_transfer = kv.value;
                errdefer {
                    self.allocator.free(@constCast(kv.key));
                    completed_transfer.deinit();
                }
                const blob = try completed_transfer.buffer.toOwnedSlice();
                errdefer self.allocator.free(blob);
                try self.completed_named_weight_blobs.put(kv.key, blob);
                std.log.info("Worker {}: Reassembly complete for request {} blob {s}. {} bytes received.", .{
                    self.node_id.?,
                    request_id,
                    blob_name,
                    blob.len,
                });
            } else {
                return error.MissingChunkState;
            }
        }
    }

    /// Handles the one-time setup message, caching the VMFB binary and shapes
    pub fn handleInitializeGraph(self: *Self, msg: MessageEnvelope) !void {
        std.log.info("Worker {} received initialize-graph request", .{self.node_id.?});

        self.freeCachedGraphState();

        // 1. Parse the JSON payload object
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };

        const artifact_key = try self.cacheVmfbFromInitializePayload(msg.request_id, payload);
        self.cached_parameter_shapes = try self.parseShapeListField(
            payload,
            "parameter_shapes",
            error.MissingParameterShapesField,
            error.InvalidParameterShapesFormat,
            error.InvalidParameterShapeFormat,
            error.InvalidParameterDimensionFormat,
        );
        if (payload.get("parameter_dtypes")) |dtypes_json| {
            self.cached_parameter_dtypes = try self.parseDtypeList(dtypes_json);
        }
        self.cached_data_input_shapes = try self.parseShapeListField(
            payload,
            "data_input_shapes",
            error.MissingShapesField,
            error.InvalidShapesFormat,
            error.InvalidShapeFormat,
            error.InvalidDimensionFormat,
        );

        // 4.5. Parse and cache the data input dtypes
        if (payload.get("data_input_dtypes")) |dtypes_json| {
            self.cached_data_input_dtypes = try self.parseDtypeList(dtypes_json);
        }

        // 5. Load weights from local path if provided (for RL generation workers)
        var is_generation_only = false;
        if (payload.get("weights_path")) |weights_path_val| {
            const weights_path = switch (weights_path_val) {
                .string => |s| s,
                else => return error.InvalidWeightsPathFormat,
            };
            const weights_format = if (payload.get("weights_format")) |format_val| switch (format_val) {
                .string => |s| s,
                else => return error.InvalidWeightsFormat,
            } else "flat";
            std.log.info("Loading initial weights from local path: {s} ({s})", .{ weights_path, weights_format });

            try self.loadGenerationWeights(weights_path, weights_format);
            self.gen_engine.markWeightsDirty();
            is_generation_only = true;
        }

        // 6. Allocate M and V state buffers (skip for generation-only workers)
        if (!is_generation_only) {
            try self.resetOptimizerStateBuffers(self.cached_parameter_shapes.?);
        }

        if (is_generation_only) {
            std.log.info("Worker {} cached VMFB, {} parameter shapes, {} data input shapes (generation-only).", .{ self.node_id.?, self.cached_parameter_shapes.?.len, self.cached_data_input_shapes.?.len });
        } else {
            std.log.info("Worker {} cached VMFB, {} parameter shapes, {} data input shapes, and allocated optimizer state buffers.", .{ self.node_id.?, self.cached_parameter_shapes.?.len, self.cached_data_input_shapes.?.len });
        }

        const program_key = if (payload.get("program_key")) |value| try parseProgramKeyValueStrict(value) else 0;

        var response_payload = std.json.ObjectMap.init(self.allocator);
        defer response_payload.deinit();
        const program_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{program_key});
        defer self.allocator.free(program_key_text);
        try response_payload.put("ok", .{ .bool = true });
        try response_payload.put("program_key", .{ .string = program_key_text });
        var response_artifact_key_text: ?[]u8 = null;
        defer if (response_artifact_key_text) |text| self.allocator.free(text);
        if (artifact_key != 0) {
            response_artifact_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{artifact_key});
            try response_payload.put("artifact_key", .{ .string = response_artifact_key_text.? });
        }

        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.INITIALIZE_GRAPH_COMPLETE,
            msg.msg_id + 1,
            contextFromMessage(msg),
            .{ .object = response_payload },
        );
        try self.sendControllerMessage(response);
        std.log.info("Worker {} finished initialize-graph request", .{self.node_id.?});
    }

    fn cacheVmfbFromInitializePayload(self: *Self, request_id: message.RequestId, payload: std.json.ObjectMap) !u64 {
        const artifact_key = if (payload.get("artifact_key")) |value| try parseProgramKeyValueStrict(value) else 0;
        const artifact_byte_hash = if (payload.get("artifact_byte_hash")) |value| try parseProgramKeyValueStrict(value) else 0;
        const vmfb_from_blob = if (payload.get("vmfb_blob")) |blob_val| switch (blob_val) {
            .bool => |value| value,
            else => false,
        } else false;
        const vmfb_from_artifact_ref = if (payload.get("vmfb_artifact_ref")) |ref_val| switch (ref_val) {
            .bool => |value| value,
            else => false,
        } else false;

        if (vmfb_from_artifact_ref) {
            if (artifact_key == 0) return error.MissingProgramArtifactKey;
            const vmfb_bytes = try self.copyCachedProgramArtifact(artifact_key);
            errdefer self.allocator.free(vmfb_bytes);
            try verifyProgramArtifactHash(vmfb_bytes, artifact_byte_hash);
            self.cached_vmfb = vmfb_bytes;
            std.log.info("✓ Loaded VMFB from program artifact cache (artifact={d}, {} bytes)", .{ artifact_key, vmfb_bytes.len });
            return artifact_key;
        }

        if (vmfb_from_blob) {
            const vmfb_bytes = try self.copyRequestScopedBlob(request_id);
            errdefer self.allocator.free(vmfb_bytes);
            try verifyProgramArtifactHash(vmfb_bytes, artifact_byte_hash);
            try self.cacheProgramArtifact(artifact_key, vmfb_bytes);
            self.cached_vmfb = vmfb_bytes;
            std.log.info("✓ Loaded VMFB from request-scoped blob (artifact={d}, {} bytes)", .{ artifact_key, vmfb_bytes.len });
            return artifact_key;
        }

        if (payload.get("vmfb_path")) |vmfb_path_val| {
            const vmfb_path = switch (vmfb_path_val) {
                .string => |value| value,
                else => return error.InvalidVmfbPathFormat,
            };
            std.log.info("Loading VMFB from path: {s}", .{vmfb_path});
            const vmfb_bytes = try std.fs.cwd().readFileAlloc(self.allocator, vmfb_path, 10 * 1024 * 1024 * 1024);
            errdefer self.allocator.free(vmfb_bytes);
            self.cached_vmfb = vmfb_bytes;
            std.log.info("✓ Loaded VMFB from disk ({} bytes)", .{vmfb_bytes.len});
            return artifact_key;
        }

        if (payload.get("vmfb")) |b64_vmfb_val| {
            const vmfb_string = switch (b64_vmfb_val) {
                .string => |value| value,
                else => return error.InvalidVmfbFormat,
            };
            const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(vmfb_string);
            const vmfb_bytes = try self.allocator.alloc(u8, decoded_len);
            errdefer self.allocator.free(vmfb_bytes);
            try std.base64.standard.Decoder.decode(vmfb_bytes, vmfb_string);
            self.cached_vmfb = vmfb_bytes;
            std.log.info("✓ Decoded VMFB from base64 ({} bytes)", .{vmfb_bytes.len});
            return artifact_key;
        }

        return error.MissingVmfbField;
    }

    fn parseShapeListField(
        self: *Self,
        payload: std.json.ObjectMap,
        field_name: []const u8,
        missing_error: anyerror,
        invalid_outer_error: anyerror,
        invalid_shape_error: anyerror,
        invalid_dim_error: anyerror,
    ) ![][]i64 {
        const shapes_json = payload.get(field_name) orelse return missing_error;
        const shapes_array = switch (shapes_json) {
            .array => |array| array,
            else => return invalid_outer_error,
        };
        if (shapes_array.items.len > protocol_limits.tensor_count_max) return error.TensorCountTooLarge;

        var shapes_list = std.ArrayList([]i64).init(self.allocator);
        errdefer {
            for (shapes_list.items) |shape| self.allocator.free(shape);
            shapes_list.deinit();
        }

        for (shapes_array.items) |shape_val| {
            const dim_array = switch (shape_val) {
                .array => |array| array,
                else => return invalid_shape_error,
            };
            var dim_list = std.ArrayList(i64).init(self.allocator);
            errdefer dim_list.deinit();
            for (dim_array.items) |dim_val| {
                const dim = switch (dim_val) {
                    .integer => |value| value,
                    else => return invalid_dim_error,
                };
                if (dim <= 0) return error.InvalidShapeDimension;
                try dim_list.append(dim);
            }
            const dims = try dim_list.toOwnedSlice();
            errdefer self.allocator.free(dims);
            try shapes_list.append(dims);
        }

        return try shapes_list.toOwnedSlice();
    }

    fn parseDtypeList(self: *Self, dtypes_json: std.json.Value) ![]tensor.DType {
        const dtypes_array = switch (dtypes_json) {
            .array => |array| array,
            else => return error.InvalidDTypesFormat,
        };
        if (dtypes_array.items.len > protocol_limits.tensor_count_max) return error.TensorCountTooLarge;

        var dtypes_list = std.ArrayList(tensor.DType).init(self.allocator);
        errdefer dtypes_list.deinit();
        for (dtypes_array.items) |value| {
            const text = switch (value) {
                .string => |inner| inner,
                else => return error.InvalidDTypeFormat,
            };
            try dtypes_list.append(parseDtypeName(text) orelse return error.InvalidDType);
        }
        return try dtypes_list.toOwnedSlice();
    }

    fn loadGenerationWeights(self: *Self, weights_path: []const u8, weights_format: []const u8) !void {
        if (self.weight_blob) |old_blob| self.allocator.free(old_blob);
        self.weight_blob = null;
        if (self.generation_weights_path) |old_path| self.allocator.free(old_path);
        self.generation_weights_path = null;
        if (self.generation_weights_format) |old_format| self.allocator.free(old_format);
        self.generation_weights_format = null;
        self.active_weight_blob_request_id = null;

        const stream_weights =
            std.mem.eql(u8, weights_format, "streamed") or
            std.mem.eql(u8, weights_format, "bf16_streamed") or
            streamed_weights.isManifestFormat(weights_format);
        if (stream_weights) {
            self.weight_blob = try self.allocator.alloc(u8, 0);
            self.generation_weights_path = try self.allocator.dupe(u8, weights_path);
            self.generation_weights_format = try self.allocator.dupe(u8, weights_format);
            std.log.info("✓ Registered streamed generation weights for local loading: {s}", .{weights_path});
            return;
        }

        const file = try std.fs.cwd().openFile(weights_path, .{ .mode = .read_only });
        defer file.close();

        const stat = try file.stat();
        if (stat.size == 0) return error.EmptyWeightsFile;
        const weight_bytes = try self.allocator.alloc(u8, stat.size);
        errdefer self.allocator.free(weight_bytes);

        const bytes_read = try file.readAll(weight_bytes);
        if (bytes_read != stat.size) return error.IncompleteWeightsRead;
        self.weight_blob = weight_bytes;
        std.log.info("✓ Loaded initial weights locally ({} MB)", .{stat.size / (1024 * 1024)});
    }

    fn resetOptimizerStateBuffers(self: *Self, parameter_shapes: [][]i64) !void {
        if (self.m_states) |states| {
            for (states) |state| self.allocator.free(state);
            self.allocator.free(states);
            self.m_states = null;
        }
        if (self.v_states) |states| {
            for (states) |state| self.allocator.free(state);
            self.allocator.free(states);
            self.v_states = null;
        }

        const num_params = parameter_shapes.len;
        const m_buffers = try self.allocator.alloc([]u8, num_params);
        errdefer {
            for (m_buffers) |buffer| self.allocator.free(buffer);
            self.allocator.free(m_buffers);
        }
        const v_buffers = try self.allocator.alloc([]u8, num_params);
        errdefer {
            for (v_buffers) |buffer| self.allocator.free(buffer);
            self.allocator.free(v_buffers);
        }

        for (parameter_shapes, 0..) |shape, index| {
            const size = try f32ShapeByteSize(shape);
            m_buffers[index] = try self.allocator.alloc(u8, size);
            @memset(m_buffers[index], 0);
            v_buffers[index] = try self.allocator.alloc(u8, size);
            @memset(v_buffers[index], 0);
        }

        self.m_states = m_buffers;
        self.v_states = v_buffers;
        self.timestep = 1.0;
    }

    fn parseProgramKeyValueStrict(value: std.json.Value) !u64 {
        return switch (value) {
            .integer => |inner| blk: {
                if (inner < 0) return error.InvalidProgramArtifactKey;
                break :blk @intCast(inner);
            },
            .string => |inner| std.fmt.parseUnsigned(u64, inner, 10) catch error.InvalidProgramArtifactKey,
            else => error.InvalidProgramArtifactKey,
        };
    }

    /// Helper: Validates numerical stability of a buffer based on dtype
    /// Returns error if NaN or Inf is detected. Handles unaligned reads safely.
    fn validateBuffer(self: *Self, name: []const u8, bytes: []const u8, dtype: tensor.DType) !void {
        _ = self;
        var nan_count: usize = 0;
        var inf_count: usize = 0;
        var max_val: f64 = 0.0;
        var sum_abs: f64 = 0.0;
        const num_elements = bytes.len / dtype.sizeInBytes();

        // Only validate floating point types
        if (dtype != .f32 and dtype != .bf16 and dtype != .f16) return;

        if (dtype == .f32) {
            var i: usize = 0;
            while (i < bytes.len) : (i += 4) {
                if (i + 4 > bytes.len) break;
                // Safe unaligned read
                const u32_val = std.mem.readInt(u32, bytes[i..][0..4], .little);
                const val: f32 = @bitCast(u32_val);

                if (std.math.isNan(val)) nan_count += 1;
                if (std.math.isInf(val)) inf_count += 1;
                const abs_v = @abs(val);
                if (abs_v > max_val) max_val = abs_v;
                sum_abs += abs_v;
            }
        } else if (dtype == .bf16) {
            var i: usize = 0;
            while (i < bytes.len) : (i += 2) {
                if (i + 2 > bytes.len) break;
                // Safe unaligned read
                const u16_val = std.mem.readInt(u16, bytes[i..][0..2], .little);
                // bf16 to f32 conversion: shift left 16 bits
                const u32_val = @as(u32, u16_val) << 16;
                const val: f32 = @bitCast(u32_val);

                if (std.math.isNan(val)) nan_count += 1;
                if (std.math.isInf(val)) inf_count += 1;
                const abs_v = @abs(val);
                if (abs_v > max_val) max_val = abs_v;
                sum_abs += abs_v;
            }
        }

        if (nan_count > 0 or inf_count > 0) {
            std.log.err("VALIDATION FAILED for {s}: {} NaNs, {} Infs (Max: {d:.4}, Avg: {d:.4})", .{ name, nan_count, inf_count, max_val, sum_abs / @as(f64, @floatFromInt(num_elements)) });
            return error.NumericalInstabilityDetected;
        }
    }

    fn accumulateGradients(dst: []u8, src: []const u8) void {
        const dst_f32 = std.mem.bytesAsSlice(f32, dst);
        if (@intFromPtr(src.ptr) % 4 == 0) {
            const src_f32 = std.mem.bytesAsSlice(f32, src);
            for (dst_f32, src_f32) |*d, s| d.* += s;
        } else {
            var i: usize = 0;
            while (i < src.len) : (i += 4) {
                const val: f32 = @bitCast(std.mem.readInt(u32, src[i..][0..4], .little));
                dst_f32[i / 4] += val;
            }
        }
    }

    fn scaleBuffer(buf: []u8, scale: f32) void {
        const slice = std.mem.bytesAsSlice(f32, buf);
        for (slice) |*v| v.* *= scale;
    }

    fn parseInnerLoopConfig(payload: std.json.ObjectMap, data_shapes: [][]i64) !InnerLoopConfig {
        if (data_shapes.len == 0) return error.MissingShapesField;
        if (data_shapes[0].len < 2) return error.InvalidShapesFormat;

        const tau = (try parsePayloadUsize(payload, "tau")) orelse return error.MissingTau;
        if (tau == 0) return error.InvalidTauFormat;
        if (tau > protocol_limits.local_steps_max) return error.LocalStepsTooLarge;

        const tokenizer_type = if (payload.get("tokenizer")) |value| switch (value) {
            .string => |text| text,
            else => return error.InvalidTokenizerFormat,
        } else "char";
        const sampling_type = if (payload.get("sampling")) |value| switch (value) {
            .string => |text| text,
            else => return error.InvalidSamplingFormat,
        } else "random";
        const dtype_name = if (payload.get("dtype")) |value| switch (value) {
            .string => |text| text,
            else => return error.InvalidDTypeFormat,
        } else "f32";
        const param_dtype = parseDtypeName(dtype_name) orelse return error.InvalidDTypeFormat;

        var micro_batch: usize = @intCast(data_shapes[0][0]);
        if (micro_batch > 1_000_000) {
            micro_batch = 64;
        }
        if (micro_batch == 0) return error.InvalidBatchSize;

        const effective_batch = (try parsePayloadUsize(payload, "effective_batch_size")) orelse micro_batch;
        if (effective_batch == 0) return error.InvalidBatchSize;
        if (effective_batch % micro_batch != 0) return error.InvalidBatchSize;

        const use_in_graph_accumulation = if (payload.get("use_in_graph_accumulation")) |value| switch (value) {
            .bool => |enabled| enabled,
            else => return error.InvalidAccumulationMode,
        } else false;

        return .{
            .tau = tau,
            .tokenizer_type = tokenizer_type,
            .sampling_type = sampling_type,
            .dtype_name = dtype_name,
            .param_dtype = param_dtype,
            .micro_batch = micro_batch,
            .effective_batch = effective_batch,
            .accumulation_steps = effective_batch / micro_batch,
            .use_in_graph_accumulation = use_in_graph_accumulation,
            .block_size = @intCast(data_shapes[0][1]),
        };
    }

    fn loadInitialInnerLoopParams(self: *Self, request_id: message.RequestId, payload: std.json.ObjectMap, arena: Allocator) ![]const u8 {
        const using_raw_params = if (payload.get("raw_params")) |value| switch (value) {
            .bool => |enabled| enabled,
            else => return error.InvalidParamsFormat,
        } else false;

        if (using_raw_params) return try self.activateWeightBlobForRequest(request_id);

        const b64_params = switch (payload.get("params") orelse return error.MissingParams) {
            .string => |text| text,
            else => return error.InvalidParamsFormat,
        };
        const capnp_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_params);
        const decoded = try arena.alloc(u8, capnp_len);
        try std.base64.standard.Decoder.decode(decoded, b64_params);
        const reader = try binary_protocol.WorkerPayload.Reader.init(decoded);
        defer reader.deinit();
        const params_from_reader = try reader.getParams();
        const params_copy = try arena.alloc(u8, params_from_reader.len);
        @memcpy(params_copy, params_from_reader);
        return params_copy;
    }

    fn copyParameterBlockAsF32(
        dtype: tensor.DType,
        src_bytes: []const u8,
        current: []u8,
        initial: []u8,
        elem_count: usize,
    ) void {
        const current_f32 = std.mem.bytesAsSlice(f32, current);
        const initial_f32 = std.mem.bytesAsSlice(f32, initial);
        switch (dtype) {
            .bf16 => {
                for (0..elem_count) |index| {
                    const bits = std.mem.readInt(u16, src_bytes[index * 2 ..][0..2], .little);
                    const value: f32 = @bitCast(@as(u32, bits) << 16);
                    current_f32[index] = value;
                    initial_f32[index] = value;
                }
            },
            .f16 => {
                for (0..elem_count) |index| {
                    const bits = std.mem.readInt(u16, src_bytes[index * 2 ..][0..2], .little);
                    const value = f16BitsToF32(bits);
                    current_f32[index] = value;
                    initial_f32[index] = value;
                }
            },
            else => {
                @memcpy(current, src_bytes);
                @memcpy(initial, src_bytes);
            },
        }
    }

    fn initInnerLoopWorkspace(
        self: *Self,
        iree_impl: *IreeBackend,
        param_shapes: [][]i64,
        initial_params: []const u8,
        dtype: tensor.DType,
    ) !InnerLoopWorkspace {
        var workspace = InnerLoopWorkspace{};
        errdefer workspace.deinit(self.allocator);

        workspace.current_params = try self.allocator.alloc([]u8, param_shapes.len);
        workspace.initial_params_f32 = try self.allocator.alloc([]u8, param_shapes.len);
        workspace.accum_grads = try self.allocator.alloc([]u8, param_shapes.len);
        workspace.device_params = try self.allocator.alloc(IreeBackend.DeviceBuffer, param_shapes.len);

        var param_offset: usize = 0;
        const src_bytes_per_element = dtype.sizeInBytes();
        for (param_shapes, 0..) |shape, index| {
            const elem_count = try shapeElementCount(shape);
            const src_size = try std.math.mul(usize, elem_count, src_bytes_per_element);
            const f32_size = try std.math.mul(usize, elem_count, @sizeOf(f32));
            if (param_offset + src_size > initial_params.len) return error.ParameterBufferTooSmall;

            workspace.current_params[index] = try self.allocator.alloc(u8, f32_size);
            workspace.current_initialized += 1;
            workspace.initial_params_f32[index] = try self.allocator.alloc(u8, f32_size);
            workspace.initial_initialized += 1;
            workspace.accum_grads[index] = try self.allocator.alloc(u8, f32_size);
            workspace.grad_initialized += 1;

            const src = initial_params[param_offset .. param_offset + src_size];
            copyParameterBlockAsF32(dtype, src, workspace.current_params[index], workspace.initial_params_f32[index], elem_count);
            param_offset += src_size;
        }
        if (param_offset != initial_params.len) return error.ParameterBufferSizeMismatch;

        for (workspace.current_params, 0..) |param_bytes, index| {
            workspace.device_params[index] = try iree_impl.moveToDevice(param_bytes, param_shapes[index], .f32);
            workspace.device_initialized += 1;
        }
        return workspace;
    }

    fn releaseDeviceBufferSlice(buffers: []IreeBackend.DeviceBuffer) void {
        for (buffers) |buffer| buffer.release();
    }

    fn computeInGraphAccumulatedGradients(
        self: *Self,
        vmfb: []const u8,
        iree_impl: *IreeBackend,
        workspace: *InnerLoopWorkspace,
        config: InnerLoopConfig,
    ) !f32 {
        const full_batch_size = config.micro_batch * config.accumulation_steps;
        const full_batch = try self.dataset.?.getBatch(full_batch_size, config.block_size);
        defer full_batch.deinit();

        var grad_inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
        defer grad_inputs.deinit();
        for (workspace.device_params) |param| {
            param.retain();
            try grad_inputs.append(param);
        }

        const folded_shape = &[_]i64{ @intCast(config.accumulation_steps), @intCast(config.micro_batch), @intCast(config.block_size) };
        const dev_input_ids = try iree_impl.moveToDevice(full_batch.inputs, folded_shape, .i64);
        try grad_inputs.append(dev_input_ids);
        const dev_targets = try iree_impl.moveToDevice(full_batch.targets, folded_shape, .i64);
        try grad_inputs.append(dev_targets);
        defer releaseDeviceBufferSlice(grad_inputs.items);

        const grad_outputs = try iree_impl.executeWithDeviceBuffers(vmfb, "compute_gradients_accumulated", grad_inputs.items);
        defer self.allocator.free(grad_outputs);

        for (0..workspace.accum_grads.len) |index| {
            const grad_bytes = try iree_impl.readToHost(grad_outputs[index]);
            defer self.allocator.free(grad_bytes);
            @memcpy(workspace.accum_grads[index], grad_bytes);
            grad_outputs[index].release();
        }

        const loss_buf = grad_outputs[workspace.accum_grads.len];
        const loss_bytes = try iree_impl.readToHost(loss_buf);
        defer self.allocator.free(loss_bytes);
        loss_buf.release();

        const total_loss = try readScalarLoss(loss_bytes);
        const scale = 1.0 / @as(f32, @floatFromInt(config.accumulation_steps));
        for (workspace.accum_grads) |buffer| scaleBuffer(buffer, scale);
        return total_loss * scale;
    }

    fn computeInCodeAccumulatedGradients(
        self: *Self,
        vmfb: []const u8,
        iree_impl: *IreeBackend,
        workspace: *InnerLoopWorkspace,
        param_shapes: [][]i64,
        config: InnerLoopConfig,
    ) !f32 {
        var device_accumulators = try self.allocator.alloc(IreeBackend.DeviceBuffer, param_shapes.len);
        var accumulator_count: usize = 0;
        defer self.allocator.free(device_accumulators);
        defer releaseDeviceBufferSlice(device_accumulators[0..accumulator_count]);

        for (param_shapes, 0..) |shape, index| {
            device_accumulators[index] = try iree_impl.allocateZerosDevice(shape, .f32);
            accumulator_count += 1;
        }

        const data_shape = &[_]i64{ @intCast(config.micro_batch), @intCast(config.block_size) };
        var total_loss: f32 = 0.0;
        for (0..config.accumulation_steps) |step| {
            const batch = try self.dataset.?.getBatch(config.micro_batch, config.block_size);
            defer batch.deinit();

            var grad_inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
            defer grad_inputs.deinit();
            for (workspace.device_params) |param| {
                param.retain();
                try grad_inputs.append(param);
            }
            const dev_input_ids = try iree_impl.moveToDevice(batch.inputs, data_shape, .i64);
            try grad_inputs.append(dev_input_ids);
            const dev_targets = try iree_impl.moveToDevice(batch.targets, data_shape, .i64);
            try grad_inputs.append(dev_targets);
            defer releaseDeviceBufferSlice(grad_inputs.items);

            const grad_outputs = try iree_impl.executeWithDeviceBuffers(vmfb, "compute_gradients", grad_inputs.items);
            defer self.allocator.free(grad_outputs);

            var acc_inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
            defer acc_inputs.deinit();
            for (device_accumulators[0..accumulator_count]) |accumulator| {
                accumulator.retain();
                try acc_inputs.append(accumulator);
            }
            for (0..param_shapes.len) |index| try acc_inputs.append(grad_outputs[index]);
            defer releaseDeviceBufferSlice(acc_inputs.items);

            const acc_outputs = try iree_impl.executeWithDeviceBuffers(vmfb, "accumulate_gradients", acc_inputs.items);
            defer self.allocator.free(acc_outputs);
            for (device_accumulators[0..accumulator_count]) |old| old.release();
            for (0..param_shapes.len) |index| device_accumulators[index] = acc_outputs[index];

            const loss_buf = grad_outputs[param_shapes.len];
            const loss_bytes = try iree_impl.readToHost(loss_buf);
            defer self.allocator.free(loss_bytes);
            loss_buf.release();
            const current_loss = try readScalarLoss(loss_bytes);
            total_loss += current_loss;
            if (step == 0) {
                std.log.info("Worker {}: Accumulation step {}/{}, loss: {d:.4}", .{
                    self.node_id.?,
                    step + 1,
                    config.accumulation_steps,
                    current_loss,
                });
            }
        }

        for (0..param_shapes.len) |index| {
            const grad_bytes = try iree_impl.readToHost(device_accumulators[index]);
            defer self.allocator.free(grad_bytes);
            @memcpy(workspace.accum_grads[index], grad_bytes);
        }
        const scale = 1.0 / @as(f32, @floatFromInt(config.accumulation_steps));
        for (workspace.accum_grads) |buffer| scaleBuffer(buffer, scale);
        return total_loss * scale;
    }

    fn computeInnerLoopGradients(
        self: *Self,
        vmfb: []const u8,
        iree_impl: *IreeBackend,
        workspace: *InnerLoopWorkspace,
        param_shapes: [][]i64,
        config: InnerLoopConfig,
    ) !f32 {
        if (config.use_in_graph_accumulation) {
            return try self.computeInGraphAccumulatedGradients(vmfb, iree_impl, workspace, config);
        }
        return try self.computeInCodeAccumulatedGradients(vmfb, iree_impl, workspace, param_shapes, config);
    }

    fn applyInnerLoopOptimizer(
        self: *Self,
        vmfb: []const u8,
        iree_impl: *IreeBackend,
        workspace: *InnerLoopWorkspace,
        param_shapes: [][]i64,
        m_states: [][]u8,
        v_states: [][]u8,
    ) !void {
        const num_params = param_shapes.len;
        var opt_inputs_list = std.ArrayList([]const u8).init(self.allocator);
        defer opt_inputs_list.deinit();
        for (workspace.current_params) |param| try opt_inputs_list.append(param);
        for (workspace.accum_grads) |grad| try opt_inputs_list.append(grad);
        for (m_states) |state| try opt_inputs_list.append(state);
        for (v_states) |state| try opt_inputs_list.append(state);
        try opt_inputs_list.append(std.mem.asBytes(&self.timestep));

        const opt_inputs = try opt_inputs_list.toOwnedSlice();
        defer self.allocator.free(opt_inputs);

        var opt_shapes = std.ArrayList([]const i64).init(self.allocator);
        defer opt_shapes.deinit();
        for (0..4) |_| {
            for (param_shapes) |shape| try opt_shapes.append(shape);
        }
        try opt_shapes.append(&[_]i64{});
        const opt_shapes_slice = try opt_shapes.toOwnedSlice();
        defer self.allocator.free(opt_shapes_slice);

        var opt_dtypes = std.ArrayList(tensor.DType).init(self.allocator);
        defer opt_dtypes.deinit();
        for (0..4) |_| {
            for (0..num_params) |_| try opt_dtypes.append(.f32);
        }
        try opt_dtypes.append(.f32);

        const opt_outputs = try self.backend.executeFunction(vmfb, "apply_optimizer", opt_inputs, opt_shapes_slice, opt_dtypes.items);
        defer freeOutputSlices(self.allocator, opt_outputs);
        for (0..num_params) |index| @memcpy(workspace.current_params[index], opt_outputs[index]);
        for (0..num_params) |index| @memcpy(m_states[index], opt_outputs[num_params + index]);
        for (0..num_params) |index| @memcpy(v_states[index], opt_outputs[2 * num_params + index]);

        var new_device_params = try self.allocator.alloc(IreeBackend.DeviceBuffer, num_params);
        var new_device_count: usize = 0;
        errdefer {
            releaseDeviceBufferSlice(new_device_params[0..new_device_count]);
            self.allocator.free(new_device_params);
        }
        for (workspace.current_params, 0..) |param, index| {
            new_device_params[index] = try iree_impl.moveToDevice(param, param_shapes[index], .f32);
            new_device_count += 1;
        }

        releaseDeviceBufferSlice(workspace.device_params[0..workspace.device_initialized]);
        self.allocator.free(workspace.device_params);
        workspace.device_params = new_device_params;
        workspace.device_initialized = new_device_count;
    }

    fn healOptimizerStates(self: *Self, m_states: [][]u8, v_states: [][]u8) void {
        var healed_count: usize = 0;
        for (m_states, 0..) |m_state, index| {
            var m_corrupt = false;
            var v_corrupt = false;
            self.validateBuffer("M-State Check", m_state, .f32) catch |err| {
                std.log.warn("Worker {}: invalid M-state for param {}: {}", .{ self.node_id.?, index, err });
                m_corrupt = true;
            };
            self.validateBuffer("V-State Check", v_states[index], .f32) catch |err| {
                std.log.warn("Worker {}: invalid V-state for param {}: {}", .{ self.node_id.?, index, err });
                v_corrupt = true;
            };
            if (!m_corrupt and !v_corrupt) continue;
            @memset(m_state, 0);
            @memset(v_states[index], 0);
            healed_count += 1;
        }
        if (healed_count == 0) return;
        std.log.warn("Worker {}: Healed {}/{} parameter states. Timestep reset.", .{
            self.node_id.?,
            healed_count,
            m_states.len,
        });
        self.timestep = 1.0;
    }

    fn appendDeltaBlock(delta_bytes: *std.ArrayList(u8), dtype: tensor.DType, initial_block: []const u8, final_block: []const u8) !void {
        const initial_f32 = std.mem.bytesAsSlice(f32, @constCast(initial_block));
        const final_f32 = std.mem.bytesAsSlice(f32, @constCast(final_block));
        for (initial_f32, final_f32) |initial, final| {
            const delta = initial - final;
            switch (dtype) {
                .bf16 => {
                    const delta_bf16: u16 = @truncate(@as(u32, @bitCast(delta)) >> 16);
                    try delta_bytes.appendSlice(std.mem.asBytes(&delta_bf16));
                },
                .f16 => {
                    const delta_f16 = f32ToF16Bits(delta);
                    try delta_bytes.appendSlice(std.mem.asBytes(&delta_f16));
                },
                else => try delta_bytes.appendSlice(std.mem.asBytes(&delta)),
            }
        }
    }

    fn buildInnerLoopDeltaBytes(self: *Self, workspace: InnerLoopWorkspace, dtype: tensor.DType) !std.ArrayList(u8) {
        var delta_bytes = std.ArrayList(u8).init(self.allocator);
        errdefer delta_bytes.deinit();
        for (workspace.current_params, 0..) |final_block, index| {
            try appendDeltaBlock(&delta_bytes, dtype, workspace.initial_params_f32[index], final_block);
        }
        return delta_bytes;
    }

    fn sendInnerLoopUpdateChunks(
        self: *Self,
        msg: MessageEnvelope,
        data: []const u8,
        final_loss: f32,
    ) !usize {
        const chunk_size = protocol_limits.chunk_bytes_max;
        const total_chunks = (data.len + chunk_size - 1) / chunk_size;
        std.log.info("Worker {}: Sending update in {} chunks ({} bytes total)", .{
            self.node_id.?,
            total_chunks,
            data.len,
        });

        var send_offset: usize = 0;
        for (0..total_chunks) |chunk_index| {
            const end = @min(send_offset + chunk_size, data.len);
            const chunk_slice = data[send_offset..end];
            const b64_len = std.base64.standard.Encoder.calcSize(chunk_slice.len);
            const b64_chunk = try self.allocator.alloc(u8, b64_len);
            defer self.allocator.free(b64_chunk);
            _ = std.base64.standard.Encoder.encode(b64_chunk, chunk_slice);

            var chunk_payload = std.json.ObjectMap.init(self.allocator);
            defer chunk_payload.deinit();
            try chunk_payload.put("chunk_index", .{ .integer = @intCast(chunk_index) });
            try chunk_payload.put("total_chunks", .{ .integer = @intCast(total_chunks) });
            try chunk_payload.put("total_bytes", .{ .integer = @intCast(data.len) });
            try chunk_payload.put("data", .{ .string = b64_chunk });
            try chunk_payload.put("loss", .{ .float = final_loss });
            try chunk_payload.put("chunk_id", .{ .integer = @intCast(self.current_chunk_id.?) });

            const chunk_msg = tcp_stream.createMessageWithContext(
                self.node_id.?,
                "worker",
                0,
                "worker_fabric",
                MessageType.UPDATE_CHUNK,
                msg.msg_id + 1 + @as(message.MessageId, @intCast(chunk_index)),
                contextFromMessage(msg),
                .{ .object = chunk_payload },
            );
            try self.sendControllerMessage(chunk_msg);
            send_offset = end;
            if (chunk_index % 5 == 4) std.time.sleep(10 * std.time.ns_per_ms);
        }
        return total_chunks;
    }

    fn sendInnerLoopComplete(self: *Self, msg: MessageEnvelope, total_chunks: usize, final_loss: f32) !void {
        var response_payload = std.json.ObjectMap.init(self.allocator);
        defer response_payload.deinit();
        try response_payload.put("chunk_id", .{ .integer = @intCast(self.current_chunk_id.?) });
        try response_payload.put("loss", .{ .float = final_loss });
        try response_payload.put("chunked", .{ .bool = true });

        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.INNER_LOOP_COMPLETE,
            msg.msg_id + 1 + @as(message.MessageId, @intCast(total_chunks)),
            contextFromMessage(msg),
            .{ .object = response_payload },
        );
        try self.sendControllerMessage(response);
    }

    /// Handle StartInnerLoop command from the worker fabric.
    /// Implements gradient accumulation: run compute_gradients N times, then apply_optimizer once
    pub fn handleStartInnerLoop(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;
        errdefer {
            if (self.state == .training) self.state = .connected;
        }

        const vmfb = self.cached_vmfb orelse return error.GraphNotInitialized;
        const param_shapes = self.cached_parameter_shapes orelse return error.ParameterShapesNotInitialized;
        const data_shapes = self.cached_data_input_shapes orelse return error.DataShapesNotInitialized;
        if (self.m_states == null) return error.OptimizerStateNotInitialized;
        if (self.v_states == null) return error.OptimizerStateNotInitialized;

        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };
        const config = try parseInnerLoopConfig(payload, data_shapes);
        if (data_shapes[0][0] > 1_000_000) {
            std.log.info("Worker {}: Dynamic batch detected, defaulting micro_batch to {}", .{ self.node_id.?, config.micro_batch });
        }

        var parsed_data_assignment: ?std.json.Parsed(data_assignment.DataAssignment) = null;
        defer if (parsed_data_assignment) |*parsed| parsed.deinit();
        const assignment = try self.resolveDataAssignmentFromPayload(payload, &parsed_data_assignment);

        self.cached_param_dtype = config.param_dtype;
        const m_states = self.m_states.?;
        const v_states = self.v_states.?;
        self.timestep = 1.0;
        for (m_states) |state| @memset(state, 0);
        for (v_states) |state| @memset(state, 0);
        try self.initializeDatasetFromAssignment(assignment, config.tokenizer_type, config.sampling_type);

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        const initial_params_temp = try self.loadInitialInnerLoopParams(msg.request_id, payload, arena.allocator());
        try self.validateBuffer("Initial Params", initial_params_temp, config.param_dtype);
        const initial_params_bytes = try self.allocator.dupe(u8, initial_params_temp);
        defer self.allocator.free(initial_params_bytes);

        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));
        var workspace = try self.initInnerLoopWorkspace(iree_impl, param_shapes, initial_params_bytes, config.param_dtype);
        defer workspace.deinit(self.allocator);

        const mode = if (config.use_in_graph_accumulation) "in-graph" else "in-code";
        std.log.info("Worker {}: Training with precision {s}, micro_batch={}, effective_batch={}, accumulation_steps={}, mode={s}", .{
            self.node_id.?,
            config.dtype_name,
            config.micro_batch,
            config.effective_batch,
            config.accumulation_steps,
            mode,
        });

        var final_loss: f32 = 0.0;
        for (0..config.tau) |step| {
            std.log.info("Worker {} inner loop step {}/{}", .{ self.node_id.?, step + 1, config.tau });
            const avg_loss = try self.computeInnerLoopGradients(vmfb, iree_impl, &workspace, param_shapes, config);
            try self.applyInnerLoopOptimizer(vmfb, iree_impl, &workspace, param_shapes, m_states, v_states);
            final_loss = avg_loss;
            self.timestep += 1.0;
            std.log.info("Worker {} step {}/{} complete, avg_loss: {d:.4}", .{ self.node_id.?, step + 1, config.tau, avg_loss });
        }

        std.log.info("Worker {} completed {} inner loop steps, final loss: {d:.4}", .{ self.node_id.?, config.tau, final_loss });
        self.healOptimizerStates(m_states, v_states);
        var delta_bytes = try self.buildInnerLoopDeltaBytes(workspace, config.param_dtype);
        defer delta_bytes.deinit();
        const total_chunks = try self.sendInnerLoopUpdateChunks(msg, delta_bytes.items, final_loss);
        std.log.info("Worker {}: All {} chunks sent, sending completion signal", .{ self.node_id.?, total_chunks });
        try self.sendInnerLoopComplete(msg, total_chunks, final_loss);
        self.state = .connected;
    }

    pub fn handleCustomLoadBundle(self: *Self, msg: MessageEnvelope) !void {
        try custom_extensions.handleCustomLoadBundle(self, msg);
    }

    pub fn handleCustomStartInference(self: *Self, msg: MessageEnvelope) !void {
        try custom_extensions.handleCustomStartInference(self, msg);
    }

    pub fn handleCustomStartTraining(self: *Self, msg: MessageEnvelope) !void {
        var observer_ctx = custom_extensions.TrainingProgressContext{
            .worker_ctx = self,
            .request_id = msg.request_id,
            .send_fn = sendCustomTrainingProgress,
        };
        try custom_extensions.handleCustomStartTraining(self, msg, &observer_ctx, custom_extensions.emitCustomTrainingProgressHook);
    }

    pub fn handleCustomStartTrainingRound(self: *Self, msg: MessageEnvelope) !void {
        var observer_ctx = custom_extensions.TrainingProgressContext{
            .worker_ctx = self,
            .request_id = msg.request_id,
            .round_id = msg.round_id,
            .task_id = msg.task_id,
            .custom_training_round = true,
            .send_fn = sendCustomTrainingProgress,
        };
        try custom_extensions.handleCustomStartTrainingRound(self, msg, &observer_ctx, custom_extensions.emitCustomTrainingProgressHook);
    }

    fn setTargetArch(self: *Self, target_arch: ?[]const u8) !void {
        if (target_arch) |target| {
            if (self.target_arch) |existing| {
                if (std.mem.eql(u8, existing, target)) return;
            }
            const owned = try self.allocator.dupe(u8, target);
            if (self.target_arch) |existing| self.allocator.free(existing);
            self.target_arch = owned;
        } else {
            if (self.target_arch) |existing| self.allocator.free(existing);
            self.target_arch = null;
        }
    }

    fn sendCustomTrainingProgress(
        worker_ctx: *anyopaque,
        observer_ctx: *const custom_extensions.TrainingProgressContext,
        progress: custom_extensions.TrainingProgressPayload,
    ) !void {
        const self: *Self = @ptrCast(@alignCast(worker_ctx));
        try custom_extensions.sendCustomExtensionTrainingProgress(self, observer_ctx, progress);
    }

    /// Handle LoadModel command for inference workers.
    pub fn handleLoadModel(self: *Self, msg: MessageEnvelope) !void {
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };

        const model_id = switch (payload.get("model_id") orelse return error.MissingModelId) {
            .string => |s| s,
            else => return error.InvalidModelId,
        };
        const vmfb_path = switch (payload.get("vmfb_path") orelse return error.MissingVmfbField) {
            .string => |s| s,
            else => return error.InvalidVmfbPathFormat,
        };
        const weights_path = switch (payload.get("weights_path") orelse return error.MissingWeightsPath) {
            .string => |s| s,
            else => return error.InvalidWeightsPath,
        };
        const weights_format = switch (payload.get("weights_format") orelse std.json.Value{ .string = "flat" }) {
            .string => |s| s,
            else => return error.InvalidWeightsFormat,
        };

        const max_context_tokens = switch (payload.get("max_context_tokens") orelse std.json.Value{ .integer = 0 }) {
            .integer => |i| @as(usize, @intCast(i)),
            else => 0,
        };

        const param_shapes_val = payload.get("parameter_shapes") orelse return error.MissingParameterShapesField;
        const param_dtypes_val = payload.get("parameter_dtypes");
        const data_shapes_val = payload.get("data_input_shapes") orelse return error.MissingShapesField;
        const data_dtypes_val = payload.get("data_input_dtypes");

        const param_shapes = try self.parseShapeArray(param_shapes_val);
        errdefer self.freeShapeArray(param_shapes);
        const param_dtypes = if (param_dtypes_val) |val|
            try self.parseDTypes(val)
        else
            null;
        errdefer if (param_dtypes) |dtypes| self.allocator.free(dtypes);
        const data_shapes = try self.parseShapeArray(data_shapes_val);
        errdefer self.freeShapeArray(data_shapes);
        const data_dtypes = if (data_dtypes_val) |val|
            try self.parseDTypes(val)
        else
            null;
        errdefer if (data_dtypes) |dtypes| self.allocator.free(dtypes);

        const vmfb_bytes = try std.fs.cwd().readFileAlloc(
            self.allocator,
            vmfb_path,
            MAX_INFERENCE_VMFB_BYTES,
        );
        errdefer self.allocator.free(vmfb_bytes);

        const stream_weights = std.mem.eql(u8, weights_format, "streamed") or
            std.mem.eql(u8, weights_format, "bf16_streamed") or
            streamed_weights.isManifestFormat(weights_format);

        const weights_bytes = if (stream_weights)
            try self.allocator.alloc(u8, 0)
        else
            try std.fs.cwd().readFileAlloc(
                self.allocator,
                weights_path,
                MAX_INFERENCE_WEIGHTS_BYTES,
            );
        errdefer self.allocator.free(weights_bytes);

        const record = generation_engine.LoadedInferenceModel{
            .model_id = try self.allocator.dupe(u8, model_id),
            .cached_vmfb = vmfb_bytes,
            .weights_blob = weights_bytes,
            .weights_path = if (stream_weights) try self.allocator.dupe(u8, weights_path) else null,
            .weights_format = try self.allocator.dupe(u8, weights_format),
            .weight_dtypes = param_dtypes,
            .param_shapes = param_shapes,
            .data_shapes = data_shapes,
            .data_dtypes = data_dtypes,
            .max_context_tokens = max_context_tokens,
        };

        self.inference_mutex.lock();
        defer self.inference_mutex.unlock();

        if (self.loaded_models.fetchRemove(record.model_id)) |kv| {
            self.freeLoadedModel(kv.value);
        }
        try self.loaded_models.put(record.model_id, record);

        var ready_payload = std.json.ObjectMap.init(self.allocator);
        defer ready_payload.deinit();
        try ready_payload.put("model_id", .{ .string = model_id });
        try ready_payload.put("ok", .{ .bool = true });

        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.MODEL_READY,
            msg.msg_id + 1,
            contextFromMessage(msg),
            .{ .object = ready_payload },
        );

        try self.sendInferenceMessage(response);
    }

    const GenerationTask = struct {
        worker: *Worker,
        request_id: message.RequestId,
        task_id: message.TaskId,
        model_id: []const u8,
        session_id: []const u8,
        prompt_tokens: []i64,
        max_new_tokens: usize,
        eos_token: i64,
        temperature: f32,
        reuse_session: bool,
        prompt_start_pos: i64,
    };

    /// Handle StartGeneration by spawning a decoding thread.
    pub fn handleStartGeneration(self: *Self, msg: MessageEnvelope) !void {
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };

        const model_id = switch (payload.get("model_id") orelse return error.MissingModelId) {
            .string => |s| s,
            else => return error.InvalidModelId,
        };
        const session_id = switch (payload.get("session_id") orelse return error.MissingSessionId) {
            .string => |s| s,
            else => return error.InvalidSessionId,
        };

        const max_new_tokens = switch (payload.get("max_new_tokens") orelse return error.MissingMaxNewTokens) {
            .integer => |i| @as(usize, @intCast(i)),
            else => return error.InvalidMaxNewTokens,
        };
        const eos_token = switch (payload.get("eos_token") orelse std.json.Value{ .integer = 0 }) {
            .integer => |i| i,
            else => 0,
        };
        const temperature = switch (payload.get("temperature") orelse std.json.Value{ .float = 0.7 }) {
            .float => |f| @as(f32, @floatCast(f)),
            .integer => |i| @as(f32, @floatFromInt(i)),
            else => 0.7,
        };
        const reuse_session = switch (payload.get("reuse_session") orelse std.json.Value{ .bool = false }) {
            .bool => |b| b,
            else => false,
        };
        const prompt_start_pos = switch (payload.get("prompt_start_pos") orelse std.json.Value{ .integer = 0 }) {
            .integer => |i| i,
            else => 0,
        };

        const prompt_json = payload.get("prompt_tokens") orelse return error.MissingPrompt;
        var prompt_list = std.ArrayList(i64).init(self.allocator);
        errdefer prompt_list.deinit();
        switch (prompt_json) {
            .array => |arr| {
                for (arr.items) |item| {
                    try prompt_list.append(item.integer);
                }
            },
            else => return error.InvalidPromptFormat,
        }

        const task = try self.allocator.create(GenerationTask);
        errdefer self.allocator.destroy(task);
        task.* = .{
            .worker = self,
            .request_id = msg.request_id,
            .task_id = msg.task_id,
            .model_id = try self.allocator.dupe(u8, model_id),
            .session_id = try self.allocator.dupe(u8, session_id),
            .prompt_tokens = try prompt_list.toOwnedSlice(),
            .max_new_tokens = max_new_tokens,
            .eos_token = eos_token,
            .temperature = temperature,
            .reuse_session = reuse_session,
            .prompt_start_pos = prompt_start_pos,
        };

        const thread = try std.Thread.spawn(.{}, runGenerationTask, .{task});
        thread.detach();
    }

    fn runGenerationTask(task: *GenerationTask) void {
        const worker = task.worker;
        defer {
            worker.allocator.free(task.model_id);
            worker.allocator.free(task.session_id);
            worker.allocator.free(task.prompt_tokens);
            worker.allocator.destroy(task);
        }

        var model: generation_engine.LoadedInferenceModel = undefined;
        var kv_cache_opt: ?[][]u8 = null;

        worker.inference_mutex.lock();
        if (worker.loaded_models.get(task.model_id)) |m| {
            model = m;
        } else {
            worker.inference_mutex.unlock();
            std.log.warn("Worker {} generation request {} failed: model {s} not loaded", .{
                worker.node_id.?,
                task.request_id,
                task.model_id,
            });
            worker.sendGenerationError(task.request_id, "model_not_loaded");
            return;
        }

        if (worker.session_kv.getPtr(task.session_id)) |session| {
            if (task.reuse_session) {
                kv_cache_opt = session.kv_cache_host;
            } else if (session.kv_cache_host) |cache| {
                for (cache) |buf| worker.allocator.free(buf);
                worker.allocator.free(cache);
                session.kv_cache_host = null;
            }
        } else {
            const session_id = worker.allocator.dupe(u8, task.session_id) catch |err| {
                worker.inference_mutex.unlock();
                worker.sendGenerationError(task.request_id, @errorName(err));
                return;
            };
            const model_id = worker.allocator.dupe(u8, task.model_id) catch |err| {
                worker.allocator.free(session_id);
                worker.inference_mutex.unlock();
                worker.sendGenerationError(task.request_id, @errorName(err));
                return;
            };
            const new_session = generation_engine.GenerationSession{
                .session_id = session_id,
                .model_id = model_id,
                .kv_cache_host = null,
                .last_pos = 0,
            };
            worker.session_kv.put(task.session_id, new_session) catch |err| {
                worker.allocator.free(session_id);
                worker.allocator.free(model_id);
                worker.inference_mutex.unlock();
                worker.sendGenerationError(task.request_id, @errorName(err));
                return;
            };
        }

        const active_state = generation_engine.ActiveGenerationState{
            .request_id = task.request_id,
            .task_id = task.task_id,
            .model_id = task.model_id,
            .session_id = task.session_id,
            .prompt_tokens = task.prompt_tokens.len,
            .completion_tokens = 0,
            .cancelled = std.atomic.Value(u8).init(0),
        };
        if (worker.active_generations.fetchRemove(task.request_id)) |kv| {
            _ = kv;
        }
        worker.active_generations.put(task.request_id, active_state) catch |err| {
            worker.inference_mutex.unlock();
            worker.sendGenerationError(task.request_id, @errorName(err));
            return;
        };

        const active_ptr = worker.active_generations.getPtr(task.request_id).?;
        const backend_type = worker.backend.getBackendType();
        var local_backend: ?*IreeBackend = null;
        defer if (local_backend) |b| b.deinit();
        const iree_impl: *IreeBackend = blk: {
            if (backend_type == .cuda or backend_type == .rocm) {
                // Create a thread-local backend to ensure a valid GPU context.
                local_backend = IreeBackend.init(worker.allocator, backend_type, 0) catch |err| {
                    worker.sendGenerationError(task.request_id, @errorName(err));
                    worker.inference_mutex.lock();
                    _ = worker.active_generations.remove(task.request_id);
                    worker.inference_mutex.unlock();
                    return;
                };
                break :blk local_backend.?;
            }
            break :blk @ptrCast(@alignCast(worker.backend.ptr));
        };

        worker.inference_mutex.unlock();

        const callback = struct {
            fn emitToken(token: i64, ctx: *anyopaque) !void {
                const task_ptr: *GenerationTask = @ptrCast(@alignCast(ctx));
                try task_ptr.worker.sendGenerationChunk(task_ptr.request_id, token);
            }
        }.emitToken;

        const result = worker.gen_engine.generateRolloutStreaming(
            iree_impl,
            &worker.sampler,
            model.cached_vmfb,
            model.param_shapes,
            model.data_shapes,
            model.data_dtypes,
            model.weights_blob,
            model.weight_dtypes,
            model.weights_path,
            model.weights_format,
            task.prompt_tokens,
            task.max_new_tokens,
            task.eos_token,
            task.temperature,
            &kv_cache_opt,
            task.prompt_start_pos,
            task.reuse_session,
            callback,
            @ptrCast(task),
            &active_ptr.cancelled,
        ) catch |err| {
            worker.sendGenerationError(task.request_id, @errorName(err));
            worker.inference_mutex.lock();
            _ = worker.active_generations.remove(task.request_id);
            worker.inference_mutex.unlock();
            return;
        };
        defer worker.allocator.free(result.tokens);

        worker.inference_mutex.lock();
        if (worker.session_kv.getPtr(task.session_id)) |session| {
            session.kv_cache_host = kv_cache_opt;
            session.last_pos = task.prompt_start_pos + @as(i64, @intCast(task.prompt_tokens.len + result.tokens.len));
        }
        _ = worker.active_generations.remove(task.request_id);
        worker.inference_mutex.unlock();

        worker.sendGenerationComplete(task.request_id, result.finish_reason, task.prompt_tokens.len, result.tokens.len);
    }

    fn sendGenerationChunk(self: *Self, request_id: message.RequestId, token: i64) !void {
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();

        var tokens = std.json.Array.init(self.allocator);
        defer tokens.deinit();
        try tokens.append(.{ .integer = token });

        try payload.put("tokens", .{ .array = tokens });
        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.GENERATION_CHUNK,
            0,
            .{ .request_id = request_id },
            .{ .object = payload },
        );
        try self.sendInferenceMessage(response);
    }

    fn sendGenerationComplete(self: *Self, request_id: message.RequestId, reason: []const u8, prompt_tokens: usize, completion_tokens: usize) void {
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        payload.put("finish_reason", .{ .string = reason }) catch |err| {
            std.log.warn("Worker {} failed to build generation-complete payload: {}", .{ self.node_id.?, err });
            return;
        };
        payload.put("prompt_tokens", .{ .integer = @intCast(prompt_tokens) }) catch |err| {
            std.log.warn("Worker {} failed to build generation-complete prompt counter: {}", .{ self.node_id.?, err });
            return;
        };
        payload.put("completion_tokens", .{ .integer = @intCast(completion_tokens) }) catch |err| {
            std.log.warn("Worker {} failed to build generation-complete completion counter: {}", .{ self.node_id.?, err });
            return;
        };

        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.GENERATION_COMPLETE,
            0,
            .{ .request_id = request_id },
            .{ .object = payload },
        );
        self.sendInferenceMessage(response) catch |err| {
            std.log.warn("Worker {} failed to send generation completion for request {}: {}", .{ self.node_id.?, request_id, err });
        };
    }

    fn sendGenerationError(self: *Self, request_id: message.RequestId, err: []const u8) void {
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        payload.put("error", .{ .string = err }) catch |payload_err| {
            std.log.warn("Worker {} failed to build generation-error payload for request {}: {}", .{ self.node_id.?, request_id, payload_err });
            return;
        };

        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.GENERATION_ERROR,
            0,
            .{ .request_id = request_id },
            .{ .object = payload },
        );
        self.sendInferenceMessage(response) catch |send_err| {
            std.log.warn("Worker {} failed to send generation error for request {}: {}", .{ self.node_id.?, request_id, send_err });
        };
    }

    pub fn handleCancelGeneration(self: *Self, msg: MessageEnvelope) void {
        self.inference_mutex.lock();
        defer self.inference_mutex.unlock();

        if (self.active_generations.getPtr(msg.request_id)) |state| {
            state.cancelled.store(1, .release);
        }
    }

    pub fn handleFlushSession(self: *Self, msg: MessageEnvelope) void {
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return,
        };
        const session_id = switch (payload.get("session_id") orelse return) {
            .string => |s| s,
            else => return,
        };

        self.inference_mutex.lock();
        defer self.inference_mutex.unlock();

        if (self.session_kv.fetchRemove(session_id)) |kv| {
            self.freeGenerationSession(kv.value);
        }
    }

    fn parseShapeArray(self: *Self, value: std.json.Value) ![][]i64 {
        const shapes_array = switch (value) {
            .array => |arr| arr,
            else => return error.InvalidShapesFormat,
        };

        var shapes_list = std.ArrayList([]i64).init(self.allocator);
        errdefer {
            for (shapes_list.items) |s| self.allocator.free(s);
            shapes_list.deinit();
        }

        for (shapes_array.items) |shape_val| {
            const dim_array = switch (shape_val) {
                .array => |arr| arr,
                else => return error.InvalidShapeFormat,
            };
            var dim_list = std.ArrayList(i64).init(self.allocator);
            for (dim_array.items) |dim_val| {
                const dim = switch (dim_val) {
                    .integer => |i| i,
                    else => return error.InvalidDimensionFormat,
                };
                try dim_list.append(dim);
            }
            try shapes_list.append(try dim_list.toOwnedSlice());
        }
        return shapes_list.toOwnedSlice();
    }

    fn freeShapeArray(self: *Self, shapes: [][]i64) void {
        for (shapes) |s| self.allocator.free(s);
        self.allocator.free(shapes);
    }

    fn parseDTypes(self: *Self, value: std.json.Value) ![]tensor.DType {
        const dtypes_array = switch (value) {
            .array => |arr| arr,
            else => return error.InvalidDTypesFormat,
        };

        var dtypes_list = std.ArrayList(tensor.DType).init(self.allocator);
        errdefer dtypes_list.deinit();

        for (dtypes_array.items) |val| {
            const s = switch (val) {
                .string => |str| str,
                else => return error.InvalidDTypeFormat,
            };
            try dtypes_list.append(parseDtypeName(s) orelse return error.InvalidDTypeFormat);
        }

        return dtypes_list.toOwnedSlice();
    }

    /// Background Thread: empties the Tx queue and sends to the worker fabric.
    fn txLoop(self: *Self) void {
        std.log.info("Worker {}: Tx Thread started.", .{self.node_id.?});
        while (self.is_running) {
            self.tx_mutex.lock();
            while (self.tx_queue.items.len == 0 and self.is_running and self.asyncTrainingModeActive()) {
                self.tx_cond.wait(&self.tx_mutex);
            }
            if (!self.is_running) {
                self.tx_mutex.unlock();
                break;
            }
            if (self.tx_queue.items.len == 0 and !self.asyncTrainingModeActive()) {
                self.tx_mutex.unlock();
                break;
            }
            // Pop the oldest message
            const msg = self.tx_queue.orderedRemove(0);
            self.tx_mutex.unlock();

            // Send over TCP (Blocking I/O happens here, off the compute thread!)
            self.sendControllerMessage(msg) catch |err| {
                std.log.err("Worker {}: Tx Thread failed to send: {}", .{ self.node_id.?, err });
            };

            // Free the cloned message now that it's sent
            var msg_mut = msg;
            msg_mut.deinitClone(self.allocator);
        }
    }

    fn asyncTrainingModeActive(self: *Self) bool {
        return self.streaming_mode or self.decoupled_mode;
    }

    fn queueDecoupledControlMessageFromRx(self: *Self, msg: MessageEnvelope) !void {
        const msg_clone = try msg.clone(self.allocator);
        errdefer msg_clone.deinitClone(self.allocator);

        self.decoupled_control_mutex.lock();
        defer self.decoupled_control_mutex.unlock();
        if (self.decoupled_control_queue.items.len >= protocol_limits.message_queue_depth_max) return error.MessageQueueFull;
        try self.decoupled_control_queue.append(msg_clone);
    }

    fn enqueueTxMessageClone(self: *Self, msg: MessageEnvelope) !void {
        const msg_clone = try msg.clone(self.allocator);
        errdefer msg_clone.deinitClone(self.allocator);

        self.tx_mutex.lock();
        defer self.tx_mutex.unlock();
        if (self.tx_queue.items.len >= protocol_limits.message_queue_depth_max) return error.MessageQueueFull;
        try self.tx_queue.append(msg_clone);
        self.tx_cond.signal();
    }

    fn popDecoupledControlMessage(self: *Self) ?MessageEnvelope {
        self.decoupled_control_mutex.lock();
        defer self.decoupled_control_mutex.unlock();
        if (self.decoupled_control_queue.items.len == 0) return null;
        return self.decoupled_control_queue.orderedRemove(0);
    }

    pub fn drainDecoupledControlMessages(self: *Self) !void {
        while (self.popDecoupledControlMessage()) |msg| {
            var msg_mut = msg;
            defer msg_mut.deinitClone(self.allocator);

            if (std.mem.eql(u8, msg.msg_type, MessageType.DECOUPLED_FRAGMENT_PULL)) {
                try self.handleDecoupledFragmentPullFromRx(msg);
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.DECOUPLED_FRAGMENT_READY)) {
                try self.handleDecoupledFragmentReadyFromRx(msg);
            }
        }
    }

    /// Background Thread: Listens to TCP and puts fragments into the Rx Map
    fn rxLoop(self: *Self) void {
        std.log.info("Worker {}: Rx Thread started.", .{self.node_id.?});
        while (self.is_running) {
            const res = self.client.receive() catch |err| {
                if (self.is_running) {
                    std.log.err("Worker {}: Rx Thread failed to receive: {}", .{ self.node_id.?, err });
                }
                self.is_running = false;
                self.rx_cond.broadcast(); // Wake up compute thread if it's waiting
                break;
            };

            const msg = res.parsed.value;

            if (std.mem.eql(u8, msg.msg_type, MessageType.DECOUPLED_FRAGMENT_PULL)) {
                std.log.info("Worker {}: received Decoupled DiLoCo fragment pull", .{self.node_id.?});
                self.queueDecoupledControlMessageFromRx(msg) catch |err| {
                    std.log.err("Worker {}: failed to queue decoupled fragment pull: {}", .{ self.node_id.?, err });
                };
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.DECOUPLED_FRAGMENT_READY)) {
                std.log.info("Worker {}: received Decoupled DiLoCo fragment ready", .{self.node_id.?});
                self.queueDecoupledControlMessageFromRx(msg) catch |err| {
                    std.log.err("Worker {}: failed to queue decoupled fragment ready: {}", .{ self.node_id.?, err });
                };
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.STOP_DECOUPLED_DILOCO_LOOP)) {
                self.handleStopDecoupledDilocoLoop(msg);
                res.parsed.deinit();
                self.allocator.free(res.buffer);
                break;
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.FRAGMENT_READY)) {
                const frag_id = @as(usize, @intCast(msg.data.object.get("fragment_id").?.integer));
                const frag_round = @as(usize, @intCast(msg.data.object.get("fragment_round").?.integer));
                const key = FragmentKey{
                    .request_id = msg.request_id,
                    .fragment_id = frag_id,
                    .fragment_round = frag_round,
                };

                // We MUST clone the message because `res.parsed` relies on temporary `res.buffer`
                if (msg.clone(self.allocator)) |msg_clone| {
                    self.rx_mutex.lock();
                    self.rx_map.put(key, msg_clone) catch |err| {
                        std.log.err("Worker {}: Failed to put fragment in Rx Map: {}", .{ self.node_id.?, err });
                        var msg_clone_mut = msg_clone;
                        msg_clone_mut.deinitClone(self.allocator);
                    };
                    self.rx_cond.broadcast(); // Wake up compute thread!
                    self.rx_mutex.unlock();

                    std.log.info("Worker {}: Network received Fragment {} round {}", .{
                        self.node_id.?, frag_id, frag_round,
                    });
                } else |err| {
                    std.log.err("Worker {}: Failed to clone FRAGMENT_READY: {}", .{ self.node_id.?, err });
                }
            } else {
                // If it's a shutdown or other async command, handle it
                if (std.mem.eql(u8, msg.msg_type, MessageType.SHUTDOWN)) {
                    self.handleShutdown(msg);
                }
            }

            res.parsed.deinit();
            self.allocator.free(res.buffer);
        }
    }

    /// Handle START_DECOUPLED_DILOCO_LOOP.
    ///
    /// This is the worker-side state machine used by the generic decoupled
    /// learner loop. The first runtime surface is intentionally metadata-first:
    /// it validates and owns learner counters, emits per-step metadata, and
    /// keeps the old streaming DiLoCo loop untouched. Device-backed one-step
    /// training and fragment pull serving reuse this state in the next runtime
    /// integration layer.
    pub fn handleStartDecoupledDilocoLoop(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;
        self.decoupled_mode = true;
        self.decoupled_request_id = msg.request_id;
        errdefer {
            self.decoupled_mode = false;
            self.decoupled_request_id = 0;
            if (self.state == .training) self.state = .connected;
        }

        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidFormat,
        };
        if (payload.get(training_window.field_name)) |window_value| {
            const training_kind = training_window.trainingKindFromWindowValue(window_value) orelse return error.MissingTrainingKind;
            if (try custom_extensions.dispatchDecoupledTrainingKind(self, msg, training_kind)) {
                return;
            }
            if (std.mem.eql(u8, training_kind, training_window.training_kind_regular)) {
                try self.handleStartRegularDecoupledDilocoLoop(msg);
                return;
            }
            return error.UnsupportedTrainingKind;
        }
        if (payload.get(data_assignment.field_name) != null or payload.get("initial_params") != null or payload.get("raw_params") != null) {
            try self.handleStartRegularDecoupledDilocoLoop(msg);
            return;
        }

        const loop_config = try parseDecoupledLoopConfig(payload, null);
        const synthetic_tokens_per_step = (try parsePayloadUsize(payload, "synthetic_tokens_per_step")) orelse 0;

        var learner_state = try decoupled_learner.DecoupledLearnerState.init(self.allocator, .{
            .num_fragments = loop_config.num_fragments,
            .sync_interval_h = loop_config.sync_interval_h,
            .overlap_tau = loop_config.overlap_tau,
            .learner_alpha = loop_config.learner_alpha,
            .max_syncer_steps = loop_config.max_syncer_steps,
            .local_node_id = self.node_id.?,
        });
        defer learner_state.deinit();

        std.log.info("Worker {}: initialized Decoupled DiLoCo learner loop P={}, H={}, tau={}, alpha={d}", .{
            self.node_id.?,
            loop_config.num_fragments,
            loop_config.sync_interval_h,
            loop_config.overlap_tau,
            loop_config.learner_alpha,
        });

        const tx_thread = try std.Thread.spawn(.{}, txLoop, .{self});
        defer {
            self.tx_mutex.lock();
            self.decoupled_mode = false;
            self.tx_cond.broadcast();
            self.tx_mutex.unlock();
            tx_thread.join();
        }

        var local_steps_run: usize = 0;
        while (self.is_running and self.decoupled_mode and learner_state.shouldContinue()) {
            if (loop_config.max_local_steps != 0 and local_steps_run >= loop_config.max_local_steps) break;
            if (synthetic_tokens_per_step == 0) break;

            try learner_state.recordLocalStep(synthetic_tokens_per_step);
            local_steps_run += 1;
            try self.queueDecoupledMetadata(learner_state, local_steps_run);
        }

        if (self.state == .training) self.state = .connected;
        self.decoupled_request_id = 0;
        std.log.info("Worker {}: Decoupled DiLoCo learner loop exited after {} local steps", .{
            self.node_id.?,
            local_steps_run,
        });
    }

    pub fn handleStopDecoupledDilocoLoop(self: *Self, msg: MessageEnvelope) void {
        if (self.decoupled_request_id != 0 and msg.request_id != 0 and self.decoupled_request_id != msg.request_id) {
            return;
        }
        self.tx_mutex.lock();
        self.decoupled_mode = false;
        self.tx_cond.broadcast();
        self.tx_mutex.unlock();
    }

    fn parsePayloadF32(payload: std.json.ObjectMap, key: []const u8) ?f32 {
        const value = payload.get(key) orelse return null;
        return switch (value) {
            .float => |inner| @floatCast(inner),
            .integer => |inner| @floatFromInt(inner),
            else => null,
        };
    }

    fn parseJsonUsize(value: std.json.Value) !usize {
        return switch (value) {
            .integer => |inner| blk: {
                if (inner < 0) return error.InvalidUnsignedInteger;
                break :blk @intCast(inner);
            },
            else => error.InvalidUnsignedInteger,
        };
    }

    fn parsePayloadUsize(payload: std.json.ObjectMap, key: []const u8) !?usize {
        const value = payload.get(key) orelse return null;
        return try parseJsonUsize(value);
    }

    fn defaultDecoupledMaxLocalSteps(max_syncer_steps: usize, sync_interval_h: usize) !usize {
        if (max_syncer_steps == 0) return 0;
        const base_steps = std.math.mul(usize, max_syncer_steps, sync_interval_h) catch return error.DecoupledStepCountOverflow;
        const overlap_window = std.math.mul(usize, base_steps, 4) catch return error.DecoupledStepCountOverflow;
        return std.math.add(usize, overlap_window, sync_interval_h) catch error.DecoupledStepCountOverflow;
    }

    pub fn parseDecoupledLoopConfig(payload: std.json.ObjectMap, loop_spec: ?training_window.DecoupledLoopSpec) !DecoupledLoopConfig {
        const num_fragments = if (loop_spec) |loop|
            loop.num_fragments
        else
            (try parsePayloadUsize(payload, "num_fragments")) orelse return error.MissingNumFragments;

        const sync_interval_h = if (loop_spec) |loop|
            loop.sync_interval_h
        else
            (try parsePayloadUsize(payload, message.DecoupledDiLoCoField.SYNC_INTERVAL_H)) orelse num_fragments;

        const overlap_tau = if (loop_spec) |loop|
            loop.overlap_tau
        else
            (try parsePayloadUsize(payload, message.DecoupledDiLoCoField.OVERLAP_TAU)) orelse 2;

        const learner_alpha = if (loop_spec) |loop|
            loop.learner_alpha
        else
            parsePayloadF32(payload, "learner_alpha") orelse 0.0;

        const max_syncer_steps = if (loop_spec) |loop|
            loop.max_syncer_steps
        else
            (try parsePayloadUsize(payload, "max_syncer_steps")) orelse 0;

        const explicit_max_local_steps = if (loop_spec) |loop|
            loop.max_local_steps
        else
            (try parsePayloadUsize(payload, "max_local_steps")) orelse 0;

        const max_local_steps = if (explicit_max_local_steps != 0)
            explicit_max_local_steps
        else
            try defaultDecoupledMaxLocalSteps(max_syncer_steps, sync_interval_h);

        const config = DecoupledLoopConfig{
            .num_fragments = num_fragments,
            .sync_interval_h = sync_interval_h,
            .overlap_tau = overlap_tau,
            .learner_alpha = learner_alpha,
            .max_syncer_steps = max_syncer_steps,
            .max_local_steps = max_local_steps,
        };
        try config.validate();
        return config;
    }

    fn parseTrainingWindowFromPayload(
        self: *Self,
        payload: std.json.ObjectMap,
        parsed_window: *?std.json.Parsed(training_window.TrainingWindowSpec),
    ) !bool {
        const window_value = payload.get(training_window.field_name) orelse return false;
        const window_json = try std.json.stringifyAlloc(self.allocator, window_value, .{});
        defer self.allocator.free(window_json);
        parsed_window.* = try std.json.parseFromSlice(
            training_window.TrainingWindowSpec,
            self.allocator,
            window_json,
            .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
        );
        return true;
    }

    pub fn decoupledTestStepDelayMs(self: *Self) u64 {
        const raw = std.process.getEnvVarOwned(self.allocator, "PCP_DECOUPLED_STEP_DELAY_MS") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => return 0,
            else => {
                std.log.warn("Worker {}: failed to read PCP_DECOUPLED_STEP_DELAY_MS: {}", .{ self.node_id orelse 0, err });
                return 0;
            },
        };
        defer self.allocator.free(raw);
        return std.fmt.parseUnsigned(u64, raw, 10) catch |err| {
            std.log.warn("Worker {}: invalid PCP_DECOUPLED_STEP_DELAY_MS='{s}': {}", .{ self.node_id orelse 0, raw, err });
            return 0;
        };
    }

    pub fn queueDecoupledMetadata(self: *Self, learner_state: decoupled_learner.DecoupledLearnerState, train_examples: usize) !void {
        std.debug.assert(self.node_id != null);
        try protocol_invariants.validateRequestContext(self.decoupled_request_id, self.decoupled_request_id);

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        const payload = try learner_state.buildMetadataPayload(arena.allocator(), train_examples);
        const msg = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            message.MessageType.DECOUPLED_LEARNER_METADATA,
            @intCast(learner_state.learner_step),
            .{
                .request_id = self.decoupled_request_id,
                .round_id = @intCast(learner_state.observed_global_step),
                .task_id = 0,
            },
            payload,
        );
        try self.enqueueTxMessageClone(msg);
    }

    fn parseRegularDecoupledRuntime(
        payload: std.json.ObjectMap,
        window_spec: ?training_window.TrainingWindowSpec,
        data_shapes: [][]i64,
    ) !RegularDecoupledRuntime {
        if (data_shapes.len == 0) return error.MissingShapesField;
        if (data_shapes[0].len < 2) return error.InvalidShapesFormat;

        const loop_spec = if (window_spec) |window| window.decoupled_loop else null;
        const regular_runtime = if (window_spec) |window| window.regular else null;
        const loop_config = try parseDecoupledLoopConfig(payload, loop_spec);

        const tokenizer_type = if (regular_runtime) |runtime|
            runtime.tokenizer
        else if (payload.get("tokenizer")) |value| switch (value) {
            .string => |text| text,
            else => return error.InvalidTokenizerFormat,
        } else "char";
        const sampling_type = if (regular_runtime) |runtime|
            runtime.sampling
        else if (payload.get("sampling")) |value| switch (value) {
            .string => |text| text,
            else => return error.InvalidSamplingFormat,
        } else "random";

        var micro_batch = if (regular_runtime) |runtime|
            runtime.micro_batch
        else
            (try parsePayloadUsize(payload, "micro_batch")) orelse @as(usize, @intCast(data_shapes[0][0]));
        if (micro_batch == 0 or micro_batch > 1_000_000) micro_batch = 4;
        const block_size: usize = @intCast(data_shapes[0][1]);

        return .{
            .loop_config = loop_config,
            .tokenizer_type = tokenizer_type,
            .sampling_type = sampling_type,
            .micro_batch = micro_batch,
            .block_size = block_size,
            .data_shape = .{ @intCast(micro_batch), @intCast(block_size) },
        };
    }

    fn loadRegularDecoupledInitialParams(
        self: *Self,
        request_id: message.RequestId,
        payload: std.json.ObjectMap,
        window_spec: ?training_window.TrainingWindowSpec,
    ) !MaybeOwnedBytes {
        if (window_spec) |window| {
            const blob = window.initial_online_weights orelse return error.MissingWeightBlobRef;
            const owned = try self.takeCompletedNamedWeightBlob(request_id, blob.name);
            errdefer self.allocator.free(owned);
            try verifyNamedBlobRef(owned, blob);
            return .{ .bytes = owned, .owned = owned };
        }

        const raw_params = if (payload.get("raw_params")) |value| switch (value) {
            .bool => |enabled| enabled,
            else => return error.InvalidParamsFormat,
        } else false;
        if (raw_params) return .{ .bytes = try self.activateWeightBlobForRequest(request_id) };

        const b64 = switch (payload.get("initial_params") orelse return error.MissingParams) {
            .string => |text| text,
            else => return error.InvalidParamsFormat,
        };
        const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(b64);
        const owned = try self.allocator.alloc(u8, decoded_size);
        errdefer self.allocator.free(owned);
        try std.base64.standard.Decoder.decode(owned, b64);
        return .{ .bytes = owned, .owned = owned };
    }

    fn ensureFreshOptimizerState(self: *Self, param_shapes: [][]i64) !void {
        if (self.m_states == null or self.v_states == null) {
            try self.initializeOptimizerStates(param_shapes);
        } else {
            for (self.m_states.?) |buffer| @memset(buffer, 0);
            for (self.v_states.?) |buffer| @memset(buffer, 0);
        }
        self.timestep = 1.0;
    }

    fn initRegularDecoupledDeviceParams(
        self: *Self,
        iree_impl: *IreeBackend,
        param_shapes: [][]i64,
        initial_params: []const u8,
    ) ![]IreeBackend.DeviceBuffer {
        if (initial_params.len % @sizeOf(f32) != 0) return error.InvalidParameterBuffer;

        var device_params = try self.allocator.alloc(IreeBackend.DeviceBuffer, param_shapes.len);
        var initialized_params: usize = 0;
        errdefer {
            for (device_params[0..initialized_params]) |buffer| buffer.release();
            self.allocator.free(device_params);
        }

        var offset: usize = 0;
        for (param_shapes, 0..) |shape, index| {
            const byte_count = try f32ShapeByteSize(shape);
            if (offset + byte_count > initial_params.len) return error.ParameterBufferTooSmall;
            device_params[index] = try iree_impl.moveToDevice(initial_params[offset .. offset + byte_count], shape, .f32);
            initialized_params += 1;
            offset += byte_count;
        }
        if (offset != initial_params.len) return error.ParameterBufferSizeMismatch;
        return device_params;
    }

    pub fn initDecoupledLearnerState(self: *Self, loop_config: DecoupledLoopConfig) !decoupled_learner.DecoupledLearnerState {
        return try decoupled_learner.DecoupledLearnerState.init(self.allocator, .{
            .num_fragments = loop_config.num_fragments,
            .sync_interval_h = loop_config.sync_interval_h,
            .overlap_tau = loop_config.overlap_tau,
            .learner_alpha = loop_config.learner_alpha,
            .max_syncer_steps = loop_config.max_syncer_steps,
            .local_node_id = self.node_id.?,
        });
    }

    pub fn spawnDecoupledIoThreads(self: *Self) !DecoupledIoThreads {
        return try DecoupledIoThreads.spawn(self);
    }

    fn runRegularDecoupledLoop(
        self: *Self,
        vmfb: []const u8,
        device_params: *[]IreeBackend.DeviceBuffer,
        learner_state: *decoupled_learner.DecoupledLearnerState,
        runtime: RegularDecoupledRuntime,
    ) !usize {
        const step_delay_ms = self.decoupledTestStepDelayMs();
        var local_steps_run: usize = 0;
        while (self.is_running and self.decoupled_mode and learner_state.shouldContinue()) {
            try self.drainDecoupledControlMessages();
            if (runtime.loop_config.max_local_steps != 0 and local_steps_run >= runtime.loop_config.max_local_steps) break;
            if (step_delay_ms > 0) std.time.sleep(step_delay_ms * std.time.ns_per_ms);

            self.decoupled_session_mutex.lock();
            var locked = true;
            defer if (locked) self.decoupled_session_mutex.unlock();

            self.current_step += 1;
            const loss = self.executeOneTrainingStep(
                vmfb,
                device_params,
                runtime.micro_batch,
                runtime.block_size,
                runtime.dataShape(),
            ) catch |err| {
                locked = false;
                self.decoupled_session_mutex.unlock();
                return err;
            };
            self.decoupled_regular_last_loss = loss;
            local_steps_run += 1;
            try learner_state.recordLocalStep(runtime.micro_batch * runtime.block_size);
            try self.queueDecoupledMetadata(learner_state.*, local_steps_run * runtime.micro_batch);

            locked = false;
            self.decoupled_session_mutex.unlock();
            try self.drainDecoupledControlMessages();
            std.time.sleep(1 * std.time.ns_per_ms);
        }
        try self.drainDecoupledControlMessages();
        return local_steps_run;
    }

    fn handleStartRegularDecoupledDilocoLoop(self: *Self, msg: MessageEnvelope) !void {
        defer {
            self.tx_mutex.lock();
            self.decoupled_mode = false;
            self.tx_cond.broadcast();
            self.tx_mutex.unlock();
            self.decoupled_request_id = 0;
            if (self.state == .training) self.state = .connected;
        }

        const vmfb = self.cached_vmfb orelse return error.GraphNotInitialized;
        const param_shapes = self.cached_parameter_shapes orelse return error.ParameterShapesNotInitialized;
        const data_shapes = self.cached_data_input_shapes orelse return error.DataShapesNotInitialized;
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };
        var parsed_window: ?std.json.Parsed(training_window.TrainingWindowSpec) = null;
        defer if (parsed_window) |*parsed| parsed.deinit();
        const has_training_window = try self.parseTrainingWindowFromPayload(payload, &parsed_window);
        const window_spec = if (has_training_window) parsed_window.?.value else null;
        const runtime = try parseRegularDecoupledRuntime(payload, window_spec, data_shapes);

        var parsed_data_assignment: ?std.json.Parsed(data_assignment.DataAssignment) = null;
        defer if (parsed_data_assignment) |*parsed| parsed.deinit();
        const assignment = if (window_spec) |window| window.data_assignment else try self.resolveDataAssignmentFromPayload(payload, &parsed_data_assignment);
        try self.initializeDatasetFromAssignment(assignment, runtime.tokenizer_type, runtime.sampling_type);
        const initial_params = try self.loadRegularDecoupledInitialParams(msg.request_id, payload, window_spec);
        defer initial_params.deinit(self.allocator);
        try self.ensureFreshOptimizerState(param_shapes);

        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));
        var device_params = try self.initRegularDecoupledDeviceParams(iree_impl, param_shapes, initial_params.bytes);
        defer {
            for (device_params) |buffer| buffer.release();
            self.allocator.free(device_params);
        }

        var learner_state = try self.initDecoupledLearnerState(runtime.loop_config);
        defer learner_state.deinit();

        self.decoupled_session_mutex.lock();
        self.decoupled_regular_device_params = device_params;
        self.decoupled_learner_state = &learner_state;
        self.decoupled_regular_last_loss = 0.0;
        self.decoupled_session_mutex.unlock();
        defer {
            self.decoupled_session_mutex.lock();
            self.decoupled_regular_device_params = null;
            self.decoupled_learner_state = null;
            self.decoupled_session_mutex.unlock();
        }

        const io_threads = try DecoupledIoThreads.spawn(self);
        defer io_threads.stopAndJoin(self);

        std.log.info("Worker {}: starting regular Decoupled DiLoCo learner loop P={}, H={}, tau={}, alpha={d}", .{
            self.node_id.?,
            runtime.loop_config.num_fragments,
            runtime.loop_config.sync_interval_h,
            runtime.loop_config.overlap_tau,
            runtime.loop_config.learner_alpha,
        });

        const local_steps_run = try self.runRegularDecoupledLoop(vmfb, &device_params, &learner_state, runtime);
        std.log.info("Worker {}: regular Decoupled DiLoCo learner loop exited after {} local steps", .{
            self.node_id.?,
            local_steps_run,
        });
    }

    pub fn handleCustomStartDecoupledTrainingLoop(self: *Self, msg: MessageEnvelope) !void {
        var observer_ctx = custom_extensions.TrainingProgressContext{
            .worker_ctx = self,
            .request_id = msg.request_id,
            .round_id = msg.round_id,
            .task_id = msg.task_id,
            .custom_decoupled_training = true,
            .send_fn = sendCustomTrainingProgress,
        };
        try custom_extensions.handleCustomStartDecoupledTrainingLoop(self, msg, &observer_ctx, custom_extensions.emitCustomTrainingProgressHook);
    }

    fn handleDecoupledFragmentPullFromRx(self: *Self, msg: MessageEnvelope) !void {
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };
        const fragment_id: usize = @intCast((payload.get(message.DecoupledDiLoCoField.FRAGMENT_ID) orelse return error.MissingFragmentId).integer);
        const syncer_step: usize = if (payload.get(message.DecoupledDiLoCoField.SYNCER_STEP)) |value|
            @intCast(value.integer)
        else
            @intCast(msg.round_id);
        const tensor_indices = switch (payload.get("tensor_indices") orelse return error.MissingTensorIndices) {
            .array => |array| array,
            else => return error.InvalidTensorIndices,
        };

        self.decoupled_session_mutex.lock();
        const regular_active = self.extension_state.decoupled_custom_session == null and self.decoupled_regular_device_params != null;
        self.decoupled_session_mutex.unlock();
        if (regular_active) {
            try self.handleRegularDecoupledFragmentPullFromRx(msg, payload, fragment_id, syncer_step, tensor_indices);
            return;
        }

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        var updates = std.json.Array.init(arena.allocator());

        self.decoupled_session_mutex.lock();
        defer self.decoupled_session_mutex.unlock();
        const session = self.extension_state.decoupled_custom_session orelse return error.DecoupledSessionNotActive;
        const learner_state = self.decoupled_learner_state orelse return error.DecoupledLearnerNotActive;
        if (fragment_id >= learner_state.steps_since_update.len) return error.InvalidFragmentId;

        for (tensor_indices.items) |tensor_value| {
            const tensor_index: usize = @intCast(tensor_value.integer);
            const tensor_values = try session.readOnlineTensor(arena.allocator(), tensor_index);
            const tensor_bytes = std.mem.sliceAsBytes(tensor_values);
            const b64_len = std.base64.standard.Encoder.calcSize(tensor_bytes.len);
            const encoded = try arena.allocator().alloc(u8, b64_len);
            _ = std.base64.standard.Encoder.encode(encoded, tensor_bytes);

            var update = std.json.ObjectMap.init(arena.allocator());
            try update.put("tensor_idx", .{ .integer = @intCast(tensor_index) });
            try update.put("data", .{ .string = encoded });
            try updates.append(.{ .object = update });
        }

        var response_payload = std.json.ObjectMap.init(arena.allocator());
        try response_payload.put(message.DecoupledDiLoCoField.FRAGMENT_ID, .{ .integer = @intCast(fragment_id) });
        try response_payload.put(message.DecoupledDiLoCoField.FRAGMENT_ROUND, .{ .integer = @intCast(syncer_step) });
        try response_payload.put(message.DecoupledDiLoCoField.SYNCER_STEP, .{ .integer = @intCast(syncer_step) });
        try response_payload.put(message.DecoupledDiLoCoField.LEARNER_STEP, .{ .integer = @intCast(learner_state.learner_step) });
        try response_payload.put(message.DecoupledDiLoCoField.STEPS_SINCE_FRAGMENT_UPDATE, .{ .integer = @intCast(learner_state.steps_since_update[fragment_id]) });
        try response_payload.put(message.DecoupledDiLoCoField.TOKENS_SINCE_FRAGMENT_UPDATE, .{ .integer = @intCast(learner_state.tokens_since_update[fragment_id]) });
        try response_payload.put(message.DecoupledDiLoCoField.TRAIN_EXAMPLES, .{ .integer = @intCast(session.trainExamples()) });
        try response_payload.put(message.DecoupledDiLoCoField.UPDATES, .{ .array = updates });

        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.DECOUPLED_FRAGMENT_UPDATE,
            @intCast(learner_state.learner_step),
            .{
                .request_id = msg.request_id,
                .round_id = @intCast(syncer_step),
                .task_id = @intCast(fragment_id + 1),
            },
            .{ .object = response_payload },
        );
        try self.enqueueTxMessageClone(response);
        std.log.info("Worker {}: queued Decoupled DiLoCo fragment {} update for syncer step {}", .{
            self.node_id.?,
            fragment_id,
            syncer_step,
        });
    }

    fn handleDecoupledFragmentReadyFromRx(self: *Self, msg: MessageEnvelope) !void {
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };
        const fragment_id: usize = @intCast((payload.get(message.DecoupledDiLoCoField.FRAGMENT_ID) orelse return error.MissingFragmentId).integer);
        const syncer_step: usize = if (payload.get(message.DecoupledDiLoCoField.SYNCER_STEP)) |value|
            @intCast(value.integer)
        else
            @intCast(msg.round_id);
        const updates = switch (payload.get(message.DecoupledDiLoCoField.UPDATES) orelse return error.MissingUpdatesField) {
            .array => |array| array,
            else => return error.InvalidUpdatesFormat,
        };
        const target_updates = if (payload.get("target_updates")) |value| switch (value) {
            .array => |array| array,
            else => return error.InvalidUpdatesFormat,
        } else null;

        self.decoupled_session_mutex.lock();
        const regular_active = self.extension_state.decoupled_custom_session == null and self.decoupled_regular_device_params != null;
        self.decoupled_session_mutex.unlock();
        if (regular_active) {
            try self.handleRegularDecoupledFragmentReadyFromRx(fragment_id, syncer_step, updates);
            return;
        }

        self.decoupled_session_mutex.lock();
        defer self.decoupled_session_mutex.unlock();
        const session = self.extension_state.decoupled_custom_session orelse return error.DecoupledSessionNotActive;
        const learner_state = self.decoupled_learner_state orelse return error.DecoupledLearnerNotActive;

        for (updates.items) |update_value| {
            const update = switch (update_value) {
                .object => |obj| obj,
                else => return error.InvalidUpdatesFormat,
            };
            const tensor_index: usize = @intCast((update.get("tensor_idx") orelse return error.MissingTensorIndex).integer);
            const data_b64 = switch (update.get("data") orelse return error.MissingDataField) {
                .string => |s| s,
                else => return error.InvalidDataField,
            };
            const values = try self.decodeBase64F32(data_b64);
            defer self.allocator.free(values);
            try session.applyOnlineTensorBlended(tensor_index, values, learner_state.config.learner_alpha);
        }

        if (target_updates) |target_array| {
            for (target_array.items) |update_value| {
                const update = switch (update_value) {
                    .object => |obj| obj,
                    else => return error.InvalidUpdatesFormat,
                };
                const tensor_index: usize = @intCast((update.get("tensor_idx") orelse return error.MissingTensorIndex).integer);
                const data_b64 = switch (update.get("data") orelse return error.MissingDataField) {
                    .string => |s| s,
                    else => return error.InvalidDataField,
                };
                const values = try self.decodeBase64F32(data_b64);
                defer self.allocator.free(values);
                try session.applyTargetTensor(tensor_index, values);
            }
        }

        try learner_state.applySyncerFragment(fragment_id, syncer_step);
        std.log.info("Worker {}: applied Decoupled DiLoCo fragment {} at syncer step {}", .{
            self.node_id.?,
            fragment_id,
            syncer_step,
        });
    }

    fn handleRegularDecoupledFragmentPullFromRx(
        self: *Self,
        msg: MessageEnvelope,
        payload: std.json.ObjectMap,
        fragment_id: usize,
        syncer_step: usize,
        tensor_indices: std.json.Array,
    ) !void {
        _ = payload;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        var updates = std.json.Array.init(arena.allocator());
        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));

        self.decoupled_session_mutex.lock();
        defer self.decoupled_session_mutex.unlock();
        const device_params = self.decoupled_regular_device_params orelse return error.DecoupledSessionNotActive;
        const learner_state = self.decoupled_learner_state orelse return error.DecoupledLearnerNotActive;
        if (fragment_id >= learner_state.steps_since_update.len) return error.InvalidFragmentId;

        for (tensor_indices.items) |tensor_value| {
            const tensor_index: usize = @intCast(tensor_value.integer);
            if (tensor_index >= device_params.len) return error.TensorIndexOutOfBounds;
            const tensor_bytes = try iree_impl.readToHost(device_params[tensor_index]);
            defer self.allocator.free(tensor_bytes);

            const b64_len = std.base64.standard.Encoder.calcSize(tensor_bytes.len);
            const encoded = try arena.allocator().alloc(u8, b64_len);
            _ = std.base64.standard.Encoder.encode(encoded, tensor_bytes);

            var update = std.json.ObjectMap.init(arena.allocator());
            try update.put("tensor_idx", .{ .integer = @intCast(tensor_index) });
            try update.put("data", .{ .string = encoded });
            try updates.append(.{ .object = update });
        }

        var response_payload = std.json.ObjectMap.init(arena.allocator());
        try response_payload.put(message.DecoupledDiLoCoField.FRAGMENT_ID, .{ .integer = @intCast(fragment_id) });
        try response_payload.put(message.DecoupledDiLoCoField.FRAGMENT_ROUND, .{ .integer = @intCast(syncer_step) });
        try response_payload.put(message.DecoupledDiLoCoField.SYNCER_STEP, .{ .integer = @intCast(syncer_step) });
        try response_payload.put(message.DecoupledDiLoCoField.LEARNER_STEP, .{ .integer = @intCast(learner_state.learner_step) });
        try response_payload.put(message.DecoupledDiLoCoField.STEPS_SINCE_FRAGMENT_UPDATE, .{ .integer = @intCast(learner_state.steps_since_update[fragment_id]) });
        try response_payload.put(message.DecoupledDiLoCoField.TOKENS_SINCE_FRAGMENT_UPDATE, .{ .integer = @intCast(learner_state.tokens_since_update[fragment_id]) });
        try response_payload.put(message.DecoupledDiLoCoField.TRAIN_EXAMPLES, .{ .integer = @intCast(learner_state.learner_step) });
        try response_payload.put(message.DecoupledDiLoCoField.UPDATES, .{ .array = updates });
        try response_payload.put("loss", .{ .float = self.decoupled_regular_last_loss });

        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.DECOUPLED_FRAGMENT_UPDATE,
            @intCast(learner_state.learner_step),
            .{
                .request_id = msg.request_id,
                .round_id = @intCast(syncer_step),
                .task_id = @intCast(fragment_id + 1),
            },
            .{ .object = response_payload },
        );
        try self.enqueueTxMessageClone(response);
    }

    fn handleRegularDecoupledFragmentReadyFromRx(
        self: *Self,
        fragment_id: usize,
        syncer_step: usize,
        updates: std.json.Array,
    ) !void {
        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));
        const param_shapes = self.cached_parameter_shapes orelse return error.ParameterShapesNotInitialized;

        self.decoupled_session_mutex.lock();
        defer self.decoupled_session_mutex.unlock();
        const device_params = self.decoupled_regular_device_params orelse return error.DecoupledSessionNotActive;
        const learner_state = self.decoupled_learner_state orelse return error.DecoupledLearnerNotActive;

        for (updates.items) |update_value| {
            const update = switch (update_value) {
                .object => |obj| obj,
                else => return error.InvalidUpdatesFormat,
            };
            const tensor_index: usize = @intCast((update.get("tensor_idx") orelse return error.MissingTensorIndex).integer);
            if (tensor_index >= device_params.len or tensor_index >= param_shapes.len) return error.TensorIndexOutOfBounds;
            const data_b64 = switch (update.get("data") orelse return error.MissingDataField) {
                .string => |s| s,
                else => return error.InvalidDataField,
            };
            const global_values = try self.decodeBase64F32(data_b64);
            defer self.allocator.free(global_values);

            const current_bytes = try iree_impl.readToHost(device_params[tensor_index]);
            defer self.allocator.free(current_bytes);
            if (current_bytes.len != global_values.len * @sizeOf(f32)) return error.FragmentLengthMismatch;
            const current_values = std.mem.bytesAsSlice(f32, current_bytes);
            if (current_values.len != global_values.len) return error.FragmentLengthMismatch;

            const blended = try self.allocator.alloc(f32, global_values.len);
            defer self.allocator.free(blended);
            for (current_values, 0..) |local, idx| {
                const value = learner_state.config.learner_alpha * local + (1.0 - learner_state.config.learner_alpha) * global_values[idx];
                blended[idx] = if (std.math.isFinite(value)) value else 0.0;
            }

            device_params[tensor_index].release();
            device_params[tensor_index] = try iree_impl.moveToDevice(std.mem.sliceAsBytes(blended), param_shapes[tensor_index], .f32);
        }

        try learner_state.applySyncerFragment(fragment_id, syncer_step);
        std.log.info("Worker {}: applied regular Decoupled DiLoCo fragment {} at syncer step {}", .{
            self.node_id.?,
            fragment_id,
            syncer_step,
        });
    }

    fn decodeBase64F32(self: *Self, data_b64: []const u8) ![]f32 {
        const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(data_b64);
        if (decoded_size % @sizeOf(f32) != 0) return error.InvalidPrivateWeightFile;
        const decoded = try self.allocator.alloc(u8, decoded_size);
        defer self.allocator.free(decoded);
        try std.base64.standard.Decoder.decode(decoded, data_b64);

        const values = try self.allocator.alloc(f32, decoded_size / @sizeOf(f32));
        errdefer self.allocator.free(values);
        for (values, 0..) |*value, idx| {
            const offset = idx * @sizeOf(f32);
            const bits = std.mem.readInt(u32, decoded[offset..][0..4], .little);
            value.* = @bitCast(bits);
        }
        return values;
    }

    /// Handle START_STREAMING_LOOP message - implements continuous streaming DiLoCo
    pub fn handleStartStreamingLoop(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;
        self.streaming_mode = true;
        self.streaming_request_id = msg.request_id;
        self.current_step = 0;
        errdefer {
            self.streaming_mode = false;
            self.streaming_request_id = 0;
            if (self.state == .training) {
                self.state = .connected;
            }
        }

        const vmfb = self.cached_vmfb orelse return error.GraphNotInitialized;
        const param_shapes = self.cached_parameter_shapes orelse return error.ParameterShapesNotInitialized;
        const data_shapes = self.cached_data_input_shapes orelse return error.DataShapesNotInitialized;

        const payload = msg.data.object;
        self.num_fragments = @intCast(payload.get("num_fragments").?.integer);
        self.inner_steps = @intCast(payload.get("inner_steps").?.integer); // H
        self.overlap_tau = @intCast(payload.get("overlap_tau").?.integer); // tau
        self.alpha = @as(f32, @floatCast(payload.get("alpha").?.float));
        const max_rounds: usize = @intCast(payload.get("max_rounds").?.integer);

        const micro_batch: usize = @intCast(payload.get("micro_batch").?.integer);

        var parsed_data_assignment: ?std.json.Parsed(data_assignment.DataAssignment) = null;
        defer if (parsed_data_assignment) |*parsed| parsed.deinit();
        const assignment = try self.resolveDataAssignmentFromPayload(payload, &parsed_data_assignment);
        const tokenizer_type = payload.get("tokenizer").?.string;
        const sampling_type = if (payload.get("sampling")) |sampling_val|
            switch (sampling_val) {
                .string => |s| s,
                else => "random",
            }
        else
            "random";

        // Check if using raw_params (chunked transfer) or base64
        const use_raw_params = payload.get("raw_params") orelse std.json.Value{ .bool = false };
        const initial_params_b64 = if (use_raw_params.bool)
            null
        else
            payload.get("initial_params").?.string;

        try self.initializeDatasetFromAssignment(assignment, tokenizer_type, sampling_type);

        std.log.info("Worker {}: Starting Continuous Streaming Loop. P={}, H={}, overlap={}", .{ self.node_id.?, self.num_fragments, self.inner_steps, self.overlap_tau });

        if (self.tensor_snapshots) |snapshots| {
            for (snapshots) |snapshot| self.allocator.free(snapshot);
            self.allocator.free(snapshots);
            self.tensor_snapshots = null;
        }
        self.clearReceivedFragments();

        // Get initial parameters (either from base64 or weight_blob)
        var decoded: []const u8 = undefined;
        var decoded_buf: ?[]u8 = null;
        var should_free_decoded = false;

        if (use_raw_params.bool) {
            // Parameters were sent via request-scoped chunked transfer
            decoded = try self.activateWeightBlobForRequest(msg.request_id);
            std.log.info("Worker {}: Using request-scoped weight blob for request {} with {} bytes", .{
                self.node_id.?,
                msg.request_id,
                decoded.len,
            });
        } else {
            // Parameters sent via base64 encoding
            const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(initial_params_b64.?);
            const mutable = try self.allocator.alloc(u8, decoded_size);
            decoded_buf = mutable;
            decoded = mutable;
            should_free_decoded = true;
            try std.base64.standard.Decoder.decode(mutable, initial_params_b64.?);
            std.log.info("Worker {}: Decoded {} bytes from base64", .{ self.node_id.?, decoded.len });
        }
        defer if (should_free_decoded) self.allocator.free(decoded_buf.?);

        // Initialize device parameters
        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));
        var device_params = try self.allocator.alloc(IreeBackend.DeviceBuffer, param_shapes.len);
        defer {
            for (device_params) |buf| buf.release();
            self.allocator.free(device_params);
        }

        // Upload initial parameters and create snapshots
        std.log.info("Worker {}: Initializing {} parameters", .{ self.node_id.?, param_shapes.len });
        self.tensor_snapshots = try self.allocator.alloc([]u8, param_shapes.len);

        // Determine the dtype size from the payload (default to f32 if not specified)
        const dtype_str = payload.get("dtype");
        const bytes_per_element: usize = if (dtype_str) |dt| blk: {
            if (std.mem.eql(u8, dt.string, "bf16")) {
                break :blk 2;
            } else if (std.mem.eql(u8, dt.string, "f16")) {
                break :blk 2;
            } else {
                break :blk 4; // f32
            }
        } else 4; // default to f32

        var param_offset: usize = 0;
        for (param_shapes, 0..) |shape, i| {
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);
            const size = elem_count * bytes_per_element;

            // Check if we have enough data remaining
            if (param_offset + size > decoded.len) {
                std.log.err("Worker {}: Parameter {} requires {} bytes but only {} available (offset={})", .{ self.node_id.?, i, size, decoded.len - param_offset, param_offset });
                return error.InsufficientParameterData;
            }

            // Extract parameter slice and convert to f32 (same as regular DiLoCo path)
            const param_bytes = decoded[param_offset .. param_offset + size];
            param_offset += size;

            const f32_size = elem_count * 4;
            const f32_buf = try self.allocator.alloc(u8, f32_size);
            defer self.allocator.free(f32_buf);

            if (bytes_per_element == 2) {
                // bf16 -> f32 conversion
                const dest = std.mem.bytesAsSlice(f32, f32_buf);
                for (0..elem_count) |k| {
                    const u16_val = std.mem.readInt(u16, param_bytes[k * 2 ..][0..2], .little);
                    dest[k] = @bitCast(@as(u32, u16_val) << 16);
                }
            } else {
                @memcpy(f32_buf, param_bytes);
            }

            // Upload f32 data to device (MLIR pipeline works in f32)
            device_params[i] = try iree_impl.moveToDevice(f32_buf, shape, .f32);

            // Create snapshot as f32 (matches what readToHost returns after optimizer)
            self.tensor_snapshots.?[i] = try self.allocator.dupe(u8, f32_buf);
        }
        std.log.info("Worker {}: Successfully initialized all parameters", .{self.node_id.?});

        // Prepare optimizer states if needed
        if (self.m_states == null) {
            std.log.info("Worker {}: Initializing optimizer states", .{self.node_id.?});
            try self.initializeOptimizerStates(param_shapes);
            std.log.info("Worker {}: Optimizer states initialized", .{self.node_id.?});
        }

        const block_size: usize = @intCast(data_shapes[0][1]);
        const data_shape = &[_]i64{ @intCast(micro_batch), @intCast(block_size) };
        std.log.info("Worker {}: Data shape: [{}, {}]", .{ self.node_id.?, micro_batch, block_size });

        // Start the asynchronous Tx thread.
        std.log.info("Worker {}: Starting streaming Tx thread", .{self.node_id.?});
        const tx_thread = try std.Thread.spawn(.{}, txLoop, .{self});
        defer {
            self.tx_mutex.lock();
            self.streaming_mode = false;
            self.tx_cond.broadcast();
            self.tx_mutex.unlock();
            tx_thread.join();
        }

        // Allow time for the final fragment merge of the last fragment.
        const last_fragment_offset = self.fragmentPhaseStart(self.num_fragments - 1);
        const max_steps = max_rounds * self.inner_steps + last_fragment_offset + self.overlap_tau;
        std.log.info("Worker {}: Entering main streaming loop (max_steps={}, max_rounds={})", .{
            self.node_id.?, max_steps, max_rounds,
        });
        while (self.current_step < max_steps and self.is_running) {
            self.current_step += 1;
            const t = self.current_step;

            std.log.info("Worker {}: Step {} starting", .{ self.node_id.?, t });

            // --- A. MERGE SCHEDULE (RECEIVE) ---
            for (0..self.num_fragments) |p| {
                const t_p = self.fragmentPhaseStart(p);
                if (t > (t_p + self.overlap_tau)) {
                    if ((t - t_p - self.overlap_tau) % self.inner_steps == 0) {
                        const fragment_round = self.fragmentReceiveRound(p, t);
                        if (fragment_round >= max_rounds) continue;
                        std.log.info("Worker {}: Step {} - Receiving fragment {}", .{ self.node_id.?, t, p });
                        try self.receiveAndMergeFragment(p, fragment_round, &device_params);
                    }
                }
            }

            // --- B. COMPUTE ONE STEP ---
            std.log.info("Worker {}: Step {} - Executing training step", .{ self.node_id.?, t });
            _ = try self.executeOneTrainingStep(vmfb, &device_params, micro_batch, block_size, data_shape);

            // --- C. SYNC SCHEDULE (SEND) ---
            for (0..self.num_fragments) |p| {
                const t_p = self.fragmentPhaseStart(p);
                if (t > t_p and (t - t_p) % self.inner_steps == 0) {
                    const fragment_round = self.fragmentSendRound(p, t);
                    if (fragment_round >= max_rounds) continue;
                    std.log.info("Worker {}: Step {} - Sending fragment {}", .{ self.node_id.?, t, p });
                    try self.extractAndSendFragment(p, fragment_round, device_params);
                }
            }
        }

        if (self.state == .training) {
            self.state = .connected;
        }
        self.streaming_request_id = 0;
        std.log.info("Worker {}: Streaming loop complete after {} steps", .{ self.node_id.?, self.current_step });
    }

    /// Executes exactly one micro-batch step on the GPU
    fn executeOneTrainingStep(self: *Self, vmfb: []const u8, device_params: *[]IreeBackend.DeviceBuffer, micro_batch: usize, block_size: usize, data_shape: *const [2]i64) !f32 {
        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));
        const num_params = device_params.len;

        // Get next batch of data
        const batch = try self.dataset.?.getBatch(micro_batch, block_size);
        defer batch.deinit();

        // Prepare inputs for gradient computation
        var grad_inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
        defer grad_inputs.deinit();

        for (device_params.*) |p| {
            p.retain();
            try grad_inputs.append(p);
        }

        const dev_input_ids = try iree_impl.moveToDevice(batch.inputs, data_shape, .i64);
        try grad_inputs.append(dev_input_ids);
        const dev_targets = try iree_impl.moveToDevice(batch.targets, data_shape, .i64);
        try grad_inputs.append(dev_targets);

        // Compute gradients
        const grad_outputs = try iree_impl.executeWithDeviceBuffers(vmfb, "compute_gradients", grad_inputs.items);
        defer {
            for (grad_outputs) |buf| buf.release();
            self.allocator.free(grad_outputs);
        }

        for (grad_inputs.items) |buf| buf.release();

        // Validate gradient outputs
        if (grad_outputs.len < num_params + 1) {
            std.log.err("Worker {}: compute_gradients returned {} outputs, expected at least {} (params + loss)", .{ self.node_id.?, grad_outputs.len, num_params + 1 });
            return error.InsufficientGradientOutputs;
        }

        // Apply AdamW optimizer update through the grouped functions emitted by GraphBuilder.
        var group_start: usize = 0;
        var group_index: usize = 0;
        while (group_start < num_params) : ({
            group_start += GraphBuilder.optimizer_group_size;
            group_index += 1;
        }) {
            const group_end = @min(group_start + GraphBuilder.optimizer_group_size, num_params);
            const group_len = group_end - group_start;

            var opt_inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
            defer opt_inputs.deinit();
            defer {
                for (opt_inputs.items) |buf| buf.release();
            }

            for (device_params.*[group_start..group_end]) |p| {
                p.retain();
                try opt_inputs.append(p);
            }
            for (group_start..group_end) |i| {
                grad_outputs[i].retain();
                try opt_inputs.append(grad_outputs[i]);
            }

            for (self.m_states.?[group_start..group_end], group_start..) |m_state, i| {
                const m_buf = try iree_impl.moveToDevice(m_state, self.cached_parameter_shapes.?[i], .f32);
                try opt_inputs.append(m_buf);
            }
            for (self.v_states.?[group_start..group_end], group_start..) |v_state, i| {
                const v_buf = try iree_impl.moveToDevice(v_state, self.cached_parameter_shapes.?[i], .f32);
                try opt_inputs.append(v_buf);
            }

            const timestep_bytes = std.mem.asBytes(&self.timestep);
            const timestep_buf = try iree_impl.moveToDevice(timestep_bytes, &[_]i64{}, .f32);
            try opt_inputs.append(timestep_buf);

            const function_name = try std.fmt.allocPrint(self.allocator, "apply_optimizer_group_{d}", .{group_index});
            defer self.allocator.free(function_name);
            const opt_outputs = try iree_impl.executeWithDeviceBuffers(vmfb, function_name, opt_inputs.items);
            defer self.allocator.free(opt_outputs);

            var release_outputs_on_error = true;
            errdefer if (release_outputs_on_error) {
                for (opt_outputs) |buf| buf.release();
            };

            const expected_opt_outputs = 3 * group_len; // params + m_states + v_states
            if (opt_outputs.len < expected_opt_outputs) {
                std.log.err("Worker {}: {s} returned {} outputs, expected at least {}", .{ self.node_id.?, function_name, opt_outputs.len, expected_opt_outputs });
                return error.InsufficientOptimizerOutputs;
            }

            for (group_start..group_end, 0..) |param_index, local_index| {
                const m_output = opt_outputs[group_len + local_index];
                const m_bytes = try iree_impl.readToHost(m_output);
                if (m_bytes.len != self.m_states.?[param_index].len) {
                    self.allocator.free(m_bytes);
                    return error.OptimizerOutputShapeMismatch;
                }
                @memcpy(self.m_states.?[param_index], m_bytes);
                self.allocator.free(m_bytes);

                const v_output = opt_outputs[2 * group_len + local_index];
                const v_bytes = try iree_impl.readToHost(v_output);
                if (v_bytes.len != self.v_states.?[param_index].len) {
                    self.allocator.free(v_bytes);
                    return error.OptimizerOutputShapeMismatch;
                }
                @memcpy(self.v_states.?[param_index], v_bytes);
                self.allocator.free(v_bytes);
            }

            for (group_start..group_end, 0..) |param_index, local_index| {
                device_params.*[param_index].release();
                device_params.*[param_index] = opt_outputs[local_index];
                opt_outputs[group_len + local_index].release();
                opt_outputs[2 * group_len + local_index].release();
            }
            release_outputs_on_error = false;
        }

        self.timestep += 1.0;

        // Extract and log loss
        const loss_buf = grad_outputs[num_params];
        const loss_bytes = try iree_impl.readToHost(loss_buf);
        defer self.allocator.free(loss_bytes);

        var loss: f32 = 0.0;
        if (loss_bytes.len == 4) {
            loss = std.mem.bytesAsSlice(f32, loss_bytes)[0];
        } else if (loss_bytes.len == 2) {
            const bf16_bits = std.mem.readInt(u16, loss_bytes[0..2], .little);
            loss = @bitCast(@as(u32, bf16_bits) << 16);
        }

        if (self.current_step % 10 == 0) {
            std.log.info("Worker {}: Step {}, loss: {d:.4}", .{ self.node_id.?, self.current_step, loss });
        }

        return loss;
    }

    /// Extracts fragment P from the GPU, computes Delta, and sends it to the worker fabric.
    fn fragmentPhaseStart(self: *Self, fragment_id: usize) usize {
        return @as(usize, @intFromFloat(@floor(
            @as(f64, @floatFromInt(fragment_id)) *
                @as(f64, @floatFromInt(self.inner_steps)) /
                @as(f64, @floatFromInt(self.num_fragments)),
        )));
    }

    fn fragmentSendRound(self: *Self, fragment_id: usize, step: usize) usize {
        const phase_start = self.fragmentPhaseStart(fragment_id);
        return ((step - phase_start) / self.inner_steps) - 1;
    }

    fn fragmentReceiveRound(self: *Self, fragment_id: usize, step: usize) usize {
        const phase_start = self.fragmentPhaseStart(fragment_id);
        return ((step - phase_start - self.overlap_tau) / self.inner_steps) - 1;
    }

    fn clearReceivedFragments(self: *Self) void {
        self.rx_mutex.lock();
        defer self.rx_mutex.unlock();

        var it = self.rx_map.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinitClone(self.allocator);
        }
        self.rx_map.clearRetainingCapacity();
    }

    fn extractAndSendFragment(
        self: *Self,
        fragment_id: usize,
        fragment_round: usize,
        device_params: []IreeBackend.DeviceBuffer,
    ) !void {
        std.log.info("Worker {}: [t={}] Extracting and sending Fragment {} round {}", .{
            self.node_id.?, self.current_step, fragment_id, fragment_round,
        });

        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));
        const param_shapes = self.cached_parameter_shapes.?;

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        var updates = std.json.Array.init(arena.allocator());

        // Iterate over all tensors belonging to this fragment (strided assignment)
        // Device params and snapshots are always f32 (MLIR pipeline works in f32)
        for (param_shapes, 0..) |shape, i| {
            if (i % self.num_fragments != fragment_id) continue;

            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);

            // Download current tensor from GPU
            const current_bytes = try iree_impl.readToHost(device_params[i]);
            defer self.allocator.free(current_bytes);

            const current_f32 = std.mem.bytesAsSlice(f32, current_bytes);
            const snapshot_f32 = std.mem.bytesAsSlice(f32, self.tensor_snapshots.?[i]);

            // Compute Delta = Snapshot - Current
            var delta_f32 = try arena.allocator().alloc(f32, elem_count);
            for (0..elem_count) |idx| {
                delta_f32[idx] = snapshot_f32[idx] - current_f32[idx];
                snapshot_f32[idx] = current_f32[idx];
            }

            // Base64 encode the delta
            const delta_bytes = std.mem.sliceAsBytes(delta_f32);
            const b64_len = std.base64.standard.Encoder.calcSize(delta_bytes.len);
            const b64_chunk = try arena.allocator().alloc(u8, b64_len);
            _ = std.base64.standard.Encoder.encode(b64_chunk, delta_bytes);

            var update = std.json.ObjectMap.init(arena.allocator());
            try update.put("tensor_idx", std.json.Value{ .integer = @intCast(i) });
            try update.put("data", std.json.Value{ .string = b64_chunk });
            try updates.append(std.json.Value{ .object = update });
        }

        // Send FRAGMENT_UPDATE to the worker fabric.
        var msg_data = std.json.ObjectMap.init(arena.allocator());
        try msg_data.put("fragment_id", std.json.Value{ .integer = @intCast(fragment_id) });
        try msg_data.put("fragment_round", std.json.Value{ .integer = @intCast(fragment_round) });
        try msg_data.put("updates", std.json.Value{ .array = updates });

        const msg = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            message.MessageType.FRAGMENT_UPDATE,
            @intCast(self.current_step),
            .{
                .request_id = self.streaming_request_id,
                .round_id = @intCast(fragment_round + 1),
                .task_id = @intCast(fragment_id + 1),
            },
            std.json.Value{ .object = msg_data },
        );

        try self.enqueueTxMessageClone(msg);

        std.log.info("Worker {}: [t={}] Pushed Fragment {} round {} to Tx Queue.", .{
            self.node_id.?, self.current_step, fragment_id, fragment_round,
        });
    }

    /// Pops merged Fragment P from the Rx Map, Alpha blends it, and uploads to GPU
    fn receiveAndMergeFragment(
        self: *Self,
        fragment_id: usize,
        fragment_round: usize,
        device_params: *[]IreeBackend.DeviceBuffer,
    ) !void {
        std.log.info("Worker {}: [t={}] Time to merge Fragment {} round {}...", .{
            self.node_id.?, self.current_step, fragment_id, fragment_round,
        });

        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));
        const param_shapes = self.cached_parameter_shapes.?;
        const desired_key = FragmentKey{
            .request_id = self.streaming_request_id,
            .fragment_id = fragment_id,
            .fragment_round = fragment_round,
        };

        var maybe_msg: ?MessageEnvelope = null;
        while (self.is_running and maybe_msg == null) {
            self.rx_mutex.lock();
            if (self.rx_map.fetchRemove(desired_key)) |stored| {
                maybe_msg = stored.value;
                self.rx_mutex.unlock();
                break;
            }
            self.rx_mutex.unlock();

            std.log.warn("Worker {}:[t={}] Waiting for Fragment {} round {}...", .{
                self.node_id.?, self.current_step, fragment_id, fragment_round,
            });

            const receive_result = self.client.receive() catch |err| {
                std.log.err("Worker {}: Streaming receive failed: {}", .{ self.node_id.?, err });
                self.is_running = false;
                return err;
            };

            const incoming = receive_result.parsed.value;
            defer receive_result.parsed.deinit();
            defer self.allocator.free(receive_result.buffer);

            if (std.mem.eql(u8, incoming.msg_type, MessageType.SHUTDOWN)) {
                self.handleShutdown(incoming);
                break;
            }

            if (!std.mem.eql(u8, incoming.msg_type, MessageType.FRAGMENT_READY)) {
                std.log.warn("Worker {}: Ignoring unexpected streaming message type {s}", .{
                    self.node_id.?, incoming.msg_type,
                });
                continue;
            }

            const incoming_key = FragmentKey{
                .request_id = incoming.request_id,
                .fragment_id = @intCast(incoming.data.object.get("fragment_id").?.integer),
                .fragment_round = @intCast(incoming.data.object.get("fragment_round").?.integer),
            };

            const incoming_clone = try incoming.clone(self.allocator);
            if (std.meta.eql(incoming_key, desired_key)) {
                maybe_msg = incoming_clone;
                break;
            }

            self.rx_mutex.lock();
            if (self.rx_map.fetchRemove(incoming_key)) |existing| {
                var replaced = existing.value;
                replaced.deinitClone(self.allocator);
            }
            try self.rx_map.put(incoming_key, incoming_clone);
            self.rx_mutex.unlock();
        }

        if (maybe_msg == null) {
            return error.WorkerStopped;
        }

        const msg = maybe_msg.?;

        defer {
            var msg_mut = msg;
            msg_mut.deinitClone(self.allocator);
        }

        const merged_payload = msg.data.object;

        // Extract and apply updates from merged_payload
        const updates = merged_payload.get("updates").?.array;

        for (updates.items) |update_value| {
            const update = update_value.object;
            const tensor_idx = @as(usize, @intCast(update.get("tensor_idx").?.integer));

            // Decode the global weights from the worker fabric.
            const data_b64 = update.get("data").?.string;
            const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(data_b64);
            const global_bytes = try self.allocator.alloc(u8, decoded_size);
            defer self.allocator.free(global_bytes);
            try std.base64.standard.Decoder.decode(global_bytes, data_b64);

            // Download current weights from device (always f32)
            const current_bytes = try iree_impl.readToHost(device_params.*[tensor_idx]);
            defer self.allocator.free(current_bytes);

            // Alpha blending: new = alpha * current + (1 - alpha) * global
            const current_f32 = std.mem.bytesAsSlice(f32, current_bytes);
            const global_f32 = std.mem.bytesAsSlice(f32, global_bytes);
            const blended = try self.allocator.alloc(u8, current_bytes.len);
            defer self.allocator.free(blended);
            const blended_f32 = std.mem.bytesAsSlice(f32, blended);

            for (0..current_f32.len) |idx| {
                blended_f32[idx] = self.alpha * current_f32[idx] + (1.0 - self.alpha) * global_f32[idx];
            }

            // Upload blended weights back to device
            device_params.*[tensor_idx].release();
            device_params.*[tensor_idx] = try iree_impl.moveToDevice(blended, param_shapes[tensor_idx], .f32);

            // Update snapshot for this tensor
            @memcpy(self.tensor_snapshots.?[tensor_idx], blended);
        }

        std.log.info("Worker {}:[t={}] Fragment {} round {} merged successfully. Resuming compute.", .{
            self.node_id.?, self.current_step, fragment_id, fragment_round,
        });
    }

    fn initializeOptimizerStates(self: *Self, param_shapes: [][]i64) !void {
        const num_params = param_shapes.len;
        self.m_states = try self.allocator.alloc([]u8, num_params);
        self.v_states = try self.allocator.alloc([]u8, num_params);

        for (param_shapes, 0..) |shape, i| {
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);
            const size = elem_count * 4; // f32

            self.m_states.?[i] = try self.allocator.alloc(u8, size);
            @memset(self.m_states.?[i], 0);

            self.v_states.?[i] = try self.allocator.alloc(u8, size);
            @memset(self.v_states.?[i], 0);
        }
    }

    /// Handle weight update from the worker fabric.
    pub fn handleUpdateWeights(self: *Self, msg: MessageEnvelope) !void {
        std.log.info("Worker {}: Receiving weight update...", .{self.node_id.?});

        const payload = switch (msg.data) {
            .object => |o| o,
            else => return error.InvalidFormat,
        };

        if (payload.get("weights_path")) |weights_path_value| {
            const weights_path = switch (weights_path_value) {
                .string => |s| s,
                else => return error.InvalidWeightsPathFormat,
            };
            const weights_format = if (payload.get("weights_format")) |format_value| switch (format_value) {
                .string => |s| s,
                else => return error.InvalidWeightsFormat,
            } else "flat";
            try self.loadGenerationWeights(weights_path, weights_format);
            self.gen_engine.markWeightsDirty();
            std.log.info("Worker {}: Weights refreshed from local path: {s}", .{ self.node_id.?, weights_path });
            return;
        }

        const weights_value = payload.get("weights") orelse return error.MissingWeights;
        const b64_weights = switch (weights_value) {
            .string => |s| s,
            else => return error.InvalidWeightsFormat,
        };

        // Decode Base64
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_weights);

        // Free old weight blob if exists
        if (self.weight_blob) |old_blob| {
            self.allocator.free(old_blob);
        }
        self.active_weight_blob_request_id = null;

        // Allocate and decode new weights
        self.weight_blob = try self.allocator.alloc(u8, decoded_len);
        errdefer {
            self.allocator.free(self.weight_blob.?);
            self.weight_blob = null;
        }

        _ = try std.base64.standard.Decoder.decode(self.weight_blob.?, b64_weights);

        // Mark device weights as dirty so they get re-uploaded on next rollout
        self.gen_engine.markWeightsDirty();

        std.log.info("Worker {}: Weights updated ({} bytes)", .{ self.node_id.?, decoded_len });
    }

    /// Handle a Rollout Request (RL Generation) with Device Residency for Weights
    /// Weights stay on GPU VRAM (~2GB saved), KV Cache uses host-side accumulation (~25MB/step)
    pub fn handleStartRollout(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;

        // 1. Parse Payload
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };

        // Get Prompt (Input IDs)
        const prompt_json = payload.get("prompt") orelse return error.MissingPrompt;
        var prompt_list = std.ArrayList(i64).init(self.allocator);
        defer prompt_list.deinit();

        switch (prompt_json) {
            .array => |arr| {
                for (arr.items) |item| {
                    try prompt_list.append(item.integer);
                }
            },
            else => return error.InvalidPromptFormat,
        }

        const max_new_tokens = if (payload.get("max_new_tokens")) |value| switch (value) {
            .integer => |i| @as(usize, @intCast(i)),
            else => return error.InvalidMaxNewTokens,
        } else 128;

        // Cast backend to IreeBackend to access device residency methods
        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));

        const param_shapes = self.cached_parameter_shapes orelse return error.ParameterShapesNotInitialized;
        const data_shapes = self.cached_data_input_shapes orelse return error.DataShapesNotInitialized;

        // 1.5 PRE-LOAD SESSION (JIT compile CUDA kernels while GPU memory is free)
        if (self.cached_vmfb) |vmfb| {
            std.log.info("Worker {}: Pre-loading session (JIT compiling kernels)...", .{self.node_id.?});
            try iree_impl.loadSession(vmfb);
        }

        if (self.weight_blob == null) {
            std.log.err("Worker {}: No weights loaded for generation!", .{self.node_id.?});
            return error.WeightsNotInitialized;
        }

        const result = try self.gen_engine.generateRollout(
            iree_impl,
            &self.sampler,
            self.cached_vmfb.?,
            param_shapes,
            data_shapes,
            self.cached_data_input_dtypes,
            self.weight_blob orelse &[_]u8{},
            self.cached_parameter_dtypes,
            self.generation_weights_path,
            self.generation_weights_format orelse "flat",
            prompt_list.items,
            max_new_tokens,
            151643,
            &self.kv_cache,
        );
        defer self.allocator.free(result.tokens);

        std.log.info("Worker {d}: Generated {d} tokens: {any}", .{
            self.node_id.?,
            result.tokens.len,
            result.tokens,
        });

        // 5. Send Results Back
        var result_payload = std.json.ObjectMap.init(self.allocator);
        defer result_payload.deinit();

        var tokens_array = std.json.Array.init(self.allocator);
        defer tokens_array.deinit();
        for (result.tokens) |t| try tokens_array.append(.{ .integer = t });

        var prompt_array = std.json.Array.init(self.allocator);
        defer prompt_array.deinit();
        for (prompt_list.items) |t| try prompt_array.append(.{ .integer = t });

        try result_payload.put("prompt", .{ .array = prompt_array });
        try result_payload.put("completion", .{ .array = tokens_array });

        const response = tcp_stream.createMessageWithContext(
            self.node_id.?,
            "worker",
            0,
            "worker_fabric",
            MessageType.ROLLOUT_COMPLETE,
            msg.msg_id + 1,
            contextFromMessage(msg),
            .{ .object = result_payload },
        );

        try self.sendControllerMessage(response);
        self.state = .connected;
    }

    /// Handle Shutdown command from the worker fabric.
    pub fn handleShutdown(self: *Self, msg: MessageEnvelope) void {
        _ = msg;
        self.state = .shutting_down;
        self.is_running = false;
        self.streaming_request_id = 0;
        self.tx_cond.broadcast();
        self.rx_cond.broadcast();
    }

    /// Send periodic heartbeat to the worker fabric.
    pub fn sendHeartbeat(self: *Self) !void {
        if (self.state != .connected) return;

        const heartbeat = tcp_stream.createMessage(
            self.node_id.?, // our node_id
            "worker", // our service
            0, // worker-fabric node_id
            "worker_fabric",
            MessageType.HEARTBEAT,
            0, // heartbeat message id
            std.json.Value{ .string = "alive" },
        );

        self.sendControllerMessage(heartbeat) catch |err| {
            std.log.err("Failed to send heartbeat: {}", .{err});
        };
    }

    /// Get current worker state
    pub fn getState(self: Self) WorkerState {
        return self.state;
    }

    /// Get assigned node ID
    pub fn getNodeId(self: Self) ?NodeId {
        return self.node_id;
    }

    /// Disconnect from the worker fabric.
    pub fn disconnect(self: *Self) void {
        self.is_running = false;
        self.state = .shutting_down;
        self.client.disconnect();
    }
};

test "worker decoupled loop config validates bounded unsigned fields" {
    var payload = std.json.ObjectMap.init(std.testing.allocator);
    defer payload.deinit();
    try payload.put("num_fragments", .{ .integer = 4 });
    try payload.put(message.DecoupledDiLoCoField.SYNC_INTERVAL_H, .{ .integer = 8 });
    try payload.put(message.DecoupledDiLoCoField.OVERLAP_TAU, .{ .integer = 2 });
    try payload.put("max_syncer_steps", .{ .integer = 3 });

    const config = try Worker.parseDecoupledLoopConfig(payload, null);
    try std.testing.expectEqual(@as(usize, 4), config.num_fragments);
    try std.testing.expectEqual(@as(usize, 8), config.sync_interval_h);
    try std.testing.expectEqual(@as(usize, 3), config.max_syncer_steps);
    try std.testing.expectEqual(@as(usize, 104), config.max_local_steps);
}

test "worker decoupled loop config rejects invalid counts" {
    var missing_payload = std.json.ObjectMap.init(std.testing.allocator);
    defer missing_payload.deinit();
    try std.testing.expectError(error.MissingNumFragments, Worker.parseDecoupledLoopConfig(missing_payload, null));

    var zero_payload = std.json.ObjectMap.init(std.testing.allocator);
    defer zero_payload.deinit();
    try zero_payload.put("num_fragments", .{ .integer = 0 });
    try std.testing.expectError(error.InvalidFragmentCount, Worker.parseDecoupledLoopConfig(zero_payload, null));

    var negative_payload = std.json.ObjectMap.init(std.testing.allocator);
    defer negative_payload.deinit();
    try negative_payload.put("num_fragments", .{ .integer = -1 });
    try std.testing.expectError(error.InvalidUnsignedInteger, Worker.parseDecoupledLoopConfig(negative_payload, null));
}
