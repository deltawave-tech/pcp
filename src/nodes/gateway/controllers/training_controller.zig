/// Gateway worker-fabric controller.
/// Manages connected workers, worker leases, message routing, and local
/// controller orchestration for training, inference, and RL services.
const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const tcp_stream = @import("../../../network/tcp_stream.zig");
const message = @import("../../../network/message.zig");
const message_queue = @import("../../../network/message_queue.zig");
const training_algorithm = @import("../../../algorithms/training_algorithm.zig");
const execution = @import("../../../backends/executor.zig");
const monitoring = @import("../../../observability/monitoring.zig");
const backend_selection = @import("../../../backends/selection.zig");
const scheduling = @import("../scheduling.zig");
const data_manager = @import("data_manager.zig");
const mlir_ctx = @import("../../../mlir/context.zig");
const tensor = @import("../../../core/tensor.zig");
const protocol_invariants = @import("../../../protocol/invariants.zig");
const protocol_limits = @import("../../../protocol/limits.zig");
const message_registry = @import("../../../protocol/message_registry.zig");
const file_util = @import("../../../protocol/file_util.zig");
const runtime_config = @import("../../../runtime/config.zig");

const TcpServer = tcp_stream.TcpServer;
const TcpStreamManager = tcp_stream.TcpStreamManager;
const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
const MessageFilter = message.MessageFilter;
const MessageContext = message.MessageContext;
const NodeId = message.NodeId;
const MessageId = message.MessageId;
const RequestId = message.RequestId;
const RoundId = message.RoundId;
const TaskId = message.TaskId;
const Executor = execution.Executor;
const Backend = backend_selection.Backend;

pub const WorkerClass = scheduling.WorkerClass;
pub const WorkerSchedulingPolicy = scheduling.SchedulingPolicy;

/// Worker readiness status
pub const WorkerStatus = enum {
    Connected, // Worker connected but not initialized
    InitializingGraph, // Worker is loading VMFB / weights / optimizer state
    GraphInitialized, // Worker has loaded VMFB and is ready for tensors
    Training, // Worker is actively training
};

pub const WorkerLeaseOwner = enum {
    inference,
    training,
    rl,

    pub fn label(self: @This()) []const u8 {
        return switch (self) {
            .inference => "Inference",
            .training => "Training",
            .rl => "RL",
        };
    }
};

/// Worker configuration identifier for compiled artifacts
pub const WorkerConfig = struct {
    backend: Backend,
    target_arch: ?[]const u8,

    pub fn hash(self: @This(), hasher: anytype) void {
        std.hash.autoHash(hasher, self.backend);
        if (self.target_arch) |target| {
            hasher.update(target);
        }
    }

    pub fn eql(self: @This(), other: @This()) bool {
        if (self.backend != other.backend) return false;
        if (self.target_arch == null and other.target_arch == null) return true;
        if (self.target_arch == null or other.target_arch == null) return false;
        return std.mem.eql(u8, self.target_arch.?, other.target_arch.?);
    }
};

pub const WorkerConfigContext = struct {
    pub fn hash(self: @This(), key: WorkerConfig) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        key.hash(&hasher);
        return hasher.final();
    }

    pub fn eql(self: @This(), a: WorkerConfig, b: WorkerConfig) bool {
        _ = self;
        return a.eql(b);
    }
};

const ProgramArtifactKind = enum {
    training_vmfb,
    external_vmfb,
};

const ProgramDescriptor = struct {
    program_key: u64,
    source_hash: u64,
};

const CompiledArtifactKey = struct {
    kind: ProgramArtifactKind,
    backend: Backend,
    target_arch_hash: u64,
    compile_options_hash: u64,
    program_key: u64,
    source_hash: u64,
};

const CompiledArtifactKeyContext = struct {
    pub fn hash(self: @This(), key: CompiledArtifactKey) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        std.hash.autoHash(&hasher, key.kind);
        std.hash.autoHash(&hasher, key.backend);
        std.hash.autoHash(&hasher, key.target_arch_hash);
        std.hash.autoHash(&hasher, key.compile_options_hash);
        std.hash.autoHash(&hasher, key.program_key);
        std.hash.autoHash(&hasher, key.source_hash);
        return hasher.final();
    }

    pub fn eql(self: @This(), a: CompiledArtifactKey, b: CompiledArtifactKey) bool {
        _ = self;
        return a.kind == b.kind and
            a.backend == b.backend and
            a.target_arch_hash == b.target_arch_hash and
            a.compile_options_hash == b.compile_options_hash and
            a.program_key == b.program_key and
            a.source_hash == b.source_hash;
    }
};

const ProgramArtifact = struct {
    kind: ProgramArtifactKind,
    backend: Backend,
    target_arch_hash: u64,
    compile_options_hash: u64,
    program_key: u64,
    source_hash: u64,
    byte_hash: u64,
    artifact_key: u64,
    bytes: []const u8,
};

const MaterializedArtifactKey = struct {
    worker_id: NodeId,
    artifact_key: u64,
};

fn hashOptionalTarget(target_arch: ?[]const u8) u64 {
    var hasher = std.hash.Wyhash.init(0);
    std.hash.autoHash(&hasher, target_arch != null);
    if (target_arch) |target| hasher.update(target);
    return hasher.final();
}

fn effectiveTargetArch(config: WorkerConfig) ?[]const u8 {
    return switch (config.backend) {
        .rocm => config.target_arch orelse "gfx942",
        .cuda => config.target_arch orelse "sm_80",
        else => config.target_arch,
    };
}

fn compileOptionsHash(config: WorkerConfig) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update("iree-compile-v1");
    hasher.update(&[_]u8{0});
    hasher.update(config.backend.toIreeCompilationTarget());
    hasher.update(&[_]u8{0});
    if (effectiveTargetArch(config)) |target| {
        std.hash.autoHash(&hasher, true);
        hasher.update(target);
    } else {
        std.hash.autoHash(&hasher, false);
    }
    return hasher.final();
}

fn compiledArtifactKey(kind: ProgramArtifactKind, program: ProgramDescriptor, config: WorkerConfig) CompiledArtifactKey {
    return .{
        .kind = kind,
        .backend = config.backend,
        .target_arch_hash = hashOptionalTarget(config.target_arch),
        .compile_options_hash = compileOptionsHash(config),
        .program_key = program.program_key,
        .source_hash = program.source_hash,
    };
}

fn artifactKindLabel(kind: ProgramArtifactKind) []const u8 {
    return switch (kind) {
        .training_vmfb => "training_vmfb",
        .external_vmfb => "external_vmfb",
    };
}

fn programArtifact(kind: ProgramArtifactKind, program: ProgramDescriptor, config: WorkerConfig, bytes: []const u8) ProgramArtifact {
    const byte_hash = std.hash.Wyhash.hash(0, bytes);
    const target_arch_hash = hashOptionalTarget(config.target_arch);
    const options_hash = compileOptionsHash(config);
    var hasher = std.hash.Wyhash.init(0);
    std.hash.autoHash(&hasher, kind);
    std.hash.autoHash(&hasher, config.backend);
    std.hash.autoHash(&hasher, target_arch_hash);
    std.hash.autoHash(&hasher, options_hash);
    std.hash.autoHash(&hasher, program.program_key);
    std.hash.autoHash(&hasher, program.source_hash);
    std.hash.autoHash(&hasher, byte_hash);
    return .{
        .kind = kind,
        .backend = config.backend,
        .target_arch_hash = target_arch_hash,
        .compile_options_hash = options_hash,
        .program_key = program.program_key,
        .source_hash = program.source_hash,
        .byte_hash = byte_hash,
        .artifact_key = hasher.final(),
        .bytes = bytes,
    };
}

/// Represents a connected worker
pub const WorkerConnection = struct {
    node_id: NodeId,
    stable_worker_id: []const u8,
    stream: net.Stream,
    last_heartbeat: i64, // timestamp
    backend: Backend,
    status: WorkerStatus,
    address: net.Address, // Worker's network address
    target_arch: ?[]const u8, // GPU target architecture (e.g., gfx942 for MI300X, sm_80 for A100)
    lease_owner: ?WorkerLeaseOwner,
    loaded_program_key: ?u64,

    const Self = @This();

    pub fn init(
        node_id: NodeId,
        stable_worker_id: []const u8,
        stream: net.Stream,
        backend: Backend,
        address: net.Address,
        target_arch: ?[]const u8,
    ) Self {
        return Self{
            .node_id = node_id,
            .stable_worker_id = stable_worker_id,
            .stream = stream,
            .last_heartbeat = std.time.timestamp(),
            .backend = backend,
            .status = .Connected,
            .address = address,
            .target_arch = target_arch,
            .lease_owner = null,
            .loaded_program_key = null,
        };
    }

    pub fn deinit(self: Self, allocator: Allocator) void {
        allocator.free(self.stable_worker_id);
        if (self.target_arch) |arch| {
            allocator.free(arch);
        }
    }

    pub fn updateHeartbeat(self: *Self) void {
        self.last_heartbeat = std.time.timestamp();
    }

    pub fn isAlive(self: Self, timeout_seconds: i64) bool {
        const now = std.time.timestamp();
        return (now - self.last_heartbeat) < timeout_seconds;
    }
};

pub const WorkerMessageHook = struct {
    ctx: *anyopaque,
    handler: *const fn (ctx: *anyopaque, worker_id: NodeId, msg: MessageEnvelope) anyerror!bool,
};

pub const WorkerConnectionHook = struct {
    ctx: *anyopaque,
    on_connect: *const fn (ctx: *anyopaque, worker_id: NodeId) anyerror!void,
    on_disconnect: ?*const fn (ctx: *anyopaque, worker_id: NodeId) void,
};

pub const WorkerChangeHook = struct {
    ctx: *anyopaque,
    on_change: *const fn (ctx: *anyopaque) void,
};

pub const ProgramReadiness = struct {
    total: usize = 0,
    ready: usize = 0,
    initializing: usize = 0,
    connected: usize = 0,
    other_program: usize = 0,
};

/// State for reassembling chunked updates from a worker
pub const UpdateChunkState = struct {
    buffer: std.ArrayList(u8),
    expected_chunks: usize,
    received_chunks: usize,
    total_bytes: usize,
    loss: f32,
    chunk_id: i64,

    pub fn init(allocator: Allocator, total_bytes: usize, expected_chunks: usize) !UpdateChunkState {
        const buffer = try std.ArrayList(u8).initCapacity(allocator, total_bytes);
        return UpdateChunkState{
            .buffer = buffer,
            .expected_chunks = expected_chunks,
            .received_chunks = 0,
            .total_bytes = total_bytes,
            .loss = 0.0,
            .chunk_id = 0,
        };
    }

    pub fn deinit(self: *UpdateChunkState) void {
        self.buffer.deinit();
    }
};

const UpdateKey = struct {
    worker_id: NodeId,
    request_id: RequestId,
    round_id: RoundId,
    task_id: TaskId,
};

pub const ReassembledUpdate = struct {
    allocator: Allocator,
    bytes: []u8,
    loss: f32,
    chunk_id: i64,

    pub fn deinit(self: *ReassembledUpdate) void {
        self.allocator.free(self.bytes);
    }
};

pub const WorkerFabricController = struct {
    allocator: Allocator,
    server: ?TcpServer,
    listen_host: ?[]const u8,
    listen_port: u16,
    worker_pool: ArrayList(WorkerConnection),
    worker_pool_mutex: std.Thread.Mutex, // Protect worker_pool from concurrent access
    next_node_id: NodeId,
    next_message_id: MessageId,
    next_request_id: RequestId,
    protocol_id_mutex: std.Thread.Mutex,
    is_running: bool,
    stop_requested: std.atomic.Value(u8),
    algorithm: ?*training_algorithm.TrainingAlgorithm, // Training algorithm interface
    executor: ?Executor, // Generic execution backend for algorithms
    compiled_artifacts: std.HashMap(CompiledArtifactKey, []const u8, CompiledArtifactKeyContext, std.hash_map.default_max_load_percentage),
    worker_materialized_artifacts: std.AutoHashMap(MaterializedArtifactKey, void),
    materialized_artifacts_mutex: std.Thread.Mutex,
    // Message queue for collecting worker results
    result_queue: ArrayList(MessageEnvelope),
    result_queue_mutex: std.Thread.Mutex,
    // Data manager for chunk-based strict partitioning
    data_manager: ?data_manager.DataManager,

    // Chunked Update Transfer: Map worker/request/round/task -> reassembly state for incoming updates
    incoming_updates: std.AutoHashMap(UpdateKey, UpdateChunkState),
    incoming_updates_mutex: std.Thread.Mutex,

    // Supervisor Pattern: Map SupervisorID -> TCP Stream
    supervisors: std.AutoHashMap(i64, net.Stream),
    // Supervisor Pattern: Map Worker NodeID -> SupervisorID
    worker_map: std.AutoHashMap(NodeId, i64),
    message_hooks: ArrayList(WorkerMessageHook),
    connection_hooks: ArrayList(WorkerConnectionHook),
    worker_change_hooks: ArrayList(WorkerChangeHook),
    scheduling_policies: [3]WorkerSchedulingPolicy,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .server = null,
            .listen_host = null,
            .listen_port = 0,
            .worker_pool = ArrayList(WorkerConnection).init(allocator),
            .worker_pool_mutex = std.Thread.Mutex{},
            .next_node_id = 1, // Start from 1, 0 is reserved for coordinator
            .next_message_id = 1,
            .next_request_id = 1,
            .protocol_id_mutex = std.Thread.Mutex{},
            .is_running = false,
            .stop_requested = std.atomic.Value(u8).init(0),
            .algorithm = null,
            .executor = null,
            .compiled_artifacts = std.HashMap(CompiledArtifactKey, []const u8, CompiledArtifactKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .worker_materialized_artifacts = std.AutoHashMap(MaterializedArtifactKey, void).init(allocator),
            .materialized_artifacts_mutex = std.Thread.Mutex{},
            .result_queue = ArrayList(MessageEnvelope).init(allocator),
            .result_queue_mutex = std.Thread.Mutex{},
            .data_manager = null, // Initialized when training starts
            .incoming_updates = std.AutoHashMap(UpdateKey, UpdateChunkState).init(allocator),
            .incoming_updates_mutex = std.Thread.Mutex{},
            .supervisors = std.AutoHashMap(i64, net.Stream).init(allocator),
            .worker_map = std.AutoHashMap(NodeId, i64).init(allocator),
            .message_hooks = ArrayList(WorkerMessageHook).init(allocator),
            .connection_hooks = ArrayList(WorkerConnectionHook).init(allocator),
            .worker_change_hooks = ArrayList(WorkerChangeHook).init(allocator),
            .scheduling_policies = .{
                .{},
                .{},
                .{},
            },
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.server) |*server| {
            server.deinit();
        }

        // Close all worker connections
        for (self.worker_pool.items) |worker| {
            worker.stream.close();
            worker.deinit(self.allocator);
        }
        self.worker_pool.deinit();

        // Clean up incoming update buffers
        var update_iter = self.incoming_updates.valueIterator();
        while (update_iter.next()) |state| {
            state.deinit();
        }
        self.incoming_updates.deinit();

        // Close all supervisor connections
        var supervisor_iter = self.supervisors.valueIterator();
        while (supervisor_iter.next()) |stream| {
            stream.close();
        }
        self.supervisors.deinit();
        self.worker_map.deinit();
        self.message_hooks.deinit();
        self.connection_hooks.deinit();
        self.worker_change_hooks.deinit();

        // Clean up compiled artifacts
        var artifact_iter = self.compiled_artifacts.valueIterator();
        while (artifact_iter.next()) |vmfb_bytes| {
            self.allocator.free(vmfb_bytes.*);
        }
        self.compiled_artifacts.deinit();
        self.worker_materialized_artifacts.deinit();

        // Clean up result queue (free cloned messages)
        for (self.result_queue.items) |*msg| {
            msg.deinitClone(self.allocator);
        }
        self.result_queue.deinit();

        // Clean up executor if owned
        if (self.executor) |executor| {
            executor.deinit();
        }

        // Clean up data manager
        if (self.data_manager) |*dm| {
            dm.deinit();
        }
    }

    pub fn allocateMessageId(self: *Self) MessageId {
        self.protocol_id_mutex.lock();
        defer self.protocol_id_mutex.unlock();

        const msg_id = self.next_message_id;
        self.next_message_id += 1;
        return msg_id;
    }

    pub fn allocateRequestId(self: *Self) RequestId {
        self.protocol_id_mutex.lock();
        defer self.protocol_id_mutex.unlock();

        const request_id = self.next_request_id;
        self.next_request_id += 1;
        return request_id;
    }

    /// Snapshot current worker IDs for routing decisions.
    pub fn snapshotWorkerIds(self: *Self, allocator: Allocator) ![]NodeId {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        const count = self.worker_pool.items.len;
        std.debug.assert(count <= protocol_limits.worker_count_max);
        var ids = try allocator.alloc(NodeId, count);
        for (self.worker_pool.items, 0..) |worker, idx| {
            ids[idx] = worker.node_id;
        }
        return ids;
    }

    fn hashProgramParts(parts: []const []const u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        for (parts) |part| {
            hasher.update(part);
            hasher.update(&[_]u8{0});
        }
        return hasher.final();
    }

    fn programKeyForMlir(mlir_source: []const u8) u64 {
        return hashProgramParts(&[_][]const u8{ "mlir", mlir_source });
    }

    fn programDescriptorForMlir(mlir_source: []const u8) ProgramDescriptor {
        return .{
            .program_key = programKeyForMlir(mlir_source),
            .source_hash = hashProgramParts(&[_][]const u8{ "source", "mlir", mlir_source }),
        };
    }

    fn programKeyForVmfbPath(vmfb_path: []const u8) u64 {
        return hashProgramParts(&[_][]const u8{ "vmfb-path", vmfb_path });
    }

    fn programDescriptorForVmfbPath(vmfb_path: []const u8) ProgramDescriptor {
        return .{
            .program_key = programKeyForVmfbPath(vmfb_path),
            .source_hash = hashProgramParts(&[_][]const u8{ "source", "vmfb-path", vmfb_path }),
        };
    }

    fn programKeyForVmfbAndWeights(vmfb_path: []const u8, weights_path: []const u8) u64 {
        return hashProgramParts(&[_][]const u8{ "vmfb-path", vmfb_path, "weights-path", weights_path });
    }

    fn findWorkerPtrUnlocked(self: *Self, node_id: NodeId) ?*WorkerConnection {
        for (self.worker_pool.items) |*worker| {
            if (worker.node_id == node_id) return worker;
        }
        return null;
    }

    fn workerIsIdleUnlocked(worker: WorkerConnection) bool {
        return worker.lease_owner == null;
    }

    fn updateKeyForMessage(worker_id: NodeId, msg: MessageEnvelope) UpdateKey {
        return .{
            .worker_id = worker_id,
            .request_id = msg.request_id,
            .round_id = msg.round_id,
            .task_id = msg.task_id,
        };
    }

    fn requirePayloadUsize(payload: std.json.ObjectMap, key: []const u8) !usize {
        const value = payload.get(key) orelse return error.MissingPayloadField;
        return switch (value) {
            .integer => |inner| blk: {
                if (inner < 0) return error.InvalidPayloadInteger;
                break :blk @intCast(inner);
            },
            else => error.InvalidPayloadInteger,
        };
    }

    fn requirePayloadI64(payload: std.json.ObjectMap, key: []const u8) !i64 {
        const value = payload.get(key) orelse return error.MissingPayloadField;
        return switch (value) {
            .integer => |inner| inner,
            else => error.InvalidPayloadInteger,
        };
    }

    fn requirePayloadString(payload: std.json.ObjectMap, key: []const u8) ![]const u8 {
        const value = payload.get(key) orelse return error.MissingPayloadField;
        return switch (value) {
            .string => |inner| inner,
            else => error.InvalidPayloadString,
        };
    }

    fn requirePayloadF32(payload: std.json.ObjectMap, key: []const u8) !f32 {
        const value = payload.get(key) orelse return error.MissingPayloadField;
        return switch (value) {
            .float => |inner| @floatCast(inner),
            .integer => |inner| @floatFromInt(inner),
            else => error.InvalidPayloadFloat,
        };
    }

    /// Start listening for worker connections
    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        self.listen_host = host;
        self.listen_port = port;
        self.server = try TcpServer.init(self.allocator, host, port);
        self.is_running = true;

        std.log.info("Training controller listening on {s}:{} for worker connections", .{ host, port });

        // Start accepting connections
        while (self.is_running) {
            if (self.server) |*server| {
                const connection = server.accept() catch |err| {
                    if (!self.is_running) break;
                    std.log.err("Failed to accept connection: {}", .{err});
                    continue;
                };

                if (!self.is_running) {
                    connection.stream.close();
                    break;
                }

                // Handle connection in a new thread, passing both stream and address
                const thread = std.Thread.spawn(.{}, handleWorkerConnection, .{ self, connection.stream, connection.address }) catch |err| {
                    std.log.err("Failed to spawn worker handler thread: {}", .{err});
                    connection.stream.close();
                    continue;
                };
                thread.detach();
            }
        }

        if (self.server) |*server| {
            server.deinit();
            self.server = null;
        }
    }

    pub fn setMessageHook(self: *Self, hook: ?WorkerMessageHook) void {
        self.message_hooks.clearRetainingCapacity();
        if (hook) |h| {
            self.message_hooks.append(h) catch unreachable;
        }
    }

    pub fn setConnectionHook(self: *Self, hook: ?WorkerConnectionHook) void {
        self.connection_hooks.clearRetainingCapacity();
        if (hook) |h| {
            self.connection_hooks.append(h) catch unreachable;
        }
    }

    pub fn addMessageHook(self: *Self, hook: WorkerMessageHook) !void {
        try self.message_hooks.append(hook);
    }

    pub fn addConnectionHook(self: *Self, hook: WorkerConnectionHook) !void {
        try self.connection_hooks.append(hook);
    }

    pub fn addWorkerChangeHook(self: *Self, hook: WorkerChangeHook) !void {
        try self.worker_change_hooks.append(hook);
    }

    pub fn setSchedulingPolicy(self: *Self, owner: WorkerLeaseOwner, policy: WorkerSchedulingPolicy) void {
        self.worker_pool_mutex.lock();
        self.scheduling_policies[ownerIndex(owner)] = policy;
        self.worker_pool_mutex.unlock();
        self.notifyWorkerChangeHooks();
    }

    pub fn getSchedulingPolicy(self: *Self, owner: WorkerLeaseOwner) WorkerSchedulingPolicy {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        return self.scheduling_policies[ownerIndex(owner)];
    }

    /// Handle a new worker connection
    fn handleWorkerConnection(self: *Self, stream: net.Stream, worker_address: net.Address) !void {
        const join_msg_result = TcpStreamManager.receive(stream, self.allocator) catch |err| {
            std.log.err("Failed to receive join message: {}", .{err});
            stream.close();
            return;
        };
        defer join_msg_result.parsed.deinit();
        defer self.allocator.free(join_msg_result.buffer);

        const join_msg = join_msg_result.parsed.value;

        if (std.mem.eql(u8, join_msg.msg_type, MessageType.SUPERVISOR_HANDSHAKE)) {
            try self.handleSupervisorConnection(stream, join_msg);
            return;
        }

        if (!std.mem.eql(u8, join_msg.msg_type, MessageType.JOIN_REQUEST)) {
            std.log.err("Expected JoinRequest or SupervisorHandshake, got: {s}", .{join_msg.msg_type});
            stream.close();
            return;
        }
        try self.handleWorkerJoin(stream, worker_address, join_msg);
    }

    fn handleSupervisorConnection(self: *Self, stream: net.Stream, join_msg: MessageEnvelope) !void {
        defer stream.close();
        const payload = switch (join_msg.data) {
            .object => |obj| obj,
            else => return error.InvalidSupervisorHandshake,
        };
        const sid = switch (payload.get("supervisor_id") orelse return error.MissingSupervisorId) {
            .integer => |value| value,
            else => return error.InvalidSupervisorId,
        };

        self.worker_pool_mutex.lock();
        try self.supervisors.put(sid, stream);
        self.worker_pool_mutex.unlock();
        std.log.info("Supervisor {} re-registered control plane.", .{sid});

        var dummy: [1]u8 = undefined;
        while (self.is_running) {
            const bytes = stream.read(&dummy) catch |err| {
                std.log.warn("Supervisor {} control plane disconnected: {}", .{ sid, err });
                break;
            };
            if (bytes == 0) break;
        }

        self.worker_pool_mutex.lock();
        _ = self.supervisors.remove(sid);
        self.worker_pool_mutex.unlock();
    }

    fn handleWorkerJoin(self: *Self, stream: net.Stream, worker_address: net.Address, join_msg: MessageEnvelope) !void {
        defer stream.close();

        const data_obj = switch (join_msg.data) {
            .object => |obj| obj,
            else => return error.InvalidWorkerJoinPayload,
        };
        const backend_str = switch (data_obj.get("backend") orelse return error.MissingBackend) {
            .string => |value| value,
            else => return error.InvalidBackend,
        };

        const worker_backend = parseWorkerBackend(backend_str);

        // Parse and duplicate target architecture string to own it
        var owned_target_arch: ?[]const u8 = null;
        if (data_obj.get("target_arch")) |target_val| {
            switch (target_val) {
                .string => |s| {
                    owned_target_arch = try self.allocator.dupe(u8, s);
                },
                else => {},
            }
        }
        errdefer if (owned_target_arch) |s| self.allocator.free(s);

        std.log.info("Worker connecting with backend: {s}", .{worker_backend.toString()});
        if (owned_target_arch) |target| {
            std.log.info("  Target architecture: {s}", .{target});
        }
        const reported_worker_id: ?[]const u8 = if (data_obj.get("worker_id")) |worker_id_val| switch (worker_id_val) {
            .string => |value| value,
            else => null,
        } else null;

        // Assign new NodeId and add worker to pool (both protected by mutex)
        self.worker_pool_mutex.lock();
        const assigned_node_id = self.next_node_id;
        self.next_node_id += 1;

        const stable_worker_id = if (reported_worker_id) |value|
            self.allocator.dupe(u8, value) catch |err| {
                self.worker_pool_mutex.unlock();
                if (owned_target_arch) |s| self.allocator.free(s);
                return err;
            }
        else
            std.fmt.allocPrint(self.allocator, "worker-{d}", .{assigned_node_id}) catch |err| {
                self.worker_pool_mutex.unlock();
                if (owned_target_arch) |s| self.allocator.free(s);
                return err;
            };
        errdefer self.allocator.free(stable_worker_id);

        const worker = WorkerConnection.init(assigned_node_id, stable_worker_id, stream, worker_backend, worker_address, owned_target_arch);
        self.worker_pool.append(worker) catch |err| {
            self.worker_pool_mutex.unlock();
            self.allocator.free(stable_worker_id);
            if (owned_target_arch) |s| self.allocator.free(s);
            std.log.err("Failed to append worker {}: {}", .{ assigned_node_id, err });
            return;
        };

        // Link worker to supervisor if supervisor_id is present
        if (data_obj.get("supervisor_id")) |sid_val| {
            const sid = sid_val.integer;
            self.worker_map.put(assigned_node_id, sid) catch |err| {
                std.log.err("Failed to map worker {} to supervisor {}: {}", .{ assigned_node_id, sid, err });
            };
            std.log.info("Worker {} linked to Supervisor {}", .{ assigned_node_id, sid });
        }

        // Update monitoring while still holding mutex
        const worker_count = self.worker_pool.items.len;
        self.worker_pool_mutex.unlock();
        monitoring.setWorkerCount(worker_count);
        self.updateWorkerInfo();
        self.notifyWorkerChangeHooks();

        std.log.info("Worker {} connected as {s}", .{ assigned_node_id, stable_worker_id });

        // Send JoinAccept response
        const empty_obj = std.json.Value{ .object = std.json.ObjectMap.init(self.allocator) };
        const join_accept = tcp_stream.createMessage(
            0, // worker-fabric node_id
            "worker_fabric",
            assigned_node_id, // worker node_id
            "worker", // worker service
            MessageType.JOIN_ACCEPT,
            self.allocateMessageId(),
            empty_obj,
        );

        const json_buffer = join_accept.asJsonString(self.allocator) catch |err| {
            std.log.err("Failed to serialize JOIN_ACCEPT: {}", .{err});
            return;
        };
        defer json_buffer.deinit();

        TcpStreamManager.send(stream, join_accept, self.allocator) catch |err| {
            std.log.err("Failed to send join accept: {}", .{err});
            return;
        };

        for (self.connection_hooks.items) |hook| {
            hook.on_connect(hook.ctx, assigned_node_id) catch |err| {
                std.log.err("Worker {} connect hook failed: {}", .{ assigned_node_id, err });
            };
        }

        // Enter worker message handling loop
        try self.handleWorkerMessages(assigned_node_id, stream);
    }

    fn parseWorkerBackend(backend_str: []const u8) backend_selection.Backend {
        if (std.mem.eql(u8, backend_str, "cuda")) return .cuda;
        if (std.mem.eql(u8, backend_str, "rocm")) return .rocm;
        if (std.mem.eql(u8, backend_str, "metal")) return .metal;
        if (std.mem.eql(u8, backend_str, "cpu")) return .cpu;
        std.log.warn("Unknown worker backend '{s}', defaulting to cpu", .{backend_str});
        return .cpu;
    }

    /// Handle ongoing messages from a worker
    fn handleWorkerMessages(self: *Self, worker_id: NodeId, stream: net.Stream) !void {
        while (self.is_running) {
            const msg_result = TcpStreamManager.receive(stream, self.allocator) catch |err| {
                std.log.warn("Worker {} disconnected: {}", .{ worker_id, err });
                self.removeWorker(worker_id);
                // Update monitoring after worker removal
                monitoring.setWorkerCount(self.getWorkerCount());
                self.updateWorkerInfo();
                return;
            };
            defer msg_result.parsed.deinit();
            defer self.allocator.free(msg_result.buffer);

            const msg = msg_result.parsed.value;
            try self.dispatchWorkerMessage(worker_id, msg);
        }
    }

    fn dispatchWorkerMessage(self: *Self, worker_id: NodeId, msg: MessageEnvelope) !void {
        if (try self.dispatchWorkerMessageHooks(worker_id, msg)) return;
        if (std.mem.eql(u8, msg.msg_type, MessageType.HEARTBEAT)) {
            self.handleHeartbeat(worker_id);
            return;
        }
        if (std.mem.eql(u8, msg.msg_type, MessageType.INITIALIZE_GRAPH_COMPLETE)) {
            try self.handleInitializeGraphComplete(worker_id, msg);
            return;
        }
        if (std.mem.eql(u8, msg.msg_type, MessageType.UPDATE_CHUNK)) {
            try self.handleUpdateChunk(worker_id, msg);
            return;
        }
        if (self.isQueuedWorkerResultMessage(msg.msg_type)) {
            try self.queueWorkerResultMessage(worker_id, msg);
            return;
        }
        std.log.warn("Unknown message type from worker {}: {s}", .{ worker_id, msg.msg_type });
    }

    fn dispatchWorkerMessageHooks(self: *Self, worker_id: NodeId, msg: MessageEnvelope) !bool {
        for (self.message_hooks.items) |hook| {
            const handled = hook.handler(hook.ctx, worker_id, msg) catch |err| {
                std.log.err("Worker {} message hook failed: {}", .{ worker_id, err });
                return err;
            };
            if (handled) return true;
        }
        return false;
    }

    fn handleInitializeGraphComplete(self: *Self, worker_id: NodeId, msg: MessageEnvelope) !void {
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidInitializeGraphCompletePayload,
        };
        const program_key = if (payload.get("program_key")) |value| try parseProgramKeyValueStrict(value) else 0;
        if (payload.get("artifact_key")) |value| {
            const artifact_key = try parseProgramKeyValueStrict(value);
            if (artifact_key != 0) try self.markWorkerMaterializedArtifact(worker_id, artifact_key);
        }
        self.setWorkerGraphInitialized(worker_id, program_key);
    }

    fn isQueuedWorkerResultMessage(self: *Self, msg_type: []const u8) bool {
        _ = self;
        return message_registry.isGatewayQueuedWorkerResultMessage(msg_type);
    }

    fn queueWorkerResultMessage(self: *Self, worker_id: NodeId, msg: MessageEnvelope) !void {
        self.handleInnerLoopComplete(worker_id, msg) catch |err| {
            std.log.err("CRITICAL: Failed to queue worker result from worker {} for type {s}: {s}", .{
                worker_id,
                msg.msg_type,
                @errorName(err),
            });
            return err;
        };
    }

    /// Handle heartbeat from worker
    fn handleHeartbeat(self: *Self, worker_id: NodeId) void {
        for (self.worker_pool.items) |*worker| {
            if (worker.node_id == worker_id) {
                worker.updateHeartbeat();
                return;
            }
        }
    }

    /// Handle chunked update transfer from worker
    /// Reassembles chunks and queues the complete result when all chunks are received
    fn handleUpdateChunk(self: *Self, worker_id: NodeId, msg: MessageEnvelope) !void {
        const payload = switch (msg.data) {
            .object => |o| o,
            else => return error.InvalidFormat,
        };

        const chunk_index = try requirePayloadUsize(payload, "chunk_index");
        const total_chunks = try requirePayloadUsize(payload, "total_chunks");
        const total_bytes = try requirePayloadUsize(payload, "total_bytes");
        const b64_data = try requirePayloadString(payload, "data");
        const loss = try requirePayloadF32(payload, "loss");
        const chunk_id = try requirePayloadI64(payload, "chunk_id");
        if (total_chunks == 0) return error.InvalidChunkCount;
        if (chunk_index >= total_chunks) return error.InvalidChunkIndex;
        const update_bytes_cap = std.math.mul(usize, protocol_limits.chunk_bytes_max, total_chunks) catch return error.UpdateTooLarge;
        if (total_bytes > update_bytes_cap) return error.UpdateTooLarge;
        const update_key = updateKeyForMessage(worker_id, msg);

        self.incoming_updates_mutex.lock();
        defer self.incoming_updates_mutex.unlock();

        // Initialize buffer on first chunk
        if (chunk_index == 0) {
            // Clean up any existing state for this worker
            if (self.incoming_updates.getPtr(update_key)) |existing| {
                existing.deinit();
            }
            var state = try UpdateChunkState.init(self.allocator, total_bytes, total_chunks);
            state.loss = loss;
            state.chunk_id = chunk_id;
            try self.incoming_updates.put(update_key, state);
            std.log.info("Worker fabric: started receiving update from worker {} request {} round {} task {} ({} bytes in {} chunks)", .{
                worker_id,
                msg.request_id,
                msg.round_id,
                msg.task_id,
                total_bytes,
                total_chunks,
            });
        }

        // Get state (must exist after first chunk)
        const state = self.incoming_updates.getPtr(update_key) orelse return error.MissingChunkState;

        // Decode and append chunk
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_data);
        const decoded = try self.allocator.alloc(u8, decoded_len);
        defer self.allocator.free(decoded);
        try std.base64.standard.Decoder.decode(decoded, b64_data);

        try state.buffer.appendSlice(decoded);
        state.received_chunks += 1;

        if (state.received_chunks % 10 == 0 or state.received_chunks == state.expected_chunks) {
            std.log.info("Worker fabric: received chunk {}/{} from worker {} request {} round {} task {}", .{
                state.received_chunks,
                state.expected_chunks,
                worker_id,
                msg.request_id,
                msg.round_id,
                msg.task_id,
            });
        }

        // When all chunks received, queue a synthetic InnerLoopComplete message
        if (state.received_chunks == state.expected_chunks) {
            std.log.info("Worker fabric: reassembly complete for worker {} request {} round {} task {}. {} bytes received.", .{
                worker_id,
                msg.request_id,
                msg.round_id,
                msg.task_id,
                state.buffer.items.len,
            });

            // Store the reassembled data - we'll reference it by worker_id in diloco
            // The InnerLoopComplete message will signal that data is ready
            // We DON'T deinit the state here - it will be consumed by collectFromWorkers
        }
    }

    /// Take reassembled update data for a worker response (called by DiLoCo after InnerLoopComplete)
    pub fn takeReassembledUpdateForMessage(self: *Self, msg: MessageEnvelope) !?ReassembledUpdate {
        self.incoming_updates_mutex.lock();
        defer self.incoming_updates_mutex.unlock();
        if (self.incoming_updates.fetchRemove(updateKeyForMessage(msg.sender_node, msg))) |kv| {
            var state = kv.value;
            defer state.deinit();

            const bytes = try std.heap.c_allocator.alloc(u8, state.buffer.items.len);
            @memcpy(bytes, state.buffer.items);
            return .{
                .allocator = std.heap.c_allocator,
                .bytes = bytes,
                .loss = state.loss,
                .chunk_id = state.chunk_id,
            };
        }
        return null;
    }

    /// Handle inner loop completion from worker
    fn handleInnerLoopComplete(self: *Self, worker_id: NodeId, msg: MessageEnvelope) !void {
        std.log.info("Worker {} sent {s} result", .{ worker_id, msg.msg_type });

        // Clone the message to own the data (msg contains pointers to temporary buffers)
        const msg_clone = try msg.clone(self.allocator);
        errdefer msg_clone.deinitClone(self.allocator);

        // Add the cloned message to the result queue for collection by the main thread
        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        if (self.result_queue.items.len >= protocol_limits.message_queue_depth_max) return error.MessageQueueFull;
        try self.result_queue.append(msg_clone);

        std.log.info("Worker {} result successfully queued. Queue len: {}", .{ worker_id, self.result_queue.items.len });
    }

    /// Remove a worker from the pool
    fn removeWorker(self: *Self, worker_id: NodeId) void {
        var removed = false;
        self.worker_pool_mutex.lock();
        for (self.worker_pool.items, 0..) |worker, i| {
            if (worker.node_id == worker_id) {
                const removed_worker = self.worker_pool.swapRemove(i);
                removed_worker.deinit(self.allocator);
                std.log.info("Worker {} disconnected", .{worker_id});
                // Update monitoring after worker removal
                monitoring.setWorkerCount(self.worker_pool.items.len);
                self.updateWorkerInfoUnlocked();
                removed = true;
                break;
            }
        }
        self.worker_pool_mutex.unlock();

        if (!removed) return;
        self.clearWorkerMaterializedArtifacts(worker_id);
        self.notifyWorkerChangeHooks();
        for (self.connection_hooks.items) |hook| {
            if (hook.on_disconnect) |cb| {
                cb(hook.ctx, worker_id);
            }
        }
    }

    fn workerHasMaterializedArtifact(self: *Self, worker_id: NodeId, artifact_key: u64) bool {
        self.materialized_artifacts_mutex.lock();
        defer self.materialized_artifacts_mutex.unlock();
        return self.worker_materialized_artifacts.contains(.{
            .worker_id = worker_id,
            .artifact_key = artifact_key,
        });
    }

    fn markWorkerMaterializedArtifact(self: *Self, worker_id: NodeId, artifact_key: u64) !void {
        self.materialized_artifacts_mutex.lock();
        defer self.materialized_artifacts_mutex.unlock();
        try self.worker_materialized_artifacts.put(.{
            .worker_id = worker_id,
            .artifact_key = artifact_key,
        }, {});
    }

    fn clearWorkerMaterializedArtifacts(self: *Self, worker_id: NodeId) void {
        self.materialized_artifacts_mutex.lock();
        defer self.materialized_artifacts_mutex.unlock();
        var keys_to_remove = ArrayList(MaterializedArtifactKey).init(self.allocator);
        defer keys_to_remove.deinit();

        var iter = self.worker_materialized_artifacts.keyIterator();
        while (iter.next()) |key| {
            if (key.worker_id == worker_id) {
                keys_to_remove.append(key.*) catch |err| {
                    std.log.warn("Failed to queue materialized artifact cleanup for worker {}: {}", .{ worker_id, err });
                    break;
                };
            }
        }

        for (keys_to_remove.items) |key| {
            _ = self.worker_materialized_artifacts.remove(key);
        }
    }

    /// Get current number of connected workers
    pub fn getWorkerCount(self: *Self) usize {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        return self.worker_pool.items.len;
    }

    pub fn getWorkerCountForLeaseOwner(self: *Self, owner: WorkerLeaseOwner) usize {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        var count: usize = 0;
        for (self.worker_pool.items) |worker| {
            if (worker.lease_owner) |lease_owner| {
                if (lease_owner == owner) count += 1;
            }
        }
        return count;
    }

    pub fn getUsableWorkerCountForLeaseOwner(self: *Self, owner: WorkerLeaseOwner) usize {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        return self.countLeasedWorkersUnlocked(owner) + self.countAdditionalAcquirableWorkersUnlocked(owner);
    }

    pub fn getReadyWorkerCountForLeaseOwner(self: *Self, owner: WorkerLeaseOwner) usize {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        var count: usize = 0;
        for (self.worker_pool.items) |worker| {
            if (worker.lease_owner) |lease_owner| {
                if (lease_owner != owner) continue;
            } else continue;
            if (worker.status == .GraphInitialized or worker.status == .Training) {
                count += 1;
            }
        }
        return count;
    }

    pub fn getAvailableWorkerCountForLeaseOwner(self: *Self, owner: WorkerLeaseOwner) usize {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        return self.countAdditionalAcquirableWorkersUnlocked(owner);
    }

    pub fn canAcquireWorker(self: *Self, node_id: NodeId, owner: WorkerLeaseOwner) bool {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        return self.canLeaseWorkerUnlocked(node_id, owner);
    }

    pub fn desiredProgramKeyForVmfbPath(_: *Self, vmfb_path: []const u8) u64 {
        return programKeyForVmfbPath(vmfb_path);
    }

    pub fn desiredProgramKeyForVmfbAndWeights(_: *Self, vmfb_path: []const u8, weights_path: []const u8) u64 {
        return programKeyForVmfbAndWeights(vmfb_path, weights_path);
    }

    pub fn getProgramReadinessForIds(self: *Self, worker_ids: []const NodeId, desired_program_key: u64) ProgramReadiness {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        var readiness = ProgramReadiness{};
        for (worker_ids) |node_id| {
            const worker = self.findWorkerPtrUnlocked(node_id) orelse continue;
            readiness.total += 1;

            const matches_program = worker.loaded_program_key != null and worker.loaded_program_key.? == desired_program_key;
            if (matches_program and (worker.status == .GraphInitialized or worker.status == .Training)) {
                readiness.ready += 1;
                continue;
            }

            switch (worker.status) {
                .InitializingGraph => {
                    if (matches_program) {
                        readiness.initializing += 1;
                    } else {
                        readiness.other_program += 1;
                    }
                },
                .Connected => readiness.connected += 1,
                .GraphInitialized, .Training => readiness.other_program += 1,
            }
        }
        return readiness;
    }

    pub fn getIdleWorkerCount(self: *Self) usize {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        var count: usize = 0;
        for (self.worker_pool.items) |worker| {
            if (workerIsIdleUnlocked(worker)) count += 1;
        }
        return count;
    }

    /// Update worker information in monitoring system (internal, assumes mutex is held)
    fn updateWorkerInfoUnlocked(self: *Self) void {
        // Create worker info array (static allocation is fine since we copy immediately)
        var worker_info_buf: [16]monitoring.WorkerInfo = undefined;
        var count: usize = 0;

        for (self.worker_pool.items) |worker| {
            if (count >= worker_info_buf.len) break;

            // Format IP address from stored address
            var ip_buf: [64]u8 = undefined;
            const ip_str = std.fmt.bufPrint(&ip_buf, "{}", .{worker.address}) catch "unknown";

            // Get backend and status strings
            const backend_str = worker.backend.toString();
            const status_str = if (worker.lease_owner) |owner|
                owner.label()
            else switch (worker.status) {
                .Connected => "Connected",
                .InitializingGraph => "InitGraph",
                .GraphInitialized => "Initialized",
                .Training => "Training",
            };

            // Create WorkerInfo with proper string copying
            var info = monitoring.WorkerInfo{
                .node_id = worker.node_id,
                .backend = [_]u8{0} ** 16,
                .backend_len = @min(backend_str.len, 16),
                .ip_address = [_]u8{0} ** 64,
                .ip_len = @min(ip_str.len, 64),
                .status = [_]u8{0} ** 16,
                .status_len = @min(status_str.len, 16),
            };

            // Copy strings into fixed buffers
            @memcpy(info.backend[0..info.backend_len], backend_str[0..info.backend_len]);
            @memcpy(info.ip_address[0..info.ip_len], ip_str[0..info.ip_len]);
            @memcpy(info.status[0..info.status_len], status_str[0..info.status_len]);

            worker_info_buf[count] = info;
            count += 1;
        }

        // Update monitoring with worker info (monitoring will copy the data)
        monitoring.setWorkerInfo(worker_info_buf[0..count]);
    }

    /// Update worker information in monitoring system (public, acquires mutex)
    pub fn updateWorkerInfo(self: *Self) void {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        self.updateWorkerInfoUnlocked();
    }

    /// Set the training algorithm
    pub fn setAlgorithm(self: *Self, algorithm: *training_algorithm.TrainingAlgorithm) void {
        self.algorithm = algorithm;
    }

    /// Set the executor for algorithms to use
    pub fn setExecutor(self: *Self, executor: Executor) void {
        self.executor = executor;
    }

    /// Get the executor for algorithms to use
    pub fn getExecutor(self: *Self) ?Executor {
        return self.executor;
    }

    /// Initialize the data manager for chunk-based data partitioning
    pub fn initDataManager(self: *Self, total_size: usize, chunk_size: usize, max_epochs: usize) !void {
        if (self.data_manager != null) return; // Already initialized
        self.data_manager = try data_manager.DataManager.init(self.allocator, total_size, chunk_size, max_epochs);
        std.log.info("DataManager initialized: {} total size, {} chunk size, {} max epochs", .{ total_size, chunk_size, max_epochs });
    }

    pub fn resetDataManager(self: *Self, total_size: usize, chunk_size: usize, max_epochs: usize) !void {
        if (self.data_manager) |*dm| {
            dm.deinit();
        }
        self.data_manager = try data_manager.DataManager.init(self.allocator, total_size, chunk_size, max_epochs);
        std.log.info("DataManager reset: {} total size, {} chunk size, {} max epochs", .{ total_size, chunk_size, max_epochs });
    }

    pub fn clearTransientJobState(self: *Self) void {
        self.clearQueuedWorkerResults();
        self.clearIncomingUpdateTransfers();
    }

    fn clearQueuedWorkerResults(self: *Self) void {
        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        for (self.result_queue.items) |*msg| {
            msg.deinitClone(self.allocator);
        }
        self.result_queue.clearRetainingCapacity();
    }

    fn clearIncomingUpdateTransfers(self: *Self) void {
        self.incoming_updates_mutex.lock();
        defer self.incoming_updates_mutex.unlock();

        var it = self.incoming_updates.valueIterator();
        while (it.next()) |state| {
            state.deinit();
        }
        self.incoming_updates.clearRetainingCapacity();
    }

    /// Start training once we have enough workers
    pub fn startTraining(self: *Self, required_workers: usize) !void {
        std.log.info("Waiting for {} workers to join...", .{required_workers});
        try self.waitForWorkerCount(protocol_limits.workerJoinBudget(required_workers));

        self.worker_pool_mutex.lock();
        const worker_count = self.worker_pool.items.len;
        self.worker_pool_mutex.unlock();
        std.log.info("Got {} workers, starting training...", .{worker_count});

        if (self.stop_requested.load(.acquire) == 1) return error.Cancelled;

        if (self.algorithm) |algo| {
            try algo.run();
            std.log.info("Training algorithm completed successfully", .{});
        } else {
            std.log.warn("No training algorithm set", .{});
        }
    }

    fn waitForWorkerCount(self: *Self, budget: protocol_limits.WaitBudget) !void {
        try budget.validate();
        const start_ms = std.time.milliTimestamp();
        var next_log_ms: u64 = 5_000;

        while (self.is_running) {
            if (self.stop_requested.load(.acquire) == 1) return error.Cancelled;

            const current_count = self.getWorkerCount();
            if (current_count >= budget.expected_count) return;

            if (budget.expired(start_ms)) {
                std.log.warn("Timed out waiting for {s}: {}/{}", .{
                    budget.reason,
                    current_count,
                    budget.expected_count,
                });
                return error.WorkerJoinTimeout;
            }

            const elapsed_ms = budget.elapsedMs(start_ms);
            if (elapsed_ms >= next_log_ms) {
                std.log.info("Waiting for {s}: {}/{} after {}ms", .{
                    budget.reason,
                    current_count,
                    budget.expected_count,
                    elapsed_ms,
                });
                next_log_ms += 5_000;
            }

            std.time.sleep(budget.sleepMs(start_ms) * std.time.ns_per_ms);
        }
        return error.Cancelled;
    }

    /// Broadcast a message to all workers
    pub fn broadcastToWorkers(self: *Self, msg_type: []const u8, data: std.json.Value) !void {
        return self.broadcastToWorkersWithContext(msg_type, data, .{});
    }

    pub fn broadcastToWorkersWithContext(self: *Self, msg_type: []const u8, data: std.json.Value, context: MessageContext) !void {
        const worker_ids = try self.snapshotWorkerIds(self.allocator);
        defer self.allocator.free(worker_ids);
        return self.broadcastToWorkerIdsWithContext(worker_ids, msg_type, data, context);
    }

    pub fn broadcastToWorkerIdsWithContext(self: *Self, worker_ids: []const NodeId, msg_type: []const u8, data: std.json.Value, context: MessageContext) !void {
        var workers_snapshot = try self.snapshotWorkersByIds(worker_ids);
        defer workers_snapshot.deinit();

        for (workers_snapshot.items) |worker| {
            const msg = tcp_stream.createMessageWithContext(
                0, // worker-fabric node_id
                "worker_fabric",
                worker.node_id, // worker node_id
                "worker", // worker service
                msg_type,
                self.allocateMessageId(),
                context,
                data,
            );

            TcpStreamManager.send(worker.stream, msg, self.allocator) catch |err| {
                std.log.err("Failed to send message to worker {}: {}", .{ worker.node_id, err });
            };
        }
    }

    /// Broadcast large binary data to all workers using chunked transfer protocol.
    /// Splits data into 10MB chunks to avoid Cap'n Proto message size limits.
    pub fn broadcastLargeData(self: *Self, data: []const u8) !void {
        return self.broadcastLargeDataWithContext(data, .{});
    }

    /// Broadcast large binary data to all workers using chunked transfer protocol
    /// while preserving request/round/task metadata for the transfer.
    pub fn broadcastLargeDataWithContext(self: *Self, data: []const u8, context: MessageContext) !void {
        const worker_ids = try self.snapshotWorkerIds(self.allocator);
        defer self.allocator.free(worker_ids);
        return self.broadcastLargeDataToWorkerIdsWithContext(worker_ids, data, context);
    }

    pub fn broadcastLargeDataToWorkerIdsWithContext(self: *Self, worker_ids: []const NodeId, data: []const u8, context: MessageContext) !void {
        return try self.broadcastLargeDataToWorkerIdsWithContextAndName(worker_ids, data, context, null);
    }

    pub fn broadcastNamedLargeDataToWorkerIdsWithContext(
        self: *Self,
        worker_ids: []const NodeId,
        data: []const u8,
        context: MessageContext,
        blob_name: []const u8,
    ) !void {
        return try self.broadcastLargeDataToWorkerIdsWithContextAndName(worker_ids, data, context, blob_name);
    }

    fn broadcastLargeDataToWorkerIdsWithContextAndName(
        self: *Self,
        worker_ids: []const NodeId,
        data: []const u8,
        context: MessageContext,
        blob_name: ?[]const u8,
    ) !void {
        const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB per chunk
        const total_size = data.len;
        const total_chunks = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        std.log.info("Broadcasting {} bytes in {} chunks as {s}...", .{ total_size, total_chunks, blob_name orelse "default" });

        var offset: usize = 0;
        var chunk_idx: usize = 0;
        while (chunk_idx < total_chunks) : (chunk_idx += 1) {
            const end = @min(offset + CHUNK_SIZE, total_size);
            const chunk_slice = data[offset..end];

            // Base64 encode the chunk
            const b64_len = std.base64.standard.Encoder.calcSize(chunk_slice.len);
            const b64_chunk = try self.allocator.alloc(u8, b64_len);
            defer self.allocator.free(b64_chunk);
            _ = std.base64.standard.Encoder.encode(b64_chunk, chunk_slice);

            // Construct Payload
            var payload = std.json.ObjectMap.init(self.allocator);
            defer payload.deinit();

            try payload.put("chunk_index", std.json.Value{ .integer = @intCast(chunk_idx) });
            try payload.put("total_chunks", std.json.Value{ .integer = @intCast(total_chunks) });
            try payload.put("data", std.json.Value{ .string = b64_chunk });
            try payload.put("total_bytes", std.json.Value{ .integer = @intCast(total_size) });
            if (blob_name) |name| {
                try payload.put("blob_name", std.json.Value{ .string = name });
            }

            // Broadcast using existing method
            try self.broadcastToWorkerIdsWithContext(worker_ids, MessageType.WEIGHT_CHUNK, .{ .object = payload }, context);

            offset = end;

            // Small sleep every 5 chunks to prevent overwhelming network buffers
            if (chunk_idx % 5 == 4) std.time.sleep(10 * std.time.ns_per_ms);
        }
        std.log.info("✓ Broadcast complete ({} chunks sent).", .{total_chunks});
    }

    /// Send a message to a specific worker by its NodeId
    pub fn sendToWorker(self: *Self, node_id: NodeId, msg_type: []const u8, data: std.json.Value) !void {
        return self.sendToWorkerWithContext(node_id, msg_type, data, .{});
    }

    pub fn sendToWorkerWithContext(self: *Self, node_id: NodeId, msg_type: []const u8, data: std.json.Value, context: MessageContext) !void {
        const worker_ids = [_]NodeId{node_id};
        var workers_snapshot = try self.snapshotWorkersByIds(&worker_ids);
        defer workers_snapshot.deinit();

        if (workers_snapshot.items.len == 0) return error.WorkerNotFound;
        const worker = workers_snapshot.items[0];
        const msg = tcp_stream.createMessageWithContext(
            0,
            "worker_fabric",
            worker.node_id,
            "worker",
            msg_type,
            self.allocateMessageId(),
            context,
            data,
        );

        TcpStreamManager.send(worker.stream, msg, self.allocator) catch |err| {
            std.log.err("Failed to send message to worker {}: {}", .{ worker.node_id, err });
        };
    }

    /// Snapshot of currently connected workers for the current training round
    pub fn snapshotWorkers(self: *Self) !ArrayList(WorkerConnection) {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        const count = self.worker_pool.items.len;
        std.debug.assert(count <= protocol_limits.worker_count_max);
        var snapshot = try ArrayList(WorkerConnection).initCapacity(self.allocator, count);
        for (self.worker_pool.items) |worker| {
            snapshot.appendAssumeCapacity(worker);
        }
        return snapshot;
    }

    pub fn snapshotWorkersByIds(self: *Self, worker_ids: []const NodeId) !ArrayList(WorkerConnection) {
        var snapshot = try ArrayList(WorkerConnection).initCapacity(self.allocator, worker_ids.len);
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        for (worker_ids) |node_id| {
            if (self.findWorkerPtrUnlocked(node_id)) |worker| {
                snapshot.appendAssumeCapacity(worker.*);
            }
        }
        return snapshot;
    }

    /// Update worker status
    pub fn setWorkerStatus(self: *Self, node_id: NodeId, status: WorkerStatus) void {
        var changed = false;
        self.worker_pool_mutex.lock();
        for (self.worker_pool.items) |*w| {
            if (w.node_id == node_id) {
                w.status = status;
                self.updateWorkerInfoUnlocked();
                changed = true;
                break;
            }
        }
        self.worker_pool_mutex.unlock();
        if (changed) {
            self.notifyWorkerChangeHooks();
        }
    }

    fn setWorkerGraphInitializing(self: *Self, node_id: NodeId, program_key: u64) void {
        var changed = false;
        self.worker_pool_mutex.lock();
        if (self.findWorkerPtrUnlocked(node_id)) |worker| {
            worker.status = .InitializingGraph;
            worker.loaded_program_key = program_key;
            self.updateWorkerInfoUnlocked();
            changed = true;
        }
        self.worker_pool_mutex.unlock();
        if (changed) {
            self.notifyWorkerChangeHooks();
        }
    }

    fn setWorkerGraphInitialized(self: *Self, node_id: NodeId, program_key: u64) void {
        var changed = false;
        self.worker_pool_mutex.lock();
        if (self.findWorkerPtrUnlocked(node_id)) |worker| {
            worker.status = .GraphInitialized;
            worker.loaded_program_key = program_key;
            self.updateWorkerInfoUnlocked();
            changed = true;
        }
        self.worker_pool_mutex.unlock();
        if (changed) {
            self.notifyWorkerChangeHooks();
        }
    }

    pub fn tryAcquireWorker(self: *Self, node_id: NodeId, owner: WorkerLeaseOwner) bool {
        var acquired = false;
        self.worker_pool_mutex.lock();
        if (self.findWorkerPtrUnlocked(node_id)) |worker| {
            if (worker.lease_owner == null and self.canLeaseWorkerUnlocked(node_id, owner)) {
                worker.lease_owner = owner;
                self.updateWorkerInfoUnlocked();
                acquired = true;
            }
        }
        self.worker_pool_mutex.unlock();
        if (acquired) {
            self.notifyWorkerChangeHooks();
        }
        return acquired;
    }

    pub fn acquireIdleWorkers(self: *Self, allocator: Allocator, owner: WorkerLeaseOwner, count: usize) ![]NodeId {
        if (count == 0) return error.InvalidWorkerCount;
        if (count > protocol_limits.worker_count_max) return error.WorkerCountTooLarge;
        var leased = try allocator.alloc(NodeId, count);
        errdefer allocator.free(leased);

        self.worker_pool_mutex.lock();

        var idx: usize = 0;
        for (self.worker_pool.items) |*worker| {
            if (!workerIsIdleUnlocked(worker.*)) continue;
            if (!self.canLeaseWorkerUnlocked(worker.node_id, owner)) continue;
            worker.lease_owner = owner;
            leased[idx] = worker.node_id;
            idx += 1;
            if (idx == count) break;
        }
        if (idx < count) {
            for (leased[0..idx]) |node_id| {
                if (self.findWorkerPtrUnlocked(node_id)) |worker| {
                    worker.lease_owner = null;
                }
            }
            if (idx > 0) {
                self.updateWorkerInfoUnlocked();
            }
            self.worker_pool_mutex.unlock();
            return error.NotEnoughIdleWorkers;
        }
        self.updateWorkerInfoUnlocked();
        self.worker_pool_mutex.unlock();
        protocol_invariants.assertWorkerCount(leased.len);
        self.notifyWorkerChangeHooks();
        return leased;
    }

    pub fn releaseWorker(self: *Self, node_id: NodeId) void {
        var changed = false;
        self.worker_pool_mutex.lock();
        if (self.findWorkerPtrUnlocked(node_id)) |worker| {
            worker.lease_owner = null;
            self.updateWorkerInfoUnlocked();
            changed = true;
        }
        self.worker_pool_mutex.unlock();
        if (changed) {
            self.notifyWorkerChangeHooks();
        }
    }

    pub fn releaseWorkers(self: *Self, worker_ids: []const NodeId) void {
        self.worker_pool_mutex.lock();
        var changed = false;
        for (worker_ids) |node_id| {
            if (self.findWorkerPtrUnlocked(node_id)) |worker| {
                worker.lease_owner = null;
                changed = true;
            }
        }
        if (changed) {
            self.updateWorkerInfoUnlocked();
        }
        self.worker_pool_mutex.unlock();
        if (changed) {
            self.notifyWorkerChangeHooks();
        }
    }

    pub fn filterIdleWorkerIds(self: *Self, allocator: Allocator, worker_ids: []const NodeId) ![]NodeId {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        var idle_count: usize = 0;
        for (worker_ids) |node_id| {
            if (self.findWorkerPtrUnlocked(node_id)) |worker| {
                if (workerIsIdleUnlocked(worker.*)) idle_count += 1;
            }
        }

        var filtered = try allocator.alloc(NodeId, idle_count);
        var idx: usize = 0;
        for (worker_ids) |node_id| {
            if (self.findWorkerPtrUnlocked(node_id)) |worker| {
                if (workerIsIdleUnlocked(worker.*)) {
                    filtered[idx] = worker.node_id;
                    idx += 1;
                }
            }
        }
        return filtered;
    }

    pub fn filterLeasedWorkerIds(self: *Self, allocator: Allocator, owner: WorkerLeaseOwner, worker_ids: []const NodeId) ![]NodeId {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        var filtered_count: usize = 0;
        for (worker_ids) |node_id| {
            const worker = self.findWorkerPtrUnlocked(node_id) orelse continue;
            if (worker.lease_owner == owner) filtered_count += 1;
        }

        var filtered = try allocator.alloc(NodeId, filtered_count);
        var idx: usize = 0;
        for (worker_ids) |node_id| {
            const worker = self.findWorkerPtrUnlocked(node_id) orelse continue;
            if (worker.lease_owner == owner) {
                filtered[idx] = worker.node_id;
                idx += 1;
            }
        }
        return filtered;
    }

    pub fn filterAcquirableWorkerIds(self: *Self, allocator: Allocator, owner: WorkerLeaseOwner, worker_ids: []const NodeId) ![]NodeId {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        var eligible_count: usize = 0;
        for (worker_ids) |node_id| {
            if (self.findWorkerPtrUnlocked(node_id)) |worker| {
                if (!workerIsIdleUnlocked(worker.*)) continue;
                if (!self.canLeaseWorkerUnlocked(node_id, owner)) continue;
                eligible_count += 1;
            }
        }

        var filtered = try allocator.alloc(NodeId, eligible_count);
        var idx: usize = 0;
        for (worker_ids) |node_id| {
            if (self.findWorkerPtrUnlocked(node_id)) |worker| {
                if (workerIsIdleUnlocked(worker.*) and self.canLeaseWorkerUnlocked(node_id, owner)) {
                    filtered[idx] = worker.node_id;
                    idx += 1;
                }
            }
        }
        return filtered;
    }

    fn canLeaseWorkerUnlocked(self: *Self, node_id: NodeId, owner: WorkerLeaseOwner) bool {
        const worker = self.findWorkerPtrUnlocked(node_id) orelse return false;
        if (worker.lease_owner != null) return false;

        const policy = self.scheduling_policies[ownerIndex(owner)];
        if (!workerMatchesPolicy(policy, worker.*)) return false;

        if (policy.max_workers) |max_workers| {
            if (self.countLeasedWorkersUnlocked(owner) >= max_workers) return false;
        }

        return self.reservationsSatisfiedAfterLeaseUnlocked(node_id, owner);
    }

    fn countAdditionalAcquirableWorkersUnlocked(self: *Self, owner: WorkerLeaseOwner) usize {
        const worker_count = self.worker_pool.items.len;
        if (worker_count == 0) return 0;
        std.debug.assert(worker_count <= protocol_limits.worker_count_max);

        var hypothetical_storage: [protocol_limits.worker_count_max]bool = undefined;
        const hypothetical = hypothetical_storage[0..worker_count];
        @memset(hypothetical, false);

        var count: usize = 0;
        while (count < worker_count) {
            var progress = false;
            for (self.worker_pool.items, 0..) |worker, idx| {
                if (!workerIsIdleUnlocked(worker)) continue;
                if (hypothetical[idx]) continue;
                if (!self.canLeaseWorkerWithHypotheticalUnlocked(idx, owner, hypothetical)) continue;
                hypothetical[idx] = true;
                count += 1;
                progress = true;
                break;
            }
            if (!progress) break;
        }

        return count;
    }

    fn canLeaseWorkerWithHypotheticalUnlocked(self: *Self, worker_index: usize, owner: WorkerLeaseOwner, hypothetical: []const bool) bool {
        const worker = self.worker_pool.items[worker_index];
        if (worker.lease_owner != null or hypothetical[worker_index]) return false;

        const policy = self.scheduling_policies[ownerIndex(owner)];
        if (!workerMatchesPolicy(policy, worker)) return false;

        if (policy.max_workers) |max_workers| {
            if (self.countLeasedWorkersWithHypotheticalUnlocked(owner, hypothetical) >= max_workers) return false;
        }

        const owners = [_]WorkerLeaseOwner{ .inference, .training, .rl };
        for (owners) |reserved_owner| {
            if (reserved_owner == owner) continue;

            const reserved_policy = self.scheduling_policies[ownerIndex(reserved_owner)];
            if (reserved_policy.reserved_workers == 0) continue;
            if (self.countCapacityForOwnerWithHypotheticalUnlocked(reserved_owner, owner, worker_index, hypothetical) < reserved_policy.reserved_workers) {
                return false;
            }
        }

        return true;
    }

    fn countCapacityForOwnerWithHypotheticalUnlocked(
        self: *Self,
        owner: WorkerLeaseOwner,
        hypothetical_owner: WorkerLeaseOwner,
        candidate_index: usize,
        hypothetical: []const bool,
    ) usize {
        const policy = self.scheduling_policies[ownerIndex(owner)];
        var count: usize = 0;
        for (self.worker_pool.items, 0..) |worker, idx| {
            if (!workerMatchesPolicy(policy, worker)) continue;

            const effective_owner: ?WorkerLeaseOwner = if (worker.lease_owner) |lease_owner|
                lease_owner
            else if (idx == candidate_index or hypothetical[idx])
                hypothetical_owner
            else
                null;

            if (effective_owner == null) {
                count += 1;
                continue;
            }
            if (effective_owner.? == owner) {
                count += 1;
            }
        }
        return count;
    }

    fn reservationsSatisfiedAfterLeaseUnlocked(self: *Self, candidate_node_id: NodeId, new_owner: WorkerLeaseOwner) bool {
        const owners = [_]WorkerLeaseOwner{ .inference, .training, .rl };
        for (owners) |owner| {
            if (owner == new_owner) continue;

            const policy = self.scheduling_policies[ownerIndex(owner)];
            if (policy.reserved_workers == 0) continue;
            if (self.countCapacityForOwnerAfterLeaseUnlocked(owner, candidate_node_id, new_owner) < policy.reserved_workers) {
                return false;
            }
        }
        return true;
    }

    fn countCapacityForOwnerAfterLeaseUnlocked(self: *Self, owner: WorkerLeaseOwner, candidate_node_id: NodeId, new_owner: WorkerLeaseOwner) usize {
        const policy = self.scheduling_policies[ownerIndex(owner)];
        var count: usize = 0;
        for (self.worker_pool.items) |worker| {
            if (!workerMatchesPolicy(policy, worker)) continue;

            const effective_owner: ?WorkerLeaseOwner = if (worker.node_id == candidate_node_id)
                new_owner
            else
                worker.lease_owner;

            if (effective_owner == null) {
                count += 1;
                continue;
            }
            if (effective_owner.? == owner) {
                count += 1;
            }
        }
        return count;
    }

    fn countLeasedWorkersUnlocked(self: *Self, owner: WorkerLeaseOwner) usize {
        var count: usize = 0;
        for (self.worker_pool.items) |worker| {
            if (worker.lease_owner) |lease_owner| {
                if (lease_owner == owner) count += 1;
            }
        }
        return count;
    }

    fn countLeasedWorkersWithHypotheticalUnlocked(self: *Self, owner: WorkerLeaseOwner, hypothetical: []const bool) usize {
        var count = self.countLeasedWorkersUnlocked(owner);
        for (hypothetical) |leased| {
            if (leased) count += 1;
        }
        return count;
    }

    fn notifyWorkerChangeHooks(self: *Self) void {
        for (self.worker_change_hooks.items) |hook| {
            hook.on_change(hook.ctx);
        }
    }

    fn countQueuedMatches(self: *Self, filter: MessageFilter, expected_max: usize) usize {
        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        std.debug.assert(self.result_queue.items.len <= protocol_limits.message_queue_depth_max);
        return message_queue.countUniqueSendersBounded(self.result_queue.items, filter, expected_max);
    }

    fn countQueuedMessages(self: *Self, filter: MessageFilter) usize {
        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        std.debug.assert(self.result_queue.items.len <= protocol_limits.message_queue_depth_max);
        return message_queue.countMatching(self.result_queue.items, filter, null);
    }

    /// Collect responses from a specific number of workers
    pub fn collectFromWorkers(self: *Self, expected_msg_type: []const u8, expected_count: usize) !ArrayList(MessageEnvelope) {
        return self.collectMatchingFromWorkers(.{
            .msg_type = expected_msg_type,
        }, expected_count);
    }

    pub fn collectMatchingFromWorkers(self: *Self, filter: MessageFilter, expected_count: usize) !ArrayList(MessageEnvelope) {
        const budget = protocol_limits.workerCollectionBudget(expected_count);
        try self.waitForMatchingWorkerMessages(filter, budget, true);
        return try self.drainUniqueMatchingWorkerMessages(filter, expected_count);
    }

    /// Nonblocking drain of queued messages matching a filter. Unlike
    /// collectMatchingFromWorkers, this intentionally allows multiple messages
    /// from the same worker and preserves unrelated queued messages.
    pub fn drainMatchingMessages(self: *Self, filter: MessageFilter, max_count: usize) !ArrayList(MessageEnvelope) {
        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        return message_queue.drainMatching(self.allocator, &self.result_queue, filter, .{ .max_count = max_count });
    }

    /// Nonblocking drain scoped to one worker. This is useful for decoupled
    /// learner metadata streams where several FIFO messages from the same
    /// worker may be consumed together.
    pub fn drainMatchingMessagesFromWorker(self: *Self, filter: MessageFilter, worker_id: NodeId, max_count: usize) !ArrayList(MessageEnvelope) {
        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        return message_queue.drainMatching(self.allocator, &self.result_queue, filter, .{
            .max_count = max_count,
            .sender_node = worker_id,
        });
    }

    /// Bounded-wait collection helper for asynchronous protocols. It returns
    /// whatever is available by timeout instead of enforcing the four-hour
    /// blocking behavior of the legacy collection helpers.
    pub fn collectMatchingMessagesBounded(self: *Self, filter: MessageFilter, expected_count: usize, timeout_ms: u64) !ArrayList(MessageEnvelope) {
        if (timeout_ms == 0) return self.drainMatchingMessages(filter, expected_count);

        const budget = protocol_limits.boundedCollectionBudget(expected_count, timeout_ms);
        try self.waitForMatchingWorkerMessages(filter, budget, false);
        return self.drainMatchingMessages(filter, expected_count);
    }

    pub fn collectMatchingMessages(self: *Self, filter: MessageFilter, expected_count: usize) !ArrayList(MessageEnvelope) {
        const budget = protocol_limits.workerCollectionBudget(expected_count);
        try self.waitForMatchingWorkerMessages(filter, budget, false);
        return self.drainMatchingMessages(filter, expected_count);
    }

    fn waitForMatchingWorkerMessages(
        self: *Self,
        filter: MessageFilter,
        budget: protocol_limits.WaitBudget,
        comptime unique_senders: bool,
    ) !void {
        try budget.validate();
        if (budget.expected_count == 0) return;

        std.log.info("Collecting {s} for type '{s}' (expecting {})...", .{
            budget.reason,
            filter.msg_type,
            budget.expected_count,
        });

        const start_ms = std.time.milliTimestamp();
        var next_log_ms: u64 = 5_000;
        while (!budget.expired(start_ms)) {
            const matching_results = if (unique_senders)
                self.countQueuedMatches(filter, budget.expected_count)
            else
                self.countQueuedMessages(filter);
            if (matching_results >= budget.expected_count) return;

            const elapsed_ms = budget.elapsedMs(start_ms);
            if (elapsed_ms >= next_log_ms) {
                std.log.info("Waiting for {s} type '{s}': {}/{} after {}ms", .{
                    budget.reason,
                    filter.msg_type,
                    matching_results,
                    budget.expected_count,
                    elapsed_ms,
                });
                next_log_ms += 5_000;
            }

            std.time.sleep(budget.sleepMs(start_ms) * std.time.ns_per_ms);
        }

        const matching_results = if (unique_senders)
            self.countQueuedMatches(filter, budget.expected_count)
        else
            self.countQueuedMessages(filter);
        if (matching_results >= budget.expected_count) return;
        if (budget.allow_partial and matching_results > 0) {
            std.log.warn("Timeout collecting {s} for type '{s}'. Proceeding with {}/{}", .{
                budget.reason,
                filter.msg_type,
                matching_results,
                budget.expected_count,
            });
            return;
        }
        return error.CollectionTimeout;
    }

    fn drainUniqueMatchingWorkerMessages(self: *Self, filter: MessageFilter, expected_count: usize) !ArrayList(MessageEnvelope) {
        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        var responses = ArrayList(MessageEnvelope).init(self.allocator);
        if (expected_count == 0) return responses;

        var idx: usize = 0;
        var seen_senders = std.AutoHashMap(NodeId, void).init(self.allocator);
        defer seen_senders.deinit();

        while (idx < self.result_queue.items.len and responses.items.len < expected_count) {
            const msg = self.result_queue.items[idx];
            if (!filter.matches(msg)) {
                idx += 1;
                continue;
            }
            if (seen_senders.contains(msg.sender_node)) {
                var duplicate = self.result_queue.orderedRemove(idx);
                duplicate.deinitClone(self.allocator);
                std.log.warn("Dropping duplicate queued message from worker {} for type '{s}' request {} round {} task {}", .{
                    msg.sender_node,
                    msg.msg_type,
                    msg.request_id,
                    msg.round_id,
                    msg.task_id,
                });
                continue;
            }

            if (filter.request_id) |request_id| {
                if (request_id != 0) protocol_invariants.assertRequestContext(request_id, msg.request_id);
            }
            try seen_senders.put(msg.sender_node, {});
            try responses.append(msg);
            _ = self.result_queue.orderedRemove(idx);
        }

        std.log.info("Collected {} results from workers", .{responses.items.len});
        return responses;
    }

    /// Compiles the MLIR source for all connected workers and ensures they are initialized.
    /// Handles caching, VMFB compilation, Base64 encoding, and JSON payload construction.
    pub fn ensureWorkersCompiled(
        self: *Self,
        mlir_source: []const u8,
        parameter_shapes: [][]i64,
        data_input_shapes: [][]i64,
    ) !void {
        const worker_ids = try self.snapshotWorkerIds(self.allocator);
        defer self.allocator.free(worker_ids);
        return self.ensureWorkersCompiledForIds(worker_ids, mlir_source, parameter_shapes, data_input_shapes);
    }

    pub fn ensureWorkersCompiledForIds(
        self: *Self,
        worker_ids: []const NodeId,
        mlir_source: []const u8,
        parameter_shapes: [][]i64,
        data_input_shapes: [][]i64,
    ) !void {
        // 1. Identify unique configs that need compilation
        var config_set = std.HashMap(WorkerConfig, void, WorkerConfigContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer config_set.deinit();

        var workers_snapshot = try self.snapshotWorkersByIds(worker_ids);
        defer workers_snapshot.deinit();
        const program = programDescriptorForMlir(mlir_source);
        const desired_program_key = program.program_key;

        for (workers_snapshot.items) |worker| {
            const config = WorkerConfig{ .backend = worker.backend, .target_arch = worker.target_arch };
            const cache_key = compiledArtifactKey(.training_vmfb, program, config);
            if (!self.compiled_artifacts.contains(cache_key)) {
                try config_set.put(config, {});
            }
        }

        // 2. Compile missing artifacts
        if (config_set.count() > 0) {
            var temp_ctx = try mlir_ctx.MLIRContext.init(self.allocator);
            defer temp_ctx.deinit();

            var it = config_set.keyIterator();
            while (it.next()) |config_ptr| {
                const config = config_ptr.*;
                const label = if (config.target_arch) |t| t else config.backend.toString();
                std.log.info("Compiling graph for target: {s}...", .{label});

                const vmfb = try temp_ctx.compileToVMFB(
                    self.allocator,
                    mlir_source,
                    config.backend.toIreeCompilationTarget(),
                    config.target_arch,
                );

                // Cache the artifact; the worker fabric owns this memory now.
                try self.compiled_artifacts.put(compiledArtifactKey(.training_vmfb, program, config), vmfb);
            }
        }

        // 3. Initialize uninitialized workers
        for (workers_snapshot.items) |worker| {
            const needs_initialize = worker.status == .Connected or worker.loaded_program_key == null or worker.loaded_program_key.? != desired_program_key;
            if (needs_initialize) {
                const config = WorkerConfig{ .backend = worker.backend, .target_arch = worker.target_arch };
                const vmfb_bytes = self.compiled_artifacts.get(compiledArtifactKey(.training_vmfb, program, config)).?; // Must exist now
                const artifact = programArtifact(.training_vmfb, program, config, vmfb_bytes);

                std.log.info("Initializing worker {} (Backend: {s})...", .{ worker.node_id, worker.backend.toString() });

                try self.sendInitializeMessageWithProgramArtifact(
                    worker.node_id,
                    artifact,
                    parameter_shapes,
                    data_input_shapes,
                    null,
                    desired_program_key,
                );
                self.setWorkerGraphInitializing(worker.node_id, desired_program_key);
            }
        }
    }

    /// Directly initializes workers with a pre-compiled VMFB path and known shapes
    /// Workers will load the VMFB from their local filesystem
    pub fn initializeWorkersWithVMFB(
        self: *Self,
        vmfb_path: []const u8,
        parameter_shapes: [][]i64,
        data_input_shapes: [][]i64,
        data_input_dtypes: []tensor.DType,
    ) !void {
        const worker_ids = try self.snapshotWorkerIds(self.allocator);
        defer self.allocator.free(worker_ids);
        return self.initializeWorkersWithVMFBForIds(worker_ids, vmfb_path, parameter_shapes, data_input_shapes, data_input_dtypes);
    }

    pub fn initializeWorkersWithVMFBForIds(
        self: *Self,
        worker_ids: []const NodeId,
        vmfb_path: []const u8,
        parameter_shapes: [][]i64,
        data_input_shapes: [][]i64,
        data_input_dtypes: []tensor.DType,
    ) !void {
        std.log.info("Distributing VMFB bytes to workers from: {s}", .{vmfb_path});
        const vmfb_bytes = try std.fs.cwd().readFileAlloc(self.allocator, vmfb_path, 2 * 1024 * 1024 * 1024);
        defer self.allocator.free(vmfb_bytes);

        var workers_snapshot = try self.snapshotWorkersByIds(worker_ids);
        defer workers_snapshot.deinit();
        const program = programDescriptorForVmfbPath(vmfb_path);
        const desired_program_key = program.program_key;

        for (workers_snapshot.items) |worker| {
            const needs_initialize = worker.status == .Connected or worker.loaded_program_key == null or worker.loaded_program_key.? != desired_program_key;
            if (!needs_initialize) continue;

            std.log.info("Initializing worker {}...", .{worker.node_id});
            const config = WorkerConfig{ .backend = worker.backend, .target_arch = worker.target_arch };
            const artifact = programArtifact(.external_vmfb, program, config, vmfb_bytes);
            try self.sendInitializeMessageWithProgramArtifact(worker.node_id, artifact, parameter_shapes, data_input_shapes, data_input_dtypes, desired_program_key);
            self.setWorkerGraphInitializing(worker.node_id, desired_program_key);
        }

        std.log.info("✓ All workers initialized with transferred VMFB bytes", .{});
    }

    /// Initialize workers with VMFB path AND weights path for local loading
    /// This is the preferred method for RL training where weights can be large (>2GB)
    pub fn initializeWorkersWithVMFBAndWeights(
        self: *Self,
        vmfb_path: []const u8,
        weights_path: []const u8,
        parameter_shapes: [][]i64,
        data_input_shapes: [][]i64,
        data_input_dtypes: []tensor.DType,
    ) !void {
        const worker_ids = try self.snapshotWorkerIds(self.allocator);
        defer self.allocator.free(worker_ids);
        return self.initializeWorkersWithVMFBAndWeightsForIds(worker_ids, vmfb_path, weights_path, "flat", parameter_shapes, null, data_input_shapes, data_input_dtypes);
    }

    pub fn initializeWorkersWithVMFBAndWeightsForIds(
        self: *Self,
        worker_ids: []const NodeId,
        vmfb_path: []const u8,
        weights_path: []const u8,
        weights_format: []const u8,
        parameter_shapes: [][]i64,
        parameter_dtypes: ?[]tensor.DType,
        data_input_shapes: [][]i64,
        data_input_dtypes: []tensor.DType,
    ) !void {
        std.log.info("Distributing VMFB path to workers: {s}", .{vmfb_path});
        std.log.info("Workers will load weights locally from: {s}", .{weights_path});
        var workers_snapshot = try self.snapshotWorkersByIds(worker_ids);
        defer workers_snapshot.deinit();
        const desired_program_key = programKeyForVmfbAndWeights(vmfb_path, weights_path);

        for (workers_snapshot.items) |worker| {
            const needs_initialize = worker.status == .Connected or worker.loaded_program_key == null or worker.loaded_program_key.? != desired_program_key;
            if (!needs_initialize) continue;

            std.log.info("Initializing worker {}...", .{worker.node_id});
            try self.sendInitializeMessageWithPathAndWeights(worker.node_id, vmfb_path, weights_path, weights_format, parameter_shapes, parameter_dtypes, data_input_shapes, data_input_dtypes, desired_program_key);
            self.setWorkerGraphInitializing(worker.node_id, desired_program_key);
        }

        std.log.info("✓ All workers initialized with VMFB + weights paths", .{});
    }

    /// Helper to send initialization message with VMFB and weights file paths
    fn sendInitializeMessageWithPathAndWeights(self: *Self, node_id: NodeId, vmfb_path: []const u8, weights_path: []const u8, weights_format: []const u8, p_shapes: [][]i64, p_dtypes: ?[]tensor.DType, d_shapes: [][]i64, d_dtypes: []tensor.DType, program_key: u64) !void {
        // Build JSON shapes
        var param_shape_array = std.json.Array.init(self.allocator);
        defer param_shape_array.deinit();
        for (p_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try param_shape_array.append(std.json.Value{ .array = dim_array });
        }

        var param_dtype_array = std.json.Array.init(self.allocator);
        defer param_dtype_array.deinit();
        if (p_dtypes) |dtypes| {
            for (dtypes) |dtype| {
                try param_dtype_array.append(std.json.Value{ .string = dtypeName(dtype) });
            }
        }

        var data_shape_array = std.json.Array.init(self.allocator);
        defer data_shape_array.deinit();
        for (d_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try data_shape_array.append(std.json.Value{ .array = dim_array });
        }

        // Serialize DTypes to string array
        var dtype_array = std.json.Array.init(self.allocator);
        defer dtype_array.deinit();
        for (d_dtypes) |dtype| {
            try dtype_array.append(std.json.Value{ .string = dtypeName(dtype) });
        }

        // Build Payload with paths instead of bytes
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        const program_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{program_key});
        defer self.allocator.free(program_key_text);
        try payload.put("vmfb_path", std.json.Value{ .string = vmfb_path });
        try payload.put("program_key", std.json.Value{ .string = program_key_text });
        try payload.put("weights_path", std.json.Value{ .string = weights_path });
        try payload.put("weights_format", std.json.Value{ .string = weights_format });
        try payload.put("parameter_shapes", std.json.Value{ .array = param_shape_array });
        if (p_dtypes != null) {
            try payload.put("parameter_dtypes", std.json.Value{ .array = param_dtype_array });
        }
        try payload.put("data_input_shapes", std.json.Value{ .array = data_shape_array });
        try payload.put("data_input_dtypes", std.json.Value{ .array = dtype_array });

        std.log.info("Sending initialize-graph paths to worker {} (vmfb={s}, weights={s})", .{ node_id, vmfb_path, weights_path });
        try self.sendToWorker(node_id, MessageType.INITIALIZE_GRAPH, .{ .object = payload });
        std.log.info("Sent initialize-graph paths to worker {}", .{node_id});
    }

    /// Helper to send initialization message with VMFB file path (for local filesystem loading)
    fn sendInitializeMessageWithPath(
        self: *Self,
        node_id: NodeId,
        vmfb_path: []const u8,
        p_shapes: [][]i64,
        d_shapes: [][]i64,
        d_dtypes: ?[]tensor.DType,
        program_key: u64,
    ) !void {
        // Build JSON shapes
        var param_shape_array = std.json.Array.init(self.allocator);
        defer param_shape_array.deinit();
        for (p_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try param_shape_array.append(std.json.Value{ .array = dim_array });
        }

        var data_shape_array = std.json.Array.init(self.allocator);
        defer data_shape_array.deinit();
        for (d_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try data_shape_array.append(std.json.Value{ .array = dim_array });
        }

        // Build Payload with path instead of bytes
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        const program_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{program_key});
        defer self.allocator.free(program_key_text);
        try payload.put("vmfb_path", std.json.Value{ .string = vmfb_path });
        try payload.put("program_key", std.json.Value{ .string = program_key_text });
        try payload.put("parameter_shapes", std.json.Value{ .array = param_shape_array });
        try payload.put("data_input_shapes", std.json.Value{ .array = data_shape_array });
        if (d_dtypes) |dtype_values| {
            var dtype_array = std.json.Array.init(self.allocator);
            defer dtype_array.deinit();
            for (dtype_values) |dtype| {
                const type_str = switch (dtype) {
                    .f32 => "f32",
                    .f64 => "f64",
                    .f16 => "f16",
                    .bf16 => "bf16",
                    .i32 => "i32",
                    .i64 => "i64",
                    .bool => "bool",
                };
                try dtype_array.append(std.json.Value{ .string = type_str });
            }
            try payload.put("data_input_dtypes", std.json.Value{ .array = dtype_array });
        }

        std.log.info("Sending initialize-graph VMFB path to worker {} ({s})", .{ node_id, vmfb_path });
        try self.sendToWorker(node_id, MessageType.INITIALIZE_GRAPH, .{ .object = payload });
        std.log.info("Sent initialize-graph VMFB path to worker {}", .{node_id});
    }

    fn sendInitializeMessageWithProgramArtifact(
        self: *Self,
        node_id: NodeId,
        artifact: ProgramArtifact,
        p_shapes: [][]i64,
        d_shapes: [][]i64,
        d_dtypes: ?[]tensor.DType,
        program_key: u64,
    ) !void {
        if (self.workerHasMaterializedArtifact(node_id, artifact.artifact_key)) {
            try self.sendInitializeMessageWithVmfbArtifactRef(node_id, artifact, p_shapes, d_shapes, d_dtypes, program_key);
            return;
        }

        try self.sendInitializeMessageWithTransferredVmfb(node_id, artifact, p_shapes, d_shapes, d_dtypes, program_key);
    }

    fn sendInitializeMessageWithTransferredVmfb(
        self: *Self,
        node_id: NodeId,
        artifact: ProgramArtifact,
        p_shapes: [][]i64,
        d_shapes: [][]i64,
        d_dtypes: ?[]tensor.DType,
        program_key: u64,
    ) !void {
        const request_id = self.allocateRequestId();
        const worker_ids = [_]NodeId{node_id};
        try self.broadcastLargeDataToWorkerIdsWithContext(worker_ids[0..], artifact.bytes, .{
            .request_id = request_id,
        });

        // Build JSON shapes
        var param_shape_array = std.json.Array.init(self.allocator);
        defer param_shape_array.deinit();
        for (p_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try param_shape_array.append(std.json.Value{ .array = dim_array });
        }

        var data_shape_array = std.json.Array.init(self.allocator);
        defer data_shape_array.deinit();
        for (d_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try data_shape_array.append(std.json.Value{ .array = dim_array });
        }

        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        const program_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{program_key});
        defer self.allocator.free(program_key_text);
        const artifact_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{artifact.artifact_key});
        defer self.allocator.free(artifact_key_text);
        const artifact_byte_hash_text = try std.fmt.allocPrint(self.allocator, "{d}", .{artifact.byte_hash});
        defer self.allocator.free(artifact_byte_hash_text);
        try payload.put("vmfb_blob", std.json.Value{ .bool = true });
        try payload.put("vmfb_bytes", std.json.Value{ .integer = @intCast(artifact.bytes.len) });
        try payload.put("artifact_kind", std.json.Value{ .string = artifactKindLabel(artifact.kind) });
        try payload.put("artifact_key", std.json.Value{ .string = artifact_key_text });
        try payload.put("artifact_byte_hash", std.json.Value{ .string = artifact_byte_hash_text });
        try payload.put("program_key", std.json.Value{ .string = program_key_text });
        try payload.put("parameter_shapes", std.json.Value{ .array = param_shape_array });
        try payload.put("data_input_shapes", std.json.Value{ .array = data_shape_array });
        if (d_dtypes) |dtype_values| {
            var dtype_array = std.json.Array.init(self.allocator);
            defer dtype_array.deinit();
            for (dtype_values) |dtype| {
                const type_str = switch (dtype) {
                    .f32 => "f32",
                    .f64 => "f64",
                    .f16 => "f16",
                    .bf16 => "bf16",
                    .i32 => "i32",
                    .i64 => "i64",
                    .bool => "bool",
                };
                try dtype_array.append(std.json.Value{ .string = type_str });
            }
            try payload.put("data_input_dtypes", std.json.Value{ .array = dtype_array });
        }

        std.log.info("Sending initialize-graph VMFB blob reference to worker {} (request={}, vmfb_bytes={})", .{
            node_id,
            request_id,
            artifact.bytes.len,
        });
        try self.sendToWorkerWithContext(node_id, MessageType.INITIALIZE_GRAPH, .{ .object = payload }, .{
            .request_id = request_id,
        });
        std.log.info("Sent initialize-graph VMFB blob reference to worker {}", .{node_id});
    }

    fn sendInitializeMessageWithVmfbArtifactRef(
        self: *Self,
        node_id: NodeId,
        artifact: ProgramArtifact,
        p_shapes: [][]i64,
        d_shapes: [][]i64,
        d_dtypes: ?[]tensor.DType,
        program_key: u64,
    ) !void {
        const request_id = self.allocateRequestId();

        // Build JSON shapes
        var param_shape_array = std.json.Array.init(self.allocator);
        defer param_shape_array.deinit();
        for (p_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try param_shape_array.append(std.json.Value{ .array = dim_array });
        }

        var data_shape_array = std.json.Array.init(self.allocator);
        defer data_shape_array.deinit();
        for (d_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try data_shape_array.append(std.json.Value{ .array = dim_array });
        }

        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        const program_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{program_key});
        defer self.allocator.free(program_key_text);
        const artifact_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{artifact.artifact_key});
        defer self.allocator.free(artifact_key_text);
        const artifact_byte_hash_text = try std.fmt.allocPrint(self.allocator, "{d}", .{artifact.byte_hash});
        defer self.allocator.free(artifact_byte_hash_text);
        try payload.put("vmfb_artifact_ref", std.json.Value{ .bool = true });
        try payload.put("artifact_kind", std.json.Value{ .string = artifactKindLabel(artifact.kind) });
        try payload.put("artifact_key", std.json.Value{ .string = artifact_key_text });
        try payload.put("artifact_byte_hash", std.json.Value{ .string = artifact_byte_hash_text });
        try payload.put("program_key", std.json.Value{ .string = program_key_text });
        try payload.put("parameter_shapes", std.json.Value{ .array = param_shape_array });
        try payload.put("data_input_shapes", std.json.Value{ .array = data_shape_array });
        if (d_dtypes) |dtype_values| {
            var dtype_array = std.json.Array.init(self.allocator);
            defer dtype_array.deinit();
            for (dtype_values) |dtype| {
                const type_str = switch (dtype) {
                    .f32 => "f32",
                    .f64 => "f64",
                    .f16 => "f16",
                    .bf16 => "bf16",
                    .i32 => "i32",
                    .i64 => "i64",
                    .bool => "bool",
                };
                try dtype_array.append(std.json.Value{ .string = type_str });
            }
            try payload.put("data_input_dtypes", std.json.Value{ .array = dtype_array });
        }

        std.log.info("Sending initialize-graph VMFB artifact reference to worker {} (request={}, artifact={d})", .{
            node_id,
            request_id,
            artifact.artifact_key,
        });
        try self.sendToWorkerWithContext(node_id, MessageType.INITIALIZE_GRAPH, .{ .object = payload }, .{
            .request_id = request_id,
        });
        std.log.info("Sent initialize-graph VMFB artifact reference to worker {}", .{node_id});
    }

    fn ensureLocalCompiledVmfbPath(
        self: *Self,
        config: WorkerConfig,
        program_key: u64,
        vmfb_bytes: []const u8,
    ) ![]u8 {
        const label = config.target_arch orelse config.backend.toString();
        const compiled_dir = try runtime_config.statePath(self.allocator, &.{"compiled"});
        defer self.allocator.free(compiled_dir);
        try file_util.ensureDirAtPath(compiled_dir);

        const filename = try std.fmt.allocPrint(self.allocator, "pcp_compiled_{s}_{d}.vmfb", .{ label, program_key });
        defer self.allocator.free(filename);
        const vmfb_path = try std.fs.path.join(self.allocator, &.{ compiled_dir, filename });

        if (file_util.fileExistsAtPath(vmfb_path)) {
            return vmfb_path;
        }

        try file_util.writeFileAtPath(vmfb_path, vmfb_bytes);
        std.log.info("Saved compiled VMFB for {s} to {s}", .{ label, vmfb_path });
        return vmfb_path;
    }

    /// Helper to construct and send the initialization JSON payload
    fn sendInitializeMessage(self: *Self, node_id: NodeId, vmfb: []const u8, p_shapes: [][]i64, d_shapes: [][]i64, program_key: u64) !void {
        // Base64 encode VMFB
        const b64_len = std.base64.standard.Encoder.calcSize(vmfb.len);
        const b64_encoded_vmfb = try self.allocator.alloc(u8, b64_len);
        defer self.allocator.free(b64_encoded_vmfb);
        const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_vmfb, vmfb).len;

        // Build JSON shapes
        var param_shape_array = std.json.Array.init(self.allocator);
        defer param_shape_array.deinit();
        for (p_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try param_shape_array.append(std.json.Value{ .array = dim_array });
        }

        var data_shape_array = std.json.Array.init(self.allocator);
        defer data_shape_array.deinit();
        for (d_shapes) |shape| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape) |dim| try dim_array.append(std.json.Value{ .integer = dim });
            try data_shape_array.append(std.json.Value{ .array = dim_array });
        }

        // Build Payload
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        const program_key_text = try std.fmt.allocPrint(self.allocator, "{d}", .{program_key});
        defer self.allocator.free(program_key_text);
        try payload.put("vmfb", std.json.Value{ .string = b64_encoded_vmfb[0..encoded_len] });
        try payload.put("program_key", std.json.Value{ .string = program_key_text });
        try payload.put("parameter_shapes", std.json.Value{ .array = param_shape_array });
        try payload.put("data_input_shapes", std.json.Value{ .array = data_shape_array });

        std.log.info("Sending initialize-graph blob to worker {} (vmfb_bytes={})", .{ node_id, vmfb.len });
        try self.sendToWorker(node_id, MessageType.INITIALIZE_GRAPH, .{ .object = payload });
        std.log.info("Sent initialize-graph blob to worker {}", .{node_id});
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

    /// Stop the coordinator
    pub fn stop(self: *Self) void {
        self.is_running = false;
        self.stop_requested.store(1, .release);

        // Send shutdown message to all workers
        const shutdown_data = std.json.Value{ .string = "shutdown" };
        self.broadcastToWorkers(MessageType.SHUTDOWN, shutdown_data) catch |err| {
            std.log.err("Failed to broadcast shutdown: {}", .{err});
        };

        if (self.listen_host) |host| {
            const address = net.Address.parseIp(host, self.listen_port) catch |err| {
                std.log.warn("Failed to parse worker fabric listen address {s}:{} during stop: {}", .{ host, self.listen_port, err });
                return;
            };
            const stream = net.tcpConnectToAddress(address) catch |err| {
                std.log.warn("Failed to wake worker fabric listener during stop: {}", .{err});
                return;
            };
            stream.close();
        }

        std.log.info("Worker fabric stopped", .{});
    }
};

pub const TrainingController = WorkerFabricController;

/// Test function for the worker-fabric controller.
pub fn testWorkerFabric(allocator: Allocator) !void {
    std.log.info("Testing worker-fabric controller...");

    var worker_fabric = WorkerFabricController.init(allocator);
    defer worker_fabric.deinit();

    // Test basic initialization
    try std.testing.expectEqual(@as(usize, 0), worker_fabric.getWorkerCount());
    try std.testing.expectEqual(@as(NodeId, 1), worker_fabric.next_node_id);

    std.log.info("✓ Worker-fabric controller test completed");
}

fn ownerIndex(owner: WorkerLeaseOwner) usize {
    return switch (owner) {
        .inference => 0,
        .training => 1,
        .rl => 2,
    };
}

fn workerClassForBackend(backend: Backend) WorkerClass {
    return switch (backend) {
        .cpu => .cpu,
        .cuda => .cuda,
        .rocm => .rocm,
        .metal => .metal,
        .vulkan => .vulkan,
    };
}

fn dtypeName(dtype: tensor.DType) []const u8 {
    return switch (dtype) {
        .f32 => "f32",
        .f64 => "f64",
        .f16 => "f16",
        .bf16 => "bf16",
        .i32 => "i32",
        .i64 => "i64",
        .bool => "bool",
    };
}

fn workerMatchesPolicy(policy: WorkerSchedulingPolicy, worker: WorkerConnection) bool {
    return policy.allowsWorker(workerClassForBackend(worker.backend), worker.target_arch);
}
