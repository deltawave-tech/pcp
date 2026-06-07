pub const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub const training_algorithm = @import("../../algorithms/training_algorithm.zig");
pub const diloco = @import("../../algorithms/diloco.zig");
pub const decoupled_diloco = @import("../../algorithms/decoupled_diloco.zig");
pub const fragments = @import("../../algorithms/fragments.zig");
pub const grpo = @import("../../algorithms/grpo.zig");
pub const adam_mlir = @import("../../optimizers/adam_mlir.zig");
pub const backend_selection = @import("../../backends/selection.zig");
pub const GraphBuilder = @import("../../compiler/graph_builder.zig").GraphBuilder;
pub const ModelSanitizer = @import("../../compiler/sanitizer.zig").ModelSanitizer;
pub const ops = @import("../../core/ops.zig");
pub const tensor = @import("../../core/tensor.zig");
pub const control_api = @import("control_plane/api.zig");
pub const control_state = @import("control_plane/state.zig");
pub const data_assignment = @import("../../data/assignment.zig");
pub const experiment_config = @import("../../workloads/training/config.zig");
pub const inference_config = @import("../../workloads/inference/config.zig");
pub const training_window = @import("../../workloads/training/window.zig");
pub const training_workload = @import("../../workloads/training/workload.zig");
pub const model_introspection = @import("../../mlir/model_introspection.zig");
pub const message = @import("../../network/message.zig");
pub const nesterov = @import("../../optimizers/nesterov.zig");
pub const file_util = @import("../../protocol/file_util.zig");
pub const protocol_limits = @import("../../protocol/limits.zig");
pub const message_registry = @import("../../protocol/message_registry.zig");
pub const training_state = @import("../../protocol/training_state.zig");
pub const wandb = @import("../../observability/wandb.zig");
pub const config = @import("config.zig");
pub const embedded_jobs = @import("embedded_jobs.zig");
pub const gateway = @import("gateway.zig");
pub const identity = @import("identity.zig");
pub const scheduling = @import("scheduling.zig");
pub const inference_controller = @import("controllers/inference_controller.zig");
pub const rl_controller = @import("controllers/rl_controller.zig");
pub const training_controller = @import("controllers/training_controller.zig");
pub const service_registry = @import("service_registry.zig");

pub const DiLoCo = diloco.DiLoCo;
pub const DiLoCoConfig = diloco.DiLoCoConfig;
pub const MLIRBuilder = ops.MLIRBuilder;
pub const WorkerFabricController = training_controller.WorkerFabricController;
pub const WorkerLeaseOwner = training_controller.WorkerLeaseOwner;
pub const WorkerSchedulingPolicy = training_controller.WorkerSchedulingPolicy;
pub const NodeId = message.NodeId;
pub const Nesterov = nesterov.Nesterov;
pub const EmbeddedJobQueue = embedded_jobs.EmbeddedJobQueue;
pub const EmbeddedQueuedJob = embedded_jobs.EmbeddedQueuedJob;
pub const EmbeddedReservation = embedded_jobs.EmbeddedReservation;
pub const EmbeddedReservationStore = embedded_jobs.EmbeddedReservationStore;
pub const readFileAllocAtPath = file_util.readFileAllocAtPath;
pub const writeFileAtPath = file_util.writeFileAtPath;
pub const ensureDirAtPath = file_util.ensureDirAtPath;
pub const fileExistsAtPath = file_util.fileExistsAtPath;

pub const ScopedServicePublisher = struct {
    allocator: Allocator,
    registry: *service_registry.ServiceRegistry,
    worker_fabric: *WorkerFabricController,
    state: *control_state.ControllerState,
    service_id: []u8,
    executor_id: []u8,
    service_type: []const u8,
    base_url: []u8,
    capabilities: []const []const u8,
    owner: WorkerLeaseOwner,
    runtime_snapshot_provider: ?control_api.SnapshotProvider,
    bind_worker_changes: bool,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        registry: *service_registry.ServiceRegistry,
        worker_fabric: *WorkerFabricController,
        state: *control_state.ControllerState,
        service_id: []const u8,
        executor_id: []const u8,
        service_type: []const u8,
        base_url: []const u8,
        capabilities: []const []const u8,
        owner: WorkerLeaseOwner,
        runtime_snapshot_provider: ?control_api.SnapshotProvider,
        bind_worker_changes: bool,
    ) !*Self {
        const publisher = try allocator.create(Self);
        publisher.* = .{
            .allocator = allocator,
            .registry = registry,
            .worker_fabric = worker_fabric,
            .state = state,
            .service_id = try allocator.dupe(u8, service_id),
            .executor_id = try allocator.dupe(u8, executor_id),
            .service_type = service_type,
            .base_url = try allocator.dupe(u8, base_url),
            .capabilities = capabilities,
            .owner = owner,
            .runtime_snapshot_provider = runtime_snapshot_provider,
            .bind_worker_changes = bind_worker_changes,
        };
        return publisher;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.service_id);
        self.allocator.free(self.executor_id);
        self.allocator.free(self.base_url);
        self.allocator.destroy(self);
    }

    pub fn bind(self: *Self) !void {
        self.state.setChangeHook(.{
            .ctx = self,
            .on_change = stateChangeHook,
        });
        if (self.bind_worker_changes) {
            try self.worker_fabric.addWorkerChangeHook(.{
                .ctx = self,
                .on_change = workerChangeHook,
            });
        }
        try self.publish();
    }

    pub fn snapshotProvider(self: *Self) control_api.SnapshotProvider {
        return .{
            .ctx = self,
            .capture = captureRuntimeSnapshotHook,
        };
    }

    pub fn publishHook(self: *Self) control_state.ChangeHook {
        return .{
            .ctx = self,
            .on_change = publishHookCallback,
        };
    }

    pub fn publish(self: *Self) !void {
        const runtime = self.captureRuntimeSnapshot();
        const status = currentStatus(self.state);
        var service = try self.registry.register(.{
            .service_id = self.service_id,
            .executor_id = self.executor_id,
            .service_type = self.service_type,
            .base_url = self.base_url,
            .auth_mode = "bearer",
            .health_status = runtime.health_status,
            .job_status = control_state.ControllerState.statusString(status),
            .workers_connected = runtime.workers_connected,
            .workers_ready = runtime.workers_ready,
            .workers_available = runtime.workers_available,
            .worker_class = runtime.worker_class,
            .target_arch = runtime.target_arch,
            .reserved_workers = runtime.reserved_workers,
            .max_workers = runtime.max_workers,
            .capabilities = self.capabilities,
        });
        service.deinit(self.allocator);
    }

    pub fn publishNoFail(self: *Self) void {
        self.publish() catch |err| {
            std.log.warn("Failed to publish embedded gateway service snapshot for {s}: {}", .{ self.service_id, err });
        };
    }

    pub fn captureRuntimeSnapshot(self: *Self) control_state.RuntimeSnapshot {
        if (self.runtime_snapshot_provider) |provider| {
            return provider.capture(provider.ctx, self.state, self.worker_fabric, self.owner) catch |err| {
                std.log.warn("Failed to capture runtime snapshot for {s}: {}", .{ self.service_id, err });
                return self.defaultRuntimeSnapshot();
            };
        }

        return self.defaultRuntimeSnapshot();
    }

    pub fn defaultRuntimeSnapshot(self: *Self) control_state.RuntimeSnapshot {
        const status = currentStatus(self.state);
        const required_workers = currentRequiredWorkers(self.state) orelse 1;
        const workers_connected = self.worker_fabric.getUsableWorkerCountForLeaseOwner(self.owner);
        const workers_ready = self.worker_fabric.getReadyWorkerCountForLeaseOwner(self.owner);
        const workers_available = self.worker_fabric.getAvailableWorkerCountForLeaseOwner(self.owner);
        const policy = self.worker_fabric.getSchedulingPolicy(self.owner);
        const idle_ready = status == .idle;
        const terminal_ready = status == .completed or status == .cancelled or status == .cancelling;
        const running_ready = status == .running and workers_ready >= required_workers;
        const ready = idle_ready or running_ready or terminal_ready;

        return .{
            .workers_connected = workers_connected,
            .workers_ready = workers_ready,
            .workers_available = workers_available,
            .health_status = healthStatus(status, ready),
            .ready = ready,
            .worker_class = if (policy.worker_class == .any) null else policy.worker_class.asString(),
            .target_arch = policy.target_arch,
            .reserved_workers = if (policy.reserved_workers > 0) policy.reserved_workers else null,
            .max_workers = policy.max_workers,
        };
    }

    pub fn captureRuntimeSnapshotHook(
        ctx: *anyopaque,
        _: *control_state.ControllerState,
        _: *WorkerFabricController,
        _: ?WorkerLeaseOwner,
    ) anyerror!control_state.RuntimeSnapshot {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.captureRuntimeSnapshot();
    }

    pub fn stateChangeHook(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.publishNoFail();
    }

    pub fn workerChangeHook(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.publishNoFail();
    }

    pub fn publishHookCallback(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.publishNoFail();
    }

    pub fn currentStatus(state: *control_state.ControllerState) control_state.JobStatus {
        state.mutex.lock();
        defer state.mutex.unlock();
        return state.status;
    }

    pub fn currentRequiredWorkers(state: *control_state.ControllerState) ?usize {
        state.mutex.lock();
        defer state.mutex.unlock();
        return state.required_workers;
    }

    pub fn healthStatus(status: control_state.JobStatus, ready: bool) []const u8 {
        if (status == .failed) return "error";
        if (status == .idle) return "ok";
        if (ready) return "ok";
        return "starting";
    }
};

pub const TrainingRunContext = struct {
    algorithm: *training_algorithm.TrainingAlgorithm,
    diloco: *DiLoCo,
    worker_fabric: *WorkerFabricController,
    state: *control_state.ControllerState,
    required_workers: usize,
    publisher: ?*ScopedServicePublisher,
};

pub const RLRunContext = struct {
    algorithm: *training_algorithm.TrainingAlgorithm,
    controller: *rl_controller.RLController,
    worker_fabric: *WorkerFabricController,
    state: *control_state.ControllerState,
    required_workers: usize,
    reserved_worker_ids: ?[]const NodeId = null,
    publisher: ?*ScopedServicePublisher,
};

pub const OwnedDecoupledLearnerUpdate = struct {
    update: decoupled_diloco.LearnerFragmentUpdate,
    tensors: []decoupled_diloco.TensorUpdate,
    values: [][]f32,

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        for (self.values) |value| allocator.free(value);
        allocator.free(self.values);
        allocator.free(self.tensors);
    }
};

pub const TimedDecoupledMessage = struct {
    envelope: message.MessageEnvelope,
    arrival_ms: u64,

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        self.envelope.deinitClone(allocator);
    }
};

pub const DecoupledRuntimeStats = struct {
    quorum_participants_total: usize = 0,
    grace_inclusions_total: usize = 0,
    skipped_learners_total: usize = 0,
    last_participant_count: usize = 0,
    last_grace_inclusion_count: usize = 0,
    last_skipped_learner_count: usize = 0,
    last_quorum_wait_ms: u64 = 0,
    last_grace_window_ms: u64 = 0,
    learner_to_syncer_fragment_bytes: usize = 0,
    syncer_to_learner_fragment_bytes: usize = 0,

    pub fn recordStep(self: *@This(), result: decoupled_diloco.FragmentSyncResult) void {
        self.quorum_participants_total += result.participant_count;
        self.grace_inclusions_total += result.grace_inclusion_count;
        self.skipped_learners_total += result.skipped_learner_count;
        self.last_participant_count = result.participant_count;
        self.last_grace_inclusion_count = result.grace_inclusion_count;
        self.last_skipped_learner_count = result.skipped_learner_count;
        self.last_quorum_wait_ms = result.quorum_wait_ms;
        self.last_grace_window_ms = result.grace_window_ms;
    }
};

pub fn makeLocalJobEnvelope(
    allocator: Allocator,
    service_id: []const u8,
    executor_id: []const u8,
    service_type: []const u8,
    job_id: []const u8,
    job_json: []const u8,
) !gateway.LocalJobEnvelope {
    return .{
        .allocator = allocator,
        .job_id = try allocator.dupe(u8, job_id),
        .service_id = try allocator.dupe(u8, service_id),
        .executor_id = try allocator.dupe(u8, executor_id),
        .service_type = try allocator.dupe(u8, service_type),
        .job_json = try allocator.dupe(u8, job_json),
    };
}

pub fn renderSyntheticJobJson(
    allocator: Allocator,
    job_type: control_state.JobType,
    status: control_state.JobStatus,
    config_path: ?[]const u8,
    run_id: ?[]const u8,
    model_id: ?[]const u8,
    resume_requested: bool,
    submitted_at: ?i64,
    started_at: i64,
    finished_at: ?i64,
    required_workers: usize,
    runtime: control_state.RuntimeSnapshot,
    cancel_requested: bool,
    last_error: ?[]const u8,
) ![]u8 {
    const health_status = switch (status) {
        .failed => "error",
        .queued, .idle, .completed, .cancelled, .cancelling => "ok",
        else => runtime.health_status,
    };
    const ready = switch (status) {
        .idle, .completed, .cancelled, .cancelling => true,
        .queued, .starting, .initializing, .waiting_for_workers, .failed => false,
        .running => runtime.ready,
    };

    return std.json.stringifyAlloc(allocator, .{
        .job_type = control_state.ControllerState.jobTypeString(job_type),
        .status = control_state.ControllerState.statusString(status),
        .health_status = health_status,
        .config_path = config_path,
        .run_id = run_id,
        .model_id = model_id,
        .@"resume" = resume_requested,
        .submitted_at = submitted_at,
        .started_at = started_at,
        .finished_at = finished_at,
        .workers_connected = runtime.workers_connected,
        .workers_ready = runtime.workers_ready,
        .workers_available = runtime.workers_available,
        .workers_dispatchable = runtime.workers_available,
        .workers_required = required_workers,
        .ready = ready,
        .readiness_phase = if (status == .queued) "queued" else runtime.readiness_phase,
        .readiness_detail = if (status == .queued) null else runtime.readiness_detail,
        .worker_class = runtime.worker_class,
        .target_arch = runtime.target_arch,
        .reserved_workers = runtime.reserved_workers,
        .max_workers = runtime.max_workers,
        .cancel_requested = cancel_requested,
        .last_error = last_error,
    }, .{});
}

pub fn makeLocalReservationResult(
    allocator: Allocator,
    service_id: []const u8,
    executor_id: []const u8,
    service_type: []const u8,
    reservation: EmbeddedReservation,
) !gateway.LocalReservationResult {
    return .{
        .allocator = allocator,
        .accepted = true,
        .reservation_id = try allocator.dupe(u8, reservation.reservation_id),
        .service_id = try allocator.dupe(u8, service_id),
        .executor_id = try allocator.dupe(u8, executor_id),
        .service_type = try allocator.dupe(u8, service_type),
        .workers_required = reservation.workers_required,
        .reserved_at = reservation.reserved_at,
    };
}

pub fn makeLocalReservationReleaseResult(
    allocator: Allocator,
    service_id: []const u8,
    executor_id: []const u8,
    service_type: []const u8,
    reservation_id: []const u8,
    status: []const u8,
) !gateway.LocalReservationReleaseResult {
    return .{
        .allocator = allocator,
        .accepted = true,
        .status = try allocator.dupe(u8, status),
        .reservation_id = try allocator.dupe(u8, reservation_id),
        .service_id = try allocator.dupe(u8, service_id),
        .executor_id = try allocator.dupe(u8, executor_id),
        .service_type = try allocator.dupe(u8, service_type),
    };
}

pub fn inferenceApiThread(controller: *inference_controller.InferenceController, host: []const u8, port: u16) !void {
    try controller.startApi(host, port);
}

pub fn controlApiThread(server: *control_api.ControlApiServer, host: []const u8, port: u16) !void {
    try server.start(host, port);
}

pub fn noOpCancel(_: *anyopaque) void {}

pub fn parsePayloadUsize(payload: std.json.ObjectMap, key: []const u8) ?usize {
    const value = payload.get(key) orelse return null;
    return switch (value) {
        .integer => |inner| if (inner >= 0) @intCast(inner) else null,
        else => null,
    };
}

pub fn parsePayloadF32(payload: std.json.ObjectMap, key: []const u8) ?f32 {
    const value = payload.get(key) orelse return null;
    return switch (value) {
        .float => |inner| @floatCast(inner),
        .integer => |inner| @floatFromInt(inner),
        else => null,
    };
}

pub fn parsePayloadBool(payload: std.json.ObjectMap, key: []const u8) ?bool {
    const value = payload.get(key) orelse return null;
    return switch (value) {
        .bool => |inner| inner,
        else => null,
    };
}

pub fn parsePayloadString(payload: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = payload.get(key) orelse return null;
    return switch (value) {
        .string => |inner| inner,
        else => null,
    };
}

pub fn makeRegularTrainingWindowValue(
    allocator: Allocator,
    model_id: []const u8,
    output_dir: []const u8,
    initial_weights: []const u8,
    assignment: data_assignment.DataAssignment,
    decoupled_loop: training_window.DecoupledLoopSpec,
    runtime: training_window.RegularRuntimeDescriptor,
) !std.json.Parsed(std.json.Value) {
    const spec = training_window.TrainingWindowSpec{
        .training_kind = training_window.training_kind_regular,
        .model_id = model_id,
        .data_assignment = assignment,
        .local_steps = decoupled_loop.max_local_steps,
        .output_contract = .{
            .kind = training_window.output_contract_kind_regular_checkpoint,
            .output_dir = output_dir,
            .include_weight_delta = false,
            .include_paths = true,
        },
        .initial_online_weights = blobRef(training_window.blob_online_weights, initial_weights),
        .decoupled_loop = decoupled_loop,
        .regular = runtime,
    };
    const json = try std.json.stringifyAlloc(allocator, spec, .{});
    defer allocator.free(json);
    return try std.json.parseFromSlice(std.json.Value, allocator, json, .{ .allocate = .alloc_always });
}

pub fn blobRef(name: []const u8, bytes: []const u8) training_window.BlobRef {
    return data_assignment.blobRef(name, bytes);
}

pub const RegularDecoupledRuntime = struct {
    allocator: Allocator,
    parameter_shapes: [][]i64,
    data_input_shapes: [][]i64,
    worker_graph_mlir_source: []u8,
    master_params: [][]f32,

    pub fn deinit(self: *@This()) void {
        for (self.master_params) |param| self.allocator.free(param);
        self.allocator.free(self.master_params);
        self.allocator.free(self.worker_graph_mlir_source);
        for (self.parameter_shapes) |shape| self.allocator.free(shape);
        self.allocator.free(self.parameter_shapes);
        for (self.data_input_shapes) |shape| self.allocator.free(shape);
        self.allocator.free(self.data_input_shapes);
        self.* = undefined;
    }
};

pub fn isRegularDecoupledDilocoConfig(cfg: experiment_config.ExperimentConfig) bool {
    const strategy = cfg.aggregation_strategy orelse return false;
    return std.mem.eql(u8, strategy, "decoupled_diloco");
}

pub fn parseRegularTrainingDType(value: []const u8) !tensor.DType {
    if (std.mem.eql(u8, value, "bf16")) return .bf16;
    if (std.mem.eql(u8, value, "f16")) return .f16;
    if (std.mem.eql(u8, value, "f32")) return .f32;
    return error.InvalidDType;
}

pub fn buildRegularDecoupledRuntime(
    allocator: Allocator,
    cfg: experiment_config.ExperimentConfig,
    mlir_builder: *MLIRBuilder,
) !RegularDecoupledRuntime {
    const dtype = try parseRegularTrainingDType(cfg.dtype);
    const raw_mlir = try readFileAllocAtPath(allocator, cfg.model_path, 10 * 1024 * 1024);
    defer allocator.free(raw_mlir);

    var current_source = try ModelSanitizer.applyStabilityPatches(allocator, raw_mlir);
    errdefer allocator.free(current_source);
    if (dtype == .bf16) {
        const decimal_source = try ModelSanitizer.sanitizeHexFloats(allocator, current_source);
        allocator.free(current_source);
        current_source = decimal_source;
    }

    const sanitized_source = try ModelSanitizer.sanitizeLargeConstants(allocator, current_source);
    allocator.free(current_source);
    defer allocator.free(sanitized_source);

    const metadata = try model_introspection.ModelInspector.inspect(
        allocator,
        mlir_builder.ctx,
        sanitized_source,
        2,
    );
    const parameter_shapes = metadata.parameter_shapes;
    const data_input_shapes = metadata.data_input_shapes;
    allocator.free(metadata.data_input_dtypes);
    if (metadata.trainable_parameter_indices) |indices| allocator.free(indices);
    errdefer {
        for (parameter_shapes) |shape| allocator.free(shape);
        allocator.free(parameter_shapes);
        for (data_input_shapes) |shape| allocator.free(shape);
        allocator.free(data_input_shapes);
    }

    const adam_config = adam_mlir.AdamMLIRConfiguration(f32){
        .learning_rate = cfg.learning_rate,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .weight_decay = 0.01,
    };
    const AdamMLIR = adam_mlir.AdamMLIR(f32);
    const adam_optimizer = try allocator.create(AdamMLIR);
    defer allocator.destroy(adam_optimizer);
    adam_optimizer.* = try AdamMLIR.init(allocator, mlir_builder, adam_config, dtype.toMLIRType(mlir_builder.ctx));
    defer adam_optimizer.deinit();

    const worker_graph_mlir_source = try GraphBuilder.buildTrainingGraph(
        allocator,
        mlir_builder,
        sanitized_source,
        adam_optimizer,
        parameter_shapes.len,
        null,
    );
    errdefer allocator.free(worker_graph_mlir_source);

    const master_params = try initRegularF32MasterParams(allocator, parameter_shapes);
    errdefer {
        for (master_params) |param| allocator.free(param);
        allocator.free(master_params);
    }

    return .{
        .allocator = allocator,
        .parameter_shapes = parameter_shapes,
        .data_input_shapes = data_input_shapes,
        .worker_graph_mlir_source = worker_graph_mlir_source,
        .master_params = master_params,
    };
}

pub fn initRegularF32MasterParams(allocator: Allocator, parameter_shapes: []const []const i64) ![][]f32 {
    const params = try allocator.alloc([]f32, parameter_shapes.len);
    var initialized: usize = 0;
    errdefer {
        for (params[0..initialized]) |param| allocator.free(param);
        allocator.free(params);
    }

    var rng = std.Random.DefaultPrng.init(12345);
    for (parameter_shapes, 0..) |shape, i| {
        const elem_count = try parameterElementCountFromShape(shape);
        params[i] = try allocator.alloc(f32, elem_count);
        initialized += 1;
        for (params[i]) |*value| {
            value.* = rng.random().floatNorm(f32) * 0.02;
        }
    }
    return params;
}

pub fn flattenF32Matrix(allocator: Allocator, matrix: []const []const f32) ![]u8 {
    var total_values: usize = 0;
    for (matrix) |values| total_values += values.len;

    const bytes = try allocator.alloc(u8, total_values * @sizeOf(f32));
    var offset: usize = 0;
    for (matrix) |values| {
        const value_bytes = std.mem.sliceAsBytes(values);
        @memcpy(bytes[offset .. offset + value_bytes.len], value_bytes);
        offset += value_bytes.len;
    }
    return bytes;
}

pub const DecoupledOutputArtifacts = struct {
    outer_optimizer_state: []u8,
    syncer_state_json: []u8,

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        allocator.free(self.outer_optimizer_state);
        allocator.free(self.syncer_state_json);
    }
};

pub fn writeDecoupledSyncerOutputArtifacts(
    allocator: Allocator,
    output_dir: []const u8,
    checkpoint: *const decoupled_diloco.SyncerCheckpoint,
) !DecoupledOutputArtifacts {
    const outer_optimizer_state = try flattenF32Matrix(allocator, checkpoint.optimizer_velocities);
    errdefer allocator.free(outer_optimizer_state);
    const outer_optimizer_state_path = try std.fs.path.join(allocator, &.{ output_dir, "outer_optimizer_state.bin" });
    defer allocator.free(outer_optimizer_state_path);
    try writeFileAtPath(outer_optimizer_state_path, outer_optimizer_state);

    const syncer_state_json = try std.json.stringifyAlloc(allocator, .{
        .schema = "pcp.decoupled_syncer_state.v1",
        .global_step = checkpoint.global_step,
        .fragment_versions = checkpoint.fragment_versions,
        .num_fragments = checkpoint.num_fragments,
        .sync_interval_h = checkpoint.sync_interval_h,
        .fragment_strategy = @tagName(checkpoint.fragment_strategy),
        .merge_strategy = @tagName(checkpoint.merge_strategy),
        .vector_clock_entries = checkpoint.vector_clock.entries.len,
        .event_tape_entries = checkpoint.event_tape.entries.len,
        .event_tape_cursor = checkpoint.event_tape_cursor,
        .outer_optimizer = "nesterov",
        .outer_optimizer_state_path = "outer_optimizer_state.bin",
    }, .{});
    errdefer allocator.free(syncer_state_json);
    const syncer_state_path = try std.fs.path.join(allocator, &.{ output_dir, "syncer_state.json" });
    defer allocator.free(syncer_state_path);
    try writeFileAtPath(syncer_state_path, syncer_state_json);

    return .{
        .outer_optimizer_state = outer_optimizer_state,
        .syncer_state_json = syncer_state_json,
    };
}

pub fn decoupledRuntimeCounters(
    stats: DecoupledRuntimeStats,
    checkpoint: *const decoupled_diloco.SyncerCheckpoint,
) training_state.RuntimeCounters {
    return .{
        .quorum_participants_total = stats.quorum_participants_total,
        .grace_inclusions_total = stats.grace_inclusions_total,
        .skipped_learners_total = stats.skipped_learners_total,
        .last_participant_count = stats.last_participant_count,
        .last_grace_inclusion_count = stats.last_grace_inclusion_count,
        .last_skipped_learner_count = stats.last_skipped_learner_count,
        .last_quorum_wait_ms = stats.last_quorum_wait_ms,
        .last_grace_window_ms = stats.last_grace_window_ms,
        .learner_to_syncer_fragment_bytes = stats.learner_to_syncer_fragment_bytes,
        .syncer_to_learner_fragment_bytes = stats.syncer_to_learner_fragment_bytes,
        .vector_clock_entries = checkpoint.vector_clock.entries.len,
        .event_tape_entries = checkpoint.event_tape.entries.len,
    };
}

pub fn regularTrainingStateDecoupledConfig(
    cfg: experiment_config.ExperimentConfig,
    syncer_steps: usize,
) training_state.DecoupledConfig {
    return .{
        .outer_loop_steps = cfg.outer_loop_steps,
        .syncer_steps = syncer_steps,
        .num_fragments = cfg.num_fragments,
        .sync_interval_h = cfg.sync_interval_h orelse cfg.tau,
        .overlap_tau = cfg.overlap_tau,
        .min_quorum = cfg.min_quorum,
        .merge_strategy = cfg.merge_strategy,
        .fragment_strategy = cfg.fragment_strategy,
        .learner_alpha = cfg.learner_alpha,
        .outer_gradient_compression = cfg.outer_gradient_compression,
        .max_recovery_syncer_steps = cfg.max_recovery_syncer_steps,
        .grace_window_ms = cfg.grace_window_ms,
        .grace_gamma = cfg.grace_gamma,
        .max_grace_steps = cfg.max_grace_steps,
        .adaptive_grace_enabled = cfg.adaptive_grace_enabled,
    };
}

pub fn resolveRegularMicroBatch(data_input_shapes: []const []const i64) usize {
    if (data_input_shapes.len == 0 or data_input_shapes[0].len == 0) return 4;
    const batch_dim = data_input_shapes[0][0];
    if (batch_dim <= 0 or batch_dim > 1_000_000) return 4;
    return @intCast(batch_dim);
}

pub fn makeRegularDecoupledConfig(cfg: experiment_config.ExperimentConfig) !decoupled_diloco.DecoupledDiLoCoConfig {
    const sync_interval_h = cfg.sync_interval_h orelse cfg.tau;
    return .{
        .num_fragments = cfg.num_fragments,
        .sync_interval_h = sync_interval_h,
        .overlap_tau = cfg.overlap_tau,
        .min_quorum = cfg.min_quorum,
        .fragment_strategy = try parseFragmentStrategy(cfg.fragment_strategy),
        .merge_strategy = try parseMergeStrategy(cfg.merge_strategy),
        .embedding_tensor_indices = cfg.embedding_tensor_indices,
        .learner_alpha = cfg.learner_alpha,
        .outer_gradient_compression = try parseOuterGradientCompression(cfg.outer_gradient_compression),
        .max_recovery_syncer_steps = cfg.max_recovery_syncer_steps,
        .outer_learning_rate = cfg.outer_learning_rate orelse 0.7,
        .outer_momentum = cfg.nesterov_momentum,
        .grace_window_ms = cfg.grace_window_ms,
        .grace_gamma = cfg.grace_gamma,
        .max_grace_steps = cfg.max_grace_steps,
        .adaptive_grace_enabled = cfg.adaptive_grace_enabled,
    };
}

pub fn writeRegularDecoupledDilocoOutputs(
    allocator: Allocator,
    cfg: experiment_config.ExperimentConfig,
    output_dir: []const u8,
    online_weights: []const u8,
    syncer: *decoupled_diloco.DecoupledDiLoCo,
    worker_count: usize,
    syncer_steps: usize,
    train_examples: usize,
    final_loss: f32,
    run_id: []const u8,
    stats: DecoupledRuntimeStats,
) !void {
    try ensureDirAtPath(output_dir);

    const online_weights_path = try std.fs.path.join(allocator, &.{ output_dir, "online_weights.bin" });
    defer allocator.free(online_weights_path);
    try writeFileAtPath(online_weights_path, online_weights);

    const checkpoint = try syncer.checkpoint(allocator);
    defer {
        var mutable = checkpoint;
        mutable.deinit();
    }
    var output_artifacts = try writeDecoupledSyncerOutputArtifacts(allocator, output_dir, &checkpoint);
    defer output_artifacts.deinit(allocator);

    const state_json = try std.json.stringifyAlloc(allocator, .{
        .schema = training_state.schema_name,
        .training_kind = training_state.training_kind_regular,
        .run_id = run_id,
        .model_id = cfg.model_path,
        .model_path = cfg.model_path,
        .data_path = cfg.data_path,
        .tokenizer = cfg.tokenizer,
        .sampling = cfg.sampling,
        .dtype = "f32",
        .steps_completed = syncer_steps,
        .train_examples = train_examples,
        .final_loss = final_loss,
        .workers = worker_count,
        .aggregation = training_state.aggregation_decoupled_diloco,
        .outer_optimizer = "nesterov",
        .artifacts = training_state.ArtifactRefs{
            .online_weights = .{
                .kind = training_state.artifact_kind_weights,
                .path = "online_weights.bin",
                .byte_count = online_weights.len,
            },
            .outer_optimizer_state = .{
                .kind = training_state.artifact_kind_outer_optimizer_state,
                .path = "outer_optimizer_state.bin",
                .byte_count = output_artifacts.outer_optimizer_state.len,
            },
            .syncer_state = .{
                .kind = training_state.artifact_kind_syncer_state,
                .path = "syncer_state.json",
                .byte_count = output_artifacts.syncer_state_json.len,
            },
        },
        .outer_optimizer_config = training_state.OuterOptimizer{
            .learning_rate = cfg.outer_learning_rate orelse 0.7,
            .momentum = cfg.nesterov_momentum,
        },
        .decoupled = regularTrainingStateDecoupledConfig(cfg, syncer_steps),
        .runtime_counters = decoupledRuntimeCounters(stats, &checkpoint),
        .regular = training_state.RegularExtension{
            .model_path = cfg.model_path,
            .data_path = cfg.data_path,
            .tokenizer = cfg.tokenizer,
            .sampling = cfg.sampling,
            .dtype = "f32",
            .final_loss = final_loss,
        },
        .syncer_steps = syncer_steps,
        .outer_loop_steps = cfg.outer_loop_steps,
        .num_fragments = cfg.num_fragments,
        .sync_interval_h = cfg.sync_interval_h orelse cfg.tau,
        .overlap_tau = cfg.overlap_tau,
        .min_quorum = cfg.min_quorum,
        .merge_strategy = cfg.merge_strategy,
        .fragment_strategy = cfg.fragment_strategy,
        .outer_learning_rate = cfg.outer_learning_rate orelse 0.7,
        .outer_momentum = cfg.nesterov_momentum,
        .learner_alpha = cfg.learner_alpha,
        .outer_gradient_compression = cfg.outer_gradient_compression,
        .max_recovery_syncer_steps = cfg.max_recovery_syncer_steps,
        .grace_window_ms = cfg.grace_window_ms,
        .grace_gamma = cfg.grace_gamma,
        .max_grace_steps = cfg.max_grace_steps,
        .adaptive_grace_enabled = cfg.adaptive_grace_enabled,
        .quorum_participants_total = stats.quorum_participants_total,
        .grace_inclusions_total = stats.grace_inclusions_total,
        .skipped_learners_total = stats.skipped_learners_total,
        .last_participant_count = stats.last_participant_count,
        .last_grace_inclusion_count = stats.last_grace_inclusion_count,
        .last_skipped_learner_count = stats.last_skipped_learner_count,
        .last_quorum_wait_ms = stats.last_quorum_wait_ms,
        .last_grace_window_ms = stats.last_grace_window_ms,
        .learner_to_syncer_fragment_bytes = stats.learner_to_syncer_fragment_bytes,
        .syncer_to_learner_fragment_bytes = stats.syncer_to_learner_fragment_bytes,
        .vector_clock_entries = checkpoint.vector_clock.entries.len,
        .event_tape_entries = checkpoint.event_tape.entries.len,
    }, .{});
    defer allocator.free(state_json);

    const training_state_path = try std.fs.path.join(allocator, &.{ output_dir, "training_state.json" });
    defer allocator.free(training_state_path);
    try writeFileAtPath(training_state_path, state_json);
}

pub fn parseFragmentStrategy(value: []const u8) !fragments.FragmentStrategy {
    if (std.mem.eql(u8, value, "strided")) return .strided;
    if (std.mem.eql(u8, value, "balanced_tensor")) return .balanced_tensor;
    return error.InvalidFragmentStrategy;
}

pub fn parseMergeStrategy(value: []const u8) !decoupled_diloco.MergeStrategy {
    if (std.mem.eql(u8, value, "weighted_average")) return .weighted_average;
    if (std.mem.eql(u8, value, "rda")) return .rda;
    if (std.mem.eql(u8, value, "avg_embedding_rda_model")) return .avg_embedding_rda_model;
    if (std.mem.eql(u8, value, "rda_embedding_avg_model")) return .rda_embedding_avg_model;
    return error.InvalidMergeStrategy;
}

pub fn parseOuterGradientCompression(value: []const u8) !decoupled_diloco.OuterGradientCompression {
    if (std.mem.eql(u8, value, "none")) return .none;
    if (std.mem.eql(u8, value, "int4")) return .int4;
    return error.InvalidOuterGradientCompression;
}

pub fn decoupledFragmentByteCount(fragment: fragments.Fragment) usize {
    var total: usize = 0;
    for (fragment.tensors) |range| {
        total += range.value_count * @sizeOf(f32);
    }
    return total;
}

pub fn saturatingAddMs(lhs: u64, rhs: u64) u64 {
    return std.math.add(u64, lhs, rhs) catch std.math.maxInt(u64);
}

pub fn elapsedMsSince(start_ms: i64) u64 {
    return @intCast(@max(std.time.milliTimestamp() - start_ms, 0));
}

pub fn collectDecoupledMessagesTimed(
    worker_fabric: *WorkerFabricController,
    allocator: Allocator,
    filter: message.MessageFilter,
    min_count: usize,
    max_count: usize,
    timeout_ms: u64,
    grace_window_ms: u64,
) !std.ArrayList(TimedDecoupledMessage) {
    var collected = std.ArrayList(TimedDecoupledMessage).init(allocator);
    errdefer {
        for (collected.items) |*msg| msg.deinit(allocator);
        collected.deinit();
    }
    if (max_count == 0) return collected;

    const start_ms = std.time.milliTimestamp();
    var grace_deadline_ms: ?u64 = if (min_count == 0) grace_window_ms else null;
    while (collected.items.len < max_count) {
        const remaining = max_count - collected.items.len;
        var drained = try worker_fabric.drainMatchingMessages(filter, remaining);
        defer drained.deinit();

        if (drained.items.len > 0) {
            const arrival_ms = elapsedMsSince(start_ms);
            var moved: usize = 0;
            errdefer {
                for (drained.items[moved..]) |*msg| msg.deinitClone(allocator);
            }
            try collected.ensureUnusedCapacity(drained.items.len);
            for (drained.items) |envelope| {
                collected.appendAssumeCapacity(.{
                    .envelope = envelope,
                    .arrival_ms = arrival_ms,
                });
                moved += 1;
            }
        }

        const elapsed_ms = elapsedMsSince(start_ms);
        if (collected.items.len >= min_count and grace_deadline_ms == null) {
            grace_deadline_ms = saturatingAddMs(elapsed_ms, grace_window_ms);
        }
        if (collected.items.len >= max_count) break;
        if (collected.items.len >= min_count) {
            if (grace_window_ms == 0) break;
            if (grace_deadline_ms) |deadline| {
                if (elapsed_ms >= deadline) break;
            }
        }
        if (timeout_ms == 0 or elapsed_ms >= timeout_ms) break;
        std.time.sleep(10 * std.time.ns_per_ms);
    }

    return collected;
}

pub fn collectDecoupledMessagesUniqueTimed(
    worker_fabric: *WorkerFabricController,
    allocator: Allocator,
    filter: message.MessageFilter,
    worker_ids: []const NodeId,
    min_count: usize,
    max_count: usize,
    timeout_ms: u64,
    grace_window_ms: u64,
) !std.ArrayList(TimedDecoupledMessage) {
    var collected = std.ArrayList(TimedDecoupledMessage).init(allocator);
    errdefer {
        for (collected.items) |*msg| msg.deinit(allocator);
        collected.deinit();
    }
    if (worker_ids.len == 0 or max_count == 0) return collected;

    const bounded_max_count = @min(max_count, worker_ids.len);
    var seen = std.AutoHashMap(NodeId, void).init(allocator);
    defer seen.deinit();

    const start_ms = std.time.milliTimestamp();
    var grace_deadline_ms: ?u64 = if (min_count == 0) grace_window_ms else null;
    while (collected.items.len < bounded_max_count) {
        for (worker_ids) |worker_id| {
            if (seen.contains(worker_id)) continue;
            var drained = try worker_fabric.drainMatchingMessagesFromWorker(filter, worker_id, 1);
            defer drained.deinit();
            if (drained.items.len == 0) continue;
            const arrival_ms = elapsedMsSince(start_ms);
            var timed = TimedDecoupledMessage{
                .envelope = drained.items[0],
                .arrival_ms = arrival_ms,
            };
            errdefer timed.deinit(allocator);
            try collected.ensureUnusedCapacity(1);
            try seen.put(worker_id, {});
            collected.appendAssumeCapacity(timed);
            if (collected.items.len >= bounded_max_count) break;
        }

        const elapsed_ms = elapsedMsSince(start_ms);
        if (collected.items.len >= min_count and grace_deadline_ms == null) {
            grace_deadline_ms = saturatingAddMs(elapsed_ms, grace_window_ms);
        }
        if (collected.items.len >= bounded_max_count) break;
        if (collected.items.len >= min_count) {
            if (grace_window_ms == 0) break;
            if (grace_deadline_ms) |deadline| {
                if (elapsed_ms >= deadline) break;
            }
        }
        if (timeout_ms == 0 or elapsed_ms >= timeout_ms) break;
        std.time.sleep(10 * std.time.ns_per_ms);
    }

    return collected;
}

pub fn loadF32WeightMatrix(
    allocator: Allocator,
    weights: []const u8,
    parameter_shapes: []const []const i64,
) ![][]f32 {
    const matrix = try allocator.alloc([]f32, parameter_shapes.len);
    var initialized: usize = 0;
    errdefer {
        for (matrix[0..initialized]) |values| allocator.free(values);
        allocator.free(matrix);
    }

    var byte_offset: usize = 0;
    for (parameter_shapes, 0..) |shape, idx| {
        const elem_count = try parameterElementCountFromShape(shape);
        const byte_count = elem_count * @sizeOf(f32);
        if (byte_offset + byte_count > weights.len) return error.WeightSizeMismatch;
        matrix[idx] = try allocator.alloc(f32, elem_count);
        initialized += 1;
        for (matrix[idx], 0..) |*value, value_idx| {
            const offset = byte_offset + value_idx * @sizeOf(f32);
            const bits = std.mem.readInt(u32, weights[offset..][0..4], .little);
            value.* = @bitCast(bits);
        }
        byte_offset += byte_count;
    }
    if (byte_offset != weights.len) return error.WeightSizeMismatch;
    return matrix;
}

pub fn freeF32Matrix(allocator: Allocator, matrix: [][]f32) void {
    for (matrix) |values| allocator.free(values);
    allocator.free(matrix);
}

pub fn parameterElementCountFromShape(shape: []const i64) !usize {
    var elem_count: usize = 1;
    for (shape) |dim| {
        if (dim <= 0) return error.DynamicParameterShapesNotSupported;
        elem_count = try std.math.mul(usize, elem_count, @intCast(dim));
    }
    return elem_count;
}

pub fn parseDecoupledLearnerUpdate(
    allocator: Allocator,
    msg: message.MessageEnvelope,
    fragment: fragments.Fragment,
    arrival_ms: u64,
) !OwnedDecoupledLearnerUpdate {
    const payload = switch (msg.data) {
        .object => |obj| obj,
        else => return error.InvalidRemoteTrainingPayload,
    };
    const updates_value = payload.get(message.DecoupledDiLoCoField.UPDATES) orelse return error.MissingUpdatesField;
    const updates_array = switch (updates_value) {
        .array => |array| array,
        else => return error.InvalidUpdatesFormat,
    };

    var tensors = try allocator.alloc(decoupled_diloco.TensorUpdate, updates_array.items.len);
    errdefer allocator.free(tensors);
    var values = try allocator.alloc([]f32, updates_array.items.len);
    var values_initialized: usize = 0;
    errdefer {
        for (values[0..values_initialized]) |value| allocator.free(value);
        allocator.free(values);
    }

    for (updates_array.items, 0..) |update_value, idx| {
        const update = switch (update_value) {
            .object => |obj| obj,
            else => return error.InvalidUpdatesFormat,
        };
        const tensor_index = @as(usize, @intCast((update.get("tensor_idx") orelse return error.MissingTensorIndex).integer));
        const range = fragment.findTensor(tensor_index) orelse return error.InvalidTensorIndex;
        const data_b64 = switch (update.get("data") orelse return error.MissingDataField) {
            .string => |s| s,
            else => return error.InvalidDataField,
        };
        const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(data_b64);
        if (decoded_size != range.value_count * @sizeOf(f32)) return error.FragmentLengthMismatch;
        const decoded = try allocator.alloc(u8, decoded_size);
        defer allocator.free(decoded);
        try std.base64.standard.Decoder.decode(decoded, data_b64);

        values[idx] = try allocator.alloc(f32, range.value_count);
        values_initialized += 1;
        for (values[idx], 0..) |*value, value_idx| {
            const offset = value_idx * @sizeOf(f32);
            const bits = std.mem.readInt(u32, decoded[offset..][0..4], .little);
            value.* = @bitCast(bits);
        }
        tensors[idx] = .{ .tensor_index = tensor_index, .values = values[idx] };
    }

    const learner_step = parsePayloadUsize(payload, message.DecoupledDiLoCoField.LEARNER_STEP) orelse 0;
    const steps_since = parsePayloadUsize(payload, message.DecoupledDiLoCoField.STEPS_SINCE_FRAGMENT_UPDATE) orelse 1;
    const tokens_since = parsePayloadUsize(payload, message.DecoupledDiLoCoField.TOKENS_SINCE_FRAGMENT_UPDATE) orelse steps_since;

    return .{
        .update = .{
            .learner_id = msg.sender_node,
            .learner_step = learner_step,
            .steps_since_fragment_update = steps_since,
            .tokens_since_fragment_update = tokens_since,
            .arrival_ms = arrival_ms,
            .tensors = tensors,
        },
        .tensors = tensors,
        .values = values,
    };
}

pub fn copySyncerFragmentToWeightBlob(
    syncer: *decoupled_diloco.DecoupledDiLoCo,
    parameter_shapes: []const []const i64,
    weights: []u8,
    fragment_id: usize,
) !void {
    if (fragment_id >= syncer.fragment_plan.fragments.len) return error.InvalidFragmentId;
    const fragment = syncer.fragment_plan.fragments[fragment_id];
    for (fragment.tensors) |range| {
        const byte_offset = try tensorByteOffset(parameter_shapes, range.tensor_index);
        const master = syncer.masterTensor(range.tensor_index);
        if (master.len != range.value_count) return error.FragmentLengthMismatch;
        for (master, 0..) |value, idx| {
            const bits: u32 = @bitCast(value);
            std.mem.writeInt(u32, weights[byte_offset + idx * @sizeOf(f32) ..][0..4], bits, .little);
        }
    }
}

pub fn tensorByteOffset(parameter_shapes: []const []const i64, tensor_index: usize) !usize {
    if (tensor_index >= parameter_shapes.len) return error.TensorIndexOutOfBounds;
    var byte_offset: usize = 0;
    for (parameter_shapes[0..tensor_index]) |shape| {
        byte_offset += try parameterElementCountFromShape(shape) * @sizeOf(f32);
    }
    return byte_offset;
}

pub fn makeDecoupledReadyValue(
    allocator: Allocator,
    syncer: *decoupled_diloco.DecoupledDiLoCo,
    parameter_shapes: []const []const i64,
    target_weights: ?[]const u8,
    fragment_id: usize,
    syncer_step: usize,
) !std.json.Parsed(std.json.Value) {
    if (fragment_id >= syncer.fragment_plan.fragments.len) return error.InvalidFragmentId;
    const fragment = syncer.fragment_plan.fragments[fragment_id];

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    var root = std.json.ObjectMap.init(scratch);
    try root.put(message.DecoupledDiLoCoField.FRAGMENT_ID, .{ .integer = @intCast(fragment_id) });
    try root.put(message.DecoupledDiLoCoField.FRAGMENT_ROUND, .{ .integer = @intCast(syncer_step) });
    try root.put(message.DecoupledDiLoCoField.SYNCER_STEP, .{ .integer = @intCast(syncer_step) });

    var updates = std.json.Array.init(scratch);
    for (fragment.tensors) |range| {
        const master = syncer.masterTensor(range.tensor_index);
        const tensor_bytes = std.mem.sliceAsBytes(master);
        const encoded_len = std.base64.standard.Encoder.calcSize(tensor_bytes.len);
        const encoded = try scratch.alloc(u8, encoded_len);
        _ = std.base64.standard.Encoder.encode(encoded, tensor_bytes);

        var update = std.json.ObjectMap.init(scratch);
        try update.put("tensor_idx", .{ .integer = @intCast(range.tensor_index) });
        try update.put("data", .{ .string = encoded });
        try updates.append(.{ .object = update });
    }
    try root.put(message.DecoupledDiLoCoField.UPDATES, .{ .array = updates });

    if (target_weights) |target| {
        var target_updates = std.json.Array.init(scratch);
        for (fragment.tensors) |range| {
            const byte_offset = try tensorByteOffset(parameter_shapes, range.tensor_index);
            const byte_count = range.value_count * @sizeOf(f32);
            if (byte_offset + byte_count > target.len) return error.WeightSizeMismatch;
            const tensor_bytes = target[byte_offset .. byte_offset + byte_count];
            const encoded_len = std.base64.standard.Encoder.calcSize(tensor_bytes.len);
            const encoded = try scratch.alloc(u8, encoded_len);
            _ = std.base64.standard.Encoder.encode(encoded, tensor_bytes);

            var update = std.json.ObjectMap.init(scratch);
            try update.put("tensor_idx", .{ .integer = @intCast(range.tensor_index) });
            try update.put("data", .{ .string = encoded });
            try target_updates.append(.{ .object = update });
        }
        try root.put("target_updates", .{ .array = target_updates });
    }

    const json = try std.json.stringifyAlloc(allocator, std.json.Value{ .object = root }, .{});
    defer allocator.free(json);
    return try std.json.parseFromSlice(std.json.Value, allocator, json, .{ .allocate = .alloc_always });
}

pub fn currentStateStatus(state: *control_state.ControllerState) control_state.JobStatus {
    state.mutex.lock();
    defer state.mutex.unlock();
    return state.status;
}

pub fn stateIsTerminal(state: *control_state.ControllerState) bool {
    return switch (currentStateStatus(state)) {
        .completed, .failed, .cancelled => true,
        else => false,
    };
}

pub fn parseBackendName(name: []const u8) !backend_selection.Backend {
    if (std.mem.eql(u8, name, "metal")) return .metal;
    if (std.mem.eql(u8, name, "cuda")) return .cuda;
    if (std.mem.eql(u8, name, "vulkan")) return .vulkan;
    if (std.mem.eql(u8, name, "rocm")) return .rocm;
    if (std.mem.eql(u8, name, "cpu")) return .cpu;
    return error.UnknownBackend;
}

pub fn applySchedulingPolicy(
    worker_fabric: *WorkerFabricController,
    owner: WorkerLeaseOwner,
    configured_worker_class: ?[]const u8,
    configured_target_arch: ?[]const u8,
    reserved_workers: ?usize,
    max_workers: ?usize,
    default_worker_class: ?[]const u8,
    default_target_arch: ?[]const u8,
) !void {
    const raw_worker_class = configured_worker_class orelse default_worker_class;
    const worker_class = if (raw_worker_class) |value|
        scheduling.WorkerClass.parse(value) orelse return error.InvalidWorkerClass
    else
        scheduling.WorkerClass.any;
    const target_arch = normalizeTargetArch(configured_target_arch) orelse normalizeTargetArch(default_target_arch);

    if (max_workers) |limit| {
        if ((reserved_workers orelse 0) > limit) {
            return error.InvalidWorkerReservation;
        }
    }

    worker_fabric.setSchedulingPolicy(owner, WorkerSchedulingPolicy{
        .worker_class = worker_class,
        .target_arch = target_arch,
        .reserved_workers = reserved_workers orelse 0,
        .max_workers = max_workers,
    });
}

pub fn acquireWorkersForEmbeddedJob(
    allocator: Allocator,
    worker_fabric: *WorkerFabricController,
    state: *control_state.ControllerState,
    owner: WorkerLeaseOwner,
    required_workers: usize,
) ![]NodeId {
    const budget = protocol_limits.workerJoinBudget(required_workers);
    try budget.validate();
    const start_ms = std.time.milliTimestamp();
    var next_log_ms: u64 = 5_000;

    while (!budget.expired(start_ms)) {
        if (state.isCancellationRequested()) return error.Cancelled;

        const available_workers = worker_fabric.getAvailableWorkerCountForLeaseOwner(owner);
        try updateScopedWaitPhase(state, worker_fabric, owner, available_workers, required_workers);
        if (worker_fabric.acquireIdleWorkers(allocator, owner, required_workers)) |workers| {
            return workers;
        } else |err| switch (err) {
            error.NotEnoughIdleWorkers => {},
            else => return err,
        }

        const elapsed_ms = budget.elapsedMs(start_ms);
        if (elapsed_ms >= next_log_ms) {
            std.log.info("Waiting for embedded job workers: available={}/{} after {}ms", .{
                available_workers,
                required_workers,
                elapsed_ms,
            });
            next_log_ms += 5_000;
        }
        std.time.sleep(budget.sleepMs(start_ms) * std.time.ns_per_ms);
    }
    return error.WorkerJoinTimeout;
}

pub fn resolveEmbeddedJobWorkers(
    allocator: Allocator,
    worker_fabric: *WorkerFabricController,
    state: *control_state.ControllerState,
    owner: WorkerLeaseOwner,
    required_workers: usize,
    reserved_worker_ids: ?[]const NodeId,
) ![]NodeId {
    if (state.isCancellationRequested()) return error.Cancelled;

    if (reserved_worker_ids) |worker_ids| {
        if (worker_ids.len != required_workers) return error.InvalidReservationRequest;
        const leased_workers = try worker_fabric.filterLeasedWorkerIds(allocator, owner, worker_ids);
        errdefer allocator.free(leased_workers);
        if (leased_workers.len != required_workers) return error.ReservationUnavailable;
        return leased_workers;
    }

    state.setStatus(.waiting_for_workers);
    return acquireWorkersForEmbeddedJob(allocator, worker_fabric, state, owner, required_workers);
}

pub fn updateScopedWaitPhase(
    state: *control_state.ControllerState,
    worker_fabric: *WorkerFabricController,
    owner: WorkerLeaseOwner,
    available_workers: usize,
    required_workers: usize,
) !void {
    const policy = worker_fabric.getSchedulingPolicy(owner);
    var detail_buf: [160]u8 = undefined;
    const detail = try std.fmt.bufPrint(
        &detail_buf,
        "available={d}/{d} connected={d} class={s} target={s}",
        .{
            available_workers,
            required_workers,
            worker_fabric.getWorkerCount(),
            policy.worker_class.asString(),
            policy.target_arch orelse "any",
        },
    );
    try state.setReadinessPhase("waiting_for_workers", detail);
}

pub fn normalizeTargetArch(raw: ?[]const u8) ?[]const u8 {
    const value = raw orelse return null;
    if (value.len == 0) return null;
    return value;
}

pub fn validateTemplateOnlyJobSubmitBody(allocator: Allocator, request_body: []const u8) !void {
    if (request_body.len == 0) return;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, request_body, .{}) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidJobRequest,
    };
    defer parsed.deinit();

    const object = switch (parsed.value) {
        .object => |value| value,
        else => return error.InvalidJobRequest,
    };

    var iterator = object.iterator();
    while (iterator.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "service_id")) continue;
        if (std.mem.eql(u8, entry.key_ptr.*, "executor_id")) continue;
        if (std.mem.eql(u8, entry.key_ptr.*, "training_kind")) continue;
        if (std.mem.eql(u8, entry.key_ptr.*, "model_type")) continue;
        return error.UnsupportedJobOverrides;
    }
}

test "template-only job submit body allows routing fields only" {
    try validateTemplateOnlyJobSubmitBody(std.testing.allocator, "");
    try validateTemplateOnlyJobSubmitBody(std.testing.allocator, "{\"training_kind\":\"regular\",\"service_id\":\"svc\"}");
    try validateTemplateOnlyJobSubmitBody(std.testing.allocator, "{\"model_type\":\"llm\",\"executor_id\":\"exec\"}");
    try std.testing.expectError(
        error.UnsupportedJobOverrides,
        validateTemplateOnlyJobSubmitBody(std.testing.allocator, "{\"learning_rate\":0.1}"),
    );
}

pub fn resolveReservationWorkersRequired(requested: ?usize, configured: usize) !usize {
    if (requested) |value| {
        if (value == 0 or value != configured) return error.InvalidReservationRequest;
        return value;
    }
    return configured;
}

pub fn publishScopedServiceIfPresent(publisher: ?*ScopedServicePublisher) void {
    const value = publisher orelse return;
    value.publishNoFail();
}

pub fn embeddedTrainingRunThread(ctx: *TrainingRunContext) !void {
    ctx.state.setStatus(.waiting_for_workers);
    const leased_workers = acquireWorkersForEmbeddedJob(
        ctx.worker_fabric.allocator,
        ctx.worker_fabric,
        ctx.state,
        .training,
        ctx.required_workers,
    ) catch |err| {
        if (err == error.Cancelled) {
            ctx.state.setCancelled();
            return;
        }
        try ctx.state.setFailed(@errorName(err));
        return err;
    };
    defer ctx.worker_fabric.allocator.free(leased_workers);
    defer {
        ctx.diloco.clearWorkerScope();
        ctx.worker_fabric.releaseWorkers(leased_workers);
        publishScopedServiceIfPresent(ctx.publisher);
    }
    try ctx.diloco.setWorkerScope(leased_workers);
    publishScopedServiceIfPresent(ctx.publisher);

    ctx.algorithm.run() catch |err| {
        if (err == error.Cancelled) {
            ctx.state.setCancelled();
            return;
        }
        try ctx.state.setFailed(@errorName(err));
        return err;
    };

    if (ctx.state.isCancellationRequested()) {
        ctx.state.setCancelled();
    } else {
        ctx.state.setStatus(.completed);
    }
}

pub fn embeddedRLRunThread(ctx: *RLRunContext) !void {
    const leased_workers = resolveEmbeddedJobWorkers(
        ctx.worker_fabric.allocator,
        ctx.worker_fabric,
        ctx.state,
        .rl,
        ctx.required_workers,
        ctx.reserved_worker_ids,
    ) catch |err| {
        if (err == error.Cancelled) {
            ctx.state.setCancelled();
            return;
        }
        try ctx.state.setFailed(@errorName(err));
        return err;
    };
    defer ctx.worker_fabric.allocator.free(leased_workers);
    defer {
        ctx.controller.clearWorkerScope();
        ctx.worker_fabric.releaseWorkers(leased_workers);
        publishScopedServiceIfPresent(ctx.publisher);
    }
    try ctx.controller.setWorkerScope(leased_workers);
    publishScopedServiceIfPresent(ctx.publisher);

    ctx.algorithm.run() catch |err| {
        if (err == error.Cancelled) {
            ctx.state.setCancelled();
            return;
        }
        try ctx.state.setFailed(@errorName(err));
        return err;
    };

    if (ctx.state.isCancellationRequested()) {
        ctx.state.setCancelled();
    } else {
        ctx.state.setStatus(.completed);
    }
}
