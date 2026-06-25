const common = @import("../embedded_common.zig");
const std = common.std;
const Allocator = common.Allocator;
const training_algorithm = common.training_algorithm;
const diloco = common.diloco;
const decoupled_diloco = common.decoupled_diloco;
const fragments = common.fragments;
const grpo = common.grpo;
const adam_mlir = common.adam_mlir;
const backend_selection = common.backend_selection;
const GraphBuilder = common.GraphBuilder;
const ModelSanitizer = common.ModelSanitizer;
const ops = common.ops;
const tensor = common.tensor;
const control_api = common.control_api;
const control_state = common.control_state;
const data_assignment = common.data_assignment;
const experiment_config = common.experiment_config;
const inference_config = common.inference_config;
const training_window = common.training_window;
const model_introspection = common.model_introspection;
const message = common.message;
const message_registry = common.message_registry;
const nesterov = common.nesterov;
const file_util = common.file_util;
const wandb = common.wandb;
const config = common.config;
const embedded_jobs = common.embedded_jobs;
const gateway = common.gateway;
const identity = common.identity;
const scheduling = common.scheduling;
const inference_controller = common.inference_controller;
const rl_controller = common.rl_controller;
const training_controller = common.training_controller;
const service_registry = common.service_registry;
const DiLoCo = common.DiLoCo;
const DiLoCoConfig = common.DiLoCoConfig;
const MLIRBuilder = common.MLIRBuilder;
const WorkerFabricController = common.WorkerFabricController;
const WorkerLeaseOwner = common.WorkerLeaseOwner;
const WorkerSchedulingPolicy = common.WorkerSchedulingPolicy;
const NodeId = common.NodeId;
const Nesterov = common.Nesterov;
const EmbeddedJobQueue = common.EmbeddedJobQueue;
const EmbeddedQueuedJob = common.EmbeddedQueuedJob;
const EmbeddedReservation = common.EmbeddedReservation;
const EmbeddedReservationStore = common.EmbeddedReservationStore;
const readFileAllocAtPath = common.readFileAllocAtPath;
const writeFileAtPath = common.writeFileAtPath;
const ensureDirAtPath = common.ensureDirAtPath;
const fileExistsAtPath = common.fileExistsAtPath;
const ScopedServicePublisher = common.ScopedServicePublisher;
const TrainingRunContext = common.TrainingRunContext;
const RLRunContext = common.RLRunContext;
const OwnedDecoupledLearnerUpdate = common.OwnedDecoupledLearnerUpdate;
const TimedDecoupledMessage = common.TimedDecoupledMessage;
const DecoupledRuntimeStats = common.DecoupledRuntimeStats;
const RegularDecoupledRuntime = common.RegularDecoupledRuntime;
const makeLocalJobEnvelope = common.makeLocalJobEnvelope;
const renderSyntheticJobJson = common.renderSyntheticJobJson;
const makeLocalReservationResult = common.makeLocalReservationResult;
const makeLocalReservationReleaseResult = common.makeLocalReservationReleaseResult;
const inferenceApiThread = common.inferenceApiThread;
const controlApiThread = common.controlApiThread;
const noOpCancel = common.noOpCancel;
const parsePayloadUsize = common.parsePayloadUsize;
const parsePayloadF32 = common.parsePayloadF32;
const parsePayloadBool = common.parsePayloadBool;
const parsePayloadString = common.parsePayloadString;
const blobRef = common.blobRef;
const isRegularDecoupledDilocoConfig = common.isRegularDecoupledDilocoConfig;
const parseRegularTrainingDType = common.parseRegularTrainingDType;
const buildRegularDecoupledRuntime = common.buildRegularDecoupledRuntime;
const initRegularF32MasterParams = common.initRegularF32MasterParams;
const flattenF32Matrix = common.flattenF32Matrix;
const resolveRegularMicroBatch = common.resolveRegularMicroBatch;
const makeRegularDecoupledConfig = common.makeRegularDecoupledConfig;
const writeRegularDecoupledDilocoOutputs = common.writeRegularDecoupledDilocoOutputs;
const parseFragmentStrategy = common.parseFragmentStrategy;
const parseMergeStrategy = common.parseMergeStrategy;
const parseOuterGradientCompression = common.parseOuterGradientCompression;
const decoupledFragmentByteCount = common.decoupledFragmentByteCount;
const saturatingAddMs = common.saturatingAddMs;
const elapsedMsSince = common.elapsedMsSince;
const collectDecoupledMessagesTimed = common.collectDecoupledMessagesTimed;
const collectDecoupledMessagesUniqueTimed = common.collectDecoupledMessagesUniqueTimed;
const loadF32WeightMatrix = common.loadF32WeightMatrix;
const freeF32Matrix = common.freeF32Matrix;
const parameterElementCountFromShape = common.parameterElementCountFromShape;
const parseDecoupledLearnerUpdate = common.parseDecoupledLearnerUpdate;
const copySyncerFragmentToWeightBlob = common.copySyncerFragmentToWeightBlob;
const tensorByteOffset = common.tensorByteOffset;
const makeDecoupledReadyValue = common.makeDecoupledReadyValue;
const currentStateStatus = common.currentStateStatus;
const stateIsTerminal = common.stateIsTerminal;
const parseBackendName = common.parseBackendName;
const applySchedulingPolicy = common.applySchedulingPolicy;
const acquireWorkersForEmbeddedJob = common.acquireWorkersForEmbeddedJob;
const resolveEmbeddedJobWorkers = common.resolveEmbeddedJobWorkers;
const updateScopedWaitPhase = common.updateScopedWaitPhase;
const normalizeTargetArch = common.normalizeTargetArch;
const validateTemplateOnlyJobSubmitBody = common.validateTemplateOnlyJobSubmitBody;
const resolveReservationWorkersRequired = common.resolveReservationWorkersRequired;
const publishScopedServiceIfPresent = common.publishScopedServiceIfPresent;
const embeddedTrainingRunThread = common.embeddedTrainingRunThread;
const embeddedRLRunThread = common.embeddedRLRunThread;

pub const EmbeddedRLService = struct {
    allocator: Allocator,
    gateway_instance: *gateway.Gateway,
    shared_worker_fabric: *WorkerFabricController,
    api_token: ?[]const u8,
    gateway_api_port: u16,
    config_result: ?experiment_config.ConfigResult = null,
    config_path: ?[]u8 = null,
    service_id: ?[]u8 = null,
    executor_id: ?[]u8 = null,
    configured_workers: usize = 0,
    training_backend: backend_selection.Backend = .cpu,
    state: ?control_state.ControllerState = null,
    api_server: ?control_api.ControlApiServer = null,
    publisher: ?*ScopedServicePublisher = null,
    manager_thread: ?std.Thread = null,
    jobs: EmbeddedJobQueue,
    reservations: EmbeddedReservationStore,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        gateway_instance: *gateway.Gateway,
        shared_worker_fabric: *WorkerFabricController,
        api_token: ?[]const u8,
        gateway_api_port: u16,
    ) Self {
        return .{
            .allocator = allocator,
            .gateway_instance = gateway_instance,
            .shared_worker_fabric = shared_worker_fabric,
            .api_token = api_token,
            .gateway_api_port = gateway_api_port,
            .jobs = EmbeddedJobQueue.init(allocator),
            .reservations = EmbeddedReservationStore.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.gateway_instance.setLocalJobHook(service_registry.ServiceType.rl.asString(), null);
        self.reservations.releaseAll(self.shared_worker_fabric);
        self.jobs.requestStop();
        if (self.state) |*state| state.requestCancel();
        if (self.api_server) |*server| server.stop();
        if (self.manager_thread) |thread| thread.join();
        if (self.state) |*state| state.setChangeHook(null);
        if (self.publisher) |publisher| publisher.deinit();
        if (self.service_id) |value| self.allocator.free(value);
        if (self.executor_id) |value| self.allocator.free(value);
        if (self.config_path) |value| self.allocator.free(value);
        if (self.state) |*state| state.deinit();
        if (self.config_result) |*result| result.deinit();
        self.jobs.deinit();
        self.reservations.deinit();
    }

    pub fn start(self: *Self, embedded_cfg: config.EmbeddedRLControllerConfig, default_workers: usize) !void {
        self.config_result = try experiment_config.loadConfig(self.allocator, embedded_cfg.config_path);
        self.config_path = try self.allocator.dupe(u8, embedded_cfg.config_path);
        const exp_config = self.config_result.?.config;
        const json_grpo = exp_config.grpo_config orelse return error.GRPOConfigRequired;
        const required_workers = embedded_cfg.workers orelse default_workers;
        self.configured_workers = required_workers;
        self.training_backend = try parseBackendName(embedded_cfg.backend);

        try applySchedulingPolicy(
            self.shared_worker_fabric,
            .rl,
            embedded_cfg.worker_class,
            embedded_cfg.target_arch,
            embedded_cfg.reserved_workers,
            embedded_cfg.max_workers,
            embedded_cfg.backend,
            null,
        );

        self.state = try control_state.ControllerState.init(
            self.allocator,
            .rl,
            .rl,
            embedded_cfg.config_path,
            null,
            exp_config.model_path,
            false,
            required_workers,
        );
        self.state.?.setRLProgress(0, json_grpo.num_iterations, 0, 0, 0.0);

        const api_host = if (embedded_cfg.api) |api_cfg|
            api_cfg.host orelse "127.0.0.1"
        else
            "127.0.0.1";
        const api_port = if (embedded_cfg.api) |api_cfg|
            api_cfg.port orelse self.gateway_api_port + 1
        else
            self.gateway_api_port + 1;
        const service_id = embedded_cfg.service_id orelse "rl-main";
        self.service_id = try self.allocator.dupe(u8, service_id);
        self.executor_id = try identity.deriveExecutorId(
            self.allocator,
            self.gateway_instance.config.gateway_id,
            "rl",
            service_id,
        );
        const base_url = try std.fmt.allocPrint(self.allocator, "http://{s}:{d}", .{ api_host, api_port });
        defer self.allocator.free(base_url);

        self.gateway_instance.setLocalJobHook(service_registry.ServiceType.rl.asString(), .{
            .service_id = self.service_id.?,
            .executor_id = self.executor_id.?,
            .service_type = service_registry.ServiceType.rl.asString(),
            .ctx = self,
            .submit = submitJobHook,
            .list = listJobsHook,
            .lookup = lookupJobHook,
            .cancel = cancelJobHook,
            .reserve = reserveJobHook,
            .commit_reservation = commitReservationHook,
            .release_reservation = releaseReservationHook,
        });

        self.publisher = try ScopedServicePublisher.init(
            self.allocator,
            &self.gateway_instance.service_registry,
            self.shared_worker_fabric,
            &self.state.?,
            service_id,
            self.executor_id.?,
            "rl",
            base_url,
            &[_][]const u8{
                message_registry.capability_rl,
                "controller.status",
                "job.current",
                "job.cancel",
            },
            .rl,
            null,
            true,
        );
        try self.publisher.?.bind();

        self.api_server = control_api.ControlApiServer.init(
            self.allocator,
            &self.state.?,
            self.shared_worker_fabric,
            self.api_token,
            self.publisher.?.snapshotProvider(),
            null,
            null,
            .rl,
            .{
                .ctx = self.shared_worker_fabric,
                .cancel = noOpCancel,
            },
        );
        const api_thread = try std.Thread.spawn(.{}, controlApiThread, .{ &self.api_server.?, api_host, api_port });
        api_thread.detach();

        self.manager_thread = try std.Thread.spawn(.{}, managerThread, .{self});
    }

    fn managerThread(self: *Self) !void {
        const exp_config = self.config_result.?.config;
        const json_grpo = exp_config.grpo_config orelse return error.GRPOConfigRequired;

        while (try self.jobs.waitForNextJob()) |job| {
            defer {
                var owned = job;
                owned.deinit(self.allocator);
            }

            self.runQueuedJob(job, json_grpo) catch |err| {
                if (self.state) |*state| {
                    if (!stateIsTerminal(state)) {
                        state.setFailed(@errorName(err)) catch |set_err| {
                            std.log.err("Failed to mark embedded RL job {s} as failed: {}", .{ job.job_id, set_err });
                        };
                    }
                }
                std.log.err("Embedded RL job {s} failed: {}", .{ job.job_id, err });
            };

            self.finalizeCompletedJob(exp_config.model_path, json_grpo.num_iterations) catch |err| {
                std.log.err("Failed to finalize embedded RL job {s}: {}", .{ job.job_id, err });
            };
        }
    }

    fn runQueuedJob(self: *Self, job: EmbeddedQueuedJob, json_grpo: experiment_config.GRPOJsonConfig) !void {
        const exp_config = self.config_result.?.config;
        const state = &self.state.?;

        try state.resetForJob(
            job.run_id,
            exp_config.model_path,
            false,
            job.required_workers,
            job.submitted_at,
            .starting,
        );
        state.setRLProgress(0, json_grpo.num_iterations, 0, 0, 0.0);
        publishScopedServiceIfPresent(self.publisher);

        var controller = try rl_controller.RLController.initWithSharedBase(self.allocator, self.shared_worker_fabric);
        defer controller.deinit();
        controller.training_backend_type = self.training_backend;

        var algorithm_impl = grpo.GRPO.init(self.allocator, &controller, .{
            .num_iterations = json_grpo.num_iterations,
            .group_size = json_grpo.group_size,
            .learning_rate = json_grpo.learning_rate,
            .beta = json_grpo.beta,
            .required_workers = job.required_workers,
            .prompt_file = json_grpo.prompt_file,
            .num_prompts = json_grpo.num_prompts,
            .rollout_max_tokens = json_grpo.rollout_max_tokens,
            .weights_path = json_grpo.weights_path,
            .training_weights_path = json_grpo.training_weights_path,
            .generation_weights_path = json_grpo.generation_weights_path,
            .generation_weights_format = json_grpo.generation_weights_format,
            .adapter_state_path = json_grpo.adapter_state_path,
            .grpo_weight_mode = json_grpo.grpo_weight_mode,
            .generation_vmfb_path = json_grpo.generation_vmfb_path,
            .generation_mlir_path = json_grpo.generation_mlir_path,
            .training_mlir_path = json_grpo.training_mlir_path,
            .num_gen_data_inputs = json_grpo.num_gen_data_inputs,
            .weight_refresh_strategy = json_grpo.weight_refresh_strategy,
            .updated_weights_path = json_grpo.updated_weights_path,
            .trainable_parameter_indices = json_grpo.trainable_parameter_indices,
        });
        var algorithm = algorithm_impl.asTrainingAlgorithm();
        defer algorithm.deinit();

        algorithm_impl.setControlState(state);

        var run_ctx = RLRunContext{
            .algorithm = &algorithm,
            .controller = &controller,
            .worker_fabric = self.shared_worker_fabric,
            .state = state,
            .required_workers = job.required_workers,
            .reserved_worker_ids = job.reserved_worker_ids,
            .publisher = self.publisher,
        };
        try embeddedRLRunThread(&run_ctx);
    }

    fn finalizeCompletedJob(self: *Self, model_id: []const u8, total_iterations: usize) !void {
        const state = &self.state.?;
        const runtime = self.captureRuntimeSnapshot();
        const job_json = try state.renderJobJsonWithRuntime(self.allocator, runtime);
        defer self.allocator.free(job_json);

        const status = control_state.ControllerState.statusString(currentStateStatus(state));
        try self.jobs.completeCurrentJob(status, job_json);

        try state.resetToIdle(model_id, self.configured_workers);
        state.setRLProgress(0, total_iterations, 0, 0, 0.0);
        publishScopedServiceIfPresent(self.publisher);
    }

    fn captureRuntimeSnapshot(self: *Self) control_state.RuntimeSnapshot {
        if (self.publisher) |publisher| {
            return publisher.captureRuntimeSnapshot();
        }

        const workers_connected = self.shared_worker_fabric.getUsableWorkerCountForLeaseOwner(.rl);
        const workers_ready = self.shared_worker_fabric.getReadyWorkerCountForLeaseOwner(.rl);
        const workers_available = self.shared_worker_fabric.getAvailableWorkerCountForLeaseOwner(.rl);
        const policy = self.shared_worker_fabric.getSchedulingPolicy(.rl);
        return .{
            .workers_connected = workers_connected,
            .workers_ready = workers_ready,
            .workers_available = workers_available,
            .health_status = "starting",
            .ready = false,
            .worker_class = if (policy.worker_class == .any) null else policy.worker_class.asString(),
            .target_arch = policy.target_arch,
            .reserved_workers = if (policy.reserved_workers > 0) policy.reserved_workers else null,
            .max_workers = policy.max_workers,
        };
    }

    fn renderCurrentEnvelopeLocked(self: *Self, allocator: Allocator, job: EmbeddedQueuedJob) !gateway.LocalJobEnvelope {
        const job_json = try self.state.?.renderJobJsonWithRuntime(allocator, self.captureRuntimeSnapshot());
        defer allocator.free(job_json);
        return try makeLocalJobEnvelope(allocator, self.service_id.?, self.executor_id.?, service_registry.ServiceType.rl.asString(), job.job_id, job_json);
    }

    fn renderQueuedEnvelopeLocked(
        self: *Self,
        allocator: Allocator,
        job: EmbeddedQueuedJob,
        status: control_state.JobStatus,
        finished_at: ?i64,
        cancel_requested: bool,
        last_error: ?[]const u8,
    ) !gateway.LocalJobEnvelope {
        const job_json = try renderSyntheticJobJson(
            allocator,
            .rl,
            status,
            self.config_path,
            job.run_id,
            self.config_result.?.config.model_path,
            false,
            job.submitted_at,
            0,
            finished_at,
            job.required_workers,
            self.captureRuntimeSnapshot(),
            cancel_requested,
            last_error,
        );
        defer allocator.free(job_json);
        return try makeLocalJobEnvelope(allocator, self.service_id.?, self.executor_id.?, service_registry.ServiceType.rl.asString(), job.job_id, job_json);
    }

    fn submitJobHook(ctx: *anyopaque, allocator: Allocator, request_body: []const u8) anyerror!gateway.LocalJobSubmitResult {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (self.reservations.hasActiveReservations()) return error.ExecutorBusy;
        try validateTemplateOnlyJobSubmitBody(self.allocator, request_body);
        const submission = try self.jobs.enqueue(self.executor_id.?, self.configured_workers, false);
        defer {
            var owned = submission.job;
            owned.deinit(self.allocator);
        }

        return .{
            .envelope = try self.renderQueuedEnvelopeLocked(allocator, submission.job, .queued, null, false, null),
            .accepted = true,
            .queue_position = submission.queue_position,
        };
    }

    fn reserveJobHook(
        ctx: *anyopaque,
        allocator: Allocator,
        request: gateway.LocalReservationRequest,
    ) anyerror!gateway.LocalReservationResult {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (self.jobs.hasPendingWork() or self.reservations.hasActiveReservations()) {
            return error.ExecutorBusy;
        }

        const workers_required = try resolveReservationWorkersRequired(request.workers_required, self.configured_workers);
        const leased_workers = self.shared_worker_fabric.acquireIdleWorkers(self.shared_worker_fabric.allocator, .rl, workers_required) catch |err| switch (err) {
            error.NotEnoughIdleWorkers => return error.ReservationUnavailable,
            else => return err,
        };
        errdefer {
            self.shared_worker_fabric.releaseWorkers(leased_workers);
            self.shared_worker_fabric.allocator.free(leased_workers);
        }
        defer self.shared_worker_fabric.allocator.free(leased_workers);

        const reservation = try self.reservations.create(self.executor_id.?, workers_required, leased_workers);
        defer {
            var owned = reservation;
            owned.deinit(self.allocator);
        }
        publishScopedServiceIfPresent(self.publisher);

        return try makeLocalReservationResult(
            allocator,
            self.service_id.?,
            self.executor_id.?,
            "rl",
            reservation,
        );
    }

    fn commitReservationHook(
        ctx: *anyopaque,
        allocator: Allocator,
        reservation_id: []const u8,
        request_body: []const u8,
    ) anyerror!?gateway.LocalJobSubmitResult {
        const self: *Self = @ptrCast(@alignCast(ctx));
        try validateTemplateOnlyJobSubmitBody(self.allocator, request_body);

        var reservation = self.reservations.take(reservation_id) orelse return null;
        defer reservation.deinit(self.allocator);

        const leased_workers = try self.shared_worker_fabric.filterLeasedWorkerIds(self.allocator, .rl, reservation.worker_ids);
        defer self.allocator.free(leased_workers);
        if (leased_workers.len != reservation.workers_required) {
            self.shared_worker_fabric.releaseWorkers(reservation.worker_ids);
            publishScopedServiceIfPresent(self.publisher);
            return error.ReservationUnavailable;
        }

        const submission = self.jobs.enqueueReserved(
            self.executor_id.?,
            reservation.workers_required,
            false,
            reservation.worker_ids,
        ) catch |err| {
            self.shared_worker_fabric.releaseWorkers(reservation.worker_ids);
            publishScopedServiceIfPresent(self.publisher);
            return err;
        };
        defer {
            var owned = submission.job;
            owned.deinit(self.allocator);
        }

        return .{
            .envelope = try self.renderQueuedEnvelopeLocked(allocator, submission.job, .queued, null, false, null),
            .accepted = true,
            .queue_position = submission.queue_position,
        };
    }

    fn releaseReservationHook(
        ctx: *anyopaque,
        allocator: Allocator,
        reservation_id: []const u8,
    ) anyerror!?gateway.LocalReservationReleaseResult {
        const self: *Self = @ptrCast(@alignCast(ctx));
        var reservation = self.reservations.take(reservation_id) orelse return null;
        defer reservation.deinit(self.allocator);

        self.shared_worker_fabric.releaseWorkers(reservation.worker_ids);
        publishScopedServiceIfPresent(self.publisher);

        return try makeLocalReservationReleaseResult(
            allocator,
            self.service_id.?,
            self.executor_id.?,
            "rl",
            reservation.reservation_id,
            "released",
        );
    }

    fn listJobsHook(ctx: *anyopaque, allocator: Allocator) anyerror![]gateway.LocalJobEnvelope {
        const self: *Self = @ptrCast(@alignCast(ctx));
        var envelopes = std.ArrayList(gateway.LocalJobEnvelope).init(allocator);
        errdefer {
            for (envelopes.items) |*envelope| envelope.deinit();
            envelopes.deinit();
        }

        self.jobs.mutex.lock();
        defer self.jobs.mutex.unlock();

        if (self.jobs.current_job) |job| {
            try envelopes.append(try self.renderCurrentEnvelopeLocked(allocator, job));
        }
        for (self.jobs.queued_jobs.items) |job| {
            try envelopes.append(try self.renderQueuedEnvelopeLocked(allocator, job, .queued, null, false, null));
        }

        var idx = self.jobs.history.items.len;
        while (idx > 0) {
            idx -= 1;
            const entry = self.jobs.history.items[idx];
            try envelopes.append(try makeLocalJobEnvelope(allocator, self.service_id.?, self.executor_id.?, service_registry.ServiceType.rl.asString(), entry.job_id, entry.job_json));
        }

        return envelopes.toOwnedSlice();
    }

    fn lookupJobHook(ctx: *anyopaque, allocator: Allocator, job_id: []const u8) anyerror!?gateway.LocalJobEnvelope {
        const self: *Self = @ptrCast(@alignCast(ctx));

        self.jobs.mutex.lock();
        defer self.jobs.mutex.unlock();

        if (self.jobs.current_job) |job| {
            if (std.mem.eql(u8, job.job_id, job_id)) {
                return try self.renderCurrentEnvelopeLocked(allocator, job);
            }
        }
        for (self.jobs.queued_jobs.items) |job| {
            if (std.mem.eql(u8, job.job_id, job_id)) {
                return try self.renderQueuedEnvelopeLocked(allocator, job, .queued, null, false, null);
            }
        }
        for (self.jobs.history.items) |entry| {
            if (std.mem.eql(u8, entry.job_id, job_id)) {
                return try makeLocalJobEnvelope(allocator, self.service_id.?, self.executor_id.?, service_registry.ServiceType.rl.asString(), entry.job_id, entry.job_json);
            }
        }

        return null;
    }

    fn cancelJobHook(ctx: *anyopaque, allocator: Allocator, job_id: []const u8) anyerror!?gateway.LocalJobCancelResult {
        const self: *Self = @ptrCast(@alignCast(ctx));

        if (self.jobs.hasCurrentJob(job_id)) {
            self.state.?.requestCancel();
            return .{
                .allocator = allocator,
                .accepted = true,
                .status = try allocator.dupe(u8, "cancelling"),
                .job_id = try allocator.dupe(u8, job_id),
                .service_id = try allocator.dupe(u8, self.service_id.?),
                .executor_id = try allocator.dupe(u8, self.executor_id.?),
                .service_type = try allocator.dupe(u8, "rl"),
            };
        }

        if (self.jobs.removeQueuedJob(job_id)) |job| {
            const job_json = try renderSyntheticJobJson(
                self.allocator,
                .rl,
                .cancelled,
                self.config_path,
                job.run_id,
                self.config_result.?.config.model_path,
                false,
                job.submitted_at,
                0,
                std.time.timestamp(),
                job.required_workers,
                self.captureRuntimeSnapshot(),
                false,
                null,
            );
            defer self.allocator.free(job_json);
            try self.jobs.archiveQueuedJob(job, "cancelled", job_json);

            return .{
                .allocator = allocator,
                .accepted = true,
                .status = try allocator.dupe(u8, "cancelled"),
                .job_id = try allocator.dupe(u8, job_id),
                .service_id = try allocator.dupe(u8, self.service_id.?),
                .executor_id = try allocator.dupe(u8, self.executor_id.?),
                .service_type = try allocator.dupe(u8, "rl"),
            };
        }

        self.jobs.mutex.lock();
        defer self.jobs.mutex.unlock();
        for (self.jobs.history.items) |entry| {
            if (std.mem.eql(u8, entry.job_id, job_id)) {
                return .{
                    .allocator = allocator,
                    .accepted = false,
                    .status = try allocator.dupe(u8, entry.status),
                    .job_id = try allocator.dupe(u8, job_id),
                    .service_id = try allocator.dupe(u8, self.service_id.?),
                    .executor_id = try allocator.dupe(u8, self.executor_id.?),
                    .service_type = try allocator.dupe(u8, "rl"),
                };
            }
        }

        return null;
    }
};
