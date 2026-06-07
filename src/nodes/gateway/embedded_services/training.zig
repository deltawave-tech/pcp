const common = @import("../embedded_common.zig");
const decoupled_session = @import("../decoupled_training_session.zig");
const decoupled_job = @import("../../../protocol/decoupled_job.zig");
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
const training_workload = common.training_workload;
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
const makeRegularTrainingWindowValue = common.makeRegularTrainingWindowValue;
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

pub const EmbeddedTrainingService = struct {
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
    default_resume_requested: bool = false,
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
        self.gateway_instance.setLocalJobHook(.training, null);
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

    pub fn start(self: *Self, embedded_cfg: config.EmbeddedTrainingControllerConfig, default_workers: usize) !void {
        self.config_result = try experiment_config.loadConfig(self.allocator, embedded_cfg.config_path);
        self.config_path = try self.allocator.dupe(u8, embedded_cfg.config_path);
        const exp_config = self.config_result.?.config;
        const required_workers = embedded_cfg.workers orelse default_workers;
        self.configured_workers = required_workers;
        self.default_resume_requested = embedded_cfg.should_resume;
        const backend = backend_selection.Backend.selectDefault();

        try applySchedulingPolicy(
            self.shared_worker_fabric,
            .training,
            embedded_cfg.worker_class,
            embedded_cfg.target_arch,
            embedded_cfg.reserved_workers,
            embedded_cfg.max_workers,
            backend.toString(),
            null,
        );

        self.state = try control_state.ControllerState.init(
            self.allocator,
            .training,
            .training,
            embedded_cfg.config_path,
            null,
            exp_config.model_path,
            false,
            required_workers,
        );
        self.state.?.setTrainingProgress(0, 0, exp_config.outer_loop_steps, 0.0, exp_config.learning_rate);

        if (self.shared_worker_fabric.getExecutor() == null) {
            self.shared_worker_fabric.setExecutor(try backend_selection.createExecutor(self.allocator, backend));
        }

        const api_host = if (embedded_cfg.api) |api_cfg|
            api_cfg.host orelse "127.0.0.1"
        else
            "127.0.0.1";
        const api_port = if (embedded_cfg.api) |api_cfg|
            api_cfg.port orelse self.gateway_api_port + 1
        else
            self.gateway_api_port + 1;
        const service_id = embedded_cfg.service_id orelse "training-main";
        self.service_id = try self.allocator.dupe(u8, service_id);
        self.executor_id = try identity.deriveExecutorId(
            self.allocator,
            self.gateway_instance.config.gateway_id,
            "training",
            service_id,
        );
        const base_url = try std.fmt.allocPrint(self.allocator, "http://{s}:{d}", .{ api_host, api_port });
        defer self.allocator.free(base_url);

        self.gateway_instance.setLocalJobHook(.training, .{
            .service_id = self.service_id.?,
            .executor_id = self.executor_id.?,
            .service_type = .training,
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
            "training",
            base_url,
            &[_][]const u8{
                message_registry.capability_regular_training,
                message_registry.capability_decoupled_training,
                message_registry.capability_streaming_diloco,
                message_registry.capability_artifact_transfer,
                training_workload.capability_regular,
                training_workload.capability_decoupled_diloco,
                "controller.status",
                "job.current",
                "job.cancel",
            },
            .training,
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
            .training,
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
        while (try self.jobs.waitForNextJob()) |job| {
            defer {
                var owned = job;
                owned.deinit(self.allocator);
            }

            self.runQueuedJob(job) catch |err| {
                if (self.state) |*state| {
                    if (!stateIsTerminal(state)) {
                        state.setFailed(@errorName(err)) catch |set_err| {
                            std.log.err("Failed to mark embedded training job {s} as failed: {}", .{ job.job_id, set_err });
                        };
                    }
                }
                std.log.err("Embedded training job {s} failed: {}", .{ job.job_id, err });
            };

            self.finalizeCompletedJob(exp_config.model_path, exp_config.outer_loop_steps, exp_config.learning_rate) catch |err| {
                std.log.err("Failed to finalize embedded training job {s}: {}", .{ job.job_id, err });
            };
        }
    }

    fn runQueuedJob(self: *Self, job: EmbeddedQueuedJob) !void {
        const exp_config = self.config_result.?.config;
        const state = &self.state.?;
        try state.resetForJob(
            job.run_id,
            exp_config.model_path,
            job.resume_requested,
            job.required_workers,
            job.submitted_at,
            .starting,
        );
        state.setTrainingProgress(0, 0, exp_config.outer_loop_steps, 0.0, exp_config.learning_rate);
        publishScopedServiceIfPresent(self.publisher);

        const leased_workers = resolveEmbeddedJobWorkers(
            self.shared_worker_fabric.allocator,
            self.shared_worker_fabric,
            state,
            .training,
            job.required_workers,
            job.reserved_worker_ids,
        ) catch |err| {
            if (err == error.Cancelled) {
                state.setCancelled();
                return;
            }
            try state.setFailed(@errorName(err));
            return err;
        };
        defer self.shared_worker_fabric.allocator.free(leased_workers);
        defer {
            self.shared_worker_fabric.releaseWorkers(leased_workers);
            publishScopedServiceIfPresent(self.publisher);
        }

        state.setStatus(.initializing);
        try state.setReadinessPhase("initializing_training_backend", exp_config.model_path);
        self.shared_worker_fabric.clearTransientJobState();

        const file = try std.fs.cwd().openFile(exp_config.data_path, .{});
        defer file.close();
        const stat = try file.stat();
        try self.shared_worker_fabric.resetDataManager(stat.size, 64 * 1024 * 1024, exp_config.max_epochs);

        if (self.shared_worker_fabric.getExecutor() == null) {
            const backend = backend_selection.Backend.selectDefault();
            self.shared_worker_fabric.setExecutor(try backend_selection.createExecutor(self.allocator, backend));
        }
        const shared_executor = self.shared_worker_fabric.getExecutor() orelse return error.ExecutorNotInitialized;
        var mlir_builder = try MLIRBuilder.init(self.allocator, shared_executor.getContext());
        defer mlir_builder.deinit();

        if (isRegularDecoupledDilocoConfig(exp_config)) {
            try self.runRegularDecoupledDilocoJob(job, exp_config, leased_workers, &mlir_builder);
            if (state.isCancellationRequested()) {
                state.setCancelled();
            } else {
                state.setStatus(.completed);
            }
            return;
        }

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
        diloco_config.resume_training = job.resume_requested;
        if (exp_config.checkpoint_dir) |checkpoint_dir| {
            diloco_config.checkpoint_dir = checkpoint_dir;
        }
        if (std.mem.eql(u8, exp_config.dtype, "bf16")) {
            diloco_config.dtype = .bf16;
        } else if (std.mem.eql(u8, exp_config.dtype, "f16")) {
            diloco_config.dtype = .f16;
        } else if (std.mem.eql(u8, exp_config.dtype, "f32")) {
            diloco_config.dtype = .f32;
        } else {
            return error.InvalidDType;
        }
        if (exp_config.effective_batch_size) |effective_batch_size| {
            diloco_config.effective_batch_size = effective_batch_size;
        }
        diloco_config.use_in_graph_accumulation = exp_config.use_in_graph_accumulation;

        var algorithm_impl = try DiLoCo.init(
            self.allocator,
            self.shared_worker_fabric,
            diloco_config,
            shared_executor,
            &mlir_builder,
        );
        var algorithm = algorithm_impl.asTrainingAlgorithm();
        defer algorithm.deinit();

        algorithm_impl.setControlState(state);
        try algorithm_impl.setWorkerScope(leased_workers);
        defer algorithm_impl.clearWorkerScope();
        publishScopedServiceIfPresent(self.publisher);

        algorithm.run() catch |err| {
            if (err == error.Cancelled) {
                state.setCancelled();
                return;
            }
            try state.setFailed(@errorName(err));
            return err;
        };

        if (state.isCancellationRequested()) {
            state.setCancelled();
        } else {
            state.setStatus(.completed);
        }
    }

    fn runRegularDecoupledDilocoJob(
        self: *Self,
        job: EmbeddedQueuedJob,
        exp_config: experiment_config.ExperimentConfig,
        leased_workers: []const NodeId,
        mlir_builder: *MLIRBuilder,
    ) !void {
        if (leased_workers.len == 0) return error.NotEnoughWorkers;
        _ = try decoupled_job.DecoupledTrainingJob.fromRegularConfig(exp_config, .{
            .run_id = job.run_id,
            .workers = leased_workers.len,
            .backend = backend_selection.Backend.selectDefault().toString(),
        });
        const state = &self.state.?;

        state.setStatus(.initializing);
        try state.setReadinessPhase("initializing_regular_decoupled_diloco", exp_config.model_path);
        publishScopedServiceIfPresent(self.publisher);

        var runtime = try buildRegularDecoupledRuntime(self.allocator, exp_config, mlir_builder);
        defer runtime.deinit();

        try self.shared_worker_fabric.ensureWorkersCompiledForIds(
            leased_workers,
            runtime.worker_graph_mlir_source,
            runtime.parameter_shapes,
            runtime.data_input_shapes,
        );

        var tensor_specs = try decoupled_diloco.tensorSpecsFromShapes(self.allocator, runtime.parameter_shapes, @sizeOf(f32));
        defer self.allocator.free(tensor_specs);
        for (exp_config.embedding_tensor_indices) |tensor_index| {
            if (tensor_index < tensor_specs.len) tensor_specs[tensor_index].role = .embedding;
        }

        var syncer = try decoupled_diloco.DecoupledDiLoCo.initWithMaster(
            self.allocator,
            try makeRegularDecoupledConfig(exp_config),
            tensor_specs,
            runtime.master_params,
        );
        defer syncer.deinit();

        const master_online = try flattenF32Matrix(self.allocator, runtime.master_params);
        defer self.allocator.free(master_online);

        const request_id = self.shared_worker_fabric.allocateRequestId();
        const total_syncer_steps = exp_config.outer_loop_steps;
        const sync_interval_h = exp_config.sync_interval_h orelse exp_config.tau;
        const micro_batch = resolveRegularMicroBatch(runtime.data_input_shapes);
        const max_local_steps = total_syncer_steps * @max(sync_interval_h, @as(usize, 1)) * 4 + sync_interval_h;
        const data_file = try std.fs.cwd().openFile(exp_config.data_path, .{});
        defer data_file.close();
        const data_file_size = (try data_file.stat()).size;

        var started_workers = std.ArrayList(NodeId).init(self.allocator);
        defer started_workers.deinit();
        defer {
            decoupled_session.stopWorkerLoops(
                self.allocator,
                self.shared_worker_fabric,
                request_id,
                started_workers.items,
                "regular Decoupled DiLoCo",
            );
        }

        try self.shared_worker_fabric.broadcastNamedLargeDataToWorkerIdsWithContext(
            leased_workers,
            master_online,
            .{ .request_id = request_id },
            training_window.blob_online_weights,
        );

        for (leased_workers, 0..) |worker_id, worker_index| {
            const chunk = if (self.shared_worker_fabric.data_manager) |*dm|
                dm.assignNextChunk(worker_id)
            else
                null;
            const assignment_offset = if (chunk) |assigned| assigned.offset else 0;
            const assignment_length = if (chunk) |assigned| assigned.length else data_file_size;
            const assignment_chunk_id = if (chunk) |assigned| assigned.id else worker_index;
            if (chunk == null) std.log.warn("No data chunk available for regular Decoupled DiLoCo worker {}; using full local data range", .{worker_id});

            const assignment = data_assignment.DataAssignment{
                .kind = data_assignment.kind_byte_range,
                .provider = data_assignment.provider_local_path,
                .path = exp_config.data_path,
                .offset = assignment_offset,
                .length = assignment_length,
                .chunk_id = assignment_chunk_id,
            };
            var window_json = try makeRegularTrainingWindowValue(
                self.allocator,
                exp_config.model_path,
                exp_config.checkpoint_dir orelse "checkpoints",
                master_online,
                assignment,
                .{
                    .num_fragments = exp_config.num_fragments,
                    .sync_interval_h = sync_interval_h,
                    .overlap_tau = exp_config.overlap_tau,
                    .learner_alpha = exp_config.learner_alpha,
                    .max_syncer_steps = total_syncer_steps,
                    .max_local_steps = max_local_steps,
                },
                .{
                    .tokenizer = exp_config.tokenizer,
                    .sampling = exp_config.sampling,
                    .dtype = "f32",
                    .micro_batch = micro_batch,
                },
            );
            defer window_json.deinit();

            var payload = std.json.ObjectMap.init(self.allocator);
            defer payload.deinit();
            try payload.put(training_window.field_name, window_json.value);

            try self.shared_worker_fabric.sendToWorkerWithContext(
                worker_id,
                message.MessageType.START_DECOUPLED_DILOCO_LOOP,
                .{ .object = payload },
                .{
                    .request_id = request_id,
                    .task_id = @intCast(worker_index + 1),
                },
            );
            try started_workers.append(worker_id);
        }

        if (started_workers.items.len < exp_config.min_quorum) return error.QuorumNotMet;

        state.setStatus(.running);
        state.setTrainingProgress(0, 0, total_syncer_steps, 0.0, exp_config.learning_rate);
        try state.setReadinessPhase("running_regular_decoupled_diloco", exp_config.model_path);
        publishScopedServiceIfPresent(self.publisher);

        var decoupled_stats = DecoupledRuntimeStats{};
        var session = decoupled_session.DecoupledTrainingSession{
            .allocator = self.allocator,
            .worker_fabric = self.shared_worker_fabric,
            .syncer = &syncer,
            .request_id = request_id,
            .worker_ids = started_workers.items,
            .min_quorum = exp_config.min_quorum,
            .parameter_shapes = runtime.parameter_shapes,
            .master_weights = master_online,
        };

        const CancellationContext = struct {
            state: *control_state.ControllerState,

            fn isCancelled(ctx: *anyopaque) bool {
                const progress: *@This() = @ptrCast(@alignCast(ctx));
                return progress.state.isCancellationRequested();
            }
        };
        const ProgressContext = struct {
            state: *control_state.ControllerState,
            publisher: ?*ScopedServicePublisher,
            total_syncer_steps: usize,
            learning_rate: f32,
            model_path: []const u8,

            fn onStep(ctx: *anyopaque, update: decoupled_session.ScheduledStepUpdate) !void {
                const progress: *@This() = @ptrCast(@alignCast(ctx));
                const final_loss = update.final_loss orelse 0.0;
                progress.state.setTrainingProgress(0, update.completed_syncer_steps, progress.total_syncer_steps, final_loss, progress.learning_rate);
                try progress.state.setReadinessPhase("running_regular_decoupled_diloco", progress.model_path);
                publishScopedServiceIfPresent(progress.publisher);

                std.log.info("Regular Decoupled DiLoCo syncer step {} fragment {} participants {} version {}", .{
                    update.step.syncer_step,
                    update.step.fragment_id,
                    update.step.participant_count,
                    update.step.version,
                });
            }
        };
        var cancellation = CancellationContext{ .state = state };
        var progress = ProgressContext{
            .state = state,
            .publisher = self.publisher,
            .total_syncer_steps = total_syncer_steps,
            .learning_rate = exp_config.learning_rate,
            .model_path = exp_config.model_path,
        };
        const run_result = try session.runScheduledSteps(.{
            .total_syncer_steps = total_syncer_steps,
            .stats = &decoupled_stats,
            .fragment_options = .{
                .collect_response_metrics = true,
            },
            .cancellation = .{
                .ctx = &cancellation,
                .is_cancelled = CancellationContext.isCancelled,
            },
            .step_hook = .{
                .ctx = &progress,
                .on_step = ProgressContext.onStep,
            },
        });

        const output_dir = exp_config.checkpoint_dir orelse "checkpoints";
        try writeRegularDecoupledDilocoOutputs(
            self.allocator,
            exp_config,
            output_dir,
            master_online,
            &syncer,
            started_workers.items.len,
            run_result.completed_syncer_steps,
            run_result.train_examples,
            run_result.final_loss orelse 0.0,
            job.run_id,
            decoupled_stats,
        );
        try state.setReadinessPhase("training_output_dir", output_dir);
    }

    fn finalizeCompletedJob(
        self: *Self,
        model_id: []const u8,
        total_steps: usize,
        learning_rate: f32,
    ) !void {
        const state = &self.state.?;
        const runtime = self.captureRuntimeSnapshot();
        const job_json = try state.renderJobJsonWithRuntime(self.allocator, runtime);
        defer self.allocator.free(job_json);

        const status = control_state.ControllerState.statusString(currentStateStatus(state));
        try self.jobs.completeCurrentJob(status, job_json);

        try state.resetToIdle(model_id, self.configured_workers);
        state.setTrainingProgress(0, 0, total_steps, 0.0, learning_rate);
        publishScopedServiceIfPresent(self.publisher);
    }

    fn captureRuntimeSnapshot(self: *Self) control_state.RuntimeSnapshot {
        if (self.publisher) |publisher| {
            return publisher.captureRuntimeSnapshot();
        }

        const workers_connected = self.shared_worker_fabric.getUsableWorkerCountForLeaseOwner(.training);
        const workers_ready = self.shared_worker_fabric.getReadyWorkerCountForLeaseOwner(.training);
        const workers_available = self.shared_worker_fabric.getAvailableWorkerCountForLeaseOwner(.training);
        const policy = self.shared_worker_fabric.getSchedulingPolicy(.training);
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
        return try makeLocalJobEnvelope(allocator, self.service_id.?, self.executor_id.?, service_registry.ServiceType.training.asString(), job.job_id, job_json);
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
            .training,
            status,
            self.config_path,
            job.run_id,
            self.config_result.?.config.model_path,
            job.resume_requested,
            job.submitted_at,
            0,
            finished_at,
            job.required_workers,
            self.captureRuntimeSnapshot(),
            cancel_requested,
            last_error,
        );
        defer allocator.free(job_json);
        return try makeLocalJobEnvelope(allocator, self.service_id.?, self.executor_id.?, service_registry.ServiceType.training.asString(), job.job_id, job_json);
    }

    fn submitJobHook(ctx: *anyopaque, allocator: Allocator, request_body: []const u8) anyerror!gateway.LocalJobSubmitResult {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (self.reservations.hasActiveReservations()) return error.ExecutorBusy;
        try validateTemplateOnlyJobSubmitBody(self.allocator, request_body);
        const submission = try self.jobs.enqueue(self.executor_id.?, self.configured_workers, self.default_resume_requested);
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
        const leased_workers = self.shared_worker_fabric.acquireIdleWorkers(self.shared_worker_fabric.allocator, .training, workers_required) catch |err| switch (err) {
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
            "training",
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

        const leased_workers = try self.shared_worker_fabric.filterLeasedWorkerIds(self.allocator, .training, reservation.worker_ids);
        defer self.allocator.free(leased_workers);
        if (leased_workers.len != reservation.workers_required) {
            self.shared_worker_fabric.releaseWorkers(reservation.worker_ids);
            publishScopedServiceIfPresent(self.publisher);
            return error.ReservationUnavailable;
        }

        const submission = self.jobs.enqueueReserved(
            self.executor_id.?,
            reservation.workers_required,
            self.default_resume_requested,
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
            "training",
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
            try envelopes.append(try makeLocalJobEnvelope(allocator, self.service_id.?, self.executor_id.?, service_registry.ServiceType.training.asString(), entry.job_id, entry.job_json));
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
                return try makeLocalJobEnvelope(allocator, self.service_id.?, self.executor_id.?, service_registry.ServiceType.training.asString(), entry.job_id, entry.job_json);
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
                .service_type = try allocator.dupe(u8, "training"),
            };
        }

        if (self.jobs.removeQueuedJob(job_id)) |job| {
            const job_json = try renderSyntheticJobJson(
                self.allocator,
                .training,
                .cancelled,
                self.config_path,
                job.run_id,
                self.config_result.?.config.model_path,
                job.resume_requested,
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
                .service_type = try allocator.dupe(u8, "training"),
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
                    .service_type = try allocator.dupe(u8, "training"),
                };
            }
        }

        return null;
    }
};
