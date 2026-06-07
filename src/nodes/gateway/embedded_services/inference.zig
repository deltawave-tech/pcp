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

pub const EmbeddedInferenceService = struct {
    allocator: Allocator,
    gateway_instance: *gateway.Gateway,
    shared_worker_fabric: *WorkerFabricController,
    api_token: ?[]const u8,
    gateway_api_port: u16,
    config_result: ?inference_config.ConfigResult = null,
    state: ?control_state.ControllerState = null,
    controller: ?inference_controller.InferenceController = null,
    publisher: ?*ScopedServicePublisher = null,
    executor_id: ?[]u8 = null,

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
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.controller) |*controller| {
            controller.setGatewayPublishHook(null);
        }
        if (self.state) |*state| {
            state.setChangeHook(null);
        }
        if (self.publisher) |publisher| {
            publisher.deinit();
        }
        if (self.executor_id) |value| self.allocator.free(value);
        if (self.controller) |*controller| {
            controller.requestShutdown();
            controller.deinit();
        }
        if (self.state) |*state| {
            state.deinit();
        }
        if (self.config_result) |*result| {
            result.deinit();
        }
    }

    pub fn start(self: *Self, embedded_cfg: config.EmbeddedInferenceControllerConfig) !void {
        const shared_api_token = self.api_token orelse return error.MissingApiToken;

        self.config_result = try inference_config.loadInferenceConfig(self.allocator, embedded_cfg.config_path);
        var inference_cfg = self.config_result.?.config;
        if (self.gateway_instance.config.resolvedApiTokenEnv()) |token_env| {
            inference_cfg.api_token_env = token_env;
        }

        try applySchedulingPolicy(
            self.shared_worker_fabric,
            .inference,
            embedded_cfg.worker_class,
            embedded_cfg.target_arch,
            embedded_cfg.reserved_workers,
            embedded_cfg.max_workers,
            inference_cfg.worker_backend,
            normalizeTargetArch(inference_cfg.worker_target_arch),
        );

        self.state = try control_state.ControllerState.init(
            self.allocator,
            .inference,
            .inference,
            embedded_cfg.config_path,
            null,
            inference_cfg.model_id,
            false,
            1,
        );
        self.state.?.setStatus(.running);

        self.controller = try inference_controller.InferenceController.initWithSharedBase(
            self.allocator,
            self.shared_worker_fabric,
            inference_cfg,
            shared_api_token,
            &self.state.?,
            null,
        );

        const api_host = if (embedded_cfg.api) |api_cfg|
            api_cfg.host orelse "127.0.0.1"
        else
            "127.0.0.1";
        const api_port = if (embedded_cfg.api) |api_cfg|
            api_cfg.port orelse self.gateway_api_port + 1
        else
            self.gateway_api_port + 1;
        const service_id = embedded_cfg.service_id orelse "inference-main";
        self.executor_id = try identity.deriveExecutorId(
            self.allocator,
            self.gateway_instance.config.gateway_id,
            "inference",
            service_id,
        );
        const base_url = try std.fmt.allocPrint(self.allocator, "http://{s}:{d}", .{ api_host, api_port });
        defer self.allocator.free(base_url);

        self.publisher = try ScopedServicePublisher.init(
            self.allocator,
            &self.gateway_instance.service_registry,
            self.shared_worker_fabric,
            &self.state.?,
            service_id,
            self.executor_id.?,
            "inference",
            base_url,
            &[_][]const u8{
                message_registry.capability_inference,
                "chat.completions",
                "models.list",
                "controller.status",
            },
            .inference,
            self.controller.?.snapshotProvider(),
            false,
        );
        try self.publisher.?.bind();
        self.controller.?.setGatewayPublishHook(self.publisher.?.publishHook());

        try self.controller.?.attachHooks();
        try self.controller.?.startMaintenance();
        const api_thread = try std.Thread.spawn(.{}, inferenceApiThread, .{ &self.controller.?, api_host, api_port });
        api_thread.detach();
    }
};
