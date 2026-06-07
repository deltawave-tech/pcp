const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;

const WorkerFabricController = @import("training_controller.zig").WorkerFabricController;
const WorkerMessageHook = @import("training_controller.zig").WorkerMessageHook;
const WorkerConnectionHook = @import("training_controller.zig").WorkerConnectionHook;
const WorkerStatus = @import("training_controller.zig").WorkerStatus;
const inference_config = @import("../../../workloads/inference/config.zig");
const inference_types = @import("../../../workloads/inference/types.zig");
const model_registry = @import("../../../workloads/inference/model_registry.zig");
const session_manager = @import("../../../workloads/inference/session_manager.zig");
const router = @import("../../../workloads/inference/router.zig");
const tokenizer_mod = @import("../../../workloads/inference/tokenizer.zig");
const backend_selection = @import("../../../backends/selection.zig");
const http_server = @import("../../../network/http_server.zig");
const control_api = @import("../control_plane/api.zig");
const control_state = @import("../control_plane/state.zig");
const gateway_service_client = @import("../service_client.zig");
const scheduling = @import("../scheduling.zig");
const tcp_stream = @import("../../../network/tcp_stream.zig");
const message = @import("../../../network/message.zig");
const model_introspection = @import("../../../mlir/model_introspection.zig");
const tensor = @import("../../../core/tensor.zig");

const ArrayList = std.ArrayList;
const MessageType = message.MessageType;
const MessageContext = message.MessageContext;
const NodeId = message.NodeId;
const RequestId = message.RequestId;
const TcpServer = tcp_stream.TcpServer;
const Backend = backend_selection.Backend;
const WorkerClass = scheduling.WorkerClass;

const WorkerSet = struct {
    allocator: Allocator,
    set: std.AutoHashMap(NodeId, void),

    fn init(allocator: Allocator) WorkerSet {
        return .{ .allocator = allocator, .set = std.AutoHashMap(NodeId, void).init(allocator) };
    }

    fn deinit(self: *WorkerSet) void {
        self.set.deinit();
    }

    fn add(self: *WorkerSet, id: NodeId) !void {
        try self.set.put(id, {});
    }

    fn remove(self: *WorkerSet, id: NodeId) void {
        _ = self.set.remove(id);
    }

    fn snapshot(self: *WorkerSet) ![]NodeId {
        var ids = try self.allocator.alloc(NodeId, self.set.count());
        var it = self.set.keyIterator();
        var idx: usize = 0;
        while (it.next()) |id| : (idx += 1) {
            ids[idx] = id.*;
        }
        return ids;
    }
};

const RequestState = struct {
    allocator: Allocator,
    request_id: RequestId,
    session_id: []const u8,
    model_id: []const u8,
    worker_id: ?NodeId,
    stream: bool,
    stream_conn: ?net.Stream,
    response: ArrayList(u8),
    created_at: i64,
    deadline_at: i64,
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_token_ids: ?[]i64,
    max_new_tokens: usize,
    temperature: f32,
    first_token_at: ?i64,
    finish_reason: ?[]const u8,
    error_message: ?[]const u8,
    done: bool,
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,

    fn init(allocator: Allocator, request_id: RequestId, session_id: []const u8, model_id: []const u8, stream: bool, deadline_at: i64) !*RequestState {
        const state = try allocator.create(RequestState);
        state.* = .{
            .allocator = allocator,
            .request_id = request_id,
            .session_id = try allocator.dupe(u8, session_id),
            .model_id = try allocator.dupe(u8, model_id),
            .worker_id = null,
            .stream = stream,
            .stream_conn = null,
            .response = ArrayList(u8).init(allocator),
            .created_at = std.time.timestamp(),
            .deadline_at = deadline_at,
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .prompt_token_ids = null,
            .max_new_tokens = 0,
            .temperature = 0.7,
            .first_token_at = null,
            .finish_reason = null,
            .error_message = null,
            .done = false,
            .mutex = std.Thread.Mutex{},
            .cond = std.Thread.Condition{},
        };
        return state;
    }

    fn deinit(self: *RequestState) void {
        self.allocator.free(self.session_id);
        self.allocator.free(self.model_id);
        if (self.finish_reason) |reason| self.allocator.free(reason);
        if (self.error_message) |err| self.allocator.free(err);
        if (self.prompt_token_ids) |tokens| self.allocator.free(tokens);
        self.response.deinit();
        self.allocator.destroy(self);
    }
};

pub const Metrics = struct {
    total_requests: u64 = 0,
    active_requests: u64 = 0,
    queued_requests: u64 = 0,
    completed_requests: u64 = 0,
    failed_requests: u64 = 0,
    tokens_generated: u64 = 0,
    prompt_tokens: u64 = 0,
    ttft_total_ms: u64 = 0,
    ttft_count: u64 = 0,
    cache_hits: u64 = 0,
    cache_misses: u64 = 0,
};

pub const InferenceController = struct {
    owned_base: ?*WorkerFabricController,
    base: *WorkerFabricController,
    config: inference_config.InferenceConfig,
    api_token: []const u8,
    tokenizer: tokenizer_mod.Tokenizer,
    models: model_registry.ModelRegistry,
    sessions: session_manager.SessionManager,
    router: router.Router,
    ready_by_model: std.StringHashMap(WorkerSet),
    active: std.AutoHashMap(RequestId, *RequestState),
    pending: ArrayList(RequestId),
    state_mutex: std.Thread.Mutex,
    metrics: Metrics,
    metrics_mutex: std.Thread.Mutex,
    operator_state: ?*control_state.ControllerState,
    gateway_client: ?*gateway_service_client.GatewayClient,
    gateway_publish_hook: ?control_state.ChangeHook,
    api_server: ?TcpServer,
    api_host: ?[]const u8,
    api_port: u16,
    api_running: std.atomic.Value(u8),
    shutdown_requested: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        cfg: inference_config.InferenceConfig,
        api_token: []const u8,
        operator_state: ?*control_state.ControllerState,
        gateway_client: ?*gateway_service_client.GatewayClient,
    ) !Self {
        const owned_base = try allocator.create(WorkerFabricController);
        errdefer allocator.destroy(owned_base);
        owned_base.* = WorkerFabricController.init(allocator);
        return initWithBase(allocator, owned_base, owned_base, cfg, api_token, operator_state, gateway_client);
    }

    pub fn initWithSharedBase(
        allocator: Allocator,
        base: *WorkerFabricController,
        cfg: inference_config.InferenceConfig,
        api_token: []const u8,
        operator_state: ?*control_state.ControllerState,
        gateway_client: ?*gateway_service_client.GatewayClient,
    ) !Self {
        return initWithBase(allocator, null, base, cfg, api_token, operator_state, gateway_client);
    }

    fn initWithBase(
        allocator: Allocator,
        owned_base: ?*WorkerFabricController,
        base: *WorkerFabricController,
        cfg: inference_config.InferenceConfig,
        api_token: []const u8,
        operator_state: ?*control_state.ControllerState,
        gateway_client: ?*gateway_service_client.GatewayClient,
    ) !Self {
        try validateOptionalFullModelFeatures(cfg);
        const tok_kind = parseTokenizerKind(cfg.tokenizer_source) catch |err| return err;
        var tokenizer = try tokenizer_mod.Tokenizer.init(allocator, tok_kind, cfg.tokenizer_path);
        errdefer tokenizer.deinit();

        var controller = Self{
            .owned_base = owned_base,
            .base = base,
            .config = try cloneConfig(allocator, cfg),
            .api_token = try allocator.dupe(u8, api_token),
            .tokenizer = tokenizer,
            .models = model_registry.ModelRegistry.init(allocator),
            .sessions = session_manager.SessionManager.init(allocator),
            .router = router.Router.init(),
            .ready_by_model = std.StringHashMap(WorkerSet).init(allocator),
            .active = std.AutoHashMap(RequestId, *RequestState).init(allocator),
            .pending = ArrayList(RequestId).init(allocator),
            .state_mutex = std.Thread.Mutex{},
            .metrics = Metrics{},
            .metrics_mutex = std.Thread.Mutex{},
            .operator_state = operator_state,
            .gateway_client = gateway_client,
            .gateway_publish_hook = null,
            .api_server = null,
            .api_host = null,
            .api_port = 0,
            .api_running = std.atomic.Value(u8).init(0),
            .shutdown_requested = std.atomic.Value(u8).init(0),
        };
        errdefer {
            controller.allocatorFreeConfig();
            allocator.free(controller.api_token);
            controller.tokenizer.deinit();
            if (owned_base) |owned| {
                owned.deinit();
                allocator.destroy(owned);
            }
        }

        try controller.resolveEosTokenFromTokenizer();
        try controller.loadModelsFromConfig();
        return controller;
    }

    pub fn deinit(self: *Self) void {
        self.tokenizer.deinit();
        self.allocatorFreeConfig();
        self.models.deinit();
        self.sessions.deinit();

        var it = self.ready_by_model.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.ready_by_model.deinit();

        var active_it = self.active.valueIterator();
        while (active_it.next()) |state| {
            state.*.deinit();
        }
        self.active.deinit();
        self.pending.deinit();

        self.base.allocator.free(self.api_token);
        if (self.owned_base) |owned| {
            owned.deinit();
            self.base.allocator.destroy(owned);
        }
    }

    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        try self.base.listen(host, port);
    }

    pub fn getReadyWorkerCount(self: *Self) usize {
        var ready_workers: ?[]NodeId = null;
        defer if (ready_workers) |ids| self.base.allocator.free(ids);

        self.state_mutex.lock();
        if (self.ready_by_model.getPtr(self.config.model_id)) |set| {
            ready_workers = set.snapshot() catch {
                self.state_mutex.unlock();
                return 0;
            };
        }
        self.state_mutex.unlock();

        if (ready_workers) |ids| {
            var workers_snapshot = self.base.snapshotWorkersByIds(ids) catch return 0;
            defer workers_snapshot.deinit();

            const policy = self.base.getSchedulingPolicy(.inference);
            var compatible_count: usize = 0;
            for (workers_snapshot.items) |worker| {
                if (policy.allowsWorker(workerClassForBackend(worker.backend), worker.target_arch)) {
                    compatible_count += 1;
                }
            }
            return compatible_count;
        }

        return 0;
    }

    pub fn getIdleReadyWorkerCount(self: *Self) usize {
        var ready_workers: ?[]NodeId = null;
        defer if (ready_workers) |ids| self.base.allocator.free(ids);

        self.state_mutex.lock();
        if (self.ready_by_model.getPtr(self.config.model_id)) |set| {
            ready_workers = set.snapshot() catch {
                self.state_mutex.unlock();
                return 0;
            };
        }
        self.state_mutex.unlock();

        if (ready_workers) |ids| {
            const available_workers = self.base.filterAcquirableWorkerIds(self.base.allocator, .inference, ids) catch return 0;
            defer self.base.allocator.free(available_workers);
            return available_workers.len;
        }
        return 0;
    }

    pub fn captureRuntimeSnapshot(self: *Self) !control_state.RuntimeSnapshot {
        const ready_workers = self.getReadyWorkerCount();
        const available_workers = self.getIdleReadyWorkerCount();
        const policy = self.base.getSchedulingPolicy(.inference);

        return .{
            .workers_connected = ready_workers,
            .workers_ready = ready_workers,
            .workers_available = available_workers,
            .health_status = if (ready_workers > 0) "ok" else "starting",
            .ready = ready_workers > 0,
            .worker_class = if (policy.worker_class == .any) null else policy.worker_class.asString(),
            .target_arch = policy.target_arch,
            .reserved_workers = if (policy.reserved_workers > 0) policy.reserved_workers else null,
            .max_workers = policy.max_workers,
        };
    }

    pub fn snapshotMetrics(self: *Self) Metrics {
        self.metrics_mutex.lock();
        defer self.metrics_mutex.unlock();
        return self.metrics;
    }

    pub fn attachHooks(self: *Self) !void {
        try self.base.addMessageHook(.{ .ctx = self, .handler = handleWorkerMessageHook });
        try self.base.addConnectionHook(.{ .ctx = self, .on_connect = handleWorkerConnectHook, .on_disconnect = handleWorkerDisconnectHook });
    }

    pub fn snapshotProvider(self: *Self) control_api.SnapshotProvider {
        return .{
            .ctx = self,
            .capture = captureInferenceRuntimeSnapshot,
        };
    }

    pub fn setGatewayPublishHook(self: *Self, hook: ?control_state.ChangeHook) void {
        self.gateway_publish_hook = hook;
    }

    pub fn startApi(self: *Self, host: []const u8, port: u16) !void {
        self.api_host = host;
        self.api_port = port;
        self.api_server = try TcpServer.init(self.base.allocator, host, port);
        self.api_running.store(1, .release);
        std.log.info("Inference API listening on {s}:{}", .{ host, port });

        while (self.api_running.load(.acquire) == 1) {
            const connection = if (self.api_server) |*server|
                server.accept() catch |err| {
                    if (self.api_running.load(.acquire) == 0) break;
                    std.log.err("Inference API accept failed: {}", .{err});
                    continue;
                }
            else
                break;

            if (self.api_running.load(.acquire) == 0) {
                connection.stream.close();
                break;
            }

            const thread = try std.Thread.spawn(.{}, handleApiConnection, .{ self, connection.stream });
            thread.detach();
        }

        if (self.api_server) |*server| {
            server.deinit();
            self.api_server = null;
        }
    }

    pub fn startMaintenance(self: *Self) !void {
        const thread = try std.Thread.spawn(.{}, maintenanceLoop, .{self});
        thread.detach();
    }

    pub fn requestShutdown(self: *Self) void {
        self.shutdown_requested.store(1, .release);
        if (self.owned_base != null) {
            self.base.stop();
        }
        self.stopApi();
    }

    pub fn isShutdownRequested(self: *Self) bool {
        return self.shutdown_requested.load(.acquire) == 1;
    }

    fn stopApi(self: *Self) void {
        self.api_running.store(0, .release);
        if (self.api_host) |host| {
            const address = net.Address.parseIp(host, self.api_port) catch return;
            const stream = net.tcpConnectToAddress(address) catch return;
            stream.close();
        }
    }

    fn allocatorFreeConfig(self: *Self) void {
        const alloc = self.base.allocator;
        alloc.free(self.config.model_id);
        alloc.free(self.config.pool_name);
        alloc.free(self.config.generation_vmfb_path);
        alloc.free(self.config.generation_mlir_path);
        alloc.free(self.config.weights_path);
        alloc.free(self.config.weights_format);
        alloc.free(self.config.tokenizer_source);
        if (self.config.tokenizer_path) |p| alloc.free(p);
        alloc.free(self.config.chat_template);
        alloc.free(self.config.worker_backend);
        alloc.free(self.config.worker_target_arch);
        alloc.free(self.config.api_token_env);
    }

    fn cloneConfig(allocator: Allocator, cfg: inference_config.InferenceConfig) !inference_config.InferenceConfig {
        return .{
            .model_id = try allocator.dupe(u8, cfg.model_id),
            .pool_name = try allocator.dupe(u8, cfg.pool_name),
            .generation_vmfb_path = try allocator.dupe(u8, cfg.generation_vmfb_path),
            .generation_mlir_path = try allocator.dupe(u8, cfg.generation_mlir_path),
            .weights_path = try allocator.dupe(u8, cfg.weights_path),
            .weights_format = try allocator.dupe(u8, cfg.weights_format),
            .tokenizer_source = try allocator.dupe(u8, cfg.tokenizer_source),
            .tokenizer_path = if (cfg.tokenizer_path) |p| try allocator.dupe(u8, p) else null,
            .chat_template = try allocator.dupe(u8, cfg.chat_template),
            .vision_enabled = cfg.vision_enabled,
            .mtp_enabled = cfg.mtp_enabled,
            .full_conditional_generation_enabled = cfg.full_conditional_generation_enabled,
            .num_gen_data_inputs = cfg.num_gen_data_inputs,
            .max_context_tokens = cfg.max_context_tokens,
            .default_max_output_tokens = cfg.default_max_output_tokens,
            .eos_token = cfg.eos_token,
            .session_ttl_seconds = cfg.session_ttl_seconds,
            .request_timeout_seconds = cfg.request_timeout_seconds,
            .worker_backend = try allocator.dupe(u8, cfg.worker_backend),
            .worker_target_arch = try allocator.dupe(u8, cfg.worker_target_arch),
            .api_token_env = try allocator.dupe(u8, cfg.api_token_env),
        };
    }

    fn parseTokenizerKind(source: []const u8) !tokenizer_mod.TokenizerKind {
        if (std.mem.eql(u8, source, "byte")) return .byte;
        if (std.mem.eql(u8, source, "char")) return .char;
        if (std.mem.eql(u8, source, "u16")) return .u16;
        if (std.mem.eql(u8, source, "qwen")) return .qwen;
        return error.UnknownTokenizerSource;
    }

    fn validateOptionalFullModelFeatures(cfg: inference_config.InferenceConfig) !void {
        if (cfg.vision_enabled) return error.VisionInferenceUnsupported;
        if (cfg.mtp_enabled) return error.MtpInferenceUnsupported;
        if (cfg.full_conditional_generation_enabled) return error.FullConditionalGenerationUnsupported;
        if (!std.mem.eql(u8, cfg.chat_template, "qwen_text_only") and
            !std.mem.eql(u8, cfg.chat_template, "legacy"))
        {
            return error.UnsupportedChatTemplate;
        }
    }

    fn resolveEosTokenFromTokenizer(self: *Self) !void {
        if (self.config.eos_token != null) return;
        self.config.eos_token = (try self.tokenizer.eosTokenId()) orelse 0;
    }

    fn loadModelsFromConfig(self: *Self) !void {
        const meta_path = try std.fmt.allocPrint(self.base.allocator, "{s}.meta.json", .{self.config.generation_mlir_path});
        defer self.base.allocator.free(meta_path);

        var meta_result: model_introspection.ModelMetadata = undefined;
        var used_cache = false;

        if (std.fs.cwd().access(meta_path, .{})) |_| {
            if (model_introspection.ModelMetadata.loadFromFile(self.base.allocator, meta_path)) |meta| {
                meta_result = meta;
                used_cache = true;
            } else |_| {}
        } else |_| {}

        if (!used_cache) {
            const mlir_source = try std.fs.cwd().readFileAlloc(
                self.base.allocator,
                self.config.generation_mlir_path,
                5 * 1024 * 1024 * 1024,
            );
            defer self.base.allocator.free(mlir_source);
            const meta = try model_introspection.ModelInspector.inspectLite(
                self.base.allocator,
                mlir_source,
                self.config.num_gen_data_inputs orelse return error.MissingGenerationDataInputMetadata,
            );
            meta_result = meta;
            try meta_result.saveToFile(meta_path);
        }

        const tokenizer_path = self.config.tokenizer_path orelse "";
        const record = inference_types.ModelRecord{
            .model_id = self.config.model_id,
            .generation_vmfb_path = self.config.generation_vmfb_path,
            .generation_mlir_path = self.config.generation_mlir_path,
            .weights_path = self.config.weights_path,
            .weights_format = self.config.weights_format,
            .tokenizer_path = tokenizer_path,
            .max_context_tokens = self.config.max_context_tokens,
            .default_max_output_tokens = self.config.default_max_output_tokens,
            .pool_name = self.config.pool_name,
        };
        try self.models.addModel(record);
        try self.ready_by_model.put(self.config.model_id, WorkerSet.init(self.base.allocator));

        meta_result.deinit();
    }

    fn handleWorkerConnectHook(ctx: *anyopaque, worker_id: NodeId) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (!try self.workerCompatibleWithInferencePolicy(worker_id)) {
            const policy = self.base.getSchedulingPolicy(.inference);
            std.log.info(
                "Skipping inference model load for worker {} because it does not satisfy policy class={s} target={s}",
                .{
                    worker_id,
                    policy.worker_class.asString(),
                    policy.target_arch orelse "any",
                },
            );
            return;
        }
        try self.dispatchLoadModel(worker_id, self.config.model_id);
    }

    fn handleWorkerDisconnectHook(ctx: *anyopaque, worker_id: NodeId) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.state_mutex.lock();

        var it = self.ready_by_model.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.remove(worker_id);
        }

        // Mark active requests on this worker as failed.
        var active_it = self.active.iterator();
        while (active_it.next()) |entry| {
            if (entry.value_ptr.*.worker_id == worker_id) {
                self.failRequestLocked(entry.value_ptr.*, "worker_disconnected");
            }
        }
        self.state_mutex.unlock();

        self.publishGatewayServiceNoFail();
    }

    fn handleWorkerMessageHook(ctx: *anyopaque, worker_id: NodeId, msg: message.MessageEnvelope) anyerror!bool {
        const self: *Self = @ptrCast(@alignCast(ctx));

        if (std.mem.eql(u8, msg.msg_type, MessageType.MODEL_READY)) {
            const payload = msg.data.object;
            const model_id = payload.get("model_id").?.string;
            self.base.setWorkerStatus(worker_id, WorkerStatus.GraphInitialized);
            self.state_mutex.lock();
            if (self.ready_by_model.getPtr(model_id)) |set| {
                try set.add(worker_id);
            }
            self.state_mutex.unlock();
            self.dispatchPending(model_id);
            self.publishGatewayServiceNoFail();
            return true;
        }

        if (std.mem.eql(u8, msg.msg_type, MessageType.GENERATION_CHUNK)) {
            const payload = msg.data.object;
            const tokens_val = payload.get("tokens") orelse return true;
            const tokens_array = tokens_val.array;
            var tokens = ArrayList(i64).init(self.base.allocator);
            defer tokens.deinit();
            for (tokens_array.items) |item| {
                try tokens.append(item.integer);
            }
            self.handleGenerationChunk(worker_id, msg.request_id, tokens.items) catch |err| {
                std.log.err("Failed to process generation chunk: {}", .{err});
            };
            return true;
        }

        if (std.mem.eql(u8, msg.msg_type, MessageType.GENERATION_COMPLETE)) {
            const payload = msg.data.object;
            const finish_reason_val = payload.get("finish_reason") orelse std.json.Value{ .string = "stop" };
            const finish_reason = switch (finish_reason_val) {
                .string => |s| s,
                else => "stop",
            };
            const prompt_tokens = payload.get("prompt_tokens") orelse std.json.Value{ .integer = 0 };
            const completion_tokens = payload.get("completion_tokens") orelse std.json.Value{ .integer = 0 };
            self.handleGenerationComplete(
                msg.request_id,
                finish_reason,
                @intCast(prompt_tokens.integer),
                @intCast(completion_tokens.integer),
            );
            return true;
        }

        if (std.mem.eql(u8, msg.msg_type, MessageType.GENERATION_ERROR)) {
            const payload = msg.data.object;
            const err_msg = payload.get("error") orelse std.json.Value{ .string = "generation_error" };
            self.handleGenerationError(msg.request_id, err_msg.string);
            return true;
        }

        return false;
    }

    fn dispatchLoadModel(self: *Self, worker_id: NodeId, model_id: []const u8) !void {
        const record = self.models.get(model_id) orelse return error.ModelNotFound;
        self.base.setWorkerStatus(worker_id, WorkerStatus.InitializingGraph);

        const meta_path = try std.fmt.allocPrint(self.base.allocator, "{s}.meta.json", .{record.generation_mlir_path});
        defer self.base.allocator.free(meta_path);
        var meta = try model_introspection.ModelMetadata.loadFromFile(self.base.allocator, meta_path);
        defer meta.deinit();

        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();
        try payload.put("model_id", .{ .string = record.model_id });
        try payload.put("vmfb_path", .{ .string = record.generation_vmfb_path });
        try payload.put("weights_path", .{ .string = record.weights_path });
        try payload.put("weights_format", .{ .string = record.weights_format });
        try payload.put("max_context_tokens", .{ .integer = @intCast(record.max_context_tokens) });

        try payload.put("parameter_shapes", try shapesToJson(self.base.allocator, meta.parameter_shapes));
        if (meta.parameter_dtypes) |param_dtypes| {
            try payload.put("parameter_dtypes", try dtypesToJson(self.base.allocator, param_dtypes));
        }
        try payload.put("data_input_shapes", try shapesToJson(self.base.allocator, meta.data_input_shapes));
        try payload.put("data_input_dtypes", try dtypesToJson(self.base.allocator, meta.data_input_dtypes));

        try self.base.sendToWorker(worker_id, MessageType.LOAD_MODEL, .{ .object = payload });
    }

    fn workerCompatibleWithInferencePolicy(self: *Self, worker_id: NodeId) !bool {
        const worker_ids = [_]NodeId{worker_id};
        var workers = try self.base.snapshotWorkersByIds(&worker_ids);
        defer workers.deinit();
        if (workers.items.len == 0) return error.WorkerNotFound;

        const worker = workers.items[0];
        const policy = self.base.getSchedulingPolicy(.inference);
        return policy.allowsWorker(workerClassForBackend(worker.backend), worker.target_arch);
    }

    fn shapesToJson(allocator: Allocator, shapes: [][]i64) !std.json.Value {
        var outer = std.json.Array.init(allocator);
        for (shapes) |shape| {
            var inner = std.json.Array.init(allocator);
            for (shape) |dim| {
                try inner.append(.{ .integer = dim });
            }
            try outer.append(.{ .array = inner });
        }
        return .{ .array = outer };
    }

    fn dtypesToJson(allocator: Allocator, dtypes: []const tensor.DType) !std.json.Value {
        var arr = std.json.Array.init(allocator);
        for (dtypes) |dt| {
            const s = switch (dt) {
                .f32 => "f32",
                .f64 => "f64",
                .bf16 => "bf16",
                .f16 => "f16",
                .i32 => "i32",
                .i64 => "i64",
                .bool => "bool",
            };
            try arr.append(.{ .string = s });
        }
        return .{ .array = arr };
    }

    fn handleApiConnection(self: *Self, stream: net.Stream) void {
        defer stream.close();

        var request = http_server.readRequest(stream, self.base.allocator, 10 * 1024 * 1024) catch |err| {
            std.log.err("API request read failed: {}", .{err});
            return;
        };
        defer request.deinit();

        self.routeApiRequest(stream, &request) catch |err| {
            std.log.err("API request failed: {}", .{err});
            _ = http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch |write_err| {
                std.log.warn("Inference API failed to write error response: {}", .{write_err});
            };
        };
    }

    fn maintenanceLoop(self: *Self) void {
        while (!self.isShutdownRequested()) {
            std.time.sleep(1 * std.time.ns_per_s);
            const now = std.time.timestamp();

            self.sessions.purgeExpired(now);

            self.state_mutex.lock();
            var it = self.active.iterator();
            while (it.next()) |entry| {
                const state = entry.value_ptr.*;
                if (!state.done and state.deadline_at <= now) {
                    if (state.worker_id) |worker_id| {
                        self.cancelRequest(worker_id, state.request_id, "timeout");
                    }
                    self.failRequestLocked(state, "timeout");
                }
            }
            self.state_mutex.unlock();

            self.publishGatewayServiceNoFail();
        }
    }

    fn routeApiRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !void {
        if (std.mem.eql(u8, req.path, "/healthz")) {
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain"}, "ok");
            return;
        }

        if (std.mem.eql(u8, req.path, "/readyz")) {
            const ready = self.isReady();
            const status = if (ready) "200 OK" else "503 Service Unavailable";
            const body = try self.renderReadyzJson();
            defer self.base.allocator.free(body);
            try http_server.writeResponse(stream, status, &.{"Content-Type: application/json"}, body);
            return;
        }

        if (std.mem.eql(u8, req.path, "/v1/models")) {
            const body = try self.renderModelsJson();
            defer self.base.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return;
        }

        if (self.operator_state) |state| {
            var operator_api = control_api.ControlApiServer.init(
                self.base.allocator,
                state,
                self.base,
                self.api_token,
                .{
                    .ctx = self,
                    .capture = captureInferenceRuntimeSnapshot,
                },
                .{
                    .ctx = self,
                    .render = renderOperatorReady,
                },
                .{
                    .ctx = self,
                    .render = renderOperatorMetrics,
                },
                null,
                .{
                    .ctx = self,
                    .cancel = cancelInference,
                },
            );
            if (try operator_api.handleRequest(stream, req)) {
                return;
            }
        }

        if (std.mem.eql(u8, req.path, "/v1/chat/completions")) {
            if (!self.authorize(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return;
            }
            std.log.info("Inference API received /v1/chat/completions request (body_bytes={})", .{req.body.len});
            try self.handleChatCompletions(stream, req);
            return;
        }

        try http_server.writeResponse(stream, "404 Not Found", &.{"Content-Type: text/plain"}, "not_found");
    }

    fn authorize(self: *Self, req: *http_server.HttpRequest) bool {
        const header = req.header("authorization") orelse return false;
        const prefix = "Bearer ";
        if (!std.mem.startsWith(u8, header, prefix)) return false;
        const token = header[prefix.len..];
        return std.mem.eql(u8, token, self.api_token);
    }

    fn isReady(self: *Self) bool {
        return self.getReadyWorkerCount() > 0;
    }

    fn renderModelsJson(self: *Self) ![]u8 {
        var buf = ArrayList(u8).init(self.base.allocator);
        errdefer buf.deinit();

        try buf.appendSlice("{\"object\":\"list\",\"data\":[");
        var first = true;
        var it = self.models.models.iterator();
        while (it.next()) |entry| {
            if (!first) try buf.appendSlice(",");
            first = false;
            try buf.appendSlice("{\"id\":\"");
            if (std.mem.startsWith(u8, entry.key_ptr.*, "pcp/")) {
                try buf.appendSlice(entry.key_ptr.*);
            } else {
                try buf.appendSlice("pcp/");
                try buf.appendSlice(entry.key_ptr.*);
            }
            try buf.appendSlice("\",\"object\":\"model\"}");
        }
        try buf.appendSlice("]}");

        return buf.toOwnedSlice();
    }

    fn renderReadyzJson(self: *Self) ![]u8 {
        const runtime = try self.captureRuntimeSnapshot();

        self.metrics_mutex.lock();
        const metrics = self.metrics;
        self.metrics_mutex.unlock();

        return std.json.stringifyAlloc(self.base.allocator, .{
            .ready = runtime.ready,
            .health_status = runtime.health_status,
            .ready_workers = runtime.workers_ready,
            .available_workers = runtime.workers_available,
            .worker_class = runtime.worker_class,
            .target_arch = runtime.target_arch,
            .reserved_workers = runtime.reserved_workers,
            .max_workers = runtime.max_workers,
            .metrics = toControlPlaneMetrics(metrics),
        }, .{});
    }

    fn renderOperatorReady(ctx: *anyopaque, allocator: Allocator, state: *control_state.ControllerState, connected_workers: usize) !control_api.ReadyResult {
        _ = connected_workers;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const runtime = try self.captureRuntimeSnapshot();
        const ready = runtime.ready;
        const body = try state.renderReadyJson(allocator, runtime);
        if (state.mode == .inference) {
            state.setInferenceMetrics(toControlPlaneMetrics(self.snapshotMetrics()));
        }
        return .{ .ready = ready, .body = body };
    }

    fn renderOperatorMetrics(ctx: *anyopaque, allocator: Allocator, state: *control_state.ControllerState, connected_workers: usize) ![]u8 {
        _ = connected_workers;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const metrics = self.snapshotMetrics();
        state.setInferenceMetrics(toControlPlaneMetrics(metrics));
        return state.renderMetricsJsonWithRuntime(allocator, try self.captureRuntimeSnapshot());
    }

    fn cancelInference(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.requestShutdown();
    }

    fn toControlPlaneMetrics(metrics: Metrics) control_state.InferenceMetrics {
        return .{
            .total_requests = metrics.total_requests,
            .active_requests = metrics.active_requests,
            .queued_requests = metrics.queued_requests,
            .completed_requests = metrics.completed_requests,
            .failed_requests = metrics.failed_requests,
            .tokens_generated = metrics.tokens_generated,
            .prompt_tokens = metrics.prompt_tokens,
            .ttft_total_ms = metrics.ttft_total_ms,
            .ttft_count = metrics.ttft_count,
            .cache_hits = metrics.cache_hits,
            .cache_misses = metrics.cache_misses,
        };
    }

    fn handleChatCompletions(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.base.allocator, req.body, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        const body = parsed.value;
        const model_val = body.object.get("model");
        const model_id = if (model_val) |val| switch (val) {
            .string => |s| normalizeModelId(s),
            else => self.config.model_id,
        } else self.config.model_id;
        if (!self.models.contains(model_id)) {
            try http_server.writeResponse(stream, "404 Not Found", &.{"Content-Type: text/plain"}, "model_not_found");
            return;
        }

        const stream_val = body.object.get("stream") orelse std.json.Value{ .bool = false };
        const use_stream = stream_val.bool;

        const max_tokens_val = body.object.get("max_tokens") orelse std.json.Value{ .integer = @intCast(self.config.default_max_output_tokens) };
        var max_tokens = @as(usize, @intCast(max_tokens_val.integer));
        const temperature_val = body.object.get("temperature") orelse std.json.Value{ .float = 0.7 };
        const temperature = switch (temperature_val) {
            .float => |f| @as(f32, @floatCast(f)),
            .integer => |i| @as(f32, @floatFromInt(i)),
            else => 0.7,
        };

        const session_id_val = body.object.get("session_id");
        const session_id = if (session_id_val) |val| val.string else null;

        const messages_val = body.object.get("messages") orelse return error.MissingMessages;
        const messages = switch (messages_val) {
            .array => |arr| arr,
            else => return error.InvalidMessages,
        };

        const prompt_text = self.tokenizer.renderChat(self.config.chat_template, messages, self.config.vision_enabled) catch |err| {
            if (isChatRequestError(err)) {
                const error_text = @errorName(err);
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, error_text);
                return;
            }
            return err;
        };
        defer self.base.allocator.free(prompt_text);
        const prompt_tokens = try self.tokenizer.encode(prompt_text);
        defer self.base.allocator.free(prompt_tokens);

        if (prompt_tokens.len > self.config.max_context_tokens) {
            try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "context_length_exceeded");
            return;
        }
        const available = self.config.max_context_tokens - prompt_tokens.len;
        if (available == 0) {
            try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "context_length_exceeded");
            return;
        }
        if (max_tokens > available) {
            max_tokens = available;
        }

        const request_id = self.base.allocateRequestId();
        const deadline = std.time.timestamp() + @as(i64, @intCast(self.config.request_timeout_seconds));

        var owned_session_id: ?[]const u8 = null;
        const new_session_id = session_id orelse blk: {
            const generated = try self.makeSessionId(request_id);
            owned_session_id = generated;
            break :blk generated;
        };
        defer if (owned_session_id) |s| self.base.allocator.free(s);
        var request_state = try RequestState.init(self.base.allocator, request_id, new_session_id, model_id, use_stream, deadline);
        errdefer request_state.deinit();

        request_state.prompt_token_ids = try self.base.allocator.alloc(i64, prompt_tokens.len);
        @memcpy(request_state.prompt_token_ids.?, prompt_tokens);
        request_state.max_new_tokens = max_tokens;
        request_state.temperature = temperature;

        self.metrics_mutex.lock();
        self.metrics.total_requests += 1;
        self.metrics.active_requests += 1;
        self.metrics_mutex.unlock();

        if (use_stream) {
            try stream.writeAll("HTTP/1.1 200 OK\r\n");
            try stream.writeAll("Content-Type: text/event-stream\r\n");
            try stream.writeAll("Cache-Control: no-cache\r\n");
            try stream.writeAll("Connection: keep-alive\r\n\r\n");
            request_state.stream_conn = stream;
        }

        const dispatch = try self.dispatchRequest(request_state);
        if (!dispatch) {
            try self.enqueueRequest(request_state);
        }

        request_state.mutex.lock();
        while (!request_state.done) {
            request_state.cond.wait(&request_state.mutex);
        }
        request_state.mutex.unlock();

        var released_worker: ?NodeId = null;
        self.state_mutex.lock();
        released_worker = request_state.worker_id;
        if (released_worker) |worker_id| {
            self.base.releaseWorker(worker_id);
            request_state.worker_id = null;
        }
        _ = self.active.remove(request_id);
        self.state_mutex.unlock();
        if (released_worker != null) {
            self.dispatchPending(request_state.model_id);
        }

        self.metrics_mutex.lock();
        self.metrics.active_requests -= 1;
        if (request_state.error_message != null) {
            self.metrics.failed_requests += 1;
        } else {
            self.metrics.completed_requests += 1;
        }
        self.metrics_mutex.unlock();

        if (!use_stream) {
            if (request_state.error_message) |err_msg| {
                std.log.warn("Inference request {} completed with error: {s}", .{ request_id, err_msg });
                http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, err_msg) catch |err| {
                    std.log.err("Failed to write inference error response for request {}: {}", .{ request_id, err });
                    return err;
                };
            } else {
                const body_json = try self.renderCompletionResponse(request_state);
                defer self.base.allocator.free(body_json);
                http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body_json) catch |err| {
                    std.log.err("Failed to write inference success response for request {}: {}", .{ request_id, err });
                    return err;
                };
                self.emitGatewayCompletionEvent(request_state, body_json) catch |err| {
                    std.log.warn("Failed to emit inference completion gateway event: {}", .{err});
                };
            }
        } else if (request_state.error_message == null) {
            self.emitGatewayCompletionEvent(request_state, null) catch |err| {
                std.log.warn("Failed to emit streamed inference completion gateway event: {}", .{err});
            };
        }

        if (released_worker != null) self.publishGatewayServiceNoFail();

        request_state.deinit();
    }

    pub fn registerGatewayService(self: *Self, host: []const u8, port: u16) !void {
        self.api_host = host;
        self.api_port = port;
        try self.refreshGatewayServiceRegistration();
    }

    fn refreshGatewayServiceRegistration(self: *Self) !void {
        const runtime = try self.captureRuntimeSnapshot();
        const job_status = if (self.operator_state) |state| blk: {
            state.mutex.lock();
            const status = state.status;
            state.mutex.unlock();
            break :blk control_state.ControllerState.statusString(status);
        } else "running";
        const client = self.gateway_client orelse return;
        const host = self.api_host orelse return;
        const port = self.api_port;
        const base_url = try std.fmt.allocPrint(self.base.allocator, "http://{s}:{d}", .{ host, port });
        defer self.base.allocator.free(base_url);
        try client.registerService(
            "inference",
            base_url,
            &[_][]const u8{ "chat.completions", "models.list", "controller.status" },
            runtime.workers_connected,
            runtime.workers_ready,
            runtime.workers_available,
            runtime.worker_class,
            runtime.target_arch,
            runtime.reserved_workers,
            runtime.max_workers,
            job_status,
            runtime.health_status,
        );
    }

    fn publishGatewayServiceNoFail(self: *Self) void {
        if (self.gateway_publish_hook) |hook| {
            hook.on_change(hook.ctx);
            return;
        }

        self.refreshGatewayServiceRegistration() catch |err| {
            std.log.warn("Failed to refresh gateway inference service registration: {}", .{err});
        };
    }

    fn captureInferenceRuntimeSnapshot(
        ctx: *anyopaque,
        _: *control_state.ControllerState,
        _: *WorkerFabricController,
        _: ?@import("training_controller.zig").WorkerLeaseOwner,
    ) anyerror!control_state.RuntimeSnapshot {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return try self.captureRuntimeSnapshot();
    }

    fn dispatchRequest(self: *Self, state: *RequestState) !bool {
        self.state_mutex.lock();
        const dispatched = try self.dispatchRequestLocked(state);
        self.state_mutex.unlock();

        if (dispatched) self.publishGatewayServiceNoFail();
        return dispatched;
    }

    fn dispatchRequestLocked(self: *Self, state: *RequestState) !bool {
        const ready_workers = if (self.ready_by_model.getPtr(state.model_id)) |set|
            try set.snapshot()
        else
            try self.base.allocator.alloc(NodeId, 0);
        defer self.base.allocator.free(ready_workers);
        const available_ready_workers = try self.base.filterAcquirableWorkerIds(self.base.allocator, .inference, ready_workers);
        defer self.base.allocator.free(available_ready_workers);

        const session_record = self.sessions.get(state.session_id);
        const decision = self.router.decide(session_record, available_ready_workers);
        if (decision.queued or decision.worker_id == null) {
            return false;
        }

        const worker_id = decision.worker_id.?;
        if (!self.base.tryAcquireWorker(worker_id, .inference)) {
            return false;
        }
        errdefer self.base.releaseWorker(worker_id);
        state.worker_id = worker_id;
        try self.active.put(state.request_id, state);

        const prompt_tokens = state.prompt_token_ids orelse return false;
        var reuse = decision.reuse_session and session_record != null;
        var prompt_start_pos: i64 = 0;
        var tokens_to_send = prompt_tokens;

        if (session_record) |session| {
            if (session.bound_worker) |old_worker| {
                if (old_worker != worker_id) {
                    self.flushSession(old_worker, state.session_id);
                }
            }
        }

        if (reuse) {
            if (session_record.?.last_prompt_tokens) |prev| {
                const lcp = longestCommonPrefix(prev, prompt_tokens);
                if (lcp == prev.len) {
                    prompt_start_pos = @intCast(lcp);
                    tokens_to_send = prompt_tokens[lcp..];
                    self.metrics_mutex.lock();
                    self.metrics.cache_hits += 1;
                    self.metrics_mutex.unlock();
                } else {
                    reuse = false;
                    prompt_start_pos = 0;
                    tokens_to_send = prompt_tokens;
                    self.metrics_mutex.lock();
                    self.metrics.cache_misses += 1;
                    self.metrics_mutex.unlock();
                    if (session_record.?.bound_worker) |old_worker| {
                        self.flushSession(old_worker, state.session_id);
                    }
                }
            }
        }

        const new_session = inference_types.SessionRecord{
            .session_id = state.session_id,
            .model_id = state.model_id,
            .bound_worker = worker_id,
            .next_round_id = 0,
            .last_prompt_hash = hashTokens(prompt_tokens),
            .last_prompt_tokens = prompt_tokens,
            .last_access_ts = std.time.timestamp(),
            .expires_at = std.time.timestamp() + @as(i64, @intCast(self.config.session_ttl_seconds)),
            .state = .active,
        };
        try self.sessions.upsert(new_session);

        state.prompt_tokens = prompt_tokens.len;
        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();
        try payload.put("model_id", .{ .string = state.model_id });
        try payload.put("session_id", .{ .string = state.session_id });
        try payload.put("max_new_tokens", .{ .integer = @intCast(state.max_new_tokens) });
        try payload.put("eos_token", .{ .integer = self.config.eos_token orelse 0 });
        try payload.put("temperature", .{ .float = state.temperature });
        try payload.put("reuse_session", .{ .bool = reuse });
        try payload.put("prompt_start_pos", .{ .integer = prompt_start_pos });

        var tokens_array = std.json.Array.init(self.base.allocator);
        defer tokens_array.deinit();
        for (tokens_to_send) |t| {
            try tokens_array.append(.{ .integer = t });
        }
        try payload.put("prompt_tokens", .{ .array = tokens_array });

        try self.base.sendToWorkerWithContext(worker_id, MessageType.START_GENERATION, .{ .object = payload }, MessageContext{
            .request_id = state.request_id,
        });
        return true;
    }

    fn enqueueRequest(self: *Self, state: *RequestState) !void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        try self.pending.append(state.request_id);
        errdefer _ = self.pending.pop();
        try self.active.put(state.request_id, state);

        self.metrics_mutex.lock();
        self.metrics.queued_requests += 1;
        self.metrics_mutex.unlock();
    }

    fn dispatchPending(self: *Self, model_id: []const u8) void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        var refreshed = false;
        var idx: usize = 0;
        while (idx < self.pending.items.len) {
            const request_id = self.pending.items[idx];
            const state = self.active.get(request_id) orelse {
                _ = self.pending.swapRemove(idx);
                continue;
            };
            if (!std.mem.eql(u8, state.model_id, model_id)) {
                idx += 1;
                continue;
            }

            if (self.dispatchRequestLocked(state) catch false) {
                _ = self.pending.swapRemove(idx);
                self.metrics_mutex.lock();
                if (self.metrics.queued_requests > 0) self.metrics.queued_requests -= 1;
                self.metrics_mutex.unlock();
                refreshed = true;
                continue;
            }
            idx += 1;
        }

        if (refreshed) {
            self.state_mutex.unlock();
            self.publishGatewayServiceNoFail();
            self.state_mutex.lock();
        }
    }

    fn handleGenerationChunk(self: *Self, worker_id: NodeId, request_id: RequestId, tokens: []const i64) !void {
        self.state_mutex.lock();
        const state = self.active.get(request_id) orelse {
            self.state_mutex.unlock();
            return;
        };
        self.state_mutex.unlock();

        if (state.first_token_at == null) {
            state.first_token_at = std.time.timestamp();
            const ttft_ms = @as(u64, @intCast((state.first_token_at.? - state.created_at) * 1000));
            self.metrics_mutex.lock();
            self.metrics.ttft_total_ms += ttft_ms;
            self.metrics.ttft_count += 1;
            self.metrics.prompt_tokens += state.prompt_tokens;
            self.metrics_mutex.unlock();
        }

        state.completion_tokens += tokens.len;
        self.metrics_mutex.lock();
        self.metrics.tokens_generated += tokens.len;
        self.metrics_mutex.unlock();

        const chunk_text = try self.tokenizer.decode(tokens);
        defer self.base.allocator.free(chunk_text);

        state.mutex.lock();
        defer state.mutex.unlock();

        if (state.stream) {
            if (state.stream_conn) |conn| {
                const json_payload = try self.renderStreamChunk(state, chunk_text, null);
                defer self.base.allocator.free(json_payload);
                writeSse(conn, json_payload) catch {
                    self.cancelRequest(worker_id, request_id, "client_disconnected");
                };
            }
        } else {
            try state.response.appendSlice(chunk_text);
        }
    }

    fn handleGenerationComplete(self: *Self, request_id: RequestId, finish_reason: []const u8, prompt_tokens: usize, completion_tokens: usize) void {
        std.log.info("Inference request {} generation complete (finish_reason={s}, prompt_tokens={}, completion_tokens={})", .{
            request_id,
            finish_reason,
            prompt_tokens,
            completion_tokens,
        });
        self.state_mutex.lock();
        const state = self.active.get(request_id) orelse {
            self.state_mutex.unlock();
            return;
        };
        self.state_mutex.unlock();

        state.mutex.lock();
        if (state.finish_reason) |reason| self.base.allocator.free(reason);
        state.finish_reason = self.base.allocator.dupe(u8, finish_reason) catch null;
        state.prompt_tokens = prompt_tokens;
        state.completion_tokens = completion_tokens;
        state.done = true;
        if (state.stream) {
            if (state.stream_conn) |conn| {
                const json_payload = self.renderStreamChunk(state, "", finish_reason) catch null;
                if (json_payload) |payload| {
                    _ = writeSse(conn, payload) catch |err| {
                        std.log.warn("Failed to write inference completion stream chunk for request {}: {}", .{ state.request_id, err });
                    };
                    self.base.allocator.free(payload);
                }
                _ = writeSse(conn, "[DONE]") catch |err| {
                    std.log.warn("Failed to write inference completion stream terminator for request {}: {}", .{ state.request_id, err });
                };
            }
        }
        state.cond.broadcast();
        state.mutex.unlock();
    }

    fn handleGenerationError(self: *Self, request_id: RequestId, err: []const u8) void {
        std.log.warn("Inference request {} generation error: {s}", .{ request_id, err });
        self.state_mutex.lock();
        const state = self.active.get(request_id) orelse {
            self.state_mutex.unlock();
            return;
        };
        self.state_mutex.unlock();
        self.failRequest(state, err);
    }

    fn failRequest(self: *Self, state: *RequestState, err: []const u8) void {
        state.mutex.lock();
        self.failRequestLocked(state, err);
        state.mutex.unlock();
    }

    fn failRequestLocked(self: *Self, state: *RequestState, err: []const u8) void {
        if (state.error_message) |old| self.base.allocator.free(old);
        state.error_message = self.base.allocator.dupe(u8, err) catch null;
        state.done = true;
        if (state.stream) {
            if (state.stream_conn) |conn| {
                _ = writeSse(conn, err) catch |write_err| {
                    std.log.warn("Failed to write inference error stream chunk for request {}: {}", .{ state.request_id, write_err });
                };
                _ = writeSse(conn, "[DONE]") catch |write_err| {
                    std.log.warn("Failed to write inference error stream terminator for request {}: {}", .{ state.request_id, write_err });
                };
            }
        }
        state.cond.broadcast();
    }

    fn cancelRequest(self: *Self, worker_id: NodeId, request_id: RequestId, reason: []const u8) void {
        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();
        payload.put("reason", .{ .string = reason }) catch |err| {
            std.log.warn("Failed to build cancel-generation payload for request {}: {}", .{ request_id, err });
            return;
        };
        _ = self.base.sendToWorkerWithContext(worker_id, MessageType.CANCEL_GENERATION, .{ .object = payload }, MessageContext{ .request_id = request_id }) catch |err| {
            std.log.warn("Failed to send cancel-generation request {} to worker {}: {}", .{ request_id, worker_id, err });
        };
    }

    fn flushSession(self: *Self, worker_id: NodeId, session_id: []const u8) void {
        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();
        payload.put("session_id", .{ .string = session_id }) catch |err| {
            std.log.warn("Failed to build flush-session payload for worker {}: {}", .{ worker_id, err });
            return;
        };
        _ = self.base.sendToWorker(worker_id, MessageType.FLUSH_SESSION, .{ .object = payload }) catch |err| {
            std.log.warn("Failed to send flush-session '{s}' to worker {}: {}", .{ session_id, worker_id, err });
        };
    }

    fn makeSessionId(self: *Self, request_id: RequestId) ![]const u8 {
        var buf: [64]u8 = undefined;
        const ts = std.time.timestamp();
        const id = try std.fmt.bufPrint(&buf, "sess_{d}_{d}", .{ ts, request_id });
        return self.base.allocator.dupe(u8, id);
    }

    fn renderCompletionResponse(self: *Self, state: *RequestState) ![]u8 {
        var root = std.json.ObjectMap.init(self.base.allocator);
        const id = try formatCompletionId(self.base.allocator, state.request_id);
        defer self.base.allocator.free(id);
        const model_id = try formatModelId(self.base.allocator, state.model_id);
        defer self.base.allocator.free(model_id);
        try root.put("id", .{ .string = id });
        try root.put("object", .{ .string = "chat.completion" });
        try root.put("created", .{ .integer = std.time.timestamp() });
        try root.put("model", .{ .string = model_id });

        var choice = std.json.ObjectMap.init(self.base.allocator);
        try choice.put("index", .{ .integer = 0 });
        var message_obj = std.json.ObjectMap.init(self.base.allocator);
        try message_obj.put("role", .{ .string = "assistant" });
        try message_obj.put("content", .{ .string = state.response.items });
        try choice.put("message", .{ .object = message_obj });
        try choice.put("finish_reason", .{ .string = state.finish_reason orelse "stop" });
        var choices = std.json.Array.init(self.base.allocator);
        try choices.append(.{ .object = choice });
        try root.put("choices", .{ .array = choices });

        var usage = std.json.ObjectMap.init(self.base.allocator);
        try usage.put("prompt_tokens", .{ .integer = @intCast(state.prompt_tokens) });
        try usage.put("completion_tokens", .{ .integer = @intCast(state.completion_tokens) });
        try usage.put("total_tokens", .{ .integer = @intCast(state.prompt_tokens + state.completion_tokens) });
        try root.put("usage", .{ .object = usage });
        try root.put("session_id", .{ .string = try self.base.allocator.dupe(u8, state.session_id) });

        return try jsonStringify(self.base.allocator, .{ .object = root });
    }

    fn emitGatewayCompletionEvent(self: *Self, state: *RequestState, response_json: ?[]const u8) !void {
        const client = self.gateway_client orelse return;
        const payload_json = if (response_json) |json|
            try self.base.allocator.dupe(u8, json)
        else
            try std.json.stringifyAlloc(self.base.allocator, .{
                .model = state.model_id,
                .session_id = state.session_id,
                .content = state.response.items,
                .finish_reason = state.finish_reason orelse "stop",
                .prompt_tokens = state.prompt_tokens,
                .completion_tokens = state.completion_tokens,
                .stream = state.stream,
            }, .{});
        defer self.base.allocator.free(payload_json);

        const provenance_json = try std.json.stringifyAlloc(self.base.allocator, .{
            .actor_id = "pcp-inference-controller",
        }, .{});
        defer self.base.allocator.free(provenance_json);

        const event_id = try std.fmt.allocPrint(self.base.allocator, "inference_completion_{d}", .{state.request_id});
        defer self.base.allocator.free(event_id);
        try client.emitEvent(event_id, "inference.completion", state.session_id, payload_json, provenance_json);
    }

    fn renderStreamChunk(self: *Self, state: *RequestState, content: []const u8, finish_reason: ?[]const u8) ![]u8 {
        var root = std.json.ObjectMap.init(self.base.allocator);
        const id = try formatCompletionId(self.base.allocator, state.request_id);
        defer self.base.allocator.free(id);
        const model_id = try formatModelId(self.base.allocator, state.model_id);
        defer self.base.allocator.free(model_id);
        try root.put("id", .{ .string = id });
        try root.put("object", .{ .string = "chat.completion.chunk" });
        try root.put("created", .{ .integer = std.time.timestamp() });
        try root.put("model", .{ .string = model_id });

        var choice = std.json.ObjectMap.init(self.base.allocator);
        try choice.put("index", .{ .integer = 0 });
        var delta = std.json.ObjectMap.init(self.base.allocator);
        if (content.len > 0) {
            try delta.put("content", .{ .string = content });
        }
        try choice.put("delta", .{ .object = delta });
        if (finish_reason) |reason| {
            try choice.put("finish_reason", .{ .string = reason });
        } else {
            try choice.put("finish_reason", .{ .null = {} });
        }
        var choices = std.json.Array.init(self.base.allocator);
        try choices.append(.{ .object = choice });
        try root.put("choices", .{ .array = choices });

        return try jsonStringify(self.base.allocator, .{ .object = root });
    }

    fn jsonStringify(allocator: Allocator, value: std.json.Value) ![]u8 {
        var buf = ArrayList(u8).init(allocator);
        errdefer buf.deinit();
        try std.json.stringify(value, .{}, buf.writer());
        return buf.toOwnedSlice();
    }

    fn writeSse(conn: net.Stream, payload: []const u8) !void {
        try conn.writeAll("data: ");
        try conn.writeAll(payload);
        try conn.writeAll("\n\n");
    }

    fn formatModelId(allocator: Allocator, model_id: []const u8) ![]const u8 {
        if (std.mem.startsWith(u8, model_id, "pcp/")) return allocator.dupe(u8, model_id);
        return std.fmt.allocPrint(allocator, "pcp/{s}", .{model_id});
    }

    fn normalizeModelId(model_id: []const u8) []const u8 {
        if (std.mem.startsWith(u8, model_id, "pcp/")) return model_id[4..];
        return model_id;
    }

    fn formatCompletionId(allocator: Allocator, request_id: RequestId) ![]const u8 {
        var buf: [32]u8 = undefined;
        const id = try std.fmt.bufPrint(&buf, "chatcmpl-{d}", .{request_id});
        return allocator.dupe(u8, id);
    }

    fn longestCommonPrefix(a: []const i64, b: []const i64) usize {
        const n = @min(a.len, b.len);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if (a[i] != b[i]) break;
        }
        return i;
    }

    fn hashTokens(tokens: []const i64) u64 {
        var hasher = std.hash.Wyhash.init(0);
        for (tokens) |t| {
            std.hash.autoHash(&hasher, t);
        }
        return hasher.final();
    }
};

fn workerClassForBackend(backend: Backend) WorkerClass {
    return switch (backend) {
        .cpu => .cpu,
        .cuda => .cuda,
        .rocm => .rocm,
        .metal => .metal,
        .vulkan => .vulkan,
    };
}

fn isChatRequestError(err: anyerror) bool {
    return switch (err) {
        error.InvalidMessage,
        error.MissingRole,
        error.InvalidRole,
        error.UnsupportedMessageRole,
        error.MissingContent,
        error.InvalidMessageContent,
        error.MultimodalMessageContentUnsupported,
        error.UnsupportedChatTemplate,
        => true,
        else => false,
    };
}

test "inference controller preserves data input dtypes in load payload" {
    const dtypes = [_]tensor.DType{ .i64, .i32, .f32, .bf16, .f16, .f64, .bool };
    var value = try InferenceController.dtypesToJson(std.testing.allocator, &dtypes);
    defer value.array.deinit();

    const expected = [_][]const u8{ "i64", "i32", "f32", "bf16", "f16", "f64", "bool" };
    try std.testing.expectEqual(expected.len, value.array.items.len);
    for (expected, value.array.items) |want, item| {
        try std.testing.expectEqualStrings(want, item.string);
    }
}

test "inference controller preserves streamed weight format in cloned config" {
    const cfg = inference_config.InferenceConfig{
        .model_id = "qwen36-27b",
        .pool_name = "default",
        .generation_vmfb_path = "models/qwen36_27b_one_token_nocache_linalg_cuda.vmfb",
        .generation_mlir_path = "models/qwen36_27b_one_token_nocache_linalg.mlir",
        .weights_path = "checkpoints/initial_weights/qwen36_27b_bf16_streamed.bin",
        .weights_format = "bf16_streamed",
        .tokenizer_source = "qwen",
        .tokenizer_path = "Qwen/Qwen3.6-27B",
        .chat_template = "qwen_text_only",
        .vision_enabled = false,
        .mtp_enabled = false,
        .full_conditional_generation_enabled = false,
        .num_gen_data_inputs = 2,
        .max_context_tokens = 64,
        .default_max_output_tokens = 1,
        .eos_token = null,
        .session_ttl_seconds = 600,
        .request_timeout_seconds = 300,
        .worker_backend = "cuda",
        .worker_target_arch = "sm_80",
        .api_token_env = "PCP_API_TOKEN",
    };

    const cloned = try InferenceController.cloneConfig(std.testing.allocator, cfg);
    defer {
        std.testing.allocator.free(cloned.model_id);
        std.testing.allocator.free(cloned.pool_name);
        std.testing.allocator.free(cloned.generation_vmfb_path);
        std.testing.allocator.free(cloned.generation_mlir_path);
        std.testing.allocator.free(cloned.weights_path);
        std.testing.allocator.free(cloned.weights_format);
        std.testing.allocator.free(cloned.tokenizer_source);
        if (cloned.tokenizer_path) |path| std.testing.allocator.free(path);
        std.testing.allocator.free(cloned.chat_template);
        std.testing.allocator.free(cloned.worker_backend);
        std.testing.allocator.free(cloned.worker_target_arch);
        std.testing.allocator.free(cloned.api_token_env);
    }

    try std.testing.expectEqualStrings("bf16_streamed", cloned.weights_format);
    try std.testing.expectEqualStrings("qwen_text_only", cloned.chat_template);
    try std.testing.expect(!cloned.vision_enabled);
    try std.testing.expect(!cloned.mtp_enabled);
    try std.testing.expect(!cloned.full_conditional_generation_enabled);
}

test "inference controller rejects qwen36 optional full-model features until implemented" {
    const base = inference_config.InferenceConfig{
        .model_id = "qwen36-27b",
        .pool_name = "default",
        .generation_vmfb_path = "models/qwen36_27b_one_token_nocache_linalg_cuda.vmfb",
        .generation_mlir_path = "models/qwen36_27b_one_token_nocache_linalg.mlir",
        .weights_path = "checkpoints/initial_weights/qwen36_27b_bf16_streamed.bin",
        .weights_format = "bf16_streamed",
        .tokenizer_source = "qwen",
        .tokenizer_path = "Qwen/Qwen3.6-27B",
        .chat_template = "qwen_text_only",
        .vision_enabled = false,
        .mtp_enabled = false,
        .full_conditional_generation_enabled = false,
        .num_gen_data_inputs = 2,
        .max_context_tokens = 64,
        .default_max_output_tokens = 1,
        .eos_token = null,
        .session_ttl_seconds = 600,
        .request_timeout_seconds = 300,
        .worker_backend = "cuda",
        .worker_target_arch = "sm_80",
        .api_token_env = "PCP_API_TOKEN",
    };

    try InferenceController.validateOptionalFullModelFeatures(base);

    var vision = base;
    vision.vision_enabled = true;
    try std.testing.expectError(error.VisionInferenceUnsupported, InferenceController.validateOptionalFullModelFeatures(vision));

    var mtp = base;
    mtp.mtp_enabled = true;
    try std.testing.expectError(error.MtpInferenceUnsupported, InferenceController.validateOptionalFullModelFeatures(mtp));

    var conditional = base;
    conditional.full_conditional_generation_enabled = true;
    try std.testing.expectError(error.FullConditionalGenerationUnsupported, InferenceController.validateOptionalFullModelFeatures(conditional));

    var chat = base;
    chat.chat_template = "qwen_vision";
    try std.testing.expectError(error.UnsupportedChatTemplate, InferenceController.validateOptionalFullModelFeatures(chat));
}
