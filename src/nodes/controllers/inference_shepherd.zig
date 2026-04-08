const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;

const Shepherd = @import("shepherd.zig").Shepherd;
const WorkerMessageHook = @import("shepherd.zig").WorkerMessageHook;
const WorkerConnectionHook = @import("shepherd.zig").WorkerConnectionHook;
const inference_config = @import("../../inference/config.zig");
const inference_types = @import("../../inference/types.zig");
const model_registry = @import("../../inference/model_registry.zig");
const session_manager = @import("../../inference/session_manager.zig");
const router = @import("../../inference/router.zig");
const tokenizer_mod = @import("../../inference/tokenizer.zig");
const http_server = @import("../../inference/http_server.zig");
const control_api = @import("../../control_plane/api.zig");
const control_state = @import("../../control_plane/state.zig");
const tcp_stream = @import("../../network/tcp_stream.zig");
const message = @import("../../network/message.zig");
const model_introspection = @import("../../mlir/model_introspection.zig");

const ArrayList = std.ArrayList;
const MessageType = message.MessageType;
const MessageContext = message.MessageContext;
const NodeId = message.NodeId;
const RequestId = message.RequestId;
const TcpServer = tcp_stream.TcpServer;

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

pub const InferenceShepherd = struct {
    base: Shepherd,
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
    api_server: ?TcpServer,
    api_host: ?[]const u8,
    api_port: u16,
    api_running: std.atomic.Value(u8),
    shutdown_requested: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(allocator: Allocator, cfg: inference_config.InferenceConfig, api_token: []const u8, operator_state: ?*control_state.ControllerState) !Self {
        const tok_kind = parseTokenizerKind(cfg.tokenizer_source) catch |err| return err;
        var tokenizer = try tokenizer_mod.Tokenizer.init(allocator, tok_kind, cfg.tokenizer_path);
        errdefer tokenizer.deinit();

        var controller = Self{
            .base = Shepherd.init(allocator),
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
            .api_server = null,
            .api_host = null,
            .api_port = 0,
            .api_running = std.atomic.Value(u8).init(0),
            .shutdown_requested = std.atomic.Value(u8).init(0),
        };

        try controller.loadModelsFromConfig();
        return controller;
    }

    pub fn deinit(self: *Self) void {
        self.tokenizer.deinit();
        self.allocatorFreeConfig();
        self.base.deinit();
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
    }

    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        try self.base.listen(host, port);
    }

    pub fn getReadyWorkerCount(self: *Self) usize {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        if (self.ready_by_model.getPtr(self.config.model_id)) |set| {
            return set.set.count();
        }
        return 0;
    }

    pub fn snapshotMetrics(self: *Self) Metrics {
        self.metrics_mutex.lock();
        defer self.metrics_mutex.unlock();
        return self.metrics;
    }

    pub fn attachHooks(self: *Self) void {
        self.base.setMessageHook(.{ .ctx = self, .handler = handleWorkerMessageHook });
        self.base.setConnectionHook(.{ .ctx = self, .on_connect = handleWorkerConnectHook, .on_disconnect = handleWorkerDisconnectHook });
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
        self.base.stop();
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
        alloc.free(self.config.tokenizer_source);
        if (self.config.tokenizer_path) |p| alloc.free(p);
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
            .tokenizer_source = try allocator.dupe(u8, cfg.tokenizer_source),
            .tokenizer_path = if (cfg.tokenizer_path) |p| try allocator.dupe(u8, p) else null,
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
                self.config.num_gen_data_inputs,
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
        try self.dispatchLoadModel(worker_id, self.config.model_id);
    }

    fn handleWorkerDisconnectHook(ctx: *anyopaque, worker_id: NodeId) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

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
    }

    fn handleWorkerMessageHook(ctx: *anyopaque, worker_id: NodeId, msg: message.MessageEnvelope) anyerror!bool {
        const self: *Self = @ptrCast(@alignCast(ctx));

        if (std.mem.eql(u8, msg.msg_type, MessageType.MODEL_READY)) {
            const payload = msg.data.object;
            const model_id = payload.get("model_id").?.string;
            self.state_mutex.lock();
            if (self.ready_by_model.getPtr(model_id)) |set| {
                try set.add(worker_id);
            }
            self.state_mutex.unlock();
            self.dispatchPending(model_id);
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

        const meta_path = try std.fmt.allocPrint(self.base.allocator, "{s}.meta.json", .{record.generation_mlir_path});
        defer self.base.allocator.free(meta_path);
        var meta = try model_introspection.ModelMetadata.loadFromFile(self.base.allocator, meta_path);
        defer meta.deinit();

        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();
        try payload.put("model_id", .{ .string = record.model_id });
        try payload.put("vmfb_path", .{ .string = record.generation_vmfb_path });
        try payload.put("weights_path", .{ .string = record.weights_path });
        try payload.put("max_context_tokens", .{ .integer = @intCast(record.max_context_tokens) });

        try payload.put("parameter_shapes", shapesToJson(self.base.allocator, meta.parameter_shapes));
        try payload.put("data_input_shapes", shapesToJson(self.base.allocator, meta.data_input_shapes));
        try payload.put("data_input_dtypes", dtypesToJson(self.base.allocator, meta.data_input_dtypes));

        try self.base.sendToWorker(worker_id, MessageType.LOAD_MODEL, .{ .object = payload });
    }

    fn shapesToJson(allocator: Allocator, shapes: [][]i64) std.json.Value {
        var outer = std.json.Array.init(allocator);
        for (shapes) |shape| {
            var inner = std.json.Array.init(allocator);
            for (shape) |dim| {
                inner.append(.{ .integer = dim }) catch {};
            }
            outer.append(.{ .array = inner }) catch {};
        }
        return .{ .array = outer };
    }

    fn dtypesToJson(allocator: Allocator, dtypes: []const @import("../../core/tensor.zig").DType) std.json.Value {
        var arr = std.json.Array.init(allocator);
        for (dtypes) |dt| {
            const s = switch (dt) {
                .f32 => "f32",
                .bf16 => "bf16",
                .f16 => "f16",
                else => "f32",
            };
            arr.append(.{ .string = s }) catch {};
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
            _ = http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch {};
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
                &self.base,
                self.api_token,
                .{
                    .ctx = self,
                    .render = renderOperatorReady,
                },
                .{
                    .ctx = self,
                    .render = renderOperatorMetrics,
                },
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
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        if (self.ready_by_model.getPtr(self.config.model_id)) |set| {
            return set.set.count() > 0;
        }
        return false;
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
        var ready_count: usize = 0;
        self.state_mutex.lock();
        if (self.ready_by_model.getPtr(self.config.model_id)) |set| {
            ready_count = set.set.count();
        }
        self.state_mutex.unlock();

        self.metrics_mutex.lock();
        const metrics = self.metrics;
        self.metrics_mutex.unlock();

        var buf = ArrayList(u8).init(self.base.allocator);
        errdefer buf.deinit();

        try buf.writer().print(
            "{{\"ready_workers\":{d},\"metrics\":{{\"total_requests\":{d},\"active_requests\":{d},\"queued_requests\":{d},\"completed_requests\":{d},\"failed_requests\":{d},\"tokens_generated\":{d},\"prompt_tokens\":{d},\"ttft_total_ms\":{d},\"ttft_count\":{d},\"cache_hits\":{d},\"cache_misses\":{d}}}}}",
            .{
                ready_count,
                metrics.total_requests,
                metrics.active_requests,
                metrics.queued_requests,
                metrics.completed_requests,
                metrics.failed_requests,
                metrics.tokens_generated,
                metrics.prompt_tokens,
                metrics.ttft_total_ms,
                metrics.ttft_count,
                metrics.cache_hits,
                metrics.cache_misses,
            },
        );

        return buf.toOwnedSlice();
    }

    fn renderOperatorReady(ctx: *anyopaque, allocator: Allocator, state: *control_state.ControllerState, connected_workers: usize) !control_api.ReadyResult {
        _ = connected_workers;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const ready = self.isReady();
        const body = try self.renderReadyzJson();
        if (state.mode == .inference) {
            state.setInferenceMetrics(toControlPlaneMetrics(self.snapshotMetrics()));
        }
        _ = allocator;
        return .{ .ready = ready, .body = body };
    }

    fn renderOperatorMetrics(ctx: *anyopaque, allocator: Allocator, state: *control_state.ControllerState, connected_workers: usize) ![]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const metrics = self.snapshotMetrics();
        state.setInferenceMetrics(toControlPlaneMetrics(metrics));
        return state.renderMetricsJson(allocator, connected_workers);
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

        const prompt_text = if (std.mem.eql(u8, self.config.tokenizer_source, "qwen"))
            try renderQwenPrompt(self.base.allocator, messages)
        else
            try renderPrompt(self.base.allocator, messages);
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
            self.enqueueRequest(request_state);
        }

        request_state.mutex.lock();
        while (!request_state.done) {
            request_state.cond.wait(&request_state.mutex);
        }
        request_state.mutex.unlock();

        self.state_mutex.lock();
        _ = self.active.remove(request_id);
        self.state_mutex.unlock();

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
                try http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, err_msg);
            } else {
                const body_json = try self.renderCompletionResponse(request_state);
                defer self.base.allocator.free(body_json);
                try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body_json);
            }
        }

        request_state.deinit();
    }

    fn dispatchRequest(self: *Self, state: *RequestState) !bool {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        return try self.dispatchRequestLocked(state);
    }

    fn dispatchRequestLocked(self: *Self, state: *RequestState) !bool {
        const ready_workers = if (self.ready_by_model.getPtr(state.model_id)) |set|
            try set.snapshot()
        else
            try self.base.allocator.alloc(NodeId, 0);
        defer self.base.allocator.free(ready_workers);

        const session_record = self.sessions.get(state.session_id);
        const decision = self.router.decide(session_record, ready_workers);
        if (decision.queued or decision.worker_id == null) {
            return false;
        }

        const worker_id = decision.worker_id.?;
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
        try payload.put("eos_token", .{ .integer = self.config.eos_token });
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

    fn enqueueRequest(self: *Self, state: *RequestState) void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        self.pending.append(state.request_id) catch {};
        self.active.put(state.request_id, state) catch {};

        self.metrics_mutex.lock();
        self.metrics.queued_requests += 1;
        self.metrics_mutex.unlock();
    }

    fn dispatchPending(self: *Self, model_id: []const u8) void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
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
                continue;
            }
            idx += 1;
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
                    _ = writeSse(conn, payload) catch {};
                    self.base.allocator.free(payload);
                }
                _ = writeSse(conn, "[DONE]") catch {};
            }
        }
        state.cond.broadcast();
        state.mutex.unlock();
    }

    fn handleGenerationError(self: *Self, request_id: RequestId, err: []const u8) void {
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
                _ = writeSse(conn, err) catch {};
                _ = writeSse(conn, "[DONE]") catch {};
            }
        }
        state.cond.broadcast();
    }

    fn cancelRequest(self: *Self, worker_id: NodeId, request_id: RequestId, reason: []const u8) void {
        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();
        payload.put("reason", .{ .string = reason }) catch {};
        _ = self.base.sendToWorkerWithContext(worker_id, MessageType.CANCEL_GENERATION, .{ .object = payload }, MessageContext{ .request_id = request_id }) catch {};
    }

    fn flushSession(self: *Self, worker_id: NodeId, session_id: []const u8) void {
        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();
        payload.put("session_id", .{ .string = session_id }) catch {};
        _ = self.base.sendToWorker(worker_id, MessageType.FLUSH_SESSION, .{ .object = payload }) catch {};
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

    fn renderPrompt(allocator: Allocator, messages: std.json.Array) ![]u8 {
        var buf = ArrayList(u8).init(allocator);
        errdefer buf.deinit();

        for (messages.items) |msg| {
            const obj = switch (msg) {
                .object => |o| o,
                else => return error.InvalidMessage,
            };
            const role_val = obj.get("role") orelse return error.MissingRole;
            const content_val = obj.get("content") orelse return error.MissingContent;
            const role = role_val.string;
            const content = content_val.string;
            if (std.mem.eql(u8, role, "system")) {
                try buf.appendSlice("System: ");
                try buf.appendSlice(content);
                try buf.appendSlice("\n");
            } else if (std.mem.eql(u8, role, "user")) {
                try buf.appendSlice("User: ");
                try buf.appendSlice(content);
                try buf.appendSlice("\n");
            } else if (std.mem.eql(u8, role, "assistant")) {
                try buf.appendSlice("Assistant: ");
                try buf.appendSlice(content);
                try buf.appendSlice("\n");
            }
        }
        try buf.appendSlice("Assistant: ");
        return buf.toOwnedSlice();
    }

    fn renderQwenPrompt(allocator: Allocator, messages: std.json.Array) ![]u8 {
        var buf = ArrayList(u8).init(allocator);
        errdefer buf.deinit();

        for (messages.items) |msg| {
            const obj = switch (msg) {
                .object => |o| o,
                else => return error.InvalidMessage,
            };
            const role_val = obj.get("role") orelse return error.MissingRole;
            const content_val = obj.get("content") orelse return error.MissingContent;
            const role = role_val.string;
            const content = content_val.string;

            if (std.mem.eql(u8, role, "system") or
                std.mem.eql(u8, role, "user") or
                std.mem.eql(u8, role, "assistant"))
            {
                try buf.appendSlice("<|im_start|>");
                try buf.appendSlice(role);
                try buf.append('\n');
                try buf.appendSlice(content);
                try buf.appendSlice("<|im_end|>\n");
            }
        }

        try buf.appendSlice("<|im_start|>assistant\n");
        return buf.toOwnedSlice();
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
