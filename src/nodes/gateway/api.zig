const std = @import("std");
const net = std.net;

const Allocator = std.mem.Allocator;
const TcpServer = @import("../../network/tcp_stream.zig").TcpServer;
const http_server = @import("../../network/http_server.zig");
const http_util = @import("../../protocol/http_util.zig");
const graph_policy_store = @import("../../protocol/federation/graph/policy_store.zig");
const graph_types = @import("../../protocol/federation/graph/types.zig");
const training_workload = @import("../../workloads/training/workload.zig");
const identity = @import("identity.zig");
const service_registry = @import("service_registry.zig");
const graph_adapter = @import("graph_adapter.zig");
const gateway_mod = @import("gateway.zig");
const event_ingest = @import("event_ingest.zig");
const custom_extensions = @import("custom_extensions.zig");
const runtime_config = @import("../../runtime/config.zig");
const bearerToken = http_util.bearerToken;
const parseHttpMethod = http_util.parseHttpMethod;
const statusText = http_util.statusText;

const ProxySelectionRequest = struct {
    service_id: ?[]const u8 = null,
    executor_id: ?[]const u8 = null,
};

const ReservationCreateRequest = struct {
    service_id: ?[]const u8 = null,
    executor_id: ?[]const u8 = null,
    workers_required: ?usize = null,
};

const QueryThenInferRequest = struct {
    graph_query: graph_types.QueryRequest,
    inference: std.json.Value,
};

const DownstreamResponse = struct {
    status: std.http.Status,
    body: []u8,
};

pub const GatewayApiServer = struct {
    allocator: Allocator,
    gateway: *gateway_mod.Gateway,
    api_token: ?[]const u8,
    api_token_env: ?[]const u8,
    api_token_file_env: ?[]const u8,
    internal_api_token: ?[]const u8,
    internal_api_token_env: ?[]const u8,
    internal_api_token_file_env: ?[]const u8,
    federation_hub_token: ?[]const u8,
    federation_hub_token_env: ?[]const u8,
    federation_hub_token_file_env: ?[]const u8,
    server: ?TcpServer,
    listen_host: ?[]const u8,
    listen_port: u16,
    is_running: std.atomic.Value(u8),
    requests_total: std.atomic.Value(u64),
    errors_total: std.atomic.Value(u64),
    duration_ns_total: std.atomic.Value(u64),

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        gateway: *gateway_mod.Gateway,
        api_token: ?[]const u8,
        api_token_env: ?[]const u8,
        api_token_file_env: ?[]const u8,
        internal_api_token: ?[]const u8,
        internal_api_token_env: ?[]const u8,
        internal_api_token_file_env: ?[]const u8,
        federation_hub_token: ?[]const u8,
        federation_hub_token_env: ?[]const u8,
        federation_hub_token_file_env: ?[]const u8,
    ) Self {
        return .{
            .allocator = allocator,
            .gateway = gateway,
            .api_token = api_token,
            .api_token_env = api_token_env,
            .api_token_file_env = api_token_file_env,
            .internal_api_token = internal_api_token,
            .internal_api_token_env = internal_api_token_env,
            .internal_api_token_file_env = internal_api_token_file_env,
            .federation_hub_token = federation_hub_token,
            .federation_hub_token_env = federation_hub_token_env,
            .federation_hub_token_file_env = federation_hub_token_file_env,
            .server = null,
            .listen_host = null,
            .listen_port = 0,
            .is_running = std.atomic.Value(u8).init(0),
            .requests_total = std.atomic.Value(u64).init(0),
            .errors_total = std.atomic.Value(u64).init(0),
            .duration_ns_total = std.atomic.Value(u64).init(0),
        };
    }

    pub fn start(self: *Self, host: []const u8, port: u16) !void {
        try self.listen(host, port);
        try self.runAcceptLoop();
    }

    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        self.listen_host = host;
        self.listen_port = port;
        std.log.info("Starting gateway API server on {s}:{d}", .{ host, port });
        self.server = TcpServer.init(self.allocator, host, port) catch |err| {
            std.log.err("Gateway API listen failed on {s}:{d}: {}", .{ host, port, err });
            return err;
        };
        self.is_running.store(1, .release);
    }

    pub fn runAcceptLoop(self: *Self) !void {
        while (self.is_running.load(.acquire) == 1) {
            const connection = if (self.server) |*server|
                server.accept() catch |err| {
                    if (self.is_running.load(.acquire) == 0) break;
                    std.log.err("Gateway API accept failed: {}", .{err});
                    continue;
                }
            else
                break;

            if (self.is_running.load(.acquire) == 0) {
                connection.stream.close();
                break;
            }

            const thread = std.Thread.spawn(.{}, handleConnection, .{ self, connection.stream }) catch |err| {
                std.log.err("Failed to spawn gateway API handler thread: {}", .{err});
                connection.stream.close();
                continue;
            };
            thread.detach();
        }

        if (self.server) |*server| {
            server.deinit();
            self.server = null;
        }
    }

    pub fn stop(self: *Self) void {
        self.is_running.store(0, .release);
        if (self.listen_host) |host| {
            const address = net.Address.parseIp(host, self.listen_port) catch return;
            const stream = net.tcpConnectToAddress(address) catch return;
            stream.close();
        }
    }

    fn handleConnection(self: *Self, stream: net.Stream) void {
        defer stream.close();

        const start_ns = std.time.nanoTimestamp();
        var count_request = false;
        var error_request = false;
        defer {
            if (count_request) {
                _ = self.requests_total.fetchAdd(1, .acq_rel);
                if (error_request) _ = self.errors_total.fetchAdd(1, .acq_rel);
                const elapsed_ns = @max(0, std.time.nanoTimestamp() - start_ns);
                _ = self.duration_ns_total.fetchAdd(@intCast(elapsed_ns), .acq_rel);
            }
        }

        var req = http_server.readRequest(stream, self.allocator, 1024 * 1024) catch |err| {
            std.log.warn("Gateway API failed to read request: {}", .{err});
            return;
        };
        defer req.deinit();
        count_request = true;

        _ = self.handleRequest(stream, &req) catch |err| {
            error_request = true;
            std.log.err("Gateway API route failed: {}", .{err});
            _ = http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch |write_err| {
                std.log.warn("Gateway API failed to write error response: {}", .{write_err});
            };
            return;
        };
    }

    fn handleRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (try self.handleProbeRequest(stream, req)) return true;
        if (try self.handleInternalFederationRequest(stream, req)) return true;
        if (!std.mem.startsWith(u8, req.path, "/v1/")) return false;
        if (!self.authorize(req)) {
            try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
            return true;
        }
        return try self.handleAuthorizedV1Request(stream, req);
    }

    fn snapshotMetrics(self: *Self) gateway_mod.ApiMetricsSnapshot {
        return .{
            .requests_total = self.requests_total.load(.acquire),
            .errors_total = self.errors_total.load(.acquire),
            .duration_ns_total = self.duration_ns_total.load(.acquire),
        };
    }

    fn handleProbeRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (std.mem.eql(u8, req.path, "/healthz")) {
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain"}, "ok");
            return true;
        }

        if (std.mem.eql(u8, req.path, "/readyz")) {
            const ready_result = try self.gateway.renderReadyJson(self.allocator, self.api_token != null);
            defer self.allocator.free(ready_result.body);
            const status = if (ready_result.ready) "200 OK" else "503 Service Unavailable";
            try http_server.writeResponse(stream, status, &.{"Content-Type: application/json"}, ready_result.body);
            return true;
        }
        if (std.mem.eql(u8, req.path, "/metrics")) {
            const body = try self.gateway.renderPrometheusMetrics(self.allocator, self.snapshotMetrics());
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain; version=0.0.4"}, body);
            return true;
        }
        return false;
    }

    fn handleInternalFederationRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/internal/events")) {
            if (!self.authorizeInternal(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }

            if (req.body.len == 0) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "missing_body");
                return true;
            }

            var parsed = try std.json.parseFromSlice(event_ingest.InternalEventsRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();

            const body = try self.gateway.event_ingester.ingestJson(self.allocator, &self.gateway.graph, parsed.value);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/internal/federation/training/jobs")) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            self.handleTrainingJobSubmit(stream, req, null) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (try self.handleCustomInternalFederationRequest(stream, req)) return true;

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/internal/federation/training/reservations")) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            self.handleReservationCreate(stream, req, service_registry.ServiceType.training.asString()) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/internal/federation/rl/jobs")) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            self.handleJobSubmit(stream, req, service_registry.ServiceType.rl.asString()) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/internal/federation/rl/reservations")) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            self.handleReservationCreate(stream, req, service_registry.ServiceType.rl.asString()) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.startsWith(u8, req.path, "/v1/internal/federation/reservations/") and std.mem.endsWith(u8, req.path, "/commit")) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            const reservation_id = req.path["/v1/internal/federation/reservations/".len .. req.path.len - "/commit".len];
            self.handleReservationCommit(stream, req, reservation_id) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.startsWith(u8, req.path, "/v1/internal/federation/reservations/") and std.mem.endsWith(u8, req.path, "/release")) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            const reservation_id = req.path["/v1/internal/federation/reservations/".len .. req.path.len - "/release".len];
            self.handleReservationRelease(stream, req, reservation_id) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.startsWith(u8, req.path, "/v1/internal/federation/jobs/") and std.mem.endsWith(u8, req.path, "/cancel")) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            const job_id = req.path["/v1/internal/federation/jobs/".len .. req.path.len - "/cancel".len];
            self.handleJobCancel(stream, req, job_id) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.startsWith(u8, req.path, "/v1/internal/federation/jobs/")) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            const job_id = req.path["/v1/internal/federation/jobs/".len..];
            self.handleJobLookup(stream, req, job_id) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        return false;
    }

    fn handleCustomInternalFederationRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, custom_extensions.route_custom_internal_training_jobs)) {
            if (!self.authorizeFederationHub(req)) {
                try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
                return true;
            }
            self.handleTrainingJobSubmit(stream, req, .custom_extension) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        return false;
    }

    fn handleAuthorizedV1Request(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/controller")) {
            const body = try self.gateway.renderControllerJson(self.allocator, self.api_token != null);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/capabilities")) {
            const body = try self.gateway.renderCapabilitiesJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/services")) {
            const body = try self.gateway.service_registry.renderJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.startsWith(u8, req.path, "/v1/services/")) {
            const executor_id = req.path["/v1/services/".len..];
            const body = try self.gateway.service_registry.renderServiceJson(self.allocator, executor_id);
            if (body) |json| {
                defer self.allocator.free(json);
                try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, json);
            } else {
                try http_server.writeResponse(stream, "404 Not Found", &.{"Content-Type: text/plain"}, "service_not_found");
            }
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/services/register")) {
            if (req.body.len == 0) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "missing_body");
                return true;
            }

            var parsed = try std.json.parseFromSlice(service_registry.RegisterRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();
            var derived_executor_id: ?[]u8 = null;
            defer if (derived_executor_id) |value| self.allocator.free(value);

            var register_request = parsed.value;
            if (register_request.executor_id == null) {
                derived_executor_id = try identity.deriveExecutorId(
                    self.allocator,
                    self.gateway.config.gateway_id,
                    register_request.service_type,
                    register_request.service_id,
                );
                register_request.executor_id = derived_executor_id.?;
            }

            var service = self.gateway.service_registry.register(register_request) catch |err| switch (err) {
                error.InvalidServiceType => {
                    try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "invalid_service_type");
                    return true;
                },
                else => return err,
            };
            defer service.deinit(self.allocator);

            const body = try std.json.stringifyAlloc(self.allocator, .{
                .accepted = true,
                .service = .{
                    .service_id = service.service_id,
                    .executor_id = service.executor_id,
                    .service_type = service.service_type,
                    .base_url = service.base_url,
                    .auth_mode = service.auth_mode,
                    .health_status = service.health_status,
                    .job_status = service.job_status,
                    .worker_count = service.worker_count,
                    .ready_worker_count = service.ready_worker_count,
                    .workers_connected = service.workers_connected,
                    .workers_ready = service.workers_ready,
                    .workers_available = service.workers_available,
                    .workers_dispatchable = service.workers_available,
                    .worker_class = service.worker_class,
                    .target_arch = service.target_arch,
                    .reserved_workers = service.reserved_workers,
                    .max_workers = service.max_workers,
                    .capabilities = service.capabilities.items,
                    .registered_at = service.registered_at,
                    .updated_at = service.updated_at,
                },
            }, .{});
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/inference/chat/completions")) {
            var service = self.requireServiceByType(
                service_registry.ServiceType.inference.asString(),
                req.header("x-pcp-service-id"),
                req.header("x-pcp-executor-id"),
            ) catch |err| {
                try writeProxyErrorResponse(stream, err);
                return true;
            };
            defer service.deinit(self.allocator);

            const downstream = self.forwardToService(service, req, "/v1/chat/completions") catch |err| {
                try writeProxyErrorResponse(stream, err);
                return true;
            };
            defer self.allocator.free(downstream.body);
            try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, downstream.body);
            return true;
        }

        if (try self.handleCustomAuthorizedV1Request(stream, req)) return true;

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/inference/query/chat/completions")) {
            self.handleQueryThenInfer(stream, req) catch |err| switch (err) {
                error.InvalidQueryMode,
                error.FederationHubNotConfigured,
                error.MissingMessages,
                error.InvalidMessages,
                error.InvalidInferenceRequest,
                error.StreamingNotSupported,
                => try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, @errorName(err)),
                error.FederationHubQueryFailed,
                error.InvalidFederationHubResponse,
                error.InvalidDownstreamResponse,
                => try http_server.writeResponse(stream, "502 Bad Gateway", &.{"Content-Type: text/plain"}, @errorName(err)),
                error.ServiceUnavailable,
                error.ServiceProxyFailed,
                error.UnsupportedMethod,
                => try writeProxyErrorResponse(stream, err),
                else => return err,
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/rl/jobs")) {
            self.handleJobSubmit(stream, req, service_registry.ServiceType.rl.asString()) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/training/jobs")) {
            self.handleTrainingJobSubmit(stream, req, null) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/jobs")) {
            const body = self.renderJobsJson(req) catch |err| {
                try writeProxyErrorResponse(stream, err);
                return true;
            };
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.startsWith(u8, req.path, "/v1/jobs/") and std.mem.endsWith(u8, req.path, "/cancel")) {
            const job_id = req.path["/v1/jobs/".len .. req.path.len - "/cancel".len];
            self.handleJobCancel(stream, req, job_id) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.startsWith(u8, req.path, "/v1/jobs/")) {
            const job_id = req.path["/v1/jobs/".len..];
            self.handleJobLookup(stream, req, job_id) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/federation/status")) {
            const body = try self.gateway.renderFederationStatusJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/federation/peers")) {
            const body = try self.gateway.federation.renderPeersJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/federation/services")) {
            const body = try self.gateway.federation.renderServicesJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/federation/replication")) {
            const body = try self.gateway.renderFederationReplicationJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/federation/connect")) {
            if (self.gateway.config.resolvedFederationHubEndpoint() == null) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "federation_hub_not_configured");
                return true;
            }

            const body = try self.gateway.renderFederationStatusJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/graph/status")) {
            const body = try self.gateway.graph.renderStatusJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/graph/policies")) {
            const body = try self.gateway.policy_store.renderJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "PUT") and std.mem.startsWith(u8, req.path, "/v1/graph/policies/")) {
            if (req.body.len == 0) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "missing_body");
                return true;
            }

            const namespace_id = req.path["/v1/graph/policies/".len..];
            var parsed = try std.json.parseFromSlice(graph_policy_store.PolicyUpdateRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();

            const updated = self.gateway.policy_store.upsert(namespace_id, parsed.value, gateway_mod.defaultPolicyVisibility(self.gateway.config)) catch |err| switch (err) {
                error.InvalidVisibility => {
                    try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "InvalidVisibility");
                    return true;
                },
                else => return err,
            };

            const body = try std.json.stringifyAlloc(self.allocator, .{ .accepted = true, .policy = updated }, .{});
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/graph/mutate")) {
            if (req.body.len == 0) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "missing_body");
                return true;
            }

            var parsed = try std.json.parseFromSlice(graph_types.MutateRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();

            const body = self.gateway.graph.applyMutationsJson(self.allocator, parsed.value) catch |err| switch (err) {
                error.InvalidMutationType, error.InvalidPayload, error.InvalidVisibility => {
                    try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, @errorName(err));
                    return true;
                },
                else => return err,
            };
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/graph/query")) {
            if (req.body.len == 0) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "missing_body");
                return true;
            }

            var parsed = try std.json.parseFromSlice(graph_types.QueryRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();

            const body = renderFederatedGraphQueryJson(
                self.allocator,
                &self.gateway.graph,
                self.gateway.config.resolvedFederationHubEndpoint(),
                self.federation_hub_token,
                parsed.value,
            ) catch |err| switch (err) {
                error.InvalidQueryMode, error.FederationHubNotConfigured => {
                    try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, @errorName(err));
                    return true;
                },
                error.FederationHubQueryFailed, error.InvalidFederationHubResponse => {
                    try http_server.writeResponse(stream, "502 Bad Gateway", &.{"Content-Type: text/plain"}, @errorName(err));
                    return true;
                },
                else => return err,
            };
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        try http_server.writeResponse(stream, "404 Not Found", &.{"Content-Type: text/plain"}, "not_found");
        return true;
    }

    fn handleCustomAuthorizedV1Request(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, custom_extensions.route_custom_predict)) {
            var service = self.requireServiceByType(
                custom_extensions.service_type_custom_inference,
                req.header("x-pcp-service-id"),
                req.header("x-pcp-executor-id"),
            ) catch |err| {
                try writeProxyErrorResponse(stream, err);
                return true;
            };
            defer service.deinit(self.allocator);

            const downstream = self.forwardToService(service, req, custom_extensions.route_custom_predict) catch |err| {
                try writeProxyErrorResponse(stream, err);
                return true;
            };
            defer self.allocator.free(downstream.body);
            try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, downstream.body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, custom_extensions.route_custom_go_no_go)) {
            var service = self.requireServiceByType(
                custom_extensions.service_type_custom_inference,
                req.header("x-pcp-service-id"),
                req.header("x-pcp-executor-id"),
            ) catch |err| {
                try writeProxyErrorResponse(stream, err);
                return true;
            };
            defer service.deinit(self.allocator);

            const downstream = self.forwardToService(service, req, custom_extensions.route_custom_go_no_go) catch |err| {
                try writeProxyErrorResponse(stream, err);
                return true;
            };
            defer self.allocator.free(downstream.body);
            try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, downstream.body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, custom_extensions.route_custom_training_jobs)) {
            self.handleTrainingJobSubmit(stream, req, .custom_extension) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        return false;
    }

    fn authorize(self: *Self, req: *http_server.HttpRequest) bool {
        if (self.api_token == null and !self.hasFileBackedSecret(self.api_token_env, self.api_token_file_env)) return true;
        const header = req.header("authorization") orelse return false;
        const token = bearerToken(header) orelse return false;
        return self.secretMatches(token, self.api_token, self.api_token_env, self.api_token_file_env);
    }

    fn authorizeInternal(self: *Self, req: *http_server.HttpRequest) bool {
        if (self.internal_api_token != null or self.hasFileBackedSecret(self.internal_api_token_env, self.internal_api_token_file_env)) {
            const header = req.header("authorization") orelse return false;
            const bearer = bearerToken(header) orelse return false;
            return self.secretMatches(bearer, self.internal_api_token, self.internal_api_token_env, self.internal_api_token_file_env);
        }
        return self.authorize(req);
    }

    fn authorizeFederationHub(self: *Self, req: *http_server.HttpRequest) bool {
        const header = req.header("authorization") orelse return false;
        const bearer = bearerToken(header) orelse return false;
        return self.secretMatches(bearer, self.federation_hub_token, self.federation_hub_token_env, self.federation_hub_token_file_env);
    }

    fn secretMatches(
        self: *Self,
        presented: []const u8,
        cached: ?[]const u8,
        env_name: ?[]const u8,
        file_env: ?[]const u8,
    ) bool {
        if (runtime_config.loadSecretFromEnvOrFile(self.allocator, env_name, file_env, false)) |maybe_secret| {
            if (maybe_secret) |secret| {
                defer self.allocator.free(secret);
                return std.mem.eql(u8, presented, secret);
            }
        } else |err| {
            std.log.warn("Failed to refresh token from file-backed secret: {}", .{err});
        }
        if (cached) |token| return std.mem.eql(u8, presented, token);
        return false;
    }

    fn hasFileBackedSecret(_: *Self, env_name: ?[]const u8, file_env: ?[]const u8) bool {
        if (file_env) |name| {
            if (std.posix.getenv(name) != null) return true;
        }
        if (env_name) |name| {
            var buf: [256]u8 = undefined;
            const derived = std.fmt.bufPrint(&buf, "{s}_FILE", .{name}) catch return false;
            return std.posix.getenv(derived) != null;
        }
        return false;
    }

    fn handleJobSubmit(self: *Self, stream: net.Stream, req: *http_server.HttpRequest, service_type: []const u8) !void {
        const selection = try parseProxySelection(self.allocator, req.body);
        defer if (selection) |parsed| parsed.deinit();

        var service = try self.requireServiceByType(
            service_type,
            if (selection) |parsed| parsed.value.service_id else null,
            if (selection) |parsed| parsed.value.executor_id else null,
        );
        defer service.deinit(self.allocator);

        if (self.gateway.localJobHookForExecutor(service.executor_id, service.service_type)) |hook| {
            var result = try hook.submit(hook.ctx, self.allocator, req.body);
            defer result.deinit();

            const body = try wrapSubmittedJobJson(self.allocator, result, service);
            defer self.allocator.free(body);

            try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
            return;
        }

        const downstream = try self.forwardRaw(service, req.method, req.header("authorization"), "/v1/job", "", null);
        defer self.allocator.free(downstream.body);

        const job_id = try identity.currentJobId(self.allocator, service.executor_id);
        defer self.allocator.free(job_id);

        const body = try wrapJobJson(self.allocator, job_id, service, downstream.body);
        defer self.allocator.free(body);

        try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
    }

    fn handleTrainingJobSubmit(
        self: *Self,
        stream: net.Stream,
        req: *http_server.HttpRequest,
        forced_kind: ?training_workload.TrainingKind,
    ) !void {
        const kind = forced_kind orelse try training_workload.trainingKindFromRequestBody(self.allocator, req.body);
        const service_type: []const u8 = switch (kind) {
            .regular => service_registry.ServiceType.training.asString(),
            .custom_extension => custom_extensions.service_type_custom_training,
        };
        try self.handleJobSubmit(stream, req, service_type);
    }

    fn handleJobLookup(self: *Self, stream: net.Stream, req: *http_server.HttpRequest, job_id: []const u8) !void {
        if (self.gateway.localJobHookByJobId(job_id)) |hook| {
            var envelope = try hook.lookup(hook.ctx, self.allocator, job_id) orelse return error.JobNotFound;
            defer envelope.deinit();

            var service = try self.gateway.service_registry.findService(self.allocator, envelope.executor_id) orelse return error.ServiceUnavailable;
            defer service.deinit(self.allocator);

            const body = try wrapJobJson(self.allocator, envelope.job_id, service, envelope.job_json);
            defer self.allocator.free(body);

            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return;
        }

        var service = try self.requireJobService(job_id);
        defer service.deinit(self.allocator);

        const downstream = try self.forwardRaw(service, "GET", req.header("authorization"), "/v1/job", "", null);
        defer self.allocator.free(downstream.body);

        const canonical_job_id = try identity.currentJobId(self.allocator, service.executor_id);
        defer self.allocator.free(canonical_job_id);

        const body = try wrapJobJson(self.allocator, canonical_job_id, service, downstream.body);
        defer self.allocator.free(body);

        try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, body);
    }

    fn handleJobCancel(self: *Self, stream: net.Stream, req: *http_server.HttpRequest, job_id: []const u8) !void {
        if (self.gateway.localJobHookByJobId(job_id)) |hook| {
            var result = try hook.cancel(hook.ctx, self.allocator, job_id) orelse return error.JobNotFound;
            defer result.deinit();

            const body = try renderLocalCancelJson(self.allocator, result);
            defer self.allocator.free(body);

            const status = if (result.accepted and std.mem.eql(u8, result.status, "cancelling"))
                "202 Accepted"
            else
                "200 OK";
            try http_server.writeResponse(stream, status, &.{"Content-Type: application/json"}, body);
            return;
        }

        var service = try self.requireJobService(job_id);
        defer service.deinit(self.allocator);

        const downstream = try self.forwardRaw(service, "POST", req.header("authorization"), "/v1/job/cancel", req.body, req.header("content-type"));
        defer self.allocator.free(downstream.body);

        const canonical_job_id = try identity.currentJobId(self.allocator, service.executor_id);
        defer self.allocator.free(canonical_job_id);

        const body = try wrapCancelJson(self.allocator, canonical_job_id, service, downstream.body);
        defer self.allocator.free(body);

        try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, body);
    }

    fn handleReservationCreate(
        self: *Self,
        stream: net.Stream,
        req: *http_server.HttpRequest,
        service_type: []const u8,
    ) !void {
        const parsed = try parseReservationCreateRequest(self.allocator, req.body);
        defer if (parsed) |value| value.deinit();

        var service = try self.requireServiceByType(
            service_type,
            if (parsed) |value| value.value.service_id else null,
            if (parsed) |value| value.value.executor_id else null,
        );
        defer service.deinit(self.allocator);

        const hook = self.gateway.localJobHookForExecutor(service.executor_id, service.service_type) orelse return error.ServiceUnavailable;
        const reserve = hook.reserve orelse return error.UnsupportedMethod;

        var result = try reserve(hook.ctx, self.allocator, .{
            .workers_required = if (parsed) |value| value.value.workers_required else null,
        });
        defer result.deinit();

        const body = try renderLocalReservationJson(self.allocator, result);
        defer self.allocator.free(body);
        try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
    }

    fn handleReservationCommit(
        self: *Self,
        stream: net.Stream,
        req: *http_server.HttpRequest,
        reservation_id: []const u8,
    ) !void {
        const hook = self.gateway.localJobHookByReservationId(reservation_id) orelse return error.ReservationNotFound;
        const commit = hook.commit_reservation orelse return error.UnsupportedMethod;

        var result = try commit(hook.ctx, self.allocator, reservation_id, req.body) orelse return error.ReservationNotFound;
        defer result.deinit();

        var service = try self.gateway.service_registry.findService(self.allocator, result.envelope.executor_id) orelse return error.ServiceUnavailable;
        defer service.deinit(self.allocator);

        const body = try wrapSubmittedJobJson(self.allocator, result, service);
        defer self.allocator.free(body);
        try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
    }

    fn handleReservationRelease(
        self: *Self,
        stream: net.Stream,
        _: *http_server.HttpRequest,
        reservation_id: []const u8,
    ) !void {
        const hook = self.gateway.localJobHookByReservationId(reservation_id) orelse return error.ReservationNotFound;
        const release = hook.release_reservation orelse return error.UnsupportedMethod;

        var result = try release(hook.ctx, self.allocator, reservation_id) orelse return error.ReservationNotFound;
        defer result.deinit();

        const body = try renderLocalReservationReleaseJson(self.allocator, result);
        defer self.allocator.free(body);
        try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
    }

    fn handleQueryThenInfer(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !void {
        if (req.body.len == 0) return error.InvalidInferenceRequest;

        var parsed = try std.json.parseFromSlice(QueryThenInferRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        const graph_body = try renderFederatedGraphQueryJson(
            self.allocator,
            &self.gateway.graph,
            self.gateway.config.resolvedFederationHubEndpoint(),
            self.federation_hub_token,
            parsed.value.graph_query,
        );
        defer self.allocator.free(graph_body);

        const graph_context_prompt = try buildGraphContextPrompt(self.allocator, graph_body);
        defer self.allocator.free(graph_context_prompt);

        const forward_body = try buildQueryThenInferBody(self.allocator, parsed.value.inference, graph_context_prompt);
        defer self.allocator.free(forward_body);

        var service = try self.requireServiceByType(
            service_registry.ServiceType.inference.asString(),
            req.header("x-pcp-service-id"),
            req.header("x-pcp-executor-id"),
        );
        defer service.deinit(self.allocator);

        const downstream = try self.forwardRaw(
            service,
            "POST",
            req.header("authorization"),
            "/v1/chat/completions",
            forward_body,
            "application/json",
        );
        defer self.allocator.free(downstream.body);

        if (downstream.status != .ok and downstream.status != .accepted and downstream.status != .created) {
            try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, downstream.body);
            return;
        }

        const body = try wrapQueryThenInferResponse(
            self.allocator,
            graph_body,
            graph_context_prompt,
            downstream.body,
        );
        defer self.allocator.free(body);
        try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
    }

    fn renderJobsJson(self: *Self, req: *http_server.HttpRequest) ![]u8 {
        const services = try self.gateway.service_registry.listServices(self.allocator);
        defer service_registry.ServiceRegistry.deinitServiceList(self.allocator, services);

        var body = std.ArrayList(u8).init(self.allocator);
        errdefer body.deinit();
        var writer = body.writer();
        try writer.writeAll("{\"jobs\":[");
        var emitted: usize = 0;

        for (services) |service| {
            if (self.gateway.localJobHookForExecutor(service.executor_id, service.service_type)) |hook| {
                const local_jobs = try hook.list(hook.ctx, self.allocator);
                defer {
                    for (local_jobs) |*job| job.deinit();
                    self.allocator.free(local_jobs);
                }

                for (local_jobs) |job| {
                    const wrapped = try wrapJobJson(self.allocator, job.job_id, service, job.job_json);
                    defer self.allocator.free(wrapped);

                    if (emitted > 0) try writer.writeByte(',');
                    try writer.writeAll(wrapped);
                    emitted += 1;
                }
                continue;
            }

            const downstream = self.forwardRaw(service, "GET", req.header("authorization"), "/v1/job", "", null) catch |err| switch (err) {
                error.ServiceProxyFailed,
                => continue,
                else => return err,
            };
            defer self.allocator.free(downstream.body);

            const job_id = try identity.currentJobId(self.allocator, service.executor_id);
            defer self.allocator.free(job_id);

            const wrapped = try wrapJobJson(self.allocator, job_id, service, downstream.body);
            defer self.allocator.free(wrapped);

            if (emitted > 0) try writer.writeByte(',');
            try writer.writeAll(wrapped);
            emitted += 1;
        }

        try writer.writeAll("]}");
        return body.toOwnedSlice();
    }

    fn requireServiceByType(
        self: *Self,
        service_type: []const u8,
        preferred_service_id: ?[]const u8,
        preferred_executor_id: ?[]const u8,
    ) !service_registry.RegisteredService {
        return try self.gateway.service_registry.selectServiceByType(
            self.allocator,
            service_type,
            preferred_service_id,
            preferred_executor_id,
        ) orelse error.ServiceUnavailable;
    }

    fn requireJobService(self: *Self, job_id: []const u8) !service_registry.RegisteredService {
        const services = try self.gateway.service_registry.listServices(self.allocator);
        defer service_registry.ServiceRegistry.deinitServiceList(self.allocator, services);

        for (services) |service| {
            const current_job_id = try identity.currentJobId(self.allocator, service.executor_id);
            defer self.allocator.free(current_job_id);
            if (!std.mem.eql(u8, current_job_id, job_id)) continue;
            return try self.gateway.service_registry.findService(self.allocator, service.executor_id) orelse error.ServiceUnavailable;
        }

        return error.JobNotFound;
    }

    fn forwardToService(self: *Self, service: service_registry.RegisteredService, req: *http_server.HttpRequest, path: []const u8) !DownstreamResponse {
        return self.forwardRaw(service, req.method, req.header("authorization"), path, req.body, req.header("content-type"));
    }

    fn forwardRaw(
        self: *Self,
        service: service_registry.RegisteredService,
        method: []const u8,
        auth_header: ?[]const u8,
        path: []const u8,
        body: []const u8,
        content_type: ?[]const u8,
    ) !DownstreamResponse {
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ std.mem.trimRight(u8, service.base_url, "/"), path });
        defer self.allocator.free(url);

        var headers = std.ArrayList(std.http.Header).init(self.allocator);
        defer headers.deinit();
        try headers.append(.{ .name = "accept", .value = "application/json" });
        if (content_type) |value| {
            try headers.append(.{ .name = "content-type", .value = value });
        }
        if (auth_header) |value| {
            try headers.append(.{ .name = "authorization", .value = value });
        }

        var response_body = std.ArrayList(u8).init(self.allocator);
        errdefer response_body.deinit();

        const result = client.fetch(.{
            .location = .{ .url = url },
            .method = try parseHttpMethod(method),
            .payload = if (body.len > 0) body else null,
            .extra_headers = headers.items,
            .response_storage = .{ .dynamic = &response_body },
        }) catch return error.ServiceProxyFailed;

        return .{
            .status = result.status,
            .body = try response_body.toOwnedSlice(),
        };
    }
};

pub const GatewayProbeServer = struct {
    allocator: Allocator,
    api_server: *GatewayApiServer,
    server: ?TcpServer,
    listen_host: ?[]const u8,
    listen_port: u16,
    is_running: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(allocator: Allocator, api_server: *GatewayApiServer) Self {
        return .{
            .allocator = allocator,
            .api_server = api_server,
            .server = null,
            .listen_host = null,
            .listen_port = 0,
            .is_running = std.atomic.Value(u8).init(0),
        };
    }

    pub fn start(self: *Self, host: []const u8, port: u16) !void {
        self.listen_host = host;
        self.listen_port = port;
        std.log.info("Starting gateway probe server on {s}:{d}", .{ host, port });
        self.server = try TcpServer.init(self.allocator, host, port);
        self.is_running.store(1, .release);

        while (self.is_running.load(.acquire) == 1) {
            const connection = if (self.server) |*server|
                server.accept() catch |err| {
                    if (self.is_running.load(.acquire) == 0) break;
                    std.log.err("Gateway probe accept failed: {}", .{err});
                    continue;
                }
            else
                break;

            if (self.is_running.load(.acquire) == 0) {
                connection.stream.close();
                break;
            }

            const thread = std.Thread.spawn(.{}, handleProbeConnection, .{ self, connection.stream }) catch |err| {
                std.log.err("Failed to spawn gateway probe handler thread: {}", .{err});
                connection.stream.close();
                continue;
            };
            thread.detach();
        }

        if (self.server) |*server| {
            server.deinit();
            self.server = null;
        }
    }

    pub fn stop(self: *Self) void {
        self.is_running.store(0, .release);
        if (self.listen_host) |host| {
            const address = net.Address.parseIp(host, self.listen_port) catch return;
            const stream = net.tcpConnectToAddress(address) catch return;
            stream.close();
        }
    }

    fn handleProbeConnection(self: *Self, stream: net.Stream) void {
        defer stream.close();

        var req = http_server.readRequest(stream, self.allocator, 1024 * 1024) catch |err| {
            std.log.warn("Gateway probe failed to read request: {}", .{err});
            return;
        };
        defer req.deinit();

        const handled = self.api_server.handleProbeRequest(stream, &req) catch |err| {
            std.log.err("Gateway probe route failed: {}", .{err});
            _ = http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch {};
            return;
        };
        if (!handled) {
            _ = http_server.writeResponse(stream, "404 Not Found", &.{"Content-Type: text/plain"}, "not_found") catch {};
        }
    }
};

fn parseProxySelection(allocator: Allocator, body: []const u8) !?std.json.Parsed(ProxySelectionRequest) {
    if (body.len == 0) return null;
    return std.json.parseFromSlice(ProxySelectionRequest, allocator, body, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidJobRequest,
    };
}

fn parseReservationCreateRequest(allocator: Allocator, body: []const u8) !?std.json.Parsed(ReservationCreateRequest) {
    if (body.len == 0) return null;
    return std.json.parseFromSlice(ReservationCreateRequest, allocator, body, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidReservationRequest,
    };
}

fn wrapJobJson(allocator: Allocator, job_id: []const u8, service: service_registry.RegisteredService, downstream_body: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, downstream_body, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidDownstreamResponse;

    const job_id_json = try std.json.stringifyAlloc(allocator, job_id, .{});
    defer allocator.free(job_id_json);
    const service_id_json = try std.json.stringifyAlloc(allocator, service.service_id, .{});
    defer allocator.free(service_id_json);
    const executor_id_json = try std.json.stringifyAlloc(allocator, service.executor_id, .{});
    defer allocator.free(executor_id_json);
    const service_type_json = try std.json.stringifyAlloc(allocator, service.service_type, .{});
    defer allocator.free(service_type_json);
    const base_url_json = try std.json.stringifyAlloc(allocator, service.base_url, .{});
    defer allocator.free(base_url_json);

    var body = std.ArrayList(u8).init(allocator);
    errdefer body.deinit();
    var writer = body.writer();
    try writer.writeAll("{\"job_id\":");
    try writer.writeAll(job_id_json);
    try writer.writeAll(",\"service_id\":");
    try writer.writeAll(service_id_json);
    try writer.writeAll(",\"executor_id\":");
    try writer.writeAll(executor_id_json);
    try writer.writeAll(",\"service_type\":");
    try writer.writeAll(service_type_json);
    try writer.writeAll(",\"base_url\":");
    try writer.writeAll(base_url_json);
    try writer.writeAll(",\"job\":");
    try writer.writeAll(downstream_body);
    try writer.writeByte('}');
    return body.toOwnedSlice();
}

fn wrapSubmittedJobJson(
    allocator: Allocator,
    result: gateway_mod.LocalJobSubmitResult,
    service: service_registry.RegisteredService,
) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, result.envelope.job_json, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidDownstreamResponse;

    const job_id_json = try std.json.stringifyAlloc(allocator, result.envelope.job_id, .{});
    defer allocator.free(job_id_json);
    const service_id_json = try std.json.stringifyAlloc(allocator, service.service_id, .{});
    defer allocator.free(service_id_json);
    const executor_id_json = try std.json.stringifyAlloc(allocator, service.executor_id, .{});
    defer allocator.free(executor_id_json);
    const service_type_json = try std.json.stringifyAlloc(allocator, service.service_type, .{});
    defer allocator.free(service_type_json);
    const base_url_json = try std.json.stringifyAlloc(allocator, service.base_url, .{});
    defer allocator.free(base_url_json);

    var body = std.ArrayList(u8).init(allocator);
    errdefer body.deinit();
    var writer = body.writer();
    try writer.writeAll("{\"accepted\":");
    try writer.writeAll(if (result.accepted) "true" else "false");
    try writer.writeAll(",\"queue_position\":");
    try writer.print("{d}", .{result.queue_position});
    try writer.writeAll(",\"job_id\":");
    try writer.writeAll(job_id_json);
    try writer.writeAll(",\"service_id\":");
    try writer.writeAll(service_id_json);
    try writer.writeAll(",\"executor_id\":");
    try writer.writeAll(executor_id_json);
    try writer.writeAll(",\"service_type\":");
    try writer.writeAll(service_type_json);
    try writer.writeAll(",\"base_url\":");
    try writer.writeAll(base_url_json);
    try writer.writeAll(",\"job\":");
    try writer.writeAll(result.envelope.job_json);
    try writer.writeByte('}');
    return body.toOwnedSlice();
}

fn wrapCancelJson(allocator: Allocator, job_id: []const u8, service: service_registry.RegisteredService, downstream_body: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, downstream_body, .{});
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |object| object,
        else => return error.InvalidDownstreamResponse,
    };

    const accepted = if (root.get("accepted")) |value|
        switch (value) {
            .bool => |inner| inner,
            else => false,
        }
    else
        false;
    const status = if (root.get("status")) |value|
        switch (value) {
            .string => |inner| inner,
            else => "unknown",
        }
    else
        "unknown";

    return std.json.stringifyAlloc(allocator, .{
        .accepted = accepted,
        .status = status,
        .job_id = job_id,
        .service_id = service.service_id,
        .executor_id = service.executor_id,
        .service_type = service.service_type,
    }, .{});
}

fn renderLocalCancelJson(allocator: Allocator, result: gateway_mod.LocalJobCancelResult) ![]u8 {
    return std.json.stringifyAlloc(allocator, .{
        .accepted = result.accepted,
        .status = result.status,
        .job_id = result.job_id,
        .service_id = result.service_id,
        .executor_id = result.executor_id,
        .service_type = result.service_type,
    }, .{});
}

fn renderLocalReservationJson(allocator: Allocator, result: gateway_mod.LocalReservationResult) ![]u8 {
    return std.json.stringifyAlloc(allocator, .{
        .accepted = result.accepted,
        .reservation_id = result.reservation_id,
        .service_id = result.service_id,
        .executor_id = result.executor_id,
        .service_type = result.service_type,
        .workers_required = result.workers_required,
        .reserved_at = result.reserved_at,
    }, .{});
}

fn renderLocalReservationReleaseJson(allocator: Allocator, result: gateway_mod.LocalReservationReleaseResult) ![]u8 {
    return std.json.stringifyAlloc(allocator, .{
        .accepted = result.accepted,
        .status = result.status,
        .reservation_id = result.reservation_id,
        .service_id = result.service_id,
        .executor_id = result.executor_id,
        .service_type = result.service_type,
    }, .{});
}

fn writeProxyErrorResponse(stream: net.Stream, err: anyerror) !void {
    const status = switch (err) {
        error.InvalidJobId,
        error.InvalidDownstreamResponse,
        error.InvalidJobRequest,
        error.InvalidReservationRequest,
        error.UnsupportedJobOverrides,
        => "400 Bad Request",
        error.JobNotFound,
        error.ReservationNotFound,
        error.ServiceUnavailable,
        => "404 Not Found",
        error.ServiceProxyFailed => "502 Bad Gateway",
        error.UnsupportedMethod => "405 Method Not Allowed",
        error.ExecutorBusy,
        error.ReservationUnavailable,
        => "409 Conflict",
        else => "500 Internal Server Error",
    };
    try http_server.writeResponse(stream, status, &.{"Content-Type: text/plain"}, @errorName(err));
}

fn buildGraphContextPrompt(allocator: Allocator, graph_body: []const u8) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "Use the following graph context when it is relevant. Do not invent facts that are not present in it.\nGraph context JSON:\n{s}",
        .{graph_body},
    );
}

fn buildQueryThenInferBody(allocator: Allocator, inference_value: std.json.Value, graph_context_prompt: []const u8) ![]u8 {
    const inference_object = switch (inference_value) {
        .object => |object| object,
        else => return error.InvalidInferenceRequest,
    };

    const messages_value = inference_object.get("messages") orelse return error.MissingMessages;
    const original_messages = switch (messages_value) {
        .array => |array| array,
        else => return error.InvalidMessages,
    };

    if (inference_object.get("stream")) |stream_value| {
        switch (stream_value) {
            .bool => |enabled| if (enabled) return error.StreamingNotSupported,
            else => return error.InvalidInferenceRequest,
        }
    }

    var forward_object = std.json.ObjectMap.init(allocator);
    defer forward_object.deinit();

    var it = inference_object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "messages")) continue;
        try forward_object.put(entry.key_ptr.*, entry.value_ptr.*);
    }

    var messages = std.json.Array.init(allocator);
    defer messages.deinit();

    var system_message = std.json.ObjectMap.init(allocator);
    defer system_message.deinit();
    try system_message.put("role", .{ .string = "system" });
    try system_message.put("content", .{ .string = graph_context_prompt });
    try messages.append(.{ .object = system_message });

    for (original_messages.items) |message| {
        try messages.append(message);
    }

    try forward_object.put("messages", .{ .array = messages });
    return jsonStringifyValue(allocator, .{ .object = forward_object });
}

fn wrapQueryThenInferResponse(
    allocator: Allocator,
    graph_body: []const u8,
    graph_context_prompt: []const u8,
    completion_body: []const u8,
) ![]u8 {
    var graph_parsed = try std.json.parseFromSlice(std.json.Value, allocator, graph_body, .{});
    defer graph_parsed.deinit();
    var completion_parsed = try std.json.parseFromSlice(std.json.Value, allocator, completion_body, .{});
    defer completion_parsed.deinit();

    if (graph_parsed.value != .object) return error.InvalidFederationHubResponse;
    if (completion_parsed.value != .object) return error.InvalidDownstreamResponse;

    var root = std.json.ObjectMap.init(allocator);
    defer root.deinit();
    try root.put("graph_query", graph_parsed.value);
    try root.put("graph_context_prompt", .{ .string = graph_context_prompt });
    try root.put("completion", completion_parsed.value);
    return jsonStringifyValue(allocator, .{ .object = root });
}

fn jsonStringifyValue(allocator: Allocator, value: std.json.Value) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    try std.json.stringify(value, .{}, buf.writer());
    return buf.toOwnedSlice();
}

fn renderFederatedGraphQueryJson(
    allocator: Allocator,
    graph: *graph_adapter.GatewayGraph,
    federation_hub_endpoint: ?[]const u8,
    federation_hub_token: ?[]const u8,
    request: graph_types.QueryRequest,
) ![]u8 {
    const mode = try request.resolvedMode();
    return switch (mode) {
        .local => graph.renderQueryJson(allocator, request.withMode(.local)),
        .global => {
            const endpoint = federation_hub_endpoint orelse return error.FederationHubNotConfigured;
            return queryFederationHub(allocator, endpoint, federation_hub_token, request.withMode(.global));
        },
        .local_plus_global => {
            const endpoint = federation_hub_endpoint orelse return error.FederationHubNotConfigured;
            const local_body = try graph.renderQueryJson(allocator, request.withMode(.local));
            defer allocator.free(local_body);
            const global_body = try queryFederationHub(allocator, endpoint, federation_hub_token, request.withMode(.global));
            defer allocator.free(global_body);
            return mergeQueryResponses(allocator, local_body, global_body);
        },
    };
}

fn queryFederationHub(
    allocator: Allocator,
    endpoint: []const u8,
    auth_token: ?[]const u8,
    request: graph_types.QueryRequest,
) ![]u8 {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const url = try std.fmt.allocPrint(
        allocator,
        "{s}/v1/global-graph/query",
        .{std.mem.trimRight(u8, endpoint, "/")},
    );
    defer allocator.free(url);

    const body = try std.json.stringifyAlloc(allocator, request, .{});
    defer allocator.free(body);

    var auth_header_value: ?[]u8 = null;
    defer if (auth_header_value) |value| allocator.free(value);

    var headers = std.ArrayList(std.http.Header).init(allocator);
    defer headers.deinit();
    try headers.append(.{ .name = "content-type", .value = "application/json" });
    try headers.append(.{ .name = "accept", .value = "application/json" });
    if (auth_token) |token| {
        auth_header_value = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token});
        try headers.append(.{ .name = "authorization", .value = auth_header_value.? });
    }

    var response_body = std.ArrayList(u8).init(allocator);
    errdefer response_body.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .extra_headers = headers.items,
        .response_storage = .{ .dynamic = &response_body },
    }) catch return error.FederationHubQueryFailed;

    if (result.status != .ok and result.status != .accepted and result.status != .created) {
        return error.FederationHubQueryFailed;
    }

    return response_body.toOwnedSlice();
}

fn mergeQueryResponses(allocator: Allocator, local_body: []const u8, global_body: []const u8) ![]u8 {
    var local_parsed = try std.json.parseFromSlice(std.json.Value, allocator, local_body, .{});
    defer local_parsed.deinit();
    var global_parsed = try std.json.parseFromSlice(std.json.Value, allocator, global_body, .{});
    defer global_parsed.deinit();

    const local_root = switch (local_parsed.value) {
        .object => |object| object,
        else => return error.InvalidFederationHubResponse,
    };
    const global_root = switch (global_parsed.value) {
        .object => |object| object,
        else => return error.InvalidFederationHubResponse,
    };

    var body = std.ArrayList(u8).init(allocator);
    errdefer body.deinit();
    var writer = body.writer();

    try writer.writeAll("{\"entities\":[");
    try writeMergedQueryArray(allocator, writer, local_root, global_root, "entities", "entity_id");
    try writer.writeAll("],\"relations\":[");
    try writeMergedQueryArray(allocator, writer, local_root, global_root, "relations", "relation_id");
    try writer.writeAll("],\"observations\":[");
    try writeMergedQueryArray(allocator, writer, local_root, global_root, "observations", "observation_id");
    try writer.writeAll("]}");

    return body.toOwnedSlice();
}

fn writeMergedQueryArray(
    allocator: Allocator,
    writer: anytype,
    local_root: std.json.ObjectMap,
    global_root: std.json.ObjectMap,
    array_name: []const u8,
    id_field: []const u8,
) !void {
    const local_items = try queryArrayField(local_root, array_name);
    const global_items = try queryArrayField(global_root, array_name);

    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    var emitted: usize = 0;
    for (local_items) |item| {
        const key = try queryItemId(item, id_field);
        if (seen.contains(key)) continue;
        try seen.put(key, {});
        if (emitted > 0) try writer.writeByte(',');
        try std.json.stringify(item, .{}, writer);
        emitted += 1;
    }

    for (global_items) |item| {
        const key = try queryItemId(item, id_field);
        if (seen.contains(key)) continue;
        try seen.put(key, {});
        if (emitted > 0) try writer.writeByte(',');
        try std.json.stringify(item, .{}, writer);
        emitted += 1;
    }
}

fn queryArrayField(root: std.json.ObjectMap, name: []const u8) ![]const std.json.Value {
    const value = root.get(name) orelse return error.InvalidFederationHubResponse;
    return switch (value) {
        .array => |array| array.items,
        else => return error.InvalidFederationHubResponse,
    };
}

fn queryItemId(item: std.json.Value, field_name: []const u8) ![]const u8 {
    const object = switch (item) {
        .object => |object| object,
        else => return error.InvalidFederationHubResponse,
    };
    const field = object.get(field_name) orelse return error.InvalidFederationHubResponse;
    return switch (field) {
        .string => |text| text,
        else => return error.InvalidFederationHubResponse,
    };
}
