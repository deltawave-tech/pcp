const std = @import("std");
const net = std.net;

const Allocator = std.mem.Allocator;
const TcpServer = @import("../../network/tcp_stream.zig").TcpServer;
const http_server = @import("../../inference/http_server.zig");
const graph_policy_store = @import("../../graph/policy_store.zig");
const graph_types = @import("../../graph/types.zig");
const service_registry = @import("service_registry.zig");
const graph_adapter = @import("graph_adapter.zig");
const gateway_mod = @import("gateway.zig");
const event_ingest = @import("event_ingest.zig");

const ProxySelectionRequest = struct {
    service_id: ?[]const u8 = null,
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
    internal_api_token: ?[]const u8,
    federation_hub_token: ?[]const u8,
    server: ?TcpServer,
    listen_host: ?[]const u8,
    listen_port: u16,
    is_running: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        gateway: *gateway_mod.Gateway,
        api_token: ?[]const u8,
        internal_api_token: ?[]const u8,
        federation_hub_token: ?[]const u8,
    ) Self {
        return .{
            .allocator = allocator,
            .gateway = gateway,
            .api_token = api_token,
            .internal_api_token = internal_api_token,
            .federation_hub_token = federation_hub_token,
            .server = null,
            .listen_host = null,
            .listen_port = 0,
            .is_running = std.atomic.Value(u8).init(0),
        };
    }

    pub fn start(self: *Self, host: []const u8, port: u16) !void {
        self.listen_host = host;
        self.listen_port = port;
        std.log.info("Starting gateway API server on {s}:{d}", .{ host, port });
        self.server = TcpServer.init(self.allocator, host, port) catch |err| {
            std.log.err("Gateway API listen failed on {s}:{d}: {}", .{ host, port, err });
            return err;
        };
        self.is_running.store(1, .release);

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

        var req = http_server.readRequest(stream, self.allocator, 1024 * 1024) catch |err| {
            std.log.warn("Gateway API failed to read request: {}", .{err});
            return;
        };
        defer req.deinit();

        _ = self.handleRequest(stream, &req) catch |err| {
            std.log.err("Gateway API route failed: {}", .{err});
            _ = http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch {};
            return;
        };
    }

    fn handleRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (std.mem.eql(u8, req.path, "/healthz")) {
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain"}, "ok");
            return true;
        }

        if (std.mem.eql(u8, req.path, "/readyz")) {
            const body = try self.gateway.renderReadyJson(self.allocator, self.api_token != null);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

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

        if (!std.mem.startsWith(u8, req.path, "/v1/")) return false;

        if (!self.authorize(req)) {
            try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
            return true;
        }

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
            const service_id = req.path["/v1/services/".len..];
            const body = try self.gateway.service_registry.renderServiceJson(self.allocator, service_id);
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

            var service = self.gateway.service_registry.register(parsed.value) catch |err| switch (err) {
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
                    .service_type = service.service_type.asString(),
                    .base_url = service.base_url,
                    .auth_mode = service.auth_mode,
                    .health_status = service.health_status,
                    .job_status = service.job_status,
                    .worker_count = service.worker_count,
                    .ready_worker_count = service.ready_worker_count,
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
            var service = self.requireServiceByType(.inference, req.header("x-pcp-service-id")) catch |err| {
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
            self.handleJobSubmit(stream, req, .rl) catch |err| {
                try writeProxyErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/training/jobs")) {
            self.handleJobSubmit(stream, req, .training) catch |err| {
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

    fn authorize(self: *Self, req: *http_server.HttpRequest) bool {
        if (self.api_token == null) return true;
        const header = req.header("authorization") orelse return false;
        if (!std.mem.startsWith(u8, header, "Bearer ")) return false;
        return std.mem.eql(u8, header["Bearer ".len..], self.api_token.?);
    }

    fn authorizeInternal(self: *Self, req: *http_server.HttpRequest) bool {
        if (self.internal_api_token) |token| {
            const header = req.header("authorization") orelse return false;
            if (!std.mem.startsWith(u8, header, "Bearer ")) return false;
            return std.mem.eql(u8, header["Bearer ".len..], token);
        }
        return self.authorize(req);
    }

    fn handleJobSubmit(self: *Self, stream: net.Stream, req: *http_server.HttpRequest, service_type: service_registry.ServiceType) !void {
        const selection = try parseProxySelection(self.allocator, req.body);
        defer if (selection) |parsed| parsed.deinit();

        var service = try self.requireServiceByType(service_type, if (selection) |parsed| parsed.value.service_id else null);
        defer service.deinit(self.allocator);

        const downstream = try self.forwardRaw(service, req.method, req.header("authorization"), "/v1/job", "", null);
        defer self.allocator.free(downstream.body);

        const job_id = try gatewayJobId(self.allocator, service.service_type, service.service_id);
        defer self.allocator.free(job_id);

        const body = try wrapJobJson(self.allocator, job_id, service, downstream.body);
        defer self.allocator.free(body);

        try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
    }

    fn handleJobLookup(self: *Self, stream: net.Stream, req: *http_server.HttpRequest, job_id: []const u8) !void {
        var service = try self.requireJobService(job_id);
        defer service.deinit(self.allocator);

        const downstream = try self.forwardRaw(service, "GET", req.header("authorization"), "/v1/job", "", null);
        defer self.allocator.free(downstream.body);

        const canonical_job_id = try gatewayJobId(self.allocator, service.service_type, service.service_id);
        defer self.allocator.free(canonical_job_id);

        const body = try wrapJobJson(self.allocator, canonical_job_id, service, downstream.body);
        defer self.allocator.free(body);

        try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, body);
    }

    fn handleJobCancel(self: *Self, stream: net.Stream, req: *http_server.HttpRequest, job_id: []const u8) !void {
        var service = try self.requireJobService(job_id);
        defer service.deinit(self.allocator);

        const downstream = try self.forwardRaw(service, "POST", req.header("authorization"), "/v1/job/cancel", req.body, req.header("content-type"));
        defer self.allocator.free(downstream.body);

        const canonical_job_id = try gatewayJobId(self.allocator, service.service_type, service.service_id);
        defer self.allocator.free(canonical_job_id);

        const body = try wrapCancelJson(self.allocator, canonical_job_id, service, downstream.body);
        defer self.allocator.free(body);

        try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, body);
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

        var service = try self.requireServiceByType(.inference, req.header("x-pcp-service-id"));
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
            const downstream = self.forwardRaw(service, "GET", req.header("authorization"), "/v1/job", "", null) catch |err| switch (err) {
                error.ServiceProxyFailed,
                => continue,
                else => return err,
            };
            defer self.allocator.free(downstream.body);

            const job_id = try gatewayJobId(self.allocator, service.service_type, service.service_id);
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

    fn requireServiceByType(self: *Self, service_type: service_registry.ServiceType, preferred_service_id: ?[]const u8) !service_registry.RegisteredService {
        return try self.gateway.service_registry.selectServiceByType(self.allocator, service_type, preferred_service_id) orelse error.ServiceUnavailable;
    }

    fn requireJobService(self: *Self, job_id: []const u8) !service_registry.RegisteredService {
        if (std.mem.indexOfScalar(u8, job_id, ':')) |sep| {
            const service_type = service_registry.ServiceType.parse(job_id[0..sep]) orelse return error.InvalidJobId;
            const service_id = job_id[sep + 1 ..];
            var service = try self.gateway.service_registry.findService(self.allocator, service_id) orelse return error.ServiceUnavailable;
            errdefer service.deinit(self.allocator);
            if (service.service_type != service_type) return error.InvalidJobId;
            return service;
        }

        return try self.gateway.service_registry.findService(self.allocator, job_id) orelse error.ServiceUnavailable;
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

fn parseProxySelection(allocator: Allocator, body: []const u8) !?std.json.Parsed(ProxySelectionRequest) {
    if (body.len == 0) return null;
    return try std.json.parseFromSlice(ProxySelectionRequest, allocator, body, .{ .ignore_unknown_fields = true });
}

fn gatewayJobId(allocator: Allocator, service_type: service_registry.ServiceType, service_id: []const u8) ![]u8 {
    _ = service_type;
    return allocator.dupe(u8, service_id);
}

fn wrapJobJson(allocator: Allocator, job_id: []const u8, service: service_registry.RegisteredService, downstream_body: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, downstream_body, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidDownstreamResponse;

    const job_id_json = try std.json.stringifyAlloc(allocator, job_id, .{});
    defer allocator.free(job_id_json);
    const service_id_json = try std.json.stringifyAlloc(allocator, service.service_id, .{});
    defer allocator.free(service_id_json);
    const service_type_json = try std.json.stringifyAlloc(allocator, service.service_type.asString(), .{});
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
    try writer.writeAll(",\"service_type\":");
    try writer.writeAll(service_type_json);
    try writer.writeAll(",\"base_url\":");
    try writer.writeAll(base_url_json);
    try writer.writeAll(",\"job\":");
    try writer.writeAll(downstream_body);
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
        .service_type = service.service_type.asString(),
    }, .{});
}

fn parseHttpMethod(method: []const u8) !std.http.Method {
    if (std.mem.eql(u8, method, "GET")) return .GET;
    if (std.mem.eql(u8, method, "POST")) return .POST;
    if (std.mem.eql(u8, method, "PUT")) return .PUT;
    if (std.mem.eql(u8, method, "DELETE")) return .DELETE;
    return error.UnsupportedMethod;
}

fn statusText(status: std.http.Status) []const u8 {
    return switch (status) {
        .ok => "200 OK",
        .created => "201 Created",
        .accepted => "202 Accepted",
        .bad_request => "400 Bad Request",
        .unauthorized => "401 Unauthorized",
        .forbidden => "403 Forbidden",
        .not_found => "404 Not Found",
        .method_not_allowed => "405 Method Not Allowed",
        .conflict => "409 Conflict",
        .unprocessable_entity => "422 Unprocessable Entity",
        .too_many_requests => "429 Too Many Requests",
        .internal_server_error => "500 Internal Server Error",
        .bad_gateway => "502 Bad Gateway",
        .service_unavailable => "503 Service Unavailable",
        .gateway_timeout => "504 Gateway Timeout",
        else => "500 Internal Server Error",
    };
}

fn writeProxyErrorResponse(stream: net.Stream, err: anyerror) !void {
    const status = switch (err) {
        error.InvalidJobId, error.InvalidDownstreamResponse => "400 Bad Request",
        error.ServiceUnavailable => "404 Not Found",
        error.ServiceProxyFailed => "502 Bad Gateway",
        error.UnsupportedMethod => "405 Method Not Allowed",
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
