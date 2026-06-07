const std = @import("std");
const net = std.net;

const Allocator = std.mem.Allocator;
const federation_types = @import("../../protocol/federation/types.zig");
const TcpServer = @import("../../network/tcp_stream.zig").TcpServer;
const http_server = @import("../../network/http_server.zig");
const http_util = @import("../../protocol/http_util.zig");
const json_util = @import("../../protocol/json_util.zig");
const training_workload = @import("../../workloads/training/workload.zig");
const training_extensions = @import("../../workloads/training/custom_extensions.zig");
const hub_mod = @import("hub.zig");
const gateway_registry = @import("gateway_registry.zig");
const graph_types = @import("../../protocol/federation/graph/types.zig");
const bearerToken = http_util.bearerToken;
const parseHttpMethod = http_util.parseHttpMethod;
const statusText = http_util.statusText;
const stringField = json_util.stringField;
const usizeField = json_util.usizeField;

const DownstreamResponse = struct {
    status: std.http.Status,
    body: []u8,
};

const ParsedGlobalJobRequest = struct {
    placement: federation_types.PlacementRequest,
    job_body: []u8,

    pub fn deinit(self: *ParsedGlobalJobRequest, allocator: Allocator) void {
        freePlacement(&self.placement, allocator);
        allocator.free(self.job_body);
    }
};

pub const FederationHubApiServer = struct {
    allocator: Allocator,
    controller: *hub_mod.FederationHub,
    api_token: ?[]const u8,
    server: ?TcpServer,
    listen_host: ?[]const u8,
    listen_port: u16,
    is_running: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(allocator: Allocator, controller: *hub_mod.FederationHub, api_token: ?[]const u8) Self {
        return .{
            .allocator = allocator,
            .controller = controller,
            .api_token = api_token,
            .server = null,
            .listen_host = null,
            .listen_port = 0,
            .is_running = std.atomic.Value(u8).init(0),
        };
    }

    pub fn start(self: *Self, host: []const u8, port: u16) !void {
        self.listen_host = host;
        self.listen_port = port;
        self.server = try TcpServer.init(self.allocator, host, port);
        self.is_running.store(1, .release);

        while (self.is_running.load(.acquire) == 1) {
            const connection = if (self.server) |*server|
                server.accept() catch |err| {
                    if (self.is_running.load(.acquire) == 0) break;
                    std.log.err("Federation Hub API accept failed: {}", .{err});
                    continue;
                }
            else
                break;

            if (self.is_running.load(.acquire) == 0) {
                connection.stream.close();
                break;
            }

            const thread = std.Thread.spawn(.{}, handleConnection, .{ self, connection.stream }) catch |err| {
                std.log.err("Failed to spawn federation hub API handler thread: {}", .{err});
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
            std.log.warn("Federation Hub API failed to read request: {}", .{err});
            return;
        };
        defer req.deinit();

        _ = self.handleRequest(stream, &req) catch |err| {
            std.log.err("Federation Hub API route failed: {}", .{err});
            _ = http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch {};
            return;
        };
    }

    fn handleRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (try self.handleProbeRequest(stream, req)) return true;
        if (!std.mem.startsWith(u8, req.path, "/v1/")) return false;
        if (!self.authorize(req)) {
            try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
            return true;
        }
        return try self.handleAuthorizedV1Request(stream, req);
    }

    fn handleProbeRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (std.mem.eql(u8, req.path, "/healthz")) {
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain"}, "ok");
            return true;
        }

        if (std.mem.eql(u8, req.path, "/readyz")) {
            const body = try self.controller.renderReadyJson(self.allocator, self.api_token != null);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }
        return false;
    }

    fn handleAuthorizedV1Request(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/controller")) {
            const body = try self.controller.renderControllerJson(self.allocator, self.api_token != null);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/training/jobs")) {
            self.handleGlobalTrainingJobSubmit(stream, req) catch |err| {
                try writeFederationJobErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/rl/jobs")) {
            self.handleGlobalJobSubmit(stream, req, "rl", "/v1/internal/federation/rl/jobs") catch |err| {
                try writeFederationJobErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/jobs")) {
            const body = self.renderGlobalJobsJson(req.header("authorization")) catch |err| {
                try writeFederationJobErrorResponse(stream, err);
                return true;
            };
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.startsWith(u8, req.path, "/v1/jobs/") and std.mem.endsWith(u8, req.path, "/cancel")) {
            const global_job_id = req.path["/v1/jobs/".len .. req.path.len - "/cancel".len];
            self.handleGlobalJobCancel(stream, req, global_job_id) catch |err| {
                try writeFederationJobErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.startsWith(u8, req.path, "/v1/jobs/")) {
            const global_job_id = req.path["/v1/jobs/".len..];
            self.handleGlobalJobLookup(stream, req, global_job_id) catch |err| {
                try writeFederationJobErrorResponse(stream, err);
            };
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/federation/connect")) {
            if (req.body.len == 0) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "missing_body");
                return true;
            }

            var parsed = try std.json.parseFromSlice(gateway_registry.ConnectRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();

            var gateway = try self.controller.registry.upsert(parsed.value);
            defer gateway.deinit(self.allocator);

            const peers_body = try self.controller.registry.renderPeersJson(self.allocator);
            defer self.allocator.free(peers_body);

            var peers = try std.json.parseFromSlice(std.json.Value, self.allocator, peers_body, .{});
            defer peers.deinit();

            const gateway_services = try federation_types.advertisementsFromServiceRecords(self.allocator, gateway.services);
            defer self.allocator.free(gateway_services);

            const body = try std.json.stringifyAlloc(self.allocator, .{
                .accepted = true,
                .gateway = .{
                    .gateway_id = gateway.gateway_id,
                    .lab_id = gateway.lab_id,
                    .base_url = gateway.base_url,
                    .graph_backend = gateway.graph_backend,
                    .status = gateway.status,
                    .registered_services = gateway.registered_services,
                    .last_sequence_no = gateway.last_sequence_no,
                    .last_replicated_sequence = gateway.last_replicated_sequence,
                    .connected_at = gateway.connected_at,
                    .last_seen_at = gateway.last_seen_at,
                    .services = gateway_services,
                },
                .peers = peers.value,
            }, .{});
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/federation/peers")) {
            const body = try self.controller.registry.renderPeersJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/federation/services")) {
            const body = try self.controller.renderServicesJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/global-graph/status")) {
            const body = try self.controller.renderGlobalGraphStatusJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/global-graph/replication")) {
            const body = try self.controller.renderReplicationJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/graph/policies")) {
            const body = try self.controller.renderPoliciesJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/global-graph/query")) {
            if (req.body.len == 0) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "missing_body");
                return true;
            }

            var parsed = try std.json.parseFromSlice(graph_types.QueryRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();

            const body = try self.controller.renderGlobalGraphQueryJson(self.allocator, parsed.value);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/federation/mutations")) {
            if (req.body.len == 0) {
                try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, "missing_body");
                return true;
            }

            var parsed = try std.json.parseFromSlice(federation_types.MutationBatchRequest, self.allocator, req.body, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();

            const ack = self.controller.applyMutationBatch(self.allocator, parsed.value) catch |err| switch (err) {
                error.UnknownGateway, error.InvalidMutationType, error.InvalidVisibility, error.PolicyRejected => {
                    try http_server.writeResponse(stream, "400 Bad Request", &.{"Content-Type: text/plain"}, @errorName(err));
                    return true;
                },
                else => return err,
            };

            const body = try std.json.stringifyAlloc(self.allocator, ack, .{});
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
        const token = bearerToken(header) orelse return false;
        return std.mem.eql(u8, token, self.api_token.?);
    }

    fn handleGlobalJobSubmit(
        self: *Self,
        stream: net.Stream,
        req: *http_server.HttpRequest,
        service_type: []const u8,
        gateway_path: []const u8,
    ) !void {
        var parsed = try parseGlobalJobRequest(self.allocator, req.body);
        defer parsed.deinit(self.allocator);

        var selected = try self.controller.registry.selectServiceByType(self.allocator, service_type, parsed.placement) orelse return error.ServiceUnavailable;
        defer selected.deinit(self.allocator);

        const gateway_body = try buildGatewayJobBody(
            self.allocator,
            parsed.job_body,
            selected.service.service_id,
            selected.service.executor_id,
        );
        defer self.allocator.free(gateway_body);

        const downstream = try self.forwardToGateway(
            selected.gateway_base_url,
            "POST",
            gateway_path,
            gateway_body,
            "application/json",
            req.header("authorization"),
        );
        defer self.allocator.free(downstream.body);
        if (downstream.status != .ok and downstream.status != .accepted and downstream.status != .created) {
            try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, downstream.body);
            return;
        }

        const child_job_id = try extractRequiredStringField(self.allocator, downstream.body, "job_id");
        defer self.allocator.free(child_job_id);

        var record = try self.controller.jobs.create(
            self.allocator,
            selected.gateway_id,
            selected.lab_id,
            selected.gateway_base_url,
            selected.service.service_id,
            selected.service.executor_id,
            selected.service.service_type,
            child_job_id,
        );
        defer record.deinit(self.allocator);

        const body = try wrapGlobalJobJson(self.allocator, record, downstream.body);
        defer self.allocator.free(body);
        try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
    }

    fn handleGlobalTrainingJobSubmit(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !void {
        var parsed = try parseGlobalJobRequest(self.allocator, req.body);
        defer parsed.deinit(self.allocator);

        const kind = try training_extensions.trainingKindFromRequestBody(self.allocator, parsed.job_body);
        const capability: []const u8 = switch (kind) {
            .regular => training_workload.capability_regular,
            .custom_extension => training_extensions.capability_custom_extension,
        };
        const gateway_path: []const u8 = switch (kind) {
            .regular => "/v1/internal/federation/training/jobs",
            .custom_extension => training_extensions.route_custom_internal_training_jobs,
        };

        var selected = try self.controller.registry.selectServiceByCapability(self.allocator, capability, parsed.placement) orelse return error.ServiceUnavailable;
        defer selected.deinit(self.allocator);

        const gateway_body = try buildGatewayJobBody(
            self.allocator,
            parsed.job_body,
            selected.service.service_id,
            selected.service.executor_id,
        );
        defer self.allocator.free(gateway_body);

        const downstream = try self.forwardToGateway(
            selected.gateway_base_url,
            "POST",
            gateway_path,
            gateway_body,
            "application/json",
            req.header("authorization"),
        );
        defer self.allocator.free(downstream.body);
        if (downstream.status != .ok and downstream.status != .accepted and downstream.status != .created) {
            try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, downstream.body);
            return;
        }

        const child_job_id = try extractRequiredStringField(self.allocator, downstream.body, "job_id");
        defer self.allocator.free(child_job_id);

        var record = try self.controller.jobs.create(
            self.allocator,
            selected.gateway_id,
            selected.lab_id,
            selected.gateway_base_url,
            selected.service.service_id,
            selected.service.executor_id,
            selected.service.service_type,
            child_job_id,
        );
        defer record.deinit(self.allocator);

        const body = try wrapGlobalJobJson(self.allocator, record, downstream.body);
        defer self.allocator.free(body);
        try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
    }

    fn handleGlobalJobLookup(self: *Self, stream: net.Stream, req: *http_server.HttpRequest, global_job_id: []const u8) !void {
        var record = try self.controller.jobs.find(self.allocator, global_job_id) orelse return error.JobNotFound;
        defer record.deinit(self.allocator);

        const path = try std.fmt.allocPrint(self.allocator, "/v1/internal/federation/jobs/{s}", .{record.child_job_id});
        defer self.allocator.free(path);

        const downstream = try self.forwardToGateway(
            record.gateway_base_url,
            "GET",
            path,
            "",
            null,
            req.header("authorization"),
        );
        defer self.allocator.free(downstream.body);
        if (downstream.status == .not_found) return error.JobNotFound;

        const body = try wrapGlobalJobJson(self.allocator, record, downstream.body);
        defer self.allocator.free(body);
        try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, body);
    }

    fn handleGlobalJobCancel(
        self: *Self,
        stream: net.Stream,
        req: *http_server.HttpRequest,
        global_job_id: []const u8,
    ) !void {
        var record = try self.controller.jobs.find(self.allocator, global_job_id) orelse return error.JobNotFound;
        defer record.deinit(self.allocator);

        const path = try std.fmt.allocPrint(self.allocator, "/v1/internal/federation/jobs/{s}/cancel", .{record.child_job_id});
        defer self.allocator.free(path);

        const downstream = try self.forwardToGateway(
            record.gateway_base_url,
            "POST",
            path,
            req.body,
            req.header("content-type"),
            req.header("authorization"),
        );
        defer self.allocator.free(downstream.body);
        if (downstream.status == .not_found) return error.JobNotFound;

        const body = try wrapGlobalJobJson(self.allocator, record, downstream.body);
        defer self.allocator.free(body);
        try http_server.writeResponse(stream, statusText(downstream.status), &.{"Content-Type: application/json"}, body);
    }

    fn renderGlobalJobsJson(self: *Self, auth_header: ?[]const u8) ![]u8 {
        const jobs = try self.controller.jobs.list(self.allocator);
        defer hub_mod.GlobalJobStore.deinitList(self.allocator, jobs);

        var body = std.ArrayList(u8).init(self.allocator);
        errdefer body.deinit();
        var writer = body.writer();
        try writer.writeAll("{\"jobs\":[");
        var emitted: usize = 0;

        for (jobs) |job| {
            const path = try std.fmt.allocPrint(self.allocator, "/v1/internal/federation/jobs/{s}", .{job.child_job_id});
            defer self.allocator.free(path);

            const downstream = self.forwardToGateway(job.gateway_base_url, "GET", path, "", null, auth_header) catch continue;
            defer self.allocator.free(downstream.body);
            if (downstream.status == .not_found) continue;

            const wrapped = try wrapGlobalJobJson(self.allocator, job, downstream.body);
            defer self.allocator.free(wrapped);

            if (emitted > 0) try writer.writeByte(',');
            try writer.writeAll(wrapped);
            emitted += 1;
        }

        try writer.writeAll("]}");
        return body.toOwnedSlice();
    }

    fn forwardToGateway(
        self: *Self,
        gateway_base_url: []const u8,
        method: []const u8,
        path: []const u8,
        body: []const u8,
        content_type: ?[]const u8,
        auth_header: ?[]const u8,
    ) !DownstreamResponse {
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ std.mem.trimRight(u8, gateway_base_url, "/"), path });
        defer self.allocator.free(url);

        var headers = std.ArrayList(std.http.Header).init(self.allocator);
        defer headers.deinit();
        try headers.append(.{ .name = "accept", .value = "application/json" });
        if (content_type) |value| {
            try headers.append(.{ .name = "content-type", .value = value });
        }
        if (auth_header) |header| {
            try headers.append(.{ .name = "authorization", .value = header });
        } else if (self.api_token) |token| {
            const auth_header_value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
            defer self.allocator.free(auth_header_value);
            try headers.append(.{ .name = "authorization", .value = auth_header_value });
        }

        var response_body = std.ArrayList(u8).init(self.allocator);
        errdefer response_body.deinit();

        const result = client.fetch(.{
            .location = .{ .url = url },
            .method = try parseHttpMethod(method),
            .payload = if (body.len > 0) body else null,
            .extra_headers = headers.items,
            .response_storage = .{ .dynamic = &response_body },
        }) catch return error.GatewayRequestFailed;

        return .{
            .status = result.status,
            .body = try response_body.toOwnedSlice(),
        };
    }
};

fn parseGlobalJobRequest(allocator: Allocator, body: []const u8) !ParsedGlobalJobRequest {
    if (body.len == 0) {
        return .{
            .placement = .{},
            .job_body = try allocator.dupe(u8, "{}"),
        };
    }

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |object| object,
        else => return error.InvalidGlobalJobRequest,
    };

    const placement_view = if (root.get("placement")) |value|
        try parsePlacementValue(value)
    else
        parsePlacementObject(root);
    const placement = try clonePlacement(allocator, placement_view);

    if (root.get("job")) |job_value| {
        return .{
            .placement = placement,
            .job_body = switch (job_value) {
                .null => try allocator.dupe(u8, "{}"),
                else => try std.json.stringifyAlloc(allocator, job_value, .{}),
            },
        };
    }

    return .{
        .placement = placement,
        .job_body = try filterLocalJobBody(allocator, root),
    };
}

fn globalTrainingKindFromRequestBody(allocator: Allocator, body: []const u8) !training_workload.TrainingKind {
    var parsed = try parseGlobalJobRequest(allocator, body);
    defer parsed.deinit(allocator);
    return try training_extensions.trainingKindFromRequestBody(allocator, parsed.job_body);
}

test "global training job parser preserves typed workload intent" {
    try std.testing.expectEqual(
        training_workload.TrainingKind.regular,
        try globalTrainingKindFromRequestBody(std.testing.allocator, ""),
    );
    if (training_extensions.parseTrainingKind(training_extensions.training_kind_custom_extension) != null) {
        const private_body = try std.fmt.allocPrint(std.testing.allocator, "{{\"training_kind\":\"{s}\"}}", .{training_extensions.training_kind_custom_extension});
        defer std.testing.allocator.free(private_body);
        try std.testing.expectEqual(
            training_workload.TrainingKind.custom_extension,
            try globalTrainingKindFromRequestBody(std.testing.allocator, private_body),
        );
        const nested_private_body = try std.fmt.allocPrint(std.testing.allocator, "{{\"placement\":{{\"workers_required\":2}},\"job\":{{\"training_kind\":\"{s}\"}}}}", .{training_extensions.training_kind_custom_extension});
        defer std.testing.allocator.free(nested_private_body);
        try std.testing.expectEqual(
            training_workload.TrainingKind.custom_extension,
            try globalTrainingKindFromRequestBody(std.testing.allocator, nested_private_body),
        );
    }
}

fn parsePlacementValue(value: std.json.Value) !federation_types.PlacementRequest {
    const object = switch (value) {
        .object => |inner| inner,
        else => return error.InvalidGlobalJobRequest,
    };
    return parsePlacementObject(object);
}

fn parsePlacementObject(object: std.json.ObjectMap) federation_types.PlacementRequest {
    return .{
        .gateway_id = stringField(object, "gateway_id"),
        .service_id = stringField(object, "service_id"),
        .executor_id = stringField(object, "executor_id"),
        .worker_class = stringField(object, "worker_class"),
        .target_arch = stringField(object, "target_arch"),
        .workers_required = usizeField(object, "workers_required"),
    };
}

fn clonePlacement(
    allocator: Allocator,
    placement: federation_types.PlacementRequest,
) !federation_types.PlacementRequest {
    var owned: federation_types.PlacementRequest = .{
        .gateway_id = null,
        .service_id = null,
        .executor_id = null,
        .worker_class = null,
        .target_arch = null,
        .workers_required = placement.workers_required,
    };
    errdefer freePlacement(&owned, allocator);

    if (placement.gateway_id) |value| owned.gateway_id = try allocator.dupe(u8, value);
    if (placement.service_id) |value| owned.service_id = try allocator.dupe(u8, value);
    if (placement.executor_id) |value| owned.executor_id = try allocator.dupe(u8, value);
    if (placement.worker_class) |value| owned.worker_class = try allocator.dupe(u8, value);
    if (placement.target_arch) |value| owned.target_arch = try allocator.dupe(u8, value);
    return owned;
}

fn freePlacement(placement: *federation_types.PlacementRequest, allocator: Allocator) void {
    if (placement.gateway_id) |value| allocator.free(value);
    if (placement.service_id) |value| allocator.free(value);
    if (placement.executor_id) |value| allocator.free(value);
    if (placement.worker_class) |value| allocator.free(value);
    if (placement.target_arch) |value| allocator.free(value);
}

fn filterLocalJobBody(allocator: Allocator, object: std.json.ObjectMap) ![]u8 {
    var body = std.ArrayList(u8).init(allocator);
    errdefer body.deinit();
    var writer = body.writer();
    try writer.writeByte('{');
    var first = true;

    var it = object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "placement")) continue;
        if (std.mem.eql(u8, entry.key_ptr.*, "gateway_id")) continue;
        if (std.mem.eql(u8, entry.key_ptr.*, "worker_class")) continue;
        if (std.mem.eql(u8, entry.key_ptr.*, "target_arch")) continue;
        if (std.mem.eql(u8, entry.key_ptr.*, "workers_required")) continue;

        if (!first) try writer.writeByte(',');
        first = false;

        const key_json = try std.json.stringifyAlloc(allocator, entry.key_ptr.*, .{});
        defer allocator.free(key_json);
        const value_json = try std.json.stringifyAlloc(allocator, entry.value_ptr.*, .{});
        defer allocator.free(value_json);

        try writer.writeAll(key_json);
        try writer.writeByte(':');
        try writer.writeAll(value_json);
    }

    try writer.writeByte('}');
    return body.toOwnedSlice();
}

fn buildGatewayJobBody(
    allocator: Allocator,
    base_job_body: []const u8,
    service_id: []const u8,
    executor_id: []const u8,
) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, base_job_body, .{});
    defer parsed.deinit();

    const object = switch (parsed.value) {
        .object => |inner| inner,
        else => return error.InvalidGlobalJobRequest,
    };

    var body = std.ArrayList(u8).init(allocator);
    errdefer body.deinit();
    var writer = body.writer();
    try writer.writeByte('{');
    var first = true;

    var it = object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "service_id")) continue;
        if (std.mem.eql(u8, entry.key_ptr.*, "executor_id")) continue;

        if (!first) try writer.writeByte(',');
        first = false;

        const key_json = try std.json.stringifyAlloc(allocator, entry.key_ptr.*, .{});
        defer allocator.free(key_json);
        const value_json = try std.json.stringifyAlloc(allocator, entry.value_ptr.*, .{});
        defer allocator.free(value_json);

        try writer.writeAll(key_json);
        try writer.writeByte(':');
        try writer.writeAll(value_json);
    }

    if (!first) try writer.writeByte(',');
    const service_id_json = try std.json.stringifyAlloc(allocator, service_id, .{});
    defer allocator.free(service_id_json);
    const executor_id_json = try std.json.stringifyAlloc(allocator, executor_id, .{});
    defer allocator.free(executor_id_json);
    try writer.writeAll("\"service_id\":");
    try writer.writeAll(service_id_json);
    try writer.writeAll(",\"executor_id\":");
    try writer.writeAll(executor_id_json);
    try writer.writeByte('}');
    return body.toOwnedSlice();
}

fn extractRequiredStringField(allocator: Allocator, body: []const u8, field_name: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const object = switch (parsed.value) {
        .object => |inner| inner,
        else => return error.InvalidDownstreamResponse,
    };
    const value = stringField(object, field_name) orelse return error.InvalidDownstreamResponse;
    return try allocator.dupe(u8, value);
}

fn wrapGlobalJobJson(
    allocator: Allocator,
    record: hub_mod.GlobalJobRecord,
    child_body: []const u8,
) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, child_body, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidDownstreamResponse;

    const global_job_id_json = try std.json.stringifyAlloc(allocator, record.global_job_id, .{});
    defer allocator.free(global_job_id_json);
    const gateway_id_json = try std.json.stringifyAlloc(allocator, record.gateway_id, .{});
    defer allocator.free(gateway_id_json);
    const lab_id_json = try std.json.stringifyAlloc(allocator, record.lab_id, .{});
    defer allocator.free(lab_id_json);
    const gateway_base_url_json = try std.json.stringifyAlloc(allocator, record.gateway_base_url, .{});
    defer allocator.free(gateway_base_url_json);
    const service_id_json = try std.json.stringifyAlloc(allocator, record.service_id, .{});
    defer allocator.free(service_id_json);
    const executor_id_json = try std.json.stringifyAlloc(allocator, record.executor_id, .{});
    defer allocator.free(executor_id_json);
    const service_type_json = try std.json.stringifyAlloc(allocator, record.service_type, .{});
    defer allocator.free(service_type_json);
    const child_job_id_json = try std.json.stringifyAlloc(allocator, record.child_job_id, .{});
    defer allocator.free(child_job_id_json);

    var body = std.ArrayList(u8).init(allocator);
    errdefer body.deinit();
    var writer = body.writer();
    try writer.writeAll("{\"global_job_id\":");
    try writer.writeAll(global_job_id_json);
    try writer.writeAll(",\"assignment\":{\"gateway_id\":");
    try writer.writeAll(gateway_id_json);
    try writer.writeAll(",\"lab_id\":");
    try writer.writeAll(lab_id_json);
    try writer.writeAll(",\"gateway_base_url\":");
    try writer.writeAll(gateway_base_url_json);
    try writer.writeAll(",\"service_id\":");
    try writer.writeAll(service_id_json);
    try writer.writeAll(",\"executor_id\":");
    try writer.writeAll(executor_id_json);
    try writer.writeAll(",\"service_type\":");
    try writer.writeAll(service_type_json);
    try writer.writeAll(",\"child_job_id\":");
    try writer.writeAll(child_job_id_json);
    try writer.writeAll("},\"submitted_at\":");
    try writer.print("{d}", .{record.submitted_at});
    try writer.writeAll(",\"updated_at\":");
    try writer.print("{d}", .{record.updated_at});
    try writer.writeAll(",\"child\":");
    try writer.writeAll(child_body);
    try writer.writeByte('}');
    return body.toOwnedSlice();
}

fn writeFederationJobErrorResponse(stream: net.Stream, err: anyerror) !void {
    const status = switch (err) {
        error.InvalidGlobalJobRequest,
        error.InvalidDownstreamResponse,
        error.UnsupportedMethod,
        => "400 Bad Request",
        error.JobNotFound,
        error.ServiceUnavailable,
        => "404 Not Found",
        error.GatewayRequestFailed,
        => "502 Bad Gateway",
        else => "500 Internal Server Error",
    };
    try http_server.writeResponse(stream, status, &.{"Content-Type: text/plain"}, @errorName(err));
}
