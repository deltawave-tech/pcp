const std = @import("std");
const net = std.net;

const Allocator = std.mem.Allocator;
const TcpServer = @import("../network/tcp_stream.zig").TcpServer;
const http_server = @import("../inference/http_server.zig");
const graph_types = @import("../graph/types.zig");
const service_registry = @import("service_registry.zig");
const gateway_mod = @import("gateway.zig");

pub const GatewayApiServer = struct {
    allocator: Allocator,
    gateway: *gateway_mod.Gateway,
    api_token: ?[]const u8,
    server: ?TcpServer,
    listen_host: ?[]const u8,
    listen_port: u16,
    is_running: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(allocator: Allocator, gateway: *gateway_mod.Gateway, api_token: ?[]const u8) Self {
        return .{
            .allocator = allocator,
            .gateway = gateway,
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
                    .endpoint = service.endpoint,
                    .status = service.status,
                    .worker_count = service.worker_count,
                    .capabilities = service.capabilities.items,
                    .registered_at = service.registered_at,
                    .updated_at = service.updated_at,
                },
            }, .{});
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/federation/status")) {
            const body = try self.gateway.renderFederationStatusJson(self.allocator);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/graph/status")) {
            const body = try self.gateway.graph.renderStatusJson(self.allocator);
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

            const body = try self.gateway.graph.renderQueryJson(self.allocator, parsed.value);
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
};
