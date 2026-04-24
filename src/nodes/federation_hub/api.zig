const std = @import("std");
const net = std.net;

const Allocator = std.mem.Allocator;
const federation_types = @import("../../federation/types.zig");
const TcpServer = @import("../../network/tcp_stream.zig").TcpServer;
const http_server = @import("../../inference/http_server.zig");
const hub_mod = @import("hub.zig");
const gateway_registry = @import("gateway_registry.zig");
const graph_types = @import("../../graph/types.zig");

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

        if (!std.mem.startsWith(u8, req.path, "/v1/")) return false;

        if (!self.authorize(req)) {
            try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/controller")) {
            const body = try self.controller.renderControllerJson(self.allocator, self.api_token != null);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
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

            const body = try std.json.stringifyAlloc(self.allocator, .{
                .accepted = true,
                .gateway = .{
                    .gateway_id = gateway.gateway_id,
                    .lab_id = gateway.lab_id,
                    .base_url = gateway.base_url,
                    .graph_backend = gateway.graph_backend,
                    .status = gateway.status,
                    .registered_services = gateway.registered_services,
                    .connected_at = gateway.connected_at,
                    .last_seen_at = gateway.last_seen_at,
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
        if (!std.mem.startsWith(u8, header, "Bearer ")) return false;
        return std.mem.eql(u8, header["Bearer ".len..], self.api_token.?);
    }
};
