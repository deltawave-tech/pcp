const std = @import("std");
const net = std.net;

const Allocator = std.mem.Allocator;
const Shepherd = @import("../nodes/controllers/shepherd.zig").Shepherd;
const WorkerStatus = @import("../nodes/controllers/shepherd.zig").WorkerStatus;
const TcpServer = @import("../network/tcp_stream.zig").TcpServer;
const http_server = @import("../inference/http_server.zig");
const control_state = @import("state.zig");

pub const ReadyResult = struct {
    ready: bool,
    body: []u8,
};

pub const ReadyRenderer = struct {
    ctx: *anyopaque,
    render: *const fn (ctx: *anyopaque, allocator: Allocator, state: *control_state.ControllerState, connected_workers: usize) anyerror!ReadyResult,
};

pub const MetricsRenderer = struct {
    ctx: *anyopaque,
    render: *const fn (ctx: *anyopaque, allocator: Allocator, state: *control_state.ControllerState, connected_workers: usize) anyerror![]u8,
};

pub const CancelHook = struct {
    ctx: *anyopaque,
    cancel: *const fn (ctx: *anyopaque) void,
};

pub const ControlApiServer = struct {
    allocator: Allocator,
    state: *control_state.ControllerState,
    shepherd: *Shepherd,
    api_token: ?[]const u8,
    ready_renderer: ?ReadyRenderer,
    metrics_renderer: ?MetricsRenderer,
    cancel_hook: ?CancelHook,
    server: ?TcpServer,
    is_running: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        state: *control_state.ControllerState,
        shepherd: *Shepherd,
        api_token: ?[]const u8,
        ready_renderer: ?ReadyRenderer,
        metrics_renderer: ?MetricsRenderer,
        cancel_hook: ?CancelHook,
    ) Self {
        return .{
            .allocator = allocator,
            .state = state,
            .shepherd = shepherd,
            .api_token = api_token,
            .ready_renderer = ready_renderer,
            .metrics_renderer = metrics_renderer,
            .cancel_hook = cancel_hook,
            .server = null,
            .is_running = std.atomic.Value(u8).init(0),
        };
    }

    pub fn start(self: *Self, host: []const u8, port: u16) !void {
        self.server = try TcpServer.init(self.allocator, host, port);
        self.is_running.store(1, .release);

        while (self.is_running.load(.acquire) == 1) {
            const connection = if (self.server) |*server|
                server.accept() catch |err| {
                    if (self.is_running.load(.acquire) == 0) break;
                    std.log.err("Control API accept failed: {}", .{err});
                    continue;
                }
            else
                break;

            const thread = std.Thread.spawn(.{}, handleConnection, .{ self, connection.stream }) catch |err| {
                std.log.err("Failed to spawn control API handler thread: {}", .{err});
                connection.stream.close();
                continue;
            };
            thread.detach();
        }
    }

    pub fn stop(self: *Self) void {
        self.is_running.store(0, .release);
        if (self.server) |*server| {
            server.deinit();
            self.server = null;
        }
    }

    pub fn handleRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        const connected_workers = self.shepherd.getWorkerCount();

        if (std.mem.eql(u8, req.path, "/healthz")) {
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain"}, "ok");
            return true;
        }

        if (std.mem.eql(u8, req.path, "/readyz")) {
            var ready_result = if (self.ready_renderer) |renderer|
                try renderer.render(renderer.ctx, self.allocator, self.state, connected_workers)
            else
                try self.renderDefaultReady(connected_workers);
            defer self.allocator.free(ready_result.body);

            const status = if (ready_result.ready) "200 OK" else "503 Service Unavailable";
            try http_server.writeResponse(stream, status, &.{"Content-Type: application/json"}, ready_result.body);
            return true;
        }

        const is_control_path = std.mem.startsWith(u8, req.path, "/v1/controller") or
            std.mem.startsWith(u8, req.path, "/v1/job") or
            std.mem.startsWith(u8, req.path, "/v1/jobs/current") or
            std.mem.startsWith(u8, req.path, "/v1/workers") or
            std.mem.startsWith(u8, req.path, "/v1/metrics");

        if (!is_control_path) return false;

        if (!self.authorize(req)) {
            try http_server.writeResponse(stream, "401 Unauthorized", &.{"Content-Type: text/plain"}, "unauthorized");
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/controller")) {
            const ready = try self.isReady(connected_workers);
            const body = try self.state.renderControllerJson(self.allocator, connected_workers, ready, self.api_token != null);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and
            (std.mem.eql(u8, req.path, "/v1/job") or std.mem.eql(u8, req.path, "/v1/jobs/current")))
        {
            const body = try self.state.renderJobJson(self.allocator, connected_workers);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/workers")) {
            const body = try self.renderWorkersJson();
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/metrics")) {
            const body = if (self.metrics_renderer) |renderer|
                try renderer.render(renderer.ctx, self.allocator, self.state, connected_workers)
            else
                try self.state.renderMetricsJson(self.allocator, connected_workers);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if (std.mem.eql(u8, req.method, "POST") and
            (std.mem.eql(u8, req.path, "/v1/job/cancel") or std.mem.eql(u8, req.path, "/v1/jobs/current/cancel")))
        {
            if (self.cancel_hook == null) {
                try http_server.writeResponse(stream, "405 Method Not Allowed", &.{"Content-Type: text/plain"}, "cancel_not_supported");
                return true;
            }

            self.state.requestCancel();
            self.cancel_hook.?.cancel(self.cancel_hook.?.ctx);

            const body = try std.json.stringifyAlloc(self.allocator, .{
                .accepted = true,
                .status = "cancelling",
            }, .{});
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "202 Accepted", &.{"Content-Type: application/json"}, body);
            return true;
        }

        try http_server.writeResponse(stream, "404 Not Found", &.{"Content-Type: text/plain"}, "not_found");
        return true;
    }

    fn handleConnection(self: *Self, stream: net.Stream) void {
        defer stream.close();

        var req = http_server.readRequest(stream, self.allocator, 1024 * 1024) catch |err| {
            std.log.warn("Control API failed to read request: {}", .{err});
            return;
        };
        defer req.deinit();

        self.handleRequest(stream, &req) catch |err| {
            std.log.err("Control API route failed: {}", .{err});
            _ = http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch {};
        };
    }

    fn authorize(self: *Self, req: *http_server.HttpRequest) bool {
        if (self.api_token == null) return true;
        const header = req.header("authorization") orelse return false;
        if (!std.mem.startsWith(u8, header, "Bearer ")) return false;
        return std.mem.eql(u8, header["Bearer ".len..], self.api_token.?);
    }

    fn isReady(self: *Self, connected_workers: usize) !bool {
        var ready_result = if (self.ready_renderer) |renderer|
            try renderer.render(renderer.ctx, self.allocator, self.state, connected_workers)
        else
            try self.renderDefaultReady(connected_workers);
        defer self.allocator.free(ready_result.body);
        return ready_result.ready;
    }

    fn renderDefaultReady(self: *Self, connected_workers: usize) !ReadyResult {
        self.state.mutex.lock();
        const required_workers = self.state.required_workers;
        const status = self.state.status;
        self.state.mutex.unlock();

        const required = required_workers orelse 1;
        const ready = connected_workers >= required;
        const body = try std.json.stringifyAlloc(self.allocator, .{
            .ready = ready,
            .status = control_state.ControllerState.statusString(status),
            .workers_connected = connected_workers,
            .workers_required = required_workers,
        }, .{});
        return .{ .ready = ready, .body = body };
    }

    fn renderWorkersJson(self: *Self) ![]u8 {
        const WorkerResponse = struct {
            node_id: u32,
            backend: []const u8,
            status: []const u8,
            address: []const u8,
            target_arch: ?[]const u8,
        };

        self.shepherd.worker_pool_mutex.lock();
        defer self.shepherd.worker_pool_mutex.unlock();

        var workers = try std.ArrayList(WorkerResponse).initCapacity(self.allocator, self.shepherd.worker_pool.items.len);
        defer workers.deinit();

        for (self.shepherd.worker_pool.items) |worker| {
            const address = try std.fmt.allocPrint(self.allocator, "{}", .{worker.address});
            errdefer self.allocator.free(address);

            try workers.append(.{
                .node_id = worker.node_id,
                .backend = worker.backend.toString(),
                .status = workerStatusString(worker.status),
                .address = address,
                .target_arch = worker.target_arch,
            });
        }
        defer {
            for (workers.items) |worker| self.allocator.free(worker.address);
        }

        return std.json.stringifyAlloc(self.allocator, .{ .workers = workers.items }, .{});
    }
};

fn workerStatusString(status: WorkerStatus) []const u8 {
    return switch (status) {
        .Connected => "connected",
        .GraphInitialized => "initialized",
        .Training => "training",
    };
}
