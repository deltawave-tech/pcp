const std = @import("std");
const net = std.net;

const Allocator = std.mem.Allocator;
const training_controller = @import("../controllers/training_controller.zig");
const WorkerFabricController = training_controller.WorkerFabricController;
const WorkerLeaseOwner = training_controller.WorkerLeaseOwner;
const WorkerStatus = training_controller.WorkerStatus;
const TcpServer = @import("../../../network/tcp_stream.zig").TcpServer;
const http_server = @import("../../../network/http_server.zig");
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

pub const SnapshotProvider = struct {
    ctx: *anyopaque,
    capture: *const fn (
        ctx: *anyopaque,
        state: *control_state.ControllerState,
        worker_fabric: *WorkerFabricController,
        worker_scope_owner: ?WorkerLeaseOwner,
    ) anyerror!control_state.RuntimeSnapshot,
};

pub const CancelHook = struct {
    ctx: *anyopaque,
    cancel: *const fn (ctx: *anyopaque) void,
};

pub const ControlApiServer = struct {
    allocator: Allocator,
    state: *control_state.ControllerState,
    worker_fabric: *WorkerFabricController,
    api_token: ?[]const u8,
    snapshot_provider: ?SnapshotProvider,
    ready_renderer: ?ReadyRenderer,
    metrics_renderer: ?MetricsRenderer,
    worker_scope_owner: ?WorkerLeaseOwner,
    cancel_hook: ?CancelHook,
    server: ?TcpServer,
    listen_host: ?[]const u8,
    listen_port: u16,
    is_running: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        state: *control_state.ControllerState,
        worker_fabric: *WorkerFabricController,
        api_token: ?[]const u8,
        snapshot_provider: ?SnapshotProvider,
        ready_renderer: ?ReadyRenderer,
        metrics_renderer: ?MetricsRenderer,
        worker_scope_owner: ?WorkerLeaseOwner,
        cancel_hook: ?CancelHook,
    ) Self {
        return .{
            .allocator = allocator,
            .state = state,
            .worker_fabric = worker_fabric,
            .api_token = api_token,
            .snapshot_provider = snapshot_provider,
            .ready_renderer = ready_renderer,
            .metrics_renderer = metrics_renderer,
            .worker_scope_owner = worker_scope_owner,
            .cancel_hook = cancel_hook,
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
                    std.log.err("Control API accept failed: {}", .{err});
                    continue;
                }
            else
                break;

            if (self.is_running.load(.acquire) == 0) {
                connection.stream.close();
                break;
            }

            const thread = std.Thread.spawn(.{}, handleConnection, .{ self, connection.stream }) catch |err| {
                std.log.err("Failed to spawn control API handler thread: {}", .{err});
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

    pub fn handleRequest(self: *Self, stream: net.Stream, req: *http_server.HttpRequest) !bool {
        const runtime = try self.captureRuntimeSnapshot();
        const connected_workers = runtime.workers_connected;

        if (std.mem.eql(u8, req.path, "/healthz")) {
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain"}, "ok");
            return true;
        }

        if (std.mem.eql(u8, req.path, "/readyz")) {
            var ready_result: ReadyResult = undefined;
            if (self.ready_renderer) |renderer| {
                ready_result = try renderer.render(renderer.ctx, self.allocator, self.state, connected_workers);
            } else {
                ready_result = .{
                    .ready = runtime.ready,
                    .body = try self.state.renderReadyJson(self.allocator, runtime),
                };
            }
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
            const body = try self.state.renderControllerJsonWithRuntime(self.allocator, runtime, self.api_token != null);
            defer self.allocator.free(body);
            try http_server.writeResponse(stream, "200 OK", &.{"Content-Type: application/json"}, body);
            return true;
        }

        if ((std.mem.eql(u8, req.method, "GET") or std.mem.eql(u8, req.method, "POST")) and
            (std.mem.eql(u8, req.path, "/v1/job") or std.mem.eql(u8, req.path, "/v1/jobs/current")))
        {
            const body = try self.state.renderJobJsonWithRuntime(self.allocator, runtime);
            defer self.allocator.free(body);
            const status = if (std.mem.eql(u8, req.method, "POST")) "202 Accepted" else "200 OK";
            try http_server.writeResponse(stream, status, &.{"Content-Type: application/json"}, body);
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
                try self.state.renderMetricsJsonWithRuntime(self.allocator, runtime);
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

        _ = self.handleRequest(stream, &req) catch |err| {
            std.log.err("Control API route failed: {}", .{err});
            _ = http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch |write_err| {
                std.log.warn("Control API failed to write error response: {}", .{write_err});
            };
            return;
        };
    }

    fn authorize(self: *Self, req: *http_server.HttpRequest) bool {
        if (self.api_token == null) return true;
        const header = req.header("authorization") orelse return false;
        const token = bearerToken(header) orelse return false;
        return std.mem.eql(u8, token, self.api_token.?);
    }

    fn isReady(self: *Self, connected_workers: usize) !bool {
        const ready_result = if (self.ready_renderer) |renderer|
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

    fn captureRuntimeSnapshot(self: *Self) !control_state.RuntimeSnapshot {
        if (self.snapshot_provider) |provider| {
            return try provider.capture(provider.ctx, self.state, self.worker_fabric, self.worker_scope_owner);
        }

        const connected_workers = if (self.worker_scope_owner) |owner|
            self.worker_fabric.getUsableWorkerCountForLeaseOwner(owner)
        else
            self.worker_fabric.getWorkerCount();

        self.state.mutex.lock();
        const required_workers = self.state.required_workers;
        self.state.mutex.unlock();

        const required = required_workers orelse 1;
        const ready = connected_workers >= required;
        return .{
            .workers_connected = connected_workers,
            .workers_ready = connected_workers,
            .workers_available = connected_workers,
            .health_status = if (ready) "ok" else "starting",
            .ready = ready,
        };
    }

    fn renderWorkersJson(self: *Self) ![]u8 {
        const WorkerResponse = struct {
            node_id: u32,
            worker_id: []const u8,
            backend: []const u8,
            status: []const u8,
            address: []const u8,
            target_arch: ?[]const u8,
            lease_owner: ?[]const u8,
            available: bool,
        };

        var workers_snapshot = try self.worker_fabric.snapshotWorkers();
        defer workers_snapshot.deinit();

        var workers = try std.ArrayList(WorkerResponse).initCapacity(self.allocator, workers_snapshot.items.len);
        defer workers.deinit();

        for (workers_snapshot.items) |worker| {
            var available = worker.lease_owner == null;
            if (self.worker_scope_owner) |owner| {
                if (worker.lease_owner) |lease_owner| {
                    if (lease_owner != owner) continue;
                    available = false;
                } else {
                    available = self.worker_fabric.canAcquireWorker(worker.node_id, owner);
                    if (!available) continue;
                }
            }

            const address = try std.fmt.allocPrint(self.allocator, "{}", .{worker.address});
            errdefer self.allocator.free(address);

            try workers.append(.{
                .node_id = worker.node_id,
                .worker_id = worker.stable_worker_id,
                .backend = worker.backend.toString(),
                .status = workerStatusString(worker.status),
                .address = address,
                .target_arch = worker.target_arch,
                .lease_owner = if (worker.lease_owner) |owner| owner.label() else null,
                .available = available,
            });
        }
        defer {
            for (workers.items) |worker| self.allocator.free(worker.address);
        }

        return std.json.stringifyAlloc(self.allocator, .{ .workers = workers.items }, .{});
    }
};

fn bearerToken(header: []const u8) ?[]const u8 {
    const scheme = "Bearer";
    if (header.len <= scheme.len) return null;
    if (!std.ascii.eqlIgnoreCase(header[0..scheme.len], scheme)) return null;
    if (header[scheme.len] != ' ') return null;
    const token = std.mem.trimLeft(u8, header[scheme.len + 1 ..], " ");
    if (token.len == 0) return null;
    return token;
}

fn workerStatusString(status: WorkerStatus) []const u8 {
    return switch (status) {
        .Connected => "connected",
        .InitializingGraph => "initializing_graph",
        .GraphInitialized => "initialized",
        .Training => "training",
    };
}
