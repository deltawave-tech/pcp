/// NodeManager: Local orchestrator that spawns and monitors multiple Supervisor processes
/// Each Supervisor manages a Worker on a specific GPU device
/// Process tree: NodeManager (1) → Supervisor (N) → Worker (N)
/// Resilience: NodeManager monitors Supervisors; Supervisors monitor Workers.
const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const TcpServer = @import("../network/tcp_stream.zig").TcpServer;
const http_server = @import("../network/http_server.zig");
const prometheus = @import("../observability/prometheus.zig");

pub const NodeManager = struct {
    allocator: Allocator,
    arena: std.heap.ArenaAllocator,
    // We no longer store child handles directly, as they are ephemeral (recreated on restart)
    // We store the configuration needed to respawn them.
    self_exe_path: []const u8,

    // Configuration to pass down
    host: []const u8,
    port: u16,
    backend: []const u8,
    target_arch: ?[]const u8,

    // Control flag
    should_run: std.atomic.Value(bool),
    expected_supervisors: std.atomic.Value(usize),
    launched_supervisors: std.atomic.Value(usize),
    child_pids_mutex: std.Thread.Mutex,
    child_pids: std.ArrayList(ManagedChildPid),
    monitor_threads: std.ArrayList(std.Thread),

    const ManagedChildPid = struct {
        device_id: usize,
        pid: std.process.Child.Id,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, host: []const u8, port: u16, backend: []const u8, target_arch: ?[]const u8) !Self {
        const self_exe = try std.fs.selfExePathAlloc(allocator);
        const arena = std.heap.ArenaAllocator.init(allocator);

        return Self{
            .allocator = allocator,
            .arena = arena,
            .self_exe_path = self_exe,
            .host = host,
            .port = port,
            .backend = backend,
            .target_arch = target_arch,
            .should_run = std.atomic.Value(bool).init(true),
            .expected_supervisors = std.atomic.Value(usize).init(0),
            .launched_supervisors = std.atomic.Value(usize).init(0),
            .child_pids_mutex = .{},
            .child_pids = std.ArrayList(ManagedChildPid).init(allocator),
            .monitor_threads = std.ArrayList(std.Thread).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.stop();
        self.joinMonitorThreads();
        self.monitor_threads.deinit();
        self.child_pids.deinit();
        self.allocator.free(self.self_exe_path);
        self.arena.deinit();
    }

    pub fn stop(self: *Self) void {
        self.should_run.store(false, .release);
        self.terminateManagedChildren();
    }

    /// Spawns N supervisor monitors, each in its own thread
    pub fn spawnSupervisors(self: *Self, count: usize) !void {
        std.log.info("NodeManager: Starting {} supervisor monitors...", .{count});
        self.expected_supervisors.store(count, .release);
        self.launched_supervisors.store(0, .release);

        for (0..count) |i| {
            // Spawn a dedicated thread to manage this GPU slot
            const thread = try std.Thread.spawn(.{}, monitorLoop, .{ self, i });
            self.monitor_threads.append(thread) catch |err| {
                thread.detach();
                return err;
            };
            _ = self.launched_supervisors.fetchAdd(1, .acq_rel);

            // Stagger start times
            std.time.sleep(200 * std.time.ns_per_ms);
        }
    }

    /// The Monitor Loop: Runs inside a thread, keeps one specific GPU slot alive
    fn monitorLoop(self: *Self, device_id: usize) void {
        // Use a local arena for argument building to avoid memory leaks over many restarts
        var loop_arena = std.heap.ArenaAllocator.init(self.allocator);
        defer loop_arena.deinit();

        while (self.should_run.load(.acquire)) {
            // 1. Reset memory at start of every iteration (handles 'continue' correctly)
            _ = loop_arena.reset(.retain_capacity);
            const child_allocator = loop_arena.allocator();

            // 2. Build Arguments Block
            var args = ArrayList([]const u8).init(child_allocator);

            const build_success = blk: {
                args.append(self.self_exe_path) catch break :blk false;
                args.append("--host") catch break :blk false;
                args.append(self.host) catch break :blk false;
                args.append("--port") catch break :blk false;
                const port_str = std.fmt.allocPrint(child_allocator, "{d}", .{self.port}) catch break :blk false;
                args.append(port_str) catch break :blk false;
                args.append("--supervise") catch break :blk false;
                args.append("--") catch break :blk false;
                args.append("--worker") catch break :blk false;

                args.append("--device-id") catch break :blk false;
                const id_str = std.fmt.allocPrint(child_allocator, "{d}", .{device_id}) catch break :blk false;
                args.append(id_str) catch break :blk false;

                args.append("--host") catch break :blk false;
                args.append(self.host) catch break :blk false;

                args.append("--port") catch break :blk false;
                args.append(port_str) catch break :blk false;

                args.append("--backend") catch break :blk false;
                args.append(self.backend) catch break :blk false;

                if (self.target_arch) |arch| {
                    args.append("--target") catch break :blk false;
                    args.append(arch) catch break :blk false;
                }
                break :blk true;
            };

            if (!build_success) {
                std.log.err("[GPU {}] OOM building arguments. Retrying in 5s...", .{device_id});
                std.time.sleep(5 * std.time.ns_per_s);
                continue;
            }

            // 3. Spawn
            var child = std.process.Child.init(args.items, self.allocator);
            child.stdin_behavior = .Ignore;
            child.stdout_behavior = .Inherit;
            child.stderr_behavior = .Inherit;

            std.log.info("[GPU {}] Launching Supervisor...", .{device_id});

            child.spawn() catch |err| {
                std.log.err("[GPU {}] Failed to spawn: {}. Retrying in 5s...", .{ device_id, err });
                std.time.sleep(5 * std.time.ns_per_s);
                continue;
            };
            self.registerChildPid(device_id, child.id) catch |err| {
                std.log.err("[GPU {}] Failed to track supervisor PID {}: {}", .{ device_id, child.id, err });
                _ = child.kill() catch {};
                continue;
            };

            // 4. Wait (Blocking)
            const child_pid = child.id;
            const term = child.wait() catch |err| blk_wait: {
                std.log.err("[GPU {}] Error waiting for supervisor: {}", .{ device_id, err });
                _ = child.kill() catch {};
                break :blk_wait std.process.Child.Term{ .Exited = 255 };
            };
            self.unregisterChildPid(child_pid);

            // 5. Handle Exit
            if (self.should_run.load(.acquire)) {
                std.log.warn("[GPU {}] Supervisor exited ({any}). Restarting in 1s...", .{ device_id, term });
                std.time.sleep(1 * std.time.ns_per_s);
            }
        }
    }

    /// Block the main thread forever (until interrupted)
    pub fn wait(self: *Self) !void {
        while (self.should_run.load(.acquire)) {
            std.time.sleep(1 * std.time.ns_per_s);
        }
    }

    fn registerChildPid(self: *Self, device_id: usize, pid: std.process.Child.Id) !void {
        self.child_pids_mutex.lock();
        defer self.child_pids_mutex.unlock();
        try self.child_pids.append(.{ .device_id = device_id, .pid = pid });
    }

    fn unregisterChildPid(self: *Self, pid: std.process.Child.Id) void {
        self.child_pids_mutex.lock();
        defer self.child_pids_mutex.unlock();
        for (self.child_pids.items, 0..) |entry, idx| {
            if (entry.pid == pid) {
                _ = self.child_pids.orderedRemove(idx);
                return;
            }
        }
    }

    fn terminateManagedChildren(self: *Self) void {
        self.child_pids_mutex.lock();
        defer self.child_pids_mutex.unlock();

        for (self.child_pids.items) |entry| {
            std.log.info("[GPU {}] Sending SIGTERM to supervisor PID {}", .{ entry.device_id, entry.pid });
            std.posix.kill(entry.pid, std.posix.SIG.TERM) catch |err| switch (err) {
                error.ProcessNotFound => {},
                else => std.log.warn("[GPU {}] Failed to terminate supervisor PID {}: {}", .{ entry.device_id, entry.pid, err }),
            };
        }
    }

    fn joinMonitorThreads(self: *Self) void {
        for (self.monitor_threads.items) |thread| {
            thread.join();
        }
        self.monitor_threads.clearRetainingCapacity();
    }

    pub fn renderReadyJson(self: *Self, allocator: Allocator) ![]u8 {
        const expected = self.expected_supervisors.load(.acquire);
        const launched = self.launched_supervisors.load(.acquire);
        const ready = expected > 0 and launched >= expected and self.should_run.load(.acquire);
        return std.json.stringifyAlloc(allocator, .{
            .ready = ready,
            .mode = "node-manager",
            .expected_supervisors = expected,
            .launched_supervisors = launched,
        }, .{});
    }

    pub fn runningSupervisorCount(self: *Self) usize {
        self.child_pids_mutex.lock();
        defer self.child_pids_mutex.unlock();
        return self.child_pids.items.len;
    }

    pub fn isReady(self: *Self) bool {
        const expected = self.expected_supervisors.load(.acquire);
        const launched = self.launched_supervisors.load(.acquire);
        return expected > 0 and launched >= expected and self.should_run.load(.acquire);
    }

    pub fn renderPrometheusMetrics(self: *Self, allocator: Allocator) ![]u8 {
        var body = std.ArrayList(u8).init(allocator);
        errdefer body.deinit();
        const writer = body.writer();
        const labels = [_]prometheus.Label{
            .{ .name = "backend", .value = self.backend },
            .{ .name = "target_arch", .value = self.target_arch orelse "default" },
        };

        const expected = self.expected_supervisors.load(.acquire);
        const launched = self.launched_supervisors.load(.acquire);
        const running = self.runningSupervisorCount();
        const ready = self.isReady();

        try prometheus.writeHelpAndType(writer, "pcp_node_manager_info", "Node-manager identity. Labels carry selected backend and target architecture.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_node_manager_info", &labels, 1);
        try prometheus.writeHelpAndType(writer, "pcp_node_manager_expected_supervisors", "Configured supervisor count.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_node_manager_expected_supervisors", &labels, expected);
        try prometheus.writeHelpAndType(writer, "pcp_node_manager_launched_supervisors", "Supervisor monitor threads launched.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_node_manager_launched_supervisors", &labels, launched);
        try prometheus.writeHelpAndType(writer, "pcp_node_manager_running_supervisors", "Supervisor child processes currently tracked.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_node_manager_running_supervisors", &labels, running);
        try prometheus.writeHelpAndType(writer, "pcp_node_manager_ready", "1 when expected supervisors have launched and manager is not stopping.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_node_manager_ready", &labels, @as(u8, if (ready) 1 else 0));

        return body.toOwnedSlice();
    }
};

pub const NodeManagerProbeServer = struct {
    allocator: Allocator,
    manager: *NodeManager,
    server: ?TcpServer,
    listen_host: ?[]const u8,
    listen_port: u16,
    is_running: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(allocator: Allocator, manager: *NodeManager) Self {
        return .{
            .allocator = allocator,
            .manager = manager,
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
                    std.log.err("NodeManager probe accept failed: {}", .{err});
                    continue;
                }
            else
                break;

            if (self.is_running.load(.acquire) == 0) {
                connection.stream.close();
                break;
            }

            const thread = std.Thread.spawn(.{}, handleProbeConnection, .{ self, connection.stream }) catch |err| {
                std.log.err("Failed to spawn node-manager probe handler: {}", .{err});
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
        var req = http_server.readRequest(stream, self.allocator, 64 * 1024) catch |err| {
            std.log.warn("NodeManager probe failed to read request: {}", .{err});
            return;
        };
        defer req.deinit();

        if (std.mem.eql(u8, req.path, "/healthz")) {
            http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain"}, "ok") catch {};
            return;
        }
        if (std.mem.eql(u8, req.path, "/readyz")) {
            const body = self.manager.renderReadyJson(self.allocator) catch |err| {
                std.log.err("NodeManager readiness render failed: {}", .{err});
                http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch {};
                return;
            };
            defer self.allocator.free(body);
            const status = if (self.manager.isReady()) "200 OK" else "503 Service Unavailable";
            http_server.writeResponse(stream, status, &.{"Content-Type: application/json"}, body) catch {};
            return;
        }
        if (std.mem.eql(u8, req.path, "/metrics")) {
            const body = self.manager.renderPrometheusMetrics(self.allocator) catch |err| {
                std.log.err("NodeManager metrics render failed: {}", .{err});
                http_server.writeResponse(stream, "500 Internal Server Error", &.{"Content-Type: text/plain"}, "error") catch {};
                return;
            };
            defer self.allocator.free(body);
            http_server.writeResponse(stream, "200 OK", &.{"Content-Type: text/plain; version=0.0.4"}, body) catch {};
            return;
        }
        http_server.writeResponse(stream, "404 Not Found", &.{"Content-Type: text/plain"}, "not found") catch {};
    }
};
