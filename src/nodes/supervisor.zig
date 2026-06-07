/// Supervisor - Parent process that manages one worker process with automatic restarts.
/// The worker registers with the gateway worker-fabric endpoint using this
/// supervisor ID so the gateway can request a restart when needed.
const std = @import("std");
const tcp_stream = @import("../network/tcp_stream.zig");
const message = @import("../network/message.zig");

const TcpClient = tcp_stream.TcpClient;
const MessageType = message.MessageType;

pub const Supervisor = struct {
    allocator: std.mem.Allocator,
    client: TcpClient,
    worker_fabric_host: []const u8,
    worker_fabric_port: u16,
    supervisor_id: i64, // Unique ID for this supervisor instance (i64 for JSON compatibility)

    child_process: ?std.process.Child,
    child_thread: ?std.Thread,
    should_run: std.atomic.Value(bool),
    child_args: std.ArrayList([]const u8),

    const Self = @This();

    /// Initialize supervisor with worker child arguments.
    pub fn init(allocator: std.mem.Allocator, host: []const u8, port: u16, child_cmd_args: []const []const u8) !Self {
        try validateChildArgs(child_cmd_args);

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const my_id = prng.random().int(i64);

        var args_list = std.ArrayList([]const u8).init(allocator);
        errdefer {
            for (args_list.items) |arg| allocator.free(arg);
            args_list.deinit();
        }

        const self_exe = try std.fs.selfExePathAlloc(allocator);
        try args_list.append(self_exe);

        for (child_cmd_args) |arg| {
            const arg_copy = try allocator.dupe(u8, arg);
            try args_list.append(arg_copy);
        }

        const id_flag = try allocator.dupe(u8, "--supervisor-id");
        try args_list.append(id_flag);

        const id_str = try std.fmt.allocPrint(allocator, "{d}", .{my_id});
        try args_list.append(id_str);

        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .worker_fabric_host = host,
            .worker_fabric_port = port,
            .supervisor_id = my_id,
            .child_process = null,
            .child_thread = null,
            .should_run = std.atomic.Value(bool).init(true),
            .child_args = args_list,
        };
    }

    pub fn deinit(self: *Self) void {
        self.stop();

        // Free allocated argument strings
        for (self.child_args.items) |arg| {
            self.allocator.free(arg);
        }
        self.child_args.deinit();

        self.client.deinit();
    }

    pub fn stop(self: *Self) void {
        self.should_run.store(false, .release);
        self.client.disconnect();
        self.killChild();
    }

    pub fn run(self: *Self) !void {
        try self.spawnChild();

        std.log.info("Supervisor {} monitoring worker process", .{self.supervisor_id});

        while (self.should_run.load(.acquire)) {
            if (!self.client.isConnected()) {
                self.connectToWorkerFabric() catch |err| {
                    if (!self.should_run.load(.acquire)) break;
                    std.log.warn("Supervisor failed to connect to worker fabric: {}. Retrying in 1s...", .{err});
                    std.time.sleep(1 * std.time.ns_per_s);
                    continue;
                };
            }

            const res = self.client.receive() catch |err| {
                if (!self.should_run.load(.acquire)) break;
                std.log.warn("Worker-fabric supervisor connection lost ({}). Attempting reconnect...", .{err});
                self.client.disconnect();
                continue;
            };

            defer res.parsed.deinit();
            defer self.allocator.free(res.buffer);

            const m = res.parsed.value;
            if (std.mem.eql(u8, m.msg_type, MessageType.RESTART_WORKER)) {
                std.log.warn("Received worker restart command.", .{});
                self.killChild();
                try self.spawnChild();
            }
        }
    }

    /// Helper to handle handshake logic
    fn connectToWorkerFabric(self: *Self) !void {
        try self.client.connect(self.worker_fabric_host, self.worker_fabric_port);
        std.log.info("Supervisor {} connected to worker fabric control plane.", .{self.supervisor_id});

        // Re-register our ID so the gateway can map the worker to this supervisor.
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        try payload.put("supervisor_id", std.json.Value{ .integer = self.supervisor_id });

        const msg = tcp_stream.createMessage(
            0,
            "supervisor",
            0,
            "worker_fabric",
            MessageType.SUPERVISOR_HANDSHAKE,
            1,
            std.json.Value{ .object = payload },
        );
        try self.client.send(msg);
    }

    /// Spawn generic child process using pre-configured arguments
    fn spawnChild(self: *Self) !void {
        if (self.child_process != null) return;

        // Use the prepared arguments list directly
        var child = std.process.Child.init(self.child_args.items, self.allocator);
        child.stdin_behavior = .Ignore;
        child.stdout_behavior = .Inherit;
        child.stderr_behavior = .Inherit;

        try child.spawn();
        self.child_process = child;
        std.log.info("Supervisor: Spawned child process PID: {}", .{child.id});

        self.child_thread = try std.Thread.spawn(.{}, monitorChild, .{ self, &self.child_process.? });
    }

    fn killChild(self: *Self) void {
        if (self.child_process) |*child| {
            _ = child.kill() catch {};
            self.child_process = null;
        }
        if (self.child_thread) |thread| {
            thread.detach();
            self.child_thread = null;
        }
    }

    fn monitorChild(self: *Self, child: *std.process.Child) !void {
        _ = child.wait() catch {};
        self.child_process = null;

        if (self.should_run.load(.acquire)) {
            std.log.warn("Child process died. Restarting in 1s...", .{});
            std.time.sleep(1 * std.time.ns_per_s);
            self.spawnChild() catch |err| std.log.err("Respawn failed: {}", .{err});
        }
    }
};

fn validateChildArgs(child_cmd_args: []const []const u8) !void {
    var saw_worker = false;
    for (child_cmd_args) |arg| {
        if (std.mem.eql(u8, arg, "--worker")) saw_worker = true;
        if (std.mem.eql(u8, arg, "--shepherd") or
            std.mem.eql(u8, arg, "--inference") or
            std.mem.eql(u8, arg, "--config") or
            std.mem.eql(u8, arg, "--inference-config") or
            std.mem.eql(u8, arg, "--model") or
            std.mem.eql(u8, arg, "--resume") or
            std.mem.eql(u8, arg, "--rl") or
            std.mem.eql(u8, arg, "--no-dashboard") or
            std.mem.eql(u8, arg, "--terminate") or
            std.mem.eql(u8, arg, "--streaming"))
        {
            return error.LegacyStandaloneModeRemoved;
        }
    }
    if (!saw_worker) return error.SupervisorRequiresWorkerChild;
}

test "supervisor rejects removed standalone child modes" {
    const removed_controller_mode = [_][]const u8{ "--shepherd", "--config", "experiment.json" };
    try std.testing.expectError(error.LegacyStandaloneModeRemoved, validateChildArgs(&removed_controller_mode));

    const legacy_inference = [_][]const u8{ "--inference", "--inference-config", "inference.json" };
    try std.testing.expectError(error.LegacyStandaloneModeRemoved, validateChildArgs(&legacy_inference));

    const legacy_worker_flag = [_][]const u8{ "--worker", "--model", "model.mlir" };
    try std.testing.expectError(error.LegacyStandaloneModeRemoved, validateChildArgs(&legacy_worker_flag));
}

test "supervisor accepts worker child args" {
    const worker_args = [_][]const u8{ "--worker", "--connect", "127.0.0.1:18080", "--backend", "cpu" };
    try validateChildArgs(&worker_args);
}
