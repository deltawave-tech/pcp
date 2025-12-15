/// Supervisor - Parent process that manages a Worker child process with automatic restarts
/// Implements the Supervisor Pattern for fault-tolerant distributed training

const std = @import("std");
const net = std.net;
const tcp_stream = @import("../network/tcp_stream.zig");
const message = @import("../network/message.zig");

const TcpClient = tcp_stream.TcpClient;
const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;

pub const Supervisor = struct {
    allocator: std.mem.Allocator,
    client: TcpClient,
    shepherd_host: []const u8,
    shepherd_port: u16,
    supervisor_id: u64, // Unique ID for this supervisor instance

    worker_process: ?std.process.Child,
    worker_thread: ?std.Thread,
    should_run: bool,
    original_args: [][:0]u8, // CLI args to pass down

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, host: []const u8, port: u16, args: [][:0]u8) !Self {
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .shepherd_host = host,
            .shepherd_port = port,
            .supervisor_id = prng.random().int(u64), // Generate Random ID
            .worker_process = null,
            .worker_thread = null,
            .should_run = true,
            .original_args = args,
        };
    }

    pub fn deinit(self: *Self) void {
        self.should_run = false;
        self.killWorker();
        self.client.deinit();
    }

    pub fn run(self: *Self) !void {
        // 1. Connect Control Plane
        try self.client.connect(self.shepherd_host, self.shepherd_port);
        std.log.info("Supervisor {} connected to Shepherd.", .{self.supervisor_id});

        // 2. Handshake
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        try payload.put("supervisor_id", std.json.Value{ .integer = @intCast(self.supervisor_id) });

        const msg = tcp_stream.createMessage(
            0, "supervisor", 0, "shepherd",
            MessageType.SUPERVISOR_HANDSHAKE, 1,
            std.json.Value{ .object = payload }
        );
        try self.client.send(msg);

        // 3. Start Data Plane (Worker)
        try self.spawnWorker();

        // 4. Control Loop (Blocking receive)
        while (self.should_run) {
            const res = self.client.receive() catch break; // Break on disconnect
            defer res.parsed.deinit();
            defer self.allocator.free(res.buffer);

            const m = res.parsed.value;
            if (std.mem.eql(u8, m.msg_type, MessageType.RESTART_WORKER)) {
                std.log.warn("Received force restart command.", .{});
                self.killWorker();
                try self.spawnWorker();
            }
        }
    }

    fn spawnWorker(self: *Self) !void {
        if (self.worker_process != null) return;

        const self_exe = try std.fs.selfExePathAlloc(self.allocator);
        defer self.allocator.free(self_exe);

        var args_list = std.ArrayList([]const u8).init(self.allocator);
        defer args_list.deinit();

        try args_list.append(self_exe);
        try args_list.append("--worker"); // Switch mode for child

        // Pass ID so Worker can tell Shepherd who owns it
        var id_buf: [32]u8 = undefined;
        const id_str = try std.fmt.bufPrint(&id_buf, "{}", .{self.supervisor_id});
        const id_owned = try self.allocator.dupe(u8, id_str);
        defer self.allocator.free(id_owned);

        try args_list.append("--supervisor-id");
        try args_list.append(id_owned);

        // Forward config args (filter out --supervisor)
        for (self.original_args) |arg| {
            if (!std.mem.eql(u8, arg, "--supervisor") and !std.mem.eql(u8, arg, "--worker")) {
                try args_list.append(arg);
            }
        }

        var child = std.process.Child.init(args_list.items, self.allocator);
        child.stdin_behavior = .Ignore;
        child.stdout_behavior = .Inherit;
        child.stderr_behavior = .Inherit;

        try child.spawn();
        self.worker_process = child;
        std.log.info("Spawned worker process PID: {}", .{child.id});

        self.worker_thread = try std.Thread.spawn(.{}, monitorWorker, .{ self, &self.worker_process.? });
    }

    fn killWorker(self: *Self) void {
        if (self.worker_process) |*child| {
            _ = child.kill() catch {};
            self.worker_process = null;
        }
        if (self.worker_thread) |thread| {
            thread.detach();
            self.worker_thread = null;
        }
    }

    fn monitorWorker(self: *Self, child: *std.process.Child) !void {
        _ = child.wait() catch {};
        self.worker_process = null;

        if (self.should_run) {
            std.log.warn("Worker died. Restarting in 1s...", .{});
            std.time.sleep(1 * std.time.ns_per_s);
            self.spawnWorker() catch |err| std.log.err("Respawn failed: {}", .{err});
        }
    }
};
