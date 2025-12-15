/// Supervisor - Parent process that manages a child process with automatic restarts
/// Implements the Supervisor Pattern for fault-tolerant distributed training
/// Generalized to supervise any child process (Worker or Shepherd)

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
    supervisor_id: i64, // Unique ID for this supervisor instance (i64 for JSON compatibility)

    child_process: ?std.process.Child,
    child_thread: ?std.Thread,
    should_run: bool,
    child_args: std.ArrayList([]const u8), // Arguments for the child process

    const Self = @This();

    /// Initialize supervisor with generic child arguments
    /// child_cmd_args should contain all arguments to pass to the child (e.g., ["--shepherd", "--config", "exp.json"])
    pub fn init(allocator: std.mem.Allocator, host: []const u8, port: u16, child_cmd_args: []const []const u8) !Self {
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

        var args_list = std.ArrayList([]const u8).init(allocator);

        // First arg is the executable path (will be set in spawnChild)
        const self_exe = try std.fs.selfExePathAlloc(allocator);
        try args_list.append(self_exe);

        // Copy child arguments
        for (child_cmd_args) |arg| {
            const arg_copy = try allocator.dupe(u8, arg);
            try args_list.append(arg_copy);
        }

        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .shepherd_host = host,
            .shepherd_port = port,
            .supervisor_id = prng.random().int(i64), // Generate Random ID (i64 for JSON compatibility)
            .child_process = null,
            .child_thread = null,
            .should_run = true,
            .child_args = args_list,
        };
    }

    pub fn deinit(self: *Self) void {
        self.should_run = false;
        self.killChild();

        // Free allocated argument strings
        for (self.child_args.items) |arg| {
            self.allocator.free(arg);
        }
        self.child_args.deinit();

        self.client.deinit();
    }

    pub fn run(self: *Self) !void {
        // 1. Detect mode
        var is_shepherd_child = false;
        for (self.child_args.items) |arg| {
            if (std.mem.eql(u8, arg, "--shepherd")) {
                is_shepherd_child = true;
                break;
            }
        }

        // 2. Start the child process immediately (Data Plane)
        // We do this BEFORE connecting to Shepherd, so the worker exists even if network is down.
        try self.spawnChild();

        // 3. Control Loop
        if (is_shepherd_child) {
            // Shepherd Supervisor: Just keep process alive
            std.log.info("Supervisor {} monitoring shepherd process", .{self.supervisor_id});
            while (self.should_run) {
                std.time.sleep(1 * std.time.ns_per_s);
            }
        } else {
            // Worker Supervisor: Maintain Control Plane connection
            std.log.info("Supervisor {} monitoring worker process", .{self.supervisor_id});

            while (self.should_run) {
                // A. Connection Loop
                if (!self.client.isConnected()) {
                    self.connectToShepherd() catch |err| {
                        std.log.warn("Supervisor failed to connect to Shepherd: {}. Retrying in 1s...", .{err});
                        std.time.sleep(1 * std.time.ns_per_s);
                        continue;
                    };
                }

                // B. Message Loop
                // receive() blocks. If connection breaks, it returns error.
                const res = self.client.receive() catch |err| {
                    std.log.warn("Control plane connection lost ({}). Attempting reconnect...", .{err});
                    self.client.disconnect(); // Clean up socket state
                    continue; // Loop back to A
                };

                defer res.parsed.deinit();
                defer self.allocator.free(res.buffer);

                const m = res.parsed.value;
                if (std.mem.eql(u8, m.msg_type, MessageType.RESTART_WORKER)) {
                    std.log.warn("Received force restart command.", .{});
                    self.killChild();
                    try self.spawnChild();
                }
            }
        }
    }

    /// Helper to handle handshake logic
    fn connectToShepherd(self: *Self) !void {
        // 1. Connect
        try self.client.connect(self.shepherd_host, self.shepherd_port);
        std.log.info("Supervisor {} connected to Shepherd control plane.", .{self.supervisor_id});

        // 2. Handshake (Re-register our ID so Shepherd knows we exist)
        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        try payload.put("supervisor_id", std.json.Value{ .integer = self.supervisor_id });

        const msg = tcp_stream.createMessage(
            0,
            "supervisor",
            0,
            "shepherd",
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

        if (self.should_run) {
            std.log.warn("Child process died. Restarting in 1s...", .{});
            std.time.sleep(1 * std.time.ns_per_s);
            self.spawnChild() catch |err| std.log.err("Respawn failed: {}", .{err});
        }
    }
};
