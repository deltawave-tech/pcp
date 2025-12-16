/// NodeManager: Local orchestrator that spawns and monitors multiple Supervisor processes
/// Each Supervisor manages a Worker on a specific GPU device
/// Process tree: NodeManager (1) → Supervisor (N) → Worker (N)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const NodeManager = struct {
    allocator: Allocator,
    arena: std.heap.ArenaAllocator,
    children: ArrayList(std.process.Child),
    self_exe_path: []const u8,

    // Configuration to pass down
    host: []const u8,
    port: u16,
    backend: []const u8,
    target_arch: ?[]const u8,

    const Self = @This();

    pub fn init(allocator: Allocator, host: []const u8, port: u16, backend: []const u8, target_arch: ?[]const u8) !Self {
        const self_exe = try std.fs.selfExePathAlloc(allocator);

        const arena = std.heap.ArenaAllocator.init(allocator);

        return Self{
            .allocator = allocator,
            .arena = arena,
            .children = ArrayList(std.process.Child).init(allocator),
            .self_exe_path = self_exe,
            .host = host,
            .port = port,
            .backend = backend,
            .target_arch = target_arch,
        };
    }

    pub fn deinit(self: *Self) void {
        self.killAll();
        self.children.deinit();
        self.allocator.free(self.self_exe_path);
        self.arena.deinit();
    }

    /// Spawns N supervisors, assigning each a sequential device_id
    pub fn spawnSupervisors(self: *Self, count: usize) !void {
        std.log.info("NodeManager: Spawning {} supervisors...", .{count});

        for (0..count) |i| {
            try self.spawnSingleSupervisor(i);
            // Stagger start times slightly to avoid "thundering herd" on disk/network
            std.time.sleep(200 * std.time.ns_per_ms);
        }
    }

    fn spawnSingleSupervisor(self: *Self, device_id: usize) !void {
        const arena_allocator = self.arena.allocator();
        var args = ArrayList([]const u8).init(arena_allocator);
        defer args.deinit();

        // Construct the command:
        // ./pcp --supervisor --supervise -- --worker --device-id <ID> --host <IP> ...

        // 1. Executable
        try args.append(self.self_exe_path);

        // 2. Supervisor Mode
        try args.append("--supervisor");

        // 3. Enable Supervision (The supervisor will manage the worker)
        try args.append("--supervise");

        // 4. Separator for Child (Worker) Args
        try args.append("--");

        // 5. Worker Arguments
        try args.append("--worker");

        // Device ID
        try args.append("--device-id");
        const id_str = try std.fmt.allocPrint(arena_allocator, "{d}", .{device_id});
        try args.append(id_str);

        // Connection details
        try args.append("--host");
        try args.append(self.host);

        try args.append("--port");
        const port_str = try std.fmt.allocPrint(arena_allocator, "{d}", .{self.port});
        try args.append(port_str);

        try args.append("--backend");
        try args.append(self.backend);

        if (self.target_arch) |arch| {
            try args.append("--target");
            try args.append(arch);
        }

        // Spawn
        var child = std.process.Child.init(args.items, self.allocator);
        child.stdin_behavior = .Ignore;
        child.stdout_behavior = .Inherit;
        child.stderr_behavior = .Inherit;

        try child.spawn();
        try self.children.append(child);

        std.log.info("NodeManager: Launched Supervisor for GPU {} (PID {})", .{ device_id, child.id });
    }

    /// Run loop - monitors children and waits for them
    pub fn wait(self: *Self) !void {
        for (self.children.items) |*child| {
            _ = try child.wait();
        }
    }

    /// Kill all children (used on shutdown)
    pub fn killAll(self: *Self) void {
        for (self.children.items) |*child| {
            _ = child.kill() catch {};
        }
    }
};
