/// NodeManager: Local orchestrator that spawns and monitors multiple Supervisor processes
/// Each Supervisor manages a Worker on a specific GPU device
/// Process tree: NodeManager (1) → Supervisor (N) → Worker (N)
/// Resilience: NodeManager monitors Supervisors; Supervisors monitor Workers.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

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
        };
    }

    pub fn deinit(self: *Self) void {
        self.should_run.store(false, .release);
        self.allocator.free(self.self_exe_path);
        self.arena.deinit();
    }

    /// Spawns N supervisor monitors, each in its own thread
    pub fn spawnSupervisors(self: *Self, count: usize) !void {
        std.log.info("NodeManager: Starting {} supervisor monitors...", .{count});

        for (0..count) |i| {
            // Spawn a dedicated thread to manage this GPU slot
            const thread = try std.Thread.spawn(.{}, monitorLoop, .{ self, i });
            thread.detach();

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

                // The Supervisor process itself maintains the control-plane connection to the Shepherd.
                // Pass host/port *before* `--supervise` so the supervisor connects to the correct endpoint
                // (otherwise it defaults to 127.0.0.1:8080 and spams ConnectionRefused).
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
                std.log.err("[GPU {}] Failed to spawn: {}. Retrying in 5s...", .{device_id, err});
                std.time.sleep(5 * std.time.ns_per_s);
                continue;
            };

            // 4. Wait (Blocking)
            const term = child.wait() catch |err| blk_wait: {
                std.log.err("[GPU {}] Error waiting for supervisor: {}", .{device_id, err});
                _ = child.kill() catch {};
                break :blk_wait std.process.Child.Term{ .Exited = 255 };
            };

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
};
