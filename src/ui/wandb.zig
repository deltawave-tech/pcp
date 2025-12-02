const std = @import("std");
const Allocator = std.mem.Allocator;

pub const WandBLogger = struct {
    child: std.process.Child,
    allocator: Allocator,
    enabled: bool,

    pub fn init(allocator: Allocator, config: WandBConfig, hyperparams: anytype) !WandBLogger {
        if (config.api_key == null) {
            std.log.warn("WandB API key not found. Logging disabled.", .{});
            return WandBLogger{
                .child = undefined,
                .allocator = allocator,
                .enabled = false,
            };
        }

        // Set up environment variables
        var env_map = try std.process.getEnvMap(allocator);
        defer env_map.deinit();
        try env_map.put("WANDB_API_KEY", config.api_key.?);
        try env_map.put("WANDB_SILENT", "true");

        const argv = [_][]const u8{ "venv/bin/python3", "tools/wandb_adapter.py" };

        var child = std.process.Child.init(&argv, allocator);
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Ignore; // Ignore stdout to keep TUI clean
        child.stderr_behavior = .Ignore; // Ignore stderr or redirect to log file
        child.env_map = &env_map;

        try child.spawn();

        var self = WandBLogger{
            .child = child,
            .allocator = allocator,
            .enabled = true,
        };

        // Send initialization config
        const init_payload = .{
            .project = config.project,
            .entity = config.entity,
            .run_name = config.run_name,
            .hyperparameters = hyperparams,
        };
        try self.sendJson(init_payload);

        return self;
    }

    pub fn log(self: *WandBLogger, metrics: anytype) void {
        if (!self.enabled) return;
        self.sendJson(metrics) catch |err| {
            std.log.err("Failed to log to WandB: {}", .{err});
            self.enabled = false; // Disable on error to prevent crashing
        };
    }

    fn sendJson(self: *WandBLogger, data: anytype) !void {
        if (self.child.stdin) |stdin| {
            const json_str = try std.json.stringifyAlloc(self.allocator, data, .{});
            defer self.allocator.free(json_str);

            try stdin.writer().writeAll(json_str);
            try stdin.writer().writeAll("\n");
        }
    }

    pub fn deinit(self: *WandBLogger) void {
        if (self.enabled) {
            // Send finish command
            self.sendJson(.{ ._command = "finish" }) catch {};
            _ = self.child.wait() catch {};
        }
    }
};

pub const WandBConfig = struct {
    project: []const u8 = "pcp-distributed",
    entity: ?[]const u8 = null,
    run_name: ?[]const u8 = null,
    api_key: ?[]const u8 = null,
};
