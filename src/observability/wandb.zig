const std = @import("std");
const Allocator = std.mem.Allocator;

pub const WandBLogger = struct {
    child: std.process.Child,
    allocator: Allocator,
    enabled: bool,

    pub fn init(allocator: Allocator, config: WandBConfig, hyperparams: anytype) !WandBLogger {
        const api_key = config.api_key orelse std.posix.getenv("WANDB_API_KEY");
        const wandb_mode = std.posix.getenv("WANDB_MODE");
        const allow_without_key = if (wandb_mode) |mode|
            std.mem.eql(u8, mode, "offline") or std.mem.eql(u8, mode, "dryrun")
        else
            false;

        if (api_key == null and !allow_without_key) {
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
        if (api_key) |key| try env_map.put("WANDB_API_KEY", key);
        if (wandb_mode) |mode| try env_map.put("WANDB_MODE", mode);
        try env_map.put("WANDB_SILENT", "true");

        const adapter_path = blk: {
            std.fs.cwd().access("tools/wandb_adapter.py", .{}) catch break :blk "wandb_adapter.py";
            break :blk "tools/wandb_adapter.py";
        };
        const argv = [_][]const u8{adapter_path};

        var child = std.process.Child.init(&argv, allocator);
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Ignore;
        child.stderr_behavior = .Inherit;
        child.env_map = &env_map;

        try child.spawn();

        var self = WandBLogger{
            .child = child,
            .allocator = allocator,
            .enabled = true,
        };

        const init_payload = .{
            .project = config.project,
            .entity = config.entity,
            .run_name = config.run_name,
            .hyperparameters = hyperparams,
        };
        try self.sendJson(init_payload);

        std.log.info("WandB logging initialized for project: {s}", .{config.project});
        return self;
    }

    pub fn log(self: *WandBLogger, metrics: anytype) void {
        if (!self.enabled) return;
        self.sendJson(metrics) catch |err| {
            std.log.err("Failed to log to WandB: {}", .{err});
            self.enabled = false;
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
