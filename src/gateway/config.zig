const std = @import("std");

const Allocator = std.mem.Allocator;

pub const GatewayConfig = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    graph_backend: []const u8 = "memory",
    global_controller_endpoint: ?[]const u8 = null,
    api_token_env: ?[]const u8 = null,
};

pub const ConfigResult = struct {
    config: GatewayConfig,
    parsed: ?std.json.Parsed(GatewayConfig),
    json_data: ?[]u8,
    allocator: Allocator,

    pub fn deinit(self: *@This()) void {
        if (self.parsed) |*p| {
            p.deinit();
        }
        if (self.json_data) |data| {
            self.allocator.free(data);
        }
    }
};

pub fn loadGatewayConfig(allocator: Allocator, path: []const u8) !ConfigResult {
    std.log.info("Loading gateway config from: {s}", .{path});
    const data = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    const parsed = try std.json.parseFromSlice(GatewayConfig, allocator, data, .{ .ignore_unknown_fields = true });
    return .{
        .config = parsed.value,
        .parsed = parsed,
        .json_data = data,
        .allocator = allocator,
    };
}
