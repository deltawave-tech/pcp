const std = @import("std");

const Allocator = std.mem.Allocator;
const gateway_registry = @import("gateway_registry.zig");

pub const GlobalController = struct {
    allocator: Allocator,
    registry: gateway_registry.GatewayRegistry,
    started_at: i64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .registry = gateway_registry.GatewayRegistry.init(allocator),
            .started_at = std.time.timestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.registry.deinit();
    }

    pub fn renderReadyJson(self: *Self, allocator: Allocator, auth_enabled: bool) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .ready = true,
            .mode = "global-controller",
            .registered_gateways = self.registry.count(),
            .auth_enabled = auth_enabled,
        }, .{});
    }

    pub fn renderControllerJson(self: *Self, allocator: Allocator, auth_enabled: bool) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .mode = "global-controller",
            .status = "running",
            .registered_gateways = self.registry.count(),
            .started_at = self.started_at,
            .auth_enabled = auth_enabled,
        }, .{});
    }

    pub fn renderGlobalGraphStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .mode = "global-controller",
            .registered_gateways = self.registry.count(),
            .replication_enabled = false,
            .max_replication_lag = 0,
            .global_graph = .{
                .namespaces = 0,
                .entities = 0,
                .relations = 0,
                .observations = 0,
            },
        }, .{});
    }
};
