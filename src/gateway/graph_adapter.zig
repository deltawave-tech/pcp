const std = @import("std");

const Allocator = std.mem.Allocator;
const gateway_config = @import("config.zig");
const graph_store = @import("../graph/store.zig");
const graph_types = @import("../graph/types.zig");

pub const GatewayGraph = struct {
    store: graph_store.GraphStore,

    const Self = @This();

    pub fn init(allocator: Allocator, config: gateway_config.GatewayConfig) !Self {
        return .{ .store = try graph_store.GraphStore.init(allocator, .{
            .backend = config.graph_backend,
            .gateway_id = config.gateway_id,
            .lab_id = config.lab_id,
            .neo4j = if (config.neo4j) |neo4j| .{
                .uri = neo4j.uri,
                .http_uri = neo4j.http_uri,
                .user = neo4j.user,
                .password_env = neo4j.password_env,
                .database = neo4j.database,
                .query_timeout_ms = neo4j.query_timeout_ms,
                .bootstrap_on_connect = neo4j.bootstrap_on_connect,
            } else null,
        }) };
    }

    pub fn deinit(self: *Self) void {
        self.store.deinit();
    }

    pub fn renderStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        return self.store.renderStatusJson(allocator);
    }

    pub fn applyMutationsJson(self: *Self, allocator: Allocator, request: graph_types.MutateRequest) ![]u8 {
        return self.store.applyMutationsJson(allocator, request);
    }

    pub fn renderQueryJson(self: *Self, allocator: Allocator, request: graph_types.QueryRequest) ![]u8 {
        return self.store.renderQueryJson(allocator, request);
    }
};
