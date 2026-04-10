const std = @import("std");

const Allocator = std.mem.Allocator;
const federation_types = @import("../federation/types.zig");
const gateway_registry = @import("gateway_registry.zig");
const graph_store = @import("../graph/store.zig");
const graph_types = @import("../graph/types.zig");

pub const GlobalController = struct {
    allocator: Allocator,
    registry: gateway_registry.GatewayRegistry,
    graph: graph_store.GraphStore,
    started_at: i64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .registry = gateway_registry.GatewayRegistry.init(allocator),
            .graph = graph_store.GraphStore.init(allocator, .{
                .backend = "memory",
                .gateway_id = "global-controller",
                .lab_id = "global",
                .neo4j = null,
            }) catch @panic("failed to initialize global graph store"),
            .started_at = std.time.timestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.graph.deinit();
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
        const counts = self.graph.counts();
        return std.json.stringifyAlloc(allocator, .{
            .mode = "global-controller",
            .registered_gateways = self.registry.count(),
            .replication_enabled = true,
            .max_replication_lag = self.registry.maxReplicationLag(),
            .global_graph = .{
                .namespaces = counts.namespaces,
                .entities = counts.entities,
                .relations = counts.relations,
                .observations = counts.observations,
            },
        }, .{});
    }

    pub fn renderGlobalGraphQueryJson(self: *Self, allocator: Allocator, request: graph_types.QueryRequest) ![]u8 {
        return self.graph.renderQueryJson(allocator, request);
    }

    pub fn applyMutationBatch(self: *Self, allocator: Allocator, request: federation_types.MutationBatchRequest) !federation_types.MutationBatchAck {
        const ack = try self.graph.applyReplicatedBatch(allocator, request.gateway_id, request.lab_id, request.mutations);
        if (!self.registry.markReplicated(request.gateway_id, ack.acked_sequence_no, request.last_sequence_no)) {
            return error.UnknownGateway;
        }
        return ack;
    }
};
