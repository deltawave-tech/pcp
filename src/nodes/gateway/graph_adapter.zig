const std = @import("std");

const Allocator = std.mem.Allocator;
const federation_types = @import("../../federation/types.zig");
const gateway_config = @import("config.zig");
const graph_store = @import("../../graph/store.zig");
const mutation_log = @import("../../graph/mutation_log.zig");
const graph_types = @import("../../graph/types.zig");

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

    pub fn counts(self: *Self) graph_types.GraphCounts {
        return self.store.counts();
    }

    pub fn lastSequence(self: *Self) u64 {
        return self.store.lastSequence();
    }

    pub fn lastReplicatedSequence(self: *Self) u64 {
        return self.store.lastReplicatedSequence();
    }

    pub fn countPendingReplications(self: *Self) usize {
        return self.store.countPendingReplications();
    }

    pub fn markReplicatedThrough(self: *Self, sequence_no: u64) void {
        self.store.markReplicatedThrough(sequence_no);
    }

    pub fn snapshotMutationsFrom(self: *Self, allocator: Allocator, after_sequence_no: u64, max_items: usize) ![]mutation_log.MutationRecord {
        return self.store.snapshotMutationsFrom(allocator, after_sequence_no, max_items);
    }

    pub fn listNamespaceMutationStats(self: *Self, allocator: Allocator, replicated_through_sequence: ?u64) ![]mutation_log.NamespaceMutationStats {
        return self.store.listNamespaceMutationStats(allocator, replicated_through_sequence);
    }

    pub fn deinitNamespaceMutationStats(allocator: Allocator, stats: []mutation_log.NamespaceMutationStats) void {
        graph_store.GraphStore.deinitNamespaceMutationStats(allocator, stats);
    }

    pub fn applyReplicatedBatch(
        self: *Self,
        allocator: Allocator,
        gateway_id: []const u8,
        lab_id: []const u8,
        batch: []const federation_types.MutationBatchItem,
    ) !federation_types.MutationBatchAck {
        return self.store.applyReplicatedBatch(allocator, gateway_id, lab_id, batch);
    }
};
