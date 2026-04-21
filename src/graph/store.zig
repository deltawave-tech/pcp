const std = @import("std");

const Allocator = std.mem.Allocator;
const federation_types = @import("../federation/types.zig");
const mutation_log = @import("mutation_log.zig");
const store_memory = @import("store_memory.zig");
const store_neo4j = @import("store_neo4j.zig");
const types = @import("types.zig");

const BackendStore = union(enum) {
    memory: store_memory.MemoryGraphStore,
    neo4j: store_neo4j.Neo4jGraphStore,
};

pub const GraphStore = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    backend: []u8,
    gateway_id: []u8,
    lab_id: []u8,
    backend_store: BackendStore,
    mutation_store: mutation_log.GraphMutationStore,

    const Self = @This();

    pub const InitOptions = struct {
        backend: []const u8,
        gateway_id: []const u8,
        lab_id: []const u8,
        neo4j: ?store_neo4j.Config = null,
    };

    pub fn init(allocator: Allocator, options: InitOptions) !Self {
        const backend_store: BackendStore = if (std.mem.eql(u8, options.backend, "memory"))
            .{ .memory = store_memory.MemoryGraphStore.init(allocator) }
        else if (std.mem.eql(u8, options.backend, "neo4j"))
            .{ .neo4j = try store_neo4j.Neo4jGraphStore.init(allocator, options.neo4j orelse return error.Neo4jConfigRequired) }
        else
            return error.UnsupportedGraphBackend;

        return .{
            .allocator = allocator,
            .mutex = .{},
            .backend = try allocator.dupe(u8, options.backend),
            .gateway_id = try allocator.dupe(u8, options.gateway_id),
            .lab_id = try allocator.dupe(u8, options.lab_id),
            .backend_store = backend_store,
            .mutation_store = mutation_log.GraphMutationStore.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.mutation_store.deinit();
        switch (self.backend_store) {
            .memory => |*store| store.deinit(),
            .neo4j => |*store| store.deinit(),
        }
        self.allocator.free(self.lab_id);
        self.allocator.free(self.gateway_id);
        self.allocator.free(self.backend);
    }

    pub fn renderStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const graph_counts = self.countsLocked();

        return std.json.stringifyAlloc(allocator, .{
            .backend = self.backend,
            .namespaces = graph_counts.namespaces,
            .entities = graph_counts.entities,
            .relations = graph_counts.relations,
            .observations = graph_counts.observations,
            .mutations = self.mutation_store.count(),
            .pending_mutations = self.mutation_store.countPending(),
            .last_replicated_sequence = self.mutation_store.lastReplicatedSequence(),
            .last_sequence = self.mutation_store.lastSequence(),
            .federation_enabled = false,
        }, .{});
    }

    pub fn applyMutationsJson(self: *Self, allocator: Allocator, request: types.MutateRequest) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var mutation_ids = std.ArrayList([]const u8).init(allocator);
        defer mutation_ids.deinit();

        for (request.mutations) |mutation| {
            const mutation_type = types.MutationType.parse(mutation.mutation_type) orelse return error.InvalidMutationType;
            const visibility = try types.Visibility.parseOrDefault(mutation.visibility);
            const now = std.time.timestamp();
            const payload_json = try jsonStringifyOrDefault(allocator, mutation.payload, "null");
            defer allocator.free(payload_json);
            const provenance_json = try jsonStringifyOrDefault(allocator, mutation.provenance, "null");
            defer allocator.free(provenance_json);

            switch (self.backend_store) {
                .memory => |*store| try store.applyMutation(self.gateway_id, self.lab_id, mutation, mutation_type, visibility, provenance_json, now),
                .neo4j => |*store| try store.applyMutation(self.gateway_id, self.lab_id, mutation, mutation_type, visibility, provenance_json, now),
            }

            const append_result = try self.mutation_store.append(self.gateway_id, self.lab_id, .{
                .mutation_id = mutation.mutation_id,
                .namespace_id = mutation.namespace_id,
                .mutation_type = mutation_type,
                .target_type = targetTypeForMutation(mutation_type),
                .target_id = mutation.target_id,
                .payload_json = payload_json,
                .visibility = visibility,
                .provenance_json = provenance_json,
                .timestamp = now,
            });
            try mutation_ids.append(append_result.mutation_id);
        }

        return std.json.stringifyAlloc(allocator, .{
            .accepted = true,
            .mutation_ids = mutation_ids.items,
        }, .{});
    }

    pub fn renderQueryJson(self: *Self, allocator: Allocator, request: types.QueryRequest) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        return switch (self.backend_store) {
            .memory => |*store| store.renderQueryJson(allocator, request),
            .neo4j => |*store| store.renderQueryJson(allocator, request),
        };
    }

    pub fn counts(self: *Self) types.GraphCounts {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.countsLocked();
    }

    pub fn lastSequence(self: *Self) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.mutation_store.lastSequence();
    }

    pub fn lastReplicatedSequence(self: *Self) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.mutation_store.lastReplicatedSequence();
    }

    pub fn countPendingReplications(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.mutation_store.countPending();
    }

    pub fn markReplicatedThrough(self: *Self, sequence_no: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.mutation_store.markReplicatedThrough(sequence_no);
    }

    pub fn snapshotMutationsFrom(self: *Self, allocator: Allocator, after_sequence_no: u64, max_items: usize) ![]mutation_log.MutationRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.mutation_store.snapshotFrom(allocator, after_sequence_no, max_items);
    }

    pub fn listNamespaceMutationStats(self: *Self, allocator: Allocator, replicated_through_sequence: ?u64) ![]mutation_log.NamespaceMutationStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.mutation_store.listNamespaceStats(allocator, replicated_through_sequence);
    }

    pub fn deinitNamespaceMutationStats(allocator: Allocator, stats: []mutation_log.NamespaceMutationStats) void {
        mutation_log.GraphMutationStore.deinitNamespaceStats(allocator, stats);
    }

    pub fn applyReplicatedBatch(
        self: *Self,
        allocator: Allocator,
        gateway_id: []const u8,
        lab_id: []const u8,
        batch: []const federation_types.MutationBatchItem,
    ) !federation_types.MutationBatchAck {
        self.mutex.lock();
        defer self.mutex.unlock();

        var applied_count: usize = 0;
        var duplicate_count: usize = 0;
        var acked_sequence_no: u64 = 0;

        for (batch) |item| {
            acked_sequence_no = @max(acked_sequence_no, item.sequence_no);
            if (self.mutation_store.containsMutationId(item.mutation_id)) {
                duplicate_count += 1;
                continue;
            }

            var payload = try std.json.parseFromSlice(std.json.Value, allocator, item.payload_json, .{});
            defer payload.deinit();
            var provenance = try std.json.parseFromSlice(std.json.Value, allocator, item.provenance_json, .{});
            defer provenance.deinit();

            const mutation_type = types.MutationType.parse(item.mutation_type) orelse return error.InvalidMutationType;
            const visibility = types.Visibility.parse(item.visibility) orelse return error.InvalidVisibility;
            const mutation = types.MutationRequest{
                .mutation_id = item.mutation_id,
                .mutation_type = item.mutation_type,
                .namespace_id = item.namespace_id,
                .target_id = item.target_id,
                .payload = payload.value,
                .visibility = item.visibility,
                .provenance = provenance.value,
            };

            switch (self.backend_store) {
                .memory => |*store| try store.applyMutation(gateway_id, lab_id, mutation, mutation_type, visibility, item.provenance_json, item.timestamp),
                .neo4j => |*store| try store.applyMutation(gateway_id, lab_id, mutation, mutation_type, visibility, item.provenance_json, item.timestamp),
            }

            _ = try self.mutation_store.append(gateway_id, lab_id, .{
                .mutation_id = item.mutation_id,
                .namespace_id = item.namespace_id,
                .mutation_type = mutation_type,
                .target_type = targetTypeForMutation(mutation_type),
                .target_id = item.target_id,
                .payload_json = item.payload_json,
                .visibility = visibility,
                .provenance_json = item.provenance_json,
                .timestamp = item.timestamp,
            });
            applied_count += 1;
        }

        return .{
            .accepted = true,
            .acked_sequence_no = acked_sequence_no,
            .applied_count = applied_count,
            .duplicate_count = duplicate_count,
        };
    }

    fn countsLocked(self: *Self) types.GraphCounts {
        return switch (self.backend_store) {
            .memory => |*store| store.counts(),
            .neo4j => |*store| store.counts(),
        };
    }
};

fn jsonStringifyOrDefault(allocator: Allocator, value: ?std.json.Value, default_json: []const u8) ![]u8 {
    if (value) |inner| {
        return std.json.stringifyAlloc(allocator, inner, .{});
    }
    return allocator.dupe(u8, default_json);
}

fn targetTypeForMutation(mutation_type: types.MutationType) []const u8 {
    return switch (mutation_type) {
        .upsert_entity, .delete_entity => "entity",
        .upsert_relation, .delete_relation => "relation",
        .append_observation => "observation",
        .policy_update => "policy",
    };
}
