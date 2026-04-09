const std = @import("std");

const Allocator = std.mem.Allocator;
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

    pub fn init(allocator: Allocator, backend: []const u8, gateway_id: []const u8, lab_id: []const u8) !Self {
        const backend_store: BackendStore = if (std.mem.eql(u8, backend, "memory"))
            .{ .memory = store_memory.MemoryGraphStore.init(allocator) }
        else if (std.mem.eql(u8, backend, "neo4j"))
            .{ .neo4j = try store_neo4j.Neo4jGraphStore.init(allocator, backend, gateway_id, lab_id) }
        else
            return error.UnsupportedGraphBackend;

        return .{
            .allocator = allocator,
            .mutex = .{},
            .backend = try allocator.dupe(u8, backend),
            .gateway_id = try allocator.dupe(u8, gateway_id),
            .lab_id = try allocator.dupe(u8, lab_id),
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

        const counts = switch (self.backend_store) {
            .memory => |*store| store.counts(),
            .neo4j => |*store| store.counts(),
        };

        return std.json.stringifyAlloc(allocator, .{
            .backend = self.backend,
            .namespaces = counts.namespaces,
            .entities = counts.entities,
            .relations = counts.relations,
            .observations = counts.observations,
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
