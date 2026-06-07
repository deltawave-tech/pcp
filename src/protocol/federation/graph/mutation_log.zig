const std = @import("std");

const Allocator = std.mem.Allocator;
const types = @import("types.zig");

pub const MutationRecord = struct {
    mutation_id: []u8,
    gateway_id: []u8,
    lab_id: []u8,
    namespace_id: []u8,
    mutation_type: types.MutationType,
    target_type: []u8,
    target_id: []u8,
    payload_json: []u8,
    visibility: types.Visibility,
    provenance_json: []u8,
    timestamp: i64,
    sequence_no: u64,

    pub fn deinit(self: *MutationRecord, allocator: Allocator) void {
        allocator.free(self.mutation_id);
        allocator.free(self.gateway_id);
        allocator.free(self.lab_id);
        allocator.free(self.namespace_id);
        allocator.free(self.target_type);
        allocator.free(self.target_id);
        allocator.free(self.payload_json);
        allocator.free(self.provenance_json);
    }
};

pub const NamespaceMutationStats = struct {
    namespace_id: []u8,
    total_mutations: usize,
    pending_mutations: usize,
    local_mutations: usize,
    shared_mutations: usize,
    global_mutations: usize,
    entity_mutations: usize,
    relation_mutations: usize,
    observation_mutations: usize,
    policy_mutations: usize,
    last_sequence_no: u64,
    replicated_through_sequence: u64,

    pub fn deinit(self: *NamespaceMutationStats, allocator: Allocator) void {
        allocator.free(self.namespace_id);
    }
};

pub fn cloneRecord(allocator: Allocator, record: MutationRecord) !MutationRecord {
    return .{
        .mutation_id = try allocator.dupe(u8, record.mutation_id),
        .gateway_id = try allocator.dupe(u8, record.gateway_id),
        .lab_id = try allocator.dupe(u8, record.lab_id),
        .namespace_id = try allocator.dupe(u8, record.namespace_id),
        .mutation_type = record.mutation_type,
        .target_type = try allocator.dupe(u8, record.target_type),
        .target_id = try allocator.dupe(u8, record.target_id),
        .payload_json = try allocator.dupe(u8, record.payload_json),
        .visibility = record.visibility,
        .provenance_json = try allocator.dupe(u8, record.provenance_json),
        .timestamp = record.timestamp,
        .sequence_no = record.sequence_no,
    };
}

pub const AppendRequest = struct {
    mutation_id: ?[]const u8 = null,
    namespace_id: []const u8,
    mutation_type: types.MutationType,
    target_type: []const u8,
    target_id: []const u8,
    payload_json: []const u8,
    visibility: types.Visibility,
    provenance_json: []const u8,
    timestamp: i64,
};

pub const AppendResult = struct {
    mutation_id: []const u8,
    sequence_no: u64,
    duplicate: bool,
};

pub const MemoryMutationStore = struct {
    allocator: Allocator,
    records: std.ArrayList(MutationRecord),
    next_sequence_no: u64,
    last_replicated_sequence: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .records = std.ArrayList(MutationRecord).init(allocator),
            .next_sequence_no = 1,
            .last_replicated_sequence = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.records.items) |*record| {
            record.deinit(self.allocator);
        }
        self.records.deinit();
    }

    pub fn append(self: *Self, gateway_id: []const u8, lab_id: []const u8, request: AppendRequest) !AppendResult {
        if (request.mutation_id) |requested_id| {
            for (self.records.items) |record| {
                if (std.mem.eql(u8, record.mutation_id, requested_id)) {
                    return .{
                        .mutation_id = record.mutation_id,
                        .sequence_no = record.sequence_no,
                        .duplicate = true,
                    };
                }
            }
        }

        const sequence_no = self.next_sequence_no;
        self.next_sequence_no += 1;
        const owned_mutation_id = if (request.mutation_id) |requested_id|
            try self.allocator.dupe(u8, requested_id)
        else
            try std.fmt.allocPrint(self.allocator, "mut_{d}", .{sequence_no});
        errdefer self.allocator.free(owned_mutation_id);

        const record = MutationRecord{
            .mutation_id = owned_mutation_id,
            .gateway_id = try self.allocator.dupe(u8, gateway_id),
            .lab_id = try self.allocator.dupe(u8, lab_id),
            .namespace_id = try self.allocator.dupe(u8, request.namespace_id),
            .mutation_type = request.mutation_type,
            .target_type = try self.allocator.dupe(u8, request.target_type),
            .target_id = try self.allocator.dupe(u8, request.target_id),
            .payload_json = try self.allocator.dupe(u8, request.payload_json),
            .visibility = request.visibility,
            .provenance_json = try self.allocator.dupe(u8, request.provenance_json),
            .timestamp = request.timestamp,
            .sequence_no = sequence_no,
        };
        errdefer {
            var cleanup = record;
            cleanup.deinit(self.allocator);
        }

        try self.records.append(record);
        return .{
            .mutation_id = self.records.items[self.records.items.len - 1].mutation_id,
            .sequence_no = sequence_no,
            .duplicate = false,
        };
    }

    pub fn count(self: *Self) usize {
        return self.records.items.len;
    }

    pub fn containsMutationId(self: *Self, mutation_id: []const u8) bool {
        for (self.records.items) |record| {
            if (std.mem.eql(u8, record.mutation_id, mutation_id)) return true;
        }
        return false;
    }

    pub fn countPending(self: *Self) usize {
        const replicated = @as(usize, @intCast(self.last_replicated_sequence));
        if (replicated >= self.records.items.len) return 0;
        return self.records.items.len - replicated;
    }

    pub fn lastSequence(self: *Self) u64 {
        if (self.next_sequence_no == 1) return 0;
        return self.next_sequence_no - 1;
    }

    pub fn markReplicatedThrough(self: *Self, sequence_no: u64) void {
        if (sequence_no > self.last_replicated_sequence) {
            self.last_replicated_sequence = @min(sequence_no, self.lastSequence());
        }
    }

    pub fn snapshotFrom(self: *Self, allocator: Allocator, after_sequence_no: u64, max_items: usize) ![]MutationRecord {
        var matches = std.ArrayList(MutationRecord).init(allocator);
        errdefer {
            for (matches.items) |*record| record.deinit(allocator);
            matches.deinit();
        }

        for (self.records.items) |record| {
            if (record.sequence_no <= after_sequence_no) continue;
            try matches.append(try cloneRecord(allocator, record));
            if (matches.items.len >= max_items) break;
        }

        return matches.toOwnedSlice();
    }

    pub fn listNamespaceStats(self: *Self, allocator: Allocator, replicated_through_sequence: ?u64) ![]NamespaceMutationStats {
        var stats = std.ArrayList(NamespaceMutationStats).init(allocator);
        errdefer {
            for (stats.items) |*item| item.deinit(allocator);
            stats.deinit();
        }

        for (self.records.items) |record| {
            const stat = try ensureNamespaceStat(&stats, allocator, record.namespace_id, replicated_through_sequence);
            stat.total_mutations += 1;
            stat.last_sequence_no = @max(stat.last_sequence_no, record.sequence_no);
            stat.replicated_through_sequence = if (replicated_through_sequence) |value|
                @min(value, stat.last_sequence_no)
            else
                stat.last_sequence_no;

            switch (record.visibility) {
                .local => stat.local_mutations += 1,
                .shared => stat.shared_mutations += 1,
                .global => stat.global_mutations += 1,
            }

            switch (record.mutation_type) {
                .upsert_entity, .delete_entity => stat.entity_mutations += 1,
                .upsert_relation, .delete_relation => stat.relation_mutations += 1,
                .append_observation => stat.observation_mutations += 1,
                .policy_update => stat.policy_mutations += 1,
            }

            if (replicated_through_sequence) |value| {
                if (record.sequence_no > value) stat.pending_mutations += 1;
            }
        }

        return stats.toOwnedSlice();
    }
};

pub const GraphMutationStore = union(enum) {
    memory: MemoryMutationStore,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{ .memory = MemoryMutationStore.init(allocator) };
    }

    pub fn deinit(self: *Self) void {
        switch (self.*) {
            .memory => |*store| store.deinit(),
        }
    }

    pub fn append(self: *Self, gateway_id: []const u8, lab_id: []const u8, request: AppendRequest) !AppendResult {
        return switch (self.*) {
            .memory => |*store| store.append(gateway_id, lab_id, request),
        };
    }

    pub fn count(self: *Self) usize {
        return switch (self.*) {
            .memory => |*store| store.count(),
        };
    }

    pub fn countPending(self: *Self) usize {
        return switch (self.*) {
            .memory => |*store| store.countPending(),
        };
    }

    pub fn containsMutationId(self: *Self, mutation_id: []const u8) bool {
        return switch (self.*) {
            .memory => |*store| store.containsMutationId(mutation_id),
        };
    }

    pub fn lastSequence(self: *Self) u64 {
        return switch (self.*) {
            .memory => |*store| store.lastSequence(),
        };
    }

    pub fn lastReplicatedSequence(self: *Self) u64 {
        return switch (self.*) {
            .memory => |*store| store.last_replicated_sequence,
        };
    }

    pub fn markReplicatedThrough(self: *Self, sequence_no: u64) void {
        switch (self.*) {
            .memory => |*store| store.markReplicatedThrough(sequence_no),
        }
    }

    pub fn snapshotFrom(self: *Self, allocator: Allocator, after_sequence_no: u64, max_items: usize) ![]MutationRecord {
        return switch (self.*) {
            .memory => |*store| store.snapshotFrom(allocator, after_sequence_no, max_items),
        };
    }

    pub fn listNamespaceStats(self: *Self, allocator: Allocator, replicated_through_sequence: ?u64) ![]NamespaceMutationStats {
        return switch (self.*) {
            .memory => |*store| store.listNamespaceStats(allocator, replicated_through_sequence),
        };
    }

    pub fn deinitNamespaceStats(allocator: Allocator, stats: []NamespaceMutationStats) void {
        for (stats) |*item| item.deinit(allocator);
        allocator.free(stats);
    }
};

fn ensureNamespaceStat(
    stats: *std.ArrayList(NamespaceMutationStats),
    allocator: Allocator,
    namespace_id: []const u8,
    replicated_through_sequence: ?u64,
) !*NamespaceMutationStats {
    for (stats.items) |*item| {
        if (std.mem.eql(u8, item.namespace_id, namespace_id)) return item;
    }

    try stats.append(.{
        .namespace_id = try allocator.dupe(u8, namespace_id),
        .total_mutations = 0,
        .pending_mutations = 0,
        .local_mutations = 0,
        .shared_mutations = 0,
        .global_mutations = 0,
        .entity_mutations = 0,
        .relation_mutations = 0,
        .observation_mutations = 0,
        .policy_mutations = 0,
        .last_sequence_no = 0,
        .replicated_through_sequence = replicated_through_sequence orelse 0,
    });
    return &stats.items[stats.items.len - 1];
}
