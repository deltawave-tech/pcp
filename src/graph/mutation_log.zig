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

pub const MutationLog = struct {
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

    pub fn countPending(self: *Self) usize {
        const replicated = @as(usize, @intCast(self.last_replicated_sequence));
        if (replicated >= self.records.items.len) return 0;
        return self.records.items.len - replicated;
    }

    pub fn lastSequence(self: *Self) u64 {
        if (self.next_sequence_no == 1) return 0;
        return self.next_sequence_no - 1;
    }
};
