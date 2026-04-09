const std = @import("std");

const Allocator = std.mem.Allocator;
const types = @import("types.zig");

pub const Neo4jGraphStore = struct {
    const Self = @This();

    pub fn init(allocator: Allocator, backend: []const u8, gateway_id: []const u8, lab_id: []const u8) !Self {
        _ = allocator;
        _ = backend;
        _ = gateway_id;
        _ = lab_id;
        return .{};
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn counts(self: *Self) types.GraphCounts {
        _ = self;
        return .{};
    }

    pub fn applyMutation(self: *Self, gateway_id: []const u8, lab_id: []const u8, mutation: types.MutationRequest, mutation_type: types.MutationType, visibility: types.Visibility, provenance_json: []const u8, now: i64) !void {
        _ = self;
        _ = gateway_id;
        _ = lab_id;
        _ = mutation;
        _ = mutation_type;
        _ = visibility;
        _ = provenance_json;
        _ = now;
        return error.Neo4jBackendNotImplemented;
    }

    pub fn renderQueryJson(self: *Self, allocator: Allocator, request: types.QueryRequest) ![]u8 {
        _ = self;
        _ = allocator;
        _ = request;
        return error.Neo4jBackendNotImplemented;
    }
};
