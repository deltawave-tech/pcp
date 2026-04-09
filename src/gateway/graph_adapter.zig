const std = @import("std");

const Allocator = std.mem.Allocator;
const graph_store = @import("../graph/store.zig");
const graph_types = @import("../graph/types.zig");

pub const MemoryGraph = struct {
    store: graph_store.MemoryGraphStore,

    const Self = @This();

    pub fn init(allocator: Allocator, graph_backend: []const u8, gateway_id: []const u8, lab_id: []const u8) !Self {
        return .{ .store = try graph_store.MemoryGraphStore.init(allocator, graph_backend, gateway_id, lab_id) };
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
