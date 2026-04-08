const std = @import("std");

const Allocator = std.mem.Allocator;

pub const MemoryGraph = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    graph_backend: []const u8,
    namespace_count: usize,
    entity_count: usize,
    relation_count: usize,
    observation_count: usize,
    mutation_count: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, graph_backend: []const u8) Self {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .graph_backend = graph_backend,
            .namespace_count = 0,
            .entity_count = 0,
            .relation_count = 0,
            .observation_count = 0,
            .mutation_count = 0,
        };
    }

    pub fn renderStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        return std.json.stringifyAlloc(allocator, .{
            .backend = self.graph_backend,
            .namespaces = self.namespace_count,
            .entities = self.entity_count,
            .relations = self.relation_count,
            .observations = self.observation_count,
            .mutations = self.mutation_count,
        }, .{});
    }
};
