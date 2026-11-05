const std = @import("std");
const json_util = @import("../protocol/json_util.zig");

const Allocator = std.mem.Allocator;

pub const NodeId = u16;
pub const Counter = u64;

pub const Entry = struct {
    node_id: NodeId,
    counter: Counter,
};

pub const Ordering = enum {
    equal,
    before,
    after,
    concurrent,
};

pub const VectorClock = struct {
    allocator: Allocator,
    entries: []Entry = &.{},

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn initWithNodes(allocator: Allocator, nodes: []const NodeId) !Self {
        var clock = init(allocator);
        errdefer clock.deinit();
        for (nodes) |node_id| {
            try clock.observe(node_id, 0);
        }
        return clock;
    }

    pub fn deinit(self: *Self) void {
        if (self.entries.len > 0) self.allocator.free(self.entries);
        self.* = undefined;
    }

    pub fn clone(self: Self, allocator: Allocator) !Self {
        return .{
            .allocator = allocator,
            .entries = try allocator.dupe(Entry, self.entries),
        };
    }

    pub fn tick(self: *Self, node_id: NodeId) !Counter {
        const current = self.counter(node_id);
        const next = try std.math.add(Counter, current, 1);
        try self.observe(node_id, next);
        return next;
    }

    pub fn observe(self: *Self, node_id: NodeId, counter_value: Counter) !void {
        const index = self.indexOf(node_id);
        if (index) |i| {
            self.entries[i].counter = @max(self.entries[i].counter, counter_value);
            return;
        }

        const insert_at = self.insertionIndex(node_id);
        var next_entries = try self.allocator.alloc(Entry, self.entries.len + 1);
        errdefer self.allocator.free(next_entries);

        @memcpy(next_entries[0..insert_at], self.entries[0..insert_at]);
        next_entries[insert_at] = .{ .node_id = node_id, .counter = counter_value };
        @memcpy(next_entries[insert_at + 1 ..], self.entries[insert_at..]);

        if (self.entries.len > 0) self.allocator.free(self.entries);
        self.entries = next_entries;
    }

    pub fn merge(self: *Self, other: Self) !void {
        for (other.entries) |entry| {
            try self.observe(entry.node_id, entry.counter);
        }
    }

    pub fn counter(self: Self, node_id: NodeId) Counter {
        if (self.indexOf(node_id)) |index| return self.entries[index].counter;
        return 0;
    }

    pub fn compare(self: Self, other: Self) Ordering {
        var saw_less = false;
        var saw_greater = false;

        var i: usize = 0;
        var j: usize = 0;
        while (i < self.entries.len or j < other.entries.len) {
            const self_node = if (i < self.entries.len) self.entries[i].node_id else std.math.maxInt(NodeId);
            const other_node = if (j < other.entries.len) other.entries[j].node_id else std.math.maxInt(NodeId);

            if (self_node == other_node) {
                const left = self.entries[i].counter;
                const right = other.entries[j].counter;
                if (left < right) saw_less = true;
                if (left > right) saw_greater = true;
                i += 1;
                j += 1;
            } else if (self_node < other_node) {
                if (self.entries[i].counter > 0) saw_greater = true;
                i += 1;
            } else {
                if (other.entries[j].counter > 0) saw_less = true;
                j += 1;
            }

            if (saw_less and saw_greater) return .concurrent;
        }

        if (saw_less) return .before;
        if (saw_greater) return .after;
        return .equal;
    }

    pub fn toJsonArray(self: Self, allocator: Allocator) !std.json.Value {
        var array = std.json.Array.init(allocator);
        errdefer array.deinit();

        for (self.entries) |entry| {
            var item = std.json.ObjectMap.init(allocator);
            errdefer item.deinit();
            try item.put("node_id", .{ .integer = entry.node_id });
            try item.put("counter", .{ .integer = @intCast(entry.counter) });
            try array.append(.{ .object = item });
        }

        return .{ .array = array };
    }

    pub fn fromJsonValue(allocator: Allocator, value: std.json.Value) !Self {
        const array = switch (value) {
            .array => |arr| arr,
            else => return error.InvalidVectorClock,
        };

        var clock = init(allocator);
        errdefer clock.deinit();
        for (array.items) |item| {
            const object = switch (item) {
                .object => |obj| obj,
                else => return error.InvalidVectorClock,
            };
            const node_value = object.get("node_id") orelse return error.InvalidVectorClock;
            const counter_json = object.get("counter") orelse return error.InvalidVectorClock;
            const node_id: NodeId = @intCast(node_value.integer);
            const counter_value: Counter = @intCast(counter_json.integer);
            try clock.observe(node_id, counter_value);
        }
        return clock;
    }

    fn indexOf(self: Self, node_id: NodeId) ?usize {
        for (self.entries, 0..) |entry, i| {
            if (entry.node_id == node_id) return i;
            if (entry.node_id > node_id) return null;
        }
        return null;
    }

    fn insertionIndex(self: Self, node_id: NodeId) usize {
        for (self.entries, 0..) |entry, i| {
            if (entry.node_id > node_id) return i;
        }
        return self.entries.len;
    }
};

test "vector clock tick observe and merge keep sorted maxima" {
    const allocator = std.testing.allocator;
    var clock = VectorClock.init(allocator);
    defer clock.deinit();

    try std.testing.expectEqual(@as(Counter, 1), try clock.tick(7));
    try clock.observe(3, 4);
    try clock.observe(7, 1);
    try clock.observe(5, 2);

    try std.testing.expectEqual(@as(usize, 3), clock.entries.len);
    try std.testing.expectEqual(@as(NodeId, 3), clock.entries[0].node_id);
    try std.testing.expectEqual(@as(Counter, 4), clock.counter(3));
    try std.testing.expectEqual(@as(Counter, 1), clock.counter(7));

    var other = try VectorClock.initWithNodes(allocator, &.{ 5, 7 });
    defer other.deinit();
    try other.observe(7, 6);
    try clock.merge(other);

    try std.testing.expectEqual(@as(Counter, 6), clock.counter(7));
    try std.testing.expectEqual(@as(Counter, 2), clock.counter(5));
}

test "vector clock comparison distinguishes causal and concurrent clocks" {
    const allocator = std.testing.allocator;
    var left = VectorClock.init(allocator);
    defer left.deinit();
    var right = VectorClock.init(allocator);
    defer right.deinit();

    try left.observe(1, 2);
    try right.observe(1, 3);
    try std.testing.expectEqual(Ordering.before, left.compare(right));

    try left.observe(2, 5);
    try std.testing.expectEqual(Ordering.concurrent, left.compare(right));

    try right.observe(2, 5);
    try std.testing.expectEqual(Ordering.before, left.compare(right));

    try left.observe(1, 3);
    try std.testing.expectEqual(Ordering.equal, left.compare(right));
}

test "vector clock round trips through JSON array" {
    const allocator = std.testing.allocator;
    var clock = VectorClock.init(allocator);
    defer clock.deinit();
    try clock.observe(2, 8);
    try clock.observe(1, 4);

    const json_value = try clock.toJsonArray(allocator);
    defer json_util.deinitJsonContainers(json_value);

    var parsed = try VectorClock.fromJsonValue(allocator, json_value);
    defer parsed.deinit();

    try std.testing.expectEqual(Ordering.equal, clock.compare(parsed));
}
