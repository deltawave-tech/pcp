const std = @import("std");

const Allocator = std.mem.Allocator;

pub fn cloneJsonValue(allocator: Allocator, value: std.json.Value) !std.json.Value {
    return switch (value) {
        .null => .null,
        .bool => |inner| .{ .bool = inner },
        .integer => |inner| .{ .integer = inner },
        .float => |inner| .{ .float = inner },
        .number_string => |inner| .{ .number_string = try allocator.dupe(u8, inner) },
        .string => |inner| .{ .string = try allocator.dupe(u8, inner) },
        .array => |inner| blk: {
            var array = std.json.Array.init(allocator);
            errdefer freeOwnedJsonValue(allocator, .{ .array = array });

            for (inner.items) |item| {
                const cloned_item = try cloneJsonValue(allocator, item);
                array.append(cloned_item) catch |err| {
                    freeOwnedJsonValue(allocator, cloned_item);
                    return err;
                };
            }

            break :blk .{ .array = array };
        },
        .object => |inner| blk: {
            var object = std.json.ObjectMap.init(allocator);
            errdefer freeOwnedJsonValue(allocator, .{ .object = object });

            var it = inner.iterator();
            while (it.next()) |entry| {
                const key = try allocator.dupe(u8, entry.key_ptr.*);
                const cloned_value = try cloneJsonValue(allocator, entry.value_ptr.*);
                object.put(key, cloned_value) catch |err| {
                    allocator.free(key);
                    freeOwnedJsonValue(allocator, cloned_value);
                    return err;
                };
            }

            break :blk .{ .object = object };
        },
    };
}

pub fn freeOwnedJsonValue(allocator: Allocator, value: std.json.Value) void {
    switch (value) {
        .null, .bool, .integer, .float => {},
        .number_string, .string => |inner| allocator.free(inner),
        .array => |inner| {
            var array = inner;
            for (array.items) |item| freeOwnedJsonValue(allocator, item);
            array.deinit();
        },
        .object => |inner| {
            var object = inner;
            var it = object.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                freeOwnedJsonValue(allocator, entry.value_ptr.*);
            }
            object.deinit();
        },
    }
}

pub fn deinitJsonContainers(value: std.json.Value) void {
    switch (value) {
        .null, .bool, .integer, .float, .number_string, .string => {},
        .array => |inner| {
            var array = inner;
            for (array.items) |item| deinitJsonContainers(item);
            array.deinit();
        },
        .object => |inner| {
            var object = inner;
            var it = object.iterator();
            while (it.next()) |entry| {
                deinitJsonContainers(entry.value_ptr.*);
            }
            object.deinit();
        },
    }
}

pub fn stringifyOrDefault(allocator: Allocator, value: ?std.json.Value, default_json: []const u8) ![]u8 {
    if (value) |inner| {
        return std.json.stringifyAlloc(allocator, inner, .{});
    }
    return allocator.dupe(u8, default_json);
}

pub fn stringField(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |inner| inner,
        else => null,
    };
}

pub fn intField(object: std.json.ObjectMap, key: []const u8) ?i64 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |inner| inner,
        else => null,
    };
}

pub fn usizeField(object: std.json.ObjectMap, key: []const u8) ?usize {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |inner| @intCast(inner),
        else => null,
    };
}

pub fn u64Field(object: std.json.ObjectMap, key: []const u8) ?u64 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |inner| @intCast(inner),
        else => null,
    };
}

test "cloneJsonValue deep copies owned strings and containers" {
    const allocator = std.testing.allocator;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"name\":\"pcp\",\"items\":[1,\"two\"]}", .{});
    defer parsed.deinit();

    const cloned = try cloneJsonValue(allocator, parsed.value);
    defer freeOwnedJsonValue(allocator, cloned);

    const object = cloned.object;
    try std.testing.expectEqualStrings("pcp", object.get("name").?.string);
    try std.testing.expectEqual(@as(i64, 1), object.get("items").?.array.items[0].integer);
    try std.testing.expectEqualStrings("two", object.get("items").?.array.items[1].string);
}

test "deinitJsonContainers frees generated numeric containers" {
    const allocator = std.testing.allocator;

    var array = std.json.Array.init(allocator);
    try array.append(.{ .integer = 1 });
    try array.append(.{ .integer = 2 });

    deinitJsonContainers(.{ .array = array });
}

test "typed field helpers read expected scalar fields" {
    const allocator = std.testing.allocator;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"name\":\"pcp\",\"count\":7}", .{});
    defer parsed.deinit();
    const object = parsed.value.object;

    try std.testing.expectEqualStrings("pcp", stringField(object, "name").?);
    try std.testing.expectEqual(@as(usize, 7), usizeField(object, "count").?);
    try std.testing.expectEqual(@as(i64, 7), intField(object, "count").?);
    try std.testing.expectEqual(@as(u64, 7), u64Field(object, "count").?);
    try std.testing.expect(stringField(object, "count") == null);
}
