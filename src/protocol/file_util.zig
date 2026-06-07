const std = @import("std");

const Allocator = std.mem.Allocator;

pub fn readFileAllocAtPath(allocator: Allocator, path: []const u8, max_bytes: usize) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();
        return try file.readToEndAlloc(allocator, max_bytes);
    }
    return try std.fs.cwd().readFileAlloc(allocator, path, max_bytes);
}

pub fn writeFileAtPath(path: []const u8, data: []const u8) !void {
    if (std.fs.path.isAbsolute(path)) {
        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();
        try file.writeAll(data);
        return;
    }
    try std.fs.cwd().writeFile(.{ .sub_path = path, .data = data });
}

pub fn ensureDirAtPath(path: []const u8) !void {
    if (std.fs.path.isAbsolute(path)) {
        var root = try std.fs.openDirAbsolute("/", .{});
        defer root.close();
        try root.makePath(path[1..]);
        return;
    }
    try std.fs.cwd().makePath(path);
}

pub fn fileExistsAtPath(path: []const u8) bool {
    if (std.fs.path.isAbsolute(path)) {
        const file = std.fs.openFileAbsolute(path, .{}) catch return false;
        file.close();
        return true;
    }
    const file = std.fs.cwd().openFile(path, .{}) catch return false;
    file.close();
    return true;
}

pub fn deleteTreeAtPath(path: []const u8) void {
    if (std.fs.path.isAbsolute(path)) {
        var root = std.fs.openDirAbsolute("/", .{}) catch |err| {
            std.log.warn("Failed to open filesystem root while deleting '{s}': {}", .{ path, err });
            return;
        };
        defer root.close();
        root.deleteTree(path[1..]) catch |err| {
            std.log.warn("Failed to delete tree '{s}': {}", .{ path, err });
        };
        return;
    }
    std.fs.cwd().deleteTree(path) catch |err| {
        std.log.warn("Failed to delete tree '{s}': {}", .{ path, err });
    };
}

test "absolute file helpers round trip" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const nested_dir = try std.fs.path.join(allocator, &.{ tmp_path, "nested" });
    defer allocator.free(nested_dir);
    try ensureDirAtPath(nested_dir);

    const file_path = try std.fs.path.join(allocator, &.{ nested_dir, "payload.bin" });
    defer allocator.free(file_path);

    try writeFileAtPath(file_path, "payload");
    try std.testing.expect(fileExistsAtPath(file_path));

    const bytes = try readFileAllocAtPath(allocator, file_path, 1024);
    defer allocator.free(bytes);
    try std.testing.expectEqualStrings("payload", bytes);

    deleteTreeAtPath(nested_dir);
    try std.testing.expect(!fileExistsAtPath(file_path));
}
