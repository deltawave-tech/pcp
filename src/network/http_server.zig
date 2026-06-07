const std = @import("std");
const net = std.net;

const Allocator = std.mem.Allocator;

pub const HttpRequest = struct {
    allocator: Allocator,
    method: []const u8,
    path: []const u8,
    headers: std.StringHashMap([]const u8),
    body: []u8,

    pub fn deinit(self: *HttpRequest) void {
        self.allocator.free(self.method);
        self.allocator.free(self.path);
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
        self.allocator.free(self.body);
    }

    pub fn header(self: *HttpRequest, name: []const u8) ?[]const u8 {
        return self.headers.get(name);
    }
};

pub fn readRequest(stream: net.Stream, allocator: Allocator, max_body: usize) !HttpRequest {
    var buffer = std.ArrayList(u8).init(allocator);
    errdefer buffer.deinit();

    var header_end: ?usize = null;
    const max_header = 64 * 1024;

    while (header_end == null) {
        var temp: [2048]u8 = undefined;
        const n = try stream.read(&temp);
        if (n == 0) return error.ConnectionClosed;
        try buffer.appendSlice(temp[0..n]);
        if (buffer.items.len > max_header) return error.HeaderTooLarge;
        if (std.mem.indexOf(u8, buffer.items, "\r\n\r\n")) |idx| {
            header_end = idx + 4;
        }
    }

    const header_bytes = buffer.items[0..header_end.?];
    const body_prefix = buffer.items[header_end.?..];

    var lines = std.mem.splitSequence(u8, header_bytes, "\r\n");
    const request_line = lines.next() orelse return error.InvalidRequest;
    var request_parts = std.mem.splitSequence(u8, request_line, " ");
    const method_raw = request_parts.next() orelse return error.InvalidRequest;
    const path_raw = request_parts.next() orelse return error.InvalidRequest;

    var headers = std.StringHashMap([]const u8).init(allocator);
    errdefer headers.deinit();

    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (std.mem.indexOfScalar(u8, line, ':')) |sep| {
            const key_raw = std.mem.trim(u8, line[0..sep], " ");
            const val_raw = std.mem.trim(u8, line[sep + 1 ..], " ");
            const key = try allocator.alloc(u8, key_raw.len);
            const val = try allocator.alloc(u8, val_raw.len);
            @memcpy(key, key_raw);
            @memcpy(val, val_raw);
            _ = std.ascii.lowerString(key, key);
            try headers.put(key, val);
        }
    }

    const method = try allocator.dupe(u8, method_raw);
    const path = try allocator.dupe(u8, path_raw);

    var content_length: usize = 0;
    if (headers.get("content-length")) |len_str| {
        content_length = std.fmt.parseInt(usize, len_str, 10) catch return error.InvalidContentLength;
        if (content_length > max_body) return error.BodyTooLarge;
    }

    const body = try allocator.alloc(u8, content_length);
    errdefer allocator.free(body);
    var copied: usize = 0;
    if (content_length > 0) {
        const prefix_len = @min(body_prefix.len, content_length);
        if (prefix_len > 0) {
            @memcpy(body[0..prefix_len], body_prefix[0..prefix_len]);
            copied = prefix_len;
        }
        while (copied < content_length) {
            const n = try stream.read(body[copied..]);
            if (n == 0) return error.ConnectionClosed;
            copied += n;
        }
    }

    buffer.deinit();

    return HttpRequest{
        .allocator = allocator,
        .method = method,
        .path = path,
        .headers = headers,
        .body = body,
    };
}

pub fn writeResponse(
    stream: net.Stream,
    status: []const u8,
    headers: []const []const u8,
    body: []const u8,
) !void {
    var response = std.ArrayList(u8).init(std.heap.page_allocator);
    defer response.deinit();

    try response.appendSlice("HTTP/1.1 ");
    try response.appendSlice(status);
    try response.appendSlice("\r\n");
    for (headers) |header| {
        try response.appendSlice(header);
        try response.appendSlice("\r\n");
    }
    if (body.len > 0) {
        var len_buf: [32]u8 = undefined;
        const len_str = try std.fmt.bufPrint(&len_buf, "Content-Length: {d}\r\n", .{body.len});
        try response.appendSlice(len_str);
    }
    try response.appendSlice("\r\n");
    try stream.writeAll(response.items);
    if (body.len > 0) {
        try stream.writeAll(body);
    }
}
