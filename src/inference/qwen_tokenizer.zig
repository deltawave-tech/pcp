const std = @import("std");

const Allocator = std.mem.Allocator;

const EncodeResponse = struct {
    ok: bool,
    tokens: ?[]i64 = null,
    error: ?[]const u8 = null,
};

const DecodeResponse = struct {
    ok: bool,
    text: ?[]const u8 = null,
    error: ?[]const u8 = null,
};

pub const QwenTokenizer = struct {
    allocator: Allocator,
    child: std.ChildProcess,
    stdin: std.fs.File.Writer,
    stdout_reader: std.io.BufferedReader(4096, std.fs.File.Reader),
    mutex: std.Thread.Mutex,

    const Self = @This();

    pub fn init(allocator: Allocator, tokenizer_path: []const u8) !Self {
        var args = std.ArrayList([]const u8).init(allocator);
        defer args.deinit();
        try args.appendSlice(&.{
            "python3",
            "-u",
            "tools/qwen_tokenizer_server.py",
            "--tokenizer-path",
            tokenizer_path,
        });

        var child = std.ChildProcess.init(args.items, allocator);
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Inherit;
        try child.spawn();

        const stdin_file = child.stdin orelse return error.TokenizerSpawnFailed;
        const stdout_file = child.stdout orelse return error.TokenizerSpawnFailed;

        return Self{
            .allocator = allocator,
            .child = child,
            .stdin = stdin_file.writer(),
            .stdout_reader = std.io.bufferedReader(stdout_file.reader()),
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self.child.kill() catch {};
        _ = self.child.wait() catch {};
    }

    pub fn encode(self: *Self, text: []const u8) ![]i64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        try payload.put("op", .{ .string = "encode" });
        try payload.put("text", .{ .string = text });

        const response_line = try self.sendAndReadLine(payload);
        defer self.allocator.free(response_line);

        var parsed = try std.json.parseFromSlice(EncodeResponse, self.allocator, response_line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        if (!parsed.value.ok) {
            return error.TokenizerFailed;
        }
        const tokens = parsed.value.tokens orelse return error.TokenizerFailed;
        const out = try self.allocator.alloc(i64, tokens.len);
        @memcpy(out, tokens);
        return out;
    }

    pub fn decode(self: *Self, tokens: []const i64) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var payload = std.json.ObjectMap.init(self.allocator);
        defer payload.deinit();
        try payload.put("op", .{ .string = "decode" });

        var token_array = std.json.Array.init(self.allocator);
        defer token_array.deinit();
        for (tokens) |t| {
            try token_array.append(.{ .integer = t });
        }
        try payload.put("tokens", .{ .array = token_array });

        const response_line = try self.sendAndReadLine(payload);
        defer self.allocator.free(response_line);

        var parsed = try std.json.parseFromSlice(DecodeResponse, self.allocator, response_line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        if (!parsed.value.ok) {
            return error.TokenizerFailed;
        }
        const text = parsed.value.text orelse return error.TokenizerFailed;
        return try self.allocator.dupe(u8, text);
    }

    fn sendAndReadLine(self: *Self, payload: std.json.ObjectMap) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        defer buffer.deinit();
        try std.json.stringify(.{ .object = payload }, .{}, buffer.writer());
        try buffer.append('\n');
        try self.stdin.writeAll(buffer.items);

        const reader = self.stdout_reader.reader();
        return try reader.readUntilDelimiterOrEofAlloc(self.allocator, '\n', 16 * 1024 * 1024) orelse error.TokenizerFailed;
    }
};
