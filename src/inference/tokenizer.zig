const std = @import("std");
const loader = @import("../data/loader.zig");

const Allocator = std.mem.Allocator;

pub const TokenizerKind = enum {
    byte,
    char,
    u16,
};

pub const Tokenizer = struct {
    allocator: Allocator,
    kind: TokenizerKind,
    char_tokenizer: ?loader.CharTokenizer,
    byte_tokenizer: ?loader.ByteTokenizer,

    const Self = @This();

    pub fn init(allocator: Allocator, kind: TokenizerKind, path: ?[]const u8) !Self {
        return switch (kind) {
            .byte => Self{
                .allocator = allocator,
                .kind = kind,
                .char_tokenizer = null,
                .byte_tokenizer = loader.ByteTokenizer.init(allocator),
            },
            .char => blk: {
                const file_path = path orelse return error.MissingTokenizerPath;
                const tokenizer = try loader.CharTokenizer.initFromFile(allocator, file_path);
                break :blk Self{
                    .allocator = allocator,
                    .kind = kind,
                    .char_tokenizer = tokenizer,
                    .byte_tokenizer = null,
                };
            },
            .u16 => Self{
                .allocator = allocator,
                .kind = kind,
                .char_tokenizer = null,
                .byte_tokenizer = loader.ByteTokenizer.init(allocator),
            },
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.char_tokenizer) |*tok| tok.deinit();
        if (self.byte_tokenizer) |*tok| tok.deinit();
    }

    pub fn encode(self: *Self, text: []const u8) ![]i64 {
        const tokens_u32 = switch (self.kind) {
            .byte, .u16 => try self.byte_tokenizer.?.encode(text),
            .char => try self.char_tokenizer.?.encode(text),
        };
        defer self.allocator.free(tokens_u32);

        const tokens_i64 = try self.allocator.alloc(i64, tokens_u32.len);
        for (tokens_u32, 0..) |t, i| {
            tokens_i64[i] = @intCast(t);
        }
        return tokens_i64;
    }

    pub fn decode(self: *Self, tokens: []const i64) ![]u8 {
        const tokens_u32 = try self.allocator.alloc(u32, tokens.len);
        defer self.allocator.free(tokens_u32);
        for (tokens, 0..) |t, i| {
            tokens_u32[i] = @intCast(t);
        }
        return switch (self.kind) {
            .byte, .u16 => try self.byte_tokenizer.?.decode(tokens_u32),
            .char => try self.char_tokenizer.?.decode(tokens_u32),
        };
    }
};
