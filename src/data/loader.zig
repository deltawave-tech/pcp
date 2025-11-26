const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Batch = struct {
    x: []u32,
    y: []u32,
};

pub const CharTokenizer = struct {
    allocator: Allocator,
    char_to_int: std.AutoHashMap(u8, u32),
    int_to_char: std.AutoHashMap(u32, u8),
    vocab_size: usize,

    // Reads a text file and builds the character vocabulary
    pub fn initFromFile(allocator: Allocator, path: []const u8) !CharTokenizer {
        const text = try std.fs.cwd().readFileAlloc(allocator, path, 2 * 1024 * 1024); // 2MB limit
        defer allocator.free(text);

        var char_set = std.AutoHashMap(u8, void).init(allocator);
        defer char_set.deinit();
        for (text) |char| {
            try char_set.put(char, {});
        }

        var char_to_int = std.AutoHashMap(u8, u32).init(allocator);
        var int_to_char = std.AutoHashMap(u32, u8).init(allocator);
        
        var it = char_set.keyIterator();
        var i = @as(u32, 0);
        while (it.next()) |char_ptr| {
            try char_to_int.put(char_ptr.*, i);
            try int_to_char.put(i, char_ptr.*);
            i += 1;
        }

        return CharTokenizer {
            .allocator = allocator,
            .char_to_int = char_to_int,
            .int_to_char = int_to_char,
            .vocab_size = char_to_int.count(),
        };
    }

    pub fn deinit(self: *CharTokenizer) void {
        self.char_to_int.deinit();
        self.int_to_char.deinit();
    }

    pub fn encode(self: *CharTokenizer, text: []const u8) ![]u32 {
        const tokens = try self.allocator.alloc(u32, text.len);
        for (text, 0..) |char, i| {
            tokens[i] = self.char_to_int.get(char) orelse return error.UnknownCharacter;
        }
        return tokens;
    }

    pub fn decode(self: *CharTokenizer, tokens: []const u32) ![]u8 {
        const text = try self.allocator.alloc(u8, tokens.len);
        for (tokens, 0..) |token, i| {
            text[i] = self.int_to_char.get(token) orelse return error.UnknownToken;
        }
        return text;
    }
};

pub const DataLoader = struct {
    allocator: Allocator,
    tokenizer: CharTokenizer,
    tokens: []u32, // All tokens for this worker's shard

    // For now, init with the full dataset. Sharding will be added later.
    pub fn init(allocator: Allocator, path: []const u8) !DataLoader {
        var tokenizer = try CharTokenizer.initFromFile(allocator, path);
        
        const text = try std.fs.cwd().readFileAlloc(allocator, path, 2 * 1024 * 1024);
        defer allocator.free(text);

        const all_tokens = try tokenizer.encode(text);

        return DataLoader {
            .allocator = allocator,
            .tokenizer = tokenizer,
            .tokens = all_tokens,
        };
    }

    pub fn deinit(self: *DataLoader) void {
        self.tokenizer.deinit();
        self.allocator.free(self.tokens);
    }
    
    // Gets a single random batch of (inputs, targets)
    pub fn getBatch(self: *DataLoader, batch_size: usize, block_size: usize) !Batch {
        const x = try self.allocator.alloc(u32, batch_size * block_size);
        const y = try self.allocator.alloc(u32, batch_size * block_size);
        
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
        const random = prng.random();

        for (0..batch_size) |i| {
            const start_index = random.uintAtMost(u32, @intCast(self.tokens.len - block_size));
            const input_slice = self.tokens[start_index .. start_index + block_size];
            const target_slice = self.tokens[start_index + 1 .. start_index + block_size + 1];

            @memcpy(x[i * block_size .. (i+1) * block_size], input_slice);
            @memcpy(y[i * block_size .. (i+1) * block_size], target_slice);
        }

        return Batch{ .x = x, .y = y };
    }
};