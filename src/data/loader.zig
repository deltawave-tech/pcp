const std = @import("std");
const Allocator = std.mem.Allocator;
const dataset_mod = @import("dataset.zig");
const Dataset = dataset_mod.Dataset;
const DatasetBatch = dataset_mod.Batch;

pub const Batch = struct {
    x: []u64,
    y: []u64,
};

pub const CharTokenizer = struct {
    allocator: Allocator,
    char_to_int: std.AutoHashMap(u8, u32),
    int_to_char: std.AutoHashMap(u32, u8),
    vocab_size: usize,

    const MAX_VOCAB_SIZE = 65;

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

        var chars = std.ArrayList(u8).init(allocator);
        defer chars.deinit();

        var it = char_set.keyIterator();
        while (it.next()) |char_ptr| {
            try chars.append(char_ptr.*);
        }

        std.sort.heap(u8, chars.items, {}, std.sort.asc(u8));

        const vocab_len: usize = @min(chars.items.len, MAX_VOCAB_SIZE);
        for (chars.items[0..vocab_len], 0..) |ch, idx| {
            const token: u32 = @intCast(idx);
            try char_to_int.put(ch, token);
            try int_to_char.put(token, ch);
        }

        if (chars.items.len > MAX_VOCAB_SIZE) {
            for (chars.items[MAX_VOCAB_SIZE..]) |ch| {
                std.log.warn("Dataset has more characters than model vocab ({}). Ignoring char '{c}'", .{ MAX_VOCAB_SIZE, ch });
            }
        }

        return CharTokenizer{
            .allocator = allocator,
            .char_to_int = char_to_int,
            .int_to_char = int_to_char,
            .vocab_size = vocab_len,
        };
    }

    pub fn deinit(self: *CharTokenizer) void {
        self.char_to_int.deinit();
        self.int_to_char.deinit();
    }

    pub fn encode(self: *CharTokenizer, text: []const u8) ![]u32 {
        const tokens = try self.allocator.alloc(u32, text.len);
        for (text, 0..) |char, i| {
            if (self.char_to_int.get(char)) |token| {
                tokens[i] = token;
            } else {
                tokens[i] = 0;
            }
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

/// ByteTokenizer provides byte-level tokenization (0-255) for dataset-agnostic models
/// This is simpler than character-level as it doesn't require building a vocab from the dataset
pub const ByteTokenizer = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) ByteTokenizer {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ByteTokenizer) void {
        _ = self;
    }

    pub fn encode(self: *ByteTokenizer, text: []const u8) ![]u32 {
        const tokens = try self.allocator.alloc(u32, text.len);
        for (text, 0..) |char, i| {
            tokens[i] = @as(u32, char); // Direct mapping 0-255
        }
        return tokens;
    }

    pub fn decode(self: *ByteTokenizer, tokens: []const u32) ![]u8 {
        const text = try self.allocator.alloc(u8, tokens.len);
        for (tokens, 0..) |token, i| {
            if (token > 255) return error.InvalidByteValue;
            text[i] = @as(u8, @intCast(token));
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
        const x = try self.allocator.alloc(u64, batch_size * block_size);
        const y = try self.allocator.alloc(u64, batch_size * block_size);

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
        const random = prng.random();

        for (0..batch_size) |i| {
            const start_index = random.uintAtMost(u32, @intCast(self.tokens.len - block_size));
            const input_slice = self.tokens[start_index .. start_index + block_size];
            const target_slice = self.tokens[start_index + 1 .. start_index + block_size + 1];

            const x_dest = x[i * block_size .. (i+1) * block_size];
            const y_dest = y[i * block_size .. (i+1) * block_size];

            for (input_slice, 0..) |token, k| x_dest[k] = @intCast(token);
            for (target_slice, 0..) |token, k| y_dest[k] = @intCast(token);
        }

        return Batch{ .x = x, .y = y };
    }
};

/// ByteDataLoader provides byte-level data loading without vocab building
pub const ByteDataLoader = struct {
    allocator: Allocator,
    tokenizer: ByteTokenizer,
    tokens: []u32, // All tokens for this worker's shard

    pub fn init(allocator: Allocator, path: []const u8) !ByteDataLoader {
        var tokenizer = ByteTokenizer.init(allocator);

        const text = try std.fs.cwd().readFileAlloc(allocator, path, 200 * 1024 * 1024); // 200MB limit for enwik8
        defer allocator.free(text);

        const all_tokens = try tokenizer.encode(text);

        return ByteDataLoader {
            .allocator = allocator,
            .tokenizer = tokenizer,
            .tokens = all_tokens,
        };
    }

    pub fn deinit(self: *ByteDataLoader) void {
        self.tokenizer.deinit();
        self.allocator.free(self.tokens);
    }

    pub fn getBatch(self: *ByteDataLoader, batch_size: usize, block_size: usize) !Batch {
        const x = try self.allocator.alloc(u64, batch_size * block_size);
        const y = try self.allocator.alloc(u64, batch_size * block_size);

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
        const random = prng.random();

        for (0..batch_size) |i| {
            const start_index = random.uintAtMost(u32, @intCast(self.tokens.len - block_size));
            const input_slice = self.tokens[start_index .. start_index + block_size];
            const target_slice = self.tokens[start_index + 1 .. start_index + block_size + 1];

            const x_dest = x[i * block_size .. (i+1) * block_size];
            const y_dest = y[i * block_size .. (i+1) * block_size];

            for (input_slice, 0..) |token, k| x_dest[k] = @intCast(token);
            for (target_slice, 0..) |token, k| y_dest[k] = @intCast(token);
        }

        return Batch{ .x = x, .y = y };
    }
};

/// TextDataset implements the Dataset interface for chunk-based data loading
pub const TextDataset = struct {
    allocator: Allocator,
    tokenizer: CharTokenizer,
    tokens: []u32, // The specific chunk of tokens for this worker
    prng: std.Random.DefaultPrng,

    const Self = @This();

    /// Initialize with explicit offset and length (Strict Partitioning)
    pub fn initChunk(allocator: Allocator, path: []const u8, offset: usize, length: usize, seed: u64) !*Self {
        var tokenizer = try CharTokenizer.initFromFile(allocator, path);

        // Read ONLY the assigned chunk
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Safety check
        const stat = try file.stat();
        if (offset >= stat.size) return error.OffsetOutOfBounds;
        const actual_len = @min(length, stat.size - offset);

        const buffer = try allocator.alloc(u8, actual_len);
        defer allocator.free(buffer);

        try file.seekTo(offset);
        _ = try file.readAll(buffer);

        // Convert chunk to tokens
        const chunk_tokens = try tokenizer.encode(buffer);

        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .tokens = chunk_tokens,
            .prng = std.Random.DefaultPrng.init(seed),
        };
        return self;
    }

    pub fn asDataset(self: *Self) Dataset {
        return Dataset{
            .ptr = self,
            .vtable = &.{ .getBatch = getBatchImpl, .deinit = deinitImpl },
        };
    }

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.tokenizer.deinit();
        self.allocator.free(self.tokens);
        self.allocator.destroy(self);
    }

    fn getBatchImpl(ptr: *anyopaque, batch_size: usize, block_size: usize) anyerror!DatasetBatch {
        const self: *Self = @ptrCast(@alignCast(ptr));

        // Allocate IREE-compatible byte buffers (u64 -> 8 bytes)
        const size_bytes = batch_size * block_size * 8;
        const x_bytes = try self.allocator.alloc(u8, size_bytes);
        errdefer self.allocator.free(x_bytes);
        const y_bytes = try self.allocator.alloc(u8, size_bytes);
        errdefer self.allocator.free(y_bytes);

        const x_flat = std.mem.bytesAsSlice(u64, x_bytes);
        const y_flat = std.mem.bytesAsSlice(u64, y_bytes);

        const random = self.prng.random();

        // Randomly sample FROM THE ASSIGNED CHUNK ONLY
        for (0..batch_size) |i| {
            if (self.tokens.len <= block_size + 1) return error.ChunkTooSmall;
            const start = random.uintAtMost(usize, self.tokens.len - block_size - 1);

            const src_x = self.tokens[start..start + block_size];
            const src_y = self.tokens[start + 1..start + block_size + 1];

            const dest_x = x_flat[i * block_size..(i + 1) * block_size];
            const dest_y = y_flat[i * block_size..(i + 1) * block_size];

            for (src_x, 0..) |tok, k| dest_x[k] = @intCast(tok);
            for (src_y, 0..) |tok, k| dest_y[k] = @intCast(tok);
        }

        return DatasetBatch{ .inputs = x_bytes, .targets = y_bytes, .allocator = self.allocator };
    }
};

/// ByteTextDataset implements Dataset interface with byte-level tokenization
/// Suitable for large datasets (enwik8, etc.) with fixed vocab size 256
pub const ByteTextDataset = struct {
    allocator: Allocator,
    tokenizer: ByteTokenizer,
    tokens: []u32, // The specific chunk of tokens for this worker
    prng: std.Random.DefaultPrng,

    const Self = @This();

    /// Initialize with explicit offset and length for distributed training
    pub fn initChunk(allocator: Allocator, path: []const u8, offset: usize, length: usize, seed: u64) !*Self {
        var tokenizer = ByteTokenizer.init(allocator);

        // Read ONLY the assigned chunk
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Safety check
        const stat = try file.stat();
        if (offset >= stat.size) return error.OffsetOutOfBounds;
        const actual_len = @min(length, stat.size - offset);

        const buffer = try allocator.alloc(u8, actual_len);
        defer allocator.free(buffer);

        try file.seekTo(offset);
        _ = try file.readAll(buffer);

        // Convert chunk to tokens (direct byte mapping)
        const chunk_tokens = try tokenizer.encode(buffer);

        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .tokens = chunk_tokens,
            .prng = std.Random.DefaultPrng.init(seed),
        };
        return self;
    }

    pub fn asDataset(self: *Self) Dataset {
        return Dataset{
            .ptr = self,
            .vtable = &.{ .getBatch = getBatchImpl, .deinit = deinitImpl },
        };
    }

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.tokenizer.deinit();
        self.allocator.free(self.tokens);
        self.allocator.destroy(self);
    }

    fn getBatchImpl(ptr: *anyopaque, batch_size: usize, block_size: usize) anyerror!DatasetBatch {
        const self: *Self = @ptrCast(@alignCast(ptr));

        // Allocate IREE-compatible byte buffers (u64 -> 8 bytes)
        const size_bytes = batch_size * block_size * 8;
        const x_bytes = try self.allocator.alloc(u8, size_bytes);
        errdefer self.allocator.free(x_bytes);
        const y_bytes = try self.allocator.alloc(u8, size_bytes);
        errdefer self.allocator.free(y_bytes);

        const x_flat = std.mem.bytesAsSlice(u64, x_bytes);
        const y_flat = std.mem.bytesAsSlice(u64, y_bytes);

        const random = self.prng.random();

        // Randomly sample FROM THE ASSIGNED CHUNK ONLY
        for (0..batch_size) |i| {
            if (self.tokens.len <= block_size + 1) return error.ChunkTooSmall;
            const start = random.uintAtMost(usize, self.tokens.len - block_size - 1);

            const src_x = self.tokens[start..start + block_size];
            const src_y = self.tokens[start + 1..start + block_size + 1];

            const dest_x = x_flat[i * block_size..(i + 1) * block_size];
            const dest_y = y_flat[i * block_size..(i + 1) * block_size];

            for (src_x, 0..) |tok, k| dest_x[k] = @intCast(tok);
            for (src_y, 0..) |tok, k| dest_y[k] = @intCast(tok);
        }

        return DatasetBatch{ .inputs = x_bytes, .targets = y_bytes, .allocator = self.allocator };
    }
};

/// U16TokenDataset implements Dataset interface for pre-tokenized u16 streams.
/// The on-disk format is a flat little-endian `u16[]` token array (nanochat IDs).
pub const U16TokenDataset = struct {
    allocator: Allocator,
    tokens: []u16, // The specific chunk of tokens for this worker
    prng: std.Random.DefaultPrng,

    const Self = @This();

    pub fn initChunk(allocator: Allocator, path: []const u8, offset: usize, length: usize, seed: u64) !*Self {
        if (offset % 2 != 0) return error.UnalignedOffset;
        if (length % 2 != 0) return error.UnalignedLength;

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        if (offset >= stat.size) return error.OffsetOutOfBounds;

        const actual_len = @min(length, stat.size - offset);
        if (actual_len % 2 != 0) return error.UnalignedLength;
        const token_count = actual_len / 2;

        const chunk_tokens = try allocator.alloc(u16, token_count);
        errdefer allocator.free(chunk_tokens);

        try file.seekTo(offset);
        const bytes = std.mem.sliceAsBytes(chunk_tokens);
        _ = try file.readAll(bytes);

        // Convert from little-endian on big-endian hosts.
        if (@import("builtin").target.cpu.arch.endian() == .big) {
            for (chunk_tokens) |*t| t.* = @byteSwap(t.*);
        }

        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .tokens = chunk_tokens,
            .prng = std.Random.DefaultPrng.init(seed),
        };
        return self;
    }

    pub fn asDataset(self: *Self) Dataset {
        return Dataset{
            .ptr = self,
            .vtable = &.{ .getBatch = getBatchImpl, .deinit = deinitImpl },
        };
    }

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.allocator.free(self.tokens);
        self.allocator.destroy(self);
    }

    fn getBatchImpl(ptr: *anyopaque, batch_size: usize, block_size: usize) anyerror!DatasetBatch {
        const self: *Self = @ptrCast(@alignCast(ptr));

        const size_bytes = batch_size * block_size * 8;
        const x_bytes = try self.allocator.alloc(u8, size_bytes);
        errdefer self.allocator.free(x_bytes);
        const y_bytes = try self.allocator.alloc(u8, size_bytes);
        errdefer self.allocator.free(y_bytes);

        const x_flat = std.mem.bytesAsSlice(u64, x_bytes);
        const y_flat = std.mem.bytesAsSlice(u64, y_bytes);

        const random = self.prng.random();

        for (0..batch_size) |i| {
            if (self.tokens.len <= block_size + 1) return error.ChunkTooSmall;
            const start = random.uintAtMost(usize, self.tokens.len - block_size - 1);

            const src_x = self.tokens[start..start + block_size];
            const src_y = self.tokens[start + 1..start + block_size + 1];

            const dest_x = x_flat[i * block_size..(i + 1) * block_size];
            const dest_y = y_flat[i * block_size..(i + 1) * block_size];

            for (src_x, 0..) |tok, k| dest_x[k] = @intCast(tok);
            for (src_y, 0..) |tok, k| dest_y[k] = @intCast(tok);
        }

        return DatasetBatch{ .inputs = x_bytes, .targets = y_bytes, .allocator = self.allocator };
    }
};

/// U16TokenDatasetFifo implements nanochat-style FIFO consumption over a token stream.
///
/// Each `getBatch(B, T)` call consumes exactly `B*T + 1` tokens from the stream:
/// - inputs are the first `B*T` tokens reshaped into `[B, T]`
/// - targets are the next-token shift over the same contiguous window (also `[B, T]`)
///
/// This matches nanochat's:
///   needed = B*T + 1
///   inputs = scratch[:-1].view(B,T)
///   targets = scratch[1:].view(B,T)
pub const U16TokenDatasetFifo = struct {
    allocator: Allocator,
    tokens: []u16, // The specific chunk of tokens for this worker
    cursor: usize,

    const Self = @This();

    pub fn initChunk(allocator: Allocator, path: []const u8, offset: usize, length: usize, seed: u64) !*Self {
        _ = seed; // FIFO mode is deterministic; seed is unused.
        if (offset % 2 != 0) return error.UnalignedOffset;
        if (length % 2 != 0) return error.UnalignedLength;

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        if (offset >= stat.size) return error.OffsetOutOfBounds;

        const actual_len = @min(length, stat.size - offset);
        if (actual_len % 2 != 0) return error.UnalignedLength;
        const token_count = actual_len / 2;

        const chunk_tokens = try allocator.alloc(u16, token_count);
        errdefer allocator.free(chunk_tokens);

        try file.seekTo(offset);
        const bytes = std.mem.sliceAsBytes(chunk_tokens);
        _ = try file.readAll(bytes);

        // Convert from little-endian on big-endian hosts.
        if (@import("builtin").target.cpu.arch.endian() == .big) {
            for (chunk_tokens) |*t| t.* = @byteSwap(t.*);
        }

        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .tokens = chunk_tokens,
            .cursor = 0,
        };
        return self;
    }

    pub fn asDataset(self: *Self) Dataset {
        return Dataset{
            .ptr = self,
            .vtable = &.{ .getBatch = getBatchImpl, .deinit = deinitImpl },
        };
    }

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.allocator.free(self.tokens);
        self.allocator.destroy(self);
    }

    fn getBatchImpl(ptr: *anyopaque, batch_size: usize, block_size: usize) anyerror!DatasetBatch {
        const self: *Self = @ptrCast(@alignCast(ptr));

        const needed_tokens = batch_size * block_size + 1;
        if (self.tokens.len < needed_tokens) return error.ChunkTooSmall;
        if (self.cursor + needed_tokens > self.tokens.len) return error.ChunkExhausted;

        const window = self.tokens[self.cursor .. self.cursor + needed_tokens];
        self.cursor += needed_tokens;

        const size_bytes = batch_size * block_size * 8;
        const x_bytes = try self.allocator.alloc(u8, size_bytes);
        errdefer self.allocator.free(x_bytes);
        const y_bytes = try self.allocator.alloc(u8, size_bytes);
        errdefer self.allocator.free(y_bytes);

        const x_flat = std.mem.bytesAsSlice(u64, x_bytes);
        const y_flat = std.mem.bytesAsSlice(u64, y_bytes);

        // Interpret the contiguous window as a flattened stream of length B*T (+1 for the shift),
        // then reshape into [B, T] exactly like nanochat.
        for (0..batch_size) |i| {
            const base = i * block_size;
            const src_x = window[base .. base + block_size];
            const src_y = window[base + 1 .. base + block_size + 1];

            const dest_x = x_flat[i * block_size .. (i + 1) * block_size];
            const dest_y = y_flat[i * block_size .. (i + 1) * block_size];

            for (src_x, 0..) |tok, k| dest_x[k] = @intCast(tok);
            for (src_y, 0..) |tok, k| dest_y[k] = @intCast(tok);
        }

        return DatasetBatch{ .inputs = x_bytes, .targets = y_bytes, .allocator = self.allocator };
    }
};