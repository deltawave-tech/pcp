const std = @import("std");
const Allocator = std.mem.Allocator;
const message = @import("../../network/message.zig");
const NodeId = message.NodeId;

pub const ChunkState = enum { Unassigned, Assigned, Completed };

pub const DataChunk = struct {
    id: usize,
    offset: usize,
    length: usize,
    state: ChunkState,
    assigned_worker: ?NodeId,
    timestamp: i64,
};

pub const ChunkSpec = struct {
    id: usize,
    offset: usize,
    length: usize,
};

pub const DataManager = struct {
    allocator: Allocator,
    chunks: std.ArrayList(DataChunk),
    total_size: usize,
    shuffle_chunks: bool,

    // Epoch tracking
    current_epoch: usize,
    max_epochs: usize,
    completed_chunks_count: usize,
    rng: std.Random.DefaultPrng,

    pub fn init(allocator: Allocator, total_size: usize, chunk_size: usize, max_epochs: usize) !DataManager {
        return initFixed(allocator, total_size, chunk_size, max_epochs, true, @intCast(std.time.timestamp()));
    }

    pub fn initFixed(
        allocator: Allocator,
        total_size: usize,
        chunk_size: usize,
        max_epochs: usize,
        shuffle_chunks: bool,
        seed: u64,
    ) !DataManager {
        var chunks = std.ArrayList(DataChunk).init(allocator);
        var offset: usize = 0;
        var id: usize = 0;

        while (offset < total_size) {
            const len = @min(chunk_size, total_size - offset);
            try chunks.append(.{
                .id = id,
                .offset = offset,
                .length = len,
                .state = .Unassigned,
                .assigned_worker = null,
                .timestamp = 0,
            });
            offset += len;
            id += 1;
        }

        return DataManager{
            .allocator = allocator,
            .chunks = chunks,
            .total_size = total_size,
            .shuffle_chunks = shuffle_chunks,
            .current_epoch = 1,
            .max_epochs = max_epochs,
            .completed_chunks_count = 0,
            .rng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn initFromChunkSpecs(
        allocator: Allocator,
        total_size: usize,
        chunk_specs: []const ChunkSpec,
        max_epochs: usize,
        shuffle_chunks: bool,
        seed: u64,
    ) !DataManager {
        var chunks = std.ArrayList(DataChunk).init(allocator);
        errdefer chunks.deinit();
        try chunks.ensureTotalCapacity(chunk_specs.len);

        for (chunk_specs) |spec| {
            try chunks.append(.{
                .id = spec.id,
                .offset = spec.offset,
                .length = spec.length,
                .state = .Unassigned,
                .assigned_worker = null,
                .timestamp = 0,
            });
        }

        return DataManager{
            .allocator = allocator,
            .chunks = chunks,
            .total_size = total_size,
            .shuffle_chunks = shuffle_chunks,
            .current_epoch = 1,
            .max_epochs = max_epochs,
            .completed_chunks_count = 0,
            .rng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn deinit(self: *DataManager) void {
        self.chunks.deinit();
    }

    /// Helper to shuffle chunks for better statistical properties across epochs
    fn shuffleChunks(self: *DataManager) void {
        if (!self.shuffle_chunks) return;
        const random = self.rng.random();
        random.shuffle(DataChunk, self.chunks.items);
    }

    /// Resets all chunks to Unassigned for the next epoch
    fn startNextEpoch(self: *DataManager) void {
        std.log.info("DataManager: Epoch {} completed. Starting Epoch {}...", .{ self.current_epoch, self.current_epoch + 1 });

        self.current_epoch += 1;
        self.completed_chunks_count = 0;

        for (self.chunks.items) |*chunk| {
            chunk.state = .Unassigned;
            chunk.assigned_worker = null;
            chunk.timestamp = 0;
        }

        // Optional: Shuffle chunks so workers don't get the exact same sequence every epoch
        self.shuffleChunks();
    }

    /// Assigns the next available chunk. Handles dead worker timeouts and Epoch cycling.
    pub fn assignNextChunk(self: *DataManager, worker_id: NodeId) ?DataChunk {
        const now = std.time.timestamp();
        const TIMEOUT = 600; // 10 minutes

        // 1. Try to find an available chunk in the current epoch
        for (self.chunks.items) |*chunk| {
            // Assign fresh chunk OR Re-assign dead chunk
            if (chunk.state == .Unassigned or
                (chunk.state == .Assigned and (now - chunk.timestamp > TIMEOUT)))
            {
                chunk.state = .Assigned;
                chunk.assigned_worker = worker_id;
                chunk.timestamp = now;
                return chunk.*;
            }
        }

        // 2. No chunks available. Check if we are done with this epoch.
        // We only roll over if ALL chunks are actually Completed (confirmed done).
        // If some are 'Assigned' but not 'Completed', we must wait for them (or their timeout).
        if (self.completed_chunks_count >= self.chunks.items.len) {
            // Epoch is done.
            if (self.current_epoch < self.max_epochs) {
                self.startNextEpoch();
                // Recursively call to get a chunk from the new epoch
                return self.assignNextChunk(worker_id);
            } else {
                std.log.info("DataManager: All epochs ({}) completed.", .{self.max_epochs});
                return null; // Training finished
            }
        }

        return null; // No data available right now (waiting for others to finish or timeout)
    }

    pub fn markComplete(self: *DataManager, chunk_id: usize) void {
        // Find chunk by ID (since they might be shuffled)
        for (self.chunks.items) |*chunk| {
            if (chunk.id == chunk_id) {
                if (chunk.state != .Completed) {
                    chunk.state = .Completed;
                    self.completed_chunks_count += 1;
                }
                return;
            }
        }
    }

    pub fn getProgress(self: *DataManager) f32 {
        if (self.chunks.items.len == 0) return 1.0;
        return @as(f32, @floatFromInt(self.completed_chunks_count)) / @as(f32, @floatFromInt(self.chunks.items.len));
    }

    /// Serialize DataManager state to JSON for checkpoint recovery
    pub fn serialize(self: *DataManager) ![]u8 {
        // Create state structure for serialization
        const State = struct {
            current_epoch: usize,
            completed_count: usize,
            chunks_state: []const ChunkState,
        };

        // Map current chunk states
        var states = try self.allocator.alloc(ChunkState, self.chunks.items.len);
        defer self.allocator.free(states);

        for (self.chunks.items, 0..) |chunk, i| {
            states[i] = chunk.state;
        }

        const state = State{
            .current_epoch = self.current_epoch,
            .completed_count = self.completed_chunks_count,
            .chunks_state = states,
        };

        return std.json.stringifyAlloc(self.allocator, state, .{});
    }

    /// Load DataManager state from JSON for checkpoint recovery
    pub fn loadState(self: *DataManager, json_data: []u8) !void {
        const parsed = try std.json.parseFromSlice(
            struct {
                current_epoch: usize,
                completed_count: usize,
                chunks_state: []ChunkState,
            },
            self.allocator,
            json_data,
            .{},
        );
        defer parsed.deinit();

        self.current_epoch = parsed.value.current_epoch;
        self.completed_chunks_count = parsed.value.completed_count;

        // Restore chunk states
        for (self.chunks.items, 0..) |*chunk, i| {
            if (i < parsed.value.chunks_state.len) {
                chunk.state = parsed.value.chunks_state[i];
                // If it was Assigned but we crashed, reset to Unassigned so it gets picked up again
                if (chunk.state == .Assigned) {
                    chunk.state = .Unassigned;
                }
            }
        }
    }
};
