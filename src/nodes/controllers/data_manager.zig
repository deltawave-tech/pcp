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

pub const DataManager = struct {
    allocator: Allocator,
    chunks: std.ArrayList(DataChunk),
    total_size: usize,

    pub fn init(allocator: Allocator, total_size: usize, chunk_size: usize) !DataManager {
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
        return DataManager{ .allocator = allocator, .chunks = chunks, .total_size = total_size };
    }

    pub fn deinit(self: *DataManager) void {
        self.chunks.deinit();
    }

    /// Assigns the next available chunk. Handles dead worker timeouts (e.g. 10 mins).
    pub fn assignNextChunk(self: *DataManager, worker_id: NodeId) ?DataChunk {
        const now = std.time.timestamp();
        const TIMEOUT = 600;

        for (self.chunks.items) |*chunk| {
            // 1. Assign fresh chunk OR 2. Re-assign dead chunk
            if (chunk.state == .Unassigned or
                (chunk.state == .Assigned and (now - chunk.timestamp > TIMEOUT)))
            {
                chunk.state = .Assigned;
                chunk.assigned_worker = worker_id;
                chunk.timestamp = now;
                return chunk.*;
            }
        }
        return null; // No data left
    }

    pub fn markComplete(self: *DataManager, chunk_id: usize) void {
        if (chunk_id < self.chunks.items.len) {
            self.chunks.items[chunk_id].state = .Completed;
        }
    }
};
