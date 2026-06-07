const std = @import("std");

pub const Batch = struct {
    inputs: []const u8, // Owned slice
    targets: []const u8, // Owned slice
    allocator: std.mem.Allocator,

    pub fn deinit(self: Batch) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.targets);
    }
};

pub const Dataset = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        getBatch: *const fn (ptr: *anyopaque, batch_size: usize, context_length: usize) anyerror!Batch,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn getBatch(self: Dataset, batch_size: usize, context_length: usize) !Batch {
        return self.vtable.getBatch(self.ptr, batch_size, context_length);
    }

    pub fn deinit(self: Dataset) void {
        self.vtable.deinit(self.ptr);
    }
};
