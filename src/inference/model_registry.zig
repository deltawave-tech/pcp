const std = @import("std");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;

pub const ModelRegistry = struct {
    allocator: Allocator,
    models: StringHashMap(types.ModelRecord),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .models = StringHashMap(types.ModelRecord).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.models.deinit();
    }

    pub fn addModel(self: *Self, record: types.ModelRecord) !void {
        try self.models.put(record.model_id, record);
    }

    pub fn get(self: *Self, model_id: []const u8) ?types.ModelRecord {
        return self.models.get(model_id);
    }

    pub fn contains(self: *Self, model_id: []const u8) bool {
        return self.models.contains(model_id);
    }
};
