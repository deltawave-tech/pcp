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
        var it = self.models.iterator();
        while (it.next()) |entry| {
            self.freeRecord(entry.value_ptr.*);
        }
        self.models.deinit();
    }

    pub fn addModel(self: *Self, record: types.ModelRecord) !void {
        const owned = try self.cloneRecord(record);
        errdefer self.freeRecord(owned);
        if (self.models.fetchRemove(owned.model_id)) |kv| {
            self.freeRecord(kv.value);
        }
        try self.models.put(owned.model_id, owned);
    }

    pub fn get(self: *Self, model_id: []const u8) ?types.ModelRecord {
        return self.models.get(model_id);
    }

    pub fn contains(self: *Self, model_id: []const u8) bool {
        return self.models.contains(model_id);
    }

    fn cloneRecord(self: *Self, record: types.ModelRecord) !types.ModelRecord {
        return .{
            .model_id = try self.allocator.dupe(u8, record.model_id),
            .generation_vmfb_path = try self.allocator.dupe(u8, record.generation_vmfb_path),
            .generation_mlir_path = try self.allocator.dupe(u8, record.generation_mlir_path),
            .weights_path = try self.allocator.dupe(u8, record.weights_path),
            .tokenizer_path = try self.allocator.dupe(u8, record.tokenizer_path),
            .max_context_tokens = record.max_context_tokens,
            .default_max_output_tokens = record.default_max_output_tokens,
            .pool_name = try self.allocator.dupe(u8, record.pool_name),
        };
    }

    fn freeRecord(self: *Self, record: types.ModelRecord) void {
        self.allocator.free(record.model_id);
        self.allocator.free(record.generation_vmfb_path);
        self.allocator.free(record.generation_mlir_path);
        self.allocator.free(record.weights_path);
        self.allocator.free(record.tokenizer_path);
        self.allocator.free(record.pool_name);
    }
};
