const std = @import("std");

const Allocator = std.mem.Allocator;

pub const field_name = "data_assignment";

pub const kind_byte_range = "byte_range";
pub const kind_record_range = "record_range";
pub const kind_record_shard = "record_shard";
pub const kind_uri_dataset = "uri_dataset";
pub const kind_materialized_blob = "materialized_blob";

pub const provider_local_path = "local_path";
pub const provider_uri = "uri";
pub const provider_materialized_blob = "materialized_blob";

pub const default_train_records_file = "train.jsonl";

pub const BlobRef = struct {
    name: []const u8,
    total_bytes: usize,
    byte_hash: u64,
};

pub const DataAssignment = struct {
    kind: []const u8,
    provider: []const u8 = provider_local_path,

    path: ?[]const u8 = null,
    uri: ?[]const u8 = null,
    blob: ?BlobRef = null,

    offset: usize = 0,
    length: usize = 0,
    chunk_id: usize = 0,

    start_record: usize = 0,
    record_count: ?usize = null,
    shard_index: usize = 0,
    shard_count: usize = 1,

    file_name: ?[]const u8 = null,
    output_name: ?[]const u8 = null,
};

pub fn blobRef(name: []const u8, bytes: []const u8) BlobRef {
    return .{
        .name = name,
        .total_bytes = bytes.len,
        .byte_hash = std.hash.Wyhash.hash(0, bytes),
    };
}

pub fn toJsonValue(allocator: Allocator, assignment: DataAssignment) !std.json.Parsed(std.json.Value) {
    const json = try std.json.stringifyAlloc(allocator, assignment, .{});
    defer allocator.free(json);
    return try std.json.parseFromSlice(std.json.Value, allocator, json, .{ .allocate = .alloc_always });
}
