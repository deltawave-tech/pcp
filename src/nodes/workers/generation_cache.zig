const std = @import("std");
const tensor = @import("../../core/tensor.zig");

pub fn dataInputDType(data_dtypes: ?[]const tensor.DType, index: usize) tensor.DType {
    if (data_dtypes) |dtypes| {
        if (index < dtypes.len) return dtypes[index];
    }
    return if (index < 2) .i64 else .f32;
}

fn tensorElementCount(shape: []const i64) !usize {
    var count: usize = 1;
    for (shape) |dim| {
        if (dim < 0) return error.InvalidTensorShape;
        count = std.math.mul(usize, count, @intCast(dim)) catch return error.TensorShapeTooLarge;
    }
    return count;
}

pub fn tensorByteSize(shape: []const i64, dtype: tensor.DType) !usize {
    return std.math.mul(usize, try tensorElementCount(shape), dtype.sizeInBytes()) catch return error.TensorShapeTooLarge;
}

fn findCacheSequenceDim(cache_shape: []const i64, update_shape: []const i64) !usize {
    if (cache_shape.len != update_shape.len) return error.CacheUpdateRankMismatch;

    var sequence_dim: ?usize = null;
    for (cache_shape, update_shape, 0..) |cache_dim, update_dim, dim| {
        if (cache_dim <= 0 or update_dim <= 0) return error.InvalidTensorShape;
        if (cache_dim == update_dim) continue;
        if (update_dim > cache_dim) return error.CacheUpdateShapeMismatch;
        if (sequence_dim != null) return error.AmbiguousCacheSequenceDimension;
        sequence_dim = dim;
    }
    return sequence_dim orelse error.MissingCacheSequenceDimension;
}

pub fn copyCacheUpdateAtPosition(
    cache_buffer: []u8,
    cache_shape: []const i64,
    cache_dtype: tensor.DType,
    update_bytes: []const u8,
    update_shape: []const i64,
    position: i64,
) !void {
    if (position < 0) return error.InvalidCachePosition;
    const sequence_dim = try findCacheSequenceDim(cache_shape, update_shape);

    const expected_cache_bytes = try tensorByteSize(cache_shape, cache_dtype);
    if (cache_buffer.len != expected_cache_bytes) return error.CacheBufferByteMismatch;

    const expected_update_bytes = try tensorByteSize(update_shape, cache_dtype);
    if (update_bytes.len != expected_update_bytes) return error.CacheUpdateByteMismatch;

    const cache_seq: usize = @intCast(cache_shape[sequence_dim]);
    const update_seq: usize = @intCast(update_shape[sequence_dim]);
    const start_pos: usize = @intCast(position);
    if (start_pos + update_seq > cache_seq) return error.CacheUpdateOutOfBounds;

    var outer_count: usize = 1;
    for (cache_shape[0..sequence_dim], update_shape[0..sequence_dim]) |cache_dim, update_dim| {
        if (cache_dim != update_dim) return error.CacheUpdateShapeMismatch;
        outer_count = std.math.mul(usize, outer_count, @intCast(cache_dim)) catch return error.TensorShapeTooLarge;
    }

    var inner_count: usize = 1;
    for (cache_shape[sequence_dim + 1 ..], update_shape[sequence_dim + 1 ..]) |cache_dim, update_dim| {
        if (cache_dim != update_dim) return error.CacheUpdateShapeMismatch;
        inner_count = std.math.mul(usize, inner_count, @intCast(cache_dim)) catch return error.TensorShapeTooLarge;
    }

    const element_size = cache_dtype.sizeInBytes();
    const row_bytes = std.math.mul(usize, inner_count, element_size) catch return error.TensorShapeTooLarge;

    for (0..outer_count) |outer| {
        for (0..update_seq) |update_pos| {
            const src_offset = ((outer * update_seq) + update_pos) * row_bytes;
            const dst_offset = ((outer * cache_seq) + start_pos + update_pos) * row_bytes;
            @memcpy(
                cache_buffer[dst_offset .. dst_offset + row_bytes],
                update_bytes[src_offset .. src_offset + row_bytes],
            );
        }
    }
}

pub fn copyStateUpdate(
    state_buffer: []u8,
    state_shape: []const i64,
    state_dtype: tensor.DType,
    update_bytes: []const u8,
    update_shape: []const i64,
    position: i64,
) !void {
    const expected_state_bytes = try tensorByteSize(state_shape, state_dtype);
    if (state_buffer.len != expected_state_bytes) return error.CacheBufferByteMismatch;

    const expected_update_bytes = try tensorByteSize(update_shape, state_dtype);
    if (update_bytes.len != expected_update_bytes) return error.CacheUpdateByteMismatch;

    if (std.mem.eql(i64, state_shape, update_shape)) {
        @memcpy(state_buffer, update_bytes);
        return;
    }

    try copyCacheUpdateAtPosition(
        state_buffer,
        state_shape,
        state_dtype,
        update_bytes,
        update_shape,
        position,
    );
}

test "generation cache sizing follows metadata dtype" {
    const qwen25_cache_shape = &.{ 1, 2, 1024, 64 };
    const qwen3_cache_shape = &.{ 1, 8, 1024, 128 };

    try std.testing.expectEqual(@as(usize, 1 * 2 * 1024 * 64 * 4), try tensorByteSize(qwen25_cache_shape, .f32));
    try std.testing.expectEqual(@as(usize, 1 * 8 * 1024 * 128 * 4), try tensorByteSize(qwen3_cache_shape, .f32));
    try std.testing.expectEqual(@as(usize, 1 * 8 * 1024 * 128 * 2), try tensorByteSize(qwen3_cache_shape, .bf16));
}

test "generation state copy replaces recurrent state when update shape matches slot shape" {
    const state_shape = &.{ 1, 2, 2 };
    var state = [_]f32{0} ** 4;
    const update = [_]f32{ 1, 2, 3, 4 };

    try copyStateUpdate(
        std.mem.sliceAsBytes(state[0..]),
        state_shape,
        .f32,
        std.mem.sliceAsBytes(update[0..]),
        state_shape,
        9,
    );

    try std.testing.expectEqual(@as(f32, 1), state[0]);
    try std.testing.expectEqual(@as(f32, 4), state[3]);
}

test "generation data input dtype falls back to legacy Qwen2 contract" {
    try std.testing.expectEqual(tensor.DType.i64, dataInputDType(null, 0));
    try std.testing.expectEqual(tensor.DType.i64, dataInputDType(null, 1));
    try std.testing.expectEqual(tensor.DType.f32, dataInputDType(null, 2));

    const dtypes = [_]tensor.DType{ .i32, .i32, .bf16 };
    try std.testing.expectEqual(tensor.DType.i32, dataInputDType(&dtypes, 0));
    try std.testing.expectEqual(tensor.DType.i32, dataInputDType(&dtypes, 1));
    try std.testing.expectEqual(tensor.DType.bf16, dataInputDType(&dtypes, 2));
    try std.testing.expectEqual(tensor.DType.f32, dataInputDType(&dtypes, 3));
}

test "generation cache copy uses Qwen2.5 metadata geometry" {
    const cache_shape = &.{ 1, 2, 4, 3 };
    const update_shape = &.{ 1, 2, 1, 3 };
    var cache = [_]f32{0} ** (1 * 2 * 4 * 3);
    const update = [_]f32{ 1, 2, 3, 4, 5, 6 };

    try copyCacheUpdateAtPosition(
        std.mem.sliceAsBytes(cache[0..]),
        cache_shape,
        .f32,
        std.mem.sliceAsBytes(update[0..]),
        update_shape,
        2,
    );

    try std.testing.expectEqual(@as(f32, 1), cache[(0 * 4 + 2) * 3 + 0]);
    try std.testing.expectEqual(@as(f32, 3), cache[(0 * 4 + 2) * 3 + 2]);
    try std.testing.expectEqual(@as(f32, 4), cache[(1 * 4 + 2) * 3 + 0]);
    try std.testing.expectEqual(@as(f32, 6), cache[(1 * 4 + 2) * 3 + 2]);
    try std.testing.expectEqual(@as(f32, 0), cache[(1 * 4 + 1) * 3 + 2]);
}

test "generation cache copy uses Qwen3 metadata geometry" {
    const cache_shape = &.{ 1, 8, 4, 5 };
    const update_shape = &.{ 1, 8, 1, 5 };
    var cache = [_]f32{0} ** (1 * 8 * 4 * 5);
    var update: [1 * 8 * 1 * 5]f32 = undefined;
    for (&update, 0..) |*value, index| value.* = @floatFromInt(index + 1);

    try copyCacheUpdateAtPosition(
        std.mem.sliceAsBytes(cache[0..]),
        cache_shape,
        .f32,
        std.mem.sliceAsBytes(update[0..]),
        update_shape,
        3,
    );

    try std.testing.expectEqual(@as(f32, 1), cache[(0 * 4 + 3) * 5 + 0]);
    try std.testing.expectEqual(@as(f32, 5), cache[(0 * 4 + 3) * 5 + 4]);
    try std.testing.expectEqual(@as(f32, 36), cache[(7 * 4 + 3) * 5 + 0]);
    try std.testing.expectEqual(@as(f32, 40), cache[(7 * 4 + 3) * 5 + 4]);
    try std.testing.expectEqual(@as(f32, 0), cache[(7 * 4 + 2) * 5 + 4]);
}
