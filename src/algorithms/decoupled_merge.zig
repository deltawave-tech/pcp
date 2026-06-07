const std = @import("std");

const Allocator = std.mem.Allocator;

pub const WeightedFragment = struct {
    values: []const f32,
    weight: f32,
};

pub const CounterWeightedFragment = struct {
    values: []const f32,
    tokens_since_update: usize,
    steps_since_update: usize,

    pub fn weighted(self: CounterWeightedFragment) WeightedFragment {
        return .{
            .values = self.values,
            .weight = tokenStepWeight(self.tokens_since_update, self.steps_since_update),
        };
    }
};

pub const MergeStats = struct {
    participant_count: usize = 0,
    total_weight: f32 = 0.0,
    direction_weight: f32 = 0.0,
    sanitized_values: usize = 0,
};

pub const CompressionStats = struct {
    value_count: usize = 0,
    byte_count: usize = 0,
    scale: f32 = 0.0,
    max_abs: f32 = 0.0,
    max_abs_error: f32 = 0.0,
    sanitized_values: usize = 0,
};

pub const Int4CompressedVector = struct {
    value_count: usize,
    scale: f32,
    bytes: []u8,

    pub fn deinit(self: *Int4CompressedVector, allocator: Allocator) void {
        allocator.free(self.bytes);
        self.* = undefined;
    }
};

pub fn tokenStepWeight(tokens_since_update: usize, steps_since_update: usize) f32 {
    if (tokens_since_update == 0) return 0.0;

    const tokens: f64 = @floatFromInt(tokens_since_update);
    const steps: f64 = @floatFromInt(@max(steps_since_update, 1));
    const weight = tokens * tokens / steps;
    const max_f32: f64 = 3.4028234663852886e38;

    if (!std.math.isFinite(weight)) return std.math.floatMax(f32);
    if (weight > max_f32) return std.math.floatMax(f32);
    return @floatCast(weight);
}

pub fn compressInt4(allocator: Allocator, values: []const f32) !Int4CompressedVector {
    var max_abs: f32 = 0.0;
    for (values) |value| {
        if (!std.math.isFinite(value)) continue;
        max_abs = @max(max_abs, @abs(value));
    }

    const packed_len = (values.len + 1) / 2;
    const packed_bytes = try allocator.alloc(u8, packed_len);
    errdefer allocator.free(packed_bytes);
    @memset(packed_bytes, 0);

    const scale = if (max_abs == 0.0) 0.0 else max_abs / 7.0;
    for (values, 0..) |value, i| {
        const finite_value = if (std.math.isFinite(value)) value else 0.0;
        const q = quantizeInt4(finite_value, scale);
        const nibble: u8 = @intCast(q + 8);
        const byte_index = i / 2;
        if (i % 2 == 0) {
            packed_bytes[byte_index] = (packed_bytes[byte_index] & 0xf0) | nibble;
        } else {
            packed_bytes[byte_index] = (packed_bytes[byte_index] & 0x0f) | (nibble << 4);
        }
    }

    return .{
        .value_count = values.len,
        .scale = scale,
        .bytes = packed_bytes,
    };
}

pub fn decompressInt4(out: []f32, compressed: Int4CompressedVector) !void {
    if (out.len != compressed.value_count) return error.OutputLengthMismatch;
    if (compressed.bytes.len != (compressed.value_count + 1) / 2) return error.InvalidCompressedLength;

    for (out, 0..) |*value, i| {
        const byte = compressed.bytes[i / 2];
        const nibble = if (i % 2 == 0) byte & 0x0f else (byte >> 4) & 0x0f;
        const q: i8 = @as(i8, @intCast(nibble)) - 8;
        value.* = if (compressed.scale == 0.0) 0.0 else @as(f32, @floatFromInt(q)) * compressed.scale;
        if (!std.math.isFinite(value.*)) value.* = 0.0;
    }
}

pub fn compressDecompressInt4InPlace(allocator: Allocator, values: []f32) !CompressionStats {
    const original = try allocator.dupe(f32, values);
    defer allocator.free(original);

    var compressed = try compressInt4(allocator, values);
    defer compressed.deinit(allocator);

    try decompressInt4(values, compressed);

    var stats = CompressionStats{
        .value_count = values.len,
        .byte_count = compressed.bytes.len,
        .scale = compressed.scale,
    };

    for (original, values) |before, after| {
        const finite_before = if (std.math.isFinite(before)) before else blk: {
            stats.sanitized_values += 1;
            break :blk 0.0;
        };
        stats.max_abs = @max(stats.max_abs, @abs(finite_before));
        stats.max_abs_error = @max(stats.max_abs_error, @abs(finite_before - after));
    }

    return stats;
}

pub fn mergeDirectTokenWeightedOuterGradient(
    out: []f32,
    global_before: []const f32,
    learners: []const CounterWeightedFragment,
) !MergeStats {
    var weighted_buf: [64]WeightedFragment = undefined;
    if (learners.len > weighted_buf.len) return error.TooManyLearnersForStackBuffer;

    for (learners, 0..) |learner, i| {
        weighted_buf[i] = learner.weighted();
    }

    return mergeDirectWeightedOuterGradient(out, global_before, weighted_buf[0..learners.len]);
}

pub fn mergeDirectWeightedOuterGradient(
    out: []f32,
    global_before: []const f32,
    learners: []const WeightedFragment,
) !MergeStats {
    try validateOutputLength(out, global_before);
    @memset(out, 0.0);

    var stats = MergeStats{};

    for (learners) |learner| {
        if (!isUsableWeight(learner.weight)) continue;
        if (learner.values.len != global_before.len) return error.FragmentLengthMismatch;

        stats.participant_count += 1;
        stats.total_weight += learner.weight;

        for (out, 0..) |*acc, i| {
            const delta = finiteDelta(global_before[i], learner.values[i]) orelse {
                stats.sanitized_values += 1;
                continue;
            };
            acc.* += learner.weight * delta;
        }
    }

    if (!isUsableWeight(stats.total_weight)) {
        @memset(out, 0.0);
        return stats;
    }

    for (out) |*value| {
        value.* /= stats.total_weight;
        if (!std.math.isFinite(value.*)) {
            value.* = 0.0;
            stats.sanitized_values += 1;
        }
    }

    return stats;
}

pub fn mergeRdaWeightedOuterGradient(
    out: []f32,
    global_before: []const f32,
    learners: []const WeightedFragment,
) !MergeStats {
    try validateOutputLength(out, global_before);
    @memset(out, 0.0);

    var stats = MergeStats{};
    var weighted_norm_sum: f64 = 0.0;

    for (learners) |learner| {
        if (!isUsableWeight(learner.weight)) continue;
        if (learner.values.len != global_before.len) return error.FragmentLengthMismatch;

        stats.participant_count += 1;
        stats.total_weight += learner.weight;

        var squared_norm: f64 = 0.0;
        for (global_before, 0..) |global_value, i| {
            const delta = finiteDelta(global_value, learner.values[i]) orelse {
                stats.sanitized_values += 1;
                continue;
            };
            const delta64: f64 = @floatCast(delta);
            squared_norm += delta64 * delta64;
        }

        const norm = @sqrt(squared_norm);
        weighted_norm_sum += @as(f64, @floatCast(learner.weight)) * norm;
        if (norm == 0.0) continue;

        stats.direction_weight += learner.weight;
        const direction_scale = learner.weight / @as(f32, @floatCast(norm));
        for (out, 0..) |*direction_acc, i| {
            const delta = finiteDelta(global_before[i], learner.values[i]) orelse 0.0;
            direction_acc.* += direction_scale * delta;
        }
    }

    if (!isUsableWeight(stats.total_weight) or !isUsableWeight(stats.direction_weight)) {
        @memset(out, 0.0);
        return stats;
    }

    var direction_squared_norm: f64 = 0.0;
    for (out) |value| {
        const value64: f64 = @floatCast(value);
        direction_squared_norm += value64 * value64;
    }

    const direction_norm = @sqrt(direction_squared_norm);
    if (direction_norm == 0.0 or !std.math.isFinite(direction_norm)) {
        @memset(out, 0.0);
        return stats;
    }

    const mean_norm = weighted_norm_sum / @as(f64, @floatCast(stats.total_weight));
    const scale = @as(f32, @floatCast(mean_norm / direction_norm));

    for (out) |*value| {
        value.* *= scale;
        if (!std.math.isFinite(value.*)) {
            value.* = 0.0;
            stats.sanitized_values += 1;
        }
    }

    return stats;
}

fn validateOutputLength(out: []const f32, global_before: []const f32) !void {
    if (out.len != global_before.len) return error.OutputLengthMismatch;
}

fn quantizeInt4(value: f32, scale: f32) i8 {
    if (scale == 0.0 or !std.math.isFinite(scale)) return 0;

    const normalized = value / scale;
    const clamped = std.math.clamp(normalized, -7.0, 7.0);
    return @intCast(@as(i32, @intFromFloat(@round(clamped))));
}

fn isUsableWeight(weight: f32) bool {
    return weight > 0.0 and std.math.isFinite(weight);
}

fn finiteDelta(global_before: f32, learner_after: f32) ?f32 {
    const delta = global_before - learner_after;
    if (!std.math.isFinite(delta)) return null;
    return delta;
}

test "token step weight is quantity times quality with step guard" {
    try std.testing.expectEqual(@as(f32, 50.0), tokenStepWeight(10, 2));
    try std.testing.expectEqual(@as(f32, 100.0), tokenStepWeight(10, 0));
    try std.testing.expectEqual(@as(f32, 0.0), tokenStepWeight(0, 5));
}

test "direct weighted outer gradient matches simple average for equal weights" {
    const global = [_]f32{ 10.0, 20.0, 30.0 };
    const learner_a = [_]f32{ 9.0, 18.0, 33.0 };
    const learner_b = [_]f32{ 7.0, 24.0, 27.0 };
    const learners = [_]WeightedFragment{
        .{ .values = &learner_a, .weight = 1.0 },
        .{ .values = &learner_b, .weight = 1.0 },
    };
    var out: [3]f32 = undefined;

    const stats = try mergeDirectWeightedOuterGradient(&out, &global, &learners);

    try std.testing.expectEqual(@as(usize, 2), stats.participant_count);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), stats.total_weight, 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[1], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 0.000001);
}

test "direct weighted outer gradient ignores zero weight learners" {
    const global = [_]f32{ 10.0, 20.0 };
    const learner_a = [_]f32{ 5.0, 10.0 };
    const learner_b = [_]f32{ -100.0, -100.0 };
    const learners = [_]WeightedFragment{
        .{ .values = &learner_a, .weight = 2.0 },
        .{ .values = &learner_b, .weight = 0.0 },
    };
    var out: [2]f32 = undefined;

    const stats = try mergeDirectWeightedOuterGradient(&out, &global, &learners);

    try std.testing.expectEqual(@as(usize, 1), stats.participant_count);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out[1], 0.000001);
}

test "direct token weighted merge uses paper weighting formula" {
    const global = [_]f32{5.0};
    const learner_a = [_]f32{3.0}; // delta 2, weight 100
    const learner_b = [_]f32{1.0}; // delta 4, weight 50
    const learners = [_]CounterWeightedFragment{
        .{ .values = &learner_a, .tokens_since_update = 10, .steps_since_update = 1 },
        .{ .values = &learner_b, .tokens_since_update = 10, .steps_since_update = 2 },
    };
    var out: [1]f32 = undefined;

    _ = try mergeDirectTokenWeightedOuterGradient(&out, &global, &learners);

    try std.testing.expectApproxEqAbs(@as(f32, 2.6666667), out[0], 0.000001);
}

test "int4 compression round trip bounds error by half a quantization bucket" {
    const allocator = std.testing.allocator;
    const values = [_]f32{ -1.0, -0.5, 0.0, 0.5, 1.0 };
    var compressed = try compressInt4(allocator, &values);
    defer compressed.deinit(allocator);

    var out: [values.len]f32 = undefined;
    try decompressInt4(&out, compressed);

    const allowed_error = compressed.scale / 2.0 + 0.000001;
    for (values, out) |expected, actual| {
        try std.testing.expect(@abs(expected - actual) <= allowed_error);
    }
    try std.testing.expectEqual(@as(usize, 3), compressed.bytes.len);
}

test "int4 in-place compression reports stats and sanitizes non-finite values" {
    const allocator = std.testing.allocator;
    var values = [_]f32{ 0.0, std.math.inf(f32), -2.0, 2.0 };

    const stats = try compressDecompressInt4InPlace(allocator, &values);

    try std.testing.expectEqual(@as(usize, 4), stats.value_count);
    try std.testing.expectEqual(@as(usize, 2), stats.byte_count);
    try std.testing.expectEqual(@as(usize, 1), stats.sanitized_values);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), values[1], 0.000001);
    try std.testing.expect(stats.max_abs_error <= stats.scale / 2.0 + 0.000001);
}

test "merge guards non-finite direct values by zeroing their contribution" {
    const global = [_]f32{ 1.0, std.math.inf(f32) };
    const learner = [_]f32{ 0.0, 1.0 };
    const learners = [_]WeightedFragment{.{ .values = &learner, .weight = 1.0 }};
    var out: [2]f32 = undefined;

    const stats = try mergeDirectWeightedOuterGradient(&out, &global, &learners);

    try std.testing.expectEqual(@as(usize, 1), stats.sanitized_values);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 0.000001);
}

test "RDA equal vectors returns the same vector" {
    const global = [_]f32{ 2.0, 3.0 };
    const learner_a = [_]f32{ 1.0, 1.0 };
    const learner_b = [_]f32{ 1.0, 1.0 };
    const learners = [_]WeightedFragment{
        .{ .values = &learner_a, .weight = 1.0 },
        .{ .values = &learner_b, .weight = 1.0 },
    };
    var out: [2]f32 = undefined;

    _ = try mergeRdaWeightedOuterGradient(&out, &global, &learners);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 0.000001);
}

test "RDA orthogonal equal-norm vectors preserves average norm" {
    const global = [_]f32{ 0.0, 0.0 };
    const learner_a = [_]f32{ -1.0, 0.0 }; // delta [1, 0]
    const learner_b = [_]f32{ 0.0, -1.0 }; // delta [0, 1]
    const learners = [_]WeightedFragment{
        .{ .values = &learner_a, .weight = 1.0 },
        .{ .values = &learner_b, .weight = 1.0 },
    };
    var out: [2]f32 = undefined;

    _ = try mergeRdaWeightedOuterGradient(&out, &global, &learners);

    try std.testing.expectApproxEqAbs(@as(f32, 0.70710677), out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.70710677), out[1], 0.000001);
}

test "RDA weighted inputs bias direction and norm" {
    const global = [_]f32{ 0.0, 0.0 };
    const learner_a = [_]f32{ -2.0, 0.0 }; // delta [2, 0]
    const learner_b = [_]f32{ 0.0, -2.0 }; // delta [0, 2]
    const learners = [_]WeightedFragment{
        .{ .values = &learner_a, .weight = 3.0 },
        .{ .values = &learner_b, .weight = 1.0 },
    };
    var out: [2]f32 = undefined;

    _ = try mergeRdaWeightedOuterGradient(&out, &global, &learners);

    try std.testing.expectApproxEqAbs(@as(f32, 1.8973666), out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6324555), out[1], 0.000001);
}

test "RDA skips zero vectors for direction and returns zero for all-zero inputs" {
    const global = [_]f32{ 1.0, 1.0 };
    const learner_zero_a = [_]f32{ 1.0, 1.0 };
    const learner_zero_b = [_]f32{ 1.0, 1.0 };
    const zero_learners = [_]WeightedFragment{
        .{ .values = &learner_zero_a, .weight = 1.0 },
        .{ .values = &learner_zero_b, .weight = 2.0 },
    };
    var zero_out: [2]f32 = undefined;

    const zero_stats = try mergeRdaWeightedOuterGradient(&zero_out, &global, &zero_learners);

    try std.testing.expectEqual(@as(usize, 2), zero_stats.participant_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), zero_out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), zero_out[1], 0.000001);

    const learner_nonzero = [_]f32{ 0.0, 1.0 }; // delta [1, 0]
    const mixed = [_]WeightedFragment{
        .{ .values = &learner_zero_a, .weight = 1.0 },
        .{ .values = &learner_nonzero, .weight = 1.0 },
    };
    var mixed_out: [2]f32 = undefined;

    _ = try mergeRdaWeightedOuterGradient(&mixed_out, &global, &mixed);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mixed_out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mixed_out[1], 0.000001);
}
