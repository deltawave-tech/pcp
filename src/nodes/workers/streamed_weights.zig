const std = @import("std");
const tensor = @import("../../core/tensor.zig");
const generation_cache = @import("generation_cache.zig");

pub const ValidationCode = enum {
    missing_dtype_metadata,
    parameter_count_mismatch,
    invalid_manifest_format,
    missing_manifest_entry,
    invalid_parameter_order,
    invalid_offset,
    invalid_shape,
    invalid_dtype,
    short_read,
    trailing_bytes,
};

pub const ValidationFailure = struct {
    code: ValidationCode,
    parameter_index: ?usize = null,
    expected_bytes: usize = 0,
    actual_bytes: usize = 0,
    dtype: ?tensor.DType = null,
};

const Manifest = struct {
    format: []const u8,
    parameters: []ManifestParameter,
};

const ManifestParameter = struct {
    index: usize,
    source_file: []const u8,
    offset: usize = 0,
    nbytes: ?usize = null,
    length: ?usize = null,
    shape: []const i64,
    dtype: []const u8,

    fn byteCount(self: ManifestParameter) !usize {
        if (self.nbytes) |value| return value;
        if (self.length) |value| return value;
        return error.MissingManifestLength;
    }
};

fn resetFailure(failure: ?*ValidationFailure) void {
    if (failure) |out| out.* = .{ .code = .short_read };
}

fn setFailure(
    failure: ?*ValidationFailure,
    code: ValidationCode,
    parameter_index: ?usize,
    expected_bytes: usize,
    actual_bytes: usize,
    dtype: ?tensor.DType,
) void {
    if (failure) |out| {
        out.* = .{
            .code = code,
            .parameter_index = parameter_index,
            .expected_bytes = expected_bytes,
            .actual_bytes = actual_bytes,
            .dtype = dtype,
        };
    }
}

pub fn validateStreamedWeightFile(
    weights_path: []const u8,
    weights_format: []const u8,
    parameter_shapes: []const []const i64,
    parameter_dtypes: ?[]const tensor.DType,
    failure: ?*ValidationFailure,
) !void {
    resetFailure(failure);

    if (isManifestFormat(weights_format)) {
        return try validateManifestWeightFile(weights_path, weights_format, parameter_shapes, parameter_dtypes, failure);
    }

    const dtypes = parameter_dtypes orelse {
        setFailure(failure, .missing_dtype_metadata, if (parameter_shapes.len > 0) 0 else null, parameter_shapes.len, 0, null);
        return error.MissingDTypeMetadata;
    };

    if (dtypes.len != parameter_shapes.len) {
        const index = @min(dtypes.len, parameter_shapes.len);
        setFailure(failure, .parameter_count_mismatch, index, parameter_shapes.len, dtypes.len, null);
        return error.ParameterCountMismatch;
    }

    const file = try std.fs.cwd().openFile(weights_path, .{ .mode = .read_only });
    defer file.close();
    const stat = try file.stat();
    const actual_bytes: usize = @intCast(stat.size);

    var expected_total: usize = 0;
    const require_bf16 = std.mem.eql(u8, weights_format, "bf16_streamed");
    for (parameter_shapes, dtypes, 0..) |shape, dtype, index| {
        if (require_bf16 and dtype != .bf16) {
            setFailure(failure, .invalid_dtype, index, @intFromEnum(tensor.DType.bf16), @intFromEnum(dtype), dtype);
            return error.InvalidStreamedWeightDType;
        }

        const byte_count = generation_cache.tensorByteSize(shape, dtype) catch |err| {
            setFailure(failure, .invalid_shape, index, 0, 0, dtype);
            return err;
        };
        const next_expected = std.math.add(usize, expected_total, byte_count) catch {
            setFailure(failure, .invalid_shape, index, 0, 0, dtype);
            return error.TensorShapeTooLarge;
        };
        if (actual_bytes < next_expected) {
            setFailure(failure, .short_read, index, next_expected, actual_bytes, dtype);
            return error.StreamedWeightShortRead;
        }
        expected_total = next_expected;
    }

    if (actual_bytes != expected_total) {
        setFailure(failure, .trailing_bytes, parameter_shapes.len, expected_total, actual_bytes, null);
        return error.StreamedWeightTrailingBytes;
    }
}

pub fn isManifestFormat(weights_format: []const u8) bool {
    return std.mem.eql(u8, weights_format, "manifest_streamed") or
        std.mem.eql(u8, weights_format, "bf16_manifest_streamed");
}

pub fn manifestRequiresBf16(weights_format: []const u8) bool {
    return std.mem.eql(u8, weights_format, "bf16_manifest_streamed");
}

fn parseDTypeName(name: []const u8) ?tensor.DType {
    if (std.mem.eql(u8, name, "f16")) return .f16;
    if (std.mem.eql(u8, name, "bf16")) return .bf16;
    if (std.mem.eql(u8, name, "f32")) return .f32;
    if (std.mem.eql(u8, name, "f64")) return .f64;
    if (std.mem.eql(u8, name, "i32")) return .i32;
    if (std.mem.eql(u8, name, "i64")) return .i64;
    if (std.mem.eql(u8, name, "bool")) return .bool;
    return null;
}

fn validateManifestWeightFile(
    manifest_path: []const u8,
    weights_format: []const u8,
    parameter_shapes: []const []const i64,
    parameter_dtypes: ?[]const tensor.DType,
    failure: ?*ValidationFailure,
) !void {
    const dtypes = parameter_dtypes orelse {
        setFailure(failure, .missing_dtype_metadata, if (parameter_shapes.len > 0) 0 else null, parameter_shapes.len, 0, null);
        return error.MissingDTypeMetadata;
    };

    if (dtypes.len != parameter_shapes.len) {
        const index = @min(dtypes.len, parameter_shapes.len);
        setFailure(failure, .parameter_count_mismatch, index, parameter_shapes.len, dtypes.len, null);
        return error.ParameterCountMismatch;
    }

    const allocator = std.heap.page_allocator;
    const manifest_bytes = try std.fs.cwd().readFileAlloc(allocator, manifest_path, 64 * 1024 * 1024);
    defer allocator.free(manifest_bytes);
    var parsed = try std.json.parseFromSlice(Manifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    const manifest = parsed.value;
    if (!std.mem.eql(u8, manifest.format, weights_format)) {
        setFailure(failure, .invalid_manifest_format, null, 0, 0, null);
        return error.InvalidManifestFormat;
    }

    if (manifest.parameters.len != parameter_shapes.len) {
        const index = @min(manifest.parameters.len, parameter_shapes.len);
        setFailure(failure, .parameter_count_mismatch, index, parameter_shapes.len, manifest.parameters.len, null);
        return error.ParameterCountMismatch;
    }

    for (manifest.parameters, parameter_shapes, dtypes, 0..) |entry, expected_shape, expected_dtype, expected_index| {
        try validateManifestEntry(manifest_path, weights_format, entry, expected_shape, expected_dtype, expected_index, failure);
    }
}

fn validateManifestEntry(
    manifest_path: []const u8,
    weights_format: []const u8,
    entry: ManifestParameter,
    expected_shape: []const i64,
    expected_dtype: tensor.DType,
    expected_index: usize,
    failure: ?*ValidationFailure,
) !void {
    if (entry.index != expected_index) {
        setFailure(failure, .invalid_parameter_order, expected_index, expected_index, entry.index, null);
        return error.InvalidManifestParameterOrder;
    }

    const manifest_dtype = parseDTypeName(entry.dtype) orelse {
        setFailure(failure, .invalid_dtype, expected_index, 0, 0, null);
        return error.InvalidManifestDType;
    };
    if (manifest_dtype != expected_dtype or (manifestRequiresBf16(weights_format) and manifest_dtype != .bf16)) {
        setFailure(failure, .invalid_dtype, expected_index, @intFromEnum(expected_dtype), @intFromEnum(manifest_dtype), manifest_dtype);
        return error.InvalidStreamedWeightDType;
    }

    if (!std.mem.eql(i64, entry.shape, expected_shape)) {
        setFailure(failure, .invalid_shape, expected_index, 0, 0, manifest_dtype);
        return error.InvalidManifestShape;
    }

    const expected_bytes = generation_cache.tensorByteSize(expected_shape, expected_dtype) catch |err| {
        setFailure(failure, .invalid_shape, expected_index, 0, 0, expected_dtype);
        return err;
    };
    const entry_bytes = entry.byteCount() catch |err| {
        setFailure(failure, .short_read, expected_index, expected_bytes, 0, expected_dtype);
        return err;
    };
    if (entry_bytes != expected_bytes) {
        setFailure(failure, .short_read, expected_index, expected_bytes, entry_bytes, expected_dtype);
        return error.StreamedWeightShortRead;
    }

    const allocator = std.heap.page_allocator;
    const source_path = try resolveManifestSourcePath(allocator, manifest_path, entry.source_file);
    defer allocator.free(source_path);
    const file = std.fs.cwd().openFile(source_path, .{ .mode = .read_only }) catch |err| {
        setFailure(failure, .missing_manifest_entry, expected_index, entry_bytes, 0, expected_dtype);
        return err;
    };
    defer file.close();
    const stat = try file.stat();
    const file_size: usize = @intCast(stat.size);
    if (entry.offset > file_size) {
        setFailure(failure, .invalid_offset, expected_index, entry_bytes, file_size, expected_dtype);
        return error.InvalidManifestOffset;
    }
    const end = std.math.add(usize, entry.offset, entry_bytes) catch {
        setFailure(failure, .invalid_offset, expected_index, entry_bytes, file_size, expected_dtype);
        return error.InvalidManifestOffset;
    };
    if (end > file_size) {
        setFailure(failure, .short_read, expected_index, end, file_size, expected_dtype);
        return error.StreamedWeightShortRead;
    }
}

fn resolveManifestSourcePath(allocator: std.mem.Allocator, manifest_path: []const u8, source_file: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(source_file)) return try allocator.dupe(u8, source_file);
    const manifest_dir = std.fs.path.dirname(manifest_path) orelse ".";
    return try std.fs.path.join(allocator, &.{ manifest_dir, source_file });
}

pub fn readManifestParameterAlloc(
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
    parameter_index: usize,
) ![]u8 {
    const manifest_bytes = try std.fs.cwd().readFileAlloc(allocator, manifest_path, 64 * 1024 * 1024);
    defer allocator.free(manifest_bytes);
    var parsed = try std.json.parseFromSlice(Manifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    if (parameter_index >= parsed.value.parameters.len) return error.MissingManifestEntry;
    const entry = parsed.value.parameters[parameter_index];
    if (entry.index != parameter_index) return error.InvalidManifestParameterOrder;
    const byte_count = try entry.byteCount();

    const source_path = try resolveManifestSourcePath(allocator, manifest_path, entry.source_file);
    defer allocator.free(source_path);
    const file = try std.fs.cwd().openFile(source_path, .{ .mode = .read_only });
    defer file.close();
    try file.seekTo(entry.offset);
    const bytes = try allocator.alloc(u8, byte_count);
    errdefer allocator.free(bytes);
    try file.reader().readNoEof(bytes);
    return bytes;
}

fn writeTempBytes(path: []const u8, bytes: []const u8) !void {
    try std.fs.cwd().makePath("zig-cache/tmp");
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(bytes);
}

fn expectValidationFailure(
    path: []const u8,
    weights_format: []const u8,
    shapes: []const []const i64,
    dtypes: ?[]const tensor.DType,
    expected_error: anyerror,
    expected_code: ValidationCode,
    expected_index: ?usize,
) !void {
    var failure: ValidationFailure = undefined;
    const result = validateStreamedWeightFile(path, weights_format, shapes, dtypes, &failure);
    try std.testing.expectError(expected_error, result);
    try std.testing.expectEqual(expected_code, failure.code);
    try std.testing.expectEqual(expected_index, failure.parameter_index);
}

test "streamed weight validation accepts exact BF16 stream" {
    const path = "zig-cache/tmp/qwen36_streamed_weights_exact.bin";
    const bytes = [_]u8{ 0, 1, 2, 3, 4, 5 };
    try writeTempBytes(path, &bytes);
    defer std.fs.cwd().deleteFile(path) catch {};

    const shape0 = [_]i64{2};
    const shape1 = [_]i64{1};
    const shapes = [_][]const i64{ &shape0, &shape1 };
    const dtypes = [_]tensor.DType{ .bf16, .bf16 };
    var failure: ValidationFailure = undefined;

    try validateStreamedWeightFile(path, "bf16_streamed", &shapes, &dtypes, &failure);
}

test "streamed weight validation reports short read parameter index" {
    const path = "zig-cache/tmp/qwen36_streamed_weights_short.bin";
    const bytes = [_]u8{ 0, 1, 2, 3, 4 };
    try writeTempBytes(path, &bytes);
    defer std.fs.cwd().deleteFile(path) catch {};

    const shape0 = [_]i64{2};
    const shape1 = [_]i64{1};
    const shapes = [_][]const i64{ &shape0, &shape1 };
    const dtypes = [_]tensor.DType{ .bf16, .bf16 };

    try expectValidationFailure(path, "bf16_streamed", &shapes, &dtypes, error.StreamedWeightShortRead, .short_read, 1);
}

test "streamed weight validation reports trailing bytes after final parameter" {
    const path = "zig-cache/tmp/qwen36_streamed_weights_trailing.bin";
    const bytes = [_]u8{ 0, 1, 2, 3, 4, 5, 6 };
    try writeTempBytes(path, &bytes);
    defer std.fs.cwd().deleteFile(path) catch {};

    const shape0 = [_]i64{2};
    const shape1 = [_]i64{1};
    const shapes = [_][]const i64{ &shape0, &shape1 };
    const dtypes = [_]tensor.DType{ .bf16, .bf16 };

    try expectValidationFailure(path, "bf16_streamed", &shapes, &dtypes, error.StreamedWeightTrailingBytes, .trailing_bytes, 2);
}

test "streamed weight validation rejects wrong dtype for BF16 format" {
    const path = "zig-cache/tmp/qwen36_streamed_weights_wrong_dtype.bin";
    const bytes = [_]u8{ 0, 1, 2, 3, 4, 5 };
    try writeTempBytes(path, &bytes);
    defer std.fs.cwd().deleteFile(path) catch {};

    const shape0 = [_]i64{2};
    const shape1 = [_]i64{1};
    const shapes = [_][]const i64{ &shape0, &shape1 };
    const dtypes = [_]tensor.DType{ .bf16, .f32 };

    try expectValidationFailure(path, "bf16_streamed", &shapes, &dtypes, error.InvalidStreamedWeightDType, .invalid_dtype, 1);
}

test "streamed weight validation reports invalid shape parameter index" {
    const path = "zig-cache/tmp/qwen36_streamed_weights_invalid_shape.bin";
    const bytes = [_]u8{ 0, 1, 2, 3 };
    try writeTempBytes(path, &bytes);
    defer std.fs.cwd().deleteFile(path) catch {};

    const shape0 = [_]i64{2};
    const shape1 = [_]i64{-1};
    const shapes = [_][]const i64{ &shape0, &shape1 };
    const dtypes = [_]tensor.DType{ .bf16, .bf16 };

    try expectValidationFailure(path, "bf16_streamed", &shapes, &dtypes, error.InvalidTensorShape, .invalid_shape, 1);
}

test "streamed weight validation reports dtype count mismatch" {
    const path = "zig-cache/tmp/qwen36_streamed_weights_dtype_count.bin";
    const bytes = [_]u8{ 0, 1, 2, 3 };
    try writeTempBytes(path, &bytes);
    defer std.fs.cwd().deleteFile(path) catch {};

    const shape0 = [_]i64{2};
    const shape1 = [_]i64{1};
    const shapes = [_][]const i64{ &shape0, &shape1 };
    const dtypes = [_]tensor.DType{.bf16};

    try expectValidationFailure(path, "bf16_streamed", &shapes, &dtypes, error.ParameterCountMismatch, .parameter_count_mismatch, 1);
}

fn writeManifest(path: []const u8, manifest: []const u8) !void {
    try std.fs.cwd().makePath("zig-cache/tmp");
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(manifest);
}

test "manifest streamed weights read two shards in parameter order" {
    const shard0_path = "zig-cache/tmp/qwen36_manifest_shard0.bin";
    const shard1_path = "zig-cache/tmp/qwen36_manifest_shard1.bin";
    const manifest_path = "zig-cache/tmp/qwen36_manifest_weights.json";
    try writeTempBytes(shard0_path, &[_]u8{ 99, 0, 1, 2, 3, 98 });
    try writeTempBytes(shard1_path, &[_]u8{ 10, 11, 12, 13 });
    try writeManifest(manifest_path,
        \\{
        \\  "format": "bf16_manifest_streamed",
        \\  "parameters": [
        \\    {"index": 0, "source_file": "qwen36_manifest_shard0.bin", "offset": 1, "nbytes": 4, "shape": [2], "dtype": "bf16"},
        \\    {"index": 1, "source_file": "qwen36_manifest_shard1.bin", "offset": 0, "length": 4, "shape": [2], "dtype": "bf16"}
        \\  ]
        \\}
    );
    defer std.fs.cwd().deleteFile(shard0_path) catch {};
    defer std.fs.cwd().deleteFile(shard1_path) catch {};
    defer std.fs.cwd().deleteFile(manifest_path) catch {};

    const shape0 = [_]i64{2};
    const shape1 = [_]i64{2};
    const shapes = [_][]const i64{ &shape0, &shape1 };
    const dtypes = [_]tensor.DType{ .bf16, .bf16 };

    try validateStreamedWeightFile(manifest_path, "bf16_manifest_streamed", &shapes, &dtypes, null);
    const param0 = try readManifestParameterAlloc(std.testing.allocator, manifest_path, 0);
    defer std.testing.allocator.free(param0);
    const param1 = try readManifestParameterAlloc(std.testing.allocator, manifest_path, 1);
    defer std.testing.allocator.free(param1);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3 }, param0);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 10, 11, 12, 13 }, param1);
}

test "manifest streamed weights report missing shard parameter index" {
    const manifest_path = "zig-cache/tmp/qwen36_manifest_missing_shard.json";
    try writeManifest(manifest_path,
        \\{
        \\  "format": "bf16_manifest_streamed",
        \\  "parameters": [
        \\    {"index": 0, "source_file": "qwen36_manifest_missing.bin", "offset": 0, "nbytes": 4, "shape": [2], "dtype": "bf16"}
        \\  ]
        \\}
    );
    defer std.fs.cwd().deleteFile(manifest_path) catch {};

    const shape0 = [_]i64{2};
    const shapes = [_][]const i64{&shape0};
    const dtypes = [_]tensor.DType{.bf16};

    try expectValidationFailure(manifest_path, "bf16_manifest_streamed", &shapes, &dtypes, error.FileNotFound, .missing_manifest_entry, 0);
}

test "manifest streamed weights report invalid offset parameter index" {
    const shard_path = "zig-cache/tmp/qwen36_manifest_offset_shard.bin";
    const manifest_path = "zig-cache/tmp/qwen36_manifest_bad_offset.json";
    try writeTempBytes(shard_path, &[_]u8{ 0, 1, 2, 3 });
    try writeManifest(manifest_path,
        \\{
        \\  "format": "bf16_manifest_streamed",
        \\  "parameters": [
        \\    {"index": 0, "source_file": "qwen36_manifest_offset_shard.bin", "offset": 9, "nbytes": 4, "shape": [2], "dtype": "bf16"}
        \\  ]
        \\}
    );
    defer std.fs.cwd().deleteFile(shard_path) catch {};
    defer std.fs.cwd().deleteFile(manifest_path) catch {};

    const shape0 = [_]i64{2};
    const shapes = [_][]const i64{&shape0};
    const dtypes = [_]tensor.DType{.bf16};

    try expectValidationFailure(manifest_path, "bf16_manifest_streamed", &shapes, &dtypes, error.InvalidManifestOffset, .invalid_offset, 0);
}

test "manifest streamed weights report short read parameter index" {
    const shard_path = "zig-cache/tmp/qwen36_manifest_short_shard.bin";
    const manifest_path = "zig-cache/tmp/qwen36_manifest_short.json";
    try writeTempBytes(shard_path, &[_]u8{ 0, 1, 2, 3, 4 });
    try writeManifest(manifest_path,
        \\{
        \\  "format": "bf16_manifest_streamed",
        \\  "parameters": [
        \\    {"index": 0, "source_file": "qwen36_manifest_short_shard.bin", "offset": 2, "nbytes": 4, "shape": [2], "dtype": "bf16"}
        \\  ]
        \\}
    );
    defer std.fs.cwd().deleteFile(shard_path) catch {};
    defer std.fs.cwd().deleteFile(manifest_path) catch {};

    const shape0 = [_]i64{2};
    const shapes = [_][]const i64{&shape0};
    const dtypes = [_]tensor.DType{.bf16};

    try expectValidationFailure(manifest_path, "bf16_manifest_streamed", &shapes, &dtypes, error.StreamedWeightShortRead, .short_read, 0);
}

test "manifest streamed weights report wrong dtype parameter index" {
    const shard_path = "zig-cache/tmp/qwen36_manifest_dtype_shard.bin";
    const manifest_path = "zig-cache/tmp/qwen36_manifest_wrong_dtype.json";
    try writeTempBytes(shard_path, &[_]u8{ 0, 1, 2, 3 });
    try writeManifest(manifest_path,
        \\{
        \\  "format": "bf16_manifest_streamed",
        \\  "parameters": [
        \\    {"index": 0, "source_file": "qwen36_manifest_dtype_shard.bin", "offset": 0, "nbytes": 4, "shape": [2], "dtype": "f32"}
        \\  ]
        \\}
    );
    defer std.fs.cwd().deleteFile(shard_path) catch {};
    defer std.fs.cwd().deleteFile(manifest_path) catch {};

    const shape0 = [_]i64{2};
    const shapes = [_][]const i64{&shape0};
    const dtypes = [_]tensor.DType{.bf16};

    try expectValidationFailure(manifest_path, "bf16_manifest_streamed", &shapes, &dtypes, error.InvalidStreamedWeightDType, .invalid_dtype, 0);
}

test "manifest streamed weights report wrong shape parameter index" {
    const shard_path = "zig-cache/tmp/qwen36_manifest_shape_shard.bin";
    const manifest_path = "zig-cache/tmp/qwen36_manifest_wrong_shape.json";
    try writeTempBytes(shard_path, &[_]u8{ 0, 1, 2, 3 });
    try writeManifest(manifest_path,
        \\{
        \\  "format": "bf16_manifest_streamed",
        \\  "parameters": [
        \\    {"index": 0, "source_file": "qwen36_manifest_shape_shard.bin", "offset": 0, "nbytes": 4, "shape": [1, 2], "dtype": "bf16"}
        \\  ]
        \\}
    );
    defer std.fs.cwd().deleteFile(shard_path) catch {};
    defer std.fs.cwd().deleteFile(manifest_path) catch {};

    const shape0 = [_]i64{2};
    const shapes = [_][]const i64{&shape0};
    const dtypes = [_]tensor.DType{.bf16};

    try expectValidationFailure(manifest_path, "bf16_manifest_streamed", &shapes, &dtypes, error.InvalidManifestShape, .invalid_shape, 0);
}

test "manifest streamed weights report parameter order mismatch" {
    const shard_path = "zig-cache/tmp/qwen36_manifest_order_shard.bin";
    const manifest_path = "zig-cache/tmp/qwen36_manifest_wrong_order.json";
    try writeTempBytes(shard_path, &[_]u8{ 0, 1, 2, 3 });
    try writeManifest(manifest_path,
        \\{
        \\  "format": "bf16_manifest_streamed",
        \\  "parameters": [
        \\    {"index": 1, "source_file": "qwen36_manifest_order_shard.bin", "offset": 0, "nbytes": 4, "shape": [2], "dtype": "bf16"}
        \\  ]
        \\}
    );
    defer std.fs.cwd().deleteFile(shard_path) catch {};
    defer std.fs.cwd().deleteFile(manifest_path) catch {};

    const shape0 = [_]i64{2};
    const shapes = [_][]const i64{&shape0};
    const dtypes = [_]tensor.DType{.bf16};

    try expectValidationFailure(manifest_path, "bf16_manifest_streamed", &shapes, &dtypes, error.InvalidManifestParameterOrder, .invalid_parameter_order, 0);
}
