const std = @import("std");
const pcp = @import("pcp");
const IreeBackend = pcp.backends.iree.IreeBackend;
const backend_selection = pcp.backend_selection;
const tensor = pcp.tensor;
const model_introspection = @import("../mlir/model_introspection.zig");

const Allocator = std.mem.Allocator;

const Config = struct {
    backend: backend_selection.Backend = .cuda,
    vmfb_path: []const u8,
    meta_path: []const u8,
    weights_path: []const u8,
    logits_out_path: ?[]const u8 = null,
    token: i64 = 9707,
    position: i64 = 0,
};

pub fn main() !void {
    var total_timer = try std.time.Timer.start();
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = try parseArgs(allocator);
    defer {
        allocator.free(config.vmfb_path);
        allocator.free(config.meta_path);
        allocator.free(config.weights_path);
        if (config.logits_out_path) |path| allocator.free(path);
    }

    var metadata = try model_introspection.ModelMetadata.loadFromFile(allocator, config.meta_path);
    defer metadata.deinit();

    const param_dtypes = metadata.parameter_dtypes orelse return error.MissingParameterDTypes;
    if (param_dtypes.len != metadata.parameter_shapes.len) return error.ParameterDTypeCountMismatch;

    const vmfb = try std.fs.cwd().readFileAlloc(allocator, config.vmfb_path, 128 * 1024 * 1024);
    defer allocator.free(vmfb);

    var backend = try IreeBackend.init(allocator, config.backend, 0);
    defer backend.deinit();

    std.debug.print("Qwen3.6-27B smoke: loading session backend={s}\n", .{config.backend.toString()});
    var stage_timer = try std.time.Timer.start();
    try backend.loadSession(vmfb);
    const session_load_ns = stage_timer.read();

    var inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(allocator);
    defer {
        for (inputs.items) |buf| buf.release();
        inputs.deinit();
    }

    stage_timer.reset();
    const uploaded_bytes = try appendWeights(allocator, backend, config.weights_path, metadata.parameter_shapes, param_dtypes, &inputs);
    const upload_ns = stage_timer.read();
    try appendScalar(allocator, backend, config.token, metadata.data_input_shapes[0], metadata.data_input_dtypes[0], &inputs);
    try appendScalar(allocator, backend, config.position, metadata.data_input_shapes[1], metadata.data_input_dtypes[1], &inputs);

    std.debug.print("Qwen3.6-27B smoke: invoking main with {} inputs\n", .{inputs.items.len});
    stage_timer.reset();
    const outputs = try backend.executeWithDeviceBuffers(vmfb, "main", inputs.items);
    const execute_ns = stage_timer.read();
    defer allocator.free(outputs);
    if (outputs.len != 1) return error.UnexpectedOutputCount;
    defer outputs[0].release();

    const logits_bytes = try backend.readToHost(outputs[0]);
    defer allocator.free(logits_bytes);
    if (logits_bytes.len == 0 or logits_bytes.len % @sizeOf(f32) != 0) return error.InvalidLogitsByteSize;
    const logits: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_bytes));
    if (logits.len == 0) return error.EmptyLogits;
    const top = try topLogit(logits);
    if (config.logits_out_path) |path| {
        try std.fs.cwd().writeFile(.{ .sub_path = path, .data = logits_bytes });
        std.debug.print("Qwen3.6-27B smoke: wrote logits {s}\n", .{path});
    }
    std.debug.print(
        "Qwen3.6-27B smoke: logits={} top_index={} top_value={d:.6}\n",
        .{ logits.len, top.index, top.value },
    );
    std.debug.print(
        "Qwen3.6-27B smoke: timings session_load_ms={d:.3} upload_ms={d:.3} execute_ms={d:.3} total_ms={d:.3} uploaded_bytes={}\n",
        .{ elapsedMs(session_load_ns), elapsedMs(upload_ns), elapsedMs(execute_ns), elapsedMs(total_timer.read()), uploaded_bytes },
    );
}

fn parseArgs(allocator: Allocator) !Config {
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next();

    var config = Config{
        .vmfb_path = try allocator.dupe(u8, "models/qwen36_27b_one_token_nocache_linalg_cuda.vmfb"),
        .meta_path = try allocator.dupe(u8, "models/qwen36_27b_one_token_nocache_linalg.mlir.meta.json"),
        .weights_path = try allocator.dupe(u8, "checkpoints/initial_weights/qwen36_27b_bf16_streamed.bin"),
    };

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--backend")) {
            config.backend = parseBackend(args.next() orelse return error.MissingBackend);
        } else if (std.mem.eql(u8, arg, "--vmfb")) {
            allocator.free(config.vmfb_path);
            config.vmfb_path = try allocator.dupe(u8, args.next() orelse return error.MissingVmfbPath);
        } else if (std.mem.eql(u8, arg, "--meta")) {
            allocator.free(config.meta_path);
            config.meta_path = try allocator.dupe(u8, args.next() orelse return error.MissingMetaPath);
        } else if (std.mem.eql(u8, arg, "--weights")) {
            allocator.free(config.weights_path);
            config.weights_path = try allocator.dupe(u8, args.next() orelse return error.MissingWeightsPath);
        } else if (std.mem.eql(u8, arg, "--logits-out")) {
            if (config.logits_out_path) |path| allocator.free(path);
            config.logits_out_path = try allocator.dupe(u8, args.next() orelse return error.MissingLogitsOutPath);
        } else if (std.mem.eql(u8, arg, "--token")) {
            config.token = try std.fmt.parseInt(i64, args.next() orelse return error.MissingToken, 10);
        } else if (std.mem.eql(u8, arg, "--position")) {
            config.position = try std.fmt.parseInt(i64, args.next() orelse return error.MissingPosition, 10);
        } else {
            return error.UnknownArgument;
        }
    }
    return config;
}

fn parseBackend(value: []const u8) backend_selection.Backend {
    if (std.mem.eql(u8, value, "cuda")) return .cuda;
    if (std.mem.eql(u8, value, "cpu")) return .cpu;
    if (std.mem.eql(u8, value, "rocm")) return .rocm;
    if (std.mem.eql(u8, value, "vulkan")) return .vulkan;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    return .cuda;
}

fn appendWeights(
    allocator: Allocator,
    backend: *IreeBackend,
    weights_path: []const u8,
    shapes: []const []const i64,
    dtypes: []const tensor.DType,
    inputs: *std.ArrayList(IreeBackend.DeviceBuffer),
) !usize {
    const file = try std.fs.cwd().openFile(weights_path, .{ .mode = .read_only });
    defer file.close();

    var buffered = std.io.bufferedReader(file.reader());
    const reader = buffered.reader();
    var uploaded_bytes: usize = 0;

    for (shapes, dtypes, 0..) |shape, dtype, index| {
        const byte_count = try tensorByteSize(shape, dtype);
        const bytes = try allocator.alloc(u8, byte_count);
        defer allocator.free(bytes);
        try reader.readNoEof(bytes);
        const dev_buf = try backend.moveToDevice(bytes, shape, dtype);
        try inputs.append(dev_buf);
        uploaded_bytes += byte_count;
        if ((index + 1) % 64 == 0 or index + 1 == shapes.len) {
            std.debug.print(
                "Qwen3.6-27B smoke: uploaded {}/{} params ({d:.2} GiB)\n",
                .{ index + 1, shapes.len, @as(f64, @floatFromInt(uploaded_bytes)) / 1024.0 / 1024.0 / 1024.0 },
            );
        }
    }
    return uploaded_bytes;
}

fn appendScalar(
    allocator: Allocator,
    backend: *IreeBackend,
    value: i64,
    shape: []const i64,
    dtype: tensor.DType,
    inputs: *std.ArrayList(IreeBackend.DeviceBuffer),
) !void {
    switch (dtype) {
        .i64 => {
            const data = try allocator.alloc(i64, elementCount(shape));
            defer allocator.free(data);
            @memset(data, value);
            try inputs.append(try backend.moveToDevice(std.mem.sliceAsBytes(data), shape, dtype));
        },
        .i32 => {
            const data = try allocator.alloc(i32, elementCount(shape));
            defer allocator.free(data);
            @memset(data, @intCast(value));
            try inputs.append(try backend.moveToDevice(std.mem.sliceAsBytes(data), shape, dtype));
        },
        else => return error.UnsupportedScalarDType,
    }
}

fn tensorByteSize(shape: []const i64, dtype: tensor.DType) !usize {
    return try std.math.mul(usize, elementCount(shape), dtype.sizeInBytes());
}

fn elementCount(shape: []const i64) usize {
    var total: usize = 1;
    for (shape) |dim| total *= @intCast(dim);
    return total;
}

fn topLogit(logits: []const f32) !struct { index: usize, value: f32 } {
    var best_index: usize = 0;
    var best_value: f32 = logits[0];
    if (!std.math.isFinite(best_value)) return error.NonFiniteLogit;
    for (logits[1..], 1..) |value, index| {
        if (!std.math.isFinite(value)) return error.NonFiniteLogit;
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return .{ .index = best_index, .value = best_value };
}

fn elapsedMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}
