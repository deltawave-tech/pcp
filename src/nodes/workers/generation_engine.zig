const std = @import("std");
const math = @import("../../core/math.zig");
const tensor = @import("../../core/tensor.zig");
const IreeBackend = @import("../../backends/iree.zig").IreeBackend;
const message = @import("../../network/message.zig");
const generation_cache = @import("generation_cache.zig");
const streamed_weights = @import("streamed_weights.zig");

const Allocator = std.mem.Allocator;

pub const LoadedInferenceModel = struct {
    model_id: []const u8,
    cached_vmfb: []const u8,
    weights_blob: []const u8,
    weights_path: ?[]const u8 = null,
    weights_format: []const u8 = "flat",
    weight_dtypes: ?[]tensor.DType = null,
    param_shapes: []const []const i64,
    data_shapes: []const []const i64,
    data_dtypes: ?[]tensor.DType,
    max_context_tokens: usize,
};

pub const GenerationSession = struct {
    session_id: []const u8,
    model_id: []const u8,
    kv_cache_host: ?[][]u8,
    last_pos: i64,
};

pub const ActiveGenerationState = struct {
    request_id: message.RequestId,
    task_id: message.TaskId,
    model_id: []const u8,
    session_id: []const u8,
    prompt_tokens: usize,
    completion_tokens: usize,
    cancelled: std.atomic.Value(u8),
};

pub const GenerationResult = struct {
    tokens: []i64,
    finish_reason: []const u8,
    prompt_tokens: usize,
    completion_tokens: usize,
};

pub const GenerationEngine = struct {
    allocator: Allocator,
    device_weights: ?[]IreeBackend.DeviceBuffer,
    device_kv_cache: ?[]IreeBackend.DeviceBuffer,
    device_weights_dirty: bool,
    current_weight_blob_ptr: ?[*]const u8,
    current_weight_blob_len: usize,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .device_weights = null,
            .device_kv_cache = null,
            .device_weights_dirty = false,
            .current_weight_blob_ptr = null,
            .current_weight_blob_len = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.device_weights) |weights| {
            for (weights) |buf| buf.release();
            self.allocator.free(weights);
        }
        if (self.device_kv_cache) |cache| {
            for (cache) |buf| buf.release();
            self.allocator.free(cache);
        }
    }

    pub fn markWeightsDirty(self: *Self) void {
        self.device_weights_dirty = true;
    }

    pub fn ensureDeviceWeights(
        self: *Self,
        backend: *IreeBackend,
        weight_blob: []const u8,
        param_shapes: []const []const i64,
        param_dtypes: ?[]const tensor.DType,
        weights_path: ?[]const u8,
        weights_format: []const u8,
    ) !void {
        if (self.device_weights != null and
            !self.device_weights_dirty and
            ((weights_path != null and self.current_weight_blob_ptr == null) or
            (weights_path == null and self.current_weight_blob_ptr != null and self.current_weight_blob_ptr.? == weight_blob.ptr and self.current_weight_blob_len == weight_blob.len)))
        {
            return;
        }

        if (self.device_weights) |old_weights| {
            for (old_weights) |buf| buf.release();
            self.allocator.free(old_weights);
            self.device_weights = null;
        }

        var dev_weights = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
        errdefer {
            for (dev_weights.items) |buf| buf.release();
            dev_weights.deinit();
        }

        if (weights_path) |path| {
            try self.loadDeviceWeightsFromPath(backend, path, weights_format, param_shapes, param_dtypes, &dev_weights);
            self.current_weight_blob_ptr = null;
            self.current_weight_blob_len = 0;
        } else {
            var offset: usize = 0;
            for (param_shapes, 0..) |shape, index| {
                const dtype = dataInputDType(param_dtypes, index, .f32);
                const size = try generation_cache.tensorByteSize(shape, dtype);

                const slice = weight_blob[offset .. offset + size];
                const dev_buf = try backend.moveToDevice(slice, shape, dtype);
                try dev_weights.append(dev_buf);

                offset += size;
            }
            if (offset != weight_blob.len) return error.WeightBufferSizeMismatch;
            self.current_weight_blob_ptr = weight_blob.ptr;
            self.current_weight_blob_len = weight_blob.len;
        }

        self.device_weights = try dev_weights.toOwnedSlice();
        self.device_weights_dirty = false;
    }

    fn loadDeviceWeightsFromPath(
        self: *Self,
        backend: *IreeBackend,
        weights_path: []const u8,
        weights_format: []const u8,
        param_shapes: []const []const i64,
        param_dtypes: ?[]const tensor.DType,
        dev_weights: *std.ArrayList(IreeBackend.DeviceBuffer),
    ) !void {
        var failure: streamed_weights.ValidationFailure = undefined;
        streamed_weights.validateStreamedWeightFile(weights_path, weights_format, param_shapes, param_dtypes, &failure) catch |err| {
            if (failure.parameter_index) |index| {
                std.log.err("Streamed weight validation failed at parameter {}: {s}", .{ index, @errorName(err) });
            } else {
                std.log.err("Streamed weight validation failed: {s}", .{@errorName(err)});
            }
            return err;
        };

        if (streamed_weights.isManifestFormat(weights_format)) {
            const dtypes = param_dtypes orelse return error.MissingDTypeMetadata;
            for (param_shapes, dtypes, 0..) |shape, dtype, index| {
                const param_bytes = try streamed_weights.readManifestParameterAlloc(
                    self.allocator,
                    weights_path,
                    index,
                );
                errdefer self.allocator.free(param_bytes);
                const dev_buf = try backend.moveToDevice(param_bytes, shape, dtype);
                try dev_weights.append(dev_buf);
                self.allocator.free(param_bytes);
            }
            return;
        }

        const file = try std.fs.cwd().openFile(weights_path, .{ .mode = .read_only });
        defer file.close();

        var buffered = std.io.bufferedReader(file.reader());
        const reader = buffered.reader();

        for (param_shapes, 0..) |shape, index| {
            const dtype = dataInputDType(param_dtypes, index, .f32);
            const byte_count = try generation_cache.tensorByteSize(shape, dtype);
            const param_bytes = try self.allocator.alloc(u8, byte_count);
            defer self.allocator.free(param_bytes);
            try reader.readNoEof(param_bytes);
            const dev_buf = try backend.moveToDevice(param_bytes, shape, dtype);
            try dev_weights.append(dev_buf);
        }
    }

    pub fn initKvCacheHost(
        self: *Self,
        data_shapes: []const []const i64,
        data_dtypes: ?[]const tensor.DType,
    ) ![][]u8 {
        const kv_shapes = data_shapes[2..];
        const kv_buffers = try self.allocator.alloc([]u8, kv_shapes.len);
        for (kv_shapes, 0..) |shape, i| {
            const dtype = generation_cache.dataInputDType(data_dtypes, i + 2);
            const size = try generation_cache.tensorByteSize(shape, dtype);
            kv_buffers[i] = try self.allocator.alloc(u8, size);
            @memset(kv_buffers[i], 0);
        }
        return kv_buffers;
    }

    pub fn generateRollout(
        self: *Self,
        backend: *IreeBackend,
        sampler: *math.Sampler,
        vmfb: []const u8,
        param_shapes: []const []const i64,
        data_shapes: []const []const i64,
        data_dtypes: ?[]const tensor.DType,
        weight_blob: []const u8,
        param_dtypes: ?[]const tensor.DType,
        weights_path: ?[]const u8,
        weights_format: []const u8,
        prompt_tokens: []const i64,
        max_new_tokens: usize,
        eos_token: i64,
        kv_cache: *?[][]u8,
    ) !GenerationResult {
        return try self.generateRolloutStreaming(
            backend,
            sampler,
            vmfb,
            param_shapes,
            data_shapes,
            data_dtypes,
            weight_blob,
            param_dtypes,
            weights_path,
            weights_format,
            prompt_tokens,
            max_new_tokens,
            eos_token,
            0.7,
            kv_cache,
            0,
            false,
            null,
            null,
            null,
        );
    }

    pub const TokenCallback = *const fn (token: i64, ctx: *anyopaque) anyerror!void;

    pub fn generateRolloutStreaming(
        self: *Self,
        backend: *IreeBackend,
        sampler: *math.Sampler,
        vmfb: []const u8,
        param_shapes: []const []const i64,
        data_shapes: []const []const i64,
        data_dtypes: ?[]const tensor.DType,
        weight_blob: []const u8,
        param_dtypes: ?[]const tensor.DType,
        weights_path: ?[]const u8,
        weights_format: []const u8,
        prompt_tokens: []const i64,
        max_new_tokens: usize,
        eos_token: i64,
        temperature: f32,
        kv_cache: *?[][]u8,
        start_pos: i64,
        reuse_cache: bool,
        on_token: ?TokenCallback,
        on_token_ctx: ?*anyopaque,
        cancelled: ?*std.atomic.Value(u8),
    ) !GenerationResult {
        try self.ensureDeviceWeights(backend, weight_blob, param_shapes, param_dtypes, weights_path, weights_format);

        if (!reuse_cache or kv_cache.* == null) {
            if (kv_cache.*) |old_cache| {
                for (old_cache) |buf| self.allocator.free(buf);
                self.allocator.free(old_cache);
            }
            kv_cache.* = try self.initKvCacheHost(data_shapes, data_dtypes);
        }

        var generated_tokens = std.ArrayList(i64).init(self.allocator);
        errdefer generated_tokens.deinit();

        var current_pos: i64 = start_pos;
        const total_len = start_pos + @as(i64, @intCast(prompt_tokens.len + max_new_tokens));
        const kv_shapes = data_shapes[2..];

        while (current_pos < total_len) {
            if (cancelled) |flag| {
                if (flag.load(.acquire) == 1) {
                    return GenerationResult{
                        .tokens = try generated_tokens.toOwnedSlice(),
                        .finish_reason = "cancelled",
                        .prompt_tokens = prompt_tokens.len,
                        .completion_tokens = generated_tokens.items.len,
                    };
                }
            }
            var input_token: i64 = 0;
            if (current_pos < start_pos + @as(i64, @intCast(prompt_tokens.len))) {
                const idx = @as(usize, @intCast(current_pos - start_pos));
                input_token = prompt_tokens[idx];
            } else if (generated_tokens.items.len > 0) {
                input_token = generated_tokens.items[generated_tokens.items.len - 1];
            }

            var inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
            defer inputs.deinit();

            for (self.device_weights.?) |w| {
                w.retain();
                try inputs.append(w);
            }

            const token_dtype = generation_cache.dataInputDType(data_dtypes, 0);
            const dev_token = try moveScalarTokenToDevice(backend, input_token, data_shapes[0], token_dtype);
            try inputs.append(dev_token);

            const pos_dtype = generation_cache.dataInputDType(data_dtypes, 1);
            const dev_pos = try moveScalarTokenToDevice(backend, current_pos, data_shapes[1], pos_dtype);
            try inputs.append(dev_pos);

            for (kv_cache.*.?, kv_shapes, 0..) |kv_buf, shape, i| {
                const kv_dtype = generation_cache.dataInputDType(data_dtypes, i + 2);
                const dev_kv = try backend.moveToDevice(kv_buf, shape, kv_dtype);
                try inputs.append(dev_kv);
            }

            const outputs = try backend.executeWithDeviceBuffers(vmfb, "main", inputs.items);
            defer self.allocator.free(outputs);

            for (inputs.items) |buf| buf.release();

            const logits_bytes = try backend.readToHost(outputs[0]);
            defer self.allocator.free(logits_bytes);
            outputs[0].release();

            const logits_f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, logits_bytes)));

            const prompt_len_i64 = @as(i64, @intCast(prompt_tokens.len));
            const prompt_end_pos = if (prompt_len_i64 > 0) start_pos + prompt_len_i64 - 1 else start_pos - 1;
            const is_last_prompt_token = (prompt_len_i64 > 0 and current_pos == prompt_end_pos);
            const is_generating = (current_pos >= start_pos + prompt_len_i64);

            var finish_reason: ?[]const u8 = null;
            if ((is_last_prompt_token and prompt_tokens.len > 0) or is_generating) {
                const next_token = sampler.sample(logits_f32, temperature);
                try generated_tokens.append(next_token);
                if (on_token) |cb| {
                    if (on_token_ctx) |ctx| {
                        try cb(next_token, ctx);
                    } else {
                        try cb(next_token, undefined);
                    }
                }

                if (next_token == eos_token) {
                    finish_reason = "stop";
                } else if (generated_tokens.items.len >= max_new_tokens) {
                    finish_reason = "length";
                }
            }

            for (outputs[1..], 0..) |output_buf, i| {
                const update_shape = try output_buf.shapeAlloc(self.allocator);
                defer self.allocator.free(update_shape);
                const update_bytes = try backend.readToHost(output_buf);
                defer self.allocator.free(update_bytes);
                output_buf.release();

                const cache_buffer = kv_cache.*.?[i];
                const cache_dtype = generation_cache.dataInputDType(data_dtypes, i + 2);
                try generation_cache.copyStateUpdate(cache_buffer, kv_shapes[i], cache_dtype, update_bytes, update_shape, current_pos);
            }

            current_pos += 1;

            if (finish_reason) |reason| {
                return GenerationResult{
                    .tokens = try generated_tokens.toOwnedSlice(),
                    .finish_reason = reason,
                    .prompt_tokens = prompt_tokens.len,
                    .completion_tokens = generated_tokens.items.len,
                };
            }
        }

        return GenerationResult{
            .tokens = try generated_tokens.toOwnedSlice(),
            .finish_reason = "length",
            .prompt_tokens = prompt_tokens.len,
            .completion_tokens = generated_tokens.items.len,
        };
    }
};

fn dataInputDType(dtypes: ?[]const tensor.DType, index: usize, fallback: tensor.DType) tensor.DType {
    if (dtypes) |values| {
        if (index < values.len) return values[index];
    }
    return fallback;
}

fn moveScalarTokenToDevice(
    backend: *IreeBackend,
    value: i64,
    shape: []const i64,
    dtype: tensor.DType,
) !IreeBackend.DeviceBuffer {
    return switch (dtype) {
        .i64 => blk: {
            var scalar: i64 = value;
            break :blk try backend.moveToDevice(std.mem.asBytes(&scalar), shape, dtype);
        },
        .i32 => blk: {
            var scalar = std.math.cast(i32, value) orelse return error.GenerationScalarOutOfRange;
            break :blk try backend.moveToDevice(std.mem.asBytes(&scalar), shape, dtype);
        },
        else => error.UnsupportedGenerationScalarDType,
    };
}
