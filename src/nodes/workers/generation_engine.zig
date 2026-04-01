const std = @import("std");
const math = @import("../../core/math.zig");
const tensor = @import("../../core/tensor.zig");
const IreeBackend = @import("../../backends/iree.zig").IreeBackend;
const message = @import("../../network/message.zig");

const Allocator = std.mem.Allocator;

pub const LoadedInferenceModel = struct {
    model_id: []const u8,
    cached_vmfb: []const u8,
    param_shapes: []const []const i64,
    data_shapes: []const []const i64,
    data_dtypes: ?[]tensor.DType,
    max_context_tokens: usize,
};

pub const GenerationSession = struct {
    session_id: []const u8,
    model_id: []const u8,
    kv_cache_host: [][]u8,
    last_pos: i64,
};

pub const ActiveGenerationState = struct {
    request_id: message.RequestId,
    task_id: message.TaskId,
    model_id: []const u8,
    session_id: []const u8,
    prompt_tokens: usize,
    completion_tokens: usize,
    cancelled: bool,
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

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .device_weights = null,
            .device_kv_cache = null,
            .device_weights_dirty = false,
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
    ) !void {
        if (self.device_weights != null and !self.device_weights_dirty) return;

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

        var offset: usize = 0;
        for (param_shapes) |shape| {
            var size: usize = 4; // f32
            for (shape) |dim| size *= @intCast(dim);

            const slice = weight_blob[offset .. offset + size];
            const dev_buf = try backend.moveToDevice(slice, shape, .f32);
            try dev_weights.append(dev_buf);

            offset += size;
        }

        self.device_weights = try dev_weights.toOwnedSlice();
        self.device_weights_dirty = false;
    }

    pub fn initKvCacheHost(
        self: *Self,
        data_shapes: []const []const i64,
    ) ![][]u8 {
        _ = self;
        const kv_shapes = data_shapes[2..];
        const kv_buffers = try self.allocator.alloc([]u8, kv_shapes.len);
        for (kv_shapes, 0..) |shape, i| {
            var size: usize = 4; // f32
            for (shape) |dim| size *= @intCast(dim);
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
        weight_blob: []const u8,
        prompt_tokens: []const i64,
        max_new_tokens: usize,
        eos_token: i64,
        kv_cache: *?[][]u8,
    ) !GenerationResult {
        try self.ensureDeviceWeights(backend, weight_blob, param_shapes);

        if (kv_cache.*) |old_cache| {
            for (old_cache) |buf| self.allocator.free(buf);
            self.allocator.free(old_cache);
        }
        kv_cache.* = try self.initKvCacheHost(data_shapes);

        var generated_tokens = std.ArrayList(i64).init(self.allocator);
        errdefer generated_tokens.deinit();

        var current_pos: i64 = 0;
        const total_len = prompt_tokens.len + max_new_tokens;
        const kv_shapes = data_shapes[2..];

        while (current_pos < total_len) {
            var input_token: i64 = 0;
            if (current_pos < prompt_tokens.len) {
                input_token = prompt_tokens[@intCast(current_pos)];
            } else if (generated_tokens.items.len > 0) {
                input_token = generated_tokens.items[generated_tokens.items.len - 1];
            }

            var inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
            defer inputs.deinit();

            for (self.device_weights.?) |w| {
                w.retain();
                try inputs.append(w);
            }

            const token_bytes = std.mem.asBytes(&input_token);
            const dev_token = try backend.moveToDevice(token_bytes, data_shapes[0], .i64);
            try inputs.append(dev_token);

            const pos_bytes = std.mem.asBytes(&current_pos);
            const dev_pos = try backend.moveToDevice(pos_bytes, data_shapes[1], .i64);
            try inputs.append(dev_pos);

            for (kv_cache.*.?, kv_shapes) |kv_buf, shape| {
                const dev_kv = try backend.moveToDevice(kv_buf, shape, .f32);
                try inputs.append(dev_kv);
            }

            const outputs = try backend.executeWithDeviceBuffers(vmfb, "main", inputs.items);
            defer self.allocator.free(outputs);

            for (inputs.items) |buf| buf.release();

            const logits_bytes = try backend.readToHost(outputs[0]);
            defer self.allocator.free(logits_bytes);
            outputs[0].release();

            const logits_f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, logits_bytes)));

            const is_last_prompt_token = (current_pos == prompt_tokens.len - 1);
            const is_generating = (current_pos >= prompt_tokens.len);

            var finish_reason: ?[]const u8 = null;
            if ((is_last_prompt_token and prompt_tokens.len > 0) or is_generating) {
                const next_token = sampler.sample(logits_f32, 0.7);
                try generated_tokens.append(next_token);

                if (next_token == eos_token) {
                    finish_reason = "stop";
                } else if (generated_tokens.items.len >= max_new_tokens) {
                    finish_reason = "length";
                }
            }

            // Update KV cache on host
            const NUM_KV_HEADS: usize = 2;
            const HEAD_DIM: usize = 64;
            const MAX_SEQ_LEN: usize = 1024;
            const bytes_per_head_slot: usize = HEAD_DIM * 4;
            const head_stride: usize = MAX_SEQ_LEN * bytes_per_head_slot;

            for (outputs[1..], 0..) |output_buf, i| {
                const update_bytes = try backend.readToHost(output_buf);
                defer self.allocator.free(update_bytes);
                output_buf.release();

                const cache_buffer = kv_cache.*.?[i];

                for (0..NUM_KV_HEADS) |head| {
                    const src_offset = head * bytes_per_head_slot;
                    const dst_offset = head * head_stride + @as(usize, @intCast(current_pos)) * bytes_per_head_slot;

                    if (dst_offset + bytes_per_head_slot <= cache_buffer.len and
                        src_offset + bytes_per_head_slot <= update_bytes.len)
                    {
                        @memcpy(
                            cache_buffer[dst_offset .. dst_offset + bytes_per_head_slot],
                            update_bytes[src_offset .. src_offset + bytes_per_head_slot],
                        );
                    }
                }
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
