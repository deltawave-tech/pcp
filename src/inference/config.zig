const std = @import("std");

const Allocator = std.mem.Allocator;

pub const InferenceConfig = struct {
    model_id: []const u8,
    pool_name: []const u8,
    generation_vmfb_path: []const u8,
    generation_mlir_path: []const u8,
    weights_path: []const u8,
    tokenizer_source: []const u8,
    tokenizer_path: ?[]const u8 = null,
    num_gen_data_inputs: usize,
    max_context_tokens: usize,
    default_max_output_tokens: usize,
    eos_token: i64 = 0,
    session_ttl_seconds: u64,
    request_timeout_seconds: u64,
    worker_backend: []const u8,
    worker_target_arch: []const u8,
    api_token_env: []const u8,
};

pub const ConfigResult = struct {
    config: InferenceConfig,
    parsed: ?std.json.Parsed(InferenceConfig),
    json_data: ?[]u8,
    allocator: Allocator,

    pub fn deinit(self: *@This()) void {
        if (self.parsed) |*p| {
            p.deinit();
        }
        if (self.json_data) |data| {
            self.allocator.free(data);
        }
    }
};

pub fn loadInferenceConfig(allocator: Allocator, path: []const u8) !ConfigResult {
    std.log.info("Loading inference config from: {s}", .{path});
    const data = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    const parsed = try std.json.parseFromSlice(InferenceConfig, allocator, data, .{ .ignore_unknown_fields = true });
    return ConfigResult{
        .config = parsed.value,
        .parsed = parsed,
        .json_data = data,
        .allocator = allocator,
    };
}
