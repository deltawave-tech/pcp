const std = @import("std");

pub const DistributedConfig = struct {
    aggregation_strategy: ?[]const u8 = null,
    outer_loop_steps: ?usize = null,
    local_steps_per_round: ?usize = null,
    learning_rate: ?f32 = null,
    outer_momentum: ?f32 = null,
    max_epochs: ?usize = null,
    dtype: ?[]const u8 = null,
    backend: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    device_id: ?usize = null,
    seed: ?u64 = null,
    log_every: ?usize = null,
    effective_batch_size: ?usize = null,
    use_in_graph_accumulation: ?bool = null,
};

pub const DecoupledDiLoCoConfig = struct {
    num_fragments: ?usize = null,
    sync_interval_h: ?usize = null,
    overlap_tau: ?usize = null,
    min_quorum: ?usize = null,
    grace_window_ms: ?u64 = null,
    grace_gamma: ?f64 = null,
    max_grace_steps: ?usize = null,
    adaptive_grace_enabled: ?bool = null,
    merge_strategy: ?[]const u8 = null,
    fragment_strategy: ?[]const u8 = null,
    learner_alpha: ?f32 = null,
    outer_learning_rate: ?f32 = null,
    outer_gradient_compression: ?[]const u8 = null,
    max_recovery_syncer_steps: ?usize = null,
    embedding_tensor_indices: ?[]const usize = null,
};

pub const ArtifactConfig = struct {
    model_id: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    bundle_path: ?[]const u8 = null,
};

pub const DatasetConfig = struct {
    path: ?[]const u8 = null,
    data_path: ?[]const u8 = null,
    split_dir: ?[]const u8 = null,
    tokenizer: ?[]const u8 = null,
    sampling: ?[]const u8 = null,
};

pub const OutputsConfig = struct {
    checkpoint_dir: ?[]const u8 = null,
    output_dir: ?[]const u8 = null,
    should_resume: ?bool = null,
    resume_from_dir: ?[]const u8 = null,
    resume_optimizer: ?bool = null,
    encoder_cache_dir: ?[]const u8 = null,
    wandb_project: ?[]const u8 = null,
    wandb_entity: ?[]const u8 = null,
    wandb_run_name: ?[]const u8 = null,
    wandb_api_key: ?[]const u8 = null,
};

pub fn select(comptime T: type, preferred: ?T, legacy: ?T, fallback: T) T {
    return preferred orelse legacy orelse fallback;
}

pub fn optionalSelect(comptime T: type, preferred: ?T, legacy: ?T) ?T {
    return preferred orelse legacy;
}

pub fn requireField(comptime T: type, value: ?T, err: anyerror) !T {
    return value orelse err;
}

pub fn warnLegacyShape(workload: []const u8, has_blocks: bool, has_legacy_fields: bool) void {
    if (!has_legacy_fields) return;
    if (has_blocks) {
        std.log.warn("{s} config uses legacy flat aliases together with converged config blocks; block values take precedence", .{workload});
    } else {
        std.log.warn("{s} config uses the legacy flat shape; prefer distributed/decoupled_diloco/artifacts/dataset/outputs blocks", .{workload});
    }
}
