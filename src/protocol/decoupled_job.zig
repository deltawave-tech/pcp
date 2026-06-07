const std = @import("std");

const experiment_config = @import("../workloads/training/config.zig");

pub const TrainingKind = enum {
    regular,
    custom_extension,
};

pub const RegularTrainingPayload = struct {
    model_path: []const u8,
    data_path: []const u8,
    tokenizer: []const u8,
    sampling: []const u8,
    checkpoint_dir: ?[]const u8,
};

pub const CustomTrainingPayload = struct {
    kind: []const u8,
    bundle_path: []const u8,
    split_dir: []const u8,
    output_dir: []const u8,
    stage: []const u8,
    resume_from_dir: ?[]const u8,
    resume_optimizer: bool,
};

pub const TrainingPayload = union(TrainingKind) {
    regular: RegularTrainingPayload,
    custom_extension: CustomTrainingPayload,
};

pub const NormalizationOptions = struct {
    run_id: []const u8,
    workers: usize,
    backend: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
};

pub const DecoupledTrainingJob = struct {
    model_id: []const u8,
    training_kind: TrainingKind,
    run_id: []const u8,
    workers: usize,
    backend: []const u8,
    target_arch: ?[]const u8,
    max_steps: usize,
    outer_loop_steps: usize,
    num_fragments: usize,
    sync_interval_h: usize,
    overlap_tau: usize,
    min_quorum: usize,
    merge_strategy: []const u8,
    fragment_strategy: []const u8,
    outer_learning_rate: f32,
    outer_momentum: f32,
    learner_alpha: f32,
    grace_window_ms: u64,
    grace_gamma: f64,
    max_grace_steps: usize,
    adaptive_grace_enabled: bool,
    outer_gradient_compression: []const u8,
    max_recovery_syncer_steps: usize,
    embedding_tensor_indices: []const usize,
    payload: TrainingPayload,

    pub fn fromRegularConfig(cfg: experiment_config.ExperimentConfig, options: NormalizationOptions) !@This() {
        const aggregation = cfg.aggregation_strategy orelse return error.InvalidDecoupledAggregation;
        if (!std.mem.eql(u8, aggregation, "decoupled_diloco")) return error.InvalidDecoupledAggregation;

        const job = @This(){
            .model_id = cfg.model_path,
            .training_kind = .regular,
            .run_id = options.run_id,
            .workers = options.workers,
            .backend = options.backend orelse "cpu",
            .target_arch = options.target_arch,
            .max_steps = cfg.outer_loop_steps,
            .outer_loop_steps = cfg.outer_loop_steps,
            .num_fragments = cfg.num_fragments,
            .sync_interval_h = cfg.sync_interval_h orelse cfg.tau,
            .overlap_tau = cfg.overlap_tau,
            .min_quorum = cfg.min_quorum,
            .merge_strategy = cfg.merge_strategy,
            .fragment_strategy = cfg.fragment_strategy,
            .outer_learning_rate = cfg.outer_learning_rate orelse 0.7,
            .outer_momentum = cfg.nesterov_momentum,
            .learner_alpha = cfg.learner_alpha,
            .grace_window_ms = cfg.grace_window_ms,
            .grace_gamma = cfg.grace_gamma,
            .max_grace_steps = cfg.max_grace_steps,
            .adaptive_grace_enabled = cfg.adaptive_grace_enabled,
            .outer_gradient_compression = cfg.outer_gradient_compression,
            .max_recovery_syncer_steps = cfg.max_recovery_syncer_steps,
            .embedding_tensor_indices = cfg.embedding_tensor_indices,
            .payload = .{
                .regular = .{
                    .model_path = cfg.model_path,
                    .data_path = cfg.data_path,
                    .tokenizer = cfg.tokenizer,
                    .sampling = cfg.sampling,
                    .checkpoint_dir = cfg.checkpoint_dir,
                },
            },
        };
        try job.validate();
        return job;
    }

    pub fn validate(self: @This()) !void {
        if (self.model_id.len == 0) return error.InvalidModelId;
        if (self.run_id.len == 0) return error.InvalidRunId;
        if (self.workers == 0) return error.InvalidWorkerCount;
        if (self.max_steps == 0) return error.InvalidMaxSteps;
        if (self.outer_loop_steps == 0) return error.InvalidOuterLoopSteps;
        if (self.num_fragments == 0) return error.InvalidFragmentCount;
        if (self.sync_interval_h == 0) return error.InvalidSyncInterval;
        if (self.overlap_tau == 0) return error.InvalidOverlapTau;
        if (self.min_quorum == 0 or self.min_quorum > self.workers) return error.InvalidQuorum;
        if (self.learner_alpha < 0.0 or self.learner_alpha > 1.0) return error.InvalidLearnerAlpha;
        if (!std.math.isFinite(self.outer_learning_rate)) return error.InvalidOuterLearningRate;
        if (!std.math.isFinite(self.outer_momentum)) return error.InvalidOuterMomentum;
        if (self.grace_gamma < 0.0 or !std.math.isFinite(self.grace_gamma)) return error.InvalidGraceGamma;
        if (self.max_grace_steps == 0) return error.InvalidMaxGraceSteps;
        try validateBackendName(self.backend);
        try validateMergeStrategy(self.merge_strategy);
        try validateFragmentStrategy(self.fragment_strategy);
        try validateOuterGradientCompression(self.outer_gradient_compression);
    }

    pub fn sharedFieldsEqual(self: @This(), other: @This()) bool {
        if (!std.mem.eql(u8, self.model_id, other.model_id)) return false;
        if (!std.mem.eql(u8, self.run_id, other.run_id)) return false;
        if (self.workers != other.workers) return false;
        if (!std.mem.eql(u8, self.backend, other.backend)) return false;
        if (!optionalStringEqual(self.target_arch, other.target_arch)) return false;
        if (self.max_steps != other.max_steps) return false;
        if (self.outer_loop_steps != other.outer_loop_steps) return false;
        if (self.num_fragments != other.num_fragments) return false;
        if (self.sync_interval_h != other.sync_interval_h) return false;
        if (self.overlap_tau != other.overlap_tau) return false;
        if (self.min_quorum != other.min_quorum) return false;
        if (!std.mem.eql(u8, self.merge_strategy, other.merge_strategy)) return false;
        if (!std.mem.eql(u8, self.fragment_strategy, other.fragment_strategy)) return false;
        if (self.outer_learning_rate != other.outer_learning_rate) return false;
        if (self.outer_momentum != other.outer_momentum) return false;
        if (self.learner_alpha != other.learner_alpha) return false;
        if (self.grace_window_ms != other.grace_window_ms) return false;
        if (self.grace_gamma != other.grace_gamma) return false;
        if (self.max_grace_steps != other.max_grace_steps) return false;
        if (self.adaptive_grace_enabled != other.adaptive_grace_enabled) return false;
        if (!std.mem.eql(u8, self.outer_gradient_compression, other.outer_gradient_compression)) return false;
        if (self.max_recovery_syncer_steps != other.max_recovery_syncer_steps) return false;
        return std.mem.eql(usize, self.embedding_tensor_indices, other.embedding_tensor_indices);
    }
};

pub fn validateBackendName(value: []const u8) !void {
    if (std.mem.eql(u8, value, "metal")) return;
    if (std.mem.eql(u8, value, "cuda")) return;
    if (std.mem.eql(u8, value, "vulkan")) return;
    if (std.mem.eql(u8, value, "rocm")) return;
    if (std.mem.eql(u8, value, "cpu")) return;
    return error.UnknownBackend;
}

pub fn validateFragmentStrategy(value: []const u8) !void {
    if (std.mem.eql(u8, value, "strided")) return;
    if (std.mem.eql(u8, value, "balanced_tensor")) return;
    return error.InvalidFragmentStrategy;
}

pub fn validateMergeStrategy(value: []const u8) !void {
    if (std.mem.eql(u8, value, "weighted_average")) return;
    if (std.mem.eql(u8, value, "rda")) return;
    if (std.mem.eql(u8, value, "avg_embedding_rda_model")) return;
    if (std.mem.eql(u8, value, "rda_embedding_avg_model")) return;
    return error.InvalidMergeStrategy;
}

pub fn validateOuterGradientCompression(value: []const u8) !void {
    if (std.mem.eql(u8, value, "none")) return;
    if (std.mem.eql(u8, value, "int4")) return;
    return error.InvalidOuterGradientCompression;
}

fn optionalStringEqual(left: ?[]const u8, right: ?[]const u8) bool {
    if (left == null and right == null) return true;
    if (left == null or right == null) return false;
    return std.mem.eql(u8, left.?, right.?);
}

fn regularFixture(embedding_tensor_indices: []const usize) experiment_config.ExperimentConfig {
    return .{
        .model_path = "model-a",
        .data_path = "data.bin",
        .tokenizer = "u16",
        .sampling = "sequential",
        .learning_rate = 0.001,
        .tau = 8,
        .outer_loop_steps = 12,
        .nesterov_momentum = 0.5,
        .max_epochs = 1,
        .dtype = "f32",
        .aggregation_strategy = "decoupled_diloco",
        .num_fragments = 4,
        .sync_interval_h = 8,
        .overlap_tau = 2,
        .min_quorum = 2,
        .grace_window_ms = 10,
        .grace_gamma = 1.25,
        .max_grace_steps = 3,
        .adaptive_grace_enabled = true,
        .merge_strategy = "rda",
        .fragment_strategy = "strided",
        .learner_alpha = 0.25,
        .outer_learning_rate = 0.7,
        .outer_gradient_compression = "int4",
        .max_recovery_syncer_steps = 5,
        .embedding_tensor_indices = embedding_tensor_indices,
        .checkpoint_dir = "out",
    };
}

test "regular decoupled configs normalize common coordinator fields" {
    const embeddings = [_]usize{ 0, 2 };
    const options = NormalizationOptions{
        .run_id = "run-1",
        .workers = 2,
        .backend = "cpu",
    };

    const regular = try DecoupledTrainingJob.fromRegularConfig(regularFixture(&embeddings), options);

    try std.testing.expectEqual(TrainingKind.regular, regular.training_kind);
    try std.testing.expectEqualStrings("data.bin", regular.payload.regular.data_path);
}

test "invalid shared decoupled fields fail for regular training" {
    const embeddings = [_]usize{0};
    const options = NormalizationOptions{
        .run_id = "run-1",
        .workers = 2,
        .backend = "cpu",
    };

    var regular = regularFixture(&embeddings);
    regular.min_quorum = 0;
    try std.testing.expectError(error.InvalidQuorum, DecoupledTrainingJob.fromRegularConfig(regular, options));

    regular = regularFixture(&embeddings);
    regular.merge_strategy = "not-a-merge";
    try std.testing.expectError(error.InvalidMergeStrategy, DecoupledTrainingJob.fromRegularConfig(regular, options));
}
