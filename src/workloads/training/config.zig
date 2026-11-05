const std = @import("std");

const Allocator = std.mem.Allocator;
const config_blocks = @import("config_blocks.zig");

pub const GRPOJsonConfig = struct {
    num_iterations: usize,
    group_size: usize,
    learning_rate: f32,
    beta: f32,
    prompt_file: []const u8,
    num_prompts: usize,
    rollout_max_tokens: usize = 128,
    weights_path: []const u8,
    training_weights_path: ?[]const u8 = null,
    generation_weights_path: ?[]const u8 = null,
    generation_weights_format: []const u8 = "flat",
    adapter_state_path: ?[]const u8 = null,
    grpo_weight_mode: []const u8 = "full",
    generation_vmfb_path: []const u8,
    generation_mlir_path: []const u8,
    training_mlir_path: []const u8,
    num_gen_data_inputs: ?usize = null,
    weight_refresh_strategy: []const u8 = "inline",
    updated_weights_path: ?[]const u8 = null,
    trainable_parameter_indices: ?[]const usize = null,
};

pub const ExperimentConfig = struct {
    model_path: []const u8,
    data_path: []const u8,
    tokenizer: []const u8,
    sampling: []const u8,
    learning_rate: f32,
    tau: usize,
    outer_loop_steps: usize,
    nesterov_momentum: f32,
    max_epochs: usize,
    dtype: []const u8,
    aggregation_strategy: ?[]const u8 = null,
    num_fragments: usize = 24,
    sync_interval_h: ?usize = null,
    overlap_tau: usize = 2,
    min_quorum: usize = 1,
    grace_window_ms: u64 = 0,
    grace_gamma: f64 = 1.0,
    max_grace_steps: usize = 1,
    adaptive_grace_enabled: bool = false,
    merge_strategy: []const u8 = "avg_embedding_rda_model",
    fragment_strategy: []const u8 = "balanced_tensor",
    learner_alpha: f32 = 0.0,
    outer_learning_rate: ?f32 = null,
    outer_gradient_compression: []const u8 = "none",
    max_recovery_syncer_steps: usize = 24,
    embedding_tensor_indices: []const usize = &.{},
    effective_batch_size: ?usize = null,
    use_in_graph_accumulation: bool = false,
    grpo_config: ?GRPOJsonConfig = null,
    wandb_project: ?[]const u8 = null,
    wandb_entity: ?[]const u8 = null,
    wandb_run_name: ?[]const u8 = null,
    wandb_api_key: ?[]const u8 = null,
    checkpoint_dir: ?[]const u8 = null,
    should_resume: bool = false,
};

pub const RawExperimentConfig = struct {
    model_path: ?[]const u8 = null,
    data_path: ?[]const u8 = null,
    tokenizer: ?[]const u8 = null,
    sampling: ?[]const u8 = null,
    learning_rate: ?f32 = null,
    tau: ?usize = null,
    outer_loop_steps: ?usize = null,
    nesterov_momentum: ?f32 = null,
    max_epochs: ?usize = null,
    dtype: ?[]const u8 = null,
    aggregation_strategy: ?[]const u8 = null,
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
    effective_batch_size: ?usize = null,
    use_in_graph_accumulation: ?bool = null,
    grpo_config: ?GRPOJsonConfig = null,
    wandb_project: ?[]const u8 = null,
    wandb_entity: ?[]const u8 = null,
    wandb_run_name: ?[]const u8 = null,
    wandb_api_key: ?[]const u8 = null,
    checkpoint_dir: ?[]const u8 = null,
    should_resume: ?bool = null,
    distributed: ?config_blocks.DistributedConfig = null,
    decoupled_diloco: ?config_blocks.DecoupledDiLoCoConfig = null,
    artifacts: ?config_blocks.ArtifactConfig = null,
    dataset: ?config_blocks.DatasetConfig = null,
    outputs: ?config_blocks.OutputsConfig = null,
};

pub const ConfigResult = struct {
    config: ExperimentConfig,
    parsed: ?std.json.Parsed(RawExperimentConfig),
    json_data: ?[]u8,
    allocator: Allocator,

    pub fn deinit(self: *@This()) void {
        if (self.parsed) |*parsed| {
            parsed.deinit();
        }
        if (self.json_data) |data| {
            self.allocator.free(data);
        }
    }
};

pub fn loadConfig(allocator: Allocator, path: ?[]const u8) !ConfigResult {
    if (path) |config_path| {
        std.log.info("Loading config from: {s}", .{config_path});
        const data = try std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
        const parsed = try std.json.parseFromSlice(RawExperimentConfig, allocator, data, .{ .ignore_unknown_fields = true });
        const config = try normalizeConfig(parsed.value);
        config_blocks.warnLegacyShape("regular training", hasConvergedBlocks(parsed.value), hasLegacyFields(parsed.value));
        return .{
            .config = config,
            .parsed = parsed,
            .json_data = data,
            .allocator = allocator,
        };
    }

    std.log.err("No config file specified. Use --config <path> to provide an experiment configuration.", .{});
    return error.ConfigFileRequired;
}

pub fn normalizeConfig(raw: RawExperimentConfig) !ExperimentConfig {
    const distributed = raw.distributed;
    const decoupled = raw.decoupled_diloco;
    const artifacts = raw.artifacts;
    const dataset = raw.dataset;
    const outputs = raw.outputs;

    return .{
        .model_path = try config_blocks.requireField([]const u8, if (artifacts) |block| block.model_path orelse raw.model_path else raw.model_path, error.MissingModelPath),
        .data_path = try config_blocks.requireField([]const u8, regularDataPath(dataset, raw.data_path), error.MissingDataPath),
        .tokenizer = try config_blocks.requireField([]const u8, if (dataset) |block| block.tokenizer orelse raw.tokenizer else raw.tokenizer, error.MissingTokenizer),
        .sampling = try config_blocks.requireField([]const u8, if (dataset) |block| block.sampling orelse raw.sampling else raw.sampling, error.MissingSampling),
        .learning_rate = try config_blocks.requireField(f32, if (distributed) |block| block.learning_rate orelse raw.learning_rate else raw.learning_rate, error.MissingLearningRate),
        .tau = try config_blocks.requireField(usize, if (distributed) |block| block.local_steps_per_round orelse raw.tau else raw.tau, error.MissingTau),
        .outer_loop_steps = try config_blocks.requireField(usize, if (distributed) |block| block.outer_loop_steps orelse raw.outer_loop_steps else raw.outer_loop_steps, error.MissingOuterLoopSteps),
        .nesterov_momentum = try config_blocks.requireField(f32, if (distributed) |block| block.outer_momentum orelse raw.nesterov_momentum else raw.nesterov_momentum, error.MissingNesterovMomentum),
        .max_epochs = try config_blocks.requireField(usize, if (distributed) |block| block.max_epochs orelse raw.max_epochs else raw.max_epochs, error.MissingMaxEpochs),
        .dtype = try config_blocks.requireField([]const u8, if (distributed) |block| block.dtype orelse raw.dtype else raw.dtype, error.MissingDType),
        .aggregation_strategy = if (distributed) |block| block.aggregation_strategy orelse raw.aggregation_strategy else raw.aggregation_strategy,
        .num_fragments = if (decoupled) |block| config_blocks.select(usize, block.num_fragments, raw.num_fragments, 24) else raw.num_fragments orelse 24,
        .sync_interval_h = if (decoupled) |block| block.sync_interval_h orelse raw.sync_interval_h else raw.sync_interval_h,
        .overlap_tau = if (decoupled) |block| config_blocks.select(usize, block.overlap_tau, raw.overlap_tau, 2) else raw.overlap_tau orelse 2,
        .min_quorum = if (decoupled) |block| config_blocks.select(usize, block.min_quorum, raw.min_quorum, 1) else raw.min_quorum orelse 1,
        .grace_window_ms = if (decoupled) |block| config_blocks.select(u64, block.grace_window_ms, raw.grace_window_ms, 0) else raw.grace_window_ms orelse 0,
        .grace_gamma = if (decoupled) |block| config_blocks.select(f64, block.grace_gamma, raw.grace_gamma, 1.0) else raw.grace_gamma orelse 1.0,
        .max_grace_steps = if (decoupled) |block| config_blocks.select(usize, block.max_grace_steps, raw.max_grace_steps, 1) else raw.max_grace_steps orelse 1,
        .adaptive_grace_enabled = if (decoupled) |block| config_blocks.select(bool, block.adaptive_grace_enabled, raw.adaptive_grace_enabled, false) else raw.adaptive_grace_enabled orelse false,
        .merge_strategy = if (decoupled) |block| config_blocks.select([]const u8, block.merge_strategy, raw.merge_strategy, "avg_embedding_rda_model") else raw.merge_strategy orelse "avg_embedding_rda_model",
        .fragment_strategy = if (decoupled) |block| config_blocks.select([]const u8, block.fragment_strategy, raw.fragment_strategy, "balanced_tensor") else raw.fragment_strategy orelse "balanced_tensor",
        .learner_alpha = if (decoupled) |block| config_blocks.select(f32, block.learner_alpha, raw.learner_alpha, 0.0) else raw.learner_alpha orelse 0.0,
        .outer_learning_rate = if (decoupled) |block| block.outer_learning_rate orelse raw.outer_learning_rate else raw.outer_learning_rate,
        .outer_gradient_compression = if (decoupled) |block| config_blocks.select([]const u8, block.outer_gradient_compression, raw.outer_gradient_compression, "none") else raw.outer_gradient_compression orelse "none",
        .max_recovery_syncer_steps = if (decoupled) |block| config_blocks.select(usize, block.max_recovery_syncer_steps, raw.max_recovery_syncer_steps, 24) else raw.max_recovery_syncer_steps orelse 24,
        .embedding_tensor_indices = if (decoupled) |block| block.embedding_tensor_indices orelse raw.embedding_tensor_indices orelse &.{} else raw.embedding_tensor_indices orelse &.{},
        .effective_batch_size = if (distributed) |block| block.effective_batch_size orelse raw.effective_batch_size else raw.effective_batch_size,
        .use_in_graph_accumulation = if (distributed) |block| config_blocks.select(bool, block.use_in_graph_accumulation, raw.use_in_graph_accumulation, false) else raw.use_in_graph_accumulation orelse false,
        .grpo_config = raw.grpo_config,
        .wandb_project = if (outputs) |block| block.wandb_project orelse raw.wandb_project else raw.wandb_project,
        .wandb_entity = if (outputs) |block| block.wandb_entity orelse raw.wandb_entity else raw.wandb_entity,
        .wandb_run_name = if (outputs) |block| block.wandb_run_name orelse raw.wandb_run_name else raw.wandb_run_name,
        .wandb_api_key = if (outputs) |block| block.wandb_api_key orelse raw.wandb_api_key else raw.wandb_api_key,
        .checkpoint_dir = if (outputs) |block| block.checkpoint_dir orelse raw.checkpoint_dir else raw.checkpoint_dir,
        .should_resume = if (outputs) |block| config_blocks.select(bool, block.should_resume, raw.should_resume, false) else raw.should_resume orelse false,
    };
}

fn regularDataPath(dataset: ?config_blocks.DatasetConfig, legacy: ?[]const u8) ?[]const u8 {
    if (dataset) |block| return block.path orelse block.data_path orelse legacy;
    return legacy;
}

fn hasConvergedBlocks(raw: RawExperimentConfig) bool {
    return raw.distributed != null or raw.decoupled_diloco != null or raw.artifacts != null or raw.dataset != null or raw.outputs != null;
}

fn hasLegacyFields(raw: RawExperimentConfig) bool {
    return raw.model_path != null or raw.data_path != null or raw.tokenizer != null or raw.sampling != null or
        raw.learning_rate != null or raw.tau != null or raw.outer_loop_steps != null or raw.nesterov_momentum != null or
        raw.max_epochs != null or raw.dtype != null or raw.aggregation_strategy != null or raw.num_fragments != null or
        raw.sync_interval_h != null or raw.checkpoint_dir != null or raw.wandb_project != null or raw.should_resume != null;
}

test "regular config normalizes legacy flat shape" {
    const raw = RawExperimentConfig{
        .model_path = "model.mlir",
        .data_path = "data.u16",
        .tokenizer = "u16",
        .sampling = "random",
        .learning_rate = 0.001,
        .tau = 4,
        .outer_loop_steps = 2,
        .nesterov_momentum = 0.9,
        .max_epochs = 1,
        .dtype = "f32",
        .aggregation_strategy = "decoupled_diloco",
        .num_fragments = 3,
        .sync_interval_h = 4,
        .checkpoint_dir = "checkpoints",
    };

    const cfg = try normalizeConfig(raw);
    try std.testing.expectEqualStrings("model.mlir", cfg.model_path);
    try std.testing.expectEqualStrings("data.u16", cfg.data_path);
    try std.testing.expectEqual(@as(usize, 3), cfg.num_fragments);
    try std.testing.expectEqual(@as(?usize, 4), cfg.sync_interval_h);
    try std.testing.expectEqualStrings("checkpoints", cfg.checkpoint_dir.?);
}

test "regular config block values override legacy aliases" {
    const raw = RawExperimentConfig{
        .model_path = "legacy.mlir",
        .data_path = "legacy.u16",
        .tokenizer = "char",
        .sampling = "sequential",
        .learning_rate = 0.001,
        .tau = 2,
        .outer_loop_steps = 1,
        .nesterov_momentum = 0.1,
        .max_epochs = 1,
        .dtype = "f32",
        .num_fragments = 2,
        .artifacts = .{ .model_path = "block.mlir" },
        .dataset = .{ .path = "block.u16", .tokenizer = "u16", .sampling = "random" },
        .distributed = .{
            .learning_rate = 0.002,
            .local_steps_per_round = 8,
            .outer_loop_steps = 3,
            .outer_momentum = 0.9,
            .max_epochs = 4,
            .dtype = "bf16",
        },
        .decoupled_diloco = .{ .num_fragments = 5 },
    };

    const cfg = try normalizeConfig(raw);
    try std.testing.expectEqualStrings("block.mlir", cfg.model_path);
    try std.testing.expectEqualStrings("block.u16", cfg.data_path);
    try std.testing.expectEqualStrings("u16", cfg.tokenizer);
    try std.testing.expectEqual(@as(usize, 8), cfg.tau);
    try std.testing.expectEqual(@as(usize, 3), cfg.outer_loop_steps);
    try std.testing.expectEqual(@as(usize, 5), cfg.num_fragments);
    try std.testing.expectEqualStrings("bf16", cfg.dtype);
}
