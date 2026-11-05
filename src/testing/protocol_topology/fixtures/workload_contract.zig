const decoupled_job = @import("../../../protocol/decoupled_job.zig");
const regular_config = @import("../../../workloads/training/config.zig");

pub const shared_embedding_indices = [_]usize{ 0, 1 };

pub fn normalizationOptions(run_id: []const u8, workers: usize) decoupled_job.NormalizationOptions {
    return .{
        .run_id = run_id,
        .workers = workers,
        .backend = "cpu",
        .target_arch = null,
    };
}

pub fn regularConfig() regular_config.ExperimentConfig {
    return .{
        .model_path = "shared-model",
        .data_path = "train.u16",
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
        .outer_gradient_compression = "none",
        .max_recovery_syncer_steps = 5,
        .embedding_tensor_indices = &shared_embedding_indices,
        .checkpoint_dir = "checkpoints",
    };
}

pub fn regularJob() !decoupled_job.DecoupledTrainingJob {
    return decoupled_job.DecoupledTrainingJob.fromRegularConfig(regularConfig(), normalizationOptions("run-shared", 2));
}
