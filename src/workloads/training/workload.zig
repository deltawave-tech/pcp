const std = @import("std");

const data_assignment = @import("../../data/assignment.zig");
const decoupled_job = @import("../../protocol/decoupled_job.zig");

const Allocator = std.mem.Allocator;

pub const TrainingKind = decoupled_job.TrainingKind;
pub const TrainingPayload = decoupled_job.TrainingPayload;
pub const RegularTrainingPayload = decoupled_job.RegularTrainingPayload;
pub const CustomTrainingPayload = decoupled_job.CustomTrainingPayload;

pub const training_kind_regular = "regular";

pub const capability_regular = "training.regular";
pub const capability_decoupled_diloco = "training.decoupled_diloco";

pub const ArtifactRefs = struct {
    model_path: ?[]const u8 = null,
    bundle_path: ?[]const u8 = null,
    checkpoint_dir: ?[]const u8 = null,
    output_dir: ?[]const u8 = null,
    resume_from_dir: ?[]const u8 = null,
};

pub const DatasetRef = struct {
    local_path: ?[]const u8 = null,
    split_dir: ?[]const u8 = null,
    assignment: ?data_assignment.DataAssignment = null,
};

pub const DecoupledConfig = struct {
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

    pub fn fromDecoupledJob(job: decoupled_job.DecoupledTrainingJob) @This() {
        return .{
            .outer_loop_steps = job.outer_loop_steps,
            .num_fragments = job.num_fragments,
            .sync_interval_h = job.sync_interval_h,
            .overlap_tau = job.overlap_tau,
            .min_quorum = job.min_quorum,
            .merge_strategy = job.merge_strategy,
            .fragment_strategy = job.fragment_strategy,
            .outer_learning_rate = job.outer_learning_rate,
            .outer_momentum = job.outer_momentum,
            .learner_alpha = job.learner_alpha,
            .grace_window_ms = job.grace_window_ms,
            .grace_gamma = job.grace_gamma,
            .max_grace_steps = job.max_grace_steps,
            .adaptive_grace_enabled = job.adaptive_grace_enabled,
            .outer_gradient_compression = job.outer_gradient_compression,
            .max_recovery_syncer_steps = job.max_recovery_syncer_steps,
            .embedding_tensor_indices = job.embedding_tensor_indices,
        };
    }
};

pub const TrainingWorkloadSpec = struct {
    model_id: []const u8,
    training_kind: TrainingKind,
    run_id: []const u8,
    workers: usize,
    backend: []const u8,
    target_arch: ?[]const u8,
    max_steps: usize,
    decoupled: DecoupledConfig,
    artifacts: ArtifactRefs,
    dataset: DatasetRef,
    payload: TrainingPayload,

    pub fn fromDecoupledJob(job: decoupled_job.DecoupledTrainingJob) !@This() {
        const spec = @This(){
            .model_id = job.model_id,
            .training_kind = job.training_kind,
            .run_id = job.run_id,
            .workers = job.workers,
            .backend = job.backend,
            .target_arch = job.target_arch,
            .max_steps = job.max_steps,
            .decoupled = DecoupledConfig.fromDecoupledJob(job),
            .artifacts = artifactRefsFromPayload(job.payload),
            .dataset = datasetRefFromPayload(job.payload),
            .payload = job.payload,
        };
        try spec.validate();
        return spec;
    }

    pub fn validate(self: @This()) !void {
        if (self.model_id.len == 0) return error.InvalidModelId;
        if (self.run_id.len == 0) return error.InvalidRunId;
        if (self.workers == 0) return error.InvalidWorkerCount;
        if (self.max_steps == 0) return error.InvalidMaxSteps;
        if (std.meta.activeTag(self.payload) != self.training_kind) return error.InvalidTrainingPayload;
        switch (self.payload) {
            .regular => |payload| {
                if (payload.model_path.len == 0) return error.InvalidModelPath;
                if (payload.data_path.len == 0) return error.InvalidDataset;
            },
            .custom_extension => |payload| {
                if (payload.kind.len == 0) return error.InvalidTrainingKind;
                if (payload.bundle_path.len == 0) return error.InvalidBundlePath;
                if (payload.split_dir.len == 0) return error.InvalidDataset;
                if (payload.output_dir.len == 0) return error.InvalidOutputDir;
            },
        }
    }
};

pub fn trainingKindString(kind: TrainingKind) []const u8 {
    return switch (kind) {
        .regular => training_kind_regular,
        .custom_extension => "custom_extension",
    };
}

pub fn parseTrainingKind(value: []const u8) ?TrainingKind {
    if (std.mem.eql(u8, value, training_kind_regular)) return .regular;
    if (std.mem.eql(u8, value, "supervised")) return .regular;
    if (std.mem.eql(u8, value, "nanochat")) return .regular;
    return null;
}

pub fn trainingKindFromRequestBody(allocator: Allocator, body: []const u8) !TrainingKind {
    if (body.len == 0) return .regular;
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidTrainingWorkloadRequest,
    };
    defer parsed.deinit();
    return try trainingKindFromValue(parsed.value);
}

pub fn trainingKindFromValue(value: std.json.Value) anyerror!TrainingKind {
    const object = switch (value) {
        .object => |inner| inner,
        else => return error.InvalidTrainingWorkloadRequest,
    };

    if (objectString(object, "training_kind")) |kind| {
        return parseTrainingKind(kind) orelse error.InvalidTrainingKind;
    }
    if (objectString(object, "model_type")) |kind| {
        return parseTrainingKind(kind) orelse error.InvalidTrainingKind;
    }
    if (object.get("job")) |job_value| {
        return trainingKindFromValue(job_value) catch |err| switch (err) {
            error.InvalidTrainingWorkloadRequest => return .regular,
            else => return err,
        };
    }
    return .regular;
}

fn objectString(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |inner| inner,
        else => null,
    };
}

fn artifactRefsFromPayload(payload: TrainingPayload) ArtifactRefs {
    return switch (payload) {
        .regular => |regular| .{
            .model_path = regular.model_path,
            .checkpoint_dir = regular.checkpoint_dir,
        },
        .custom_extension => |private| .{
            .bundle_path = private.bundle_path,
            .output_dir = private.output_dir,
            .resume_from_dir = private.resume_from_dir,
        },
    };
}

fn datasetRefFromPayload(payload: TrainingPayload) DatasetRef {
    return switch (payload) {
        .regular => |regular| .{ .local_path = regular.data_path },
        .custom_extension => |private| .{ .split_dir = private.split_dir },
    };
}

test "training workload request kind defaults to regular and rejects private extension kinds" {
    try std.testing.expectEqual(TrainingKind.regular, try trainingKindFromRequestBody(std.testing.allocator, ""));
    try std.testing.expectEqual(TrainingKind.regular, try trainingKindFromRequestBody(std.testing.allocator, "{}"));
    try std.testing.expectError(error.InvalidTrainingKind, trainingKindFromRequestBody(std.testing.allocator, "{\"training_kind\":\"private\"}"));
    try std.testing.expectError(error.InvalidTrainingKind, trainingKindFromRequestBody(std.testing.allocator, "{\"model_type\":\"private\"}"));
    try std.testing.expectError(error.InvalidTrainingKind, trainingKindFromRequestBody(std.testing.allocator, "{\"training_kind\":\"unknown\"}"));
}

test "training workload request kind can be nested under job for regular training" {
    try std.testing.expectEqual(
        TrainingKind.regular,
        try trainingKindFromRequestBody(std.testing.allocator, "{\"placement\":{\"workers_required\":1},\"job\":{\"training_kind\":\"regular\"}}"),
    );
}

test "training workload spec preserves decoupled payload boundaries" {
    const embeddings = [_]usize{ 0, 1 };
    const job = decoupled_job.DecoupledTrainingJob{
        .model_id = "model-a",
        .training_kind = .regular,
        .run_id = "run-a",
        .workers = 2,
        .backend = "cpu",
        .target_arch = null,
        .max_steps = 4,
        .outer_loop_steps = 4,
        .num_fragments = 2,
        .sync_interval_h = 2,
        .overlap_tau = 1,
        .min_quorum = 1,
        .merge_strategy = "weighted_average",
        .fragment_strategy = "strided",
        .outer_learning_rate = 0.7,
        .outer_momentum = 0.9,
        .learner_alpha = 0.0,
        .grace_window_ms = 0,
        .grace_gamma = 1.0,
        .max_grace_steps = 1,
        .adaptive_grace_enabled = false,
        .outer_gradient_compression = "none",
        .max_recovery_syncer_steps = 1,
        .embedding_tensor_indices = &embeddings,
        .payload = .{
            .regular = .{
                .model_path = "model-a",
                .data_path = "data.bin",
                .tokenizer = "u16",
                .sampling = "sequential",
                .checkpoint_dir = "out",
            },
        },
    };

    const spec = try TrainingWorkloadSpec.fromDecoupledJob(job);
    try std.testing.expectEqual(TrainingKind.regular, spec.training_kind);
    try std.testing.expectEqualStrings("model-a", spec.artifacts.model_path.?);
    try std.testing.expectEqualStrings("data.bin", spec.dataset.local_path.?);
    try std.testing.expectEqual(@as(usize, 2), spec.decoupled.num_fragments);
}
