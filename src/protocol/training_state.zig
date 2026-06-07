const std = @import("std");

pub const schema_name = "pcp.training_state.v1";
pub const aggregation_decoupled_diloco = "decoupled_diloco";
pub const training_kind_regular = "regular";

pub const artifact_kind_weights = "weights";
pub const artifact_kind_target_weights = "target_weights";
pub const artifact_kind_inner_optimizer_state = "inner_optimizer_state";
pub const artifact_kind_outer_optimizer_state = "outer_optimizer_state";
pub const artifact_kind_syncer_state = "syncer_state";

pub const ArtifactRef = struct {
    kind: []const u8,
    path: []const u8,
    byte_count: usize = 0,
};

pub const ArtifactRefs = struct {
    online_weights: ArtifactRef,
    target_weights: ?ArtifactRef = null,
    outer_optimizer_state: ?ArtifactRef = null,
    syncer_state: ?ArtifactRef = null,
    inner_optimizer_state: ?ArtifactRef = null,
};

pub const OuterOptimizer = struct {
    kind: []const u8 = "nesterov",
    learning_rate: f32,
    momentum: f32,
};

pub const DecoupledConfig = struct {
    outer_loop_steps: usize,
    syncer_steps: usize,
    num_fragments: usize,
    sync_interval_h: usize,
    overlap_tau: usize,
    min_quorum: usize,
    merge_strategy: []const u8,
    fragment_strategy: []const u8,
    learner_alpha: f32,
    outer_gradient_compression: []const u8,
    max_recovery_syncer_steps: usize,
    grace_window_ms: u64,
    grace_gamma: f64,
    max_grace_steps: usize,
    adaptive_grace_enabled: bool,
};

pub const RuntimeCounters = struct {
    quorum_participants_total: usize,
    grace_inclusions_total: usize,
    skipped_learners_total: usize,
    last_participant_count: usize,
    last_grace_inclusion_count: usize,
    last_skipped_learner_count: usize,
    last_quorum_wait_ms: u64,
    last_grace_window_ms: u64,
    learner_to_syncer_fragment_bytes: usize,
    syncer_to_learner_fragment_bytes: usize,
    vector_clock_entries: usize,
    event_tape_entries: usize,
};

pub const RegularExtension = struct {
    model_path: []const u8,
    data_path: []const u8,
    tokenizer: []const u8,
    sampling: []const u8,
    dtype: []const u8,
    final_loss: f32,
};

pub const ResumeExpectation = struct {
    training_kind: ?[]const u8 = null,
    model_id: ?[]const u8 = null,
    extension_object_key: ?[]const u8 = null,
    extension_stage: ?[]const u8 = null,
};

pub fn readResumeStepsFromFile(
    allocator: std.mem.Allocator,
    path: []const u8,
    expectation: ResumeExpectation,
) !usize {
    const data = try std.fs.cwd().readFileAlloc(allocator, path, 8 * 1024 * 1024);
    defer allocator.free(data);
    return try readResumeStepsFromSlice(allocator, data, expectation);
}

pub fn readResumeStepsFromSlice(
    allocator: std.mem.Allocator,
    data: []const u8,
    expectation: ResumeExpectation,
) !usize {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, data, .{});
    defer parsed.deinit();

    const object = switch (parsed.value) {
        .object => |*object| object,
        else => return error.InvalidResumeTrainingState,
    };
    try validateResumeExpectation(object, expectation);
    return jsonObjectUsize(object, "steps_completed") orelse return error.InvalidResumeTrainingState;
}

fn validateResumeExpectation(object: *const std.json.ObjectMap, expectation: ResumeExpectation) !void {
    if (expectation.training_kind) |expected_kind| {
        if (jsonObjectString(object, "training_kind")) |kind| {
            if (!std.mem.eql(u8, kind, expected_kind)) return error.ResumeTrainingKindMismatch;
        }
    }

    if (expectation.model_id) |expected_model_id| {
        const model_id = jsonObjectString(object, "model_id") orelse jsonObjectString(object, "model_path") orelse return error.InvalidResumeTrainingState;
        if (!std.mem.eql(u8, model_id, expected_model_id)) return error.ResumeModelIdMismatch;
    }

    if (expectation.extension_stage) |expected_stage| {
        const stage = if (expectation.extension_object_key) |key|
            nestedObjectString(object, key, "stage") orelse jsonObjectString(object, "stage") orelse return error.InvalidResumeTrainingState
        else
            jsonObjectString(object, "stage") orelse return error.InvalidResumeTrainingState;
        if (!std.mem.eql(u8, stage, expected_stage)) return error.ResumeStageMismatch;
    }
}

fn nestedObjectString(object: *const std.json.ObjectMap, object_key: []const u8, field_key: []const u8) ?[]const u8 {
    const nested = object.get(object_key) orelse return null;
    const nested_object = switch (nested) {
        .object => |inner| inner,
        else => return null,
    };
    return jsonObjectString(&nested_object, field_key);
}

fn jsonObjectString(object: *const std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |string| string,
        else => null,
    };
}

fn jsonObjectUsize(object: *const std.json.ObjectMap, key: []const u8) ?usize {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |integer| if (integer >= 0) @intCast(integer) else null,
        else => null,
    };
}

test "resume steps parser accepts common regular training state" {
    const data =
        \\{
        \\  "schema": "pcp.training_state.v1",
        \\  "training_kind": "regular",
        \\  "model_id": "nanochat",
        \\  "steps_completed": 7
        \\}
    ;
    const steps = try readResumeStepsFromSlice(std.testing.allocator, data, .{
        .training_kind = training_kind_regular,
        .model_id = "nanochat",
    });
    try std.testing.expectEqual(@as(usize, 7), steps);
}

test "resume steps parser accepts generic extension stage expectations" {
    const data =
        \\{
        \\  "schema": "pcp.training_state.v1",
        \\  "training_kind": "custom_extension",
        \\  "model_id": "extension-model",
        \\  "steps_completed": 11,
        \\  "extension": { "stage": "stage_a" }
        \\}
    ;
    try std.testing.expectEqual(@as(usize, 11), try readResumeStepsFromSlice(std.testing.allocator, data, .{
        .training_kind = "custom_extension",
        .model_id = "extension-model",
        .extension_object_key = "extension",
        .extension_stage = "stage_a",
    }));
}
