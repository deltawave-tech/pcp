const std = @import("std");
pub const data_assignment_mod = @import("../../data/assignment.zig");

pub const field_name = "training_window";
pub const spec_kind = "pcp_training_window_v1";

pub const training_kind_regular = "regular";

pub const output_contract_kind_regular_checkpoint = "regular_checkpoint";

pub const blob_online_weights = "online_weights";
pub const blob_target_weights = "target_weights";
pub const blob_optimizer_state = "optimizer_state";

pub const ProgramArtifactRef = struct {
    name: []const u8,
    artifact_kind: []const u8,
    backend: []const u8,
    target_arch: ?[]const u8 = null,
    source_hash: u64 = 0,
    compile_options_hash: u64 = 0,
    byte_hash: u64 = 0,
    program_key: u64 = 0,
};

pub const BlobRef = data_assignment_mod.BlobRef;

pub const OutputContract = struct {
    kind: []const u8 = output_contract_kind_regular_checkpoint,
    output_dir: []const u8,
    include_weight_delta: bool = false,
    include_paths: bool = true,
};

pub const DecoupledLoopSpec = struct {
    num_fragments: usize,
    sync_interval_h: usize,
    overlap_tau: usize,
    learner_alpha: f32 = 0.0,
    max_syncer_steps: usize = 0,
    max_local_steps: usize = 0,
};

pub const RegularRuntimeDescriptor = struct {
    tokenizer: []const u8 = "char",
    sampling: []const u8 = "random",
    dtype: []const u8 = "f32",
    micro_batch: usize = 0,
};

pub const TrainingWindowSpec = struct {
    kind: []const u8 = spec_kind,
    training_kind: []const u8 = training_kind_regular,
    model_id: ?[]const u8 = null,
    program_artifacts: []const ProgramArtifactRef = &.{},
    data_assignment: data_assignment_mod.DataAssignment,
    local_steps: usize = 0,
    previous_steps_completed: usize = 0,
    output_contract: OutputContract,
    initial_online_weights: ?BlobRef = null,
    target_weights: ?BlobRef = null,
    optimizer_state: ?BlobRef = null,
    outer_optimizer_state: ?BlobRef = null,
    decoupled_loop: ?DecoupledLoopSpec = null,
    regular: ?RegularRuntimeDescriptor = null,
};

pub fn trainingKindFromWindowValue(value: std.json.Value) ?[]const u8 {
    const object = switch (value) {
        .object => |obj| obj,
        else => return null,
    };
    if (object.get("training_kind")) |kind_value| {
        return switch (kind_value) {
            .string => |kind| kind,
            else => null,
        };
    }
    return null;
}

test "training window kind detection supports unified regular windows" {
    var unified = std.json.ObjectMap.init(std.testing.allocator);
    defer unified.deinit();
    try unified.put("kind", .{ .string = spec_kind });
    try unified.put("training_kind", .{ .string = training_kind_regular });
    try std.testing.expectEqualStrings(training_kind_regular, trainingKindFromWindowValue(.{ .object = unified }).?);
}
