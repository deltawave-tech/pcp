const std = @import("std");
const decoupled_job = @import("../../protocol/decoupled_job.zig");

pub const training_kind_custom_extension = "";
pub const capability_custom_extension = "";
pub const route_custom_internal_training_jobs = "";

pub fn parseTrainingKind(_: []const u8) ?decoupled_job.TrainingKind {
    return null;
}

pub fn trainingKindFromRequestBody(_: std.mem.Allocator, _: []const u8) !decoupled_job.TrainingKind {
    return .regular;
}

pub fn trainingKindFromValue(_: std.json.Value) anyerror!decoupled_job.TrainingKind {
    return .regular;
}
