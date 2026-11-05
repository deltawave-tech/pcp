const std = @import("std");
const message = @import("../../network/message.zig");

pub const ExtensionWorkerState = struct {
    pub fn init(_: std.mem.Allocator) ExtensionWorkerState {
        return .{};
    }

    pub fn deinit(_: *ExtensionWorkerState, _: std.mem.Allocator) void {}
};

pub const TrainingProgressPayload = struct {
    stage: []const u8 = "",
    step: usize = 0,
    total_steps: usize = 0,
    loss: f32 = 0,
    learning_rate: f32 = 0,
    train_examples: usize = 0,
};

pub const TrainingProgressContext = struct {
    worker_ctx: *anyopaque,
    request_id: message.RequestId,
    round_id: message.RoundId = 0,
    task_id: message.TaskId = 0,
    custom_training_round: bool = false,
    custom_decoupled_training: bool = false,
    send_fn: *const fn (*anyopaque, *const TrainingProgressContext, TrainingProgressPayload) anyerror!void,
};

pub fn dispatchTask(_: anytype, _: message.MessageEnvelope) !bool {
    return false;
}

pub fn handleCustomLoadBundle(_: anytype, _: message.MessageEnvelope) !void {
    return error.UnsupportedPrivateExtension;
}

pub fn handleCustomStartInference(_: anytype, _: message.MessageEnvelope) !void {
    return error.UnsupportedPrivateExtension;
}

pub fn handleCustomStartTraining(_: anytype, _: message.MessageEnvelope, _: *TrainingProgressContext, _: anytype) !void {
    return error.UnsupportedPrivateExtension;
}

pub fn handleCustomStartTrainingRound(_: anytype, _: message.MessageEnvelope, _: *TrainingProgressContext, _: anytype) !void {
    return error.UnsupportedPrivateExtension;
}

pub fn handleCustomStartDecoupledTrainingLoop(_: anytype, _: message.MessageEnvelope, _: *TrainingProgressContext, _: anytype) !void {
    return error.UnsupportedPrivateExtension;
}

pub fn emitCustomTrainingProgressHook(_: *anyopaque, _: anytype) anyerror!void {}

pub fn sendCustomExtensionTrainingProgress(_: anytype, _: *const TrainingProgressContext, _: TrainingProgressPayload) !void {}

pub fn dispatchDecoupledTrainingKind(_: anytype, _: message.MessageEnvelope, _: []const u8) !bool {
    return false;
}
