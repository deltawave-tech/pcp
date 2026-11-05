const std = @import("std");
const message = @import("../../network/message.zig");
const message_registry = @import("../../protocol/message_registry.zig");

pub const graph = @import("task_handlers/graph.zig");
pub const transfer = @import("task_handlers/transfer.zig");
pub const regular_training = @import("task_handlers/regular_training.zig");
pub const decoupled_training = @import("task_handlers/decoupled_training.zig");
pub const streaming_diloco = @import("task_handlers/streaming_diloco.zig");
pub const rl = @import("task_handlers/rl.zig");
pub const inference = @import("task_handlers/inference.zig");
pub const lifecycle = @import("task_handlers/lifecycle.zig");

const MessageType = message.MessageType;

pub const TaskFamily = message_registry.WorkerHandlerFamily;
pub const registry = message_registry.worker_handler_registry;

pub fn familyForMessageType(msg_type: []const u8) TaskFamily {
    return message_registry.workerHandlerForMessageType(msg_type);
}

pub fn ownsMessageType(family: TaskFamily, msg_type: []const u8) bool {
    return message_registry.messageHasWorkerHandler(msg_type, family);
}

test "worker task registry maps gateway messages to handler families" {
    try std.testing.expectEqual(TaskFamily.graph, familyForMessageType(MessageType.INITIALIZE_GRAPH));
    try std.testing.expectEqual(TaskFamily.transfer, familyForMessageType(MessageType.WEIGHT_CHUNK));
    try std.testing.expectEqual(TaskFamily.regular_training, familyForMessageType(MessageType.START_INNER_LOOP));
    try std.testing.expectEqual(TaskFamily.decoupled_training, familyForMessageType(MessageType.START_DECOUPLED_DILOCO_LOOP));
    try std.testing.expectEqual(TaskFamily.streaming_diloco, familyForMessageType(MessageType.START_STREAMING_LOOP));
    try std.testing.expectEqual(TaskFamily.rl, familyForMessageType(MessageType.START_ROLLOUT));
    try std.testing.expectEqual(TaskFamily.inference, familyForMessageType(MessageType.START_GENERATION));
    try std.testing.expectEqual(TaskFamily.lifecycle, familyForMessageType(MessageType.SHUTDOWN));
    try std.testing.expectEqual(TaskFamily.unknown, familyForMessageType("pcp.unknown"));
}

test "worker task registry has no duplicate message ownership" {
    for (registry, 0..) |entry, i| {
        for (entry.messages) |msg_type| {
            try std.testing.expectEqual(entry.family, familyForMessageType(msg_type));
        }
        for (registry[i + 1 ..]) |other| {
            for (entry.messages) |left| {
                for (other.messages) |right| {
                    try std.testing.expect(!std.mem.eql(u8, left, right));
                }
            }
        }
    }
}
