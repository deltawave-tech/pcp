const std = @import("std");
const message = @import("../../../network/message.zig");
const message_registry = @import("../../../protocol/message_registry.zig");

const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
pub const family = message_registry.WorkerHandlerFamily.rl;

pub fn ownsMessageType(msg_type: []const u8) bool {
    return message_registry.messageHasWorkerHandler(msg_type, family);
}

pub fn dispatch(worker: anytype, msg: MessageEnvelope) !bool {
    if (std.mem.eql(u8, msg.msg_type, MessageType.START_ROLLOUT)) {
        try worker.handleStartRollout(msg);
        return true;
    }
    if (std.mem.eql(u8, msg.msg_type, MessageType.UPDATE_WEIGHTS)) {
        try worker.handleUpdateWeights(msg);
        return true;
    }
    return false;
}

test "RL handler owns rollout and weight update messages" {
    try std.testing.expect(ownsMessageType(MessageType.START_ROLLOUT));
    try std.testing.expect(ownsMessageType(MessageType.UPDATE_WEIGHTS));
    try std.testing.expect(!ownsMessageType(MessageType.START_GENERATION));
}
