const std = @import("std");
const message = @import("../../../network/message.zig");
const message_registry = @import("../../../protocol/message_registry.zig");

const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
pub const family = message_registry.WorkerHandlerFamily.lifecycle;

pub fn ownsMessageType(msg_type: []const u8) bool {
    return message_registry.messageHasWorkerHandler(msg_type, family);
}

pub fn dispatch(worker: anytype, msg: MessageEnvelope) !bool {
    if (ownsMessageType(msg.msg_type)) {
        worker.handleShutdown(msg);
        return true;
    }
    return false;
}

test "lifecycle handler owns shutdown" {
    try std.testing.expect(ownsMessageType(MessageType.SHUTDOWN));
    try std.testing.expect(!ownsMessageType(MessageType.WEIGHT_CHUNK));
}
