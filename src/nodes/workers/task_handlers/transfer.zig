const std = @import("std");
const message = @import("../../../network/message.zig");
const message_registry = @import("../../../protocol/message_registry.zig");

const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
pub const family = message_registry.WorkerHandlerFamily.transfer;

pub fn ownsMessageType(msg_type: []const u8) bool {
    return message_registry.messageHasWorkerHandler(msg_type, family);
}

pub fn dispatch(worker: anytype, msg: MessageEnvelope) !bool {
    if (ownsMessageType(msg.msg_type)) {
        try worker.handleWeightChunk(msg);
        return true;
    }
    return false;
}

test "transfer handler owns request scoped blob transfer" {
    try std.testing.expect(ownsMessageType(MessageType.WEIGHT_CHUNK));
    try std.testing.expect(!ownsMessageType(MessageType.INITIALIZE_GRAPH));
}
