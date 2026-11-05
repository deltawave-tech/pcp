const std = @import("std");
const message = @import("../../../network/message.zig");
const message_registry = @import("../../../protocol/message_registry.zig");

const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
pub const family = message_registry.WorkerHandlerFamily.streaming_diloco;

pub fn ownsMessageType(msg_type: []const u8) bool {
    return message_registry.messageHasWorkerHandler(msg_type, family);
}

pub fn dispatch(worker: anytype, msg: MessageEnvelope) !bool {
    if (ownsMessageType(msg.msg_type)) {
        try worker.handleStartStreamingLoop(msg);
        return true;
    }
    return false;
}

test "streaming DiLoCo handler owns streaming-loop starts" {
    try std.testing.expect(ownsMessageType(MessageType.START_STREAMING_LOOP));
    try std.testing.expect(!ownsMessageType(MessageType.START_DECOUPLED_DILOCO_LOOP));
}
