const std = @import("std");
const message = @import("../../../network/message.zig");
const message_registry = @import("../../../protocol/message_registry.zig");

const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
pub const family = message_registry.WorkerHandlerFamily.inference;

pub fn ownsMessageType(msg_type: []const u8) bool {
    return message_registry.messageHasWorkerHandler(msg_type, family);
}

pub fn dispatch(worker: anytype, msg: MessageEnvelope) !bool {
    if (std.mem.eql(u8, msg.msg_type, MessageType.LOAD_MODEL)) {
        try worker.handleLoadModel(msg);
        return true;
    }
    if (std.mem.eql(u8, msg.msg_type, MessageType.START_GENERATION)) {
        try worker.handleStartGeneration(msg);
        return true;
    }
    if (std.mem.eql(u8, msg.msg_type, MessageType.CANCEL_GENERATION)) {
        worker.handleCancelGeneration(msg);
        return true;
    }
    if (std.mem.eql(u8, msg.msg_type, MessageType.FLUSH_SESSION)) {
        worker.handleFlushSession(msg);
        return true;
    }
    return false;
}

test "inference handler owns generation lifecycle messages" {
    try std.testing.expect(ownsMessageType(MessageType.LOAD_MODEL));
    try std.testing.expect(ownsMessageType(MessageType.START_GENERATION));
    try std.testing.expect(ownsMessageType(MessageType.CANCEL_GENERATION));
    try std.testing.expect(ownsMessageType(MessageType.FLUSH_SESSION));
    try std.testing.expect(!ownsMessageType(MessageType.START_ROLLOUT));
}
