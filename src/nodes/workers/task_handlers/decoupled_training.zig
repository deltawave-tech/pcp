const std = @import("std");
const message = @import("../../../network/message.zig");
const message_registry = @import("../../../protocol/message_registry.zig");

const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
pub const family = message_registry.WorkerHandlerFamily.decoupled_training;

pub fn ownsMessageType(msg_type: []const u8) bool {
    return message_registry.messageHasWorkerHandler(msg_type, family);
}

pub fn dispatch(worker: anytype, msg: MessageEnvelope) !bool {
    if (std.mem.eql(u8, msg.msg_type, MessageType.START_DECOUPLED_DILOCO_LOOP)) {
        try worker.handleStartDecoupledDilocoLoop(msg);
        return true;
    }
    if (std.mem.eql(u8, msg.msg_type, MessageType.STOP_DECOUPLED_DILOCO_LOOP)) {
        worker.handleStopDecoupledDilocoLoop(msg);
        return true;
    }
    return false;
}

test "decoupled training handler owns decoupled lifecycle messages" {
    try std.testing.expect(ownsMessageType(MessageType.START_DECOUPLED_DILOCO_LOOP));
    try std.testing.expect(ownsMessageType(MessageType.STOP_DECOUPLED_DILOCO_LOOP));
    try std.testing.expect(!ownsMessageType(MessageType.START_INNER_LOOP));
}
