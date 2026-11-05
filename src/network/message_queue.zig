const std = @import("std");
const message = @import("message.zig");
const protocol_limits = @import("../protocol/limits.zig");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const MessageEnvelope = message.MessageEnvelope;
const MessageFilter = message.MessageFilter;
const NodeId = message.NodeId;

comptime {
    std.debug.assert(protocol_limits.worker_count_max <= std.math.maxInt(NodeId));
    std.debug.assert(protocol_limits.message_queue_depth_max > 0);
}

pub const DrainOptions = struct {
    max_count: usize = std.math.maxInt(usize),
    sender_node: ?NodeId = null,
};

pub fn countMatching(queue: []const MessageEnvelope, filter: MessageFilter, sender_node: ?NodeId) usize {
    var count: usize = 0;
    for (queue) |msg| {
        if (sender_node) |expected_sender| {
            if (msg.sender_node != expected_sender) continue;
        }
        if (filter.matches(msg)) count += 1;
    }
    return count;
}

pub fn countUniqueSenders(allocator: Allocator, queue: []const MessageEnvelope, filter: MessageFilter) !usize {
    var seen_senders = std.AutoHashMap(NodeId, void).init(allocator);
    defer seen_senders.deinit();

    for (queue) |msg| {
        if (!filter.matches(msg)) continue;
        try seen_senders.put(msg.sender_node, {});
    }

    return seen_senders.count();
}

pub fn countUniqueSendersBounded(queue: []const MessageEnvelope, filter: MessageFilter, expected_max: usize) usize {
    std.debug.assert(expected_max <= protocol_limits.worker_count_max);

    var seen_senders = std.StaticBitSet(std.math.maxInt(NodeId) + 1).initEmpty();
    var unique_count: usize = 0;
    for (queue) |msg| {
        if (!filter.matches(msg)) continue;
        const sender_index: usize = msg.sender_node;
        if (seen_senders.isSet(sender_index)) continue;
        seen_senders.set(sender_index);
        unique_count += 1;
        if (unique_count == expected_max) break;
    }

    std.debug.assert(unique_count <= expected_max);
    return unique_count;
}

pub fn drainMatching(
    allocator: Allocator,
    queue: *ArrayList(MessageEnvelope),
    filter: MessageFilter,
    options: DrainOptions,
) !ArrayList(MessageEnvelope) {
    var drained = ArrayList(MessageEnvelope).init(allocator);
    if (options.max_count == 0) return drained;
    if (options.max_count != std.math.maxInt(usize)) {
        std.debug.assert(options.max_count <= protocol_limits.message_queue_depth_max);
    }

    var idx: usize = 0;
    while (idx < queue.items.len and drained.items.len < options.max_count) {
        const msg = queue.items[idx];
        if (options.sender_node) |expected_sender| {
            if (msg.sender_node != expected_sender) {
                idx += 1;
                continue;
            }
        }
        if (!filter.matches(msg)) {
            idx += 1;
            continue;
        }

        try drained.append(msg);
        _ = queue.orderedRemove(idx);
    }

    return drained;
}

fn testMsg(sender: NodeId, msg_type: []const u8, request_id: message.RequestId, task_id: message.TaskId) MessageEnvelope {
    return .{
        .recipient_node = 0,
        .recipient_service = "controller",
        .sender_node = sender,
        .sender_service = "worker",
        .msg_type = msg_type,
        .msg_id = task_id,
        .request_id = request_id,
        .round_id = 0,
        .task_id = task_id,
        .data = .null,
    };
}

test "drainMatching collects multiple messages from the same worker and preserves unrelated messages" {
    const allocator = std.testing.allocator;
    var queue = ArrayList(MessageEnvelope).init(allocator);
    defer queue.deinit();

    try queue.append(testMsg(1, message.MessageType.DECOUPLED_LEARNER_METADATA, 7, 1));
    try queue.append(testMsg(1, message.MessageType.HEARTBEAT, 7, 2));
    try queue.append(testMsg(1, message.MessageType.DECOUPLED_LEARNER_METADATA, 7, 3));
    try queue.append(testMsg(2, message.MessageType.DECOUPLED_LEARNER_METADATA, 8, 4));

    var drained = try drainMatching(
        allocator,
        &queue,
        .{ .msg_type = message.MessageType.DECOUPLED_LEARNER_METADATA, .request_id = 7 },
        .{ .sender_node = 1 },
    );
    defer drained.deinit();

    try std.testing.expectEqual(@as(usize, 2), drained.items.len);
    try std.testing.expectEqual(@as(message.TaskId, 1), drained.items[0].task_id);
    try std.testing.expectEqual(@as(message.TaskId, 3), drained.items[1].task_id);
    try std.testing.expectEqual(@as(usize, 2), queue.items.len);
    try std.testing.expectEqualStrings(message.MessageType.HEARTBEAT, queue.items[0].msg_type);
    try std.testing.expectEqual(@as(NodeId, 2), queue.items[1].sender_node);
}

test "drainMatching honors max count and FIFO order" {
    const allocator = std.testing.allocator;
    var queue = ArrayList(MessageEnvelope).init(allocator);
    defer queue.deinit();

    try queue.append(testMsg(1, message.MessageType.DECOUPLED_FRAGMENT_UPDATE, 9, 1));
    try queue.append(testMsg(1, message.MessageType.DECOUPLED_FRAGMENT_UPDATE, 9, 2));
    try queue.append(testMsg(1, message.MessageType.DECOUPLED_FRAGMENT_UPDATE, 9, 3));

    var drained = try drainMatching(
        allocator,
        &queue,
        .{ .msg_type = message.MessageType.DECOUPLED_FRAGMENT_UPDATE, .request_id = 9 },
        .{ .max_count = 2 },
    );
    defer drained.deinit();

    try std.testing.expectEqual(@as(usize, 2), drained.items.len);
    try std.testing.expectEqual(@as(message.TaskId, 1), drained.items[0].task_id);
    try std.testing.expectEqual(@as(message.TaskId, 2), drained.items[1].task_id);
    try std.testing.expectEqual(@as(usize, 1), queue.items.len);
    try std.testing.expectEqual(@as(message.TaskId, 3), queue.items[0].task_id);
}

test "countUniqueSenders counts quorum senders without consuming messages" {
    const allocator = std.testing.allocator;
    var queue = ArrayList(MessageEnvelope).init(allocator);
    defer queue.deinit();

    try queue.append(testMsg(1, message.MessageType.DECOUPLED_LEARNER_METADATA, 11, 1));
    try queue.append(testMsg(1, message.MessageType.DECOUPLED_LEARNER_METADATA, 11, 2));
    try queue.append(testMsg(2, message.MessageType.DECOUPLED_LEARNER_METADATA, 11, 3));
    try queue.append(testMsg(3, message.MessageType.DECOUPLED_LEARNER_METADATA, 12, 4));

    const unique = try countUniqueSenders(
        allocator,
        queue.items,
        .{ .msg_type = message.MessageType.DECOUPLED_LEARNER_METADATA, .request_id = 11 },
    );

    try std.testing.expectEqual(@as(usize, 2), unique);
    try std.testing.expectEqual(@as(usize, 4), queue.items.len);
}

test "countUniqueSendersBounded counts quorum senders without heap allocation" {
    const allocator = std.testing.allocator;
    var queue = ArrayList(MessageEnvelope).init(allocator);
    defer queue.deinit();

    try queue.append(testMsg(1, message.MessageType.DECOUPLED_LEARNER_METADATA, 11, 1));
    try queue.append(testMsg(1, message.MessageType.DECOUPLED_LEARNER_METADATA, 11, 2));
    try queue.append(testMsg(2, message.MessageType.DECOUPLED_LEARNER_METADATA, 11, 3));
    try queue.append(testMsg(3, message.MessageType.DECOUPLED_LEARNER_METADATA, 11, 4));

    const unique = countUniqueSendersBounded(
        queue.items,
        .{ .msg_type = message.MessageType.DECOUPLED_LEARNER_METADATA, .request_id = 11 },
        2,
    );

    try std.testing.expectEqual(@as(usize, 2), unique);
    try std.testing.expectEqual(@as(usize, 4), queue.items.len);
}
