const std = @import("std");
const message = @import("../network/message.zig");
const custom_extensions = @import("custom_extensions.zig");
pub const types = @import("message_registry_types.zig");

const MessageType = message.MessageType;

pub const MessageFamily = types.MessageFamily;
pub const MessageDirection = types.MessageDirection;
pub const WorkerHandlerFamily = types.WorkerHandlerFamily;
pub const MessageEntry = types.MessageEntry;
pub const MessageFamilyCapability = types.MessageFamilyCapability;
pub const FederationOperation = types.FederationOperation;
pub const WorkerHandlerRegistration = types.WorkerHandlerRegistration;

pub const capability_worker_control = "message_family.worker_control";
pub const capability_artifact_transfer = "message_family.artifact_transfer";
pub const capability_regular_training = "message_family.regular_training";
pub const capability_decoupled_training = "message_family.decoupled_training";
pub const capability_streaming_diloco = "message_family.streaming_diloco";
pub const capability_inference = "message_family.inference";
pub const capability_rl = "message_family.rl";
pub const capability_custom_extension = custom_extensions.capability_custom_extension;
pub const capability_federation = "message_family.federation";

pub const public_family_capability_registry = [_]MessageFamilyCapability{
    .{ .family = .worker_control, .capability = capability_worker_control },
    .{ .family = .artifact_transfer, .capability = capability_artifact_transfer },
    .{ .family = .regular_training, .capability = capability_regular_training },
    .{ .family = .decoupled_training, .capability = capability_decoupled_training },
    .{ .family = .streaming_diloco, .capability = capability_streaming_diloco },
    .{ .family = .inference, .capability = capability_inference },
    .{ .family = .rl, .capability = capability_rl },
    .{ .family = .federation, .capability = capability_federation },
};

pub const family_capability_registry = public_family_capability_registry ++ custom_extensions.family_capabilities;

pub const graph_worker_messages = [_][]const u8{
    MessageType.INITIALIZE_GRAPH,
};

pub const transfer_worker_messages = [_][]const u8{
    MessageType.WEIGHT_CHUNK,
};

pub const regular_training_worker_messages = [_][]const u8{
    MessageType.START_INNER_LOOP,
};

pub const decoupled_training_worker_messages = [_][]const u8{
    MessageType.START_DECOUPLED_DILOCO_LOOP,
    MessageType.STOP_DECOUPLED_DILOCO_LOOP,
};

pub const streaming_diloco_worker_messages = [_][]const u8{
    MessageType.START_STREAMING_LOOP,
};

pub const rl_worker_messages = [_][]const u8{
    MessageType.START_ROLLOUT,
    MessageType.UPDATE_WEIGHTS,
};

pub const inference_worker_messages = [_][]const u8{
    MessageType.LOAD_MODEL,
    MessageType.START_GENERATION,
    MessageType.CANCEL_GENERATION,
    MessageType.FLUSH_SESSION,
};

pub const custom_extension_worker_messages = custom_extensions.custom_extension_worker_messages;

pub const lifecycle_worker_messages = [_][]const u8{
    MessageType.SHUTDOWN,
};

pub const public_worker_handler_registry = [_]WorkerHandlerRegistration{
    .{ .family = .graph, .messages = &graph_worker_messages },
    .{ .family = .transfer, .messages = &transfer_worker_messages },
    .{ .family = .regular_training, .messages = &regular_training_worker_messages },
    .{ .family = .decoupled_training, .messages = &decoupled_training_worker_messages },
    .{ .family = .streaming_diloco, .messages = &streaming_diloco_worker_messages },
    .{ .family = .rl, .messages = &rl_worker_messages },
    .{ .family = .inference, .messages = &inference_worker_messages },
    .{ .family = .lifecycle, .messages = &lifecycle_worker_messages },
};

pub const worker_handler_registry = public_worker_handler_registry ++ custom_extensions.worker_handler_registry;

pub const public_protocol_messages = [_]MessageEntry{
    .{ .msg_type = MessageType.JOIN_REQUEST, .family = .worker_control, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.JOIN_ACCEPT, .family = .worker_control, .direction = .controller_to_worker },
    .{ .msg_type = MessageType.INITIALIZE_GRAPH, .family = .worker_control, .direction = .controller_to_worker, .worker_handler = .graph },
    .{ .msg_type = MessageType.INITIALIZE_GRAPH_COMPLETE, .family = .worker_control, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.SHUTDOWN, .family = .worker_control, .direction = .controller_to_worker, .worker_handler = .lifecycle },
    .{ .msg_type = MessageType.HEARTBEAT, .family = .worker_control, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.SUPERVISOR_HANDSHAKE, .family = .worker_control, .direction = .supervisor_to_controller },
    .{ .msg_type = MessageType.RESTART_WORKER, .family = .worker_control, .direction = .controller_to_supervisor },

    .{ .msg_type = MessageType.WEIGHT_CHUNK, .family = .artifact_transfer, .direction = .controller_to_worker, .worker_handler = .transfer },
    .{ .msg_type = MessageType.UPDATE_CHUNK, .family = .artifact_transfer, .direction = .worker_to_controller },

    .{ .msg_type = MessageType.START_INNER_LOOP, .family = .regular_training, .direction = .controller_to_worker, .worker_handler = .regular_training },
    .{ .msg_type = MessageType.INNER_LOOP_COMPLETE, .family = .regular_training, .direction = .worker_to_controller },

    .{ .msg_type = MessageType.START_DECOUPLED_DILOCO_LOOP, .family = .decoupled_training, .direction = .controller_to_worker, .worker_handler = .decoupled_training },
    .{ .msg_type = MessageType.DECOUPLED_LEARNER_METADATA, .family = .decoupled_training, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.DECOUPLED_FRAGMENT_PULL, .family = .decoupled_training, .direction = .controller_to_worker },
    .{ .msg_type = MessageType.DECOUPLED_FRAGMENT_UPDATE, .family = .decoupled_training, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.DECOUPLED_FRAGMENT_READY, .family = .decoupled_training, .direction = .controller_to_worker },
    .{ .msg_type = MessageType.STOP_DECOUPLED_DILOCO_LOOP, .family = .decoupled_training, .direction = .controller_to_worker, .worker_handler = .decoupled_training },

    .{ .msg_type = MessageType.START_STREAMING_LOOP, .family = .streaming_diloco, .direction = .controller_to_worker, .worker_handler = .streaming_diloco },
    .{ .msg_type = MessageType.FRAGMENT_UPDATE, .family = .streaming_diloco, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.FRAGMENT_READY, .family = .streaming_diloco, .direction = .controller_to_worker },

    .{ .msg_type = MessageType.LOAD_MODEL, .family = .inference, .direction = .controller_to_worker, .worker_handler = .inference },
    .{ .msg_type = MessageType.MODEL_READY, .family = .inference, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.START_GENERATION, .family = .inference, .direction = .controller_to_worker, .worker_handler = .inference },
    .{ .msg_type = MessageType.GENERATION_CHUNK, .family = .inference, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.GENERATION_COMPLETE, .family = .inference, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.GENERATION_ERROR, .family = .inference, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.CANCEL_GENERATION, .family = .inference, .direction = .controller_to_worker, .worker_handler = .inference },
    .{ .msg_type = MessageType.FLUSH_SESSION, .family = .inference, .direction = .controller_to_worker, .worker_handler = .inference },

    .{ .msg_type = MessageType.START_ROLLOUT, .family = .rl, .direction = .controller_to_worker, .worker_handler = .rl },
    .{ .msg_type = MessageType.ROLLOUT_COMPLETE, .family = .rl, .direction = .worker_to_controller },
    .{ .msg_type = MessageType.UPDATE_WEIGHTS, .family = .rl, .direction = .controller_to_worker, .worker_handler = .rl },
};

pub const protocol_messages = public_protocol_messages ++ custom_extensions.protocol_messages;

pub const federation_operations = [_]FederationOperation{
    .{ .name = "gateway_connect", .method = "POST", .path_pattern = "/v1/federation/connect", .direction = .gateway_to_hub },
    .{ .name = "mutation_batch", .method = "POST", .path_pattern = "/v1/federation/mutations", .direction = .gateway_to_hub },
    .{ .name = "global_training_job", .method = "POST", .path_pattern = "/v1/internal/federation/training/jobs", .direction = .hub_to_gateway },
    .{ .name = "global_training_reservation", .method = "POST", .path_pattern = "/v1/internal/federation/training/reservations", .direction = .hub_to_gateway },
    .{ .name = "global_rl_job", .method = "POST", .path_pattern = "/v1/internal/federation/rl/jobs", .direction = .hub_to_gateway },
    .{ .name = "global_rl_reservation", .method = "POST", .path_pattern = "/v1/internal/federation/rl/reservations", .direction = .hub_to_gateway },
    .{ .name = "reservation_commit", .method = "POST", .path_pattern = "/v1/internal/federation/reservations/:id/commit", .direction = .hub_to_gateway },
    .{ .name = "reservation_release", .method = "POST", .path_pattern = "/v1/internal/federation/reservations/:id/release", .direction = .hub_to_gateway },
    .{ .name = "global_job_cancel", .method = "POST", .path_pattern = "/v1/internal/federation/jobs/:id/cancel", .direction = .hub_to_gateway },
    .{ .name = "global_job_status", .method = "GET", .path_pattern = "/v1/internal/federation/jobs/:id", .direction = .hub_to_gateway },
};

pub fn entryForMessageType(msg_type: []const u8) ?MessageEntry {
    inline for (protocol_messages) |entry| {
        if (std.mem.eql(u8, msg_type, entry.msg_type)) return entry;
    }
    return null;
}

pub fn familyForMessageType(msg_type: []const u8) MessageFamily {
    return if (entryForMessageType(msg_type)) |entry| entry.family else .unknown;
}

pub fn workerHandlerForMessageType(msg_type: []const u8) WorkerHandlerFamily {
    return if (entryForMessageType(msg_type)) |entry| entry.worker_handler else .unknown;
}

pub fn messageHasFamily(msg_type: []const u8, family: MessageFamily) bool {
    return familyForMessageType(msg_type) == family;
}

pub fn messageHasWorkerHandler(msg_type: []const u8, family: WorkerHandlerFamily) bool {
    return workerHandlerForMessageType(msg_type) == family;
}

pub fn capabilityForFamily(family: MessageFamily) ?[]const u8 {
    inline for (family_capability_registry) |entry| {
        if (entry.family == family) return entry.capability;
    }
    return null;
}

pub fn isGatewayQueuedWorkerResultMessage(msg_type: []const u8) bool {
    if (std.mem.eql(u8, msg_type, MessageType.INNER_LOOP_COMPLETE)) return true;
    if (custom_extensions.isGatewayQueuedWorkerResultMessage(msg_type)) return true;
    if (std.mem.eql(u8, msg_type, MessageType.ROLLOUT_COMPLETE)) return true;
    if (std.mem.eql(u8, msg_type, MessageType.FRAGMENT_UPDATE)) return true;
    if (std.mem.eql(u8, msg_type, MessageType.DECOUPLED_LEARNER_METADATA)) return true;
    if (std.mem.eql(u8, msg_type, MessageType.DECOUPLED_FRAGMENT_UPDATE)) return true;
    return false;
}

pub fn isDecoupledWorkerResultMessage(msg_type: []const u8) bool {
    return std.mem.eql(u8, msg_type, MessageType.DECOUPLED_LEARNER_METADATA) or
        std.mem.eql(u8, msg_type, MessageType.DECOUPLED_FRAGMENT_UPDATE);
}

pub const public_message_types = [_][]const u8{
    MessageType.JOIN_REQUEST,
    MessageType.JOIN_ACCEPT,
    MessageType.INITIALIZE_GRAPH,
    MessageType.START_INNER_LOOP,
    MessageType.INNER_LOOP_COMPLETE,
    MessageType.SHUTDOWN,
    MessageType.HEARTBEAT,
    MessageType.SUPERVISOR_HANDSHAKE,
    MessageType.RESTART_WORKER,
    MessageType.START_ROLLOUT,
    MessageType.ROLLOUT_COMPLETE,
    MessageType.UPDATE_WEIGHTS,
    MessageType.WEIGHT_CHUNK,
    MessageType.UPDATE_CHUNK,
    MessageType.START_STREAMING_LOOP,
    MessageType.FRAGMENT_UPDATE,
    MessageType.FRAGMENT_READY,
    MessageType.START_DECOUPLED_DILOCO_LOOP,
    MessageType.DECOUPLED_LEARNER_METADATA,
    MessageType.DECOUPLED_FRAGMENT_PULL,
    MessageType.DECOUPLED_FRAGMENT_UPDATE,
    MessageType.DECOUPLED_FRAGMENT_READY,
    MessageType.STOP_DECOUPLED_DILOCO_LOOP,
    MessageType.LOAD_MODEL,
    MessageType.MODEL_READY,
    MessageType.INITIALIZE_GRAPH_COMPLETE,
    MessageType.START_GENERATION,
    MessageType.GENERATION_CHUNK,
    MessageType.GENERATION_COMPLETE,
    MessageType.GENERATION_ERROR,
    MessageType.CANCEL_GENERATION,
    MessageType.FLUSH_SESSION,
};

pub const all_message_types = public_message_types ++ custom_extensions.all_message_types;

fn declaredWorkerHandlerForMessage(msg_type: []const u8) WorkerHandlerFamily {
    for (worker_handler_registry) |registration| {
        for (registration.messages) |registered_type| {
            if (std.mem.eql(u8, msg_type, registered_type)) return registration.family;
        }
    }
    return .unknown;
}

test "protocol message registry covers every MessageType constant" {
    try std.testing.expectEqual(all_message_types.len, protocol_messages.len);
    for (all_message_types) |msg_type| {
        const entry = entryForMessageType(msg_type) orelse return error.MissingMessageRegistryEntry;
        try std.testing.expect(entry.family != .unknown);
    }
}

test "protocol message registry has no duplicate message types" {
    for (protocol_messages, 0..) |entry, i| {
        for (protocol_messages[i + 1 ..]) |other| {
            try std.testing.expect(!std.mem.eql(u8, entry.msg_type, other.msg_type));
        }
    }
}

test "worker handler declarations match protocol registry" {
    for (worker_handler_registry) |registration| {
        try std.testing.expect(registration.family != .unknown);
        for (registration.messages) |msg_type| {
            try std.testing.expectEqual(registration.family, workerHandlerForMessageType(msg_type));
        }
    }

    for (protocol_messages) |entry| {
        if (entry.worker_handler == .unknown) continue;
        try std.testing.expectEqual(entry.worker_handler, declaredWorkerHandlerForMessage(entry.msg_type));
    }
}

test "message families classify distributed training and serving surfaces" {
    try std.testing.expect(messageHasFamily(MessageType.START_INNER_LOOP, .regular_training));
    try std.testing.expect(messageHasFamily(MessageType.DECOUPLED_FRAGMENT_UPDATE, .decoupled_training));
    try std.testing.expect(messageHasFamily(MessageType.FRAGMENT_READY, .streaming_diloco));
    try std.testing.expect(messageHasFamily(MessageType.START_GENERATION, .inference));
    try std.testing.expect(messageHasFamily(MessageType.START_ROLLOUT, .rl));
    try std.testing.expect(messageHasFamily(MessageType.SUPERVISOR_HANDSHAKE, .worker_control));
    try std.testing.expectEqual(MessageFamily.unknown, familyForMessageType("pcp.unknown"));
}

test "message family capabilities cover protocol families" {
    try std.testing.expectEqualStrings(capability_worker_control, capabilityForFamily(.worker_control).?);
    try std.testing.expectEqualStrings(capability_artifact_transfer, capabilityForFamily(.artifact_transfer).?);
    try std.testing.expectEqualStrings(capability_regular_training, capabilityForFamily(.regular_training).?);
    try std.testing.expectEqualStrings(capability_decoupled_training, capabilityForFamily(.decoupled_training).?);
    try std.testing.expectEqualStrings(capability_streaming_diloco, capabilityForFamily(.streaming_diloco).?);
    try std.testing.expectEqualStrings(capability_inference, capabilityForFamily(.inference).?);
    try std.testing.expectEqualStrings(capability_rl, capabilityForFamily(.rl).?);
    if (capabilityForFamily(.custom_extension)) |capability| {
        try std.testing.expectEqualStrings(capability_custom_extension, capability);
    }
    try std.testing.expectEqualStrings(capability_federation, capabilityForFamily(.federation).?);
    try std.testing.expectEqual(@as(?[]const u8, null), capabilityForFamily(.unknown));
}

test "gateway queued worker-result classification is explicit" {
    try std.testing.expect(isGatewayQueuedWorkerResultMessage(MessageType.INNER_LOOP_COMPLETE));
    try std.testing.expect(isGatewayQueuedWorkerResultMessage(MessageType.ROLLOUT_COMPLETE));
    try std.testing.expect(isGatewayQueuedWorkerResultMessage(MessageType.FRAGMENT_UPDATE));
    try std.testing.expect(isGatewayQueuedWorkerResultMessage(MessageType.DECOUPLED_LEARNER_METADATA));
    try std.testing.expect(!isGatewayQueuedWorkerResultMessage(MessageType.UPDATE_CHUNK));
    try std.testing.expect(!isGatewayQueuedWorkerResultMessage(MessageType.GENERATION_COMPLETE));
}

test "federation coordination operations are registered separately from worker messages" {
    try std.testing.expect(federation_operations.len > 0);
    for (federation_operations) |operation| {
        try std.testing.expect(operation.direction == .gateway_to_hub or operation.direction == .hub_to_gateway);
        try std.testing.expect(operation.method.len > 0);
        try std.testing.expect(operation.path_pattern.len > 0);
    }
}
