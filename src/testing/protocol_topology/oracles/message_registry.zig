const std = @import("std");

const message = @import("../../../network/message.zig");
const message_registry = @import("../../../protocol/message_registry.zig");
const topology = @import("../topology.zig");

const MessageType = message.MessageType;
pub const WorkerHandlerFamily = message_registry.WorkerHandlerFamily;

pub const DispatchCase = enum {
    decoupled_start,
    inference_start,
    decoupled_result,
    unknown_message,
};

pub const DispatchOutput = enum {
    accept_decoupled,
    accept_inference,
    reject,
};

pub const Violation = enum {
    none,
    unknown_message,
    no_worker_handler,
    wrong_worker_handler,
    wrong_direction,
    duplicate_worker_handler_owner,
    worker_handler_not_declared,
};

pub const MessageRegistryTopologyTask = struct {
    input: topology.Complex,
    protocol: topology.Complex,
    output: topology.Complex,
    carrier: topology.CarrierMap,
    decisions: []const topology.DecisionEdge,
};

const decoupled_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = MessageType.START_DECOUPLED_DILOCO_LOOP } },
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "handler.decoupled_training" } },
};
const inference_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = MessageType.START_GENERATION } },
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "handler.inference" } },
};
const decoupled_result_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = MessageType.DECOUPLED_FRAGMENT_UPDATE } },
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = "controller.result_queue" } },
};
const unknown_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = "pcp.unknown" } },
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "handler.unknown" } },
};

const decoupled_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "dispatch.decoupled_training" } },
};
const inference_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "dispatch.inference" } },
};
const result_reject_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "dispatch.reject.worker_result" } },
};
const unknown_reject_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "dispatch.reject.unknown" } },
};

const decoupled_output_vertices = [_]topology.Vertex{
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "accepted.decoupled_training" } },
};
const inference_output_vertices = [_]topology.Vertex{
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "accepted.inference" } },
};
const rejected_output_vertices = [_]topology.Vertex{
    .{ .color = .{ .worker = 0 }, .label = .{ .symbol = "rejected" } },
};

const input_simplexes = [_]topology.Simplex{
    .{ .vertices = &decoupled_input_vertices },
    .{ .vertices = &inference_input_vertices },
    .{ .vertices = &decoupled_result_input_vertices },
    .{ .vertices = &unknown_input_vertices },
};
const protocol_simplexes = [_]topology.Simplex{
    .{ .vertices = &decoupled_protocol_vertices },
    .{ .vertices = &inference_protocol_vertices },
    .{ .vertices = &result_reject_protocol_vertices },
    .{ .vertices = &unknown_reject_protocol_vertices },
};
const output_simplexes = [_]topology.Simplex{
    .{ .vertices = &decoupled_output_vertices },
    .{ .vertices = &inference_output_vertices },
    .{ .vertices = &rejected_output_vertices },
};

const decoupled_outputs = [_]usize{0};
const inference_outputs = [_]usize{1};
const rejected_outputs = [_]usize{2};
const carrier_edges = [_]topology.CarrierEdge{
    .{ .input_index = 0, .output_indices = &decoupled_outputs },
    .{ .input_index = 1, .output_indices = &inference_outputs },
    .{ .input_index = 2, .output_indices = &rejected_outputs },
    .{ .input_index = 3, .output_indices = &rejected_outputs },
};
const default_decisions = [_]topology.DecisionEdge{
    .{ .input_index = 0, .protocol_index = 0, .output_index = 0 },
    .{ .input_index = 1, .protocol_index = 1, .output_index = 1 },
    .{ .input_index = 2, .protocol_index = 2, .output_index = 2 },
    .{ .input_index = 3, .protocol_index = 3, .output_index = 2 },
};

pub fn topologyTask() MessageRegistryTopologyTask {
    const input = topology.Complex{ .simplexes = &input_simplexes };
    const output = topology.Complex{ .simplexes = &output_simplexes };
    return .{
        .input = input,
        .protocol = .{ .simplexes = &protocol_simplexes },
        .output = output,
        .carrier = .{
            .input = input,
            .output = output,
            .carries = &carrier_edges,
        },
        .decisions = &default_decisions,
    };
}

pub fn inputIndex(case: DispatchCase) usize {
    return switch (case) {
        .decoupled_start => 0,
        .inference_start => 1,
        .decoupled_result => 2,
        .unknown_message => 3,
    };
}

pub fn protocolIndex(case: DispatchCase) usize {
    return inputIndex(case);
}

pub fn outputIndex(output: DispatchOutput) usize {
    return switch (output) {
        .accept_decoupled => 0,
        .accept_inference => 1,
        .reject => 2,
    };
}

pub fn checkWorkerHandlerAcceptance(msg_type: []const u8, claimed_handler: WorkerHandlerFamily) Violation {
    const entry = message_registry.entryForMessageType(msg_type) orelse return .unknown_message;
    if (entry.worker_handler == .unknown) return .no_worker_handler;
    if (entry.worker_handler != claimed_handler) return .wrong_worker_handler;
    if (entry.direction != .controller_to_worker) return .wrong_direction;
    return .none;
}

pub fn checkTopologyDecision(
    allocator: std.mem.Allocator,
    dispatch_case: DispatchCase,
    output: DispatchOutput,
) !void {
    const task = topologyTask();
    const protocol_simplex = [_]topology.Simplex{
        task.protocol.simplexes[protocolIndex(dispatch_case)],
    };
    const decisions = [_]topology.DecisionEdge{.{
        .input_index = inputIndex(dispatch_case),
        .protocol_index = 0,
        .output_index = outputIndex(output),
    }};

    try topology.validateDecisionMap(
        allocator,
        .{ .simplexes = &protocol_simplex },
        task.carrier,
        &decisions,
    );
}

pub fn checkWorkerHandlerRegistrySync() Violation {
    for (message_registry.worker_handler_registry, 0..) |registration, registration_index| {
        for (registration.messages) |msg_type| {
            if (checkWorkerHandlerAcceptance(msg_type, registration.family) != .none) {
                return .worker_handler_not_declared;
            }
            for (message_registry.worker_handler_registry[registration_index + 1 ..]) |other| {
                for (other.messages) |other_msg_type| {
                    if (std.mem.eql(u8, msg_type, other_msg_type)) return .duplicate_worker_handler_owner;
                }
            }
        }
    }

    for (message_registry.protocol_messages) |entry| {
        if (entry.worker_handler == .unknown) continue;
        if (!workerHandlerRegistryDeclares(entry.worker_handler, entry.msg_type)) {
            return .worker_handler_not_declared;
        }
    }

    return .none;
}

fn workerHandlerRegistryDeclares(handler: WorkerHandlerFamily, msg_type: []const u8) bool {
    for (message_registry.worker_handler_registry) |registration| {
        if (registration.family != handler) continue;
        for (registration.messages) |registered_msg_type| {
            if (std.mem.eql(u8, msg_type, registered_msg_type)) return true;
        }
    }
    return false;
}

test "message registry topology task validates default dispatch decisions" {
    const task = topologyTask();
    try topology.validateDecisionMap(std.testing.allocator, task.protocol, task.carrier, task.decisions);
}

test "oracle accepts only the real worker handler for controller-to-worker messages" {
    try std.testing.expectEqual(
        Violation.none,
        checkWorkerHandlerAcceptance(MessageType.START_DECOUPLED_DILOCO_LOOP, .decoupled_training),
    );
    try std.testing.expectEqual(
        Violation.none,
        checkWorkerHandlerAcceptance(MessageType.START_GENERATION, .inference),
    );
}

test "oracle rejects wrong worker handler unknown tags and result messages" {
    try std.testing.expectEqual(
        Violation.wrong_worker_handler,
        checkWorkerHandlerAcceptance(MessageType.START_GENERATION, .decoupled_training),
    );
    try std.testing.expectEqual(
        Violation.unknown_message,
        checkWorkerHandlerAcceptance("pcp.unknown", .decoupled_training),
    );
    try std.testing.expectEqual(
        Violation.no_worker_handler,
        checkWorkerHandlerAcceptance(MessageType.DECOUPLED_FRAGMENT_UPDATE, .decoupled_training),
    );
}

test "oracle rejects dispatch decisions outside the message carrier" {
    try std.testing.expectError(
        error.DecisionOutsideCarrier,
        checkTopologyDecision(std.testing.allocator, .decoupled_start, .accept_inference),
    );
    try std.testing.expectError(
        error.DecisionOutsideCarrier,
        checkTopologyDecision(std.testing.allocator, .unknown_message, .accept_decoupled),
    );
    try checkTopologyDecision(std.testing.allocator, .unknown_message, .reject);
}

test "oracle mirrors real worker handler registry ownership" {
    try std.testing.expectEqual(Violation.none, checkWorkerHandlerRegistrySync());
}
