const std = @import("std");

const common = @import("common.zig");
const message = @import("../../../network/message.zig");
const oracle = @import("../oracles/message_registry.zig");
const message_registry = @import("../../../protocol/message_registry.zig");
const tasks = @import("../tasks.zig");
const topology = @import("../topology.zig");
const trace = @import("../trace.zig");

const Event = trace.Event;
const MessageType = message.MessageType;

pub const MessageExtractCase = enum {
    decoupled_start,
    inference_start,
    worker_result,
    unknown_message,
};

pub const TraceWitness = struct {
    case: MessageExtractCase,
    input_index: usize,
    protocol_index: usize,
    event_count: usize,
    handler: message_registry.WorkerHandlerFamily,
    violation: oracle.Violation,
    state_hash: u64,
};

pub const OwnedExtractedMessageCarrier = struct {
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
    task: tasks.OwnedProtocolTask,
    protocol: topology.OwnedComplex,
    execution: topology.OwnedSimplicialCarrierMap,
    decision_entries: []topology.VertexMapEntry,
    traces: []trace.OwnedTrace,
    witnesses: []TraceWitness,
    metadata: []common.SourceMetadata,

    pub fn protocolComplex(self: *const OwnedExtractedMessageCarrier) topology.Complex {
        return self.protocol.complex();
    }

    pub fn executionCarrier(self: *const OwnedExtractedMessageCarrier) topology.SimplicialCarrierMap {
        return self.execution.carrierMap();
    }

    pub fn decisionMap(self: *const OwnedExtractedMessageCarrier) topology.VertexMap {
        return .{
            .domain = self.protocol.complex(),
            .codomain = self.task.task().output,
            .entries = self.decision_entries,
        };
    }

    pub fn validate(self: *const OwnedExtractedMessageCarrier) !void {
        try self.execution.carrierMap().validate(self.allocator, .{
            .require_monotone = true,
            .require_chromatic = true,
        });
        try topology.validateCarriedDecisionMap(
            self.allocator,
            self.execution.carrierMap(),
            self.task.task().carrier,
            self.decisionMap(),
        );
        try validateWitnessesCoverProtocol(self.allocator, self.protocol.complex(), self.witnesses);
        try common.validateMetadataSet(self.metadata, self.traces.len);
        for (self.traces) |trace_value| try common.validateTraceWithinBound(trace_value, self.bound);
    }

    pub fn deinit(self: *OwnedExtractedMessageCarrier) void {
        for (self.traces) |*trace_value| trace_value.deinit();
        self.allocator.free(self.metadata);
        self.allocator.free(self.traces);
        self.allocator.free(self.witnesses);
        self.allocator.free(self.decision_entries);
        self.execution.deinit();
        self.protocol.deinit();
        self.task.deinit();
        self.* = undefined;
    }
};

const GeneratedCase = struct {
    case: MessageExtractCase,
    msg_type: []const u8,
    handler: message_registry.WorkerHandlerFamily,
    input: topology.Simplex,
    protocol: topology.Simplex,
    output: topology.Simplex,
};

const CaseRun = struct {
    trace_value: trace.OwnedTrace,
    handler: message_registry.WorkerHandlerFamily,
    violation: oracle.Violation,
};

const v = vertex;

const msg_decoupled_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, MessageType.START_DECOUPLED_DILOCO_LOOP),
    v(.{ .worker = 0 }, "handler.decoupled_training"),
};
const msg_inference_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, MessageType.START_GENERATION),
    v(.{ .worker = 0 }, "handler.inference"),
};
const msg_result_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "queue.worker_result"),
    v(.{ .worker = 0 }, MessageType.DECOUPLED_FRAGMENT_UPDATE),
};
const msg_unknown_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "message.unknown"),
    v(.{ .worker = 0 }, "handler.unknown"),
};

const msg_decoupled_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.dispatch.controller_to_worker"),
    v(.{ .worker = 0 }, "view.handler.decoupled_training"),
};
const msg_inference_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.dispatch.controller_to_worker"),
    v(.{ .worker = 0 }, "view.handler.inference"),
};
const msg_result_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.dispatch.gateway_queue_only"),
    v(.{ .worker = 0 }, "view.reject.worker_result_to_worker"),
};
const msg_unknown_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.dispatch.fail_closed"),
    v(.{ .worker = 0 }, "view.reject.unknown_message"),
};

const msg_decoupled_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "dispatch.controller_to_worker"),
    v(.{ .worker = 0 }, "accepted.decoupled_training"),
};
const msg_inference_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "dispatch.controller_to_worker"),
    v(.{ .worker = 0 }, "accepted.inference"),
};
const msg_result_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "dispatch.gateway_queue_only"),
    v(.{ .worker = 0 }, "rejected.worker_result_to_worker"),
};
const msg_unknown_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "dispatch.fail_closed"),
    v(.{ .worker = 0 }, "rejected.unknown_message"),
};

const generated_cases = [_]GeneratedCase{
    .{
        .case = .decoupled_start,
        .msg_type = MessageType.START_DECOUPLED_DILOCO_LOOP,
        .handler = .decoupled_training,
        .input = .{ .vertices = &msg_decoupled_input },
        .protocol = .{ .vertices = &msg_decoupled_protocol },
        .output = .{ .vertices = &msg_decoupled_output },
    },
    .{
        .case = .inference_start,
        .msg_type = MessageType.START_GENERATION,
        .handler = .inference,
        .input = .{ .vertices = &msg_inference_input },
        .protocol = .{ .vertices = &msg_inference_protocol },
        .output = .{ .vertices = &msg_inference_output },
    },
    .{
        .case = .worker_result,
        .msg_type = MessageType.DECOUPLED_FRAGMENT_UPDATE,
        .handler = .unknown,
        .input = .{ .vertices = &msg_result_input },
        .protocol = .{ .vertices = &msg_result_protocol },
        .output = .{ .vertices = &msg_result_output },
    },
    .{
        .case = .unknown_message,
        .msg_type = "pcp.unknown",
        .handler = .unknown,
        .input = .{ .vertices = &msg_unknown_input },
        .protocol = .{ .vertices = &msg_unknown_protocol },
        .output = .{ .vertices = &msg_unknown_output },
    },
};

pub fn extractMessageDispatchCarrier(
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
) !OwnedExtractedMessageCarrier {
    try common.validateCaseTraceBounds(bound, generated_cases.len, generated_cases.len);
    var task = try tasks.buildMessageDispatchTask(allocator);
    errdefer task.deinit();
    return generateForCases(allocator, task, bound, &generated_cases);
}

fn generateForCases(
    allocator: std.mem.Allocator,
    task: tasks.OwnedProtocolTask,
    bound: common.ExtractionBound,
    cases: []const GeneratedCase,
) !OwnedExtractedMessageCarrier {
    var protocol_builder = topology.ComplexBuilder.init(allocator);
    errdefer protocol_builder.deinit();
    for (cases) |case| try protocol_builder.add(case.protocol);

    var protocol = try protocol_builder.toOwnedComplex();
    errdefer protocol.deinit();

    var execution = try buildExecutionCarrier(allocator, task.task().input, protocol.complex(), cases);
    errdefer execution.deinit();

    const decision_entries = try buildDecisionEntries(allocator, protocol.complex(), cases);
    errdefer allocator.free(decision_entries);

    const traces = try allocator.alloc(trace.OwnedTrace, cases.len);
    errdefer allocator.free(traces);
    const witnesses = try allocator.alloc(TraceWitness, cases.len);
    errdefer allocator.free(witnesses);
    const metadata = try allocator.alloc(common.SourceMetadata, cases.len);
    errdefer allocator.free(metadata);

    var initialized_traces: usize = 0;
    errdefer {
        for (traces[0..initialized_traces]) |*trace_value| trace_value.deinit();
    }

    for (cases, 0..) |case, index| {
        var run = try runCase(allocator, case);
        errdefer run.trace_value.deinit();

        const input_index = task.task().input.findSimplexIndex(case.input) orelse return error.ExtractedInputMissingFromTask;
        const protocol_index = protocol.complex().findSimplexIndex(case.protocol) orelse return error.ExtractedProtocolMissing;
        traces[index] = run.trace_value;
        initialized_traces += 1;
        try common.validateTraceWithinBound(traces[index], bound);
        witnesses[index] = .{
            .case = case.case,
            .input_index = input_index,
            .protocol_index = protocol_index,
            .event_count = traces[index].events.len,
            .handler = run.handler,
            .violation = run.violation,
            .state_hash = messageStateHash(run.handler, run.violation),
        };
        metadata[index] = common.sourceMetadata(
            "src/testing/protocol_topology/extract/message_registry.zig",
            fixtureForCase(case.case),
            traces[index].seed,
            bound,
            traceIdForCase(case.case),
            .production_fixture,
            "src/protocol/message_registry.zig + src/nodes/workers/task_handlers.zig",
        );
    }

    var generated = OwnedExtractedMessageCarrier{
        .allocator = allocator,
        .bound = bound,
        .task = task,
        .protocol = protocol,
        .execution = execution,
        .decision_entries = decision_entries,
        .traces = traces,
        .witnesses = witnesses,
        .metadata = metadata,
    };
    errdefer generated.deinit();

    try generated.validate();
    return generated;
}

fn fixtureForCase(case: MessageExtractCase) []const u8 {
    return switch (case) {
        .decoupled_start => "message_registry.START_DECOUPLED_DILOCO_LOOP",
        .inference_start => "message_registry.START_GENERATION",
        .worker_result => "message_registry.DECOUPLED_FRAGMENT_UPDATE",
        .unknown_message => "message_registry.unknown",
    };
}

fn traceIdForCase(case: MessageExtractCase) []const u8 {
    return switch (case) {
        .decoupled_start => "message.dispatch.decoupled_start",
        .inference_start => "message.dispatch.inference_start",
        .worker_result => "message.dispatch.worker_result_reject",
        .unknown_message => "message.dispatch.unknown_reject",
    };
}

fn buildExecutionCarrier(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    protocol: topology.Complex,
    cases: []const GeneratedCase,
) !topology.OwnedSimplicialCarrierMap {
    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    var builder = topology.SimplicialCarrierMapBuilder.init(allocator, input, protocol);
    errdefer builder.deinit();

    for (input_faces.simplexes) |face| {
        var image = try imageForInputFace(allocator, protocol, cases, face);
        defer image.deinit();
        try builder.add(face, .{ .parent = protocol, .simplexes = image.simplexes });
    }

    return builder.toOwned();
}

fn imageForInputFace(
    allocator: std.mem.Allocator,
    protocol: topology.Complex,
    cases: []const GeneratedCase,
    face: topology.Simplex,
) !topology.OwnedComplex {
    var builder = topology.ComplexBuilder.init(allocator);
    errdefer builder.deinit();

    if (face.isEmpty()) {
        try builder.add(.{ .vertices = &[_]topology.Vertex{} });
        return builder.toOwnedComplex();
    }

    var matched = false;
    for (cases) |case| {
        if (!face.isSubsetOf(case.input)) continue;
        matched = true;
        var projected = try projectProtocolToInputFace(allocator, face, case.protocol);
        defer projected.deinit();
        if (!protocol.containsSimplex(projected.simplex())) return error.ProtocolProjectionOutsideProtocolComplex;
        try builder.add(projected.simplex());
    }
    if (!matched) return error.UncoveredExtractedInputFace;

    return builder.toOwnedComplex();
}

fn buildDecisionEntries(
    allocator: std.mem.Allocator,
    protocol: topology.Complex,
    cases: []const GeneratedCase,
) ![]topology.VertexMapEntry {
    var protocol_vertices = try protocol.vertices(allocator);
    defer protocol_vertices.deinit();

    var entries = std.ArrayList(topology.VertexMapEntry).init(allocator);
    errdefer entries.deinit();

    for (protocol_vertices.vertices) |source| {
        const target = targetForProtocolVertex(cases, source) orelse return error.MissingDecisionTarget;
        try entries.append(.{ .source = source, .target = target });
    }

    return entries.toOwnedSlice();
}

fn targetForProtocolVertex(cases: []const GeneratedCase, source: topology.Vertex) ?topology.Vertex {
    var target: ?topology.Vertex = null;
    for (cases) |case| {
        if (!case.protocol.containsVertex(source)) continue;
        const candidate = case.output.vertexWithColor(source.color) orelse return null;
        if (target) |existing| {
            if (!topology.Vertex.eql(existing, candidate)) return null;
        } else {
            target = candidate;
        }
    }
    return target;
}

fn projectProtocolToInputFace(
    allocator: std.mem.Allocator,
    face: topology.Simplex,
    protocol: topology.Simplex,
) !topology.OwnedSimplex {
    const projected = try allocator.alloc(topology.Vertex, face.vertices.len);
    errdefer allocator.free(projected);

    for (face.vertices, 0..) |input_vertex, index| {
        projected[index] = protocol.vertexWithColor(input_vertex.color) orelse return error.ProtocolFacetMissingColor;
    }

    std.mem.sort(topology.Vertex, projected, {}, topology.Vertex.lessThan);
    return .{ .allocator = allocator, .vertices = projected };
}

fn runCase(allocator: std.mem.Allocator, case: GeneratedCase) !CaseRun {
    const violation = oracle.checkWorkerHandlerAcceptance(case.msg_type, case.handler);
    const events = switch (case.case) {
        .decoupled_start, .inference_start => &[_]Event{
            Event.init(.send, 0, 1, @intCast(@intFromEnum(case.case) + 1)),
            Event.init(.deliver, 1, 0, @intCast(@intFromEnum(case.case) + 1)),
        },
        .worker_result => &[_]Event{
            Event.init(.send, 1, 0, 4),
            Event.init(.drop, 0, 1, 4),
        },
        .unknown_message => &[_]Event{
            Event.init(.send, 0, 1, 5),
            Event.init(.drop, 0, 1, 5),
        },
    };

    return .{
        .trace_value = try ownedTraceFromEvents(allocator, case.case, events, case.handler, violation),
        .handler = case.handler,
        .violation = violation,
    };
}

fn ownedTraceFromEvents(
    allocator: std.mem.Allocator,
    case: MessageExtractCase,
    events: []const Event,
    handler: message_registry.WorkerHandlerFamily,
    violation: oracle.Violation,
) !trace.OwnedTrace {
    return .{
        .allocator = allocator,
        .seed = @intFromEnum(case),
        .events = try allocator.dupe(Event, events),
        .final_snapshot = .{
            .steps = @intCast(events.len),
            .node_count = 2,
            .online_mask = 0b11,
            .checksum = messageStateHash(handler, violation),
        },
    };
}

fn messageStateHash(handler: message_registry.WorkerHandlerFamily, violation: oracle.Violation) u64 {
    var hasher = std.hash.Wyhash.init(0x6d73_675f_7869);
    hashUsize(&hasher, @intFromEnum(handler));
    hashUsize(&hasher, @intFromEnum(violation));
    return hasher.final();
}

fn validateWitnessesCoverProtocol(
    allocator: std.mem.Allocator,
    protocol: topology.Complex,
    witnesses: []const TraceWitness,
) !void {
    var seen = try allocator.alloc(bool, protocol.simplexes.len);
    defer allocator.free(seen);
    @memset(seen, false);

    for (witnesses) |witness| {
        if (witness.protocol_index >= seen.len) return error.WitnessProtocolIndexOutOfBounds;
        seen[witness.protocol_index] = true;
        if (witness.event_count == 0) return error.EmptyMessageTraceWitness;
    }
    for (seen) |item| {
        if (!item) return error.ProtocolSimplexMissingTraceWitness;
    }
}

fn hashUsize(hasher: anytype, value: usize) void {
    var buffer: [8]u8 = undefined;
    std.mem.writeInt(u64, &buffer, @intCast(value), .little);
    hasher.update(&buffer);
}

fn vertex(color: topology.Color, label: []const u8) topology.Vertex {
    return .{ .color = color, .label = .{ .symbol = label } };
}

test "message dispatch extraction mirrors real registry handler ownership" {
    const bound = common.ExtractionBound{
        .max_cases = 4,
        .max_traces = 4,
        .max_events = 2,
        .max_depth = 2,
    };
    var extracted = try extractMessageDispatchCarrier(std.testing.allocator, bound);
    defer extracted.deinit();

    try std.testing.expectEqual(@as(usize, generated_cases.len), extracted.protocol.simplexes.len);
    try std.testing.expectEqual(oracle.Violation.none, extracted.witnesses[0].violation);
    try std.testing.expectEqual(oracle.Violation.none, extracted.witnesses[1].violation);
    try std.testing.expectEqual(oracle.Violation.no_worker_handler, extracted.witnesses[2].violation);
    try std.testing.expectEqual(oracle.Violation.unknown_message, extracted.witnesses[3].violation);
    try std.testing.expectEqual(oracle.Violation.none, oracle.checkWorkerHandlerRegistrySync());
    try std.testing.expectEqualStrings("message.dispatch.decoupled_start", extracted.metadata[0].trace_id);
    try std.testing.expectEqual(tasks.CoverageEvidence.production_fixture, extracted.metadata[0].evidence);
}
