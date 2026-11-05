const std = @import("std");

const common = @import("common.zig");
const model = @import("../models/federation_placement.zig");
const tasks = @import("../tasks.zig");
const topology = @import("../topology.zig");
const trace = @import("../trace.zig");

const Event = trace.Event;

pub const TraceWitness = struct {
    trace_case: model.ExecutionTraceCase,
    input_index: usize,
    protocol_index: usize,
    event_count: usize,
    final_decision: model.Decision,
    violation: model.Violation,
    state_hash: u64,
};

pub const OwnedExtractedFederationCarrier = struct {
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
    task: tasks.OwnedProtocolTask,
    protocol: topology.OwnedComplex,
    execution: topology.OwnedSimplicialCarrierMap,
    decision_entries: []topology.VertexMapEntry,
    traces: []trace.OwnedTrace,
    witnesses: []TraceWitness,
    metadata: []common.SourceMetadata,

    pub fn protocolComplex(self: *const OwnedExtractedFederationCarrier) topology.Complex {
        return self.protocol.complex();
    }

    pub fn executionCarrier(self: *const OwnedExtractedFederationCarrier) topology.SimplicialCarrierMap {
        return self.execution.carrierMap();
    }

    pub fn decisionMap(self: *const OwnedExtractedFederationCarrier) topology.VertexMap {
        return .{
            .domain = self.protocol.complex(),
            .codomain = self.task.task().output,
            .entries = self.decision_entries,
        };
    }

    pub fn validate(self: *const OwnedExtractedFederationCarrier) !void {
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

    pub fn deinit(self: *OwnedExtractedFederationCarrier) void {
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
    trace_case: model.ExecutionTraceCase,
    input: topology.Simplex,
    protocol: topology.Simplex,
    output: topology.Simplex,
};

const CaseRun = struct {
    trace_value: trace.OwnedTrace,
    observed: model.ExecutionTraceObservation,
};

const v = vertex;

const federation_compatible_input = [_]topology.Vertex{
    v(.{ .hub = 0 }, "registry.current"),
    v(.{ .gateway = 1 }, "gateway.connected.compatible"),
    v(.{ .job = 30 }, "job.requires.training"),
};
const federation_stale_input = [_]topology.Vertex{
    v(.{ .hub = 0 }, "registry.stale_gateway"),
    v(.{ .gateway = 1 }, "gateway.stale.compatible"),
    v(.{ .job = 31 }, "job.requires.training"),
};
const federation_incompatible_input = [_]topology.Vertex{
    v(.{ .hub = 0 }, "registry.current.incompatible"),
    v(.{ .gateway = 1 }, "gateway.connected.incompatible"),
    v(.{ .job = 32 }, "job.requires.training"),
};

const federation_placed_protocol = [_]topology.Vertex{
    v(.{ .hub = 0 }, "view.placement.selected"),
    v(.{ .gateway = 1 }, "view.gateway.route.accepted"),
    v(.{ .job = 30 }, "view.job.forwarded"),
};
const federation_loss_protocol = [_]topology.Vertex{
    v(.{ .hub = 0 }, "view.placement.failed_after_loss"),
    v(.{ .gateway = 1 }, "view.gateway.route_lost"),
    v(.{ .job = 30 }, "view.job.failed_retryable"),
};
const federation_stale_protocol = [_]topology.Vertex{
    v(.{ .hub = 0 }, "view.placement.rejected.stale"),
    v(.{ .gateway = 1 }, "view.gateway.not_selected"),
    v(.{ .job = 31 }, "view.job.rejected"),
};
const federation_incompatible_protocol = [_]topology.Vertex{
    v(.{ .hub = 0 }, "view.placement.rejected.incompatible"),
    v(.{ .gateway = 1 }, "view.gateway.not_selected"),
    v(.{ .job = 32 }, "view.job.rejected"),
};

const federation_placed_output = [_]topology.Vertex{
    v(.{ .hub = 0 }, "placement.placed"),
    v(.{ .gateway = 1 }, "gateway.route.accepted"),
    v(.{ .job = 30 }, "job.forwarded"),
};
const federation_loss_output = [_]topology.Vertex{
    v(.{ .hub = 0 }, "placement.failed_after_loss"),
    v(.{ .gateway = 1 }, "gateway.route_lost"),
    v(.{ .job = 30 }, "job.failed_retryable"),
};
const federation_stale_output = [_]topology.Vertex{
    v(.{ .hub = 0 }, "placement.rejected.stale"),
    v(.{ .gateway = 1 }, "gateway.not_selected"),
    v(.{ .job = 31 }, "job.rejected"),
};
const federation_incompatible_output = [_]topology.Vertex{
    v(.{ .hub = 0 }, "placement.rejected.incompatible"),
    v(.{ .gateway = 1 }, "gateway.not_selected"),
    v(.{ .job = 32 }, "job.rejected"),
};

const generated_cases = [_]GeneratedCase{
    .{
        .trace_case = .compatible_placement,
        .input = .{ .vertices = &federation_compatible_input },
        .protocol = .{ .vertices = &federation_placed_protocol },
        .output = .{ .vertices = &federation_placed_output },
    },
    .{
        .trace_case = .compatible_placement_loss,
        .input = .{ .vertices = &federation_compatible_input },
        .protocol = .{ .vertices = &federation_loss_protocol },
        .output = .{ .vertices = &federation_loss_output },
    },
    .{
        .trace_case = .stale_gateway_rejection,
        .input = .{ .vertices = &federation_stale_input },
        .protocol = .{ .vertices = &federation_stale_protocol },
        .output = .{ .vertices = &federation_stale_output },
    },
    .{
        .trace_case = .incompatible_gateway_rejection,
        .input = .{ .vertices = &federation_incompatible_input },
        .protocol = .{ .vertices = &federation_incompatible_protocol },
        .output = .{ .vertices = &federation_incompatible_output },
    },
};

pub fn extractFederationPlacementCarrier(
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
) !OwnedExtractedFederationCarrier {
    try common.validateCaseTraceBounds(bound, generated_cases.len, generated_cases.len);
    var task = try tasks.buildFederationPlacementTask(allocator);
    errdefer task.deinit();
    return generateForCases(allocator, task, bound, &generated_cases);
}

fn generateForCases(
    allocator: std.mem.Allocator,
    task: tasks.OwnedProtocolTask,
    bound: common.ExtractionBound,
    cases: []const GeneratedCase,
) !OwnedExtractedFederationCarrier {
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
        var run = try runCase(allocator, case.trace_case);
        errdefer run.trace_value.deinit();

        const input_index = task.task().input.findSimplexIndex(case.input) orelse return error.ExtractedInputMissingFromTask;
        const protocol_index = protocol.complex().findSimplexIndex(case.protocol) orelse return error.ExtractedProtocolMissing;
        traces[index] = run.trace_value;
        initialized_traces += 1;
        try common.validateTraceWithinBound(traces[index], bound);
        witnesses[index] = .{
            .trace_case = case.trace_case,
            .input_index = input_index,
            .protocol_index = protocol_index,
            .event_count = traces[index].events.len,
            .final_decision = run.observed.final_decision,
            .violation = run.observed.violation,
            .state_hash = federationStateHash(run.observed),
        };
        metadata[index] = common.sourceMetadata(
            "src/testing/protocol_topology/extract/federation_placement.zig",
            fixtureForCase(case.trace_case),
            traces[index].seed,
            bound,
            traceIdForCase(case.trace_case),
            .bounded_model,
            "src/nodes/federation_hub/gateway_registry.zig",
        );
    }

    var generated = OwnedExtractedFederationCarrier{
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

fn fixtureForCase(case: model.ExecutionTraceCase) []const u8 {
    return switch (case) {
        .compatible_placement => "gateway_registry.compatible_placement",
        .compatible_placement_loss => "gateway_registry.routed_completion_loss",
        .stale_gateway_rejection => "gateway_registry.stale_gateway",
        .incompatible_gateway_rejection => "gateway_registry.incompatible_gateway",
    };
}

fn traceIdForCase(case: model.ExecutionTraceCase) []const u8 {
    return switch (case) {
        .compatible_placement => "federation.placement.compatible",
        .compatible_placement_loss => "federation.placement.loss",
        .stale_gateway_rejection => "federation.placement.stale_gateway",
        .incompatible_gateway_rejection => "federation.placement.incompatible",
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

fn runCase(allocator: std.mem.Allocator, case: model.ExecutionTraceCase) !CaseRun {
    const observed = try model.runExecutionTrace(case);
    const events = switch (case) {
        .compatible_placement => &[_]Event{
            Event.init(.join, 1, null, 30),
            Event.init(.send, 0, 1, 30),
            Event.init(.deliver, 1, 0, 30),
        },
        .compatible_placement_loss => &[_]Event{
            Event.init(.join, 1, null, 30),
            Event.init(.send, 0, 1, 30),
            Event.init(.crash, 1, null, 30),
            Event.init(.timeout, 0, null, 30),
        },
        .stale_gateway_rejection => &[_]Event{
            Event.init(.heartbeat, 1, 0, 31),
            Event.init(.timeout, 0, null, 31),
        },
        .incompatible_gateway_rejection => &[_]Event{
            Event.init(.join, 1, null, 32),
            Event.init(.drop, 0, 1, 32),
        },
    };

    return .{
        .trace_value = try ownedTraceFromEvents(allocator, case, events, observed),
        .observed = observed,
    };
}

fn ownedTraceFromEvents(
    allocator: std.mem.Allocator,
    case: model.ExecutionTraceCase,
    events: []const Event,
    observed: model.ExecutionTraceObservation,
) !trace.OwnedTrace {
    return .{
        .allocator = allocator,
        .seed = @intFromEnum(case),
        .events = try allocator.dupe(Event, events),
        .final_snapshot = .{
            .steps = @intCast(events.len),
            .node_count = 3,
            .online_mask = if (observed.final_decision == .failed_after_placement_loss) 0b101 else 0b111,
            .checksum = federationStateHash(observed),
        },
    };
}

fn federationStateHash(observed: model.ExecutionTraceObservation) u64 {
    var hasher = std.hash.Wyhash.init(0x6665_645f_7869);
    hashUsize(&hasher, @intFromEnum(observed.trace_case));
    hashUsize(&hasher, @intFromEnum(observed.final_decision));
    hashUsize(&hasher, @intFromEnum(observed.violation));
    hashUsize(&hasher, observed.input_index);
    hashUsize(&hasher, observed.protocol_index);
    hashUsize(&hasher, observed.output_index);
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
        if (witness.event_count == 0) return error.EmptyFederationTraceWitness;
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

test "federation extraction turns registry placement traces into carried views" {
    const bound = common.ExtractionBound{
        .max_cases = 4,
        .max_traces = 4,
        .max_events = 4,
        .max_depth = 4,
    };
    var extracted = try extractFederationPlacementCarrier(std.testing.allocator, bound);
    defer extracted.deinit();

    try std.testing.expectEqual(@as(usize, generated_cases.len), extracted.protocol.simplexes.len);
    try std.testing.expectEqual(model.Decision.placed, extracted.witnesses[0].final_decision);
    try std.testing.expectEqual(model.Decision.failed_after_placement_loss, extracted.witnesses[1].final_decision);
    try std.testing.expectEqual(model.Violation.stale_gateway, extracted.witnesses[2].violation);
    try std.testing.expectEqual(model.Violation.incompatible_gateway, extracted.witnesses[3].violation);
    try std.testing.expectEqualStrings("federation.placement.loss", extracted.metadata[1].trace_id);
    try std.testing.expectEqual(tasks.CoverageEvidence.bounded_model, extracted.metadata[0].evidence);
}
