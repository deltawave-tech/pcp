const std = @import("std");

const common = @import("common.zig");
const oracle = @import("../oracles/decoupled_merge.zig");
const tasks = @import("../tasks.zig");
const topology = @import("../topology.zig");
const trace = @import("../trace.zig");

const Event = trace.Event;

pub const RdaExtractCase = enum {
    direction_norm,
    degenerate_zero,
    below_quorum,
};

pub const TraceWitness = struct {
    case: RdaExtractCase,
    input_index: usize,
    protocol_index: usize,
    event_count: usize,
    participant_mask: u64,
    participant_count: usize,
    total_weight: f64,
    direction_weight: f64,
    mean_norm: f64,
    output_norm: f64,
    violation: oracle.Violation,
    state_hash: u64,
};

pub const OwnedExtractedRdaCarrier = struct {
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
    task: tasks.OwnedProtocolTask,
    protocol: topology.OwnedComplex,
    execution: topology.OwnedSimplicialCarrierMap,
    decision_entries: []topology.VertexMapEntry,
    traces: []trace.OwnedTrace,
    witnesses: []TraceWitness,
    metadata: []common.SourceMetadata,

    pub fn protocolComplex(self: *const OwnedExtractedRdaCarrier) topology.Complex {
        return self.protocol.complex();
    }

    pub fn executionCarrier(self: *const OwnedExtractedRdaCarrier) topology.SimplicialCarrierMap {
        return self.execution.carrierMap();
    }

    pub fn decisionMap(self: *const OwnedExtractedRdaCarrier) topology.VertexMap {
        return .{
            .domain = self.protocol.complex(),
            .codomain = self.task.task().output,
            .entries = self.decision_entries,
        };
    }

    pub fn validate(self: *const OwnedExtractedRdaCarrier) !void {
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

    pub fn deinit(self: *OwnedExtractedRdaCarrier) void {
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
    case: RdaExtractCase,
    input: topology.Simplex,
    protocol: topology.Simplex,
    output: topology.Simplex,
};

const CaseRun = struct {
    trace_value: trace.OwnedTrace,
    result: oracle.MergeResult,
};

const v = vertex;

const rda_direction_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge_kind.rda.direction"),
    v(.{ .session = 50 }, "window.quorum"),
    v(.{ .worker = 0 }, "participants.direction_nonzero"),
};
const rda_degenerate_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge_kind.rda.degenerate"),
    v(.{ .session = 51 }, "window.quorum"),
    v(.{ .worker = 0 }, "participants.direction_degenerate"),
};
const rda_below_quorum_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge_kind.rda.below_quorum"),
    v(.{ .session = 52 }, "window.below_quorum"),
    v(.{ .worker = 0 }, "participants.insufficient"),
};

const rda_direction_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.rda.direction_norm_carried"),
    v(.{ .session = 50 }, "view.merge.committed"),
    v(.{ .worker = 0 }, "view.participants.weighted"),
};
const rda_degenerate_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.rda.degenerate_zero"),
    v(.{ .session = 51 }, "view.merge.committed"),
    v(.{ .worker = 0 }, "view.participants.canceling_direction"),
};
const rda_below_quorum_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.rda.below_quorum"),
    v(.{ .session = 52 }, "view.merge.deferred"),
    v(.{ .worker = 0 }, "view.participants.insufficient"),
};

const rda_direction_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "rda.output.direction_norm_carried"),
    v(.{ .session = 50 }, "merge.committed"),
    v(.{ .worker = 0 }, "participants.used"),
};
const rda_degenerate_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "rda.output.zero"),
    v(.{ .session = 51 }, "merge.committed"),
    v(.{ .worker = 0 }, "participants.canceling_direction"),
};
const rda_below_quorum_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "rda.output.none"),
    v(.{ .session = 52 }, "merge.deferred"),
    v(.{ .worker = 0 }, "participants.insufficient"),
};

const generated_cases = [_]GeneratedCase{
    .{
        .case = .direction_norm,
        .input = .{ .vertices = &rda_direction_input },
        .protocol = .{ .vertices = &rda_direction_protocol },
        .output = .{ .vertices = &rda_direction_output },
    },
    .{
        .case = .degenerate_zero,
        .input = .{ .vertices = &rda_degenerate_input },
        .protocol = .{ .vertices = &rda_degenerate_protocol },
        .output = .{ .vertices = &rda_degenerate_output },
    },
    .{
        .case = .below_quorum,
        .input = .{ .vertices = &rda_below_quorum_input },
        .protocol = .{ .vertices = &rda_below_quorum_protocol },
        .output = .{ .vertices = &rda_below_quorum_output },
    },
};

pub fn extractRdaMergeCarrier(
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
) !OwnedExtractedRdaCarrier {
    try common.validateCaseTraceBounds(bound, generated_cases.len, generated_cases.len);
    var task = try tasks.buildRdaMergeTask(allocator);
    errdefer task.deinit();
    return generateForCases(allocator, task, bound, &generated_cases);
}

fn generateForCases(
    allocator: std.mem.Allocator,
    task: tasks.OwnedProtocolTask,
    bound: common.ExtractionBound,
    cases: []const GeneratedCase,
) !OwnedExtractedRdaCarrier {
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
        var run = try runCase(allocator, case.case);
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
            .participant_mask = run.result.participant_mask,
            .participant_count = run.result.participant_count,
            .total_weight = run.result.total_weight,
            .direction_weight = run.result.direction_weight,
            .mean_norm = run.result.mean_norm,
            .output_norm = run.result.output_norm,
            .violation = run.result.violation,
            .state_hash = rdaStateHash(run.result),
        };
        metadata[index] = common.sourceMetadata(
            "src/testing/protocol_topology/extract/rda_merge.zig",
            fixtureForCase(case.case),
            traces[index].seed,
            bound,
            traceIdForCase(case.case),
            .production_fixture,
            "src/algorithms/decoupled_merge.zig",
        );
    }

    var generated = OwnedExtractedRdaCarrier{
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

fn fixtureForCase(case: RdaExtractCase) []const u8 {
    return switch (case) {
        .direction_norm => "decoupled_merge.rda.direction_norm",
        .degenerate_zero => "decoupled_merge.rda.degenerate_zero",
        .below_quorum => "decoupled_merge.rda.below_quorum",
    };
}

fn traceIdForCase(case: RdaExtractCase) []const u8 {
    return switch (case) {
        .direction_norm => "rda.merge.direction_norm",
        .degenerate_zero => "rda.merge.degenerate_zero",
        .below_quorum => "rda.merge.below_quorum",
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

fn runCase(allocator: std.mem.Allocator, case: RdaExtractCase) !CaseRun {
    return switch (case) {
        .direction_norm => runDirectionNorm(allocator),
        .degenerate_zero => runDegenerateZero(allocator),
        .below_quorum => runBelowQuorum(allocator),
    };
}

fn runDirectionNorm(allocator: std.mem.Allocator) !CaseRun {
    const window = oracle.Window{ .syncer_step = 1, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 2 };
    const a = [_]i64{ 2, 0 };
    const b = [_]i64{ 0, 2 };
    const updates = [_]oracle.Update{
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &a, .tokens_since_update = 10, .steps_since_update = 1 },
        .{ .learner_id = 1, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 2, .values = &b, .tokens_since_update = 10, .steps_since_update = 2 },
    };
    const result = oracle.compute(window, .rda, &updates);
    const events = [_]Event{
        Event.init(.deliver, 0, 0, 1),
        Event.init(.deliver, 1, 0, 1),
        Event.init(.local_step_complete, 0, null, 1),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, .direction_norm, &events, result),
        .result = result,
    };
}

fn runDegenerateZero(allocator: std.mem.Allocator) !CaseRun {
    const window = oracle.Window{ .syncer_step = 2, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 2 };
    const a = [_]i64{ 1, 0 };
    const b = [_]i64{ -1, 0 };
    const updates = [_]oracle.Update{
        .{ .learner_id = 0, .syncer_step = 2, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 2, .fragment_id = 0, .arrival_tick = 2, .values = &b },
    };
    const result = oracle.compute(window, .rda, &updates);
    const events = [_]Event{
        Event.init(.deliver, 0, 0, 2),
        Event.init(.deliver, 1, 0, 2),
        Event.init(.local_step_complete, 0, null, 2),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, .degenerate_zero, &events, result),
        .result = result,
    };
}

fn runBelowQuorum(allocator: std.mem.Allocator) !CaseRun {
    const window = oracle.Window{ .syncer_step = 3, .fragment_id = 0, .min_quorum = 3, .grace_deadline_tick = 3, .dimensions = 1 };
    const a = [_]i64{0};
    const b = [_]i64{10};
    const updates = [_]oracle.Update{
        .{ .learner_id = 0, .syncer_step = 3, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 3, .fragment_id = 0, .arrival_tick = 2, .values = &b },
    };
    const result = oracle.compute(window, .rda, &updates);
    const events = [_]Event{
        Event.init(.deliver, 0, 0, 3),
        Event.init(.deliver, 1, 0, 3),
        Event.init(.timeout, 0, null, 3),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, .below_quorum, &events, result),
        .result = result,
    };
}

fn ownedTraceFromEvents(
    allocator: std.mem.Allocator,
    case: RdaExtractCase,
    events: []const Event,
    result: oracle.MergeResult,
) !trace.OwnedTrace {
    return .{
        .allocator = allocator,
        .seed = @intFromEnum(case),
        .events = try allocator.dupe(Event, events),
        .final_snapshot = .{
            .steps = @intCast(events.len),
            .node_count = 3,
            .online_mask = 0b111,
            .checksum = rdaStateHash(result),
        },
    };
}

fn rdaStateHash(result: oracle.MergeResult) u64 {
    var hasher = std.hash.Wyhash.init(0x7264_615f_7869);
    hashUsize(&hasher, result.participant_count);
    hashUsize(&hasher, result.non_participant_count);
    hashUsize(&hasher, @intCast(result.participant_mask));
    hashUsize(&hasher, @intFromEnum(result.violation));
    hashF64(&hasher, result.total_weight);
    hashF64(&hasher, result.direction_weight);
    hashF64(&hasher, result.mean_norm);
    hashF64(&hasher, result.output_norm);
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
        if (witness.event_count == 0) return error.EmptyRdaTraceWitness;
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

fn hashF64(hasher: anytype, value: f64) void {
    var buffer: [8]u8 = undefined;
    std.mem.writeInt(u64, &buffer, @bitCast(value), .little);
    hasher.update(&buffer);
}

fn vertex(color: topology.Color, label: []const u8) topology.Vertex {
    return .{ .color = color, .label = .{ .symbol = label } };
}

test "RDA extraction preserves participant masks weights direction and degenerate cases" {
    const bound = common.ExtractionBound{
        .max_cases = 3,
        .max_traces = 3,
        .max_events = 3,
        .max_depth = 3,
    };
    var extracted = try extractRdaMergeCarrier(std.testing.allocator, bound);
    defer extracted.deinit();

    try std.testing.expectEqual(@as(usize, generated_cases.len), extracted.protocol.simplexes.len);
    try std.testing.expectEqual(oracle.Violation.none, extracted.witnesses[0].violation);
    try std.testing.expectEqual(@as(u64, 0b11), extracted.witnesses[0].participant_mask);
    try std.testing.expect(extracted.witnesses[0].total_weight > extracted.witnesses[0].direction_weight - oracle.epsilon);
    try std.testing.expect(extracted.witnesses[0].output_norm > 0.0);
    try std.testing.expectEqual(oracle.Violation.none, extracted.witnesses[1].violation);
    try std.testing.expectEqual(@as(f64, 0.0), extracted.witnesses[1].output_norm);
    try std.testing.expectEqual(oracle.Violation.below_quorum, extracted.witnesses[2].violation);
    try std.testing.expectEqualStrings("rda.merge.direction_norm", extracted.metadata[0].trace_id);
    try std.testing.expectEqual(tasks.CoverageEvidence.production_fixture, extracted.metadata[0].evidence);
}
