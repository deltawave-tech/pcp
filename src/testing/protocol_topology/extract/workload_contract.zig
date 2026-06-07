const std = @import("std");

const common = @import("common.zig");
const fixtures = @import("../fixtures/workload_contract.zig");
const model = @import("../models/workload_contract.zig");
const tasks = @import("../tasks.zig");
const topology = @import("../topology.zig");
const trace = @import("../trace.zig");

const Event = trace.Event;

pub const WorkloadExtractCase = enum {
    regular_request,
    invalid_request,
};

pub const TraceWitness = struct {
    case: WorkloadExtractCase,
    input_index: usize,
    protocol_index: usize,
    event_count: usize,
    route: model.WorkloadRoute,
    status: model.WorkloadStatus,
    resume_label: model.ResumeLabel,
    state_hash: u64,
};

pub const OwnedExtractedWorkloadCarrier = struct {
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
    task: tasks.OwnedProtocolTask,
    protocol: topology.OwnedComplex,
    execution: topology.OwnedSimplicialCarrierMap,
    decision_entries: []topology.VertexMapEntry,
    traces: []trace.OwnedTrace,
    witnesses: []TraceWitness,
    metadata: []common.SourceMetadata,

    pub fn protocolComplex(self: *const OwnedExtractedWorkloadCarrier) topology.Complex {
        return self.protocol.complex();
    }

    pub fn executionCarrier(self: *const OwnedExtractedWorkloadCarrier) topology.SimplicialCarrierMap {
        return self.execution.carrierMap();
    }

    pub fn decisionMap(self: *const OwnedExtractedWorkloadCarrier) topology.VertexMap {
        return .{
            .domain = self.protocol.complex(),
            .codomain = self.task.task().output,
            .entries = self.decision_entries,
        };
    }

    pub fn validate(self: *const OwnedExtractedWorkloadCarrier) !void {
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

    pub fn deinit(self: *OwnedExtractedWorkloadCarrier) void {
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
    case: WorkloadExtractCase,
    input: topology.Simplex,
    protocol: topology.Simplex,
    output: topology.Simplex,
};

const CaseRun = struct {
    trace_value: trace.OwnedTrace,
    output: model.WorkloadOutput,
};

const v = vertex;

const workload_regular_input = [_]topology.Vertex{
    v(.{ .api_caller = 0 }, "request.regular"),
    v(.{ .gateway = 0 }, "parser.training.regular"),
    v(.{ .job = 1 }, "job.training.regular"),
};
const workload_invalid_input = [_]topology.Vertex{
    v(.{ .api_caller = 0 }, "request.invalid"),
    v(.{ .gateway = 0 }, "parser.training.invalid"),
    v(.{ .job = 1 }, "job.training.invalid"),
};

const workload_regular_protocol = [_]topology.Vertex{
    v(.{ .api_caller = 0 }, "view.request.accepted"),
    v(.{ .gateway = 0 }, "view.route.regular_training"),
    v(.{ .job = 1 }, "view.job.normalized.regular"),
};
const workload_rejected_protocol = [_]topology.Vertex{
    v(.{ .api_caller = 0 }, "view.request.rejected"),
    v(.{ .gateway = 0 }, "view.route.rejected"),
    v(.{ .job = 1 }, "view.job.rejected"),
};

const workload_regular_output = [_]topology.Vertex{
    v(.{ .api_caller = 0 }, "status.accepted"),
    v(.{ .gateway = 0 }, "route.regular_training"),
    v(.{ .job = 1 }, "job.normalized.regular"),
};
const workload_rejected_output = [_]topology.Vertex{
    v(.{ .api_caller = 0 }, "status.rejected"),
    v(.{ .gateway = 0 }, "route.rejected"),
    v(.{ .job = 1 }, "job.rejected"),
};

const generated_cases = [_]GeneratedCase{
    .{
        .case = .regular_request,
        .input = .{ .vertices = &workload_regular_input },
        .protocol = .{ .vertices = &workload_regular_protocol },
        .output = .{ .vertices = &workload_regular_output },
    },
    .{
        .case = .invalid_request,
        .input = .{ .vertices = &workload_invalid_input },
        .protocol = .{ .vertices = &workload_rejected_protocol },
        .output = .{ .vertices = &workload_rejected_output },
    },
};

pub fn extractWorkloadExecutionCarrier(
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
) !OwnedExtractedWorkloadCarrier {
    try common.validateCaseTraceBounds(bound, generated_cases.len, generated_cases.len);
    var task = try tasks.buildWorkloadNormalizationTask(allocator);
    errdefer task.deinit();
    return generateForCases(allocator, task, bound, &generated_cases);
}

fn generateForCases(
    allocator: std.mem.Allocator,
    task: tasks.OwnedProtocolTask,
    bound: common.ExtractionBound,
    cases: []const GeneratedCase,
) !OwnedExtractedWorkloadCarrier {
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
            .route = run.output.route,
            .status = run.output.status,
            .resume_label = run.output.resume_label,
            .state_hash = workloadOutputHash(run.output),
        };
        metadata[index] = common.sourceMetadata(
            "src/testing/protocol_topology/extract/workload_contract.zig",
            fixtureForCase(case.case),
            traces[index].seed,
            bound,
            traceIdForCase(case.case),
            .production_fixture,
            "src/protocol/decoupled_job.zig + src/workloads/training/workload.zig",
        );
    }

    var generated = OwnedExtractedWorkloadCarrier{
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

fn fixtureForCase(case: WorkloadExtractCase) []const u8 {
    return switch (case) {
        .regular_request => "fixtures.regularJob",
        .invalid_request => "fixtures.invalidRequest",
    };
}

fn traceIdForCase(case: WorkloadExtractCase) []const u8 {
    return switch (case) {
        .regular_request => "workload.regular.request",
        .invalid_request => "workload.invalid.request",
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

fn runCase(allocator: std.mem.Allocator, case: WorkloadExtractCase) !CaseRun {
    switch (case) {
        .regular_request => {
            const job = try fixtures.regularJob();
            const output = model.acceptedOutput(job);
            const events = [_]Event{
                Event.init(.send, 0, null, 10),
                Event.init(.deliver, 0, 0, 10),
            };
            return .{
                .trace_value = try ownedTraceFromEvents(allocator, case, &events, output),
                .output = output,
            };
        },
        .invalid_request => {
            const output = model.rejectedOutput("invalid-session");
            const events = [_]Event{
                Event.init(.send, 0, null, 12),
                Event.init(.drop, 0, null, 12),
            };
            return .{
                .trace_value = try ownedTraceFromEvents(allocator, case, &events, output),
                .output = output,
            };
        },
    }
}

fn ownedTraceFromEvents(
    allocator: std.mem.Allocator,
    case: WorkloadExtractCase,
    events: []const Event,
    output: model.WorkloadOutput,
) !trace.OwnedTrace {
    return .{
        .allocator = allocator,
        .seed = @intFromEnum(case),
        .events = try allocator.dupe(Event, events),
        .final_snapshot = .{
            .steps = @intCast(events.len),
            .node_count = 3,
            .online_mask = 0b111,
            .checksum = workloadOutputHash(output),
        },
    };
}

fn workloadOutputHash(output: model.WorkloadOutput) u64 {
    var hasher = std.hash.Wyhash.init(0x776f_726b_6c6f_6164);
    hashUsize(&hasher, if (output.kind) |kind| @as(usize, @intFromEnum(kind)) + 1 else 0);
    hashUsize(&hasher, @intFromEnum(output.route));
    hashUsize(&hasher, @intFromEnum(output.status));
    hashUsize(&hasher, @intFromEnum(output.resume_label));
    hasher.update(output.session_id);
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
        if (witness.event_count == 0) return error.EmptyWorkloadTraceWitness;
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

test "workload extraction builds carried execution views from production-like fixtures" {
    const bound = common.ExtractionBound{
        .max_cases = 2,
        .max_traces = 2,
        .max_events = 3,
        .max_depth = 3,
    };
    var extracted = try extractWorkloadExecutionCarrier(std.testing.allocator, bound);
    defer extracted.deinit();

    try std.testing.expectEqual(@as(usize, generated_cases.len), extracted.protocol.simplexes.len);
    try std.testing.expectEqual(model.WorkloadRoute.regular_training_service, extracted.witnesses[0].route);
    try std.testing.expectEqual(model.WorkloadRoute.rejected, extracted.witnesses[1].route);
    try std.testing.expectEqual(model.ResumeLabel.compatible, extracted.witnesses[0].resume_label);
    try std.testing.expectEqual(model.ResumeLabel.none, extracted.witnesses[1].resume_label);
    try std.testing.expectEqualStrings("workload.regular.request", extracted.metadata[0].trace_id);
    try std.testing.expectEqual(tasks.CoverageEvidence.production_fixture, extracted.metadata[0].evidence);
}
