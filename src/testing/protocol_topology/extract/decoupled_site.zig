const std = @import("std");

const common = @import("common.zig");
const model = @import("../models/decoupled_site.zig");
const tasks = @import("../tasks.zig");
const topology = @import("../topology.zig");
const trace = @import("../trace.zig");

const Event = trace.Event;

pub const DecoupledTraceCase = enum {
    fragment_quorum_commit,
    below_quorum_timeout,
    duplicate_queued_response,
    context_payload_mismatch,
    stale_vector_clock,
    post_cancel_update,
    crash_restart_resume,
    incompatible_resume,
};

pub const InjectedBug = enum {
    below_quorum_commit,
    stale_vector_clock_acceptance,
};

pub const TraceWitness = struct {
    case: DecoupledTraceCase,
    input_index: usize,
    protocol_index: usize,
    event_count: usize,
    decision: model.Decision,
    violation: model.Violation,
    accepted_updates: usize,
    committed_windows: usize,
    tape_entries: usize,
    state_hash: u64,
    rda_bridge: bool = false,
};

pub const ExplorationSummary = struct {
    checked: usize = 0,
    decision_mask: u32 = 0,
    violation_mask: u32 = 0,
    committed_windows: usize = 0,
    failed_states: usize = 0,

    pub fn record(self: *ExplorationSummary, snapshot_value: model.Snapshot) void {
        self.checked += 1;
        self.decision_mask |= bitForEnum(snapshot_value.decision);
        self.violation_mask |= bitForEnum(snapshot_value.violation);
        self.committed_windows += snapshot_value.committed_windows;
        if (snapshot_value.violation != .none) self.failed_states += 1;
    }

    pub fn sawDecision(self: ExplorationSummary, decision: model.Decision) bool {
        return (self.decision_mask & bitForEnum(decision)) != 0;
    }

    pub fn sawViolation(self: ExplorationSummary, violation: model.Violation) bool {
        return (self.violation_mask & bitForEnum(violation)) != 0;
    }

    pub fn eql(a: ExplorationSummary, b: ExplorationSummary) bool {
        return a.checked == b.checked and
            a.decision_mask == b.decision_mask and
            a.violation_mask == b.violation_mask and
            a.committed_windows == b.committed_windows and
            a.failed_states == b.failed_states;
    }
};

pub const BugWitness = struct {
    allocator: std.mem.Allocator,
    bug: InjectedBug,
    violation: model.Violation,
    trace_value: trace.OwnedTrace,

    pub fn deinit(self: *BugWitness) void {
        self.trace_value.deinit();
        self.* = undefined;
    }
};

pub const RdaBridgeWitness = struct {
    decoupled_case: DecoupledTraceCase,
    decoupled_protocol_index: usize,
    rda_input_index: usize,
    replay_state_hash: u64,
};

pub const OwnedGeneratedDecoupledCarrier = struct {
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
    task: tasks.OwnedProtocolTask,
    protocol: topology.OwnedComplex,
    execution: topology.OwnedSimplicialCarrierMap,
    decision_entries: []topology.VertexMapEntry,
    traces: []trace.OwnedTrace,
    witnesses: []TraceWitness,
    metadata: []common.SourceMetadata,

    pub fn protocolComplex(self: *const OwnedGeneratedDecoupledCarrier) topology.Complex {
        return self.protocol.complex();
    }

    pub fn executionCarrier(self: *const OwnedGeneratedDecoupledCarrier) topology.SimplicialCarrierMap {
        return self.execution.carrierMap();
    }

    pub fn decisionMap(self: *const OwnedGeneratedDecoupledCarrier) topology.VertexMap {
        return .{
            .domain = self.protocol.complex(),
            .codomain = self.task.task().output,
            .entries = self.decision_entries,
        };
    }

    pub fn validate(self: *const OwnedGeneratedDecoupledCarrier) !void {
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

    pub fn rdaBridge(self: *const OwnedGeneratedDecoupledCarrier, allocator: std.mem.Allocator) !RdaBridgeWitness {
        var rda_task = try tasks.buildRdaMergeTask(allocator);
        defer rda_task.deinit();

        const rda_index = rda_task.task().input.findSimplexIndex(.{ .vertices = &rda_direction_input }) orelse {
            return error.RdaInputMissing;
        };

        for (self.witnesses) |witness| {
            if (!witness.rda_bridge) continue;
            return .{
                .decoupled_case = witness.case,
                .decoupled_protocol_index = witness.protocol_index,
                .rda_input_index = rda_index,
                .replay_state_hash = witness.state_hash,
            };
        }
        return error.MissingCommittedWindowBridge;
    }

    pub fn deinit(self: *OwnedGeneratedDecoupledCarrier) void {
        for (self.traces) |*trace_value| {
            trace_value.deinit();
        }
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
    case: DecoupledTraceCase,
    input: topology.Simplex,
    protocol: topology.Simplex,
    output: topology.Simplex,
};

const CaseRun = struct {
    trace_value: trace.OwnedTrace,
    snapshot: model.Snapshot,
    collection: model.CollectionSummary = .{},
    committed_replay_present: bool = false,
};

const v = vertex;

const fragment_fresh_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "window.open.quorum"),
    v(.{ .session = 40 }, "session.active"),
    v(.{ .worker = 0 }, "fragment.fresh.unique_sender.quorum"),
};
const fragment_below_quorum_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "window.open.below_quorum"),
    v(.{ .session = 41 }, "session.active"),
    v(.{ .worker = 0 }, "fragment.fresh.unique_sender.below_quorum"),
};
const fragment_duplicate_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "window.open.duplicate_sender"),
    v(.{ .session = 42 }, "session.active"),
    v(.{ .worker = 0 }, "fragment.duplicate_update"),
};
const fragment_context_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "window.open.context_check"),
    v(.{ .session = 43 }, "session.active"),
    v(.{ .worker = 0 }, "fragment.context_payload_mismatch"),
};
const fragment_clock_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "window.open.vector_clock"),
    v(.{ .session = 44 }, "session.active"),
    v(.{ .worker = 0 }, "fragment.stale_vector_clock"),
};
const fragment_canceled_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "window.canceled"),
    v(.{ .session = 45 }, "session.canceled"),
    v(.{ .worker = 0 }, "fragment.post_cancel"),
};

const fragment_commit_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.merge.committed.event_tape_replay"),
    v(.{ .session = 40 }, "view.session.progressed"),
    v(.{ .worker = 0 }, "view.update.accepted.quorum"),
};
const fragment_defer_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.merge.deferred.timeout"),
    v(.{ .session = 41 }, "view.session.waiting"),
    v(.{ .worker = 0 }, "view.update.pending.below_quorum"),
};
const fragment_duplicate_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.merge.not_committed.duplicate_sender"),
    v(.{ .session = 42 }, "view.session.active"),
    v(.{ .worker = 0 }, "view.update.rejected.duplicate"),
};
const fragment_context_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.merge.not_committed.context_payload_mismatch"),
    v(.{ .session = 43 }, "view.session.active"),
    v(.{ .worker = 0 }, "view.update.rejected.context"),
};
const fragment_clock_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.merge.not_committed.stale_vector_clock"),
    v(.{ .session = 44 }, "view.session.active"),
    v(.{ .worker = 0 }, "view.update.rejected.vector_clock"),
};
const fragment_canceled_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.merge.canceled.post_cancel"),
    v(.{ .session = 45 }, "view.session.canceled"),
    v(.{ .worker = 0 }, "view.update.rejected.post_cancel"),
};

const fragment_commit_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge.committed"),
    v(.{ .session = 40 }, "session.progressed"),
    v(.{ .worker = 0 }, "update.accepted"),
};
const fragment_defer_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge.deferred"),
    v(.{ .session = 41 }, "session.waiting"),
    v(.{ .worker = 0 }, "update.pending"),
};
const fragment_duplicate_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge.not_committed"),
    v(.{ .session = 42 }, "session.active"),
    v(.{ .worker = 0 }, "update.rejected.duplicate"),
};
const fragment_context_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge.not_committed"),
    v(.{ .session = 43 }, "session.active"),
    v(.{ .worker = 0 }, "update.rejected.context"),
};
const fragment_clock_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge.not_committed"),
    v(.{ .session = 44 }, "session.active"),
    v(.{ .worker = 0 }, "update.rejected.vector_clock"),
};
const fragment_canceled_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge.canceled"),
    v(.{ .session = 45 }, "session.canceled"),
    v(.{ .worker = 0 }, "update.rejected.post_cancel"),
};

const generated_cases = [_]GeneratedCase{
    .{
        .case = .fragment_quorum_commit,
        .input = .{ .vertices = &fragment_fresh_input },
        .protocol = .{ .vertices = &fragment_commit_protocol },
        .output = .{ .vertices = &fragment_commit_output },
    },
    .{
        .case = .below_quorum_timeout,
        .input = .{ .vertices = &fragment_below_quorum_input },
        .protocol = .{ .vertices = &fragment_defer_protocol },
        .output = .{ .vertices = &fragment_defer_output },
    },
    .{
        .case = .duplicate_queued_response,
        .input = .{ .vertices = &fragment_duplicate_input },
        .protocol = .{ .vertices = &fragment_duplicate_protocol },
        .output = .{ .vertices = &fragment_duplicate_output },
    },
    .{
        .case = .context_payload_mismatch,
        .input = .{ .vertices = &fragment_context_input },
        .protocol = .{ .vertices = &fragment_context_protocol },
        .output = .{ .vertices = &fragment_context_output },
    },
    .{
        .case = .stale_vector_clock,
        .input = .{ .vertices = &fragment_clock_input },
        .protocol = .{ .vertices = &fragment_clock_protocol },
        .output = .{ .vertices = &fragment_clock_output },
    },
    .{
        .case = .post_cancel_update,
        .input = .{ .vertices = &fragment_canceled_input },
        .protocol = .{ .vertices = &fragment_canceled_protocol },
        .output = .{ .vertices = &fragment_canceled_output },
    },
};

const rda_direction_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "merge_kind.rda.direction"),
    v(.{ .session = 50 }, "window.quorum"),
    v(.{ .worker = 0 }, "participants.direction_nonzero"),
};

pub fn generateFragmentExecutionCarrier(
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
) !OwnedGeneratedDecoupledCarrier {
    try common.validateCaseTraceBounds(bound, generated_cases.len, generated_cases.len);
    var task = try tasks.buildDecoupledFragmentTask(allocator);
    errdefer task.deinit();
    return generateForCases(allocator, task, bound, &generated_cases);
}

pub fn runExhaustiveTwoWorkerOneWindow(allocator: std.mem.Allocator, depth: usize) !ExplorationSummary {
    if (depth > 5) return error.DepthTooLarge;
    const alphabet = [_]model.Action{
        .{ .tag = .assign, .worker_id = 0 },
        .{ .tag = .assign, .worker_id = 1 },
        .{ .tag = .local_step_complete, .worker_id = 0 },
        .{ .tag = .local_step_complete, .worker_id = 1 },
        .{ .tag = .submit_fragment, .worker_id = 0 },
        .{ .tag = .submit_fragment, .worker_id = 1 },
        .{ .tag = .timeout_window },
        .{ .tag = .cancel_session },
        .{ .tag = .crash, .worker_id = 0 },
        .{ .tag = .restart, .worker_id = 0 },
        .{ .tag = .resume_worker, .worker_id = 0, .session_id = 7 },
    };

    var buffer: [5]model.Action = undefined;
    var summary = ExplorationSummary{};
    try exploreRecursive(allocator, &alphabet, &buffer, 0, depth, &summary);
    return summary;
}

pub fn runSeededExploration(
    allocator: std.mem.Allocator,
    seed: u64,
    trace_count: usize,
    depth: usize,
) !ExplorationSummary {
    const alphabet = [_]model.Action{
        .{ .tag = .assign, .worker_id = 0 },
        .{ .tag = .assign, .worker_id = 1 },
        .{ .tag = .assign, .worker_id = 2 },
        .{ .tag = .assign, .worker_id = 3 },
        .{ .tag = .local_step_complete, .worker_id = 0 },
        .{ .tag = .local_step_complete, .worker_id = 1 },
        .{ .tag = .local_step_complete, .worker_id = 2 },
        .{ .tag = .local_step_complete, .worker_id = 3 },
        .{ .tag = .submit_fragment, .worker_id = 0 },
        .{ .tag = .submit_fragment, .worker_id = 1 },
        .{ .tag = .submit_fragment, .worker_id = 2 },
        .{ .tag = .submit_fragment, .worker_id = 3 },
        .{ .tag = .timeout_window },
        .{ .tag = .cancel_session },
        .{ .tag = .crash, .worker_id = 3 },
        .{ .tag = .restart, .worker_id = 3 },
        .{ .tag = .leave, .worker_id = 2 },
        .{ .tag = .join, .worker_id = 2 },
        .{ .tag = .resume_worker, .worker_id = 3, .session_id = 11 },
        .{ .tag = .resume_worker, .worker_id = 3, .session_id = 99 },
    };

    var prng = std.rand.DefaultPrng.init(seed);
    var summary = ExplorationSummary{};

    var trace_index: usize = 0;
    while (trace_index < trace_count) : (trace_index += 1) {
        var site = try model.DecoupledSiteModel.init(allocator, .{
            .session_id = 11,
            .worker_count = 4,
            .min_quorum = 2,
            .num_fragments = 3,
        });
        defer site.deinit();

        var step: usize = 0;
        while (step < depth) : (step += 1) {
            const action = alphabet[prng.random().uintLessThan(usize, alphabet.len)];
            try site.apply(action);
            if (site.violation != .none) break;
        }
        summary.record(site.snapshot());
    }

    return summary;
}

pub fn injectedBugWitness(allocator: std.mem.Allocator, bug: InjectedBug) !BugWitness {
    switch (bug) {
        .below_quorum_commit => {
            var site = try model.DecoupledSiteModel.init(allocator, .{
                .session_id = 60,
                .worker_count = 2,
                .min_quorum = 2,
                .num_fragments = 1,
            });
            defer site.deinit();

            try site.assignWorker(0);
            try site.localStepComplete(0);
            try site.submitCurrent(0);
            try site.forceCommitWindow();

            const events = [_]Event{
                Event.init(.join, 0, null, 60),
                Event.init(.local_step_complete, 0, null, 60),
                Event.init(.deliver, 0, 0, 60),
                Event.init(.timeout, 0, null, 60),
            };
            return bugWitnessFromEvents(allocator, bug, site.snapshot().violation, &events);
        },
        .stale_vector_clock_acceptance => {
            var site = try model.DecoupledSiteModel.init(allocator, .{
                .session_id = 61,
                .worker_count = 1,
                .min_quorum = 1,
                .num_fragments = 1,
            });
            defer site.deinit();

            try site.assignWorker(0);
            try site.localStepComplete(0);
            var stale = currentQueuedResponse(&site, 0);
            stale.update.clock_counter = 0;
            _ = try site.collectQueuedFragmentResponses(&.{stale}, 1);

            const events = [_]Event{
                Event.init(.join, 0, null, 61),
                Event.init(.local_step_complete, 0, null, 61),
                Event.init(.deliver, 0, 0, 61),
            };
            return bugWitnessFromEvents(allocator, bug, site.snapshot().violation, &events);
        },
    }
}

fn generateForCases(
    allocator: std.mem.Allocator,
    task: tasks.OwnedProtocolTask,
    bound: common.ExtractionBound,
    cases: []const GeneratedCase,
) !OwnedGeneratedDecoupledCarrier {
    var protocol_builder = topology.ComplexBuilder.init(allocator);
    errdefer protocol_builder.deinit();
    for (cases) |case| {
        try protocol_builder.add(case.protocol);
    }

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
        var run = try runTraceCase(allocator, case.case);
        errdefer run.trace_value.deinit();

        const input_index = task.task().input.findSimplexIndex(case.input) orelse return error.GeneratedInputMissingFromTask;
        const protocol_index = protocol.complex().findSimplexIndex(case.protocol) orelse return error.GeneratedProtocolMissing;
        traces[index] = run.trace_value;
        initialized_traces += 1;
        try common.validateTraceWithinBound(traces[index], bound);
        witnesses[index] = .{
            .case = case.case,
            .input_index = input_index,
            .protocol_index = protocol_index,
            .event_count = traces[index].events.len,
            .decision = run.snapshot.decision,
            .violation = run.snapshot.violation,
            .accepted_updates = run.snapshot.accepted_updates,
            .committed_windows = run.snapshot.committed_windows,
            .tape_entries = run.snapshot.tape_entries,
            .state_hash = decoupledSnapshotHash(run.snapshot),
            .rda_bridge = run.committed_replay_present,
        };
        metadata[index] = common.sourceMetadata(
            "src/testing/protocol_topology/extract/decoupled_site.zig",
            fixtureForCase(case.case),
            traces[index].seed,
            bound,
            traceIdForCase(case.case),
            .generated_xi,
            task.definition.production_surface,
        );
    }

    var generated = OwnedGeneratedDecoupledCarrier{
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

fn fixtureForCase(case: DecoupledTraceCase) []const u8 {
    return switch (case) {
        .fragment_quorum_commit => "decoupled_site.fragment_quorum_commit",
        .below_quorum_timeout => "decoupled_site.below_quorum_timeout",
        .duplicate_queued_response => "decoupled_site.duplicate_queued_response",
        .context_payload_mismatch => "decoupled_site.context_payload_mismatch",
        .stale_vector_clock => "decoupled_site.stale_vector_clock",
        .post_cancel_update => "decoupled_site.post_cancel_update",
        .crash_restart_resume => "decoupled_site.crash_restart_resume",
        .incompatible_resume => "decoupled_site.incompatible_resume",
    };
}

fn traceIdForCase(case: DecoupledTraceCase) []const u8 {
    return switch (case) {
        .fragment_quorum_commit => "decoupled.fragment.quorum_commit",
        .below_quorum_timeout => "decoupled.fragment.below_quorum_timeout",
        .duplicate_queued_response => "decoupled.fragment.duplicate_queued_response",
        .context_payload_mismatch => "decoupled.fragment.context_payload_mismatch",
        .stale_vector_clock => "decoupled.fragment.stale_vector_clock",
        .post_cancel_update => "decoupled.fragment.post_cancel_update",
        .crash_restart_resume => "decoupled.recovery.compatible_resume",
        .incompatible_resume => "decoupled.recovery.incompatible_resume",
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

        try builder.add(face, .{
            .parent = protocol,
            .simplexes = image.simplexes,
        });
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
    if (!matched) return error.UncoveredGeneratedInputFace;

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

fn runTraceCase(allocator: std.mem.Allocator, case: DecoupledTraceCase) !CaseRun {
    return switch (case) {
        .fragment_quorum_commit => runQuorumCommit(allocator),
        .below_quorum_timeout => runBelowQuorumTimeout(allocator),
        .duplicate_queued_response => runDuplicateQueuedResponse(allocator),
        .context_payload_mismatch => runContextPayloadMismatch(allocator),
        .stale_vector_clock => runStaleVectorClock(allocator),
        .post_cancel_update => runPostCancelUpdate(allocator),
        .crash_restart_resume => runRecoveryTrace(allocator, true),
        .incompatible_resume => runRecoveryTrace(allocator, false),
    };
}

fn runQuorumCommit(allocator: std.mem.Allocator) !CaseRun {
    var site = try model.DecoupledSiteModel.init(allocator, .{
        .session_id = 40,
        .worker_count = 2,
        .min_quorum = 2,
        .num_fragments = 1,
    });
    defer site.deinit();

    try site.assignWorker(0);
    try site.assignWorker(1);
    try site.localStepComplete(0);
    try site.localStepComplete(1);
    const queue = [_]model.QueuedFragmentResponse{
        currentQueuedResponse(&site, 0),
        currentQueuedResponse(&site, 1),
    };
    const collection = try site.collectQueuedFragmentResponses(&queue, 2);
    try site.timeoutWindow();

    site.tape.resetReplay();
    const committed_replay_present = site.replayCommittedWindow(1, 0) != null;
    const events = [_]Event{
        Event.init(.join, 0, null, 40),
        Event.init(.join, 1, null, 40),
        Event.init(.local_step_complete, 0, null, 40),
        Event.init(.local_step_complete, 1, null, 40),
        Event.init(.deliver, 0, 0, 40),
        Event.init(.deliver, 1, 0, 40),
        Event.init(.timeout, 0, null, 40),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, caseSeed(.fragment_quorum_commit), &events, site.snapshot()),
        .snapshot = site.snapshot(),
        .collection = collection,
        .committed_replay_present = committed_replay_present,
    };
}

fn runBelowQuorumTimeout(allocator: std.mem.Allocator) !CaseRun {
    var site = try model.DecoupledSiteModel.init(allocator, .{
        .session_id = 41,
        .worker_count = 2,
        .min_quorum = 2,
        .num_fragments = 1,
    });
    defer site.deinit();

    try site.assignWorker(0);
    try site.localStepComplete(0);
    const queue = [_]model.QueuedFragmentResponse{currentQueuedResponse(&site, 0)};
    const collection = try site.collectQueuedFragmentResponses(&queue, 2);
    try site.timeoutWindow();

    const events = [_]Event{
        Event.init(.join, 0, null, 41),
        Event.init(.local_step_complete, 0, null, 41),
        Event.init(.deliver, 0, 0, 41),
        Event.init(.timeout, 0, null, 41),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, caseSeed(.below_quorum_timeout), &events, site.snapshot()),
        .snapshot = site.snapshot(),
        .collection = collection,
    };
}

fn runDuplicateQueuedResponse(allocator: std.mem.Allocator) !CaseRun {
    var site = try model.DecoupledSiteModel.init(allocator, .{
        .session_id = 42,
        .worker_count = 2,
        .min_quorum = 2,
        .num_fragments = 1,
    });
    defer site.deinit();

    try site.assignWorker(0);
    try site.assignWorker(1);
    try site.localStepComplete(0);
    const first = currentQueuedResponse(&site, 0);
    const queue = [_]model.QueuedFragmentResponse{ first, first };
    const collection = try site.collectQueuedFragmentResponses(&queue, 2);
    try site.timeoutWindow();

    const events = [_]Event{
        Event.init(.join, 0, null, 42),
        Event.init(.join, 1, null, 42),
        Event.init(.local_step_complete, 0, null, 42),
        Event.init(.deliver, 0, 0, 42),
        Event.init(.duplicate, 0, 0, 42),
        Event.init(.timeout, 0, null, 42),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, caseSeed(.duplicate_queued_response), &events, site.snapshot()),
        .snapshot = site.snapshot(),
        .collection = collection,
    };
}

fn runContextPayloadMismatch(allocator: std.mem.Allocator) !CaseRun {
    var site = try model.DecoupledSiteModel.init(allocator, .{
        .session_id = 43,
        .worker_count = 1,
        .min_quorum = 1,
        .num_fragments = 2,
    });
    defer site.deinit();

    try site.assignWorker(0);
    try site.localStepComplete(0);
    var mismatch = currentQueuedResponse(&site, 0);
    mismatch.update.fragment_id = 1;
    const collection = try site.collectQueuedFragmentResponses(&.{mismatch}, 1);

    const events = [_]Event{
        Event.init(.join, 0, null, 43),
        Event.init(.local_step_complete, 0, null, 43),
        Event.init(.deliver, 0, 0, 43),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, caseSeed(.context_payload_mismatch), &events, site.snapshot()),
        .snapshot = site.snapshot(),
        .collection = collection,
    };
}

fn runStaleVectorClock(allocator: std.mem.Allocator) !CaseRun {
    var site = try model.DecoupledSiteModel.init(allocator, .{
        .session_id = 44,
        .worker_count = 1,
        .min_quorum = 1,
        .num_fragments = 1,
    });
    defer site.deinit();

    try site.assignWorker(0);
    try site.localStepComplete(0);
    var stale = currentQueuedResponse(&site, 0);
    stale.update.clock_counter = 0;
    const collection = try site.collectQueuedFragmentResponses(&.{stale}, 1);

    const events = [_]Event{
        Event.init(.join, 0, null, 44),
        Event.init(.local_step_complete, 0, null, 44),
        Event.init(.deliver, 0, 0, 44),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, caseSeed(.stale_vector_clock), &events, site.snapshot()),
        .snapshot = site.snapshot(),
        .collection = collection,
    };
}

fn runPostCancelUpdate(allocator: std.mem.Allocator) !CaseRun {
    var site = try model.DecoupledSiteModel.init(allocator, .{
        .session_id = 45,
        .worker_count = 1,
        .min_quorum = 1,
        .num_fragments = 1,
    });
    defer site.deinit();

    try site.assignWorker(0);
    site.cancelSession();
    try site.submitCurrent(0);

    const events = [_]Event{
        Event.init(.join, 0, null, 45),
        Event.init(.cancel, 0, null, 45),
        Event.init(.deliver, 0, 0, 45),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, caseSeed(.post_cancel_update), &events, site.snapshot()),
        .snapshot = site.snapshot(),
    };
}

fn runRecoveryTrace(allocator: std.mem.Allocator, compatible: bool) !CaseRun {
    var site = try model.DecoupledSiteModel.init(allocator, .{
        .session_id = 46,
        .worker_count = 2,
        .min_quorum = 1,
        .num_fragments = 1,
    });
    defer site.deinit();

    try site.crashWorker(0);
    try site.restartWorker(0);
    try site.resumeWorker(0, if (compatible) 46 else 99);

    const case: DecoupledTraceCase = if (compatible) .crash_restart_resume else .incompatible_resume;
    const events = [_]Event{
        Event.init(.crash, 0, null, 46),
        Event.init(.restart, 0, null, 46),
        Event.init(.join, 0, null, if (compatible) 46 else 99),
    };
    return .{
        .trace_value = try ownedTraceFromEvents(allocator, caseSeed(case), &events, site.snapshot()),
        .snapshot = site.snapshot(),
    };
}

fn currentQueuedResponse(site: *const model.DecoupledSiteModel, sender_id: model.WorkerId) model.QueuedFragmentResponse {
    const worker = site.workers[sender_id];
    return .{
        .sender_id = sender_id,
        .context = .{
            .request_id = site.session_id,
            .round_id = @intCast(site.current_syncer_step),
            .task_id = @intCast(site.current_fragment + 1),
        },
        .update = .{
            .worker_id = sender_id,
            .session_id = site.session_id,
            .learner_id = sender_id,
            .fragment_id = site.current_fragment,
            .syncer_step = site.current_syncer_step,
            .learner_step = worker.local_step,
            .clock_counter = site.clock.counter(sender_id),
            .tokens_since_fragment_update = worker.tokens_since_update,
        },
    };
}

fn exploreRecursive(
    allocator: std.mem.Allocator,
    alphabet: []const model.Action,
    buffer: *[5]model.Action,
    cursor: usize,
    depth: usize,
    summary: *ExplorationSummary,
) !void {
    if (cursor == depth) {
        var site = try model.DecoupledSiteModel.init(allocator, .{
            .session_id = 7,
            .worker_count = 2,
            .min_quorum = 2,
            .num_fragments = 1,
        });
        defer site.deinit();

        for (buffer[0..depth]) |action| {
            try site.apply(action);
            if (site.violation != .none) break;
        }
        summary.record(site.snapshot());
        return;
    }

    for (alphabet) |action| {
        buffer[cursor] = action;
        try exploreRecursive(allocator, alphabet, buffer, cursor + 1, depth, summary);
    }
}

fn bugWitnessFromEvents(
    allocator: std.mem.Allocator,
    bug: InjectedBug,
    violation: model.Violation,
    events: []const Event,
) !BugWitness {
    return .{
        .allocator = allocator,
        .bug = bug,
        .violation = violation,
        .trace_value = .{
            .allocator = allocator,
            .seed = @intFromEnum(bug),
            .events = try allocator.dupe(Event, events),
            .final_snapshot = .{
                .steps = @intCast(events.len),
                .node_count = 2,
                .online_mask = 1,
                .checksum = @intFromEnum(violation),
            },
        },
    };
}

fn ownedTraceFromEvents(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    events: []const Event,
    snapshot_value: model.Snapshot,
) !trace.OwnedTrace {
    return .{
        .allocator = allocator,
        .seed = seed,
        .events = try allocator.dupe(Event, events),
        .final_snapshot = .{
            .steps = @intCast(events.len),
            .node_count = @intCast(@min(snapshot_value.worker_count + 1, 255)),
            .online_mask = onlineMask(snapshot_value),
            .checksum = decoupledSnapshotHash(snapshot_value),
        },
    };
}

fn onlineMask(snapshot_value: model.Snapshot) u64 {
    if (snapshot_value.worker_count >= 63) return std.math.maxInt(u64);
    return (@as(u64, 1) << @intCast(snapshot_value.worker_count + 1)) - 1;
}

fn decoupledSnapshotHash(snapshot_value: model.Snapshot) u64 {
    var hasher = std.hash.Wyhash.init(0x6469_6c6f_636f_7869);
    hashUsize(&hasher, snapshot_value.session_id);
    hashUsize(&hasher, snapshot_value.current_syncer_step);
    hashUsize(&hasher, snapshot_value.current_fragment);
    hashUsize(&hasher, snapshot_value.committed_windows);
    hashUsize(&hasher, snapshot_value.accepted_updates);
    hashUsize(&hasher, snapshot_value.worker_count);
    hashUsize(&hasher, snapshot_value.min_quorum);
    hashUsize(&hasher, if (snapshot_value.canceled) 1 else 0);
    hashUsize(&hasher, @intFromEnum(snapshot_value.decision));
    hashUsize(&hasher, @intFromEnum(snapshot_value.violation));
    hashUsize(&hasher, snapshot_value.tape_entries);
    return hasher.final();
}

fn hashUsize(hasher: anytype, value: usize) void {
    var buffer: [8]u8 = undefined;
    std.mem.writeInt(u64, &buffer, @intCast(value), .little);
    hasher.update(&buffer);
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
        if (witness.event_count == 0) return error.EmptyDecoupledTraceWitness;
    }
    for (seen) |item| {
        if (!item) return error.ProtocolSimplexMissingTraceWitness;
    }
}

fn vertex(color: topology.Color, label: []const u8) topology.Vertex {
    return .{ .color = color, .label = .{ .symbol = label } };
}

fn bitForEnum(value: anytype) u32 {
    return @as(u32, 1) << @intCast(@intFromEnum(value));
}

fn caseSeed(case: DecoupledTraceCase) trace.Seed {
    return @intFromEnum(case);
}

test "generated Decoupled DiLoCo fragment execution carrier validates against task carrier" {
    const bound = common.ExtractionBound{
        .max_cases = 6,
        .max_traces = 6,
        .max_events = 7,
        .max_depth = 7,
    };
    var generated = try generateFragmentExecutionCarrier(std.testing.allocator, bound);
    defer generated.deinit();

    try std.testing.expectEqual(@as(usize, generated_cases.len), generated.protocol.simplexes.len);
    try std.testing.expectEqual(@as(usize, generated_cases.len), generated.witnesses.len);
    try std.testing.expectEqual(model.Decision.merge_committed, generated.witnesses[0].decision);
    try std.testing.expectEqual(@as(usize, 1), generated.witnesses[0].committed_windows);
    try std.testing.expect(generated.witnesses[0].rda_bridge);
    try std.testing.expectEqual(model.Decision.merge_deferred, generated.witnesses[1].decision);
    try std.testing.expectEqual(model.Violation.context_payload_mismatch, generated.witnesses[3].violation);
    try std.testing.expectEqual(model.Violation.stale_vector_clock, generated.witnesses[4].violation);
    try std.testing.expectEqual(model.Violation.post_cancel_update, generated.witnesses[5].violation);
    try std.testing.expectEqualStrings("decoupled.fragment.quorum_commit", generated.metadata[0].trace_id);
    try std.testing.expectEqual(tasks.CoverageEvidence.generated_xi, generated.metadata[0].evidence);
}

test "committed fragment trace exposes event-tape replay bridge into RDA task" {
    const bound = common.ExtractionBound{
        .max_cases = 6,
        .max_traces = 6,
        .max_events = 7,
        .max_depth = 7,
    };
    var generated = try generateFragmentExecutionCarrier(std.testing.allocator, bound);
    defer generated.deinit();

    const bridge = try generated.rdaBridge(std.testing.allocator);
    try std.testing.expectEqual(DecoupledTraceCase.fragment_quorum_commit, bridge.decoupled_case);
    try std.testing.expectEqual(@as(usize, 0), bridge.decoupled_protocol_index);
    try std.testing.expectEqual(@as(usize, 0), bridge.rda_input_index);
    try std.testing.expect(bridge.replay_state_hash != 0);
}

test "bounded Decoupled exploration covers updates cancellation resume and failures" {
    const exhaustive = try runExhaustiveTwoWorkerOneWindow(std.testing.allocator, 3);
    try std.testing.expectEqual(@as(usize, 1331), exhaustive.checked);
    try std.testing.expect(exhaustive.sawDecision(.assigned));
    try std.testing.expect(exhaustive.sawDecision(.update_accepted));
    try std.testing.expect(exhaustive.sawDecision(.update_rejected));
    try std.testing.expect(exhaustive.sawDecision(.canceled));
    try std.testing.expect(exhaustive.sawDecision(.resumable));
    try std.testing.expect(exhaustive.sawViolation(.unassigned_worker));
    try std.testing.expect(exhaustive.sawViolation(.post_cancel_update));

    const left = try runSeededExploration(std.testing.allocator, 0xdec0, 64, 18);
    const right = try runSeededExploration(std.testing.allocator, 0xdec0, 64, 18);
    try std.testing.expect(ExplorationSummary.eql(left, right));
    try std.testing.expect(left.failed_states > 0);
    try std.testing.expect(left.sawViolation(.incompatible_resume) or left.sawViolation(.unassigned_worker));
}

test "recovery traces distinguish compatible and incompatible resumes" {
    var compatible = try runTraceCase(std.testing.allocator, .crash_restart_resume);
    defer compatible.trace_value.deinit();
    try std.testing.expectEqual(model.Decision.resumable, compatible.snapshot.decision);
    try std.testing.expectEqual(model.Violation.none, compatible.snapshot.violation);
    try std.testing.expect(compatible.snapshot.tape_entries >= 2);

    var incompatible = try runTraceCase(std.testing.allocator, .incompatible_resume);
    defer incompatible.trace_value.deinit();
    try std.testing.expectEqual(model.Decision.update_rejected, incompatible.snapshot.decision);
    try std.testing.expectEqual(model.Violation.incompatible_resume, incompatible.snapshot.violation);
}

test "injected Decoupled bugs produce compact replay witnesses" {
    var below_quorum = try injectedBugWitness(std.testing.allocator, .below_quorum_commit);
    defer below_quorum.deinit();
    try std.testing.expectEqual(model.Violation.below_quorum_commit, below_quorum.violation);
    try std.testing.expect(below_quorum.trace_value.events.len <= 4);

    var stale_clock = try injectedBugWitness(std.testing.allocator, .stale_vector_clock_acceptance);
    defer stale_clock.deinit();
    try std.testing.expectEqual(model.Violation.stale_vector_clock, stale_clock.violation);
    try std.testing.expect(stale_clock.trace_value.events.len <= 3);
}
