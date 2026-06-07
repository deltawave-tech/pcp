const std = @import("std");

const common = @import("common.zig");
const scheduling_oracle = @import("../oracles/scheduling.zig");
const model = @import("../models/gateway_scheduling.zig");
const tasks = @import("../tasks.zig");
const topology = @import("../topology.zig");
const trace = @import("../trace.zig");

const Event = trace.Event;

pub const SchedulingTraceCase = enum {
    lease_available,
    lease_maxed,
    lease_cross_owner_reserved,
    queue_admission,
    reservation_commit,
    reservation_release,
    reservation_expire,
    reservation_cancel,
    active_start,
    active_complete,
    worker_loss_failed,
};

pub const InjectedBug = enum {
    duplicate_worker_reservation,
    assignment_after_worker_loss,
    cancellation_cleanup_leak,
};

pub const TraceWitness = struct {
    case: SchedulingTraceCase,
    input_index: usize,
    protocol_index: usize,
    event_count: usize,
    decision: model.Decision,
    violation: model.Violation,
    state_hash: u64,
};

pub const ExplorationSummary = struct {
    checked: usize = 0,
    decision_mask: u32 = 0,
    violation_mask: u32 = 0,
    failed_states: usize = 0,

    pub fn record(self: *ExplorationSummary, snapshot_value: model.Snapshot) void {
        self.checked += 1;
        self.decision_mask |= bitForEnum(snapshot_value.decision);
        self.violation_mask |= bitForEnum(snapshot_value.violation);
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
            a.failed_states == b.failed_states;
    }
};

pub const BugWitness = struct {
    allocator: std.mem.Allocator,
    bug: InjectedBug,
    violation: scheduling_oracle.Violation,
    trace_value: trace.OwnedTrace,

    pub fn deinit(self: *BugWitness) void {
        self.trace_value.deinit();
        self.* = undefined;
    }
};

pub const OwnedGeneratedSchedulingCarrier = struct {
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
    task: tasks.OwnedProtocolTask,
    protocol: topology.OwnedComplex,
    execution: topology.OwnedSimplicialCarrierMap,
    decision_entries: []topology.VertexMapEntry,
    traces: []trace.OwnedTrace,
    witnesses: []TraceWitness,
    metadata: []common.SourceMetadata,

    pub fn protocolComplex(self: *const OwnedGeneratedSchedulingCarrier) topology.Complex {
        return self.protocol.complex();
    }

    pub fn executionCarrier(self: *const OwnedGeneratedSchedulingCarrier) topology.SimplicialCarrierMap {
        return self.execution.carrierMap();
    }

    pub fn decisionMap(self: *const OwnedGeneratedSchedulingCarrier) topology.VertexMap {
        return .{
            .domain = self.protocol.complex(),
            .codomain = self.task.task().output,
            .entries = self.decision_entries,
        };
    }

    pub fn validate(self: *const OwnedGeneratedSchedulingCarrier) !void {
        const task_value = self.task.task();
        try self.execution.carrierMap().validate(self.allocator, .{
            .require_monotone = true,
            .require_chromatic = true,
        });
        try topology.validateCarriedDecisionMap(
            self.allocator,
            self.execution.carrierMap(),
            task_value.carrier,
            self.decisionMap(),
        );
        try validateWitnessesCoverProtocol(self.allocator, self.protocol.complex(), self.witnesses);
        try common.validateMetadataSet(self.metadata, self.traces.len);
        for (self.traces) |trace_value| try common.validateTraceWithinBound(trace_value, self.bound);
    }

    pub fn deinit(self: *OwnedGeneratedSchedulingCarrier) void {
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
    case: SchedulingTraceCase,
    input: topology.Simplex,
    protocol: topology.Simplex,
    output: topology.Simplex,
};

const CaseRun = struct {
    trace_value: trace.OwnedTrace,
    snapshot: model.Snapshot,
};

const v = vertex;

const lease_available_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "pool.idle_capacity"),
    v(.{ .job = 10 }, "lease.training.request"),
    v(.{ .worker = 0 }, "worker.cuda.idle.available"),
};
const lease_maxed_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "policy.training.max_workers_reached"),
    v(.{ .job = 11 }, "lease.training.request"),
    v(.{ .worker = 0 }, "worker.cuda.idle.maxed"),
};
const lease_cross_owner_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "policy.inference.reserved_workers"),
    v(.{ .job = 12 }, "lease.training.request"),
    v(.{ .worker = 0 }, "worker.cuda.last_inference_capacity"),
};
const lease_available_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.lease.reserved"),
    v(.{ .job = 10 }, "view.reservation.active"),
    v(.{ .worker = 0 }, "view.worker.leased.training"),
};
const lease_maxed_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.lease.rejected.max_workers"),
    v(.{ .job = 11 }, "view.reservation.rejected.maxed"),
    v(.{ .worker = 0 }, "view.worker.unleased.maxed"),
};
const lease_cross_owner_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.lease.deferred.preserve_capacity"),
    v(.{ .job = 12 }, "view.reservation.deferred.capacity"),
    v(.{ .worker = 0 }, "view.worker.reserved_for_inference"),
};
const lease_available_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "lease.reserved"),
    v(.{ .job = 10 }, "reservation.active"),
    v(.{ .worker = 0 }, "worker.leased.training"),
};
const lease_maxed_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "lease.rejected.max_workers"),
    v(.{ .job = 11 }, "reservation.rejected"),
    v(.{ .worker = 0 }, "worker.unleased"),
};
const lease_cross_owner_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "lease.deferred.preserve_capacity"),
    v(.{ .job = 12 }, "reservation.deferred"),
    v(.{ .worker = 0 }, "worker.reserved_for_inference"),
};
const lease_cases = [_]GeneratedCase{
    .{
        .case = .lease_available,
        .input = .{ .vertices = &lease_available_input },
        .protocol = .{ .vertices = &lease_available_protocol },
        .output = .{ .vertices = &lease_available_output },
    },
    .{
        .case = .lease_maxed,
        .input = .{ .vertices = &lease_maxed_input },
        .protocol = .{ .vertices = &lease_maxed_protocol },
        .output = .{ .vertices = &lease_maxed_output },
    },
    .{
        .case = .lease_cross_owner_reserved,
        .input = .{ .vertices = &lease_cross_owner_input },
        .protocol = .{ .vertices = &lease_cross_owner_protocol },
        .output = .{ .vertices = &lease_cross_owner_output },
    },
};

const reservation_commit_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "reservation.active"),
    v(.{ .job = 20 }, "reservation.commit"),
    v(.{ .worker = 0 }, "worker.reserved.training"),
};
const reservation_release_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "reservation.active.release"),
    v(.{ .job = 21 }, "reservation.release"),
    v(.{ .worker = 0 }, "worker.reserved.training.release"),
};
const reservation_expire_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "reservation.active.expire"),
    v(.{ .job = 22 }, "reservation.expire"),
    v(.{ .worker = 0 }, "worker.reserved.training.expire"),
};
const reservation_cancel_input = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "queue.reserved_job"),
    v(.{ .job = 23 }, "reservation.cancel"),
    v(.{ .worker = 0 }, "worker.leased.queued_job"),
};
const reservation_commit_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.reservation.committed.to_queue"),
    v(.{ .job = 20 }, "view.job.queued_reserved"),
    v(.{ .worker = 0 }, "view.worker.leased.after_commit"),
};
const reservation_release_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.reservation.released"),
    v(.{ .job = 21 }, "view.job.not_queued.release"),
    v(.{ .worker = 0 }, "view.worker.idle.release"),
};
const reservation_expire_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.reservation.expired"),
    v(.{ .job = 22 }, "view.job.not_queued.expire"),
    v(.{ .worker = 0 }, "view.worker.idle.expire"),
};
const reservation_cancel_protocol = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "view.reservation.canceled.cleanup_complete"),
    v(.{ .job = 23 }, "view.job.canceled"),
    v(.{ .worker = 0 }, "view.worker.idle.cancel"),
};
const reservation_commit_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "reservation.committed"),
    v(.{ .job = 20 }, "job.queued_reserved"),
    v(.{ .worker = 0 }, "worker.leased.training"),
};
const reservation_release_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "reservation.released"),
    v(.{ .job = 21 }, "job.not_queued"),
    v(.{ .worker = 0 }, "worker.idle"),
};
const reservation_expire_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "reservation.expired"),
    v(.{ .job = 22 }, "job.not_queued"),
    v(.{ .worker = 0 }, "worker.idle"),
};
const reservation_cancel_output = [_]topology.Vertex{
    v(.{ .gateway = 0 }, "reservation.canceled.cleanup_complete"),
    v(.{ .job = 23 }, "job.canceled"),
    v(.{ .worker = 0 }, "worker.idle"),
};
const reservation_cases = [_]GeneratedCase{
    .{
        .case = .reservation_commit,
        .input = .{ .vertices = &reservation_commit_input },
        .protocol = .{ .vertices = &reservation_commit_protocol },
        .output = .{ .vertices = &reservation_commit_output },
    },
    .{
        .case = .reservation_release,
        .input = .{ .vertices = &reservation_release_input },
        .protocol = .{ .vertices = &reservation_release_protocol },
        .output = .{ .vertices = &reservation_release_output },
    },
    .{
        .case = .reservation_expire,
        .input = .{ .vertices = &reservation_expire_input },
        .protocol = .{ .vertices = &reservation_expire_protocol },
        .output = .{ .vertices = &reservation_expire_output },
    },
    .{
        .case = .reservation_cancel,
        .input = .{ .vertices = &reservation_cancel_input },
        .protocol = .{ .vertices = &reservation_cancel_protocol },
        .output = .{ .vertices = &reservation_cancel_output },
    },
};

pub fn generateLeaseExecutionCarrier(
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
) !OwnedGeneratedSchedulingCarrier {
    try common.validateCaseTraceBounds(bound, lease_cases.len, lease_cases.len);
    var task = try tasks.buildGatewayLeaseSchedulingTask(allocator);
    errdefer task.deinit();
    return generateForCases(allocator, task, bound, &lease_cases);
}

pub fn generateReservationExecutionCarrier(
    allocator: std.mem.Allocator,
    bound: common.ExtractionBound,
) !OwnedGeneratedSchedulingCarrier {
    try common.validateCaseTraceBounds(bound, reservation_cases.len, reservation_cases.len);
    var task = try tasks.buildReservationLifecycleTask(allocator);
    errdefer task.deinit();
    return generateForCases(allocator, task, bound, &reservation_cases);
}

pub fn runExhaustiveSmallExploration() !ExplorationSummary {
    const action_count = std.meta.fields(SchedulingTraceCase).len;
    var summary = ExplorationSummary{};

    inline for (std.meta.fields(SchedulingTraceCase)) |first_field| {
        inline for (std.meta.fields(SchedulingTraceCase)) |second_field| {
            var scheduling = try initialExplorationModel();
            try applyExplorationCase(&scheduling, @as(SchedulingTraceCase, @enumFromInt(first_field.value)));
            try applyExplorationCase(&scheduling, @as(SchedulingTraceCase, @enumFromInt(second_field.value)));
            summary.record(scheduling.snapshot());
        }
    }

    if (summary.checked != action_count * action_count) return error.ExhaustiveSchedulingCoverageMismatch;
    return summary;
}

pub fn runSeededExploration(seed: u64, trace_count: usize, depth: usize) !ExplorationSummary {
    var rng = std.rand.DefaultPrng.init(seed);
    const action_count = std.meta.fields(SchedulingTraceCase).len;
    var summary = ExplorationSummary{};

    var trace_index: usize = 0;
    while (trace_index < trace_count) : (trace_index += 1) {
        var scheduling = try initialExplorationModel();
        var step: usize = 0;
        while (step < depth) : (step += 1) {
            const action = @as(SchedulingTraceCase, @enumFromInt(rng.random().uintLessThan(usize, action_count)));
            try applyExplorationCase(&scheduling, action);
        }
        summary.record(scheduling.snapshot());
    }

    return summary;
}

pub fn injectedBugWitness(allocator: std.mem.Allocator, bug: InjectedBug) !BugWitness {
    switch (bug) {
        .duplicate_worker_reservation => {
            var left = model.Reservation{
                .id = 1,
                .job_id = 70,
                .workers_required = 1,
                .worker_count = 1,
            };
            left.worker_ids[0] = 0;
            var right = model.Reservation{
                .id = 2,
                .job_id = 71,
                .workers_required = 1,
                .worker_count = 1,
            };
            right.worker_ids[0] = 0;

            const events = [_]Event{
                Event.init(.lease_reserve, 0, 0, 70),
                Event.init(.duplicate, 0, 0, 71),
            };
            return bugWitnessFromEvents(
                allocator,
                bug,
                scheduling_oracle.checkNoDuplicateActiveWorkers(&.{ left, right }),
                &events,
            );
        },
        .assignment_after_worker_loss => {
            var scheduling = model.GatewaySchedulingModel.init();
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            const reserved = (try scheduling.reserve(model.makeRequirement(72, 1, .cuda, "sm_80"))).?;
            scheduling.loseWorker(0);
            _ = scheduling.commitReservation(reserved.id);

            const events = [_]Event{
                Event.init(.lease_reserve, 0, 0, 72),
                Event.init(.crash, 0, null, 0),
                Event.init(.lease_commit, 0, 0, 72),
            };
            return bugWitnessFromEvents(
                allocator,
                bug,
                scheduling_oracle.checkAssignmentResult(scheduling.snapshot()),
                &events,
            );
        },
        .cancellation_cleanup_leak => {
            var scheduling = model.GatewaySchedulingModel.init();
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            scheduling.workers[0].state = .reserved;
            scheduling.workers[0].lease_owner = .training;
            scheduling.workers[0].job_id = 73;

            const events = [_]Event{
                Event.init(.lease_reserve, 0, 0, 73),
                Event.init(.cancel, 0, 0, 73),
            };
            return bugWitnessFromEvents(
                allocator,
                bug,
                scheduling_oracle.checkCanceledJobCleanup(scheduling, 73),
                &events,
            );
        },
    }
}

fn generateForCases(
    allocator: std.mem.Allocator,
    task: tasks.OwnedProtocolTask,
    bound: common.ExtractionBound,
    cases: []const GeneratedCase,
) !OwnedGeneratedSchedulingCarrier {
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
            .state_hash = schedulingSnapshotHash(run.snapshot),
        };
        metadata[index] = common.sourceMetadata(
            "src/testing/protocol_topology/extract/scheduling.zig",
            fixtureForCase(case.case),
            traces[index].seed,
            bound,
            traceIdForCase(case.case),
            .generated_xi,
            task.definition.production_surface,
        );
    }

    var generated = OwnedGeneratedSchedulingCarrier{
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

fn fixtureForCase(case: SchedulingTraceCase) []const u8 {
    return switch (case) {
        .lease_available => "gateway_scheduling.lease_available",
        .lease_maxed => "gateway_scheduling.lease_maxed",
        .lease_cross_owner_reserved => "gateway_scheduling.cross_owner_reserved",
        .queue_admission => "gateway_scheduling.queue_admission",
        .reservation_commit => "gateway_scheduling.reservation_commit",
        .reservation_release => "gateway_scheduling.reservation_release",
        .reservation_expire => "gateway_scheduling.reservation_expire",
        .reservation_cancel => "gateway_scheduling.reservation_cancel",
        .active_start => "gateway_scheduling.active_start",
        .active_complete => "gateway_scheduling.active_complete",
        .worker_loss_failed => "gateway_scheduling.worker_loss_failed",
    };
}

fn traceIdForCase(case: SchedulingTraceCase) []const u8 {
    return switch (case) {
        .lease_available => "scheduling.lease.available",
        .lease_maxed => "scheduling.lease.maxed",
        .lease_cross_owner_reserved => "scheduling.lease.cross_owner_reserved",
        .queue_admission => "scheduling.queue.admission",
        .reservation_commit => "reservation.lifecycle.commit",
        .reservation_release => "reservation.lifecycle.release",
        .reservation_expire => "reservation.lifecycle.expire",
        .reservation_cancel => "reservation.lifecycle.cancel",
        .active_start => "scheduling.active.start",
        .active_complete => "scheduling.active.complete",
        .worker_loss_failed => "scheduling.worker_loss.failed",
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

fn runTraceCase(allocator: std.mem.Allocator, case: SchedulingTraceCase) !CaseRun {
    var scheduling = model.GatewaySchedulingModel.init();
    var events: []const Event = &.{};

    switch (case) {
        .lease_available => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            _ = try scheduling.reserve(model.makeRequirement(10, 1, .cuda, "sm_80"));
            events = &[_]Event{Event.init(.lease_reserve, 0, 0, 10)};
        },
        .lease_maxed => {
            scheduling.setSchedulingPolicy(.training, .{
                .worker_class = .cuda,
                .target_arch = "sm_80",
                .max_workers = 1,
            });
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            _ = try scheduling.reserve(model.makeRequirement(1, 1, .cuda, "sm_80"));
            _ = try scheduling.reserve(model.makeRequirement(11, 1, .cuda, "sm_80"));
            events = &[_]Event{
                Event.init(.lease_reserve, 0, 0, 1),
                Event.init(.lease_reserve, 0, 0, 11),
            };
        },
        .lease_cross_owner_reserved => {
            scheduling.setSchedulingPolicy(.inference, .{
                .worker_class = .cuda,
                .target_arch = "sm_80",
                .reserved_workers = 1,
            });
            scheduling.setSchedulingPolicy(.training, .{
                .worker_class = .cuda,
                .target_arch = "sm_80",
            });
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            try scheduling.addWorker(model.makeWorker(1, .cuda, "sm_80"));
            _ = try scheduling.reserve(model.makeRequirement(12, 2, .cuda, "sm_80"));
            events = &[_]Event{Event.init(.lease_reserve, 0, 0, 12)};
        },
        .queue_admission => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            _ = try scheduling.enqueueJob(model.makeRequirement(30, 1, .cuda, "sm_80"));
            events = &[_]Event{Event.init(.send, 0, null, 30)};
        },
        .reservation_commit => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            const reserved = (try scheduling.reserve(model.makeRequirement(20, 1, .cuda, "sm_80"))).?;
            _ = scheduling.commitReservationToQueue(reserved.id);
            events = &[_]Event{
                Event.init(.lease_reserve, 0, 0, 20),
                Event.init(.lease_commit, 0, 0, 20),
            };
        },
        .reservation_release => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            const reserved = (try scheduling.reserve(model.makeRequirement(21, 1, .cuda, "sm_80"))).?;
            _ = scheduling.releaseReservation(reserved.id);
            events = &[_]Event{
                Event.init(.lease_reserve, 0, 0, 21),
                Event.init(.lease_release, 0, 0, 21),
            };
        },
        .reservation_expire => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            const reserved = (try scheduling.reserve(model.makeRequirement(22, 1, .cuda, "sm_80"))).?;
            _ = scheduling.expireReservation(reserved.id);
            events = &[_]Event{
                Event.init(.lease_reserve, 0, 0, 22),
                Event.init(.lease_expire, 0, 0, 22),
            };
        },
        .reservation_cancel => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            const reserved = (try scheduling.reserve(model.makeRequirement(23, 1, .cuda, "sm_80"))).?;
            _ = scheduling.commitReservationToQueue(reserved.id);
            _ = scheduling.cancelQueuedJob(23);
            events = &[_]Event{
                Event.init(.lease_reserve, 0, 0, 23),
                Event.init(.lease_commit, 0, 0, 23),
                Event.init(.cancel, 0, 0, 23),
            };
        },
        .active_start => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            _ = try scheduling.enqueueJob(model.makeRequirement(31, 1, .cuda, "sm_80"));
            _ = scheduling.startNextQueuedJob();
            events = &[_]Event{
                Event.init(.send, 0, null, 31),
                Event.init(.deliver, 0, 0, 31),
            };
        },
        .active_complete => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            _ = try scheduling.enqueueJob(model.makeRequirement(32, 1, .cuda, "sm_80"));
            _ = scheduling.startNextQueuedJob();
            _ = scheduling.completeActiveJob(32);
            events = &[_]Event{
                Event.init(.send, 0, null, 32),
                Event.init(.deliver, 0, 0, 32),
                Event.init(.local_step_complete, 0, 0, 32),
            };
        },
        .worker_loss_failed => {
            try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
            const reserved = (try scheduling.reserve(model.makeRequirement(33, 1, .cuda, "sm_80"))).?;
            scheduling.loseWorker(0);
            _ = scheduling.commitReservation(reserved.id);
            events = &[_]Event{
                Event.init(.lease_reserve, 0, 0, 33),
                Event.init(.crash, 0, null, 0),
                Event.init(.lease_commit, 0, 0, 33),
            };
        },
    }

    return .{
        .trace_value = try ownedTraceFromEvents(allocator, case, events, scheduling.snapshot()),
        .snapshot = scheduling.snapshot(),
    };
}

fn initialExplorationModel() !model.GatewaySchedulingModel {
    var scheduling = model.GatewaySchedulingModel.init();
    scheduling.setSchedulingPolicy(.inference, .{ .worker_class = .cuda, .target_arch = "sm_80" });
    scheduling.setSchedulingPolicy(.training, .{ .worker_class = .cuda, .target_arch = "sm_80" });
    scheduling.setSchedulingPolicy(.rl, .{ .worker_class = .cuda, .target_arch = "sm_80" });
    try scheduling.addWorker(model.makeWorker(0, .cuda, "sm_80"));
    try scheduling.addWorker(model.makeWorker(1, .cuda, "sm_80"));
    return scheduling;
}

fn applyExplorationCase(scheduling: *model.GatewaySchedulingModel, case: SchedulingTraceCase) !void {
    switch (case) {
        .lease_available => {
            _ = try scheduling.reserve(model.makeRequirement(10, 1, .cuda, "sm_80"));
        },
        .lease_maxed => {
            scheduling.setSchedulingPolicy(.training, .{
                .worker_class = .cuda,
                .target_arch = "sm_80",
                .max_workers = 1,
            });
            _ = try scheduling.reserve(model.makeRequirement(11, 1, .cuda, "sm_80"));
        },
        .lease_cross_owner_reserved => {
            scheduling.setSchedulingPolicy(.inference, .{
                .worker_class = .cuda,
                .target_arch = "sm_80",
                .reserved_workers = 1,
            });
            _ = try scheduling.reserve(model.makeRequirement(12, 2, .cuda, "sm_80"));
        },
        .queue_admission => {
            _ = try scheduling.enqueueJob(model.makeRequirement(30, 1, .cuda, "sm_80"));
        },
        .reservation_commit => {
            const reserved = (try scheduling.reserve(model.makeRequirement(20, 1, .cuda, "sm_80"))) orelse return;
            _ = scheduling.commitReservationToQueue(reserved.id);
        },
        .reservation_release => {
            const reserved = (try scheduling.reserve(model.makeRequirement(21, 1, .cuda, "sm_80"))) orelse return;
            _ = scheduling.releaseReservation(reserved.id);
        },
        .reservation_expire => {
            const reserved = (try scheduling.reserve(model.makeRequirement(22, 1, .cuda, "sm_80"))) orelse return;
            _ = scheduling.expireReservation(reserved.id);
        },
        .reservation_cancel => {
            const reserved = (try scheduling.reserve(model.makeRequirement(23, 1, .cuda, "sm_80"))) orelse return;
            _ = scheduling.commitReservationToQueue(reserved.id);
            _ = scheduling.cancelQueuedJob(23);
        },
        .active_start => {
            _ = try scheduling.enqueueJob(model.makeRequirement(31, 1, .cuda, "sm_80"));
            _ = scheduling.startNextQueuedJob();
        },
        .active_complete => {
            _ = try scheduling.enqueueJob(model.makeRequirement(32, 1, .cuda, "sm_80"));
            _ = scheduling.startNextQueuedJob();
            _ = scheduling.completeActiveJob(32);
        },
        .worker_loss_failed => {
            const reserved = (try scheduling.reserve(model.makeRequirement(33, 1, .cuda, "sm_80"))) orelse return;
            scheduling.loseWorker(0);
            _ = scheduling.commitReservation(reserved.id);
        },
    }
}

fn bugWitnessFromEvents(
    allocator: std.mem.Allocator,
    bug: InjectedBug,
    violation: scheduling_oracle.Violation,
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
    case: SchedulingTraceCase,
    events: []const Event,
    snapshot_value: model.Snapshot,
) !trace.OwnedTrace {
    return .{
        .allocator = allocator,
        .seed = @intFromEnum(case),
        .events = try allocator.dupe(Event, events),
        .final_snapshot = .{
            .steps = @intCast(events.len),
            .node_count = @intCast(@min(snapshot_value.workers_total + 1, 255)),
            .online_mask = onlineMask(snapshot_value),
            .checksum = schedulingSnapshotHash(snapshot_value),
        },
    };
}

fn onlineMask(snapshot_value: model.Snapshot) u64 {
    if (snapshot_value.workers_total >= 63) return std.math.maxInt(u64);
    return (@as(u64, 1) << @intCast(snapshot_value.workers_total + 1)) - 1;
}

fn schedulingSnapshotHash(snapshot_value: model.Snapshot) u64 {
    var hasher = std.hash.Wyhash.init(0x7363_6865_645f_7869);
    hashUsize(&hasher, snapshot_value.workers_total);
    hashUsize(&hasher, snapshot_value.idle_workers);
    hashUsize(&hasher, snapshot_value.reserved_workers);
    hashUsize(&hasher, snapshot_value.assigned_workers);
    hashUsize(&hasher, snapshot_value.active_reservations);
    hashUsize(&hasher, snapshot_value.queued_jobs);
    hashUsize(&hasher, snapshot_value.active_jobs);
    hashUsize(&hasher, snapshot_value.canceled_jobs);
    for (snapshot_value.owner_leases) |value| hashUsize(&hasher, value);
    for (snapshot_value.owner_available) |value| hashUsize(&hasher, value);
    hashUsize(&hasher, @intFromEnum(snapshot_value.decision));
    hashUsize(&hasher, @intFromEnum(snapshot_value.violation));
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
        if (witness.event_count == 0) return error.EmptySchedulingTraceWitness;
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

test "generated lease scheduling execution carrier validates against task carrier" {
    const bound = common.ExtractionBound{
        .max_cases = 3,
        .max_traces = 3,
        .max_events = 2,
        .max_depth = 2,
    };
    var generated = try generateLeaseExecutionCarrier(std.testing.allocator, bound);
    defer generated.deinit();

    try std.testing.expectEqual(@as(usize, lease_cases.len), generated.protocol.simplexes.len);
    try std.testing.expectEqual(@as(usize, lease_cases.len), generated.witnesses.len);
    try std.testing.expectEqual(model.Decision.reserved, generated.witnesses[0].decision);
    try std.testing.expectEqual(model.Decision.rejected, generated.witnesses[1].decision);
    try std.testing.expectEqual(model.Violation.reservation_unavailable, generated.witnesses[2].violation);
    try std.testing.expectEqualStrings("scheduling.lease.available", generated.metadata[0].trace_id);
    try std.testing.expectEqual(tasks.CoverageEvidence.generated_xi, generated.metadata[0].evidence);
}

test "generated reservation execution carrier validates against task carrier" {
    const bound = common.ExtractionBound{
        .max_cases = 4,
        .max_traces = 4,
        .max_events = 3,
        .max_depth = 3,
    };
    var generated = try generateReservationExecutionCarrier(std.testing.allocator, bound);
    defer generated.deinit();

    try std.testing.expectEqual(@as(usize, reservation_cases.len), generated.protocol.simplexes.len);
    try std.testing.expectEqual(@as(usize, reservation_cases.len), generated.witnesses.len);
    try std.testing.expectEqual(model.Decision.queued, generated.witnesses[0].decision);
    try std.testing.expectEqual(model.Decision.released, generated.witnesses[1].decision);
    try std.testing.expectEqual(model.Decision.expired, generated.witnesses[2].decision);
    try std.testing.expectEqual(model.Decision.canceled, generated.witnesses[3].decision);
    try std.testing.expectEqualStrings("reservation.lifecycle.commit", generated.metadata[0].trace_id);
}

test "bounded scheduling exploration sees queue start complete and failure states" {
    const exhaustive = try runExhaustiveSmallExploration();
    try std.testing.expect(exhaustive.checked > 0);
    try std.testing.expect(exhaustive.sawDecision(.reserved));
    try std.testing.expect(exhaustive.sawDecision(.queued));
    try std.testing.expect(exhaustive.sawDecision(.started));
    try std.testing.expect(exhaustive.sawDecision(.completed));
    try std.testing.expect(exhaustive.sawDecision(.canceled));
    try std.testing.expect(exhaustive.sawDecision(.rejected));
    try std.testing.expect(exhaustive.sawViolation(.assignment_after_worker_loss));

    const left = try runSeededExploration(0x5151, 64, 4);
    const right = try runSeededExploration(0x5151, 64, 4);
    try std.testing.expect(ExplorationSummary.eql(left, right));
    try std.testing.expect(left.failed_states > 0);
}

test "injected scheduling bugs produce compact replay witnesses" {
    var duplicate = try injectedBugWitness(std.testing.allocator, .duplicate_worker_reservation);
    defer duplicate.deinit();
    try std.testing.expectEqual(scheduling_oracle.Violation.duplicate_worker_reservation, duplicate.violation);
    try std.testing.expect(duplicate.trace_value.events.len <= 2);

    var worker_loss = try injectedBugWitness(std.testing.allocator, .assignment_after_worker_loss);
    defer worker_loss.deinit();
    try std.testing.expectEqual(scheduling_oracle.Violation.assignment_after_worker_loss, worker_loss.violation);
    try std.testing.expect(worker_loss.trace_value.events.len <= 3);

    var cleanup = try injectedBugWitness(std.testing.allocator, .cancellation_cleanup_leak);
    defer cleanup.deinit();
    try std.testing.expectEqual(scheduling_oracle.Violation.cancellation_cleanup_leak, cleanup.violation);
    try std.testing.expect(cleanup.trace_value.events.len <= 2);
}
