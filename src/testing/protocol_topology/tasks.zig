const std = @import("std");

const message = @import("../../network/message.zig");
const topology = @import("topology.zig");

const MessageType = message.MessageType;

pub const TaskKind = enum {
    message_dispatch,
    workload_normalization,
    gateway_lease_scheduling,
    reservation_lifecycle,
    federation_placement,
    decoupled_fragment,
    rda_merge,
    multi_gateway_global_job,
};

pub const CoverageEvidence = enum {
    symbolic_carrier,
    production_fixture,
    bounded_model,
    generated_xi,
    solver_checked,
    smoke_replay,
};

pub fn coverageEvidenceLabel(evidence: CoverageEvidence) []const u8 {
    return switch (evidence) {
        .symbolic_carrier => "symbolic_carrier",
        .production_fixture => "production_fixture",
        .bounded_model => "bounded_model",
        .generated_xi => "generated_Xi",
        .solver_checked => "solver_checked",
        .smoke_replay => "smoke_replay",
    };
}

pub const TaskDefinition = struct {
    kind: TaskKind,
    name: []const u8,
    production_surface: []const u8,
    coverage_evidence: CoverageEvidence,
    coverage_paths: []const []const u8,
    design_only: bool = false,
};

const message_dispatch_coverage_paths = [_][]const u8{
    "message_registry.protocol_messages",
    "message_registry.worker_handler_registry",
    "message_registry.family_capabilities",
};

const workload_coverage_paths = [_][]const u8{
    "workload.request_kind",
    "workload.route",
    "workload.status",
    "workload.resume_label",
};

const gateway_lease_coverage_paths = [_][]const u8{
    "scheduling.lease_owner",
    "scheduling.worker_state",
    "scheduling.decision.admission",
    "scheduling.violation.admission",
};

const reservation_coverage_paths = [_][]const u8{
    "reservation.decision.lifecycle",
    "reservation.violation.lifecycle",
    "reservation.queue_state",
};

const federation_coverage_paths = [_][]const u8{
    "federation.operations.gateway_connect",
    "federation.operations.mutation_batch",
    "federation.gateway_status",
    "federation.service_health",
    "federation.placement_decision",
    "federation.placement_violation",
};

const decoupled_coverage_paths = [_][]const u8{
    "decoupled.worker_status",
    "decoupled.action_tag",
    "decoupled.decision",
    "decoupled.violation",
    "decoupled.crash_restart_resume",
};

const rda_coverage_paths = [_][]const u8{
    "rda.merge_kind",
    "rda.violation",
    "rda.local_view_compatibility",
};

const global_job_coverage_paths = [_][]const u8{
    "federation.operations.global_job",
    "global_job.design_state",
    "global_job.partial_loss",
};

pub const catalog = [_]TaskDefinition{
    .{
        .kind = .message_dispatch,
        .name = "message dispatch",
        .production_surface = "src/protocol/message_registry.zig + src/nodes/workers/task_handlers.zig",
        .coverage_evidence = .production_fixture,
        .coverage_paths = &message_dispatch_coverage_paths,
    },
    .{
        .kind = .workload_normalization,
        .name = "workload normalization",
        .production_surface = "src/protocol/decoupled_job.zig + src/workloads/training/workload.zig",
        .coverage_evidence = .production_fixture,
        .coverage_paths = &workload_coverage_paths,
    },
    .{
        .kind = .gateway_lease_scheduling,
        .name = "gateway lease-owner scheduling",
        .production_surface = "src/nodes/gateway/scheduling.zig + src/nodes/gateway/controllers/training_controller.zig",
        .coverage_evidence = .generated_xi,
        .coverage_paths = &gateway_lease_coverage_paths,
    },
    .{
        .kind = .reservation_lifecycle,
        .name = "reservation lifecycle",
        .production_surface = "src/nodes/gateway/embedded_jobs.zig + src/nodes/gateway/scheduling.zig",
        .coverage_evidence = .generated_xi,
        .coverage_paths = &reservation_coverage_paths,
    },
    .{
        .kind = .federation_placement,
        .name = "federation placement",
        .production_surface = "src/nodes/federation_hub/gateway_registry.zig",
        .coverage_evidence = .bounded_model,
        .coverage_paths = &federation_coverage_paths,
    },
    .{
        .kind = .decoupled_fragment,
        .name = "gateway-local Decoupled DiLoCo fragment",
        .production_surface = "src/nodes/gateway/decoupled_training_session.zig + src/algorithms/event_tape.zig",
        .coverage_evidence = .generated_xi,
        .coverage_paths = &decoupled_coverage_paths,
    },
    .{
        .kind = .rda_merge,
        .name = "RDA merge",
        .production_surface = "src/algorithms/decoupled_merge.zig",
        .coverage_evidence = .production_fixture,
        .coverage_paths = &rda_coverage_paths,
    },
    .{
        .kind = .multi_gateway_global_job,
        .name = "future multi-gateway global-job contract",
        .production_surface = "future hub -> gateway -> scheduling -> worker composition",
        .coverage_evidence = .symbolic_carrier,
        .coverage_paths = &global_job_coverage_paths,
        .design_only = true,
    },
};

pub const ProtocolTask = struct {
    definition: TaskDefinition,
    input: topology.Complex,
    output: topology.Complex,
    carrier: topology.SimplicialCarrierMap,
};

pub const OwnedProtocolTask = struct {
    allocator: std.mem.Allocator,
    definition: TaskDefinition,
    input: topology.OwnedComplex,
    output: topology.OwnedComplex,
    carrier: topology.OwnedSimplicialCarrierMap,

    pub fn task(self: *const OwnedProtocolTask) ProtocolTask {
        return .{
            .definition = self.definition,
            .input = self.input.complex(),
            .output = self.output.complex(),
            .carrier = self.carrier.carrierMap(),
        };
    }

    pub fn deinit(self: *OwnedProtocolTask) void {
        self.carrier.deinit();
        self.output.deinit();
        self.input.deinit();
        self.* = undefined;
    }
};

pub const CaseSpec = struct {
    input: topology.Simplex,
    outputs: []const topology.Simplex,
};

pub fn definitionFor(kind: TaskKind) TaskDefinition {
    for (catalog) |item| {
        if (item.kind == kind) return item;
    }
    unreachable;
}

pub fn build(allocator: std.mem.Allocator, kind: TaskKind) !OwnedProtocolTask {
    return switch (kind) {
        .message_dispatch => buildMessageDispatchTask(allocator),
        .workload_normalization => buildWorkloadNormalizationTask(allocator),
        .gateway_lease_scheduling => buildGatewayLeaseSchedulingTask(allocator),
        .reservation_lifecycle => buildReservationLifecycleTask(allocator),
        .federation_placement => buildFederationPlacementTask(allocator),
        .decoupled_fragment => buildDecoupledFragmentTask(allocator),
        .rda_merge => buildRdaMergeTask(allocator),
        .multi_gateway_global_job => buildMultiGatewayGlobalJobTask(allocator),
    };
}

pub fn buildMessageDispatchTask(allocator: std.mem.Allocator) !OwnedProtocolTask {
    return buildFromCases(allocator, definitionFor(.message_dispatch), &message_dispatch_cases);
}

pub fn buildWorkloadNormalizationTask(allocator: std.mem.Allocator) !OwnedProtocolTask {
    return buildFromCases(allocator, definitionFor(.workload_normalization), &workload_cases);
}

pub fn buildGatewayLeaseSchedulingTask(allocator: std.mem.Allocator) !OwnedProtocolTask {
    return buildFromCases(allocator, definitionFor(.gateway_lease_scheduling), &gateway_lease_cases);
}

pub fn buildReservationLifecycleTask(allocator: std.mem.Allocator) !OwnedProtocolTask {
    return buildFromCases(allocator, definitionFor(.reservation_lifecycle), &reservation_cases);
}

pub fn buildFederationPlacementTask(allocator: std.mem.Allocator) !OwnedProtocolTask {
    return buildFromCases(allocator, definitionFor(.federation_placement), &federation_cases);
}

pub fn buildDecoupledFragmentTask(allocator: std.mem.Allocator) !OwnedProtocolTask {
    return buildFromCases(allocator, definitionFor(.decoupled_fragment), &decoupled_fragment_cases);
}

pub fn buildRdaMergeTask(allocator: std.mem.Allocator) !OwnedProtocolTask {
    return buildFromCases(allocator, definitionFor(.rda_merge), &rda_cases);
}

pub fn buildMultiGatewayGlobalJobTask(allocator: std.mem.Allocator) !OwnedProtocolTask {
    return buildFromCases(allocator, definitionFor(.multi_gateway_global_job), &global_job_cases);
}

pub fn buildFromCases(
    allocator: std.mem.Allocator,
    definition: TaskDefinition,
    cases: []const CaseSpec,
) !OwnedProtocolTask {
    var input = try buildInputComplex(allocator, cases);
    errdefer input.deinit();

    var output = try buildOutputComplex(allocator, cases);
    errdefer output.deinit();

    var carrier = try buildCarrier(allocator, input.complex(), output.complex(), cases);
    errdefer carrier.deinit();

    try carrier.carrierMap().validate(allocator, .{
        .require_monotone = true,
        .require_chromatic = true,
    });

    return .{
        .allocator = allocator,
        .definition = definition,
        .input = input,
        .output = output,
        .carrier = carrier,
    };
}

fn buildInputComplex(allocator: std.mem.Allocator, cases: []const CaseSpec) !topology.OwnedComplex {
    var builder = topology.ComplexBuilder.init(allocator);
    errdefer builder.deinit();

    for (cases) |case| {
        try builder.add(case.input);
    }

    return builder.toOwnedComplex();
}

fn buildOutputComplex(allocator: std.mem.Allocator, cases: []const CaseSpec) !topology.OwnedComplex {
    var builder = topology.ComplexBuilder.init(allocator);
    errdefer builder.deinit();

    for (cases) |case| {
        for (case.outputs) |output| {
            try builder.add(output);
        }
    }

    return builder.toOwnedComplex();
}

fn buildCarrier(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    output: topology.Complex,
    cases: []const CaseSpec,
) !topology.OwnedSimplicialCarrierMap {
    var faces = try input.faces(allocator);
    defer faces.deinit();

    var builder = topology.SimplicialCarrierMapBuilder.init(allocator, input, output);
    errdefer builder.deinit();

    for (faces.simplexes) |face| {
        var image = try imageForInputFace(allocator, output, cases, face);
        defer image.deinit();

        try builder.add(face, .{
            .parent = output,
            .simplexes = image.simplexes,
        });
    }

    return builder.toOwned();
}

fn imageForInputFace(
    allocator: std.mem.Allocator,
    output: topology.Complex,
    cases: []const CaseSpec,
    face: topology.Simplex,
) !topology.OwnedComplex {
    var builder = topology.ComplexBuilder.init(allocator);
    errdefer builder.deinit();

    if (face.isEmpty()) {
        try builder.add(.{ .vertices = &[_]topology.Vertex{} });
        return builder.toOwnedComplex();
    }

    const case = uniqueContainingCase(cases, face) orelse return error.UncoveredTaskFace;
    for (case.outputs) |output_simplex| {
        var projected = try projectOutputToInputFace(allocator, face, output_simplex);
        defer projected.deinit();
        if (!output.containsSimplex(projected.simplex())) return error.TaskOutputProjectionOutsideOutputComplex;
        try builder.add(projected.simplex());
    }

    return builder.toOwnedComplex();
}

fn uniqueContainingCase(cases: []const CaseSpec, face: topology.Simplex) ?CaseSpec {
    var result: ?CaseSpec = null;
    for (cases) |case| {
        if (!face.isSubsetOf(case.input)) continue;
        if (result != null) return null;
        result = case;
    }
    return result;
}

fn projectOutputToInputFace(
    allocator: std.mem.Allocator,
    face: topology.Simplex,
    output: topology.Simplex,
) !topology.OwnedSimplex {
    const projected = try allocator.alloc(topology.Vertex, face.vertices.len);
    errdefer allocator.free(projected);

    for (face.vertices, 0..) |input_vertex, index| {
        projected[index] = output.vertexWithColor(input_vertex.color) orelse return error.OutputFacetMissingColor;
    }

    std.mem.sort(topology.Vertex, projected, {}, topology.Vertex.lessThan);
    return .{ .allocator = allocator, .vertices = projected };
}

fn v(color: topology.Color, label: []const u8) topology.Vertex {
    return .{ .color = color, .label = .{ .symbol = label } };
}

const empty_simplex = topology.Simplex{ .vertices = &[_]topology.Vertex{} };

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
const msg_decoupled_outputs = [_]topology.Simplex{.{ .vertices = &msg_decoupled_output }};
const msg_inference_outputs = [_]topology.Simplex{.{ .vertices = &msg_inference_output }};
const msg_result_outputs = [_]topology.Simplex{.{ .vertices = &msg_result_output }};
const msg_unknown_outputs = [_]topology.Simplex{.{ .vertices = &msg_unknown_output }};
const message_dispatch_cases = [_]CaseSpec{
    .{ .input = .{ .vertices = &msg_decoupled_input }, .outputs = &msg_decoupled_outputs },
    .{ .input = .{ .vertices = &msg_inference_input }, .outputs = &msg_inference_outputs },
    .{ .input = .{ .vertices = &msg_result_input }, .outputs = &msg_result_outputs },
    .{ .input = .{ .vertices = &msg_unknown_input }, .outputs = &msg_unknown_outputs },
};

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
const workload_regular_outputs = [_]topology.Simplex{.{ .vertices = &workload_regular_output }};
const workload_rejected_outputs = [_]topology.Simplex{.{ .vertices = &workload_rejected_output }};
const workload_cases = [_]CaseSpec{
    .{ .input = .{ .vertices = &workload_regular_input }, .outputs = &workload_regular_outputs },
    .{ .input = .{ .vertices = &workload_invalid_input }, .outputs = &workload_rejected_outputs },
};

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
const lease_available_outputs = [_]topology.Simplex{.{ .vertices = &lease_available_output }};
const lease_maxed_outputs = [_]topology.Simplex{.{ .vertices = &lease_maxed_output }};
const lease_cross_owner_outputs = [_]topology.Simplex{.{ .vertices = &lease_cross_owner_output }};
const gateway_lease_cases = [_]CaseSpec{
    .{ .input = .{ .vertices = &lease_available_input }, .outputs = &lease_available_outputs },
    .{ .input = .{ .vertices = &lease_maxed_input }, .outputs = &lease_maxed_outputs },
    .{ .input = .{ .vertices = &lease_cross_owner_input }, .outputs = &lease_cross_owner_outputs },
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
const reservation_commit_outputs = [_]topology.Simplex{.{ .vertices = &reservation_commit_output }};
const reservation_release_outputs = [_]topology.Simplex{.{ .vertices = &reservation_release_output }};
const reservation_expire_outputs = [_]topology.Simplex{.{ .vertices = &reservation_expire_output }};
const reservation_cancel_outputs = [_]topology.Simplex{.{ .vertices = &reservation_cancel_output }};
const reservation_cases = [_]CaseSpec{
    .{ .input = .{ .vertices = &reservation_commit_input }, .outputs = &reservation_commit_outputs },
    .{ .input = .{ .vertices = &reservation_release_input }, .outputs = &reservation_release_outputs },
    .{ .input = .{ .vertices = &reservation_expire_input }, .outputs = &reservation_expire_outputs },
    .{ .input = .{ .vertices = &reservation_cancel_input }, .outputs = &reservation_cancel_outputs },
};

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
const federation_compatible_outputs = [_]topology.Simplex{
    .{ .vertices = &federation_placed_output },
    .{ .vertices = &federation_loss_output },
};
const federation_stale_outputs = [_]topology.Simplex{.{ .vertices = &federation_stale_output }};
const federation_incompatible_outputs = [_]topology.Simplex{.{ .vertices = &federation_incompatible_output }};
const federation_cases = [_]CaseSpec{
    .{ .input = .{ .vertices = &federation_compatible_input }, .outputs = &federation_compatible_outputs },
    .{ .input = .{ .vertices = &federation_stale_input }, .outputs = &federation_stale_outputs },
    .{ .input = .{ .vertices = &federation_incompatible_input }, .outputs = &federation_incompatible_outputs },
};

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
const fragment_commit_outputs = [_]topology.Simplex{.{ .vertices = &fragment_commit_output }};
const fragment_defer_outputs = [_]topology.Simplex{.{ .vertices = &fragment_defer_output }};
const fragment_duplicate_outputs = [_]topology.Simplex{.{ .vertices = &fragment_duplicate_output }};
const fragment_context_outputs = [_]topology.Simplex{.{ .vertices = &fragment_context_output }};
const fragment_clock_outputs = [_]topology.Simplex{.{ .vertices = &fragment_clock_output }};
const fragment_canceled_outputs = [_]topology.Simplex{.{ .vertices = &fragment_canceled_output }};
const decoupled_fragment_cases = [_]CaseSpec{
    .{ .input = .{ .vertices = &fragment_fresh_input }, .outputs = &fragment_commit_outputs },
    .{ .input = .{ .vertices = &fragment_below_quorum_input }, .outputs = &fragment_defer_outputs },
    .{ .input = .{ .vertices = &fragment_duplicate_input }, .outputs = &fragment_duplicate_outputs },
    .{ .input = .{ .vertices = &fragment_context_input }, .outputs = &fragment_context_outputs },
    .{ .input = .{ .vertices = &fragment_clock_input }, .outputs = &fragment_clock_outputs },
    .{ .input = .{ .vertices = &fragment_canceled_input }, .outputs = &fragment_canceled_outputs },
};

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
const rda_direction_outputs = [_]topology.Simplex{.{ .vertices = &rda_direction_output }};
const rda_degenerate_outputs = [_]topology.Simplex{.{ .vertices = &rda_degenerate_output }};
const rda_below_quorum_outputs = [_]topology.Simplex{.{ .vertices = &rda_below_quorum_output }};
const rda_cases = [_]CaseSpec{
    .{ .input = .{ .vertices = &rda_direction_input }, .outputs = &rda_direction_outputs },
    .{ .input = .{ .vertices = &rda_degenerate_input }, .outputs = &rda_degenerate_outputs },
    .{ .input = .{ .vertices = &rda_below_quorum_input }, .outputs = &rda_below_quorum_outputs },
};

const global_commit_input = [_]topology.Vertex{
    v(.{ .hub = 0 }, "global_job.accepted"),
    v(.{ .job = 60 }, "job.requires_two_gateways"),
    v(.{ .gateway = 1 }, "gateway_a.placed.commit"),
    v(.{ .gateway = 2 }, "gateway_b.placed.commit"),
};
const global_partial_loss_input = [_]topology.Vertex{
    v(.{ .hub = 0 }, "global_job.accepted.partial_loss"),
    v(.{ .job = 61 }, "job.requires_two_gateways"),
    v(.{ .gateway = 1 }, "gateway_a.placed.partial_loss"),
    v(.{ .gateway = 2 }, "gateway_b.lost"),
};
const global_cancel_input = [_]topology.Vertex{
    v(.{ .hub = 0 }, "global_job.cancel_requested"),
    v(.{ .job = 62 }, "job.canceling"),
    v(.{ .gateway = 1 }, "gateway_a.placed.cancel"),
    v(.{ .gateway = 2 }, "gateway_b.queued"),
};
const global_capacity_input = [_]topology.Vertex{
    v(.{ .hub = 0 }, "global_job.capacity_wait"),
    v(.{ .job = 63 }, "job.requires_two_gateways"),
    v(.{ .gateway = 1 }, "gateway_a.placed.capacity_wait"),
    v(.{ .gateway = 2 }, "gateway_b.no_capacity"),
};
const global_commit_output = [_]topology.Vertex{
    v(.{ .hub = 0 }, "global_job.committed"),
    v(.{ .job = 60 }, "job.running"),
    v(.{ .gateway = 1 }, "gateway_a.committed"),
    v(.{ .gateway = 2 }, "gateway_b.committed"),
};
const global_rollback_output = [_]topology.Vertex{
    v(.{ .hub = 0 }, "global_job.rollback"),
    v(.{ .job = 61 }, "job.retryable"),
    v(.{ .gateway = 1 }, "gateway_a.release"),
    v(.{ .gateway = 2 }, "gateway_b.failed"),
};
const global_cancel_output = [_]topology.Vertex{
    v(.{ .hub = 0 }, "global_job.canceled"),
    v(.{ .job = 62 }, "job.canceled"),
    v(.{ .gateway = 1 }, "gateway_a.cleanup"),
    v(.{ .gateway = 2 }, "gateway_b.cleanup"),
};
const global_deferred_output = [_]topology.Vertex{
    v(.{ .hub = 0 }, "global_job.deferred"),
    v(.{ .job = 63 }, "job.queued"),
    v(.{ .gateway = 1 }, "gateway_a.hold_or_release"),
    v(.{ .gateway = 2 }, "gateway_b.await_capacity"),
};
const global_commit_outputs = [_]topology.Simplex{.{ .vertices = &global_commit_output }};
const global_rollback_outputs = [_]topology.Simplex{.{ .vertices = &global_rollback_output }};
const global_cancel_outputs = [_]topology.Simplex{.{ .vertices = &global_cancel_output }};
const global_deferred_outputs = [_]topology.Simplex{.{ .vertices = &global_deferred_output }};
const global_job_cases = [_]CaseSpec{
    .{ .input = .{ .vertices = &global_commit_input }, .outputs = &global_commit_outputs },
    .{ .input = .{ .vertices = &global_partial_loss_input }, .outputs = &global_rollback_outputs },
    .{ .input = .{ .vertices = &global_cancel_input }, .outputs = &global_cancel_outputs },
    .{ .input = .{ .vertices = &global_capacity_input }, .outputs = &global_deferred_outputs },
};

test "catalog exposes every milestone PCP task" {
    try std.testing.expectEqual(@as(usize, 8), catalog.len);

    for (catalog) |item| {
        try std.testing.expect(item.name.len > 0);
        try std.testing.expect(item.production_surface.len > 0);
    }
}

test "every PCP task builds as a chromatic monotone carrier" {
    for (catalog) |item| {
        var owned = try build(std.testing.allocator, item.kind);
        defer owned.deinit();

        const task_value = owned.task();
        try task_value.input.validate();
        try task_value.output.validate();
        try task_value.carrier.validate(std.testing.allocator, .{
            .require_monotone = true,
            .require_chromatic = true,
        });
    }
}

test "message dispatch task fails closed for worker-result and unknown messages" {
    var owned = try buildMessageDispatchTask(std.testing.allocator);
    defer owned.deinit();
    const task_value = owned.task();

    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &msg_decoupled_input },
        .{ .vertices = &msg_decoupled_output },
    ));
    try std.testing.expect(!task_value.carrier.carries(
        .{ .vertices = &msg_result_input },
        .{ .vertices = &msg_decoupled_output },
    ));
    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &msg_result_input },
        .{ .vertices = &msg_result_output },
    ));
    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &msg_unknown_input },
        .{ .vertices = &msg_unknown_output },
    ));
}

test "reservation lifecycle task separates commit release expiry and cancellation cleanup" {
    var owned = try buildReservationLifecycleTask(std.testing.allocator);
    defer owned.deinit();
    const task_value = owned.task();

    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &reservation_commit_input },
        .{ .vertices = &reservation_commit_output },
    ));
    try std.testing.expect(!task_value.carrier.carries(
        .{ .vertices = &reservation_cancel_input },
        .{ .vertices = &reservation_commit_output },
    ));
    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &reservation_cancel_input },
        .{ .vertices = &reservation_cancel_output },
    ));
}

test "federation placement task allows only compatible placement loss alternatives" {
    var owned = try buildFederationPlacementTask(std.testing.allocator);
    defer owned.deinit();
    const task_value = owned.task();

    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &federation_compatible_input },
        .{ .vertices = &federation_placed_output },
    ));
    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &federation_compatible_input },
        .{ .vertices = &federation_loss_output },
    ));
    try std.testing.expect(!task_value.carrier.carries(
        .{ .vertices = &federation_stale_input },
        .{ .vertices = &federation_placed_output },
    ));
}

test "RDA task carries direction norm and degenerate zero outputs separately" {
    var owned = try buildRdaMergeTask(std.testing.allocator);
    defer owned.deinit();
    const task_value = owned.task();

    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &rda_direction_input },
        .{ .vertices = &rda_direction_output },
    ));
    try std.testing.expect(!task_value.carrier.carries(
        .{ .vertices = &rda_direction_input },
        .{ .vertices = &rda_degenerate_output },
    ));
    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &rda_degenerate_input },
        .{ .vertices = &rda_degenerate_output },
    ));
}

test "global job task includes rollback cancel and capacity states" {
    var owned = try buildMultiGatewayGlobalJobTask(std.testing.allocator);
    defer owned.deinit();
    const task_value = owned.task();

    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &global_commit_input },
        .{ .vertices = &global_commit_output },
    ));
    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &global_partial_loss_input },
        .{ .vertices = &global_rollback_output },
    ));
    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &global_cancel_input },
        .{ .vertices = &global_cancel_output },
    ));
    try std.testing.expect(task_value.carrier.carries(
        .{ .vertices = &global_capacity_input },
        .{ .vertices = &global_deferred_output },
    ));
}
