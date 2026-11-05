const std = @import("std");

const message_registry = @import("../../protocol/message_registry.zig");
const tasks = @import("tasks.zig");

const decoupled_model = @import("models/decoupled_site.zig");
const federation_model = @import("models/federation_placement.zig");
const rda_oracle = @import("oracles/decoupled_merge.zig");
const scheduling_model = @import("models/gateway_scheduling.zig");
const workload_model = @import("models/workload_contract.zig");

pub const CoveragePath = struct {
    id: []const u8,
    task: ?tasks.TaskKind,
    source: []const u8,
    design_only: bool = false,
};

pub const CoverageMapping = struct {
    path_id: []const u8,
    task: tasks.TaskKind,
    evidence: tasks.CoverageEvidence,
    design_only: bool,
};

pub const CoverageSummary = struct {
    total: usize = 0,
    implemented: usize = 0,
    covered: usize = 0,
    design_only: usize = 0,
    uncovered: usize = 0,
};

pub const GlobalJobDesignState = enum {
    submitted,
    routed,
    partially_committed,
    rolled_back,
    status_polled,
    canceled,
};

pub const default_paths = [_]CoveragePath{
    .{
        .id = "message_registry.protocol_messages",
        .task = .message_dispatch,
        .source = "src/protocol/message_registry.zig",
    },
    .{
        .id = "message_registry.worker_handler_registry",
        .task = .message_dispatch,
        .source = "src/protocol/message_registry.zig",
    },
    .{
        .id = "message_registry.family_capabilities",
        .task = .message_dispatch,
        .source = "src/protocol/message_registry.zig",
    },
    .{
        .id = "workload.request_kind",
        .task = .workload_normalization,
        .source = "src/testing/protocol_topology/models/workload_contract.zig",
    },
    .{
        .id = "workload.route",
        .task = .workload_normalization,
        .source = "src/workloads/training/workload.zig",
    },
    .{
        .id = "workload.status",
        .task = .workload_normalization,
        .source = "src/testing/protocol_topology/models/workload_contract.zig",
    },
    .{
        .id = "workload.resume_label",
        .task = .workload_normalization,
        .source = "src/protocol/decoupled_job.zig",
    },
    .{
        .id = "scheduling.lease_owner",
        .task = .gateway_lease_scheduling,
        .source = "src/nodes/gateway/controllers/training_controller.zig",
    },
    .{
        .id = "scheduling.worker_state",
        .task = .gateway_lease_scheduling,
        .source = "src/testing/protocol_topology/models/gateway_scheduling.zig",
    },
    .{
        .id = "scheduling.decision.admission",
        .task = .gateway_lease_scheduling,
        .source = "src/testing/protocol_topology/models/gateway_scheduling.zig",
    },
    .{
        .id = "scheduling.violation.admission",
        .task = .gateway_lease_scheduling,
        .source = "src/testing/protocol_topology/models/gateway_scheduling.zig",
    },
    .{
        .id = "reservation.decision.lifecycle",
        .task = .reservation_lifecycle,
        .source = "src/nodes/gateway/embedded_jobs.zig + src/nodes/gateway/scheduling.zig",
    },
    .{
        .id = "reservation.violation.lifecycle",
        .task = .reservation_lifecycle,
        .source = "src/testing/protocol_topology/models/gateway_scheduling.zig",
    },
    .{
        .id = "reservation.queue_state",
        .task = .reservation_lifecycle,
        .source = "src/testing/protocol_topology/models/gateway_scheduling.zig",
    },
    .{
        .id = "federation.operations.gateway_connect",
        .task = .federation_placement,
        .source = "src/protocol/message_registry.zig",
    },
    .{
        .id = "federation.operations.mutation_batch",
        .task = .federation_placement,
        .source = "src/protocol/message_registry.zig",
    },
    .{
        .id = "federation.gateway_status",
        .task = .federation_placement,
        .source = "src/nodes/federation_hub/gateway_registry.zig",
    },
    .{
        .id = "federation.service_health",
        .task = .federation_placement,
        .source = "src/protocol/federation/types.zig",
    },
    .{
        .id = "federation.placement_decision",
        .task = .federation_placement,
        .source = "src/testing/protocol_topology/models/federation_placement.zig",
    },
    .{
        .id = "federation.placement_violation",
        .task = .federation_placement,
        .source = "src/testing/protocol_topology/models/federation_placement.zig",
    },
    .{
        .id = "decoupled.worker_status",
        .task = .decoupled_fragment,
        .source = "src/testing/protocol_topology/models/decoupled_site.zig",
    },
    .{
        .id = "decoupled.action_tag",
        .task = .decoupled_fragment,
        .source = "src/testing/protocol_topology/models/decoupled_site.zig",
    },
    .{
        .id = "decoupled.decision",
        .task = .decoupled_fragment,
        .source = "src/testing/protocol_topology/models/decoupled_site.zig",
    },
    .{
        .id = "decoupled.violation",
        .task = .decoupled_fragment,
        .source = "src/testing/protocol_topology/models/decoupled_site.zig",
    },
    .{
        .id = "decoupled.crash_restart_resume",
        .task = .decoupled_fragment,
        .source = "src/algorithms/event_tape.zig + src/testing/protocol_topology/models/decoupled_site.zig",
    },
    .{
        .id = "rda.merge_kind",
        .task = .rda_merge,
        .source = "src/algorithms/decoupled_merge.zig",
    },
    .{
        .id = "rda.violation",
        .task = .rda_merge,
        .source = "src/testing/protocol_topology/oracles/decoupled_merge.zig",
    },
    .{
        .id = "rda.local_view_compatibility",
        .task = .rda_merge,
        .source = "src/testing/protocol_topology/oracles/decoupled_merge.zig",
    },
    .{
        .id = "federation.operations.global_job",
        .task = .multi_gateway_global_job,
        .source = "src/protocol/message_registry.zig",
        .design_only = true,
    },
    .{
        .id = "global_job.design_state",
        .task = .multi_gateway_global_job,
        .source = "src/testing/protocol_topology/tasks.zig",
        .design_only = true,
    },
    .{
        .id = "global_job.partial_loss",
        .task = .multi_gateway_global_job,
        .source = "src/testing/protocol_topology/reporting.zig",
        .design_only = true,
    },
};

pub fn summarize(paths: []const CoveragePath) CoverageSummary {
    var summary = CoverageSummary{ .total = paths.len };
    for (paths) |path| {
        if (path.design_only) {
            summary.design_only += 1;
            continue;
        }

        summary.implemented += 1;
        if (path.task) |_| {
            summary.covered += 1;
        } else {
            summary.uncovered += 1;
        }
    }
    return summary;
}

pub fn writeCiSummary(writer: anytype, paths: []const CoveragePath) !void {
    const summary = summarize(paths);
    try writer.print(
        "pcp topology coverage summary: total={d} implemented={d} covered={d} design_only={d} uncovered={d}\n",
        .{ summary.total, summary.implemented, summary.covered, summary.design_only, summary.uncovered },
    );

    inline for (tasks.catalog) |definition| {
        const count = countPathsForTask(paths, definition.kind);
        try writer.print(
            "- {s}: paths={d} evidence={s}",
            .{ @tagName(definition.kind), count, tasks.coverageEvidenceLabel(definition.coverage_evidence) },
        );
        if (definition.design_only) try writer.writeAll(" design_only=true");
        try writer.writeByte('\n');
    }

    for (paths) |path| {
        if (path.task == null and !path.design_only) {
            try writer.print("- uncovered: {s} source={s}\n", .{ path.id, path.source });
        }
    }
}

pub fn validateDefaultCoverage() !void {
    try validateTaskMetadata();
    try validateCoveragePaths(&default_paths);
    try validateStrictImplementedCoverage(&default_paths);
}

pub fn validateStrictImplementedCoverage(paths: []const CoveragePath) !void {
    for (paths) |path| {
        if (path.task == null and !path.design_only) return error.UncoveredImplementedProductionPath;
    }
}

pub fn coverageForPath(id: []const u8) ?CoverageMapping {
    for (default_paths) |path| {
        if (!std.mem.eql(u8, path.id, id)) continue;
        const kind = path.task orelse return null;
        return mappingFromPath(path, kind);
    }
    return null;
}

pub fn coverageForMessageEntry(entry: message_registry.MessageEntry) ?CoverageMapping {
    if (entry.family == .unknown) return null;
    return mappingForTask("message_registry.protocol_messages", .message_dispatch);
}

pub fn coverageForMessageFamily(family: message_registry.MessageFamily) ?CoverageMapping {
    return switch (family) {
        .unknown => null,
        else => mappingForTask("message_registry.family_capabilities", .message_dispatch),
    };
}

pub fn coverageForWorkerHandlerFamily(family: message_registry.WorkerHandlerFamily) ?CoverageMapping {
    return switch (family) {
        .unknown => null,
        else => mappingForTask("message_registry.worker_handler_registry", .message_dispatch),
    };
}

pub fn coverageForFederationOperation(operation: message_registry.FederationOperation) ?CoverageMapping {
    if (std.mem.eql(u8, operation.name, "gateway_connect")) {
        return mappingForTask("federation.operations.gateway_connect", .federation_placement);
    }
    if (std.mem.eql(u8, operation.name, "mutation_batch")) {
        return mappingForTask("federation.operations.mutation_batch", .federation_placement);
    }
    if (std.mem.eql(u8, operation.name, "reservation_commit") or
        std.mem.eql(u8, operation.name, "reservation_release"))
    {
        return mappingForTask("reservation.decision.lifecycle", .reservation_lifecycle);
    }
    if (std.mem.startsWith(u8, operation.name, "global_")) {
        return mappingForTask("federation.operations.global_job", .multi_gateway_global_job);
    }
    return null;
}

pub fn coverageForWorkloadRequestCase(_: workload_model.RequestCase) CoverageMapping {
    return mappingForTask("workload.request_kind", .workload_normalization);
}

pub fn coverageForWorkloadRoute(_: workload_model.WorkloadRoute) CoverageMapping {
    return mappingForTask("workload.route", .workload_normalization);
}

pub fn coverageForWorkloadStatus(_: workload_model.WorkloadStatus) CoverageMapping {
    return mappingForTask("workload.status", .workload_normalization);
}

pub fn coverageForResumeLabel(_: workload_model.ResumeLabel) CoverageMapping {
    return mappingForTask("workload.resume_label", .workload_normalization);
}

pub fn coverageForLeaseOwner(_: scheduling_model.LeaseOwner) CoverageMapping {
    return mappingForTask("scheduling.lease_owner", .gateway_lease_scheduling);
}

pub fn coverageForSchedulingWorkerState(_: scheduling_model.WorkerState) CoverageMapping {
    return mappingForTask("scheduling.worker_state", .gateway_lease_scheduling);
}

pub fn coverageForSchedulingDecision(decision: scheduling_model.Decision) CoverageMapping {
    return switch (decision) {
        .idle, .reserved, .queued, .rejected => mappingForTask(
            "scheduling.decision.admission",
            .gateway_lease_scheduling,
        ),
        .committed, .started, .completed, .released, .expired, .canceled => mappingForTask(
            "reservation.decision.lifecycle",
            .reservation_lifecycle,
        ),
    };
}

pub fn coverageForSchedulingViolation(violation: scheduling_model.Violation) CoverageMapping {
    return switch (violation) {
        .none,
        .invalid_worker,
        .invalid_request,
        .reservation_unavailable,
        .assignment_after_worker_loss,
        .worker_already_busy,
        .queue_full,
        => mappingForTask("scheduling.violation.admission", .gateway_lease_scheduling),
        .duplicate_job_reservation,
        .duplicate_worker_reservation,
        .duplicate_queued_job,
        .reservation_not_found,
        .queued_job_not_found,
        .active_job_not_found,
        => mappingForTask("reservation.violation.lifecycle", .reservation_lifecycle),
    };
}

pub fn coverageForFederationGatewayStatus(_: federation_model.GatewayStatus) CoverageMapping {
    return mappingForTask("federation.gateway_status", .federation_placement);
}

pub fn coverageForFederationServiceHealth(_: federation_model.ServiceHealth) CoverageMapping {
    return mappingForTask("federation.service_health", .federation_placement);
}

pub fn coverageForFederationDecision(_: federation_model.Decision) CoverageMapping {
    return mappingForTask("federation.placement_decision", .federation_placement);
}

pub fn coverageForFederationViolation(_: federation_model.Violation) CoverageMapping {
    return mappingForTask("federation.placement_violation", .federation_placement);
}

pub fn coverageForDecoupledWorkerStatus(_: decoupled_model.WorkerStatus) CoverageMapping {
    return mappingForTask("decoupled.worker_status", .decoupled_fragment);
}

pub fn coverageForDecoupledActionTag(tag: decoupled_model.ActionTag) CoverageMapping {
    return switch (tag) {
        .crash, .restart, .leave, .join, .resume_worker => mappingForTask(
            "decoupled.crash_restart_resume",
            .decoupled_fragment,
        ),
        else => mappingForTask("decoupled.action_tag", .decoupled_fragment),
    };
}

pub fn coverageForDecoupledDecision(_: decoupled_model.Decision) CoverageMapping {
    return mappingForTask("decoupled.decision", .decoupled_fragment);
}

pub fn coverageForDecoupledViolation(_: decoupled_model.Violation) CoverageMapping {
    return mappingForTask("decoupled.violation", .decoupled_fragment);
}

pub fn coverageForRdaMergeKind(_: rda_oracle.MergeKind) CoverageMapping {
    return mappingForTask("rda.merge_kind", .rda_merge);
}

pub fn coverageForRdaViolation(violation: rda_oracle.Violation) CoverageMapping {
    return switch (violation) {
        .incompatible_local_views => mappingForTask("rda.local_view_compatibility", .rda_merge),
        else => mappingForTask("rda.violation", .rda_merge),
    };
}

pub fn coverageForGlobalJobDesignState(_: GlobalJobDesignState) CoverageMapping {
    return mappingForTask("global_job.design_state", .multi_gateway_global_job);
}

fn validateTaskMetadata() !void {
    inline for (tasks.catalog) |definition| {
        if (definition.coverage_paths.len == 0) return error.MissingTaskCoveragePaths;
        if (definition.design_only and definition.kind != .multi_gateway_global_job) {
            return error.UnexpectedDesignOnlyTask;
        }

        for (definition.coverage_paths) |path_id| {
            const path = findPath(path_id) orelse return error.StaleTaskCoveragePath;
            if (path.task == null) return error.TaskCoveragePathUnmapped;
            if (path.task.? != definition.kind) return error.TaskCoveragePathMismatch;
            if (path.design_only != definition.design_only) return error.TaskCoverageDesignOnlyMismatch;
        }
    }
}

fn validateCoveragePaths(paths: []const CoveragePath) !void {
    for (paths, 0..) |path, index| {
        if (path.id.len == 0) return error.EmptyCoveragePathId;
        if (path.source.len == 0) return error.EmptyCoveragePathSource;

        for (paths[index + 1 ..]) |other| {
            if (std.mem.eql(u8, path.id, other.id)) return error.DuplicateCoveragePath;
        }

        if (path.task) |kind| {
            const definition = tasks.definitionFor(kind);
            if (definition.design_only != path.design_only) return error.CoveragePathDesignOnlyMismatch;
        }
    }
}

fn findPath(id: []const u8) ?CoveragePath {
    for (default_paths) |path| {
        if (std.mem.eql(u8, path.id, id)) return path;
    }
    return null;
}

fn mappingForTask(path_id: []const u8, kind: tasks.TaskKind) CoverageMapping {
    const definition = tasks.definitionFor(kind);
    return .{
        .path_id = path_id,
        .task = kind,
        .evidence = definition.coverage_evidence,
        .design_only = definition.design_only,
    };
}

fn mappingFromPath(path: CoveragePath, kind: tasks.TaskKind) CoverageMapping {
    const definition = tasks.definitionFor(kind);
    return .{
        .path_id = path.id,
        .task = kind,
        .evidence = definition.coverage_evidence,
        .design_only = path.design_only,
    };
}

fn countPathsForTask(paths: []const CoveragePath, kind: tasks.TaskKind) usize {
    var count: usize = 0;
    for (paths) |path| {
        if (path.task == kind) count += 1;
    }
    return count;
}

fn expectMapped(mapping: ?CoverageMapping) !CoverageMapping {
    return mapping orelse error.MissingTopologyCoverageMapping;
}

test "default coverage metadata is strict for implemented production paths" {
    try validateDefaultCoverage();
}

test "production message registry entries and handlers map to topology coverage" {
    for (message_registry.protocol_messages) |entry| {
        const mapping = try expectMapped(coverageForMessageEntry(entry));
        try std.testing.expectEqual(tasks.TaskKind.message_dispatch, mapping.task);
    }

    for (message_registry.family_capability_registry) |entry| {
        const mapping = try expectMapped(coverageForMessageFamily(entry.family));
        try std.testing.expectEqual(tasks.TaskKind.message_dispatch, mapping.task);
    }

    for (message_registry.worker_handler_registry) |registration| {
        const mapping = try expectMapped(coverageForWorkerHandlerFamily(registration.family));
        try std.testing.expectEqual(tasks.TaskKind.message_dispatch, mapping.task);
    }

    try std.testing.expectEqual(@as(?CoverageMapping, null), coverageForWorkerHandlerFamily(.unknown));
}

test "federation operation registry maps implemented and design-only paths" {
    var saw_design_only = false;
    for (message_registry.federation_operations) |operation| {
        const mapping = try expectMapped(coverageForFederationOperation(operation));
        if (mapping.design_only) saw_design_only = true;
    }
    try std.testing.expect(saw_design_only);
}

test "bounded model state labels map to topology coverage" {
    inline for (std.meta.fields(workload_model.RequestCase)) |field| {
        _ = coverageForWorkloadRequestCase(@as(workload_model.RequestCase, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(workload_model.WorkloadRoute)) |field| {
        _ = coverageForWorkloadRoute(@as(workload_model.WorkloadRoute, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(workload_model.WorkloadStatus)) |field| {
        _ = coverageForWorkloadStatus(@as(workload_model.WorkloadStatus, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(workload_model.ResumeLabel)) |field| {
        _ = coverageForResumeLabel(@as(workload_model.ResumeLabel, @enumFromInt(field.value)));
    }

    for (scheduling_model.lease_owners) |owner| {
        _ = coverageForLeaseOwner(owner);
    }
    inline for (std.meta.fields(scheduling_model.WorkerState)) |field| {
        _ = coverageForSchedulingWorkerState(@as(scheduling_model.WorkerState, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(scheduling_model.Decision)) |field| {
        _ = coverageForSchedulingDecision(@as(scheduling_model.Decision, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(scheduling_model.Violation)) |field| {
        _ = coverageForSchedulingViolation(@as(scheduling_model.Violation, @enumFromInt(field.value)));
    }

    inline for (std.meta.fields(federation_model.GatewayStatus)) |field| {
        _ = coverageForFederationGatewayStatus(@as(federation_model.GatewayStatus, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(federation_model.ServiceHealth)) |field| {
        _ = coverageForFederationServiceHealth(@as(federation_model.ServiceHealth, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(federation_model.Decision)) |field| {
        _ = coverageForFederationDecision(@as(federation_model.Decision, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(federation_model.Violation)) |field| {
        _ = coverageForFederationViolation(@as(federation_model.Violation, @enumFromInt(field.value)));
    }

    inline for (std.meta.fields(decoupled_model.WorkerStatus)) |field| {
        _ = coverageForDecoupledWorkerStatus(@as(decoupled_model.WorkerStatus, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(decoupled_model.ActionTag)) |field| {
        _ = coverageForDecoupledActionTag(@as(decoupled_model.ActionTag, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(decoupled_model.Decision)) |field| {
        _ = coverageForDecoupledDecision(@as(decoupled_model.Decision, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(decoupled_model.Violation)) |field| {
        _ = coverageForDecoupledViolation(@as(decoupled_model.Violation, @enumFromInt(field.value)));
    }

    inline for (std.meta.fields(rda_oracle.MergeKind)) |field| {
        _ = coverageForRdaMergeKind(@as(rda_oracle.MergeKind, @enumFromInt(field.value)));
    }
    inline for (std.meta.fields(rda_oracle.Violation)) |field| {
        _ = coverageForRdaViolation(@as(rda_oracle.Violation, @enumFromInt(field.value)));
    }
}

test "design-only global job mappings are reported outside implemented coverage" {
    const summary = summarize(&default_paths);
    try std.testing.expectEqual(@as(usize, 31), summary.total);
    try std.testing.expectEqual(@as(usize, 28), summary.implemented);
    try std.testing.expectEqual(@as(usize, 28), summary.covered);
    try std.testing.expectEqual(@as(usize, 3), summary.design_only);
    try std.testing.expectEqual(@as(usize, 0), summary.uncovered);

    inline for (std.meta.fields(GlobalJobDesignState)) |field| {
        const mapping = coverageForGlobalJobDesignState(
            @as(GlobalJobDesignState, @enumFromInt(field.value)),
        );
        try std.testing.expect(mapping.design_only);
    }
}

test "strict coverage rejects implemented paths without a topology mapping" {
    const paths = [_]CoveragePath{
        .{
            .id = "production.unmapped",
            .task = null,
            .source = "src/production/new_surface.zig",
        },
    };
    try std.testing.expectError(
        error.UncoveredImplementedProductionPath,
        validateStrictImplementedCoverage(&paths),
    );
}

test "coverage ci summary format is stable" {
    var rendered = std.ArrayList(u8).init(std.testing.allocator);
    defer rendered.deinit();

    try writeCiSummary(rendered.writer(), &default_paths);

    try std.testing.expectEqualStrings(
        "pcp topology coverage summary: total=31 implemented=28 covered=28 design_only=3 uncovered=0\n" ++
            "- message_dispatch: paths=3 evidence=production_fixture\n" ++
            "- workload_normalization: paths=4 evidence=production_fixture\n" ++
            "- gateway_lease_scheduling: paths=4 evidence=generated_Xi\n" ++
            "- reservation_lifecycle: paths=3 evidence=generated_Xi\n" ++
            "- federation_placement: paths=6 evidence=bounded_model\n" ++
            "- decoupled_fragment: paths=5 evidence=generated_Xi\n" ++
            "- rda_merge: paths=3 evidence=production_fixture\n" ++
            "- multi_gateway_global_job: paths=3 evidence=symbolic_carrier design_only=true\n",
        rendered.items,
    );
}
