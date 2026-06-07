const std = @import("std");

const decoupled_job = @import("../../../protocol/decoupled_job.zig");
const topology = @import("../topology.zig");
const workload = @import("../../../workloads/training/workload.zig");

pub const WorkloadRoute = enum {
    regular_training_service,
    rejected,
};

pub const WorkloadStatus = enum {
    accepted,
    rejected,
    completed,
};

pub const ResumeLabel = enum {
    none,
    compatible,
    stale_session,
};

pub const RequestCase = enum {
    regular,
    invalid,
};

pub const WorkloadOutput = struct {
    kind: ?decoupled_job.TrainingKind,
    route: WorkloadRoute,
    status: WorkloadStatus,
    resume_label: ResumeLabel,
    session_id: []const u8,
};

pub const WorkloadTopologyTask = struct {
    input: topology.Complex,
    protocol: topology.Complex,
    output: topology.Complex,
    carrier: topology.CarrierMap,
    decisions: []const topology.DecisionEdge,
};

const regular_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .api_caller = 0 }, .label = .{ .symbol = "request.regular" } },
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = "gateway.training" } },
};
const invalid_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .api_caller = 0 }, .label = .{ .symbol = "request.invalid" } },
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = "gateway.training" } },
};

const regular_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = "normalized.regular" } },
    .{ .color = .{ .session = 0 }, .label = .{ .symbol = "session.accepted" } },
};
const rejected_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = "normalized.rejected" } },
    .{ .color = .{ .session = 0 }, .label = .{ .symbol = "session.rejected" } },
};

const regular_output_vertices = [_]topology.Vertex{
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = "route.regular_training" } },
    .{ .color = .{ .session = 0 }, .label = .{ .symbol = "status.accepted" } },
};
const rejected_output_vertices = [_]topology.Vertex{
    .{ .color = .{ .gateway = 0 }, .label = .{ .symbol = "route.rejected" } },
    .{ .color = .{ .session = 0 }, .label = .{ .symbol = "status.rejected" } },
};

const input_simplexes = [_]topology.Simplex{
    .{ .vertices = &regular_input_vertices },
    .{ .vertices = &invalid_input_vertices },
};
const protocol_simplexes = [_]topology.Simplex{
    .{ .vertices = &regular_protocol_vertices },
    .{ .vertices = &rejected_protocol_vertices },
};
const output_simplexes = [_]topology.Simplex{
    .{ .vertices = &regular_output_vertices },
    .{ .vertices = &rejected_output_vertices },
};

const regular_outputs = [_]usize{0};
const rejected_outputs = [_]usize{1};
const carrier_edges = [_]topology.CarrierEdge{
    .{ .input_index = 0, .output_indices = &regular_outputs },
    .{ .input_index = 1, .output_indices = &rejected_outputs },
};
const default_decisions = [_]topology.DecisionEdge{
    .{ .input_index = 0, .protocol_index = 0, .output_index = 0 },
    .{ .input_index = 1, .protocol_index = 1, .output_index = 1 },
};

pub fn topologyTask() WorkloadTopologyTask {
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

pub fn inputIndex(case: RequestCase) usize {
    return switch (case) {
        .regular => 0,
        .invalid => 1,
    };
}

pub fn protocolIndex(case: RequestCase) usize {
    return inputIndex(case);
}

pub fn outputIndex(route: WorkloadRoute) usize {
    return switch (route) {
        .regular_training_service => 0,
        .rejected => 1,
    };
}

pub fn requestCaseForKind(kind: decoupled_job.TrainingKind) RequestCase {
    if (kind == .regular) return .regular;
    return .invalid;
}

pub fn routeForKind(kind: decoupled_job.TrainingKind) WorkloadRoute {
    if (kind == .regular) return .regular_training_service;
    return .rejected;
}

pub fn routeForJob(job: decoupled_job.DecoupledTrainingJob) WorkloadRoute {
    if (std.meta.activeTag(job.payload) != job.training_kind) return .rejected;
    return routeForKind(job.training_kind);
}

pub fn acceptedOutput(job: decoupled_job.DecoupledTrainingJob) WorkloadOutput {
    return .{
        .kind = job.training_kind,
        .route = routeForJob(job),
        .status = .accepted,
        .resume_label = .compatible,
        .session_id = job.run_id,
    };
}

pub fn rejectedOutput(session_id: []const u8) WorkloadOutput {
    return .{
        .kind = null,
        .route = .rejected,
        .status = .rejected,
        .resume_label = .none,
        .session_id = session_id,
    };
}

pub fn specFromJob(job: decoupled_job.DecoupledTrainingJob) !workload.TrainingWorkloadSpec {
    return workload.TrainingWorkloadSpec.fromDecoupledJob(job);
}

test "workload topology task validates default decisions" {
    const task = topologyTask();
    try topology.validateDecisionMap(std.testing.allocator, task.protocol, task.carrier, task.decisions);
}

test "workload request parser keeps regular intent" {
    try std.testing.expectEqual(
        decoupled_job.TrainingKind.regular,
        try workload.trainingKindFromRequestBody(std.testing.allocator, "{}"),
    );
}
