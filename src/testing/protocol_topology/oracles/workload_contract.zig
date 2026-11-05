const std = @import("std");

const decoupled_job = @import("../../../protocol/decoupled_job.zig");
const model = @import("../models/workload_contract.zig");
const topology = @import("../topology.zig");

pub fn checkRoute(kind: decoupled_job.TrainingKind, route: model.WorkloadRoute) !void {
    if (route != model.routeForKind(kind)) return error.WrongWorkloadRoute;
}

pub fn checkNormalizedJob(job: decoupled_job.DecoupledTrainingJob) !model.WorkloadOutput {
    try job.validate();

    const spec = try model.specFromJob(job);
    if (spec.training_kind != job.training_kind) return error.TrainingKindMismatch;

    const route = model.routeForJob(job);
    try checkRoute(job.training_kind, route);

    return model.acceptedOutput(job);
}

pub fn checkSessionOutput(expected_session: []const u8, output: model.WorkloadOutput) !void {
    if (!std.mem.eql(u8, expected_session, output.session_id)) {
        return error.StaleSessionOutput;
    }
}

pub fn checkStableOutputLabels(output: model.WorkloadOutput) !void {
    if (output.route == .rejected) {
        if (output.status != .rejected) return error.InvalidOutputStatus;
        if (output.resume_label != .none) return error.InvalidResumeLabel;
        return;
    }

    if (output.status != .accepted and output.status != .completed) return error.InvalidOutputStatus;
    if (output.resume_label != .compatible) return error.InvalidResumeLabel;
}

pub fn checkTopologyDecision(
    allocator: std.mem.Allocator,
    request_case: model.RequestCase,
    route: model.WorkloadRoute,
) !void {
    const task = model.topologyTask();
    const protocol_simplexes = [_]topology.Simplex{
        task.protocol.simplexes[model.protocolIndex(request_case)],
    };
    const decisions = [_]topology.DecisionEdge{.{
        .input_index = model.inputIndex(request_case),
        .protocol_index = 0,
        .output_index = model.outputIndex(route),
    }};

    try topology.validateDecisionMap(
        allocator,
        .{ .simplexes = &protocol_simplexes },
        task.carrier,
        &decisions,
    );
}
