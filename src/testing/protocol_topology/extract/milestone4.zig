const std = @import("std");

const common = @import("common.zig");
const decoupled_site = @import("decoupled_site.zig");
const federation_placement = @import("federation_placement.zig");
const message_registry = @import("message_registry.zig");
const rda_merge = @import("rda_merge.zig");
const scheduling = @import("scheduling.zig");
const tasks = @import("../tasks.zig");
const workload_contract = @import("workload_contract.zig");

pub const SurfaceExtractionSummary = struct {
    workload_traces: usize,
    message_traces: usize,
    lease_traces: usize,
    reservation_traces: usize,
    federation_traces: usize,
    decoupled_traces: usize,
    rda_traces: usize,
};

pub fn validateMilestone4ExtractionCoverage(allocator: std.mem.Allocator) !SurfaceExtractionSummary {
    const workload_bound = common.ExtractionBound{ .max_cases = 2, .max_traces = 2, .max_events = 3, .max_depth = 3 };
    var workload = try workload_contract.extractWorkloadExecutionCarrier(allocator, workload_bound);
    defer workload.deinit();
    try requireEvidence(workload.metadata, .production_fixture);

    const message_bound = common.ExtractionBound{ .max_cases = 4, .max_traces = 4, .max_events = 2, .max_depth = 2 };
    var messages = try message_registry.extractMessageDispatchCarrier(allocator, message_bound);
    defer messages.deinit();
    try requireEvidence(messages.metadata, .production_fixture);

    const lease_bound = common.ExtractionBound{ .max_cases = 3, .max_traces = 3, .max_events = 2, .max_depth = 2 };
    var lease = try scheduling.generateLeaseExecutionCarrier(allocator, lease_bound);
    defer lease.deinit();
    try requireEvidence(lease.metadata, .generated_xi);

    const reservation_bound = common.ExtractionBound{ .max_cases = 4, .max_traces = 4, .max_events = 3, .max_depth = 3 };
    var reservation = try scheduling.generateReservationExecutionCarrier(allocator, reservation_bound);
    defer reservation.deinit();
    try requireEvidence(reservation.metadata, .generated_xi);

    const federation_bound = common.ExtractionBound{ .max_cases = 4, .max_traces = 4, .max_events = 4, .max_depth = 4 };
    var federation = try federation_placement.extractFederationPlacementCarrier(allocator, federation_bound);
    defer federation.deinit();
    try requireEvidence(federation.metadata, .bounded_model);

    const decoupled_bound = common.ExtractionBound{ .max_cases = 6, .max_traces = 6, .max_events = 7, .max_depth = 7 };
    var decoupled = try decoupled_site.generateFragmentExecutionCarrier(allocator, decoupled_bound);
    defer decoupled.deinit();
    try requireEvidence(decoupled.metadata, .generated_xi);

    const rda_bound = common.ExtractionBound{ .max_cases = 3, .max_traces = 3, .max_events = 3, .max_depth = 3 };
    var rda = try rda_merge.extractRdaMergeCarrier(allocator, rda_bound);
    defer rda.deinit();
    try requireEvidence(rda.metadata, .production_fixture);

    return .{
        .workload_traces = workload.traces.len,
        .message_traces = messages.traces.len,
        .lease_traces = lease.traces.len,
        .reservation_traces = reservation.traces.len,
        .federation_traces = federation.traces.len,
        .decoupled_traces = decoupled.traces.len,
        .rda_traces = rda.traces.len,
    };
}

fn requireEvidence(metadata: []const common.SourceMetadata, expected: tasks.CoverageEvidence) !void {
    if (metadata.len == 0) return error.MissingExtractionMetadata;
    for (metadata) |item| {
        try item.validate();
        if (item.evidence != expected) return error.ExtractionEvidenceMismatch;
    }
}

test "milestone 4 extraction coverage spans every implemented topology surface" {
    const summary = try validateMilestone4ExtractionCoverage(std.testing.allocator);

    try std.testing.expect(summary.workload_traces >= 2);
    try std.testing.expect(summary.message_traces >= 4);
    try std.testing.expect(summary.lease_traces >= 3);
    try std.testing.expect(summary.reservation_traces >= 4);
    try std.testing.expect(summary.federation_traces >= 4);
    try std.testing.expect(summary.decoupled_traces >= 6);
    try std.testing.expect(summary.rda_traces >= 3);
}
