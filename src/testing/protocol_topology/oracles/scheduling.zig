const std = @import("std");

const model = @import("../models/gateway_scheduling.zig");

pub const Violation = enum {
    none,
    duplicate_worker_reservation,
    release_without_reservation,
    assignment_after_worker_loss,
    nondeterministic_admission,
    owner_max_workers_exceeded,
    reserved_capacity_violation,
    cancellation_cleanup_leak,
};

pub fn checkNoDuplicateActiveWorkers(reservations: []const model.Reservation) Violation {
    var seen: u64 = 0;
    for (reservations) |reservation| {
        if (!reservation.active) continue;
        for (reservation.worker_ids[0..reservation.worker_count]) |worker_id| {
            if (worker_id >= 64) continue;
            const bit = @as(u64, 1) << @intCast(worker_id);
            if ((seen & bit) != 0) return .duplicate_worker_reservation;
            seen |= bit;
        }
    }
    return .none;
}

pub fn checkReleaseResult(snapshot_value: model.Snapshot) Violation {
    return if (snapshot_value.violation == .reservation_not_found)
        .release_without_reservation
    else
        .none;
}

pub fn checkAssignmentResult(snapshot_value: model.Snapshot) Violation {
    return if (snapshot_value.violation == .assignment_after_worker_loss)
        .assignment_after_worker_loss
    else
        .none;
}

pub fn checkDeterministicAdmission(
    left: model.ReservationView,
    right: model.ReservationView,
) Violation {
    if (left.worker_count != right.worker_count) return .nondeterministic_admission;
    for (left.workerSlice(), right.workerSlice()) |left_worker, right_worker| {
        if (left_worker != right_worker) return .nondeterministic_admission;
    }
    return .none;
}

pub fn checkOwnerPolicyBounds(scheduling: model.GatewaySchedulingModel) Violation {
    for (model.lease_owners) |owner| {
        const policy = scheduling.getSchedulingPolicy(owner);
        if (policy.max_workers) |limit| {
            if (scheduling.countLeasedWorkersForOwner(owner) > limit) return .owner_max_workers_exceeded;
        }
        if (policy.reserved_workers > 0 and scheduling.countCapacityForOwner(owner) < policy.reserved_workers) {
            return .reserved_capacity_violation;
        }
    }
    return .none;
}

pub fn checkCanceledJobCleanup(scheduling: model.GatewaySchedulingModel, job_id: model.JobId) Violation {
    if (scheduling.hasQueuedJob(job_id)) return .cancellation_cleanup_leak;
    if (scheduling.hasLeasedWorkersForJob(job_id)) return .cancellation_cleanup_leak;
    return .none;
}

test "oracle catches duplicate active worker reservations" {
    var left = model.Reservation{
        .id = 1,
        .job_id = 10,
        .workers_required = 1,
        .worker_count = 1,
    };
    left.worker_ids[0] = 7;

    var right = model.Reservation{
        .id = 2,
        .job_id = 20,
        .workers_required = 1,
        .worker_count = 1,
    };
    right.worker_ids[0] = 7;

    try std.testing.expectEqual(Violation.duplicate_worker_reservation, checkNoDuplicateActiveWorkers(&.{ left, right }));
}

test "oracle maps release without reservation to scheduling violation" {
    var scheduling = model.GatewaySchedulingModel.init();
    _ = scheduling.releaseReservation(123);

    try std.testing.expectEqual(Violation.release_without_reservation, checkReleaseResult(scheduling.snapshot()));
}

test "oracle maps assignment after worker loss to scheduling violation" {
    var scheduling = model.GatewaySchedulingModel.init();
    try scheduling.addWorker(model.makeWorker(1, .cuda, "sm_80"));

    const reserved = (try scheduling.reserve(model.makeRequirement(1, 1, .cuda, "sm_80"))).?;
    scheduling.loseWorker(1);
    _ = scheduling.commitReservation(reserved.id);

    try std.testing.expectEqual(Violation.assignment_after_worker_loss, checkAssignmentResult(scheduling.snapshot()));
}

test "oracle accepts deterministic repeated admission" {
    var left = model.GatewaySchedulingModel.init();
    var right = model.GatewaySchedulingModel.init();
    for ([_]model.Worker{
        model.makeWorker(1, .cpu, null),
        model.makeWorker(2, .cuda, "sm_80"),
        model.makeWorker(3, .cuda, "sm_80"),
    }) |item| {
        try left.addWorker(item);
        try right.addWorker(item);
    }

    const left_reservation = (try left.reserve(model.makeRequirement(77, 2, .cuda, "sm_80"))).?;
    const right_reservation = (try right.reserve(model.makeRequirement(77, 2, .cuda, "sm_80"))).?;

    try std.testing.expectEqual(Violation.none, checkDeterministicAdmission(left_reservation, right_reservation));
}

test "oracle catches owner max worker violations" {
    var scheduling = model.GatewaySchedulingModel.init();
    scheduling.setSchedulingPolicy(.training, .{ .max_workers = 1 });
    try scheduling.addWorker(model.makeWorker(1, .cuda, "sm_80"));
    try scheduling.addWorker(model.makeWorker(2, .cuda, "sm_80"));
    scheduling.workers[0].state = .reserved;
    scheduling.workers[0].lease_owner = .training;
    scheduling.workers[1].state = .assigned;
    scheduling.workers[1].lease_owner = .training;

    try std.testing.expectEqual(Violation.owner_max_workers_exceeded, checkOwnerPolicyBounds(scheduling));
}

test "oracle catches cross owner reserved capacity loss" {
    var scheduling = model.GatewaySchedulingModel.init();
    scheduling.setSchedulingPolicy(.inference, .{
        .worker_class = .cuda,
        .target_arch = "sm_80",
        .reserved_workers = 1,
    });
    try scheduling.addWorker(model.makeWorker(1, .cuda, "sm_80"));
    scheduling.workers[0].state = .reserved;
    scheduling.workers[0].lease_owner = .training;

    try std.testing.expectEqual(Violation.reserved_capacity_violation, checkOwnerPolicyBounds(scheduling));
}

test "oracle accepts queued reserved cancellation cleanup" {
    var scheduling = model.GatewaySchedulingModel.init();
    try scheduling.addWorker(model.makeWorker(1, .cuda, "sm_80"));

    const reservation = (try scheduling.reserve(model.makeRequirement(88, 1, .cuda, "sm_80"))).?;
    _ = scheduling.commitReservationToQueue(reservation.id).?;
    _ = scheduling.cancelQueuedJob(88).?;

    try std.testing.expectEqual(Violation.none, checkCanceledJobCleanup(scheduling, 88));
}

test "oracle catches cancellation cleanup leaks" {
    var scheduling = model.GatewaySchedulingModel.init();
    try scheduling.addWorker(model.makeWorker(1, .cuda, "sm_80"));
    scheduling.workers[0].state = .reserved;
    scheduling.workers[0].lease_owner = .training;
    scheduling.workers[0].job_id = 99;

    try std.testing.expectEqual(Violation.cancellation_cleanup_leak, checkCanceledJobCleanup(scheduling, 99));
}
