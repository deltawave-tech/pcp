const std = @import("std");

const real_scheduling = @import("../../../nodes/gateway/scheduling.zig");
const real_training_controller = @import("../../../nodes/gateway/controllers/training_controller.zig");

pub const max_workers = 8;
pub const max_reservations = 8;
pub const max_queued_jobs = 8;

pub const WorkerClass = real_scheduling.WorkerClass;
pub const SchedulingPolicy = real_scheduling.SchedulingPolicy;
pub const LeaseOwner = real_training_controller.WorkerLeaseOwner;
pub const lease_owners = [_]LeaseOwner{ .inference, .training, .rl };
pub const owner_count = lease_owners.len;
pub const WorkerId = u16;
pub const JobId = u32;
pub const ReservationId = u32;

pub const WorkerState = enum {
    offline,
    idle,
    reserved,
    assigned,
};

pub const Decision = enum {
    idle,
    reserved,
    committed,
    queued,
    started,
    completed,
    released,
    expired,
    canceled,
    rejected,
};

pub const Violation = enum {
    none,
    invalid_worker,
    invalid_request,
    duplicate_job_reservation,
    duplicate_worker_reservation,
    duplicate_queued_job,
    reservation_not_found,
    reservation_unavailable,
    assignment_after_worker_loss,
    worker_already_busy,
    queue_full,
    queued_job_not_found,
    active_job_not_found,
};

pub const Worker = struct {
    id: WorkerId,
    worker_class: WorkerClass,
    target_arch: ?[]const u8 = null,
    state: WorkerState = .idle,
    lease_owner: ?LeaseOwner = null,
    reservation_id: ?ReservationId = null,
    job_id: ?JobId = null,
};

pub const JobRequirement = struct {
    job_id: JobId,
    owner: LeaseOwner = .training,
    workers_required: usize,
    policy: SchedulingPolicy = .{},
};

pub const Reservation = struct {
    id: ReservationId,
    job_id: JobId,
    owner: LeaseOwner = .training,
    workers_required: usize,
    worker_ids: [max_workers]WorkerId = [_]WorkerId{0} ** max_workers,
    worker_count: usize = 0,
    active: bool = true,
    committed: bool = false,
};

pub const ReservationView = struct {
    id: ReservationId,
    job_id: JobId,
    owner: LeaseOwner = .training,
    worker_ids: [max_workers]WorkerId = [_]WorkerId{0} ** max_workers,
    worker_count: usize = 0,

    pub fn workerSlice(self: *const ReservationView) []const WorkerId {
        return self.worker_ids[0..self.worker_count];
    }
};

pub const QueuedJob = struct {
    job_id: JobId,
    owner: LeaseOwner = .training,
    workers_required: usize,
    policy: SchedulingPolicy = .{},
    worker_ids: [max_workers]WorkerId = [_]WorkerId{0} ** max_workers,
    worker_count: usize = 0,
    reserved_workers: bool = false,
};

pub const QueuedJobView = struct {
    job_id: JobId,
    owner: LeaseOwner = .training,
    queue_position: usize = 0,
    worker_ids: [max_workers]WorkerId = [_]WorkerId{0} ** max_workers,
    worker_count: usize = 0,
    reserved_workers: bool = false,

    pub fn workerSlice(self: *const QueuedJobView) []const WorkerId {
        return self.worker_ids[0..self.worker_count];
    }
};

pub const Snapshot = struct {
    workers_total: usize = 0,
    idle_workers: usize = 0,
    reserved_workers: usize = 0,
    assigned_workers: usize = 0,
    active_reservations: usize = 0,
    queued_jobs: usize = 0,
    active_jobs: usize = 0,
    canceled_jobs: usize = 0,
    owner_leases: [owner_count]usize = [_]usize{0} ** owner_count,
    owner_available: [owner_count]usize = [_]usize{0} ** owner_count,
    decision: Decision = .idle,
    violation: Violation = .none,

    pub fn eql(a: Snapshot, b: Snapshot) bool {
        return a.workers_total == b.workers_total and
            a.idle_workers == b.idle_workers and
            a.reserved_workers == b.reserved_workers and
            a.assigned_workers == b.assigned_workers and
            a.active_reservations == b.active_reservations and
            a.queued_jobs == b.queued_jobs and
            a.active_jobs == b.active_jobs and
            a.canceled_jobs == b.canceled_jobs and
            std.mem.eql(usize, a.owner_leases[0..], b.owner_leases[0..]) and
            std.mem.eql(usize, a.owner_available[0..], b.owner_available[0..]) and
            a.decision == b.decision and
            a.violation == b.violation;
    }
};

pub const GatewaySchedulingModel = struct {
    workers: [max_workers]Worker = undefined,
    worker_count: usize = 0,
    reservations: [max_reservations]Reservation = undefined,
    reservation_count: usize = 0,
    queued_jobs: [max_queued_jobs]QueuedJob = undefined,
    queued_job_count: usize = 0,
    active_job: ?QueuedJob = null,
    canceled_job_count: usize = 0,
    scheduling_policies: [owner_count]SchedulingPolicy = [_]SchedulingPolicy{ .{}, .{}, .{} },
    next_reservation_id: ReservationId = 1,
    decision: Decision = .idle,
    violation: Violation = .none,

    pub fn init() GatewaySchedulingModel {
        return .{};
    }

    pub fn setSchedulingPolicy(self: *GatewaySchedulingModel, owner: LeaseOwner, policy: SchedulingPolicy) void {
        self.scheduling_policies[ownerIndex(owner)] = policy;
    }

    pub fn getSchedulingPolicy(self: GatewaySchedulingModel, owner: LeaseOwner) SchedulingPolicy {
        return self.scheduling_policies[ownerIndex(owner)];
    }

    pub fn addWorker(self: *GatewaySchedulingModel, worker: Worker) !void {
        if (self.worker_count >= max_workers) return error.TooManyWorkers;
        if (self.findWorkerIndex(worker.id) != null) return error.DuplicateWorker;
        self.workers[self.worker_count] = worker;
        self.worker_count += 1;
    }

    pub fn reserve(self: *GatewaySchedulingModel, requirement: JobRequirement) !?ReservationView {
        if (self.violation != .none) return null;
        if (requirement.workers_required == 0 or requirement.workers_required > max_workers) {
            self.reject(.invalid_request);
            return null;
        }
        if (self.activeReservationForJob(requirement.job_id) != null) {
            self.reject(.duplicate_job_reservation);
            return null;
        }
        if (self.queuedJobIndex(requirement.job_id) != null or self.activeJobIs(requirement.job_id)) {
            self.reject(.duplicate_queued_job);
            return null;
        }
        if (self.reservation_count >= max_reservations) return error.TooManyReservations;

        const reservation_id = self.next_reservation_id;
        var selected: [max_workers]WorkerId = [_]WorkerId{0} ** max_workers;
        const selected_count = self.acquireWorkersForRequirement(requirement, .reserved, reservation_id, &selected);
        if (selected_count != requirement.workers_required) {
            self.releaseSelectedWorkers(selected[0..selected_count]);
            self.reject(.reservation_unavailable);
            return null;
        }

        self.next_reservation_id += 1;
        var reservation = Reservation{
            .id = reservation_id,
            .job_id = requirement.job_id,
            .owner = requirement.owner,
            .workers_required = requirement.workers_required,
            .worker_count = selected_count,
        };
        @memcpy(reservation.worker_ids[0..selected_count], selected[0..selected_count]);

        self.reservations[self.reservation_count] = reservation;
        self.reservation_count += 1;
        self.decision = .reserved;
        return reservationView(reservation);
    }

    pub fn enqueueJob(self: *GatewaySchedulingModel, requirement: JobRequirement) !?QueuedJobView {
        if (self.violation != .none) return null;
        if (requirement.workers_required == 0 or requirement.workers_required > max_workers) {
            self.reject(.invalid_request);
            return null;
        }
        if (self.queuedJobIndex(requirement.job_id) != null or self.activeJobIs(requirement.job_id)) {
            self.reject(.duplicate_queued_job);
            return null;
        }
        if (self.queued_job_count >= max_queued_jobs) {
            self.reject(.queue_full);
            return null;
        }

        const job = QueuedJob{
            .job_id = requirement.job_id,
            .owner = requirement.owner,
            .workers_required = requirement.workers_required,
            .policy = requirement.policy,
        };
        const queue_position = self.queued_job_count;
        self.queued_jobs[self.queued_job_count] = job;
        self.queued_job_count += 1;
        self.decision = .queued;
        return queuedJobView(job, queue_position);
    }

    pub fn commitReservation(self: *GatewaySchedulingModel, reservation_id: ReservationId) ?ReservationView {
        const reservation = self.activeReservation(reservation_id) orelse {
            self.reject(.reservation_not_found);
            return null;
        };

        for (reservation.worker_ids[0..reservation.worker_count]) |worker_id| {
            const worker = self.getWorker(worker_id) orelse {
                self.reject(.assignment_after_worker_loss);
                return null;
            };
            if (worker.state != .reserved or worker.reservation_id == null or worker.reservation_id.? != reservation.id or worker.lease_owner != reservation.owner) {
                self.reject(.assignment_after_worker_loss);
                return null;
            }
        }

        for (reservation.worker_ids[0..reservation.worker_count]) |worker_id| {
            const worker = self.getWorker(worker_id).?;
            worker.state = .assigned;
        }
        reservation.active = false;
        reservation.committed = true;
        self.decision = .committed;
        return reservationView(reservation.*);
    }

    pub fn commitReservationToQueue(self: *GatewaySchedulingModel, reservation_id: ReservationId) ?QueuedJobView {
        const reservation = self.activeReservation(reservation_id) orelse {
            self.reject(.reservation_not_found);
            return null;
        };
        if (self.queued_job_count >= max_queued_jobs) {
            self.reject(.queue_full);
            return null;
        }
        if (self.queuedJobIndex(reservation.job_id) != null or self.activeJobIs(reservation.job_id)) {
            self.reject(.duplicate_queued_job);
            return null;
        }

        for (reservation.worker_ids[0..reservation.worker_count]) |worker_id| {
            const worker = self.getWorker(worker_id) orelse {
                self.reject(.reservation_unavailable);
                return null;
            };
            if (worker.state != .reserved or worker.reservation_id == null or worker.reservation_id.? != reservation.id or worker.lease_owner != reservation.owner) {
                self.reject(.reservation_unavailable);
                return null;
            }
        }

        var job = QueuedJob{
            .job_id = reservation.job_id,
            .owner = reservation.owner,
            .workers_required = reservation.workers_required,
            .worker_count = reservation.worker_count,
            .reserved_workers = true,
        };
        @memcpy(job.worker_ids[0..reservation.worker_count], reservation.worker_ids[0..reservation.worker_count]);
        self.insertQueuedJobFront(job);

        reservation.active = false;
        reservation.committed = true;
        self.decision = .queued;
        return queuedJobView(job, 0);
    }

    pub fn startNextQueuedJob(self: *GatewaySchedulingModel) ?QueuedJobView {
        if (self.queued_job_count == 0) {
            self.reject(.queued_job_not_found);
            return null;
        }
        if (self.active_job != null) {
            self.reject(.worker_already_busy);
            return null;
        }

        var job = self.queued_jobs[0];
        self.removeQueuedJobAt(0);

        if (job.reserved_workers) {
            if (!self.reservedQueuedJobStillLeased(job)) {
                self.releaseJobWorkers(job);
                self.reject(.reservation_unavailable);
                return null;
            }
            for (job.worker_ids[0..job.worker_count]) |worker_id| {
                const worker = self.getWorker(worker_id).?;
                worker.state = .assigned;
            }
        } else {
            var selected: [max_workers]WorkerId = [_]WorkerId{0} ** max_workers;
            const selected_count = self.acquireWorkersForRequirement(.{
                .job_id = job.job_id,
                .owner = job.owner,
                .workers_required = job.workers_required,
                .policy = job.policy,
            }, .assigned, null, &selected);
            if (selected_count != job.workers_required) {
                self.releaseSelectedWorkers(selected[0..selected_count]);
                self.reject(.reservation_unavailable);
                return null;
            }
            job.worker_count = selected_count;
            @memcpy(job.worker_ids[0..selected_count], selected[0..selected_count]);
        }

        self.active_job = job;
        self.decision = .started;
        return queuedJobView(job, 0);
    }

    pub fn completeActiveJob(self: *GatewaySchedulingModel, job_id: JobId) ?QueuedJobView {
        const job = self.active_job orelse {
            self.reject(.active_job_not_found);
            return null;
        };
        if (job.job_id != job_id) {
            self.reject(.active_job_not_found);
            return null;
        }

        const view = queuedJobView(job, 0);
        self.releaseJobWorkers(job);
        self.active_job = null;
        self.decision = .completed;
        return view;
    }

    pub fn cancelQueuedJob(self: *GatewaySchedulingModel, job_id: JobId) ?QueuedJobView {
        const index = self.queuedJobIndex(job_id) orelse {
            self.reject(.queued_job_not_found);
            return null;
        };

        const job = self.queued_jobs[index];
        const view = queuedJobView(job, index);
        if (job.reserved_workers) self.releaseJobWorkers(job);
        self.removeQueuedJobAt(index);
        self.canceled_job_count += 1;
        self.decision = .canceled;
        return view;
    }

    pub fn cancelActiveJob(self: *GatewaySchedulingModel, job_id: JobId) ?QueuedJobView {
        const job = self.active_job orelse {
            self.reject(.active_job_not_found);
            return null;
        };
        if (job.job_id != job_id) {
            self.reject(.active_job_not_found);
            return null;
        }

        const view = queuedJobView(job, 0);
        self.releaseJobWorkers(job);
        self.active_job = null;
        self.canceled_job_count += 1;
        self.decision = .canceled;
        return view;
    }

    pub fn releaseReservation(self: *GatewaySchedulingModel, reservation_id: ReservationId) ?ReservationView {
        const reservation = self.activeReservation(reservation_id) orelse {
            self.reject(.reservation_not_found);
            return null;
        };

        const view = reservationView(reservation.*);
        self.releaseReservationWorkers(reservation.*);
        reservation.active = false;
        self.decision = .released;
        return view;
    }

    pub fn expireReservation(self: *GatewaySchedulingModel, reservation_id: ReservationId) ?ReservationView {
        const reservation = self.activeReservation(reservation_id) orelse {
            self.reject(.reservation_not_found);
            return null;
        };

        const view = reservationView(reservation.*);
        self.releaseReservationWorkers(reservation.*);
        reservation.active = false;
        self.decision = .expired;
        return view;
    }

    pub fn loseWorker(self: *GatewaySchedulingModel, worker_id: WorkerId) void {
        const worker = self.getWorker(worker_id) orelse {
            self.reject(.invalid_worker);
            return;
        };
        worker.state = .offline;
    }

    pub fn restoreWorker(self: *GatewaySchedulingModel, worker_id: WorkerId) void {
        const worker = self.getWorker(worker_id) orelse {
            self.reject(.invalid_worker);
            return;
        };
        worker.state = .idle;
        worker.lease_owner = null;
        worker.reservation_id = null;
        worker.job_id = null;
    }

    pub fn snapshot(self: GatewaySchedulingModel) Snapshot {
        var snapshot_value = Snapshot{
            .workers_total = self.worker_count,
            .queued_jobs = self.queued_job_count,
            .active_jobs = if (self.active_job == null) 0 else 1,
            .canceled_jobs = self.canceled_job_count,
            .decision = self.decision,
            .violation = self.violation,
        };

        for (self.workers[0..self.worker_count]) |worker| {
            switch (worker.state) {
                .offline => {},
                .idle => snapshot_value.idle_workers += 1,
                .reserved => snapshot_value.reserved_workers += 1,
                .assigned => snapshot_value.assigned_workers += 1,
            }
            if (worker.state != .offline) {
                if (worker.lease_owner) |owner| {
                    snapshot_value.owner_leases[ownerIndex(owner)] += 1;
                }
            }
        }
        for (self.reservations[0..self.reservation_count]) |reservation| {
            if (reservation.active) snapshot_value.active_reservations += 1;
        }
        for (lease_owners) |owner| {
            snapshot_value.owner_available[ownerIndex(owner)] = self.availableWorkerCountForOwner(owner);
        }

        return snapshot_value;
    }

    pub fn activeReservationSlice(self: *const GatewaySchedulingModel) []const Reservation {
        return self.reservations[0..self.reservation_count];
    }

    pub fn queuedJobSlice(self: *const GatewaySchedulingModel) []const QueuedJob {
        return self.queued_jobs[0..self.queued_job_count];
    }

    pub fn countLeasedWorkersForOwner(self: GatewaySchedulingModel, owner: LeaseOwner) usize {
        var count: usize = 0;
        for (self.workers[0..self.worker_count]) |worker| {
            if (worker.state == .offline) continue;
            if (worker.lease_owner) |lease_owner| {
                if (lease_owner == owner) count += 1;
            }
        }
        return count;
    }

    pub fn countCapacityForOwner(self: GatewaySchedulingModel, owner: LeaseOwner) usize {
        const policy = self.scheduling_policies[ownerIndex(owner)];
        var count: usize = 0;
        for (self.workers[0..self.worker_count]) |worker| {
            if (worker.state == .offline) continue;
            if (!policy.allowsWorker(worker.worker_class, worker.target_arch)) continue;

            if (worker.lease_owner == null) {
                count += 1;
                continue;
            }
            if (worker.lease_owner.? == owner) count += 1;
        }
        return count;
    }

    pub fn availableWorkerCountForOwner(self: GatewaySchedulingModel, owner: LeaseOwner) usize {
        if (self.worker_count == 0) return 0;

        var hypothetical_storage: [max_workers]bool = [_]bool{false} ** max_workers;
        const hypothetical = hypothetical_storage[0..self.worker_count];

        var count: usize = 0;
        while (count < self.worker_count) {
            var progress = false;
            for (self.workers[0..self.worker_count], 0..) |_, index| {
                if (!self.canLeaseWorkerWithHypothetical(index, owner, hypothetical)) continue;
                hypothetical[index] = true;
                count += 1;
                progress = true;
                break;
            }
            if (!progress) break;
        }
        return count;
    }

    pub fn hasLeasedWorkersForJob(self: GatewaySchedulingModel, job_id: JobId) bool {
        for (self.workers[0..self.worker_count]) |worker| {
            if (worker.job_id != null and worker.job_id.? == job_id and worker.lease_owner != null) return true;
        }
        return false;
    }

    pub fn hasQueuedJob(self: GatewaySchedulingModel, job_id: JobId) bool {
        return self.queuedJobIndex(job_id) != null;
    }

    fn acquireWorkersForRequirement(
        self: *GatewaySchedulingModel,
        requirement: JobRequirement,
        worker_state: WorkerState,
        reservation_id: ?ReservationId,
        selected: *[max_workers]WorkerId,
    ) usize {
        var selected_count: usize = 0;
        for (self.workers[0..self.worker_count]) |worker| {
            if (!self.canLeaseWorkerForRequirement(worker.id, requirement)) continue;
            const selected_worker = self.getWorker(worker.id).?;
            selected_worker.state = worker_state;
            selected_worker.lease_owner = requirement.owner;
            selected_worker.reservation_id = reservation_id;
            selected_worker.job_id = requirement.job_id;
            selected[selected_count] = worker.id;
            selected_count += 1;
            if (selected_count == requirement.workers_required) break;
        }
        return selected_count;
    }

    fn canLeaseWorkerForRequirement(self: GatewaySchedulingModel, worker_id: WorkerId, requirement: JobRequirement) bool {
        const worker_index = self.findWorkerIndex(worker_id) orelse return false;
        const worker = self.workers[worker_index];
        if (worker.state != .idle or worker.lease_owner != null) return false;

        const owner_policy = self.scheduling_policies[ownerIndex(requirement.owner)];
        if (!owner_policy.allowsWorker(worker.worker_class, worker.target_arch)) return false;
        if (!requirement.policy.allowsWorker(worker.worker_class, worker.target_arch)) return false;

        if (owner_policy.max_workers) |limit| {
            if (self.countLeasedWorkersForOwner(requirement.owner) >= limit) return false;
        }

        return self.reservationsSatisfiedAfterLease(worker_id, requirement.owner);
    }

    fn canLeaseWorkerWithHypothetical(
        self: GatewaySchedulingModel,
        worker_index: usize,
        owner: LeaseOwner,
        hypothetical: []const bool,
    ) bool {
        const worker = self.workers[worker_index];
        if (worker.state != .idle or worker.lease_owner != null or hypothetical[worker_index]) return false;

        const policy = self.scheduling_policies[ownerIndex(owner)];
        if (!policy.allowsWorker(worker.worker_class, worker.target_arch)) return false;

        if (policy.max_workers) |limit| {
            if (self.countLeasedWorkersForOwner(owner) + countHypothetical(hypothetical) >= limit) return false;
        }

        for (lease_owners) |reserved_owner| {
            if (reserved_owner == owner) continue;

            const reserved_policy = self.scheduling_policies[ownerIndex(reserved_owner)];
            if (reserved_policy.reserved_workers == 0) continue;
            if (self.countCapacityForOwnerWithHypothetical(reserved_owner, owner, worker_index, hypothetical) < reserved_policy.reserved_workers) {
                return false;
            }
        }

        return true;
    }

    fn reservationsSatisfiedAfterLease(self: GatewaySchedulingModel, candidate_worker_id: WorkerId, new_owner: LeaseOwner) bool {
        for (lease_owners) |owner| {
            if (owner == new_owner) continue;

            const policy = self.scheduling_policies[ownerIndex(owner)];
            if (policy.reserved_workers == 0) continue;
            if (self.countCapacityForOwnerAfterLease(owner, candidate_worker_id, new_owner) < policy.reserved_workers) {
                return false;
            }
        }
        return true;
    }

    fn countCapacityForOwnerAfterLease(
        self: GatewaySchedulingModel,
        owner: LeaseOwner,
        candidate_worker_id: WorkerId,
        new_owner: LeaseOwner,
    ) usize {
        const policy = self.scheduling_policies[ownerIndex(owner)];
        var count: usize = 0;
        for (self.workers[0..self.worker_count]) |worker| {
            if (worker.state == .offline) continue;
            if (!policy.allowsWorker(worker.worker_class, worker.target_arch)) continue;

            const effective_owner: ?LeaseOwner = if (worker.id == candidate_worker_id)
                new_owner
            else
                worker.lease_owner;

            if (effective_owner == null) {
                count += 1;
                continue;
            }
            if (effective_owner.? == owner) count += 1;
        }
        return count;
    }

    fn countCapacityForOwnerWithHypothetical(
        self: GatewaySchedulingModel,
        owner: LeaseOwner,
        hypothetical_owner: LeaseOwner,
        candidate_index: usize,
        hypothetical: []const bool,
    ) usize {
        const policy = self.scheduling_policies[ownerIndex(owner)];
        var count: usize = 0;
        for (self.workers[0..self.worker_count], 0..) |worker, index| {
            if (worker.state == .offline) continue;
            if (!policy.allowsWorker(worker.worker_class, worker.target_arch)) continue;

            const effective_owner: ?LeaseOwner = if (worker.lease_owner) |lease_owner|
                lease_owner
            else if (index == candidate_index or hypothetical[index])
                hypothetical_owner
            else
                null;

            if (effective_owner == null) {
                count += 1;
                continue;
            }
            if (effective_owner.? == owner) count += 1;
        }
        return count;
    }

    fn reservedQueuedJobStillLeased(self: GatewaySchedulingModel, job: QueuedJob) bool {
        if (job.worker_count != job.workers_required) return false;
        for (job.worker_ids[0..job.worker_count]) |worker_id| {
            const index = self.findWorkerIndex(worker_id) orelse return false;
            const worker = self.workers[index];
            if (worker.state != .reserved) return false;
            if (worker.lease_owner == null or worker.lease_owner.? != job.owner) return false;
            if (worker.job_id == null or worker.job_id.? != job.job_id) return false;
        }
        return true;
    }

    fn releaseReservationWorkers(self: *GatewaySchedulingModel, reservation: Reservation) void {
        for (reservation.worker_ids[0..reservation.worker_count]) |worker_id| {
            if (self.getWorker(worker_id)) |worker| {
                if (worker.reservation_id != null and worker.reservation_id.? == reservation.id) {
                    clearWorkerLease(worker);
                }
            }
        }
    }

    fn releaseJobWorkers(self: *GatewaySchedulingModel, job: QueuedJob) void {
        for (job.worker_ids[0..job.worker_count]) |worker_id| {
            if (self.getWorker(worker_id)) |worker| {
                if (worker.job_id != null and worker.job_id.? == job.job_id and worker.lease_owner == job.owner) {
                    clearWorkerLease(worker);
                }
            }
        }
    }

    fn releaseSelectedWorkers(self: *GatewaySchedulingModel, worker_ids: []const WorkerId) void {
        for (worker_ids) |worker_id| {
            if (self.getWorker(worker_id)) |worker| clearWorkerLease(worker);
        }
    }

    fn insertQueuedJobFront(self: *GatewaySchedulingModel, job: QueuedJob) void {
        std.debug.assert(self.queued_job_count < max_queued_jobs);
        var index = self.queued_job_count;
        while (index > 0) : (index -= 1) {
            self.queued_jobs[index] = self.queued_jobs[index - 1];
        }
        self.queued_jobs[0] = job;
        self.queued_job_count += 1;
    }

    fn removeQueuedJobAt(self: *GatewaySchedulingModel, index: usize) void {
        std.debug.assert(index < self.queued_job_count);
        var cursor = index;
        while (cursor + 1 < self.queued_job_count) : (cursor += 1) {
            self.queued_jobs[cursor] = self.queued_jobs[cursor + 1];
        }
        self.queued_job_count -= 1;
    }

    fn activeReservation(self: *GatewaySchedulingModel, reservation_id: ReservationId) ?*Reservation {
        for (self.reservations[0..self.reservation_count]) |*reservation| {
            if (reservation.id == reservation_id and reservation.active) return reservation;
        }
        return null;
    }

    fn activeReservationForJob(self: GatewaySchedulingModel, job_id: JobId) ?Reservation {
        for (self.reservations[0..self.reservation_count]) |reservation| {
            if (reservation.active and reservation.job_id == job_id) return reservation;
        }
        return null;
    }

    fn queuedJobIndex(self: GatewaySchedulingModel, job_id: JobId) ?usize {
        for (self.queued_jobs[0..self.queued_job_count], 0..) |job, index| {
            if (job.job_id == job_id) return index;
        }
        return null;
    }

    fn activeJobIs(self: GatewaySchedulingModel, job_id: JobId) bool {
        if (self.active_job) |job| return job.job_id == job_id;
        return false;
    }

    fn getWorker(self: *GatewaySchedulingModel, worker_id: WorkerId) ?*Worker {
        const index = self.findWorkerIndex(worker_id) orelse return null;
        return &self.workers[index];
    }

    fn findWorkerIndex(self: GatewaySchedulingModel, worker_id: WorkerId) ?usize {
        for (self.workers[0..self.worker_count], 0..) |worker, index| {
            if (worker.id == worker_id) return index;
        }
        return null;
    }

    fn reject(self: *GatewaySchedulingModel, violation: Violation) void {
        self.violation = violation;
        self.decision = .rejected;
    }
};

pub fn makeWorker(id: WorkerId, worker_class: WorkerClass, target_arch: ?[]const u8) Worker {
    return .{
        .id = id,
        .worker_class = worker_class,
        .target_arch = target_arch,
    };
}

pub fn makeRequirement(
    job_id: JobId,
    workers_required: usize,
    worker_class: WorkerClass,
    target_arch: ?[]const u8,
) JobRequirement {
    return makeRequirementForOwner(.training, job_id, workers_required, worker_class, target_arch);
}

pub fn makeRequirementForOwner(
    owner: LeaseOwner,
    job_id: JobId,
    workers_required: usize,
    worker_class: WorkerClass,
    target_arch: ?[]const u8,
) JobRequirement {
    return .{
        .job_id = job_id,
        .owner = owner,
        .workers_required = workers_required,
        .policy = .{
            .worker_class = worker_class,
            .target_arch = target_arch,
        },
    };
}

pub fn ownerIndex(owner: LeaseOwner) usize {
    return switch (owner) {
        .inference => 0,
        .training => 1,
        .rl => 2,
    };
}

fn reservationView(reservation: Reservation) ReservationView {
    var view = ReservationView{
        .id = reservation.id,
        .job_id = reservation.job_id,
        .owner = reservation.owner,
        .worker_count = reservation.worker_count,
    };
    @memcpy(view.worker_ids[0..reservation.worker_count], reservation.worker_ids[0..reservation.worker_count]);
    return view;
}

fn queuedJobView(job: QueuedJob, queue_position: usize) QueuedJobView {
    var view = QueuedJobView{
        .job_id = job.job_id,
        .owner = job.owner,
        .queue_position = queue_position,
        .worker_count = job.worker_count,
        .reserved_workers = job.reserved_workers,
    };
    @memcpy(view.worker_ids[0..job.worker_count], job.worker_ids[0..job.worker_count]);
    return view;
}

fn clearWorkerLease(worker: *Worker) void {
    worker.state = .idle;
    worker.lease_owner = null;
    worker.reservation_id = null;
    worker.job_id = null;
}

fn countHypothetical(hypothetical: []const bool) usize {
    var count: usize = 0;
    for (hypothetical) |leased| {
        if (leased) count += 1;
    }
    return count;
}

test "model reserves first matching idle workers deterministically" {
    var left = GatewaySchedulingModel.init();
    var right = GatewaySchedulingModel.init();
    for ([_]Worker{
        makeWorker(7, .cpu, null),
        makeWorker(8, .cuda, "sm_80"),
        makeWorker(9, .cuda, "sm_80"),
    }) |item| {
        try left.addWorker(item);
        try right.addWorker(item);
    }

    const left_reservation = (try left.reserve(makeRequirement(1, 2, .cuda, "sm_80"))).?;
    const right_reservation = (try right.reserve(makeRequirement(1, 2, .cuda, "sm_80"))).?;

    try std.testing.expectEqual(@as(usize, 2), left_reservation.worker_count);
    try std.testing.expectEqual(@as(WorkerId, 8), left_reservation.worker_ids[0]);
    try std.testing.expectEqual(@as(WorkerId, 9), left_reservation.worker_ids[1]);
    try std.testing.expectEqualSlices(WorkerId, left_reservation.workerSlice(), right_reservation.workerSlice());
    try std.testing.expect(Snapshot.eql(left.snapshot(), right.snapshot()));
}

test "model rejects duplicate active reservation for one job" {
    var model = GatewaySchedulingModel.init();
    try model.addWorker(makeWorker(1, .cuda, "sm_80"));
    try model.addWorker(makeWorker(2, .cuda, "sm_80"));

    _ = try model.reserve(makeRequirement(42, 1, .cuda, "sm_80"));
    _ = try model.reserve(makeRequirement(42, 1, .cuda, "sm_80"));

    try std.testing.expectEqual(Violation.duplicate_job_reservation, model.snapshot().violation);
}

test "model catches release without reservation" {
    var model = GatewaySchedulingModel.init();
    try model.addWorker(makeWorker(1, .cpu, null));

    _ = model.releaseReservation(99);

    try std.testing.expectEqual(Violation.reservation_not_found, model.snapshot().violation);
}

test "model catches assignment after worker loss" {
    var model = GatewaySchedulingModel.init();
    try model.addWorker(makeWorker(1, .cuda, "sm_80"));

    const reserved = (try model.reserve(makeRequirement(1, 1, .cuda, "sm_80"))).?;
    model.loseWorker(1);
    _ = model.commitReservation(reserved.id);

    try std.testing.expectEqual(Violation.assignment_after_worker_loss, model.snapshot().violation);
}

test "reservation expiry releases workers" {
    var model = GatewaySchedulingModel.init();
    try model.addWorker(makeWorker(1, .cpu, null));
    try model.addWorker(makeWorker(2, .cpu, null));

    const reserved = (try model.reserve(makeRequirement(7, 2, .cpu, null))).?;
    _ = model.expireReservation(reserved.id);

    const snapshot_value = model.snapshot();
    try std.testing.expectEqual(Decision.expired, snapshot_value.decision);
    try std.testing.expectEqual(@as(usize, 2), snapshot_value.idle_workers);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.active_reservations);
}

test "owner max workers caps additional leases" {
    var model = GatewaySchedulingModel.init();
    model.setSchedulingPolicy(.training, .{
        .worker_class = .cuda,
        .target_arch = "sm_80",
        .max_workers = 1,
    });
    try model.addWorker(makeWorker(1, .cuda, "sm_80"));
    try model.addWorker(makeWorker(2, .cuda, "sm_80"));

    _ = try model.reserve(makeRequirement(10, 1, .cuda, "sm_80"));
    _ = try model.reserve(makeRequirement(11, 1, .cuda, "sm_80"));

    const snapshot_value = model.snapshot();
    try std.testing.expectEqual(Violation.reservation_unavailable, snapshot_value.violation);
    try std.testing.expectEqual(@as(usize, 1), snapshot_value.owner_leases[ownerIndex(.training)]);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.owner_available[ownerIndex(.training)]);
}

test "cross owner reserved capacity is preserved" {
    var model = GatewaySchedulingModel.init();
    model.setSchedulingPolicy(.inference, .{
        .worker_class = .cuda,
        .target_arch = "sm_80",
        .reserved_workers = 1,
    });
    model.setSchedulingPolicy(.training, .{
        .worker_class = .cuda,
        .target_arch = "sm_80",
    });
    try model.addWorker(makeWorker(1, .cuda, "sm_80"));
    try model.addWorker(makeWorker(2, .cuda, "sm_80"));

    _ = try model.reserve(makeRequirement(20, 2, .cuda, "sm_80"));

    const snapshot_value = model.snapshot();
    try std.testing.expectEqual(Violation.reservation_unavailable, snapshot_value.violation);
    try std.testing.expectEqual(@as(usize, 2), snapshot_value.idle_workers);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.owner_leases[ownerIndex(.training)]);
    try std.testing.expectEqual(@as(usize, 1), snapshot_value.owner_available[ownerIndex(.training)]);
    try std.testing.expectEqual(@as(usize, 2), model.countCapacityForOwner(.inference));
}

test "reserved job commit is queued ahead of normal jobs" {
    var model = GatewaySchedulingModel.init();
    try model.addWorker(makeWorker(1, .cuda, "sm_80"));
    try model.addWorker(makeWorker(2, .cuda, "sm_80"));

    _ = try model.enqueueJob(makeRequirement(30, 1, .cuda, "sm_80"));
    const reservation = (try model.reserve(makeRequirement(31, 1, .cuda, "sm_80"))).?;
    const queued = model.commitReservationToQueue(reservation.id).?;

    try std.testing.expectEqual(@as(JobId, 31), queued.job_id);
    try std.testing.expectEqual(@as(usize, 0), queued.queue_position);
    try std.testing.expectEqual(@as(usize, 2), model.snapshot().queued_jobs);

    const started = model.startNextQueuedJob().?;
    try std.testing.expectEqual(@as(JobId, 31), started.job_id);
    try std.testing.expectEqual(@as(usize, 1), model.snapshot().assigned_workers);
}

test "canceling queued reserved job releases leased workers" {
    var model = GatewaySchedulingModel.init();
    try model.addWorker(makeWorker(1, .cuda, "sm_80"));

    const reservation = (try model.reserve(makeRequirement(40, 1, .cuda, "sm_80"))).?;
    _ = model.commitReservationToQueue(reservation.id).?;
    _ = model.cancelQueuedJob(40).?;

    const snapshot_value = model.snapshot();
    try std.testing.expectEqual(Decision.canceled, snapshot_value.decision);
    try std.testing.expectEqual(@as(usize, 1), snapshot_value.idle_workers);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.reserved_workers);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.queued_jobs);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.owner_leases[ownerIndex(.training)]);
    try std.testing.expect(!model.hasLeasedWorkersForJob(40));
}

test "canceling active queued job releases assigned workers" {
    var model = GatewaySchedulingModel.init();
    try model.addWorker(makeWorker(1, .cuda, "sm_80"));

    _ = try model.enqueueJob(makeRequirement(50, 1, .cuda, "sm_80"));
    _ = model.startNextQueuedJob().?;
    _ = model.cancelActiveJob(50).?;

    const snapshot_value = model.snapshot();
    try std.testing.expectEqual(Decision.canceled, snapshot_value.decision);
    try std.testing.expectEqual(@as(usize, 1), snapshot_value.idle_workers);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.assigned_workers);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.active_jobs);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.owner_leases[ownerIndex(.training)]);
}

test "model policy behavior matches gateway scheduling worker class helpers" {
    try std.testing.expectEqual(WorkerClass.cuda, WorkerClass.parse("cuda").?);
    try std.testing.expectEqual(WorkerClass.rocm, WorkerClass.parse("hip").?);
    try std.testing.expect(WorkerClass.any.matchesBackendName("cuda"));
    try std.testing.expect(!WorkerClass.cuda.matchesBackendName("cpu"));

    const policy = SchedulingPolicy{ .worker_class = .cuda, .target_arch = "sm_80" };
    try std.testing.expect(policy.allowsWorker(.cuda, "sm_80"));
    try std.testing.expect(!policy.allowsWorker(.cuda, "sm_90"));
    try std.testing.expect(!policy.allowsWorker(.cpu, null));
}
