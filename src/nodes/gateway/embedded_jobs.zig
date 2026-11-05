const std = @import("std");

const Allocator = std.mem.Allocator;
const identity = @import("identity.zig");
const message = @import("../../network/message.zig");

const NodeId = message.NodeId;

pub const EmbeddedQueuedJob = struct {
    job_id: []u8,
    run_id: []u8,
    submitted_at: i64,
    required_workers: usize,
    resume_requested: bool,
    reserved_worker_ids: ?[]NodeId = null,

    pub fn clone(self: @This(), allocator: Allocator) !@This() {
        return .{
            .job_id = try allocator.dupe(u8, self.job_id),
            .run_id = try allocator.dupe(u8, self.run_id),
            .submitted_at = self.submitted_at,
            .required_workers = self.required_workers,
            .resume_requested = self.resume_requested,
            .reserved_worker_ids = if (self.reserved_worker_ids) |worker_ids| try allocator.dupe(NodeId, worker_ids) else null,
        };
    }

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        allocator.free(self.job_id);
        allocator.free(self.run_id);
        if (self.reserved_worker_ids) |worker_ids| allocator.free(worker_ids);
    }
};

pub const EmbeddedJobHistoryEntry = struct {
    job_id: []u8,
    run_id: []u8,
    submitted_at: i64,
    status: []u8,
    job_json: []u8,

    pub fn clone(self: @This(), allocator: Allocator) !@This() {
        return .{
            .job_id = try allocator.dupe(u8, self.job_id),
            .run_id = try allocator.dupe(u8, self.run_id),
            .submitted_at = self.submitted_at,
            .status = try allocator.dupe(u8, self.status),
            .job_json = try allocator.dupe(u8, self.job_json),
        };
    }

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        allocator.free(self.job_id);
        allocator.free(self.run_id);
        allocator.free(self.status);
        allocator.free(self.job_json);
    }
};

pub const EmbeddedJobQueueSnapshot = struct {
    allocator: Allocator,
    current_job: ?EmbeddedQueuedJob,
    queued_jobs: []EmbeddedQueuedJob,
    history: []EmbeddedJobHistoryEntry,

    pub fn deinit(self: *@This()) void {
        if (self.current_job) |*job| job.deinit(self.allocator);
        for (self.queued_jobs) |*job| {
            job.deinit(self.allocator);
        }
        self.allocator.free(self.queued_jobs);
        for (self.history) |*entry| {
            entry.deinit(self.allocator);
        }
        self.allocator.free(self.history);
    }
};

pub const EmbeddedJobQueue = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,
    stop_requested: bool,
    next_job_seq: usize,
    current_job: ?EmbeddedQueuedJob,
    queued_jobs: std.ArrayList(EmbeddedQueuedJob),
    history: std.ArrayList(EmbeddedJobHistoryEntry),

    pub fn init(allocator: Allocator) @This() {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .cond = .{},
            .stop_requested = false,
            .next_job_seq = 1,
            .current_job = null,
            .queued_jobs = std.ArrayList(EmbeddedQueuedJob).init(allocator),
            .history = std.ArrayList(EmbeddedJobHistoryEntry).init(allocator),
        };
    }

    pub fn deinit(self: *@This()) void {
        if (self.current_job) |*job| job.deinit(self.allocator);
        for (self.queued_jobs.items) |*job| {
            job.deinit(self.allocator);
        }
        self.queued_jobs.deinit();
        for (self.history.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.history.deinit();
    }

    pub fn requestStop(self: *@This()) void {
        self.mutex.lock();
        self.stop_requested = true;
        self.cond.broadcast();
        self.mutex.unlock();
    }

    pub fn enqueue(
        self: *@This(),
        executor_id: []const u8,
        required_workers: usize,
        resume_requested: bool,
    ) !struct {
        job: EmbeddedQueuedJob,
        queue_position: usize,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.stop_requested) return error.ServiceUnavailable;

        const sequence = self.next_job_seq;
        self.next_job_seq += 1;

        const job_id = try identity.queuedJobId(self.allocator, executor_id, sequence);
        errdefer self.allocator.free(job_id);

        const run_id = try self.allocator.dupe(u8, job_id);
        errdefer self.allocator.free(run_id);

        const record = EmbeddedQueuedJob{
            .job_id = job_id,
            .run_id = run_id,
            .submitted_at = std.time.timestamp(),
            .required_workers = required_workers,
            .resume_requested = resume_requested,
            .reserved_worker_ids = null,
        };
        const queue_position = self.queued_jobs.items.len + @as(usize, if (self.current_job != null) 1 else 0);
        try self.queued_jobs.append(record);
        self.cond.signal();

        return .{
            .job = try record.clone(self.allocator),
            .queue_position = queue_position,
        };
    }

    pub fn enqueueReserved(
        self: *@This(),
        executor_id: []const u8,
        required_workers: usize,
        resume_requested: bool,
        reserved_worker_ids: []const NodeId,
    ) !struct {
        job: EmbeddedQueuedJob,
        queue_position: usize,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.stop_requested) return error.ServiceUnavailable;

        const sequence = self.next_job_seq;
        self.next_job_seq += 1;

        const job_id = try identity.queuedJobId(self.allocator, executor_id, sequence);
        errdefer self.allocator.free(job_id);

        const run_id = try self.allocator.dupe(u8, job_id);
        errdefer self.allocator.free(run_id);

        const owned_reserved_ids = try self.allocator.dupe(NodeId, reserved_worker_ids);
        errdefer self.allocator.free(owned_reserved_ids);

        const record = EmbeddedQueuedJob{
            .job_id = job_id,
            .run_id = run_id,
            .submitted_at = std.time.timestamp(),
            .required_workers = required_workers,
            .resume_requested = resume_requested,
            .reserved_worker_ids = owned_reserved_ids,
        };
        const queue_position = @as(usize, if (self.current_job != null) 1 else 0);
        try self.queued_jobs.insert(0, record);
        self.cond.signal();

        return .{
            .job = try record.clone(self.allocator),
            .queue_position = queue_position,
        };
    }

    pub fn waitForNextJob(self: *@This()) !?EmbeddedQueuedJob {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (!self.stop_requested and self.queued_jobs.items.len == 0) {
            self.cond.wait(&self.mutex);
        }

        if (self.stop_requested and self.queued_jobs.items.len == 0) return null;

        std.debug.assert(self.current_job == null);
        self.current_job = self.queued_jobs.orderedRemove(0);
        return try self.current_job.?.clone(self.allocator);
    }

    pub fn completeCurrentJob(self: *@This(), status: []const u8, job_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const current = self.current_job orelse return;
        self.current_job = null;
        defer {
            var owned = current;
            owned.deinit(self.allocator);
        }

        try self.history.append(.{
            .job_id = try self.allocator.dupe(u8, current.job_id),
            .run_id = try self.allocator.dupe(u8, current.run_id),
            .submitted_at = current.submitted_at,
            .status = try self.allocator.dupe(u8, status),
            .job_json = try self.allocator.dupe(u8, job_json),
        });
    }

    pub fn archiveQueuedJob(self: *@This(), job: EmbeddedQueuedJob, status: []const u8, job_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        defer {
            var owned = job;
            owned.deinit(self.allocator);
        }

        try self.history.append(.{
            .job_id = try self.allocator.dupe(u8, job.job_id),
            .run_id = try self.allocator.dupe(u8, job.run_id),
            .submitted_at = job.submitted_at,
            .status = try self.allocator.dupe(u8, status),
            .job_json = try self.allocator.dupe(u8, job_json),
        });
    }

    pub fn removeQueuedJob(self: *@This(), job_id: []const u8) ?EmbeddedQueuedJob {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.queued_jobs.items, 0..) |job, idx| {
            if (std.mem.eql(u8, job.job_id, job_id)) {
                return self.queued_jobs.orderedRemove(idx);
            }
        }

        return null;
    }

    pub fn hasCurrentJob(self: *@This(), job_id: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_job) |job| {
            return std.mem.eql(u8, job.job_id, job_id);
        }
        return false;
    }

    pub fn hasPendingWork(self: *@This()) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.current_job != null or self.queued_jobs.items.len > 0;
    }

    pub fn snapshot(self: *@This(), allocator: Allocator) !EmbeddedJobQueueSnapshot {
        self.mutex.lock();
        defer self.mutex.unlock();

        var current: ?EmbeddedQueuedJob = null;
        if (self.current_job) |job| {
            current = try job.clone(allocator);
        }

        const queued_jobs = try allocator.alloc(EmbeddedQueuedJob, self.queued_jobs.items.len);
        var queued_count: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < queued_count) : (idx += 1) {
                queued_jobs[idx].deinit(allocator);
            }
            allocator.free(queued_jobs);
            if (current) |*job| job.deinit(allocator);
        }
        for (self.queued_jobs.items, 0..) |job, idx| {
            queued_jobs[idx] = try job.clone(allocator);
            queued_count += 1;
        }

        const history = try allocator.alloc(EmbeddedJobHistoryEntry, self.history.items.len);
        var history_count: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < history_count) : (idx += 1) {
                history[idx].deinit(allocator);
            }
            allocator.free(history);
            idx = 0;
            while (idx < queued_count) : (idx += 1) {
                queued_jobs[idx].deinit(allocator);
            }
            allocator.free(queued_jobs);
            if (current) |*job| job.deinit(allocator);
        }
        for (self.history.items, 0..) |entry, idx| {
            history[idx] = try entry.clone(allocator);
            history_count += 1;
        }

        return .{
            .allocator = allocator,
            .current_job = current,
            .queued_jobs = queued_jobs,
            .history = history,
        };
    }
};

pub const EmbeddedReservation = struct {
    reservation_id: []u8,
    reserved_at: i64,
    workers_required: usize,
    worker_ids: []NodeId,

    pub fn clone(self: @This(), allocator: Allocator) !@This() {
        return .{
            .reservation_id = try allocator.dupe(u8, self.reservation_id),
            .reserved_at = self.reserved_at,
            .workers_required = self.workers_required,
            .worker_ids = try allocator.dupe(NodeId, self.worker_ids),
        };
    }

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        allocator.free(self.reservation_id);
        allocator.free(self.worker_ids);
    }
};

pub const EmbeddedReservationStore = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    next_reservation_seq: usize,
    active: std.ArrayList(EmbeddedReservation),

    pub fn init(allocator: Allocator) @This() {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .next_reservation_seq = 1,
            .active = std.ArrayList(EmbeddedReservation).init(allocator),
        };
    }

    pub fn deinit(self: *@This()) void {
        for (self.active.items) |*reservation| {
            reservation.deinit(self.allocator);
        }
        self.active.deinit();
    }

    pub fn create(
        self: *@This(),
        executor_id: []const u8,
        workers_required: usize,
        worker_ids: []const NodeId,
    ) !EmbeddedReservation {
        self.mutex.lock();
        defer self.mutex.unlock();

        const reservation = EmbeddedReservation{
            .reservation_id = try identity.reservationId(self.allocator, executor_id, self.next_reservation_seq),
            .reserved_at = std.time.timestamp(),
            .workers_required = workers_required,
            .worker_ids = try self.allocator.dupe(NodeId, worker_ids),
        };
        self.next_reservation_seq += 1;
        try self.active.append(reservation);
        return try reservation.clone(self.allocator);
    }

    pub fn take(self: *@This(), reservation_id: []const u8) ?EmbeddedReservation {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.active.items, 0..) |reservation, idx| {
            if (std.mem.eql(u8, reservation.reservation_id, reservation_id)) {
                return self.active.orderedRemove(idx);
            }
        }
        return null;
    }

    pub fn hasActiveReservations(self: *@This()) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.active.items.len > 0;
    }

    pub fn releaseAll(self: *@This(), worker_fabric: anytype) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.active.items) |reservation| {
            worker_fabric.releaseWorkers(reservation.worker_ids);
        }
        for (self.active.items) |*reservation| {
            reservation.deinit(self.allocator);
        }
        self.active.clearRetainingCapacity();
    }
};

test "embedded job queue moves jobs through current and history" {
    const allocator = std.testing.allocator;
    var queue = EmbeddedJobQueue.init(allocator);
    defer queue.deinit();

    const submission = try queue.enqueue("executor", 2, false);
    defer {
        var owned = submission.job;
        owned.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 0), submission.queue_position);
    try std.testing.expect(queue.hasPendingWork());

    var current = (try queue.waitForNextJob()).?;
    defer current.deinit(allocator);

    try std.testing.expect(queue.hasCurrentJob(current.job_id));
    try queue.completeCurrentJob("completed", "{\"status\":\"completed\"}");
    try std.testing.expect(!queue.hasPendingWork());

    var snapshot = try queue.snapshot(allocator);
    defer snapshot.deinit();

    try std.testing.expect(snapshot.current_job == null);
    try std.testing.expectEqual(@as(usize, 0), snapshot.queued_jobs.len);
    try std.testing.expectEqual(@as(usize, 1), snapshot.history.len);
    try std.testing.expectEqualStrings("completed", snapshot.history[0].status);
}

test "reserved jobs are inserted ahead of normal queued jobs" {
    const allocator = std.testing.allocator;
    var queue = EmbeddedJobQueue.init(allocator);
    defer queue.deinit();

    const normal = try queue.enqueue("executor", 2, false);
    defer {
        var owned = normal.job;
        owned.deinit(allocator);
    }

    const reserved_ids = [_]NodeId{ 7, 8 };
    const reserved = try queue.enqueueReserved("executor", 2, true, &reserved_ids);
    defer {
        var owned = reserved.job;
        owned.deinit(allocator);
    }

    var current = (try queue.waitForNextJob()).?;
    defer current.deinit(allocator);

    try std.testing.expectEqualStrings(reserved.job.job_id, current.job_id);
    try std.testing.expect(current.resume_requested);
    try std.testing.expectEqual(@as(NodeId, 7), current.reserved_worker_ids.?[0]);
}

test "reservation store returns active reservations exactly once" {
    const allocator = std.testing.allocator;
    var store = EmbeddedReservationStore.init(allocator);
    defer store.deinit();

    const worker_ids = [_]NodeId{ 3, 4 };
    var reservation = try store.create("executor", 2, &worker_ids);
    defer reservation.deinit(allocator);

    try std.testing.expect(store.hasActiveReservations());

    var taken = store.take(reservation.reservation_id).?;
    defer taken.deinit(allocator);

    try std.testing.expect(!store.hasActiveReservations());
    try std.testing.expectEqualStrings(reservation.reservation_id, taken.reservation_id);
    try std.testing.expect(store.take(reservation.reservation_id) == null);
}
