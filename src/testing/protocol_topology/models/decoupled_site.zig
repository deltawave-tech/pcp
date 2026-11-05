const std = @import("std");

const event_tape = @import("../../../algorithms/event_tape.zig");
const message = @import("../../../network/message.zig");
const vector_clock = @import("../../../algorithms/vector_clock.zig");

pub const max_workers = 4;
pub const max_fragments = 4;
pub const tokens_per_local_step = 8;

pub const WorkerId = vector_clock.NodeId;
pub const SessionId = u32;
pub const syncer_node_id: WorkerId = std.math.maxInt(WorkerId);

pub const WorkerStatus = enum {
    offline,
    online,
    crashed,
};

pub const Decision = enum {
    pending,
    assigned,
    update_accepted,
    update_rejected,
    merge_committed,
    merge_deferred,
    canceled,
    failed,
    resumable,
};

pub const Violation = enum {
    none,
    invalid_worker,
    invalid_session,
    invalid_learner,
    invalid_fragment,
    unassigned_worker,
    unavailable_worker,
    duplicate_update,
    context_payload_mismatch,
    stale_update,
    stale_vector_clock,
    below_quorum_commit,
    post_cancel_update,
    incompatible_resume,
};

pub const InitOptions = struct {
    session_id: SessionId = 1,
    worker_count: usize,
    min_quorum: usize,
    num_fragments: usize,
};

pub const WorkerState = struct {
    id: WorkerId,
    status: WorkerStatus = .online,
    assigned: bool = false,
    local_step: usize = 0,
    tokens_since_update: usize = 0,
    last_submitted_syncer_step: [max_fragments]usize = [_]usize{0} ** max_fragments,
};

pub const FragmentUpdate = struct {
    worker_id: WorkerId,
    session_id: SessionId,
    learner_id: WorkerId,
    fragment_id: usize,
    syncer_step: usize,
    learner_step: usize,
    clock_counter: vector_clock.Counter,
    tokens_since_fragment_update: usize,
};

pub const AcceptedUpdate = struct {
    worker_id: WorkerId,
    fragment_id: usize,
    syncer_step: usize,
    learner_step: usize,
    clock_counter: vector_clock.Counter,
    tokens_since_fragment_update: usize,
};

pub const Snapshot = struct {
    session_id: SessionId,
    current_syncer_step: usize,
    current_fragment: usize,
    committed_windows: usize,
    accepted_updates: usize,
    worker_count: usize,
    min_quorum: usize,
    canceled: bool,
    decision: Decision,
    violation: Violation,
    tape_entries: usize,

    pub fn eql(a: Snapshot, b: Snapshot) bool {
        return a.session_id == b.session_id and
            a.current_syncer_step == b.current_syncer_step and
            a.current_fragment == b.current_fragment and
            a.committed_windows == b.committed_windows and
            a.accepted_updates == b.accepted_updates and
            a.worker_count == b.worker_count and
            a.min_quorum == b.min_quorum and
            a.canceled == b.canceled and
            a.decision == b.decision and
            a.violation == b.violation and
            a.tape_entries == b.tape_entries;
    }
};

pub const ActionTag = enum {
    assign,
    local_step_complete,
    submit_fragment,
    timeout_window,
    cancel_session,
    crash,
    restart,
    leave,
    join,
    resume_worker,
    force_commit,
};

pub const Action = struct {
    tag: ActionTag,
    worker_id: WorkerId = 0,
    session_id: SessionId = 1,
};

pub const QueuedLearnerMetadata = struct {
    sender_id: WorkerId,
    context: message.MessageContext,
    learner_step: usize,
    clock_counter: vector_clock.Counter,
};

pub const QueuedFragmentResponse = struct {
    sender_id: WorkerId,
    context: message.MessageContext,
    update: FragmentUpdate,
};

pub const CollectionSummary = struct {
    matched_envelopes: usize = 0,
    accepted_count: usize = 0,
    unique_senders: usize = 0,
    duplicate_senders: usize = 0,
    ignored_context: usize = 0,
    violation: Violation = .none,
};

pub const ExplorationSummary = struct {
    checked: usize = 0,
    first_violation: Violation = .none,
};

pub const DecoupledSiteModel = struct {
    allocator: std.mem.Allocator,
    session_id: SessionId,
    worker_count: usize,
    min_quorum: usize,
    num_fragments: usize,
    current_syncer_step: usize = 1,
    current_fragment: usize = 0,
    committed_windows: usize = 0,
    canceled: bool = false,
    workers: [max_workers]WorkerState = undefined,
    accepted: [max_workers]AcceptedUpdate = undefined,
    accepted_count: usize = 0,
    decision: Decision = .pending,
    violation: Violation = .none,
    clock: vector_clock.VectorClock,
    tape: event_tape.EventTape,

    pub fn init(allocator: std.mem.Allocator, options: InitOptions) !DecoupledSiteModel {
        if (options.worker_count == 0 or options.worker_count > max_workers) return error.InvalidWorkerCount;
        if (options.num_fragments == 0 or options.num_fragments > max_fragments) return error.InvalidFragmentCount;
        if (options.min_quorum == 0 or options.min_quorum > options.worker_count) return error.InvalidQuorum;

        var model = DecoupledSiteModel{
            .allocator = allocator,
            .session_id = options.session_id,
            .worker_count = options.worker_count,
            .min_quorum = options.min_quorum,
            .num_fragments = options.num_fragments,
            .clock = vector_clock.VectorClock.init(allocator),
            .tape = event_tape.EventTape.init(allocator),
        };
        errdefer model.deinit();

        for (0..options.worker_count) |index| {
            const worker_id: WorkerId = @intCast(index);
            model.workers[index] = .{ .id = worker_id };
            try model.clock.observe(worker_id, 0);
        }
        try model.clock.observe(syncer_node_id, 0);

        return model;
    }

    pub fn deinit(self: *DecoupledSiteModel) void {
        self.clock.deinit();
        self.tape.deinit();
        self.* = undefined;
    }

    pub fn apply(self: *DecoupledSiteModel, action: Action) !void {
        switch (action.tag) {
            .assign => try self.assignWorker(action.worker_id),
            .local_step_complete => try self.localStepComplete(action.worker_id),
            .submit_fragment => try self.submitCurrent(action.worker_id),
            .timeout_window => try self.timeoutWindow(),
            .cancel_session => self.cancelSession(),
            .crash => try self.crashWorker(action.worker_id),
            .restart => try self.restartWorker(action.worker_id),
            .leave => try self.leaveWorker(action.worker_id),
            .join => try self.joinWorker(action.worker_id),
            .resume_worker => try self.resumeWorker(action.worker_id, action.session_id),
            .force_commit => try self.forceCommitWindow(),
        }
    }

    pub fn assignWorker(self: *DecoupledSiteModel, worker_id: WorkerId) !void {
        if (self.canceled) return self.reject(.post_cancel_update);
        const worker = self.getWorker(worker_id) orelse return self.reject(.invalid_worker);
        if (worker.status != .online) return self.reject(.unavailable_worker);
        worker.assigned = true;
        self.decision = .assigned;
    }

    pub fn localStepComplete(self: *DecoupledSiteModel, worker_id: WorkerId) !void {
        if (self.canceled) return self.reject(.post_cancel_update);
        const worker = self.getWorker(worker_id) orelse return self.reject(.invalid_worker);
        if (worker.status != .online) return self.reject(.unavailable_worker);
        if (!worker.assigned) return self.reject(.unassigned_worker);

        worker.local_step += 1;
        worker.tokens_since_update += tokens_per_local_step;
        _ = try self.clock.tick(worker_id);
        _ = try self.tape.appendLearnerApply(
            self.current_syncer_step,
            self.current_fragment,
            worker_id,
            self.clock,
        );
    }

    pub fn submitCurrent(self: *DecoupledSiteModel, worker_id: WorkerId) !void {
        const worker = self.getWorker(worker_id) orelse return self.reject(.invalid_worker);
        return self.submitFragment(.{
            .worker_id = worker_id,
            .session_id = self.session_id,
            .learner_id = worker_id,
            .fragment_id = self.current_fragment,
            .syncer_step = self.current_syncer_step,
            .learner_step = worker.local_step,
            .clock_counter = self.clock.counter(worker_id),
            .tokens_since_fragment_update = worker.tokens_since_update,
        });
    }

    pub fn submitFragment(self: *DecoupledSiteModel, update: FragmentUpdate) !void {
        if (self.canceled) return self.reject(.post_cancel_update);
        const worker = self.getWorker(update.worker_id) orelse return self.reject(.invalid_worker);
        if (worker.status != .online) return self.reject(.unavailable_worker);
        if (!worker.assigned) return self.reject(.unassigned_worker);
        if (update.session_id != self.session_id) return self.reject(.invalid_session);
        if (update.learner_id != update.worker_id) return self.reject(.invalid_learner);
        if (update.fragment_id >= self.num_fragments) return self.reject(.invalid_fragment);
        if (update.fragment_id != self.current_fragment) return self.reject(.stale_update);
        if (update.syncer_step != self.current_syncer_step) return self.reject(.stale_update);
        if (update.clock_counter < self.clock.counter(update.worker_id)) return self.reject(.stale_vector_clock);
        if (worker.last_submitted_syncer_step[update.fragment_id] >= update.syncer_step) {
            return self.reject(.duplicate_update);
        }
        if (self.hasAccepted(update.worker_id)) return self.reject(.duplicate_update);

        try self.clock.observe(update.worker_id, update.clock_counter);
        worker.last_submitted_syncer_step[update.fragment_id] = update.syncer_step;
        worker.tokens_since_update = 0;
        self.accepted[self.accepted_count] = .{
            .worker_id = update.worker_id,
            .fragment_id = update.fragment_id,
            .syncer_step = update.syncer_step,
            .learner_step = update.learner_step,
            .clock_counter = update.clock_counter,
            .tokens_since_fragment_update = update.tokens_since_fragment_update,
        };
        self.accepted_count += 1;

        _ = try self.tape.appendInFlightMessage(
            self.current_syncer_step,
            @intCast(self.current_syncer_step),
            update.worker_id,
            self.clock,
        );
        self.decision = .update_accepted;
    }

    pub fn collectUniqueMetadata(
        self: *DecoupledSiteModel,
        queue: []const QueuedLearnerMetadata,
        expected_count: usize,
    ) !CollectionSummary {
        var summary = CollectionSummary{};
        var seen_mask: u64 = 0;
        const bounded_expected = @min(expected_count, self.worker_count);

        for (queue) |metadata| {
            if (summary.accepted_count >= bounded_expected) break;
            if (!self.metadataContextMatches(metadata.context)) {
                summary.ignored_context += 1;
                continue;
            }
            summary.matched_envelopes += 1;
            if (metadata.sender_id >= self.worker_count) return self.finishCollectionReject(&summary, .invalid_worker);

            const bit = workerMask(metadata.sender_id);
            if ((seen_mask & bit) != 0) {
                summary.duplicate_senders += 1;
                continue;
            }
            if (metadata.clock_counter < self.clock.counter(metadata.sender_id)) {
                return self.finishCollectionReject(&summary, .stale_vector_clock);
            }

            seen_mask |= bit;
            summary.accepted_count += 1;
            summary.unique_senders += 1;
            try self.clock.observe(metadata.sender_id, metadata.clock_counter);
        }

        return summary;
    }

    pub fn collectQueuedFragmentResponses(
        self: *DecoupledSiteModel,
        queue: []const QueuedFragmentResponse,
        expected_count: usize,
    ) !CollectionSummary {
        var summary = CollectionSummary{};
        var seen_mask: u64 = 0;
        const bounded_expected = @min(expected_count, self.worker_count);

        for (queue) |response| {
            if (summary.accepted_count >= bounded_expected) break;
            if (!self.fragmentEnvelopeMatches(response.context)) {
                summary.ignored_context += 1;
                continue;
            }
            summary.matched_envelopes += 1;
            if (response.sender_id >= self.worker_count) return self.finishCollectionReject(&summary, .invalid_worker);

            const bit = workerMask(response.sender_id);
            if ((seen_mask & bit) != 0) {
                summary.duplicate_senders += 1;
                continue;
            }
            if (!self.fragmentPayloadMatchesEnvelope(response)) {
                return self.finishCollectionReject(&summary, .context_payload_mismatch);
            }
            if (response.update.clock_counter < self.clock.counter(response.sender_id)) {
                return self.finishCollectionReject(&summary, .stale_vector_clock);
            }

            try self.submitFragment(response.update);
            if (self.violation != .none) {
                summary.violation = self.violation;
                return summary;
            }

            seen_mask |= bit;
            summary.accepted_count += 1;
            summary.unique_senders += 1;
        }

        return summary;
    }

    pub fn timeoutWindow(self: *DecoupledSiteModel) !void {
        if (self.canceled) {
            self.decision = .canceled;
            return;
        }
        if (self.accepted_count < self.min_quorum) {
            self.decision = .merge_deferred;
            return;
        }
        try self.commitWindow();
    }

    pub fn forceCommitWindow(self: *DecoupledSiteModel) !void {
        try self.commitWindow();
    }

    pub fn cancelSession(self: *DecoupledSiteModel) void {
        self.canceled = true;
        self.accepted_count = 0;
        self.decision = .canceled;
    }

    pub fn crashWorker(self: *DecoupledSiteModel, worker_id: WorkerId) !void {
        const worker = self.getWorker(worker_id) orelse return self.reject(.invalid_worker);
        worker.status = .crashed;
        worker.assigned = false;
        _ = try self.tape.appendLearnerFailure(self.current_syncer_step, worker_id, self.clock);
        self.decision = .resumable;
    }

    pub fn restartWorker(self: *DecoupledSiteModel, worker_id: WorkerId) !void {
        const worker = self.getWorker(worker_id) orelse return self.reject(.invalid_worker);
        worker.status = .online;
        _ = try self.tape.appendLearnerRecovery(self.current_syncer_step, worker_id, self.clock);
        self.decision = .resumable;
    }

    pub fn resumeWorker(self: *DecoupledSiteModel, worker_id: WorkerId, session_id: SessionId) !void {
        if (session_id != self.session_id) return self.reject(.incompatible_resume);
        const worker = self.getWorker(worker_id) orelse return self.reject(.invalid_worker);
        if (worker.status != .online) return self.reject(.unavailable_worker);
        worker.assigned = true;
        _ = try self.tape.appendLearnerRecovery(self.current_syncer_step, worker_id, self.clock);
        self.decision = .resumable;
    }

    pub fn leaveWorker(self: *DecoupledSiteModel, worker_id: WorkerId) !void {
        const worker = self.getWorker(worker_id) orelse return self.reject(.invalid_worker);
        worker.status = .offline;
        worker.assigned = false;
        self.decision = .resumable;
    }

    pub fn joinWorker(self: *DecoupledSiteModel, worker_id: WorkerId) !void {
        const worker = self.getWorker(worker_id) orelse return self.reject(.invalid_worker);
        worker.status = .online;
        self.decision = .resumable;
    }

    pub fn snapshot(self: DecoupledSiteModel) Snapshot {
        return .{
            .session_id = self.session_id,
            .current_syncer_step = self.current_syncer_step,
            .current_fragment = self.current_fragment,
            .committed_windows = self.committed_windows,
            .accepted_updates = self.accepted_count,
            .worker_count = self.worker_count,
            .min_quorum = self.min_quorum,
            .canceled = self.canceled,
            .decision = self.decision,
            .violation = self.violation,
            .tape_entries = self.tape.entries.items.len,
        };
    }

    pub fn replayCommittedWindow(self: *DecoupledSiteModel, syncer_step: usize, fragment_id: usize) ?*const event_tape.EventEntry {
        return self.tape.replayFragmentSync(syncer_step, fragment_id);
    }

    fn commitWindow(self: *DecoupledSiteModel) !void {
        if (self.accepted_count < self.min_quorum) return self.reject(.below_quorum_commit);
        _ = try self.clock.tick(syncer_node_id);

        var participants: [max_workers]event_tape.ParticipantRecord = undefined;
        for (self.accepted[0..self.accepted_count], 0..) |accepted, index| {
            participants[index] = .{
                .learner_id = accepted.worker_id,
                .learner_step = accepted.learner_step,
                .steps_since_fragment_update = accepted.learner_step,
                .tokens_since_fragment_update = accepted.tokens_since_fragment_update,
                .merge_weight = @floatFromInt(@max(accepted.tokens_since_fragment_update, 1)),
            };
        }

        _ = try self.tape.appendFragmentSync(
            self.current_syncer_step,
            self.current_fragment,
            participants[0..self.accepted_count],
            self.clock,
        );

        self.committed_windows += 1;
        self.accepted_count = 0;
        self.current_syncer_step += 1;
        self.current_fragment = (self.current_fragment + 1) % self.num_fragments;
        self.decision = .merge_committed;
    }

    fn metadataContextMatches(self: DecoupledSiteModel, context: message.MessageContext) bool {
        return context.request_id == self.session_id;
    }

    fn fragmentEnvelopeMatches(self: DecoupledSiteModel, context: message.MessageContext) bool {
        return context.request_id == self.session_id and
            context.round_id == self.current_syncer_step and
            context.task_id == self.current_fragment + 1;
    }

    fn fragmentPayloadMatchesEnvelope(self: DecoupledSiteModel, response: QueuedFragmentResponse) bool {
        return response.update.session_id == self.session_id and
            response.update.worker_id == response.sender_id and
            response.update.learner_id == response.sender_id and
            response.update.syncer_step == response.context.round_id and
            response.update.fragment_id + 1 == response.context.task_id and
            response.update.syncer_step == self.current_syncer_step and
            response.update.fragment_id == self.current_fragment;
    }

    fn getWorker(self: *DecoupledSiteModel, worker_id: WorkerId) ?*WorkerState {
        if (worker_id >= self.worker_count) return null;
        return &self.workers[worker_id];
    }

    fn hasAccepted(self: DecoupledSiteModel, worker_id: WorkerId) bool {
        for (self.accepted[0..self.accepted_count]) |accepted| {
            if (accepted.worker_id == worker_id) return true;
        }
        return false;
    }

    fn reject(self: *DecoupledSiteModel, violation: Violation) void {
        self.violation = violation;
        self.decision = .update_rejected;
    }

    fn finishCollectionReject(self: *DecoupledSiteModel, partial: *const CollectionSummary, violation: Violation) CollectionSummary {
        self.reject(violation);
        var summary = partial.*;
        summary.violation = violation;
        return summary;
    }
};

fn workerMask(worker_id: WorkerId) u64 {
    return @as(u64, 1) << @intCast(worker_id);
}

pub fn runActions(
    allocator: std.mem.Allocator,
    options: InitOptions,
    actions: []const Action,
) !Snapshot {
    var model = try DecoupledSiteModel.init(allocator, options);
    defer model.deinit();

    for (actions) |action| {
        try model.apply(action);
        if (model.violation != .none) break;
    }
    return model.snapshot();
}

pub fn exploreTwoWorkerOneWindow(
    allocator: std.mem.Allocator,
    depth: usize,
) !ExplorationSummary {
    if (depth > 6) return error.DepthTooLarge;

    const alphabet = [_]Action{
        .{ .tag = .assign, .worker_id = 0 },
        .{ .tag = .assign, .worker_id = 1 },
        .{ .tag = .local_step_complete, .worker_id = 0 },
        .{ .tag = .local_step_complete, .worker_id = 1 },
        .{ .tag = .submit_fragment, .worker_id = 0 },
        .{ .tag = .submit_fragment, .worker_id = 1 },
        .{ .tag = .timeout_window },
    };

    var buffer: [6]Action = undefined;
    var summary = ExplorationSummary{};
    try exploreRecursive(allocator, &alphabet, &buffer, 0, depth, &summary);
    return summary;
}

fn exploreRecursive(
    allocator: std.mem.Allocator,
    alphabet: []const Action,
    buffer: *[6]Action,
    cursor: usize,
    depth: usize,
    summary: *ExplorationSummary,
) !void {
    if (cursor == depth) {
        summary.checked += 1;
        const snapshot_value = try runActions(
            allocator,
            .{ .session_id = 7, .worker_count = 2, .min_quorum = 2, .num_fragments = 1 },
            buffer[0..depth],
        );
        if (snapshot_value.violation != .none and summary.first_violation == .none) {
            summary.first_violation = snapshot_value.violation;
        }
        return;
    }

    for (alphabet) |action| {
        buffer[cursor] = action;
        try exploreRecursive(allocator, alphabet, buffer, cursor + 1, depth, summary);
    }
}

pub fn randomScheduleSnapshot(
    allocator: std.mem.Allocator,
    seed: u64,
    action_count: usize,
) !Snapshot {
    const alphabet = [_]Action{
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
        .{ .tag = .crash, .worker_id = 3 },
        .{ .tag = .restart, .worker_id = 3 },
        .{ .tag = .resume_worker, .worker_id = 3, .session_id = 11 },
    };

    var prng = std.rand.DefaultPrng.init(seed);
    var model = try DecoupledSiteModel.init(allocator, .{
        .session_id = 11,
        .worker_count = 4,
        .min_quorum = 2,
        .num_fragments = 3,
    });
    defer model.deinit();

    for (0..action_count) |_| {
        const index = prng.random().uintLessThan(usize, alphabet.len);
        try model.apply(alphabet[index]);
        if (model.violation != .none) break;
    }
    return model.snapshot();
}

fn fragmentContext(session_id: SessionId, syncer_step: usize, fragment_id: usize) message.MessageContext {
    return .{
        .request_id = session_id,
        .round_id = @intCast(syncer_step),
        .task_id = @intCast(fragment_id + 1),
    };
}

fn currentQueuedResponse(model: *const DecoupledSiteModel, sender_id: WorkerId) QueuedFragmentResponse {
    const worker = model.workers[sender_id];
    return .{
        .sender_id = sender_id,
        .context = fragmentContext(model.session_id, model.current_syncer_step, model.current_fragment),
        .update = .{
            .worker_id = sender_id,
            .session_id = model.session_id,
            .learner_id = sender_id,
            .fragment_id = model.current_fragment,
            .syncer_step = model.current_syncer_step,
            .learner_step = worker.local_step,
            .clock_counter = model.clock.counter(sender_id),
            .tokens_since_fragment_update = worker.tokens_since_update,
        },
    };
}

fn clockEntryCounter(entries: []const vector_clock.Entry, node_id: WorkerId) ?vector_clock.Counter {
    for (entries) |entry| {
        if (entry.node_id == node_id) return entry.counter;
    }
    return null;
}

test "gateway-local model commits quorum fragment and replays event tape" {
    var model = try DecoupledSiteModel.init(std.testing.allocator, .{
        .session_id = 10,
        .worker_count = 3,
        .min_quorum = 2,
        .num_fragments = 2,
    });
    defer model.deinit();

    try model.assignWorker(0);
    try model.assignWorker(1);
    try model.localStepComplete(0);
    try model.localStepComplete(1);
    try model.submitCurrent(0);
    try model.submitCurrent(1);
    try model.timeoutWindow();

    const snapshot_value = model.snapshot();
    try std.testing.expectEqual(Decision.merge_committed, snapshot_value.decision);
    try std.testing.expectEqual(Violation.none, snapshot_value.violation);
    try std.testing.expectEqual(@as(usize, 1), snapshot_value.committed_windows);
    try std.testing.expectEqual(@as(vector_clock.Counter, 1), model.clock.counter(0));
    try std.testing.expectEqual(@as(vector_clock.Counter, 1), model.clock.counter(1));
    try std.testing.expectEqual(@as(vector_clock.Counter, 1), model.clock.counter(syncer_node_id));

    model.tape.resetReplay();
    const replay = model.replayCommittedWindow(1, 0) orelse return error.MissingReplayEntry;
    try std.testing.expectEqual(event_tape.EventKind.fragment_sync, replay.kind);
    try std.testing.expectEqual(@as(usize, 2), replay.participants.len);
    try std.testing.expectEqual(@as(vector_clock.Counter, 1), clockEntryCounter(replay.clock_entries, syncer_node_id).?);
}

test "small exhaustive gateway-local schedules run bounded state space" {
    const summary = try exploreTwoWorkerOneWindow(std.testing.allocator, 3);

    try std.testing.expectEqual(@as(usize, 343), summary.checked);
    try std.testing.expect(summary.first_violation != .none);
}

test "random four-worker three-window schedules are deterministic by seed" {
    const first = try randomScheduleSnapshot(std.testing.allocator, 42, 18);
    const second = try randomScheduleSnapshot(std.testing.allocator, 42, 18);

    try std.testing.expect(Snapshot.eql(first, second));
}

test "model catches below-quorum merge commits" {
    const snapshot_value = try runActions(
        std.testing.allocator,
        .{ .session_id = 3, .worker_count = 2, .min_quorum = 2, .num_fragments = 1 },
        &[_]Action{
            .{ .tag = .assign, .worker_id = 0 },
            .{ .tag = .local_step_complete, .worker_id = 0 },
            .{ .tag = .submit_fragment, .worker_id = 0 },
            .{ .tag = .force_commit },
        },
    );

    try std.testing.expectEqual(Violation.below_quorum_commit, snapshot_value.violation);
}

test "model catches stale fragment acceptance" {
    var model = try DecoupledSiteModel.init(std.testing.allocator, .{
        .session_id = 3,
        .worker_count = 2,
        .min_quorum = 1,
        .num_fragments = 1,
    });
    defer model.deinit();

    try model.assignWorker(0);
    try model.localStepComplete(0);
    try model.submitCurrent(0);
    try model.timeoutWindow();

    try model.submitFragment(.{
        .worker_id = 0,
        .session_id = 3,
        .learner_id = 0,
        .fragment_id = 0,
        .syncer_step = 1,
        .learner_step = 1,
        .clock_counter = model.clock.counter(0),
        .tokens_since_fragment_update = tokens_per_local_step,
    });

    try std.testing.expectEqual(Violation.stale_update, model.snapshot().violation);
}

test "model catches duplicate fragment updates in one window" {
    const snapshot_value = try runActions(
        std.testing.allocator,
        .{ .session_id = 3, .worker_count = 2, .min_quorum = 1, .num_fragments = 1 },
        &[_]Action{
            .{ .tag = .assign, .worker_id = 0 },
            .{ .tag = .local_step_complete, .worker_id = 0 },
            .{ .tag = .submit_fragment, .worker_id = 0 },
            .{ .tag = .submit_fragment, .worker_id = 0 },
        },
    );

    try std.testing.expectEqual(Violation.duplicate_update, snapshot_value.violation);
}

test "queued learner metadata is collected by unique sender" {
    var model = try DecoupledSiteModel.init(std.testing.allocator, .{
        .session_id = 44,
        .worker_count = 3,
        .min_quorum = 2,
        .num_fragments = 1,
    });
    defer model.deinit();

    try model.assignWorker(0);
    try model.assignWorker(1);
    try model.localStepComplete(0);
    try model.localStepComplete(1);

    const queue = [_]QueuedLearnerMetadata{
        .{
            .sender_id = 0,
            .context = .{ .request_id = 44, .round_id = 0, .task_id = 0 },
            .learner_step = 1,
            .clock_counter = 1,
        },
        .{
            .sender_id = 0,
            .context = .{ .request_id = 44, .round_id = 0, .task_id = 0 },
            .learner_step = 1,
            .clock_counter = 1,
        },
        .{
            .sender_id = 2,
            .context = .{ .request_id = 99, .round_id = 0, .task_id = 0 },
            .learner_step = 1,
            .clock_counter = 0,
        },
        .{
            .sender_id = 1,
            .context = .{ .request_id = 44, .round_id = 0, .task_id = 0 },
            .learner_step = 1,
            .clock_counter = 1,
        },
    };

    const summary = try model.collectUniqueMetadata(&queue, 3);
    try std.testing.expectEqual(@as(usize, 3), summary.matched_envelopes);
    try std.testing.expectEqual(@as(usize, 2), summary.accepted_count);
    try std.testing.expectEqual(@as(usize, 2), summary.unique_senders);
    try std.testing.expectEqual(@as(usize, 1), summary.duplicate_senders);
    try std.testing.expectEqual(@as(usize, 1), summary.ignored_context);
    try std.testing.expectEqual(Violation.none, summary.violation);
}

test "duplicate queued fragment responses do not satisfy quorum" {
    var model = try DecoupledSiteModel.init(std.testing.allocator, .{
        .session_id = 45,
        .worker_count = 2,
        .min_quorum = 2,
        .num_fragments = 1,
    });
    defer model.deinit();

    try model.assignWorker(0);
    try model.assignWorker(1);
    try model.localStepComplete(0);
    try model.localStepComplete(1);

    const first = currentQueuedResponse(&model, 0);
    const queue = [_]QueuedFragmentResponse{ first, first };

    const summary = try model.collectQueuedFragmentResponses(&queue, 2);
    try std.testing.expectEqual(@as(usize, 2), summary.matched_envelopes);
    try std.testing.expectEqual(@as(usize, 1), summary.accepted_count);
    try std.testing.expectEqual(@as(usize, 1), summary.unique_senders);
    try std.testing.expectEqual(@as(usize, 1), summary.duplicate_senders);
    try std.testing.expectEqual(Violation.none, summary.violation);

    try model.timeoutWindow();
    const snapshot_value = model.snapshot();
    try std.testing.expectEqual(Decision.merge_deferred, snapshot_value.decision);
    try std.testing.expectEqual(@as(usize, 1), snapshot_value.accepted_updates);
    try std.testing.expectEqual(@as(usize, 0), snapshot_value.committed_windows);
}

test "queued fragment responses must match envelope and payload context" {
    var model = try DecoupledSiteModel.init(std.testing.allocator, .{
        .session_id = 46,
        .worker_count = 1,
        .min_quorum = 1,
        .num_fragments = 2,
    });
    defer model.deinit();

    try model.assignWorker(0);
    try model.localStepComplete(0);

    var wrong_envelope = currentQueuedResponse(&model, 0);
    wrong_envelope.context.round_id = 99;
    const ignored = try model.collectQueuedFragmentResponses(&.{wrong_envelope}, 1);
    try std.testing.expectEqual(@as(usize, 0), ignored.matched_envelopes);
    try std.testing.expectEqual(@as(usize, 1), ignored.ignored_context);
    try std.testing.expectEqual(Violation.none, ignored.violation);

    var payload_mismatch = currentQueuedResponse(&model, 0);
    payload_mismatch.update.fragment_id = 1;
    const rejected = try model.collectQueuedFragmentResponses(&.{payload_mismatch}, 1);
    try std.testing.expectEqual(Violation.context_payload_mismatch, rejected.violation);
    try std.testing.expectEqual(Violation.context_payload_mismatch, model.snapshot().violation);
}

test "queued fragment responses require fresh learner vector clocks" {
    var model = try DecoupledSiteModel.init(std.testing.allocator, .{
        .session_id = 47,
        .worker_count = 1,
        .min_quorum = 1,
        .num_fragments = 1,
    });
    defer model.deinit();

    try model.assignWorker(0);
    try model.localStepComplete(0);

    var stale = currentQueuedResponse(&model, 0);
    stale.update.clock_counter = 0;
    const summary = try model.collectQueuedFragmentResponses(&.{stale}, 1);

    try std.testing.expectEqual(Violation.stale_vector_clock, summary.violation);
    try std.testing.expectEqual(Violation.stale_vector_clock, model.snapshot().violation);
}

test "model catches post-cancel worker updates" {
    const snapshot_value = try runActions(
        std.testing.allocator,
        .{ .session_id = 3, .worker_count = 2, .min_quorum = 1, .num_fragments = 1 },
        &[_]Action{
            .{ .tag = .assign, .worker_id = 0 },
            .{ .tag = .cancel_session },
            .{ .tag = .submit_fragment, .worker_id = 0 },
        },
    );

    try std.testing.expectEqual(Violation.post_cancel_update, snapshot_value.violation);
}

test "model accepts only compatible worker resume sessions" {
    const bad_snapshot = try runActions(
        std.testing.allocator,
        .{ .session_id = 12, .worker_count = 2, .min_quorum = 1, .num_fragments = 1 },
        &[_]Action{
            .{ .tag = .crash, .worker_id = 0 },
            .{ .tag = .restart, .worker_id = 0 },
            .{ .tag = .resume_worker, .worker_id = 0, .session_id = 99 },
        },
    );
    try std.testing.expectEqual(Violation.incompatible_resume, bad_snapshot.violation);

    const ok_snapshot = try runActions(
        std.testing.allocator,
        .{ .session_id = 12, .worker_count = 2, .min_quorum = 1, .num_fragments = 1 },
        &[_]Action{
            .{ .tag = .crash, .worker_id = 0 },
            .{ .tag = .restart, .worker_id = 0 },
            .{ .tag = .resume_worker, .worker_id = 0, .session_id = 12 },
        },
    );
    try std.testing.expectEqual(Violation.none, ok_snapshot.violation);
    try std.testing.expectEqual(Decision.resumable, ok_snapshot.decision);
}
