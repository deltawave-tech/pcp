const std = @import("std");
const vector_clock = @import("vector_clock.zig");

const Allocator = std.mem.Allocator;
pub const NodeId = vector_clock.NodeId;

pub const EventKind = enum {
    fragment_sync,
    learner_apply,
    learner_failure,
    learner_recovery,
    snapshot_marker,
    in_flight_message,
};

pub const ParticipantRecord = struct {
    learner_id: NodeId,
    learner_step: usize,
    steps_since_fragment_update: usize,
    tokens_since_fragment_update: usize,
    merge_weight: f32,
};

pub const EventEntry = struct {
    id: u64,
    kind: EventKind,
    syncer_step: usize,
    fragment_id: ?usize = null,
    learner_id: ?NodeId = null,
    snapshot_id: ?u64 = null,
    participants: []ParticipantRecord = &.{},
    clock_entries: []vector_clock.Entry = &.{},

    pub fn clone(self: EventEntry, allocator: Allocator) !EventEntry {
        return .{
            .id = self.id,
            .kind = self.kind,
            .syncer_step = self.syncer_step,
            .fragment_id = self.fragment_id,
            .learner_id = self.learner_id,
            .snapshot_id = self.snapshot_id,
            .participants = try allocator.dupe(ParticipantRecord, self.participants),
            .clock_entries = try allocator.dupe(vector_clock.Entry, self.clock_entries),
        };
    }

    pub fn deinit(self: *EventEntry, allocator: Allocator) void {
        if (self.participants.len > 0) allocator.free(self.participants);
        if (self.clock_entries.len > 0) allocator.free(self.clock_entries);
        self.* = undefined;
    }
};

pub const EventTapeSnapshot = struct {
    entries: []EventEntry,
    next_event_id: u64,
    replay_cursor: usize,

    pub fn deinit(self: *EventTapeSnapshot, allocator: Allocator) void {
        for (self.entries) |*entry| entry.deinit(allocator);
        if (self.entries.len > 0) allocator.free(self.entries);
        self.* = undefined;
    }
};

pub const SnapshotParticipantStatus = enum {
    pending_marker,
    marker_returned,
    failed,
};

pub const RecoveryPlan = struct {
    target_syncer_step: usize,
    started_syncer_step: usize,
    max_recovery_syncer_steps: usize,

    pub fn acceptsSyncerFragment(self: RecoveryPlan, syncer_step: usize) bool {
        return syncer_step >= self.target_syncer_step;
    }

    pub fn mustRestart(self: RecoveryPlan, current_syncer_step: usize) bool {
        return current_syncer_step > self.started_syncer_step + self.max_recovery_syncer_steps;
    }
};

pub const EventTape = struct {
    allocator: Allocator,
    entries: std.ArrayList(EventEntry),
    next_event_id: u64 = 1,
    replay_cursor: usize = 0,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .entries = std.ArrayList(EventEntry).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.clearEntries();
        self.entries.deinit();
        self.* = undefined;
    }

    pub fn appendFragmentSync(
        self: *Self,
        syncer_step: usize,
        fragment_id: usize,
        participants: []const ParticipantRecord,
        clock: vector_clock.VectorClock,
    ) !u64 {
        return try self.append(.{
            .kind = .fragment_sync,
            .syncer_step = syncer_step,
            .fragment_id = fragment_id,
            .participants = participants,
            .clock = clock,
        });
    }

    pub fn appendLearnerApply(
        self: *Self,
        syncer_step: usize,
        fragment_id: usize,
        learner_id: NodeId,
        clock: vector_clock.VectorClock,
    ) !u64 {
        return try self.append(.{
            .kind = .learner_apply,
            .syncer_step = syncer_step,
            .fragment_id = fragment_id,
            .learner_id = learner_id,
            .clock = clock,
        });
    }

    pub fn appendLearnerFailure(self: *Self, syncer_step: usize, learner_id: NodeId, clock: vector_clock.VectorClock) !u64 {
        return try self.append(.{
            .kind = .learner_failure,
            .syncer_step = syncer_step,
            .learner_id = learner_id,
            .clock = clock,
        });
    }

    pub fn appendLearnerRecovery(self: *Self, syncer_step: usize, learner_id: NodeId, clock: vector_clock.VectorClock) !u64 {
        return try self.append(.{
            .kind = .learner_recovery,
            .syncer_step = syncer_step,
            .learner_id = learner_id,
            .clock = clock,
        });
    }

    pub fn appendSnapshotMarker(self: *Self, syncer_step: usize, snapshot_id: u64, clock: vector_clock.VectorClock) !u64 {
        return try self.append(.{
            .kind = .snapshot_marker,
            .syncer_step = syncer_step,
            .snapshot_id = snapshot_id,
            .clock = clock,
        });
    }

    pub fn appendInFlightMessage(
        self: *Self,
        syncer_step: usize,
        snapshot_id: u64,
        learner_id: NodeId,
        clock: vector_clock.VectorClock,
    ) !u64 {
        return try self.append(.{
            .kind = .in_flight_message,
            .syncer_step = syncer_step,
            .learner_id = learner_id,
            .snapshot_id = snapshot_id,
            .clock = clock,
        });
    }

    pub fn replayFragmentSync(self: *Self, syncer_step: usize, fragment_id: usize) ?*const EventEntry {
        while (self.replay_cursor < self.entries.items.len) : (self.replay_cursor += 1) {
            const entry = &self.entries.items[self.replay_cursor];
            if (entry.kind != .fragment_sync) continue;
            if (entry.syncer_step != syncer_step) continue;
            if (entry.fragment_id == null or entry.fragment_id.? != fragment_id) continue;

            self.replay_cursor += 1;
            return entry;
        }
        return null;
    }

    pub fn resetReplay(self: *Self) void {
        self.replay_cursor = 0;
    }

    pub fn snapshot(self: Self, allocator: Allocator) !EventTapeSnapshot {
        const entries = try allocator.alloc(EventEntry, self.entries.items.len);
        var initialized: usize = 0;
        errdefer {
            for (entries[0..initialized]) |*entry| entry.deinit(allocator);
            allocator.free(entries);
        }

        for (self.entries.items, 0..) |entry, i| {
            entries[i] = try entry.clone(allocator);
            initialized += 1;
        }

        return .{
            .entries = entries,
            .next_event_id = self.next_event_id,
            .replay_cursor = self.replay_cursor,
        };
    }

    pub fn restoreSnapshot(self: *Self, snapshot_value: EventTapeSnapshot) !void {
        self.clearEntries();
        try self.entries.ensureTotalCapacity(snapshot_value.entries.len);
        for (snapshot_value.entries) |entry| {
            try self.entries.append(try entry.clone(self.allocator));
        }
        self.next_event_id = snapshot_value.next_event_id;
        self.replay_cursor = snapshot_value.replay_cursor;
    }

    fn clearEntries(self: *Self) void {
        for (self.entries.items) |*entry| entry.deinit(self.allocator);
        self.entries.clearRetainingCapacity();
    }

    fn append(self: *Self, event: AppendEvent) !u64 {
        const id = self.next_event_id;
        self.next_event_id = try std.math.add(u64, self.next_event_id, 1);

        const participants = try self.allocator.dupe(ParticipantRecord, event.participants);
        errdefer self.allocator.free(participants);
        const clock_entries = try self.allocator.dupe(vector_clock.Entry, event.clock.entries);
        errdefer self.allocator.free(clock_entries);

        try self.entries.append(.{
            .id = id,
            .kind = event.kind,
            .syncer_step = event.syncer_step,
            .fragment_id = event.fragment_id,
            .learner_id = event.learner_id,
            .snapshot_id = event.snapshot_id,
            .participants = participants,
            .clock_entries = clock_entries,
        });
        return id;
    }
};

const AppendEvent = struct {
    kind: EventKind,
    syncer_step: usize,
    fragment_id: ?usize = null,
    learner_id: ?NodeId = null,
    snapshot_id: ?u64 = null,
    participants: []const ParticipantRecord = &.{},
    clock: vector_clock.VectorClock,
};

pub fn snapshotCanComplete(statuses: []const SnapshotParticipantStatus) bool {
    for (statuses) |status| {
        if (status == .pending_marker) return false;
    }
    return true;
}

test "event tape replays fragment participant choices and weights" {
    const allocator = std.testing.allocator;
    var clock = vector_clock.VectorClock.init(allocator);
    defer clock.deinit();
    try clock.observe(0, 3);

    var tape = EventTape.init(allocator);
    defer tape.deinit();

    const participants = [_]ParticipantRecord{
        .{ .learner_id = 1, .learner_step = 5, .steps_since_fragment_update = 2, .tokens_since_fragment_update = 16, .merge_weight = 128.0 },
        .{ .learner_id = 3, .learner_step = 4, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 8, .merge_weight = 64.0 },
    };
    _ = try tape.appendFragmentSync(7, 2, &participants, clock);
    _ = try tape.appendLearnerFailure(8, 4, clock);

    const replay = tape.replayFragmentSync(7, 2) orelse return error.MissingReplayEntry;
    try std.testing.expectEqual(EventKind.fragment_sync, replay.kind);
    try std.testing.expectEqual(@as(usize, 2), replay.participants.len);
    try std.testing.expectEqual(@as(NodeId, 3), replay.participants[1].learner_id);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), replay.participants[1].merge_weight, 0.000001);
    try std.testing.expect(tape.replayFragmentSync(7, 2) == null);
}

test "event tape snapshot restores entries and replay cursor" {
    const allocator = std.testing.allocator;
    var clock = vector_clock.VectorClock.init(allocator);
    defer clock.deinit();
    try clock.observe(0, 1);

    var tape = EventTape.init(allocator);
    defer tape.deinit();
    _ = try tape.appendSnapshotMarker(3, 99, clock);
    _ = try tape.appendFragmentSync(4, 0, &.{}, clock);
    _ = tape.replayFragmentSync(4, 0) orelse return error.MissingReplayEntry;

    var snap = try tape.snapshot(allocator);
    defer snap.deinit(allocator);

    _ = try tape.appendLearnerRecovery(5, 2, clock);
    try tape.restoreSnapshot(snap);

    try std.testing.expectEqual(@as(usize, 2), tape.entries.items.len);
    try std.testing.expectEqual(@as(usize, 2), tape.replay_cursor);
    try std.testing.expectEqual(EventKind.snapshot_marker, tape.entries.items[0].kind);
}

test "snapshot completion skips failed learners" {
    const statuses = [_]SnapshotParticipantStatus{
        .marker_returned,
        .failed,
        .marker_returned,
    };
    try std.testing.expect(snapshotCanComplete(&statuses));

    const pending = [_]SnapshotParticipantStatus{
        .marker_returned,
        .pending_marker,
    };
    try std.testing.expect(!snapshotCanComplete(&pending));
}

test "recovery plan waits for target stamp and expires after H steps" {
    const plan = RecoveryPlan{
        .target_syncer_step = 10,
        .started_syncer_step = 8,
        .max_recovery_syncer_steps = 4,
    };

    try std.testing.expect(!plan.acceptsSyncerFragment(9));
    try std.testing.expect(plan.acceptsSyncerFragment(10));
    try std.testing.expect(!plan.mustRestart(12));
    try std.testing.expect(plan.mustRestart(13));
}
