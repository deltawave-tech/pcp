const std = @import("std");

const trace = @import("trace.zig");

pub const EventPredicate = *const fn (trace.Seed, []const trace.Event) bool;

pub const OwnedEventShrink = struct {
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    events: []trace.Event,
    original_event_count: usize,

    pub fn deinit(self: *OwnedEventShrink) void {
        self.allocator.free(self.events);
        self.events = &.{};
        self.seed = 0;
        self.original_event_count = 0;
    }

    pub fn deletedEvents(self: OwnedEventShrink) usize {
        return self.original_event_count - self.events.len;
    }
};

pub const OwnedIndexSet = struct {
    allocator: std.mem.Allocator,
    indices: []usize,
    original_count: usize,

    pub fn deinit(self: *OwnedIndexSet) void {
        self.allocator.free(self.indices);
        self.indices = &.{};
        self.original_count = 0;
    }

    pub fn removedCount(self: OwnedIndexSet) usize {
        return self.original_count - self.indices.len;
    }
};

pub const CounterexampleReport = struct {
    title: []const u8 = "topology simulator counterexample",
    input_simplex: []const u8,
    final_protocol_simplex: []const u8,
    output_decision: []const u8,
    violated_rule: []const u8,
    seed: trace.Seed,
    events: []const trace.Event = &.{},

    pub fn writeHuman(self: CounterexampleReport, writer: anytype) !void {
        try writer.print("{s}\n", .{self.title});
        try writer.print("input_simplex: {s}\n", .{self.input_simplex});
        try writer.print("final_protocol_simplex: {s}\n", .{self.final_protocol_simplex});
        try writer.print("output_decision: {s}\n", .{self.output_decision});
        try writer.print("violated_rule: {s}\n", .{self.violated_rule});
        try writer.print("seed: {d}\n", .{self.seed});
        try writer.print("events: {d}\n", .{self.events.len});
        for (self.events, 0..) |event, index| {
            try event.writeText(writer, index);
        }
    }

    pub fn writeReplayFixture(self: CounterexampleReport, writer: anytype) !void {
        try writer.writeAll("{\"schema\":\"pcp.topology.replay.v1\"");
        try writer.print(",\"seed\":{d}", .{self.seed});
        try writer.writeAll(",\"input_simplex\":");
        try std.json.stringify(self.input_simplex, .{}, writer);
        try writer.writeAll(",\"final_protocol_simplex\":");
        try std.json.stringify(self.final_protocol_simplex, .{}, writer);
        try writer.writeAll(",\"output_decision\":");
        try std.json.stringify(self.output_decision, .{}, writer);
        try writer.writeAll(",\"violated_rule\":");
        try std.json.stringify(self.violated_rule, .{}, writer);
        try writer.writeAll(",\"events\":[");
        for (self.events, 0..) |event, index| {
            if (index != 0) try writer.writeByte(',');
            try writer.writeAll("{\"tag\":");
            try std.json.stringify(@tagName(event.tag), .{}, writer);
            try writer.print(",\"actor\":{d},\"peer\":", .{event.actor});
            if (event.peerOrNull()) |peer| {
                try writer.print("{d}", .{peer});
            } else {
                try writer.writeAll("null");
            }
            try writer.print(",\"message_id\":{d}}}", .{event.message_id});
        }
        try writer.writeAll("]}");
    }
};

pub fn shrinkEvents(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    events: []const trace.Event,
    predicate: EventPredicate,
) !OwnedEventShrink {
    if (!predicate(seed, events)) return error.PredicateDoesNotHold;

    var current = try allocator.dupe(trace.Event, events);
    errdefer allocator.free(current);

    var changed = true;
    while (changed) {
        changed = false;
        if (current.len == 0) break;

        var index: usize = 0;
        while (index < current.len) : (index += 1) {
            const candidate = try removeEventAt(allocator, current, index);
            errdefer allocator.free(candidate);

            if (predicate(seed, candidate)) {
                allocator.free(current);
                current = candidate;
                changed = true;
                break;
            }

            allocator.free(candidate);
        }
    }

    const minimized_seed = minimizeSeed(seed, current, predicate);
    return .{
        .allocator = allocator,
        .seed = minimized_seed,
        .events = current,
        .original_event_count = events.len,
    };
}

pub fn shrinkIndices(
    allocator: std.mem.Allocator,
    item_count: usize,
    context: anytype,
    predicate: anytype,
) !OwnedIndexSet {
    var current = try allocator.alloc(usize, item_count);
    errdefer allocator.free(current);
    for (current, 0..) |*index, value| {
        index.* = value;
    }

    if (!predicate(context, current)) return error.PredicateDoesNotHold;

    var changed = true;
    while (changed) {
        changed = false;
        if (current.len == 0) break;

        var position: usize = 0;
        while (position < current.len) : (position += 1) {
            const candidate = try removeIndexAt(allocator, current, position);
            errdefer allocator.free(candidate);

            if (predicate(context, candidate)) {
                allocator.free(current);
                current = candidate;
                changed = true;
                break;
            }

            allocator.free(candidate);
        }
    }

    return .{
        .allocator = allocator,
        .indices = current,
        .original_count = item_count,
    };
}

pub fn shrinkParticipants(
    allocator: std.mem.Allocator,
    participant_count: usize,
    context: anytype,
    predicate: anytype,
) !OwnedIndexSet {
    return shrinkIndices(allocator, participant_count, context, predicate);
}

pub fn shrinkWindows(
    allocator: std.mem.Allocator,
    window_count: usize,
    context: anytype,
    predicate: anytype,
) !OwnedIndexSet {
    return shrinkIndices(allocator, window_count, context, predicate);
}

fn minimizeSeed(seed: trace.Seed, events: []const trace.Event, predicate: EventPredicate) trace.Seed {
    var current = seed;
    if (current != 0 and predicate(0, events)) return 0;
    if (current > 1 and predicate(1, events)) current = 1;

    var candidate = current / 2;
    while (candidate > 1) : (candidate /= 2) {
        if (candidate < current and predicate(candidate, events)) {
            current = candidate;
        }
    }
    return current;
}

fn removeEventAt(
    allocator: std.mem.Allocator,
    events: []const trace.Event,
    remove_index: usize,
) ![]trace.Event {
    std.debug.assert(remove_index < events.len);
    const candidate = try allocator.alloc(trace.Event, events.len - 1);
    var out_index: usize = 0;
    for (events, 0..) |event, index| {
        if (index == remove_index) continue;
        candidate[out_index] = event;
        out_index += 1;
    }
    return candidate;
}

fn removeIndexAt(
    allocator: std.mem.Allocator,
    indices: []const usize,
    remove_position: usize,
) ![]usize {
    std.debug.assert(remove_position < indices.len);
    const candidate = try allocator.alloc(usize, indices.len - 1);
    var out_index: usize = 0;
    for (indices, 0..) |index_value, position| {
        if (position == remove_position) continue;
        candidate[out_index] = index_value;
        out_index += 1;
    }
    return candidate;
}

fn twoNodeInvalidDelivery(seed: trace.Seed, events: []const trace.Event) bool {
    var pending_messages: u32 = 0;
    var online_mask: u64 = 0b11;
    _ = seed;

    for (events) |event| {
        if (event.actor >= 2) return false;
        const actor_mask = @as(u64, 1) << @intCast(event.actor);
        if ((online_mask & actor_mask) == 0 and event.tag != .restart and event.tag != .join) return false;

        switch (event.tag) {
            .send => {
                if (event.peerOrNull() == null or pending_messages != 0) return false;
                pending_messages += 1;
            },
            .deliver => {
                if (event.peerOrNull() == null) return false;
                if (pending_messages == 0) return true;
                pending_messages -= 1;
            },
            .duplicate => {
                if (pending_messages == 0) return true;
                pending_messages += 1;
            },
            .drop => {
                if (pending_messages == 0) return true;
                pending_messages -= 1;
            },
            .crash, .leave => online_mask &= ~actor_mask,
            .restart, .join => online_mask |= actor_mask,
            .cancel => pending_messages = 0,
            .noop,
            .timeout,
            .heartbeat,
            .health_update,
            .local_step_complete,
            .lease_reserve,
            .lease_commit,
            .lease_release,
            .lease_expire,
            => {},
        }
    }
    return false;
}

const QuorumBugContext = struct {
    min_quorum: usize,
    participants: []const bool,
};

fn belowQuorumBugStillFails(context: *const QuorumBugContext, retained: []const usize) bool {
    var participant_count: usize = 0;
    for (retained) |index| {
        if (context.participants[index]) participant_count += 1;
    }
    return participant_count > 0 and participant_count < context.min_quorum;
}

fn duplicateReservationStillFails(_: void, retained: []const usize) bool {
    const reservation_worker_ids = [_]u16{ 7, 8, 7, 9 };
    for (retained, 0..) |left_index, left_position| {
        for (retained[left_position + 1 ..]) |right_index| {
            if (reservation_worker_ids[left_index] == reservation_worker_ids[right_index]) {
                return true;
            }
        }
    }
    return false;
}

fn windowFailureStillFails(_: void, retained: []const usize) bool {
    const failing_window: usize = 2;
    for (retained) |window| {
        if (window == failing_window) return true;
    }
    return false;
}

test "event deletion shrinking keeps the minimal replayable failure" {
    const events = [_]trace.Event{
        trace.Event.init(.timeout, 0, null, 0),
        trace.Event.init(.heartbeat, 1, 0, 0),
        trace.Event.init(.deliver, 1, 0, 1),
        trace.Event.init(.send, 0, 1, 1),
    };

    var shrunk = try shrinkEvents(std.testing.allocator, 777, &events, twoNodeInvalidDelivery);
    defer shrunk.deinit();

    try std.testing.expectEqual(@as(trace.Seed, 0), shrunk.seed);
    try std.testing.expectEqual(@as(usize, 1), shrunk.events.len);
    try std.testing.expectEqual(trace.EventTag.deliver, shrunk.events[0].tag);
    try std.testing.expectEqual(@as(usize, 3), shrunk.deletedEvents());
}

test "participant shrinking minimizes a deliberately injected below quorum bug" {
    const participants = [_]bool{ true, false, false, false };
    const context = QuorumBugContext{
        .min_quorum = 2,
        .participants = &participants,
    };

    var shrunk = try shrinkParticipants(
        std.testing.allocator,
        4,
        &context,
        belowQuorumBugStillFails,
    );
    defer shrunk.deinit();

    try std.testing.expectEqual(@as(usize, 1), shrunk.indices.len);
    try std.testing.expectEqual(@as(usize, 3), shrunk.removedCount());
}

test "participant shrinking minimizes a duplicate reservation counterexample" {
    var shrunk = try shrinkParticipants(
        std.testing.allocator,
        4,
        {},
        duplicateReservationStillFails,
    );
    defer shrunk.deinit();

    try std.testing.expectEqual(@as(usize, 2), shrunk.indices.len);
    try std.testing.expectEqual(@as(usize, 2), shrunk.removedCount());
    try std.testing.expectEqual(@as(usize, 0), shrunk.indices[0]);
    try std.testing.expectEqual(@as(usize, 2), shrunk.indices[1]);
}

test "window shrinking keeps only the failing window" {
    var shrunk = try shrinkWindows(
        std.testing.allocator,
        5,
        {},
        windowFailureStillFails,
    );
    defer shrunk.deinit();

    try std.testing.expectEqual(@as(usize, 1), shrunk.indices.len);
    try std.testing.expectEqual(@as(usize, 2), shrunk.indices[0]);
}

test "counterexample report includes topology decision and replay fixture fields" {
    const events = [_]trace.Event{
        trace.Event.init(.deliver, 1, 0, 1),
    };
    const report = CounterexampleReport{
        .input_simplex = "input.gateway.connected",
        .final_protocol_simplex = "protocol.delivery.invalid",
        .output_decision = "output.rejected",
        .violated_rule = "oracle.invalid_delivery",
        .seed = 0,
        .events = &events,
    };

    var human = std.ArrayList(u8).init(std.testing.allocator);
    defer human.deinit();
    try report.writeHuman(human.writer());

    try std.testing.expect(std.mem.indexOf(u8, human.items, "input_simplex: input.gateway.connected") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "final_protocol_simplex: protocol.delivery.invalid") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "output_decision: output.rejected") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "violated_rule: oracle.invalid_delivery") != null);

    var fixture = std.ArrayList(u8).init(std.testing.allocator);
    defer fixture.deinit();
    try report.writeReplayFixture(fixture.writer());

    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"schema\":\"pcp.topology.replay.v1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"input_simplex\":\"input.gateway.connected\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"final_protocol_simplex\":\"protocol.delivery.invalid\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"output_decision\":\"output.rejected\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"violated_rule\":\"oracle.invalid_delivery\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"tag\":\"deliver\"") != null);
}
