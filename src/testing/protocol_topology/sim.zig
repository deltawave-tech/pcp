const std = @import("std");
const trace = @import("trace.zig");

pub const Role = enum(u8) {
    hub,
    gateway,
    worker,
    api_caller,
};

pub const NodeState = enum(u8) {
    online,
    crashed,
};

pub const Node = struct {
    id: trace.NodeId,
    role: Role,
    state: NodeState = .online,
};

pub const DeterministicRng = struct {
    seed: trace.Seed,
    prng: std.rand.DefaultPrng,

    pub fn init(seed: trace.Seed) DeterministicRng {
        return .{
            .seed = seed,
            .prng = std.rand.DefaultPrng.init(seed),
        };
    }

    pub fn nextIndex(self: *DeterministicRng, upper_bound: usize) usize {
        std.debug.assert(upper_bound > 0);
        return self.prng.random().uintLessThan(usize, upper_bound);
    }
};

pub const TwoNodeModel = struct {
    seed: trace.Seed,
    nodes: [2]Node,
    steps: u32 = 0,
    pending_messages: u32 = 0,
    delivered_messages: u32 = 0,
    checksum: u64,
    failure: trace.Failure = .none,

    pub fn init(seed: trace.Seed) TwoNodeModel {
        return .{
            .seed = seed,
            .nodes = .{
                .{ .id = 0, .role = .gateway },
                .{ .id = 1, .role = .worker },
            },
            .checksum = seed ^ 0x544f_504f_5f53_494d,
        };
    }

    pub fn apply(self: *TwoNodeModel, event: trace.Event) trace.Failure {
        if (self.failure != .none) return self.failure;

        self.steps += 1;
        self.checksum = mixChecksum(self.checksum, event.hashValue());

        if (event.actor >= self.nodes.len) {
            self.failure = .invalid_actor;
            return self.failure;
        }

        const peer = event.peerOrNull();
        if (peer) |peer_id| {
            if (peer_id >= self.nodes.len) {
                self.failure = .invalid_peer;
                return self.failure;
            }
        }

        const actor = &self.nodes[event.actor];
        if (actor.state == .crashed and event.tag != .restart and event.tag != .join) {
            self.failure = .event_on_crashed_node;
            return self.failure;
        }

        switch (event.tag) {
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
            .send => {
                if (peer == null) {
                    self.failure = .invalid_peer;
                } else if (self.pending_messages != 0) {
                    self.failure = .duplicate_send;
                } else {
                    self.pending_messages += 1;
                }
            },
            .deliver => {
                if (peer == null) {
                    self.failure = .invalid_peer;
                } else if (self.pending_messages == 0) {
                    self.failure = .invalid_delivery;
                } else {
                    self.pending_messages -= 1;
                    self.delivered_messages += 1;
                }
            },
            .duplicate => {
                if (self.pending_messages == 0) {
                    self.failure = .invalid_delivery;
                } else {
                    self.pending_messages += 1;
                }
            },
            .drop => {
                if (self.pending_messages == 0) {
                    self.failure = .invalid_delivery;
                } else {
                    self.pending_messages -= 1;
                }
            },
            .crash, .leave => {
                actor.state = .crashed;
            },
            .restart, .join => {
                actor.state = .online;
            },
            .cancel => {
                self.pending_messages = 0;
            },
        }

        return self.failure;
    }

    pub fn snapshot(self: TwoNodeModel) trace.Snapshot {
        var online_mask: u64 = 0;
        for (self.nodes, 0..) |node, index| {
            if (node.state == .online) {
                online_mask |= @as(u64, 1) << @intCast(index);
            }
        }

        return .{
            .steps = self.steps,
            .node_count = self.nodes.len,
            .online_mask = online_mask,
            .pending_messages = self.pending_messages,
            .delivered_messages = self.delivered_messages,
            .checksum = self.checksum,
            .failure = self.failure,
        };
    }
};

pub const ExplorationResult = struct {
    checked: usize = 0,
    first_failure: ?trace.OwnedTrace = null,

    pub fn deinit(self: *ExplorationResult) void {
        if (self.first_failure) |*failure| {
            failure.deinit();
        }
        self.first_failure = null;
    }
};

pub const ExplorationOptions = struct {
    include_invalid_events: bool = false,
};

const valid_two_node_alphabet = [_]trace.Event{
    trace.Event.init(.send, 0, 1, 1),
    trace.Event.init(.deliver, 1, 0, 1),
    trace.Event.init(.timeout, 0, null, 0),
};

const control_two_node_alphabet = [_]trace.Event{
    trace.Event.init(.timeout, 0, null, 0),
    trace.Event.init(.heartbeat, 1, 0, 0),
    trace.Event.init(.local_step_complete, 1, null, 0),
};

pub fn runTwoNodeTrace(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    events: []const trace.Event,
) !trace.OwnedTrace {
    var recorder = trace.Recorder.init(allocator, seed);
    defer recorder.deinit();

    var model = TwoNodeModel.init(seed);
    for (events) |event| {
        try recorder.record(event);
        _ = model.apply(event);
        if (model.failure != .none) break;
    }

    recorder.finish(model.snapshot());
    return recorder.toOwnedTrace();
}

pub fn replayTwoNode(seed: trace.Seed, events: []const trace.Event) trace.Snapshot {
    var model = TwoNodeModel.init(seed);
    for (events) |event| {
        _ = model.apply(event);
        if (model.failure != .none) break;
    }
    return model.snapshot();
}

pub fn randomTwoNodeTrace(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    event_count: usize,
) !trace.OwnedTrace {
    var rng = DeterministicRng.init(seed);
    const events = try allocator.alloc(trace.Event, event_count);
    defer allocator.free(events);

    for (events) |*event| {
        event.* = control_two_node_alphabet[rng.nextIndex(control_two_node_alphabet.len)];
    }

    return runTwoNodeTrace(allocator, seed, events);
}

pub fn exploreTwoNodeExhaustive(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    depth: usize,
    options: ExplorationOptions,
) !ExplorationResult {
    if (depth > 8) return error.DepthTooLarge;

    var result = ExplorationResult{};
    var buffer: [8]trace.Event = undefined;
    const alphabet = if (options.include_invalid_events)
        valid_two_node_alphabet[0..]
    else
        control_two_node_alphabet[0..];

    try exploreTwoNodeRecursive(allocator, seed, alphabet, &buffer, 0, depth, &result);
    return result;
}

fn exploreTwoNodeRecursive(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    alphabet: []const trace.Event,
    buffer: *[8]trace.Event,
    cursor: usize,
    depth: usize,
    result: *ExplorationResult,
) !void {
    if (cursor == depth) {
        result.checked += 1;

        var owned = try runTwoNodeTrace(allocator, seed, buffer[0..depth]);
        if (!owned.final_snapshot.ok() and result.first_failure == null) {
            result.first_failure = owned;
        } else {
            owned.deinit();
        }
        return;
    }

    for (alphabet) |event| {
        buffer[cursor] = event;
        try exploreTwoNodeRecursive(allocator, seed, alphabet, buffer, cursor + 1, depth, result);
    }
}

fn mixChecksum(previous: u64, value: u64) u64 {
    var mixed = previous ^ value;
    mixed ^= mixed >> 33;
    mixed *%= 0xff51afd7ed558ccd;
    mixed ^= mixed >> 33;
    mixed *%= 0xc4ceb9fe1a85ec53;
    mixed ^= mixed >> 33;
    return mixed;
}

test "two node valid trace replays to the same final snapshot" {
    const events = [_]trace.Event{
        trace.Event.init(.send, 0, 1, 1),
        trace.Event.init(.deliver, 1, 0, 1),
    };

    var owned = try runTwoNodeTrace(std.testing.allocator, 1234, &events);
    defer owned.deinit();

    const replayed = replayTwoNode(owned.seed, owned.events);

    try std.testing.expect(owned.final_snapshot.ok());
    try std.testing.expect(trace.Snapshot.eql(owned.final_snapshot, replayed));
    try std.testing.expectEqual(@as(u32, 1), replayed.delivered_messages);
}

test "seeded random two node traces are deterministic" {
    var first = try randomTwoNodeTrace(std.testing.allocator, 777, 5);
    defer first.deinit();

    var second = try randomTwoNodeTrace(std.testing.allocator, 777, 5);
    defer second.deinit();

    try std.testing.expectEqual(first.events.len, second.events.len);
    for (first.events, second.events) |a, b| {
        try std.testing.expect(trace.Event.eql(a, b));
    }
    try std.testing.expect(trace.Snapshot.eql(first.final_snapshot, second.final_snapshot));
}

test "exhaustive two node schedules produce replayable failure traces" {
    var result = try exploreTwoNodeExhaustive(std.testing.allocator, 2026, 2, .{
        .include_invalid_events = true,
    });
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 9), result.checked);
    try std.testing.expect(result.first_failure != null);

    const failure = result.first_failure.?;
    const replayed = replayTwoNode(failure.seed, failure.events);

    try std.testing.expect(!failure.final_snapshot.ok());
    try std.testing.expect(trace.Snapshot.eql(failure.final_snapshot, replayed));

    var rendered = std.ArrayList(u8).init(std.testing.allocator);
    defer rendered.deinit();

    try failure.writeText(rendered.writer());

    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "topology-simulator-trace v1 seed=2026 events=") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "failure=") != null);
}

test "exhaustive control schedules stay inside the valid state space" {
    var result = try exploreTwoNodeExhaustive(std.testing.allocator, 2026, 3, .{});
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 27), result.checked);
    try std.testing.expect(result.first_failure == null);
}
