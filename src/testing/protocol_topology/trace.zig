const std = @import("std");

pub const Seed = u64;
pub const NodeId = u16;
pub const MessageId = u32;
pub const no_node: NodeId = std.math.maxInt(NodeId);

pub const EventTag = enum(u8) {
    noop,
    send,
    deliver,
    drop,
    duplicate,
    timeout,
    crash,
    restart,
    join,
    leave,
    cancel,
    heartbeat,
    health_update,
    local_step_complete,
    lease_reserve,
    lease_commit,
    lease_release,
    lease_expire,
};

pub const Failure = enum(u8) {
    none,
    invalid_actor,
    invalid_peer,
    invalid_delivery,
    duplicate_send,
    event_on_crashed_node,
};

pub const Event = struct {
    tag: EventTag,
    actor: NodeId,
    peer: NodeId = no_node,
    message_id: MessageId = 0,

    pub fn init(tag: EventTag, actor: NodeId, peer: ?NodeId, message_id: MessageId) Event {
        return .{
            .tag = tag,
            .actor = actor,
            .peer = peer orelse no_node,
            .message_id = message_id,
        };
    }

    pub fn peerOrNull(self: Event) ?NodeId {
        if (self.peer == no_node) return null;
        return self.peer;
    }

    pub fn eql(a: Event, b: Event) bool {
        return a.tag == b.tag and
            a.actor == b.actor and
            a.peer == b.peer and
            a.message_id == b.message_id;
    }

    pub fn hashValue(self: Event) u64 {
        var hash: u64 = 14695981039346656037;
        hash = (hash ^ @as(u64, @intFromEnum(self.tag))) *% 1099511628211;
        hash = (hash ^ @as(u64, self.actor)) *% 1099511628211;
        hash = (hash ^ @as(u64, self.peer)) *% 1099511628211;
        hash = (hash ^ @as(u64, self.message_id)) *% 1099511628211;
        return hash;
    }

    pub fn writeText(self: Event, writer: anytype, index: usize) !void {
        try writer.print("{d}: {s} actor={d}", .{ index, @tagName(self.tag), self.actor });
        if (self.peerOrNull()) |peer| {
            try writer.print(" peer={d}", .{peer});
        } else {
            try writer.writeAll(" peer=-");
        }
        try writer.print(" message={d}\n", .{self.message_id});
    }
};

pub const Snapshot = struct {
    steps: u32 = 0,
    node_count: u8 = 0,
    online_mask: u64 = 0,
    pending_messages: u32 = 0,
    delivered_messages: u32 = 0,
    checksum: u64 = 0,
    failure: Failure = .none,

    pub fn ok(self: Snapshot) bool {
        return self.failure == .none;
    }

    pub fn eql(a: Snapshot, b: Snapshot) bool {
        return a.steps == b.steps and
            a.node_count == b.node_count and
            a.online_mask == b.online_mask and
            a.pending_messages == b.pending_messages and
            a.delivered_messages == b.delivered_messages and
            a.checksum == b.checksum and
            a.failure == b.failure;
    }

    pub fn writeText(self: Snapshot, writer: anytype) !void {
        try writer.print(
            "final steps={d} nodes={d} online_mask={d} pending={d} delivered={d} checksum={d} failure={s}\n",
            .{
                self.steps,
                self.node_count,
                self.online_mask,
                self.pending_messages,
                self.delivered_messages,
                self.checksum,
                @tagName(self.failure),
            },
        );
    }

    pub fn hashValue(self: Snapshot) u64 {
        var hasher = std.hash.Wyhash.init(0x7072_6f74_6f5f_7374);
        var buf: [8]u8 = undefined;
        var small: [4]u8 = undefined;

        std.mem.writeInt(u32, &small, self.steps, .little);
        hasher.update(&small);
        hasher.update(&.{self.node_count});
        std.mem.writeInt(u64, &buf, self.online_mask, .little);
        hasher.update(&buf);
        std.mem.writeInt(u32, &small, self.pending_messages, .little);
        hasher.update(&small);
        std.mem.writeInt(u32, &small, self.delivered_messages, .little);
        hasher.update(&small);
        std.mem.writeInt(u64, &buf, self.checksum, .little);
        hasher.update(&buf);
        hasher.update(&.{@intFromEnum(self.failure)});

        return hasher.final();
    }
};

pub const OwnedTrace = struct {
    allocator: std.mem.Allocator,
    seed: Seed,
    events: []Event,
    final_snapshot: Snapshot,

    pub fn deinit(self: *OwnedTrace) void {
        self.allocator.free(self.events);
        self.events = &.{};
        self.final_snapshot = .{};
        self.seed = 0;
    }

    pub fn writeText(self: OwnedTrace, writer: anytype) !void {
        try writer.print("topology-simulator-trace v1 seed={d} events={d}\n", .{ self.seed, self.events.len });
        for (self.events, 0..) |event, index| {
            try event.writeText(writer, index);
        }
        try self.final_snapshot.writeText(writer);
    }
};

pub const Recorder = struct {
    allocator: std.mem.Allocator,
    seed: Seed,
    events: std.ArrayList(Event),
    final_snapshot: ?Snapshot = null,

    pub fn init(allocator: std.mem.Allocator, seed: Seed) Recorder {
        return .{
            .allocator = allocator,
            .seed = seed,
            .events = std.ArrayList(Event).init(allocator),
        };
    }

    pub fn deinit(self: *Recorder) void {
        self.events.deinit();
        self.final_snapshot = null;
    }

    pub fn record(self: *Recorder, event: Event) !void {
        try self.events.append(event);
    }

    pub fn finish(self: *Recorder, snapshot: Snapshot) void {
        self.final_snapshot = snapshot;
    }

    pub fn toOwnedTrace(self: *Recorder) !OwnedTrace {
        const events = try self.events.toOwnedSlice();
        self.events = std.ArrayList(Event).init(self.allocator);
        return .{
            .allocator = self.allocator,
            .seed = self.seed,
            .events = events,
            .final_snapshot = self.final_snapshot orelse .{},
        };
    }
};

test "event text format is stable" {
    const event = Event.init(.send, 1, 2, 42);
    var rendered = std.ArrayList(u8).init(std.testing.allocator);
    defer rendered.deinit();

    try event.writeText(rendered.writer(), 7);

    try std.testing.expectEqualStrings("7: send actor=1 peer=2 message=42\n", rendered.items);
}

test "owned trace text includes replay fields" {
    var owned = OwnedTrace{
        .allocator = std.testing.allocator,
        .seed = 99,
        .events = try std.testing.allocator.dupe(Event, &[_]Event{
            Event.init(.timeout, 0, null, 0),
        }),
        .final_snapshot = .{
            .steps = 1,
            .node_count = 2,
            .online_mask = 3,
            .checksum = 100,
        },
    };
    defer owned.deinit();

    var rendered = std.ArrayList(u8).init(std.testing.allocator);
    defer rendered.deinit();

    try owned.writeText(rendered.writer());

    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "topology-simulator-trace v1 seed=99 events=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "final steps=1 nodes=2") != null);
}

test "snapshot hash is stable for canonical state comparison" {
    const left = Snapshot{
        .steps = 2,
        .node_count = 2,
        .online_mask = 3,
        .pending_messages = 1,
        .delivered_messages = 1,
        .checksum = 99,
    };
    const right = left;
    const different = Snapshot{
        .steps = 2,
        .node_count = 2,
        .online_mask = 1,
        .pending_messages = 1,
        .delivered_messages = 1,
        .checksum = 99,
    };

    try std.testing.expectEqual(left.hashValue(), right.hashValue());
    try std.testing.expect(left.hashValue() != different.hashValue());
}
