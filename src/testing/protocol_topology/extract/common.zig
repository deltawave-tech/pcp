const std = @import("std");

const tasks = @import("../tasks.zig");
const trace = @import("../trace.zig");

pub const ExtractionBound = struct {
    max_cases: usize,
    max_traces: usize,
    max_events: usize,
    max_depth: usize,

    pub fn validate(self: ExtractionBound) !void {
        if (self.max_cases == 0) return error.EmptyExtractionCaseBound;
        if (self.max_traces == 0) return error.EmptyExtractionTraceBound;
        if (self.max_events == 0) return error.EmptyExtractionEventBound;
    }
};

pub const SourceMetadata = struct {
    source_module: []const u8,
    fixture: []const u8,
    seed: trace.Seed,
    bound: ExtractionBound,
    trace_id: []const u8,
    evidence: tasks.CoverageEvidence,
    production_path: []const u8,

    pub fn validate(self: SourceMetadata) !void {
        try self.bound.validate();
        if (self.source_module.len == 0) return error.EmptySourceModule;
        if (self.fixture.len == 0) return error.EmptyFixture;
        if (self.trace_id.len == 0) return error.EmptyTraceId;
        if (self.production_path.len == 0) return error.EmptyProductionPath;
    }
};

pub const OwnedSmokeReplayFixture = struct {
    allocator: std.mem.Allocator,
    trace_id: []const u8,
    input_label: []const u8,
    output_label: []const u8,
    metadata: SourceMetadata,
    trace_value: trace.OwnedTrace,

    pub fn deinit(self: *OwnedSmokeReplayFixture) void {
        self.trace_value.deinit();
        self.allocator.free(self.output_label);
        self.allocator.free(self.input_label);
        self.allocator.free(self.trace_id);
        self.* = undefined;
    }
};

pub fn validateCaseTraceBounds(bound: ExtractionBound, case_count: usize, trace_count: usize) !void {
    try bound.validate();
    if (case_count > bound.max_cases) return error.ExtractionCaseBoundExceeded;
    if (trace_count > bound.max_traces) return error.ExtractionTraceBoundExceeded;
}

pub fn validateTraceWithinBound(trace_value: trace.OwnedTrace, bound: ExtractionBound) !void {
    try bound.validate();
    if (trace_value.events.len > bound.max_events) return error.ExtractionEventBoundExceeded;
    if (trace_value.final_snapshot.steps > bound.max_depth) return error.ExtractionDepthBoundExceeded;
}

pub fn validateMetadataSet(metadata: []const SourceMetadata, expected_count: usize) !void {
    if (metadata.len != expected_count) return error.ExtractionMetadataCountMismatch;
    for (metadata) |item| try item.validate();
}

pub fn sourceMetadata(
    source_module: []const u8,
    fixture: []const u8,
    seed: trace.Seed,
    bound: ExtractionBound,
    trace_id: []const u8,
    evidence: tasks.CoverageEvidence,
    production_path: []const u8,
) SourceMetadata {
    return .{
        .source_module = source_module,
        .fixture = fixture,
        .seed = seed,
        .bound = bound,
        .trace_id = trace_id,
        .evidence = evidence,
        .production_path = production_path,
    };
}

pub fn importSmokeReplayFixture(
    allocator: std.mem.Allocator,
    json_bytes: []const u8,
    bound: ExtractionBound,
    source_module: []const u8,
    production_path: []const u8,
) !OwnedSmokeReplayFixture {
    try bound.validate();

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();

    const object = parsed.value.object;
    const schema = stringField(object, "schema") orelse return error.MissingSmokeReplaySchema;
    if (!std.mem.eql(u8, schema, "pcp.topology.smoke_replay.v1")) return error.UnsupportedSmokeReplaySchema;

    const trace_id = try allocator.dupe(u8, stringField(object, "trace_id") orelse return error.MissingSmokeReplayTraceId);
    errdefer allocator.free(trace_id);

    const input_label = try allocator.dupe(u8, stringField(object, "input") orelse return error.MissingSmokeReplayInput);
    errdefer allocator.free(input_label);

    const output_label = try allocator.dupe(u8, stringField(object, "output") orelse return error.MissingSmokeReplayOutput);
    errdefer allocator.free(output_label);

    const seed_value = integerField(object, "seed") orelse return error.MissingSmokeReplaySeed;
    if (seed_value < 0) return error.InvalidSmokeReplaySeed;
    const seed: trace.Seed = @intCast(seed_value);

    const events_value = object.get("events") orelse return error.MissingSmokeReplayEvents;
    if (events_value != .array) return error.InvalidSmokeReplayEvents;
    if (events_value.array.items.len > bound.max_events) return error.ExtractionEventBoundExceeded;

    const events = try allocator.alloc(trace.Event, events_value.array.items.len);
    errdefer allocator.free(events);
    for (events_value.array.items, 0..) |event_value, index| {
        events[index] = try parseEvent(event_value);
    }

    const snapshot = trace.Snapshot{
        .steps = @intCast(events.len),
        .node_count = 0,
        .online_mask = 0,
        .checksum = smokeReplayHash(seed, input_label, output_label, events),
    };
    if (snapshot.steps > bound.max_depth) return error.ExtractionDepthBoundExceeded;

    const metadata = sourceMetadata(
        source_module,
        input_label,
        seed,
        bound,
        trace_id,
        .smoke_replay,
        production_path,
    );
    try metadata.validate();

    return .{
        .allocator = allocator,
        .trace_id = trace_id,
        .input_label = input_label,
        .output_label = output_label,
        .metadata = metadata,
        .trace_value = .{
            .allocator = allocator,
            .seed = seed,
            .events = events,
            .final_snapshot = snapshot,
        },
    };
}

fn parseEvent(value: std.json.Value) !trace.Event {
    if (value != .object) return error.InvalidSmokeReplayEvent;
    const object = value.object;

    const tag_name = stringField(object, "tag") orelse return error.MissingSmokeReplayEventTag;
    const actor = integerField(object, "actor") orelse return error.MissingSmokeReplayEventActor;
    const message_id = integerField(object, "message_id") orelse return error.MissingSmokeReplayEventMessageId;
    if (actor < 0 or message_id < 0) return error.InvalidSmokeReplayEvent;

    const peer_value = object.get("peer") orelse return error.MissingSmokeReplayEventPeer;
    const peer: ?trace.NodeId = switch (peer_value) {
        .null => null,
        .integer => |value_int| blk: {
            if (value_int < 0) return error.InvalidSmokeReplayEvent;
            break :blk @intCast(value_int);
        },
        else => return error.InvalidSmokeReplayEvent,
    };

    return trace.Event.init(
        try parseEventTag(tag_name),
        @intCast(actor),
        peer,
        @intCast(message_id),
    );
}

fn parseEventTag(name: []const u8) !trace.EventTag {
    inline for (std.meta.fields(trace.EventTag)) |field| {
        if (std.mem.eql(u8, name, field.name)) {
            return @as(trace.EventTag, @enumFromInt(field.value));
        }
    }
    return error.UnknownSmokeReplayEventTag;
}

fn stringField(object: std.json.ObjectMap, name: []const u8) ?[]const u8 {
    const value = object.get(name) orelse return null;
    return switch (value) {
        .string => |string| string,
        else => null,
    };
}

fn integerField(object: std.json.ObjectMap, name: []const u8) ?i64 {
    const value = object.get(name) orelse return null;
    return switch (value) {
        .integer => |int_value| int_value,
        else => null,
    };
}

fn smokeReplayHash(seed: trace.Seed, input_label: []const u8, output_label: []const u8, events: []const trace.Event) u64 {
    var hasher = std.hash.Wyhash.init(0x736d_6f6b_655f_7869);
    hashU64(&hasher, seed);
    hasher.update(input_label);
    hasher.update(output_label);
    for (events) |event| hashU64(&hasher, event.hashValue());
    return hasher.final();
}

fn hashU64(hasher: *std.hash.Wyhash, value: u64) void {
    var buffer: [8]u8 = undefined;
    std.mem.writeInt(u64, &buffer, value, .little);
    hasher.update(&buffer);
}

test "smoke replay fixture import preserves stable trace metadata and bounds" {
    const bound = ExtractionBound{
        .max_cases = 1,
        .max_traces = 1,
        .max_events = 4,
        .max_depth = 4,
    };
    const json =
        \\{
        \\  "schema": "pcp.topology.smoke_replay.v1",
        \\  "trace_id": "gateway-training-smoke-001",
        \\  "seed": 42,
        \\  "input": "workload.regular",
        \\  "output": "status.accepted",
        \\  "events": [
        \\    {"tag": "send", "actor": 0, "peer": 1, "message_id": 10},
        \\    {"tag": "deliver", "actor": 1, "peer": 0, "message_id": 10}
        \\  ]
        \\}
    ;

    var fixture = try importSmokeReplayFixture(
        std.testing.allocator,
        json,
        bound,
        "test/nanochat/test_gateway_topology_integration.py",
        "gateway.training",
    );
    defer fixture.deinit();

    try std.testing.expectEqual(@as(trace.Seed, 42), fixture.trace_value.seed);
    try std.testing.expectEqual(@as(usize, 2), fixture.trace_value.events.len);
    try std.testing.expectEqual(trace.EventTag.send, fixture.trace_value.events[0].tag);
    try std.testing.expectEqualStrings("gateway-training-smoke-001", fixture.metadata.trace_id);
    try std.testing.expectEqual(tasks.CoverageEvidence.smoke_replay, fixture.metadata.evidence);
    try validateTraceWithinBound(fixture.trace_value, bound);
}

test "smoke replay fixture import rejects event bound overflow" {
    const bound = ExtractionBound{
        .max_cases = 1,
        .max_traces = 1,
        .max_events = 1,
        .max_depth = 1,
    };
    const json =
        \\{
        \\  "schema": "pcp.topology.smoke_replay.v1",
        \\  "trace_id": "too-long",
        \\  "seed": 1,
        \\  "input": "workload.regular",
        \\  "output": "status.accepted",
        \\  "events": [
        \\    {"tag": "send", "actor": 0, "peer": 1, "message_id": 10},
        \\    {"tag": "deliver", "actor": 1, "peer": 0, "message_id": 10}
        \\  ]
        \\}
    ;

    try std.testing.expectError(
        error.ExtractionEventBoundExceeded,
        importSmokeReplayFixture(
            std.testing.allocator,
            json,
            bound,
            "test/nanochat/test_gateway_topology_integration.py",
            "gateway.training",
        ),
    );
}
