const std = @import("std");

const shrink = @import("shrink.zig");
const topology = @import("topology.zig");
const trace = @import("trace.zig");

pub const max_generated_trace_depth = 10;

pub const BoundedProcess = struct {
    id: trace.NodeId,
    color: topology.Color,
};

pub const BoundedProcessSet = struct {
    processes: []const BoundedProcess,
    max_processes: usize = 64,

    pub fn validate(self: BoundedProcessSet) !void {
        if (self.processes.len > self.max_processes) return error.ProcessBoundExceeded;
        if (self.processes.len > 64) return error.ProcessBoundExceeded;

        for (self.processes, 0..) |left, left_index| {
            for (self.processes[left_index + 1 ..]) |right| {
                if (left.id == right.id) return error.DuplicateProcessId;
                if (topology.Color.eql(left.color, right.color)) return error.DuplicateProcessColor;
            }
        }
    }

    pub fn colorForNode(self: BoundedProcessSet, id: trace.NodeId) ?topology.Color {
        for (self.processes) |process| {
            if (process.id == id) return process.color;
        }
        return null;
    }

    pub fn containsNode(self: BoundedProcessSet, id: trace.NodeId) bool {
        return self.colorForNode(id) != null;
    }
};

pub const EventAlphabet = struct {
    events: []const trace.Event,

    pub fn validate(self: EventAlphabet, processes: BoundedProcessSet) !void {
        try processes.validate();
        for (self.events) |event| {
            if (!processes.containsNode(event.actor)) return error.EventActorOutsideProcessSet;
            if (event.peerOrNull()) |peer| {
                if (!processes.containsNode(peer)) return error.EventPeerOutsideProcessSet;
            }
        }
    }

    pub fn containsTag(self: EventAlphabet, tag: trace.EventTag) bool {
        for (self.events) |event| {
            if (event.tag == tag) return true;
        }
        return false;
    }
};

pub const GenerationOptions = struct {
    seed: trace.Seed = 0,
    depth: usize = 0,
    partial_order_reduction: bool = true,
    require_nonempty_inputs: bool = true,
};

pub const RandomGenerationOptions = struct {
    seed: trace.Seed,
    depth: usize,
    trace_count: usize,
    require_nonempty_inputs: bool = true,
};

pub const ProtocolTraceWitness = struct {
    input_index: usize,
    protocol_index: usize,
    trace_index: usize,
    event_count: usize,
    state_hash: u64,
};

pub const OwnedGeneratedProtocolComplex = struct {
    allocator: std.mem.Allocator,
    input: topology.Complex,
    protocol: topology.OwnedComplex,
    execution: topology.OwnedSimplicialCarrierMap,
    traces: []trace.OwnedTrace,
    witnesses: []ProtocolTraceWitness,

    pub fn protocolComplex(self: *const OwnedGeneratedProtocolComplex) topology.Complex {
        return self.protocol.complex();
    }

    pub fn executionCarrier(self: *const OwnedGeneratedProtocolComplex) topology.SimplicialCarrierMap {
        return self.execution.carrierMap();
    }

    pub fn deinit(self: *OwnedGeneratedProtocolComplex) void {
        for (self.traces) |*trace_value| {
            trace_value.deinit();
        }
        self.allocator.free(self.traces);
        self.allocator.free(self.witnesses);
        self.execution.deinit();
        self.protocol.deinit();
        self.traces = &.{};
        self.witnesses = &.{};
    }
};

const GenerationRecord = struct {
    input_index: usize,
    trace_value: trace.OwnedTrace,
    simplex: topology.OwnedSimplex,
    state_hash: u64,
};

pub fn generateExhaustiveProtocolComplex(
    allocator: std.mem.Allocator,
    processes: BoundedProcessSet,
    input: topology.Complex,
    alphabet: EventAlphabet,
    options: GenerationOptions,
    context: anytype,
    comptime runTrace: anytype,
    comptime extractLocalViews: anytype,
) !OwnedGeneratedProtocolComplex {
    if (options.depth > max_generated_trace_depth) return error.DepthTooLarge;
    try input.validate();
    try alphabet.validate(processes);

    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    var records = std.ArrayList(GenerationRecord).init(allocator);
    defer deinitRecords(&records);

    var protocol_builder = topology.ComplexBuilder.init(allocator);
    errdefer protocol_builder.deinit();

    const buffer = try allocator.alloc(trace.Event, options.depth);
    defer allocator.free(buffer);

    for (input_faces.simplexes, 0..) |input_simplex, input_index| {
        const before_count = records.items.len;
        const supported = try supportedEvents(allocator, processes, input_simplex, alphabet.events);
        defer allocator.free(supported);

        if (options.depth == 0) {
            try addGeneratedTrace(
                allocator,
                options.seed,
                input_index,
                input_simplex,
                buffer[0..0],
                &records,
                &protocol_builder,
                context,
                runTrace,
                extractLocalViews,
            );
        } else if (supported.len != 0) {
            try exploreExhaustive(
                allocator,
                options,
                input_index,
                input_simplex,
                supported,
                buffer,
                0,
                &records,
                &protocol_builder,
                context,
                runTrace,
                extractLocalViews,
            );
        }

        if (options.require_nonempty_inputs and
            !input_simplex.isEmpty() and
            records.items.len == before_count)
        {
            return error.EmptyExecutionCarrier;
        }
    }

    return finishGeneration(allocator, input, input_faces.complex(), &records, &protocol_builder);
}

pub fn generateSeededRandomProtocolComplex(
    allocator: std.mem.Allocator,
    processes: BoundedProcessSet,
    input: topology.Complex,
    alphabet: EventAlphabet,
    options: RandomGenerationOptions,
    context: anytype,
    comptime runTrace: anytype,
    comptime extractLocalViews: anytype,
) !OwnedGeneratedProtocolComplex {
    if (options.depth > max_generated_trace_depth) return error.DepthTooLarge;
    try input.validate();
    try alphabet.validate(processes);

    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    var records = std.ArrayList(GenerationRecord).init(allocator);
    defer deinitRecords(&records);

    var protocol_builder = topology.ComplexBuilder.init(allocator);
    errdefer protocol_builder.deinit();

    const buffer = try allocator.alloc(trace.Event, options.depth);
    defer allocator.free(buffer);

    var rng = std.rand.DefaultPrng.init(options.seed);

    for (input_faces.simplexes, 0..) |input_simplex, input_index| {
        const before_count = records.items.len;
        const supported = try supportedEvents(allocator, processes, input_simplex, alphabet.events);
        defer allocator.free(supported);

        if (options.depth != 0 and supported.len == 0) {
            if (options.require_nonempty_inputs and !input_simplex.isEmpty()) {
                return error.EmptyExecutionCarrier;
            }
            continue;
        }

        var trace_index: usize = 0;
        while (trace_index < options.trace_count) : (trace_index += 1) {
            for (buffer) |*event| {
                event.* = supported[rng.random().uintLessThan(usize, supported.len)];
            }
            try addGeneratedTrace(
                allocator,
                mixSeed(options.seed, input_index, trace_index),
                input_index,
                input_simplex,
                buffer,
                &records,
                &protocol_builder,
                context,
                runTrace,
                extractLocalViews,
            );
        }

        if (options.require_nonempty_inputs and
            !input_simplex.isEmpty() and
            records.items.len == before_count)
        {
            return error.EmptyExecutionCarrier;
        }
    }

    return finishGeneration(allocator, input, input_faces.complex(), &records, &protocol_builder);
}

pub fn eventsIndependent(left: trace.Event, right: trace.Event) bool {
    if (left.message_id != 0 and left.message_id == right.message_id) return false;
    if (eventTouchesNode(left, right.actor)) return false;
    if (right.peerOrNull()) |peer| {
        if (eventTouchesNode(left, peer)) return false;
    }
    return true;
}

pub fn partialOrderAllows(prefix: []const trace.Event, candidate: trace.Event) bool {
    if (prefix.len == 0) return true;
    const previous = prefix[prefix.len - 1];
    if (!eventsIndependent(previous, candidate)) return true;
    return eventOrder(previous) <= eventOrder(candidate);
}

pub fn shrinkTraceToSameProtocolSimplex(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    input_simplex: topology.Simplex,
    target_simplex: topology.Simplex,
    events: []const trace.Event,
    context: anytype,
    comptime runTrace: anytype,
    comptime extractLocalViews: anytype,
) !shrink.OwnedEventShrink {
    if (!(try traceEndsAtSimplex(
        allocator,
        seed,
        input_simplex,
        target_simplex,
        events,
        context,
        runTrace,
        extractLocalViews,
    ))) {
        return error.PredicateDoesNotHold;
    }

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

            if (try traceEndsAtSimplex(
                allocator,
                seed,
                input_simplex,
                target_simplex,
                candidate,
                context,
                runTrace,
                extractLocalViews,
            )) {
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
        .seed = seed,
        .events = current,
        .original_event_count = events.len,
    };
}

fn exploreExhaustive(
    allocator: std.mem.Allocator,
    options: GenerationOptions,
    input_index: usize,
    input_simplex: topology.Simplex,
    alphabet: []const trace.Event,
    buffer: []trace.Event,
    cursor: usize,
    records: *std.ArrayList(GenerationRecord),
    protocol_builder: *topology.ComplexBuilder,
    context: anytype,
    comptime runTrace: anytype,
    comptime extractLocalViews: anytype,
) !void {
    if (cursor == options.depth) {
        try addGeneratedTrace(
            allocator,
            options.seed,
            input_index,
            input_simplex,
            buffer[0..options.depth],
            records,
            protocol_builder,
            context,
            runTrace,
            extractLocalViews,
        );
        return;
    }

    for (alphabet) |event| {
        if (options.partial_order_reduction and !partialOrderAllows(buffer[0..cursor], event)) continue;
        buffer[cursor] = event;
        try exploreExhaustive(
            allocator,
            options,
            input_index,
            input_simplex,
            alphabet,
            buffer,
            cursor + 1,
            records,
            protocol_builder,
            context,
            runTrace,
            extractLocalViews,
        );
    }
}

fn addGeneratedTrace(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    input_index: usize,
    input_simplex: topology.Simplex,
    events: []const trace.Event,
    records: *std.ArrayList(GenerationRecord),
    protocol_builder: *topology.ComplexBuilder,
    context: anytype,
    comptime runTrace: anytype,
    comptime extractLocalViews: anytype,
) !void {
    var trace_value = try runTrace(context, allocator, seed, input_simplex, events);
    errdefer trace_value.deinit();

    var final_simplex = try extractLocalViews(context, allocator, input_simplex, &trace_value);
    errdefer final_simplex.deinit();

    try final_simplex.simplex().validate();
    try protocol_builder.add(final_simplex.simplex());
    try records.append(.{
        .input_index = input_index,
        .trace_value = trace_value,
        .simplex = final_simplex,
        .state_hash = trace_value.final_snapshot.hashValue(),
    });
}

fn finishGeneration(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    input_faces: topology.Complex,
    records: *std.ArrayList(GenerationRecord),
    protocol_builder: *topology.ComplexBuilder,
) !OwnedGeneratedProtocolComplex {
    var protocol_owned = try protocol_builder.toOwnedComplex();
    errdefer protocol_owned.deinit();

    const protocol_complex = protocol_owned.complex();
    var carrier_builder = topology.SimplicialCarrierMapBuilder.init(allocator, input, protocol_complex);
    errdefer carrier_builder.deinit();

    for (input_faces.simplexes, 0..) |input_simplex, input_index| {
        var image_builder = topology.ComplexBuilder.init(allocator);
        defer image_builder.deinit();

        for (records.items) |record| {
            if (record.input_index == input_index) try image_builder.add(record.simplex.simplex());
        }

        var image_owned = try image_builder.toOwnedComplex();
        defer image_owned.deinit();

        try carrier_builder.add(input_simplex, .{
            .parent = protocol_complex,
            .simplexes = image_owned.simplexes,
        });
    }

    var execution_owned = try carrier_builder.toOwned();
    errdefer execution_owned.deinit();
    try execution_owned.carrierMap().validate(allocator, .{});

    const traces = try allocator.alloc(trace.OwnedTrace, records.items.len);
    errdefer allocator.free(traces);
    const witnesses = try allocator.alloc(ProtocolTraceWitness, records.items.len);
    errdefer allocator.free(witnesses);

    var initialized_traces: usize = 0;
    errdefer {
        for (traces[0..initialized_traces]) |*trace_value| {
            trace_value.deinit();
        }
    }

    for (records.items, 0..) |*record, index| {
        const protocol_index = protocol_complex.findSimplexIndex(record.simplex.simplex()) orelse {
            return error.GeneratedSimplexMissingFromProtocolComplex;
        };
        traces[index] = record.trace_value;
        initialized_traces += 1;
        record.trace_value.events = &.{};
        witnesses[index] = .{
            .input_index = record.input_index,
            .protocol_index = protocol_index,
            .trace_index = index,
            .event_count = traces[index].events.len,
            .state_hash = record.state_hash,
        };
    }

    return .{
        .allocator = allocator,
        .input = input,
        .protocol = protocol_owned,
        .execution = execution_owned,
        .traces = traces,
        .witnesses = witnesses,
    };
}

fn supportedEvents(
    allocator: std.mem.Allocator,
    processes: BoundedProcessSet,
    input_simplex: topology.Simplex,
    events: []const trace.Event,
) ![]trace.Event {
    var supported = std.ArrayList(trace.Event).init(allocator);
    errdefer supported.deinit();

    for (events) |event| {
        if (try inputSupportsEvent(processes, input_simplex, event)) {
            try supported.append(event);
        }
    }

    return supported.toOwnedSlice();
}

fn inputSupportsEvent(
    processes: BoundedProcessSet,
    input_simplex: topology.Simplex,
    event: trace.Event,
) !bool {
    const actor_color = processes.colorForNode(event.actor) orelse return error.EventActorOutsideProcessSet;
    if (input_simplex.vertexWithColor(actor_color) == null) return false;

    if (event.peerOrNull()) |peer| {
        const peer_color = processes.colorForNode(peer) orelse return error.EventPeerOutsideProcessSet;
        if (input_simplex.vertexWithColor(peer_color) == null) return false;
    }

    return true;
}

fn eventTouchesNode(event: trace.Event, node_id: trace.NodeId) bool {
    if (event.actor == node_id) return true;
    if (event.peerOrNull()) |peer| {
        if (peer == node_id) return true;
    }
    return false;
}

fn eventOrder(event: trace.Event) u64 {
    var order: u64 = @intFromEnum(event.tag);
    order = order * 257 + event.actor;
    order = order * 257 + event.peer;
    order = order * 4_294_967_291 + event.message_id;
    return order;
}

fn mixSeed(seed: trace.Seed, input_index: usize, trace_index: usize) trace.Seed {
    var mixed = seed ^ (@as(u64, input_index) *% 0x9e37_79b9_7f4a_7c15);
    mixed ^= @as(u64, trace_index) *% 0xbf58_476d_1ce4_e5b9;
    mixed ^= mixed >> 33;
    mixed *%= 0xff51_afd7_ed55_8ccd;
    mixed ^= mixed >> 33;
    return mixed;
}

fn traceEndsAtSimplex(
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    input_simplex: topology.Simplex,
    target_simplex: topology.Simplex,
    events: []const trace.Event,
    context: anytype,
    comptime runTrace: anytype,
    comptime extractLocalViews: anytype,
) !bool {
    var trace_value = try runTrace(context, allocator, seed, input_simplex, events);
    defer trace_value.deinit();

    var simplex = try extractLocalViews(context, allocator, input_simplex, &trace_value);
    defer simplex.deinit();

    return topology.Simplex.eql(simplex.simplex(), target_simplex);
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

fn deinitRecords(records: *std.ArrayList(GenerationRecord)) void {
    for (records.items) |*record| {
        record.trace_value.deinit();
        record.simplex.deinit();
    }
    records.deinit();
}

const all_milestone_event_tags = [_]trace.EventTag{
    .send,
    .deliver,
    .drop,
    .duplicate,
    .timeout,
    .crash,
    .restart,
    .join,
    .leave,
    .cancel,
    .heartbeat,
    .local_step_complete,
    .lease_reserve,
    .lease_commit,
    .lease_release,
    .lease_expire,
};

fn toyRunTrace(
    _: void,
    allocator: std.mem.Allocator,
    seed: trace.Seed,
    input_simplex: topology.Simplex,
    events: []const trace.Event,
) !trace.OwnedTrace {
    var recorder = trace.Recorder.init(allocator, seed);
    defer recorder.deinit();

    var local_steps: u32 = 0;
    var checksum = seed ^ @as(u64, input_simplex.vertices.len);
    for (events) |event| {
        try recorder.record(event);
        if (event.tag == .local_step_complete) local_steps += 1;
        checksum ^= event.hashValue();
        checksum *%= 0x1000_0000_01b3;
    }

    recorder.finish(.{
        .steps = @intCast(events.len),
        .node_count = @intCast(input_simplex.vertices.len),
        .online_mask = if (input_simplex.vertices.len == 0) 0 else (@as(u64, 1) << @intCast(input_simplex.vertices.len)) - 1,
        .delivered_messages = local_steps,
        .checksum = checksum,
    });
    return recorder.toOwnedTrace();
}

fn toyExtractLocalViews(
    _: void,
    allocator: std.mem.Allocator,
    input_simplex: topology.Simplex,
    trace_value: *const trace.OwnedTrace,
) !topology.OwnedSimplex {
    const vertices = try allocator.alloc(topology.Vertex, input_simplex.vertices.len);
    errdefer allocator.free(vertices);

    for (input_simplex.vertices, 0..) |input_vertex, index| {
        vertices[index] = .{
            .color = input_vertex.color,
            .label = .{ .protocol = @as(i64, trace_value.final_snapshot.delivered_messages) },
        };
    }
    return .{ .allocator = allocator, .vertices = vertices };
}

test "event alphabet covers milestone execution vocabulary" {
    const process = [_]BoundedProcess{.{ .id = 0, .color = .{ .worker = 0 } }};
    const events = [_]trace.Event{
        trace.Event.init(.send, 0, 0, 1),
        trace.Event.init(.deliver, 0, 0, 1),
        trace.Event.init(.drop, 0, null, 1),
        trace.Event.init(.duplicate, 0, null, 1),
        trace.Event.init(.timeout, 0, null, 0),
        trace.Event.init(.crash, 0, null, 0),
        trace.Event.init(.restart, 0, null, 0),
        trace.Event.init(.join, 0, null, 0),
        trace.Event.init(.leave, 0, null, 0),
        trace.Event.init(.cancel, 0, null, 0),
        trace.Event.init(.heartbeat, 0, null, 0),
        trace.Event.init(.local_step_complete, 0, null, 0),
        trace.Event.init(.lease_reserve, 0, null, 10),
        trace.Event.init(.lease_commit, 0, null, 10),
        trace.Event.init(.lease_release, 0, null, 10),
        trace.Event.init(.lease_expire, 0, null, 10),
    };
    const alphabet = EventAlphabet{ .events = &events };

    try alphabet.validate(.{ .processes = &process });
    for (all_milestone_event_tags) |tag| {
        try std.testing.expect(alphabet.containsTag(tag));
    }
}

test "exhaustive generation builds protocol complex execution carrier and witnesses" {
    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const p1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const input_vertices = [_]topology.Vertex{ p0, p1 };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_vertices }} };
    const processes = [_]BoundedProcess{
        .{ .id = 0, .color = .{ .process = 0 } },
        .{ .id = 1, .color = .{ .process = 1 } },
    };
    const events = [_]trace.Event{
        trace.Event.init(.local_step_complete, 0, null, 0),
        trace.Event.init(.local_step_complete, 1, null, 0),
    };

    var generated = try generateExhaustiveProtocolComplex(
        std.testing.allocator,
        .{ .processes = &processes },
        input,
        .{ .events = &events },
        .{ .seed = 7, .depth = 1 },
        {},
        toyRunTrace,
        toyExtractLocalViews,
    );
    defer generated.deinit();

    try generated.protocolComplex().validate();
    try generated.executionCarrier().validate(std.testing.allocator, .{});
    try std.testing.expectEqual(@as(usize, 4), generated.witnesses.len);
    try std.testing.expect(generated.protocolComplex().containsSimplex(.{
        .vertices = &[_]topology.Vertex{
            .{ .color = .{ .process = 0 }, .label = .{ .protocol = 1 } },
            .{ .color = .{ .process = 1 }, .label = .{ .protocol = 1 } },
        },
    }));
}

test "partial-order reduction skips swapped independent schedules" {
    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const p1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const input_vertices = [_]topology.Vertex{ p0, p1 };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_vertices }} };
    const processes = [_]BoundedProcess{
        .{ .id = 0, .color = .{ .process = 0 } },
        .{ .id = 1, .color = .{ .process = 1 } },
    };
    const events = [_]trace.Event{
        trace.Event.init(.local_step_complete, 0, null, 0),
        trace.Event.init(.local_step_complete, 1, null, 0),
    };

    try std.testing.expect(eventsIndependent(events[0], events[1]));
    try std.testing.expect(partialOrderAllows(&[_]trace.Event{events[0]}, events[1]));
    try std.testing.expect(!partialOrderAllows(&[_]trace.Event{events[1]}, events[0]));

    var generated = try generateExhaustiveProtocolComplex(
        std.testing.allocator,
        .{ .processes = &processes },
        input,
        .{ .events = &events },
        .{ .seed = 7, .depth = 2, .partial_order_reduction = true },
        {},
        toyRunTrace,
        toyExtractLocalViews,
    );
    defer generated.deinit();

    try std.testing.expectEqual(@as(usize, 5), generated.witnesses.len);
}

test "seeded random generation is deterministic" {
    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const input_vertices = [_]topology.Vertex{p0};
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_vertices }} };
    const processes = [_]BoundedProcess{.{ .id = 0, .color = .{ .process = 0 } }};
    const events = [_]trace.Event{
        trace.Event.init(.heartbeat, 0, null, 0),
        trace.Event.init(.local_step_complete, 0, null, 0),
    };

    var left = try generateSeededRandomProtocolComplex(
        std.testing.allocator,
        .{ .processes = &processes },
        input,
        .{ .events = &events },
        .{ .seed = 11, .depth = 3, .trace_count = 4 },
        {},
        toyRunTrace,
        toyExtractLocalViews,
    );
    defer left.deinit();

    var right = try generateSeededRandomProtocolComplex(
        std.testing.allocator,
        .{ .processes = &processes },
        input,
        .{ .events = &events },
        .{ .seed = 11, .depth = 3, .trace_count = 4 },
        {},
        toyRunTrace,
        toyExtractLocalViews,
    );
    defer right.deinit();

    try std.testing.expectEqual(left.witnesses.len, right.witnesses.len);
    for (left.witnesses, right.witnesses) |a, b| {
        try std.testing.expectEqual(a.state_hash, b.state_hash);
        try std.testing.expectEqual(a.event_count, b.event_count);
    }
}

test "trace shrinking preserves the same generated protocol simplex" {
    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const input_vertices = [_]topology.Vertex{p0};
    const input_simplex = topology.Simplex{ .vertices = &input_vertices };
    const events = [_]trace.Event{
        trace.Event.init(.heartbeat, 0, null, 0),
        trace.Event.init(.local_step_complete, 0, null, 0),
    };
    const target_vertices = [_]topology.Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 1 } },
    };

    var shrunk = try shrinkTraceToSameProtocolSimplex(
        std.testing.allocator,
        5,
        input_simplex,
        .{ .vertices = &target_vertices },
        &events,
        {},
        toyRunTrace,
        toyExtractLocalViews,
    );
    defer shrunk.deinit();

    try std.testing.expectEqual(@as(usize, 1), shrunk.events.len);
    try std.testing.expectEqual(trace.EventTag.local_step_complete, shrunk.events[0].tag);
    try std.testing.expectEqual(@as(usize, 1), shrunk.deletedEvents());
}
