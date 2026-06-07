const std = @import("std");

const coverage = @import("coverage.zig");
const tasks = @import("tasks.zig");
const topology = @import("topology.zig");
const trace = @import("trace.zig");

pub const OwnedCarrierEscape = struct {
    allocator: std.mem.Allocator,
    input_simplex: topology.OwnedSimplex,
    offending_output: topology.OwnedSimplex,
    decided_image: topology.OwnedComplex,
    allowed_image: topology.OwnedComplex,

    pub fn deinit(self: *OwnedCarrierEscape) void {
        self.allowed_image.deinit();
        self.decided_image.deinit();
        self.offending_output.deinit();
        self.input_simplex.deinit();
        self.* = undefined;
    }
};

pub const AnalysisReport = struct {
    title: []const u8 = "pcp topology counterexample",
    violated_rule: []const u8,
    input: topology.Complex,
    protocol: topology.Complex,
    output: topology.Complex,
    execution: topology.SimplicialCarrierMap,
    task: topology.SimplicialCarrierMap,
    decision: topology.VertexMap,
    witness: ?*const OwnedCarrierEscape = null,
    seed: trace.Seed = 0,
    events: []const trace.Event = &.{},
    excerpt_limit: usize = 8,

    pub fn writeHuman(self: AnalysisReport, allocator: std.mem.Allocator, writer: anytype) !void {
        try writer.print("{s}\n", .{self.title});
        try writer.print("violated_rule: {s}\n", .{self.violated_rule});
        try writer.print("seed: {d}\n", .{self.seed});
        try writer.print("events: {d}\n\n", .{self.events.len});

        try writeComplexExcerpt(allocator, writer, "I", self.input, self.excerpt_limit);
        try writeComplexExcerpt(allocator, writer, "P", self.protocol, self.excerpt_limit);
        try writeComplexExcerpt(allocator, writer, "O", self.output, self.excerpt_limit);
        try writeCarrierExcerpt(allocator, writer, "Xi", self.execution, self.excerpt_limit);
        try writeCarrierExcerpt(allocator, writer, "Delta", self.task, self.excerpt_limit);
        try writeDecisionMapExcerpt(allocator, writer, "delta", self.decision, self.excerpt_limit);

        if (self.witness) |witness| {
            try writer.writeAll("minimal_offending_simplex:\n");
            try writer.writeAll("  input: ");
            try writeSimplex(writer, witness.input_simplex.simplex());
            try writer.writeByte('\n');
            try writer.writeAll("  offending_output: ");
            try writeSimplex(writer, witness.offending_output.simplex());
            try writer.writeByte('\n');
            try writeComplexExcerpt(allocator, writer, "  decided_image", witness.decided_image.complex(), self.excerpt_limit);
            try writeComplexExcerpt(allocator, writer, "  allowed_image", witness.allowed_image.complex(), self.excerpt_limit);
        }

        if (self.events.len != 0) {
            try writer.writeAll("replay_events:\n");
            for (self.events, 0..) |event, index| {
                try writer.writeAll("  ");
                try event.writeText(writer, index);
            }
        }
    }

    pub fn writeReplayFixture(self: AnalysisReport, allocator: std.mem.Allocator, writer: anytype) !void {
        try writer.writeAll("{\"schema\":\"pcp.topology.analysis.v1\"");
        try writer.writeAll(",\"title\":");
        try std.json.stringify(self.title, .{}, writer);
        try writer.writeAll(",\"violated_rule\":");
        try std.json.stringify(self.violated_rule, .{}, writer);
        try writer.print(",\"seed\":{d}", .{self.seed});
        try writer.writeAll(",\"counts\":{");
        try writer.print("\"input_simplexes\":{d}", .{(try faceCount(allocator, self.input))});
        try writer.print(",\"protocol_simplexes\":{d}", .{(try faceCount(allocator, self.protocol))});
        try writer.print(",\"output_simplexes\":{d}", .{(try faceCount(allocator, self.output))});
        try writer.writeAll("}");

        if (self.witness) |witness| {
            try writer.writeAll(",\"witness\":{");
            try writer.writeAll("\"input_simplex\":");
            try writeJsonSimplexString(allocator, writer, witness.input_simplex.simplex());
            try writer.writeAll(",\"offending_output\":");
            try writeJsonSimplexString(allocator, writer, witness.offending_output.simplex());
            try writer.writeAll("}");
        }

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

pub const ChangeKind = enum {
    unchanged,
    changed,
    added,
    removed,
};

pub const ArtifactChange = struct {
    name: []const u8,
    kind: ChangeKind,
    before_signature: ?u64 = null,
    after_signature: ?u64 = null,
};

pub const RegressionCase = struct {
    id: []const u8,
    task: tasks.TaskKind,
    violated_rule: []const u8,
    description: []const u8,
};

pub const regression_corpus = [_]RegressionCase{
    .{
        .id = "message.worker_result_to_worker",
        .task = .message_dispatch,
        .violated_rule = "message dispatch fails closed for worker-result messages",
        .description = "A queued worker-result message must not be dispatched as a controller-to-worker task.",
    },
    .{
        .id = "scheduling.cross_owner_capacity",
        .task = .gateway_lease_scheduling,
        .violated_rule = "reserved capacity is preserved across lease owners",
        .description = "Training admission must not consume capacity reserved for inference or RL owners.",
    },
    .{
        .id = "reservation.cancel_cleanup",
        .task = .reservation_lifecycle,
        .violated_rule = "cancellation removes queued job and leased worker state",
        .description = "Canceling a reserved queued job must release all associated leases.",
    },
    .{
        .id = "federation.stale_gateway",
        .task = .federation_placement,
        .violated_rule = "placements require current compatible gateways",
        .description = "A stale gateway cannot carry a placed output.",
    },
    .{
        .id = "diloco.duplicate_quorum",
        .task = .decoupled_fragment,
        .violated_rule = "duplicate fragment responses do not satisfy quorum",
        .description = "A duplicate sender contributes one learner view at most.",
    },
    .{
        .id = "rda.direct_average_escape",
        .task = .rda_merge,
        .violated_rule = "RDA output follows direction/norm carrier",
        .description = "RDA is not validated by coordinate convex-envelope checks alone.",
    },
    .{
        .id = "global.partial_loss_without_rollback",
        .task = .multi_gateway_global_job,
        .violated_rule = "partial placement loss requires rollback or deferred global state",
        .description = "A multi-gateway job cannot remain globally committed after one gateway route is lost.",
    },
};

pub fn findCarrierEscape(
    allocator: std.mem.Allocator,
    execution: topology.SimplicialCarrierMap,
    task: topology.SimplicialCarrierMap,
    decision: topology.VertexMap,
) !?OwnedCarrierEscape {
    try execution.validate(allocator, .{});
    try task.validate(allocator, .{});
    try decision.validate(allocator, .{ .require_color_preserving = true });

    var decided = try execution.composeWithMap(allocator, decision);
    defer decided.deinit();
    const decided_carrier = decided.carrierMap();

    var domain_faces = try execution.domain.faces(allocator);
    defer domain_faces.deinit();
    for (domain_faces.simplexes) |input_simplex| {
        const decided_image = decided_carrier.imageOf(input_simplex) orelse return error.MissingCarrierSimplex;
        const allowed_image = task.imageOf(input_simplex) orelse return error.MissingCarrierSimplex;

        var decided_faces = try decided_image.complex().faces(allocator);
        defer decided_faces.deinit();
        for (decided_faces.simplexes) |output_simplex| {
            if (allowed_image.complex().containsSimplex(output_simplex)) continue;
            return .{
                .allocator = allocator,
                .input_simplex = try input_simplex.canonicalize(allocator),
                .offending_output = try output_simplex.canonicalize(allocator),
                .decided_image = try cloneClosedComplex(allocator, decided_image.complex()),
                .allowed_image = try cloneClosedComplex(allocator, allowed_image.complex()),
            };
        }
    }

    return null;
}

pub fn writeComplexDot(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
    graph_name: []const u8,
    writer: anytype,
) !void {
    try complex.validate();
    var vertices = try complex.vertices(allocator);
    defer vertices.deinit();

    try writer.print("graph \"{s}\" {{\n", .{graph_name});
    for (vertices.vertices, 0..) |vertex, index| {
        try writer.print("  v{d} [label=\"", .{index});
        try writeVertex(writer, vertex);
        try writer.writeAll("\"];\n");
    }

    for (vertices.vertices, 0..) |left, left_index| {
        for (vertices.vertices[left_index + 1 ..], left_index + 1..) |right, right_index| {
            const edge_vertices = [_]topology.Vertex{ left, right };
            if (!complex.containsSimplex(.{ .vertices = &edge_vertices })) continue;
            try writer.print("  v{d} -- v{d};\n", .{ left_index, right_index });
        }
    }
    try writer.writeAll("}\n");
}

pub fn writeCiSummary(writer: anytype, changes: []const ArtifactChange) !void {
    try writer.print("pcp topology artifact summary: {d} item(s)\n", .{changes.len});
    for (changes) |change| {
        try writer.print("- {s}: {s}", .{ change.name, @tagName(change.kind) });
        if (change.before_signature) |signature| {
            try writer.print(" before={x}", .{signature});
        }
        if (change.after_signature) |signature| {
            try writer.print(" after={x}", .{signature});
        }
        try writer.writeByte('\n');
    }
}

pub fn writeCiSummaryWithCoverage(
    writer: anytype,
    changes: []const ArtifactChange,
    coverage_paths: []const coverage.CoveragePath,
) !void {
    try writeCiSummary(writer, changes);
    try coverage.writeCiSummary(writer, coverage_paths);
}

pub fn carrierSignature(
    allocator: std.mem.Allocator,
    carrier: topology.SimplicialCarrierMap,
) !u64 {
    try carrier.validate(allocator, .{});
    var hasher = std.hash.Wyhash.init(0x6372_7272_5f73_6967);

    var domain_faces = try carrier.domain.faces(allocator);
    defer domain_faces.deinit();
    for (domain_faces.simplexes) |input_simplex| {
        try updateHashWithSimplex(allocator, &hasher, input_simplex);
        const image = carrier.imageOf(input_simplex) orelse return error.MissingCarrierSimplex;
        var image_faces = try image.complex().faces(allocator);
        defer image_faces.deinit();
        for (image_faces.simplexes) |output_simplex| {
            try updateHashWithSimplex(allocator, &hasher, output_simplex);
        }
    }

    return hasher.final();
}

pub fn mapSignature(
    allocator: std.mem.Allocator,
    map: topology.VertexMap,
) !u64 {
    try map.validate(allocator, .{});
    var hasher = std.hash.Wyhash.init(0x6d61_705f_7369_67);

    var vertices = try map.domain.vertices(allocator);
    defer vertices.deinit();
    for (vertices.vertices) |source| {
        try updateHashWithVertex(&hasher, source);
        const target = map.targetForVertex(source) orelse return error.MissingVertexMapEntry;
        try updateHashWithVertex(&hasher, target);
    }

    return hasher.final();
}

pub fn writeComplexExcerpt(
    allocator: std.mem.Allocator,
    writer: anytype,
    name: []const u8,
    complex: topology.Complex,
    limit: usize,
) !void {
    var faces = try complex.faces(allocator);
    defer faces.deinit();

    try writer.print("{s}: simplexes={d} dimension={d}\n", .{ name, faces.simplexes.len, complex.dimension() });
    const count = @min(limit, faces.simplexes.len);
    for (faces.simplexes[0..count], 0..) |simplex, index| {
        try writer.print("  {d}: ", .{index});
        try writeSimplex(writer, simplex);
        try writer.writeByte('\n');
    }
    if (count < faces.simplexes.len) {
        try writer.print("  ... {d} more\n", .{faces.simplexes.len - count});
    }
}

pub fn writeCarrierExcerpt(
    allocator: std.mem.Allocator,
    writer: anytype,
    name: []const u8,
    carrier: topology.SimplicialCarrierMap,
    limit: usize,
) !void {
    try carrier.validate(allocator, .{});
    try writer.print("{s}: carrier_entries={d}\n", .{ name, carrier.entries.len });
    const count = @min(limit, carrier.entries.len);
    for (carrier.entries[0..count], 0..) |entry, index| {
        try writer.print("  {d}: ", .{index});
        try writeSimplex(writer, entry.simplex);
        try writer.writeAll(" -> ");
        try writeSubcomplexInline(writer, entry.image);
        try writer.writeByte('\n');
    }
    if (count < carrier.entries.len) {
        try writer.print("  ... {d} more\n", .{carrier.entries.len - count});
    }
}

pub fn writeDecisionMapExcerpt(
    allocator: std.mem.Allocator,
    writer: anytype,
    name: []const u8,
    map: topology.VertexMap,
    limit: usize,
) !void {
    try map.validate(allocator, .{});
    try writer.print("{s}: vertex_map_entries={d}\n", .{ name, map.entries.len });
    const count = @min(limit, map.entries.len);
    for (map.entries[0..count], 0..) |entry, index| {
        try writer.print("  {d}: ", .{index});
        try writeVertex(writer, entry.source);
        try writer.writeAll(" -> ");
        try writeVertex(writer, entry.target);
        try writer.writeByte('\n');
    }
    if (count < map.entries.len) {
        try writer.print("  ... {d} more\n", .{map.entries.len - count});
    }
}

pub fn writeSimplex(writer: anytype, simplex: topology.Simplex) !void {
    try writer.writeByte('{');
    for (simplex.vertices, 0..) |vertex, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeVertex(writer, vertex);
    }
    try writer.writeByte('}');
}

fn writeSubcomplexInline(writer: anytype, subcomplex: topology.Subcomplex) !void {
    try writer.writeByte('[');
    for (subcomplex.simplexes, 0..) |simplex, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeSimplex(writer, simplex);
    }
    try writer.writeByte(']');
}

fn writeVertex(writer: anytype, vertex: topology.Vertex) !void {
    try writeColor(writer, vertex.color);
    try writer.writeByte(':');
    try writeLabel(writer, vertex.label);
}

fn writeColor(writer: anytype, color: topology.Color) !void {
    switch (color) {
        .hub => |value| try writer.print("hub.{d}", .{value}),
        .gateway => |value| try writer.print("gateway.{d}", .{value}),
        .worker => |value| try writer.print("worker.{d}", .{value}),
        .api_caller => |value| try writer.print("api_caller.{d}", .{value}),
        .job => |value| try writer.print("job.{d}", .{value}),
        .session => |value| try writer.print("session.{d}", .{value}),
        .process => |value| try writer.print("process.{d}", .{value}),
    }
}

fn writeLabel(writer: anytype, label: topology.Label) !void {
    switch (label) {
        .empty => try writer.writeAll("empty"),
        .input => |value| try writer.print("input.{d}", .{value}),
        .protocol => |value| try writer.print("protocol.{d}", .{value}),
        .output => |value| try writer.print("output.{d}", .{value}),
        .symbol => |value| try writer.writeAll(value),
    }
}

fn writeJsonSimplexString(
    allocator: std.mem.Allocator,
    writer: anytype,
    simplex: topology.Simplex,
) !void {
    var rendered = std.ArrayList(u8).init(allocator);
    defer rendered.deinit();
    try writeSimplex(rendered.writer(), simplex);
    try std.json.stringify(rendered.items, .{}, writer);
}

fn cloneClosedComplex(allocator: std.mem.Allocator, complex: topology.Complex) !topology.OwnedComplex {
    var faces = try complex.faces(allocator);
    defer faces.deinit();
    var builder = topology.ComplexBuilder.init(allocator);
    errdefer builder.deinit();
    for (faces.simplexes) |simplex| try builder.add(simplex);
    return builder.toOwnedComplex();
}

fn faceCount(allocator: std.mem.Allocator, complex: topology.Complex) !usize {
    var faces = try complex.faces(allocator);
    defer faces.deinit();
    return faces.simplexes.len;
}

fn updateHashWithSimplex(
    allocator: std.mem.Allocator,
    hasher: *std.hash.Wyhash,
    simplex: topology.Simplex,
) !void {
    var canonical = try simplex.canonicalize(allocator);
    defer canonical.deinit();
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, canonical.vertices.len, .little);
    hasher.update(&buf);
    for (canonical.vertices) |vertex| {
        try updateHashWithVertex(hasher, vertex);
    }
}

fn updateHashWithVertex(hasher: *std.hash.Wyhash, vertex: topology.Vertex) !void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, vertex.color.hash(), .little);
    hasher.update(&buf);
    std.mem.writeInt(u64, &buf, vertex.label.hash(), .little);
    hasher.update(&buf);
}

test "carrier escape report includes book objects and replay fixture fields" {
    const in_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const protocol_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 0 } };
    const good_output = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 1 } };
    const bad_output = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 2 } };
    const empty_vertices = [_]topology.Vertex{};
    const input_vertices = [_]topology.Vertex{in_vertex};
    const protocol_vertices = [_]topology.Vertex{protocol_vertex};
    const good_vertices = [_]topology.Vertex{good_output};
    const bad_vertices = [_]topology.Vertex{bad_output};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_vertices }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_vertices }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &good_vertices },
        .{ .vertices = &bad_vertices },
    } };

    var execution_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, protocol);
    defer execution_builder.deinit();
    try execution_builder.add(empty, .{ .parent = protocol, .simplexes = &[_]topology.Simplex{empty} });
    try execution_builder.add(.{ .vertices = &input_vertices }, .{
        .parent = protocol,
        .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_vertices }},
    });
    var execution = try execution_builder.toOwned();
    defer execution.deinit();

    var task_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, output);
    defer task_builder.deinit();
    try task_builder.add(empty, .{ .parent = output, .simplexes = &[_]topology.Simplex{empty} });
    try task_builder.add(.{ .vertices = &input_vertices }, .{
        .parent = output,
        .simplexes = &[_]topology.Simplex{.{ .vertices = &good_vertices }},
    });
    var task = try task_builder.toOwned();
    defer task.deinit();

    const decision_entries = [_]topology.VertexMapEntry{.{
        .source = protocol_vertex,
        .target = bad_output,
    }};
    const decision = topology.VertexMap{
        .domain = protocol,
        .codomain = output,
        .entries = &decision_entries,
    };

    var witness = (try findCarrierEscape(
        std.testing.allocator,
        execution.carrierMap(),
        task.carrierMap(),
        decision,
    )).?;
    defer witness.deinit();

    const events = [_]trace.Event{trace.Event.init(.deliver, 1, 0, 77)};
    const report = AnalysisReport{
        .violated_rule = "decision outside Delta",
        .input = input,
        .protocol = protocol,
        .output = output,
        .execution = execution.carrierMap(),
        .task = task.carrierMap(),
        .decision = decision,
        .witness = &witness,
        .events = &events,
    };

    var human = std.ArrayList(u8).init(std.testing.allocator);
    defer human.deinit();
    try report.writeHuman(std.testing.allocator, human.writer());

    try std.testing.expect(std.mem.indexOf(u8, human.items, "I: simplexes=") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "P: simplexes=") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "O: simplexes=") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "Xi: carrier_entries=") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "Delta: carrier_entries=") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "delta: vertex_map_entries=") != null);
    try std.testing.expect(std.mem.indexOf(u8, human.items, "minimal_offending_simplex") != null);

    var fixture = std.ArrayList(u8).init(std.testing.allocator);
    defer fixture.deinit();
    try report.writeReplayFixture(std.testing.allocator, fixture.writer());

    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"schema\":\"pcp.topology.analysis.v1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"violated_rule\":\"decision outside Delta\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"witness\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, fixture.items, "\"tag\":\"deliver\"") != null);
}

test "carrier escape finder returns null for carried decision maps" {
    const in_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const protocol_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 0 } };
    const output_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 1 } };
    const empty_vertices = [_]topology.Vertex{};
    const input_vertices = [_]topology.Vertex{in_vertex};
    const protocol_vertices = [_]topology.Vertex{protocol_vertex};
    const output_vertices = [_]topology.Vertex{output_vertex};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_vertices }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_vertices }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &output_vertices }} };

    var execution_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, protocol);
    defer execution_builder.deinit();
    try execution_builder.add(empty, .{ .parent = protocol, .simplexes = &[_]topology.Simplex{empty} });
    try execution_builder.add(.{ .vertices = &input_vertices }, .{
        .parent = protocol,
        .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_vertices }},
    });
    var execution = try execution_builder.toOwned();
    defer execution.deinit();

    var task_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, output);
    defer task_builder.deinit();
    try task_builder.add(empty, .{ .parent = output, .simplexes = &[_]topology.Simplex{empty} });
    try task_builder.add(.{ .vertices = &input_vertices }, .{
        .parent = output,
        .simplexes = &[_]topology.Simplex{.{ .vertices = &output_vertices }},
    });
    var task = try task_builder.toOwned();
    defer task.deinit();

    const decision_entries = [_]topology.VertexMapEntry{.{
        .source = protocol_vertex,
        .target = output_vertex,
    }};
    const decision = topology.VertexMap{
        .domain = protocol,
        .codomain = output,
        .entries = &decision_entries,
    };

    const escape = try findCarrierEscape(std.testing.allocator, execution.carrierMap(), task.carrierMap(), decision);
    try std.testing.expect(escape == null);
}

test "DOT export renders the one skeleton of a complex" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const edge = [_]topology.Vertex{ a, b };
    const complex = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &edge }} };

    var rendered = std.ArrayList(u8).init(std.testing.allocator);
    defer rendered.deinit();
    try writeComplexDot(std.testing.allocator, complex, "interval", rendered.writer());

    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "graph \"interval\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "v0 [label=\"process.0:input.0\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "v0 -- v1") != null);
}

test "CI summary and signatures are stable review artifacts" {
    var task = try tasks.buildMessageDispatchTask(std.testing.allocator);
    defer task.deinit();
    const signature = try carrierSignature(std.testing.allocator, task.task().carrier);
    const change = ArtifactChange{
        .name = "message dispatch Delta",
        .kind = .changed,
        .before_signature = signature,
        .after_signature = signature ^ 0xff,
    };

    var rendered = std.ArrayList(u8).init(std.testing.allocator);
    defer rendered.deinit();
    try writeCiSummaryWithCoverage(rendered.writer(), &.{change}, &coverage.default_paths);

    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "pcp topology artifact summary: 1 item(s)") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "message dispatch Delta: changed") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "before=") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "after=") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "pcp topology coverage summary:") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "multi_gateway_global_job: paths=3 evidence=symbolic_carrier design_only=true") != null);
}

test "regression corpus names known invalid protocol designs" {
    try std.testing.expect(regression_corpus.len >= 7);

    var saw_rda = false;
    var saw_global = false;
    for (regression_corpus) |case| {
        try std.testing.expect(case.id.len > 0);
        try std.testing.expect(case.description.len > 0);
        if (case.task == .rda_merge) saw_rda = true;
        if (case.task == .multi_gateway_global_job) saw_global = true;
    }

    try std.testing.expect(saw_rda);
    try std.testing.expect(saw_global);
}
