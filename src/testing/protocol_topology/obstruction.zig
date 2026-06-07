const std = @import("std");

const decision_search = @import("decision_search.zig");
const invariants = @import("invariants.zig");
const reporting = @import("reporting.zig");
const topology = @import("topology.zig");

pub const PCPWitnessKind = enum {
    generic,
    partition,
    stale_gateway,
    duplicate_quorum,
    cancellation_hole,
    global_rollback_gap,
    lease_owner_conflict,
};

pub const FailedInvariant = enum {
    candidate_carrier,
    connectivity,
    betti_number,
    induced_homology,
    local_view_compatibility,
    decision_search,
    unsupported_dimension,
};

pub const ObstructionKind = enum {
    carrier_nonexistence,
    disconnected_task_output,
    hole_preservation_failure,
    hole_creation_failure,
    incompatible_local_view_identification,
    solver_unsat,
    unsupported_dimension,
};

pub const ObstructionOptions = struct {
    max_homology_dimension: usize = 2,
    prime: u16 = 2,
    max_assignments: usize = 100_000,
    max_unsat_core_vertices: usize = 10,
};

pub const MapInvariantContext = struct {
    input_face: topology.Simplex,
    protocol_carrier_image: topology.Complex,
    output_carrier_image: topology.Complex,
    map: topology.VertexMap,
    pcp_witness: PCPWitnessKind = .generic,
};

pub const OwnedObstructionReport = struct {
    allocator: std.mem.Allocator,
    kind: ObstructionKind,
    pcp_witness: PCPWitnessKind,
    failed_invariant: FailedInvariant,
    input_face: topology.OwnedSimplex,
    protocol_carrier_image: topology.OwnedComplex,
    output_carrier_image: topology.OwnedComplex,
    protocol_betti: []usize,
    output_betti: []usize,
    induced_ranks: []usize,
    unsat_reason: ?decision_search.UnsatReason = null,
    unsupported_dimension: ?isize = null,
    homology_witness: ?invariants.OwnedHomologyMapWitness = null,
    unsat_core_vertices: []topology.Vertex,

    pub fn deinit(self: *OwnedObstructionReport) void {
        if (self.homology_witness) |*witness| witness.deinit();
        self.allocator.free(self.unsat_core_vertices);
        self.allocator.free(self.induced_ranks);
        self.allocator.free(self.output_betti);
        self.allocator.free(self.protocol_betti);
        self.output_carrier_image.deinit();
        self.protocol_carrier_image.deinit();
        self.input_face.deinit();
        self.unsat_core_vertices = &.{};
        self.induced_ranks = &.{};
        self.output_betti = &.{};
        self.protocol_betti = &.{};
        self.homology_witness = null;
    }

    pub fn writeHuman(self: OwnedObstructionReport, allocator: std.mem.Allocator, writer: anytype) !void {
        try writer.writeAll("bounded topology obstruction\n");
        try writer.print("kind: {s}\n", .{@tagName(self.kind)});
        try writer.print("pcp_witness: {s}\n", .{@tagName(self.pcp_witness)});
        try writer.print("failed_invariant: {s}\n", .{@tagName(self.failed_invariant)});
        if (self.unsat_reason) |reason| {
            try writer.print("unsat_reason: {s}\n", .{@tagName(reason)});
        }
        if (self.unsupported_dimension) |dimension| {
            try writer.print("unsupported_dimension: {d}\n", .{dimension});
        }

        try writer.writeAll("input_face: ");
        try reporting.writeSimplex(writer, self.input_face.simplex());
        try writer.writeByte('\n');

        try writer.writeAll("protocol_betti: ");
        try writeUsizeList(writer, self.protocol_betti);
        try writer.writeByte('\n');
        try writer.writeAll("output_betti: ");
        try writeUsizeList(writer, self.output_betti);
        try writer.writeByte('\n');
        if (self.induced_ranks.len != 0) {
            try writer.writeAll("induced_ranks: ");
            try writeUsizeList(writer, self.induced_ranks);
            try writer.writeByte('\n');
        }

        if (self.unsat_core_vertices.len != 0) {
            try writer.writeAll("unsat_core_vertices:");
            for (self.unsat_core_vertices) |vertex| {
                try writer.writeByte(' ');
                try writeVertexInline(writer, vertex);
            }
            try writer.writeByte('\n');
        }

        if (self.homology_witness) |witness| {
            try writer.print("homology_witness: {s} dim={d} prime={d}\n", .{
                @tagName(witness.kind),
                witness.dimension,
                witness.prime,
            });
            if (witness.source_cycle) |cycle| {
                try writer.print("source_cycle_simplexes: {d}\n", .{cycle.simplexes.len});
            }
            if (witness.image_cycle) |cycle| {
                try writer.print("image_cycle_simplexes: {d}\n", .{cycle.simplexes.len});
            }
        }

        try reporting.writeComplexExcerpt(
            allocator,
            writer,
            "protocol_carrier_image",
            self.protocol_carrier_image.complex(),
            8,
        );
        try reporting.writeComplexExcerpt(
            allocator,
            writer,
            "output_carrier_image",
            self.output_carrier_image.complex(),
            8,
        );
    }
};

pub fn diagnoseBoundedDecisionProblem(
    allocator: std.mem.Allocator,
    problem: decision_search.DecisionSearchProblem,
    pcp_witness: PCPWitnessKind,
    options: ObstructionOptions,
) !?OwnedObstructionReport {
    try problem.validate(allocator);

    var search = try decision_search.enumerateDecisionMaps(allocator, problem, .{
        .max_solutions = 1,
        .max_assignments = options.max_assignments,
        .max_unsat_core_vertices = options.max_unsat_core_vertices,
    });
    defer search.deinit();

    if (search.status == .satisfiable) return null;

    if (search.unsat_core.reason == .empty_candidate_set) {
        var input_face = try selectInputFaceForCore(allocator, problem.execution, search.unsat_core.vertices);
        defer input_face.deinit();
        return try buildReportFromProblem(
            allocator,
            problem,
            pcp_witness,
            .carrier_nonexistence,
            .candidate_carrier,
            input_face.simplex(),
            options,
            search.unsat_core.reason,
            search.unsat_core.vertices,
            null,
            null,
        );
    }

    var connectivity_maybe = try invariants.findConnectivityObstruction(allocator, problem.execution, problem.task);
    if (connectivity_maybe) |*connectivity| {
        defer connectivity.deinit();
        return try buildReportFromProblem(
            allocator,
            problem,
            pcp_witness,
            .disconnected_task_output,
            .connectivity,
            connectivity.inputSimplex(),
            options,
            search.unsat_core.reason,
            search.unsat_core.vertices,
            null,
            null,
        );
    }

    var homology_failure_maybe = try findCarrierHomologyFailure(allocator, problem, options);
    if (homology_failure_maybe) |*failure| {
        defer failure.deinit();
        return try buildReportFromProblem(
            allocator,
            problem,
            pcp_witness,
            failure.kind,
            failure.failed_invariant,
            failure.input_face.simplex(),
            options,
            search.unsat_core.reason,
            search.unsat_core.vertices,
            failure.unsupported_dimension,
            null,
        );
    }

    var input_face = try selectInputFaceForCore(allocator, problem.execution, search.unsat_core.vertices);
    defer input_face.deinit();

    const fallback_kind: ObstructionKind = switch (search.unsat_core.reason) {
        .local_inconsistency => .incompatible_local_view_identification,
        else => .solver_unsat,
    };
    const fallback_invariant: FailedInvariant = switch (search.unsat_core.reason) {
        .local_inconsistency => .local_view_compatibility,
        else => .decision_search,
    };

    return try buildReportFromProblem(
        allocator,
        problem,
        pcp_witness,
        fallback_kind,
        fallback_invariant,
        input_face.simplex(),
        options,
        search.unsat_core.reason,
        search.unsat_core.vertices,
        null,
        null,
    );
}

pub fn diagnoseMapInducedInvariantFailure(
    allocator: std.mem.Allocator,
    context: MapInvariantContext,
    options: ObstructionOptions,
) !?OwnedObstructionReport {
    var homology = try invariants.checkBettiPreservingMap(
        allocator,
        context.map,
        options.max_homology_dimension,
        options.prime,
    );
    defer homology.deinit();

    if (homology.preserves_homology) return null;

    var witness = try invariants.findHomologyMapWitness(
        allocator,
        context.map,
        options.max_homology_dimension,
        options.prime,
    );
    errdefer if (witness) |*owned| owned.deinit();

    const kind: ObstructionKind = if (witness) |owned| switch (owned.kind) {
        .source_cycle_lost => .hole_preservation_failure,
        .image_cycle_created => .hole_creation_failure,
    } else .hole_preservation_failure;

    var report = try buildReportFromImages(
        allocator,
        context.pcp_witness,
        kind,
        .induced_homology,
        context.input_face,
        context.protocol_carrier_image,
        context.output_carrier_image,
        options,
        null,
        &.{},
        null,
        witness,
    );
    witness = null;
    errdefer report.deinit();

    allocator.free(report.induced_ranks);
    report.induced_ranks = try allocator.dupe(usize, homology.induced_ranks);
    return report;
}

const CarrierHomologyFailure = struct {
    allocator: std.mem.Allocator,
    input_face: topology.OwnedSimplex,
    kind: ObstructionKind,
    failed_invariant: FailedInvariant,
    unsupported_dimension: ?isize = null,

    fn deinit(self: *CarrierHomologyFailure) void {
        self.input_face.deinit();
    }
};

fn findCarrierHomologyFailure(
    allocator: std.mem.Allocator,
    problem: decision_search.DecisionSearchProblem,
    options: ObstructionOptions,
) !?CarrierHomologyFailure {
    var domain_faces = try problem.execution.domain.faces(allocator);
    defer domain_faces.deinit();

    for (domain_faces.simplexes) |input_face| {
        if (input_face.isEmpty()) continue;

        const protocol_image = problem.execution.imageOf(input_face) orelse return error.MissingCarrierSimplex;
        const output_image = problem.task.imageOf(input_face) orelse return error.MissingCarrierSimplex;

        const largest_dimension = @max(protocol_image.complex().dimension(), output_image.complex().dimension());
        if (largest_dimension > @as(isize, @intCast(options.max_homology_dimension))) {
            return .{
                .allocator = allocator,
                .input_face = try input_face.canonicalize(allocator),
                .kind = .unsupported_dimension,
                .failed_invariant = .unsupported_dimension,
                .unsupported_dimension = largest_dimension,
            };
        }

        var protocol_betti = try invariants.bettiNumbers(
            allocator,
            protocol_image.complex(),
            options.max_homology_dimension,
            options.prime,
        );
        defer protocol_betti.deinit();
        var output_betti = try invariants.bettiNumbers(
            allocator,
            output_image.complex(),
            options.max_homology_dimension,
            options.prime,
        );
        defer output_betti.deinit();

        var dimension: usize = 1;
        while (dimension <= options.max_homology_dimension) : (dimension += 1) {
            if (protocol_betti.values[dimension] > output_betti.values[dimension]) {
                return .{
                    .allocator = allocator,
                    .input_face = try input_face.canonicalize(allocator),
                    .kind = .hole_preservation_failure,
                    .failed_invariant = .betti_number,
                };
            }
            if (output_betti.values[dimension] > protocol_betti.values[dimension]) {
                return .{
                    .allocator = allocator,
                    .input_face = try input_face.canonicalize(allocator),
                    .kind = .hole_creation_failure,
                    .failed_invariant = .betti_number,
                };
            }
        }
    }

    return null;
}

fn buildReportFromProblem(
    allocator: std.mem.Allocator,
    problem: decision_search.DecisionSearchProblem,
    pcp_witness: PCPWitnessKind,
    kind: ObstructionKind,
    failed_invariant: FailedInvariant,
    input_face: topology.Simplex,
    options: ObstructionOptions,
    unsat_reason: ?decision_search.UnsatReason,
    unsat_core_vertices: []const topology.Vertex,
    unsupported_dimension: ?isize,
    homology_witness: ?invariants.OwnedHomologyMapWitness,
) !OwnedObstructionReport {
    const protocol_image = problem.execution.imageOf(input_face) orelse return error.MissingCarrierSimplex;
    const output_image = problem.task.imageOf(input_face) orelse return error.MissingCarrierSimplex;
    return buildReportFromImages(
        allocator,
        pcp_witness,
        kind,
        failed_invariant,
        input_face,
        protocol_image.complex(),
        output_image.complex(),
        options,
        unsat_reason,
        unsat_core_vertices,
        unsupported_dimension,
        homology_witness,
    );
}

fn buildReportFromImages(
    allocator: std.mem.Allocator,
    pcp_witness: PCPWitnessKind,
    kind: ObstructionKind,
    failed_invariant: FailedInvariant,
    input_face: topology.Simplex,
    protocol_image: topology.Complex,
    output_image: topology.Complex,
    options: ObstructionOptions,
    unsat_reason: ?decision_search.UnsatReason,
    unsat_core_vertices: []const topology.Vertex,
    unsupported_dimension: ?isize,
    homology_witness: ?invariants.OwnedHomologyMapWitness,
) !OwnedObstructionReport {
    var owned_input = try input_face.canonicalize(allocator);
    errdefer owned_input.deinit();
    var owned_protocol = try cloneClosedComplex(allocator, protocol_image);
    errdefer owned_protocol.deinit();
    var owned_output = try cloneClosedComplex(allocator, output_image);
    errdefer owned_output.deinit();

    var protocol_betti = try invariants.bettiNumbers(
        allocator,
        owned_protocol.complex(),
        options.max_homology_dimension,
        options.prime,
    );
    defer protocol_betti.deinit();
    const protocol_betti_values = try allocator.dupe(usize, protocol_betti.values);
    errdefer allocator.free(protocol_betti_values);

    var output_betti = try invariants.bettiNumbers(
        allocator,
        owned_output.complex(),
        options.max_homology_dimension,
        options.prime,
    );
    defer output_betti.deinit();
    const output_betti_values = try allocator.dupe(usize, output_betti.values);
    errdefer allocator.free(output_betti_values);

    const induced_ranks = try allocator.alloc(usize, 0);
    errdefer allocator.free(induced_ranks);

    const core = try allocator.dupe(topology.Vertex, unsat_core_vertices);
    errdefer allocator.free(core);

    return .{
        .allocator = allocator,
        .kind = kind,
        .pcp_witness = pcp_witness,
        .failed_invariant = failed_invariant,
        .input_face = owned_input,
        .protocol_carrier_image = owned_protocol,
        .output_carrier_image = owned_output,
        .protocol_betti = protocol_betti_values,
        .output_betti = output_betti_values,
        .induced_ranks = induced_ranks,
        .unsat_reason = unsat_reason,
        .unsupported_dimension = unsupported_dimension,
        .homology_witness = homology_witness,
        .unsat_core_vertices = core,
    };
}

fn selectInputFaceForCore(
    allocator: std.mem.Allocator,
    execution: topology.SimplicialCarrierMap,
    core_vertices: []const topology.Vertex,
) !topology.OwnedSimplex {
    var domain_faces = try execution.domain.faces(allocator);
    defer domain_faces.deinit();

    if (core_vertices.len != 0) {
        for (domain_faces.simplexes) |input_face| {
            if (input_face.isEmpty()) continue;
            const image = execution.imageOf(input_face) orelse return error.MissingCarrierSimplex;
            if (subcomplexContainsAllVertices(image, core_vertices)) {
                return input_face.canonicalize(allocator);
            }
        }
        for (domain_faces.simplexes) |input_face| {
            if (input_face.isEmpty()) continue;
            const image = execution.imageOf(input_face) orelse return error.MissingCarrierSimplex;
            if (subcomplexContainsAnyVertex(image, core_vertices)) {
                return input_face.canonicalize(allocator);
            }
        }
    }

    for (domain_faces.simplexes) |input_face| {
        if (!input_face.isEmpty()) return input_face.canonicalize(allocator);
    }
    return domain_faces.simplexes[0].canonicalize(allocator);
}

fn subcomplexContainsAllVertices(subcomplex: topology.Subcomplex, vertices: []const topology.Vertex) bool {
    for (vertices) |vertex| {
        if (!subcomplexContainsVertex(subcomplex, vertex)) return false;
    }
    return true;
}

fn subcomplexContainsAnyVertex(subcomplex: topology.Subcomplex, vertices: []const topology.Vertex) bool {
    for (vertices) |vertex| {
        if (subcomplexContainsVertex(subcomplex, vertex)) return true;
    }
    return false;
}

fn subcomplexContainsVertex(subcomplex: topology.Subcomplex, vertex: topology.Vertex) bool {
    const vertices = [_]topology.Vertex{vertex};
    return subcomplex.containsSimplex(.{ .vertices = &vertices });
}

fn cloneClosedComplex(allocator: std.mem.Allocator, complex: topology.Complex) !topology.OwnedComplex {
    var faces = try complex.faces(allocator);
    defer faces.deinit();

    var builder = topology.ComplexBuilder.init(allocator);
    errdefer builder.deinit();
    for (faces.simplexes) |simplex| try builder.add(simplex);
    return builder.toOwnedComplex();
}

fn writeUsizeList(writer: anytype, values: []const usize) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{value});
    }
    try writer.writeByte(']');
}

fn writeVertexInline(writer: anytype, vertex: topology.Vertex) !void {
    const simplex = topology.Simplex{ .vertices = &[_]topology.Vertex{vertex} };
    try reporting.writeSimplex(writer, simplex);
}

fn v(color: topology.Color, label: []const u8) topology.Vertex {
    return .{ .color = color, .label = .{ .symbol = label } };
}

fn addEmptyCarrierEntry(
    builder: *topology.SimplicialCarrierMapBuilder,
    codomain: topology.Complex,
) !void {
    const empty = topology.Simplex{ .vertices = &[_]topology.Vertex{} };
    try builder.add(empty, .{ .parent = codomain, .simplexes = &[_]topology.Simplex{empty} });
}

test "bounded obstruction report explains canonical disconnected-output toy task" {
    const in0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const in1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 0 } };
    const p1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .protocol = 1 } };
    const out0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const input_edge = [_]topology.Vertex{ in0, in1 };
    const input0 = [_]topology.Vertex{in0};
    const input1 = [_]topology.Vertex{in1};
    const protocol_edge = [_]topology.Vertex{ p0, p1 };
    const protocol0 = [_]topology.Vertex{p0};
    const protocol1 = [_]topology.Vertex{p1};
    const output0 = [_]topology.Vertex{out0};
    const output1 = [_]topology.Vertex{out1};

    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_edge }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &output0 },
        .{ .vertices = &output1 },
    } };

    var execution_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, protocol);
    defer execution_builder.deinit();
    try addEmptyCarrierEntry(&execution_builder, protocol);
    try execution_builder.add(.{ .vertices = &input0 }, .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} });
    try execution_builder.add(.{ .vertices = &input1 }, .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol1 }} });
    try execution_builder.add(.{ .vertices = &input_edge }, .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} });
    var execution = try execution_builder.toOwned();
    defer execution.deinit();

    var task_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, output);
    defer task_builder.deinit();
    try addEmptyCarrierEntry(&task_builder, output);
    try task_builder.add(.{ .vertices = &input0 }, .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output0 }} });
    try task_builder.add(.{ .vertices = &input1 }, .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output1 }} });
    try task_builder.add(.{ .vertices = &input_edge }, .{ .parent = output, .simplexes = &[_]topology.Simplex{
        .{ .vertices = &output0 },
        .{ .vertices = &output1 },
    } });
    var task = try task_builder.toOwned();
    defer task.deinit();

    var report = (try diagnoseBoundedDecisionProblem(std.testing.allocator, .{
        .execution = execution.carrierMap(),
        .task = task.carrierMap(),
    }, .partition, .{})).?;
    defer report.deinit();

    try std.testing.expectEqual(ObstructionKind.disconnected_task_output, report.kind);
    try std.testing.expectEqual(FailedInvariant.connectivity, report.failed_invariant);
    try std.testing.expectEqual(PCPWitnessKind.partition, report.pcp_witness);
    try std.testing.expect(topology.Simplex.eql(report.input_face.simplex(), .{ .vertices = &input_edge }));
    try std.testing.expect(report.protocol_carrier_image.complex().containsSimplex(.{ .vertices = &protocol_edge }));
    try std.testing.expect(report.output_carrier_image.complex().containsSimplex(.{ .vertices = &output0 }));
    try std.testing.expect(!report.output_carrier_image.complex().containsSimplex(.{ .vertices = &[_]topology.Vertex{ out0, out1 } }));

    var rendered = std.ArrayList(u8).init(std.testing.allocator);
    defer rendered.deinit();
    try report.writeHuman(std.testing.allocator, rendered.writer());
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "input_face:") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "protocol_carrier_image") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "output_carrier_image") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "failed_invariant: connectivity") != null);
}

test "bounded obstruction report names PCP global rollback gap under partial gateway loss" {
    const ih = v(.{ .hub = 0 }, "global_job.accepted.partial_loss");
    const ij = v(.{ .job = 61 }, "job.requires_two_gateways");
    const iga = v(.{ .gateway = 1 }, "gateway_a.placed.partial_loss");
    const igb = v(.{ .gateway = 2 }, "gateway_b.lost");
    const ph = v(.{ .hub = 0 }, "view.global_commit_candidate");
    const pj = v(.{ .job = 61 }, "view.job.running_candidate");
    const pga = v(.{ .gateway = 1 }, "view.gateway_a.committed");
    const pgb = v(.{ .gateway = 2 }, "view.gateway_b.lost");
    const oh = v(.{ .hub = 0 }, "global_job.committed_without_rollback");
    const oj = v(.{ .job = 61 }, "job.running_without_rollback");
    const oga = v(.{ .gateway = 1 }, "gateway_a.committed_without_rollback");
    const ogb = v(.{ .gateway = 2 }, "gateway_b.lost_without_rollback");

    const input_vertices = [_]topology.Vertex{ ih, ij, iga, igb };
    const protocol_vertices = [_]topology.Vertex{ ph, pj, pga, pgb };
    const output_h_j_ga = [_]topology.Vertex{ oh, oj, oga };
    const output_h_j_gb = [_]topology.Vertex{ oh, oj, ogb };
    const output_h_ga_gb = [_]topology.Vertex{ oh, oga, ogb };
    const output_j_ga_gb = [_]topology.Vertex{ oj, oga, ogb };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_vertices }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_vertices }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &output_h_j_ga },
        .{ .vertices = &output_h_j_gb },
        .{ .vertices = &output_h_ga_gb },
        .{ .vertices = &output_j_ga_gb },
    } };

    var execution_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, protocol);
    defer execution_builder.deinit();
    try addColorProjectionSimplexCarrierEntries(
        std.testing.allocator,
        &execution_builder,
        input,
        protocol,
        &[_]topology.Vertex{ ph, pj, pga, pgb },
        true,
    );
    var execution = try execution_builder.toOwned();
    defer execution.deinit();

    var task_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, output);
    defer task_builder.deinit();
    try addColorProjectionBoundaryCarrierEntries(
        std.testing.allocator,
        &task_builder,
        input,
        output,
        &[_]topology.Vertex{ oh, oj, oga, ogb },
    );
    var task = try task_builder.toOwned();
    defer task.deinit();

    var report = (try diagnoseBoundedDecisionProblem(std.testing.allocator, .{
        .execution = execution.carrierMap(),
        .task = task.carrierMap(),
    }, .global_rollback_gap, .{ .max_homology_dimension = 3 })).?;
    defer report.deinit();

    try std.testing.expectEqual(ObstructionKind.hole_creation_failure, report.kind);
    try std.testing.expectEqual(FailedInvariant.betti_number, report.failed_invariant);
    try std.testing.expectEqual(PCPWitnessKind.global_rollback_gap, report.pcp_witness);
    try std.testing.expect(topology.Simplex.eql(report.input_face.simplex(), .{ .vertices = &input_vertices }));
    try std.testing.expect(report.protocol_carrier_image.complex().containsSimplex(.{ .vertices = &protocol_vertices }));
    try std.testing.expect(!report.output_carrier_image.complex().containsSimplex(.{ .vertices = &[_]topology.Vertex{ oh, oj, oga, ogb } }));
    try std.testing.expectEqual(@as(usize, 0), report.protocol_betti[2]);
    try std.testing.expectEqual(@as(usize, 1), report.output_betti[2]);
}

test "bounded obstruction report distinguishes carrier nonexistence" {
    const input_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const protocol_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 0 } };
    const output_vertex = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const input0 = [_]topology.Vertex{input_vertex};
    const protocol0 = [_]topology.Vertex{protocol_vertex};
    const output1 = [_]topology.Vertex{output_vertex};
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input0 }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &output1 }} };

    var execution_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, protocol);
    defer execution_builder.deinit();
    try addEmptyCarrierEntry(&execution_builder, protocol);
    try execution_builder.add(.{ .vertices = &input0 }, .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} });
    var execution = try execution_builder.toOwned();
    defer execution.deinit();

    var task_builder = topology.SimplicialCarrierMapBuilder.init(std.testing.allocator, input, output);
    defer task_builder.deinit();
    try addEmptyCarrierEntry(&task_builder, output);
    try task_builder.add(.{ .vertices = &input0 }, .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output1 }} });
    var task = try task_builder.toOwned();
    defer task.deinit();

    var report = (try diagnoseBoundedDecisionProblem(std.testing.allocator, .{
        .execution = execution.carrierMap(),
        .task = task.carrierMap(),
    }, .generic, .{})).?;
    defer report.deinit();

    try std.testing.expectEqual(ObstructionKind.carrier_nonexistence, report.kind);
    try std.testing.expectEqual(FailedInvariant.candidate_carrier, report.failed_invariant);
    try std.testing.expectEqual(decision_search.UnsatReason.empty_candidate_set, report.unsat_reason.?);
}

test "map-induced obstruction report carries a concrete lost cycle witness" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const point = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const ab = [_]topology.Vertex{ a, b };
    const bc = [_]topology.Vertex{ b, c };
    const ac = [_]topology.Vertex{ a, c };
    const point_vertices = [_]topology.Vertex{point};
    const input_face_vertices = [_]topology.Vertex{ a, b, c };
    const domain = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &ab },
        .{ .vertices = &bc },
        .{ .vertices = &ac },
    } };
    const codomain = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &point_vertices }} };
    const entries = [_]topology.VertexMapEntry{
        .{ .source = a, .target = point },
        .{ .source = b, .target = point },
        .{ .source = c, .target = point },
    };

    var report = (try diagnoseMapInducedInvariantFailure(std.testing.allocator, .{
        .input_face = .{ .vertices = &input_face_vertices },
        .protocol_carrier_image = domain,
        .output_carrier_image = codomain,
        .map = .{ .domain = domain, .codomain = codomain, .entries = &entries },
        .pcp_witness = .cancellation_hole,
    }, .{})).?;
    defer report.deinit();

    try std.testing.expectEqual(ObstructionKind.hole_preservation_failure, report.kind);
    try std.testing.expectEqual(FailedInvariant.induced_homology, report.failed_invariant);
    try std.testing.expectEqual(PCPWitnessKind.cancellation_hole, report.pcp_witness);
    try std.testing.expectEqual(invariants.HomologyMapWitnessKind.source_cycle_lost, report.homology_witness.?.kind);
    try std.testing.expectEqual(@as(usize, 3), report.homology_witness.?.source_cycle.?.simplexes.len);
    try std.testing.expectEqual(@as(usize, 0), report.induced_ranks[1]);
}

fn addColorProjectionSimplexCarrierEntries(
    allocator: std.mem.Allocator,
    builder: *topology.SimplicialCarrierMapBuilder,
    input: topology.Complex,
    codomain: topology.Complex,
    targets: []const topology.Vertex,
    connected_images: bool,
) !void {
    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    for (input_faces.simplexes) |face| {
        if (face.isEmpty()) {
            try addEmptyCarrierEntry(builder, codomain);
            continue;
        }

        var projected = std.ArrayList(topology.Vertex).init(allocator);
        defer projected.deinit();
        for (face.vertices) |source| {
            const before = projected.items.len;
            for (targets) |target| {
                if (topology.Color.eql(source.color, target.color)) {
                    try projected.append(target);
                    break;
                }
            }
            if (projected.items.len == before) return error.MissingColorProjection;
        }

        if (connected_images) {
            try builder.add(face, .{
                .parent = codomain,
                .simplexes = &[_]topology.Simplex{.{ .vertices = projected.items }},
            });
        } else {
            var singleton_simplexes = std.ArrayList(topology.Simplex).init(allocator);
            defer singleton_simplexes.deinit();
            for (projected.items, 0..) |_, index| {
                try singleton_simplexes.append(.{ .vertices = projected.items[index .. index + 1] });
            }
            try builder.add(face, .{
                .parent = codomain,
                .simplexes = singleton_simplexes.items,
            });
        }
    }
}

fn addColorProjectionBoundaryCarrierEntries(
    allocator: std.mem.Allocator,
    builder: *topology.SimplicialCarrierMapBuilder,
    input: topology.Complex,
    codomain: topology.Complex,
    targets: []const topology.Vertex,
) !void {
    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    for (input_faces.simplexes) |face| {
        if (face.isEmpty()) {
            try addEmptyCarrierEntry(builder, codomain);
            continue;
        }

        var projected = std.ArrayList(topology.Vertex).init(allocator);
        defer projected.deinit();
        for (face.vertices) |source| {
            const before = projected.items.len;
            for (targets) |target| {
                if (topology.Color.eql(source.color, target.color)) {
                    try projected.append(target);
                    break;
                }
            }
            if (projected.items.len == before) return error.MissingColorProjection;
        }

        const projected_simplex = topology.Simplex{ .vertices = projected.items };
        if (codomain.containsSimplex(projected_simplex)) {
            try builder.add(face, .{
                .parent = codomain,
                .simplexes = &[_]topology.Simplex{projected_simplex},
            });
        } else {
            try builder.add(face, .{
                .parent = codomain,
                .simplexes = codomain.simplexes,
            });
        }
    }
}
