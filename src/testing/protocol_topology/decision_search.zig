const std = @import("std");

const topology = @import("topology.zig");

pub const DecisionSearchProblem = struct {
    execution: topology.SimplicialCarrierMap,
    task: topology.SimplicialCarrierMap,

    pub fn validate(self: DecisionSearchProblem, allocator: std.mem.Allocator) !void {
        try self.execution.validate(allocator, .{});
        try self.task.validate(allocator, .{});
        if (!(try complexesEqualByClosure(allocator, self.execution.domain, self.task.domain))) {
            return error.ComplexMismatch;
        }
    }
};

pub const DecisionSearchOptions = struct {
    max_solutions: usize = 1,
    max_assignments: usize = 100_000,
    max_unsat_core_vertices: usize = 10,
};

pub const DecisionSearchStatus = enum {
    satisfiable,
    unsatisfiable,
    limit_exceeded,
};

pub const UnsatReason = enum {
    empty_candidate_set,
    local_inconsistency,
    global_inconsistency,
    search_limit_exceeded,
};

pub const CandidateSet = struct {
    source: topology.Vertex,
    targets: []topology.Vertex,
};

pub const SmtLibExportOptions = struct {
    timeout_ms: ?u64 = null,
    produce_unsat_cores: bool = false,
    include_check_sat: bool = true,
    include_model_query: bool = false,
    include_unsat_core_query: bool = false,
};

pub const OwnedUnsatCore = struct {
    allocator: std.mem.Allocator,
    vertices: []topology.Vertex,
    reason: UnsatReason,

    pub fn deinit(self: *OwnedUnsatCore) void {
        if (self.vertices.len != 0) self.allocator.free(self.vertices);
        self.vertices = &.{};
    }
};

pub const OwnedDecisionSearchResult = struct {
    allocator: std.mem.Allocator,
    status: DecisionSearchStatus,
    candidate_sets: []CandidateSet,
    solutions: []topology.OwnedVertexMap,
    unsat_core: OwnedUnsatCore,
    explored_assignments: usize,

    pub fn firstSolution(self: *const OwnedDecisionSearchResult) ?topology.VertexMap {
        if (self.solutions.len == 0) return null;
        return self.solutions[0].vertexMap();
    }

    pub fn deinit(self: *OwnedDecisionSearchResult) void {
        for (self.solutions) |*solution| {
            solution.deinit();
        }
        self.allocator.free(self.solutions);
        deinitCandidateSets(self.allocator, self.candidate_sets);
        self.unsat_core.deinit();
        self.solutions = &.{};
        self.candidate_sets = &.{};
        self.explored_assignments = 0;
    }
};

pub fn solveDecisionMap(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
) !OwnedDecisionSearchResult {
    return enumerateDecisionMaps(allocator, problem, .{ .max_solutions = 1 });
}

pub fn enumerateDecisionMaps(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    options: DecisionSearchOptions,
) !OwnedDecisionSearchResult {
    try problem.validate(allocator);

    const candidate_sets = try buildCandidateSets(allocator, problem);
    errdefer deinitCandidateSets(allocator, candidate_sets);

    var solutions = std.ArrayList(topology.OwnedVertexMap).init(allocator);
    defer solutions.deinit();
    errdefer deinitSolutionBuilder(&solutions);

    var search_state = SearchState{
        .allocator = allocator,
        .problem = problem,
        .options = options,
        .candidate_sets = candidate_sets,
        .solutions = &solutions,
    };

    if (!hasEmptyCandidateSet(candidate_sets) and options.max_solutions != 0) {
        const order = try variableOrder(allocator, candidate_sets);
        defer allocator.free(order);
        const assignment = try allocator.alloc(?usize, candidate_sets.len);
        defer allocator.free(assignment);
        @memset(assignment, null);

        try search_state.search(order, assignment, 0);
    }

    var status: DecisionSearchStatus = .unsatisfiable;
    if (solutions.items.len != 0) {
        status = .satisfiable;
    } else if (search_state.limit_exceeded) {
        status = .limit_exceeded;
    }

    var unsat_core = try emptyUnsatCore(allocator);
    errdefer unsat_core.deinit();
    if (status != .satisfiable) {
        unsat_core.deinit();
        unsat_core = try computeMinimalUnsatCore(allocator, problem, candidate_sets, options, status);
    }

    const owned_solutions = try solutions.toOwnedSlice();
    errdefer {
        for (owned_solutions) |*solution| {
            solution.deinit();
        }
        allocator.free(owned_solutions);
    }

    return .{
        .allocator = allocator,
        .status = status,
        .candidate_sets = candidate_sets,
        .solutions = owned_solutions,
        .unsat_core = unsat_core,
        .explored_assignments = search_state.explored_assignments,
    };
}

pub fn writeSmtLib2(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    writer: anytype,
) !void {
    try writeSmtLib2WithOptions(allocator, problem, writer, .{});
}

pub fn writeSmtLib2WithOptions(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    writer: anytype,
    options: SmtLibExportOptions,
) !void {
    try problem.validate(allocator);
    const candidate_sets = try buildDecisionCandidateSets(allocator, problem);
    defer deinitDecisionCandidateSets(allocator, candidate_sets);

    if (options.timeout_ms) |timeout_ms| {
        try writer.print("(set-option :timeout {d})\n", .{timeout_ms});
    }
    if (options.produce_unsat_cores) {
        try writer.writeAll("(set-option :produce-unsat-cores true)\n");
    }
    try writer.writeAll("(set-logic QF_BOOL)\n");
    try writer.writeAll("; x_i_j means protocol vertex i maps to candidate j\n");

    var assertions = AssertionWriter{ .name_assertions = options.produce_unsat_cores };

    for (candidate_sets, 0..) |candidate_set, source_index| {
        try writer.print("; source {d} color={d}:{d} candidates={d}\n", .{
            source_index,
            candidate_set.source.color.order(),
            candidate_set.source.color.value(),
            candidate_set.targets.len,
        });
        for (candidate_set.targets, 0..) |_, target_index| {
            try writer.print("(declare-const x_{d}_{d} Bool)\n", .{ source_index, target_index });
        }
    }

    for (candidate_sets, 0..) |candidate_set, source_index| {
        if (candidate_set.targets.len == 0) {
            try assertions.begin(writer);
            try writer.writeAll("false");
            try assertions.end(writer);
            try writer.writeAll("; no color-and-carrier candidate\n");
            continue;
        }
        try assertions.begin(writer);
        try writer.writeAll("(or");
        for (candidate_set.targets, 0..) |_, target_index| {
            try writer.print(" x_{d}_{d}", .{ source_index, target_index });
        }
        try writer.writeAll(")");
        try assertions.end(writer);

        for (candidate_set.targets, 0..) |_, left_index| {
            for (candidate_set.targets[left_index + 1 ..], left_index + 1..) |_, right_index| {
                try assertions.begin(writer);
                try writer.print(
                    "(not (and x_{d}_{d} x_{d}_{d}))",
                    .{ source_index, left_index, source_index, right_index },
                );
                try assertions.end(writer);
            }
        }
    }

    var protocol_faces = try problem.execution.codomain.faces(allocator);
    defer protocol_faces.deinit();
    for (protocol_faces.simplexes) |protocol_face| {
        try writeInvalidSimplexClauses(
            allocator,
            candidate_sets,
            protocol_face,
            problem.task.codomain,
            &assertions,
            writer,
        );
    }

    for (problem.execution.entries) |entry| {
        const task_image = problem.task.imageOf(entry.simplex) orelse return error.MissingCarrierSimplex;
        var reachable_faces = try entry.image.complex().faces(allocator);
        defer reachable_faces.deinit();
        for (reachable_faces.simplexes) |reachable_face| {
            try writeInvalidSimplexClauses(
                allocator,
                candidate_sets,
                reachable_face,
                task_image.complex(),
                &assertions,
                writer,
            );
        }
    }

    if (options.include_check_sat) try writer.writeAll("(check-sat)\n");
    if (options.include_model_query) {
        try writer.writeAll("(get-value (");
        var first = true;
        for (candidate_sets, 0..) |candidate_set, source_index| {
            for (candidate_set.targets, 0..) |_, target_index| {
                if (!first) try writer.writeAll(" ");
                first = false;
                try writer.print("x_{d}_{d}", .{ source_index, target_index });
            }
        }
        try writer.writeAll("))\n");
    }
    if (options.include_unsat_core_query) try writer.writeAll("(get-unsat-core)\n");
}

pub fn buildDecisionCandidateSets(allocator: std.mem.Allocator, problem: DecisionSearchProblem) ![]CandidateSet {
    try problem.validate(allocator);
    return buildCandidateSets(allocator, problem);
}

pub fn deinitDecisionCandidateSets(allocator: std.mem.Allocator, candidate_sets: []CandidateSet) void {
    deinitCandidateSets(allocator, candidate_sets);
}

pub fn ownedMapFromCandidateAssignment(
    allocator: std.mem.Allocator,
    domain: topology.Complex,
    codomain: topology.Complex,
    candidate_sets: []const CandidateSet,
    assignment: []const ?usize,
) !topology.OwnedVertexMap {
    return ownedMapFromAssignment(allocator, domain, codomain, candidate_sets, assignment);
}

const SearchState = struct {
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    options: DecisionSearchOptions,
    candidate_sets: []const CandidateSet,
    solutions: *std.ArrayList(topology.OwnedVertexMap),
    explored_assignments: usize = 0,
    limit_exceeded: bool = false,

    fn search(self: *SearchState, order: []const usize, assignment: []?usize, depth: usize) !void {
        if (self.solutions.items.len >= self.options.max_solutions) return;
        if (self.limit_exceeded) return;

        if (depth == order.len) {
            self.explored_assignments += 1;
            if (self.explored_assignments > self.options.max_assignments) {
                self.limit_exceeded = true;
                return;
            }

            var decision = try ownedMapFromAssignment(
                self.allocator,
                self.problem.execution.codomain,
                self.problem.task.codomain,
                self.candidate_sets,
                assignment,
            );
            const valid = validDecisionMap(self.allocator, self.problem, decision.vertexMap());
            if (valid) {
                self.solutions.append(decision) catch |err| {
                    decision.deinit();
                    return err;
                };
            } else {
                decision.deinit();
            }
            return;
        }

        const source_index = order[depth];
        for (self.candidate_sets[source_index].targets, 0..) |_, candidate_index| {
            assignment[source_index] = candidate_index;
            if (try isAssignmentConsistent(
                self.allocator,
                self.problem,
                self.candidate_sets,
                assignment,
            )) {
                try self.search(order, assignment, depth + 1);
            }
            assignment[source_index] = null;

            if (self.solutions.items.len >= self.options.max_solutions or self.limit_exceeded) return;
        }
    }
};

fn buildCandidateSets(allocator: std.mem.Allocator, problem: DecisionSearchProblem) ![]CandidateSet {
    var protocol_vertices = try problem.execution.codomain.vertices(allocator);
    defer protocol_vertices.deinit();
    var output_vertices = try problem.task.codomain.vertices(allocator);
    defer output_vertices.deinit();

    const candidate_sets = try allocator.alloc(CandidateSet, protocol_vertices.vertices.len);
    errdefer allocator.free(candidate_sets);
    var initialized: usize = 0;
    errdefer {
        for (candidate_sets[0..initialized]) |candidate_set| {
            if (candidate_set.targets.len != 0) allocator.free(candidate_set.targets);
        }
    }

    for (protocol_vertices.vertices, 0..) |source, index| {
        var candidates = std.ArrayList(topology.Vertex).init(allocator);
        errdefer candidates.deinit();

        for (output_vertices.vertices) |target| {
            if (topology.Color.eql(source.color, target.color)) {
                try candidates.append(target);
            }
        }

        for (problem.execution.entries) |execution_entry| {
            if (!subcomplexContainsVertex(execution_entry.image, source)) continue;

            const task_image = problem.task.imageOf(execution_entry.simplex) orelse return error.MissingCarrierSimplex;
            try restrictCandidatesToCarrier(&candidates, task_image);
        }

        candidate_sets[index] = .{
            .source = source,
            .targets = try candidates.toOwnedSlice(),
        };
        initialized += 1;
    }

    return candidate_sets;
}

fn restrictCandidatesToCarrier(candidates: *std.ArrayList(topology.Vertex), carrier: topology.Subcomplex) !void {
    var retained = std.ArrayList(topology.Vertex).init(candidates.allocator);
    errdefer retained.deinit();

    for (candidates.items) |target| {
        if (subcomplexContainsVertex(carrier, target)) {
            try retained.append(target);
        }
    }

    candidates.deinit();
    candidates.* = retained;
}

fn isAssignmentConsistent(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    candidate_sets: []const CandidateSet,
    assignment: []const ?usize,
) !bool {
    var protocol_faces = try problem.execution.codomain.faces(allocator);
    defer protocol_faces.deinit();
    for (protocol_faces.simplexes) |protocol_face| {
        var mapped = try mappedSimplexIfAssigned(allocator, candidate_sets, assignment, protocol_face);
        defer if (mapped) |*simplex| simplex.deinit();
        if (mapped) |*simplex| {
            if (!problem.task.codomain.containsSimplex(simplex.simplex())) return false;
        }
    }

    for (problem.execution.entries) |execution_entry| {
        const task_image = problem.task.imageOf(execution_entry.simplex) orelse return error.MissingCarrierSimplex;
        var reachable_faces = try execution_entry.image.complex().faces(allocator);
        defer reachable_faces.deinit();

        for (reachable_faces.simplexes) |reachable_face| {
            var mapped = try mappedSimplexIfAssigned(allocator, candidate_sets, assignment, reachable_face);
            defer if (mapped) |*simplex| simplex.deinit();
            if (mapped) |*simplex| {
                if (!task_image.containsSimplex(simplex.simplex())) return false;
            }
        }
    }

    return true;
}

fn mappedSimplexIfAssigned(
    allocator: std.mem.Allocator,
    candidate_sets: []const CandidateSet,
    assignment: []const ?usize,
    simplex: topology.Simplex,
) !?topology.OwnedSimplex {
    const vertices = try allocator.alloc(topology.Vertex, simplex.vertices.len);
    errdefer allocator.free(vertices);

    for (simplex.vertices, 0..) |source, out_index| {
        const source_index = indexOfSource(candidate_sets, source) orelse return error.SourceVertexOutsideDomain;
        const candidate_index = assignment[source_index] orelse {
            allocator.free(vertices);
            return null;
        };
        vertices[out_index] = candidate_sets[source_index].targets[candidate_index];
    }

    return topology.OwnedSimplex{ .allocator = allocator, .vertices = vertices };
}

fn ownedMapFromAssignment(
    allocator: std.mem.Allocator,
    domain: topology.Complex,
    codomain: topology.Complex,
    candidate_sets: []const CandidateSet,
    assignment: []const ?usize,
) !topology.OwnedVertexMap {
    const entries = try allocator.alloc(topology.VertexMapEntry, candidate_sets.len);
    errdefer allocator.free(entries);

    for (candidate_sets, 0..) |candidate_set, index| {
        const candidate_index = assignment[index] orelse return error.IncompleteAssignment;
        entries[index] = .{
            .source = candidate_set.source,
            .target = candidate_set.targets[candidate_index],
        };
    }

    return .{
        .allocator = allocator,
        .domain = domain,
        .codomain = codomain,
        .entries = entries,
    };
}

fn validDecisionMap(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    decision: topology.VertexMap,
) bool {
    topology.validateCarriedDecisionMap(allocator, problem.execution, problem.task, decision) catch return false;
    return true;
}

fn computeMinimalUnsatCore(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    candidate_sets: []const CandidateSet,
    options: DecisionSearchOptions,
    status: DecisionSearchStatus,
) !OwnedUnsatCore {
    if (status == .limit_exceeded) {
        return fullUnsatCore(allocator, candidate_sets, .search_limit_exceeded);
    }

    for (candidate_sets) |candidate_set| {
        if (candidate_set.targets.len == 0) {
            return unsatCoreFromVertices(allocator, &[_]topology.Vertex{candidate_set.source}, .empty_candidate_set);
        }
    }

    if (candidate_sets.len >= @bitSizeOf(usize)) {
        return fullUnsatCore(allocator, candidate_sets, .global_inconsistency);
    }

    const max_size = @min(candidate_sets.len, options.max_unsat_core_vertices);
    var size: usize = 1;
    while (size <= max_size) : (size += 1) {
        const mask_limit = @as(usize, 1) << @intCast(candidate_sets.len);
        var mask: usize = 1;
        while (mask < mask_limit) : (mask += 1) {
            if (@popCount(mask) != size) continue;
            if (!(try subsetSatisfiable(allocator, problem, candidate_sets, mask, options.max_assignments))) {
                return unsatCoreFromMask(allocator, candidate_sets, mask, .local_inconsistency);
            }
        }
    }

    return fullUnsatCore(allocator, candidate_sets, .global_inconsistency);
}

fn subsetSatisfiable(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    candidate_sets: []const CandidateSet,
    mask: usize,
    max_assignments: usize,
) !bool {
    var order = std.ArrayList(usize).init(allocator);
    defer order.deinit();

    for (candidate_sets, 0..) |_, index| {
        const bit = @as(usize, 1) << @intCast(index);
        if ((mask & bit) != 0) try order.append(index);
    }
    std.mem.sort(usize, order.items, candidate_sets, lessCandidateIndex);

    const assignment = try allocator.alloc(?usize, candidate_sets.len);
    defer allocator.free(assignment);
    @memset(assignment, null);

    var explored: usize = 0;
    return subsetSearch(allocator, problem, candidate_sets, order.items, assignment, 0, max_assignments, &explored);
}

fn subsetSearch(
    allocator: std.mem.Allocator,
    problem: DecisionSearchProblem,
    candidate_sets: []const CandidateSet,
    order: []const usize,
    assignment: []?usize,
    depth: usize,
    max_assignments: usize,
    explored: *usize,
) !bool {
    if (depth == order.len) {
        explored.* += 1;
        if (explored.* > max_assignments) return true;
        return isAssignmentConsistent(allocator, problem, candidate_sets, assignment);
    }

    const source_index = order[depth];
    for (candidate_sets[source_index].targets, 0..) |_, candidate_index| {
        assignment[source_index] = candidate_index;
        if (try isAssignmentConsistent(allocator, problem, candidate_sets, assignment)) {
            if (try subsetSearch(allocator, problem, candidate_sets, order, assignment, depth + 1, max_assignments, explored)) {
                assignment[source_index] = null;
                return true;
            }
        }
        assignment[source_index] = null;
    }

    return false;
}

fn writeInvalidSimplexClauses(
    allocator: std.mem.Allocator,
    candidate_sets: []const CandidateSet,
    simplex: topology.Simplex,
    allowed: topology.Complex,
    assertions: *AssertionWriter,
    writer: anytype,
) !void {
    if (simplex.vertices.len == 0) return;

    const variables = try allocator.alloc(usize, simplex.vertices.len);
    defer allocator.free(variables);

    for (simplex.vertices, 0..) |vertex, index| {
        variables[index] = indexOfSource(candidate_sets, vertex) orelse return error.SourceVertexOutsideDomain;
    }

    const selected = try allocator.alloc(usize, variables.len);
    defer allocator.free(selected);

    try writeInvalidClauseCombinations(allocator, candidate_sets, variables, selected, 0, allowed, assertions, writer);
}

fn writeInvalidClauseCombinations(
    allocator: std.mem.Allocator,
    candidate_sets: []const CandidateSet,
    variables: []const usize,
    selected: []usize,
    depth: usize,
    allowed: topology.Complex,
    assertions: *AssertionWriter,
    writer: anytype,
) !void {
    if (depth == variables.len) {
        const mapped_vertices = try allocator.alloc(topology.Vertex, variables.len);
        defer allocator.free(mapped_vertices);

        for (variables, 0..) |source_index, index| {
            mapped_vertices[index] = candidate_sets[source_index].targets[selected[index]];
        }
        if (allowed.containsSimplex(.{ .vertices = mapped_vertices })) return;

        try assertions.begin(writer);
        try writer.writeAll("(not (and");
        for (variables, selected) |source_index, target_index| {
            try writer.print(" x_{d}_{d}", .{ source_index, target_index });
        }
        try writer.writeAll("))");
        try assertions.end(writer);
        return;
    }

    const source_index = variables[depth];
    for (candidate_sets[source_index].targets, 0..) |_, candidate_index| {
        selected[depth] = candidate_index;
        try writeInvalidClauseCombinations(allocator, candidate_sets, variables, selected, depth + 1, allowed, assertions, writer);
    }
}

const AssertionWriter = struct {
    name_assertions: bool = false,
    next_name: usize = 0,

    fn begin(self: *AssertionWriter, writer: anytype) !void {
        if (self.name_assertions) {
            try writer.writeAll("(assert (! ");
        } else {
            try writer.writeAll("(assert ");
        }
    }

    fn end(self: *AssertionWriter, writer: anytype) !void {
        if (self.name_assertions) {
            const name = self.next_name;
            self.next_name += 1;
            try writer.print(" :named c_{d}))\n", .{name});
        } else {
            try writer.writeAll(")\n");
        }
    }
};

fn variableOrder(allocator: std.mem.Allocator, candidate_sets: []const CandidateSet) ![]usize {
    const order = try allocator.alloc(usize, candidate_sets.len);
    for (order, 0..) |*slot, index| {
        slot.* = index;
    }
    std.mem.sort(usize, order, candidate_sets, lessCandidateIndex);
    return order;
}

fn lessCandidateIndex(candidate_sets: []const CandidateSet, left: usize, right: usize) bool {
    if (candidate_sets[left].targets.len != candidate_sets[right].targets.len) {
        return candidate_sets[left].targets.len < candidate_sets[right].targets.len;
    }
    return topology.Vertex.lessThan({}, candidate_sets[left].source, candidate_sets[right].source);
}

fn subcomplexContainsVertex(subcomplex: topology.Subcomplex, vertex: topology.Vertex) bool {
    const vertices = [_]topology.Vertex{vertex};
    return subcomplex.containsSimplex(.{ .vertices = &vertices });
}

fn hasEmptyCandidateSet(candidate_sets: []const CandidateSet) bool {
    for (candidate_sets) |candidate_set| {
        if (candidate_set.targets.len == 0) return true;
    }
    return false;
}

fn indexOfSource(candidate_sets: []const CandidateSet, source: topology.Vertex) ?usize {
    for (candidate_sets, 0..) |candidate_set, index| {
        if (topology.Vertex.eql(candidate_set.source, source)) return index;
    }
    return null;
}

fn complexesEqualByClosure(allocator: std.mem.Allocator, left: topology.Complex, right: topology.Complex) !bool {
    var left_faces = try left.faces(allocator);
    defer left_faces.deinit();
    var right_faces = try right.faces(allocator);
    defer right_faces.deinit();

    for (left_faces.simplexes) |simplex| {
        if (!right_faces.complex().containsSimplex(simplex)) return false;
    }
    for (right_faces.simplexes) |simplex| {
        if (!left_faces.complex().containsSimplex(simplex)) return false;
    }
    return true;
}

fn emptyUnsatCore(allocator: std.mem.Allocator) !OwnedUnsatCore {
    return .{
        .allocator = allocator,
        .vertices = try allocator.alloc(topology.Vertex, 0),
        .reason = .global_inconsistency,
    };
}

fn fullUnsatCore(
    allocator: std.mem.Allocator,
    candidate_sets: []const CandidateSet,
    reason: UnsatReason,
) !OwnedUnsatCore {
    const vertices = try allocator.alloc(topology.Vertex, candidate_sets.len);
    for (candidate_sets, 0..) |candidate_set, index| {
        vertices[index] = candidate_set.source;
    }
    return .{ .allocator = allocator, .vertices = vertices, .reason = reason };
}

fn unsatCoreFromMask(
    allocator: std.mem.Allocator,
    candidate_sets: []const CandidateSet,
    mask: usize,
    reason: UnsatReason,
) !OwnedUnsatCore {
    var vertices = std.ArrayList(topology.Vertex).init(allocator);
    errdefer vertices.deinit();

    for (candidate_sets, 0..) |candidate_set, index| {
        const bit = @as(usize, 1) << @intCast(index);
        if ((mask & bit) != 0) try vertices.append(candidate_set.source);
    }

    return .{ .allocator = allocator, .vertices = try vertices.toOwnedSlice(), .reason = reason };
}

fn unsatCoreFromVertices(
    allocator: std.mem.Allocator,
    vertices: []const topology.Vertex,
    reason: UnsatReason,
) !OwnedUnsatCore {
    return .{
        .allocator = allocator,
        .vertices = try allocator.dupe(topology.Vertex, vertices),
        .reason = reason,
    };
}

fn deinitCandidateSets(allocator: std.mem.Allocator, candidate_sets: []CandidateSet) void {
    for (candidate_sets) |candidate_set| {
        if (candidate_set.targets.len != 0) allocator.free(candidate_set.targets);
    }
    allocator.free(candidate_sets);
}

fn deinitSolutionBuilder(solutions: *std.ArrayList(topology.OwnedVertexMap)) void {
    for (solutions.items) |*solution| {
        solution.deinit();
    }
}

fn expectTarget(map: topology.VertexMap, source: topology.Vertex, target: topology.Vertex) !void {
    const actual = map.targetForVertex(source) orelse return error.MissingVertexMapEntry;
    try std.testing.expect(topology.Vertex.eql(actual, target));
}

test "decision search finds a carried color-preserving map" {
    const in0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const in1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } };
    const p1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .protocol = 20 } };
    const out0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const extra0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 9 } };
    const empty_vertices = [_]topology.Vertex{};
    const input_edge = [_]topology.Vertex{ in0, in1 };
    const input0 = [_]topology.Vertex{in0};
    const input1 = [_]topology.Vertex{in1};
    const protocol_edge = [_]topology.Vertex{ p0, p1 };
    const protocol0 = [_]topology.Vertex{p0};
    const protocol1 = [_]topology.Vertex{p1};
    const output_edge = [_]topology.Vertex{ out0, out1 };
    const output0 = [_]topology.Vertex{out0};
    const output1 = [_]topology.Vertex{out1};
    const extra_output0 = [_]topology.Vertex{extra0};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_edge }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &output_edge },
        .{ .vertices = &extra_output0 },
    } };
    const execution_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} } },
        .{ .simplex = .{ .vertices = &input1 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol1 }} } },
        .{ .simplex = .{ .vertices = &input_edge }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} } },
    };
    const task_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output0 }} } },
        .{ .simplex = .{ .vertices = &input1 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output1 }} } },
        .{ .simplex = .{ .vertices = &input_edge }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output_edge }} } },
    };

    var result = try solveDecisionMap(std.testing.allocator, .{
        .execution = .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .task = .{ .domain = input, .codomain = output, .entries = &task_entries },
    });
    defer result.deinit();

    try std.testing.expectEqual(DecisionSearchStatus.satisfiable, result.status);
    try std.testing.expectEqual(@as(usize, 1), result.solutions.len);
    const solution = result.firstSolution().?;
    try expectTarget(solution, p0, out0);
    try expectTarget(solution, p1, out1);
    try topology.validateCarriedDecisionMap(std.testing.allocator, .{
        .domain = input,
        .codomain = protocol,
        .entries = &execution_entries,
    }, .{
        .domain = input,
        .codomain = output,
        .entries = &task_entries,
    }, solution);
}

test "decision search enumerates multiple carried vertex maps" {
    const input_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const protocol_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 1 } };
    const out0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out1 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 1 } };
    const empty_vertices = [_]topology.Vertex{};
    const input0 = [_]topology.Vertex{input_vertex};
    const protocol0 = [_]topology.Vertex{protocol_vertex};
    const output0 = [_]topology.Vertex{out0};
    const output1 = [_]topology.Vertex{out1};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input0 }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &output0 },
        .{ .vertices = &output1 },
    } };
    const execution_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} } },
    };
    const task_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{
            .{ .vertices = &output0 },
            .{ .vertices = &output1 },
        } } },
    };

    var result = try enumerateDecisionMaps(std.testing.allocator, .{
        .execution = .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .task = .{ .domain = input, .codomain = output, .entries = &task_entries },
    }, .{ .max_solutions = 8 });
    defer result.deinit();

    try std.testing.expectEqual(DecisionSearchStatus.satisfiable, result.status);
    try std.testing.expectEqual(@as(usize, 2), result.solutions.len);
}

test "decision search reports minimal local unsat core" {
    const in0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const in1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } };
    const p1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .protocol = 20 } };
    const out0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const empty_vertices = [_]topology.Vertex{};
    const input_edge = [_]topology.Vertex{ in0, in1 };
    const input0 = [_]topology.Vertex{in0};
    const input1 = [_]topology.Vertex{in1};
    const protocol_edge = [_]topology.Vertex{ p0, p1 };
    const protocol0 = [_]topology.Vertex{p0};
    const protocol1 = [_]topology.Vertex{p1};
    const output0 = [_]topology.Vertex{out0};
    const output1 = [_]topology.Vertex{out1};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_edge }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &output0 },
        .{ .vertices = &output1 },
    } };
    const execution_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} } },
        .{ .simplex = .{ .vertices = &input1 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol1 }} } },
        .{ .simplex = .{ .vertices = &input_edge }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} } },
    };
    const task_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output0 }} } },
        .{ .simplex = .{ .vertices = &input1 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output1 }} } },
        .{ .simplex = .{ .vertices = &input_edge }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{
            .{ .vertices = &output0 },
            .{ .vertices = &output1 },
        } } },
    };

    var result = try enumerateDecisionMaps(std.testing.allocator, .{
        .execution = .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .task = .{ .domain = input, .codomain = output, .entries = &task_entries },
    }, .{ .max_solutions = 4 });
    defer result.deinit();

    try std.testing.expectEqual(DecisionSearchStatus.unsatisfiable, result.status);
    try std.testing.expectEqual(UnsatReason.local_inconsistency, result.unsat_core.reason);
    try std.testing.expectEqual(@as(usize, 2), result.unsat_core.vertices.len);
}

test "decision search reports empty candidate unsat core" {
    const input_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const protocol_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 1 } };
    const out1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const empty_vertices = [_]topology.Vertex{};
    const input0 = [_]topology.Vertex{input_vertex};
    const protocol0 = [_]topology.Vertex{protocol_vertex};
    const output1 = [_]topology.Vertex{out1};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input0 }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &output1 }} };
    const execution_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} } },
    };
    const task_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output1 }} } },
    };

    var result = try solveDecisionMap(std.testing.allocator, .{
        .execution = .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .task = .{ .domain = input, .codomain = output, .entries = &task_entries },
    });
    defer result.deinit();

    try std.testing.expectEqual(DecisionSearchStatus.unsatisfiable, result.status);
    try std.testing.expectEqual(UnsatReason.empty_candidate_set, result.unsat_core.reason);
    try std.testing.expectEqual(@as(usize, 1), result.unsat_core.vertices.len);
}

test "decision search exports SMT friendly constraints" {
    const input_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const protocol_vertex = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 1 } };
    const out0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out1 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 1 } };
    const empty_vertices = [_]topology.Vertex{};
    const input0 = [_]topology.Vertex{input_vertex};
    const protocol0 = [_]topology.Vertex{protocol_vertex};
    const output0 = [_]topology.Vertex{out0};
    const output1 = [_]topology.Vertex{out1};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input0 }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &output0 },
        .{ .vertices = &output1 },
    } };
    const execution_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} } },
    };
    const task_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{
            .{ .vertices = &output0 },
            .{ .vertices = &output1 },
        } } },
    };

    var rendered = std.ArrayList(u8).init(std.testing.allocator);
    defer rendered.deinit();

    try writeSmtLib2(std.testing.allocator, .{
        .execution = .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .task = .{ .domain = input, .codomain = output, .entries = &task_entries },
    }, rendered.writer());

    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "(declare-const x_0_0 Bool)") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "(assert (or x_0_0 x_0_1))") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "(check-sat)") != null);
}
