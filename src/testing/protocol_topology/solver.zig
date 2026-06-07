const std = @import("std");

const decision_search = @import("decision_search.zig");
const topology = @import("topology.zig");

pub const SolverBackend = enum {
    z3,
    cvc5,
    custom,
};

pub const SolverStatus = enum {
    satisfiable,
    unsatisfiable,
    unknown,
};

pub const SolverOptions = struct {
    backend: SolverBackend = .z3,
    executable: ?[]const u8 = null,
    timeout_ms: ?u64 = 30_000,
    max_output_bytes: usize = 1024 * 1024,
    request_unsat_core: bool = true,

    fn executableName(self: SolverOptions) []const u8 {
        if (self.executable) |name| return name;
        return switch (self.backend) {
            .z3 => "z3",
            .cvc5 => "cvc5",
            .custom => "z3",
        };
    }
};

pub const OwnedSolverDecisionResult = struct {
    allocator: std.mem.Allocator,
    status: SolverStatus,
    solutions: []topology.OwnedVertexMap,
    unsat_core_names: [][]const u8,

    pub fn firstSolution(self: *const OwnedSolverDecisionResult) ?topology.VertexMap {
        if (self.solutions.len == 0) return null;
        return self.solutions[0].vertexMap();
    }

    pub fn deinit(self: *OwnedSolverDecisionResult) void {
        for (self.solutions) |*solution| solution.deinit();
        self.allocator.free(self.solutions);
        for (self.unsat_core_names) |name| self.allocator.free(name);
        self.allocator.free(self.unsat_core_names);
        self.solutions = &.{};
        self.unsat_core_names = &.{};
    }
};

const OwnedSolverProcessResult = struct {
    allocator: std.mem.Allocator,
    stdout: []u8,
    stderr: []u8,

    fn deinit(self: *OwnedSolverProcessResult) void {
        self.allocator.free(self.stdout);
        self.allocator.free(self.stderr);
        self.stdout = &.{};
        self.stderr = &.{};
    }
};

const SolverVariable = struct {
    source_index: usize,
    target_index: usize,
};

pub fn solveDecisionMap(
    allocator: std.mem.Allocator,
    problem: decision_search.DecisionSearchProblem,
    options: SolverOptions,
) !OwnedSolverDecisionResult {
    try problem.validate(allocator);

    const candidate_sets = try decision_search.buildDecisionCandidateSets(allocator, problem);
    defer decision_search.deinitDecisionCandidateSets(allocator, candidate_sets);

    var status_smt = std.ArrayList(u8).init(allocator);
    defer status_smt.deinit();
    try decision_search.writeSmtLib2WithOptions(allocator, problem, status_smt.writer(), .{
        .timeout_ms = options.timeout_ms,
    });

    var status_run = try runSolver(allocator, status_smt.items, options);
    defer status_run.deinit();

    const status = parseSolverStatus(status_run.stdout) orelse return error.SolverStatusMissing;
    return switch (status) {
        .satisfiable => solveSatModel(allocator, problem, candidate_sets, options),
        .unsatisfiable => solveUnsatCore(allocator, problem, options),
        .unknown => emptyResult(allocator, .unknown),
    };
}

fn solveSatModel(
    allocator: std.mem.Allocator,
    problem: decision_search.DecisionSearchProblem,
    candidate_sets: []const decision_search.CandidateSet,
    options: SolverOptions,
) !OwnedSolverDecisionResult {
    var model_smt = std.ArrayList(u8).init(allocator);
    defer model_smt.deinit();
    try decision_search.writeSmtLib2WithOptions(allocator, problem, model_smt.writer(), .{
        .timeout_ms = options.timeout_ms,
        .include_model_query = true,
    });

    var model_run = try runSolver(allocator, model_smt.items, options);
    defer model_run.deinit();

    const model_status = parseSolverStatus(model_run.stdout) orelse return error.SolverStatusMissing;
    if (model_status != .satisfiable) return error.SolverModelQueryNotSatisfiable;

    const assignment = try allocator.alloc(?usize, candidate_sets.len);
    defer allocator.free(assignment);
    @memset(assignment, null);
    try parseModelAssignments(model_run.stdout, candidate_sets, assignment);

    var solution = try decision_search.ownedMapFromCandidateAssignment(
        allocator,
        problem.execution.codomain,
        problem.task.codomain,
        candidate_sets,
        assignment,
    );
    errdefer solution.deinit();
    try topology.validateCarriedDecisionMap(allocator, problem.execution, problem.task, solution.vertexMap());

    const solutions = try allocator.alloc(topology.OwnedVertexMap, 1);
    errdefer allocator.free(solutions);
    solutions[0] = solution;

    return .{
        .allocator = allocator,
        .status = .satisfiable,
        .solutions = solutions,
        .unsat_core_names = try allocator.alloc([]const u8, 0),
    };
}

fn solveUnsatCore(
    allocator: std.mem.Allocator,
    problem: decision_search.DecisionSearchProblem,
    options: SolverOptions,
) !OwnedSolverDecisionResult {
    if (!options.request_unsat_core) return emptyResult(allocator, .unsatisfiable);

    var core_smt = std.ArrayList(u8).init(allocator);
    defer core_smt.deinit();
    try decision_search.writeSmtLib2WithOptions(allocator, problem, core_smt.writer(), .{
        .timeout_ms = options.timeout_ms,
        .produce_unsat_cores = true,
        .include_unsat_core_query = true,
    });

    var core_run = try runSolver(allocator, core_smt.items, options);
    defer core_run.deinit();

    const core_status = parseSolverStatus(core_run.stdout) orelse return error.SolverStatusMissing;
    if (core_status != .unsatisfiable) return error.SolverCoreQueryNotUnsatisfiable;

    return .{
        .allocator = allocator,
        .status = .unsatisfiable,
        .solutions = try allocator.alloc(topology.OwnedVertexMap, 0),
        .unsat_core_names = try parseUnsatCoreNames(allocator, core_run.stdout),
    };
}

fn emptyResult(allocator: std.mem.Allocator, status: SolverStatus) !OwnedSolverDecisionResult {
    return .{
        .allocator = allocator,
        .status = status,
        .solutions = try allocator.alloc(topology.OwnedVertexMap, 0),
        .unsat_core_names = try allocator.alloc([]const u8, 0),
    };
}

fn runSolver(
    allocator: std.mem.Allocator,
    smt_lib: []const u8,
    options: SolverOptions,
) !OwnedSolverProcessResult {
    const temp_path = try std.fmt.allocPrint(
        allocator,
        "/tmp/pcp-topology-solver-{d}.smt2",
        .{std.time.nanoTimestamp()},
    );
    defer allocator.free(temp_path);
    defer std.fs.deleteFileAbsolute(temp_path) catch {};

    {
        const file = try std.fs.createFileAbsolute(temp_path, .{ .exclusive = true });
        defer file.close();
        try file.writeAll(smt_lib);
    }

    var argv_buffer: [4][]const u8 = undefined;
    const argv = solverArgv(&argv_buffer, options, temp_path);
    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = argv,
        .max_output_bytes = options.max_output_bytes,
    });
    errdefer allocator.free(result.stdout);
    errdefer allocator.free(result.stderr);

    if (result.term != .Exited or result.term.Exited != 0) return error.SolverProcessFailed;
    return .{
        .allocator = allocator,
        .stdout = result.stdout,
        .stderr = result.stderr,
    };
}

fn solverArgv(buffer: *[4][]const u8, options: SolverOptions, temp_path: []const u8) []const []const u8 {
    const executable = options.executableName();
    return switch (options.backend) {
        .z3 => blk: {
            buffer[0] = executable;
            buffer[1] = "-smt2";
            buffer[2] = temp_path;
            break :blk buffer[0..3];
        },
        .cvc5 => blk: {
            buffer[0] = executable;
            buffer[1] = "--lang";
            buffer[2] = "smt2";
            buffer[3] = temp_path;
            break :blk buffer[0..4];
        },
        .custom => blk: {
            buffer[0] = executable;
            buffer[1] = temp_path;
            break :blk buffer[0..2];
        },
    };
}

fn parseSolverStatus(output: []const u8) ?SolverStatus {
    var cursor = TokenCursor{ .input = output };
    while (cursor.next()) |token| {
        if (std.mem.eql(u8, token, "sat")) return .satisfiable;
        if (std.mem.eql(u8, token, "unsat")) return .unsatisfiable;
        if (std.mem.eql(u8, token, "unknown")) return .unknown;
    }
    return null;
}

fn parseModelAssignments(
    output: []const u8,
    candidate_sets: []const decision_search.CandidateSet,
    assignment: []?usize,
) !void {
    var pending: ?SolverVariable = null;
    var cursor = TokenCursor{ .input = output };
    while (cursor.next()) |token| {
        if (parseSolverVariable(token)) |variable| {
            if (variable.source_index >= assignment.len) return error.SolverModelVariableOutOfRange;
            if (variable.target_index >= candidate_sets[variable.source_index].targets.len) {
                return error.SolverModelVariableOutOfRange;
            }
            pending = variable;
            continue;
        }

        if (std.mem.eql(u8, token, "true")) {
            const variable = pending orelse continue;
            if (assignment[variable.source_index] != null) return error.SolverModelDuplicateAssignment;
            assignment[variable.source_index] = variable.target_index;
            pending = null;
            continue;
        }
        if (std.mem.eql(u8, token, "false")) {
            pending = null;
        }
    }

    for (assignment) |candidate| {
        if (candidate == null) return error.SolverModelIncompleteAssignment;
    }
}

fn parseUnsatCoreNames(allocator: std.mem.Allocator, output: []const u8) ![][]const u8 {
    var names = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (names.items) |name| allocator.free(name);
        names.deinit();
    }

    var cursor = TokenCursor{ .input = output };
    while (cursor.next()) |token| {
        if (!std.mem.startsWith(u8, token, "c_")) continue;
        _ = parseCoreName(token) catch continue;
        try names.append(try allocator.dupe(u8, token));
    }

    return names.toOwnedSlice();
}

fn parseSolverVariable(token: []const u8) ?SolverVariable {
    if (!std.mem.startsWith(u8, token, "x_")) return null;
    const rest = token[2..];
    const split = std.mem.indexOfScalar(u8, rest, '_') orelse return null;
    const source = std.fmt.parseInt(usize, rest[0..split], 10) catch return null;
    const target = std.fmt.parseInt(usize, rest[split + 1 ..], 10) catch return null;
    return .{ .source_index = source, .target_index = target };
}

fn parseCoreName(token: []const u8) !usize {
    if (!std.mem.startsWith(u8, token, "c_")) return error.InvalidSolverCoreName;
    return std.fmt.parseInt(usize, token[2..], 10);
}

const TokenCursor = struct {
    input: []const u8,
    index: usize = 0,

    fn next(self: *TokenCursor) ?[]const u8 {
        while (self.index < self.input.len and isDelimiter(self.input[self.index])) {
            self.index += 1;
        }
        if (self.index >= self.input.len) return null;

        const start = self.index;
        while (self.index < self.input.len and !isDelimiter(self.input[self.index])) {
            self.index += 1;
        }
        return self.input[start..self.index];
    }

    fn isDelimiter(byte: u8) bool {
        return switch (byte) {
            ' ', '\n', '\t', '\r', '(', ')' => true,
            else => false,
        };
    }
};

fn solverAvailable(allocator: std.mem.Allocator, options: SolverOptions) bool {
    var argv_buffer: [2][]const u8 = undefined;
    const executable = options.executableName();
    const argv: []const []const u8 = switch (options.backend) {
        .z3 => blk: {
            argv_buffer[0] = executable;
            argv_buffer[1] = "-version";
            break :blk argv_buffer[0..2];
        },
        .cvc5, .custom => blk: {
            argv_buffer[0] = executable;
            argv_buffer[1] = "--version";
            break :blk argv_buffer[0..2];
        },
    };
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = argv,
        .max_output_bytes = 16 * 1024,
    }) catch return false;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return result.term == .Exited and result.term.Exited == 0;
}

test "solver parser reads statuses assignments and unsat cores" {
    try std.testing.expectEqual(SolverStatus.satisfiable, parseSolverStatus("sat\n").?);
    try std.testing.expectEqual(SolverStatus.unsatisfiable, parseSolverStatus("unsat\n(c_0)").?);
    try std.testing.expectEqual(SolverStatus.unknown, parseSolverStatus("unknown\n").?);

    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 0 } };
    const out0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out1 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 1 } };
    var targets = [_]topology.Vertex{ out0, out1 };
    const candidate_sets = [_]decision_search.CandidateSet{.{
        .source = p0,
        .targets = &targets,
    }};
    var assignment = [_]?usize{null};
    try parseModelAssignments(
        \\sat
        \\((x_0_0 false)
        \\ (x_0_1 true))
    , &candidate_sets, &assignment);
    try std.testing.expectEqual(@as(?usize, 1), assignment[0]);

    const names = try parseUnsatCoreNames(std.testing.allocator, "unsat\n(c_0 c_12)\n");
    defer {
        for (names) |name| std.testing.allocator.free(name);
        std.testing.allocator.free(names);
    }
    try std.testing.expectEqual(@as(usize, 2), names.len);
    try std.testing.expectEqualStrings("c_0", names[0]);
    try std.testing.expectEqualStrings("c_12", names[1]);
}

test "solver SMT-LIB export supports model and unsat-core queries" {
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

    try decision_search.writeSmtLib2WithOptions(std.testing.allocator, .{
        .execution = .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .task = .{ .domain = input, .codomain = output, .entries = &task_entries },
    }, rendered.writer(), .{
        .timeout_ms = 1000,
        .produce_unsat_cores = true,
        .include_model_query = true,
        .include_unsat_core_query = true,
    });

    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "(set-option :timeout 1000)") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, ":named c_") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "(get-value (x_0_0 x_0_1))") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered.items, "(get-unsat-core)") != null);
}

test "z3 solver finds and natively rechecks a carried decision map" {
    const options = SolverOptions{};
    if (!solverAvailable(std.testing.allocator, options)) return error.SkipZigTest;

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
    const output_edge = [_]topology.Vertex{ out0, out1 };
    const output0 = [_]topology.Vertex{out0};
    const output1 = [_]topology.Vertex{out1};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_edge }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &output_edge }} };
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

    const problem = decision_search.DecisionSearchProblem{
        .execution = .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .task = .{ .domain = input, .codomain = output, .entries = &task_entries },
    };

    var native = try decision_search.solveDecisionMap(std.testing.allocator, problem);
    defer native.deinit();
    var solved = try solveDecisionMap(std.testing.allocator, problem, options);
    defer solved.deinit();

    try std.testing.expectEqual(decision_search.DecisionSearchStatus.satisfiable, native.status);
    try std.testing.expectEqual(SolverStatus.satisfiable, solved.status);
    try std.testing.expectEqual(@as(usize, 1), solved.solutions.len);
    try topology.validateCarriedDecisionMap(std.testing.allocator, problem.execution, problem.task, solved.firstSolution().?);
}

test "z3 solver reports unsat and imports an unsat core" {
    const options = SolverOptions{};
    if (!solverAvailable(std.testing.allocator, options)) return error.SkipZigTest;

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

    var solved = try solveDecisionMap(std.testing.allocator, .{
        .execution = .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .task = .{ .domain = input, .codomain = output, .entries = &task_entries },
    }, options);
    defer solved.deinit();

    try std.testing.expectEqual(SolverStatus.unsatisfiable, solved.status);
    try std.testing.expectEqual(@as(usize, 0), solved.solutions.len);
    try std.testing.expect(solved.unsat_core_names.len != 0);
}
