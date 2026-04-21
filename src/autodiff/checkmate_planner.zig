const std = @import("std");
const glpk = @cImport({
    @cInclude("glpk.h");
});

const Allocator = std.mem.Allocator;

pub const SolverBackend = enum {
    milp_glpk,
    exact_bounded,
    greedy_stage_aware,
    milp_external,
};

pub const MilpExternalOptions = struct {
    solver_command: ?[]const u8 = null,
    lp_path: ?[]const u8 = null,
    solution_path: ?[]const u8 = null,
    keep_artifacts: bool = false,
};

pub const SolveOptions = struct {
    backend: SolverBackend = .exact_bounded,
    milp_external: MilpExternalOptions = .{},
};

pub const Candidate = struct {
    bytes: u64,
    weighted_recompute_cost: f64,
    op_index: usize,
    last_forward_use_index: usize,
    producer_candidate_indices: []usize,
    user_candidate_indices: []usize,
    use_stage_indices: []usize,
    backward_use_stage_indices: []usize,
};

pub const PlanningProblem = struct {
    allocator: Allocator,
    candidates: []Candidate,
    candidate_metadata_indices: []usize,
    base_stage_bytes: []u64,
    base_retained_bytes: u64,
    total_retained_bytes_before: u64,
    budget: u64,
    forward_stage_count: usize,
    backward_stage_count: usize,

    pub fn deinit(self: *PlanningProblem) void {
        for (self.candidates) |candidate| {
            self.allocator.free(candidate.producer_candidate_indices);
            self.allocator.free(candidate.user_candidate_indices);
            self.allocator.free(candidate.use_stage_indices);
            self.allocator.free(candidate.backward_use_stage_indices);
        }
        self.allocator.free(self.candidates);
        self.allocator.free(self.candidate_metadata_indices);
        self.allocator.free(self.base_stage_bytes);
    }
};

pub const DependencyEdge = struct {
    producer_var_index: usize,
    user_var_index: usize,
};

pub const LinearTerm = struct {
    variable_index: usize,
    coefficient: f64,
};

pub const LinearConstraint = struct {
    allocator: Allocator,
    name: []const u8,
    terms: []LinearTerm,
    rhs: f64,
    sense: Sense,

    pub const Sense = enum {
        less_equal,
        greater_equal,
    };

    pub fn deinit(self: *LinearConstraint) void {
        self.allocator.free(self.name);
        self.allocator.free(self.terms);
    }
};

pub const MilpModel = struct {
    allocator: Allocator,
    variable_names: [][]const u8,
    objective_coefficients: []f64,
    objective_constant: f64,
    keep_variable_count: usize,
    recompute_variables: []RecomputeVariable,
    stage_constraints: []LinearConstraint,
    availability_constraints: []LinearConstraint,
    dependency_constraints: []LinearConstraint,
    dependency_edges: []DependencyEdge,

    pub const RecomputeVariable = struct {
        candidate_index: usize,
        stage_index: usize,
    };

    pub fn deinit(self: *MilpModel) void {
        for (self.variable_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.variable_names);
        self.allocator.free(self.objective_coefficients);
        self.allocator.free(self.recompute_variables);
        for (self.stage_constraints) |*constraint| {
            constraint.deinit();
        }
        self.allocator.free(self.stage_constraints);
        for (self.availability_constraints) |*constraint| {
            constraint.deinit();
        }
        self.allocator.free(self.availability_constraints);
        for (self.dependency_constraints) |*constraint| {
            constraint.deinit();
        }
        self.allocator.free(self.dependency_constraints);
        self.allocator.free(self.dependency_edges);
    }

    pub fn writeLp(self: *const MilpModel, writer: anytype) !void {
        try writer.writeAll("Minimize\n");
        try writer.writeAll(" obj:");
        var first_term = true;
        for (self.objective_coefficients, 0..) |coefficient, variable_idx| {
            if (coefficient == 0.0) continue;
            if (!first_term) {
                try writer.writeAll(" ");
            } else {
                try writer.writeAll(" ");
                first_term = false;
            }
            try writeSignedTerm(writer, coefficient, self.variable_names[variable_idx]);
        }
        if (first_term) {
            try writer.writeAll(" 0");
        }
        try writer.writeAll("\nSubject To\n");
        for (self.stage_constraints) |constraint| {
            try writeConstraint(writer, self.variable_names, constraint);
        }
        for (self.availability_constraints) |constraint| {
            try writeConstraint(writer, self.variable_names, constraint);
        }
        for (self.dependency_constraints) |constraint| {
            try writeConstraint(writer, self.variable_names, constraint);
        }
        if (self.dependency_edges.len > 0) {
            try writer.writeAll("\\ Dependency edges for future precedence constraints:\n");
            for (self.dependency_edges) |edge| {
                try writer.print(
                    "\\ keep_{d} depends_on keep_{d}\n",
                    .{ edge.user_var_index, edge.producer_var_index },
                );
            }
        }
        try writer.writeAll("Binary\n");
        for (self.variable_names) |name| {
            try writer.print(" {s}\n", .{name});
        }
        try writer.writeAll("End\n");
    }
};

pub const ExactPlanResult = struct {
    keep_mask: []bool,
    total_retained_bytes: u64,
    peak_stage_bytes: u64,
    total_recompute_cost: f64,

    pub fn deinit(self: *ExactPlanResult, allocator: Allocator) void {
        allocator.free(self.keep_mask);
    }
};

const MAX_EXACT_CANDIDATES: usize = 24;

pub fn buildMilpModel(allocator: Allocator, problem: *const PlanningProblem) !MilpModel {
    const keep_variable_count = problem.candidates.len;
    var recompute_var_count: usize = 0;
    for (problem.candidates) |candidate| {
        recompute_var_count += candidate.last_forward_use_index - candidate.op_index + 1;
        recompute_var_count += candidate.backward_use_stage_indices.len;
    }
    const total_variable_count = keep_variable_count + recompute_var_count;

    var variable_names = try allocator.alloc([]const u8, total_variable_count);
    errdefer allocator.free(variable_names);
    for (problem.candidates, 0..) |_, candidate_idx| {
        variable_names[candidate_idx] = try std.fmt.allocPrintZ(allocator, "keep_{d}", .{candidate_idx});
    }
    errdefer {
        for (variable_names) |name| {
            allocator.free(name);
        }
    }

    const objective_coefficients = try allocator.alloc(f64, total_variable_count);
    errdefer allocator.free(objective_coefficients);
    @memset(objective_coefficients, 0.0);
    const objective_constant: f64 = 0.0;
    const recompute_variables = try allocator.alloc(MilpModel.RecomputeVariable, recompute_var_count);
    errdefer allocator.free(recompute_variables);

    var next_var_idx = keep_variable_count;
    for (problem.candidates, 0..) |candidate, candidate_idx| {
        for (candidate.op_index..candidate.last_forward_use_index + 1) |stage_idx| {
            recompute_variables[next_var_idx - keep_variable_count] = .{
                .candidate_index = candidate_idx,
                .stage_index = stage_idx,
            };
            variable_names[next_var_idx] = try std.fmt.allocPrintZ(
                allocator,
                "remat_{d}_{d}",
                .{ candidate_idx, stage_idx },
            );
            objective_coefficients[next_var_idx] = candidate.weighted_recompute_cost;
            next_var_idx += 1;
        }
        for (candidate.backward_use_stage_indices) |stage_idx| {
            recompute_variables[next_var_idx - keep_variable_count] = .{
                .candidate_index = candidate_idx,
                .stage_index = stage_idx,
            };
            variable_names[next_var_idx] = try std.fmt.allocPrintZ(
                allocator,
                "remat_{d}_{d}",
                .{ candidate_idx, stage_idx },
            );
            objective_coefficients[next_var_idx] = candidate.weighted_recompute_cost;
            next_var_idx += 1;
        }
    }

    var stage_constraints = std.ArrayList(LinearConstraint).init(allocator);
    errdefer {
        for (stage_constraints.items) |*constraint| {
            constraint.deinit();
        }
        stage_constraints.deinit();
    }

    for (problem.base_stage_bytes, 0..) |base_bytes, stage_idx| {
        var terms = std.ArrayList(LinearTerm).init(allocator);
        errdefer terms.deinit();
        for (problem.candidates, 0..) |candidate, candidate_idx| {
            if (candidateKeptAtStage(candidate, stage_idx, problem.forward_stage_count)) {
                try terms.append(.{
                    .variable_index = candidate_idx,
                    .coefficient = @floatFromInt(candidate.bytes),
                });
            }
        }
        for (recompute_variables, 0..) |recompute_var, recompute_idx| {
            if (recompute_var.stage_index == stage_idx) {
                try terms.append(.{
                    .variable_index = keep_variable_count + recompute_idx,
                    .coefficient = @floatFromInt(problem.candidates[recompute_var.candidate_index].bytes),
                });
            }
        }
        const rhs = @as(f64, @floatFromInt(problem.budget)) - @as(f64, @floatFromInt(base_bytes));
        try stage_constraints.append(.{
            .allocator = allocator,
            .name = try std.fmt.allocPrintZ(allocator, "stage_{d}", .{stage_idx}),
            .terms = try terms.toOwnedSlice(),
            .rhs = rhs,
            .sense = .less_equal,
        });
    }

    var availability_constraints = std.ArrayList(LinearConstraint).init(allocator);
    errdefer {
        for (availability_constraints.items) |*constraint| {
            constraint.deinit();
        }
        availability_constraints.deinit();
    }
    for (problem.candidates, 0..) |candidate, candidate_idx| {
        for (candidate.use_stage_indices) |use_stage_idx| {
            var terms = std.ArrayList(LinearTerm).init(allocator);
            errdefer terms.deinit();
            try terms.append(.{
                .variable_index = candidate_idx,
                .coefficient = 1.0,
            });
            for (recompute_variables, 0..) |recompute_var, recompute_idx| {
                if (recompute_var.candidate_index == candidate_idx and recompute_var.stage_index <= use_stage_idx) {
                    try terms.append(.{
                        .variable_index = keep_variable_count + recompute_idx,
                        .coefficient = 1.0,
                    });
                }
            }
            try availability_constraints.append(.{
                .allocator = allocator,
                .name = try std.fmt.allocPrintZ(allocator, "avail_{d}_{d}", .{ candidate_idx, use_stage_idx }),
                .terms = try terms.toOwnedSlice(),
                .rhs = 1.0,
                .sense = .greater_equal,
            });
        }
        for (candidate.backward_use_stage_indices) |use_stage_idx| {
            var terms = std.ArrayList(LinearTerm).init(allocator);
            errdefer terms.deinit();
            try terms.append(.{
                .variable_index = candidate_idx,
                .coefficient = 1.0,
            });
            for (recompute_variables, 0..) |recompute_var, recompute_idx| {
                if (recompute_var.candidate_index == candidate_idx and recompute_var.stage_index <= use_stage_idx) {
                    try terms.append(.{
                        .variable_index = keep_variable_count + recompute_idx,
                        .coefficient = 1.0,
                    });
                }
            }
            try availability_constraints.append(.{
                .allocator = allocator,
                .name = try std.fmt.allocPrintZ(allocator, "avail_{d}_{d}", .{ candidate_idx, use_stage_idx }),
                .terms = try terms.toOwnedSlice(),
                .rhs = 1.0,
                .sense = .greater_equal,
            });
        }
    }

    var dependency_constraints = std.ArrayList(LinearConstraint).init(allocator);
    errdefer {
        for (dependency_constraints.items) |*constraint| {
            constraint.deinit();
        }
        dependency_constraints.deinit();
    }
    for (recompute_variables, 0..) |recompute_var, recompute_idx| {
        const candidate = problem.candidates[recompute_var.candidate_index];
        for (candidate.producer_candidate_indices) |producer_idx| {
            var terms = std.ArrayList(LinearTerm).init(allocator);
            errdefer terms.deinit();
            try terms.append(.{
                .variable_index = keep_variable_count + recompute_idx,
                .coefficient = 1.0,
            });
            try terms.append(.{
                .variable_index = producer_idx,
                .coefficient = -1.0,
            });
            for (recompute_variables, 0..) |producer_recompute_var, producer_recompute_idx| {
                if (producer_recompute_var.candidate_index == producer_idx and
                    producer_recompute_var.stage_index <= recompute_var.stage_index)
                {
                    try terms.append(.{
                        .variable_index = keep_variable_count + producer_recompute_idx,
                        .coefficient = -1.0,
                    });
                }
            }
            try dependency_constraints.append(.{
                .allocator = allocator,
                .name = try std.fmt.allocPrintZ(
                    allocator,
                    "dep_{d}_{d}_{d}",
                    .{ recompute_var.candidate_index, recompute_var.stage_index, producer_idx },
                ),
                .terms = try terms.toOwnedSlice(),
                .rhs = 0.0,
                .sense = .less_equal,
            });
        }
    }

    var dependency_edges = std.ArrayList(DependencyEdge).init(allocator);
    defer dependency_edges.deinit();
    for (problem.candidates, 0..) |candidate, user_idx| {
        for (candidate.producer_candidate_indices) |producer_idx| {
            try dependency_edges.append(.{
                .producer_var_index = producer_idx,
                .user_var_index = user_idx,
            });
        }
    }

    return .{
        .allocator = allocator,
        .variable_names = variable_names,
        .objective_coefficients = objective_coefficients,
        .objective_constant = objective_constant,
        .keep_variable_count = keep_variable_count,
        .recompute_variables = recompute_variables,
        .stage_constraints = try stage_constraints.toOwnedSlice(),
        .availability_constraints = try availability_constraints.toOwnedSlice(),
        .dependency_constraints = try dependency_constraints.toOwnedSlice(),
        .dependency_edges = try dependency_edges.toOwnedSlice(),
    };
}

pub fn solveExactSmallGraph(
    allocator: Allocator,
    candidates: []const Candidate,
    base_stage_bytes: []const u64,
    budget: u64,
) !ExactPlanResult {
    if (base_stage_bytes.len == 0) {
        const keep_mask = try allocator.alloc(bool, candidates.len);
        @memset(keep_mask, false);
        return .{
            .keep_mask = keep_mask,
            .total_retained_bytes = 0,
            .peak_stage_bytes = 0,
            .total_recompute_cost = 0.0,
        };
    }

    if (candidates.len > MAX_EXACT_CANDIDATES) {
        return error.CheckmateGraphTooLarge;
    }

    for (base_stage_bytes) |stage_bytes| {
        if (stage_bytes > budget) {
            return error.NoFeasibleRematPlan;
        }
    }

    var best_mask: u64 = 0;
    var best_cost: f64 = std.math.inf(f64);
    var best_peak: u64 = std.math.maxInt(u64);
    var found: bool = false;

    const subset_count: u64 = @as(u64, 1) << @intCast(candidates.len);
    var mask: u64 = 0;
    while (mask < subset_count) : (mask += 1) {
        const stage_eval = evaluateMask(candidates, base_stage_bytes, mask);
        if (stage_eval.peak_stage_bytes > budget) continue;

        if (!found or stage_eval.total_recompute_cost < best_cost or
            (stage_eval.total_recompute_cost == best_cost and stage_eval.peak_stage_bytes < best_peak) or
            (stage_eval.total_recompute_cost == best_cost and stage_eval.peak_stage_bytes == best_peak and
            stageWeightedKeepSum(candidates, mask) > stageWeightedKeepSum(candidates, best_mask)))
        {
            found = true;
            best_mask = mask;
            best_cost = stage_eval.total_recompute_cost;
            best_peak = stage_eval.peak_stage_bytes;
        }
    }

    if (!found) {
        return error.NoFeasibleRematPlan;
    }

    const keep_mask = try allocator.alloc(bool, candidates.len);
    for (candidates, 0..) |_, candidate_idx| {
        keep_mask[candidate_idx] = ((best_mask >> @intCast(candidate_idx)) & 1) == 1;
    }

    const final_eval = evaluateMask(candidates, base_stage_bytes, best_mask);
    return .{
        .keep_mask = keep_mask,
        .total_retained_bytes = final_eval.total_retained_bytes,
        .peak_stage_bytes = final_eval.peak_stage_bytes,
        .total_recompute_cost = final_eval.total_recompute_cost,
    };
}

pub fn solvePlanningProblemExact(
    allocator: Allocator,
    problem: *const PlanningProblem,
) !ExactPlanResult {
    if (problem.base_stage_bytes.len == 0) {
        const keep_mask = try allocator.alloc(bool, problem.candidates.len);
        @memset(keep_mask, false);
        return .{
            .keep_mask = keep_mask,
            .total_retained_bytes = 0,
            .peak_stage_bytes = 0,
            .total_recompute_cost = 0.0,
        };
    }

    if (problem.candidates.len > MAX_EXACT_CANDIDATES) {
        return error.CheckmateGraphTooLarge;
    }

    for (problem.base_stage_bytes) |stage_bytes| {
        if (stage_bytes > problem.budget) {
            return error.NoFeasibleRematPlan;
        }
    }

    var best_mask: u64 = 0;
    var best_cost: f64 = std.math.inf(f64);
    var best_peak: u64 = std.math.maxInt(u64);
    var found: bool = false;

    const subset_count: u64 = @as(u64, 1) << @intCast(problem.candidates.len);
    var mask: u64 = 0;
    while (mask < subset_count) : (mask += 1) {
        const stage_eval = evaluateProblemMask(problem, mask);
        if (stage_eval.peak_stage_bytes > problem.budget) continue;

        if (!found or stage_eval.total_recompute_cost < best_cost or
            (stage_eval.total_recompute_cost == best_cost and stage_eval.peak_stage_bytes < best_peak) or
            (stage_eval.total_recompute_cost == best_cost and stage_eval.peak_stage_bytes == best_peak and
            stageWeightedKeepSum(problem.candidates, mask) > stageWeightedKeepSum(problem.candidates, best_mask)))
        {
            found = true;
            best_mask = mask;
            best_cost = stage_eval.total_recompute_cost;
            best_peak = stage_eval.peak_stage_bytes;
        }
    }

    if (!found) {
        return error.NoFeasibleRematPlan;
    }

    const keep_mask = try allocator.alloc(bool, problem.candidates.len);
    for (problem.candidates, 0..) |_, candidate_idx| {
        keep_mask[candidate_idx] = ((best_mask >> @intCast(candidate_idx)) & 1) == 1;
    }

    const final_eval = evaluateProblemMask(problem, best_mask);
    return .{
        .keep_mask = keep_mask,
        .total_retained_bytes = final_eval.total_retained_bytes,
        .peak_stage_bytes = final_eval.peak_stage_bytes,
        .total_recompute_cost = final_eval.total_recompute_cost,
    };
}

pub fn solvePlanningProblem(
    allocator: Allocator,
    problem: *const PlanningProblem,
    options: SolveOptions,
) !ExactPlanResult {
    return switch (options.backend) {
        .milp_glpk => solvePlanningProblemGlpk(allocator, problem),
        .exact_bounded => solvePlanningProblemExact(allocator, problem),
        .greedy_stage_aware => solvePlanningProblemGreedy(allocator, problem),
        .milp_external => solvePlanningProblemMilpExternal(allocator, problem, options.milp_external),
    };
}

pub fn solvePlanningProblemGlpk(
    allocator: Allocator,
    problem: *const PlanningProblem,
) !ExactPlanResult {
    if (problem.base_stage_bytes.len == 0) {
        const keep_mask = try allocator.alloc(bool, problem.candidates.len);
        @memset(keep_mask, false);
        return .{
            .keep_mask = keep_mask,
            .total_retained_bytes = 0,
            .peak_stage_bytes = 0,
            .total_recompute_cost = 0.0,
        };
    }

    for (problem.base_stage_bytes) |stage_bytes| {
        if (stage_bytes > problem.budget) {
            return error.NoFeasibleRematPlan;
        }
    }

    var model = try buildMilpModel(allocator, problem);
    defer model.deinit();

    const lp = glpk.glp_create_prob() orelse return error.GlpkInitFailed;
    defer glpk.glp_delete_prob(lp);

    glpk.glp_set_obj_dir(lp, glpk.GLP_MIN);

    const row_count_total = model.stage_constraints.len + model.availability_constraints.len + model.dependency_constraints.len;
    const row_count: c_int = @intCast(row_count_total);
    const col_count: c_int = @intCast(model.variable_names.len);

    _ = glpk.glp_add_rows(lp, row_count);
    var row_idx: usize = 0;
    for (model.stage_constraints) |constraint| {
        try configureGlpkRow(lp, row_idx + 1, constraint);
        row_idx += 1;
    }
    for (model.availability_constraints) |constraint| {
        try configureGlpkRow(lp, row_idx + 1, constraint);
        row_idx += 1;
    }
    for (model.dependency_constraints) |constraint| {
        try configureGlpkRow(lp, row_idx + 1, constraint);
        row_idx += 1;
    }

    _ = glpk.glp_add_cols(lp, col_count);
    for (model.variable_names, 0..) |name, col_idx| {
        const is_keep_var = col_idx < model.keep_variable_count;
        glpk.glp_set_col_name(lp, @intCast(col_idx + 1), model.variable_names[col_idx].ptr);
        glpk.glp_set_col_kind(lp, @intCast(col_idx + 1), glpk.GLP_BV);
        glpk.glp_set_col_bnds(lp, @intCast(col_idx + 1), glpk.GLP_DB, 0.0, 1.0);
        glpk.glp_set_obj_coef(lp, @intCast(col_idx + 1), model.objective_coefficients[col_idx]);
        _ = name;
        _ = is_keep_var;
    }

    var nonzeros: usize = 0;
    for (model.stage_constraints) |constraint| {
        nonzeros += constraint.terms.len;
    }
    for (model.availability_constraints) |constraint| {
        nonzeros += constraint.terms.len;
    }
    for (model.dependency_constraints) |constraint| {
        nonzeros += constraint.terms.len;
    }

    const ia = try allocator.alloc(c_int, nonzeros + 1);
    defer allocator.free(ia);
    const ja = try allocator.alloc(c_int, nonzeros + 1);
    defer allocator.free(ja);
    const ar = try allocator.alloc(f64, nonzeros + 1);
    defer allocator.free(ar);

    var nz_index: usize = 1;
    row_idx = 0;
    for (model.stage_constraints) |constraint| {
        appendConstraintTermsToGlpkMatrix(constraint, row_idx + 1, ia, ja, ar, &nz_index);
        row_idx += 1;
    }
    for (model.availability_constraints) |constraint| {
        appendConstraintTermsToGlpkMatrix(constraint, row_idx + 1, ia, ja, ar, &nz_index);
        row_idx += 1;
    }
    for (model.dependency_constraints) |constraint| {
        appendConstraintTermsToGlpkMatrix(constraint, row_idx + 1, ia, ja, ar, &nz_index);
        row_idx += 1;
    }

    glpk.glp_load_matrix(lp, @intCast(nonzeros), ia.ptr, ja.ptr, ar.ptr);

    var smcp: glpk.glp_smcp = undefined;
    glpk.glp_init_smcp(&smcp);
    smcp.msg_lev = glpk.GLP_MSG_OFF;
    const simplex_rc = glpk.glp_simplex(lp, &smcp);
    if (simplex_rc != 0) {
        return error.GlpkSimplexFailed;
    }

    var iocp: glpk.glp_iocp = undefined;
    glpk.glp_init_iocp(&iocp);
    iocp.msg_lev = glpk.GLP_MSG_OFF;
    iocp.presolve = glpk.GLP_ON;
    const intopt_rc = glpk.glp_intopt(lp, &iocp);
    if (intopt_rc != 0) {
        return error.GlpkIntoptFailed;
    }

    const mip_status = glpk.glp_mip_status(lp);
    if (mip_status != glpk.GLP_OPT) {
        if (mip_status == glpk.GLP_NOFEAS) return error.NoFeasibleRematPlan;
        return error.GlpkNoOptimalSolution;
    }

    const keep_mask = try allocator.alloc(bool, problem.candidates.len);
    errdefer allocator.free(keep_mask);
    for (problem.candidates, 0..) |_, candidate_idx| {
        const value = glpk.glp_mip_col_val(lp, @intCast(candidate_idx + 1));
        keep_mask[candidate_idx] = value >= 0.5;
    }
    const recompute_mask = try allocator.alloc(bool, model.recompute_variables.len);
    defer allocator.free(recompute_mask);
    for (model.recompute_variables, 0..) |_, recompute_idx| {
        const value = glpk.glp_mip_col_val(lp, @intCast(model.keep_variable_count + recompute_idx + 1));
        recompute_mask[recompute_idx] = value >= 0.5;
    }

    const final_eval = evaluateScheduledSolution(problem, &model, keep_mask, recompute_mask);
    if (final_eval.peak_stage_bytes > problem.budget) {
        return error.InvalidMilpSolution;
    }

    return .{
        .keep_mask = keep_mask,
        .total_retained_bytes = final_eval.total_retained_bytes,
        .peak_stage_bytes = final_eval.peak_stage_bytes,
        .total_recompute_cost = final_eval.total_recompute_cost,
    };
}

fn configureGlpkRow(lp: ?*glpk.glp_prob, row_idx: usize, constraint: LinearConstraint) !void {
    glpk.glp_set_row_name(lp, @intCast(row_idx), constraint.name.ptr);
    switch (constraint.sense) {
        .less_equal => {
            if (constraint.rhs < 0.0) return error.NoFeasibleRematPlan;
            glpk.glp_set_row_bnds(lp, @intCast(row_idx), glpk.GLP_UP, 0.0, constraint.rhs);
        },
        .greater_equal => {
            glpk.glp_set_row_bnds(lp, @intCast(row_idx), glpk.GLP_LO, constraint.rhs, 0.0);
        },
    }
}

fn appendConstraintTermsToGlpkMatrix(
    constraint: LinearConstraint,
    row_idx: usize,
    ia: []c_int,
    ja: []c_int,
    ar: []f64,
    nz_index: *usize,
) void {
    for (constraint.terms) |term| {
        ia[nz_index.*] = @intCast(row_idx);
        ja[nz_index.*] = @intCast(term.variable_index + 1);
        ar[nz_index.*] = term.coefficient;
        nz_index.* += 1;
    }
}

pub fn solvePlanningProblemMilpExternal(
    allocator: Allocator,
    problem: *const PlanningProblem,
    options: MilpExternalOptions,
) !ExactPlanResult {
    var model = try buildMilpModel(allocator, problem);
    defer model.deinit();

    const lp_path = try resolveArtifactPath(allocator, options.lp_path, ".lp");
    defer allocator.free(lp_path);
    const solution_path = try resolveArtifactPath(allocator, options.solution_path, ".sol");
    defer allocator.free(solution_path);

    {
        const lp_file = try std.fs.createFileAbsolute(lp_path, .{ .truncate = true });
        defer lp_file.close();
        try model.writeLp(lp_file.writer());
    }

    errdefer if (!options.keep_artifacts and options.lp_path == null) {
        std.fs.deleteFileAbsolute(lp_path) catch {};
    };
    errdefer if (!options.keep_artifacts and options.solution_path == null) {
        std.fs.deleteFileAbsolute(solution_path) catch {};
    };

    const solver_command = options.solver_command orelse return error.MilpSolverUnavailable;
    try runExternalMilpSolver(allocator, solver_command, lp_path, solution_path);

    const keep_mask = try parseExternalMilpSolution(allocator, problem, solution_path);
    errdefer allocator.free(keep_mask);
    const final_eval = evaluateKeepPlan(problem, keep_mask);

    if (!options.keep_artifacts and options.lp_path == null) {
        std.fs.deleteFileAbsolute(lp_path) catch {};
    }
    if (!options.keep_artifacts and options.solution_path == null) {
        std.fs.deleteFileAbsolute(solution_path) catch {};
    }

    return .{
        .keep_mask = keep_mask,
        .total_retained_bytes = final_eval.total_retained_bytes,
        .peak_stage_bytes = final_eval.peak_stage_bytes,
        .total_recompute_cost = final_eval.total_recompute_cost,
    };
}

pub fn solvePlanningProblemGreedy(
    allocator: Allocator,
    problem: *const PlanningProblem,
) !ExactPlanResult {
    if (problem.base_stage_bytes.len == 0) {
        const keep_mask = try allocator.alloc(bool, problem.candidates.len);
        @memset(keep_mask, false);
        return .{
            .keep_mask = keep_mask,
            .total_retained_bytes = 0,
            .peak_stage_bytes = 0,
            .total_recompute_cost = 0.0,
        };
    }

    for (problem.base_stage_bytes) |stage_bytes| {
        if (stage_bytes > problem.budget) {
            return error.NoFeasibleRematPlan;
        }
    }

    const keep_mask = try allocator.alloc(bool, problem.candidates.len);
    errdefer allocator.free(keep_mask);
    @memset(keep_mask, true);

    const candidate_order = try allocator.alloc(usize, problem.candidates.len);
    defer allocator.free(candidate_order);
    for (candidate_order, 0..) |*entry, idx| {
        entry.* = idx;
    }

    std.sort.block(usize, candidate_order, problem.candidates, compareGreedyDropScore);

    for (candidate_order) |candidate_idx| {
        const current_eval = evaluateKeepPlan(problem, keep_mask);
        if (current_eval.peak_stage_bytes <= problem.budget) break;
        keep_mask[candidate_idx] = false;
    }

    const final_eval = evaluateKeepPlan(problem, keep_mask);
    if (final_eval.peak_stage_bytes > problem.budget) {
        return error.NoFeasibleRematPlan;
    }

    return .{
        .keep_mask = keep_mask,
        .total_retained_bytes = final_eval.total_retained_bytes,
        .peak_stage_bytes = final_eval.peak_stage_bytes,
        .total_recompute_cost = final_eval.total_recompute_cost,
    };
}

const StageEval = struct {
    total_retained_bytes: u64,
    peak_stage_bytes: u64,
    total_recompute_cost: f64,
};

fn evaluateMask(candidates: []const Candidate, base_stage_bytes: []const u64, mask: u64) StageEval {
    var stage_bytes: u64 = base_stage_bytes[0];
    var peak_stage_bytes: u64 = stage_bytes;
    var total_recompute_cost: f64 = 0.0;
    var total_retained_bytes: u64 = 0;

    for (candidates, 0..) |candidate, candidate_idx| {
        const keep = ((mask >> @intCast(candidate_idx)) & 1) == 1;
        if (keep) {
            total_retained_bytes += candidate.bytes;
        } else {
            total_recompute_cost += candidate.weighted_recompute_cost;
        }
    }

    for (base_stage_bytes, 0..) |base_bytes, stage_idx| {
        stage_bytes = base_bytes;
        for (candidates, 0..) |candidate, candidate_idx| {
            const keep = ((mask >> @intCast(candidate_idx)) & 1) == 1;
            if (keep and candidate.op_index <= stage_idx and candidate.last_forward_use_index >= stage_idx) {
                stage_bytes += candidate.bytes;
            }
        }
        peak_stage_bytes = @max(peak_stage_bytes, stage_bytes);
    }

    return .{
        .total_retained_bytes = total_retained_bytes,
        .peak_stage_bytes = peak_stage_bytes,
        .total_recompute_cost = total_recompute_cost,
    };
}

fn evaluateKeepMask(candidates: []const Candidate, base_stage_bytes: []const u64, keep_mask: []const bool) StageEval {
    var stage_bytes: u64 = base_stage_bytes[0];
    var peak_stage_bytes: u64 = stage_bytes;
    var total_recompute_cost: f64 = 0.0;
    var total_retained_bytes: u64 = 0;

    for (candidates, 0..) |candidate, candidate_idx| {
        if (keep_mask[candidate_idx]) {
            total_retained_bytes += candidate.bytes;
        } else {
            total_recompute_cost += candidate.weighted_recompute_cost;
        }
    }

    for (base_stage_bytes, 0..) |base_bytes, stage_idx| {
        stage_bytes = base_bytes;
        for (candidates, 0..) |candidate, candidate_idx| {
            if (keep_mask[candidate_idx] and candidate.op_index <= stage_idx and candidate.last_forward_use_index >= stage_idx) {
                stage_bytes += candidate.bytes;
            }
        }
        peak_stage_bytes = @max(peak_stage_bytes, stage_bytes);
    }

    return .{
        .total_retained_bytes = total_retained_bytes,
        .peak_stage_bytes = peak_stage_bytes,
        .total_recompute_cost = total_recompute_cost,
    };
}

fn evaluateKeepPlan(problem: *const PlanningProblem, keep_mask: []const bool) StageEval {
    var stage_bytes: u64 = problem.base_stage_bytes[0];
    var peak_stage_bytes: u64 = stage_bytes;
    var total_recompute_cost: f64 = 0.0;
    var total_retained_bytes: u64 = 0;

    for (problem.candidates, 0..) |candidate, candidate_idx| {
        if (keep_mask[candidate_idx]) {
            total_retained_bytes += candidate.bytes;
        } else {
            total_recompute_cost += candidate.weighted_recompute_cost;
        }
    }

    for (problem.base_stage_bytes, 0..) |base_bytes, stage_idx| {
        stage_bytes = base_bytes;
        for (problem.candidates, 0..) |candidate, candidate_idx| {
            if (keep_mask[candidate_idx] and candidateKeptAtStage(candidate, stage_idx, problem.forward_stage_count)) {
                stage_bytes += candidate.bytes;
            }
        }
        peak_stage_bytes = @max(peak_stage_bytes, stage_bytes);
    }

    return .{
        .total_retained_bytes = total_retained_bytes,
        .peak_stage_bytes = peak_stage_bytes,
        .total_recompute_cost = total_recompute_cost,
    };
}

fn evaluateProblemMask(problem: *const PlanningProblem, mask: u64) StageEval {
    var stage_bytes: u64 = problem.base_stage_bytes[0];
    var peak_stage_bytes: u64 = stage_bytes;
    var total_recompute_cost: f64 = 0.0;
    var total_retained_bytes: u64 = 0;

    for (problem.candidates, 0..) |candidate, candidate_idx| {
        const keep = ((mask >> @intCast(candidate_idx)) & 1) == 1;
        if (keep) {
            total_retained_bytes += candidate.bytes;
        } else {
            total_recompute_cost += candidate.weighted_recompute_cost;
        }
    }

    for (problem.base_stage_bytes, 0..) |base_bytes, stage_idx| {
        stage_bytes = base_bytes;
        for (problem.candidates, 0..) |candidate, candidate_idx| {
            const keep = ((mask >> @intCast(candidate_idx)) & 1) == 1;
            if (keep and candidateKeptAtStage(candidate, stage_idx, problem.forward_stage_count)) {
                stage_bytes += candidate.bytes;
            }
        }
        peak_stage_bytes = @max(peak_stage_bytes, stage_bytes);
    }

    return .{
        .total_retained_bytes = total_retained_bytes,
        .peak_stage_bytes = peak_stage_bytes,
        .total_recompute_cost = total_recompute_cost,
    };
}

fn candidateKeptAtStage(candidate: Candidate, stage_idx: usize, forward_stage_count: usize) bool {
    if (stage_idx < forward_stage_count) {
        return candidate.op_index <= stage_idx and candidate.last_forward_use_index >= stage_idx;
    }
    if (candidate.backward_use_stage_indices.len == 0) return false;

    var max_backward_stage = candidate.backward_use_stage_indices[0];
    for (candidate.backward_use_stage_indices[1..]) |backward_stage| {
        max_backward_stage = @max(max_backward_stage, backward_stage);
    }
    return stage_idx <= max_backward_stage;
}

fn evaluateScheduledSolution(
    problem: *const PlanningProblem,
    model: *const MilpModel,
    keep_mask: []const bool,
    recompute_mask: []const bool,
) StageEval {
    var peak_stage_bytes: u64 = 0;
    var total_retained_bytes: u64 = 0;
    var total_recompute_cost: f64 = 0.0;

    for (problem.candidates, 0..) |candidate, candidate_idx| {
        if (keep_mask[candidate_idx]) {
            total_retained_bytes += candidate.bytes;
        }
    }

    for (model.recompute_variables, 0..) |recompute_var, recompute_idx| {
        if (recompute_mask[recompute_idx]) {
            total_recompute_cost += problem.candidates[recompute_var.candidate_index].weighted_recompute_cost;
        }
    }

    for (problem.base_stage_bytes, 0..) |base_bytes, stage_idx| {
        var stage_bytes = base_bytes;
        for (problem.candidates, 0..) |candidate, candidate_idx| {
            if (keep_mask[candidate_idx] and candidateKeptAtStage(candidate, stage_idx, problem.forward_stage_count)) {
                stage_bytes += candidate.bytes;
            }
        }
        for (model.recompute_variables, 0..) |recompute_var, recompute_idx| {
            if (recompute_mask[recompute_idx] and recompute_var.stage_index == stage_idx) {
                stage_bytes += problem.candidates[recompute_var.candidate_index].bytes;
            }
        }
        peak_stage_bytes = @max(peak_stage_bytes, stage_bytes);
    }

    return .{
        .total_retained_bytes = total_retained_bytes,
        .peak_stage_bytes = peak_stage_bytes,
        .total_recompute_cost = total_recompute_cost,
    };
}

fn stageWeightedKeepSum(candidates: []const Candidate, mask: u64) usize {
    var total: usize = 0;
    for (candidates, 0..) |candidate, candidate_idx| {
        const keep = ((mask >> @intCast(candidate_idx)) & 1) == 1;
        if (keep) {
            total += candidate.last_forward_use_index;
        }
    }
    return total;
}

fn compareGreedyDropScore(candidates: []const Candidate, lhs_idx: usize, rhs_idx: usize) bool {
    const lhs = candidates[lhs_idx];
    const rhs = candidates[rhs_idx];
    const lhs_score = greedyDropScore(lhs);
    const rhs_score = greedyDropScore(rhs);
    if (lhs_score == rhs_score) {
        return lhs.last_forward_use_index > rhs.last_forward_use_index;
    }
    return lhs_score > rhs_score;
}

fn greedyDropScore(candidate: Candidate) f64 {
    const bytes_saved = @as(f64, @floatFromInt(candidate.bytes));
    return bytes_saved / candidate.weighted_recompute_cost;
}

fn resolveArtifactPath(allocator: Allocator, configured_path: ?[]const u8, extension: []const u8) ![]u8 {
    if (configured_path) |path| return allocator.dupe(u8, path);
    return std.fmt.allocPrint(
        allocator,
        "/tmp/pcp-checkmate-{d}{s}",
        .{ std.time.nanoTimestamp(), extension },
    );
}

fn runExternalMilpSolver(
    allocator: Allocator,
    solver_command: []const u8,
    lp_path: []const u8,
    solution_path: []const u8,
) !void {
    var env_map = try std.process.getEnvMap(allocator);
    defer env_map.deinit();
    try env_map.put("PCP_CHECKMATE_LP_PATH", lp_path);
    try env_map.put("PCP_CHECKMATE_SOLUTION_PATH", solution_path);

    var child = std.process.Child.init(&.{ "/bin/sh", "-c", solver_command }, allocator);
    child.env_map = &env_map;
    const term = try child.spawnAndWait();
    switch (term) {
        .Exited => |code| if (code != 0) return error.ExternalMilpSolverFailed,
        else => return error.ExternalMilpSolverFailed,
    }
}

fn parseExternalMilpSolution(
    allocator: Allocator,
    problem: *const PlanningProblem,
    solution_path: []const u8,
) ![]bool {
    const solution_file = try std.fs.openFileAbsolute(solution_path, .{});
    defer solution_file.close();
    const contents = try solution_file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(contents);

    const keep_mask = try allocator.alloc(bool, problem.candidates.len);
    @memset(keep_mask, false);

    var saw_status = false;
    var saw_assignment = false;
    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        if (std.mem.eql(u8, line, "status=optimal")) {
            saw_status = true;
            continue;
        }

        if (std.mem.indexOfScalar(u8, line, '=')) |eq_idx| {
            const key = std.mem.trim(u8, line[0..eq_idx], " \t");
            const value = std.mem.trim(u8, line[eq_idx + 1 ..], " \t");
            if (!std.mem.startsWith(u8, key, "keep_")) continue;

            const idx = try std.fmt.parseInt(usize, key["keep_".len..], 10);
            if (idx >= keep_mask.len) return error.InvalidMilpSolution;
            if (std.mem.eql(u8, value, "1")) {
                keep_mask[idx] = true;
            } else if (std.mem.eql(u8, value, "0")) {
                keep_mask[idx] = false;
            } else {
                return error.InvalidMilpSolution;
            }
            saw_assignment = true;
        }
    }

    if (!saw_status or !saw_assignment) return error.InvalidMilpSolution;
    return keep_mask;
}

fn writeSignedTerm(writer: anytype, coefficient: f64, variable_name: []const u8) !void {
    if (coefficient < 0.0) {
        try writer.print("- {d:.6} {s}", .{ -coefficient, variable_name });
    } else {
        try writer.print("+ {d:.6} {s}", .{ coefficient, variable_name });
    }
}

fn writeConstraint(writer: anytype, variable_names: [][]const u8, constraint: LinearConstraint) !void {
    try writer.print(" {s}:", .{constraint.name});
    if (constraint.terms.len == 0) {
        try writer.writeAll(" 0");
    } else {
        for (constraint.terms, 0..) |term, term_idx| {
            if (term_idx > 0) try writer.writeAll(" ");
            try writeSignedTerm(writer, term.coefficient, variable_names[term.variable_index]);
        }
    }
    const sense_text = switch (constraint.sense) {
        .less_equal => "<=",
        .greater_equal => ">=",
    };
    try writer.print(" {s} {d:.6}\n", .{ sense_text, constraint.rhs });
}

fn dupCandidates(allocator: Allocator, candidates: []const Candidate) ![]Candidate {
    var owned = try allocator.alloc(Candidate, candidates.len);
    errdefer allocator.free(owned);

    for (candidates, 0..) |candidate, idx| {
        owned[idx] = .{
            .bytes = candidate.bytes,
            .weighted_recompute_cost = candidate.weighted_recompute_cost,
            .op_index = candidate.op_index,
            .last_forward_use_index = candidate.last_forward_use_index,
            .producer_candidate_indices = try allocator.dupe(usize, candidate.producer_candidate_indices),
            .user_candidate_indices = try allocator.dupe(usize, candidate.user_candidate_indices),
            .use_stage_indices = try allocator.dupe(usize, candidate.use_stage_indices),
            .backward_use_stage_indices = try allocator.dupe(usize, candidate.backward_use_stage_indices),
        };
    }

    return owned;
}

test "stage-aware exact planner prefers keeping long-lived expensive candidate" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{
            .bytes = 8,
            .weighted_recompute_cost = 100.0,
            .op_index = 0,
            .last_forward_use_index = 2,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{2},
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 1.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{1},
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 1.0,
            .op_index = 2,
            .last_forward_use_index = 2,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{2},
            .backward_use_stage_indices = &.{},
        },
    };

    var result = try solveExactSmallGraph(allocator, &candidates, &.{ 0, 0, 0 }, 8);
    defer result.deinit(allocator);

    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, result.keep_mask);
    try std.testing.expectEqual(@as(u64, 8), result.peak_stage_bytes);
}

test "stage-aware exact planner rejects infeasible base stage" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{};
    try std.testing.expectError(
        error.NoFeasibleRematPlan,
        solveExactSmallGraph(allocator, &candidates, &.{16}, 8),
    );
}

test "milp model captures stage constraints and dependency edges" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{
            .bytes = 8,
            .weighted_recompute_cost = 10.0,
            .op_index = 0,
            .last_forward_use_index = 2,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{1},
            .use_stage_indices = &.{ 1, 2 },
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 4,
            .weighted_recompute_cost = 3.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{0},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{1},
            .backward_use_stage_indices = &.{},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try dupCandidates(allocator, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{ 0, 1 }),
        .base_stage_bytes = try allocator.dupe(u64, &.{ 0, 2, 2 }),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 14,
        .budget = 8,
        .forward_stage_count = 3,
        .backward_stage_count = 0,
    };
    defer problem.deinit();

    var model = try buildMilpModel(allocator, &problem);
    defer model.deinit();

    try std.testing.expectEqual(@as(f64, 0.0), model.objective_constant);
    try std.testing.expectEqual(@as(usize, 2), model.keep_variable_count);
    try std.testing.expectEqual(@as(usize, 3), model.recompute_variables.len);
    try std.testing.expectEqual(@as(usize, 5), model.variable_names.len);
    try std.testing.expectEqual(@as(usize, 3), model.stage_constraints.len);
    try std.testing.expectEqual(@as(usize, 3), model.availability_constraints.len);
    try std.testing.expectEqual(@as(usize, 2), model.dependency_constraints.len);
    try std.testing.expectEqual(@as(usize, 1), model.dependency_edges.len);
    try std.testing.expectEqual(@as(usize, 4), model.stage_constraints[1].terms.len);
    try std.testing.expectEqual(@as(f64, 6.0), model.stage_constraints[0].rhs);
    try std.testing.expectEqualStrings("keep_0", model.variable_names[0]);
    try std.testing.expectEqualStrings("remat_0_0", model.variable_names[2]);
    try std.testing.expectEqual(LinearConstraint.Sense.greater_equal, model.availability_constraints[0].sense);
}

test "milp model writes lp scaffold" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{
            .bytes = 8,
            .weighted_recompute_cost = 5.0,
            .op_index = 0,
            .last_forward_use_index = 0,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{0},
            .backward_use_stage_indices = &.{},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try dupCandidates(allocator, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{0}),
        .base_stage_bytes = try allocator.dupe(u64, &.{0}),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 8,
        .budget = 8,
        .forward_stage_count = 1,
        .backward_stage_count = 0,
    };
    defer problem.deinit();

    var model = try buildMilpModel(allocator, &problem);
    defer model.deinit();

    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    try model.writeLp(buffer.writer());

    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Minimize") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "stage_0") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "avail_0_0") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Binary") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "keep_0") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "remat_0_0") != null);
}

test "external milp solution parser reads keep assignments" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{
            .bytes = 8,
            .weighted_recompute_cost = 5.0,
            .op_index = 0,
            .last_forward_use_index = 0,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{0},
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 5.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{1},
            .backward_use_stage_indices = &.{},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try dupCandidates(allocator, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{ 0, 1 }),
        .base_stage_bytes = try allocator.dupe(u64, &.{ 0, 0 }),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 16,
        .budget = 8,
        .forward_stage_count = 2,
        .backward_stage_count = 0,
    };
    defer problem.deinit();

    const solution_path = try resolveArtifactPath(allocator, null, ".sol");
    defer {
        std.fs.deleteFileAbsolute(solution_path) catch {};
        allocator.free(solution_path);
    }
    {
        const file = try std.fs.createFileAbsolute(solution_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll(
            \\status=optimal
            \\keep_0=1
            \\keep_1=0
            \\
        );
    }

    const keep_mask = try parseExternalMilpSolution(allocator, &problem, solution_path);
    defer allocator.free(keep_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, keep_mask);
}

test "planning problem exact solver handles branching lifetime tradeoff" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{
            .bytes = 8,
            .weighted_recompute_cost = 50.0,
            .op_index = 0,
            .last_forward_use_index = 3,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{1},
            .use_stage_indices = &.{ 1, 3 },
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 2.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{0},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{1},
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 2.0,
            .op_index = 2,
            .last_forward_use_index = 2,
            .producer_candidate_indices = &.{0},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{2},
            .backward_use_stage_indices = &.{},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try dupCandidates(allocator, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{ 0, 1, 2 }),
        .base_stage_bytes = try allocator.dupe(u64, &.{ 0, 0, 0, 0 }),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 24,
        .budget = 8,
        .forward_stage_count = 4,
        .backward_stage_count = 0,
    };
    defer problem.deinit();

    var result = try solvePlanningProblemExact(allocator, &problem);
    defer result.deinit(allocator);

    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, result.keep_mask);
    try std.testing.expectEqual(@as(u64, 8), result.peak_stage_bytes);
}

test "exact backend beats greedy on overlapping intervals" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{
            .bytes = 8,
            .weighted_recompute_cost = 10.0,
            .op_index = 0,
            .last_forward_use_index = 2,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{2},
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 6.0,
            .op_index = 0,
            .last_forward_use_index = 0,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{0},
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 6.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{1},
            .backward_use_stage_indices = &.{},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try dupCandidates(allocator, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{ 0, 1, 2 }),
        .base_stage_bytes = try allocator.dupe(u64, &.{ 0, 0, 0 }),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 24,
        .budget = 8,
        .forward_stage_count = 3,
        .backward_stage_count = 0,
    };
    defer problem.deinit();

    var exact = try solvePlanningProblem(allocator, &problem, .{ .backend = .exact_bounded });
    defer exact.deinit(allocator);
    var greedy = try solvePlanningProblem(allocator, &problem, .{ .backend = .greedy_stage_aware });
    defer greedy.deinit(allocator);

    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, exact.keep_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, greedy.keep_mask);
    try std.testing.expect(exact.total_recompute_cost < greedy.total_recompute_cost);
}

test "glpk backend matches exact bounded solver on small problem" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{
            .bytes = 8,
            .weighted_recompute_cost = 10.0,
            .op_index = 0,
            .last_forward_use_index = 2,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{2},
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 6.0,
            .op_index = 0,
            .last_forward_use_index = 0,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{0},
            .backward_use_stage_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 6.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{1},
            .backward_use_stage_indices = &.{},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try dupCandidates(allocator, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{ 0, 1, 2 }),
        .base_stage_bytes = try allocator.dupe(u64, &.{ 0, 0, 0 }),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 24,
        .budget = 8,
        .forward_stage_count = 3,
        .backward_stage_count = 0,
    };
    defer problem.deinit();

    var exact = try solvePlanningProblem(allocator, &problem, .{ .backend = .exact_bounded });
    defer exact.deinit(allocator);
    var glpk_result = try solvePlanningProblem(allocator, &problem, .{ .backend = .milp_glpk });
    defer glpk_result.deinit(allocator);

    try std.testing.expectEqualSlices(bool, exact.keep_mask, glpk_result.keep_mask);
    try std.testing.expectEqual(exact.peak_stage_bytes, glpk_result.peak_stage_bytes);
}

test "exact and glpk honor backward-stage checkpoint pressure" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{
            .bytes = 8,
            .weighted_recompute_cost = 10.0,
            .op_index = 0,
            .last_forward_use_index = 0,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{0},
            .backward_use_stage_indices = &.{1},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 2.0,
            .op_index = 0,
            .last_forward_use_index = 0,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
            .use_stage_indices = &.{0},
            .backward_use_stage_indices = &.{1},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try dupCandidates(allocator, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{ 0, 1 }),
        .base_stage_bytes = try allocator.dupe(u64, &.{ 0, 0 }),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 16,
        .budget = 8,
        .forward_stage_count = 1,
        .backward_stage_count = 1,
    };
    defer problem.deinit();

    var exact = try solvePlanningProblem(allocator, &problem, .{ .backend = .exact_bounded });
    defer exact.deinit(allocator);
    var glpk_result = try solvePlanningProblem(allocator, &problem, .{ .backend = .milp_glpk });
    defer glpk_result.deinit(allocator);

    try std.testing.expectEqualSlices(bool, &.{ true, false }, exact.keep_mask);
    try std.testing.expectEqualSlices(bool, exact.keep_mask, glpk_result.keep_mask);
    try std.testing.expectEqual(@as(u64, 8), exact.peak_stage_bytes);
    try std.testing.expectEqual(exact.peak_stage_bytes, glpk_result.peak_stage_bytes);
}
