const std = @import("std");

const Allocator = std.mem.Allocator;

pub const SolverBackend = enum {
    exact_bounded,
    greedy_stage_aware,
};

pub const Candidate = struct {
    bytes: u64,
    weighted_recompute_cost: f64,
    op_index: usize,
    last_forward_use_index: usize,
    producer_candidate_indices: []usize,
    user_candidate_indices: []usize,
};

pub const PlanningProblem = struct {
    allocator: Allocator,
    candidates: []Candidate,
    candidate_metadata_indices: []usize,
    base_stage_bytes: []u64,
    base_retained_bytes: u64,
    total_retained_bytes_before: u64,
    budget: u64,

    pub fn deinit(self: *PlanningProblem) void {
        for (self.candidates) |candidate| {
            self.allocator.free(candidate.producer_candidate_indices);
            self.allocator.free(candidate.user_candidate_indices);
        }
        self.allocator.free(self.candidates);
        self.allocator.free(self.candidate_metadata_indices);
        self.allocator.free(self.base_stage_bytes);
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
    return solveExactSmallGraph(
        allocator,
        problem.candidates,
        problem.base_stage_bytes,
        problem.budget,
    );
}

pub fn solvePlanningProblem(
    allocator: Allocator,
    problem: *const PlanningProblem,
    backend: SolverBackend,
) !ExactPlanResult {
    return switch (backend) {
        .exact_bounded => solvePlanningProblemExact(allocator, problem),
        .greedy_stage_aware => solvePlanningProblemGreedy(allocator, problem),
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

    var candidate_order = try allocator.alloc(usize, problem.candidates.len);
    defer allocator.free(candidate_order);
    for (candidate_order, 0..) |*entry, idx| {
        entry.* = idx;
    }

    std.sort.block(usize, candidate_order, problem.candidates, compareGreedyDropScore);

    for (candidate_order) |candidate_idx| {
        const current_eval = evaluateKeepMask(problem.candidates, problem.base_stage_bytes, keep_mask);
        if (current_eval.peak_stage_bytes <= problem.budget) break;
        keep_mask[candidate_idx] = false;
    }

    const final_eval = evaluateKeepMask(problem.candidates, problem.base_stage_bytes, keep_mask);
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
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 1.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 1.0,
            .op_index = 2,
            .last_forward_use_index = 2,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
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
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 2.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{0},
            .user_candidate_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 2.0,
            .op_index = 2,
            .last_forward_use_index = 2,
            .producer_candidate_indices = &.{0},
            .user_candidate_indices = &.{},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try allocator.dupe(Candidate, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{ 0, 1, 2 }),
        .base_stage_bytes = try allocator.dupe(u64, &.{ 0, 0, 0, 0 }),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 24,
        .budget = 8,
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
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 6.0,
            .op_index = 0,
            .last_forward_use_index = 0,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
        },
        .{
            .bytes = 8,
            .weighted_recompute_cost = 6.0,
            .op_index = 1,
            .last_forward_use_index = 1,
            .producer_candidate_indices = &.{},
            .user_candidate_indices = &.{},
        },
    };
    var problem = PlanningProblem{
        .allocator = allocator,
        .candidates = try allocator.dupe(Candidate, &candidates),
        .candidate_metadata_indices = try allocator.dupe(usize, &.{ 0, 1, 2 }),
        .base_stage_bytes = try allocator.dupe(u64, &.{ 0, 0, 0 }),
        .base_retained_bytes = 0,
        .total_retained_bytes_before = 24,
        .budget = 8,
    };
    defer problem.deinit();

    var exact = try solvePlanningProblem(allocator, &problem, .exact_bounded);
    defer exact.deinit(allocator);
    var greedy = try solvePlanningProblem(allocator, &problem, .greedy_stage_aware);
    defer greedy.deinit(allocator);

    try std.testing.expectEqualSlices(bool, &.{ false, true, true }, exact.keep_mask);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, greedy.keep_mask);
    try std.testing.expect(exact.total_recompute_cost < greedy.total_recompute_cost);
}
