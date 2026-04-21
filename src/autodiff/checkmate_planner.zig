const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Candidate = struct {
    bytes: u64,
    weighted_recompute_cost: f64,
    op_index: usize,
    last_forward_use_index: usize,
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

test "stage-aware exact planner prefers keeping long-lived expensive candidate" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{ .bytes = 8, .weighted_recompute_cost = 100.0, .op_index = 0, .last_forward_use_index = 2 },
        .{ .bytes = 8, .weighted_recompute_cost = 1.0, .op_index = 1, .last_forward_use_index = 1 },
        .{ .bytes = 8, .weighted_recompute_cost = 1.0, .op_index = 2, .last_forward_use_index = 2 },
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
