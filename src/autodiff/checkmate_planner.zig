const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Candidate = struct {
    bytes: u64,
    weighted_recompute_cost: f64,
    last_forward_use_index: usize,
};

pub const ExactPlanResult = struct {
    keep_mask: []bool,
    total_retained_bytes: u64,
    total_recompute_cost: f64,

    pub fn deinit(self: *ExactPlanResult, allocator: Allocator) void {
        allocator.free(self.keep_mask);
    }
};

const MAX_EXACT_CANDIDATES = 24;

pub fn solveExactSmallGraph(
    allocator: Allocator,
    candidates: []const Candidate,
    base_retained_bytes: u64,
    budget: u64,
) !ExactPlanResult {
    if (candidates.len > MAX_EXACT_CANDIDATES) {
        return error.CheckmateGraphTooLarge;
    }
    if (base_retained_bytes > budget) {
        return error.NoFeasibleRematPlan;
    }

    var best_mask: u64 = 0;
    var best_cost = std.math.inf(f64);
    var best_retained = base_retained_bytes;
    var found = false;

    const subset_count: u64 = @as(u64, 1) << @intCast(candidates.len);
    var mask: u64 = 0;
    while (mask < subset_count) : (mask += 1) {
        var retained = base_retained_bytes;
        var recompute_cost: f64 = 0.0;
        var stage_weighted_keep_sum: usize = 0;

        for (candidates, 0..) |candidate, candidate_idx| {
            const keep = ((mask >> @intCast(candidate_idx)) & 1) == 1;
            if (keep) {
                retained += candidate.bytes;
                stage_weighted_keep_sum += candidate.last_forward_use_index;
            } else {
                recompute_cost += candidate.weighted_recompute_cost;
            }
        }

        if (retained > budget) continue;

        if (!found or recompute_cost < best_cost or
            (recompute_cost == best_cost and retained > best_retained) or
            (recompute_cost == best_cost and retained == best_retained and
            stage_weighted_keep_sum > stageWeightedKeepSum(candidates, best_mask)))
        {
            found = true;
            best_mask = mask;
            best_cost = recompute_cost;
            best_retained = retained;
        }
    }

    if (!found) {
        return error.NoFeasibleRematPlan;
    }

    const keep_mask = try allocator.alloc(bool, candidates.len);
    for (candidates, 0..) |_, candidate_idx| {
        keep_mask[candidate_idx] = ((best_mask >> @intCast(candidate_idx)) & 1) == 1;
    }

    return .{
        .keep_mask = keep_mask,
        .total_retained_bytes = best_retained,
        .total_recompute_cost = best_cost,
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

test "exact planner keeps expensive candidate under budget" {
    const allocator = std.testing.allocator;

    const candidates = [_]Candidate{
        .{ .bytes = 8, .weighted_recompute_cost = 100.0, .last_forward_use_index = 2 },
        .{ .bytes = 8, .weighted_recompute_cost = 1.0, .last_forward_use_index = 1 },
        .{ .bytes = 8, .weighted_recompute_cost = 1.0, .last_forward_use_index = 0 },
    };

    var result = try solveExactSmallGraph(allocator, &candidates, 0, 8);
    defer result.deinit(allocator);

    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, result.keep_mask);
    try std.testing.expectEqual(@as(u64, 8), result.total_retained_bytes);
}
