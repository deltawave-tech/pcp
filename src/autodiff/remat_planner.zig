const std = @import("std");
const mlir = @import("../mlir/wrapper.zig");
const tensor = @import("../core/tensor.zig");
const checkmate_planner = @import("checkmate_planner.zig");

const Allocator = std.mem.Allocator;
const c = @import("../mlir/c.zig").c;

pub const RematPolicy = enum {
    legacy_threshold,
    budget_greedy,
    checkmate_optimal,
};

pub const RematCostModel = enum {
    static_heuristic,
    profiled,
};

pub const RematPlannerConfig = struct {
    policy: RematPolicy = .legacy_threshold,
    activation_memory_budget_bytes: ?u64 = null,
    cost_model: RematCostModel = .static_heuristic,
    remat_allow_expensive_ops: bool = true,
};

pub const RematPlan = struct {
    should_keep: std.AutoHashMap(mlir.Operation, bool),
    stats: PlanStats,

    pub fn init(allocator: Allocator) RematPlan {
        return .{
            .should_keep = std.AutoHashMap(mlir.Operation, bool).init(allocator),
            .stats = .{},
        };
    }

    pub fn deinit(self: *RematPlan) void {
        self.should_keep.deinit();
    }

    pub fn shouldRematerialize(self: *const RematPlan, op: mlir.Operation) bool {
        return !(self.should_keep.get(op) orelse true);
    }
};

pub const PlanStats = struct {
    policy: RematPolicy = .legacy_threshold,
    activation_memory_budget_bytes: ?u64 = null,
    estimated_retained_bytes_before: u64 = 0,
    estimated_retained_bytes_after: u64 = 0,
    estimated_peak_stage_bytes_after: u64 = 0,
    kept_candidate_ops: usize = 0,
    rematerialized_candidate_ops: usize = 0,
};

pub const CostClass = enum {
    cheap,
    moderate,
    expensive,
};

pub const OpMetadata = struct {
    op: mlir.Operation,
    op_index: usize,
    result_bytes: u64,
    fanout: usize,
    dependency_penalty: f64,
    recompute_cost: f64,
    weighted_recompute_cost: f64,
    remat_legal: bool,
    candidate: bool,
    cost_class: CostClass,
    producer_indices: []usize,
    user_indices: []usize,
    last_forward_use_index: usize,
};

pub const GraphMetadata = struct {
    allocator: Allocator,
    ops_in_forward_order: []mlir.Operation,
    op_metadata: []OpMetadata,
    base_stage_bytes: []u64,

    pub fn deinit(self: *GraphMetadata) void {
        for (self.op_metadata) |entry| {
            self.allocator.free(entry.producer_indices);
            self.allocator.free(entry.user_indices);
        }
        self.allocator.free(self.op_metadata);
        self.allocator.free(self.ops_in_forward_order);
        self.allocator.free(self.base_stage_bytes);
    }
};

pub fn buildRematPlan(
    allocator: Allocator,
    ops_forward: []const mlir.Operation,
    config: RematPlannerConfig,
) !RematPlan {
    var plan = RematPlan.init(allocator);
    errdefer plan.deinit();

    plan.stats.policy = config.policy;
    plan.stats.activation_memory_budget_bytes = config.activation_memory_budget_bytes;

    var graph_metadata = try collectGraphMetadata(allocator, ops_forward, config);
    defer graph_metadata.deinit();

    switch (config.policy) {
        .legacy_threshold => try buildLegacyThresholdPlan(&plan, graph_metadata.op_metadata),
        .budget_greedy => try buildBudgetGreedyPlan(allocator, &plan, graph_metadata.op_metadata, config),
        .checkmate_optimal => try buildCheckmateExactPlan(allocator, &plan, &graph_metadata, config),
    }

    return plan;
}

pub fn collectGraphMetadata(
    allocator: Allocator,
    ops_forward: []const mlir.Operation,
    config: RematPlannerConfig,
) !GraphMetadata {
    const owned_ops = try allocator.dupe(mlir.Operation, ops_forward);
    errdefer allocator.free(owned_ops);

    var op_metadata = try allocator.alloc(OpMetadata, owned_ops.len);
    errdefer allocator.free(op_metadata);
    var base_stage_bytes = try allocator.alloc(u64, owned_ops.len);
    errdefer allocator.free(base_stage_bytes);

    var op_to_index = std.AutoHashMap(mlir.Operation, usize).init(allocator);
    defer op_to_index.deinit();

    for (owned_ops, 0..) |op, i| {
        try op_to_index.put(op, i);
        const result_bytes = computeResultBytes(op);
        op_metadata[i] = .{
            .op = op,
            .op_index = i,
            .result_bytes = result_bytes,
            .fanout = 0,
            .dependency_penalty = 1.0,
            .recompute_cost = 0.0,
            .weighted_recompute_cost = 0.0,
            .remat_legal = isRematerializableOp(op),
            .candidate = false,
            .cost_class = .cheap,
            .producer_indices = try allocator.alloc(usize, 0),
            .user_indices = try allocator.alloc(usize, 0),
            .last_forward_use_index = i,
        };
        base_stage_bytes[i] = 0;
    }

    for (owned_ops, 0..) |op, op_idx| {
        var producer_indices = std.ArrayList(usize).init(allocator);
        defer producer_indices.deinit();

        for (0..op.getNumOperands()) |operand_idx| {
            const operand = op.getOperand(operand_idx);
            if (!c.mlirValueIsAOpResult(operand.handle)) continue;
            const producer = mlir.Operation{ .handle = c.mlirOpResultGetOwner(operand.handle) };
            if (op_to_index.get(producer)) |producer_idx| {
                try producer_indices.append(producer_idx);
                op_metadata[producer_idx].fanout += 1;

                var user_indices = std.ArrayList(usize).init(allocator);
                defer user_indices.deinit();
                try user_indices.appendSlice(op_metadata[producer_idx].user_indices);
                try user_indices.append(op_idx);
                allocator.free(op_metadata[producer_idx].user_indices);
                op_metadata[producer_idx].user_indices = try user_indices.toOwnedSlice();
                op_metadata[producer_idx].last_forward_use_index = @max(op_metadata[producer_idx].last_forward_use_index, op_idx);
            }
        }

        allocator.free(op_metadata[op_idx].producer_indices);
        op_metadata[op_idx].producer_indices = try producer_indices.toOwnedSlice();
    }

    for (op_metadata) |*entry| {
        entry.cost_class = costClassForOp(entry.op.getName());
        entry.recompute_cost = estimateRecomputeCost(entry.op, entry.result_bytes, config);

        const expensive_allowed = config.remat_allow_expensive_ops or entry.cost_class != .expensive;
        entry.candidate = entry.remat_legal and entry.result_bytes > 0 and expensive_allowed;

        var dependency_penalty: f64 = 1.0;
        for (entry.producer_indices) |producer_idx| {
            if (op_metadata[producer_idx].remat_legal) {
                dependency_penalty += 0.5;
            }
        }
        entry.dependency_penalty = dependency_penalty;

        const fanout_penalty = 1.0 + 0.35 * @as(f64, @floatFromInt(entry.fanout));
        entry.weighted_recompute_cost = entry.recompute_cost * fanout_penalty * entry.dependency_penalty;
    }

    for (op_metadata) |entry| {
        if (entry.result_bytes == 0) continue;
        for (entry.op_index..entry.last_forward_use_index + 1) |stage_idx| {
            base_stage_bytes[stage_idx] += entry.result_bytes;
        }
    }

    return .{
        .allocator = allocator,
        .ops_in_forward_order = owned_ops,
        .op_metadata = op_metadata,
        .base_stage_bytes = base_stage_bytes,
    };
}

fn buildLegacyThresholdPlan(plan: *RematPlan, metadata: []const OpMetadata) !void {
    for (metadata) |entry| {
        const keep = !legacyShouldRematerialize(entry.op);
        try plan.should_keep.put(entry.op, keep);
        if (entry.candidate) {
            if (keep) {
                plan.stats.kept_candidate_ops += 1;
            } else {
                plan.stats.rematerialized_candidate_ops += 1;
            }
        }
        plan.stats.estimated_retained_bytes_before += entry.result_bytes;
        if (keep) {
            plan.stats.estimated_retained_bytes_after += entry.result_bytes;
        }
    }
}

fn buildBudgetGreedyPlan(
    allocator: Allocator,
    plan: *RematPlan,
    metadata: []const OpMetadata,
    config: RematPlannerConfig,
) !void {
    if (config.activation_memory_budget_bytes == null) {
        try buildLegacyThresholdPlan(plan, metadata);
        return;
    }

    var current_retained_bytes: u64 = 0;
    for (metadata) |entry| {
        current_retained_bytes += entry.result_bytes;
    }
    plan.stats.estimated_retained_bytes_before = current_retained_bytes;

    const budget = config.activation_memory_budget_bytes.?;

    var candidate_indices = std.ArrayList(usize).init(allocator);
    defer candidate_indices.deinit();

    for (metadata, 0..) |entry, i| {
        try plan.should_keep.put(entry.op, true);
        if (entry.candidate) {
            try candidate_indices.append(i);
        }
    }

    std.sort.block(usize, candidate_indices.items, metadata, compareCandidateScore);

    var kept_candidates: usize = candidate_indices.items.len;
    for (candidate_indices.items) |idx| {
        if (current_retained_bytes <= budget) break;
        const entry = metadata[idx];
        const current_keep = plan.should_keep.get(entry.op) orelse true;
        if (!current_keep) continue;
        try plan.should_keep.put(entry.op, false);
        current_retained_bytes -|= entry.result_bytes;
        kept_candidates -= 1;
    }

    plan.stats.estimated_retained_bytes_after = current_retained_bytes;
    plan.stats.kept_candidate_ops = kept_candidates;
    plan.stats.rematerialized_candidate_ops = candidate_indices.items.len - kept_candidates;
}

fn buildCheckmateExactPlan(allocator: Allocator, plan: *RematPlan, graph_metadata: *const GraphMetadata, config: RematPlannerConfig) !void {
    const metadata = graph_metadata.op_metadata;
    var problem = try buildCheckmatePlanningProblem(allocator, graph_metadata, config);
    defer problem.deinit();

    for (metadata) |entry| {
        try plan.should_keep.put(entry.op, true);
    }

    plan.stats.estimated_retained_bytes_before = problem.total_retained_bytes_before;

    if (config.activation_memory_budget_bytes == null) {
        plan.stats.kept_candidate_ops = problem.candidates.len;
        plan.stats.rematerialized_candidate_ops = 0;
        plan.stats.estimated_retained_bytes_after = problem.total_retained_bytes_before;
        return;
    }

    var exact_result = try checkmate_planner.solvePlanningProblem(allocator, &problem, .exact_bounded);
    defer exact_result.deinit(allocator);

    var kept_candidates: usize = 0;
    for (problem.candidate_metadata_indices, 0..) |metadata_idx, candidate_idx| {
        const keep = exact_result.keep_mask[candidate_idx];
        try plan.should_keep.put(metadata[metadata_idx].op, keep);
        if (keep) kept_candidates += 1;
    }

    plan.stats.kept_candidate_ops = kept_candidates;
    plan.stats.rematerialized_candidate_ops = problem.candidates.len - kept_candidates;
    plan.stats.estimated_retained_bytes_after = problem.base_retained_bytes + exact_result.total_retained_bytes;
    plan.stats.estimated_peak_stage_bytes_after = exact_result.peak_stage_bytes;
}

fn buildCheckmatePlanningProblem(
    allocator: Allocator,
    graph_metadata: *const GraphMetadata,
    config: RematPlannerConfig,
) !checkmate_planner.PlanningProblem {
    const metadata = graph_metadata.op_metadata;
    const budget = config.activation_memory_budget_bytes orelse std.math.maxInt(u64);

    var total_retained_before: u64 = 0;
    var base_retained_bytes: u64 = 0;
    var candidate_metadata_indices = std.ArrayList(usize).init(allocator);
    defer candidate_metadata_indices.deinit();
    var candidates = std.ArrayList(checkmate_planner.Candidate).init(allocator);
    errdefer {
        for (candidates.items) |candidate| {
            allocator.free(candidate.producer_candidate_indices);
            allocator.free(candidate.user_candidate_indices);
        }
        candidates.deinit();
    }
    var base_stage_bytes = try allocator.dupe(u64, graph_metadata.base_stage_bytes);
    errdefer allocator.free(base_stage_bytes);
    var candidate_index_by_metadata = try allocator.alloc(?usize, metadata.len);
    defer allocator.free(candidate_index_by_metadata);

    for (candidate_index_by_metadata) |*entry| {
        entry.* = null;
    }

    for (metadata, 0..) |entry, idx| {
        total_retained_before += entry.result_bytes;
        if (!entry.candidate) {
            base_retained_bytes += entry.result_bytes;
            continue;
        }

        candidate_index_by_metadata[idx] = candidates.items.len;
        try candidate_metadata_indices.append(idx);
        try candidates.append(.{
            .bytes = entry.result_bytes,
            .weighted_recompute_cost = entry.weighted_recompute_cost,
            .op_index = entry.op_index,
            .last_forward_use_index = entry.last_forward_use_index,
            .producer_candidate_indices = try allocator.alloc(usize, 0),
            .user_candidate_indices = try allocator.alloc(usize, 0),
        });
        for (entry.op_index..entry.last_forward_use_index + 1) |stage_idx| {
            base_stage_bytes[stage_idx] -|= entry.result_bytes;
        }
    }

    for (candidate_metadata_indices.items, 0..) |metadata_idx, candidate_idx| {
        const entry = metadata[metadata_idx];
        allocator.free(candidates.items[candidate_idx].producer_candidate_indices);
        candidates.items[candidate_idx].producer_candidate_indices = try collectCandidateNeighborIndices(
            allocator,
            entry.producer_indices,
            candidate_index_by_metadata,
        );
        allocator.free(candidates.items[candidate_idx].user_candidate_indices);
        candidates.items[candidate_idx].user_candidate_indices = try collectCandidateNeighborIndices(
            allocator,
            entry.user_indices,
            candidate_index_by_metadata,
        );
    }

    return .{
        .allocator = allocator,
        .candidates = try candidates.toOwnedSlice(),
        .candidate_metadata_indices = try candidate_metadata_indices.toOwnedSlice(),
        .base_stage_bytes = base_stage_bytes,
        .base_retained_bytes = base_retained_bytes,
        .total_retained_bytes_before = total_retained_before,
        .budget = budget,
    };
}

fn collectCandidateNeighborIndices(
    allocator: Allocator,
    metadata_indices: []const usize,
    candidate_index_by_metadata: []const ?usize,
) ![]usize {
    var neighbors = std.ArrayList(usize).init(allocator);
    defer neighbors.deinit();

    for (metadata_indices) |metadata_idx| {
        if (candidate_index_by_metadata[metadata_idx]) |candidate_idx| {
            try neighbors.append(candidate_idx);
        }
    }

    return neighbors.toOwnedSlice();
}

fn compareCandidateScore(metadata: []const OpMetadata, lhs_idx: usize, rhs_idx: usize) bool {
    const lhs = metadata[lhs_idx];
    const rhs = metadata[rhs_idx];

    const lhs_score = candidateScore(lhs);
    const rhs_score = candidateScore(rhs);
    if (lhs_score == rhs_score) {
        return lhs.result_bytes > rhs.result_bytes;
    }
    return lhs_score > rhs_score;
}

fn candidateScore(entry: OpMetadata) f64 {
    const bytes_saved = @as(f64, @floatFromInt(entry.result_bytes));
    return bytes_saved / entry.weighted_recompute_cost;
}

fn isRematerializableOp(op: mlir.Operation) bool {
    const name = op.getName();
    if (std.mem.eql(u8, name, "stablehlo.constant") or std.mem.eql(u8, name, "func.return")) {
        return false;
    }
    if (op.getNumResults() != 1) return false;

    const result_type = op.getResult(0).getType();
    if (!c.mlirTypeIsARankedTensor(result_type.handle)) return false;

    if (!isSupportedRematOpName(name)) return false;
    return true;
}

fn isSupportedRematOpName(name: []const u8) bool {
    return std.mem.eql(u8, name, "stablehlo.dot_general") or
        std.mem.eql(u8, name, "stablehlo.exponential") or
        std.mem.eql(u8, name, "stablehlo.divide") or
        std.mem.eql(u8, name, "stablehlo.logistic") or
        std.mem.eql(u8, name, "stablehlo.tanh") or
        std.mem.eql(u8, name, "stablehlo.maximum") or
        std.mem.eql(u8, name, "stablehlo.add");
}

fn legacyShouldRematerialize(op: mlir.Operation) bool {
    const name = op.getName();

    const is_heavy_math = std.mem.eql(u8, name, "stablehlo.dot_general") or
        std.mem.eql(u8, name, "stablehlo.exponential") or
        std.mem.eql(u8, name, "stablehlo.divide") or
        std.mem.eql(u8, name, "stablehlo.logistic") or
        std.mem.eql(u8, name, "stablehlo.tanh") or
        std.mem.eql(u8, name, "stablehlo.maximum") or
        std.mem.eql(u8, name, "stablehlo.add");

    if (!is_heavy_math) return false;
    if (op.getNumResults() == 0) return false;

    const type_val = op.getResult(0).getType();
    if (!c.mlirTypeIsARankedTensor(type_val.handle)) return false;

    const rank = c.mlirShapedTypeGetRank(type_val.handle);
    if (rank >= 3) {
        var total_elements: i64 = 1;
        for (0..@intCast(rank)) |i| {
            const dim = c.mlirShapedTypeGetDimSize(type_val.handle, @intCast(i));
            if (dim > 0) total_elements *= dim;
        }
        return total_elements >= 1_000_000;
    }

    return false;
}

fn computeResultBytes(op: mlir.Operation) u64 {
    if (op.getNumResults() != 1) return 0;
    const ranked_type = op.getResult(0).getType().as(mlir.RankedTensorType) orelse return 0;
    return computeTypeBytes(ranked_type);
}

fn estimateRecomputeCost(op: mlir.Operation, result_bytes: u64, config: RematPlannerConfig) f64 {
    _ = config;
    const bytes_scale = 1.0 + @as(f64, @floatFromInt(@max(result_bytes, 1))) / (1024.0 * 1024.0);
    const base: f64 = switch (costClassForOp(op.getName())) {
        .cheap => 1.0,
        .moderate => 2.0,
        .expensive => 12.0,
    };
    return base * bytes_scale;
}

fn costClassForOp(name: []const u8) CostClass {
    if (std.mem.eql(u8, name, "stablehlo.dot_general")) return .expensive;
    if (std.mem.eql(u8, name, "stablehlo.exponential") or
        std.mem.eql(u8, name, "stablehlo.divide") or
        std.mem.eql(u8, name, "stablehlo.logistic") or
        std.mem.eql(u8, name, "stablehlo.tanh"))
    {
        return .moderate;
    }
    return .cheap;
}

test "budget greedy prefers larger cheaper candidates" {
    const candidate_a = OpMetadata{
        .op = undefined,
        .op_index = 0,
        .result_bytes = 32 * 1024 * 1024,
        .fanout = 1,
        .dependency_penalty = 1.0,
        .recompute_cost = 2.0,
        .weighted_recompute_cost = 2.7,
        .remat_legal = true,
        .candidate = true,
        .cost_class = .moderate,
        .producer_indices = &.{},
        .user_indices = &.{},
        .last_forward_use_index = 3,
    };
    const candidate_b = OpMetadata{
        .op = undefined,
        .op_index = 1,
        .result_bytes = 8 * 1024 * 1024,
        .fanout = 3,
        .dependency_penalty = 2.0,
        .recompute_cost = 12.0,
        .weighted_recompute_cost = 44.4,
        .remat_legal = true,
        .candidate = true,
        .cost_class = .expensive,
        .producer_indices = &.{},
        .user_indices = &.{},
        .last_forward_use_index = 4,
    };

    try std.testing.expect(candidateScore(candidate_a) > candidateScore(candidate_b));
}

test "result bytes handles scalar and dynamic shapes conservatively" {
    const ctx = try mlir.Context.init();
    defer ctx.deinit();

    const scalar_type = mlir.Type.rankedTensorType(ctx, &.{}, mlir.Type.f32Type(ctx));
    const scalar_ranked = scalar_type.as(mlir.RankedTensorType).?;
    try std.testing.expectEqual(@as(u64, 4), computeTypeBytes(scalar_ranked));

    const dynamic_type = mlir.Type.rankedTensorType(ctx, &.{ -1, 32 }, mlir.Type.f32Type(ctx));
    const dynamic_ranked = dynamic_type.as(mlir.RankedTensorType).?;
    try std.testing.expectEqual(@as(u64, 0), computeTypeBytes(dynamic_ranked));
}

test "checkmate planning problem preserves candidate edges and base stages" {
    const allocator = std.testing.allocator;

    var op_metadata = try allocator.alloc(OpMetadata, 3);
    errdefer allocator.free(op_metadata);
    const ops_in_forward_order = try allocator.alloc(mlir.Operation, 0);
    errdefer allocator.free(ops_in_forward_order);
    const base_stage_bytes = try allocator.dupe(u64, &.{ 8, 16, 12, 8 });
    errdefer allocator.free(base_stage_bytes);

    op_metadata[0] = .{
        .op = undefined,
        .op_index = 0,
        .result_bytes = 8,
        .fanout = 2,
        .dependency_penalty = 1.0,
        .recompute_cost = 10.0,
        .weighted_recompute_cost = 10.0,
        .remat_legal = true,
        .candidate = true,
        .cost_class = .moderate,
        .producer_indices = try allocator.dupe(usize, &.{}),
        .user_indices = try allocator.dupe(usize, &.{ 1, 2 }),
        .last_forward_use_index = 3,
    };
    op_metadata[1] = .{
        .op = undefined,
        .op_index = 1,
        .result_bytes = 8,
        .fanout = 0,
        .dependency_penalty = 1.0,
        .recompute_cost = 2.0,
        .weighted_recompute_cost = 2.0,
        .remat_legal = true,
        .candidate = true,
        .cost_class = .cheap,
        .producer_indices = try allocator.dupe(usize, &.{0}),
        .user_indices = try allocator.dupe(usize, &.{}),
        .last_forward_use_index = 1,
    };
    op_metadata[2] = .{
        .op = undefined,
        .op_index = 2,
        .result_bytes = 4,
        .fanout = 0,
        .dependency_penalty = 1.0,
        .recompute_cost = 1.0,
        .weighted_recompute_cost = 1.0,
        .remat_legal = true,
        .candidate = false,
        .cost_class = .cheap,
        .producer_indices = try allocator.dupe(usize, &.{0}),
        .user_indices = try allocator.dupe(usize, &.{}),
        .last_forward_use_index = 2,
    };

    var graph_metadata = GraphMetadata{
        .allocator = allocator,
        .ops_in_forward_order = ops_in_forward_order,
        .op_metadata = op_metadata,
        .base_stage_bytes = base_stage_bytes,
    };
    defer graph_metadata.deinit();

    var problem = try buildCheckmatePlanningProblem(allocator, &graph_metadata, .{
        .policy = .checkmate_optimal,
        .activation_memory_budget_bytes = 8,
    });
    defer problem.deinit();

    try std.testing.expectEqual(@as(u64, 20), problem.total_retained_bytes_before);
    try std.testing.expectEqual(@as(u64, 4), problem.base_retained_bytes);
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 4, 0 }, problem.base_stage_bytes);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, problem.candidate_metadata_indices);
    try std.testing.expectEqualSlices(usize, &.{1}, problem.candidates[0].user_candidate_indices);
    try std.testing.expectEqualSlices(usize, &.{0}, problem.candidates[1].producer_candidate_indices);
}

fn computeTypeBytes(ranked_type: mlir.RankedTensorType) u64 {
    var elements: u64 = 1;
    const rank = ranked_type.getRank();
    if (rank == 0) {
        elements = 1;
    } else {
        for (0..rank) |dim_idx| {
            const dim = ranked_type.getDimension(dim_idx);
            if (dim <= 0) return 0;
            elements *= @intCast(dim);
        }
    }

    const dtype = tensor.DType.fromMlirType(ranked_type.getElementType());
    return elements * dtype.sizeInBytes();
}
