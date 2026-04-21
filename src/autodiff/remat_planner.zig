const std = @import("std");
const mlir = @import("../mlir/wrapper.zig");
const tensor = @import("../core/tensor.zig");

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
    kept_candidate_ops: usize = 0,
    rematerialized_candidate_ops: usize = 0,
};

const CostClass = enum {
    cheap,
    moderate,
    expensive,
};

const OpMetadata = struct {
    op: mlir.Operation,
    result_bytes: u64,
    fanout: usize,
    dependency_penalty: f64,
    recompute_cost: f64,
    remat_legal: bool,
    candidate: bool,
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

    var metadata = try allocator.alloc(OpMetadata, ops_forward.len);
    defer allocator.free(metadata);

    var op_to_index = std.AutoHashMap(mlir.Operation, usize).init(allocator);
    defer op_to_index.deinit();

    for (ops_forward, 0..) |op, i| {
        try op_to_index.put(op, i);
        metadata[i] = .{
            .op = op,
            .result_bytes = computeResultBytes(op),
            .fanout = 0,
            .dependency_penalty = 1.0,
            .recompute_cost = 0.0,
            .remat_legal = isRematerializableOp(op),
            .candidate = false,
        };
    }

    for (ops_forward) |op| {
        for (0..op.getNumOperands()) |operand_idx| {
            const operand = op.getOperand(operand_idx);
            if (!c.mlirValueIsAOpResult(operand.handle)) continue;
            const producer = mlir.Operation{ .handle = c.mlirOpResultGetOwner(operand.handle) };
            if (op_to_index.get(producer)) |producer_idx| {
                metadata[producer_idx].fanout += 1;
            }
        }
    }

    for (metadata, 0..) |*entry, i| {
        entry.recompute_cost = estimateRecomputeCost(entry.op, entry.result_bytes, config);
        const cost_class = costClassForOp(entry.op.getName());
        const expensive_allowed = config.remat_allow_expensive_ops or cost_class != .expensive;
        entry.candidate = entry.remat_legal and entry.result_bytes > 0 and expensive_allowed;

        var dependency_penalty: f64 = 1.0;
        for (0..entry.op.getNumOperands()) |operand_idx| {
            const operand = entry.op.getOperand(operand_idx);
            if (!c.mlirValueIsAOpResult(operand.handle)) continue;
            const producer = mlir.Operation{ .handle = c.mlirOpResultGetOwner(operand.handle) };
            if (op_to_index.get(producer)) |producer_idx| {
                if (metadata[producer_idx].remat_legal) {
                    dependency_penalty += 0.5;
                }
            }
        }
        _ = i;
        entry.dependency_penalty = dependency_penalty;
    }

    switch (config.policy) {
        .legacy_threshold => try buildLegacyThresholdPlan(&plan, metadata),
        .budget_greedy => try buildBudgetGreedyPlan(allocator, &plan, metadata, config),
        .checkmate_optimal => return error.CheckmatePlannerNotImplemented,
    }

    return plan;
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
    const fanout_penalty = 1.0 + 0.35 * @as(f64, @floatFromInt(entry.fanout));
    return bytes_saved / (entry.recompute_cost * fanout_penalty * entry.dependency_penalty);
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
        .result_bytes = 32 * 1024 * 1024,
        .fanout = 1,
        .dependency_penalty = 1.0,
        .recompute_cost = 2.0,
        .remat_legal = true,
        .candidate = true,
    };
    const candidate_b = OpMetadata{
        .op = undefined,
        .result_bytes = 8 * 1024 * 1024,
        .fanout = 3,
        .dependency_penalty = 2.0,
        .recompute_cost = 12.0,
        .remat_legal = true,
        .candidate = true,
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
