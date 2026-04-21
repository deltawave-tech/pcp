const std = @import("std");
const pcp = @import("pcp");
const remat_planner = @import("../autodiff/remat_planner.zig");

const mlir = pcp.mlir;
const mlir_ctx = pcp.mlir_ctx;
const ops = pcp.ops;
const autodiff = pcp.autodiff;
const MLIRBuilder = ops.MLIRBuilder;

fn collectOpsInForwardOrder(allocator: std.mem.Allocator, fn_op: mlir.Operation) ![]mlir.Operation {
    var operations = std.ArrayList(mlir.Operation).init(allocator);
    errdefer operations.deinit();

    const func_body_region = fn_op.getRegion(0);
    const func_body_block = func_body_region.getBlock(0);

    var maybe_op = func_body_block.getFirstOp();
    while (maybe_op) |op| {
        try operations.append(op);
        maybe_op = op.getNext();
    }

    return operations.toOwnedSlice();
}

fn countRematerialized(plan: *const remat_planner.RematPlan, ops_forward: []const mlir.Operation) usize {
    var count: usize = 0;
    for (ops_forward) |op| {
        if (plan.shouldRematerialize(op)) count += 1;
    }
    return count;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var full_ctx = try mlir_ctx.MLIRContext.init(allocator);
    defer full_ctx.deinit();
    const ctx = full_ctx.getContext();

    var builder = try MLIRBuilder.init(allocator, ctx);
    defer builder.deinit();

    const f32_type = mlir.Type.f32Type(ctx);
    const matrix_type = mlir.Type.rankedTensorType(ctx, &.{ 1024, 1024 }, f32_type);
    const func_type = try mlir.Type.functionType(allocator, ctx, &.{ matrix_type, matrix_type }, &.{matrix_type});

    const fwd_result = try builder.createFunction("remat_forward", func_type);
    builder.setInsertionBlock(fwd_result.entry_block);

    const a = try builder.newTensor(fwd_result.entry_block.getArgument(0));
    const b = try builder.newTensor(fwd_result.entry_block.getArgument(1));

    const sum0 = try ops.add(&builder, a, b);
    const act0 = try ops.tanh(&builder, sum0);
    const sum1 = try ops.add(&builder, act0, a);
    _ = try builder.createAndAttach("func.return", &.{sum1.value}, &.{}, .{});

    const ops_forward = try collectOpsInForwardOrder(allocator, fwd_result.func_op);
    defer allocator.free(ops_forward);

    var legacy_plan = try remat_planner.buildRematPlan(allocator, ops_forward, .{
        .policy = .legacy_threshold,
    });
    defer legacy_plan.deinit();

    var greedy_plan = try remat_planner.buildRematPlan(allocator, ops_forward, .{
        .policy = .budget_greedy,
        .activation_memory_budget_bytes = 8 * 1024 * 1024,
        .cost_model = .static_heuristic,
        .remat_allow_expensive_ops = true,
    });
    defer greedy_plan.deinit();

    var checkmate_plan = try remat_planner.buildRematPlan(allocator, ops_forward, .{
        .policy = .checkmate_optimal,
        .activation_memory_budget_bytes = 8 * 1024 * 1024,
        .cost_model = .static_heuristic,
        .remat_allow_expensive_ops = true,
    });
    defer checkmate_plan.deinit();

    const legacy_remat = countRematerialized(&legacy_plan, ops_forward);
    const greedy_remat = countRematerialized(&greedy_plan, ops_forward);
    const checkmate_remat = countRematerialized(&checkmate_plan, ops_forward);

    try std.testing.expectEqual(@as(usize, 0), legacy_remat);
    try std.testing.expect(greedy_remat >= 1);
    try std.testing.expect(checkmate_remat >= 1);
    try std.testing.expect(greedy_plan.stats.estimated_retained_bytes_after <= 8 * 1024 * 1024);
    try std.testing.expect(checkmate_plan.stats.estimated_retained_bytes_after <= 8 * 1024 * 1024);

    _ = try autodiff.buildGradientGraphWithConfig(
        allocator,
        &builder,
        fwd_result.func_op,
        -100.0,
        100.0,
        .{
            .policy = .budget_greedy,
            .activation_memory_budget_bytes = 8 * 1024 * 1024,
            .cost_model = .static_heuristic,
            .remat_allow_expensive_ops = true,
        },
    );

    std.debug.print(
        "legacy_remat={} greedy_remat={} checkmate_remat={} retained_before={} greedy_after={} checkmate_after={}\n",
        .{
            legacy_remat,
            greedy_remat,
            checkmate_remat,
            greedy_plan.stats.estimated_retained_bytes_before,
            greedy_plan.stats.estimated_retained_bytes_after,
            checkmate_plan.stats.estimated_retained_bytes_after,
        },
    );
}
