const std = @import("std");
const pcp = @import("pcp");
const GraphBuilder = pcp.compiler.graph_builder.GraphBuilder;

const usage =
    \\usage: grpo_backward_builder <forward.mlir> <backward.mlir> <num_params> <num_buffers>
    \\
;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len != 5) {
        std.debug.print("{s}", .{usage});
        return error.InvalidArguments;
    }

    const forward_path = args[1];
    const backward_path = args[2];
    const num_params = try std.fmt.parseInt(usize, args[3], 10);
    const num_buffers = try std.fmt.parseInt(usize, args[4], 10);

    const forward_mlir = try std.fs.cwd().readFileAlloc(allocator, forward_path, 2 * 1024 * 1024 * 1024);
    defer allocator.free(forward_mlir);

    var build_ctx = try pcp.mlir_ctx.MLIRContext.init(allocator);
    defer build_ctx.deinit();

    var builder = try pcp.ops.MLIRBuilder.init(allocator, build_ctx.getContext());
    defer builder.deinit();

    const backward_mlir = try GraphBuilder.buildGrpoBackwardPass(
        allocator,
        &builder,
        forward_mlir,
        num_params,
        num_buffers,
    );
    defer allocator.free(backward_mlir);

    try std.fs.cwd().writeFile(.{
        .sub_path = backward_path,
        .data = backward_mlir,
    });

    std.debug.print("Wrote GRPO backward MLIR: {s} ({} bytes)\n", .{ backward_path, backward_mlir.len });
}
