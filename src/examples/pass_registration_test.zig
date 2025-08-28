// src/examples/pass_registration_test.zig
const std = @import("std");
const pcp = @import("pcp");
const mlir_ctx = pcp.mlir_ctx;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("--- Running Minimal Pass Registration Test ---\n", .{});
    // This line should trigger the crash if the linking is wrong.
    var ctx = try mlir_ctx.MLIRContext.init(allocator);
    defer ctx.deinit();

    std.debug.print("--- SUCCESS: MLIRContext with full GPU pipeline initialized. ---\n", .{});
}