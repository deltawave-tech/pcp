const std = @import("std");
const pcp = @import("src/main.zig");
const mlir_ctx = pcp.mlir_ctx;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🍎 M3 MLIR → SPIR-V → MSL → Metal Pipeline Test\n", .{});
    std.debug.print("============================================\n", .{});

    try mlir_ctx.testMLIRGPUPipeline(allocator);
    
    std.debug.print("\n🌚 M3 pipeline test completed!\n", .{});
}