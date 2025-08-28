const std = @import("std");
const pcp = @import("pcp");
const mlir_ctx = pcp.mlir_ctx;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🍎 M3 MLIR → SPIR-V → MSL → Metal Pipeline Test\n", .{});
    std.debug.print("============================================\n", .{});

    std.debug.print("Starting MLIR GPU pipeline test...\n", .{});
    
    mlir_ctx.testMLIRGPUPipeline(allocator) catch |err| {
        std.debug.print("❌ Pipeline test failed with error: {}\n", .{err});
        return;
    };
    
    std.debug.print("\n🎉 M3 pipeline test completed successfully!\n", .{});
}