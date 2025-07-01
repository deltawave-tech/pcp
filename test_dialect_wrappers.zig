const std = @import("std");
const ops = @import("src/ops.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("Testing MLIR dialect wrappers...\n", .{});
    
    try ops.testMLIROpGeneration(allocator);
    
    std.debug.print("Dialect wrapper test completed successfully!\n", .{});
}