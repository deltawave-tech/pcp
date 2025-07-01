const std = @import("std");
const autodiff = @import("src/autodiff.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("Starting MLIR autodiff test...\n", .{});
    
    try autodiff.testMLIRAutoDiff(allocator);
    
    std.debug.print("MLIR autodiff test completed successfully!\n", .{});
}