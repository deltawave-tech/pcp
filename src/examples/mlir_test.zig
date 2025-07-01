const std = @import("std");
const pcp = @import("pcp");

/// Simple test program to verify MLIR integration
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("PCP MLIR Integration Test\n", .{});
    std.debug.print("========================\n", .{});
    
    // Test MLIR context initialization
    std.debug.print("\n1. Testing MLIR Context...\n", .{});
    try pcp.mlir_ctx.testMLIRIntegration(allocator);
    
    // Test MLIR operations
    std.debug.print("\n2. Testing MLIR Operations...\n", .{});
    try pcp.mlir.testMLIROperations(allocator);
    
    // Check compile-time MLIR availability
    std.debug.print("\n3. Compile-time Configuration...\n", .{});
    std.debug.print("MLIR enabled: true (always available)\n", .{});
    
    std.debug.print("✓ MLIR is available and ready for use\n", .{});
    std.debug.print("✓ You can now proceed with tensor MLIR migration (Phase 1)\n", .{});
    
    std.debug.print("\n=== MLIR Integration Test Complete ===\n", .{});
}