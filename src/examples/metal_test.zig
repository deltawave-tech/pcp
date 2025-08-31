/// Simple Metal backend test
/// Tests basic Metal device detection and MLIR context integration

const std = @import("std");
const pcp = @import("pcp");
const metal = pcp.backends.metal;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🍎 Metal Backend Test on Apple M3\n", .{});
    std.debug.print("================================\n\n", .{});

    // Test 1: Basic Metal device detection
    std.debug.print("Test 1: Metal device detection...\n", .{});
    
    // Initialize Metal backend
    metal.init(allocator) catch |err| {
        std.debug.print("💣 Metal initialization failed: {}\n", .{err});
        std.debug.print("Note: This is expected if not running on Apple hardware\n", .{});
        return;
    };
    defer metal.deinit();
    
    std.debug.print("🌙 Metal backend initialized successfully!\n", .{});
    
    // Test 2: Get execution engine
    std.debug.print("\nTest 2: Getting Metal execution engine...\n", .{});
    const engine = metal.getExecutionEngine() catch |err| {
        std.debug.print("💣 Failed to get Metal execution engine: {}\n", .{err});
        return;
    };
    
    std.debug.print("🌙 Metal execution engine obtained!\n", .{});
    
    // Test 3: Get MLIR context from executor
    std.debug.print("\nTest 3: Getting shared MLIR context...\n", .{});
    const executor = engine.asExecutor();
    const mlir_context = executor.getContext();
    
    std.debug.print("🌙 Shared MLIR context obtained: handle={}\n", .{@intFromPtr(mlir_context.handle)});
    
    std.debug.print("\n🌚 All Metal backend tests passed!\n", .{});
    std.debug.print("Your Apple M3 system is ready for MLIR → Metal execution.\n", .{});
}