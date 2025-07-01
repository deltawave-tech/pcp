const std = @import("std");
const pcp = @import("pcp");

/// Test the new MLIR-based tensor architecture
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("=== MLIR Tensor Architecture Test ===\n", .{});
    
    // Test DType conversion
    std.debug.print("1. Testing DType...\n", .{});
    const dtype = pcp.tensor_mlir.DType.f32;
    std.debug.print("   DType.f32 size: {} bytes\n", .{dtype.sizeInBytes()});
    
    // Test Shape creation
    std.debug.print("2. Testing Shape...\n", .{});
    const dims = [_]i64{2, 3, 4};
    var shape = try pcp.tensor_mlir.Shape.init(allocator, &dims, dtype);
    defer shape.deinit();
    
    std.debug.print("   Shape: rank={}, dims=[", .{shape.rank()});
    for (shape.dims) |dim| {
        std.debug.print("{}, ", .{dim});
    }
    std.debug.print("], elements={}\n", .{shape.elemCount()});
    
    // Test Tensor type creation (compilation test)
    std.debug.print("3. Testing Tensor type...\n", .{});
    _ = pcp.tensor_mlir.Tensor(f32); // Just test that it compiles
    std.debug.print("   TensorF32 type created successfully\n", .{});
    
    // Note: Can't actually create tensor instances yet since we need MLIRBuilder
    // which requires full MLIR wrappers to be implemented
    
    std.debug.print("✓ MLIR Tensor architecture compiles successfully!\n", .{});
    std.debug.print("  Next: Implement missing MLIR wrappers (OpBuilder, Location, Value, etc.)\n", .{});
}