const std = @import("std");
const pcp = @import("pcp");
const tensorModule = pcp.tensor;
const autodiff = pcp.autodiff;

const Allocator = std.mem.Allocator;
const Tensor = tensorModule.Tensor;
const DType = tensorModule.DType;
const Node = autodiff.Node;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("Starting autodiff memory leak test\n", .{});
    
    // Create tensors
    var dims = [_]usize{ 2, 2 };
    
    var a_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a_tensor.setScalar(&[_]usize{0, 0}, 1.0);
    try a_tensor.setScalar(&[_]usize{0, 1}, 2.0);
    try a_tensor.setScalar(&[_]usize{1, 0}, 3.0);
    try a_tensor.setScalar(&[_]usize{1, 1}, 4.0);
    
    var b_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try b_tensor.setScalar(&[_]usize{0, 0}, 5.0);
    try b_tensor.setScalar(&[_]usize{0, 1}, 6.0);
    try b_tensor.setScalar(&[_]usize{1, 0}, 7.0);
    try b_tensor.setScalar(&[_]usize{1, 1}, 8.0);
    
    // Create nodes
    std.debug.print("Creating nodes\n", .{});
    var a = try autodiff.variable(allocator, a_tensor, true);
    var b = try autodiff.variable(allocator, b_tensor, true);
    
    // Forward pass: c = a + b
    std.debug.print("Creating add operation\n", .{});
    var c = try autodiff.add(allocator, a, b);
    
    // Check result
    std.debug.print("c[0,0] = {d}\n", .{try c.tensor.getScalar(&[_]usize{0, 0})});
    
    // Backward pass
    std.debug.print("Running backward pass\n", .{});
    try autodiff.backward(allocator, c);
    
    // Verify gradients
    std.debug.print("Gradient a[0,0] = {d}\n", .{try a.grad.?.getScalar(&[_]usize{0, 0})});
    
    // Test subtraction
    std.debug.print("\nTesting subtraction\n", .{});
    var a2_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a2_tensor.setScalar(&[_]usize{0, 0}, 10.0);
    try a2_tensor.setScalar(&[_]usize{0, 1}, 11.0);
    
    var b2_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try b2_tensor.setScalar(&[_]usize{0, 0}, 3.0);
    try b2_tensor.setScalar(&[_]usize{0, 1}, 4.0);
    
    var a2 = try autodiff.variable(allocator, a2_tensor, true);
    var b2 = try autodiff.variable(allocator, b2_tensor, true);
    
    var c2 = try autodiff.subtract(allocator, a2, b2);
    try autodiff.backward(allocator, c2);
    
    // Test ReLU
    std.debug.print("\nTesting ReLU\n", .{});
    var a3_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a3_tensor.setScalar(&[_]usize{0, 0}, -1.0);
    try a3_tensor.setScalar(&[_]usize{0, 1}, 2.0);
    
    var a3 = try autodiff.variable(allocator, a3_tensor, true);
    var c3 = try autodiff.relu(allocator, a3);
    try autodiff.backward(allocator, c3);
    
    // Test matmul
    std.debug.print("\nTesting matmul\n", .{});
    var a4_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a4_tensor.setScalar(&[_]usize{0, 0}, 1.0);
    try a4_tensor.setScalar(&[_]usize{0, 1}, 2.0);
    
    var b4_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try b4_tensor.setScalar(&[_]usize{0, 0}, 3.0);
    try b4_tensor.setScalar(&[_]usize{0, 1}, 4.0);
    
    var a4 = try autodiff.variable(allocator, a4_tensor, true);
    var b4 = try autodiff.variable(allocator, b4_tensor, true);
    
    var c4 = try autodiff.matmul(allocator, a4, b4);
    try autodiff.backward(allocator, c4);
    
    // Test softmax
    std.debug.print("\nTesting softmax\n", .{});
    var a5_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a5_tensor.setScalar(&[_]usize{0, 0}, 1.0);
    try a5_tensor.setScalar(&[_]usize{0, 1}, 2.0);
    
    var a5 = try autodiff.variable(allocator, a5_tensor, true);
    var c5 = try autodiff.softmax(allocator, a5);
    try autodiff.backward(allocator, c5);
    
    // Clean up
    std.debug.print("\nCleaning up all nodes\n", .{});
    a.deinit();
    b.deinit();
    c.deinit();
    
    a2.deinit();
    b2.deinit();
    c2.deinit();
    
    a3.deinit();
    c3.deinit();
    
    a4.deinit();
    b4.deinit();
    c4.deinit();
    
    a5.deinit();
    c5.deinit();
    
    std.debug.print("\nTest completed successfully\n", .{});
}