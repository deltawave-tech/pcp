const std = @import("std");
const pcp = @import("pcp");
const tensor = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;

/// Helper function for pointer casting
fn ptrCastHelper(comptime T: type, ptr: anytype) T {
    // We still need to use alignCast for pointers that require higher alignment
    return @ptrCast(@alignCast(ptr));
}

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const DType = tensor.DType;

// Helper function to print tensor contents
fn printTensor(t: Tensor) void {
    const buf = ptrCastHelper([*]f32, t.buffer.data.ptr)[0..t.shape.elemCount()];
    
    std.debug.print("Shape: [", .{});
    for (t.shape.dims) |dim| {
        std.debug.print("{}, ", .{dim});
    }
    std.debug.print("]\n", .{});
    
    if (t.shape.rank() == 2) {
        const rows = t.shape.dims[0];
        const cols = t.shape.dims[1];
        
        for (0..rows) |i| {
            std.debug.print("[ ", .{});
            for (0..cols) |j| {
                std.debug.print("{d:.4} ", .{buf[i * cols + j]});
            }
            std.debug.print("]\n", .{});
        }
    } else {
        // For other ranks, just print all values
        std.debug.print("[ ", .{});
        for (buf) |val| {
            std.debug.print("{d:.4} ", .{val});
        }
        std.debug.print("]\n", .{});
    }
}

// Test a simple model with Plan-based approach
fn testPlanBasedModel(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Plan-Based Model ===\n", .{});
    
    // Create simple tensors
    var dims = [_]usize{ 2, 2 };
    
    // Create tensors for parameters
    var w1_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer w1_tensor.deinit();
    
    try w1_tensor.setScalar(&[_]usize{0, 0}, 0.1);
    try w1_tensor.setScalar(&[_]usize{0, 1}, 0.2);
    try w1_tensor.setScalar(&[_]usize{1, 0}, 0.3);
    try w1_tensor.setScalar(&[_]usize{1, 1}, 0.4);
    
    var w2_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer w2_tensor.deinit();
    
    try w2_tensor.setScalar(&[_]usize{0, 0}, 0.5);
    try w2_tensor.setScalar(&[_]usize{0, 1}, 0.6);
    try w2_tensor.setScalar(&[_]usize{1, 0}, 0.7);
    try w2_tensor.setScalar(&[_]usize{1, 1}, 0.8);
    
    // Create input
    var x_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer x_tensor.deinit();
    
    try x_tensor.setScalar(&[_]usize{0, 0}, 1.0);
    try x_tensor.setScalar(&[_]usize{0, 1}, 2.0);
    try x_tensor.setScalar(&[_]usize{1, 0}, 3.0);
    try x_tensor.setScalar(&[_]usize{1, 1}, 4.0);
    
    // Target values
    var y_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer y_tensor.deinit();
    
    try y_tensor.setScalar(&[_]usize{0, 0}, 0.5);
    try y_tensor.setScalar(&[_]usize{0, 1}, 0.5);
    try y_tensor.setScalar(&[_]usize{1, 0}, 0.5);
    try y_tensor.setScalar(&[_]usize{1, 1}, 0.5);
    
    // Create the autodiff plans
    const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend, f32, null, null, null);
    var matmul_plan1 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan1.deinit();
    
    var matmul_plan2 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan2.deinit();
    
    const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend, f32, null);
    var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
    defer relu_plan.deinit();
    
    // Forward pass using plans
    std.debug.print("Running forward pass with plans...\n", .{});
    
    // h_pre = x @ w1
    const h_pre = try matmul_plan1.forward(.{ .a = x_tensor, .b = w1_tensor });
    defer h_pre.deinit();
    
    // h = relu(h_pre)
    const h = try relu_plan.forward(h_pre);
    defer h.deinit();
    
    // y_pred = h @ w2
    const y_pred = try matmul_plan2.forward(.{ .a = h, .b = w2_tensor });
    defer y_pred.deinit();
    
    // Print prediction
    std.debug.print("\nPrediction (Plan-based):\n", .{});
    printTensor(y_pred);
    
    // Compute MSE loss manually
    var diff = try ops.subtract(allocator, y_pred, y_tensor);
    defer diff.deinit();
    
    var diff_squared = try ops.multiply(allocator, diff, diff);
    defer diff_squared.deinit();
    
    // Calculate mean manually
    const diff_squared_buf = ptrCastHelper([*]f32, diff_squared.buffer.data.ptr)[0..diff_squared.shape.elemCount()];
    var sum: f32 = 0.0;
    for (diff_squared_buf) |val| {
        sum += val;
    }
    const mse = sum / @as(f32, @floatFromInt(diff_squared.shape.elemCount()));
    
    std.debug.print("\nMSE Loss: {d:.6}\n", .{mse});
    
    // Create gradient with ones
    var grad_ones = try Tensor.filled(allocator, y_pred.shape.dims, y_pred.dtype, 1.0, y_pred.backend);
    defer grad_ones.deinit();
    
    // Backward pass
    std.debug.print("\nRunning backward pass with plans...\n", .{});
    
    // Backprop through second matmul
    const grads2 = try matmul_plan2.backward(grad_ones);
    defer grads2.da.deinit();
    defer grads2.db.deinit();
    
    // Backprop through relu
    const relu_grad = try relu_plan.backward(grads2.da);
    defer relu_grad.deinit();
    
    // Backprop through first matmul
    const grads1 = try matmul_plan1.backward(relu_grad);
    defer grads1.da.deinit();
    defer grads1.db.deinit();
    
    // Print gradients
    std.debug.print("\nGradients for w1 (Plan-based):\n", .{});
    printTensor(grads1.db);
    
    std.debug.print("\nGradients for w2 (Plan-based):\n", .{});
    printTensor(grads2.db);
    
    std.debug.print("\nPlan-based model test completed successfully\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("Starting autodiff with Plan-based approach...\n", .{});
    
    // Test Plan-based model
    try testPlanBasedModel(allocator);
    
    std.debug.print("\nAll tests completed successfully!\n", .{});
}