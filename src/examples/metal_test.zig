const std = @import("std");
const pcp = @import("pcp");
const tensor = pcp.tensor;
const ops = pcp.ops;
const metal = @import("../backends/metal.zig");

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

// Test Metal backend operations
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize Metal backend
    try metal.init(allocator);
    defer metal.deinit();

    std.debug.print("Testing Metal backend operations...\n", .{});

    // Create test tensors
    var dims = [_]usize{ 2, 3 };

    // Create a tensor with CPU backend first
    var a_cpu = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a_cpu.deinit();

    // Set some values
    try a_cpu.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try a_cpu.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a_cpu.setScalar(&[_]usize{ 0, 2 }, 3.0);
    try a_cpu.setScalar(&[_]usize{ 1, 0 }, 4.0);
    try a_cpu.setScalar(&[_]usize{ 1, 1 }, 5.0);
    try a_cpu.setScalar(&[_]usize{ 1, 2 }, 6.0);

    // Create another tensor
    var b_cpu = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer b_cpu.deinit();

    try b_cpu.setScalar(&[_]usize{ 0, 0 }, 6.0);
    try b_cpu.setScalar(&[_]usize{ 0, 1 }, 5.0);
    try b_cpu.setScalar(&[_]usize{ 0, 2 }, 4.0);
    try b_cpu.setScalar(&[_]usize{ 1, 0 }, 3.0);
    try b_cpu.setScalar(&[_]usize{ 1, 1 }, 2.0);
    try b_cpu.setScalar(&[_]usize{ 1, 2 }, 1.0);

    // Create Metal tensors from CPU tensors
    var a_metal = try Tensor.zeros(allocator, &dims, .f32, .metal);
    defer a_metal.deinit();

    // Copy data from CPU tensor to Metal tensor
    const a_cpu_buf = ptrCastHelper([*]f32, a_cpu.buffer.data.ptr)[0..a_cpu.shape.elemCount()];
    const a_metal_buf = ptrCastHelper([*]f32, a_metal.buffer.data.ptr)[0..a_metal.shape.elemCount()];
    for (a_cpu_buf, 0..) |val, i| {
        a_metal_buf[i] = val;
    }

    var b_metal = try Tensor.zeros(allocator, &dims, .f32, .metal);
    defer b_metal.deinit();

    // Copy data from CPU tensor to Metal tensor
    const b_cpu_buf = ptrCastHelper([*]f32, b_cpu.buffer.data.ptr)[0..b_cpu.shape.elemCount()];
    const b_metal_buf = ptrCastHelper([*]f32, b_metal.buffer.data.ptr)[0..b_metal.shape.elemCount()];
    for (b_cpu_buf, 0..) |val, i| {
        b_metal_buf[i] = val;
    }

    // Print input tensors
    std.debug.print("\nTensor A (CPU):\n", .{});
    printTensor(a_cpu);

    std.debug.print("\nTensor B (CPU):\n", .{});
    printTensor(b_cpu);

    // Test element-wise operations

    // 1. Add
    std.debug.print("\nTesting element-wise addition on Metal...\n", .{});

    // Create test plans for Metal backend
    const AddPlanType = ops.AddPlan(metal.MetalBackend, f32, null);
    var add_plan = AddPlanType.init(allocator);

    // Execute operation
    const add_result = try add_plan.run(.{ .a = a_metal, .b = b_metal });
    defer add_result.deinit();

    // Print result
    std.debug.print("\nA + B (Metal):\n", .{});
    printTensor(add_result);

    // Compare with CPU result
    const add_cpu_result = try ops.add(allocator, a_cpu, b_cpu);
    defer add_cpu_result.deinit();

    std.debug.print("\nA + B (CPU for comparison):\n", .{});
    printTensor(add_cpu_result);

    // 2. Multiply
    std.debug.print("\nTesting element-wise multiplication on Metal...\n", .{});

    const MultiplyPlanType = ops.MultiplyPlan(metal.MetalBackend, f32, null);
    var multiply_plan = MultiplyPlanType.init(allocator);

    const multiply_result = try multiply_plan.run(.{ .a = a_metal, .b = b_metal });
    defer multiply_result.deinit();

    std.debug.print("\nA * B (Metal):\n", .{});
    printTensor(multiply_result);

    // Compare with CPU result
    const multiply_cpu_result = try ops.multiply(allocator, a_cpu, b_cpu);
    defer multiply_cpu_result.deinit();

    std.debug.print("\nA * B (CPU for comparison):\n", .{});
    printTensor(multiply_cpu_result);

    // 3. Test ReLU
    std.debug.print("\nTesting ReLU on Metal...\n", .{});

    // Create a tensor with negative values
    var c_dims = [_]usize{6};
    var c_cpu = try Tensor.zeros(allocator, &c_dims, .f32, .cpu);
    defer c_cpu.deinit();

    try c_cpu.setScalar(&[_]usize{0}, -2.0);
    try c_cpu.setScalar(&[_]usize{1}, -1.0);
    try c_cpu.setScalar(&[_]usize{2}, 0.0);
    try c_cpu.setScalar(&[_]usize{3}, 1.0);
    try c_cpu.setScalar(&[_]usize{4}, 2.0);
    try c_cpu.setScalar(&[_]usize{5}, 3.0);

    var c_metal = try Tensor.zeros(allocator, &c_dims, .f32, .metal);
    defer c_metal.deinit();

    // Copy data from CPU tensor to Metal tensor
    const c_cpu_buf = ptrCastHelper([*]f32, c_cpu.buffer.data.ptr)[0..c_cpu.shape.elemCount()];
    const c_metal_buf = ptrCastHelper([*]f32, c_metal.buffer.data.ptr)[0..c_metal.shape.elemCount()];
    for (c_cpu_buf, 0..) |val, i| {
        c_metal_buf[i] = val;
    }

    std.debug.print("\nInput tensor with negative values:\n", .{});
    printTensor(c_cpu);

    const ReluPlanType = ops.ReluPlan(metal.MetalBackend, f32, null);
    var relu_plan = ReluPlanType.init(allocator);

    const relu_result = try relu_plan.run(c_metal);
    defer relu_result.deinit();

    std.debug.print("\nReLU(C) (Metal):\n", .{});
    printTensor(relu_result);

    // Compare with CPU result
    const relu_cpu_result = try ops.relu(allocator, c_cpu);
    defer relu_cpu_result.deinit();

    std.debug.print("\nReLU(C) (CPU for comparison):\n", .{});
    printTensor(relu_cpu_result);

    // 4. Test matrix multiplication
    std.debug.print("\nTesting matrix multiplication on Metal...\n", .{});

    // Create matrices for matmul
    var mat_a_dims = [_]usize{ 2, 3 };
    var mat_a_cpu = try Tensor.zeros(allocator, &mat_a_dims, .f32, .cpu);
    defer mat_a_cpu.deinit();

    try mat_a_cpu.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try mat_a_cpu.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try mat_a_cpu.setScalar(&[_]usize{ 0, 2 }, 3.0);
    try mat_a_cpu.setScalar(&[_]usize{ 1, 0 }, 4.0);
    try mat_a_cpu.setScalar(&[_]usize{ 1, 1 }, 5.0);
    try mat_a_cpu.setScalar(&[_]usize{ 1, 2 }, 6.0);

    var mat_b_dims = [_]usize{ 3, 2 };
    var mat_b_cpu = try Tensor.zeros(allocator, &mat_b_dims, .f32, .cpu);
    defer mat_b_cpu.deinit();

    try mat_b_cpu.setScalar(&[_]usize{ 0, 0 }, 7.0);
    try mat_b_cpu.setScalar(&[_]usize{ 0, 1 }, 8.0);
    try mat_b_cpu.setScalar(&[_]usize{ 1, 0 }, 9.0);
    try mat_b_cpu.setScalar(&[_]usize{ 1, 1 }, 10.0);
    try mat_b_cpu.setScalar(&[_]usize{ 2, 0 }, 11.0);
    try mat_b_cpu.setScalar(&[_]usize{ 2, 1 }, 12.0);

    var mat_a_metal = try Tensor.zeros(allocator, &mat_a_dims, .f32, .metal);
    defer mat_a_metal.deinit();

    // Copy data from CPU tensor to Metal tensor
    const mat_a_cpu_buf = ptrCastHelper([*]f32, mat_a_cpu.buffer.data.ptr)[0..mat_a_cpu.shape.elemCount()];
    const mat_a_metal_buf = ptrCastHelper([*]f32, mat_a_metal.buffer.data.ptr)[0..mat_a_metal.shape.elemCount()];
    for (mat_a_cpu_buf, 0..) |val, i| {
        mat_a_metal_buf[i] = val;
    }

    var mat_b_metal = try Tensor.zeros(allocator, &mat_b_dims, .f32, .metal);
    defer mat_b_metal.deinit();

    // Copy data from CPU tensor to Metal tensor
    const mat_b_cpu_buf = ptrCastHelper([*]f32, mat_b_cpu.buffer.data.ptr)[0..mat_b_cpu.shape.elemCount()];
    const mat_b_metal_buf = ptrCastHelper([*]f32, mat_b_metal.buffer.data.ptr)[0..mat_b_metal.shape.elemCount()];
    for (mat_b_cpu_buf, 0..) |val, i| {
        mat_b_metal_buf[i] = val;
    }

    std.debug.print("\nMatrix A:\n", .{});
    printTensor(mat_a_cpu);

    std.debug.print("\nMatrix B:\n", .{});
    printTensor(mat_b_cpu);

    const MatmulPlanType = ops.MatmulPlan(metal.MetalBackend, f32, null, null, null);
    var matmul_plan = MatmulPlanType.init(allocator);

    const matmul_result = try matmul_plan.run(.{ .a = mat_a_metal, .b = mat_b_metal });
    defer matmul_result.deinit();

    std.debug.print("\nA @ B (Metal):\n", .{});
    printTensor(matmul_result);

    // Compare with CPU result
    const matmul_cpu_result = try ops.matmul(allocator, mat_a_cpu, mat_b_cpu);
    defer matmul_cpu_result.deinit();

    std.debug.print("\nA @ B (CPU for comparison):\n", .{});
    printTensor(matmul_cpu_result);

    std.debug.print("\nAll Metal tests completed successfully!\n", .{});
}
