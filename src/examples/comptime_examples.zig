const std = @import("std");
const pcp = @import("pcp");
const tensor = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;

const Allocator = std.mem.Allocator;
const DataType = f32;
const Tensor = tensor.Tensor(DataType);

/// Helper function to print tensor values
fn printTensor(t: Tensor) void {
    const buf = t.buffer.data.ptr;

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
        for (t.buffer.data) |val| {
            std.debug.print("{d:.4} ", .{val});
        }
        std.debug.print("]\n", .{});
    }
}

/// Example showing basic comptime plan usage
fn demoBasicPlans(allocator: Allocator) !void {
    std.debug.print("\n=== Basic Comptime Operation Plans ===\n", .{});

    // Create tensors with known shapes
    const dims = [_]usize{ 2, 2 };

    // Create input tensors
    var a = try Tensor.zeros(allocator, &dims, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 3.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 4.0);

    var b = try Tensor.zeros(allocator, &dims, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{ 0, 0 }, 5.0);
    try b.setScalar(&[_]usize{ 0, 1 }, 6.0);
    try b.setScalar(&[_]usize{ 1, 0 }, 7.0);
    try b.setScalar(&[_]usize{ 1, 1 }, 8.0);

    std.debug.print("Input tensor A:\n", .{});
    printTensor(a);

    std.debug.print("\nInput tensor B:\n", .{});
    printTensor(b);

    // Create an AddPlan with the CPU backend and fixed shape
    const shape_opt: ?[]const usize = &dims;
    var plan = ops.AddPlan(ops.CpuBackend(DataType), DataType, shape_opt).init(allocator);

    // Execute the plan
    const result = try plan.run(.{ .a = a, .b = b });
    defer result.deinit();

    std.debug.print("\nResult of A + B:\n", .{});
    printTensor(result);

    // Create a MatMulPlan
    const M: ?usize = 2;
    const N: ?usize = 2;
    const P: ?usize = 2;
    var matmul_plan = ops.MatmulPlan(ops.CpuBackend(DataType), DataType, M, N, P).init(allocator);

    // Execute the matmul plan
    const matmul_result = try matmul_plan.run(.{ .a = a, .b = b });
    defer matmul_result.deinit();

    std.debug.print("\nResult of A @ B (matmul):\n", .{});
    printTensor(matmul_result);

    // Create a ReluPlan
    var relu_plan = ops.ReluPlan(ops.CpuBackend(DataType), DataType, shape_opt).init(allocator);

    // Create a tensor with negative values to test ReLU
    var c = try Tensor.zeros(allocator, &dims, .cpu);
    defer c.deinit();
    try c.setScalar(&[_]usize{ 0, 0 }, -1.0);
    try c.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try c.setScalar(&[_]usize{ 1, 0 }, -3.0);
    try c.setScalar(&[_]usize{ 1, 1 }, 4.0);

    std.debug.print("\nInput tensor C (with negative values):\n", .{});
    printTensor(c);

    // Execute the relu plan
    const relu_result = try relu_plan.run(c);
    defer relu_result.deinit();

    std.debug.print("\nResult of ReLU(C):\n", .{});
    printTensor(relu_result);
}

/// Example showing automatic differentiation with comptime plans
fn demoAutoDiff(allocator: Allocator) !void {
    std.debug.print("\n=== Comptime AutoDiff with Plans ===\n", .{});

    // Create tensors with known shapes
    const dims = [_]usize{ 2, 2 };

    // Create input tensors
    var a = try Tensor.zeros(allocator, &dims, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{ 0, 0 }, 1.0);
    try a.setScalar(&[_]usize{ 0, 1 }, 2.0);
    try a.setScalar(&[_]usize{ 1, 0 }, 3.0);
    try a.setScalar(&[_]usize{ 1, 1 }, 4.0);

    var b = try Tensor.zeros(allocator, &dims, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{ 0, 0 }, 5.0);
    try b.setScalar(&[_]usize{ 0, 1 }, 6.0);
    try b.setScalar(&[_]usize{ 1, 0 }, 7.0);
    try b.setScalar(&[_]usize{ 1, 1 }, 8.0);

    // Create an AddPlan with gradient support
    const shape_opt: ?[]const usize = &dims;

    // Use the predefined WithGrad plan that adds proper GradType declarations
    const PlanType = autodiff.AddPlanWithGrad(ops.CpuBackend(DataType), DataType, shape_opt);

    // Now we use the official WithGrad plan types that have proper GradType and op_type declarations

    var auto_diff = autodiff.AutoDiffPlan(PlanType, DataType).init(allocator);
    defer auto_diff.deinit();

    // Forward pass
    std.debug.print("Running forward pass (A + B)...\n", .{});
    const output = try auto_diff.forward(.{ .a = a, .b = b });
    defer output.deinit();

    std.debug.print("\nForward output:\n", .{});
    printTensor(output);

    // Create gradient tensor (all ones for simplicity)
    var grad_out = try Tensor.filled(allocator, &dims, 1.0, .cpu);
    defer grad_out.deinit();

    std.debug.print("\nUpstream gradient (all ones):\n", .{});
    printTensor(grad_out);

    // Backward pass
    std.debug.print("\nRunning backward pass...\n", .{});
    const grads = try auto_diff.backward(grad_out);
    defer grads.da.deinit();
    defer grads.db.deinit();

    // Print gradients
    std.debug.print("\nGradient for A:\n", .{});
    printTensor(grads.da);

    std.debug.print("\nGradient for B:\n", .{});
    printTensor(grads.db);

    // Now try with a more complex operation: MatMul with gradients
    const M: ?usize = 2;
    const N: ?usize = 2;
    const P: ?usize = 2;

    // Use WithGrad plan that has proper GradType declaration
    const MatMulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, M, N, P);

    var matmul_autodiff = autodiff.AutoDiffPlan(MatMulPlanType, DataType).init(allocator);
    defer matmul_autodiff.deinit();

    // Forward pass for matmul
    std.debug.print("\nRunning forward pass for MatMul (A @ B)...\n", .{});
    const matmul_output = try matmul_autodiff.forward(.{ .a = a, .b = b });
    defer matmul_output.deinit();

    std.debug.print("\nMatMul output:\n", .{});
    printTensor(matmul_output);

    // Create gradient tensor for matmul output
    var matmul_grad = try Tensor.filled(allocator, matmul_output.shape.dims, 1.0, .cpu);
    defer matmul_grad.deinit();

    // Backward pass for matmul
    std.debug.print("\nRunning backward pass for MatMul...\n", .{});
    const matmul_grads = try matmul_autodiff.backward(matmul_grad);
    defer matmul_grads.da.deinit();
    defer matmul_grads.db.deinit();

    // Print matmul gradients
    std.debug.print("\nMatMul gradient for A:\n", .{});
    printTensor(matmul_grads.da);

    std.debug.print("\nMatMul gradient for B:\n", .{});
    printTensor(matmul_grads.db);
}

/// Simple example of a multi-layer network using comptime plans
fn demoSimpleNetwork(allocator: Allocator) !void {
    std.debug.print("\n=== Simple Neural Network with Comptime Plans ===\n", .{});

    // Network architecture: 2x2 input -> MatMul -> ReLU -> MatMul -> output

    // Create input tensor (2x2)
    const input_dims = [_]usize{ 2, 2 };
    var input = try Tensor.zeros(allocator, &input_dims, .cpu);
    defer input.deinit();
    try input.setScalar(&[_]usize{ 0, 0 }, 0.5);
    try input.setScalar(&[_]usize{ 0, 1 }, -0.5);
    try input.setScalar(&[_]usize{ 1, 0 }, 1.0);
    try input.setScalar(&[_]usize{ 1, 1 }, -1.0);

    std.debug.print("Network input:\n", .{});
    printTensor(input);

    // Create weight tensors
    // Weight 1: 2x2
    var w1 = try Tensor.zeros(allocator, &input_dims, .cpu);
    defer w1.deinit();
    try w1.setScalar(&[_]usize{ 0, 0 }, 0.1);
    try w1.setScalar(&[_]usize{ 0, 1 }, 0.2);
    try w1.setScalar(&[_]usize{ 1, 0 }, 0.3);
    try w1.setScalar(&[_]usize{ 1, 1 }, 0.4);

    // Weight 2: 2x2
    var w2 = try Tensor.zeros(allocator, &input_dims, .cpu);
    defer w2.deinit();
    try w2.setScalar(&[_]usize{ 0, 0 }, 0.5);
    try w2.setScalar(&[_]usize{ 0, 1 }, 0.6);
    try w2.setScalar(&[_]usize{ 1, 0 }, 0.7);
    try w2.setScalar(&[_]usize{ 1, 1 }, 0.8);

    // Set up the first layer (MatMul + ReLU)
    const M1: ?usize = 2;
    const N1: ?usize = 2;
    const P1: ?usize = 2;

    // Use WithGrad plans for proper autodiff support
    const Matmul1PlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, M1, N1, P1);
    const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend(DataType), DataType, &input_dims);
    const Matmul2PlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend(DataType), DataType, M1, N1, P1);

    // Create AutoDiff wrappers directly without storing unused plans

    // Create AutoDiff wrappers
    var matmul1_autodiff = autodiff.AutoDiffPlan(Matmul1PlanType, DataType).init(allocator);
    defer matmul1_autodiff.deinit();

    var relu_autodiff = autodiff.AutoDiffPlan(ReluPlanType, DataType).init(allocator);
    defer relu_autodiff.deinit();

    var matmul2_autodiff = autodiff.AutoDiffPlan(Matmul2PlanType, DataType).init(allocator);
    defer matmul2_autodiff.deinit();

    // Forward pass through the network
    std.debug.print("\nRunning forward pass through the network...\n", .{});

    // Layer 1: MatMul(input, w1)
    const layer1_out = try matmul1_autodiff.forward(.{ .a = input, .b = w1 });
    defer layer1_out.deinit();

    std.debug.print("\nAfter first matrix multiplication:\n", .{});
    printTensor(layer1_out);

    // Apply ReLU
    const relu_out = try relu_autodiff.forward(layer1_out);
    defer relu_out.deinit();

    std.debug.print("\nAfter ReLU activation:\n", .{});
    printTensor(relu_out);

    // Layer 2: MatMul(relu_out, w2)
    const output = try matmul2_autodiff.forward(.{ .a = relu_out, .b = w2 });
    defer output.deinit();

    std.debug.print("\nNetwork output:\n", .{});
    printTensor(output);

    // Define a target output for simple MSE loss
    var target = try Tensor.filled(allocator, output.shape.dims, 0.5, .cpu);
    defer target.deinit();

    std.debug.print("\nTarget output:\n", .{});
    printTensor(target);

    // Compute a simple loss: MSE = (output - target)^2
    // Use WithGrad plans for proper autodiff support
    const SubtractPlanType = autodiff.SubtractPlanWithGrad(ops.CpuBackend(DataType), DataType, &input_dims);
    const MultiplyPlanType = autodiff.MultiplyPlanWithGrad(ops.CpuBackend(DataType), DataType, &input_dims);

    // Create AutoDiff wrappers directly without storing unused plans

    // Create AutoDiff wrappers
    var subtract_autodiff = autodiff.AutoDiffPlan(SubtractPlanType, DataType).init(allocator);
    defer subtract_autodiff.deinit();

    var multiply_autodiff = autodiff.AutoDiffPlan(MultiplyPlanType, DataType).init(allocator);
    defer multiply_autodiff.deinit();

    // Calculate difference
    const diff = try subtract_autodiff.forward(.{ .a = output, .b = target });
    defer diff.deinit();

    std.debug.print("\nDifference (output - target):\n", .{});
    printTensor(diff);

    // Square the difference
    const loss = try multiply_autodiff.forward(.{ .a = diff, .b = diff });
    defer loss.deinit();

    std.debug.print("\nLoss (MSE):\n", .{});
    printTensor(loss);

    // Backward pass through the network
    std.debug.print("\nRunning backward pass to compute gradients...\n", .{});

    // Create gradient tensor (all ones for simplicity)
    var loss_grad = try Tensor.filled(allocator, loss.shape.dims, 1.0, .cpu);
    defer loss_grad.deinit();

    // Backpropagation for each layer in reverse order
    const multiply_grads = try multiply_autodiff.backward(loss_grad);
    defer multiply_grads.da.deinit();
    defer multiply_grads.db.deinit();

    const subtract_grads = try subtract_autodiff.backward(multiply_grads.da);
    defer subtract_grads.da.deinit();
    defer subtract_grads.db.deinit();

    const matmul2_grads = try matmul2_autodiff.backward(subtract_grads.da);
    defer matmul2_grads.da.deinit();
    defer matmul2_grads.db.deinit();

    // ReLU returns a single gradient tensor since it's a unary operation
    const relu_grad = try relu_autodiff.backward(matmul2_grads.da);
    defer relu_grad.deinit();

    const matmul1_grads = try matmul1_autodiff.backward(relu_grad);
    defer matmul1_grads.da.deinit();
    defer matmul1_grads.db.deinit();

    // Print weight gradients
    std.debug.print("\nGradient for weight 1 (first layer):\n", .{});
    printTensor(matmul1_grads.db);

    std.debug.print("\nGradient for weight 2 (second layer):\n", .{});
    printTensor(matmul2_grads.db);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Comptime Plan Examples ===\n", .{});

    // Demonstrate basic operations with comptime plans
    try demoBasicPlans(allocator);

    // Demonstrate autodiff with comptime plans
    try demoAutoDiff(allocator);

    // Demonstrate a simple neural network with comptime plans
    try demoSimpleNetwork(allocator);

    std.debug.print("\nAll examples completed successfully!\n", .{});
}
