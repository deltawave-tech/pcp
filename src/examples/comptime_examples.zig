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

/// Helper function to print tensor values
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

/// Example showing basic comptime plan usage
fn demoBasicPlans(allocator: Allocator) !void {
    std.debug.print("\n=== Basic Comptime Operation Plans ===\n", .{});
    
    // Create tensors with known shapes
    const dims = [_]usize{ 2, 2 };
    
    // Create input tensors
    var a = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{0, 0}, 1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{1, 0}, 3.0);
    try a.setScalar(&[_]usize{1, 1}, 4.0);
    
    var b = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{0, 0}, 5.0);
    try b.setScalar(&[_]usize{0, 1}, 6.0);
    try b.setScalar(&[_]usize{1, 0}, 7.0);
    try b.setScalar(&[_]usize{1, 1}, 8.0);
    
    std.debug.print("Input tensor A:\n", .{});
    printTensor(a);
    
    std.debug.print("\nInput tensor B:\n", .{});
    printTensor(b);
    
    // Create an AddPlan with the CPU backend and fixed shape
    const shape_opt: ?[]const usize = &dims;
    var plan = ops.AddPlan(ops.CpuBackend, f32, shape_opt).init(allocator);
    
    // Execute the plan
    const result = try plan.run(.{ .a = a, .b = b });
    defer result.deinit();
    
    std.debug.print("\nResult of A + B:\n", .{});
    printTensor(result);
    
    // Create a MatMulPlan
    const M: ?usize = 2;
    const N: ?usize = 2;
    const P: ?usize = 2;
    var matmul_plan = ops.MatmulPlan(ops.CpuBackend, f32, M, N, P).init(allocator);
    
    // Execute the matmul plan
    const matmul_result = try matmul_plan.run(.{ .a = a, .b = b });
    defer matmul_result.deinit();
    
    std.debug.print("\nResult of A @ B (matmul):\n", .{});
    printTensor(matmul_result);
    
    // Create a ReluPlan
    var relu_plan = ops.ReluPlan(ops.CpuBackend, f32, shape_opt).init(allocator);
    
    // Create a tensor with negative values to test ReLU
    var c = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer c.deinit();
    try c.setScalar(&[_]usize{0, 0}, -1.0);
    try c.setScalar(&[_]usize{0, 1}, 2.0);
    try c.setScalar(&[_]usize{1, 0}, -3.0);
    try c.setScalar(&[_]usize{1, 1}, 4.0);
    
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
    var a = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer a.deinit();
    try a.setScalar(&[_]usize{0, 0}, 1.0);
    try a.setScalar(&[_]usize{0, 1}, 2.0);
    try a.setScalar(&[_]usize{1, 0}, 3.0);
    try a.setScalar(&[_]usize{1, 1}, 4.0);
    
    var b = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer b.deinit();
    try b.setScalar(&[_]usize{0, 0}, 5.0);
    try b.setScalar(&[_]usize{0, 1}, 6.0);
    try b.setScalar(&[_]usize{1, 0}, 7.0);
    try b.setScalar(&[_]usize{1, 1}, 8.0);
    
    // Create an AddPlan with gradient support
    const shape_opt: ?[]const usize = &dims;
    
    // For direct ops.AddPlan
    const BasePlanType = ops.AddPlan(ops.CpuBackend, f32, shape_opt);
    
    // Instead of adding with autodiff.AddPlanWithGrad, define the plan type inline for debugging
    const PlanType = struct {
        const Self = @This();
        const InputType = BasePlanType.InputType;
        const op_type = BasePlanType.op_type;
        const GradType = struct { da: Tensor, db: Tensor };
        
        base: BasePlanType,
        
        pub fn init(alloc: Allocator) Self {
            return .{ .base = BasePlanType.init(alloc) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
    
    var auto_diff = autodiff.AutoDiffPlan(PlanType).init(allocator);
    defer auto_diff.deinit();
    
    // Forward pass
    std.debug.print("Running forward pass (A + B)...\n", .{});
    const output = try auto_diff.forward(.{ .a = a, .b = b });
    defer output.deinit();
    
    std.debug.print("\nForward output:\n", .{});
    printTensor(output);
    
    // Create gradient tensor (all ones for simplicity)
    var grad_out = try Tensor.filled(allocator, &dims, .f32, 1.0, .cpu);
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
    
    // For direct ops.MatmulPlan
    const BaseMatmulPlanType = ops.MatmulPlan(ops.CpuBackend, f32, M, N, P);
    
    // Create the MatMul plan type inline for debugging
    const MatMulPlanType = struct {
        const Self = @This();
        const InputType = BaseMatmulPlanType.InputType;
        const op_type = BaseMatmulPlanType.op_type;
        const GradType = struct { da: Tensor, db: Tensor };
        
        base: BaseMatmulPlanType,
        
        pub fn init(alloc: Allocator) Self {
            return .{ .base = BaseMatmulPlanType.init(alloc) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
    
    var matmul_autodiff = autodiff.AutoDiffPlan(MatMulPlanType).init(allocator);
    defer matmul_autodiff.deinit();
    
    // Forward pass for matmul
    std.debug.print("\nRunning forward pass for MatMul (A @ B)...\n", .{});
    const matmul_output = try matmul_autodiff.forward(.{ .a = a, .b = b });
    defer matmul_output.deinit();
    
    std.debug.print("\nMatMul output:\n", .{});
    printTensor(matmul_output);
    
    // Create gradient tensor for matmul output
    var matmul_grad = try Tensor.filled(allocator, matmul_output.shape.dims, .f32, 1.0, .cpu);
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
    var input = try Tensor.zeros(allocator, &input_dims, .f32, .cpu);
    defer input.deinit();
    try input.setScalar(&[_]usize{0, 0}, 0.5);
    try input.setScalar(&[_]usize{0, 1}, -0.5);
    try input.setScalar(&[_]usize{1, 0}, 1.0);
    try input.setScalar(&[_]usize{1, 1}, -1.0);
    
    std.debug.print("Network input:\n", .{});
    printTensor(input);
    
    // Create weight tensors
    // Weight 1: 2x2
    var w1 = try Tensor.zeros(allocator, &input_dims, .f32, .cpu);
    defer w1.deinit();
    try w1.setScalar(&[_]usize{0, 0}, 0.1);
    try w1.setScalar(&[_]usize{0, 1}, 0.2);
    try w1.setScalar(&[_]usize{1, 0}, 0.3);
    try w1.setScalar(&[_]usize{1, 1}, 0.4);
    
    // Weight 2: 2x2
    var w2 = try Tensor.zeros(allocator, &input_dims, .f32, .cpu);
    defer w2.deinit();
    try w2.setScalar(&[_]usize{0, 0}, 0.5);
    try w2.setScalar(&[_]usize{0, 1}, 0.6);
    try w2.setScalar(&[_]usize{1, 0}, 0.7);
    try w2.setScalar(&[_]usize{1, 1}, 0.8);
    
    // Set up the first layer (MatMul + ReLU)
    const M1: ?usize = 2;
    const N1: ?usize = 2;
    const P1: ?usize = 2;
    
    // Create direct plan types
    const BaseMatmul1PlanType = ops.MatmulPlan(ops.CpuBackend, f32, M1, N1, P1);
    const BaseReluPlanType = ops.ReluPlan(ops.CpuBackend, f32, &input_dims);
    const BaseMatmul2PlanType = ops.MatmulPlan(ops.CpuBackend, f32, M1, N1, P1);
    
    // Define wrapper types with GradType
    const Matmul1PlanType = struct {
        const Self = @This();
        const InputType = BaseMatmul1PlanType.InputType;
        const op_type = BaseMatmul1PlanType.op_type;
        const GradType = struct { da: Tensor, db: Tensor };
        
        base: BaseMatmul1PlanType,
        
        pub fn init(alloc: Allocator) Self {
            return .{ .base = BaseMatmul1PlanType.init(alloc) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
    
    const ReluPlanType = struct {
        const Self = @This();
        const InputType = Tensor;
        const op_type = BaseReluPlanType.op_type;
        const GradType = Tensor;
        
        base: BaseReluPlanType,
        
        pub fn init(alloc: Allocator) Self {
            return .{ .base = BaseReluPlanType.init(alloc) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
    
    const Matmul2PlanType = struct {
        const Self = @This();
        const InputType = BaseMatmul2PlanType.InputType;
        const op_type = BaseMatmul2PlanType.op_type;
        const GradType = struct { da: Tensor, db: Tensor };
        
        base: BaseMatmul2PlanType,
        
        pub fn init(alloc: Allocator) Self {
            return .{ .base = BaseMatmul2PlanType.init(alloc) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
    
    // Create plan instances
    const matmul1_plan = Matmul1PlanType.init(allocator);
    const relu_plan = ReluPlanType.init(allocator);
    const matmul2_plan = Matmul2PlanType.init(allocator);
    
    // Create AutoDiff wrappers
    var matmul1_autodiff = autodiff.AutoDiffPlan(Matmul1PlanType).init(allocator);
    defer matmul1_autodiff.deinit();
    
    var relu_autodiff = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
    defer relu_autodiff.deinit();
    
    var matmul2_autodiff = autodiff.AutoDiffPlan(Matmul2PlanType).init(allocator);
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
    var target = try Tensor.filled(allocator, output.shape.dims, .f32, 0.5, .cpu);
    defer target.deinit();
    
    std.debug.print("\nTarget output:\n", .{});
    printTensor(target);
    
    // Compute a simple loss: MSE = (output - target)^2
    // Create direct plan types
    const BaseSubtractPlanType = ops.SubtractPlan(ops.CpuBackend, f32, &input_dims);
    const BaseMultiplyPlanType = ops.MultiplyPlan(ops.CpuBackend, f32, &input_dims);
    
    // Define wrapper types with GradType
    const SubtractPlanType = struct {
        const Self = @This();
        const InputType = BaseSubtractPlanType.InputType;
        const op_type = BaseSubtractPlanType.op_type;
        const GradType = struct { da: Tensor, db: Tensor };
        
        base: BaseSubtractPlanType,
        
        pub fn init(alloc: Allocator) Self {
            return .{ .base = BaseSubtractPlanType.init(alloc) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
    
    const MultiplyPlanType = struct {
        const Self = @This();
        const InputType = BaseMultiplyPlanType.InputType;
        const op_type = BaseMultiplyPlanType.op_type;
        const GradType = struct { da: Tensor, db: Tensor };
        
        base: BaseMultiplyPlanType,
        
        pub fn init(alloc: Allocator) Self {
            return .{ .base = BaseMultiplyPlanType.init(alloc) };
        }
        
        pub fn run(self: Self, input: InputType) !Tensor {
            return self.base.run(input);
        }
    };
    
    // Create plan instances
    const subtract_plan = SubtractPlanType.init(allocator);
    const multiply_plan = MultiplyPlanType.init(allocator);
    
    // Create AutoDiff wrappers
    var subtract_autodiff = autodiff.AutoDiffPlan(SubtractPlanType).init(allocator);
    defer subtract_autodiff.deinit();
    
    var multiply_autodiff = autodiff.AutoDiffPlan(MultiplyPlanType).init(allocator);
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
    var loss_grad = try Tensor.filled(allocator, loss.shape.dims, .f32, 1.0, .cpu);
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