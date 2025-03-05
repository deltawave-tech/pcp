// PCP Plan Example
// Demonstrates the comptime plan generation approach

const std = @import("std");
const Allocator = std.mem.Allocator;

// A simplified tensor for demonstration
pub const Tensor = struct {
    shape: []const usize,
    data: []f32,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, shape: []const usize, data: []f32) !Tensor {
        const shape_copy = try allocator.dupe(usize, shape);
        const data_copy = try allocator.dupe(f32, data);
        
        return Tensor{
            .shape = shape_copy,
            .data = data_copy,
            .allocator = allocator,
        };
    }
    
    pub fn zeros(allocator: Allocator, shape: []const usize) !Tensor {
        var size: usize = 1;
        for (shape) |dim| {
            size *= dim;
        }
        
        const data = try allocator.alloc(f32, size);
        std.mem.set(f32, data, 0);
        
        const shape_copy = try allocator.dupe(usize, shape);
        
        return Tensor{
            .shape = shape_copy,
            .data = data,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.shape);
        self.allocator.free(self.data);
    }
    
    pub fn print(self: Tensor) void {
        std.debug.print("Tensor shape: ", .{});
        for (self.shape) |dim| {
            std.debug.print("{} ", .{dim});
        }
        std.debug.print("\nData: ", .{});
        for (self.data) |val| {
            std.debug.print("{d:.2} ", .{val});
        }
        std.debug.print("\n", .{});
    }
};

// Operation types
pub const ElementwiseOp = enum {
    add,
    multiply,
    relu,
    tanh,
};

pub const ReduceOp = enum {
    sum,
    max,
    mean,
};

// Backend interface
pub fn Backend(comptime T: type) type {
    return struct {
        // Base implementation
        pub fn reshape(tensor: TensorSpec, new_shape: []const usize) Operation {
            return Operation{
                .op_type = .reshape,
                .inputs = &[_]TensorSpec{tensor},
                .parameters = .{
                    .reshape = .{
                        .shape = new_shape,
                    },
                },
            };
        }
        
        pub fn elementwiseBinary(a: TensorSpec, b: TensorSpec, op: ElementwiseOp) Operation {
            return Operation{
                .op_type = .binary,
                .inputs = &[_]TensorSpec{ a, b },
                .parameters = .{
                    .binary = .{
                        .op = op,
                    },
                },
            };
        }
        
        pub fn reduce(tensor: TensorSpec, op: ReduceOp, axes: []const usize) Operation {
            return Operation{
                .op_type = .reduce,
                .inputs = &[_]TensorSpec{tensor},
                .parameters = .{
                    .reduce = .{
                        .op = op,
                        .axes = axes,
                    },
                },
            };
        }
        
        // CPU-specific optimizations
        pub fn optimize(plan: Plan) Plan {
            // Here we could apply CPU-specific optimizations
            // For now, just return the original plan
            return plan;
        }
    };
}

// Operation and plan structures
pub const OperationType = enum {
    reshape,
    binary,
    reduce,
};

pub const Operation = struct {
    op_type: OperationType,
    inputs: []const TensorSpec,
    parameters: Parameters,
    
    const Parameters = union {
        reshape: struct {
            shape: []const usize,
        },
        binary: struct {
            op: ElementwiseOp,
        },
        reduce: struct {
            op: ReduceOp,
            axes: []const usize,
        },
    };
};

pub const TensorSpec = struct {
    name: []const u8,
    shape: ?[]const usize = null,
};

pub const Plan = struct {
    operations: []const Operation,
    inputs: []const []const u8,
    output: TensorSpec,
    
    pub fn execute(self: Plan, allocator: Allocator, inputs: []const Tensor) !Tensor {
        // In a real implementation, this would execute the plan
        // For this example, we'll just print the plan and return a dummy tensor
        
        std.debug.print("Executing plan with {} operations\n", .{self.operations.len});
        for (self.operations, 0..) |op, i| {
            std.debug.print("  Op {}: {} with {} inputs\n", .{i, op.op_type, op.inputs.len});
        }
        
        // Return a dummy tensor
        return try Tensor.zeros(allocator, &[_]usize{2, 2});
    }
};

// High-level operation plan generators
pub fn matmulPlan(comptime a_shape: []const usize, comptime b_shape: []const usize) Plan {
    if (a_shape[1] != b_shape[0]) {
        @compileError("Matrix dimensions don't match for multiplication");
    }
    
    // Input tensors
    const a_spec = TensorSpec{ .name = "A", .shape = a_shape };
    const b_spec = TensorSpec{ .name = "B", .shape = b_shape };
    
    // Output shape
    const out_shape = [_]usize{ a_shape[0], b_shape[1] };
    
    // Create operations for the plan
    const reshape_a = Backend(void).reshape(a_spec, &[_]usize{ a_shape[0], a_shape[1], 1 });
    const reshape_b = Backend(void).reshape(b_spec, &[_]usize{ 1, b_shape[0], b_shape[1] });
    
    // In a full implementation, we'd handle broadcasting here
    const multiply = Backend(void).elementwiseBinary(
        TensorSpec{ .name = "reshaped_A" },
        TensorSpec{ .name = "reshaped_B" },
        .multiply
    );
    
    const reduce_sum = Backend(void).reduce(
        TensorSpec{ .name = "product" },
        .sum,
        &[_]usize{1}
    );
    
    // Compile into a plan
    return Plan{
        .operations = &[_]Operation{ reshape_a, reshape_b, multiply, reduce_sum },
        .inputs = &[_][]const u8{ "A", "B" },
        .output = TensorSpec{ .name = "C", .shape = &out_shape },
    };
}

// Simple softmax example using elementwise operations
pub fn softmaxPlan(comptime shape: []const usize, comptime axis: usize) Plan {
    const input = TensorSpec{ .name = "input", .shape = shape };
    
    // max(x) for numerical stability
    const max_reduce = Backend(void).reduce(input, .max, &[_]usize{axis});
    
    // x - max(x)
    const subtract = Backend(void).elementwiseBinary(
        input,
        TensorSpec{ .name = "max_values" },
        .add // Assuming subtract is implemented as add with negative
    );
    
    // exp(x - max(x))
    const exp = Backend(void).elementwiseBinary(
        TensorSpec{ .name = "shifted" },
        TensorSpec{ .name = "dummy" }, // Not used for unary ops
        .tanh // As an example, real implementation would use exp
    );
    
    // sum(exp(x - max(x)))
    const sum = Backend(void).reduce(
        TensorSpec{ .name = "exp_values" },
        .sum,
        &[_]usize{axis}
    );
    
    // exp(x - max(x)) / sum(exp(x - max(x)))
    const divide = Backend(void).elementwiseBinary(
        TensorSpec{ .name = "exp_values" },
        TensorSpec{ .name = "sum_values" },
        .multiply // Assuming divide is implemented as multiply with reciprocal
    );
    
    return Plan{
        .operations = &[_]Operation{ max_reduce, subtract, exp, sum, divide },
        .inputs = &[_][]const u8{"input"},
        .output = TensorSpec{ .name = "softmax", .shape = shape },
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    // Create input tensors
    const a_shape = [_]usize{ 2, 3 };
    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var tensor_a = try Tensor.init(allocator, &a_shape, &a_data);
    defer tensor_a.deinit();
    
    const b_shape = [_]usize{ 3, 2 };
    const b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var tensor_b = try Tensor.init(allocator, &b_shape, &b_data);
    defer tensor_b.deinit();
    
    // Create a matrix multiplication plan at compile time
    const plan = comptime matmulPlan(&[_]usize{ 2, 3 }, &[_]usize{ 3, 2 });
    
    // Execute the plan
    var result = try plan.execute(allocator, &[_]Tensor{ tensor_a, tensor_b });
    defer result.deinit();
    
    // In a real implementation, this would compute:
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
    // = [58, 64]
    //   [139, 154]
    
    std.debug.print("\nComptime plan generation example:\n", .{});
    std.debug.print("--------------------------------\n", .{});
    std.debug.print("Input A: ", .{});
    tensor_a.print();
    std.debug.print("Input B: ", .{});
    tensor_b.print();
    std.debug.print("Result: ", .{});
    result.print();
    
    // Note: In a real implementation, the result would be computed correctly
    // This example just demonstrates the structure and concept
}

test "matrix multiplication plan" {
    // Compile-time check for valid dimensions
    _ = comptime matmulPlan(&[_]usize{ 2, 3 }, &[_]usize{ 3, 4 });
    
    // This would cause a compile error:
    // _ = comptime matmulPlan(&[_]usize{2, 3}, &[_]usize{4, 4});
}