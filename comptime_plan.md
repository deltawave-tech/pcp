# PCP Comptime Operation Plans

This document outlines our strategy for adapting PCP to use comptime-generated operation plans and gradient rules instead of a runtime graph of operations.

## Motivation

While a UOp graph approach (like tinygrad's) offers benefits for backend simplification and optimization opportunities, we can better leverage Zig's unique strengths - particularly comptime - to create a more elegant solution that:

1. Maintains or improves performance by using Zig's zero-cost abstractions
2. Simplifies runtime logic by moving optimization to compile time
3. Enhances memory safety through static analysis and explicit tensor lifetimes
4. Creates a PCP-specific design that aligns with our distributed tensor computation goals

## Core Concept: Comptime-Generated Plans

Instead of representing operations and gradients as a runtime graph, we'll use Zig's comptime features to generate optimized execution plans at compile time:

1. Define a small set of primitive operations that backends must implement
2. Generate forward plans for high-level operations based on tensor shapes and types
3. Generate gradient rules that define how to compute gradients for each primitive
4. Automatically derive backward passes from forward plans at compile time
5. Move validation and compatibility checks to compile time where possible
6. Minimize runtime overhead by executing pre-compiled plans

## Implementation Status

### ✅ Phase 1: Refactor ops.zig (Completed)

1. **Primitive Operations**
   - Defined a `Primitives` struct with core operations backends must implement
   - Included basic operations like add, multiply, matmul, relu
   - Provided default CPU implementations for each primitive

2. **Plan Structure**
   - Created a generic `PlanType` function for generating plan types
   - Each plan contains compile-time knowledge of shapes and types
   - Plans include init and run methods with minimal runtime checks

3. **Operation Plans**
   - Replaced each high-level operation with a comptime Plan generator
   - Moved validation to compile time where possible
   - Kept minimal runtime validation for dynamic aspects

4. **Backend Interface**
   - Defined a standard backend interface that includes primitive operations
   - Created a CpuBackend implementation that adopts the Primitives

### ✅ Phase 2: Implement Comptime Gradient Generator (Completed)

1. **Gradient Rules**
   - Defined gradient computation rules for each primitive operation
   - Built direct gradient methods into each operation plan
   - Implemented rules for key operations: add, subtract, multiply, divide, matmul, relu, softmax, transpose, embedding_lookup

2. **AutoDiffPlan Generator**
   - Created the `AutoDiffPlan` function that wraps forward plans with backward capabilities
   - Ensured AutoDiffPlan tracks the inputs/outputs needed for gradient computation
   - Built a consistent interface for all operations through WithGrad wrappers

3. **Integration with ops.zig**
   - Made backward passes reuse the same primitive operations
   - Established consistent interfaces between forward and backward passes
   - Support both runtime and comptime shape information
   - Created WithGrad wrappers for all operation plans

4. **Memory Management**
   - Implemented proper cleanup of intermediate tensors during gradient computation
   - Added reference tracking for tensors through forward and backward passes
   - Used errdefer for clean error handling in gradient computation

### 🔜 Phase 3: Update Tensor and Backend Support

1. **Tensor Updates**
   - Update tensor.zig to support comptime shape initialization
   - Add support for statically-known shapes at compile time
   - Optimize tensor operations for the Plan-based approach

2. **Metal Backend**
   - Update metal.zig to implement the primitive operations
   - Optimize Metal-specific implementations
   - Ensure plan compatibility across backends

3. **Backend Optimizations**
   - Add backend-specific optimization capabilities
   - Implement operation fusion where beneficial
   - Use comptime to generate optimized kernels

### 🔄 Phase 4: Examples and Testing (In Progress)

1. **Update Examples**
   - ✅ Created dedicated Plan-based test showing full forward/backward functionality
   - ✅ Updated GPT-2 model implementation to use Plan-based approach
   - ✅ Added real gradient-based training to GPT-2 training example
   - 🔄 Continue adapting remaining examples to the Plan-based approach
   - 🔄 Update shakespeare_training.zig to use Plan-based gradients

2. **Testing**
   - ✅ Added tests for core plan types and their gradients
   - ✅ Implemented proper test for Plan-based autodiff with various operations
   - ✅ Verified memory safety during forward and backward passes
   - 🔄 Expand testing coverage to all operations
   - 🔄 Add cross-backend compatibility testing

## Implementation Details

### Primitive Operations

The core primitive operations will include:

```zig
pub const Primitives = struct {
    fn add(comptime T: type, a: Tensor(T), b: Tensor(T), result: Tensor(T)) void { ... }
    fn multiply(comptime T: type, a: Tensor(T), b: Tensor(T), result: Tensor(T)) void { ... }
    fn matmul(comptime T: type, a: Tensor(T), b: Tensor(T), result: Tensor(T)) void { ... }
    fn relu(comptime T: type, a: Tensor(T), result: Tensor(T)) void { ... }
    // Additional primitives: subtract, divide, transpose, exp, log, reduce, etc.
};
```

### Plan Structure

The generic Plan type:

```zig
fn Plan(comptime Backend: type, comptime Input: type, comptime Output: type) type {
    return struct {
        const Self = @This();
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        pub fn run(self: Self, input: Input) !Tensor(Output) {
            // To be overridden by specific plans
            _ = self; _ = input;
            return OpError.NotImplemented;
        }
    };
}
```

### Operation Plan Example

A matrix multiplication plan:

```zig
pub fn MatmulPlan(comptime Backend: type, comptime T: type, comptime M: usize, comptime N: usize, comptime P: usize) type {
    comptime {
        if (!Backend.hasPrimitive("matmul")) @compileError("Backend must implement matmul primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }

    return struct {
        const Base = Plan(Backend, struct { a: Tensor(T), b: Tensor(T) }, T);
        base: Base,

        pub fn init(allocator: Allocator) @This() {
            return .{ .base = Base.init(allocator) };
        }

        pub fn run(self: @This(), input: struct { a: Tensor(T), b: Tensor(T) }) !Tensor(T) {
            // Runtime validation
            if (input.a.shape.rank() != 2 or input.b.shape.rank() != 2 or
                input.a.shape.dims[0] != M or input.a.shape.dims[1] != N or
                input.b.shape.dims[0] != N or input.b.shape.dims[1] != P) {
                return OpError.ShapeMismatch;
            }
            if (input.a.dtype != T or input.b.dtype != T) {
                return OpError.UnsupportedDataType;
            }

            // Allocate result
            const result_dims = [_]usize{ M, P };
            var result = try Tensor(T).zeros(self.base.allocator, &result_dims, T, input.a.backend);
            errdefer result.deinit();

            // Execute primitive
            Backend.matmul(T, input.a, input.b, result);
            return result;
        }
    };
}
```

### Backend Implementation

A simple CPU backend:

```zig
pub const CpuBackend = struct {
    pub fn hasPrimitive(comptime name: []const u8) bool {
        return std.mem.eql(u8, name, "add") or
               std.mem.eql(u8, name, "multiply") or
               std.mem.eql(u8, name, "matmul") or
               std.mem.eql(u8, name, "relu");
    }

    pub usingnamespace Primitives; // Include default implementations
};
```

## Memory Management

Memory management will be simplified with the Plan-based approach:

1. Plans are responsible for allocating result tensors
2. Error handling is streamlined with `errdefer` for cleanup
3. Memory is allocated only when necessary and properly cleaned up

## Migration Strategy

1. Begin by refactoring ops.zig to implement the core Plan structure
2. Update tensor.zig to support comptime shape information
3. Implement primitive operations in CPU and Metal backends
4. Update autodiff.zig to use the new Plan-based approach
5. Convert example code to use Plans
6. Add comprehensive tests for all operations

## Architectural Improvements

Based on our experience with the implementation so far, we're planning the following architectural improvements:

### 1. Centralizing Plan Definitions ✅

We've moved gradient computation directly into operation plans in `ops.zig` to create a more consistent structure:

```zig
// In ops.zig
pub fn AddPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // ... comptime validation ...
    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.add;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        // ... implementation ...
        
        // Gradient computation built into the plan
        pub fn gradient(self: @This(), grad_out: Tensor, _: InputType) !GradType {
            // Both inputs receive the same gradient
            var da = try grad_out.clone();
            errdefer da.deinit();
            
            var db = try grad_out.clone();
            errdefer db.deinit();
            
            return .{ .da = da, .db = db };
        }
    };
}
```

This approach ensures:
- All plan-related functionality is in one place
- GradType and op_type are always defined consistently
- Gradient computation is directly tied to the operation that produces it
- AutoDiffPlan directly calls the plan's gradient method
- WithGrad plan wrappers simply pass through to the base plan's gradient method

The code is more maintainable because adding a new operation only requires implementing one plan type with its gradient method, not modifying the autodiff engine.

### 2. Comptime Fusion for Complex Gradients

For operations with complex gradient rules (like matmul and softmax), we'll implement comptime fusion to optimize the computation:

```zig
pub fn matmul_gradient(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
    // Current implementation: Multiple operations
    const b_t = try ops.transpose(allocator, b);
    errdefer b_t.deinit();
    const da = try ops.matmul(allocator, grad_out, b_t);
    b_t.deinit();

    const a_t = try ops.transpose(allocator, a);
    errdefer a_t.deinit();
    const db = try ops.matmul(allocator, a_t, grad_out);
    a_t.deinit();

    return .{ .da = da, .db = db };
}

// Fused implementation
pub fn matmul_gradient_fused(comptime Backend: type, comptime T: type) type {
    return struct {
        pub fn compute(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
            // One-shot optimized implementation
            var result = try compute_fused_matmul_grad(allocator, grad_out, a, b);
            return result;
        }
    };
}
```

Benefits:
- Reduced temporary tensor allocations
- Fewer individual operations
- Potential for backend-specific optimizations

### 3. Enhanced Shape Validation

We'll improve shape validation by combining comptime checks with robust runtime validation:

```zig
pub fn MatmulPlan(comptime Backend: type, comptime T: type, comptime M: ?usize, comptime N: ?usize, comptime P: ?usize) type {
    // ... implementation ...
    
    pub fn run(self: Self, input: InputType) !Tensor {
        // Comptime shape validation when dimensions are known
        if (M != null and N != null and P != null) {
            // Static validation
            comptime {
                if (/* validation logic */) {
                    @compileError("Invalid static dimensions");
                }
            }
        } else {
            // Runtime validation for dynamic dimensions
            if (/* validation logic */) {
                return error.ShapeMismatch;
            }
        }
        
        // ... implementation ...
    }
}
```

This approach provides:
- Compile-time safety when shapes are known
- Runtime safety for dynamic shapes
- Consistent error handling

## Future Optimizations

1. **Kernel Fusion**: Backends can implement sophisticated plan optimization
2. **Type Specialization**: Generated code can be optimized for specific types
3. **SIMD Utilization**: Backends can use SIMD instructions for primitives
4. **Distribution**: Plans can include sharding strategies for distributed execution

## New AutoDiffPlan Implementation

One of the key new concepts introduced in our implementation is the AutoDiffPlan wrapper:

```zig
/// AutoDiffPlan wraps a forward plan and provides backward functionality
pub fn AutoDiffPlan(comptime ForwardPlanType: type) type {
    // Comptime validation
    comptime {
        if (!@hasDecl(ForwardPlanType, "GradType")) {
            @compileError("Forward plan must define GradType for autodiff");
        }
        if (!@hasDecl(ForwardPlanType, "op_type")) {
            @compileError("Forward plan must define op_type for autodiff");
        }
    }
    
    return struct {
        const Self = @This();
        
        // The forward plan
        forward_plan: ForwardPlanType,
        
        // Keep track of input and output for backward pass
        last_input: ?ForwardPlanType.InputType = null,
        last_output: ?Tensor = null,
        
        allocator: Allocator,
        
        pub fn init(allocator: Allocator) Self {
            return .{
                .forward_plan = ForwardPlanType.init(allocator),
                .allocator = allocator,
            };
        }
        
        pub fn deinit(self: *Self) void {
            if (self.last_output != null) {
                self.last_output.?.deinit();
                self.last_output = null;
            }
        }
        
        /// Forward pass: run the forward plan and cache inputs/outputs for backward
        pub fn forward(self: *Self, input: ForwardPlanType.InputType) !Tensor {
            // Run the forward plan
            const output = try self.forward_plan.run(input);
            
            // Clean up previous output if it exists
            if (self.last_output != null) {
                self.last_output.?.deinit();
            }
            
            // Store input and output
            self.last_input = input;
            self.last_output = try output.clone();
            
            return output;
        }
        
        /// Backward pass: compute gradients with respect to inputs
        pub fn backward(self: *Self, grad_out: Tensor) !ForwardPlanType.GradType {
            // Ensure we have inputs and outputs from a previous forward pass
            if (self.last_input == null or self.last_output == null) {
                return error.NoPreviousForward;
            }
            
            // Call the plan's gradient method directly
            return try self.forward_plan.gradient(grad_out, self.last_input.?);
        }
    };
}
```

This wrapper provides several key advantages:

1. **Automatic input/output tracking** - Stores the inputs and outputs needed for gradient computation
2. **Memory safety** - Properly handles tensor lifetimes and cleanup
3. **Simplified interface** - Provides a consistent backward() method for all operation types
4. **Compile-time validation** - Ensures all plans have the necessary gradient information
5. **Clear separation of concerns** - Plans provide gradient rules, wrapper handles tracking

## Conclusion

The comptime Plan-based approach offers significant advantages over a runtime UOp graph:

1. Better performance due to compile-time optimization
2. Enhanced safety through static validation
3. Simplified runtime logic with minimal overhead
4. Elegant architecture leveraging Zig's unique strengths
5. Maintainable codebase with clear separation of concerns
6. Consistent interface for gradient computation
7. Proper memory management with explicit tensor lifetimes