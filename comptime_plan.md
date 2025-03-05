# PCP Comptime Operation Plans

This document outlines our strategy for adapting PCP to use comptime-generated operation plans instead of a runtime graph of operations.

## Motivation

While a UOp graph approach (like tinygrad's) offers benefits for backend simplification and optimization opportunities, we can better leverage Zig's unique strengths - particularly comptime - to create a more elegant solution that:

1. Maintains or improves performance by using Zig's zero-cost abstractions
2. Simplifies runtime logic by moving optimization to compile time
3. Enhances memory safety through static analysis and explicit tensor lifetimes
4. Creates a PCP-specific design that aligns with our distributed tensor computation goals

## Core Concept: Comptime-Generated Operation Plans

Instead of representing operations as a runtime graph of UOps, we'll use Zig's comptime features to generate optimized execution plans at compile time:

1. Define a small set of primitive operations that backends must implement
2. Generate plans for high-level operations at compile time based on tensor shapes and types
3. Move validation and compatibility checks to compile time where possible
4. Minimize runtime overhead by executing pre-compiled plans

## Implementation Plan

### Phase 1: Refactor ops.zig

1. **Primitive Operations**
   - Define a `Primitives` struct with core operations backends must implement
   - Include basic operations like add, multiply, matmul, relu
   - Provide default CPU implementations for each primitive

2. **Plan Structure**
   - Create a generic `Plan` type parameterized by Backend, Input, and Output
   - Each plan contains compile-time knowledge of shapes and types
   - Plans include init and run methods with minimal runtime checks

3. **Operation Plans**
   - Replace each high-level operation with a comptime Plan generator
   - Move validation to compile time where possible
   - Keep minimal runtime validation for dynamic aspects

4. **Backend Interface**
   - Define a standard backend interface that includes primitive operations
   - Create a CpuBackend implementation that adopts the Primitives

### Phase 2: Update Tensor Implementation

1. **Tensor Updates**
   - Update tensor.zig to support comptime shape initialization
   - Ensure tensor operations work with the new Plan-based approach
   - Add support for statically-known shapes at compile time

2. **Memory Management**
   - Optimize memory management for plan-based operations
   - Ensure proper cleanup of intermediate tensors
   - Maintain reference counting for tensors across plan execution

### Phase 3: Extend Backend Support

1. **Metal Backend**
   - Update metal.zig to implement the primitive operations
   - Optimize Metal-specific implementations
   - Ensure plan compatibility across backends

2. **Backend Optimizations**
   - Add backend-specific optimization capabilities
   - Implement operation fusion where beneficial
   - Use comptime to generate optimized kernels

### Phase 4: Autodiff Integration

1. **Gradient Plans**
   - Generate backward plans at compile time based on forward plans
   - Ensure proper gradient flow through the computation
   - Maintain memory safety during backward pass

2. **Autodiff Module Updates**
   - Update autodiff.zig to use the plan-based approach
   - Ensure compatibility with existing models
   - Optimize memory usage during backward pass

### Phase 5: Examples and Tests

1. **Update Examples**
   - Adapt example code to use the new plan-based approach
   - Demonstrate benefits in terms of performance and safety
   - Provide clear usage patterns for new developers

2. **Testing**
   - Add comprehensive tests for all plan types
   - Ensure compatibility across backends
   - Validate memory safety and correct operation

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

## Future Optimizations

1. **Kernel Fusion**: Backends can implement sophisticated plan optimization
2. **Type Specialization**: Generated code can be optimized for specific types
3. **SIMD Utilization**: Backends can use SIMD instructions for primitives
4. **Distribution**: Plans can include sharding strategies for distributed execution

## Conclusion

The comptime Plan-based approach offers significant advantages over a runtime UOp graph:

1. Better performance due to compile-time optimization
2. Enhanced safety through static validation
3. Simplified runtime logic with minimal overhead
4. Elegant architecture leveraging Zig's unique strengths
5. Maintainable codebase with clear separation of concerns