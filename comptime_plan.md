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

1. Define a small set of primitive operations as comptime-callable functions
2. Compose high-level operations using these primitives at compile time
3. Generate optimized, backend-specific execution plans
4. Execute these plans at runtime with minimal overhead

## Implementation Plan

### Phase 1: Core Architecture Changes

1. **Primitive Operations**
   - Define a minimal set of primitive operations in `src/primitives.zig`
   - Each primitive will have a comptime interface for plan generation
   - Categories: movement, elementwise, and reduction operations

2. **Plan Structure**
   - Create a `Plan` type in `src/plan.zig` to represent execution plans
   - Plans will contain backend-specific instructions for tensor operations
   - Include mechanisms for static fusion of operations

3. **Backend Interface Updates**
   - Update `src/backends/backend.zig` to define the interface for backend implementations
   - Each backend must implement the primitive operations
   - Add capabilities for backends to provide optimization hints during plan generation

### Phase 2: High-Level Operation Reimplementation

1. **Plan Generators**
   - Create comptime functions for each high-level operation (e.g., matmul, softmax)
   - These functions will generate optimized plans based on tensor shapes and backend

2. **Tensor Interface Updates**
   - Update `src/tensor.zig` to work with compiled plans
   - Add methods for executing plans on tensors

3. **Update Examples**
   - Adapt existing examples to use the new plan-based approach

### Phase 3: Autodiff Integration

1. **Gradient Plans**
   - Extend the Plan type to include gradient computation
   - Generate backward plans at compile time using the chain rule

2. **Autodiff Module Updates**
   - Rewrite `src/autodiff.zig` to use comptime plan generation for gradients
   - Ensure proper memory management throughout the backward pass

### Phase 4: Optimization and Distribution

1. **Static Optimizations**
   - Implement comptime optimization passes for plans
   - Add fusion logic for combining operations where beneficial

2. **Distribution Strategy**
   - Add primitives for distributed tensor operations
   - Implement comptime sharding strategies for distributed execution

## Implementation Details

### Primitive Operations

We'll define the following primitive operations:

```zig
// Movement operations
fn reshape(comptime Backend: type, tensor: Tensor, comptime new_shape: []const usize) Plan(Backend) {...}
fn transpose(comptime Backend: type, tensor: Tensor, comptime axes: []const usize) Plan(Backend) {...}
fn slice(comptime Backend: type, tensor: Tensor, comptime start: []const usize, comptime end: []const usize) Plan(Backend) {...}

// Elementwise operations
fn elementwiseUnary(comptime Backend: type, tensor: Tensor, comptime op: ElementwiseOp) Plan(Backend) {...}
fn elementwiseBinary(comptime Backend: type, a: Tensor, b: Tensor, comptime op: ElementwiseOp) Plan(Backend) {...}

// Reduction operations
fn reduce(comptime Backend: type, tensor: Tensor, comptime op: ReduceOp, comptime axes: []const usize) Plan(Backend) {...}
```

### Plan Structure

A Plan will be a struct containing:
- Input tensor specifications
- Output tensor specification
- A sequence of operations with backend-specific optimizations
- Memory management instructions

```zig
pub fn Plan(comptime Backend: type) type {
    return struct {
        const Self = @This();
        
        inputs: []const TensorSpec,
        output: TensorSpec,
        operations: []const Operation(Backend),
        
        pub fn run(self: Self, inputs: []const Tensor) Tensor {
            // Execute the plan with the given input tensors
            // ...
        }
    };
}
```

### High-Level Operation Example: Matrix Multiplication

```zig
pub fn matmulPlan(comptime Backend: type, comptime a_shape: []const usize, comptime b_shape: []const usize) Plan(Backend) {
    if (a_shape[1] != b_shape[0]) {
        @compileError("Matrix dimensions don't match for multiplication");
    }
    
    // Create a plan for: C[i,k] = sum_j A[i,j] * B[j,k]
    const reshapeA = reshape(Backend, "A", [a_shape[0], a_shape[1], 1]);
    const reshapeB = reshape(Backend, "B", [1, b_shape[0], b_shape[1]]);
    const broadcast = [_]usize{a_shape[0], a_shape[1], b_shape[1]};
    const mul = elementwiseBinary(Backend, reshapeA, reshapeB, .multiply, broadcast);
    const result = reduce(Backend, mul, .sum, [_]usize{1});
    
    // Allow the backend to optimize this plan
    return Backend.optimize(result);
}
```

### Backend Implementation Example: CPU

```zig
pub const CPUBackend = struct {
    // Primitive implementations
    pub fn reshape(tensor: Tensor, new_shape: []const usize) PlanOp {
        return PlanOp{
            .op = .reshape,
            .inputs = [_]TensorRef{tensor.ref()},
            .shapes = [_][]const usize{new_shape},
        };
    }
    
    // Other primitives...
    
    // Plan optimization
    pub fn optimize(plan: Plan(CPUBackend)) Plan(CPUBackend) {
        // Apply CPU-specific optimizations
        // e.g., loop fusion, SIMD intrinsics, etc.
        return optimized_plan;
    }
};
```

## Memory Management

One of the key benefits of this approach is improved memory management:

1. Tensor lifetimes are clearer since operations are composed at compile time
2. Memory allocations and deallocations can be planned in advance
3. The runtime execution has minimal overhead without complex graph traversal

We'll continue to use reference counting, but with more static guarantees:

```zig
// In plan.run()
fn run(self: Self, inputs: []const Tensor) Tensor {
    var intermediates = std.ArrayList(Tensor).init(allocator);
    defer {
        // Clean up all intermediate tensors
        for (intermediates.items) |t| {
            t.deinit();
        }
        intermediates.deinit();
    }
    
    // Execute operations...
    
    // Return the final result tensor
    return result;
}
```

## Migration Strategy

To transition from our current implementation to the comptime plan approach:

1. First, implement the core primitives and plan structure
2. Create plan generators for basic operations (add, multiply, etc.)
3. Implement the CPU backend for these primitives
4. Update one example (e.g., autodiff_test.zig) to use the new approach
5. Gradually migrate other operations and examples
6. Update the Metal backend to use the new architecture
7. Finally, integrate the updated autodiff system

## Timeline

- **Week 1**: Core architecture - primitives, plan structure, CPU backend
- **Week 2**: Basic operations and first example migration
- **Week 3**: Autodiff integration and more operations
- **Week 4**: Metal backend and optimization passes
- **Week 5**: Distributed primitives and testing

## Conclusion

This comptime-based approach offers a more elegant and Zig-specific solution than a direct UOp graph port from tinygrad. It leverages Zig's strengths while maintaining the benefits of operation decomposition and backend simplification.

The implementation will require significant changes to our architecture, but the resulting system will be more performant, safer, and better aligned with our vision for planetary-scale tensor computation.