# Planetary Compute Protocol (PCP) Documentation

A comprehensive guide to the PCP tensor computation framework with automatic differentiation.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
   - [Tensor Module](#tensor-module)
   - [Operations Module](#operations-module)
   - [Autodiff Engine](#autodiff-engine)
   - [Model Implementation](#model-implementation)
4. [Best Practices](#best-practices)
5. [Implementation Details](#implementation-details)
   - [Memory Management](#memory-management)
   - [Error Handling](#error-handling)
   - [Embedding Implementation](#embedding-implementation)
   - [Neural Network Training](#neural-network-training)
   - [Compile-Time Planning](#compile-time-planning)
6. [Zig Version](#zig-version)
7. [Examples](#examples)

## Introduction

Planetary Compute Protocol (PCP) is a distributed tensor computation framework written in Zig. It provides a foundation for building and training neural networks with proper memory management and automatic differentiation. PCP is designed for performance and memory safety, making it suitable for both research and production environments.

Key features:
- Pure Zig implementation with no external dependencies
- Reference-counted tensor management
- Automatic differentiation with computational graphs
- Compile-time operation planning and validation
- Support for CPU computations (with Metal/GPU support in development)
- Memory-safe tensor operations with proper cleanup

## Architecture Overview

PCP follows a layered architecture design:

1. **Core Tensor Layer**: Low-level tensor operations, memory management
2. **Operations Layer**: Mathematical functions on tensors with compile-time plans
3. **Autodiff Layer**: Gradient computation through a computational graph or via compile-time plans
4. **Model Layer**: Neural network model implementations (like GPT-2)
5. **Training Layer**: Example training loops and optimization algorithms

This layered approach allows for clean separation of concerns and modular development.

## Core Components

### Tensor Module

The tensor module (`tensor.zig`) implements the fundamental data structure for all computations - the Tensor.

Key components:

- **DType**: Supported data types
  ```zig
  pub const DType = enum {
      f16,
      f32,
      f64,
      i32,
      i64,
      bool,
      
      pub fn sizeInBytes(self: DType) usize {
          return switch (self) {
              .f16 => 2,
              .f32 => 4,
              .f64 => 8,
              .i32 => 4,
              .i64 => 8,
              .bool => 1,
          };
      }
  };
  ```

- **Shape**: Tensor dimensions
  ```zig
  pub const Shape = struct {
      dims: []const usize,
      allocator: Allocator,
      
      pub fn init(allocator: Allocator, dimensions: []const usize) !Shape {
          const dims_copy = try allocator.dupe(usize, dimensions);
          return Shape{
              .dims = dims_copy,
              .allocator = allocator,
          };
      }
      
      pub fn deinit(self: *Shape) void {
          self.allocator.free(self.dims);
      }
      
      // Other methods...
  };
  ```

- **Tensor**: Core tensor struct with reference counting
  ```zig
  pub const Tensor = struct {
      shape: Shape,
      dtype: DType,
      buffer: Buffer,
      requires_grad: bool = false,
      backend: BackendType = .cpu,
      ref_count: *usize, // Reference count
      allocator: Allocator, // Store allocator for reference count management
      
      // Methods for creation, manipulation, and memory management...
  };
  ```

### Operations Module

The operations module (`ops.zig`) provides mathematical operations on tensors with both a traditional interface and a modern compile-time plan-based approach.

#### Primitive Operations
Basic operations implemented by all backends:

```zig
pub const Primitives = struct {
    /// Add two tensors element-wise
    pub fn add(comptime T: type, a: Tensor, b: Tensor, result: *Tensor) void {
        // Implementation...
    }
    
    /// Matrix multiplication for 2D tensors
    pub fn matmul(comptime T: type, a: Tensor, b: Tensor, result: *Tensor) void {
        // Implementation...
    }
    
    // Other primitives: subtract, multiply, divide, relu, softmax, transpose, embedding_lookup
};
```

#### Operation Plans
Modern operation interface using compile-time validation:

```zig
/// Generic type for comptime-generated operation plans
pub fn PlanType(comptime Backend: type, comptime InputType: type, comptime OutputType: type) type {
    return struct {
        pub allocator: Allocator,
        
        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }
        
        pub fn run(self: Self, input: InputType) !OutputType {
            // To be overridden by specific plans
        }
        
        // Default gradient type for operations without gradients
        pub const GradType = void;
    };
}

/// Example plan implementation for matrix multiplication
pub fn MatmulPlan(comptime Backend: type, comptime T: type, 
                 comptime M: ?usize, comptime N: ?usize, comptime P: ?usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("matmul")) @compileError("Backend must implement matmul primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }
    
    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.matmul;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,
        
        // Implementation...
    };
}
```

#### Legacy Interface
Backward compatibility with direct operation functions:

```zig
/// Matrix multiplication (legacy interface)
pub fn matmul(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
    // Runtime validation and implementation...
}
```

### Autodiff Engine

The autodiff module (`autodiff.zig`) provides automatic differentiation capabilities using the Compile-Time Plan Approach:

- **Plan Types with Gradient Methods**: Each operation plan includes its own gradient computation method:
  ```zig
  pub fn AddPlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
      // Comptime validation...
      return struct {
          pub const InputType = struct { a: Tensor, b: Tensor };
          pub const op_type = OpType.add;
          pub const GradType = struct { da: Tensor, db: Tensor };
          
          const Base = PlanType(Backend, InputType, Tensor);
          base: Base,
          
          // Implementation...
          
          /// Compute gradients for add operation: da = grad_out, db = grad_out
          pub fn gradient(self: @This(), grad_out: Tensor, _: InputType) !GradType {
              _ = self; // Not used
              
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

- **AutoDiffPlan**: Wrapper for operation plans that adds gradient functionality
  ```zig
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
          // The forward plan
          forward_plan: ForwardPlanType,
          
          // Keep track of input and output for backward pass
          last_input: ?ForwardPlanType.InputType = null,
          last_output: ?Tensor = null,
          
          allocator: Allocator,
          
          // Methods for forward and backward passes...
          
          /// Comptime function to compute gradients
          fn computeGradient(self: *Self, grad_out: Tensor) !ForwardPlanType.GradType {
              const input = self.last_input.?;
              
              // Use the plan's gradient method directly
              return try self.forward_plan.gradient(grad_out, input);
          }
      };
  }
  ```

The key improvements in this design:
1. Gradient computation is tied directly to the operation that produces it
2. Each plan type includes its own gradient method
3. The AutoDiffPlan directly calls the plan's gradient method
4. Memory management is handled consistently with errdefer for cleanup
5. The code is more maintainable as adding a new operation only requires implementing one plan type

The project has completely migrated from using a Node-based autodiff approach to this modern Plan-based approach, which offers:
1. Better compile-time guarantees and type safety
2. Simplified memory management (no graph to clean up)
3. Closer integration between operations and their gradient implementations
4. More natural fit with Zig's compile-time metaprogramming capabilities
5. Better performance through operation specialization

### Model Implementation

The project includes implementations of neural network models, with GPT-2 as a primary example.

Key components:

- **GPT2Config**: Model configuration
  ```zig
  pub const GPT2Config = struct {
      vocab_size: usize = 50257,
      n_positions: usize = 1024,
      n_embd: usize = 768,
      n_layer: usize = 12,
      n_head: usize = 12,
      layer_norm_epsilon: f32 = 1e-5,
      initializer_range: f32 = 0.02,
  };
  ```

- **GPT2**: Main model implementation with Plan-based forward and gradient methods:
  ```zig
  pub const GPT2 = struct {
      wte: Tensor, // Token embeddings
      wpe: Tensor, // Position embeddings
      blocks: []Block, // Transformer blocks
      ln_f: LayerNorm, // Final layer norm
      
      // AutoDiff plans for different operations
      matmul_plan: ?*autodiff.AutoDiffPlan(autodiff.MatmulPlanWithGrad(...)) = null,
      embed_lookup_plan: ?*autodiff.AutoDiffPlan(autodiff.EmbeddingLookupPlanWithGrad(...)) = null,
      add_plan: ?*autodiff.AutoDiffPlan(autodiff.AddPlanWithGrad(...)) = null,
      
      // Regular forward pass
      pub fn forward(self: *GPT2, input_ids: Tensor) !Tensor {
          // Implementation using regular tensor operations
      }
      
      // Forward pass with gradient tracking
      pub fn forwardWithGrad(self: *GPT2, input_ids: Tensor) !Tensor {
          // Implementation using Plan operations instead of regular ones
      }
  };
  ```

- **Attention and MLP Components**: Also implemented with the Plan-based approach:
  ```zig
  pub const Attention = struct {
      // Parameters
      c_attn_weight: Tensor,
      c_attn_bias: Tensor,
      // ...
      
      // Plans for autodiff
      matmul_plan: ?*autodiff.AutoDiffPlan(...) = null,
      // ...
      
      // Forward methods
      pub fn forward(self: *Attention, x: Tensor) !Tensor {
          // Implementation with regular tensor operations
      }
      
      pub fn forwardWithGrad(self: *Attention, x: Tensor) !Tensor {
          // Implementation with Plan-based operations for gradient tracking
      }
  };
  ```

## Best Practices

Throughout the PCP codebase, several best practices are employed:

1. **Explicit Memory Management**: All resources are tracked and freed appropriately
   ```zig
   pub fn deinit(self: *Node) void {
       // Clean up inputs array
       self.inputs.deinit();
       
       // Clean up gradient if it exists
       if (self.grad) |*g| {
           g.deinit();
       }
       
       // Release the tensor
       self.tensor.deinit();
       
       // Free the Node itself
       self.allocator.destroy(self);
   }
   ```

2. **Reference Counting**: Safe sharing of tensors between operations
   ```zig
   pub fn retain(self: Tensor) Tensor {
       self.ref_count.* += 1;
       return self;
   }
   
   pub fn release(self: *const Tensor) void {
       // Safety check: don't decrement if already zero
       if (self.ref_count.* == 0) {
           std.debug.print("Warning: Trying to release tensor with zero reference count\n", .{});
           return;
       }
       
       self.ref_count.* -= 1;
       
       if (self.ref_count.* == 0) {
           // Clean up all resources
           // ...
       }
   }
   ```

3. **Comprehensive Error Handling**: All errors are properly propagated and resources cleaned up
   ```zig
   if (buffer_size == std.math.maxInt(usize)) {
       // Clean up shape to avoid memory leak
       var shape_copy = shape;
       shape_copy.deinit();
       return error.TensorTooLarge;
   }
   ```

4. **Compile-time Validation**: Use Zig's comptime to verify interfaces at compile time
   ```zig
   comptime {
       if (!Backend.hasPrimitive("matmul")) @compileError("Backend must implement matmul primitive");
       if (T != f32) @compileError("Only f32 supported for now");
   }
   ```

5. **Factory Methods**: Static creation methods for objects
   ```zig
   pub fn zeros(allocator: Allocator, dims: []const usize, dtype: DType, backend: BackendType) !Tensor {
       const tensor = try init(allocator, dims, dtype, backend);
       // Fill with zeros
       @memset(tensor.buffer.data, 0);
       return tensor;
   }
   ```

## Implementation Details

### Memory Management

PCP uses reference counting for memory management, a critical aspect for tensor operations where the same tensor may be used in multiple places.

The core of the reference counting system:

```zig
pub fn retain(self: Tensor) Tensor {
    self.ref_count.* += 1;
    return self;
}

pub fn release(self: *const Tensor) void {
    if (self.ref_count.* == 0) {
        std.debug.print("Warning: Trying to release tensor with zero reference count\n", .{});
        return;
    }
    
    self.ref_count.* -= 1;
    
    if (self.ref_count.* == 0) {
        // Clean up all resources when reference count reaches zero
        var shape_copy = self.shape;
        var buffer_copy = self.buffer;
        const allocator = self.allocator;
        
        // Free shape and buffer
        shape_copy.deinit();
        buffer_copy.deinit();
        
        // Free the reference count itself
        allocator.destroy(self.ref_count);
    }
}
```

This system allows safe sharing of tensor data across operations while ensuring memory is freed when no longer needed.

### Error Handling

PCP uses Zig's error handling mechanism to handle errors gracefully with proper resource cleanup.

Example error handling pattern:

```zig
pub fn getScalar(self: Tensor, indices: []const usize) !f32 {
    if (indices.len != self.shape.rank()) {
        return error.InvalidIndices;
    }
    
    // Calculate linear index
    var linear_idx: usize = 0;
    var stride: usize = 1;
    
    var i: usize = self.shape.rank();
    while (i > 0) {
        i -= 1;
        if (indices[i] >= self.shape.dims[i]) {
            return error.IndexOutOfBounds;
        }
        linear_idx += indices[i] * stride;
        stride *= self.shape.dims[i];
    }
    
    switch (self.dtype) {
        .f32 => {
            const f32_buf = ptrCastHelper([*]f32, self.buffer.data.ptr);
            return f32_buf[linear_idx];
        },
        else => {
            return error.UnsupportedDataTypeForGetScalar;
        }
    }
}
```

This approach ensures:
- Errors are explicitly returned
- Client code must handle or propagate errors
- No resources are leaked on error paths

### Embedding Implementation

Embedding layers require special handling for gradient flow and memory management. PCP implements this with careful reference counting and specialized gradient accumulation:

```zig
/// Gradient for embedding lookup: accumulate gradients at token positions
pub fn embedding_lookup(allocator: Allocator, grad_out: Tensor, 
                       params: Tensor, indices: Tensor) !Tensor {
    // Gradient for params is sparse: only positions referenced by indices get gradients
    var d_params = try Tensor.zeros(allocator, params.shape.dims, params.dtype, params.backend);
    errdefer d_params.deinit();
    
    // Extract dimensions
    const vocab_size = params.shape.dims[0];
    const embed_dim = params.shape.dims[1];
    
    // For each token in the batch, accumulate gradients back to the embedding table
    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            // Get token ID, clamping to vocab size
            const token_id = /* ... */;
            
            // Accumulate gradients for this embedding
            for (0..embed_dim) |d| {
                const grad_pos = (b * seq_len + s) * embed_dim + d;
                const param_pos = token_id * embed_dim + d;
                d_params_buf[param_pos] += grad_buf[grad_pos];
            }
        }
    }
    
    return d_params;
}
```

### Neural Network Training

The PCP framework provides examples of neural network training loops:

```zig
// Training loop
for (0..num_epochs) |epoch| {
    std.debug.print("Epoch {}/{}:\n", .{epoch + 1, num_epochs});
    
    var batch_offset: usize = 0;
    
    while (batch_offset < dataset.inputs.shape.dims[0]) {
        // Get batch
        var batch = dataset.getBatch(batch_size, batch_offset) catch |err| {
            if (err == error.BatchSizeZero) break;
            return err;
        };
        defer batch.deinit();
        
        // Forward pass
        var logits = try model.forward(input_copy);
        
        // Compute the loss
        var loss = try computeCrossEntropy(allocator, logits, batch.targets);
        defer loss.deinit();
        
        // Backward pass
        try autodiff.backward(allocator, loss);
        
        // Update parameters
        if (model.wte_node) |wte_node| {
            if (wte_node.grad != null) {
                try optimizer.step(&model.wte, wte_node.grad.?);
            }
        }
        
        // Move to next batch
        batch_offset += batch_size;
    }
}
```

### Compile-Time Planning

One of the key innovations in PCP is the use of Zig's comptime features to generate operation plans that validate and optimize operations at compile time.

#### Plan Generation

```zig
// Creating a plan with static dimensions
const MatmulOp = MatmulPlan(CpuBackend, f32, 128, 256, 64);
var matmul_op = MatmulOp.init(allocator);

// Wrapping with autodiff
const MatmulWithGrad = AutoDiffPlan(MatmulOp);
var autodiff_op = MatmulWithGrad.init(allocator);
```

#### Direct Gradient Methods in Plans

The improved design directly includes gradient computation in each plan:

```zig
pub fn MatmulPlan(comptime Backend: type, comptime T: type, 
               comptime M: ?usize, comptime N: ?usize, comptime P: ?usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("matmul")) @compileError("Backend must implement matmul primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }
    
    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.matmul;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        // ... implementation of run() method ...
        
        /// Compute gradients for matmul: da = grad_out @ b^T, db = a^T @ grad_out
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const a = input.a;
            const b = input.b;
            
            // Compute b transpose
            const b_transpose = try ops.transpose(self.allocator, b);
            errdefer b_transpose.deinit();
            
            // Gradient for a is grad_out @ b^T
            var da = try ops.matmul(self.allocator, grad_out, b_transpose);
            errdefer da.deinit();
            
            // Compute a transpose
            const a_transpose = try ops.transpose(self.allocator, a);
            errdefer a_transpose.deinit();
            
            // Gradient for b is a^T @ grad_out
            var db = try ops.matmul(self.allocator, a_transpose, grad_out);
            errdefer db.deinit();
            
            // Clean up temporaries
            b_transpose.deinit();
            a_transpose.deinit();
            
            return .{ .da = da, .db = db };
        }
    };
}
```

#### Simplified Autodiff Backward

The AutoDiffPlan's backward method now directly calls the plan's gradient method:

```zig
/// Backward pass: compute gradients with respect to inputs
pub fn backward(self: *Self, grad_out: Tensor) !ForwardPlanType.GradType {
    // Ensure we have inputs and outputs from a previous forward pass
    if (self.last_input == null or self.last_output == null) {
        return error.NoPreviousForward;
    }
    
    // Call the plan's gradient method directly
    return try self.forward_plan.gradient(grad_out, self.last_input.?);
}
```

This approach ensures:
1. Each operation plan fully encapsulates both its forward and backward behavior
2. Operations are validated at compile time
3. The gradient implementation is directly tied to the operation itself
4. Type safety is maintained throughout the system
5. New operations can be added consistently by implementing a single plan type

## Zig Version

This project requires Zig 0.14.x or later.

- The codebase uses Zig 0.14's `std.Random` module for random number generation
- Pointer casting uses a helper function for consistent syntax and proper alignment:
  ```zig
  fn ptrCastHelper(comptime T: type, ptr: anytype) T {
      return @ptrCast(@alignCast(ptr));
  }
  ```

## Examples

### Recommended Plan-Based Approach

Here's a complete example demonstrating the recommended Plan-based approach for a simple neural network:

```zig
const std = @import("std");
const pcp = @import("pcp");
const tensor = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // 1. Create input and weight tensors
    var dims = [_]usize{ 2, 2 };
    
    var x = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer x.deinit();
    try x.setScalar(&[_]usize{0, 0}, 1.0);
    try x.setScalar(&[_]usize{0, 1}, 2.0);
    try x.setScalar(&[_]usize{1, 0}, 3.0);
    try x.setScalar(&[_]usize{1, 1}, 4.0);
    
    var w1 = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer w1.deinit();
    try w1.setScalar(&[_]usize{0, 0}, 0.1);
    try w1.setScalar(&[_]usize{0, 1}, 0.2);
    try w1.setScalar(&[_]usize{1, 0}, 0.3);
    try w1.setScalar(&[_]usize{1, 1}, 0.4);
    
    var w2 = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer w2.deinit();
    try w2.setScalar(&[_]usize{0, 0}, 0.5);
    try w2.setScalar(&[_]usize{0, 1}, 0.6);
    try w2.setScalar(&[_]usize{1, 0}, 0.7);
    try w2.setScalar(&[_]usize{1, 1}, 0.8);
    
    // 2. Set up plans for each operation in the network
    const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend, f32, null, null, null);
    var matmul_plan1 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan1.deinit();
    
    var matmul_plan2 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
    defer matmul_plan2.deinit();
    
    const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend, f32, null);
    var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
    defer relu_plan.deinit();
    
    // 3. Perform forward pass with plans
    std.debug.print("Running forward pass with plans...\n", .{});
    
    // h1 = x @ w1
    const h1 = try matmul_plan1.forward(.{ .a = x, .b = w1 });
    defer h1.deinit();
    
    // h2 = relu(h1)
    const h2 = try relu_plan.forward(h1);
    defer h2.deinit();
    
    // output = h2 @ w2
    const output = try matmul_plan2.forward(.{ .a = h2, .b = w2 });
    defer output.deinit();
    
    std.debug.print("Output: ", .{});
    printTensor(output);
    
    // 4. Create target and compute loss
    var target = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    defer target.deinit();
    try target.setScalar(&[_]usize{0, 0}, 0.5);
    try target.setScalar(&[_]usize{0, 1}, 0.5);
    try target.setScalar(&[_]usize{1, 0}, 0.5);
    try target.setScalar(&[_]usize{1, 1}, 0.5);
    
    // Compute MSE loss components
    var diff = try ops.subtract(allocator, output, target);
    defer diff.deinit();
    
    var diff_squared = try ops.multiply(allocator, diff, diff);
    defer diff_squared.deinit();
    
    // 5. Perform backward pass to compute gradients
    std.debug.print("\nRunning backward pass...\n", .{});
    
    // Create gradient with ones
    var grad_ones = try Tensor.filled(allocator, output.shape.dims, output.dtype, 1.0, output.backend);
    defer grad_ones.deinit();
    
    // Backprop through output layer
    const grads2 = try matmul_plan2.backward(grad_ones);
    defer grads2.da.deinit();
    defer grads2.db.deinit();
    
    // Backprop through hidden activation
    const relu_grad = try relu_plan.backward(grads2.da);
    defer relu_grad.deinit();
    
    // Backprop through input layer
    const grads1 = try matmul_plan1.backward(relu_grad);
    defer grads1.da.deinit();
    defer grads1.db.deinit();
    
    // 6. Print gradients
    std.debug.print("\nGradients for w1:\n", .{});
    printTensor(grads1.db);
    
    std.debug.print("\nGradients for w2:\n", .{});
    printTensor(grads2.db);
    
    // 7. Update weights
    // In a real training loop, you would update weights using:
    // w1 = w1 - learning_rate * grads1.db
    // w2 = w2 - learning_rate * grads2.db
}

// Helper to print tensor contents
fn printTensor(t: Tensor) void {
    const buf = @ptrCast([*]f32, @alignCast(@alignOf([*]f32), t.buffer.data.ptr))[0..t.shape.elemCount()];
    
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
        std.debug.print("[ ", .{});
        for (buf) |val| {
            std.debug.print("{d:.4} ", .{val});
        }
        std.debug.print("]\n", .{});
    }
}
```

Key points in this example:

1. Each operation in the network (matmul, relu) has its own plan
2. Plans are created with `AutoDiffPlan(PlanType).init(allocator)`
3. Forward pass calls `plan.forward(input)` instead of direct operations
4. Backward pass calls `plan.backward(gradient)` to compute gradients
5. Resources are properly managed with `defer plan.deinit()` and `defer tensor.deinit()`

This approach provides:
- Complete gradient tracking through the computation
- Proper memory management
- Type safety for all operations
- Consistency between forward and backward passes

### Project Examples

The project includes several example applications:

1. **plan_based_test.zig**: Dedicated test of the Plan-based autodiff approach
2. **gpt2_training.zig**: GPT-2 implementation with Plan-based training
3. **shakespeare_training.zig**: Training a language model on Shakespeare text using the Plan-based approach
4. **autodiff_test.zig**: Automatic differentiation tests using the Plan-based approach
5. **comptime_examples.zig**: Examples demonstrating the compile-time planning features

These examples demonstrate how to use the framework for real-world tasks and provide templates for building more complex applications.

---

This documentation provides an overview of the PCP framework. For more detailed information, refer to the source code and comments in the specific modules.