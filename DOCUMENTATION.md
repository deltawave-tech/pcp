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

The autodiff module (`autodiff.zig`) provides automatic differentiation capabilities through two approaches:

#### 1. Computational Graph Approach (Legacy)

- **Node**: A node in the computational graph
  ```zig
  pub const Node = struct {
      // Output of this node
      tensor: Tensor,
      
      // Operation that created this node
      op_type: ?OpType = null,
      
      // Inputs to this operation
      inputs: std.ArrayList(*Node),
      
      // Gradient of the loss with respect to this node's output
      grad: ?Tensor = null,
      
      // Whether this node requires gradient computation
      requires_grad: bool,
      
      // Allocator for managing memory
      allocator: Allocator,
      
      // Methods for initialization, gradient handling, etc.
  };
  ```

- **Backward Pass**: Gradient computation through the graph
  ```zig
  pub fn backward(allocator: Allocator, node: *Node) !void {
      // Initialize gradient
      try node.initGradOnes();
      
      // Build a topological sort of the graph
      var visited = std.AutoHashMap(*Node, void).init(allocator);
      defer visited.deinit();
      
      var topo_order = std.ArrayList(*Node).init(allocator);
      defer topo_order.deinit();
      
      // DFS to build topological sort
      try buildTopoSort(node, &visited, &topo_order);
      
      // Backward pass in reverse topological order
      var i: usize = topo_order.items.len;
      while (i > 0) {
          i -= 1;
          const current = topo_order.items[i];
          
          // Skip nodes that don't require gradients
          if (!current.requires_grad) continue;
          
          // Process according to operation type
          if (current.op_type) |op| {
              switch (op) {
                  .add => try backwardAdd(allocator, current),
                  .subtract => try backwardSubtract(allocator, current),
                  .multiply => try backwardMultiply(allocator, current),
                  .matmul => try backwardMatmul(allocator, current),
                  // More operations...
                  else => return error.UnsupportedOperationBackward,
              }
          }
      }
  }
  ```

#### 2. Compile-Time Plan Approach (Modern)

- **GradRules**: Generic gradient computation rules
  ```zig
  pub const GradRules = struct {
      /// Gradient for add: da = grad_out, db = grad_out
      pub fn add(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
          // Implementation...
      }
      
      /// Gradient for matmul: da = grad_out @ b^T, db = a^T @ grad_out
      pub fn matmul(allocator: Allocator, grad_out: Tensor, a: Tensor, b: Tensor) !struct { da: Tensor, db: Tensor } {
          // Implementation...
      }
      
      // Other gradient rules...
  };
  ```

- **AutoDiffPlan**: Wrapper for operation plans that adds gradient functionality
  ```zig
  pub fn AutoDiffPlan(comptime ForwardPlanType: type) type {
      // Comptime validation
      comptime {
          if (!@hasField(ForwardPlanType, "GradType")) {
              @compileError("Forward plan must define GradType for autodiff");
          }
          if (!@hasField(ForwardPlanType, "op_type")) {
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
          
          // Methods for forward and backward...
          
          /// Comptime function to select and apply the appropriate gradient rule
          fn computeGradient(self: *Self, grad_out: Tensor) !ForwardPlanType.GradType {
              // Use comptime type information to select the right gradient function
              switch (ForwardPlanType.op_type) {
                  .add => {
                      return try GradRules.add(self.allocator, grad_out, input.a, input.b);
                  },
                  // Other operation types...
              }
          }
      };
  }
  ```

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

- **GPT2**: Main model implementation with attention, MLP, and layer normalization components

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

#### Compile-Time Type Selection for Gradients

The core of the autodiff system uses comptime to select the appropriate gradient computation:

```zig
fn computeGradient(self: *Self, grad_out: Tensor) !ForwardPlanType.GradType {
    // Use comptime type information to select the right gradient function
    switch (ForwardPlanType.op_type) {
        .add => {
            return try GradRules.add(self.allocator, grad_out, input.a, input.b);
        },
        .subtract => {
            return try GradRules.subtract(self.allocator, grad_out, input.a, input.b);
        },
        .matmul => {
            return try GradRules.matmul(self.allocator, grad_out, input.a, input.b);
        },
        // Other operations...
    }
}
```

This approach ensures:
1. Operations are validated at compile time
2. The correct gradient function is selected at compile time
3. Runtime overhead is minimized
4. Type safety is maintained throughout the system

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

The project includes several example applications:

1. **autodiff_test.zig**: Basic automatic differentiation and memory management tests
2. **shakespeare_training.zig**: Training a language model on Shakespeare text
3. **gpt2_training.zig**: GPT-2 implementation with training utilities

These examples demonstrate how to use the framework for real-world tasks and provide templates for building more complex applications.

---

This documentation provides an overview of the PCP framework. For more detailed information, refer to the source code and comments in the specific modules.