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
6. [Compatibility Notes](#compatibility-notes)
7. [Examples](#examples)

## Introduction

Planetary Compute Protocol (PCP) is a distributed tensor computation framework written in Zig. It provides a foundation for building and training neural networks with proper memory management and automatic differentiation. PCP is designed for performance and memory safety, making it suitable for both research and production environments.

Key features:
- Pure Zig implementation with no external dependencies
- Reference-counted tensor management
- Automatic differentiation with computational graphs
- Support for CPU computations (with Metal/GPU support in development)
- Memory-safe tensor operations with proper cleanup

## Architecture Overview

PCP follows a layered architecture design:

1. **Core Tensor Layer**: Low-level tensor operations, memory management
2. **Operations Layer**: Mathematical functions on tensors
3. **Autodiff Layer**: Gradient computation through a computational graph
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

The operations module (`ops.zig`) provides mathematical operations on tensors.

Key operations:
- add/subtract (element-wise)
- multiply (element-wise)
- matmul (matrix multiplication)
- relu/softmax (activation functions)
- transpose

Example implementation:

```zig
pub fn matmul(allocator: Allocator, a: Tensor, b: Tensor) !Tensor {
    // Check dimensions for matrix multiplication: (m, n) x (n, p) -> (m, p)
    if (a.shape.rank() != 2 or b.shape.rank() != 2) {
        return OpError.DimensionMismatch;
    }
    
    if (a.shape.dims[1] != b.shape.dims[0]) {
        return OpError.ShapeMismatch;
    }
    
    // Result dimensions: [a.dims[0], b.dims[1]]
    const result_dims = [_]usize{ a.shape.dims[0], b.shape.dims[1] };
    var result = try Tensor.zeros(allocator, &result_dims, a.dtype, a.backend);
    
    switch (a.dtype) {
        .f32 => {
            const a_buf = @ptrCast([*]f32, a.buffer.data.ptr);
            const b_buf = @ptrCast([*]f32, b.buffer.data.ptr);
            const result_buf = @ptrCast([*]f32, result.buffer.data.ptr);
            
            const m = a.shape.dims[0];
            const n = a.shape.dims[1]; // also b.shape.dims[0]
            const p = b.shape.dims[1];
            
            // Simple 3-loop matrix multiplication
            for (0..m) |i| {
                for (0..p) |j| {
                    var sum: f32 = 0.0;
                    for (0..n) |k| {
                        sum += a_buf[i * n + k] * b_buf[k * p + j];
                    }
                    result_buf[i * p + j] = sum;
                }
            }
        },
        else => {
            // Clean up the result tensor on error
            result.deinit();
            return OpError.UnsupportedDataType;
        }
    }
    
    return result;
}
```

### Autodiff Engine

The autodiff module (`autodiff.zig`) provides automatic differentiation capabilities through a computational graph.

Key components:

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

- **Operation Nodes**: Functions that create operation nodes in the graph
  ```zig
  pub fn matmul(allocator: Allocator, a: *Node, b: *Node) !*Node {
      // Perform the operation
      var result_tensor = try ops.matmul(allocator, a.tensor, b.tensor);
      result_tensor.requires_grad = a.requires_grad or b.requires_grad;
      
      // Create the result node
      var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
      result.op_type = .matmul;
      
      // Save input nodes for backward pass
      try result.inputs.append(a);
      try result.inputs.append(b);
      
      return result;
  }
  ```

- **Backward Pass**: Gradient computation through the graph
  ```zig
  pub fn backward(allocator: Allocator, node: *Node) !void {
      // Initialize gradient of the output node to ones
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

4. **Builder Pattern**: Operations chain together to build the computation graph
   ```zig
   var h_pre = try autodiff.matmul(allocator, x, w1); // Linear layer
   var h = try autodiff.relu(allocator, h_pre); // Activation
   var y_pred = try autodiff.matmul(allocator, h, w2); // Second linear layer
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
            const f32_buf = @ptrCast([*]f32, self.buffer.data.ptr);
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

Embedding layers require special handling for gradient flow and memory management. PCP implements this with careful reference counting:

```zig
pub fn embedding_lookup(allocator: Allocator, params: *Node, indices: Tensor) !*Node {
    // Create output tensor with shape [batch_size, seq_len, embed_dim]
    var result_dims = [_]usize{ batch_size, seq_len, embed_dim };
    var result_tensor = try Tensor.zeros(allocator, &result_dims, params.tensor.dtype, params.tensor.backend);
    
    // Perform lookup - simplified for readability
    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            const token_id = // Get token ID from indices
            
            // Copy embedding for this token
            for (0..embed_dim) |d| {
                const src_idx = token_id * embed_dim + d;
                const dst_idx = (b * seq_len + s) * embed_dim + d;
                result_buf[dst_idx] = params_buf[src_idx];
            }
        }
    }
    
    // Create result node
    var result = try Node.init(allocator, result_tensor, params.requires_grad);
    result.op_type = .embedding_lookup;
    
    // Save input nodes for backward pass
    try result.inputs.append(params);
    
    // Create a copy of the indices tensor that we'll own
    const indices_for_copy = indices.retain();
    const indices_node = try Node.init(allocator, indices_for_copy, false);
    try result.inputs.append(indices_node);
    
    return result;
}
```

The backward pass for embeddings is complex due to the sparse nature of gradients:

```zig
fn backwardEmbeddingLookup(allocator: Allocator, node: *Node) !void {
    // For each token in the batch, accumulate gradients back to the embedding table
    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            const token_id = // Get token ID from indices
            
            // Accumulate gradients for this embedding
            for (0..embed_dim) |d| {
                const grad_pos = (b * seq_len + s) * embed_dim + d;
                const param_pos = token_id * embed_dim + d;
                params_grad_buf[param_pos] += grad_buf[grad_pos];
            }
        }
    }
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

The optimizer implementation (Adam) shows how gradients are applied to parameters:

```zig
pub fn step(self: *Adam, parameter: *Tensor, gradient: Tensor) !void {
    // Increment time step
    self.t += 1;
    
    // Get moment vectors
    var m = self.m_map.get(parameter).?;
    var v = self.v_map.get(parameter).?;
    
    // Get pointers to the data
    const param_buf = @ptrCast([*]f32, parameter.buffer.data.ptr)[0..parameter.shape.elemCount()];
    const grad_buf = @ptrCast([*]f32, gradient.buffer.data.ptr)[0..gradient.shape.elemCount()];
    const m_buf = @ptrCast([*]f32, m.buffer.data.ptr)[0..m.shape.elemCount()];
    const v_buf = @ptrCast([*]f32, v.buffer.data.ptr)[0..v.shape.elemCount()];
    
    // Update parameters using Adam update rule
    const lr = self.learning_rate;
    const beta1 = self.beta1;
    const beta2 = self.beta2;
    const epsilon = self.epsilon;
    
    // Apply Adam update to each parameter
    for (param_buf, 0..) |*param, i| {
        const g = grad_buf[i];
        
        // Update biased first moment estimate
        m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g;
        
        // Update biased second raw moment estimate
        v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g * g;
        
        // Apply update
        param.* -= lr_t * m_buf[i] / (@sqrt(v_buf[i]) + epsilon);
    }
}
```

## Compatibility Notes

The project supports multiple versions of Zig with recent compatibility updates:

- Compatible with Zig 0.12.0 through 0.14.x
- Updated `std.rand` to `std.Random` for random number generation
- Updated pointer cast syntax:
  - Old: `@as([*]f32, @ptrCast(@alignCast(ptr)))[0..len]`
  - New: `@ptrCast([*]f32, ptr)[0..len]`

These changes maintain backward compatibility while enabling the code to work with newer Zig versions.

## Examples

The project includes several example applications:

1. **autodiff_test.zig**: Basic automatic differentiation and memory management tests
2. **shakespeare_training.zig**: Training a language model on Shakespeare text
3. **gpt2_training.zig**: GPT-2 implementation with training utilities

These examples demonstrate how to use the framework for real-world tasks and provide templates for building more complex applications.

---

This documentation provides an overview of the PCP framework. For more detailed information, refer to the source code and comments in the specific modules.