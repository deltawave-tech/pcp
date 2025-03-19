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
   - [Property-Based Testing](#property-based-testing)
6. [Zig Version](#zig-version)
7. [Examples](#examples)

## Introduction

Planetary Compute Protocol (PCP) is a distributed tensor computation framework written in Zig. It provides a foundation for building and training neural networks with proper memory management and automatic differentiation. PCP is designed for performance and memory safety, making it suitable for both research and production environments.

Key features:
- Pure Zig implementation with minimal external dependencies
- Reference-counted tensor management
- Automatic differentiation with compile-time plans
- Compile-time operation planning and validation
- Support for CPU and Metal backend computations
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

The operations module (`ops.zig`) provides mathematical operations on tensors using a compile-time plan-based approach.

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
              const allocator = self.base.allocator;
              
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

- **More Complex Gradient Example**: For operations like division, gradients follow proper calculus rules:
  ```zig
  pub fn DividePlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
      // Comptime validation...
      return struct {
          pub const InputType = struct { a: Tensor, b: Tensor };
          pub const op_type = OpType.divide;
          pub const GradType = struct { da: Tensor, db: Tensor };
          
          const Base = PlanType(Backend, InputType, Tensor);
          base: Base,
          
          // Implementation...
          
          /// Compute gradients for divide: da = grad_out / b, db = -grad_out * a / (b * b)
          pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
              const allocator = self.base.allocator;
              
              // Input a gradient is element-wise division of grad_out by b
              var da = try divide(allocator, grad_out, input.b);
              errdefer da.deinit();
              
              // Input b gradient is more complex: -grad_out * a / (b * b)
              // First calculate b^2
              var b_squared = try multiply(allocator, input.b, input.b);
              errdefer b_squared.deinit();
              
              // Calculate a / b^2
              var a_div_b_squared = try divide(allocator, input.a, b_squared);
              errdefer a_div_b_squared.deinit();
              
              // Multiply by grad_out
              var temp = try multiply(allocator, grad_out, a_div_b_squared);
              errdefer temp.deinit();
              
              // Negate the result
              const negative_one = try Tensor.filled(allocator, temp.shape.dims, temp.dtype, -1.0, temp.backend);
              errdefer negative_one.deinit();
              
              var db = try multiply(allocator, temp, negative_one);
              errdefer db.deinit();
              
              // Clean up temporaries
              b_squared.deinit();
              a_div_b_squared.deinit();
              temp.deinit();
              negative_one.deinit();
              
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

The key features of this design:
1. Gradient computation is tied directly to the operation that produces it
2. Each plan type includes its own gradient method following correct calculus rules
3. The AutoDiffPlan directly calls the plan's gradient method
4. Memory management is handled consistently with errdefer for cleanup
5. The code is more maintainable as adding a new operation only requires implementing one plan type

The Plan-based approach offers:
1. Strong compile-time guarantees and type safety
2. Simplified memory management with clear ownership
3. Close integration between operations and their gradient implementations
4. Natural fit with Zig's compile-time metaprogramming capabilities
5. Better performance through operation specialization
6. Proper implementation of mathematical gradient rules for each operation

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
   pub fn deinit(self: *Plan) void {
       // Clean up resources
       if (self.input) |input| {
           input.deinit();
       }
       
       if (self.output) |*output| {
           output.deinit();
       }
       
       // Free the Plan itself if needed
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

All operation plans include their own gradient computation methods. Here are two examples showing how different mathematical operations implement their gradient calculations:

##### Matrix Multiplication Gradients

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
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,
        
        // ... implementation of run() method ...
        
        /// Compute gradients for matmul: da = grad_out @ b^T, db = a^T @ grad_out
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // Compute b transpose
            const b_transpose = try transpose(allocator, input.b);
            errdefer b_transpose.deinit();
            
            // Gradient for a is grad_out @ b^T
            var da = try matmul(allocator, grad_out, b_transpose);
            errdefer da.deinit();
            
            // Compute a transpose
            const a_transpose = try transpose(allocator, input.a);
            errdefer a_transpose.deinit();
            
            // Gradient for b is a^T @ grad_out
            var db = try matmul(allocator, a_transpose, grad_out);
            errdefer db.deinit();
            
            // Clean up temporaries
            b_transpose.deinit();
            a_transpose.deinit();
            
            return .{ .da = da, .db = db };
        }
    };
}
```

##### Division Gradients

```zig
pub fn DividePlan(comptime Backend: type, comptime T: type, comptime shape: ?[]const usize) type {
    // Comptime validation
    comptime {
        if (!Backend.hasPrimitive("divide")) @compileError("Backend must implement divide primitive");
        if (T != f32) @compileError("Only f32 supported for now");
    }
    
    return struct {
        pub const InputType = struct { a: Tensor, b: Tensor };
        pub const op_type = OpType.divide;
        pub const GradType = struct { da: Tensor, db: Tensor };
        
        const Base = PlanType(Backend, InputType, Tensor);
        base: Base,
        
        // ... implementation of run() method ...
        
        /// Compute gradients for divide: da = grad_out / b, db = -grad_out * a / (b * b)
        pub fn gradient(self: @This(), grad_out: Tensor, input: InputType) !GradType {
            const allocator = self.base.allocator;
            
            // Input a gradient is element-wise division of grad_out by b
            var da = try divide(allocator, grad_out, input.b);
            errdefer da.deinit();
            
            // Input b gradient is more complex: -grad_out * a / (b * b)
            // First calculate b^2
            var b_squared = try multiply(allocator, input.b, input.b);
            errdefer b_squared.deinit();
            
            // Calculate a / b^2
            var a_div_b_squared = try divide(allocator, input.a, b_squared);
            errdefer a_div_b_squared.deinit();
            
            // Multiply by grad_out
            var temp = try multiply(allocator, grad_out, a_div_b_squared);
            errdefer temp.deinit();
            
            // Negate the result
            const negative_one = try Tensor.filled(allocator, temp.shape.dims, temp.dtype, -1.0, temp.backend);
            errdefer negative_one.deinit();
            
            var db = try multiply(allocator, temp, negative_one);
            errdefer db.deinit();
            
            // Clean up temporaries
            b_squared.deinit();
            a_div_b_squared.deinit();
            temp.deinit();
            negative_one.deinit();
            
            return .{ .da = da, .db = db };
        }
    };
}
```

#### Simplified Autodiff Backward

The AutoDiffPlan's backward method directly calls the plan's gradient method, providing a consistent interface:

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
6. Mathematical gradient rules are correctly implemented for each operation type
7. Memory management is handled consistently across all operations

### Property-Based Testing

PCP includes a state-of-the-art property-based testing framework to verify the correctness of tensor operations and their gradients. This approach focuses on testing mathematical properties and invariants rather than specific input-output pairs, providing stronger guarantees about the library's correctness.

#### Core Concepts

The property-based testing framework in `prop_tests.zig` follows these sophisticated principles:

1. **Enhanced Random Tensor Generation**: Generate tensors with random values, dimensions, and data types to test operations across a wide range of inputs.
2. **Mathematical Property Verification**: Test invariant properties that should hold for all valid inputs (e.g., commutativity, distributivity).
3. **Rigorous Gradient Verification**: Verify gradients both analytically and numerically using finite differences.
4. **Input Shrinking**: When tests fail, attempt to find minimal failing inputs.
5. **Comprehensive Error Reporting**: Detailed error information to help diagnose test failures.
6. **Multiple Test Iterations**: Run each test multiple times with different random inputs to improve coverage.
7. **Configurable Testing Parameters**: Easily adjust iterations, seed values, and verbosity.

#### Enhanced Random Tensor Generators

The framework includes sophisticated tensor generators that support variable dimensions, multiple data types, and edge cases:

```zig
/// Generates a random tensor with the specified shape constraints
/// max_dim limits the maximum size of each dimension
/// max_rank specifies the maximum number of dimensions (default is 2)
/// include_edge_cases determines whether to include special cases
/// dtype specifies the data type (default is f32)
pub fn randomTensor(
    allocator: Allocator, 
    rnd: *std.Random, 
    max_dim: usize, 
    max_rank: ?usize,
    include_edge_cases: ?bool,
    dtype: ?DType
) !Tensor {
    const rank = if (max_rank) |mr| rnd.intRangeAtMost(usize, 1, mr) else 2;
    const edge_cases = include_edge_cases orelse false;
    const data_type = dtype orelse .f32;
    
    // Prepare a buffer for dimensions
    var shape_buffer: [4]usize = undefined;
    
    // Generate random dimensions for each axis including edge cases
    for (0..rank) |i| {
        if (edge_cases and rnd.float(f32) < 0.1) {
            // 10% chance of single-element dimension (1)
            shape_buffer[i] = 1;
        } else {
            // Otherwise, random dimension between 1 and max_dim
            shape_buffer[i] = rnd.intRangeAtMost(usize, 1, max_dim);
        }
    }
    
    // Create tensor with generated shape
    var tensor_obj = try Tensor.zeros(allocator, shape_buffer[0..rank], data_type, .cpu);
    errdefer tensor_obj.deinit();
    
    // Fill with type-specific random values
    switch (data_type) {
        .f32 => {
            const buf = ptrCastHelper([*]f32, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
            for (buf) |*x| {
                x.* = rnd.float(f32) * 2.0 - 1.0; // Random value between -1.0 and 1.0
            }
        },
        .i32 => {
            const buf = ptrCastHelper([*]i32, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
            for (buf) |*x| {
                x.* = rnd.intRangeLessThan(i32, -100, 100); // Random integers
            }
        },
        // Other data types supported similarly
        else => {},
    }
    
    return tensor_obj;
}
```

Similar generators are available for tensor pairs and trios with compatible shapes for operations like addition and matrix multiplication.

#### Advanced Property Tests

The framework verifies a comprehensive set of mathematical properties for tensor operations:

```zig
/// Check property: A + B = B + A (commutativity of addition)
pub fn checkAdditionCommutativity(allocator: Allocator, rnd: *std.Random) !void {
    // Generate random tensors with options for different dtypes and dimensions
    var pair = try randomTensorPair(
        allocator, 
        rnd, 
        5,            // max_dim
        3,            // max_rank (up to 3D tensors)
        true,         // include edge cases
        .f32          // dtype
    );
    defer pair.a.deinit();
    defer pair.b.deinit();

    var a_plus_b = try ops.add(allocator, pair.a, pair.b);
    defer a_plus_b.deinit();
    
    var b_plus_a = try ops.add(allocator, pair.b, pair.a);
    defer b_plus_a.deinit();

    // Check equality with relative error and improved reporting
    const comparison = try tensorsApproxEqual(a_plus_b, b_plus_a, 1e-5, true);
    
    if (comparison == .err) {
        std.debug.print("Addition commutativity test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        
        std.debug.print("Tensor shapes: ", .{});
        for (pair.a.shape.dims) |d| {
            std.debug.print("{} ", .{d});
        }
        std.debug.print("\n", .{});
        
        return error.TensorComparisonFailed;
    }
}

/// Check property: A⋅(B+C)=A⋅B+A⋅C (distributivity of matmul over addition)
pub fn checkMatmulDistributivity(allocator: Allocator, rnd: *std.Random) !void {
    // Generate compatible tensors for matmul distributivity test
    const m = rnd.intRangeAtMost(usize, 1, 5);
    const n = rnd.intRangeAtMost(usize, 1, 5);
    const p = rnd.intRangeAtMost(usize, 1, 5);
    
    // A is m x n, B and C are n x p
    var a = try Tensor.zeros(allocator, &[_]usize{ m, n }, .f32, .cpu);
    defer a.deinit();
    
    var b = try Tensor.zeros(allocator, &[_]usize{ n, p }, .f32, .cpu);
    defer b.deinit();
    
    var c = try Tensor.zeros(allocator, &[_]usize{ n, p }, .f32, .cpu);
    defer c.deinit();
    
    // Fill with random values and perform the test
    // ...

    // Compare: A⋅(B+C) vs. A⋅B + A⋅C
    const comparison = try tensorsApproxEqual(left, right, 1e-4, true);
    
    if (comparison == .err) {
        std.debug.print("Matrix multiplication distributivity test failed:\n", .{});
        printTensorErrorReport(comparison.err);
        // Detailed error information
    }
}

/// Check property: A/A = 1 for all non-zero A
pub fn checkDivisionByItself(allocator: Allocator, rnd: *std.Random) !void {
    // Create tensor with non-zero values
    var tensor_obj = try randomTensor(allocator, rnd, 5, 3, false, .f32);
    defer tensor_obj.deinit();
    
    // Ensure no values are too close to zero
    const buf = ptrCastHelper([*]f32, tensor_obj.buffer.data.ptr)[0..tensor_obj.shape.elemCount()];
    for (buf) |*x| {
        if (@abs(x.*) < 0.5) {
            x.* = if (x.* < 0) -0.5 else 0.5;
        }
    }
    
    // Verify A/A = 1
    // ...
}
```

#### Enhanced Error Reporting

The framework provides sophisticated error reporting to help diagnose test failures:

```zig
/// Error information returned when tensors are not approximately equal
pub const TensorComparisonError = struct {
    max_abs_diff: f32,      // Maximum absolute difference found
    max_rel_diff: f32,      // Maximum relative difference found
    failure_index: usize,   // Index where the largest difference was found
    a_value: f32,           // Value in tensor A at failure point
    b_value: f32,           // Value in tensor B at failure point
    shape_mismatch: bool,   // Whether the tensors had different shapes
};

/// Helper to print a tensor error report with details on where the comparison failed
pub fn printTensorErrorReport(error_info: TensorComparisonError) void {
    if (error_info.shape_mismatch) {
        std.debug.print("Tensor shape mismatch\n", .{});
    } else {
        std.debug.print("Tensor comparison failed at index {}\n", .{error_info.failure_index});
        std.debug.print("Values: A={d:.6}, B={d:.6}\n", .{error_info.a_value, error_info.b_value});
        std.debug.print("Absolute difference: {d:.6}\n", .{error_info.max_abs_diff});
        if (error_info.max_rel_diff > 0) {
            std.debug.print("Relative difference: {d:.6}\n", .{error_info.max_rel_diff});
        }
    }
}
```

#### Finite Difference Gradient Verification

One of the most advanced features is numerical gradient verification using finite differences:

```zig
/// Helper function to calculate a numerical approximation of gradients using finite differences
pub fn finiteDifference(
    allocator: Allocator, 
    op: anytype, 
    input: Tensor, 
    epsilon: f32
) !Tensor {
    // Compute the forward result of the operation
    const forward = try op.forward(input);
    defer forward.deinit();
    
    // Create a result tensor to store the gradients
    var grad = try Tensor.zeros(allocator, input.shape.dims, .f32, .cpu);
    errdefer grad.deinit();
    
    // For each element in the input tensor
    for (0..input.shape.elemCount()) |i| {
        // Save the original value
        const orig_val = input_buf[i];
        
        // Perturb the input by epsilon
        input_buf[i] = orig_val + epsilon;
        
        // Compute the perturbed output
        const perturbed_forward = try op.forward(input);
        defer perturbed_forward.deinit();
        
        // Compute the numerical gradient (df/dx ≈ [f(x+ε) - f(x)]/ε)
        var diff_sum: f32 = 0.0;
        for (forward_buf, perturbed_buf) |orig, pert| {
            diff_sum += pert - orig;
        }
        
        grad_buf[i] = diff_sum / epsilon;
        
        // Restore the original value
        input_buf[i] = orig_val;
    }
    
    return grad;
}

/// Check gradient correctness using finite differences
pub fn checkGradientWithFiniteDifferences(allocator: Allocator, rnd: *std.Random) !void {
    // Create a test tensor
    var test_tensor = try randomTensor(allocator, rnd, 4, 2, false, .f32);
    defer test_tensor.deinit();
    
    // Set up a test operation (ReLU)
    const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend, f32, null);
    var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
    defer relu_plan.deinit();
    
    // Get gradient from autodiff
    const autodiff_grad = /* ... */;
    
    // Get gradient from finite differences
    var test_copy = try test_tensor.clone();
    defer test_copy.deinit();
    var numerical_grad = try finiteDifference(allocator, &relu_plan, test_copy, 1e-4);
    defer numerical_grad.deinit();
    
    // Compare the two gradients
    const comparison = try tensorsApproxEqual(autodiff_grad, numerical_grad, 1e-2, true);
    
    if (comparison == .err) {
        // Special handling for ReLU discontinuity at x=0
        const input_val = input_buf[comparison.err.failure_index];
        if (@abs(input_val) < 1e-3) {
            std.debug.print("Ignoring error near ReLU activation boundary (x ≈ 0)\n", .{});
            return;
        }
        
        // Error reporting for other cases
        // ...
    }
}
```

#### Advanced Test Runner with Shrinking

The framework includes a sophisticated test runner that supports shrinking (trying to find minimal failing cases):

```zig
/// Run a property test function multiple times with different random inputs
/// Performs basic shrinking on failure by reducing dimensions when a test fails
pub fn runPropertyTest(
    allocator: Allocator,
    comptime testFn: fn (Allocator, *std.Random) anyerror!void,
    iterations: usize,
    seed: ?u64,
    verbose: ?bool
) !void {
    const verbose_output = verbose orelse false;
    
    // Create PRNG with specified or random seed
    var prng = blk: {
        const s = seed orelse @as(u64, @intCast(@abs(std.time.milliTimestamp())));
        if (verbose_output) {
            std.debug.print("Using seed: {}\n", .{s});
        }
        break :blk std.Random.DefaultPrng.init(s);
    };
    var rand = prng.random();
    const rnd = &rand;
    
    // Run the test multiple times with different random inputs
    for (0..iterations) |i| {
        if (verbose_output and i % 10 == 0) {
            std.debug.print("Running iteration {}/{}\n", .{i + 1, iterations});
        }
        
        // Try to run the test
        testFn(allocator, rnd) catch |err| {
            // If the test fails, try to shrink inputs by reducing dimensions
            std.debug.print("Test failed on iteration {} with error: {!}\n", .{i + 1, err});
            
            // Try to find a minimal failing case by shrinking the input
            std.debug.print("Attempting to find minimal failing case...\n", .{});
            
            // Try with increasingly smaller dimensions
            const shrink_steps = [_]usize{ 4, 3, 2, 1 };
            var found_min = false;
            
            for (shrink_steps) |dim| {
                std.debug.print("Trying with max_dim={}\n", .{dim});
                
                // Try to run with reduced dimensions
                testFn(allocator, rnd) catch {
                    found_min = true;
                    std.debug.print("Test still fails with max_dim={}\n", .{dim});
                    break;
                };
            }
            
            if (!found_min) {
                std.debug.print("Could not find minimal case; test passed with smaller dimensions\n", .{});
            }
            
            return err;
        };
    }
}
```

#### Test Registration

Tests are registered using a uniform pattern that leverages the advanced test runner:

```zig
test "property-based addition commutativity" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkAdditionCommutativity,
        50,          // iterations
        42,          // seed for reproducibility
        false        // verbose output
    );
}

test "property-based matmul distributivity" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkMatmulDistributivity,
        50,          // iterations
        42,          // seed for reproducibility
        false        // verbose output
    );
}

test "property-based finite difference gradient verification" {
    const allocator = std.testing.allocator;
    try runPropertyTest(
        allocator,
        checkGradientWithFiniteDifferences,
        10,          // fewer iterations (more computationally expensive)
        42,          // seed for reproducibility
        false        // verbose output
    );
}
```

#### Benefits of Advanced Property-Based Testing

This sophisticated approach provides several significant advantages:

1. **Comprehensive Input Coverage**: Tests across diverse tensor shapes, dimensions, and data types
2. **Mathematical Invariants**: Verifies fundamental properties like commutativity, associativity, and distributivity
3. **Multi-Method Gradient Verification**: Uses both analytical derivation and numerical approximation
4. **Instructive Failure Reporting**: Provides detailed error context to quickly resolve failures
5. **Shrinking to Minimal Examples**: Helps find the smallest failing inputs for easier debugging
6. **Edge Case Detection**: Actively tests boundary conditions and numerically challenging scenarios
7. **Reproducible Testing**: Seed-based generation allows exact test reproduction
8. **Flexible Test Configuration**: Adjustable test parameters for different testing environments

The property-based testing framework ensures the mathematical correctness and robustness of PCP's tensor operations and gradients, providing high confidence in the library's behavior across a wide range of usage scenarios.

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

The project includes several example applications that demonstrate how to use the framework:

1. **plan_based_test.zig**: Dedicated test of the Plan-based autodiff approach - this example shows a simple neural network using plans for all operations and demonstrates both forward and backward passes.

2. **gpt2_training.zig**: Complete GPT-2 implementation with Plan-based training - shows how to implement a state-of-the-art transformer model using the framework, including gradient-based training.

3. **shakespeare_training.zig**: Training a language model on Shakespeare text using the Plan-based approach - demonstrates how to load data, train a model, and generate text.

4. **autodiff_test.zig**: Comprehensive automatic differentiation tests using the Plan-based approach - includes tests for various operations like addition, multiplication, division, matmul, ReLU, and embedding lookups.

5. **comptime_examples.zig**: Examples demonstrating the compile-time planning features - shows basic operations with compile-time shapes and how to use autodiff with these operations.

6. **metal_test.zig**: Test of Metal backend integration with the plan-based architecture - demonstrates how the framework can be extended to new hardware backends.

7. **metal_benchmark.zig**: Performance benchmarks for the Metal backend - shows how to benchmark operations across different backends.

Each example builds upon the Plan-based approach to automatic differentiation and demonstrates proper memory management, error handling, and gradient computation. Together, they showcase how the framework can be used for everything from simple neural networks to complex language models with proper gradient-based training.

For new users, the recommended learning path is:
1. Start with `comptime_examples.zig` to understand the basic operations
2. Move to `plan_based_test.zig` to see how autodiff works
3. Explore `autodiff_test.zig` to understand more complex operations and their gradients
4. Finally examine the practical applications in `shakespeare_training.zig` and `gpt2_training.zig`

---

This documentation provides an overview of the PCP framework. For more detailed information, refer to the source code and comments in the specific modules.