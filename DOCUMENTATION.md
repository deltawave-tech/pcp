# PCP Documentation

A comprehensive guide to the MLIR-based tensor computation framework.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [MLIR Integration](#mlir-integration)
4. [Automatic Differentiation](#automatic-differentiation)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Best Practices](#best-practices)

## Architecture Overview

PCP transforms high-level tensor operations into optimized MLIR computation graphs:

```
Zig Tensor API → MLIRBuilder → StableHLO MLIR → Execution
      ↓             ↓              ↓             ↓
  VJP Rules → Gradient Graph → Optimized Code → Results
```

### Core Layers

1. **Tensor Abstraction**: High-level tensor operations with shape validation
2. **MLIR Builder**: Constructs computation graphs using MLIR C API
3. **StableHLO Dialect**: Clean Zig wrapper for StableHLO operations
4. **VJP Autodiff**: Vector-Jacobian Product automatic differentiation
5. **Execution Engine**: MLIR-based compilation and execution

## Core Components

### MLIRBuilder

Central component for building computation graphs:

```zig
const builder = try ops.MLIRBuilder.init(allocator);
defer builder.deinit();

// Create tensors and operations
const result = try ops.add(&builder, tensor_a, tensor_b);
```

**Key methods:**
- `init(allocator)`: Initialize with MLIR context
- `createConstant(value, type)`: Create constant values
- `createOp(name, operands, result_type)`: Create generic operations
- `deinit()`: Clean up MLIR context

### StableHLO Dialect Wrapper

Clean Zig interface to MLIR StableHLO operations in `src/mlir/dialects/stablehlo.zig`:

```zig
// Element-wise operations
pub fn add(ctx: Context, lhs: Value, rhs: Value, loc: Location) Operation
pub fn multiply(ctx: Context, lhs: Value, rhs: Value, loc: Location) Operation

// Matrix operations  
pub fn dot_general(ctx: Context, lhs: Value, rhs: Value, args: DotArgs) Operation

// Activation functions
pub fn maximum(ctx: Context, lhs: Value, rhs: Value, loc: Location) Operation // ReLU

// Tensor manipulation
pub fn transpose(ctx: Context, operand: Value, permutation: []const i64, loc: Location) Operation
pub fn reshape(ctx: Context, operand: Value, result_shape: []const i64, loc: Location) Operation
```

### Tensor Operations

High-level tensor operations in `src/ops.zig`:

```zig
// Element-wise operations
pub fn add(builder: *MLIRBuilder, a: Tensor, b: Tensor) !Tensor
pub fn subtract(builder: *MLIRBuilder, a: Tensor, b: Tensor) !Tensor 
pub fn multiply(builder: *MLIRBuilder, a: Tensor, b: Tensor) !Tensor
pub fn divide(builder: *MLIRBuilder, a: Tensor, b: Tensor) !Tensor

// Matrix operations
pub fn matmul(builder: *MLIRBuilder, a: Tensor, b: Tensor) !Tensor

// Activation functions
pub fn relu(builder: *MLIRBuilder, a: Tensor) !Tensor

// Tensor manipulation
pub fn transpose(builder: *MLIRBuilder, a: Tensor, permutation: []const i64) !Tensor
pub fn reshape(builder: *MLIRBuilder, a: Tensor, new_shape: []const i64) !Tensor
```

## MLIR Integration

### C API Bindings

`src/mlir/c.zig` provides comprehensive MLIR C API bindings:

```zig
// Context management
extern fn mlirContextCreate() *MlirContext;
extern fn mlirContextDestroy(ctx: *MlirContext) void;

// Operation creation  
extern fn mlirOperationCreate(state: *MlirOperationState) *MlirOperation;

// Type system
extern fn mlirRankedTensorTypeGet(rank: isize, shape: [*]const i64, elementType: *MlirType) *MlirType;

// Value operations
extern fn mlirValueGetType(value: *MlirValue) *MlirType;
```

### MLIR Wrapper Types

`src/mlir.zig` provides safe Zig wrappers:

```zig
pub const Context = struct {
    handle: *c.MlirContext,
    
    pub fn init() !Self
    pub fn deinit(self: Self) void
};

pub const Operation = struct {
    handle: *c.MlirOperation,
    
    pub fn create(ctx: Context, op_name: []const u8, args: CreateArgs) Self
    pub fn getResult(self: Self, index: usize) Value
    pub fn deinit(self: Self) void
};

pub const Value = struct {
    handle: *c.MlirValue,
    
    pub fn getType(self: Self) Type
};
```

## Automatic Differentiation

### VJP-Based Reverse Mode AD

PCP implements Vector-Jacobian Product (VJP) automatic differentiation:

```zig
// VJP function signature
const VJPFn = *const fn(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) anyerror![]mlir.Value;
```

### Compile-Time VJP Dispatch

Operations are mapped to their VJP rules at compile time:

```zig
fn getVjpFn(comptime op_name: []const u8) VJPFn {
    if (std.mem.eql(u8, op_name, "stablehlo.add")) {
        return addVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.multiply")) {
        return multiplyVJP;
    // ... more operations
    } else {
        @compileError("No VJP rule defined for: " ++ op_name);
    }
}
```

### VJP Rules

Each operation implements its mathematical gradient:

```zig
// Addition: both inputs get same gradient
fn addVJP(builder: *MLIRBuilder, op: Operation, adjoints: []const Value) ![]Value {
    const grad_out = adjoints[0];
    return &.{ grad_out, grad_out }; // da = db = grad_out
}

// Multiplication: da = grad_out * b, db = grad_out * a  
fn multiplyVJP(builder: *MLIRBuilder, op: Operation, adjoints: []const Value) ![]Value {
    const grad_out = adjoints[0];
    const a = op.getOperand(0);
    const b = op.getOperand(1);
    
    const grad_a = try builder.createOp("stablehlo.multiply", &.{ grad_out, b }, grad_out.getType());
    const grad_b = try builder.createOp("stablehlo.multiply", &.{ grad_out, a }, grad_out.getType());
    
    return &.{ grad_a.getResult(0), grad_b.getResult(0) };
}

// Matrix multiplication: da = grad_out @ b^T, db = a^T @ grad_out
fn matmulVJP(builder: *MLIRBuilder, op: Operation, adjoints: []const Value) ![]Value {
    const grad_out = adjoints[0];
    const a = op.getOperand(0);
    const b = op.getOperand(1);
    
    const b_t = try builder.createOp("stablehlo.transpose", &.{b}, b.getType());
    const a_t = try builder.createOp("stablehlo.transpose", &.{a}, a.getType()); 
    
    const grad_a = try builder.createOp("stablehlo.dot_general", &.{ grad_out, b_t.getResult(0) }, grad_out.getType());
    const grad_b = try builder.createOp("stablehlo.dot_general", &.{ a_t.getResult(0), grad_out }, grad_out.getType());
    
    return &.{ grad_a.getResult(0), grad_b.getResult(0) };
}
```

### AutoDiff API

High-level automatic differentiation interface:

```zig
var autodiff = AutoDiff.init(allocator, &builder);

// Transform forward function to gradient function
const grad_fn = try autodiff.grad(forward_fn);

// Get both value and gradients
const value_and_grad_fn = try autodiff.valueAndGrad(forward_fn);
```

## API Reference

### Tensor Creation

```zig
// Create tensors
var tensor = try Tensor.zeros(allocator, &[_]usize{2, 3}, .f32, .cpu);
var tensor = try Tensor.ones(allocator, &[_]usize{2, 3}, .f32, .cpu);
var tensor = try Tensor.filled(allocator, &[_]usize{2, 3}, .f32, 5.0, .cpu);
var tensor = try Tensor.random(allocator, &[_]usize{2, 3}, .f32, .cpu);

// From data
const data = &[_]f32{ 1.0, 2.0, 3.0, 4.0 };
var tensor = try Tensor.fromSlice(allocator, data, &[_]usize{2, 2}, .cpu);
```

### Basic Operations

```zig
// Arithmetic operations
const c = try ops.add(&builder, a, b);
const c = try ops.subtract(&builder, a, b);  
const c = try ops.multiply(&builder, a, b);
const c = try ops.divide(&builder, a, b);

// Matrix operations
const c = try ops.matmul(&builder, a, b);

// Activation functions
const activated = try ops.relu(&builder, input);

// Tensor manipulation
const transposed = try ops.transpose(&builder, tensor, &[_]i64{1, 0});
const reshaped = try ops.reshape(&builder, tensor, &[_]i64{4, 2});
```

### Memory Management

```zig
// All tensors must be properly freed
var tensor = try Tensor.zeros(allocator, shape, .f32, .cpu);
defer tensor.deinit();

// MLIRBuilder manages MLIR context
var builder = try ops.MLIRBuilder.init(allocator);
defer builder.deinit();
```

## Examples

### Basic Neural Network

```zig
const std = @import("std");
const pcp = @import("pcp");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{});
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create MLIR builder
    var builder = try pcp.ops.MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    // Create tensors
    const shape = &[_]usize{2, 2};
    var x = try pcp.tensor.Tensor.ones(allocator, shape, .f32, .cpu);
    var w = try pcp.tensor.Tensor.random(allocator, shape, .f32, .cpu);
    defer x.deinit();
    defer w.deinit();
    
    // Forward pass: output = relu(x @ w)
    const matmul_result = try pcp.ops.matmul(&builder, x, w);
    const output = try pcp.ops.relu(&builder, matmul_result);
    defer matmul_result.deinit();
    defer output.deinit();
    
    // Automatic differentiation
    var autodiff = pcp.autodiff.AutoDiff.init(allocator, &builder);
    const grad_fn = try autodiff.grad(output.operation);
    
    std.debug.print("Forward pass completed!\n", .{});
}
```

### GPT-2 Model

```zig
// Create GPT-2 model with MLIR operations
var model = try GPT2.init(allocator, config, .cpu);
defer model.deinit();

// Forward pass
const logits = try model.forward(&builder, input_ids);
defer logits.deinit();

// Compute gradients
var autodiff = AutoDiff.init(allocator, &builder);
const grad_fn = try autodiff.grad(logits.operation);
```

## Best Practices

### Memory Management

1. **Always use defer**: Every tensor and builder must be cleaned up
   ```zig
   var tensor = try Tensor.zeros(allocator, shape, .f32, .cpu);
   defer tensor.deinit(); // Always defer cleanup
   ```

2. **One builder per computation**: Create MLIRBuilder once per computation graph
   ```zig
   var builder = try MLIRBuilder.init(allocator);
   defer builder.deinit();
   // Use same builder for all related operations
   ```

3. **Clean up intermediate results**: Free temporary tensors promptly
   ```zig
   const temp = try ops.add(&builder, a, b);
   defer temp.deinit(); // Clean up intermediate results
   const result = try ops.multiply(&builder, temp, c);
   ```

### Error Handling

1. **Propagate errors**: Use `try` for all tensor operations
   ```zig
   const result = try ops.matmul(&builder, a, b); // Propagate errors
   ```

2. **Handle shape mismatches**: Operations validate tensor shapes
   ```zig
   // This will return error.IncompatibleShapes if shapes don't match
   const result = try ops.add(&builder, a, b);
   ```

### Performance

1. **Batch operations**: Group related operations in single builder
2. **Use appropriate data types**: f32 for most ML, f16 for memory efficiency
3. **Leverage MLIR optimizations**: Let MLIR optimize the computation graph

### MLIR Integration

1. **Understand operation mapping**: Know how Zig ops map to StableHLO
2. **Use dialect wrappers**: Prefer wrapper functions over raw C API
3. **Handle MLIR resources**: Ensure proper cleanup of MLIR contexts

This documentation covers the essential aspects of using PCP's MLIR-based tensor computation framework. For implementation details, refer to the source code and examples.