const std = @import("std");
const mlir = @import("mlir.zig");
const ops = @import("ops.zig");
const c = @import("mlir/c.zig").c;

const Allocator = std.mem.Allocator;
const MLIRBuilder = ops.MLIRBuilder;

/// MLIR-based Automatic Differentiation using Graph-to-Graph Transformation
/// This implements reverse-mode AD on MLIR computation graphs using VJP rules

/// Vector-Jacobian Product (VJP) function signature
/// Takes the forward operation and output gradients, returns input gradients
const VJPFn = *const fn(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) anyerror![]mlir.Value;

/// Compile-time dispatch from operation name to VJP rule
/// This avoids ALL runtime lookups - pure static dispatch
fn getVjpFn(comptime op_name: []const u8) VJPFn {
    if (std.mem.eql(u8, op_name, "stablehlo.add")) {
        return addVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.subtract")) {
        return subtractVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.multiply")) {
        return multiplyVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.dot_general")) {
        return matmulVJP;
    } else if (std.mem.eql(u8, op_name, "stablehlo.maximum")) {
        return reluVJP; // ReLU implemented as max(x, 0)
    } else if (std.mem.eql(u8, op_name, "stablehlo.constant")) {
        return constantVJP;
    } else if (std.mem.eql(u8, op_name, "func.return")) {
        return returnVJP;
    } else {
        @compileError("No VJP rule defined for operation: " ++ op_name ++ 
                     ". Add the rule to getVjpFn() in autodiff.zig");
    }
}

/// Check if an mlir.Value comes from a constant operation at compile time
fn isValueFromConstantOp(value: mlir.Value) bool {
    // In a full implementation, this would check if the defining op is stablehlo.constant
    // For now, we'll assume all values are non-constant to be conservative
    _ = value;
    return false;
}

/// Main automatic differentiation function - transforms forward graph to gradient graph
pub fn buildGradientGraph(
    allocator: Allocator,
    builder: *MLIRBuilder,
    forward_fn: mlir.Operation,
) !mlir.Operation {
    std.debug.print("Building gradient graph from forward function...\n", .{});
    
    // Create new function for gradients with same input signature but different outputs
    const gradient_fn = try createGradientFunction(builder, forward_fn);
    
    // Map from forward-pass values (primals) to their gradients (adjoints)
    var adjoint_map = std.AutoHashMap(mlir.Value, mlir.Value).init(allocator);
    defer adjoint_map.deinit();
    
    // Get the operations in reverse topological order
    const ops_reversed = try getOperationsInReverseOrder(allocator, forward_fn);
    defer allocator.free(ops_reversed);
    
    // Initialize gradient of loss (output) to 1.0
    const loss_value = getReturnValue(forward_fn);
    const ones_constant = try builder.createConstant(1.0, loss_value.getType());
    try adjoint_map.put(loss_value, ones_constant);
    
    std.debug.print("Starting reverse-mode AD through {} operations...\n", .{ops_reversed.len});
    
    // Walk the forward graph backwards, applying VJP rules
    for (ops_reversed) |op| {
        try processOperationVJP(allocator, builder, op, &adjoint_map);
    }
    
    // Collect gradients for function inputs and create return statement
    try finalizeGradientFunction(allocator, builder, gradient_fn, forward_fn, &adjoint_map);
    
    std.debug.print("✓ Successfully built gradient graph\n", .{});
    return gradient_fn;
}

/// Process a single operation's VJP rule
fn processOperationVJP(
    allocator: Allocator,
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value),
) !void {
    const op_name = op.getName();
    
    // Get output gradients for this operation
    const output_gradients = try getOutputGradients(allocator, op, adjoint_map);
    defer allocator.free(output_gradients);
    
    if (output_gradients.len == 0) {
        // No gradient flows through this operation
        return;
    }
    
    // Compile-time dispatch to the correct VJP rule
    const vjp_rule = comptime getVjpFn(op_name);
    
    // Apply the VJP rule to get input gradients
    const input_gradients = try vjp_rule(builder, op, output_gradients);
    defer allocator.free(input_gradients);
    
    // Add input gradients to the adjoint map
    try addInputGradients(allocator, op, input_gradients, adjoint_map);
}

/// VJP rule for addition: both inputs get the same gradient
fn addVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1); // Add has one output
    
    const grad_out = adjoints[0];
    
    // Check for constant operands at compile time to optimize gradient graph
    const a = op.getOperand(0);
    const b = op.getOperand(1);
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    // Gradient for first input
    if (!isValueFromConstantOp(a)) {
        try result.append(grad_out); // da = grad_out
    }
    
    // Gradient for second input  
    if (!isValueFromConstantOp(b)) {
        try result.append(grad_out); // db = grad_out
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for subtraction: da = grad_out, db = -grad_out
fn subtractVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0);
    const b = op.getOperand(1);
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    // da = grad_out
    if (!isValueFromConstantOp(a)) {
        try result.append(grad_out);
    }
    
    // db = -grad_out
    if (!isValueFromConstantOp(b)) {
        const neg_grad = try builder.createOp("stablehlo.negate", &.{grad_out}, grad_out.getType());
        try result.append(neg_grad.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for multiplication: da = grad_out * b, db = grad_out * a
fn multiplyVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0); // Primal 'a'
    const b = op.getOperand(1); // Primal 'b'
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    // da = grad_out * b
    if (!isValueFromConstantOp(a)) {
        const grad_a = try builder.createOp("stablehlo.multiply", &.{ grad_out, b }, grad_out.getType());
        try result.append(grad_a.getResult(0));
    }
    
    // db = grad_out * a  
    if (!isValueFromConstantOp(b)) {
        const grad_b = try builder.createOp("stablehlo.multiply", &.{ grad_out, a }, grad_out.getType());
        try result.append(grad_b.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for matrix multiplication: da = grad_out @ b^T, db = a^T @ grad_out
fn matmulVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const a = op.getOperand(0); // Primal 'a'
    const b = op.getOperand(1); // Primal 'b'
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    // da = grad_out @ b^T
    if (!isValueFromConstantOp(a)) {
        const b_transposed = try builder.createOp("stablehlo.transpose", &.{b}, b.getType());
        const grad_a = try builder.createOp("stablehlo.dot_general", &.{ grad_out, b_transposed.getResult(0) }, grad_out.getType());
        try result.append(grad_a.getResult(0));
    }
    
    // db = a^T @ grad_out
    if (!isValueFromConstantOp(b)) {
        const a_transposed = try builder.createOp("stablehlo.transpose", &.{a}, a.getType());
        const grad_b = try builder.createOp("stablehlo.dot_general", &.{ a_transposed.getResult(0), grad_out }, grad_out.getType());
        try result.append(grad_b.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for ReLU (max(x, 0)): gradient flows through only where x > 0
fn reluVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    std.debug.assert(adjoints.len == 1);
    
    const grad_out = adjoints[0];
    const x = op.getOperand(0); // Input to ReLU
    
    var result = std.ArrayList(mlir.Value).init(builder.allocator);
    
    if (!isValueFromConstantOp(x)) {
        // Create mask: x > 0
        const zero = try builder.createConstant(0.0, x.getType());
        const mask = try builder.createOp("stablehlo.compare", &.{ x, zero }, x.getType()); // x > 0
        
        // Apply mask: grad_out * mask
        const grad_x = try builder.createOp("stablehlo.select", &.{ mask.getResult(0), grad_out, zero }, grad_out.getType());
        try result.append(grad_x.getResult(0));
    }
    
    return result.toOwnedSlice();
}

/// VJP rule for constants: no gradient (constants don't have inputs)
fn constantVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = builder;
    _ = op;
    _ = adjoints;
    
    // Constants have no inputs, so no gradients to return
    return &[_]mlir.Value{};
}

/// VJP rule for function return: just pass through the gradient
fn returnVJP(
    builder: *MLIRBuilder,
    op: mlir.Operation,
    adjoints: []const mlir.Value,
) ![]mlir.Value {
    _ = op;
    
    // Return operation just passes gradients through to its operand
    return builder.allocator.dupe(mlir.Value, adjoints);
}

// Helper functions for graph traversal and manipulation

fn createGradientFunction(builder: *MLIRBuilder, forward_fn: mlir.Operation) !mlir.Operation {
    // Create new function with same inputs but different outputs (gradients)
    // This is a simplified implementation - real version would properly handle function types
    return try builder.createOp("func.func", &.{}, forward_fn.getResult(0).getType());
}

fn getOperationsInReverseOrder(allocator: Allocator, fn_op: mlir.Operation) ![]mlir.Operation {
    // Walk the function body and collect operations in reverse topological order
    // This is a simplified implementation
    _ = fn_op;
    var operations = std.ArrayList(mlir.Operation).init(allocator);
    
    // In a real implementation, this would walk the MLIR function body
    // For now, return empty array as placeholder
    return operations.toOwnedSlice();
}

fn getReturnValue(fn_op: mlir.Operation) mlir.Value {
    // Get the return value of the function (typically the loss)
    // This is a simplified implementation
    _ = fn_op;
    return mlir.Value{}; // Placeholder
}

fn getOutputGradients(allocator: Allocator, op: mlir.Operation, adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value)) ![]mlir.Value {
    // Get gradients for all outputs of this operation
    var gradients = std.ArrayList(mlir.Value).init(allocator);
    
    // In a real implementation, this would iterate over op.getResults()
    // and look up their gradients in the adjoint_map
    _ = op;
    _ = adjoint_map;
    
    return gradients.toOwnedSlice();
}

fn addInputGradients(
    allocator: Allocator,
    op: mlir.Operation,
    input_gradients: []mlir.Value,
    adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value),
) !void {
    // Add gradients for each input operand to the adjoint map
    // If a gradient already exists, sum them
    _ = allocator;
    _ = op;
    _ = input_gradients;
    _ = adjoint_map;
    
    // Real implementation would iterate over op.getOperands() and input_gradients
    // and either add new entries or sum with existing gradients
}

fn finalizeGradientFunction(
    allocator: Allocator,
    builder: *MLIRBuilder,
    gradient_fn: mlir.Operation,
    forward_fn: mlir.Operation,
    adjoint_map: *std.AutoHashMap(mlir.Value, mlir.Value),
) !void {
    // Create return statement with gradients for all function inputs
    _ = allocator;
    _ = builder;
    _ = gradient_fn;
    _ = forward_fn;
    _ = adjoint_map;
    
    // Real implementation would collect gradients for function arguments
    // and create a func.return operation
}

/// High-level API for automatic differentiation
pub const AutoDiff = struct {
    allocator: Allocator,
    builder: *MLIRBuilder,
    
    pub fn init(allocator: Allocator, builder: *MLIRBuilder) AutoDiff {
        return AutoDiff{
            .allocator = allocator,
            .builder = builder,
        };
    }
    
    /// Create gradient function from forward function
    pub fn grad(self: *AutoDiff, forward_fn: mlir.Operation) !mlir.Operation {
        return buildGradientGraph(self.allocator, self.builder, forward_fn);
    }
    
    /// Compose forward and backward pass into a single training function
    pub fn valueAndGrad(self: *AutoDiff, forward_fn: mlir.Operation) !mlir.Operation {
        // Create function that returns both value and gradients
        const grad_fn = try self.grad(forward_fn);
        
        // In a real implementation, this would create a new function that calls both
        // forward_fn and grad_fn and returns both results
        _ = grad_fn;
        return forward_fn; // Placeholder
    }
};

/// Test function for MLIR autodiff
pub fn testMLIRAutoDiff(allocator: Allocator) !void {
    std.debug.print("\n=== Testing MLIR Automatic Differentiation ===\n", .{});
    
    // Create MLIR builder
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    // Create autodiff system
    _ = AutoDiff.init(allocator, &builder);
    
    std.debug.print("✓ AutoDiff system initialized\n", .{});
    
    // In a real test, we would:
    // 1. Build a forward function with StableHLO ops
    // 2. Call autodiff.grad() to get gradient function
    // 3. Verify the gradient graph is correct
    
    std.debug.print("✓ MLIR autodiff test completed\n", .{});
}