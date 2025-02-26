const std = @import("std");
const tensor = @import("tensor.zig");
const ops = @import("ops.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const DType = tensor.DType;
const Shape = tensor.Shape;
const BackendType = tensor.BackendType;

/// Operation type for the computational graph
pub const OpType = enum {
    add,
    subtract,
    multiply,
    divide,
    matmul,
    relu,
    softmax,
    transpose,
    // More operations will be added
};

/// A node in the computational graph
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
    
    pub fn init(allocator: Allocator, tensor_val: Tensor, requires_grad: bool) !*Node {
        // Create the node
        const node = try allocator.create(Node);
        node.* = Node{
            .tensor = tensor_val, // With ref counting, this is now safe
            .inputs = std.ArrayList(*Node).init(allocator),
            .requires_grad = requires_grad,
            .allocator = allocator,
        };
        return node;
    }
    
    pub fn deinit(self: *Node) void {
        // Clean up inputs array
        self.inputs.deinit();
        
        // Clean up gradient if it exists
        if (self.grad) |*g| {
            g.deinit();
        }
        
        // Release the tensor (decrement ref count)
        // With reference counting, we don't need to worry about who "owns" the tensor
        // It will be freed when all references are gone
        self.tensor.deinit();
        
        // Finally, deallocate the Node itself
        self.allocator.destroy(self);
    }
    
    /// Create a gradient tensor with the same shape as this node's tensor
    pub fn initGrad(self: *Node) !void {
        if (self.grad == null and self.requires_grad) {
            self.grad = try Tensor.zeros(
                self.allocator, 
                self.tensor.shape.dims, 
                self.tensor.dtype, 
                self.tensor.backend
            );
        }
    }
    
    /// Initialize gradient with ones (for loss nodes)
    pub fn initGradOnes(self: *Node) !void {
        if (self.grad == null and self.requires_grad) {
            self.grad = try Tensor.filled(
                self.allocator, 
                self.tensor.shape.dims, 
                self.tensor.dtype, 
                1.0,
                self.tensor.backend
            );
        }
    }
};

/// Create a computational graph node from a tensor
/// With reference counting, this function doesn't "take ownership" in the same way.
/// It simply uses the tensor, and the tensor will be freed when all references to it are gone.
/// The node will increment the reference count of the tensor.
pub fn variable(allocator: Allocator, t: Tensor, requires_grad: bool) !*Node {
    // The tensor's reference count is already 1 from its creation
    // We don't need to explicitly increment it when creating a node
    return Node.init(allocator, t, requires_grad);
}

/// Add two nodes element-wise
pub fn add(allocator: Allocator, a: *Node, b: *Node) !*Node {
    // Perform the operation
    var result_tensor = try ops.add(allocator, a.tensor, b.tensor);
    result_tensor.requires_grad = a.requires_grad or b.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .add;
    
    // Save input nodes for backward pass
    try result.inputs.append(a);
    try result.inputs.append(b);
    
    return result;
}

/// Subtract two nodes element-wise
pub fn subtract(allocator: Allocator, a: *Node, b: *Node) !*Node {
    // Perform the operation
    var result_tensor = try ops.subtract(allocator, a.tensor, b.tensor);
    result_tensor.requires_grad = a.requires_grad or b.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .subtract;
    
    // Save input nodes for backward pass
    try result.inputs.append(a);
    try result.inputs.append(b);
    
    return result;
}

/// Multiply two nodes element-wise
pub fn multiply(allocator: Allocator, a: *Node, b: *Node) !*Node {
    // Perform the operation
    var result_tensor = try ops.multiply(allocator, a.tensor, b.tensor);
    result_tensor.requires_grad = a.requires_grad or b.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .multiply;
    
    // Save input nodes for backward pass
    try result.inputs.append(a);
    try result.inputs.append(b);
    
    return result;
}

/// Matrix multiplication of two nodes
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

/// Apply ReLU activation to a node
pub fn relu(allocator: Allocator, a: *Node) !*Node {
    // Perform the operation
    var result_tensor = try ops.relu(allocator, a.tensor);
    result_tensor.requires_grad = a.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .relu;
    
    // Save input node for backward pass
    try result.inputs.append(a);
    
    return result;
}

/// Apply softmax activation to a node
pub fn softmax(allocator: Allocator, a: *Node) !*Node {
    // Perform the operation
    var result_tensor = try ops.softmax(allocator, a.tensor);
    result_tensor.requires_grad = a.requires_grad;
    
    // Create the result node
    var result = try Node.init(allocator, result_tensor, result_tensor.requires_grad);
    result.op_type = .softmax;
    
    // Save input node for backward pass
    try result.inputs.append(a);
    
    return result;
}

/// Compute gradients through the computational graph
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
                .relu => try backwardRelu(allocator, current),
                .softmax => try backwardSoftmax(allocator, current),
                // Add more operation types as they are implemented
                else => return error.UnsupportedOperationBackward,
            }
        }
    }
}

/// Helper function to build a topological sort
fn buildTopoSort(
    node: *Node, 
    visited: *std.AutoHashMap(*Node, void), 
    topo_order: *std.ArrayList(*Node)
) !void {
    // Check if node was already visited
    if (visited.contains(node)) return;
    
    // Mark as visited
    try visited.put(node, {});
    
    // Visit all inputs first
    for (node.inputs.items) |input| {
        try buildTopoSort(input, visited, topo_order);
    }
    
    // Add current node to the order
    try topo_order.append(node);
}

/// Backward pass for addition
fn backwardAdd(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    const b = node.inputs.items[1];
    
    if (node.grad) |grad| {
        // dL/da = dL/dc * dc/da = dL/dc * 1
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // If we already have gradients, accumulate
                const temp = try ops.add(allocator, a_grad.*, grad);
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
        
        // dL/db = dL/dc * dc/db = dL/dc * 1
        if (b.requires_grad) {
            try b.initGrad();
            if (b.grad) |*b_grad| {
                // If we already have gradients, accumulate
                const temp = try ops.add(allocator, b_grad.*, grad);
                b_grad.deinit();
                b_grad.* = temp;
            }
        }
    }
}

/// Backward pass for subtraction
fn backwardSubtract(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    const b = node.inputs.items[1];
    
    if (node.grad) |grad| {
        // dL/da = dL/dc * dc/da = dL/dc * 1
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // If we already have gradients, accumulate
                const temp = try ops.add(allocator, a_grad.*, grad);
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
        
        // dL/db = dL/dc * dc/db = dL/dc * (-1)
        if (b.requires_grad) {
            try b.initGrad();
            if (b.grad) |*b_grad| {
                // For subtraction, the gradient for b is negated
                // Create the constant tensor for -1.0
                const negative_one = try Tensor.filled(
                    allocator,
                    grad.shape.dims,
                    grad.dtype,
                    -1.0,
                    grad.backend
                );
                
                // Negate the gradient
                const neg_grad = try ops.multiply(allocator, grad, negative_one);
                // Clean up the temporary tensor
                negative_one.deinit();
                
                // Accumulate gradients
                const temp = try ops.add(allocator, b_grad.*, neg_grad);
                // Clean up the negated gradient tensor
                neg_grad.deinit();
                
                b_grad.deinit();
                b_grad.* = temp;
            }
        }
    }
}

/// Backward pass for element-wise multiplication
fn backwardMultiply(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    const b = node.inputs.items[1];
    
    if (node.grad) |grad| {
        // dL/da = dL/dc * dc/da = dL/dc * b
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                const temp_grad = try ops.multiply(allocator, grad, b.tensor);
                const temp = try ops.add(allocator, a_grad.*, temp_grad);
                // Clean up the temporary gradient
                temp_grad.deinit();
                
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
        
        // dL/db = dL/dc * dc/db = dL/dc * a
        if (b.requires_grad) {
            try b.initGrad();
            if (b.grad) |*b_grad| {
                const temp_grad = try ops.multiply(allocator, grad, a.tensor);
                const temp = try ops.add(allocator, b_grad.*, temp_grad);
                // Clean up the temporary gradient
                temp_grad.deinit();
                
                b_grad.deinit();
                b_grad.* = temp;
            }
        }
    }
}

/// Backward pass for matrix multiplication
fn backwardMatmul(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 2) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    const b = node.inputs.items[1];
    
    if (node.grad) |grad| {
        // dL/da = dL/dc * dc/da = dL/dc * b^T
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // Compute b transpose - with reference counting, this creates a new tensor
                const b_transpose = try ops.transpose(allocator, b.tensor);
                
                // Compute gradient contribution - with reference counting, this creates a new tensor
                const temp_grad = try ops.matmul(allocator, grad, b_transpose);
                
                // Release the transpose tensor (decrement ref count)
                b_transpose.deinit();
                
                // Accumulate gradient - with reference counting, this creates a new tensor
                const temp = try ops.add(allocator, a_grad.*, temp_grad);
                
                // Release the temporary gradient (decrement ref count)
                temp_grad.deinit();
                
                // Replace the old gradient with the new one
                // Old gradient's ref count is decremented
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
        
        // dL/db = dL/dc * dc/db = a^T * dL/dc
        if (b.requires_grad) {
            try b.initGrad();
            if (b.grad) |*b_grad| {
                // Compute a transpose - with reference counting, this creates a new tensor
                const a_transpose = try ops.transpose(allocator, a.tensor);
                
                // Compute gradient contribution - with reference counting, this creates a new tensor
                const temp_grad = try ops.matmul(allocator, a_transpose, grad);
                
                // Release the transpose tensor (decrement ref count)
                a_transpose.deinit();
                
                // Accumulate gradient - with reference counting, this creates a new tensor
                const temp = try ops.add(allocator, b_grad.*, temp_grad);
                
                // Release the temporary gradient (decrement ref count)
                temp_grad.deinit();
                
                // Replace the old gradient with the new one
                // Old gradient's ref count is decremented
                b_grad.deinit();
                b_grad.* = temp;
            }
        }
    }
}

/// Backward pass for ReLU
fn backwardRelu(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 1) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    
    if (node.grad) |grad| {
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // Create ReLU derivative mask: 1 where input > 0, 0 elsewhere
                const mask = try Tensor.zeros(allocator, a.tensor.shape.dims, a.tensor.dtype, a.tensor.backend);
                
                const input_buf = @as([*]f32, @ptrCast(@alignCast(a.tensor.buffer.data.ptr)))[0..a.tensor.shape.elemCount()];
                const mask_buf = @as([*]f32, @ptrCast(@alignCast(mask.buffer.data.ptr)))[0..mask.shape.elemCount()];
                
                for (input_buf, 0..) |val, i| {
                    mask_buf[i] = if (val > 0) 1.0 else 0.0;
                }
                
                // Gradient is only passed where input was positive
                const temp_grad = try ops.multiply(allocator, grad, mask);
                // Clean up the mask tensor
                mask.deinit();
                
                const temp = try ops.add(allocator, a_grad.*, temp_grad);
                // Clean up the temporary gradient
                temp_grad.deinit();
                
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
    }
}

/// Backward pass for softmax
fn backwardSoftmax(allocator: Allocator, node: *Node) !void {
    if (node.inputs.items.len != 1) return error.InvalidInputCount;
    
    const a = node.inputs.items[0];
    
    if (node.grad) |grad| {
        if (a.requires_grad) {
            try a.initGrad();
            if (a.grad) |*a_grad| {
                // Softmax gradient is more complex
                // For a batch of vectors, we need to compute for each row:
                // dL/dx_i = Σ_j (dL/dy_j * dy_j/dx_i)
                // For softmax: dy_j/dx_i = y_j * (δ_ij - y_i)
                
                const batch_size = node.tensor.shape.dims[0];
                const feature_size = node.tensor.shape.dims[1];
                
                // Create a new tensor for the gradients
                const new_grad = try Tensor.zeros(allocator, a.tensor.shape.dims, a.tensor.dtype, a.tensor.backend);
                
                const softmax_out = @as([*]f32, @ptrCast(@alignCast(node.tensor.buffer.data.ptr)));
                const upstream_grad = @as([*]f32, @ptrCast(@alignCast(grad.buffer.data.ptr)));
                const new_grad_buf = @as([*]f32, @ptrCast(@alignCast(new_grad.buffer.data.ptr)));
                
                // For each batch item
                for (0..batch_size) |b_idx| {
                    // Calculate jacobian vector product
                    for (0..feature_size) |i| {
                        var sum: f32 = 0.0;
                        
                        for (0..feature_size) |j| {
                            const s_i = softmax_out[b_idx * feature_size + i];
                            const s_j = softmax_out[b_idx * feature_size + j];
                            const g = upstream_grad[b_idx * feature_size + j];
                            
                            // dy_j/dx_i = y_j * (δ_ij - y_i)
                            const delta_ij: f32 = if (i == j) 1.0 else 0.0;
                            sum += g * s_j * (delta_ij - s_i);
                        }
                        
                        new_grad_buf[b_idx * feature_size + i] = sum;
                    }
                }
                
                // Add to existing gradient
                const temp = try ops.add(allocator, a_grad.*, new_grad);
                // Clean up the new_grad tensor after use
                new_grad.deinit();
                
                a_grad.deinit();
                a_grad.* = temp;
            }
        }
    }
}

test "variable creation" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create a tensor that we'll copy for the variable
    var t = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    
    try t.setScalar(&[_]usize{0, 0}, 1.0);
    try t.setScalar(&[_]usize{0, 1}, 2.0);
    try t.setScalar(&[_]usize{1, 0}, 3.0);
    try t.setScalar(&[_]usize{1, 1}, 4.0);
    
    // Create a copy for the variable since variable() takes ownership
    const t_copy = try Tensor.init(allocator, t.shape.dims, t.dtype, t.backend);
    
    // Copy the data
    @memcpy(t_copy.buffer.data, t.buffer.data[0..t.buffer.data.len]);
    
    // Now we can create the variable with our copy and keep the original
    var node = try variable(allocator, t_copy, true);
    defer node.deinit();
    
    // Now we can safely use t
    defer t.deinit();
    
    try std.testing.expect(node.requires_grad);
    try std.testing.expectEqual(@as(usize, 2), node.tensor.shape.rank());
    try std.testing.expectEqual(@as(f32, 1.0), try node.tensor.getScalar(&[_]usize{0, 0}));
}

test "simple addition and backward" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create tensors
    var a_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a_tensor.setScalar(&[_]usize{0, 0}, 1.0);
    try a_tensor.setScalar(&[_]usize{0, 1}, 2.0);
    try a_tensor.setScalar(&[_]usize{1, 0}, 3.0);
    try a_tensor.setScalar(&[_]usize{1, 1}, 4.0);
    
    var b_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try b_tensor.setScalar(&[_]usize{0, 0}, 5.0);
    try b_tensor.setScalar(&[_]usize{0, 1}, 6.0);
    try b_tensor.setScalar(&[_]usize{1, 0}, 7.0);
    try b_tensor.setScalar(&[_]usize{1, 1}, 8.0);
    
    // Create copies of tensors for the variables
    const a_tensor_copy = try Tensor.init(allocator, a_tensor.shape.dims, a_tensor.dtype, a_tensor.backend);
    @memcpy(a_tensor_copy.buffer.data, a_tensor.buffer.data[0..a_tensor.buffer.data.len]);
    
    const b_tensor_copy = try Tensor.init(allocator, b_tensor.shape.dims, b_tensor.dtype, b_tensor.backend);
    @memcpy(b_tensor_copy.buffer.data, b_tensor.buffer.data[0..b_tensor.buffer.data.len]);
    
    // Create nodes with the copies
    var a = try variable(allocator, a_tensor_copy, true);
    var b = try variable(allocator, b_tensor_copy, true);
    
    // Now we can safely use the original tensors
    defer a_tensor.deinit();
    defer b_tensor.deinit();
    
    // Forward pass: c = a + b
    var c = try add(allocator, a, b);
    
    // Check result: c = [[6, 8], [10, 12]]
    try std.testing.expectEqual(@as(f32, 6.0), try c.tensor.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 8.0), try c.tensor.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 10.0), try c.tensor.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 12.0), try c.tensor.getScalar(&[_]usize{1, 1}));
    
    // Backward pass
    try backward(allocator, c);
    
    // For addition, gradients are 1s
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{1, 1}));
    
    try std.testing.expectEqual(@as(f32, 1.0), try b.grad.?.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try b.grad.?.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), try b.grad.?.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try b.grad.?.getScalar(&[_]usize{1, 1}));
    
    // Cleanup
    // The nodes will clean up their tensors (a_tensor_copy, b_tensor_copy)
    // and the gradients when they are deinited
    a.deinit();
    b.deinit();
    c.deinit();
}

test "multiplication and backward" {
    const allocator = std.testing.allocator;
    var dims = [_]usize{ 2, 2 };
    
    // Create tensors
    var a_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try a_tensor.setScalar(&[_]usize{0, 0}, 2.0);
    try a_tensor.setScalar(&[_]usize{0, 1}, 3.0);
    try a_tensor.setScalar(&[_]usize{1, 0}, 4.0);
    try a_tensor.setScalar(&[_]usize{1, 1}, 5.0);
    
    var b_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
    try b_tensor.setScalar(&[_]usize{0, 0}, 0.5);
    try b_tensor.setScalar(&[_]usize{0, 1}, 1.0);
    try b_tensor.setScalar(&[_]usize{1, 0}, 1.5);
    try b_tensor.setScalar(&[_]usize{1, 1}, 2.0);
    
    // Create copies of tensors for the variables
    const a_tensor_copy = try Tensor.init(allocator, a_tensor.shape.dims, a_tensor.dtype, a_tensor.backend);
    @memcpy(a_tensor_copy.buffer.data, a_tensor.buffer.data[0..a_tensor.buffer.data.len]);
    
    const b_tensor_copy = try Tensor.init(allocator, b_tensor.shape.dims, b_tensor.dtype, b_tensor.backend);
    @memcpy(b_tensor_copy.buffer.data, b_tensor.buffer.data[0..b_tensor.buffer.data.len]);
    
    // Create nodes with the copies
    var a = try variable(allocator, a_tensor_copy, true);
    var b = try variable(allocator, b_tensor_copy, true);
    
    // Now we can safely use the original tensors
    defer a_tensor.deinit();
    defer b_tensor.deinit();
    
    // Forward pass: c = a * b (element-wise)
    var c = try multiply(allocator, a, b);
    
    // Check result: c = [[1, 3], [6, 10]]
    try std.testing.expectEqual(@as(f32, 1.0), try c.tensor.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 3.0), try c.tensor.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 6.0), try c.tensor.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 10.0), try c.tensor.getScalar(&[_]usize{1, 1}));
    
    // Backward pass
    try backward(allocator, c);
    
    // For element-wise multiplication, dL/da = dL/dc * b
    try std.testing.expectEqual(@as(f32, 0.5), try a.grad.?.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), try a.grad.?.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.5), try a.grad.?.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 2.0), try a.grad.?.getScalar(&[_]usize{1, 1}));
    
    // And dL/db = dL/dc * a
    try std.testing.expectEqual(@as(f32, 2.0), try b.grad.?.getScalar(&[_]usize{0, 0}));
    try std.testing.expectEqual(@as(f32, 3.0), try b.grad.?.getScalar(&[_]usize{0, 1}));
    try std.testing.expectEqual(@as(f32, 4.0), try b.grad.?.getScalar(&[_]usize{1, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), try b.grad.?.getScalar(&[_]usize{1, 1}));
    
    // Cleanup nodes (which will also clean up their tensors)
    a.deinit();
    b.deinit();
    c.deinit();
}