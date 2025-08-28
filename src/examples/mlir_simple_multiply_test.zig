const std = @import("std");
const pcp = @import("pcp");
const mlir = pcp.mlir;
const ops = pcp.ops;

const Allocator = std.mem.Allocator;
const MLIRBuilder = ops.MLIRBuilder;

/// Simple test that just builds and dumps a multiply operation
pub fn testSimpleMultiply(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Simple Multiply: f(x, y) = x * y ===\n", .{});
    
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();

    // Define function signature
    const f32_type = mlir.Type.f32Type(builder.ctx);
    const scalar_type = mlir.Type.rankedTensorType(builder.ctx, &.{1}, f32_type);

    // Create arguments
    const x = try builder.addBlockArgument(scalar_type);
    const y = try builder.addBlockArgument(scalar_type);

    std.debug.print("Created function arguments: x and y\n", .{});

    // Build the forward pass: x * y using ops.multiply
    const x_tensor = try builder.newTensor(x);
    const y_tensor = try builder.newTensor(y);
    const result_tensor = try ops.multiply(&builder, x_tensor, y_tensor);
    
    std.debug.print("Built forward pass: x * y\n", .{});

    // Create return operation
    const return_op = mlir.Operation.create(builder.ctx, "func.return", .{
        .operands = &.{result_tensor.value},
        .location = builder.loc,
    });
    
    // CRITICAL FIX: Use the builder's pre-established insertion block instead of accessing module regions directly
    const body_block = builder.insertion_block;
    body_block.appendOwnedOperation(return_op);

    std.debug.print("Added func.return to forward graph\n", .{});

    // Print the forward graph
    const forward_fn_op = builder.module.op();
    std.debug.print("--- Forward Graph ---\n", .{});
    forward_fn_op.dump();
    
    std.debug.print("✓ Simple multiply test completed successfully!\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try testSimpleMultiply(allocator);
}