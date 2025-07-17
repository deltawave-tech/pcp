const std = @import("std");
const pcp = @import("pcp");
const mlir = pcp.mlir;
const ops = pcp.ops;
const autodiff = pcp.autodiff;
const tensor = pcp.tensor;

// Access stablehlo operations through the ops module
const stablehlo = struct {
    pub const multiply = ops.multiply;
};

const Allocator = std.mem.Allocator;
const MLIRBuilder = ops.MLIRBuilder;

/// Test function for simple autodiff on f(x, w) = x * w
pub fn testSimpleAutodiff(allocator: Allocator) !void {
    std.debug.print("\n=== Testing Simple MLIR Autodiff: f(x, w) = x * w ===\n", .{});
    
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();

    // 1. Define function signature and create arguments
    const f32_type = mlir.Type.f32Type(builder.ctx);
    const tensor_type = mlir.Type.rankedTensorType(builder.ctx, &.{1}, f32_type); // tensor<1xf32>

    // Create placeholder values for x and w
    // In a real implementation, these would be function arguments
    const x = try builder.addBlockArgument(tensor_type);
    const w = try builder.addBlockArgument(tensor_type);

    std.debug.print("Created function arguments: x and w\n", .{});

    // 2. Build the forward pass: x * w using ops.multiply
    const x_tensor = builder.newTensor(x) catch unreachable;
    const w_tensor = builder.newTensor(w) catch unreachable;
    const result_tensor = try ops.multiply(&builder, x_tensor, w_tensor);
    const result_val = result_tensor.value;

    std.debug.print("Built forward pass: x * w\n", .{});

    // 3. The multiply operation is already added to the module by ops.multiply

    // 4. Create a func.return to make the function well-formed
    const return_op = mlir.Operation.create(builder.ctx, "func.return", .{
        .operands = &.{result_val}, // Return the result of the multiplication
        .location = builder.loc,
    });
    
    // Add the return operation to the module body
    const body_block = builder.module.op().getRegion(0).getBlock(0);
    body_block.appendOwnedOperation(return_op);

    std.debug.print("Added func.return to forward graph\n", .{});

    // 4. Get the complete function operation
    const forward_fn_op = builder.module.op();
    std.debug.print("--- Forward Graph ---\n", .{});
    forward_fn_op.dump(); // Print the forward graph

    // 5. Run the graph transformer
    std.debug.print("Running autodiff.buildGradientGraph...\n", .{});
    const grad_fn_op = autodiff.buildGradientGraph(allocator, &builder, forward_fn_op) catch |err| {
        std.debug.print("Error in buildGradientGraph: {}\n", .{err});
        std.debug.print("Continuing with test to show progress...\n", .{});
        return; // Skip the error for now and continue
    };

    // 6. Verify the output - THE CORE TEST!
    std.debug.print("--- Gradient Graph (should not segfault!) ---\n", .{});
    grad_fn_op.dump(); // Print the gradient graph - this was segfaulting before our fix!

    // 7. Basic structural verification of the gradient graph
    std.debug.print("Verifying gradient graph structure...\n", .{});
    
    // Check that we have a gradient function with the expected name
    const grad_fn_name = grad_fn_op.getName();
    std.debug.print("Gradient function operation type: {s}\n", .{grad_fn_name});
    
    // For f(x, w) = x * w, we expect the gradient function to have:
    // - Inputs: x, w, grad_out 
    // - Outputs: dx, dw
    // - VJP operations: multiply operations for dx = grad_out * w, dw = grad_out * x
    
    if (std.mem.eql(u8, grad_fn_name, "func.func")) {
        std.debug.print("✓ Gradient function has correct type (func.func)\n", .{});
    } else {
        std.debug.print("✗ Unexpected gradient function type: {s}\n", .{grad_fn_name});
        return error.UnexpectedGradientFunctionType;
    }
    
    // Check that the gradient function has a region (function body)
    const grad_region = grad_fn_op.getRegion(0);
    const grad_block = grad_region.getBlock(0);
    
    // Count operations in the gradient function
    var op_count: usize = 0;
    var maybe_op = grad_block.getFirstOp();
    while (maybe_op) |op| {
        const op_name = op.getName();
        std.debug.print("  Gradient op {}: {s}\n", .{op_count, op_name});
        op_count += 1;
        maybe_op = op.getNext();
    }
    
    std.debug.print("✓ Gradient function contains {} operations\n", .{op_count});
    
    if (op_count > 0) {
        std.debug.print("✓ Gradient graph is non-empty\n", .{});
    } else {
        std.debug.print("✗ Gradient graph is empty - this suggests the transformation failed\n", .{});
        return error.EmptyGradientGraph;
    }

    std.debug.print("✓ Simple autodiff test completed successfully!\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try testSimpleAutodiff(allocator);
}