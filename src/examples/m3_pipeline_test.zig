// In src/examples/m3_pipeline_test.zig

const std = @import("std");
const pcp = @import("pcp");
const mlir_ctx = pcp.mlir_ctx;
const mlir = pcp.mlir;
// Correct way to import files from the main 'pcp' module
const backend_selection = pcp.backend_selection; 
const IreeBackend = pcp.backends.iree.IreeBackend;

/// This test now verifies the complete IREE-based pipeline:
/// 1. Build a StableHLO MLIR module in Zig.
/// 2. Call `iree-compile` to compile it to a .vmfb artifact for the Metal backend.
/// 3. Initialize the IREE runtime backend in `iree.zig`.
/// 4. Execute the .vmfb artifact on the M3 GPU via the IREE runtime.
/// 5. Verify the numerical output is correct.
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🍎 M3 IREE-based MLIR -> GPU Pipeline Test\n", .{});
    std.debug.print("==================================================\n", .{});

    // --- Test Data ---
    const input_a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // 2x2 matrix
    const input_b_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 }; // 2x2 matrix
    const expected_output_data = [_]f32{ 19.0, 22.0, 43.0, 50.0 }; // Expected A @ B

    // 1. Initialize MLIR context
    var mlir_context = try mlir_ctx.MLIRContext.init(allocator);
    defer mlir_context.deinit();

    // 2. Create the MLIR module for a simple matrix multiplication
    const stablehlo_module_str =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
        \\    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    return %0 : tensor<2x2xf32>
        \\  }
        \\}
    ;
    const module = try mlir.Module.parse(mlir_context.getContext(), stablehlo_module_str);
    defer module.deinit();
    std.debug.print("✓ Created StableHLO module\n", .{});

    // 3. Compile MLIR to a VMFB artifact using IREE
    //    We will replace the body of 'lowerToSPIRV' later, for now we can adapt the call.
    //    Let's find the IREE target for the current OS.
    const target_backend = backend_selection.Backend.selectDefault();
    std.debug.print("Compiling MLIR to VMFB for backend: {s}...\n", .{target_backend.toIreeDriverName()});

    // The 'compileToVMFB' function (which you may have from a previous step) handles this.
    // If not, you would implement it in mlir_ctx.zig to call 'iree-compile'.
    const vmfb_binary = try mlir_context.compileToVMFB(allocator, module, target_backend.toIreeDriverName());
    defer allocator.free(vmfb_binary);
    std.debug.print("✓ Compiled to VMFB artifact ({} bytes)\n", .{vmfb_binary.len});

    // 4. Initialize the correct IREE runtime backend
    var iree_backend = try IreeBackend.init(allocator, target_backend);
    defer iree_backend.deinit();
    std.debug.print("✓ Initialized IREE runtime for {s}\n", .{target_backend.toString()});

    // 5. Prepare inputs and execute the VMFB
    var inputs_data = [_][]const u8{
        std.mem.sliceAsBytes(&input_a_data),
        std.mem.sliceAsBytes(&input_b_data),
    };
    // The IREE backend also needs the shapes for the inputs.
    var input_shapes = [_][]const i64{
        &[_]i64{ 2, 2 },
        &[_]i64{ 2, 2 },
    };

    std.debug.print("👻 Dispatching VMFB to IREE runtime for execution...\n", .{});
    const outputs = try iree_backend.execute(vmfb_binary, &inputs_data, &input_shapes);
    defer {
        for (outputs) |o| allocator.free(o);
        allocator.free(outputs);
    }
    std.debug.print("🌙 IREE execution finished.\n", .{});

    // 6. Verify the result from the GPU/CPU via IREE
    if (outputs.len != 1) return error.UnexpectedOutputCount;
    if (outputs[0].len != expected_output_data.len * @sizeOf(f32)) return error.IncorrectOutputSize;

    const gpu_result_slice: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, outputs[0]));

    std.debug.print("Verifying IREE output against expected result...\n", .{});
    std.debug.print("   IREE Result: {any}\n", .{gpu_result_slice});
    std.debug.print("   Expected:    {any}\n", .{expected_output_data});

    const tolerance = 1e-6;
    for (gpu_result_slice, expected_output_data) |gpu_val, expected_val| {
        if (@abs(gpu_val - expected_val) > tolerance) {
            std.debug.print("💣 IREE verification failed! Computed: {}, Expected: {}\n", .{ gpu_val, expected_val });
            return error.GPUVerificationFailed;
        }
    }
    std.debug.print("🌙 Verification successful! The result from the IREE runtime is correct.\n", .{});
    std.debug.print("\n🌚 M3 IREE pipeline test completed successfully!\n", .{});
}
