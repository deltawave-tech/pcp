const std = @import("std");
const pcp = @import("pcp");
const mlir_ctx = pcp.mlir_ctx;
const mlir = pcp.mlir;
const backend_selection = pcp.backend_selection;
const IreeBackend = pcp.backends.iree.IreeBackend;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🌚 CUDA IREE-based MLIR -> GPU Pipeline Test\n", .{});
    std.debug.print("==============================================\n", .{});

    const input_a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input_b_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const expected_output_data = [_]f32{ 19.0, 22.0, 43.0, 50.0 };

    var mlir_context = try mlir_ctx.MLIRContext.init(allocator);
    defer mlir_context.deinit();

    const stablehlo_module_str =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
        \\    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    return %0 : tensor<2x2xf32>
        \\  }
        \\}
    ;
    std.debug.print("✓ Created StableHLO module\n", .{});

    // Select the CUDA backend for compilation and execution
    const target_backend = backend_selection.Backend.cuda;
    std.debug.print("Compiling MLIR to VMFB for backend: {s}...\n", .{target_backend.toIreeCompilationTarget()});

    const vmfb_binary = try mlir_context.compileToVMFB(allocator, stablehlo_module_str, target_backend.toIreeCompilationTarget());
    defer allocator.free(vmfb_binary);
    std.debug.print("✓ Compiled to VMFB artifact ({} bytes)\n", .{vmfb_binary.len});

    var iree_backend = try IreeBackend.init(allocator, target_backend);
    defer iree_backend.deinit();
    std.debug.print("✓ Initialized IREE runtime for {s}\n", .{target_backend.toString()});

    var inputs_data = [_][]const u8{
        std.mem.sliceAsBytes(&input_a_data),
        std.mem.sliceAsBytes(&input_b_data),
    };
    var input_shapes = [_][]const i64{
        &[_]i64{ 2, 2 },
        &[_]i64{ 2, 2 },
    };

    std.debug.print("👻 Dispatching VMFB to IREE runtime for execution...\n", .{});
    const outputs = try iree_backend.execute(vmfb_binary, "main", &inputs_data, &input_shapes);
    defer {
        for (outputs) |o| allocator.free(o);
        allocator.free(outputs);
    }
    std.debug.print("🌙 IREE execution finished.\n", .{});

    const cuda_result_slice: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, outputs[0]));

    std.debug.print("Verifying IREE output against expected result...\n", .{});
    std.debug.print("   IREE Result: {any}\n", .{cuda_result_slice});
    std.debug.print("   Expected:    {any}\n", .{expected_output_data});

    const tolerance = 1e-6;
    for (cuda_result_slice, expected_output_data) |cuda_val, expected_val| {
        if (@abs(cuda_val - expected_val) > tolerance) {
            std.debug.print("💣 IREE verification failed! Computed: {}, Expected: {}\n", .{ cuda_val, expected_val });
            return error.CudaVerificationFailed;
        }
    }
    std.debug.print("🌙 Verification successful! The result from the IREE runtime is correct.\n", .{});
    std.debug.print("\n🌚 CUDA IREE pipeline test completed successfully!\n", .{});
}
