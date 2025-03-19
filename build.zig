const std = @import("std");
const builtin = @import("builtin");

const test_targets = [_]std.Target.Query{
    .{}, // native
    .{ .cpu_arch = .aarch64, .os_tag = .macos },
    .{ .cpu_arch = .aarch64, .os_tag = .linux },
    .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu },
    .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl },
    .{ .cpu_arch = .x86_64, .os_tag = .windows },
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the Metal bridge object file if on macOS
    var metal_bridge_lib: ?*std.Build.Step.Compile = null;
    if (builtin.os.tag == .macos) {
        metal_bridge_lib = b.addStaticLibrary(.{
            .name = "metal_bridge",
            .target = target,
            .optimize = optimize,
        });

        metal_bridge_lib.?.addCSourceFile(.{
            .file = b.path("src/backends/metal_bridge.m"),
            .flags = &[_][]const u8{"-fobjc-arc"},
        });

        // Add frameworks for macOS
        metal_bridge_lib.?.linkFramework("Foundation");
        metal_bridge_lib.?.linkFramework("Metal");
    }

    // Create the main PCP module
    const pcp_module = b.addModule("pcp", .{
        .root_source_file = b.path("src/main.zig"),
    });

    // Create the GPT-2 module
    const gpt2_module = b.addModule("gpt2", .{
        .root_source_file = b.path("src/models/gpt2.zig"),
    });

    // Add dependency from GPT-2 to PCP
    gpt2_module.addImport("pcp", pcp_module);

    const test_step = b.step("test", "Run unit tests");
    for (test_targets) |t_target| {
        // Add unit tests
        inline for (.{ "src/tensor.zig", "src/autodiff.zig", "src/ops.zig" }) |module| {
            const unit_tests = b.addTest(.{
                .root_source_file = b.path(module),
                .target = b.resolveTargetQuery(t_target),
                .optimize = optimize,
            });
            const run_unit_tests = b.addRunArtifact(unit_tests);
            run_unit_tests.skip_foreign_checks = true;
            test_step.dependOn(&run_unit_tests.step);
        }
    }

    // GPT-2 training example executable
    const gpt2_example = b.addExecutable(.{
        .name = "gpt2_example",
        .root_source_file = b.path("src/examples/gpt2_training.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for GPT-2 example
    gpt2_example.root_module.addImport("pcp", pcp_module);
    gpt2_example.root_module.addImport("gpt2", gpt2_module);

    // Shakespeare training example executable
    const shakespeare_example = b.addExecutable(.{
        .name = "shakespeare_example",
        .root_source_file = b.path("src/examples/shakespeare_training.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for Shakespeare example
    shakespeare_example.root_module.addImport("pcp", pcp_module);
    shakespeare_example.root_module.addImport("gpt2", gpt2_module);

    // Install the executables
    b.installArtifact(gpt2_example);
    b.installArtifact(shakespeare_example);

    // Run step for GPT-2 example (default)
    const run_gpt2_cmd = b.addRunArtifact(gpt2_example);
    run_gpt2_cmd.step.dependOn(&gpt2_example.step); // Only depend on the gpt2_example

    const run_gpt2_step = b.step("run", "Run the GPT-2 training example");
    run_gpt2_step.dependOn(&run_gpt2_cmd.step);

    // Run step for Shakespeare example
    const run_shakespeare_cmd = b.addRunArtifact(shakespeare_example);
    run_shakespeare_cmd.step.dependOn(b.getInstallStep());

    const run_shakespeare_step = b.step("run-shakespeare", "Run the Shakespeare training example");
    run_shakespeare_step.dependOn(&run_shakespeare_cmd.step);

    // Autodiff test executable
    const autodiff_test = b.addExecutable(.{
        .name = "autodiff_test",
        .root_source_file = b.path("src/examples/autodiff_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for autodiff test
    autodiff_test.root_module.addImport("pcp", pcp_module);

    // Install the test executable
    b.installArtifact(autodiff_test);

    // Run step for autodiff test
    const run_autodiff_test_cmd = b.addRunArtifact(autodiff_test);
    run_autodiff_test_cmd.step.dependOn(&autodiff_test.step);

    const run_autodiff_test_step = b.step("run-autodiff-test", "Run the autodiff Plan-based tests");
    run_autodiff_test_step.dependOn(&run_autodiff_test_cmd.step);

    // Comptime Plan examples executable
    const comptime_examples = b.addExecutable(.{
        .name = "comptime_examples",
        .root_source_file = b.path("src/examples/comptime_examples.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for comptime examples
    comptime_examples.root_module.addImport("pcp", pcp_module);

    // Install the executable
    b.installArtifact(comptime_examples);

    // Run step for comptime examples
    const run_comptime_examples_cmd = b.addRunArtifact(comptime_examples);
    run_comptime_examples_cmd.step.dependOn(b.getInstallStep());

    const run_comptime_examples_step = b.step("run-comptime-examples", "Run the comptime plan examples");
    run_comptime_examples_step.dependOn(&run_comptime_examples_cmd.step);

    // Metal backend test executable
    const metal_test = b.addExecutable(.{
        .name = "metal_test",
        .root_source_file = b.path("src/examples/metal_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for Metal test
    metal_test.root_module.addImport("pcp", pcp_module);

    // Link Metal bridge on macOS
    if (builtin.os.tag == .macos) {
        metal_test.linkFramework("Foundation");
        metal_test.linkFramework("Metal");
        metal_test.linkLibrary(metal_bridge_lib.?);
    }

    // Install the executable
    b.installArtifact(metal_test);

    // Run step for Metal test
    const run_metal_test_cmd = b.addRunArtifact(metal_test);
    run_metal_test_cmd.step.dependOn(&metal_test.step);

    const run_metal_test_step = b.step("run-metal-test", "Run the Metal backend tests");
    run_metal_test_step.dependOn(&run_metal_test_cmd.step);

    // Metal benchmark executable
    const metal_benchmark = b.addExecutable(.{
        .name = "metal_benchmark",
        .root_source_file = b.path("src/examples/metal_benchmark.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for Metal benchmark
    metal_benchmark.root_module.addImport("pcp", pcp_module);

    // Link Metal bridge on macOS with extra debugging
    if (builtin.os.tag == .macos) {
        std.debug.print("Adding Metal frameworks to benchmark...\n", .{});
        metal_benchmark.linkFramework("Foundation");
        metal_benchmark.linkFramework("Metal");
        metal_benchmark.linkLibrary(metal_bridge_lib.?);

        // Set dynamic link flags to ensure visibility of Metal symbols
        metal_benchmark.addCSourceFile(.{
            .file = b.path("src/backends/metal_bridge.m"),
            .flags = &[_][]const u8{"-fobjc-arc"},
        });
    }

    // Install the executable
    b.installArtifact(metal_benchmark);

    // Run step for Metal benchmark
    const run_metal_benchmark_cmd = b.addRunArtifact(metal_benchmark);
    run_metal_benchmark_cmd.step.dependOn(&metal_benchmark.step);

    const run_metal_benchmark_step = b.step("run-metal-benchmark", "Run the Metal backend benchmarks");
    run_metal_benchmark_step.dependOn(&run_metal_benchmark_cmd.step);

    // Create a new Zig file for our Plan-based test
    const plan_test_src =
        \\const std = @import("std");
        \\const pcp = @import("pcp");
        \\const tensor = pcp.tensor;
        \\const ops = pcp.ops;
        \\const autodiff = pcp.autodiff;
        \\
        \\/// Helper function for pointer casting
        \\fn ptrCastHelper(comptime T: type, ptr: anytype) T {
        \\    // We still need to use alignCast for pointers that require higher alignment
        \\    return @ptrCast(@alignCast(ptr));
        \\}
        \\
        \\const Allocator = std.mem.Allocator;
        \\const Tensor = tensor.Tensor;
        \\const DType = tensor.DType;
        \\
        \\// Helper function to print tensor contents
        \\fn printTensor(t: Tensor) void {
        \\    const buf = ptrCastHelper([*]f32, t.buffer.data.ptr)[0..t.shape.elemCount()];
        \\    
        \\    std.debug.print("Shape: [", .{});
        \\    for (t.shape.dims) |dim| {
        \\        std.debug.print("{}, ", .{dim});
        \\    }
        \\    std.debug.print("]\n", .{});
        \\    
        \\    if (t.shape.rank() == 2) {
        \\        const rows = t.shape.dims[0];
        \\        const cols = t.shape.dims[1];
        \\        
        \\        for (0..rows) |i| {
        \\            std.debug.print("[ ", .{});
        \\            for (0..cols) |j| {
        \\                std.debug.print("{d:.4} ", .{buf[i * cols + j]});
        \\            }
        \\            std.debug.print("]\n", .{});
        \\        }
        \\    } else {
        \\        // For other ranks, just print all values
        \\        std.debug.print("[ ", .{});
        \\        for (buf) |val| {
        \\            std.debug.print("{d:.4} ", .{val});
        \\        }
        \\        std.debug.print("]\n", .{});
        \\    }
        \\}
        \\
        \\// Test a simple model with Plan-based approach
        \\fn testPlanBasedModel(allocator: Allocator) !void {
        \\    std.debug.print("\n=== Testing Plan-Based Model ===\n", .{});
        \\    
        \\    // Create simple tensors
        \\    var dims = [_]usize{ 2, 2 };
        \\    
        \\    // Create tensors for parameters
        \\    var w1_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
        \\    defer w1_tensor.deinit();
        \\    
        \\    try w1_tensor.setScalar(&[_]usize{0, 0}, 0.1);
        \\    try w1_tensor.setScalar(&[_]usize{0, 1}, 0.2);
        \\    try w1_tensor.setScalar(&[_]usize{1, 0}, 0.3);
        \\    try w1_tensor.setScalar(&[_]usize{1, 1}, 0.4);
        \\    
        \\    var w2_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
        \\    defer w2_tensor.deinit();
        \\    
        \\    try w2_tensor.setScalar(&[_]usize{0, 0}, 0.5);
        \\    try w2_tensor.setScalar(&[_]usize{0, 1}, 0.6);
        \\    try w2_tensor.setScalar(&[_]usize{1, 0}, 0.7);
        \\    try w2_tensor.setScalar(&[_]usize{1, 1}, 0.8);
        \\    
        \\    // Create input
        \\    var x_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
        \\    defer x_tensor.deinit();
        \\    
        \\    try x_tensor.setScalar(&[_]usize{0, 0}, 1.0);
        \\    try x_tensor.setScalar(&[_]usize{0, 1}, 2.0);
        \\    try x_tensor.setScalar(&[_]usize{1, 0}, 3.0);
        \\    try x_tensor.setScalar(&[_]usize{1, 1}, 4.0);
        \\    
        \\    // Target values
        \\    var y_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
        \\    defer y_tensor.deinit();
        \\    
        \\    try y_tensor.setScalar(&[_]usize{0, 0}, 0.5);
        \\    try y_tensor.setScalar(&[_]usize{0, 1}, 0.5);
        \\    try y_tensor.setScalar(&[_]usize{1, 0}, 0.5);
        \\    try y_tensor.setScalar(&[_]usize{1, 1}, 0.5);
        \\    
        \\    // Create the autodiff plans
        \\    const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend, f32, null, null, null);
        \\    var matmul_plan1 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
        \\    defer matmul_plan1.deinit();
        \\    
        \\    var matmul_plan2 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
        \\    defer matmul_plan2.deinit();
        \\    
        \\    const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend, f32, null);
        \\    var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
        \\    defer relu_plan.deinit();
        \\    
        \\    // Forward pass using plans
        \\    std.debug.print("Running forward pass with plans...\n", .{});
        \\    
        \\    // h_pre = x @ w1
        \\    const h_pre = try matmul_plan1.forward(.{ .a = x_tensor, .b = w1_tensor });
        \\    defer h_pre.deinit();
        \\    
        \\    // h = relu(h_pre)
        \\    const h = try relu_plan.forward(h_pre);
        \\    defer h.deinit();
        \\    
        \\    // y_pred = h @ w2
        \\    const y_pred = try matmul_plan2.forward(.{ .a = h, .b = w2_tensor });
        \\    defer y_pred.deinit();
        \\    
        \\    // Print prediction
        \\    std.debug.print("\nPrediction (Plan-based):\n", .{});
        \\    printTensor(y_pred);
        \\    
        \\    // Compute MSE loss manually
        \\    var diff = try ops.subtract(allocator, y_pred, y_tensor);
        \\    defer diff.deinit();
        \\    
        \\    var diff_squared = try ops.multiply(allocator, diff, diff);
        \\    defer diff_squared.deinit();
        \\    
        \\    // Calculate mean manually
        \\    const diff_squared_buf = ptrCastHelper([*]f32, diff_squared.buffer.data.ptr)[0..diff_squared.shape.elemCount()];
        \\    var sum: f32 = 0.0;
        \\    for (diff_squared_buf) |val| {
        \\        sum += val;
        \\    }
        \\    const mse = sum / @as(f32, @floatFromInt(diff_squared.shape.elemCount()));
        \\    
        \\    std.debug.print("\nMSE Loss: {d:.6}\n", .{mse});
        \\    
        \\    // Create gradient with ones
        \\    var grad_ones = try Tensor.filled(allocator, y_pred.shape.dims, y_pred.dtype, 1.0, y_pred.backend);
        \\    defer grad_ones.deinit();
        \\    
        \\    // Backward pass
        \\    std.debug.print("\nRunning backward pass with plans...\n", .{});
        \\    
        \\    // Backprop through second matmul
        \\    const grads2 = try matmul_plan2.backward(grad_ones);
        \\    defer grads2.da.deinit();
        \\    defer grads2.db.deinit();
        \\    
        \\    // Backprop through relu
        \\    const relu_grad = try relu_plan.backward(grads2.da);
        \\    defer relu_grad.deinit();
        \\    
        \\    // Backprop through first matmul
        \\    const grads1 = try matmul_plan1.backward(relu_grad);
        \\    defer grads1.da.deinit();
        \\    defer grads1.db.deinit();
        \\    
        \\    // Print gradients
        \\    std.debug.print("\nGradients for w1 (Plan-based):\n", .{});
        \\    printTensor(grads1.db);
        \\    
        \\    std.debug.print("\nGradients for w2 (Plan-based):\n", .{});
        \\    printTensor(grads2.db);
        \\    
        \\    std.debug.print("\nPlan-based model test completed successfully\n", .{});
        \\}
        \\
        \\pub fn main() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\    
        \\    std.debug.print("Starting autodiff with Plan-based approach...\n", .{});
        \\    
        \\    // Test Plan-based model
        \\    try testPlanBasedModel(allocator);
        \\    
        \\    std.debug.print("\nAll tests completed successfully!\n", .{});
        \\}
    ;

    // Write the source to a file
    const plan_test_path = "src/examples/plan_based_test.zig";
    const plan_test_file = b.addWriteFile(plan_test_path, plan_test_src);

    // Create an executable from the new file
    const plan_test = b.addExecutable(.{
        .name = "plan_test",
        .root_source_file = b.path(plan_test_path),
        .target = target,
        .optimize = optimize,
    });

    // Make sure the file is written before compilation
    plan_test.step.dependOn(&plan_test_file.step);

    // Add module dependencies
    plan_test.root_module.addImport("pcp", pcp_module);

    // Install the executable
    b.installArtifact(plan_test);

    // Run step for Plan test
    const run_plan_test_cmd = b.addRunArtifact(plan_test);
    run_plan_test_cmd.step.dependOn(&plan_test.step); // Only depend on the plan_test

    const run_plan_test_step = b.step("run-plan-test", "Run the Plan-based autodiff test");
    run_plan_test_step.dependOn(&run_plan_test_cmd.step);
}
