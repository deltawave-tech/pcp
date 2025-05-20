const std = @import("std");
const pcp = @import("pcp");
const tensor = pcp.tensor;
const ops = pcp.ops;
const builtin = @import("builtin");

/// Helper function for pointer casting
fn ptrCastHelper(comptime T: type, ptr: anytype) T {
    // We still need to use alignCast for pointers that require higher alignment
    return @ptrCast(@alignCast(ptr));
}

// Simple timer to measure performance
const Timer = struct {
    start_time: i128,

    pub fn start() Timer {
        return Timer{
            .start_time = std.time.nanoTimestamp(),
        };
    }

    pub fn elapsed(self: Timer) f64 {
        const end_time = std.time.nanoTimestamp();
        const duration_ns = end_time - self.start_time;
        return @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0; // Convert to milliseconds
    }
};

// CPU benchmark as reference for Metal implementation
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== CPU Tensor Performance Benchmark ===\n\n", .{});
    std.debug.print("This benchmark measures CPU performance as a baseline for future Metal GPU acceleration.\n\n", .{});

    // Create some test tensors
    const m = 512;
    const n = 512;
    const p = 512;

    std.debug.print("Initializing tensors of size {d}x{d} and {d}x{d} for matmul benchmark...\n", .{ m, n, n, p });

    var a_dims = [_]usize{ m, n };
    var a = try tensor.Tensor(f32).random(allocator, &a_dims, .cpu);
    defer a.deinit();

    var b_dims = [_]usize{ n, p };
    var b = try tensor.Tensor(f32).random(allocator, &b_dims, .cpu);
    defer b.deinit();

    // Run CPU matmul
    std.debug.print("\nBenchmarking CPU matrix multiplication...\n", .{});

    const MatmulPlanType = ops.MatmulPlan(ops.CpuBackend, f32, null, null, null);
    var matmul_plan = MatmulPlanType.init(allocator);

    // Warm-up
    var warm_up_result = try matmul_plan.run(.{ .a = a, .b = b });
    warm_up_result.deinit();

    // Benchmark
    var timer = Timer.start();
    const result = try matmul_plan.run(.{ .a = a, .b = b });
    const elapsed = timer.elapsed();
    defer result.deinit();

    std.debug.print("CPU matrix multiplication ({d}x{d} @ {d}x{d}):\n", .{ m, n, n, p });
    std.debug.print("  Time: {d:.2} ms\n", .{elapsed});
    std.debug.print("  Throughput: {d:.2} GFLOPS\n", .{2.0 * @as(f64, @floatFromInt(m * n * p)) / elapsed / 1_000_000.0});

    // Large elementwise operation benchmark
    const elem_size = 10 * 1024 * 1024;
    std.debug.print("\nBenchmarking large elementwise operations ({d} elements)...\n", .{elem_size});

    var elem_dims = [_]usize{elem_size};
    var e1 = try tensor.Tensor.random(allocator, &elem_dims, .f32, .cpu);
    defer e1.deinit();

    var e2 = try tensor.Tensor.random(allocator, &elem_dims, .f32, .cpu);
    defer e2.deinit();

    // Add operation
    {
        const AddPlanType = ops.AddPlan(ops.CpuBackend, f32, null);
        var add_plan = AddPlanType.init(allocator);

        // Warm-up
        var add_warm_up = try add_plan.run(.{ .a = e1, .b = e2 });
        add_warm_up.deinit();

        // Benchmark
        timer = Timer.start();
        const add_result = try add_plan.run(.{ .a = e1, .b = e2 });
        const add_elapsed = timer.elapsed();
        defer add_result.deinit();

        std.debug.print("CPU elementwise addition:\n", .{});
        std.debug.print("  Time: {d:.2} ms\n", .{add_elapsed});
        std.debug.print("  Throughput: {d:.2} GB/s\n", .{3.0 * @as(f64, @floatFromInt(elem_size)) * @sizeOf(f32) / add_elapsed / 1_000_000.0});
    }

    // Multiply operation
    {
        const MulPlanType = ops.MultiplyPlan(ops.CpuBackend, f32, null);
        var mul_plan = MulPlanType.init(allocator);

        // Warm-up
        var mul_warm_up = try mul_plan.run(.{ .a = e1, .b = e2 });
        mul_warm_up.deinit();

        // Benchmark
        timer = Timer.start();
        const mul_result = try mul_plan.run(.{ .a = e1, .b = e2 });
        const mul_elapsed = timer.elapsed();
        defer mul_result.deinit();

        std.debug.print("CPU elementwise multiplication:\n", .{});
        std.debug.print("  Time: {d:.2} ms\n", .{mul_elapsed});
        std.debug.print("  Throughput: {d:.2} GB/s\n", .{3.0 * @as(f64, @floatFromInt(elem_size)) * @sizeOf(f32) / mul_elapsed / 1_000_000.0});
    }

    // ReLU operation
    {
        const ReluPlanType = ops.ReluPlan(ops.CpuBackend, f32, null);
        var relu_plan = ReluPlanType.init(allocator);

        // Warm-up
        var relu_warm_up = try relu_plan.run(e1);
        relu_warm_up.deinit();

        // Benchmark
        timer = Timer.start();
        const relu_result = try relu_plan.run(e1);
        const relu_elapsed = timer.elapsed();
        defer relu_result.deinit();

        std.debug.print("CPU ReLU activation:\n", .{});
        std.debug.print("  Time: {d:.2} ms\n", .{relu_elapsed});
        std.debug.print("  Throughput: {d:.2} GB/s\n", .{2.0 * @as(f64, @floatFromInt(elem_size)) * @sizeOf(f32) / relu_elapsed / 1_000_000.0});
    }

    std.debug.print("\n=== Metal Backend Development Roadmap ===\n", .{});
    std.debug.print("1. Fix Metal initialization and device creation issues\n", .{});
    std.debug.print("2. Implement and debug Metal kernel compilation\n", .{});
    std.debug.print("3. Complete tensor operation implementations with Metal shaders\n", .{});
    std.debug.print("4. Compare CPU vs Metal GPU performance\n", .{});
    std.debug.print("5. Optimize Metal kernels for maximum performance\n", .{});
    std.debug.print("\nBenchmark completed successfully.\n", .{});
}
