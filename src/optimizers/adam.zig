/// Implementation of the optimizer *Adam* as introduced in "Kingma and Ba 2014: Adam: A method for
/// stochastic optimization." (https://arxiv.org/pdf/1412.6980). We apply an optimization suggested
/// by "Loshchilov and Hutter 2019: Decoupled Weight Decay Regularization"
/// (https://arxiv.org/pdf/1711.05101).

// The optimizer is used as the "inner Optimizer" in the DiLoCo algorithm

const std = @import("std");

/// Return a Adam optimizer for type 'T'
pub fn Adam(comptime param_count: usize, comptime T: type) type {
    return struct {
        alpha: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        timestep: usize,
        m: []T,
        v: []T,

        /// Initialize the optimizer. Literature suggests 'alpha=0.001', 'beta1=0.9', 'beta2=0.999' and 'epsilon=1e-8'.
        pub fn init(allocator: std.mem.Allocator, alpha: T, beta1: T, beta2: T, epsilon: T) !@This() {
            return @This(){
                .alpha = alpha,
                .beta1 = beta1,
                .beta2 = beta2,
                .epsilon = epsilon,
                .timestep = 0,
                .m = try allocator.alloc(T, param_count),
                .v = try allocator.alloc(T, param_count),
            };
        }

        pub fn update(self: *@This(), params: []T, grads: []const T) void {
            self.timestep += 1;
            const t: T = @floatFromInt(self.timestep);

            for (params, 0..) |*param, i| {
                const g = grads[i];
                // Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
                // Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
                // Compute bias-corrected first moment estimate
                const m_hat = self.m[i] / (1.0 - std.math.pow(T, self.beta1, t));
                // Compute bias-corrected second raw moment estimate
                const v_hat = self.v[i] / (1.0 - std.math.pow(T, self.beta2, t));

                // Update parameters
                param.* -= self.alpha * m_hat / (std.math.sqrt(v_hat) + self.epsilon);
            }
        }

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            allocator.free(self.m);
            allocator.free(self.v);
        }
    };
}

test "adam optimizer converges to minimum of (x - 3)^2" {
    // The test optimizes a simple 1D function.  Within 1000 iterations, it asserts that the
    // parameter x is close to 3 (within 1e−3).  The optimizer is fully reusable and deallocates
    // internal state after the test.

    const allocator = std.testing.allocator;

    // Initial parameter
    const x: f32 = 0.0;
    var params = [_]f32{x};

    // Gradient: df/dx = 2(x - 3)
    var grads = [_]f32{0.0};

    var adam = try Adam(1, f32).init(allocator, 0.1, 0.9, 0.999, 1e-8);

    defer adam.deinit(allocator);

    for (0..1000) |_| {
        grads[0] = 2.0 * (params[0] - 3.0); // compute gradient
        // std.debug.print("Run {}: grads: {}, params: {}\n", .{ n, grads[0], params[0] });
        adam.update(&params, &grads);
        if (@abs(params[0] - 3.0) < 1e-3) break;
    }

    const diff = @abs(params[0] - 3.0);
    std.testing.expect(diff < 1e-3) catch {
        std.debug.print("Final value: {}\n", .{params[0]});
        return error.TestExpectedEqual;
    };
}

test "adam optimizer converges on flattened 2D parameter matrix" {
    const allocator = std.testing.allocator;
    const Optimizer = Adam(4, f64);

    // Initial parameter values
    var params = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    var grads = [_]f64{ 0, 0, 0, 0 };

    var opt = try Optimizer.init(allocator, 0.1, 0.9, 0.999, 1e-8);
    defer opt.deinit(allocator);

    // Optimize toward target matrix: [[1, 2], [3, 4]]
    const target = [_]f64{ 1, 2, 3, 4 };

    for (0..1000) |_| {
        var diff_okay: bool = true;
        for (params, 0..) |p, i| {
            grads[i] = 2.0 * (p - target[i]);
            const diff = @abs(p - target[i]);
            if (diff_okay and diff > 1e-3) {
                diff_okay = false;
            }
        }
        if (diff_okay) break;
        opt.update(&params, &grads);
    }

    for (params, 0..) |p, i| {
        const diff = @abs(p - target[i]);
        std.testing.expect(diff < 1e-3) catch {
            std.debug.print("Mismatch at index {}: got {}, want {}\n", .{ i, p, target[i] });
            return error.TestExpectedEqual;
        };
    }
}
