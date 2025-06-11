const std = @import("std");

pub fn NesterovAcceleretedGradient(comptime DataType: type) type {
    return struct {
        allocator: std.mem.Allocator,
        learning_rate: DataType,
        momentum: DataType,
        velocity: []DataType,

        const Self = @This();

        pub fn init(
            allocator: std.mem.Allocator,
            param_count: usize,
            learning_rate: DataType,
            momentum: DataType,
        ) !Self {
            const velocity = try allocator.alloc(DataType, param_count);
            for (velocity) |*v| {
                v.* = 0;
            }

            return Self{
                .allocator = allocator,
                .learning_rate = learning_rate,
                .momentum = momentum,
                .velocity = velocity,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.velocity);
        }

        pub fn update(
            self: *Self,
            params: []DataType,
            grads: []DataType,
        ) !void {
            // Step 1: create lookahead parameters
            const lookahead = try self.allocator.alloc(DataType, params.len);
            defer self.allocator.free(lookahead);

            for (params, self.velocity, 0..) |theta, vi, i| {
                lookahead[i] = theta - self.momentum * vi;
            }

            // NOTE: In practice, gradients should be recomputed at `lookahead`.
            // This demo assumes gradients are already computed at lookahead.

            // Step 2: update velocity and parameters
            for (self.velocity, grads, params) |*vi, gi, *thetai| {
                vi.* = self.momentum * vi.* + self.learning_rate * gi;
                thetai.* -= vi.*;
            }
        }
    };
}
