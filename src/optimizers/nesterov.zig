// File: src/optimizers/nesterov.zig
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const NesterovConfig = struct {
    learning_rate: f32 = 0.01,
    momentum: f32 = 0.9,
};

/// Host-side Nesterov Optimizer (Pure Zig, no MLIR)
/// Used for Shepherd/Coordinator updates on the CPU.
pub const Nesterov = struct {
    allocator: Allocator,
    config: NesterovConfig,
    // One velocity array per parameter tensor
    velocities: [][]f32,
    initialized: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, config: NesterovConfig) Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .velocities = &[_][]f32{},
            .initialized = false,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.initialized) {
            for (self.velocities) |v| {
                self.allocator.free(v);
            }
            self.allocator.free(self.velocities);
        }
    }

    /// Allocate velocity buffers based on parameter shapes.
    /// Should be called once before the first update.
    pub fn initParameters(self: *Self, shapes: [][]const i64) !void {
        if (self.initialized) return;

        self.velocities = try self.allocator.alloc([]f32, shapes.len);

        for (shapes, 0..) |shape, i| {
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);

            self.velocities[i] = try self.allocator.alloc(f32, elem_count);
            @memset(self.velocities[i], 0.0);
        }
        self.initialized = true;
    }

    /// Reset velocities to zero (e.g., if model diverges or manual reset required)
    pub fn resetVelocity(self: *Self) void {
        if (!self.initialized) return;
        for (self.velocities) |v| {
            @memset(v, 0.0);
        }
    }

    /// Perform the update on a specific parameter tensor by index.
    ///
    /// In DiLoCo: gradient = (master_param - averaged_worker_param)
    /// update:
    ///    v = momentum * v + gradient
    ///    p_new = p_old - lr * v
    ///
    /// Arguments:
    ///   - index: The index of the parameter tensor (matches initParameters order)
    ///   - master_param: Slice containing current master values (in/out)
    ///   - averaged_worker_param: Slice containing averaged worker values
    pub fn update(self: *Self, index: usize, master_param: []f32, averaged_worker_param: []const f32) !void {
        if (!self.initialized) return error.NotInitialized;
        if (index >= self.velocities.len) return error.OutOfBounds;
        if (master_param.len != averaged_worker_param.len) return error.DimensionMismatch;

        const velocity = self.velocities[index];
        if (master_param.len != velocity.len) return error.DimensionMismatch;

        const lr = self.config.learning_rate;
        const mu = self.config.momentum;

        // Vectorized loop optimization (auto-vectorized by Zig ReleaseFast)
        for (0..master_param.len) |i| {
            const p_master = master_param[i];
            const p_worker = averaged_worker_param[i];

            // 1. Compute pseudo-gradient (direction towards workers)
            // If workers moved right, p_worker > p_master.
            // We want grad to be negative so -lr*grad moves right.
            // Standard SGD: p -= lr * grad.
            // Here: grad = p_master - p_worker
            const grad = p_master - p_worker;

            // 2. Update Velocity
            // v_{t+1} = mu * v_t + grad
            const v_new = mu * velocity[i] + grad;
            velocity[i] = v_new;

            // 3. Apply Update
            // p_{t+1} = p_t - lr * v_{t+1}
            master_param[i] = p_master - (lr * v_new);
        }
    }
};
