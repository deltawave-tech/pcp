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

    /// Perform the Nesterov update using a pre-calculated outer gradient (Delta).
    ///
    /// Arguments:
    ///   - index: The index of the parameter tensor.
    ///   - master_param: Slice containing current master values (in/out).
    ///   - outer_gradient: The averaged delta from all workers (1/k * sum(theta_old - theta_new)).
    pub fn update(self: *Self, index: usize, master_param: []f32, outer_gradient: []const f32) !void {
        if (!self.initialized) return error.NotInitialized;
        if (index >= self.velocities.len) return error.OutOfBounds;
        if (master_param.len != outer_gradient.len) return error.DimensionMismatch;

        const velocity = self.velocities[index];
        if (master_param.len != velocity.len) return error.DimensionMismatch;

        const lr = self.config.learning_rate;
        const mu = self.config.momentum;

        // Vectorized loop optimization
        for (0..master_param.len) |i| {
            const grad = outer_gradient[i];
            const p_master = master_param[i];

            // 1. Update Velocity (Polyak)
            // v_{t+1} = mu * v_t + grad
            const v_prev = velocity[i];
            const v_new = mu * v_prev + grad;
            velocity[i] = v_new;

            // 2. Apply Nesterov Update (Sutskever formulation)
            // theta_{t+1} = theta_t - lr * (grad + mu * v_{t+1})
            // Note: The gradient here acts as the direction *up* the hill of the loss function
            // defined by the difference, so we subtract it.
            master_param[i] = p_master - lr * (grad + mu * v_new);
        }
    }

    /// Serialize optimizer state (velocities) to binary format for checkpoint recovery
    pub fn serialize(self: *Nesterov) ![]u8 {
        if (!self.initialized) return error.NotInitialized;

        var buffer = std.ArrayList(u8).init(self.allocator);

        // Write raw bytes of all velocity tensors sequentially
        for (self.velocities) |v| {
            const bytes = std.mem.sliceAsBytes(v);
            try buffer.appendSlice(bytes);
        }

        return buffer.toOwnedSlice();
    }

    /// Deserialize optimizer state (velocities) from binary format for checkpoint recovery
    pub fn deserialize(self: *Nesterov, data: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;

        var offset: usize = 0;
        for (self.velocities) |v| {
            const size = v.len * @sizeOf(f32);
            if (offset + size > data.len) return error.CorruptSaveFile;

            const src = data[offset .. offset + size];
            @memcpy(std.mem.sliceAsBytes(v), src);
            offset += size;
        }
    }
};
