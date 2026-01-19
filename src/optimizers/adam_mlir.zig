/// FILE: src/optimizers/adam_mlir.zig (AdamW with Weight Decay)
const std = @import("std");
const mlir = @import("../mlir/wrapper.zig");
const ops = @import("../core/ops.zig");
const tensor = @import("../core/tensor.zig");

const MLIRBuilder = ops.MLIRBuilder;
const Tensor = tensor.Tensor(void);

pub fn AdamMLIRConfiguration(comptime DataType: type) type {
    return struct {
        learning_rate: DataType = 0.001,
        beta1: DataType = 0.9,
        beta2: DataType = 0.999,
        epsilon: DataType = 1e-6,
        weight_decay: DataType = 0.01,
        max_grad_norm: DataType = 1.0,
        gradient_clip_min: DataType = -100.0,
        gradient_clip_max: DataType = 100.0,
        pub fn default_configuration() @This() { return .{}; }
    };
}

pub fn AdamMLIR(comptime T: type) type {
    const ConfT = AdamMLIRConfiguration(T);

    return struct {
        allocator: std.mem.Allocator,
        builder: *MLIRBuilder,
        conf: ConfT,
        element_type: mlir.Type,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, builder: *MLIRBuilder, conf: ConfT, element_type: mlir.Type) !Self {
            return Self{
                .allocator = allocator,
                .builder = builder,
                .conf = conf,
                .element_type = element_type,
            };
        }

        pub fn deinit(self: *Self) void {
            _ = self; // Nothing to deinit in stateless version
        }

        /// Stateless update function for Graph construction (Mixed Precision).
        /// All math is performed in f32. M/V states stay in f32. Params are cast back to original type.
        pub fn update(
            self: *Self,
            params: Tensor,
            grads: Tensor,
            m_state: Tensor,
            v_state: Tensor,
            timestep: Tensor,
        ) !struct { new_params: Tensor, new_m: Tensor, new_v: Tensor } {
            const b = self.builder;
            const f32_type = mlir.Type.f32Type(b.ctx);
            const original_param_type = params.value.getType().as(mlir.RankedTensorType).?.getElementType();

            const params_f32 = try ops.convert(b, params, f32_type);
            const grads_f32 = try ops.convert(b, grads, f32_type);

            const dims = try params.shape.getDims(b.allocator);
            defer b.allocator.free(dims);

            const eps_val = @as(f64, @floatCast(self.conf.epsilon));
            const beta1 = try ops.constant(b, @as(f64, @floatCast(self.conf.beta1)), dims, f32_type);
            const beta2 = try ops.constant(b, @as(f64, @floatCast(self.conf.beta2)), dims, f32_type);
            const one = try ops.constant(b, 1.0, dims, f32_type);
            const lr = try ops.constant(b, @as(f64, @floatCast(self.conf.learning_rate)), dims, f32_type);
            const eps = try ops.constant(b, eps_val, dims, f32_type);
            const decay = try ops.constant(b, @as(f64, @floatCast(self.conf.weight_decay)), dims, f32_type);

            // Tensor-wise L2 Gradient Clipping with Iterative Dimensional Reduction
            const scalar_shape = &[_]i64{};

            // Pre-clamp gradients to prevent norm overflow (g^2 must fit in f32)
            const clamp_min = try ops.constant(b, -1000.0, scalar_shape, f32_type);
            const clamp_max = try ops.constant(b, 1000.0, scalar_shape, f32_type);
            const g_clamped_1 = try ops.maximum(b, grads_f32, clamp_min);
            const g_for_norm = try ops.minimum(b, g_clamped_1, clamp_max);

            // Use clamped gradients for norm calculation only
            const g_squared = try ops.multiply(b, g_for_norm, g_for_norm);

            // Iterative Reduction: reduce one dimension at a time to avoid shared memory overflow
            var sum_sq = g_squared;
            const rank = params.shape.rank();
            var i: usize = 0;
            while (i < rank) : (i += 1) {
                const reduce_dim = [_]i64{0};
                sum_sq = try ops.reduceSum(b, sum_sq, &reduce_dim, false);
            }

            const norm_raw = try ops.sqrt(b, sum_sq);

            const eps_scalar = try ops.constant(b, eps_val, scalar_shape, f32_type);
            const norm = try ops.add(b, norm_raw, eps_scalar);

            const max_norm_scalar = try ops.constant(b, @as(f64, @floatCast(self.conf.max_grad_norm)), scalar_shape, f32_type);
            const max_val = try ops.maximum(b, norm, max_norm_scalar);
            const clip_scale = try ops.divide(b, max_norm_scalar, max_val);

            // Apply scale to ORIGINAL grads (preserve direction, just scale magnitude)
            const clipped_grads = try ops.multiply(b, grads_f32, clip_scale);

            const one_minus_beta1 = try ops.subtract(b, one, beta1);
            const term1_m = try ops.multiply(b, beta1, m_state);
            const term2_m = try ops.multiply(b, one_minus_beta1, clipped_grads);
            const new_m = try ops.add(b, term1_m, term2_m);

            const one_minus_beta2 = try ops.subtract(b, one, beta2);
            const term1_v = try ops.multiply(b, beta2, v_state);
            const clipped_grads_sq = try ops.multiply(b, clipped_grads, clipped_grads);
            const term2_v = try ops.multiply(b, one_minus_beta2, clipped_grads_sq);
            const new_v = try ops.add(b, term1_v, term2_v);

            const timestep_broadcast_val = try ops.broadcastToShape(b, timestep.value, dims);
            const timestep_broadcast = try b.newTensor(timestep_broadcast_val);
            const beta1_pow = try ops.power(b, beta1, timestep_broadcast);
            const beta2_pow = try ops.power(b, beta2, timestep_broadcast);
            const m_corr_denom = try ops.subtract(b, one, beta1_pow);
            const v_corr_denom = try ops.subtract(b, one, beta2_pow);

            const m_corr_safe = try ops.maximum(b, m_corr_denom, eps);
            const v_corr_safe = try ops.maximum(b, v_corr_denom, eps);
            const m_hat = try ops.divide(b, new_m, m_corr_safe);
            const v_hat = try ops.divide(b, new_v, v_corr_safe);

            const sqrt_v = try ops.sqrt(b, v_hat);
            const denom = try ops.add(b, sqrt_v, eps);

            const ratio = try ops.divide(b, m_hat, denom);
            const decay_term = try ops.multiply(b, decay, params_f32);
            const combined = try ops.add(b, ratio, decay_term);

            const scaled_step = try ops.multiply(b, lr, combined);
            const new_params_f32 = try ops.subtract(b, params_f32, scaled_step);

            const new_params = try ops.convert(b, new_params_f32, original_param_type);

            return .{
                .new_params = new_params,
                .new_m = new_m,
                .new_v = new_v,
            };
        }
    };
}

/// MLIR-based Adam optimizer map for managing multiple optimizers
pub fn AdamMLIRMap(comptime K: type, comptime DataType: type) type {
    const AdamT = AdamMLIR(DataType);
    const MapT = std.AutoHashMap(K, AdamT);
    const ConfT = AdamMLIRConfiguration(DataType);

    return struct {
        allocator: std.mem.Allocator,
        builder: *MLIRBuilder,
        map: MapT,
        element_type: mlir.Type,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, builder: *MLIRBuilder, element_type: mlir.Type) Self {
            return Self{
                .allocator = allocator,
                .builder = builder,
                .map = MapT.init(allocator),
                .element_type = element_type,
            };
        }

        pub fn deinit(self: *Self) void {
            var it = self.map.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.*.deinit();
            }
            self.map.clearAndFree();
        }

        pub fn add(self: *Self, key: K, conf: ConfT) !*AdamT {
            const optimizer = try AdamT.init(self.allocator, self.builder, conf, self.element_type);
            try self.map.put(key, optimizer);
            return self.map.getPtr(key).?;
        }

        pub fn get(self: *Self, key: K) ?*AdamT {
            return self.map.getPtr(key);
        }

        pub fn update(
            self: *Self,
            key: K,
            params: Tensor,
            grads: Tensor,
            m_state: Tensor,
            v_state: Tensor,
            timestep: Tensor
        ) !struct { new_params: Tensor, new_m: Tensor, new_v: Tensor } {
            const optimizer = self.map.getPtr(key) orelse return error.OptimizerNotFound;
            return try optimizer.update(params, grads, m_state, v_state, timestep);
        }
    };
}

/// Test function for MLIR Adam optimizer
pub fn testAdamMLIR(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing MLIR Adam Optimizer ===\n", .{});

    // Create MLIR context for this test
    var ctx = try mlir.Context.init();
    defer ctx.deinit();
    const c_api = @import("../mlir/c.zig").c;
    c_api.mlirContextSetAllowUnregisteredDialects(ctx.handle, true);

    var builder = try MLIRBuilder.init(allocator, ctx);
    defer builder.deinit();

    const element_type = mlir.Type.f32Type(builder.ctx);
    const conf = AdamMLIRConfiguration(f32).default_configuration();

    var adam = try AdamMLIR(f32).init(allocator, &builder, conf, element_type);
    defer adam.deinit();

    std.debug.print("✓ MLIR Adam optimizer initialized\n", .{});
    std.debug.print("✓ MLIR Adam optimizer test completed\n", .{});
}
