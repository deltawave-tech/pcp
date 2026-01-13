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
        epsilon: DataType = 1e-8,
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

        /// Stateless update function for Graph construction.
        /// Returns new values for {params, m, v} to be passed to the next step.
        pub fn update(
            self: *Self,
            params: Tensor,
            grads: Tensor,
            m_state: Tensor,
            v_state: Tensor,
            timestep: Tensor,
        ) !struct { new_params: Tensor, new_m: Tensor, new_v: Tensor } {
            const b = self.builder;
            const dims = try params.shape.getDims(b.allocator);
            defer b.allocator.free(dims);

            // BF16 Safety: Clamp epsilon to prevent underflow to 0.0
            // BF16 resolution is low; 1e-8 becomes 0.0, causing div-by-zero in the first step.
            var eps_val = @as(f64, @floatCast(self.conf.epsilon));
            if (self.element_type.isBF16(b.ctx)) {
                if (eps_val < 1e-4) {
                    eps_val = 1e-4;
                }
            }

            const beta1 = try ops.constant(b, @as(f64, @floatCast(self.conf.beta1)), dims, self.element_type);
            const beta2 = try ops.constant(b, @as(f64, @floatCast(self.conf.beta2)), dims, self.element_type);
            const one = try ops.constant(b, 1.0, dims, self.element_type);
            const lr = try ops.constant(b, @as(f64, @floatCast(self.conf.learning_rate)), dims, self.element_type);
            const eps = try ops.constant(b, eps_val, dims, self.element_type);
            const decay = try ops.constant(b, @as(f64, @floatCast(self.conf.weight_decay)), dims, self.element_type);
            const max_norm = try ops.constant(b, @as(f64, @floatCast(self.conf.max_grad_norm)), dims, self.element_type);

            // Tensor-wise L2 Gradient Clipping
            const g_squared = try ops.multiply(b, grads, grads);

            var all_dims = std.ArrayList(i64).init(b.allocator);
            defer all_dims.deinit();
            for (0..params.shape.rank()) |i| try all_dims.append(@intCast(i));

            const sum_sq = try ops.reduceSum(b, g_squared, all_dims.items, true);

            const norm_raw = try ops.sqrt(b, sum_sq);
            const norm = try ops.add(b, norm_raw, eps);

            const max_val = try ops.maximum(b, norm, max_norm);
            const clip_scale = try ops.divide(b, max_norm, max_val);

            const clipped_grads = try ops.multiply(b, grads, clip_scale);

            // Update m_state
            const one_minus_beta1 = try ops.subtract(b, one, beta1);
            const term1_m = try ops.multiply(b, beta1, m_state);
            const term2_m = try ops.multiply(b, one_minus_beta1, clipped_grads);
            const new_m = try ops.add(b, term1_m, term2_m);

            // Update v_state
            const one_minus_beta2 = try ops.subtract(b, one, beta2);
            const term1_v = try ops.multiply(b, beta2, v_state);
            const clipped_grads_sq = try ops.multiply(b, clipped_grads, clipped_grads);
            const term2_v = try ops.multiply(b, one_minus_beta2, clipped_grads_sq);
            const new_v = try ops.add(b, term1_v, term2_v);

            // Bias Correction
            // Convert timestep to model element type if needed (timestep may be f32 while model is bf16)
            const timestep_converted = try ops.convert(b, timestep, self.element_type);
            const timestep_broadcast_val = try ops.broadcastToShape(b, timestep_converted.value, dims);
            const timestep_broadcast = try b.newTensor(timestep_broadcast_val);
            const beta1_pow = try ops.power(b, beta1, timestep_broadcast);
            const beta2_pow = try ops.power(b, beta2, timestep_broadcast);
            const m_corr_denom = try ops.subtract(b, one, beta1_pow);
            const v_corr_denom = try ops.subtract(b, one, beta2_pow);

            const m_hat = try ops.divide(b, new_m, m_corr_denom);
            const v_hat = try ops.divide(b, new_v, v_corr_denom);

            // AdamW Update
            const sqrt_v = try ops.sqrt(b, v_hat);
            const denom = try ops.add(b, sqrt_v, eps);

            const ratio = try ops.divide(b, m_hat, denom);
            const decay_term = try ops.multiply(b, decay, params);
            const combined = try ops.add(b, ratio, decay_term);

            const scaled_step = try ops.multiply(b, lr, combined);
            const new_params = try ops.subtract(b, params, scaled_step);

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
