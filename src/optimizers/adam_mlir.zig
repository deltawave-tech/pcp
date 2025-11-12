/// FILE: src/optimizers/adam_mlir.zig (Numerically Stable Version)
const std = @import("std");
const mlir = @import("../mlir.zig");
const ops = @import("../ops.zig");
const tensor = @import("../tensor.zig");

const MLIRBuilder = ops.MLIRBuilder;
const Tensor = tensor.Tensor(void);

// AdamMLIRConfiguration remains the same
pub fn AdamMLIRConfiguration(comptime DataType: type) type {
    return struct {
        learning_rate: DataType = 0.001,
        beta1: DataType = 0.9,
        beta2: DataType = 0.999,
        epsilon: DataType = 1e-8,
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

        // The update function remains stateless.
        pub fn update(
            self: *Self,
            params: Tensor,
            grads: Tensor,
            m_state: Tensor,
            v_state: Tensor,
            timestep: Tensor,
        ) !struct { new_params: Tensor, new_m: Tensor, new_v: Tensor } {
            const dims = try params.shape.getDims(self.builder.allocator);
            defer self.builder.allocator.free(dims);
            
            // Create constants
            const beta1 = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.beta1)), dims, self.element_type);
            const beta2 = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.beta2)), dims, self.element_type);
            const one = try ops.constant(self.builder, 1.0, dims, self.element_type);
            const lr = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.learning_rate)), dims, self.element_type);
            const epsilon = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.epsilon)), dims, self.element_type);

            // Update m_state: m = beta1 * m + (1 - beta1) * g
            const one_minus_beta1 = try ops.subtract(self.builder, one, beta1);
            const new_m = try ops.add(self.builder, try ops.multiply(self.builder, beta1, m_state), try ops.multiply(self.builder, one_minus_beta1, grads));

            // Update v_state: v = beta2 * v + (1 - beta2) * g^2
            const one_minus_beta2 = try ops.subtract(self.builder, one, beta2);
            const grads_squared = try ops.multiply(self.builder, grads, grads);
            const new_v = try ops.add(self.builder, try ops.multiply(self.builder, beta2, v_state), try ops.multiply(self.builder, one_minus_beta2, grads_squared));

            // Bias correction
            const beta1_pow_t = try ops.power(self.builder, beta1, timestep);
            const beta2_pow_t = try ops.power(self.builder, beta2, timestep);
            const m_hat_denom = try ops.subtract(self.builder, one, beta1_pow_t);
            const v_hat_denom = try ops.subtract(self.builder, one, beta2_pow_t);
            const m_hat = try ops.divide(self.builder, new_m, m_hat_denom);
            const v_hat = try ops.divide(self.builder, new_v, v_hat_denom);

            // Use the direct stablehlo.sqrt operation
            const sqrt_v_hat = try ops.sqrt(self.builder, v_hat);

            const final_denom = try ops.add(self.builder, sqrt_v_hat, epsilon);
            const update_term = try ops.divide(self.builder, m_hat, final_denom);
            const scaled_update = try ops.multiply(self.builder, lr, update_term);
            const new_params = try ops.subtract(self.builder, params, scaled_update);
            
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