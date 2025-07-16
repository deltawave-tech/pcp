/// MLIR-based implementation of the Adam optimizer
/// Implements the Adam optimizer using MLIR/StableHLO operations for compilation and optimization

const std = @import("std");
const mlir = @import("../mlir.zig");
const ops = @import("../ops.zig");
const tensor = @import("../tensor.zig");
const hlo = @import("../mlir/dialects/stablehlo.zig");

const MLIRBuilder = ops.MLIRBuilder;
const Tensor = tensor.Tensor(void);

pub fn AdamMLIRConfiguration(comptime DataType: type) type {
    return struct {
        learning_rate: DataType = 0.001,
        beta1: DataType = 0.9,
        beta2: DataType = 0.999,
        epsilon: DataType = 1e-8,

        pub fn default_configuration() @This() {
            return @This(){
                .learning_rate = 0.001,
                .beta1 = 0.9,
                .beta2 = 0.999,
                .epsilon = 1e-8,
            };
        }
    };
}

pub fn AdamMLIR(comptime T: type) type {
    const ConfT = AdamMLIRConfiguration(T);

    return struct {
        allocator: std.mem.Allocator,
        builder: *MLIRBuilder,
        conf: ConfT,
        timestep: usize,
        element_type: mlir.Type,
        m_state: ?Tensor,
        v_state: ?Tensor,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, builder: *MLIRBuilder, conf: ConfT, element_type: mlir.Type) !Self {
            return Self{
                .allocator = allocator,
                .builder = builder,
                .conf = conf,
                .timestep = 0,
                .element_type = element_type,
                .m_state = null,
                .v_state = null,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.m_state) |m| {
                m.deinit();
            }
            if (self.v_state) |v| {
                v.deinit();
            }
        }

        /// Initialize state tensors to match parameter shape
        fn initializeState(self: *Self, params: Tensor) !void {
            // Initialize m and v to zeros with same shape as params
            const zero_op = hlo.zeroConstant(self.builder.ctx, params.shape.dims, self.element_type);
            
            // Create m state (first moment)
            const m_op = hlo.zeroConstant(self.builder.ctx, params.shape.dims, self.element_type);
            self.m_state = try self.builder.newTensor(m_op.getResult(0));
            
            // Create v state (second moment)
            const v_op = hlo.zeroConstant(self.builder.ctx, params.shape.dims, self.element_type);
            self.v_state = try self.builder.newTensor(v_op.getResult(0));
        }

        /// Update parameters using Adam algorithm implemented in MLIR
        pub fn update(self: *Self, params: Tensor, grads: Tensor) !Tensor {
            // Initialize state if first call
            if (self.m_state == null) {
                try self.initializeState(params);
            }

            self.timestep += 1;
            const t_val: T = @floatFromInt(self.timestep);

            // Create scalar constants for Adam parameters
            const beta1_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.beta1)), params.shape.dims, self.element_type);
            const beta2_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.beta2)), params.shape.dims, self.element_type);
            const one_tensor = try ops.constant(self.builder, 1.0, params.shape.dims, self.element_type);
            const lr_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.learning_rate)), params.shape.dims, self.element_type);
            const epsilon_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.epsilon)), params.shape.dims, self.element_type);

            // Calculate (1 - beta1) and (1 - beta2)
            const one_minus_beta1 = try ops.subtract(self.builder, one_tensor, beta1_tensor);
            const one_minus_beta2 = try ops.subtract(self.builder, one_tensor, beta2_tensor);

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
            const m_decay = try ops.multiply(self.builder, beta1_tensor, self.m_state.?);
            const m_update = try ops.multiply(self.builder, one_minus_beta1, grads);
            const new_m = try ops.add(self.builder, m_decay, m_update);

            // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
            const v_decay = try ops.multiply(self.builder, beta2_tensor, self.v_state.?);
            const grads_squared = try ops.multiply(self.builder, grads, grads);
            const v_update = try ops.multiply(self.builder, one_minus_beta2, grads_squared);
            const new_v = try ops.add(self.builder, v_decay, v_update);

            // Bias correction terms
            const t_tensor = try ops.constant(self.builder, @as(f64, @floatCast(t_val)), params.shape.dims, self.element_type);
            
            // Calculate beta1^t and beta2^t using exponential operation
            const beta1_power_t = try self.computePower(beta1_tensor, t_tensor);
            const beta2_power_t = try self.computePower(beta2_tensor, t_tensor);
            
            // Calculate bias correction: 1 - beta^t
            const one_minus_beta1_t = try ops.subtract(self.builder, one_tensor, beta1_power_t);
            const one_minus_beta2_t = try ops.subtract(self.builder, one_tensor, beta2_power_t);

            // Compute bias-corrected estimates
            const m_hat = try ops.divide(self.builder, new_m, one_minus_beta1_t);
            const v_hat = try ops.divide(self.builder, new_v, one_minus_beta2_t);

            // Compute sqrt(v_hat) + epsilon
            const sqrt_v_hat = try self.computeSqrt(v_hat);
            const sqrt_v_hat_plus_eps = try ops.add(self.builder, sqrt_v_hat, epsilon_tensor);

            // Update parameters: params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
            const update_direction = try ops.divide(self.builder, m_hat, sqrt_v_hat_plus_eps);
            const scaled_update = try ops.multiply(self.builder, lr_tensor, update_direction);
            const new_params = try ops.subtract(self.builder, params, scaled_update);

            // Update internal state
            self.m_state.?.deinit();
            self.v_state.?.deinit();
            self.m_state = new_m;
            self.v_state = new_v;

            return new_params;
        }

        /// Compute power operation using exponential and logarithm
        fn computePower(self: *Self, base: Tensor, exponent: Tensor) !Tensor {
            // power(base, exp) = exp(exp * log(base))
            const log_base = try self.computeLog(base);
            const exp_times_log = try ops.multiply(self.builder, exponent, log_base);
            return try ops.exp(self.builder, exp_times_log);
        }

        /// Compute square root using power operation
        fn computeSqrt(self: *Self, input: Tensor) !Tensor {
            // sqrt(x) = x^0.5
            const half_tensor = try ops.constant(self.builder, 0.5, input.shape.dims, self.element_type);
            return try self.computePower(input, half_tensor);
        }

        /// Compute natural logarithm (placeholder - needs StableHLO log operation)
        fn computeLog(self: *Self, input: Tensor) !Tensor {
            // Use StableHLO log operation
            const operation = hlo.log(self.builder.ctx, input.value, self.builder.loc);
            return try self.builder.newTensor(operation.getResult(0));
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

        pub fn update(self: *Self, key: K, params: Tensor, grads: Tensor) !Tensor {
            const optimizer = self.map.getPtr(key) orelse return error.OptimizerNotFound;
            return try optimizer.update(params, grads);
        }
    };
}

/// Test function for MLIR Adam optimizer
pub fn testAdamMLIR(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing MLIR Adam Optimizer ===\n", .{});

    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();

    const element_type = mlir.Type.f32Type(builder.ctx);
    const conf = AdamMLIRConfiguration(f32).default_configuration();
    
    var adam = try AdamMLIR(f32).init(allocator, &builder, conf, element_type);
    defer adam.deinit();

    std.debug.print("✓ MLIR Adam optimizer initialized\n", .{});
    std.debug.print("✓ MLIR Adam optimizer test completed\n", .{});
}