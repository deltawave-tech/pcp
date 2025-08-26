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

/// Builds a self-contained `func.func @apply_adam(...)` that encapsulates the Adam optimizer logic.
/// The function signature will be:
/// func(%params, %grads, %m_state, %v_state, %timestep) -> (%new_params, %new_m, %new_v)
fn buildAdamUpdateFunction(
    builder: *MLIRBuilder,
    conf: anytype,
    tensor_type: mlir.Type,
    element_type: mlir.Type,
) ![]const u8 {
    // 1. Define the function signature
    const scalar_type = mlir.Type.rankedTensorType(builder.ctx, &[_]i64{}, element_type); // scalar type
    const input_types = [_]mlir.Type{ tensor_type, tensor_type, tensor_type, tensor_type, scalar_type }; // params, grads, m_state, v_state, timestep
    const result_types = [_]mlir.Type{ tensor_type, tensor_type, tensor_type }; // new_params, new_m, new_v
    const func_type = mlir.Type.functionType(builder.ctx, &input_types, &result_types);

    const func_name = "apply_adam_update";

    // 2. Create the func.func operation
    const func_op = mlir.Operation.create(builder.ctx, "func.func", .{
        .attributes = &.{
            .{ "function_type", mlir.Attribute.typeAttr(func_type) },
            .{ "sym_name", mlir.Attribute.stringAttr(builder.ctx, func_name) },
        },
        .location = builder.loc,
    });
    // Add the function to the main module's body
    builder.module.op().getRegion(0).getBlock(0).appendOwnedOperation(func_op);

    // 3. Create the function body
    const region = func_op.getRegion(0);
    const block = try builder.createBlock();
    region.appendOwnedBlock(block);

    // 4. Get handles to the function arguments
    const params_arg = block.addArgument(tensor_type, builder.loc);
    const grads_arg = block.addArgument(tensor_type, builder.loc);
    const m_state_arg = block.addArgument(tensor_type, builder.loc);
    const v_state_arg = block.addArgument(tensor_type, builder.loc);
    const timestep_arg = block.addArgument(scalar_type, builder.loc);

    const params = try builder.newTensor(params_arg);
    const grads = try builder.newTensor(grads_arg);
    const m_state = try builder.newTensor(m_state_arg);
    const v_state = try builder.newTensor(v_state_arg);
    const timestep = try builder.newTensor(timestep_arg);

    // 5. Build the Adam optimizer logic inside this function
    const beta1_tensor = try ops.constant(builder, @as(f64, @floatCast(conf.beta1)), params.shape.dims, element_type);
    const beta2_tensor = try ops.constant(builder, @as(f64, @floatCast(conf.beta2)), params.shape.dims, element_type);
    const one_tensor = try ops.constant(builder, 1.0, params.shape.dims, element_type);
    const lr_tensor = try ops.constant(builder, @as(f64, @floatCast(conf.learning_rate)), params.shape.dims, element_type);
    const epsilon_tensor = try ops.constant(builder, @as(f64, @floatCast(conf.epsilon)), params.shape.dims, element_type);

    // Calculate (1 - beta1) and (1 - beta2)
    const one_minus_beta1 = try ops.subtract(builder, one_tensor, beta1_tensor);
    const one_minus_beta2 = try ops.subtract(builder, one_tensor, beta2_tensor);

    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
    const m_decay = try ops.multiply(builder, beta1_tensor, m_state);
    const m_update = try ops.multiply(builder, one_minus_beta1, grads);
    const new_m = try ops.add(builder, m_decay, m_update);

    // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
    const v_decay = try ops.multiply(builder, beta2_tensor, v_state);
    const grads_squared = try ops.multiply(builder, grads, grads);
    const v_update = try ops.multiply(builder, one_minus_beta2, grads_squared);
    const new_v = try ops.add(builder, v_decay, v_update);

    // Broadcast timestep to tensor shape for power computation
    const t_tensor = try ops.broadcast(builder, timestep, params.shape.dims);
    
    // Calculate beta1^t and beta2^t using exponential operation
    const beta1_power_t = try ops.exp(builder, try ops.multiply(builder, try ops.log(builder, beta1_tensor), t_tensor));
    const beta2_power_t = try ops.exp(builder, try ops.multiply(builder, try ops.log(builder, beta2_tensor), t_tensor));
    
    // Calculate bias correction: 1 - beta^t
    const one_minus_beta1_t = try ops.subtract(builder, one_tensor, beta1_power_t);
    const one_minus_beta2_t = try ops.subtract(builder, one_tensor, beta2_power_t);

    // Compute bias-corrected estimates
    const m_hat = try ops.divide(builder, new_m, one_minus_beta1_t);
    const v_hat = try ops.divide(builder, new_v, one_minus_beta2_t);

    // Compute sqrt(v_hat) + epsilon
    const half_tensor = try ops.constant(builder, 0.5, params.shape.dims, element_type);
    const sqrt_v_hat = try ops.exp(builder, try ops.multiply(builder, try ops.log(builder, v_hat), half_tensor));
    const sqrt_v_hat_plus_eps = try ops.add(builder, sqrt_v_hat, epsilon_tensor);

    // Update parameters: params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
    const update_direction = try ops.divide(builder, m_hat, sqrt_v_hat_plus_eps);
    const scaled_update = try ops.multiply(builder, lr_tensor, update_direction);
    const new_params = try ops.subtract(builder, params, scaled_update);

    // 6. Add the return operation
    _ = try builder.createAndAttach("func.return", &.{ new_params.value, new_m.value, new_v.value }, &.{});

    return func_name;
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
        
        // NEW FIELD: Store the name of our pre-built optimizer function
        update_fn_name: []const u8,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, builder: *MLIRBuilder, conf: ConfT, element_type: mlir.Type) !Self {
            // To build the function, we need a representative tensor type.
            // The optimizer logic is shape-agnostic, but MLIR needs a concrete type.
            // We'll use a placeholder shape [1] which works because the logic is element-wise.
            const placeholder_shape = &[_]i64{1};
            const tensor_type = mlir.Type.rankedTensorType(builder.ctx, placeholder_shape, element_type);

            // Build the reusable update function and get its name
            const fn_name = try buildAdamUpdateFunction(builder, conf, tensor_type, element_type);

            return Self{
                .allocator = allocator,
                .builder = builder,
                .conf = conf,
                .timestep = 0,
                .element_type = element_type,
                .m_state = null,
                .v_state = null,
                .update_fn_name = fn_name, // Store the name
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

            // The logic is now encapsulated in the pre-built function.
            // We just need to call it.
            
            // Create scalar timestep tensor
            const scalar_shape = &[_]i64{}; // scalar shape
            const scalar_type = mlir.Type.rankedTensorType(self.builder.ctx, scalar_shape, self.element_type);
            const timestep_tensor = try ops.constant(self.builder, @as(f64, @floatCast(t_val)), scalar_shape, self.element_type);
            
            // 1. Prepare operands for the call
            const operands = [_]mlir.Value{
                params.value,
                grads.value,
                self.m_state.?.value,
                self.v_state.?.value,
                timestep_tensor.value,
            };

            // 2. Define the result types based on the function's signature
            const tensor_type = params.value.getType();
            const result_types = [_]mlir.Type{
                tensor_type, // new_params
                tensor_type, // new_m
                tensor_type, // new_v
            };

            // 3. Create the func.call operation
            const call_op = self.builder.createAndAttach("func.call", &operands, &result_types, .{
                .attributes = &.{
                    .{ "callee", mlir.Attribute.stringAttr(self.builder.ctx, self.update_fn_name) },
                },
            }) catch |err| {
                std.log.err("Failed to create func.call for Adam optimizer: {}", .{err});
                // Dump the module to see what functions are available
                self.builder.module.op().dump();
                return err;
            };
            
            // 4. Extract results from the call
            const new_params_val = call_op.getResult(0);
            const new_m_val = call_op.getResult(1);
            const new_v_val = call_op.getResult(2);

            const new_params = try self.builder.newTensor(new_params_val);
            const new_m = try self.builder.newTensor(new_m_val);
            const new_v = try self.builder.newTensor(new_v_val);

            // 5. Update internal state
            if (self.m_state) |m| m.deinit();
            if (self.v_state) |v| v.deinit();
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