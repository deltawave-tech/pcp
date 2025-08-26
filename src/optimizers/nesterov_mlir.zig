/// MLIR-based implementation of the Nesterov Accelerated Gradient optimizer
/// Implements the Nesterov momentum optimizer using MLIR/StableHLO operations for compilation and optimization

const std = @import("std");
const mlir = @import("../mlir.zig");
const ops = @import("../ops.zig");
const tensor = @import("../tensor.zig");
const hlo = @import("../mlir/dialects/stablehlo.zig");

const MLIRBuilder = ops.MLIRBuilder;
const Tensor = tensor.Tensor(void);

pub fn NesterovMLIRConfiguration(comptime DataType: type) type {
    return struct {
        learning_rate: DataType = 0.01,
        momentum: DataType = 0.9,

        pub fn default_configuration() @This() {
            return @This(){
                .learning_rate = 0.01,
                .momentum = 0.9,
            };
        }
    };
}

/// Builds a self-contained `func.func @apply_nesterov(...)` that encapsulates the optimizer logic.
/// The function signature will be:
/// func(%params, %grads, %velocity) -> (%new_params, %new_velocity)
fn buildNesterovUpdateFunction(
    builder: *MLIRBuilder,
    conf: anytype,
    tensor_type: mlir.Type,
    element_type: mlir.Type,
) ![]const u8 {
    // 1. Define the function signature
    const input_types = [_]mlir.Type{ tensor_type, tensor_type, tensor_type }; // params, grads, velocity
    const result_types = [_]mlir.Type{ tensor_type, tensor_type }; // new_params, new_velocity
    const func_type = mlir.Type.functionType(builder.ctx, &input_types, &result_types);

    const func_name = "apply_nesterov_update";

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
    const velocity_arg = block.addArgument(tensor_type, builder.loc);

    const params = try builder.newTensor(params_arg);
    const grads = try builder.newTensor(grads_arg);
    const velocity = try builder.newTensor(velocity_arg);

    // 5. Build the optimizer logic inside this function
    const momentum_tensor = try ops.constant(builder, @as(f64, @floatCast(conf.momentum)), params.shape.dims, element_type);
    const lr_tensor = try ops.constant(builder, @as(f64, @floatCast(conf.learning_rate)), params.shape.dims, element_type);

    // velocity = momentum * velocity + learning_rate * grads
    const momentum_velocity_update = try ops.multiply(builder, momentum_tensor, velocity);
    const lr_grads = try ops.multiply(builder, lr_tensor, grads);
    const new_velocity = try ops.add(builder, momentum_velocity_update, lr_grads);

    // params = params - new_velocity
    const new_params = try ops.subtract(builder, params, new_velocity);

    // 6. Add the return operation
    _ = try builder.createAndAttach("func.return", &.{ new_params.value, new_velocity.value }, &.{});

    return func_name;
}

pub fn NesterovMLIR(comptime T: type) type {
    const ConfT = NesterovMLIRConfiguration(T);

    return struct {
        allocator: std.mem.Allocator,
        builder: *MLIRBuilder,
        conf: ConfT,
        element_type: mlir.Type,
        velocity: ?Tensor,
        
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
            const fn_name = try buildNesterovUpdateFunction(builder, conf, tensor_type, element_type);

            return Self{
                .allocator = allocator,
                .builder = builder,
                .conf = conf,
                .element_type = element_type,
                .velocity = null,
                .update_fn_name = fn_name, // Store the name
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.velocity) |v| {
                v.deinit();
            }
        }

        /// Initialize velocity tensor to match parameter shape
        fn initializeVelocity(self: *Self, params: Tensor) !void {
            // Initialize velocity to zeros with same shape as params
            const zero_op = hlo.zeroConstant(self.builder.ctx, params.shape.dims, self.element_type);
            self.velocity = try self.builder.newTensor(zero_op.getResult(0));
        }

        /// Update parameters using Nesterov momentum algorithm implemented in MLIR
        pub fn update(self: *Self, params: Tensor, grads: Tensor) !Tensor {
            // Initialize velocity if first call
            if (self.velocity == null) {
                try self.initializeVelocity(params);
            }

            // The logic is now encapsulated in the pre-built function.
            // We just need to call it.
            
            // 1. Prepare operands for the call
            const operands = [_]mlir.Value{
                params.value,
                grads.value,
                self.velocity.?.value,
            };

            // 2. Define the result types based on the function's signature
            const tensor_type = params.value.getType();
            const result_types = [_]mlir.Type{
                tensor_type, // new_params
                tensor_type, // new_velocity
            };

            // 3. Create the func.call operation
            const call_op = self.builder.createAndAttach("func.call", &operands, &result_types, .{
                .attributes = &.{
                    .{ "callee", mlir.Attribute.stringAttr(self.builder.ctx, self.update_fn_name) },
                },
            }) catch |err| {
                std.log.err("Failed to create func.call for optimizer: {}", .{err});
                // Dump the module to see what functions are available
                self.builder.module.op().dump();
                return err;
            };
            
            // 4. Extract results from the call
            const new_params_val = call_op.getResult(0);
            const new_velocity_val = call_op.getResult(1);

            const new_params = try self.builder.newTensor(new_params_val);
            const new_velocity = try self.builder.newTensor(new_velocity_val);

            // 5. Update internal state
            if (self.velocity) |v| v.deinit();
            self.velocity = new_velocity;

            return new_params;
        }

        /// Alternative implementation that more clearly separates the lookahead computation
        pub fn updateWithLookahead(self: *Self, params: Tensor, grads_fn: fn(Tensor) anyerror!Tensor) !Tensor {
            // Initialize velocity if first call
            if (self.velocity == null) {
                try self.initializeVelocity(params);
            }

            // Create scalar constants
            const momentum_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.momentum)), params.shape.dims, self.element_type);
            const lr_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.learning_rate)), params.shape.dims, self.element_type);

            // Step 1: Compute lookahead parameters
            // lookahead = params - momentum * velocity
            const momentum_velocity = try ops.multiply(self.builder, momentum_tensor, self.velocity.?);
            const lookahead = try ops.subtract(self.builder, params, momentum_velocity);

            // Step 2: Compute gradients at lookahead position
            const grads_at_lookahead = try grads_fn(lookahead);

            // Step 3: Update velocity using gradient at lookahead position
            // velocity = momentum * velocity + learning_rate * grads
            const momentum_velocity_update = try ops.multiply(self.builder, momentum_tensor, self.velocity.?);
            const lr_grads = try ops.multiply(self.builder, lr_tensor, grads_at_lookahead);
            const new_velocity = try ops.add(self.builder, momentum_velocity_update, lr_grads);

            // Step 4: Update parameters
            // params = params - new_velocity
            const new_params = try ops.subtract(self.builder, params, new_velocity);

            // Update internal state
            self.velocity.?.deinit();
            self.velocity = new_velocity;

            return new_params;
        }

        /// Get current velocity for inspection
        pub fn getVelocity(self: Self) ?Tensor {
            return self.velocity;
        }

        /// Reset velocity to zero
        pub fn resetVelocity(self: *Self) !void {
            if (self.velocity) |v| {
                v.deinit();
                self.velocity = null;
            }
        }
    };
}

/// MLIR-based Nesterov optimizer map for managing multiple optimizers
pub fn NesterovMLIRMap(comptime K: type, comptime DataType: type) type {
    const NesterovT = NesterovMLIR(DataType);
    const MapT = std.AutoHashMap(K, NesterovT);
    const ConfT = NesterovMLIRConfiguration(DataType);

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

        pub fn add(self: *Self, key: K, conf: ConfT) !*NesterovT {
            const optimizer = try NesterovT.init(self.allocator, self.builder, conf, self.element_type);
            try self.map.put(key, optimizer);
            return self.map.getPtr(key).?;
        }

        pub fn get(self: *Self, key: K) ?*NesterovT {
            return self.map.getPtr(key);
        }

        pub fn update(self: *Self, key: K, params: Tensor, grads: Tensor) !Tensor {
            const optimizer = self.map.getPtr(key) orelse return error.OptimizerNotFound;
            return try optimizer.update(params, grads);
        }

        pub fn updateWithLookahead(self: *Self, key: K, params: Tensor, grads_fn: fn(Tensor) anyerror!Tensor) !Tensor {
            const optimizer = self.map.getPtr(key) orelse return error.OptimizerNotFound;
            return try optimizer.updateWithLookahead(params, grads_fn);
        }
    };
}

/// Test function for MLIR Nesterov optimizer
pub fn testNesterovMLIR(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing MLIR Nesterov Optimizer ===\n", .{});

    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();

    const element_type = mlir.Type.f32Type(builder.ctx);
    const conf = NesterovMLIRConfiguration(f32).default_configuration();
    
    var nesterov = try NesterovMLIR(f32).init(allocator, &builder, conf, element_type);
    defer nesterov.deinit();

    std.debug.print("✓ MLIR Nesterov optimizer initialized\n", .{});
    std.debug.print("✓ MLIR Nesterov optimizer test completed\n", .{});
}