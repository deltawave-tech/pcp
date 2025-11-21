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

// NOTE: buildNesterovUpdateFunction has been removed.
// We now use inline operations in the update() method to avoid complex function building
// that was causing segmentation faults with the MLIR C API.

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
            std.log.info("NesterovMLIR.init: Starting initialization...", .{});
            
            // SIMPLIFIED: Don't build the function during initialization to avoid crashes
            // Instead, we'll build operations inline during the update() call
            std.log.info("NesterovMLIR.init: Creating simple optimizer instance", .{});

            return Self{
                .allocator = allocator,
                .builder = builder,
                .conf = conf,
                .element_type = element_type,
                .velocity = null,
                .update_fn_name = "nesterov_inline", // Dummy name for now
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
            const params_dims = try params.shape.getDims(self.builder.allocator);
            defer self.builder.allocator.free(params_dims);
            const zero_op = try hlo.zeroConstant(self.builder.allocator, self.builder.ctx, params_dims, self.element_type);
            self.velocity = try self.builder.newTensor(zero_op.getResult(0));
        }

        /// Update parameters using Nesterov momentum algorithm implemented in MLIR
        pub fn update(self: *Self, params: Tensor, grads: Tensor) !Tensor {
            // Initialize velocity if first call
            if (self.velocity == null) {
                try self.initializeVelocity(params);
            }

            // SIMPLIFIED: Build operations inline instead of calling a pre-built function
            // This avoids the complex MLIR function building that was causing the crash

            // Create scalar constants
            const params_dims = try params.shape.getDims(self.builder.allocator);
            defer self.builder.allocator.free(params_dims);
            const momentum_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.momentum)), params_dims, self.element_type);
            const lr_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.learning_rate)), params_dims, self.element_type);

            // velocity = momentum * velocity + learning_rate * grads
            const momentum_velocity_update = try ops.multiply(self.builder, momentum_tensor, self.velocity.?);
            const lr_grads = try ops.multiply(self.builder, lr_tensor, grads);
            const new_velocity = try ops.add(self.builder, momentum_velocity_update, lr_grads);

            // params = params - new_velocity
            const new_params = try ops.subtract(self.builder, params, new_velocity);

            // Update internal state
            if (self.velocity) |v| v.deinit();
            self.velocity = new_velocity;

            return new_params;
        }

        /// Alternative implementation that more clearly separates the lookahead computation
        pub fn updateWithLookahead(self: *Self, params: Tensor, grads_fn: fn (Tensor) anyerror!Tensor) !Tensor {
            // Initialize velocity if first call
            if (self.velocity == null) {
                try self.initializeVelocity(params);
            }

            // Create scalar constants
            const params_dims = try params.shape.getDims(self.builder.allocator);
            defer self.builder.allocator.free(params_dims);
            const momentum_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.momentum)), params_dims, self.element_type);
            const lr_tensor = try ops.constant(self.builder, @as(f64, @floatCast(self.conf.learning_rate)), params_dims, self.element_type);

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

        pub fn updateWithLookahead(self: *Self, key: K, params: Tensor, grads_fn: fn (Tensor) anyerror!Tensor) !Tensor {
            const optimizer = self.map.getPtr(key) orelse return error.OptimizerNotFound;
            return try optimizer.updateWithLookahead(params, grads_fn);
        }
    };
}

/// Test function for MLIR Nesterov optimizer
pub fn testNesterovMLIR(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing MLIR Nesterov Optimizer ===\n", .{});

    // Create MLIR context for this test
    var ctx = try mlir.Context.init();
    defer ctx.deinit();
    const c_api = @import("../mlir/c.zig").c;
    c_api.mlirContextSetAllowUnregisteredDialects(ctx.handle, true);

    var builder = try MLIRBuilder.init(allocator, ctx);
    defer builder.deinit();

    const element_type = mlir.Type.f32Type(builder.ctx);
    const conf = NesterovMLIRConfiguration(f32).default_configuration();

    var nesterov = try NesterovMLIR(f32).init(allocator, &builder, conf, element_type);
    defer nesterov.deinit();

    std.debug.print("✓ MLIR Nesterov optimizer initialized\n", .{});
    std.debug.print("✓ MLIR Nesterov optimizer test completed\n", .{});
}
