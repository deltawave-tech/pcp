/// Demo Backend - Simulates distributed training execution for demonstration
/// This backend provides realistic behavior without requiring actual MLIR compilation

const std = @import("std");
const Allocator = std.mem.Allocator;
const worker_backend = @import("worker_backend.zig");
const execution = @import("../execution.zig");
const mlir = @import("../mlir.zig");
const tensor = @import("../tensor.zig");
const mlir_ctx = @import("../mlir_ctx.zig");
const ops = @import("../ops.zig");

const WorkerBackend = worker_backend.WorkerBackend;
const Executor = execution.Executor;

/// Demo backend that simulates MLIR training execution
pub const DemoBackend = struct {
    allocator: Allocator,
    worker_id: u32,
    execution_count: u32,
    real_executor: ?Executor,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) !*Self {
        const backend = try allocator.create(Self);
        backend.* = Self{
            .allocator = allocator,
            .worker_id = @intCast(@mod(std.time.timestamp(), 1000)),
            .execution_count = 0,
            .real_executor = null,
        };
        return backend;
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }
    
    pub fn setRealExecutor(self: *Self, executor: Executor) void {
        self.real_executor = executor;
    }
    
    pub fn asExecutor(self: *Self) Executor {
        return Executor{
            .ptr = self,
            .vtable = &.{
                .materialize = demoMaterialize,
                .materialize_module = demoMaterializeModule,
                .getContext = demoGetContext,
                .deinit = demoExecutorDeinit,
            },
        };
    }
    
    pub fn asWorkerBackend(self: *Self) WorkerBackend {
        return WorkerBackend{
            .ptr = self,
            .vtable = &.{
                .executeTrainingStep = executeTrainingStep,
                .deinit = deinitInterface,
            },
        };
    }
    
    /// Simulate training step execution with realistic timing and results
    fn executeTrainingStep(ptr: *anyopaque, _: []const u8, inputs: [][]const u8, _: [][]const i64) anyerror![][]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.execution_count += 1;
        
        std.log.info("Demo Backend Worker-{}: Executing training step #{}", .{ self.worker_id, self.execution_count });
        
        // Simulate realistic execution time (GPU compilation + execution)
        const execution_time_ms = 50 + @mod(self.execution_count * 17, 100); // 50-150ms variation
        std.time.sleep(execution_time_ms * std.time.ns_per_ms);
        
        // Validate inputs
        if (inputs.len < 3) {
            return error.InvalidInputs;
        }
        
        const params_bytes = inputs[0];
        const input_data = inputs[1]; 
        const target_data = inputs[2];
        
        std.log.info("Demo Backend: Processing {} param bytes, {} input bytes, {} target bytes", .{
            params_bytes.len, input_data.len, target_data.len
        });
        
        // Simulate parameter updates (realistic parameter evolution)
        const updated_params = try self.simulateParameterUpdate(params_bytes);
        
        // Simulate loss computation (realistic loss decay)
        const loss = try self.simulateLossComputation();
        
        // Package outputs: [updated_params, loss]
        const outputs = try self.allocator.alloc([]u8, 2);
        outputs[0] = updated_params;
        outputs[1] = loss;
        
        std.log.info("Demo Backend Worker-{}: Completed training step (simulated loss: {d:.4})", .{ 
            self.worker_id, 
            @as(f32, @bitCast(std.mem.readInt(u32, loss[0..4], .little)))
        });
        
        return outputs;
    }
    
    /// Simulate realistic parameter updates
    fn simulateParameterUpdate(self: *Self, original_params: []const u8) ![]u8 {
        const param_count = original_params.len / @sizeOf(f32);
        const params: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, original_params));
        
        const updated = try self.allocator.alloc(f32, param_count);
        
        // Simulate gradient descent with noise
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp() + self.worker_id));
        const learning_rate = 0.01;
        
        for (params, updated, 0..) |param, *updated_param, i| {
            // Simulate gradient with worker-specific variation
            const gradient = rng.random().floatNorm(f32) * 0.1 + @sin(@as(f32, @floatFromInt(i + self.execution_count)) * 0.1) * 0.05;
            updated_param.* = param - learning_rate * gradient;
        }
        
        return std.mem.sliceAsBytes(updated);
    }
    
    /// Simulate realistic loss computation with decay
    fn simulateLossComputation(self: *Self) ![]u8 {
        // Simulate realistic loss decay with noise
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp() + self.worker_id * 7));
        
        // Base loss starts around 2.0 and decays
        const base_loss = 2.0 * std.math.exp(-@as(f32, @floatFromInt(self.execution_count)) * 0.05);
        const noise = rng.random().floatNorm(f32) * 0.1;
        const worker_variation = (@as(f32, @floatFromInt(self.worker_id % 10)) - 5.0) * 0.02;
        
        const loss = @max(0.01, base_loss + noise + worker_variation);
        
        const loss_bytes = try self.allocator.alloc(u8, @sizeOf(f32));
        std.mem.writeInt(u32, loss_bytes[0..4], @bitCast(loss), .little);
        
        return loss_bytes;
    }
    
    fn getContextInterface(ptr: *anyopaque) mlir.Context {
        return demoGetContext(ptr);
    }
    
    fn deinitInterface(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
    
    // Executor interface implementations that delegate to the real executor
    fn demoMaterialize(ptr: *anyopaque, t: tensor.Tensor(void)) ![]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (self.real_executor) |real| {
            return real.materialize(t);
        } else {
            return error.NoRealExecutor;
        }
    }

    fn demoMaterializeModule(ptr: *anyopaque, module: mlir.Module) ![]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (self.real_executor) |real| {
            return real.materializeModule(module);
        } else {
            // Fallback to simulation for demo
            return simulateModuleSerialization(self.allocator, "demo execution");
        }
    }

    fn demoGetContext(ptr: *anyopaque) mlir.Context {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (self.real_executor) |real| {
            return real.getContext();
        } else {
            @panic("Demo backend requires real executor for MLIR context");
        }
    }

    fn demoExecutorDeinit(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
};

/// Create a realistic demo MLIR module using actual MLIR infrastructure
pub fn simulateModuleSerialization(allocator: Allocator, description: []const u8) ![]u8 {
    // Create a real MLIR context and builder to generate an actual module
    var ctx = mlir.Context.init() catch {
        // Fallback to simple string if MLIR initialization fails
        return try std.fmt.allocPrint(allocator, "// Demo fallback: {s} (MLIR unavailable)", .{description});
    };
    defer ctx.deinit();
    
    // Allow unregistered dialects for flexibility in demo
    const c_api = @import("../mlir/c.zig").c;
    c_api.contextSetAllowUnregisteredDialects(ctx.handle, true);
    
    var builder = ops.MLIRBuilder.init(allocator, ctx) catch {
        // Fallback if builder creation fails
        return try std.fmt.allocPrint(allocator, "// Demo fallback: {s} (Builder unavailable)", .{description});
    };
    defer builder.deinit();
    
    // Build a realistic demo training module with actual MLIR operations
    try buildDemoTrainingModule(&builder, description);
    
    // Use the real serialization function from mlir_ctx
    return mlir_ctx.serializeMLIRModule(allocator, builder.module);
}

/// Build a realistic demo training module with actual MLIR operations
fn buildDemoTrainingModule(builder: *ops.MLIRBuilder, description: []const u8) !void {
    const f32_type = mlir.Type.f32Type(builder.ctx);
    const param_count = 1000;
    
    // Define tensor types
    const params_type = mlir.Type.rankedTensorType(builder.ctx, &.{param_count}, f32_type);
    const data_type = mlir.Type.rankedTensorType(builder.ctx, &.{4, 12}, f32_type);
    const loss_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type); // Scalar loss
    
    // Create forward+loss function
    const forward_input_types = [_]mlir.Type{ params_type, data_type, data_type };
    const forward_result_types = [_]mlir.Type{loss_type};
    const forward_func_type = mlir.Type.functionType(builder.ctx, &forward_input_types, &forward_result_types);
    
    const forward_result = try builder.createFunction("forward_and_loss_fn", forward_func_type);
    const forward_func = forward_result.func_op;
    const forward_block = forward_result.entry_block;
    
    // Build forward function body
    const original_block = builder.getInsertionBlock();
    builder.setInsertionBlock(forward_block);
    
    // Get function arguments
    const params_arg = forward_block.getArgument(0);
    const inputs_arg = forward_block.getArgument(1);
    const targets_arg = forward_block.getArgument(2);
    
    // Create simple loss computation: MSE between prediction and targets
    const param_tensor = try builder.newTensor(params_arg);
    const input_tensor = try builder.newTensor(inputs_arg);
    const target_tensor = try builder.newTensor(targets_arg);
    
    // Flatten inputs to match param dimensions for demo
    const input_flat = try ops.reshape(builder, input_tensor, &.{param_count});
    const target_flat = try ops.reshape(builder, target_tensor, &.{param_count});
    
    // Simple prediction: params + inputs (demo computation)
    const prediction = try ops.add(builder, param_tensor, input_flat);
    
    // MSE loss: (prediction - target)^2
    const diff = try ops.subtract(builder, prediction, target_flat);
    const squared = try ops.multiply(builder, diff, diff);
    const loss = try ops.reduceSum(builder, squared, &.{0}, false);
    
    // Return loss
    _ = try builder.createAndAttach("func.return", &.{loss.value}, &.{}, .{});
    
    // Restore insertion block and attach function to module
    builder.setInsertionBlock(original_block);
    builder.module_body.appendOwnedOperation(forward_func);
    
    // Add a comment attribute to identify this as a demo module
    const comment_attr = mlir.Attribute.stringAttr(builder.ctx, description);
    _ = @import("../mlir/c.zig").c.operationSetAttributeByName(builder.module.op().handle, "demo.comment", comment_attr.handle);
}