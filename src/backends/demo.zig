/// Demo Backend - Simulates distributed training execution for demonstration
/// This backend provides realistic behavior without requiring actual MLIR compilation

const std = @import("std");
const Allocator = std.mem.Allocator;
const worker_backend = @import("worker_backend.zig");
const execution = @import("../execution.zig");
const mlir = @import("../mlir.zig");
const tensor = @import("../tensor.zig");

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
    fn executeTrainingStep(ptr: *anyopaque, _: mlir.Module, inputs: [][]const u8) anyerror![][]u8 {
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

/// Module serialization simulation for demo
pub fn simulateModuleSerialization(allocator: Allocator, description: []const u8) ![]u8 {
    // Create a realistic "serialized" MLIR module for demo
    const module_data = try std.fmt.allocPrint(allocator, 
        \\// Demo MLIR Module: {s}
        \\module {{
        \\  func.func @forward_and_loss_fn(%params: tensor<1000xf32>, %inputs: tensor<4x12xf32>, %targets: tensor<4x12xf32>) -> tensor<f32> {{
        \\    // Simulated forward pass
        \\    %loss = "demo.compute_loss"(%params, %inputs, %targets) : (tensor<1000xf32>, tensor<4x12xf32>, tensor<4x12xf32>) -> tensor<f32>
        \\    return %loss : tensor<f32>
        \\  }}
        \\  
        \\  func.func @forward_and_loss_fn_grad(%params: tensor<1000xf32>, %inputs: tensor<4x12xf32>, %targets: tensor<4x12xf32>, %seed: tensor<f32>) -> (tensor<1000xf32>, tensor<4x12xf32>, tensor<4x12xf32>) {{
        \\    // Simulated gradient computation  
        \\    %grad_params, %grad_inputs, %grad_targets = "demo.compute_gradients"(%params, %inputs, %targets, %seed) : (tensor<1000xf32>, tensor<4x12xf32>, tensor<4x12xf32>, tensor<f32>) -> (tensor<1000xf32>, tensor<4x12xf32>, tensor<4x12xf32>)
        \\    return %grad_params, %grad_inputs, %grad_targets : tensor<1000xf32>, tensor<4x12xf32>, tensor<4x12xf32>
        \\  }}
        \\  
        \\  func.func @main(%initial_params: tensor<1000xf32>, %inputs: tensor<4x12xf32>, %targets: tensor<4x12xf32>) -> (tensor<1000xf32>, tensor<f32>) {{
        \\    // Simulated main training function
        \\    %loss = call @forward_and_loss_fn(%initial_params, %inputs, %targets) : (tensor<1000xf32>, tensor<4x12xf32>, tensor<4x12xf32>) -> tensor<f32>
        \\    %one = arith.constant 1.0 : tensor<f32>
        \\    %grad_params, %grad_inputs, %grad_targets = call @forward_and_loss_fn_grad(%initial_params, %inputs, %targets, %one) : (tensor<1000xf32>, tensor<4x12xf32>, tensor<4x12xf32>, tensor<f32>) -> (tensor<1000xf32>, tensor<4x12xf32>, tensor<4x12xf32>)
        \\    %updated_params = "demo.apply_gradients"(%initial_params, %grad_params) : (tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
        \\    return %updated_params, %loss : tensor<1000xf32>, tensor<f32>
        \\  }}
        \\}}
        \\// Demo module size: {} bytes
    , .{ description, 1000 }); // Fixed size since module_data is not yet defined
    
    return module_data;
}