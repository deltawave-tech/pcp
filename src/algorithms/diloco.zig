/// DiLoCo (Distributed Low-Communication) Algorithm Implementation
/// Implements the DiLoCo training algorithm for distributed learning with real MLIR optimizers

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const training_algorithm = @import("training_algorithm.zig");
const shepherd = @import("../controllers/shepherd.zig");
const message = @import("../network/message.zig");
const nesterov_mlir = @import("../optimizers/nesterov_mlir.zig");
const autodiff = @import("../autodiff.zig");
const ops = @import("../ops.zig");
const mlir = @import("../mlir.zig");
const tensor = @import("../tensor.zig");
const execution = @import("../execution.zig"); // NEW import
const monitoring = @import("../monitoring.zig");

const TrainingAlgorithm = training_algorithm.TrainingAlgorithm;
const TrainingStatus = training_algorithm.TrainingStatus;
const TrainingConfig = training_algorithm.TrainingConfig;
const TrainingMetrics = training_algorithm.TrainingMetrics;
const Shepherd = shepherd.Shepherd;
const MessageType = message.MessageType;
const NesterovMLIR = nesterov_mlir.NesterovMLIR(f32);
const MLIRBuilder = ops.MLIRBuilder;
const Tensor = tensor.Tensor(void);
const Executor = execution.Executor; // NEW: Generic executor interface

/// DiLoCo algorithm configuration
pub const DiLoCoConfig = struct {
    base_config: TrainingConfig,
    tau: usize, // Inner loop steps
    nesterov_momentum: f32,
    parameter_averaging: bool,
    param_count: usize, // Number of model parameters
    
    pub fn default() DiLoCoConfig {
        return DiLoCoConfig{
            .base_config = TrainingConfig.default(),
            .tau = 10, // Default inner loop steps
            .nesterov_momentum = 0.9,
            .parameter_averaging = true,
            .param_count = 1000, // Default parameter count
        };
    }
};

/// DiLoCo algorithm implementation with real MLIR optimizers
/// Now backend-agnostic using the generic Executor interface
pub const DiLoCo = struct {
    allocator: Allocator,
    coordinator: *Shepherd,
    config: DiLoCoConfig,
    status: TrainingStatus,
    metrics: TrainingMetrics,
    current_epoch: usize,
    
    // MLIR infrastructure
    mlir_builder: *MLIRBuilder,
    nesterov_optimizer: *NesterovMLIR,
    element_type: mlir.Type,
    
    // Master parameters as MLIR tensors
    master_parameters: ?Tensor,
    parameter_shape: []const i64,
    
    // NEW: Generic executor - replaces direct backend knowledge
    executor: Executor,
    
    const Self = @This();
    
    // Change the init signature to accept the generic executor
    pub fn init(allocator: Allocator, coordinator: *Shepherd, config: DiLoCoConfig, executor: Executor) !Self {
        // Initialize MLIR infrastructure
        const mlir_builder = try allocator.create(MLIRBuilder);
        mlir_builder.* = try MLIRBuilder.init(allocator);
        
        const element_type = mlir.Type.f32Type(mlir_builder.ctx);
        
        // Configure and create Nesterov optimizer
        const nesterov_config = nesterov_mlir.NesterovMLIRConfiguration(f32){
            .learning_rate = config.base_config.learning_rate,
            .momentum = config.nesterov_momentum,
        };
        
        const nesterov_optimizer = try allocator.create(NesterovMLIR);
        nesterov_optimizer.* = try NesterovMLIR.init(allocator, mlir_builder, nesterov_config, element_type);
        
        // Set up parameter shape
        const parameter_shape = try allocator.alloc(i64, 1);
        parameter_shape[0] = @intCast(config.param_count);
        
        return Self{
            .allocator = allocator,
            .coordinator = coordinator,
            .config = config,
            .status = .not_started,
            .metrics = TrainingMetrics.init(),
            .current_epoch = 0,
            .mlir_builder = mlir_builder,
            .nesterov_optimizer = nesterov_optimizer,
            .element_type = element_type,
            .master_parameters = null,
            .parameter_shape = parameter_shape,
            .executor = executor, // Store the generic executor
        };
    }
    
    pub fn deinit(self: *Self) void {
        if (self.master_parameters) |params| {
            params.deinit();
        }
        
        self.nesterov_optimizer.deinit();
        self.allocator.destroy(self.nesterov_optimizer);
        
        self.mlir_builder.deinit();
        self.allocator.destroy(self.mlir_builder);
        
        self.allocator.free(self.parameter_shape);
    }
    
    /// Get TrainingAlgorithm interface
    pub fn asTrainingAlgorithm(self: *Self) TrainingAlgorithm {
        return TrainingAlgorithm{
            .ptr = self,
            .vtable = &.{
                .run = run,
                .deinit = deinitInterface,
                .getName = getName,
                .getStatus = getStatus,
            },
        };
    }
    
    /// Main DiLoCo training loop
    pub fn run(ptr: *anyopaque) !void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        
        self.status = .initializing;
        monitoring.setStatus(.initializing);
        monitoring.setModelInfo(self.config.param_count, self.config.base_config.learning_rate);
        std.log.info("Starting DiLoCo training algorithm with MLIR optimizers", .{});
        
        // Initialize master parameters
        try self.initializeMasterParameters();
        
        self.status = .running;
        monitoring.setStatus(.running);
        
        // Main outer loop
        for (0..self.config.base_config.outer_loop_steps) |outer_step| {
            const start_time = std.time.milliTimestamp();
            std.log.info("DiLoCo outer loop step {}/{}", .{ outer_step + 1, self.config.base_config.outer_loop_steps });
            
            // Phase 1: Broadcast master parameters to all workers
            try self.broadcastMasterParameters();
            
            // Phase 2: Collect results from workers after inner loop
            const worker_results = try self.collectWorkerResults();
            defer self.cleanupWorkerResults(worker_results);
            
            // Phase 3: Update master parameters using MLIR Nesterov optimizer
            try self.updateMasterParametersMLIR(worker_results);
            
            // Update metrics
            self.metrics.outer_loop_count += 1;
            self.current_epoch += 1;
            
            // Calculate epoch time and update monitoring
            const end_time = std.time.milliTimestamp();
            const epoch_time_ms: u64 = @intCast(end_time - start_time);
            monitoring.setEpochTime(epoch_time_ms);
            monitoring.setMetrics(self.current_epoch, self.metrics.loss, self.coordinator.getWorkerCount());
            
            // Check convergence or stopping conditions
            if (self.shouldStop()) {
                break;
            }
        }
        
        self.status = .completed;
        monitoring.setStatus(.completed);
        std.log.info("DiLoCo training completed", .{});
    }
    
    /// Initialize master parameters using MLIR
    fn initializeMasterParameters(self: *Self) !void {
        // Create initial parameter values
        const param_data = try self.allocator.alloc(f32, self.config.param_count);
        defer self.allocator.free(param_data);
        
        // Initialize with Xavier/Glorot initialization
        var rng = std.Random.DefaultPrng.init(12345);
        const scale = std.math.sqrt(2.0 / @as(f32, @floatFromInt(self.config.param_count)));
        
        for (param_data) |*param| {
            param.* = rng.random().floatNorm(f32) * scale;
        }
        
        // Convert to MLIR tensor
        self.master_parameters = try self.createTensorFromArray(param_data);
        
        std.log.info("Initialized master parameters with {} elements using Xavier initialization", .{self.config.param_count});
    }
    
    /// Create a dedicated forward+loss function that will be differentiated
    /// This is a well-formed func.func operation that autodiff can process
    fn buildForwardAndLossFunction(self: *Self, builder: *ops.MLIRBuilder) !mlir.Operation {
        const f32_type = mlir.Type.f32Type(builder.ctx);
        const param_count = self.config.param_count;
        
        // Define types
        const params_type = mlir.Type.rankedTensorType(builder.ctx, &.{@intCast(param_count)}, f32_type);
        const data_type = mlir.Type.rankedTensorType(builder.ctx, &.{4, 12}, f32_type);
        const loss_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type); // Scalar loss
        
        // Define the function type: func(params, inputs, targets) -> (loss)
        const input_types = [_]mlir.Type{ params_type, data_type, data_type };
        const result_types = [_]mlir.Type{ loss_type };
        const func_type = mlir.Type.functionType(builder.ctx, &input_types, &result_types);
        
        // Create the func.func operation with attributes
        const func_op = mlir.Operation.create(builder.ctx, "func.func", .{
            .operands = &.{},
            .results = &.{},
            .attributes = &.{
                .{ "function_type", mlir.Attribute.typeAttr(func_type) },
                .{ "sym_name", mlir.Attribute.stringAttr(builder.ctx, "forward_and_loss_fn") },
            },
            .location = builder.loc,
        });
        
        // Create a block for the function body
        const region = func_op.getRegion(0);
        const block = try builder.createBlock();
        region.appendOwnedBlock(block);
        
        // Add block arguments
        const params_arg = block.addArgument(params_type, builder.loc);
        const inputs_arg = block.addArgument(data_type, builder.loc);
        _ = block.addArgument(data_type, builder.loc); // targets_arg (unused in simple example)
        
        // Build the forward/loss computation inside this function
        const param_tensor = try builder.newTensor(params_arg);
        const input_flat = try ops.reshape(builder, try builder.newTensor(inputs_arg), &.{@intCast(param_count)});
        
        // Simple MSE loss computation
        const diff = try ops.subtract(builder, param_tensor, input_flat);
        const squared = try ops.multiply(builder, diff, diff);
        const loss = try ops.reduceSum(builder, squared, &.{0});
        
        // Return the loss
        _ = try builder.createAndAttach("func.return", &.{loss.value}, &.{});
        
        return func_op;
    }

    /// Build the complete worker training graph as MLIR module using Function-as-a-Unit pattern
    /// This creates: forward_fn -> autodiff -> main_fn that orchestrates the full training step
    fn buildWorkerTrainingGraph(self: *Self) ![]u8 {
        std.log.info("DiLoCo: Building REAL autodiff-enabled worker training graph with Function-as-a-Unit pattern...", .{});
        var builder = try ops.MLIRBuilder.init(self.allocator);
        defer builder.deinit();

        // === Part 1: Create the forward+loss function to be differentiated ===
        std.log.info("Creating forward+loss function...", .{});
        const forward_fn = try self.buildForwardAndLossFunction(&builder);
        
        // === Part 2: Differentiate the forward function ===  
        std.log.info("Differentiating forward function with autodiff.buildGradientGraph...", .{});
        _ = try autodiff.buildGradientGraph(self.allocator, &builder, forward_fn);
        
        // The autodiff system will name the gradient function predictably
        // For now, we'll use the convention that it names it "forward_and_loss_fn_grad"
        const grad_fn_name = "forward_and_loss_fn_grad";
        
        // === Part 3: Create the main worker function that orchestrates everything ===
        std.log.info("Creating main worker orchestration function...", .{});
        
        // Define types for main function
        const f32_type = mlir.Type.f32Type(builder.ctx);
        const param_count = self.config.param_count;
        const params_type = mlir.Type.rankedTensorType(builder.ctx, &.{@intCast(param_count)}, f32_type);
        const data_type = mlir.Type.rankedTensorType(builder.ctx, &.{4, 12}, f32_type);
        const loss_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type);
        
        // Create main function: func(params, inputs, targets) -> (new_params, loss)
        const main_input_types = [_]mlir.Type{ params_type, data_type, data_type };
        const main_result_types = [_]mlir.Type{ params_type, loss_type };
        const main_func_type = mlir.Type.functionType(builder.ctx, &main_input_types, &main_result_types);
        
        const main_func_op = mlir.Operation.create(builder.ctx, "func.func", .{
            .operands = &.{},
            .results = &.{},
            .attributes = &.{
                .{ "function_type", mlir.Attribute.typeAttr(main_func_type) },
                .{ "sym_name", mlir.Attribute.stringAttr(builder.ctx, "main") },
            },
            .location = builder.loc,
        });
        
        // Create main function body
        const main_region = main_func_op.getRegion(0);
        const main_block = try builder.createBlock();
        main_region.appendOwnedBlock(main_block);
        
        // Add main function arguments
        const initial_params = main_block.addArgument(params_type, builder.loc);
        const inputs = main_block.addArgument(data_type, builder.loc);
        const targets = main_block.addArgument(data_type, builder.loc);
        
        // Call the forward function to get the loss
        const forward_call_op = mlir.Operation.create(builder.ctx, "func.call", .{
            .operands = &.{initial_params, inputs, targets},
            .results = &.{loss_type},
            .attributes = &.{
                .{ "callee", mlir.Attribute.stringAttr(builder.ctx, "forward_and_loss_fn") },
            },
            .location = builder.loc,
        });
        const loss_val = forward_call_op.getResult(0);
        
        // Call the gradient function to get gradients
        const one = try ops.constant(&builder, 1.0, &.{}, f32_type); // Scalar 1.0 for loss gradient seed
        const grad_call_op = mlir.Operation.create(builder.ctx, "func.call", .{
            .operands = &.{initial_params, inputs, targets, one.value},
            .results = &.{params_type, data_type, data_type},
            .attributes = &.{
                .{ "callee", mlir.Attribute.stringAttr(builder.ctx, grad_fn_name) },
            },
            .location = builder.loc,
        });
        const param_grads = grad_call_op.getResult(0); // Gradients w.r.t. parameters
        
        // Apply simple gradient descent update
        const learning_rate = try ops.constant(&builder, 0.01, &.{@intCast(param_count)}, f32_type);
        const grad_scaled = try ops.multiply(&builder, try builder.newTensor(param_grads), learning_rate);
        const updated_params = try ops.subtract(&builder, try builder.newTensor(initial_params), grad_scaled);
        
        // Return both updated parameters and loss from main function
        _ = try builder.createAndAttach("func.return", &.{updated_params.value, loss_val}, &.{});
        
        // === Part 4: Debug and serialize the complete module ===
        std.log.info("Complete worker module with {} functions created successfully", .{3}); // forward_fn, grad_fn, main
        
        // Dump the module for debugging
        builder.module.op().dump();
        
        const serialized = try @import("../mlir_ctx.zig").serializeMLIRModule(self.allocator, builder.module);
        
        std.log.info("✓ DiLoCo REAL autodiff worker training graph built and serialized ({} bytes).", .{serialized.len});
        return serialized;
    }
    
    /// Broadcast master parameters and training graph to all workers
    fn broadcastMasterParameters(self: *Self) !void {
        if (self.master_parameters == null) {
            return error.ParametersNotInitialized;
        }
        
        std.log.info("Materializing master parameters for broadcast...", .{});
        
        // Materialize parameters to bytes
        const serialized_params = try self.executor.materialize(self.master_parameters.?);
        defer self.allocator.free(serialized_params);
        
        // Build the complete worker training graph
        const worker_graph_str = try self.buildWorkerTrainingGraph();
        defer self.allocator.free(worker_graph_str);
        
        // Create simple delimiter-based payload to avoid complex JSON parsing
        var payload_list = std.ArrayList(u8).init(self.allocator);
        defer payload_list.deinit();
        try payload_list.appendSlice(worker_graph_str);
        try payload_list.appendSlice("|||PARAMS|||"); // Delimiter
        try payload_list.appendSlice(serialized_params);
        const payload_bytes = try payload_list.toOwnedSlice();
        defer self.allocator.free(payload_bytes);
        
        const json_payload = std.json.Value{.string = payload_bytes};
        
        // Broadcast StartInnerLoop message with graph + parameters
        try self.coordinator.broadcastToWorkers(MessageType.START_INNER_LOOP, json_payload);
        
        std.log.info("Training graph + parameters broadcasted to {} workers", .{self.coordinator.getWorkerCount()});
    }
    
    /// Collect results from workers after inner loop
    fn collectWorkerResults(self: *Self) !ArrayList(WorkerResult) {
        std.log.info("Collecting results from workers", .{});
        
        // Collect InnerLoopComplete messages from all workers
        const responses = try self.coordinator.collectFromWorkers(MessageType.INNER_LOOP_COMPLETE);
        defer responses.deinit();
        
        var results = ArrayList(WorkerResult).init(self.allocator);
        
        for (responses.items) |response| {
            // Parse worker result from message - new format with delimiter
            const data_bytes = switch (response.data) {
                .string => |s| s,
                else => return error.InvalidMessageFormat,
            };
            
            const delimiter = "|||LOSS|||";
            const split_point = std.mem.indexOf(u8, data_bytes, delimiter) orelse return error.InvalidPayload;

            const params = try self.allocator.dupe(u8, data_bytes[0..split_point]);
            const loss = try self.allocator.dupe(u8, data_bytes[split_point + delimiter.len..]);

            const result = WorkerResult{
                .node_id = response.sender_node,
                .parameter_bytes = params,
                .loss_bytes = loss,
                .steps_completed = self.config.tau,
            };
            
            try results.append(result);
        }
        
        std.log.info("Collected results from {} workers", .{results.items.len});
        return results;
    }
    
    
    /// Update master parameters using MLIR Nesterov optimizer
    fn updateMasterParametersMLIR(self: *Self, worker_results: ArrayList(WorkerResult)) !void {
        if (worker_results.items.len == 0) {
            return error.NoWorkerResults;
        }
        
        std.log.info("Updating master parameters using MLIR Nesterov optimizer with {} worker results", .{worker_results.items.len});
        
        // Average worker parameters
        const averaged_params_bytes = try self.averageWorkerParameterBytes(worker_results);
        defer self.allocator.free(averaged_params_bytes);
        const param_slice: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, averaged_params_bytes));
        const averaged_tensor = try self.createTensorFromArray(param_slice);
        defer averaged_tensor.deinit();
        
        // Compute gradients (difference between averaged and master parameters)
        const gradient_tensor = try ops.subtract(self.mlir_builder, averaged_tensor, self.master_parameters.?);
        defer gradient_tensor.deinit();
        
        // Apply Nesterov momentum update using MLIR optimizer
        const updated_params = try self.nesterov_optimizer.update(self.master_parameters.?, gradient_tensor);
        
        // Replace master parameters with updated ones
        self.master_parameters.?.deinit();
        self.master_parameters = updated_params;
        
        // Update metrics
        self.metrics.loss = self.calculateAverageLoss(worker_results);
        
        std.log.info("Master parameters updated using MLIR Nesterov, average loss: {d:.4}", .{self.metrics.loss});
    }
    
    /// Average parameters from all workers (new byte-based version)
    fn averageWorkerParameterBytes(self: *Self, worker_results: ArrayList(WorkerResult)) ![]u8 {
        const param_count = self.config.param_count;
        const averaged = try self.allocator.alloc(f32, param_count);
        defer self.allocator.free(averaged);
        
        // Initialize to zero
        for (averaged) |*param| {
            param.* = 0.0;
        }
        
        // Sum all worker parameters
        for (worker_results.items) |result| {
            const worker_params: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, result.parameter_bytes));
            for (averaged, 0..) |*avg_param, i| {
                if (i < worker_params.len) {
                    avg_param.* += worker_params[i];
                }
            }
        }
        
        // Divide by number of workers to get average
        const num_workers: f32 = @floatFromInt(worker_results.items.len);
        for (averaged) |*avg_param| {
            avg_param.* /= num_workers;
        }
        
        // Convert back to bytes
        const result_bytes = try self.allocator.alloc(u8, averaged.len * @sizeOf(f32));
        const result_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, result_bytes));
        @memcpy(result_f32, averaged);
        
        return result_bytes;
    }
    
    /// Calculate average loss across workers
    fn calculateAverageLoss(self: *Self, worker_results: ArrayList(WorkerResult)) f32 {
        _ = self;
        var total_loss: f32 = 0.0;
        
        for (worker_results.items) |result| {
            // Parse loss from bytes - assume it's a single f32
            if (result.loss_bytes.len >= @sizeOf(f32)) {
                const loss_value: f32 = @bitCast(std.mem.readInt(u32, result.loss_bytes[0..4], .little));
                total_loss += loss_value;
            }
        }
        
        return total_loss / @as(f32, @floatFromInt(worker_results.items.len));
    }
    
    /// Create MLIR tensor from parameter array
    fn createTensorFromArray(self: *Self, array: []const f32) !Tensor {
        const byte_data = std.mem.sliceAsBytes(array);
        const tensor_type = mlir.Type.rankedTensorType(self.mlir_builder.ctx, self.parameter_shape, self.element_type);
        
        var shape = try tensor.Shape.init(self.allocator, self.parameter_shape, tensor.DType.f32);
        defer shape.deinit();
        const value = try self.mlir_builder.createConstant(byte_data, tensor_type, shape);
        return try self.mlir_builder.newTensor(value);
    }
    
    /// Extract tensor values to array
    fn extractTensorToArray(self: *Self, _: Tensor) ![]f32 {
        // In a real implementation, this would extract values from the MLIR tensor
        // For now, simulate by creating array with current values
        const array = try self.allocator.alloc(f32, self.config.param_count);
        
        // This is a placeholder - real implementation would extract from MLIR tensor
        var rng = std.Random.DefaultPrng.init(@intCast(self.current_epoch));
        for (array) |*param| {
            param.* = rng.random().floatNorm(f32) * 0.1;
        }
        
        return array;
    }
    
    /// Serialize parameters to string
    fn serializeParameters(self: *Self, params: []const f32) ![]u8 {
        var buffer = ArrayList(u8).init(self.allocator);
        defer buffer.deinit();
        
        // Simple binary serialization
        try buffer.appendSlice(std.mem.asBytes(&params.len));
        for (params) |param| {
            try buffer.appendSlice(std.mem.asBytes(&param));
        }
        
        return try self.allocator.dupe(u8, buffer.items);
    }
    
    /// Deserialize parameters from string
    fn deserializeParameters(self: *Self, data: []const u8) ![]f32 {
        if (data.len < @sizeOf(usize)) {
            return error.InvalidData;
        }
        
        const param_count = std.mem.readInt(usize, data[0..@sizeOf(usize)], .little);
        const params = try self.allocator.alloc(f32, param_count);
        
        var offset: usize = @sizeOf(usize);
        for (params) |*param| {
            if (offset + @sizeOf(f32) > data.len) {
                return error.InvalidData;
            }
            param.* = @bitCast(std.mem.readInt(u32, data[offset..offset + @sizeOf(f32)][0..4], .little));
            offset += @sizeOf(f32);
        }
        
        return params;
    }
    
    /// Clean up worker results
    fn cleanupWorkerResults(self: *Self, worker_results: ArrayList(WorkerResult)) void {
        for (worker_results.items) |result| {
            self.allocator.free(result.parameter_bytes);
            self.allocator.free(result.loss_bytes);
        }
        worker_results.deinit();
    }
    
    /// Check if training should stop
    fn shouldStop(self: *Self) bool {
        if (self.current_epoch >= self.config.base_config.max_epochs) {
            return true;
        }
        
        if (self.metrics.loss < 0.01) {
            return true;
        }
        
        return false;
    }
    
    /// Interface implementations
    fn deinitInterface(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
    
    fn getName(ptr: *anyopaque) []const u8 {
        _ = ptr;
        return "DiLoCo-MLIR";
    }
    
    fn getStatus(ptr: *anyopaque) TrainingStatus {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.status;
    }
};

/// Result from a worker after inner loop completion
const WorkerResult = struct {
    node_id: message.NodeId,
    parameter_bytes: []u8,
    loss_bytes: []u8,
    steps_completed: usize,
};

/// Test function for DiLoCo algorithm
pub fn testDiLoCo(allocator: Allocator) !void {
    std.log.info("Testing DiLoCo algorithm with MLIR optimizers...");
    
    // Create a mock coordinator
    var mock_coordinator = shepherd.Shepherd.init(allocator);
    defer mock_coordinator.deinit();
    
    // Create DiLoCo algorithm
    const config = DiLoCoConfig.default();
    var diloco = try DiLoCo.init(allocator, &mock_coordinator, config);
    defer diloco.deinit();
    
    // Test initialization
    try std.testing.expectEqual(TrainingStatus.not_started, diloco.status);
    try std.testing.expectEqual(@as(usize, 0), diloco.current_epoch);
    try std.testing.expectEqual(@as(usize, 1000), diloco.config.param_count);
    
    std.log.info("✓ DiLoCo MLIR algorithm test completed");
}