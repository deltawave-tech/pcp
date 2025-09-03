/// DiLoCo (Distributed Low-Communication) Algorithm Implementation
/// Implements the DiLoCo training algorithm for distributed learning with real MLIR optimizers
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const training_algorithm = @import("training_algorithm.zig");
const shepherd = @import("../controllers/shepherd.zig");
const message = @import("../network/message.zig");
const binary_protocol = @import("../network/capnp_zig_wrapper.zig");
const nesterov_mlir = @import("../optimizers/nesterov_mlir.zig");
const autodiff = @import("../autodiff.zig");
const ops = @import("../ops.zig");
const mlir = @import("../mlir.zig");
const tensor = @import("../tensor.zig");
const execution = @import("../execution.zig");
const monitoring = @import("../monitoring.zig");
const gpt2 = @import("../models/gpt2.zig");
const data_loader = @import("../data_loader.zig");

// GPT-2 model imports for easier access
const GPT2Config = gpt2.GPT2Config;
const GPT2Model = gpt2.GPT2(f32);
const getParameterShapes = gpt2.getParameterShapes;

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
const DataLoader = data_loader.DataLoader;

/// DiLoCo algorithm configuration
pub const DiLoCoConfig = struct {
    base_config: TrainingConfig,
    tau: usize, // Inner loop steps
    nesterov_momentum: f32,
    parameter_averaging: bool,
    gpt2_config: GPT2Config, // GPT-2 model configuration

    pub fn default() DiLoCoConfig {
        return DiLoCoConfig{
            .base_config = TrainingConfig.default(),
            .tau = 10, // Default inner loop steps
            .nesterov_momentum = 0.9,
            .parameter_averaging = true,
            .gpt2_config = GPT2Config.nano(), // Use nano config by default
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
    mlir_builder: *MLIRBuilder, // Now a pointer, as it's owned externally
    nesterov_optimizer: *NesterovMLIR,
    element_type: mlir.Type,

    // GPT-2 master parameters as multiple MLIR tensors
    master_parameters: ?[]Tensor,
    parameter_shapes: [][]i64, // Multiple parameter shapes for GPT-2

    // NEW: Generic executor - replaces direct backend knowledge
    executor: Executor,

    // NEW: Cached serialized worker graph for one-time initialization
    serialized_worker_graph: []u8,

    // Data loading for real training batches
    data_loader: *DataLoader,

    const Self = @This();

    // Change the init signature to accept the generic executor and DataLoader
    pub fn init(allocator: Allocator, coordinator: *Shepherd, config: DiLoCoConfig, executor: Executor, mlir_builder: *MLIRBuilder, dataset: *DataLoader) !Self {
        std.log.info("DiLoCo.init: Starting initialization...", .{});
        
        // The MLIRBuilder is now passed in, ensuring a single context.
        std.log.info("DiLoCo.init: Creating element type...", .{});
        std.log.info("DiLoCo.init: mlir_builder pointer = {}", .{@intFromPtr(mlir_builder)});
        std.log.info("DiLoCo.init: mlir_builder.ctx handle = {}", .{@intFromPtr(mlir_builder.ctx.handle)});
        std.log.info("DiLoCo.init: About to call f32Type...", .{});
        const element_type = mlir.Type.f32Type(mlir_builder.ctx);
        std.log.info("DiLoCo.init: f32Type call succeeded", .{});

        // Configure and create Nesterov optimizer
        std.log.info("DiLoCo.init: Configuring Nesterov optimizer...", .{});
        const nesterov_config = nesterov_mlir.NesterovMLIRConfiguration(f32){
            .learning_rate = config.base_config.learning_rate,
            .momentum = config.nesterov_momentum,
        };

        std.log.info("DiLoCo.init: Allocating Nesterov optimizer memory...", .{});
        const nesterov_optimizer = try allocator.create(NesterovMLIR);
        std.log.info("DiLoCo.init: Memory allocated, calling NesterovMLIR.init...", .{});
        nesterov_optimizer.* = try NesterovMLIR.init(allocator, mlir_builder, nesterov_config, element_type);
        std.log.info("DiLoCo.init: Nesterov optimizer created successfully", .{});

        // Set up parameter shapes for GPT-2 model
        std.log.info("DiLoCo.init: Setting up parameter shapes for GPT-2 model...", .{});
        const parameter_shapes = try getParameterShapes(allocator, config.gpt2_config);
        std.log.info("DiLoCo.init: Created {} parameter shapes for GPT-2", .{parameter_shapes.len});

        // Create a partial instance to build the worker graph
        std.log.info("DiLoCo.init: Creating partial instance...", .{});
        var partial_self = Self{
            .allocator = allocator,
            .coordinator = undefined, // Not used during graph building
            .config = config,
            .status = .not_started,
            .metrics = TrainingMetrics.init(),
            .current_epoch = 0,
            .mlir_builder = mlir_builder,
            .nesterov_optimizer = nesterov_optimizer,
            .element_type = element_type,
            .master_parameters = null,
            .parameter_shapes = parameter_shapes,
            .executor = executor,
            .serialized_worker_graph = undefined, // Will be set below
            .data_loader = dataset,
        };

        // Build the worker graph ONCE during initialization using the safe function builder
        std.log.info("DiLoCo.init: Building worker training graph...", .{});
        const graph = try partial_self.buildWorkerTrainingGraph(mlir_builder);
        mlir_builder.module.op().dump(); // Debug print the state of the shared builder's module
        std.log.info("DiLoCo.init: Worker training graph built successfully", .{});

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
            .parameter_shapes = parameter_shapes,
            .executor = executor, // Store the generic executor
            .serialized_worker_graph = graph, // Store the pre-built graph
            .data_loader = dataset,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.master_parameters) |params| {
            for (params) |param| {
                param.deinit();
            }
            self.allocator.free(params);
        }

        self.nesterov_optimizer.deinit();
        self.allocator.destroy(self.nesterov_optimizer);

        // We no longer deinit the builder, as we don't own it.

        // Free the stored serialized graph
        self.allocator.free(self.serialized_worker_graph);

        // Free parameter shapes
        for (self.parameter_shapes) |shape| {
            self.allocator.free(shape);
        }
        self.allocator.free(self.parameter_shapes);
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

        std.log.info("👻 DiLoCo.run() starting...", .{});

        self.status = .initializing;
        monitoring.setStatus(.initializing);
        
        // Calculate total parameter count from GPT-2 config
        const total_param_count = gpt2.countTotalParameters(self.config.gpt2_config);
        monitoring.setModelInfo(total_param_count, self.config.base_config.learning_rate);
        std.log.info("✓ DiLoCo status and monitoring initialized with {} GPT-2 parameters", .{total_param_count});

        // Initialize master parameters
        std.log.info("Initializing master parameters...", .{});
        try self.initializeMasterParameters();
        std.log.info("✓ Master parameters initialized successfully", .{});

        // PHASE 0: SETUP WORKERS (ONE-TIME)
        std.log.info("Setting up workers...", .{});
        try self.setupWorkers();
        std.log.info("✓ Workers setup completed", .{});

        self.status = .running;
        monitoring.setStatus(.running);

        // Main outer loop
        for (0..self.config.base_config.outer_loop_steps) |outer_step| {
            const start_time = std.time.milliTimestamp();
            std.log.info("DiLoCo outer loop step {}/{}", .{ outer_step + 1, self.config.base_config.outer_loop_steps });

            // Phase 0: Get a fresh batch of training data
            const batch_size = 4;
            const block_size = 8;
            const batch = try self.data_loader.getBatch(batch_size, block_size);
            defer self.allocator.free(batch.x);
            defer self.allocator.free(batch.y);
            std.log.info("Loaded batch with {} input tokens and {} target tokens", .{ batch.x.len, batch.y.len });

            // Phase 1: Broadcast master parameters and data batch to all workers
            try self.broadcastMasterParametersAndBatch(batch);

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

    /// Initialize master parameters using MLIR for GPT-2 model
    fn initializeMasterParameters(self: *Self) !void {
        const total_param_count = gpt2.countTotalParameters(self.config.gpt2_config);
        std.log.info("🎲 Initializing {} GPT-2 parameter tensors (total {} elements)...", .{ self.parameter_shapes.len, total_param_count });

        // Allocate array for master parameter tensors
        const param_tensors = try self.allocator.alloc(Tensor, self.parameter_shapes.len);
        var rng = std.Random.DefaultPrng.init(12345);

        // Initialize each parameter tensor with appropriate shape
        for (param_tensors, 0..) |*param_tensor, i| {
            const shape = self.parameter_shapes[i];
            
            // Calculate element count for this tensor
            var element_count: usize = 1;
            for (shape) |dim| {
                element_count *= @intCast(dim);
            }

            std.log.info("Initializing parameter {} with shape [{d}] ({} elements)", .{ i, shape, element_count });

            // Create parameter data with Xavier initialization
            const param_data = try self.allocator.alloc(f32, element_count);
            defer self.allocator.free(param_data);

            const scale = std.math.sqrt(2.0 / @as(f32, @floatFromInt(element_count)));
            for (param_data) |*param| {
                param.* = rng.random().floatNorm(f32) * scale;
            }

            // Convert to MLIR tensor
            param_tensor.* = try self.createTensorFromArrayWithShape(param_data, shape);
        }

        self.master_parameters = param_tensors;
        std.log.info("🌙 Initialized {} GPT-2 parameter tensors with Xavier initialization", .{param_tensors.len});
    }

    /// Create a dedicated forward+loss function that will be differentiated
    /// This is a well-formed func.func operation that autodiff can process
    /// Now builds the complete GPT-2 training graph: forward pass + cross entropy loss
    fn buildForwardAndLossFunction(self: *Self, builder: *ops.MLIRBuilder) !mlir.Operation {
        std.log.info("buildForwardAndLossFunction: Building FULL GPT-2 training graph...", .{});
        const config = self.config.gpt2_config;
        const f32_type = mlir.Type.f32Type(builder.ctx);
        const i32_type = mlir.Type.i32Type(builder.ctx);

        // Define batch and sequence sizes for training
        const batch_size = 4;
        const block_size = 8;

        // 1. Create input types: all GPT-2 parameters + input_ids + targets
        var func_input_types = ArrayList(mlir.Type).init(self.allocator);
        defer func_input_types.deinit();

        // Add types for all trainable parameters (same order as getParameterShapes)
        for (self.parameter_shapes) |shape| {
            try func_input_types.append(mlir.Type.rankedTensorType(builder.ctx, shape, f32_type));
        }

        // Add types for input data (x) and targets (y)
        const data_shape = &[_]i64{ batch_size, block_size };
        try func_input_types.append(mlir.Type.rankedTensorType(builder.ctx, data_shape, i32_type));
        try func_input_types.append(mlir.Type.rankedTensorType(builder.ctx, data_shape, i32_type));

        // Define the function's output types (just the scalar loss)
        const scalar_f32_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type);
        const func_output_types = &[_]mlir.Type{scalar_f32_type};

        const func_type = mlir.Type.functionType(builder.ctx, func_input_types.items, func_output_types);

        // 2. Create the function and get its entry block
        const result = try builder.createFunction("gpt2_forward_and_loss", func_type);
        const func_op = result.func_op;
        const entry_block = result.entry_block;

        // 3. Set the insertion point to inside our new function
        const original_insertion_block = builder.getInsertionBlock();
        builder.setInsertionBlock(entry_block);
        defer builder.setInsertionBlock(original_insertion_block);

        // 4. Get all arguments (parameters, x, y)
        var all_func_args = try entry_block.getArguments(self.allocator);
        defer self.allocator.free(all_func_args);
        var params_slice = all_func_args[0..self.parameter_shapes.len];
        const x = try builder.newTensor(all_func_args[self.parameter_shapes.len]);
        const y = try builder.newTensor(all_func_args[self.parameter_shapes.len + 1]);

        std.log.info("buildForwardAndLossFunction: Instantiating GPT-2 model with {} parameters...", .{params_slice.len});

        // 5. Instantiate the GPT-2 model and build the graph
        var model = try GPT2Model.init(self.allocator, config, builder, &params_slice);
        defer model.deinit();
        const outputs = try model.forwardWithLoss(x, y, builder);

        std.log.info("buildForwardAndLossFunction: GPT-2 forward pass and loss completed", .{});

        // 6. Return the loss
        _ = try builder.createAndAttach("func.return", &.{outputs.loss.value}, &.{}, .{});

        std.log.info("buildForwardAndLossFunction: GPT-2 training graph built successfully.", .{});
        return func_op;
    }

    /// Extract raw byte data from a tensor defined by a stablehlo.constant operation.
    fn extractTensorData(self: *Self, t: Tensor) ![]u8 {
        const c = @import("../mlir/c.zig").c;

        // 1. Get the MLIR Value from the Tensor
        if (!c.mlirValueIsAOpResult(t.value.handle)) {
            std.log.err("Cannot extract data: Tensor value is not an operation result.", .{});
            return error.NotAnOperationResult;
        }
        const defining_op_handle = c.mlirOpResultGetOwner(t.value.handle);
        const defining_op = mlir.Operation{ .handle = defining_op_handle };

        // 2. Verify the operation is a constant
        const op_identifier = c.mlirOperationGetName(defining_op.handle);
        const op_name_ref = c.mlirIdentifierStr(op_identifier);
        const op_name = c.fromStringRef(op_name_ref);

        if (!std.mem.eql(u8, op_name, "stablehlo.constant")) {
            std.log.err("Cannot extract data: Tensor is not defined by a stablehlo.constant op. It is a {s}", .{op_name});
            return error.NotAConstantOp;
        }

        // 3. Get the 'value' attribute, which must be a DenseElementsAttr
        const value_attr = c.operationGetAttributeByName(defining_op.handle, "value");
        if (@intFromPtr(value_attr) == 0 or !c.mlirAttributeIsADenseElements(value_attr)) {
            return error.InvalidConstantAttribute;
        }

        // 4. Extract the raw data pointer and size
        const raw_data_ptr = c.mlirDenseElementsAttrGetRawData(value_attr);
        const elem_count = t.shape.elemCount();
        const dtype_size = t.shape.dtype.sizeInBytes();
        const num_bytes = elem_count * dtype_size;
        
        std.log.info("DEBUG extractTensorData: shape.elemCount()={}, dtype.sizeInBytes()={}, calculated num_bytes={}", .{ elem_count, dtype_size, num_bytes });
        
        // Sanity check - if num_bytes is gigantic, there's a bug in the calculation
        if (num_bytes > 100 * 1024 * 1024) { // 100MB per tensor is way too large
            std.log.err("ERROR: extractTensorData calculated tensor size {} MB - this is definitely wrong!", .{ num_bytes / (1024 * 1024) });
            return error.TensorSizeCalculationError;
        }
        
        const data_slice: [*]const u8 = @ptrCast(raw_data_ptr);

        // 5. Return a duplicate of the data for the caller to own
        std.log.info("DEBUG extractTensorData: Successfully extracting {} bytes", .{num_bytes});
        return self.allocator.dupe(u8, data_slice[0..num_bytes]);
    }

    /// Build the complete worker training graph as MLIR module using Function-as-a-Unit pattern
    /// This creates: forward_fn -> autodiff -> main_fn that orchestrates the full training step
    fn buildWorkerTrainingGraph(self: *Self, builder: *MLIRBuilder) ![]u8 {
        std.log.info("DiLoCo: Building REAL autodiff-enabled worker training graph with Function-as-a-Unit pattern...", .{});

        // === Part 1: Create the forward+loss function to be differentiated ===
        std.log.info("Creating forward+loss function...", .{});
        std.log.info("buildWorkerTrainingGraph: builder pointer = {}", .{@intFromPtr(builder)});
        std.log.info("buildWorkerTrainingGraph: About to call buildForwardAndLossFunction...", .{});
        const forward_fn_op = try self.buildForwardAndLossFunction(builder);
        std.log.info("Forward+loss function created successfully", .{});
        // NOTE: Function is already automatically attached to module by createFunction

        // === Part 2: Differentiate the forward function ===
        std.log.info("Differentiating forward function with autodiff.buildGradientGraph...", .{});
        
        // DEBUG: Verify function structure before calling autodiff
        std.log.info("DEBUG: Verifying forward function structure...", .{});
        const func_region = forward_fn_op.getRegion(0);
        const func_block = func_region.getBlock(0);
        
        // Debug: Check all operations in the block
        std.log.info("DEBUG: Walking through all operations in function block...", .{});
        var maybe_op = func_block.getFirstOp();
        var op_count: usize = 0;
        while (maybe_op) |op| {
            const op_name = op.getName();
            std.log.info("DEBUG: Operation {}: {s}", .{ op_count, op_name });
            if (std.mem.eql(u8, op_name, "func.return")) {
                std.log.info("DEBUG: Found func.return operation!", .{});
            }
            maybe_op = op.getNext();
            op_count += 1;
        }
        std.log.info("DEBUG: Total operations in function block: {}", .{op_count});
        
        const maybe_terminator = func_block.getLastOp();
        const maybe_last_op = func_block.getLastOpGeneric();
        if (maybe_terminator) |terminator| {
            std.log.info("DEBUG: Found terminator operation in forward function", .{});
            terminator.dump();
        } else {
            std.log.info("DEBUG: ERROR - No terminator found via getLastOp()!", .{});
        }
        
        if (maybe_last_op) |last_op| {
            const last_op_name = last_op.getName();
            std.log.info("DEBUG: Last operation via getLastOpGeneric(): {s}", .{last_op_name});
            if (std.mem.eql(u8, last_op_name, "func.return")) {
                std.log.info("DEBUG: Last operation IS func.return!", .{});
            }
        } else {
            std.log.info("DEBUG: ERROR - No last operation found via getLastOpGeneric()!", .{});
        }
        
        // The autodiff function will add its own generated function to the module.
        _ = try autodiff.buildGradientGraph(self.allocator, builder, forward_fn_op);

        // The autodiff system will name the gradient function predictably
        // Updated for GPT-2 function name
        const grad_fn_name = "gpt2_forward_and_loss_grad";

        // === Part 3: Create the main worker function that orchestrates everything ===
        std.log.info("Creating main worker orchestration function...", .{});

        // Define types for main function - now with all GPT-2 parameters
        const f32_type = mlir.Type.f32Type(builder.ctx);
        const i32_type = mlir.Type.i32Type(builder.ctx);
        const loss_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type);
        const data_shape = &[_]i64{ 4, 8 }; // batch=4, seq_len=8 for nano config
        const data_type = mlir.Type.rankedTensorType(builder.ctx, data_shape, i32_type);

        // Create input types: all GPT-2 parameters + inputs + targets
        var main_input_types = ArrayList(mlir.Type).init(self.allocator);
        defer main_input_types.deinit();
        
        // Add all parameter types
        for (self.parameter_shapes) |shape| {
            try main_input_types.append(mlir.Type.rankedTensorType(builder.ctx, shape, f32_type));
        }
        // Add input data and targets
        try main_input_types.append(data_type);
        try main_input_types.append(data_type);

        // Create output types: all updated parameters + loss
        var main_result_types = ArrayList(mlir.Type).init(self.allocator);
        defer main_result_types.deinit();
        
        // Add all parameter types (updated parameters)
        for (self.parameter_shapes) |shape| {
            try main_result_types.append(mlir.Type.rankedTensorType(builder.ctx, shape, f32_type));
        }
        // Add loss
        try main_result_types.append(loss_type);

        const main_func_type = mlir.Type.functionType(builder.ctx, main_input_types.items, main_result_types.items);

        // --- REFACTORED MAIN FUNCTION CREATION ---
        // Use the safe helper to create the main function
        std.log.info("Creating main orchestration function with safe helper...", .{});
        const main_result = try builder.createFunction("main", main_func_type);
        const main_block = main_result.entry_block;

        // NOTE: Function is already automatically attached to module by createFunction

        // CRITICAL: Set the insertion point to the main function's body.
        const original_insertion_block = builder.getInsertionBlock();
        builder.setInsertionBlock(main_block);
        defer builder.setInsertionBlock(original_insertion_block);

        // Get main function arguments (all GPT-2 parameters + inputs + targets)
        var main_func_args = try main_block.getArguments(self.allocator);
        defer self.allocator.free(main_func_args);
        
        const param_count = self.parameter_shapes.len;
        const initial_params = main_func_args[0..param_count];
        const inputs = main_func_args[param_count];
        const targets = main_func_args[param_count + 1];
        std.log.info("Main function created with {} parameter arguments", .{param_count});
        // --- END REFACTORED MAIN FUNCTION CREATION ---

        // Build forward call operands: all parameters + inputs + targets
        var forward_operands = ArrayList(mlir.Value).init(self.allocator);
        defer forward_operands.deinit();
        
        for (initial_params) |param| {
            try forward_operands.append(param);
        }
        try forward_operands.append(inputs);
        try forward_operands.append(targets);

        // Call the forward function to get the loss
        const forward_callee_str = try std.fmt.allocPrint(self.allocator, "@\"{s}\"", .{"gpt2_forward_and_loss"});
        defer self.allocator.free(forward_callee_str);
        const forward_callee_attr = try mlir.Attribute.fromParseString(builder.ctx, forward_callee_str);

        const forward_call_op = try builder.createAndAttach("func.call", forward_operands.items, &.{loss_type}, .{
            .attributes = &.{
                .{ "callee", forward_callee_attr },
            },
        });
        const loss_val = forward_call_op.getResult(0);

        // Call the gradient function to get gradients
        const one = try ops.constant(builder, 1.0, &.{}, f32_type); // Scalar 1.0 for loss gradient seed
        
        // Build gradient call operands: all parameters + inputs + targets + loss_grad
        var grad_operands = ArrayList(mlir.Value).init(self.allocator);
        defer grad_operands.deinit();
        
        for (initial_params) |param| {
            try grad_operands.append(param);
        }
        try grad_operands.append(inputs);
        try grad_operands.append(targets);
        try grad_operands.append(one.value);

        // Build gradient result types: gradients for all parameters + data + targets
        var grad_result_types = ArrayList(mlir.Type).init(self.allocator);
        defer grad_result_types.deinit();
        
        for (self.parameter_shapes) |shape| {
            try grad_result_types.append(mlir.Type.rankedTensorType(builder.ctx, shape, f32_type));
        }
        try grad_result_types.append(data_type); // input gradients
        try grad_result_types.append(data_type); // target gradients

        const grad_callee_str = try std.fmt.allocPrint(self.allocator, "@\"{s}\"", .{grad_fn_name});
        defer self.allocator.free(grad_callee_str);
        const grad_callee_attr = try mlir.Attribute.fromParseString(builder.ctx, grad_callee_str);

        const grad_call_op = try builder.createAndAttach("func.call", grad_operands.items, grad_result_types.items, .{
            .attributes = &.{
                .{ "callee", grad_callee_attr },
            },
        });

        // Apply gradient descent update to each parameter
        var updated_param_values = ArrayList(mlir.Value).init(self.allocator);
        defer updated_param_values.deinit();
        
        const learning_rate_scalar = try ops.constant(builder, 0.01, &.{}, f32_type);
        
        for (initial_params, 0..) |param, i| {
            const param_grad = grad_call_op.getResult(i);
            
            // Broadcast learning rate to parameter shape if needed
            const param_tensor = try builder.newTensor(param);
            const grad_tensor = try builder.newTensor(param_grad);
            
            // Scale gradient: lr * grad
            const grad_scaled = try ops.multiply(builder, grad_tensor, learning_rate_scalar);
            
            // Update parameter: param - lr * grad
            const updated_param = try ops.subtract(builder, param_tensor, grad_scaled);
            try updated_param_values.append(updated_param.value);
        }

        // Add loss to return values
        try updated_param_values.append(loss_val);

        // Return all updated parameters + loss from main function
        _ = try builder.createAndAttach("func.return", updated_param_values.items, &.{}, .{});

        // === Part 4: Debug and serialize the complete module ===
        std.log.info("Complete worker module with {} functions created successfully", .{3}); // forward_fn, grad_fn, main

        // Dump the module for debugging
        builder.module.op().dump();

        const serialized = try @import("../mlir_ctx.zig").serializeMLIRModule(self.allocator, builder.module);

        std.log.info("✓ DiLoCo REAL autodiff worker training graph built and serialized ({} bytes).", .{serialized.len});
        return serialized;
    }

    /// Broadcasts the pre-built training graph to all workers for initialization
    fn setupWorkers(self: *Self) !void {
        std.log.info("Broadcasting training graph to workers for setup...", .{});

        // Base64 encode the serialized graph for safe JSON transport
        const b64_len = std.base64.standard.Encoder.calcSize(self.serialized_worker_graph.len);
        const b64_encoded_graph = try self.allocator.alloc(u8, b64_len);
        defer self.allocator.free(b64_encoded_graph);

        const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_graph, self.serialized_worker_graph).len;

        // Create JSON payload with the Base64-encoded string
        const json_payload = std.json.Value{ .string = b64_encoded_graph[0..encoded_len] };

        // Broadcast InitializeGraph message with the encoded graph
        try self.coordinator.broadcastToWorkers(MessageType.INITIALIZE_GRAPH, json_payload);

        std.log.info("Training graph sent to {} workers for caching.", .{self.coordinator.getWorkerCount()});
    }

    /// Broadcast master parameters to all workers using Cap'n Proto
    fn broadcastMasterParameters(self: *Self) !void {
        if (self.master_parameters == null) {
            return error.ParametersNotInitialized;
        }

        const param_tensors = self.master_parameters.?;
        
        // Extract and concatenate all parameter tensor data
        var total_param_bytes = ArrayList(u8).init(self.allocator);
        defer total_param_bytes.deinit();
        
        for (param_tensors) |param_tensor| {
            const tensor_data = try self.extractTensorData(param_tensor);
            defer self.allocator.free(tensor_data);
            try total_param_bytes.appendSlice(tensor_data);
        }

        // Create WorkerPayload and serialize with Cap'n Proto
        const worker_payload = binary_protocol.WorkerPayload{
            .params = total_param_bytes.items,
        };
        const capnp_bytes = try worker_payload.serialize(self.allocator);
        defer self.allocator.free(capnp_bytes);

        // Base64 encode the binary data
        const b64_len = std.base64.standard.Encoder.calcSize(capnp_bytes.len);
        const b64_encoded_payload = try self.allocator.alloc(u8, b64_len);
        defer self.allocator.free(b64_encoded_payload);

        const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_payload, capnp_bytes).len;

        // Create JSON payload with Base64-encoded binary data
        const json_payload = std.json.Value{ .string = b64_encoded_payload[0..encoded_len] };

        // Broadcast StartInnerLoop message with Cap'n Proto
        try self.coordinator.broadcastToWorkers(MessageType.START_INNER_LOOP, json_payload);

        std.log.debug("Broadcasted Cap'n Proto encoded parameters ({} tensors) to {} workers", .{ param_tensors.len, self.coordinator.getWorkerCount() });
    }

    /// Broadcast master parameters and data batch to all workers using Cap'n Proto
    fn broadcastMasterParametersAndBatch(self: *Self, batch: data_loader.Batch) !void {
        if (self.master_parameters == null) {
            return error.ParametersNotInitialized;
        }

        const param_tensors = self.master_parameters.?;
        
        // Extract and concatenate all parameter tensor data
        var total_param_bytes = ArrayList(u8).init(self.allocator);
        defer total_param_bytes.deinit();
        
        for (param_tensors) |param_tensor| {
            const tensor_data = try self.extractTensorData(param_tensor);
            defer self.allocator.free(tensor_data);
            try total_param_bytes.appendSlice(tensor_data);
        }

        std.log.info("DEBUG: Total raw parameter bytes: {} ({} MB)", .{ total_param_bytes.items.len, total_param_bytes.items.len / (1024 * 1024) });

        // Serialize batch data as raw bytes (u32 tokens -> bytes)
        const input_ids_bytes = std.mem.sliceAsBytes(batch.x);
        const targets_bytes = std.mem.sliceAsBytes(batch.y);
        
        std.log.info("DEBUG: Batch sizes - input_ids: {} bytes, targets: {} bytes", .{ input_ids_bytes.len, targets_bytes.len });

        // Create WorkerPayload with parameters and batch data
        const worker_payload = binary_protocol.WorkerPayload{
            .params = total_param_bytes.items,
            .input_ids = input_ids_bytes,
            .targets = targets_bytes,
        };
        
        std.log.info("DEBUG: About to serialize WorkerPayload with Cap'n Proto...", .{});
        const capnp_bytes = try worker_payload.serialize(self.allocator);
        defer self.allocator.free(capnp_bytes);
        
        std.log.info("DEBUG: Cap'n Proto serialization result: {} bytes ({} MB)", .{ capnp_bytes.len, capnp_bytes.len / (1024 * 1024) });

        // Base64 encode the binary data
        const b64_len = std.base64.standard.Encoder.calcSize(capnp_bytes.len);
        const b64_encoded_payload = try self.allocator.alloc(u8, b64_len);
        defer self.allocator.free(b64_encoded_payload);

        const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_payload, capnp_bytes).len;
        
        std.log.info("DEBUG: Base64 encoding result: {} bytes ({} MB)", .{ encoded_len, encoded_len / (1024 * 1024) });

        // Create JSON payload with Base64-encoded binary data
        const json_payload = std.json.Value{ .string = b64_encoded_payload[0..encoded_len] };

        // Broadcast StartInnerLoop message with Cap'n Proto
        try self.coordinator.broadcastToWorkers(MessageType.START_INNER_LOOP, json_payload);

        std.log.debug("Broadcasted parameters ({} tensors) + batch ({} inputs, {} targets) to {} workers", .{ 
            param_tensors.len, 
            batch.x.len, 
            batch.y.len, 
            self.coordinator.getWorkerCount() 
        });
    }

    /// Collect results from workers after inner loop using Cap'n Proto
    fn collectWorkerResults(self: *Self) !ArrayList(WorkerResult) {
        std.log.info("Collecting Cap'n Proto results from workers", .{});
        const responses = try self.coordinator.collectFromWorkers(MessageType.INNER_LOOP_COMPLETE);
        defer responses.deinit();

        var results = ArrayList(WorkerResult).init(self.allocator);
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        for (responses.items) |response| {
            // 1. Extract the Base64 string payload
            const b64_encoded_payload = switch (response.data) {
                .string => |s| s,
                else => return error.InvalidMessageFormat,
            };

            // 2. Base64-decode the payload to get the binary Cap'n Proto data
            const capnp_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_encoded_payload);
            const capnp_bytes = try arena.allocator().alloc(u8, capnp_len);
            try std.base64.standard.Decoder.decode(capnp_bytes, b64_encoded_payload);

            // 3. Deserialize the binary data using Cap'n Proto
            const reader = try binary_protocol.ShepherdPayload.Reader.init(capnp_bytes);
            defer reader.deinit();

            // 4. Extract the data and build the WorkerResult
            const params_bytes = try reader.getUpdatedParams();
            const loss_value = try reader.getLoss();

            // The WorkerResult struct expects owned slices, so we must dupe the data.
            const result = WorkerResult{
                .node_id = response.sender_node,
                .parameter_bytes = try self.allocator.dupe(u8, params_bytes),
                .loss = loss_value,
                .steps_completed = self.config.tau,
            };
            try results.append(result);
        }

        std.log.info("Collected {} Cap'n Proto results from workers", .{results.items.len});
        return results;
    }

    /// Update master parameters using MLIR Nesterov optimizer
    fn updateMasterParametersMLIR(self: *Self, worker_results: ArrayList(WorkerResult)) !void {
        if (worker_results.items.len == 0) {
            return error.NoWorkerResults;
        }

        std.log.info("Updating {} GPT-2 parameter tensors using MLIR Nesterov optimizer with {} worker results", .{ self.parameter_shapes.len, worker_results.items.len });

        const param_tensors = self.master_parameters.?;

        // Split averaged worker parameters back into individual tensors and update each
        const averaged_params_bytes = try self.averageWorkerParameterBytes(worker_results);
        defer self.allocator.free(averaged_params_bytes);
        
        // Convert concatenated bytes back to individual parameter tensors
        var byte_offset: usize = 0;
        for (param_tensors, 0..) |*param_tensor, i| {
            const shape = self.parameter_shapes[i];
            
            // Calculate size for this parameter tensor
            var element_count: usize = 1;
            for (shape) |dim| {
                element_count *= @intCast(dim);
            }
            const tensor_bytes = element_count * @sizeOf(f32);
            
            // Extract this tensor's data
            const tensor_data_bytes = averaged_params_bytes[byte_offset..byte_offset + tensor_bytes];
            const param_slice: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, tensor_data_bytes));
            
            // Create averaged tensor for this parameter
            const averaged_tensor = try self.createTensorFromArrayWithShape(param_slice, shape);
            defer averaged_tensor.deinit();

            // Compute gradients (difference between averaged and master parameters)
            const gradient_tensor = try ops.subtract(self.mlir_builder, averaged_tensor, param_tensor.*);
            defer gradient_tensor.deinit();

            // Apply Nesterov momentum update using MLIR optimizer
            const updated_param = try self.nesterov_optimizer.update(param_tensor.*, gradient_tensor);

            // Replace this parameter with updated one
            param_tensor.deinit();
            param_tensor.* = updated_param;
            
            byte_offset += tensor_bytes;
        }

        // Update metrics
        self.metrics.loss = self.calculateAverageLoss(worker_results);

        std.log.info("All {} GPT-2 parameters updated using MLIR Nesterov, average loss: {d:.4}", .{ param_tensors.len, self.metrics.loss });
    }

    /// Average parameters from all workers (new byte-based version)
    fn averageWorkerParameterBytes(self: *Self, worker_results: ArrayList(WorkerResult)) ![]u8 {
        // Calculate total parameter count from GPT-2 config
        const total_param_count = gpt2.countTotalParameters(self.config.gpt2_config);
        const averaged = try self.allocator.alloc(f32, total_param_count);
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
            total_loss += result.loss;
        }

        return total_loss / @as(f32, @floatFromInt(worker_results.items.len));
    }

    /// Create MLIR tensor from parameter array with specific shape
    fn createTensorFromArrayWithShape(self: *Self, array: []const f32, shape: []const i64) !Tensor {
        const byte_data = std.mem.sliceAsBytes(array);
        const tensor_type = mlir.Type.rankedTensorType(self.mlir_builder.ctx, shape, self.element_type);

        var tensor_shape = try tensor.Shape.initWithDims(self.mlir_builder.ctx, shape, tensor.DType.f32);
        defer tensor_shape.deinit();
        const value = try self.mlir_builder.createConstant(byte_data, tensor_type, tensor_shape);
        return try self.mlir_builder.newTensor(value);
    }

    /// Create MLIR tensor from parameter array (legacy method, kept for compatibility)
    fn createTensorFromArray(self: *Self, array: []const f32) !Tensor {
        if (self.parameter_shapes.len == 0) return error.NoParameterShapes;
        return self.createTensorFromArrayWithShape(array, self.parameter_shapes[0]);
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
            param.* = @bitCast(std.mem.readInt(u32, data[offset .. offset + @sizeOf(f32)][0..4], .little));
            offset += @sizeOf(f32);
        }

        return params;
    }

    /// Clean up worker results
    fn cleanupWorkerResults(self: *Self, worker_results: ArrayList(WorkerResult)) void {
        for (worker_results.items) |result| {
            self.allocator.free(result.parameter_bytes);
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
    loss: f32,
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
    
    // Test GPT-2 configuration
    const expected_param_count = gpt2.countTotalParameters(config.gpt2_config);
    const actual_param_shapes = diloco.parameter_shapes.len;
    std.log.info("GPT-2 nano has {} parameter tensors with total {} parameters", .{ actual_param_shapes, expected_param_count });

    std.log.info("✓ DiLoCo MLIR algorithm test completed");
}
