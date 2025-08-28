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
    param_count: usize,

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
    mlir_builder: *MLIRBuilder, // Now a pointer, as it's owned externally
    nesterov_optimizer: *NesterovMLIR,
    element_type: mlir.Type,

    // Master parameters as MLIR tensors
    master_parameters: ?Tensor,
    parameter_shape: []const i64,

    // NEW: Generic executor - replaces direct backend knowledge
    executor: Executor,

    // NEW: Cached serialized worker graph for one-time initialization
    serialized_worker_graph: []u8,

    const Self = @This();

    // Change the init signature to accept the generic executor
    pub fn init(allocator: Allocator, coordinator: *Shepherd, config: DiLoCoConfig, executor: Executor, mlir_builder: *MLIRBuilder) !Self {
        std.log.info("DiLoCo.init: Starting initialization...", .{});
        
        // The MLIRBuilder is now passed in, ensuring a single context.
        std.log.info("DiLoCo.init: Creating element type...", .{});
        const element_type = mlir.Type.f32Type(mlir_builder.ctx);

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

        // Set up parameter shape
        std.log.info("DiLoCo.init: Setting up parameter shape...", .{});
        const parameter_shape = try allocator.alloc(i64, 1);
        parameter_shape[0] = @intCast(config.param_count);

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
            .parameter_shape = parameter_shape,
            .executor = executor,
            .serialized_worker_graph = undefined, // Will be set below
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
            .parameter_shape = parameter_shape,
            .executor = executor, // Store the generic executor
            .serialized_worker_graph = graph, // Store the pre-built graph
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.master_parameters) |params| {
            params.deinit();
        }

        self.nesterov_optimizer.deinit();
        self.allocator.destroy(self.nesterov_optimizer);

        // We no longer deinit the builder, as we don't own it.

        // Free the stored serialized graph
        self.allocator.free(self.serialized_worker_graph);

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

        std.log.info("🚀 DiLoCo.run() starting...", .{});

        self.status = .initializing;
        monitoring.setStatus(.initializing);
        monitoring.setModelInfo(self.config.param_count, self.config.base_config.learning_rate);
        std.log.info("✓ DiLoCo status and monitoring initialized", .{});

        // Initialize master parameters
        std.log.info("🔧 Initializing master parameters...", .{});
        try self.initializeMasterParameters();
        std.log.info("✓ Master parameters initialized successfully", .{});

        // PHASE 0: SETUP WORKERS (ONE-TIME)
        std.log.info("👥 Setting up workers...", .{});
        try self.setupWorkers();
        std.log.info("✓ Workers setup completed", .{});

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
        std.log.info("🎲 Allocating parameter data array ({} elements)...", .{self.config.param_count});

        // Create initial parameter values
        const param_data = try self.allocator.alloc(f32, self.config.param_count);
        defer self.allocator.free(param_data);
        std.log.info("✓ Parameter array allocated", .{});

        // Initialize with Xavier/Glorot initialization
        std.log.info("🔢 Computing Xavier initialization...", .{});
        var rng = std.Random.DefaultPrng.init(12345);
        const scale = std.math.sqrt(2.0 / @as(f32, @floatFromInt(self.config.param_count)));

        for (param_data) |*param| {
            param.* = rng.random().floatNorm(f32) * scale;
        }
        std.log.info("✓ Parameter values generated", .{});

        // Convert to MLIR tensor
        std.log.info("🔧 Converting to MLIR tensor...", .{});
        self.master_parameters = try self.createTensorFromArray(param_data);
        std.log.info("✓ MLIR tensor created successfully", .{});

        std.log.info("✅ Initialized master parameters with {} elements using Xavier initialization", .{self.config.param_count});
    }

    /// Create a dedicated forward+loss function that will be differentiated
    /// This is a well-formed func.func operation that autodiff can process
    fn buildForwardAndLossFunction(self: *Self, builder: *ops.MLIRBuilder) !mlir.Operation {
        std.log.info("buildForwardAndLossFunction: Starting...", .{});
        const f32_type = mlir.Type.f32Type(builder.ctx);
        const param_count = self.config.param_count;

        // Define types
        std.log.info("buildForwardAndLossFunction: Creating tensor types...", .{});
        const params_type = mlir.Type.rankedTensorType(builder.ctx, &.{@intCast(param_count)}, f32_type);
        const data_type = mlir.Type.rankedTensorType(builder.ctx, &.{ 4, 12 }, f32_type);
        const loss_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type); // Scalar loss

        // Define the function type: func(params, inputs, targets) -> (loss)
        std.log.info("buildForwardAndLossFunction: Creating function type...", .{});
        const input_types = [_]mlir.Type{ params_type, data_type, data_type };
        const result_types = [_]mlir.Type{loss_type};
        const func_type = mlir.Type.functionType(builder.ctx, &input_types, &result_types);
        std.log.info("buildForwardAndLossFunction: Function type created", .{});

        // --- REFACTORED CODE ---
        // Use the new safe helper to create the function and get its entry block
        std.log.info("buildForwardAndLossFunction: Using safe createFunction helper...", .{});
        const result = try builder.createFunction("forward_and_loss_fn", func_type);
        const func_op = result.func_op;
        const block = result.entry_block;

        // Get block arguments (which are now created for us)
        const params_arg = block.getArgument(0);
        const inputs_arg = block.getArgument(1);
        // const targets_arg = block.getArgument(2); // (unused)
        std.log.info("buildForwardAndLossFunction: Function created with safe helper", .{});
        // --- END REFACTORED CODE ---

        // CRITICAL: Save the original insertion block and set the new one.
        const original_insertion_block = builder.getInsertionBlock();
        builder.setInsertionBlock(block);
        defer builder.setInsertionBlock(original_insertion_block); // Restore on exit

        // Build the forward/loss computation inside this function
        std.log.info("buildForwardAndLossFunction: Building forward computation in function body...", .{});
        const param_tensor = try builder.newTensor(params_arg);
        const input_flat = try ops.reshape(builder, try builder.newTensor(inputs_arg), &.{@intCast(param_count)});

        // Simple MSE loss computation
        const diff = try ops.subtract(builder, param_tensor, input_flat);
        const squared = try ops.multiply(builder, diff, diff);
        const loss = try ops.reduceSum(builder, squared, &.{0});

        // Return the loss
        std.log.info("buildForwardAndLossFunction: Adding func.return terminator...", .{});
        _ = try builder.createAndAttach("func.return", &.{loss.value}, &.{});
        std.log.info("buildForwardAndLossFunction: func.return terminator added", .{});

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
        const num_bytes = t.shape.elemCount() * t.shape.dtype.sizeInBytes();
        const data_slice: [*]const u8 = @ptrCast(raw_data_ptr);

        // 5. Return a duplicate of the data for the caller to own
        return self.allocator.dupe(u8, data_slice[0..num_bytes]);
    }

    /// Build the complete worker training graph as MLIR module using Function-as-a-Unit pattern
    /// This creates: forward_fn -> autodiff -> main_fn that orchestrates the full training step
    fn buildWorkerTrainingGraph(self: *Self, builder: *MLIRBuilder) ![]u8 {
        std.log.info("DiLoCo: Building REAL autodiff-enabled worker training graph with Function-as-a-Unit pattern...", .{});

        // === Part 1: Create the forward+loss function to be differentiated ===
        std.log.info("Creating forward+loss function...", .{});
        const forward_fn_op = try self.buildForwardAndLossFunction(builder);
        std.log.info("Forward+loss function created successfully", .{});
        // CRITICAL: Attach the newly created function to the module's body.
        std.log.info("Attaching function to module body...", .{});
        builder.module_body.appendOwnedOperation(forward_fn_op);
        std.log.info("Function attached successfully", .{});

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
        // For now, we'll use the convention that it names it "forward_and_loss_fn_grad"
        const grad_fn_name = "forward_and_loss_fn_grad";

        // === Part 3: Create the main worker function that orchestrates everything ===
        std.log.info("Creating main worker orchestration function...", .{});

        // Define types for main function
        const f32_type = mlir.Type.f32Type(builder.ctx);
        const param_count = self.config.param_count;
        const params_type = mlir.Type.rankedTensorType(builder.ctx, &.{@intCast(param_count)}, f32_type);
        const data_type = mlir.Type.rankedTensorType(builder.ctx, &.{ 4, 12 }, f32_type);
        const loss_type = mlir.Type.rankedTensorType(builder.ctx, &.{}, f32_type);

        // Create main function: func(params, inputs, targets) -> (new_params, loss)
        const main_input_types = [_]mlir.Type{ params_type, data_type, data_type };
        const main_result_types = [_]mlir.Type{ params_type, loss_type };
        const main_func_type = mlir.Type.functionType(builder.ctx, &main_input_types, &main_result_types);

        // --- REFACTORED MAIN FUNCTION CREATION ---
        // Use the safe helper to create the main function
        std.log.info("Creating main orchestration function with safe helper...", .{});
        const main_result = try builder.createFunction("main", main_func_type);
        const main_func_op = main_result.func_op;
        const main_block = main_result.entry_block;

        // CRITICAL: Attach the main function to the module's body.
        builder.module_body.appendOwnedOperation(main_func_op);

        // CRITICAL: Set the insertion point to the main function's body.
        const original_insertion_block = builder.getInsertionBlock();
        builder.setInsertionBlock(main_block);
        defer builder.setInsertionBlock(original_insertion_block);

        // Get main function arguments (created by the safe helper)
        const initial_params = main_block.getArgument(0);
        const inputs = main_block.getArgument(1);
        const targets = main_block.getArgument(2);
        std.log.info("Main function created with safe helper", .{});
        // --- END REFACTORED MAIN FUNCTION CREATION ---

        // Call the forward function to get the loss
        const forward_call_op = mlir.Operation.create(builder.ctx, "func.call", .{
            .operands = &.{ initial_params, inputs, targets },
            .results = &.{loss_type},
            .attributes = &.{
                .{ "callee", mlir.Attribute.stringAttr(builder.ctx, "forward_and_loss_fn") },
            },
            .location = builder.loc,
        });
        builder.insertion_block.appendOwnedOperation(forward_call_op);
        const loss_val = forward_call_op.getResult(0);

        // Call the gradient function to get gradients
        const one = try ops.constant(builder, 1.0, &.{}, f32_type); // Scalar 1.0 for loss gradient seed
        const grad_call_op = mlir.Operation.create(builder.ctx, "func.call", .{
            .operands = &.{ initial_params, inputs, targets, one.value },
            .results = &.{ params_type, data_type, data_type },
            .attributes = &.{
                .{ "callee", mlir.Attribute.stringAttr(builder.ctx, grad_fn_name) },
            },
            .location = builder.loc,
        });
        builder.insertion_block.appendOwnedOperation(grad_call_op);
        const param_grads = grad_call_op.getResult(0); // Gradients w.r.t. parameters

        // Apply simple gradient descent update
        const learning_rate = try ops.constant(builder, 0.01, &.{@intCast(param_count)}, f32_type);
        const grad_scaled = try ops.multiply(builder, try builder.newTensor(param_grads), learning_rate);
        const updated_params = try ops.subtract(builder, try builder.newTensor(initial_params), grad_scaled);

        // Return both updated parameters and loss from main function
        _ = try builder.createAndAttach("func.return", &.{ updated_params.value, loss_val }, &.{});

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

        const json_payload = std.json.Value{ .string = self.serialized_worker_graph };

        // Broadcast InitializeGraph message with the graph
        try self.coordinator.broadcastToWorkers(MessageType.INITIALIZE_GRAPH, json_payload);

        std.log.info("Training graph sent to {} workers for caching.", .{self.coordinator.getWorkerCount()});
    }

    /// Broadcast master parameters to all workers using Cap'n Proto
    fn broadcastMasterParameters(self: *Self) !void {
        if (self.master_parameters == null) {
            return error.ParametersNotInitialized;
        }

        // CORRECT APPROACH: Extract the raw data directly from the constant tensor.
        // This avoids misusing the executor and creating invalid IR.
        const serialized_params = try self.extractTensorData(self.master_parameters.?);
        defer self.allocator.free(serialized_params);

        // Create WorkerPayload and serialize with Cap'n Proto
        const worker_payload = binary_protocol.WorkerPayload{
            .params = serialized_params,
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

        std.log.debug("Broadcasted Cap'n Proto encoded parameters to {} workers", .{self.coordinator.getWorkerCount()});
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
            total_loss += result.loss;
        }

        return total_loss / @as(f32, @floatFromInt(worker_results.items.len));
    }

    /// Create MLIR tensor from parameter array
    fn createTensorFromArray(self: *Self, array: []const f32) !Tensor {
        const byte_data = std.mem.sliceAsBytes(array);
        const tensor_type = mlir.Type.rankedTensorType(self.mlir_builder.ctx, self.parameter_shape, self.element_type);

        var shape = try tensor.Shape.initWithDims(self.mlir_builder.ctx, self.parameter_shape, tensor.DType.f32);
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
    try std.testing.expectEqual(@as(usize, 1000), diloco.config.param_count);

    std.log.info("✓ DiLoCo MLIR algorithm test completed");
}
