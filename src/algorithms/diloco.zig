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
const data_loader = @import("../data_loader.zig");
const backend_selection = @import("../backend_selection.zig");

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
    model_mlir_path: []const u8, // Path to the model MLIR file

    pub fn default() DiLoCoConfig {
        return DiLoCoConfig{
            .base_config = TrainingConfig.default(),
            .tau = 10, // Default inner loop steps
            .nesterov_momentum = 0.9,
            .parameter_averaging = true,
            .model_mlir_path = "src/models/nanogpt_forward.mlir", // Default path
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

    // NEW: MLIR source for on-demand compilation
    worker_graph_mlir_source: []u8,

    // Data loading for real training batches
    data_loader: *DataLoader,

    const Self = @This();

    /// Helper function to find a func.func operation by its string name
    fn findFunctionByName(module: mlir.Module, name: []const u8) !mlir.Operation {
        const c = @import("../mlir/c.zig").c;

        const module_body = module.op().getRegion(0).getBlock(0);

        var maybe_op = module_body.getFirstOp();
        while (maybe_op) |op| {
            if (std.mem.eql(u8, op.getName(), "func.func")) {
                // It's a function, check its name
                const sym_name_attr = c.operationGetAttributeByName(op.handle, "sym_name");
                if (@intFromPtr(sym_name_attr) != 0 and c.attributeIsAString(sym_name_attr)) {
                    const string_attr = @as(*c.MlirStringAttribute, @ptrCast(sym_name_attr));
                    const func_name_ref = c.stringAttributeGetValue(string_attr);
                    const func_name = c.fromStringRef(func_name_ref);
                    if (std.mem.eql(u8, func_name, name)) {
                        return op; // Found it!
                    }
                }
            }
            maybe_op = op.getNext();
        }

        return error.FunctionNotFound;
    }

    // Change the init signature to accept the generic executor and DataLoader
    pub fn init(allocator: Allocator, coordinator: *Shepherd, config: DiLoCoConfig, executor: Executor, mlir_builder: *MLIRBuilder, dataset: *DataLoader) !Self {
        // The MLIRBuilder is now passed in, ensuring a single context.
        const element_type = mlir.Type.f32Type(mlir_builder.ctx);

        // === NEW: INTROSPECT MODEL SHAPES FROM .MLIR FILE ===
        std.debug.print("Introspecting model parameters from: {s}\n", .{config.model_mlir_path});
        const forward_pass_mlir_source = try std.fs.cwd().readFileAlloc(allocator, config.model_mlir_path, 10 * 1024 * 1024);
        defer allocator.free(forward_pass_mlir_source);

        const temp_module = try mlir.Module.parse(mlir_builder.ctx, forward_pass_mlir_source);
        defer temp_module.deinit();

        const forward_fn = try findFunctionByName(temp_module, "main");
        const func_type = forward_fn.getType().as(mlir.FunctionType) orelse return error.NotAFunctionType;

        var introspected_shapes = std.ArrayList([]i64).init(allocator);
        errdefer {
            for (introspected_shapes.items) |s| allocator.free(s);
            introspected_shapes.deinit();
        }

        // ASSUMPTION: The trainable parameters are the first N inputs to the function.
        // The final inputs are the data batches (e.g., input_ids, targets).
        // Let's assume for now that all but the last 2 inputs are trainable parameters.
        const num_data_inputs = 2; // e.g., input_ids, targets
        if (func_type.getNumInputs() <= num_data_inputs) {
            return error.NotEnoughInputsInModel;
        }
        const num_params = func_type.getNumInputs() - num_data_inputs;

        for (0..num_params) |i| {
            const input_type = func_type.getInput(i);
            const ranked_type = input_type.as(mlir.RankedTensorType) orelse return error.ParameterIsNotATensor;
            const shape = try ranked_type.getShape(allocator);
            try introspected_shapes.append(shape);
        }
        const parameter_shapes = try introspected_shapes.toOwnedSlice();
        std.debug.print("✓ Introspection complete. Found {} trainable parameter tensors.\n", .{parameter_shapes.len});

        // === END NEW SECTION ===

        // Configure and create Nesterov optimizer
        const nesterov_config = nesterov_mlir.NesterovMLIRConfiguration(f32){
            .learning_rate = config.base_config.learning_rate,
            .momentum = config.nesterov_momentum,
        };

        const nesterov_optimizer = try allocator.create(NesterovMLIR);
        nesterov_optimizer.* = try NesterovMLIR.init(allocator, mlir_builder, nesterov_config, element_type);

        // Create a partial instance to build the worker graph
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
            .worker_graph_mlir_source = undefined, // Will be set below
            .data_loader = dataset,
        };

        // Build the worker graph ONCE during initialization using the safe function builder
        const graph = try partial_self.buildWorkerTrainingGraph(mlir_builder);

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
            .worker_graph_mlir_source = graph, // Store the MLIR source
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

        // Free the stored MLIR source
        self.allocator.free(self.worker_graph_mlir_source);

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

        self.status = .initializing;
        monitoring.setStatus(.initializing);

        // Calculate total parameter count from introspected shapes
        var total_param_count: usize = 0;
        for (self.parameter_shapes) |shape| {
            var element_count: usize = 1;
            for (shape) |dim| {
                element_count *= @intCast(dim);
            }
            total_param_count += element_count;
        }
        monitoring.setModelInfo(total_param_count, self.config.base_config.learning_rate);

        // Initialize master parameters
        try self.initializeMasterParameters();

        // --- NEW COMPILE & DISTRIBUTE LOGIC ---
        // This happens AFTER workers have connected but BEFORE training starts.

        // 1. Wait for workers to connect (this logic is in Shepherd.startTraining,
        //    which calls this run() function. So by the time we are here, workers are present.)
        
        // 2. Identify unique backends in the connected worker pool
        var backend_set = std.AutoHashMap(backend_selection.Backend, void).init(self.allocator);
        defer backend_set.deinit();

        for (self.coordinator.worker_pool.items) |worker| {
            try backend_set.put(worker.backend, {});
        }

        // 3. Compile the MLIR source once for EACH unique backend
        var compiled_artifacts = std.AutoHashMap(backend_selection.Backend, []const u8).init(self.allocator);
        defer {
            var it = compiled_artifacts.iterator();
            while (it.next()) |entry| self.allocator.free(entry.value_ptr.*);
            compiled_artifacts.deinit();
        }
        
        var temp_mlir_ctx = try @import("../mlir_ctx.zig").MLIRContext.init(self.allocator);
        defer temp_mlir_ctx.deinit();

        var backend_it = backend_set.keyIterator();
        while (backend_it.next()) |backend_ptr| {
            const backend = backend_ptr.*;
            std.log.info("Compiling worker graph for backend: {s}", .{backend.toString()});

            const vmfb_bytes = try temp_mlir_ctx.compileToVMFB(
                self.allocator,
                self.worker_graph_mlir_source,
                backend.toString(),
            );
            try compiled_artifacts.put(backend, vmfb_bytes);
        }
        
        // 4. Distribute the correct compiled artifact to each worker (replaces setupWorkers)
        std.log.info("Distributing compiled artifacts to workers...", .{});
        for (self.coordinator.worker_pool.items) |worker| {
            const vmfb_bytes = compiled_artifacts.get(worker.backend).?;

            const b64_len = std.base64.standard.Encoder.calcSize(vmfb_bytes.len);
            const b64_encoded_vmfb = try self.allocator.alloc(u8, b64_len);
            defer self.allocator.free(b64_encoded_vmfb);

            const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_vmfb, vmfb_bytes).len;
            const json_payload = std.json.Value{ .string = b64_encoded_vmfb[0..encoded_len] };

            // Send a TARGETED message to this specific worker
            try self.coordinator.sendToWorker(worker.node_id, MessageType.INITIALIZE_GRAPH, json_payload);
        }
        
        // The old setupWorkers() is now fully replaced. We can delete it.
        // --- END NEW LOGIC ---

        self.status = .running;
        monitoring.setStatus(.running);

        // Main outer loop
        for (0..self.config.base_config.outer_loop_steps) |_| {
            const start_time = std.time.milliTimestamp();

            // Phase 0: Get a fresh batch of training data
            const batch_size = 4;
            const block_size = 8;
            const batch = try self.data_loader.getBatch(batch_size, block_size);
            defer self.allocator.free(batch.x);
            defer self.allocator.free(batch.y);

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
    }

    /// Initialize master parameters using MLIR for the loaded model
    fn initializeMasterParameters(self: *Self) !void {

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

        // Debug: Log extraction details
        std.log.debug("Extracting tensor data: elements={}, dtype_size={}, total_bytes={}", .{ elem_count, dtype_size, num_bytes });

        // Sanity check - if num_bytes is gigantic, there's a bug in the calculation
        if (num_bytes > 100 * 1024 * 1024) { // 100MB per tensor is way too large
            std.log.err("Tensor size calculation error: {} bytes (elements={}, dtype_size={})", .{ num_bytes, elem_count, dtype_size });
            return error.TensorSizeCalculationError;
        }

        // Check if raw data pointer is valid
        if (@intFromPtr(raw_data_ptr) == 0) {
            std.log.err("MLIR returned null pointer for tensor raw data", .{});
            return error.InvalidRawDataPointer;
        }

        const data_slice: [*]const u8 = @ptrCast(raw_data_ptr);
        const result = self.allocator.dupe(u8, data_slice[0..num_bytes]);

        // Debug: Check for patterns in the extracted data
        if (num_bytes > 0) {
            const sample_size = @min(16, num_bytes);
            std.log.debug("First {} bytes: {any}", .{ sample_size, data_slice[0..sample_size] });
        }

        // 5. Return a duplicate of the data for the caller to own
        return result;
    }

    /// Build the complete worker training graph as MLIR module using loaded .mlir file
    /// This loads a .mlir file, clones the forward function, and creates the complete training pipeline
    fn buildWorkerTrainingGraph(self: *Self, builder: *MLIRBuilder) ![]u8 {
        // === PHASE 1: LOAD AND PARSE THE FORWARD PASS MODULE ===
        std.debug.print("Loading and parsing forward pass from {s}...\n", .{self.config.model_mlir_path});

        // 1. Read the .mlir file from disk using the path from config
        const forward_pass_mlir_source = try std.fs.cwd().readFileAlloc(self.allocator, self.config.model_mlir_path, // Use the config path
            10 * 1024 * 1024 // 10MB limit
        );
        defer self.allocator.free(forward_pass_mlir_source);

        // 2. Parse the text file into a *temporary* MLIR module object.
        //    We use the builder's context to ensure compatibility.
        const forward_pass_module = try mlir.Module.parse(builder.ctx, forward_pass_mlir_source);
        defer forward_pass_module.deinit();

        std.debug.print("✓ Forward pass MLIR parsed successfully.\n", .{});

        // === PHASE 2: CLONE FORWARD PASS INTO OUR MAIN MODULE ===

        // 3. Find the main forward function in the parsed module.
        const forward_fn_to_clone = try findFunctionByName(forward_pass_module, "main");

        // 4. Clone the function into our builder's module. This makes it available for our graph.
        const c = @import("../mlir/c.zig").c;
        const cloned_forward_fn = mlir.Operation{ .handle = c.operationClone(forward_fn_to_clone.handle) };

        // 5. IMPORTANT: Change the name to avoid conflicts (e.g., with our future 'main' orchestrator)
        const new_fn_name = "model_forward_pass";
        const new_name_attr = mlir.Attribute.stringAttr(builder.ctx, new_fn_name);
        c.operationSetAttributeByName(cloned_forward_fn.handle, "sym_name", new_name_attr.handle);

        // 6. Add the cloned function to our main module.
        builder.module_body.appendOwnedOperation(cloned_forward_fn);

        std.debug.print("✓ Cloned forward function into main module as '{s}'.\n", .{new_fn_name});

        // === PHASE 3: APPLY AUTODIFF AND BUILD THE ORCHESTRATOR ===
        std.debug.print("Building gradient graph from the cloned forward function...\n", .{});

        const grad_fn_op = try autodiff.buildGradientGraph(self.allocator, builder, cloned_forward_fn);
        const grad_fn_name = "model_forward_pass_grad"; // Name given by autodiff

        // === PHASE 4: CREATE THE MAIN ORCHESTRATOR FUNCTION ===
        std.debug.print("Building the 'main' orchestrator function...\n", .{});

        // 4.1: INTROSPECT the forward function to define our main function's signature
        const forward_fn_type = cloned_forward_fn.getType().as(mlir.FunctionType) orelse return error.NotAFunctionType;
        const num_forward_inputs = forward_fn_type.getNumInputs();

        var main_input_types = std.ArrayList(mlir.Type).init(self.allocator);
        defer main_input_types.deinit();

        // Main inputs = all inputs of the forward function (params, data, etc.)
        for (0..num_forward_inputs) |i| {
            try main_input_types.append(forward_fn_type.getInput(i));
        }

        // Main outputs = all gradients of the forward function's inputs + the final loss
        const grad_fn_type = grad_fn_op.getType().as(mlir.FunctionType) orelse return error.NotAFunctionType;
        var main_output_types = std.ArrayList(mlir.Type).init(self.allocator);
        defer main_output_types.deinit();

        // Add gradient types for each of the forward function's inputs
        for (0..(grad_fn_type.getNumResults())) |i| {
            try main_output_types.append(grad_fn_type.getResult(i));
        }
        // Add the loss type (the single output of the forward function)
        try main_output_types.append(forward_fn_type.getResult(0));

        const main_func_type = mlir.Type.functionType(builder.ctx, main_input_types.items, main_output_types.items);

        // 4.2: Create the 'main' function
        const main_result = try builder.createFunction("main", main_func_type);
        const main_block = main_result.entry_block;
        builder.setInsertionBlock(main_block); // Set insertion point to 'main'

        // 4.3: Get all arguments for 'main'
        const main_func_args = try main_block.getArguments(self.allocator);
        defer self.allocator.free(main_func_args);

        // 4.4: Call the forward function to get the loss
        const forward_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, new_fn_name);
        const forward_call_op = try builder.createAndAttach("func.call", main_func_args, &.{forward_fn_type.getResult(0)}, .{
            .attributes = &.{.{ "callee", forward_callee_attr }},
        });
        const loss_val = forward_call_op.getResult(0);

        // 4.5: Call the gradient function to get gradients
        var grad_operands = std.ArrayList(mlir.Value).init(self.allocator);
        defer grad_operands.deinit();

        // Grad func inputs = forward func inputs + gradient of the output (loss)
        try grad_operands.appendSlice(main_func_args);
        const one = try ops.constant(builder, 1.0, &.{}, loss_val.getType().as(mlir.RankedTensorType).?.getElementType());
        try grad_operands.append(one.value);

        // Prepare grad function result types
        var grad_result_types = std.ArrayList(mlir.Type).init(self.allocator);
        defer grad_result_types.deinit();
        for (0..grad_fn_type.getNumResults()) |i| {
            try grad_result_types.append(grad_fn_type.getResult(i));
        }

        const grad_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, grad_fn_name);
        const grad_call_op = try builder.createAndAttach("func.call", grad_operands.items, grad_result_types.items, .{
            .attributes = &.{.{ "callee", grad_callee_attr }},
        });

        // 4.6: Return all gradients and the loss from 'main'
        var final_return_values = std.ArrayList(mlir.Value).init(self.allocator);
        defer final_return_values.deinit();

        // Add all results from the grad call
        for (0..grad_call_op.getNumResults()) |i| {
            try final_return_values.append(grad_call_op.getResult(i));
        }
        // Add the loss
        try final_return_values.append(loss_val);

        _ = try builder.createAndAttach("func.return", final_return_values.items, &.{}, .{});

        // === PHASE 5: FINALIZE AND SERIALIZE ===
        std.debug.print("✓ Final worker graph constructed successfully.\n", .{});

        // VERIFY the final module before serializing. This is a critical debugging step.
        if (!builder.module.op().verify()) {
            std.log.err("FINAL MODULE VERIFICATION FAILED. Dumping module:", .{});
            builder.module.op().dump();
            return error.ModuleVerificationFailed;
        }
        std.log.info("✓ Final module verification successful!", .{});

        return try @import("../mlir_ctx.zig").serializeMLIRModule(self.allocator, builder.module);
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

        // Serialize batch data as raw bytes (u32 tokens -> bytes)
        const input_ids_bytes = std.mem.sliceAsBytes(batch.x);
        const targets_bytes = std.mem.sliceAsBytes(batch.y);

        // VALIDATION: Check that we have valid data before serialization
        if (total_param_bytes.items.len == 0) {
            std.log.err("Empty parameters detected in broadcastMasterParametersAndBatch", .{});
            return error.EmptyParameters;
        }
        if (input_ids_bytes.len == 0) {
            std.log.err("Empty input_ids detected in broadcastMasterParametersAndBatch", .{});
            return error.EmptyInputIds;
        }
        if (targets_bytes.len == 0) {
            std.log.err("Empty targets detected in broadcastMasterParametersAndBatch", .{});
            return error.EmptyTargets;
        }

        std.log.info("Broadcasting params: {} bytes, input_ids: {} bytes, targets: {} bytes", .{ total_param_bytes.items.len, input_ids_bytes.len, targets_bytes.len });

        // Create WorkerPayload with parameters and batch data
        const worker_payload = binary_protocol.WorkerPayload{
            .params = total_param_bytes.items,
            .input_ids = input_ids_bytes,
            .targets = targets_bytes,
        };

        const capnp_bytes = try worker_payload.serialize(self.allocator);
        defer self.allocator.free(capnp_bytes);

        // Debug: Check Cap'n Proto serialization output
        std.log.info("Cap'n Proto serialized to {} bytes", .{capnp_bytes.len});
        if (capnp_bytes.len > 0) {
            const sample_size = @min(32, capnp_bytes.len);
            std.log.info("Cap'n Proto first {} bytes: {any}", .{ sample_size, capnp_bytes[0..sample_size] });

            // Check if data is mostly zeros (which would become 'A's in Base64)
            var zero_count: usize = 0;
            const check_size = @min(1024, capnp_bytes.len);
            for (capnp_bytes[0..check_size]) |byte| {
                if (byte == 0) zero_count += 1;
            }
            std.log.info("Cap'n Proto zero density: {}/{} ({d:.1}%)", .{ zero_count, check_size, (@as(f32, @floatFromInt(zero_count)) / @as(f32, @floatFromInt(check_size))) * 100.0 });
        }

        // Base64 encode the binary data
        const b64_len = std.base64.standard.Encoder.calcSize(capnp_bytes.len);
        const b64_encoded_payload = try self.allocator.alloc(u8, b64_len);
        defer self.allocator.free(b64_encoded_payload);

        const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_payload, capnp_bytes).len;

        // Create JSON payload with Base64-encoded binary data
        const json_payload = std.json.Value{ .string = b64_encoded_payload[0..encoded_len] };

        // Broadcast StartInnerLoop message with Cap'n Proto
        try self.coordinator.broadcastToWorkers(MessageType.START_INNER_LOOP, json_payload);
    }

    /// Collect results from workers after inner loop using Cap'n Proto
    fn collectWorkerResults(self: *Self) !ArrayList(WorkerResult) {
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

        return results;
    }

    /// Update master parameters using MLIR Nesterov optimizer
    fn updateMasterParametersMLIR(self: *Self, worker_results: ArrayList(WorkerResult)) !void {
        if (worker_results.items.len == 0) {
            return error.NoWorkerResults;
        }

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
            const tensor_data_bytes = averaged_params_bytes[byte_offset .. byte_offset + tensor_bytes];
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

        std.log.info("Parameters updated, loss: {d:.4}", .{self.metrics.loss});
    }

    /// Average parameters from all workers (new byte-based version)
    fn averageWorkerParameterBytes(self: *Self, worker_results: ArrayList(WorkerResult)) ![]u8 {
        // Calculate total parameter count from introspected shapes
        var total_param_count: usize = 0;
        for (self.parameter_shapes) |shape| {
            var element_count: usize = 1;
            for (shape) |dim| {
                element_count *= @intCast(dim);
            }
            total_param_count += element_count;
        }
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
