/// DiLoCo (Distributed Low-Communication) Algorithm Implementation
/// Implements the DiLoCo training algorithm for distributed learning with real MLIR optimizers
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const training_algorithm = @import("training_algorithm.zig");
const shepherd = @import("../nodes/controllers/shepherd.zig");
const message = @import("../network/message.zig");
const binary_protocol = @import("../network/capnp_zig_wrapper.zig");
const nesterov_mlir = @import("../optimizers/nesterov_mlir.zig");
const adam_mlir = @import("../optimizers/adam_mlir.zig");
const nesterov_host = @import("../optimizers/nesterov.zig");
const NesterovHost = nesterov_host.Nesterov;
const AdamMLIR = adam_mlir.AdamMLIR(f32);
const autodiff = @import("../autodiff/engine.zig");
const ops = @import("../core/ops.zig");
const mlir = @import("../mlir/wrapper.zig");
const tensor = @import("../core/tensor.zig");
const execution = @import("../execution.zig");
const monitoring = @import("../ui/monitoring.zig");
const data_loader = @import("../data/loader.zig");
const backend_selection = @import("../backends/selection.zig");
const wandb = @import("../ui/wandb.zig");

const TrainingAlgorithm = training_algorithm.TrainingAlgorithm;
const TrainingStatus = training_algorithm.TrainingStatus;
const TrainingConfig = training_algorithm.TrainingConfig;
const TrainingMetrics = training_algorithm.TrainingMetrics;
const Shepherd = shepherd.Shepherd;
const WorkerConfig = shepherd.WorkerConfig;
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
    // NEW FIELDS for experiment configuration
    data_path: []const u8,
    batch_size: usize,
    block_size: usize,

    // WandB configuration
    wandb_project: []const u8,
    wandb_entity: ?[]const u8,
    wandb_run_name: ?[]const u8,
    wandb_api_key: ?[]const u8,

    pub fn default() DiLoCoConfig {
        return DiLoCoConfig{
            .base_config = TrainingConfig.default(),
            .tau = 10, // Default inner loop steps
            .nesterov_momentum = 0.9,
            .parameter_averaging = true,
            .model_mlir_path = "src/models/nanogpt_forward.mlir", // Default path
            .data_path = "data/tiny_shakespeare.txt",
            .batch_size = 64,
            .block_size = 64, // Typically matches model context window
            .wandb_project = "pcp-distributed",
            .wandb_entity = null,
            .wandb_run_name = null,
            .wandb_api_key = null,
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
    adam_optimizer: *AdamMLIR, // Inner optimizer for worker training graph (AdamW)
    host_optimizer: NesterovHost, // Outer optimizer for master parameter updates (Nesterov)
    element_type: mlir.Type,

    // GPT-2 master parameters as multiple MLIR tensors
    master_parameters: ?[]Tensor,
    parameter_shapes: [][]i64, // Multiple parameter shapes for GPT-2
    
    // NEW: Data input shapes for worker buffer creation
    data_input_shapes: [][]i64, // Shapes for input_ids, targets, etc.

    // NEW: Generic executor - replaces direct backend knowledge
    executor: Executor,

    // NEW: MLIR source for on-demand compilation
    worker_graph_mlir_source: []u8,

    // Data loading for real training batches
    data_loader: *DataLoader,

    // WandB logging
    wandb_logger: wandb.WandBLogger,

    const Self = @This();

    /// Helper function to find a func.func operation by its string name
    fn findFunctionByName(module: mlir.Module, name: []const u8) !mlir.Operation {
        const c = @import("../mlir/c.zig").c;

        const module_body = module.op().getRegion(0).getBlock(0);

        var maybe_op = module_body.getFirstOp();
        while (maybe_op) |op| {
            if (std.mem.eql(u8, op.getName(), "func.func")) {
                // It's a function, check its name
                const sym_name_ref = c.stringRefFromString("sym_name");
                const sym_name_attr = c.operationGetAttributeByName(op.handle, sym_name_ref);
                if (@intFromPtr(sym_name_attr.ptr) != 0 and c.attributeIsAString(sym_name_attr)) {
                    const func_name_ref = c.stringAttributeGetValue(sym_name_attr);
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

    // Change the init signature to accept the generic executor (dataset now loaded by workers)
    pub fn init(allocator: Allocator, coordinator: *Shepherd, config: DiLoCoConfig, executor: Executor, mlir_builder: *MLIRBuilder) !Self {
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
        // Add errdefer to cleanup parameter_shapes if subsequent allocations fail
        errdefer {
            for (parameter_shapes) |s| allocator.free(s);
            allocator.free(parameter_shapes);
        }
        std.debug.print("✓ Introspection complete. Found {} trainable parameter tensors.\n", .{parameter_shapes.len});

        // Extract data input shapes (the last num_data_inputs)
        var data_shapes = std.ArrayList([]i64).init(allocator);
        errdefer {
            for (data_shapes.items) |s| allocator.free(s);
            data_shapes.deinit();
        }

        for (num_params..func_type.getNumInputs()) |i| {
            const input_type = func_type.getInput(i);
            const ranked_type = input_type.as(mlir.RankedTensorType) orelse return error.DataInputIsNotATensor;
            const shape = try ranked_type.getShape(allocator);
            try data_shapes.append(shape);
        }
        const data_input_shapes = try data_shapes.toOwnedSlice();
        errdefer {
            for (data_input_shapes) |s| allocator.free(s);
            allocator.free(data_input_shapes);
        }
        std.debug.print("✓ Found {} data input shapes.\n", .{data_input_shapes.len});

        // === END NEW SECTION ===

        // Configure and create AdamW optimizer for INNER loop (workers)
        // Per DiLoCo paper: "the inner optimizer is AdamW"
        const adam_config = adam_mlir.AdamMLIRConfiguration(f32){
            .learning_rate = config.base_config.learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .weight_decay = 0.01, // AdamW weight decay
        };

        const adam_optimizer = try allocator.create(AdamMLIR);
        adam_optimizer.* = try AdamMLIR.init(allocator, mlir_builder, adam_config, element_type);

        // Initialize Host Optimizer for OUTER loop (master updates on CPU)
        // Per DiLoCo paper: "the outer optimizer is Nesterov momentum"
        var host_opt = NesterovHost.init(allocator, .{
            .learning_rate = config.base_config.learning_rate,
            .momentum = config.nesterov_momentum,
        });

        // Pre-allocate velocity buffers based on introspection
        try host_opt.initParameters(parameter_shapes);

        // Initialize WandB logger
        const wb_config = wandb.WandBConfig{
            .project = config.wandb_project,
            .entity = config.wandb_entity,
            .run_name = config.wandb_run_name,
            .api_key = config.wandb_api_key orelse std.posix.getenv("WANDB_API_KEY"),
        };

        const logger = try wandb.WandBLogger.init(allocator, wb_config, .{
            .learning_rate = config.base_config.learning_rate,
            .batch_size = config.batch_size,
            .block_size = config.block_size,
            .tau = config.tau,
            .nesterov_momentum = config.nesterov_momentum,
            .optimizer = "Nesterov-AdamW",
        });

        // Create a partial instance to build the worker graph
        var partial_self = Self{
            .allocator = allocator,
            .coordinator = undefined, // Not used during graph building
            .config = config,
            .status = .not_started,
            .metrics = TrainingMetrics.init(),
            .current_epoch = 0,
            .mlir_builder = mlir_builder,
            .adam_optimizer = adam_optimizer,
            .host_optimizer = host_opt,
            .element_type = element_type,
            .master_parameters = null,
            .parameter_shapes = parameter_shapes,
            .data_input_shapes = data_input_shapes,
            .executor = executor,
            .worker_graph_mlir_source = undefined, // Will be set below
            .data_loader = undefined, // No longer used - workers load data locally
            .wandb_logger = logger,
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
            .adam_optimizer = adam_optimizer,
            .host_optimizer = host_opt,
            .element_type = element_type,
            .master_parameters = null,
            .parameter_shapes = parameter_shapes,
            .data_input_shapes = data_input_shapes,
            .executor = executor, // Store the generic executor
            .worker_graph_mlir_source = graph, // Store the MLIR source
            .data_loader = undefined, // No longer used - workers load data locally
            .wandb_logger = logger,
        };
    }

    pub fn deinit(self: *Self) void {
        // Deinit WandB logger first
        self.wandb_logger.deinit();

        if (self.master_parameters) |params| {
            for (params) |param| {
                param.deinit();
            }
            self.allocator.free(params);
        }

        self.adam_optimizer.deinit();
        self.allocator.destroy(self.adam_optimizer);

        // Deinit host optimizer
        self.host_optimizer.deinit();

        // We no longer deinit the builder, as we don't own it.

        // Free the stored MLIR source
        self.allocator.free(self.worker_graph_mlir_source);

        // Free parameter shapes
        for (self.parameter_shapes) |shape| {
            self.allocator.free(shape);
        }
        self.allocator.free(self.parameter_shapes);

        // Free data input shapes
        for (self.data_input_shapes) |shape| {
            self.allocator.free(shape);
        }
        self.allocator.free(self.data_input_shapes);
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

        // 2. Identify unique (backend, target_arch) combinations in the connected worker pool
        var config_set = std.HashMap(WorkerConfig, void, std.hash_map.AutoContext(WorkerConfig), std.hash_map.default_max_load_percentage).init(self.allocator);
        defer config_set.deinit();

        self.coordinator.worker_pool_mutex.lock();
        for (self.coordinator.worker_pool.items) |worker| {
            const config = WorkerConfig{
                .backend = worker.backend,
                .target_arch = worker.target_arch,
            };
            try config_set.put(config, {});
        }
        self.coordinator.worker_pool_mutex.unlock();

        // 3. Compile the MLIR source once for EACH unique (backend, target_arch) combination
        var compiled_artifacts = std.HashMap(WorkerConfig, []const u8, std.hash_map.AutoContext(WorkerConfig), std.hash_map.default_max_load_percentage).init(self.allocator);
        defer {
            var it = compiled_artifacts.iterator();
            while (it.next()) |entry| self.allocator.free(entry.value_ptr.*);
            compiled_artifacts.deinit();
        }

        var temp_mlir_ctx = try @import("../mlir/context.zig").MLIRContext.init(self.allocator);
        defer temp_mlir_ctx.deinit();

        var config_it = config_set.keyIterator();
        while (config_it.next()) |config_ptr| {
            const config = config_ptr.*;
            if (config.target_arch) |target| {
                std.log.info("Compiling worker graph for backend: {s} with target: {s}", .{config.backend.toString(), target});
            } else {
                std.log.info("Compiling worker graph for backend: {s}", .{config.backend.toString()});
            }

            const vmfb_bytes = try temp_mlir_ctx.compileToVMFB(
                self.allocator,
                self.worker_graph_mlir_source,
                config.backend.toIreeCompilationTarget(),
                config.target_arch,
            );
            try compiled_artifacts.put(config, vmfb_bytes);
        }

        // Copy compiled artifacts to shepherd for later worker reconnections
        // We need to duplicate the VMFB bytes since the local compiled_artifacts will be freed
        var artifact_it = compiled_artifacts.iterator();
        while (artifact_it.next()) |entry| {
            const vmfb_copy = try self.allocator.dupe(u8, entry.value_ptr.*);
            try self.coordinator.compiled_artifacts.put(entry.key_ptr.*, vmfb_copy);
        }

        // 4. Create the JSON payload for BOTH parameter shapes and data input shapes
        var param_shape_array = std.json.Array.init(self.allocator);
        defer param_shape_array.deinit();
        for (self.parameter_shapes) |shape_slice| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape_slice) |dim| {
                try dim_array.append(std.json.Value{ .integer = dim });
            }
            try param_shape_array.append(std.json.Value{ .array = dim_array });
        }
        const param_shapes_json = std.json.Value{ .array = param_shape_array };

        var data_shape_array = std.json.Array.init(self.allocator);
        defer data_shape_array.deinit();
        for (self.data_input_shapes) |shape_slice| {
            var dim_array = std.json.Array.init(self.allocator);
            for (shape_slice) |dim| {
                try dim_array.append(std.json.Value{ .integer = dim });
            }
            try data_shape_array.append(std.json.Value{ .array = dim_array });
        }
        const data_shapes_json = std.json.Value{ .array = data_shape_array };

        // 5. Distribute the compiled artifacts AND the data input shapes
        std.log.info("Distributing compiled artifacts and input shapes to workers...", .{});

        // Copy worker list to avoid holding mutex while sending messages
        self.coordinator.worker_pool_mutex.lock();
        const worker_list = try self.allocator.dupe(shepherd.WorkerConnection, self.coordinator.worker_pool.items);
        self.coordinator.worker_pool_mutex.unlock();
        defer self.allocator.free(worker_list);

        for (worker_list) |worker| {
            const worker_config = WorkerConfig{
                .backend = worker.backend,
                .target_arch = worker.target_arch,
            };
            const vmfb_bytes = compiled_artifacts.get(worker_config).?;

            const b64_len = std.base64.standard.Encoder.calcSize(vmfb_bytes.len);
            const b64_encoded_vmfb = try self.allocator.alloc(u8, b64_len);
            defer self.allocator.free(b64_encoded_vmfb);

            const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_vmfb, vmfb_bytes).len;

            // Create a JSON object containing vmfb, parameter shapes, and data input shapes
            var payload = std.json.ObjectMap.init(self.allocator);
            defer payload.deinit();
            try payload.put("vmfb", std.json.Value{ .string = b64_encoded_vmfb[0..encoded_len] });
            try payload.put("parameter_shapes", param_shapes_json);
            try payload.put("data_input_shapes", data_shapes_json);

            // Send the combined payload in a single message (sendToWorker will lock internally)
            try self.coordinator.sendToWorker(worker.node_id, MessageType.INITIALIZE_GRAPH, .{ .object = payload });
        }
        
        // The old setupWorkers() is now fully replaced. We can delete it.
        // --- END NEW LOGIC ---

        self.status = .running;
        monitoring.setStatus(.running);

        // Main outer loop with snapshot-based dynamic topology
        for (0..self.config.base_config.outer_loop_steps) |step| {
            const start_time = std.time.milliTimestamp();

            // Take snapshot of currently connected workers
            const participants = try self.coordinator.snapshotWorkers();
            defer participants.deinit();

            if (participants.items.len == 0) {
                std.log.warn("No workers connected. Waiting...", .{});
                std.time.sleep(1 * std.time.ns_per_s);
                continue;
            }

            // Initialize any new workers that haven't received the graph yet
            try self.initializeNewWorkers(participants.items);

            // Broadcast to snapshot participants only (workers will load data locally)
            const workers_assigned = try self.broadcastToSnapshot(participants.items);

            if (workers_assigned == 0) {
                std.log.warn("Round {} failed: No workers assigned. Skipping update.", .{step});
                continue;
            }

            // Streaming Accumulation: Collect and apply gradients on the fly
            try self.accumulateAndApplyGradients(workers_assigned);

            // Save checkpoint every 10 steps or at the end
            if ((step + 1) % 10 == 0 or (step + 1) == self.config.base_config.outer_loop_steps) {
                try self.saveCheckpoint(step + 1);
            }

            // Update metrics
            self.metrics.outer_loop_count += 1;

            // Get true epoch from DataManager
            if (self.coordinator.data_manager) |*dm| {
                self.current_epoch = dm.current_epoch;
            } else {
                // Fallback if no DM
                self.current_epoch = 1;
            }

            // Calculate epoch time and update monitoring
            const end_time = std.time.milliTimestamp();
            const epoch_time_ms: u64 = @intCast(end_time - start_time);
            monitoring.setEpochTime(epoch_time_ms);
            monitoring.setMetrics(self.current_epoch, self.metrics.loss, self.coordinator.getWorkerCount());

            // Log to WandB
            self.wandb_logger.log(.{
                .outer_step = step,
                .epoch = self.current_epoch,
                .loss = self.metrics.loss,
                .min_worker_loss = self.metrics.loss, // Note: Individual worker losses not tracked in streaming mode
                .max_worker_loss = self.metrics.loss, // Note: Individual worker losses not tracked in streaming mode
                .active_workers = workers_assigned,
                .learning_rate = self.host_optimizer.config.learning_rate,
                .epoch_time_ms = epoch_time_ms,
            });

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
        const value_ref = c.stringRefFromString("value");
        const value_attr = c.operationGetAttributeByName(defining_op.handle, value_ref);
        if (@intFromPtr(value_attr.ptr) == 0 or !c.mlirAttributeIsADenseElements(value_attr)) {
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

    /// Save current master parameters to a binary file
    fn saveCheckpoint(self: *Self, step: usize) !void {
        var path_buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "checkpoint_{d}.bin", .{step});

        std.log.info("Saving checkpoint to {s}...", .{path});
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var writer = file.writer();

        // 1. Header
        try writer.writeAll("PCPCHECK");
        try writer.writeInt(u32, 1, .little); // Version

        if (self.master_parameters) |params| {
            // 2. Number of tensors
            try writer.writeInt(u32, @intCast(params.len), .little);

            for (params) |t| {
                // 3a. Write Shape info
                const rank = t.shape.rank();
                try writer.writeInt(u8, @intCast(rank), .little);

                const dims = try t.shape.getDims(self.allocator);
                defer self.allocator.free(dims);
                for (dims) |d| {
                    try writer.writeInt(i64, d, .little);
                }

                // 3b. Write Data
                const raw_bytes = try self.extractTensorData(t);
                defer self.allocator.free(raw_bytes);

                // Write size of data block followed by data
                try writer.writeInt(u64, @intCast(raw_bytes.len), .little);
                try writer.writeAll(raw_bytes);
            }
        }
        std.log.info("✓ Checkpoint saved.", .{});
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
        const sym_name_attr_ref = c.stringRefFromString("sym_name");
        c.operationSetAttributeByName(cloned_forward_fn.handle, sym_name_attr_ref, new_name_attr.handle);

        // 6. Add the cloned function to our main module.
        builder.module_body.appendOwnedOperation(cloned_forward_fn);

        std.debug.print("✓ Cloned forward function into main module as '{s}'.\n", .{new_fn_name});

        // === PHASE 3: APPLY AUTODIFF AND BUILD THE ORCHESTRATOR ===
        std.debug.print("Building gradient graph from the cloned forward function...\n", .{});

        _ = try autodiff.buildGradientGraph(self.allocator, builder, cloned_forward_fn);
        const grad_fn_name = "model_forward_pass_grad"; // Name given by autodiff

        // === PHASE 4: CREATE THE MAIN ORCHESTRATOR FUNCTION ===
        std.debug.print("Building the 'main' orchestrator function with AdamW state...\n", .{});

        // 4.1: INTROSPECT the forward function to define our main function's signature
        const forward_fn_type = cloned_forward_fn.getType().as(mlir.FunctionType) orelse return error.NotAFunctionType;
        const num_forward_inputs = forward_fn_type.getNumInputs();

        var main_input_types = std.ArrayList(mlir.Type).init(self.allocator);
        defer main_input_types.deinit();

        const num_params = self.parameter_shapes.len;

        // Main inputs = Parameters (N) + M States (N) + V States (N) + Timestep (1) + Data inputs
        // A. Parameters
        for (0..num_params) |i| {
            try main_input_types.append(forward_fn_type.getInput(i));
        }
        // B. M States (same shape as params)
        for (0..num_params) |i| {
            try main_input_types.append(forward_fn_type.getInput(i));
        }
        // C. V States (same shape as params)
        for (0..num_params) |i| {
            try main_input_types.append(forward_fn_type.getInput(i));
        }
        // D. Timestep (scalar tensor)
        const scalar_type = mlir.Type.rankedTensorType(self.mlir_builder.ctx, &.{}, self.element_type);
        try main_input_types.append(scalar_type);
        // E. Data inputs (from original forward function, skipping parameters)
        for (num_params..num_forward_inputs) |i| {
            try main_input_types.append(forward_fn_type.getInput(i));
        }

        // Main outputs = Parameters (N) + M States (N) + V States (N) + Loss (1)
        var main_output_types = std.ArrayList(mlir.Type).init(self.allocator);
        defer main_output_types.deinit();

        // Return updated parameters (same types as parameter inputs)
        for (0..num_params) |i| {
            try main_output_types.append(forward_fn_type.getInput(i));
        }
        // Return updated M states
        for (0..num_params) |i| {
            try main_output_types.append(forward_fn_type.getInput(i));
        }
        // Return updated V states
        for (0..num_params) |i| {
            try main_output_types.append(forward_fn_type.getInput(i));
        }
        // Add the loss type (the single output of the forward function)
        try main_output_types.append(forward_fn_type.getResult(0));

        const main_func_type = try mlir.Type.functionType(builder.allocator, builder.ctx, main_input_types.items, main_output_types.items);

        // 4.2: Create the 'main' function
        const main_result = try builder.createFunction("main", main_func_type);
        const main_block = main_result.entry_block;
        builder.setInsertionBlock(main_block); // Set insertion point to 'main'

        // 4.3: Get all arguments for 'main' and slice them
        const main_func_args = try main_block.getArguments(self.allocator);
        defer self.allocator.free(main_func_args);

        // Slice arguments: [Params (N), Ms (N), Vs (N), Timestep (1), Data (D)]
        const params_in = main_func_args[0..num_params];
        const m_in = main_func_args[num_params .. num_params * 2];
        const v_in = main_func_args[num_params * 2 .. num_params * 3];
        const timestep = main_func_args[num_params * 3];
        const data_in = main_func_args[num_params * 3 + 1 ..]; // Data inputs

        // 4.4: Call the forward function to get the loss (only params + data)
        var forward_operands = std.ArrayList(mlir.Value).init(self.allocator);
        defer forward_operands.deinit();
        try forward_operands.appendSlice(params_in);
        try forward_operands.appendSlice(data_in);

        const forward_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, new_fn_name);
        const forward_call_op = try builder.createAndAttach("func.call", forward_operands.items, &.{forward_fn_type.getResult(0)}, .{
            .attributes = &.{.{ "callee", forward_callee_attr }},
        });
        const loss_val = forward_call_op.getResult(0);

        // 4.5: Call the gradient function to get gradients (params + data + loss_grad)
        var grad_operands = std.ArrayList(mlir.Value).init(self.allocator);
        defer grad_operands.deinit();

        try grad_operands.appendSlice(params_in);
        try grad_operands.appendSlice(data_in);
        const one = try ops.constant(builder, 1.0, &.{}, loss_val.getType().as(mlir.RankedTensorType).?.getElementType());
        try grad_operands.append(one.value);

        // Prepare grad function result types
        // Note: Autodiff returns gradients for ALL inputs (params + data), not just parameters
        var grad_result_types = std.ArrayList(mlir.Type).init(self.allocator);
        defer grad_result_types.deinit();
        for (params_in) |p| try grad_result_types.append(p.getType());
        for (data_in) |d| try grad_result_types.append(d.getType());

        const grad_callee_attr = mlir.Attribute.symbolRefAttr(builder.ctx, grad_fn_name);
        const grad_call_op = try builder.createAndAttach("func.call", grad_operands.items, grad_result_types.items, .{
            .attributes = &.{.{ "callee", grad_callee_attr }},
        });

        // Note: grad_call_op returns [param_grads..., data_grads...]
        // We only need the parameter gradients (first num_params results)

        // 4.6: Apply AdamW optimizer updates to get new parameters and states
        std.debug.print("Adding AdamW optimizer update step to worker graph...\n", .{});

        var final_return_values = std.ArrayList(mlir.Value).init(self.allocator);
        defer final_return_values.deinit();

        const timestep_tensor = try builder.newTensor(timestep);

        // Collect results from optimizer updates
        var new_params = std.ArrayList(mlir.Value).init(self.allocator);
        var new_ms = std.ArrayList(mlir.Value).init(self.allocator);
        var new_vs = std.ArrayList(mlir.Value).init(self.allocator);
        defer new_params.deinit();
        defer new_ms.deinit();
        defer new_vs.deinit();

        // For each parameter: apply AdamW optimizer update ONCE and collect all three outputs
        for (0..num_params) |i| {
            const param_tensor = try builder.newTensor(params_in[i]);
            const grad_tensor = try builder.newTensor(grad_call_op.getResult(i));
            const m_tensor = try builder.newTensor(m_in[i]);
            const v_tensor = try builder.newTensor(v_in[i]);

            // Apply AdamW update with all 5 parameters (returns new_params, new_m, new_v)
            const result = try self.adam_optimizer.update(param_tensor, grad_tensor, m_tensor, v_tensor, timestep_tensor);

            try new_params.append(result.new_params.value);
            try new_ms.append(result.new_m.value);
            try new_vs.append(result.new_v.value);
        }

        // Return in order: params, M states, V states, loss
        try final_return_values.appendSlice(new_params.items);
        try final_return_values.appendSlice(new_ms.items);
        try final_return_values.appendSlice(new_vs.items);
        try final_return_values.append(loss_val);

        _ = try builder.createAndAttach("func.return", final_return_values.items, &.{}, .{});

        std.debug.print("✓ Worker graph now returns updated parameters + M + V + loss\n", .{});

        // === PHASE 5: FINALIZE AND SERIALIZE ===
        std.debug.print("✓ Final worker graph constructed successfully.\n", .{});

        // VERIFY the final module before serializing. This is a critical debugging step.
        if (!builder.module.op().verify()) {
            std.log.err("FINAL MODULE VERIFICATION FAILED. Dumping module:", .{});
            builder.module.op().dump();
            return error.ModuleVerificationFailed;
        }
        std.log.info("✓ Final module verification successful!", .{});

        // === DUMP MLIR TO FILE FOR DEBUGGING ===
        const serialized_mlir = try @import("../mlir/context.zig").serializeMLIRModule(self.allocator, builder.module);

        // Write the MLIR to project root as worker_graph_after_autodiff.mlir
        const mlir_dump_path = "worker_graph_after_autodiff.mlir";
        std.log.info("Dumping MLIR to {s} for analysis...", .{mlir_dump_path});
        try std.fs.cwd().writeFile(.{ .sub_path = mlir_dump_path, .data = serialized_mlir });
        std.log.info("✓ MLIR dumped to {s}", .{mlir_dump_path});

        return serialized_mlir;
    }

    /// Initialize any new workers in the snapshot that need the graph
    fn initializeNewWorkers(self: *Self, workers: []const shepherd.WorkerConnection) !void {
        for (workers) |worker| {
            if (worker.status == .Connected) {
                std.log.info("Initializing new worker {} (Backend: {s})...", .{worker.node_id, worker.backend.toString()});

                // Get compiled artifact for this worker's backend
                const worker_config = WorkerConfig{
                    .backend = worker.backend,
                    .target_arch = worker.target_arch,
                };
                const vmfb_bytes = self.coordinator.compiled_artifacts.get(worker_config) orelse return error.VMFBNotFound;

                const b64_len = std.base64.standard.Encoder.calcSize(vmfb_bytes.len);
                const b64_encoded_vmfb = try self.allocator.alloc(u8, b64_len);
                defer self.allocator.free(b64_encoded_vmfb);

                const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_vmfb, vmfb_bytes).len;

                // Create JSON payload with VMFB and shapes
                var param_shape_array = std.json.Array.init(self.allocator);
                defer param_shape_array.deinit();
                for (self.parameter_shapes) |shape_slice| {
                    var dim_array = std.json.Array.init(self.allocator);
                    for (shape_slice) |dim| {
                        try dim_array.append(std.json.Value{ .integer = dim });
                    }
                    try param_shape_array.append(std.json.Value{ .array = dim_array });
                }

                var data_shape_array = std.json.Array.init(self.allocator);
                defer data_shape_array.deinit();
                for (self.data_input_shapes) |shape_slice| {
                    var dim_array = std.json.Array.init(self.allocator);
                    for (shape_slice) |dim| {
                        try dim_array.append(std.json.Value{ .integer = dim });
                    }
                    try data_shape_array.append(std.json.Value{ .array = dim_array });
                }

                var payload = std.json.ObjectMap.init(self.allocator);
                defer payload.deinit();
                try payload.put("vmfb", std.json.Value{ .string = b64_encoded_vmfb[0..encoded_len] });
                try payload.put("parameter_shapes", std.json.Value{ .array = param_shape_array });
                try payload.put("data_input_shapes", std.json.Value{ .array = data_shape_array });

                try self.coordinator.sendToWorker(worker.node_id, MessageType.INITIALIZE_GRAPH, .{ .object = payload });
                self.coordinator.setWorkerStatus(worker.node_id, .GraphInitialized);
            }
        }
    }

    /// Broadcast to snapshot participants only
    /// Returns the number of workers that were actually assigned chunks
    fn broadcastToSnapshot(self: *Self, workers: []const shepherd.WorkerConnection) !usize {
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

        // Create WorkerPayload with parameters (no batch data - workers load locally)
        const empty_slice: []const u8 = &[_]u8{};
        const worker_payload = binary_protocol.WorkerPayload{
            .params = total_param_bytes.items,
            .input_ids = empty_slice,
            .targets = empty_slice,
        };
        const capnp_bytes = try worker_payload.serialize(self.allocator);
        defer self.allocator.free(capnp_bytes);

        // Base64 encode the Cap'n Proto message for JSON transport
        const b64_len = std.base64.standard.Encoder.calcSize(capnp_bytes.len);
        const b64_encoded_params = try self.allocator.alloc(u8, b64_len);
        defer self.allocator.free(b64_encoded_params);
        const encoded_len = std.base64.standard.Encoder.encode(b64_encoded_params, capnp_bytes).len;

        // Send to each worker with their assigned data chunk
        var workers_assigned: usize = 0;
        for (workers) |worker| {
            // Assign data chunk for this worker
            const chunk = if (self.coordinator.data_manager) |*dm|
                dm.assignNextChunk(worker.node_id)
            else
                null;

            if (chunk == null) {
                std.log.warn("No data chunk available for worker {}", .{worker.node_id});
                continue;
            }

            // Build JSON payload with params and chunk info
            var payload_map = std.json.ObjectMap.init(self.allocator);
            defer payload_map.deinit();

            try payload_map.put("params", std.json.Value{ .string = b64_encoded_params[0..encoded_len] });
            try payload_map.put("offset", std.json.Value{ .integer = @intCast(chunk.?.offset) });
            try payload_map.put("length", std.json.Value{ .integer = @intCast(chunk.?.length) });
            try payload_map.put("chunk_id", std.json.Value{ .integer = @intCast(chunk.?.id) });
            try payload_map.put("data_path", std.json.Value{ .string = self.config.data_path });

            const json_payload = std.json.Value{ .object = payload_map };

            self.coordinator.sendToWorker(worker.node_id, MessageType.START_INNER_LOOP, json_payload) catch |err| {
                std.log.err("Failed to send task to worker {}: {}", .{worker.node_id, err});
            };

            std.log.info("Assigned chunk {} (offset: {}, len: {}) to worker {}", .{
                chunk.?.id,
                chunk.?.offset,
                chunk.?.length,
                worker.node_id,
            });

            workers_assigned += 1;
        }

        return workers_assigned;
    }

    /// Collect results from workers, accumulate gradients on the fly, and apply update.
    /// This keeps memory usage constant O(ModelSize) regardless of worker count.
    fn accumulateAndApplyGradients(self: *Self, expected_count: usize) !void {
        if (expected_count == 0) return;
        if (self.master_parameters == null) return error.ParametersNotInitialized;

        // 1. Initialize Accumulator (Zeroed buffer matching total parameter size)
        // We calculate total size first
        var total_param_bytes: usize = 0;
        for (self.parameter_shapes) |shape| {
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);
            total_param_bytes += elem_count * @sizeOf(f32);
        }

        const accumulator = try self.allocator.alloc(u8, total_param_bytes);
        defer self.allocator.free(accumulator);
        @memset(accumulator, 0); // Important: Init to zero

        var collected_count: usize = 0;
        var total_loss: f32 = 0.0;

        std.log.info("Streaming results from {} workers...", .{expected_count});

        // 2. Collection Loop
        while (collected_count < expected_count) {
            // Fetch one message at a time
            const responses = try self.coordinator.collectFromWorkers(MessageType.INNER_LOOP_COMPLETE, 1);
            defer {
                for (responses.items) |*msg| msg.deinitClone(self.allocator);
                responses.deinit();
            }

            if (responses.items.len == 0) continue; // Safety check

            const response = responses.items[0];

            // Handle chunk completion tracking
            if (response.data == .object) {
                if (response.data.object.get("chunk_id")) |chunk_val| {
                    if (chunk_val == .integer) {
                        const chunk_id: usize = @intCast(chunk_val.integer);
                        if (self.coordinator.data_manager) |*dm| {
                            dm.markComplete(chunk_id);
                            std.log.info("Worker {} completed chunk {}", .{ response.sender_node, chunk_id });
                        }
                    }
                }
            }

            // Decode Payload
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            // Extract base64 payload
            const b64_payload = switch (response.data) {
                .string => |s| s,
                .object => |obj| blk: {
                    if (obj.get("params")) |p| {
                        break :blk switch (p) {
                            .string => |s| s,
                            else => return error.InvalidMessageFormat,
                        };
                    }
                    return error.InvalidMessageFormat;
                },
                else => return error.InvalidMessageFormat,
            };

            const capnp_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_payload);
            const capnp_bytes = try arena.allocator().alloc(u8, capnp_len);
            try std.base64.standard.Decoder.decode(capnp_bytes, b64_payload);

            const reader = try binary_protocol.ShepherdPayload.Reader.init(capnp_bytes);
            defer reader.deinit();

            const delta_bytes = try reader.getUpdatedParams(); // This is now the Delta
            const loss_val = try reader.getLoss();

            // 3. Accumulate: Accumulator += Delta
            if (delta_bytes.len != total_param_bytes) {
                std.log.err("Worker delta size mismatch! Expected {}, got {}", .{ total_param_bytes, delta_bytes.len });
                return error.DimensionMismatch;
            }

            const acc_f32: [*]f32 = @ptrCast(@alignCast(accumulator.ptr));
            const delta_f32: [*]const f32 = @ptrCast(@alignCast(delta_bytes.ptr));
            const num_floats = total_param_bytes / @sizeOf(f32);

            // Vectorize addition
            for (0..num_floats) |i| {
                acc_f32[i] += delta_f32[i];
            }

            total_loss += loss_val;
            collected_count += 1;

            // Message is freed here by the defer/arena logic, keeping memory low
        }

        // 4. Average the Accumulator
        const acc_f32: [*]f32 = @ptrCast(@alignCast(accumulator.ptr));
        const num_floats = total_param_bytes / @sizeOf(f32);
        const divisor = @as(f32, @floatFromInt(collected_count));

        for (0..num_floats) |i| {
            acc_f32[i] /= divisor;
        }

        // 5. Apply to Master Parameters (Per Tensor)
        var byte_offset: usize = 0;
        const param_tensors = self.master_parameters.?;

        for (param_tensors, 0..) |*param_tensor_ptr, i| {
            const shape = self.parameter_shapes[i];
            var elem_count: usize = 1;
            for (shape) |dim| elem_count *= @intCast(dim);
            const tensor_size = elem_count * @sizeOf(f32);

            // Get current master data
            const master_bytes = try self.extractTensorData(param_tensor_ptr.*);
            defer self.allocator.free(master_bytes);
            const master_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, master_bytes));

            // Get corresponding averaged delta
            const grad_slice = acc_f32[byte_offset / 4 .. (byte_offset + tensor_size) / 4];

            // Update using Nesterov (Host Optimizer)
            // Note: master_f32 is modified in-place
            try self.host_optimizer.update(i, master_f32, grad_slice);

            // Wrap back to MLIR Tensor
            param_tensor_ptr.deinit();
            param_tensor_ptr.* = try self.createTensorFromArrayWithShape(master_f32, shape);

            byte_offset += tensor_size;
        }

        // Update metrics
        self.metrics.loss = total_loss / divisor;
        std.log.info("Round complete. Avg Loss: {d:.4}", .{self.metrics.loss});
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

    /// Check if training should stop
    fn shouldStop(self: *Self) bool {
        // Stop if we've reached the configured outer loop steps limit
        // (Note: We keep this to allow "train for X steps", but if user wants
        // pure epoch-based training, they set outer_loop_steps very high)
        if (self.metrics.outer_loop_count >= self.config.base_config.outer_loop_steps) {
            std.log.info("Stopping: Reached maximum outer loop steps ({})", .{self.config.base_config.outer_loop_steps});
            return true;
        }

        // Stop if DataManager says we are done (Max epochs reached)
        if (self.coordinator.data_manager) |*dm| {
            if (dm.current_epoch > dm.max_epochs) {
                std.log.info("Stopping: Reached maximum epochs ({})", .{dm.max_epochs});
                return true;
            }
        }

        if (self.metrics.loss < 0.01) {
            std.log.info("Stopping: Converged (loss < 0.01)", .{});
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
