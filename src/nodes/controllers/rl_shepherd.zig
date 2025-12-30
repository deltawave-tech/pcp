const std = @import("std");
const Allocator = std.mem.Allocator;
const Shepherd = @import("shepherd.zig").Shepherd;
const MessageType = @import("../../network/message.zig").MessageType;
const MessageEnvelope = @import("../../network/message.zig").MessageEnvelope;
const ArrayList = std.ArrayList;
const backend_selection = @import("../../backends/selection.zig");
const IreeBackend = @import("../../backends/iree.zig").IreeBackend;
const mlir_ctx = @import("../../mlir/context.zig");
const Nesterov = @import("../../optimizers/nesterov.zig").Nesterov;

pub const RolloutData = struct {
    prompt: []const i64,
    completion: []const i64,
    allocator: Allocator,

    pub fn deinit(self: *RolloutData) void {
        self.allocator.free(self.prompt);
        self.allocator.free(self.completion);
    }
};

pub const RLShepherd = struct {
    base: Shepherd, // Composition

    // Weight management
    weight_buffer: ?[]u8, // Raw weight file buffer
    master_parameters: ?[][]const u8, // Sliced parameter tensors
    parameter_shapes: ?[][]const i64, // Shapes for each parameter

    // Training Execution
    training_backend: ?*IreeBackend,
    training_vmfb: ?[]u8,
    optimizer: Nesterov, // Pure Zig optimizer

    // Generation Model State
    generation_vmfb: ?[]u8,
    gen_param_shapes: ?[][]i64,
    gen_data_shapes: ?[][]i64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .base = Shepherd.init(allocator),
            .weight_buffer = null,
            .master_parameters = null,
            .parameter_shapes = null,
            .training_backend = null,
            .training_vmfb = null,
            .optimizer = Nesterov.init(allocator, .{ .learning_rate = 1e-5 }),
            .generation_vmfb = null,
            .gen_param_shapes = null,
            .gen_data_shapes = null,
        };
    }

    pub fn deinit(self: *Self) void {
        // Clean up weight management
        if (self.master_parameters) |params| {
            self.base.allocator.free(params);
        }
        if (self.parameter_shapes) |shapes| {
            for (shapes) |shape| {
                self.base.allocator.free(shape);
            }
            self.base.allocator.free(shapes);
        }
        if (self.weight_buffer) |buffer| {
            self.base.allocator.free(buffer);
        }

        // Clean up training backend
        if (self.training_backend) |backend| {
            backend.deinit();
        }
        if (self.training_vmfb) |vmfb| {
            self.base.allocator.free(vmfb);
        }
        self.optimizer.deinit();

        // Clean up generation backend
        if (self.generation_vmfb) |vmfb| {
            self.base.allocator.free(vmfb);
        }
        if (self.gen_param_shapes) |shapes| {
            for (shapes) |shape| {
                self.base.allocator.free(shape);
            }
            self.base.allocator.free(shapes);
        }
        if (self.gen_data_shapes) |shapes| {
            for (shapes) |shape| {
                self.base.allocator.free(shape);
            }
            self.base.allocator.free(shapes);
        }

        self.base.deinit();
    }

    /// Passthrough to base listen
    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        try self.base.listen(host, port);
    }

    /// Load model weights from a flat binary file
    /// parameter_shapes: Array of shapes for each parameter tensor
    pub fn loadWeightsFromFile(self: *Self, path: []const u8, parameter_shapes: []const []const i64) !void {
        std.log.info("Loading weights from {s}...", .{path});

        // 1. Read the full binary file
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const file_size = stat.size;

        // Allocate buffer for all weights
        const weight_buffer = try self.base.allocator.alloc(u8, file_size);
        errdefer self.base.allocator.free(weight_buffer);
        // In a real prod environment, use mmap instead of alloc+read
        _ = try file.readAll(weight_buffer);

        // 2. Slice the buffer into tensors based on shapes
        // We assume the shapes match exactly what was in the MLIR export

        var offset: usize = 0;
        var loaded_tensors = try self.base.allocator.alloc([]const u8, parameter_shapes.len);
        errdefer self.base.allocator.free(loaded_tensors);

        // 3. Copy and store shapes
        var shapes_copy = try self.base.allocator.alloc([]const i64, parameter_shapes.len);
        errdefer self.base.allocator.free(shapes_copy);

        for (parameter_shapes, 0..) |shape, i| {
            // Calculate size: product(dims) * 4 bytes (f32)
            var num_elements: usize = 1;
            for (shape) |dim| num_elements *= @intCast(dim);
            const byte_size = num_elements * 4;

            if (offset + byte_size > file_size) {
                return error.WeightFileTooSmall;
            }

            const raw_slice = weight_buffer[offset .. offset + byte_size];
            loaded_tensors[i] = raw_slice;

            // Copy shape
            const shape_copy = try self.base.allocator.alloc(i64, shape.len);
            @memcpy(shape_copy, shape);
            shapes_copy[i] = shape_copy;

            offset += byte_size;
        }

        // Store the buffer, slices, and shapes
        self.weight_buffer = weight_buffer;
        self.master_parameters = loaded_tensors;
        self.parameter_shapes = shapes_copy;

        std.log.info("✓ Successfully loaded {} parameter tensors ({} MB total).", .{ self.master_parameters.?.len, file_size / 1024 / 1024 });
    }

    /// Initialize the training backend (compile MLIR and set up IREE)
    pub fn initTrainingBackend(self: *Self, model_path: []const u8) !void {
        std.log.info("Initializing Training Backend from {s}...", .{model_path});

        // 1. Initialize Backend (Use CPU or CUDA based on availability)
        const backend_type = backend_selection.Backend.selectDefault();
        // Use device 0 for Shepherd training
        const backend = try IreeBackend.init(self.base.allocator, backend_type, 0);
        self.training_backend = backend;

        // 2. Load or Compile Training Graph
        if (std.mem.endsWith(u8, model_path, ".vmfb")) {
            // Load pre-compiled VMFB directly (instant!)
            std.log.info("🚀 Loading pre-compiled VMFB artifact directly...", .{});

            const vmfb_bytes = try std.fs.cwd().readFileAlloc(
                self.base.allocator,
                model_path,
                2 * 1024 * 1024 * 1024, // Allow up to 2GB
            );
            self.training_vmfb = vmfb_bytes;
            std.log.info("✓ Loaded VMFB ({} bytes)", .{vmfb_bytes.len});
        } else {
            // Compile from MLIR source (SLOW - 10-20 minutes)
            std.log.warn("⚠️  Source is .mlir. Compiling at runtime... (This will be SLOW)", .{});

            const mlir_source = try std.fs.cwd().readFileAlloc(
                self.base.allocator,
                model_path,
                100 * 1024 * 1024,
            );
            defer self.base.allocator.free(mlir_source);

            var ctx = try mlir_ctx.MLIRContext.init(self.base.allocator);
            defer ctx.deinit();

            const vmfb = try ctx.compileToVMFB(
                self.base.allocator,
                mlir_source,
                backend_type.toIreeCompilationTarget(),
                null, // target_arch autodetect
            );
            self.training_vmfb = vmfb;
        }

        std.log.info("✓ Training Backend Ready.", .{});
    }

    /// Send a rollout request to a worker
    pub fn requestRollout(self: *Self, prompt: []const i64) !void {
        // Round-robin or random assignment
        // For MVP, broadcast to first available worker

        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();

        var prompt_arr = std.json.Array.init(self.base.allocator);
        defer prompt_arr.deinit();
        for (prompt) |t| try prompt_arr.append(.{ .integer = t });

        try payload.put("prompt", .{ .array = prompt_arr });

        // Use base functionality to broadcast (or sendToWorker)
        try self.base.broadcastToWorkers(MessageType.START_ROLLOUT, .{ .object = payload });
    }

    /// Collect rollout results
    pub fn collectRollouts(self: *Self, expected: usize) !ArrayList(RolloutData) {
        // Use base collect functionality
        const msgs = try self.base.collectFromWorkers(MessageType.ROLLOUT_COMPLETE, expected);
        defer {
            for (msgs.items) |msg| {
                msg.deinitClone(self.base.allocator);
            }
            msgs.deinit();
        }

        var results = ArrayList(RolloutData).init(self.base.allocator);
        errdefer {
            for (results.items) |*r| r.deinit();
            results.deinit();
        }

        for (msgs.items) |msg| {
            // Parse completion from msg.data
            const obj = switch (msg.data) {
                .object => |o| o,
                else => continue,
            };

            const completion_value = obj.get("completion") orelse continue;
            const completion_array = switch (completion_value) {
                .array => |arr| arr,
                else => continue,
            };

            var comp_list = ArrayList(i64).init(self.base.allocator);
            errdefer comp_list.deinit();

            for (completion_array.items) |item| {
                switch (item) {
                    .integer => |i| try comp_list.append(i),
                    else => {},
                }
            }

            const completion_slice = try comp_list.toOwnedSlice();

            // Create placeholder prompt (in real implementation, track this)
            const placeholder_prompt = try self.base.allocator.alloc(i64, 0);

            try results.append(.{
                .prompt = placeholder_prompt,
                .completion = completion_slice,
                .allocator = self.base.allocator,
            });
        }

        return results;
    }

    /// Execute a GRPO training step
    /// rollouts: List of {prompt, completion}
    /// rewards: List of float scores corresponding to rollouts
    pub fn trainStep(self: *Self, rollouts: ArrayList(RolloutData), rewards: ArrayList(f32)) !void {
        std.log.info("Running GRPO Training Step on {} samples...", .{rollouts.items.len});

        // 1. Compute Advantages (Group Normalized)
        // Assume rollouts are ordered by group (e.g., 4 completions per prompt)
        const group_size = 4;
        var advantages = try self.base.allocator.alloc(f32, rewards.items.len);
        defer self.base.allocator.free(advantages);

        var i: usize = 0;
        while (i < rewards.items.len) : (i += group_size) {
            // Get slice for this group
            const group_rewards = rewards.items[i..@min(i + group_size, rewards.items.len)];

            // Calculate Mean/Std
            var sum: f32 = 0;
            for (group_rewards) |r| sum += r;
            const mean = sum / @as(f32, @floatFromInt(group_rewards.len));

            var sum_sq_diff: f32 = 0;
            for (group_rewards) |r| sum_sq_diff += (r - mean) * (r - mean);
            // Add epsilon to std to prevent div by zero
            const std_dev = std.math.sqrt(sum_sq_diff / @as(f32, @floatFromInt(group_rewards.len))) + 1e-8;

            // Normalize
            for (group_rewards, 0..) |r, k| {
                advantages[i + k] = (r - mean) / std_dev;
            }
        }

        // 2. Prepare Tensors for IREE
        // We need to batch 'group_size' items together for the static graph

        // Loop over data in chunks of GROUP_SIZE
        var batch_idx: usize = 0;
        while (batch_idx < rollouts.items.len) : (batch_idx += group_size) {

            // A. Prepare Input IDs (Pad/Truncate to 512)
            const seq_len = 512;
            const input_ids_bytes = try self.base.allocator.alloc(u8, group_size * seq_len * 8); // i64
            const mask_bytes = try self.base.allocator.alloc(u8, group_size * seq_len * 8); // i64
            defer self.base.allocator.free(input_ids_bytes);
            defer self.base.allocator.free(mask_bytes);

            // Zero init
            @memset(input_ids_bytes, 0);
            @memset(mask_bytes, 0);

            const ids_view = std.mem.bytesAsSlice(i64, input_ids_bytes);
            const mask_view = std.mem.bytesAsSlice(i64, mask_bytes);

            for (0..group_size) |k| {
                if (batch_idx + k >= rollouts.items.len) break;

                const r = rollouts.items[batch_idx + k];
                // Concatenate Prompt + Completion
                // Note: Real implementation needs careful handling of arraylists
                // Here assuming r.completion includes prompt or we concat

                // Copy to buffer
                var pos: usize = 0;
                for (r.prompt) |token| {
                    if (pos >= seq_len) break;
                    ids_view[k * seq_len + pos] = token;
                    mask_view[k * seq_len + pos] = 1;
                    pos += 1;
                }
                for (r.completion) |token| {
                    if (pos >= seq_len) break;
                    ids_view[k * seq_len + pos] = token;
                    mask_view[k * seq_len + pos] = 1;
                    pos += 1;
                }
            }

            // B. Prepare Advantages
            const adv_slice = advantages[batch_idx..@min(batch_idx + group_size, advantages.len)];
            const adv_bytes = std.mem.sliceAsBytes(adv_slice);

            // C. Execute Training Graph
            var inputs_list = ArrayList([]const u8).init(self.base.allocator);
            defer inputs_list.deinit();
            var shapes_list = ArrayList([]const i64).init(self.base.allocator);
            defer shapes_list.deinit();

            // 1. Add Parameters
            if (self.master_parameters) |params| {
                for (params, 0..) |param_bytes, param_idx| {
                    try inputs_list.append(param_bytes);
                    if (self.parameter_shapes) |shapes| {
                        try shapes_list.append(shapes[param_idx]);
                    } else {
                        // Fallback: empty shape if static shapes are compiled in
                        try shapes_list.append(&[_]i64{});
                    }
                }
            }

            // 2. Add Data Inputs
            try inputs_list.append(input_ids_bytes);
            try shapes_list.append(&[_]i64{ group_size, seq_len }); // [4, 512]

            try inputs_list.append(mask_bytes);
            try shapes_list.append(&[_]i64{ group_size, seq_len }); // [4, 512]

            try inputs_list.append(adv_bytes);
            try shapes_list.append(&[_]i64{group_size}); // [4]

            // 3. Execute!
            std.log.info("Executing Backward Pass...", .{});
            const gradients = try self.training_backend.?.execute(
                self.training_vmfb.?,
                "main",
                inputs_list.items,
                shapes_list.items,
                null,
            );
            defer {
                for (gradients) |g| self.base.allocator.free(g);
                self.base.allocator.free(gradients);
            }

            // 4. Update Weights (Zig Optimizer)
            std.log.info("Applying Gradients...", .{});

            // Gradients should match parameters 1-to-1
            if (self.master_parameters) |params| {
                if (gradients.len != params.len) {
                    return error.GradientCountMismatch;
                }

                for (params, 0..) |param_bytes, param_idx| {
                    const grad_bytes = gradients[param_idx];

                    // Cast to f32 slices
                    // param_bytes is const, we need to cast away const to update in-place
                    // (Since we own the buffer in weight_buffer)
                    const param_ptr = @constCast(param_bytes.ptr);
                    const param_slice = @as([*]f32, @ptrCast(@alignCast(param_ptr)))[0 .. param_bytes.len / 4];

                    const grad_slice = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, grad_bytes)));

                    // Call Zig Optimizer
                    try self.optimizer.update(param_idx, param_slice, grad_slice);
                }
            }
        }

        std.log.info("GRPO Step Complete. Weights Updated.", .{});
    }

    /// Broadcast updated weights to all workers
    pub fn broadcastNewWeights(self: *Self) !void {
        std.log.info("Broadcasting updated weights to workers...", .{});

        // Get the updated weight buffer
        const weight_data = self.weight_buffer orelse return error.NoWeightsLoaded;

        // Encode to Base64 for JSON transport
        const b64_len = std.base64.standard.Encoder.calcSize(weight_data.len);
        const b64_weights = try self.base.allocator.alloc(u8, b64_len);
        defer self.base.allocator.free(b64_weights);

        _ = std.base64.standard.Encoder.encode(b64_weights, weight_data);

        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();

        try payload.put("weights", std.json.Value{ .string = b64_weights });

        try self.base.broadcastToWorkers(MessageType.UPDATE_WEIGHTS, .{ .object = payload });

        std.log.info("✓ Weight broadcast complete ({} bytes)", .{weight_data.len});
    }

    /// Initialize the generation backend for workers
    pub fn initGenerationBackend(self: *Self, vmfb_path: []const u8, mlir_path: []const u8) !void {
        std.log.info("Loading Generation Backend...", .{});

        // 1. Read VMFB
        self.generation_vmfb = try std.fs.cwd().readFileAlloc(
            self.base.allocator,
            vmfb_path,
            10 * 1024 * 1024 * 1024, // Allow up to 10GB for generation model
        );
        std.log.info("✓ Loaded Generation VMFB ({} bytes)", .{self.generation_vmfb.?.len});

        // 2. Read MLIR for Shape Introspection
        const mlir_source = try std.fs.cwd().readFileAlloc(
            self.base.allocator,
            mlir_path,
            5 * 1024 * 1024 * 1024, // Allow up to 5GB for large generation MLIRs
        );
        defer self.base.allocator.free(mlir_source);

        var ctx = try mlir_ctx.MLIRContext.init(self.base.allocator);
        defer ctx.deinit();

        const model_introspection = @import("../../mlir/model_introspection.zig");

        // In qwen_rl_generation, weights are frozen constants (not inputs)
        // All 51 function arguments are runtime data inputs (temp, input_ids, pos_ids, KV caches)
        // Count them first
        const mlir_wrapper = @import("../../mlir/wrapper.zig");
        var ctx_for_count = try mlir_ctx.MLIRContext.init(self.base.allocator);
        defer ctx_for_count.deinit();
        const temp_module = try mlir_wrapper.Module.parse(ctx_for_count.getContext(), mlir_source);
        defer temp_module.deinit();
        const forward_fn = try temp_module.findFunction("main");
        const mlir_types = @import("../../mlir/wrapper.zig");
        const func_type = forward_fn.getType().as(mlir_types.FunctionType) orelse return error.NotAFunctionType;
        const num_total_inputs = func_type.getNumInputs();

        const metadata = try model_introspection.ModelInspector.inspect(
            self.base.allocator,
            ctx.getContext(),
            mlir_source,
            num_total_inputs, // All inputs are data inputs (no trainable parameters)
        );

        self.gen_data_shapes = metadata.data_input_shapes;
        self.gen_param_shapes = metadata.parameter_shapes; // Should be empty

        std.log.info("✓ Introspected Generation Graph: {} data inputs", .{self.gen_data_shapes.?.len});
    }

    /// Distribute the generation graph to workers
    pub fn prepareWorkersForGeneration(self: *Self, vmfb_path: []const u8) !void {
        if (self.gen_data_shapes == null) {
            return error.GenerationBackendNotInitialized;
        }

        std.log.info("Preparing workers for RL generation...", .{});

        // Use parameter shapes (should be empty for generation graph with frozen weights)
        const param_shapes = self.gen_param_shapes orelse {
            // Allocate empty mutable slice if needed
            const empty = try self.base.allocator.alloc([]i64, 0);
            defer self.base.allocator.free(empty);
            return try self.base.initializeWorkersWithVMFB(
                vmfb_path,
                empty,
                self.gen_data_shapes.?,
            );
        };

        try self.base.initializeWorkersWithVMFB(
            vmfb_path,
            param_shapes,
            self.gen_data_shapes.?,
        );

        std.log.info("✓ Workers ready for generation", .{});
    }
};
