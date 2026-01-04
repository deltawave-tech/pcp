const std = @import("std");
const Allocator = std.mem.Allocator;
const Shepherd = @import("shepherd.zig").Shepherd;
const MessageType = @import("../../network/message.zig").MessageType;
const MessageEnvelope = @import("../../network/message.zig").MessageEnvelope;
const ArrayList = std.ArrayList;
const backend_selection = @import("../../backends/selection.zig");
const IreeBackend = @import("../../backends/iree.zig").IreeBackend;
const mlir_ctx = @import("../../mlir/context.zig");
const model_introspection = @import("../../mlir/model_introspection.zig");
const Nesterov = @import("../../optimizers/nesterov.zig").Nesterov;
const tensor = @import("../../core/tensor.zig");

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
    weight_buffer: ?[]align(4096) u8, // Raw weight file buffer (page-aligned for mmap)
    master_parameters: ?[][]const u8, // Sliced parameter tensors
    parameter_shapes: ?[][]const i64, // Shapes for each parameter
    weights_mmapped: bool, // Track if weights are mmapped for correct cleanup

    // Training Execution
    training_backend: ?*IreeBackend,
    training_vmfb: ?[]u8,
    training_backend_type: ?backend_selection.Backend, // Store the backend type
    optimizer: Nesterov, // Pure Zig optimizer

    // Generation Model State
    generation_vmfb: ?[]u8,
    gen_param_shapes: ?[][]i64,
    gen_data_shapes: ?[][]i64,
    gen_data_dtypes: ?[]tensor.DType,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .base = Shepherd.init(allocator),
            .weight_buffer = null,
            .master_parameters = null,
            .parameter_shapes = null,
            .weights_mmapped = false,
            .training_backend = null,
            .training_vmfb = null,
            .training_backend_type = null,
            .optimizer = Nesterov.init(allocator, .{ .learning_rate = 1e-5 }),
            .generation_vmfb = null,
            .gen_param_shapes = null,
            .gen_data_shapes = null,
            .gen_data_dtypes = null,
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
            if (self.weights_mmapped) {
                // Cross-version compatibility for mmap unmapping
                const munmap = if (@hasDecl(std, "posix")) std.posix.munmap else std.os.munmap;
                munmap(buffer);
            } else {
                self.base.allocator.free(buffer);
            }
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
        if (self.gen_data_dtypes) |dtypes| {
            self.base.allocator.free(dtypes);
        }

        self.base.deinit();
    }

    /// Passthrough to base listen
    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        try self.base.listen(host, port);
    }

    /// Load model weights from a flat binary file using mmap
    /// parameter_shapes: Array of shapes for each parameter tensor
    pub fn loadWeightsFromFile(self: *Self, path: []const u8, parameter_shapes: []const []const i64) !void {
        std.log.info("Loading weights from {s} using mmap...", .{path});

        // 1. Open the file and get its size
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const file_size = stat.size;

        // 2. Memory-map the file (Cross-version compatibility for Zig 0.11-0.13+)
        const posix = if (@hasDecl(std, "posix")) std.posix else std.os;
        const PROT = posix.PROT;
        // MAP_PRIVATE creates a Copy-On-Write (COW) mapping.
        // We can write to memory (for optimizer updates) without modifying the file.
        const flags = if (@hasDecl(std, "posix")) .{ .TYPE = .PRIVATE } else posix.MAP.PRIVATE;

        // Map the file: Read + Write permission is required because optimizer updates weights in-place
        const weight_buffer = try posix.mmap(
            null,
            file_size,
            PROT.READ | PROT.WRITE,
            flags,
            file.handle,
            0,
        );
        errdefer {
            const munmap = if (@hasDecl(std, "posix")) std.posix.munmap else std.os.munmap;
            munmap(weight_buffer);
        }

        self.weight_buffer = weight_buffer;
        self.weights_mmapped = true;
        std.log.info("✓ mmap successful, now slicing into {} parameter tensors...", .{parameter_shapes.len});

        // 2. Slice the buffer into tensors based on shapes
        // We assume the shapes match exactly what was in the MLIR export

        var offset: usize = 0;
        var loaded_tensors = try self.base.allocator.alloc([]const u8, parameter_shapes.len);
        errdefer self.base.allocator.free(loaded_tensors);
        std.log.info("✓ Allocated loaded_tensors array", .{});

        // 3. Copy and store shapes
        var shapes_copy = try self.base.allocator.alloc([]const i64, parameter_shapes.len);
        errdefer self.base.allocator.free(shapes_copy);
        std.log.info("✓ Allocated shapes_copy array, processing {} shapes...", .{parameter_shapes.len});

        for (parameter_shapes, 0..) |shape, i| {
            if (i % 10 == 0 or i >= 240) {
                std.log.info("Processing parameter {}/{}", .{ i, parameter_shapes.len });
            }
            // Calculate size: product(dims) * 4 bytes (f32)
            var num_elements: usize = 1;
            for (shape) |dim| num_elements *= @intCast(dim);
            const byte_size = num_elements * 4;

            if (offset + byte_size > file_size) {
                std.log.err("❌ Weight file too small at parameter {}: offset={} byte_size={} file_size={}", .{ i, offset, byte_size, file_size });
                return error.WeightFileTooSmall;
            }

            const raw_slice = weight_buffer[offset .. offset + byte_size];
            loaded_tensors[i] = raw_slice;

            // Copy shape
            if (i >= 240) {
                std.log.info("  Allocating shape copy for parameter {}, shape.len={}", .{ i, shape.len });
            }
            const shape_copy = try self.base.allocator.alloc(i64, shape.len);
            if (i >= 240) {
                std.log.info("  Copying shape data for parameter {}", .{i});
            }
            @memcpy(shape_copy, shape);
            if (i >= 240) {
                std.log.info("  Storing shape for parameter {}", .{i});
            }
            shapes_copy[i] = shape_copy;

            offset += byte_size;
            if (i >= 240) {
                std.log.info("  Completed parameter {}, new offset={}", .{ i, offset });
            }
        }

        std.log.info("Loop completed, storing slices and shapes...", .{});
        // Store the slices and shapes
        self.master_parameters = loaded_tensors;
        self.parameter_shapes = shapes_copy;

        std.log.info("✓ Successfully mapped {} parameter tensors ({} MB total).", .{ self.master_parameters.?.len, file_size / 1024 / 1024 });
    }

    pub fn initTrainingBackend(self: *Self, model_path: []const u8, backend: backend_selection.Backend) !void {
        std.log.info("Initializing Training Backend from {s} using {s}...", .{ model_path, backend.toString() });

        // 1. Initialize Backend
        self.training_backend = try IreeBackend.init(self.base.allocator, backend, 0);
        self.training_backend_type = backend;

        // 2. Metadata Cache Handling
        const meta_path = try std.fmt.allocPrint(self.base.allocator, "{s}.meta.json", .{model_path});
        defer self.base.allocator.free(meta_path);

        var metadata_loaded = false;
        if (std.fs.cwd().access(meta_path, .{})) |_| {
            if (model_introspection.ModelMetadata.loadFromFile(self.base.allocator, meta_path)) |meta| {
                self.parameter_shapes = meta.parameter_shapes;
                for (meta.data_input_shapes) |s| self.base.allocator.free(s);
                self.base.allocator.free(meta.data_input_shapes);
                self.base.allocator.free(meta.data_input_dtypes);
                metadata_loaded = true;
                std.log.info("✓ Loaded metadata from cache.", .{});
            } else |_| {}
        } else |_| {}

        // 3. Fallback Introspection
        if (!metadata_loaded) {
            std.log.info("Cache miss. Introspecting...", .{});
            const mlir_source = try std.fs.cwd().readFileAlloc(self.base.allocator, model_path, 2 * 1024 * 1024 * 1024);
            defer self.base.allocator.free(mlir_source);

            // 5 non-parameter inputs: causal_mask buffer, inv_freq buffer, input_ids, mask, advantages
            const num_data_inputs = 5;
            var meta_result = model_introspection.ModelInspector.inspectLite(
                self.base.allocator,
                mlir_source,
                num_data_inputs,
            ) catch |err| blk: {
                std.log.warn("Lite introspection failed ({}), falling back to Full MLIR parsing...", .{err});
                var ctx = try mlir_ctx.MLIRContext.init(self.base.allocator);
                defer ctx.deinit();
                break :blk try model_introspection.ModelInspector.inspect(
                    self.base.allocator,
                    ctx.getContext(),
                    mlir_source,
                    num_data_inputs,
                );
            };

            self.parameter_shapes = meta_result.parameter_shapes;
            meta_result.saveToFile(meta_path) catch {};

            for (meta_result.data_input_shapes) |s| self.base.allocator.free(s);
            self.base.allocator.free(meta_result.data_input_shapes);
            self.base.allocator.free(meta_result.data_input_dtypes);
        }

        // 4. Backend-Specific VMFB Caching
        const base_name = if (std.mem.endsWith(u8, model_path, ".mlir"))
            model_path[0 .. model_path.len - 5]
        else
            model_path;

        const vmfb_cache_path = try std.fmt.allocPrint(
            self.base.allocator,
            "{s}.{s}.vmfb",
            .{ base_name, backend.toString() },
        );
        defer self.base.allocator.free(vmfb_cache_path);

        var vmfb_loaded = false;

        if (std.fs.cwd().access(vmfb_cache_path, .{})) |_| {
            std.log.info("Found cached VMFB: {s}", .{vmfb_cache_path});
            const vmfb_bytes = try std.fs.cwd().readFileAlloc(self.base.allocator, vmfb_cache_path, 2 * 1024 * 1024 * 1024);
            self.training_vmfb = vmfb_bytes;
            vmfb_loaded = true;
            std.log.info("✓ Loaded VMFB from cache ({} bytes)", .{vmfb_bytes.len});
        } else |_| {}

        if (!vmfb_loaded) {
            std.log.warn("Compiling MLIR for {s}... (This runs once)", .{backend.toIreeCompilationTarget()});

            const mlir_source = try std.fs.cwd().readFileAlloc(self.base.allocator, model_path, 2 * 1024 * 1024 * 1024);
            defer self.base.allocator.free(mlir_source);

            var ctx = try mlir_ctx.MLIRContext.init(self.base.allocator);
            defer ctx.deinit();

            const vmfb = try ctx.compileToVMFB(
                self.base.allocator,
                mlir_source,
                backend.toIreeCompilationTarget(),
                null,
            );
            self.training_vmfb = vmfb;

            std.log.info("Saving compiled VMFB to {s}...", .{vmfb_cache_path});
            std.fs.cwd().writeFile(.{ .sub_path = vmfb_cache_path, .data = vmfb }) catch |err| {
                std.log.warn("Failed to save VMFB cache: {}", .{err});
            };
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

    pub fn trainStep(self: *Self, rollouts: ArrayList(RolloutData), rewards: ArrayList(f32), group_size: usize) !void {
        std.debug.print(">>> trainStep START. Rollouts: {}\n", .{rollouts.items.len});
        var advantages = try self.base.allocator.alloc(f32, rewards.items.len);
        defer self.base.allocator.free(advantages);

        var i: usize = 0;
        while (i < rewards.items.len) : (i += group_size) {
            const group_rewards = rewards.items[i..@min(i + group_size, rewards.items.len)];
            var sum: f32 = 0;
            for (group_rewards) |r| sum += r;
            const mean = sum / @as(f32, @floatFromInt(group_rewards.len));

            var sum_sq_diff: f32 = 0;
            for (group_rewards) |r| sum_sq_diff += (r - mean) * (r - mean);
            const std_dev = std.math.sqrt(sum_sq_diff / @as(f32, @floatFromInt(group_rewards.len))) + 1e-8;

            for (group_rewards, 0..) |r, k| {
                advantages[i + k] = (r - mean) / std_dev;
            }
        }

        // 2. Prepare Tensors for IREE
        var batch_idx: usize = 0;
        while (batch_idx < rollouts.items.len) : (batch_idx += group_size) {
            std.debug.print(">>> Preparing batch {}\n", .{batch_idx});

            // A. Prepare Input IDs (i64) and Mask (f32)
            const seq_len = 512;
            const input_ids_bytes = try self.base.allocator.alloc(u8, group_size * seq_len * 8);
            const mask_bytes = try self.base.allocator.alloc(u8, group_size * seq_len * 4);
            defer self.base.allocator.free(input_ids_bytes);
            defer self.base.allocator.free(mask_bytes);

            @memset(input_ids_bytes, 0);
            @memset(mask_bytes, 0);

            const ids_view = std.mem.bytesAsSlice(i64, input_ids_bytes);
            const mask_view = std.mem.bytesAsSlice(f32, mask_bytes);

            for (0..group_size) |k| {
                if (batch_idx + k >= rollouts.items.len) break;
                const r = rollouts.items[batch_idx + k];
                var pos: usize = 0;
                for (r.prompt) |token| {
                    if (pos >= seq_len) break;
                    ids_view[k * seq_len + pos] = token;
                    mask_view[k * seq_len + pos] = 1.0;
                    pos += 1;
                }
                for (r.completion) |token| {
                    if (pos >= seq_len) break;
                    ids_view[k * seq_len + pos] = token;
                    mask_view[k * seq_len + pos] = 1.0;
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
            var dtypes_list = ArrayList(tensor.DType).init(self.base.allocator);
            defer dtypes_list.deinit();

            // Add Parameters
            if (self.master_parameters) |params| {
                for (params, 0..) |param_bytes, param_idx| {
                    try inputs_list.append(param_bytes);
                    try dtypes_list.append(.f32);
                    if (self.parameter_shapes) |shapes| {
                        try shapes_list.append(shapes[param_idx]);
                    } else {
                        try shapes_list.append(&[_]i64{});
                    }
                }
            }

            const gs: i64 = @intCast(group_size);
            try inputs_list.append(input_ids_bytes);
            try shapes_list.append(&[_]i64{ gs, seq_len });
            try dtypes_list.append(.i64);

            try inputs_list.append(mask_bytes);
            try shapes_list.append(&[_]i64{ gs, seq_len });
            try dtypes_list.append(.f32);

            try inputs_list.append(adv_bytes);
            try shapes_list.append(&[_]i64{gs});
            try dtypes_list.append(.f32);

            // Execute backward pass
            std.debug.print(">>> Executing IREE Backward Pass...\n", .{});
            const gradients = try self.training_backend.?.execute(
                self.training_vmfb.?,
                "main",
                inputs_list.items,
                shapes_list.items,
                dtypes_list.items,
            );
            defer self.base.allocator.free(gradients);

            // Update Weights
            std.log.info("IREE backward pass returned {} gradients", .{gradients.len});

            if (self.master_parameters) |params| {
                if (gradients.len != params.len) {
                    std.log.err("CRITICAL: Gradient count mismatch! Params: {}, Grads: {}", .{ params.len, gradients.len });
                    return error.GradientCountMismatch;
                }

                // Compute gradient statistics
                var total_grad_norm: f64 = 0.0;
                var total_grad_elements: usize = 0;
                var max_grad: f32 = 0.0;
                var nan_count: usize = 0;
                var inf_count: usize = 0;

                for (gradients) |grad_bytes_raw| {
                    var grad_slice_stats: []const f32 = undefined;
                    if (@intFromPtr(grad_bytes_raw.ptr) % 4 != 0) {
                        const aligned = try self.base.allocator.alloc(u8, grad_bytes_raw.len);
                        defer self.base.allocator.free(aligned);
                        @memcpy(aligned, grad_bytes_raw);
                        grad_slice_stats = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, aligned)));
                    } else {
                        grad_slice_stats = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, grad_bytes_raw)));
                    }
                    for (grad_slice_stats) |g| {
                        if (std.math.isNan(g)) {
                            nan_count += 1;
                        } else if (std.math.isInf(g)) {
                            inf_count += 1;
                        } else {
                            total_grad_norm += @as(f64, g) * @as(f64, g);
                            if (@abs(g) > max_grad) max_grad = @abs(g);
                        }
                        total_grad_elements += 1;
                    }
                }
                total_grad_norm = std.math.sqrt(total_grad_norm);

                std.log.info("=== Gradient Statistics ===", .{});
                std.log.info("  L2 Norm: {d:.6}", .{total_grad_norm});
                std.log.info("  Max |grad|: {d:.6}", .{max_grad});
                std.log.info("  Elements: {}", .{total_grad_elements});
                if (nan_count > 0 or inf_count > 0) {
                    std.log.warn("  NaN count: {}, Inf count: {}", .{ nan_count, inf_count });
                }

                // Apply gradients
                var params_updated: usize = 0;
                for (params, 0..) |_, param_idx| {
                    const param_bytes = params[param_idx];
                    const grad_bytes_raw = gradients[param_idx];

                    // Check alignment of param_bytes
                    if (@intFromPtr(param_bytes.ptr) % 4 != 0) {
                        std.log.err("CRITICAL: Param {} unaligned!", .{param_idx});
                        return error.MisalignedParameter;
                    }
                    const param_ptr = @constCast(param_bytes.ptr);
                    const param_slice = @as([*]f32, @ptrCast(@alignCast(param_ptr)))[0 .. param_bytes.len / 4];

                    // Handle potentially unaligned gradients
                    var grad_slice: []const f32 = undefined;
                    var aligned_grad_buffer: ?[]u8 = null;

                    if (@intFromPtr(grad_bytes_raw.ptr) % 4 != 0) {
                        // Unaligned: create aligned copy
                        aligned_grad_buffer = try self.base.allocator.alloc(u8, grad_bytes_raw.len);
                        @memcpy(aligned_grad_buffer.?, grad_bytes_raw);
                        grad_slice = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, aligned_grad_buffer.?)));
                    } else {
                        grad_slice = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, grad_bytes_raw)));
                    }

                    // Update weights
                    try self.optimizer.update(param_idx, param_slice, grad_slice);
                    params_updated += 1;

                    // Clean up
                    if (aligned_grad_buffer) |buf| {
                        self.base.allocator.free(buf);
                    }
                    self.base.allocator.free(grad_bytes_raw);
                }
                std.log.info("  Parameters updated: {}/{}", .{ params_updated, params.len });
            }
        }

        std.log.info("=== Training Step Complete ===", .{});
    }

    /// Broadcast updated weights to all workers
    /// Weights are sent directly - no prepending needed since MLIR no longer has dead inv_freq input
    pub fn broadcastNewWeights(self: *Self) !void {
        std.log.info("Broadcasting updated weights to workers...", .{});

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
        std.log.info("Initializing Generation Backend...", .{});

        // 1. Read VMFB
        // Use 10GB limit as defined before
        self.generation_vmfb = try std.fs.cwd().readFileAlloc(
            self.base.allocator,
            vmfb_path,
            10 * 1024 * 1024 * 1024,
        );
        std.log.info("✓ Loaded Generation VMFB ({} bytes)", .{self.generation_vmfb.?.len});

        // 2. Metadata Cache Handling
        const meta_path = try std.fmt.allocPrint(self.base.allocator, "{s}.meta.json", .{mlir_path});
        defer self.base.allocator.free(meta_path);

        var metadata_loaded = false;

        // Try Loading Cache First
        if (std.fs.cwd().access(meta_path, .{})) |_| {
            std.log.info("Found generation metadata cache: {s}", .{meta_path});
            if (model_introspection.ModelMetadata.loadFromFile(self.base.allocator, meta_path)) |meta| {
                self.gen_data_shapes = meta.data_input_shapes;
                self.gen_data_dtypes = meta.data_input_dtypes;
                self.gen_param_shapes = meta.parameter_shapes; // Should be empty
                metadata_loaded = true;
                std.log.info("✓ Loaded generation metadata from cache: {} data inputs (instant!)", .{meta.data_input_shapes.len});
            } else |err| {
                std.log.warn("Failed to load metadata cache: {}. Will introspect.", .{err});
            }
        } else |_| {}

        // 3. Fallback to Standard Introspection (Source Scan)
        if (!metadata_loaded) {
            std.log.info("Cache miss. Reading MLIR source for introspection...", .{});

            // Read source (3.7GB)
            const mlir_source = try std.fs.cwd().readFileAlloc(
                self.base.allocator,
                mlir_path,
                5 * 1024 * 1024 * 1024,
            );
            defer self.base.allocator.free(mlir_source);

            std.log.info("Source loaded ({} bytes). Running Standard Inspector...", .{mlir_source.len});

            // Use standard inspector that separates params from data inputs
            // Qwen 0.5B has 24 layers: data inputs = Token(1) + Pos(1) + KV(2*24) = 50
            const num_data_inputs = 50;

            var meta_result = try model_introspection.ModelInspector.inspectLite(
                self.base.allocator,
                mlir_source,
                num_data_inputs,
            );

            std.log.info("Introspection complete. Found {} params, {} data inputs.", .{
                meta_result.parameter_shapes.len,
                meta_result.data_input_shapes.len,
            });
            std.log.info("Saving cache to {s}...", .{meta_path});
            try meta_result.saveToFile(meta_path);

            self.gen_data_shapes = meta_result.data_input_shapes;
            self.gen_data_dtypes = meta_result.data_input_dtypes;
            self.gen_param_shapes = meta_result.parameter_shapes;
        }

        const num_params = if (self.gen_param_shapes) |ps| ps.len else 0;
        std.log.info("✓ Generation Backend Ready: {} params, {} data inputs.", .{ num_params, self.gen_data_shapes.?.len });
    }

    /// Distribute the generation graph to workers
    pub fn prepareWorkersForGeneration(self: *Self, vmfb_path: []const u8) !void {
        if (self.gen_data_shapes == null or self.gen_data_dtypes == null) {
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
                self.gen_data_dtypes.?,
            );
        };

        try self.base.initializeWorkersWithVMFB(
            vmfb_path,
            param_shapes,
            self.gen_data_shapes.?,
            self.gen_data_dtypes.?,
        );

        std.log.info("✓ Workers ready for generation", .{});
    }
};
