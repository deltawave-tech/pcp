const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const TrainingAlgorithm = @import("training_algorithm.zig").TrainingAlgorithm;
const TrainingStatus = @import("training_algorithm.zig").TrainingStatus;
const backend_selection = @import("../backends/selection.zig");
const control_state_mod = @import("../nodes/gateway/control_plane/state.zig");
const gateway_service_client = @import("../nodes/gateway/service_client.zig");
const file_util = @import("../protocol/file_util.zig");
const protocol_limits = @import("../protocol/limits.zig");

// RL orchestration is now exposed under the gateway controller namespace.
pub const RLController = @import("../nodes/gateway/controllers/rl_controller.zig").RLController;
pub const RolloutData = @import("../nodes/gateway/controllers/rl_controller.zig").RolloutData;

pub const GRPOConfig = struct {
    num_iterations: usize,
    group_size: usize,
    learning_rate: f32,
    beta: f32,
    required_workers: usize,
    prompt_file: []const u8,
    num_prompts: usize,
    rollout_max_tokens: usize = 128,
    // Model paths
    weights_path: []const u8,
    training_weights_path: ?[]const u8 = null,
    generation_weights_path: ?[]const u8 = null,
    generation_weights_format: []const u8 = "flat",
    adapter_state_path: ?[]const u8 = null,
    grpo_weight_mode: []const u8 = "full",
    generation_vmfb_path: []const u8,
    generation_mlir_path: []const u8,
    training_mlir_path: []const u8,
    // Generation model config
    num_gen_data_inputs: ?usize = null, // Optional fallback when generation metadata is absent.
    weight_refresh_strategy: []const u8 = "inline",
    updated_weights_path: ?[]const u8 = null,
    trainable_parameter_indices: ?[]const usize = null,
};

pub const GRPO = struct {
    allocator: Allocator,
    controller: *RLController,
    config: GRPOConfig,
    status: TrainingStatus,
    prompts: ArrayList([]const i64),
    control_state: ?*control_state_mod.ControllerState,
    gateway_client: ?*gateway_service_client.GatewayClient,

    const Self = @This();

    pub fn init(allocator: Allocator, controller: *RLController, config: GRPOConfig) Self {
        return Self{
            .allocator = allocator,
            .controller = controller,
            .config = config,
            .status = .not_started,
            .prompts = ArrayList([]const i64).init(allocator),
            .control_state = null,
            .gateway_client = null,
        };
    }

    pub fn setControlState(self: *Self, state: *control_state_mod.ControllerState) void {
        self.control_state = state;
    }

    pub fn setGatewayClient(self: *Self, client: ?*gateway_service_client.GatewayClient) void {
        self.gateway_client = client;
    }

    /// Load prompts from binary file (created by tools/prepare_rl_dataset.py)
    pub fn loadPrompts(self: *Self, path: []const u8) !void {
        std.log.info("Loading prompts from {s}...", .{path});

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var reader = file.reader();

        // Read count (u32 little endian)
        const count = try reader.readInt(u32, .little);
        std.log.info("Found {} prompts in dataset", .{count});

        for (0..count) |i| {
            const len = try reader.readInt(u32, .little);
            const tokens = try self.allocator.alloc(i64, len);
            errdefer self.allocator.free(tokens);

            for (0..len) |j| {
                // Read as u64, cast to i64
                const tok_u64 = try reader.readInt(u64, .little);
                tokens[j] = @intCast(tok_u64);
            }

            try self.prompts.append(tokens);
            std.log.info("  Prompt {}: {} tokens", .{ i + 1, len });
        }

        std.log.info("✓ Loaded {} prompts", .{self.prompts.items.len});
    }

    pub fn asTrainingAlgorithm(self: *Self) TrainingAlgorithm {
        return TrainingAlgorithm{
            .ptr = self,
            .vtable = &.{
                .run = run,
                .deinit = deinit,
                .getName = getName,
                .getStatus = getStatus,
            },
        };
    }

    fn run(ptr: *anyopaque) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.status = .running;
        if (self.control_state) |state| {
            state.setStatus(.initializing);
            state.setRLProgress(0, self.config.num_iterations, 0, 0, 0.0);
            try state.setReadinessPhase("loading_prompts", self.config.prompt_file);
        }

        std.log.info("Starting GRPO Training Loop...", .{});

        // 1. Load Prompts
        if (self.prompts.items.len == 0) {
            try self.loadPrompts(self.config.prompt_file);
        }

        if (self.prompts.items.len == 0) {
            std.log.err("No prompts loaded! Cannot start training.", .{});
            return error.NoPromptsLoaded;
        }

        self.waitForWorkersToConnect() catch |err| {
            if (err == error.Cancelled) return;
            return err;
        };

        if (self.control_state) |state| {
            state.setStatus(.initializing);
        }

        if (self.controller.training_backend == null) {
            const backend_type = self.controller.training_backend_type orelse {
                std.log.err("training_backend_type not set! Use --backend flag.", .{});
                return error.BackendNotSet;
            };

            if (self.config.trainable_parameter_indices) |indices| {
                try self.controller.setTrainableParameterIndices(indices);
                std.log.info("✓ Restricted GRPO optimizer/update path to {} trainable parameters", .{indices.len});
            }

            if (self.control_state) |state| {
                try state.setReadinessPhase("initializing_training_backend", backend_type.toString());
            }

            // Qwen model: no buffers (causal_mask and RoPE are constants in the MLIR)
            // Data inputs: input_ids, mask, advantages (3 data inputs)
            try self.controller.initTrainingBackend(
                self.config.training_mlir_path,
                backend_type,
                0, // num_buffers (none - all constants are baked in)
                3, // num_data_inputs
            );

            if (self.control_state) |state| {
                if (state.isCancellationRequested()) {
                    self.status = .completed;
                    state.setCancelled();
                    return;
                }
            }

            // Load initial weights after initializing backend
            if (self.controller.parameter_shapes) |shapes| {
                const training_weights_path = try self.initialTrainingWeightsPath();
                defer self.allocator.free(training_weights_path);
                std.log.info("Loading initial trainable weights from {s}...", .{training_weights_path});
                if (self.control_state) |state| {
                    try state.setReadinessPhase("loading_training_weights", training_weights_path);
                }
                try self.controller.loadWeightsFromFile(training_weights_path, shapes);
                std.log.info("✓ Loaded initial trainable weights successfully", .{});

                // Initialize optimizer velocity buffers
                if (self.control_state) |state| {
                    try state.setReadinessPhase("initializing_optimizer", null);
                }
                self.controller.optimizer.config.learning_rate = self.config.learning_rate;
                if (self.controller.trainable_parameter_indices) |indices| {
                    try self.controller.optimizer.initParametersForIndices(shapes, indices);
                    std.log.info("✓ Optimizer initialized for {} trainable parameters", .{indices.len});
                } else {
                    try self.controller.optimizer.initParameters(shapes);
                    std.log.info("✓ Optimizer initialized for {} parameters", .{shapes.len});
                }
                try self.resumeAdapterOptimizerState();
            } else {
                std.log.err("Failed to load weights: parameter shapes not initialized", .{});
                return error.ParameterShapesNotInitialized;
            }
        }

        // 4. Initialize Generation Backend (Distributed to Workers)
        if (self.controller.generation_vmfb == null) {
            if (self.control_state) |state| {
                try state.setReadinessPhase("initializing_generation_backend", self.config.generation_vmfb_path);
            }
            try self.controller.initGenerationBackend(
                self.config.generation_vmfb_path,
                self.config.generation_mlir_path,
                self.config.num_gen_data_inputs,
            );
        }
        const generation_vmfb_path = self.controller.getEffectiveGenerationVmfbPath(self.config.generation_vmfb_path);
        const generation_weights_path = self.generationWeightsPath();

        // 5. Ensure Workers are Ready - send VMFB path and weights path for local loading
        std.log.info("Initializing Workers with Generation Model...", .{});
        if (self.control_state) |state| {
            try state.setReadinessPhase("preparing_generation_workers", generation_vmfb_path);
        }
        try self.controller.prepareWorkersForGeneration(
            generation_vmfb_path,
            generation_weights_path,
            self.config.generation_weights_format,
        );

        self.waitForGenerationWorkers(generation_vmfb_path, generation_weights_path) catch |err| {
            if (err == error.Cancelled) return;
            return err;
        };
        std.log.info("Workers initialized. Starting training loop...", .{});
        if (self.control_state) |state| {
            state.setStatus(.running);
            try state.setReadinessPhase("running", "rollouts_active");
        }

        // 5. Training Loop
        for (0..self.config.num_iterations) |iter| {
            if (self.control_state) |state| {
                if (state.isCancellationRequested()) {
                    self.status = .completed;
                    state.setCancelled();
                    return;
                }
                state.setRLProgress(iter, self.config.num_iterations, 0, 0, 0.0);
            }
            std.log.info("=== GRPO Iteration {}/{} ===", .{ iter + 1, self.config.num_iterations });

            // A. Use Loaded Prompts
            const max_prompts = @min(self.config.num_prompts, self.prompts.items.len);
            const prompts = self.prompts.items[0..max_prompts];

            // B. Request Rollouts
            // We need 'group_size' completions per prompt
            var total_requests: usize = 0;
            for (prompts) |prompt| {
                for (0..self.config.group_size) |_| {
                    try self.controller.requestRollout(prompt, self.config.rollout_max_tokens);
                    total_requests += 1;
                }
            }
            if (self.control_state) |state| {
                state.setRLProgress(iter + 1, self.config.num_iterations, total_requests, 0, 0.0);
            }

            // C. Collect Experience
            std.log.info("Waiting for {} rollouts...", .{total_requests});
            var rollouts = try self.controller.collectRollouts(total_requests);
            defer rollouts.deinit();
            if (self.control_state) |state| {
                state.setRLProgress(iter + 1, self.config.num_iterations, total_requests, total_requests, 0.0);
            }

            // D. Compute Rewards (Format Check + Quality Heuristics)
            var rewards = std.ArrayList(f32).init(self.allocator);
            defer rewards.deinit();

            // Qwen 2.5 EOS token ID
            const eos_id: i64 = 151643;

            for (rollouts.items, 0..) |r, rollout_idx| {
                var score: f32 = 0.0;

                // 1. Length Scoring
                // Penalize very short responses (likely incomplete)
                // Reward moderate-length responses
                // Slightly penalize very long responses (may be repetitive)
                const len = r.completion.len;
                if (len < 5) {
                    score -= 2.0; // Strong penalty for too short
                } else if (len >= 5 and len <= 50) {
                    score += 0.5 + (@as(f32, @floatFromInt(len)) * 0.02); // Reward reasonable length
                } else if (len > 50 and len <= 128) {
                    score += 1.0; // Good length range
                } else {
                    score += 0.5; // Long but acceptable
                }

                // 2. EOS Token Bonus (proper termination)
                var has_eos = false;
                for (r.completion) |token| {
                    if (token == eos_id) {
                        has_eos = true;
                        break;
                    }
                }

                if (has_eos) {
                    score += 1.0; // Bonus for proper termination
                } else {
                    score -= 0.5; // Penalty for not terminating properly
                }

                // 3. Diversity Bonus (count unique tokens)
                var unique_tokens = std.AutoHashMap(i64, void).init(self.allocator);
                defer unique_tokens.deinit();
                for (r.completion) |token| {
                    try unique_tokens.put(token, {});
                }
                const diversity = @as(f32, @floatFromInt(unique_tokens.count())) / @as(f32, @floatFromInt(@max(len, 1)));
                score += diversity * 0.5; // Bonus for diverse token usage

                try rewards.append(score);

                // Log detailed rollout info
                std.log.info("--- Rollout {} ---", .{rollout_idx + 1});
                std.log.info("  Prompt tokens ({}): {any}", .{ r.prompt.len, r.prompt[0..@min(r.prompt.len, 20)] });
                std.log.info("  Completion tokens ({}): first20={any}", .{ len, r.completion[0..@min(len, 20)] });
                if (len > 20) {
                    std.log.info("  Completion tokens (cont): last20={any}", .{r.completion[@max(len, 20) - 20 .. len]});
                }
                std.log.info("  Reward: {d:.3} (len={}, eos={}, diversity={d:.2})", .{ score, len, has_eos, diversity });
            }

            // Iteration summary stats
            var reward_sum: f32 = 0.0;
            var reward_min: f32 = std.math.floatMax(f32);
            var reward_max: f32 = -std.math.floatMax(f32);
            for (rewards.items) |r| {
                reward_sum += r;
                if (r < reward_min) reward_min = r;
                if (r > reward_max) reward_max = r;
            }
            const reward_mean = reward_sum / @as(f32, @floatFromInt(rewards.items.len));
            var reward_var: f32 = 0.0;
            for (rewards.items) |r| {
                reward_var += (r - reward_mean) * (r - reward_mean);
            }
            const reward_std = std.math.sqrt(reward_var / @as(f32, @floatFromInt(rewards.items.len)));
            if (self.control_state) |state| {
                state.setRLProgress(iter + 1, self.config.num_iterations, total_requests, total_requests, reward_mean);
            }
            std.log.info("=== Iteration {} Summary ===", .{iter + 1});
            std.log.info("  Rewards: mean={d:.3}, std={d:.3}, min={d:.3}, max={d:.3}", .{ reward_mean, reward_std, reward_min, reward_max });
            self.emitRewardEvent(iter + 1, total_requests, reward_mean, reward_min, reward_max, reward_std) catch |err| {
                std.log.warn("Failed to emit RL reward gateway event: {}", .{err});
            };

            try self.controller.trainStep(rollouts, rewards, self.config.group_size);

            // F. Refresh worker generation weights.
            // Qwen3-8B uses a local path refresh to avoid base64-broadcasting
            // the full 31GB f32 checkpoint through the controller fabric.
            try self.refreshWorkerWeights(iter + 1);
        }

        self.status = .completed;
        if (self.control_state) |state| {
            if (state.isCancellationRequested()) {
                state.setCancelled();
            } else {
                state.setStatus(.completed);
            }
        }
        std.log.info("GRPO Training Completed.", .{});
    }

    fn refreshWorkerWeights(self: *Self, iteration: usize) !void {
        if (std.mem.eql(u8, self.config.grpo_weight_mode, "adapter_only")) {
            const path = try self.adapterStatePath(iteration);
            defer self.allocator.free(path);
            try self.controller.writeWeightsSnapshot(path);
            std.log.info("✓ Adapter-only GRPO state written to {s}; frozen generation weights were not broadcast", .{path});
            const optimizer_path = try self.adapterOptimizerStatePath(iteration);
            defer self.allocator.free(optimizer_path);
            try self.writeOptimizerStateSnapshot(optimizer_path);
            std.log.info("✓ Adapter-only GRPO optimizer state written to {s}", .{optimizer_path});
            return;
        }

        if (std.mem.eql(u8, self.config.weight_refresh_strategy, "local_path")) {
            const path = try self.updatedWeightsPath(iteration);
            defer self.allocator.free(path);
            try self.controller.refreshGenerationWeightsFromLocalSnapshot(path);
            return;
        }

        if (std.mem.eql(u8, self.config.weight_refresh_strategy, "inline")) {
            try self.controller.broadcastNewWeights();
            return;
        }

        std.log.err("Unsupported GRPO weight_refresh_strategy: {s}", .{self.config.weight_refresh_strategy});
        return error.UnsupportedWeightRefreshStrategy;
    }

    fn updatedWeightsPath(self: *Self, iteration: usize) ![]u8 {
        if (self.config.updated_weights_path) |path| {
            return self.allocator.dupe(u8, path);
        }
        return std.fmt.allocPrint(self.allocator, "/tmp/pcp_grpo_updated_weights_iter_{d}.bin", .{iteration});
    }

    fn trainingWeightsPath(self: *Self) []const u8 {
        return self.config.training_weights_path orelse self.config.weights_path;
    }

    fn initialTrainingWeightsPath(self: *Self) ![]u8 {
        if (std.mem.eql(u8, self.config.grpo_weight_mode, "adapter_only")) {
            if (self.config.adapter_state_path) |path| {
                if (file_util.fileExistsAtPath(path)) {
                    std.log.info("Resuming adapter-only GRPO from adapter state {s}", .{path});
                    return self.allocator.dupe(u8, path);
                }
            }
        }
        return self.allocator.dupe(u8, self.trainingWeightsPath());
    }

    fn generationWeightsPath(self: *Self) []const u8 {
        return self.config.generation_weights_path orelse self.config.weights_path;
    }

    fn adapterStatePath(self: *Self, iteration: usize) ![]u8 {
        if (self.config.adapter_state_path) |path| {
            return self.allocator.dupe(u8, path);
        }
        return std.fmt.allocPrint(self.allocator, "/tmp/pcp_grpo_adapter_state_iter_{d}.bin", .{iteration});
    }

    fn adapterOptimizerStatePath(self: *Self, iteration: usize) ![]u8 {
        if (self.config.adapter_state_path) |path| {
            return std.fmt.allocPrint(self.allocator, "{s}.optimizer_state.bin", .{path});
        }
        return std.fmt.allocPrint(self.allocator, "/tmp/pcp_grpo_adapter_state_iter_{d}.optimizer_state.bin", .{iteration});
    }

    fn resumeAdapterOptimizerState(self: *Self) !void {
        if (!std.mem.eql(u8, self.config.grpo_weight_mode, "adapter_only")) return;

        const optimizer_path = try self.adapterOptimizerStatePath(0);
        defer self.allocator.free(optimizer_path);
        if (!file_util.fileExistsAtPath(optimizer_path)) return;

        const optimizer_state = try file_util.readFileAllocAtPath(self.allocator, optimizer_path, 1024 * 1024 * 1024);
        defer self.allocator.free(optimizer_state);
        try self.controller.optimizer.deserialize(optimizer_state);
        std.log.info("Resumed adapter-only GRPO optimizer state from {s} ({} bytes)", .{ optimizer_path, optimizer_state.len });
    }

    fn writeOptimizerStateSnapshot(self: *Self, optimizer_path: []const u8) !void {
        const optimizer_state = try self.controller.optimizer.serialize();
        defer self.allocator.free(optimizer_state);

        if (std.fs.path.dirname(optimizer_path)) |dir| {
            if (dir.len > 0) try file_util.ensureDirAtPath(dir);
        }

        const tmp_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}.tmp.{d}",
            .{ optimizer_path, std.time.nanoTimestamp() },
        );
        defer self.allocator.free(tmp_path);

        try file_util.writeFileAtPath(tmp_path, optimizer_state);
        if (std.fs.path.isAbsolute(tmp_path)) {
            try std.fs.renameAbsolute(tmp_path, optimizer_path);
        } else {
            try std.fs.cwd().rename(tmp_path, optimizer_path);
        }
    }

    fn waitForWorkersToConnect(self: *Self) !void {
        const budget = protocol_limits.workerJoinBudget(self.config.required_workers);
        try budget.validate();
        const start_ms = std.time.milliTimestamp();
        var last_worker_count: ?usize = null;

        std.log.info("Waiting for workers to connect...", .{});
        if (self.control_state) |state| {
            state.setStatus(.waiting_for_workers);
            try updateWorkerWaitPhase(state, self.controller.getWorkerCount(), self.config.required_workers);
        }

        while (self.controller.base.is_running) {
            if (self.markCancelledIfRequested()) return error.Cancelled;
            const current_count = self.controller.getWorkerCount();
            if (current_count >= self.config.required_workers) {
                std.log.info("✓ {} workers connected", .{current_count});
                return;
            }
            if (self.control_state) |state| {
                if (last_worker_count == null or last_worker_count.? != current_count) {
                    try updateWorkerWaitPhase(state, current_count, self.config.required_workers);
                    last_worker_count = current_count;
                }
            }
            if (budget.expired(start_ms)) {
                std.log.warn("Timed out waiting for GRPO workers: {}/{}", .{ current_count, self.config.required_workers });
                return error.WorkerJoinTimeout;
            }
            std.time.sleep(budget.sleepMs(start_ms) * std.time.ns_per_ms);
        }
        return error.Cancelled;
    }

    fn waitForGenerationWorkers(self: *Self, generation_vmfb_path: []const u8, generation_weights_path: []const u8) !void {
        const budget = protocol_limits.generationWorkerReadyBudget(self.config.required_workers);
        try budget.validate();
        const start_ms = std.time.milliTimestamp();

        while (self.controller.base.is_running) {
            if (self.markCancelledIfRequested()) return error.Cancelled;
            const readiness = try self.controller.getGenerationWorkerReadiness(
                generation_vmfb_path,
                generation_weights_path,
            );
            if (self.control_state) |state| try updateGenerationWorkerPhase(state, readiness);
            if (readiness.ready >= self.config.required_workers) return;
            if (budget.expired(start_ms)) {
                std.log.warn("Timed out waiting for generation workers: ready={}/{} connected={}", .{
                    readiness.ready,
                    self.config.required_workers,
                    readiness.connected,
                });
                return error.GenerationWorkerReadyTimeout;
            }
            std.time.sleep(budget.sleepMs(start_ms) * std.time.ns_per_ms);
        }
        return error.Cancelled;
    }

    fn markCancelledIfRequested(self: *Self) bool {
        if (self.control_state) |state| {
            if (state.isCancellationRequested()) {
                self.status = .completed;
                state.setCancelled();
                return true;
            }
        }
        return false;
    }

    fn updateWorkerWaitPhase(state: *control_state_mod.ControllerState, connected: usize, required: usize) !void {
        var detail_buf: [96]u8 = undefined;
        const detail = try std.fmt.bufPrint(&detail_buf, "{d}/{d} workers connected", .{ connected, required });
        try state.setReadinessPhase("waiting_for_workers", detail);
    }

    fn updateGenerationWorkerPhase(state: *control_state_mod.ControllerState, readiness: @import("../nodes/gateway/controllers/training_controller.zig").ProgramReadiness) !void {
        var detail_buf: [160]u8 = undefined;
        const detail = try std.fmt.bufPrint(
            &detail_buf,
            "ready={d}/{d} init={d} connected={d} other_program={d}",
            .{
                readiness.ready,
                readiness.total,
                readiness.initializing,
                readiness.connected,
                readiness.other_program,
            },
        );
        try state.setReadinessPhase("waiting_for_generation_workers", detail);
    }

    fn deinit(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        // Clean up prompts
        for (self.prompts.items) |prompt| {
            self.allocator.free(prompt);
        }
        self.prompts.deinit();
    }

    fn getName(ptr: *anyopaque) []const u8 {
        _ = ptr;
        return "GRPO";
    }

    fn getStatus(ptr: *anyopaque) TrainingStatus {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.status;
    }

    fn emitRewardEvent(self: *Self, iteration: usize, rollouts_completed: usize, reward_mean: f32, reward_min: f32, reward_max: f32, reward_std: f32) !void {
        const client = self.gateway_client orelse return;
        const payload_json = try std.json.stringifyAlloc(self.allocator, .{
            .iteration = iteration,
            .rollouts_completed = rollouts_completed,
            .reward_mean = reward_mean,
            .reward_min = reward_min,
            .reward_max = reward_max,
            .reward_std = reward_std,
        }, .{});
        defer self.allocator.free(payload_json);

        const provenance_json = try std.json.stringifyAlloc(self.allocator, .{
            .actor_id = "pcp-rl-controller",
        }, .{});
        defer self.allocator.free(provenance_json);

        const event_id = try std.fmt.allocPrint(self.allocator, "rl_reward_{d}", .{iteration});
        defer self.allocator.free(event_id);
        const job_id = try std.fmt.allocPrint(self.allocator, "grpo-{d}", .{iteration});
        defer self.allocator.free(job_id);
        try client.emitEvent(event_id, "rl.reward_evaluated", job_id, payload_json, provenance_json);
    }
};
