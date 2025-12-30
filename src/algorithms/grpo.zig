const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const TrainingAlgorithm = @import("training_algorithm.zig").TrainingAlgorithm;
const TrainingStatus = @import("training_algorithm.zig").TrainingStatus;

// Forward declaration - RLShepherd will be defined in src/nodes/controllers/rl_shepherd.zig
pub const RLShepherd = @import("../nodes/controllers/rl_shepherd.zig").RLShepherd;
pub const RolloutData = @import("../nodes/controllers/rl_shepherd.zig").RolloutData;

pub const GRPOConfig = struct {
    num_iterations: usize = 10,
    group_size: usize = 4, // Number of generations per prompt
    learning_rate: f32 = 1e-6,
    beta: f32 = 0.1, // KL penalty coefficient
};

pub const GRPO = struct {
    allocator: Allocator,
    controller: *RLShepherd,
    config: GRPOConfig,
    status: TrainingStatus,
    prompts: ArrayList([]const i64),

    const Self = @This();

    pub fn init(allocator: Allocator, controller: *RLShepherd, config: GRPOConfig) Self {
        return Self{
            .allocator = allocator,
            .controller = controller,
            .config = config,
            .status = .not_started,
            .prompts = ArrayList([]const i64).init(allocator),
        };
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

        std.log.info("Starting GRPO Training Loop...", .{});

        // 1. Load Prompts
        if (self.prompts.items.len == 0) {
            try self.loadPrompts("data/rl_prompts.bin");
        }

        if (self.prompts.items.len == 0) {
            std.log.err("No prompts loaded! Cannot start training.", .{});
            return error.NoPromptsLoaded;
        }

        // 2. Initialize Training Backend (Shepherd-local)
        if (self.controller.training_backend == null) {
            try self.controller.initTrainingBackend("models/qwen_grpo_training.vmfb");
        }

        // 3. Initialize Generation Backend (Distributed to Workers)
        if (self.controller.generation_vmfb == null) {
            try self.controller.initGenerationBackend(
                "models/qwen_rl_generation.vmfb",
                "models/qwen_rl_generation.mlir",
            );
        }

        // 4. Ensure Workers are Ready
        std.log.info("Initializing Workers with Generation Model...", .{});
        try self.controller.prepareWorkersForGeneration("models/qwen_rl_generation.vmfb");

        // Give workers a moment to load the VMFB and allocate the KV cache
        std.time.sleep(2 * std.time.ns_per_s);
        std.log.info("Workers initialized. Starting training loop...", .{});

        // 5. Training Loop
        for (0..self.config.num_iterations) |iter| {
            std.log.info("=== GRPO Iteration {}/{} ===", .{ iter + 1, self.config.num_iterations });

            // A. Use Loaded Prompts
            const prompts = self.prompts.items;

            // B. Request Rollouts
            // We need 'group_size' completions per prompt
            var total_requests: usize = 0;
            for (prompts) |prompt| {
                for (0..self.config.group_size) |_| {
                    try self.controller.requestRollout(prompt);
                    total_requests += 1;
                }
            }

            // C. Collect Experience
            std.log.info("Waiting for {} rollouts...", .{total_requests});
            var rollouts = try self.controller.collectRollouts(total_requests);
            defer rollouts.deinit();

            // D. Compute Rewards (Format Check + Quality Heuristics)
            var rewards = std.ArrayList(f32).init(self.allocator);
            defer rewards.deinit();

            // Qwen 2.5 EOS token ID
            const eos_id: i64 = 151643;

            for (rollouts.items) |r| {
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

                std.log.info("Rollout reward: {d:.2} (len={}, eos={}, diversity={d:.2})", .{ score, len, has_eos, diversity });
            }

            // E. Train Step (Backprop + Update)
            // This runs on the Shepherd using the Training Graph
            try self.controller.trainStep(rollouts, rewards);

            // F. Broadcast Updated Weights to Workers
            try self.controller.broadcastNewWeights();
        }

        self.status = .completed;
        std.log.info("GRPO Training Completed.", .{});
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
};
