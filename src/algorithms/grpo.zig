const std = @import("std");
const Allocator = std.mem.Allocator;
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

    const Self = @This();

    pub fn init(allocator: Allocator, controller: *RLShepherd, config: GRPOConfig) Self {
        return Self{
            .allocator = allocator,
            .controller = controller,
            .config = config,
            .status = .not_started,
        };
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

        std.log.info("Starting GRPO Loop...", .{});

        // 1. Distribute Weights (Once at start, or every iter)
        // Assume weights loaded in controller

        for (0..self.config.num_iterations) |iter| {
            std.log.info("GRPO Iteration {}", .{iter});

            // A. Sample Prompts
            const prompts = [_][]const i64{
                &.{ 1, 2, 3 }, // Dummy prompt 1
                &.{ 4, 5, 6 }, // Dummy prompt 2
            };

            // B. Request Rollouts (Group Size G)
            for (prompts) |prompt| {
                // Send G requests to workers
                for (0..self.config.group_size) |_| {
                    try self.controller.requestRollout(prompt);
                }
            }

            // C. Collect Experience
            const expected_responses = prompts.len * self.config.group_size;
            const rollouts = try self.controller.collectRollouts(expected_responses);
            defer rollouts.deinit();

            // D. Compute Rewards (Simple Length Reward for MVP)
            var rewards = std.ArrayList(f32).init(self.allocator);
            defer rewards.deinit();

            for (rollouts.items) |r| {
                const len_reward = @as(f32, @floatFromInt(r.completion.len)) * 0.1;
                try rewards.append(len_reward);
            }

            // E. Compute Advantages (Group Normalization)
            // (Skipped for brevity: mean/std calculation)

            // F. Update Policy
            // Run training step on Controller's local Executor using the gathered data
            // try self.controller.trainStep(rollouts, advantages);

            std.log.info("Completed GRPO iteration {} with {} rollouts", .{ iter, rollouts.items.len });
        }

        self.status = .completed;
    }

    fn deinit(ptr: *anyopaque) void {
        _ = ptr;
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
