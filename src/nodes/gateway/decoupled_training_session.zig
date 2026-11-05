const common = @import("embedded_common.zig");

const std = common.std;
const Allocator = common.Allocator;
const DecoupledRuntimeStats = common.DecoupledRuntimeStats;
const NodeId = common.NodeId;
const WorkerFabricController = common.WorkerFabricController;
const decoupled_diloco = common.decoupled_diloco;
const message = common.message;
const protocol_invariants = @import("../../protocol/invariants.zig");
const protocol_limits = @import("../../protocol/limits.zig");

pub const FragmentStepResult = struct {
    syncer_step: usize,
    fragment_id: usize,
    metadata_count: usize,
    response_count: usize,
    participant_count: usize,
    version: usize,
    average_loss: ?f32,
    train_examples: usize,
};

pub const FragmentStepOptions = struct {
    collect_response_metrics: bool = false,
};

const ParsedLearnerUpdates = struct {
    owned_updates: []common.OwnedDecoupledLearnerUpdate,
    owned_count: usize,
    learner_updates: []decoupled_diloco.LearnerFragmentUpdate,
    availability: []decoupled_diloco.LearnerAvailability,
    average_loss: ?f32,
    train_examples: usize,

    fn deinit(self: *@This(), allocator: Allocator) void {
        for (self.owned_updates[0..self.owned_count]) |*owned| {
            owned.deinit(allocator);
        }
        allocator.free(self.owned_updates);
        allocator.free(self.learner_updates);
        allocator.free(self.availability);
    }
};

pub const ScheduledStepUpdate = struct {
    step: FragmentStepResult,
    completed_syncer_steps: usize,
    train_examples: usize,
    final_loss: ?f32,
};

pub const ScheduledRunResult = struct {
    completed_syncer_steps: usize = 0,
    executed_steps: usize = 0,
    train_examples: usize = 0,
    final_loss: ?f32 = null,
};

pub const CancellationHook = struct {
    ctx: *anyopaque,
    is_cancelled: *const fn (ctx: *anyopaque) bool,
};

pub const StepHook = struct {
    ctx: *anyopaque,
    on_step: *const fn (ctx: *anyopaque, update: ScheduledStepUpdate) anyerror!void,
};

pub const ScheduledRunOptions = struct {
    total_syncer_steps: usize,
    stats: *DecoupledRuntimeStats,
    fragment_options: FragmentStepOptions = .{},
    cancellation: ?CancellationHook = null,
    step_hook: ?StepHook = null,
};

pub const DecoupledTrainingSession = struct {
    allocator: Allocator,
    worker_fabric: *WorkerFabricController,
    syncer: *decoupled_diloco.DecoupledDiLoCo,
    request_id: message.RequestId,
    worker_ids: []const NodeId,
    min_quorum: usize,
    parameter_shapes: []const []const i64,
    master_weights: []u8,
    target_weights: ?[]u8 = null,
    target_ema_momentum: ?f32 = null,
    collect_timeout_ms: u64 = protocol_limits.decoupled_collection_timeout_ms,

    const Self = @This();

    pub fn runScheduledSteps(self: *Self, options: ScheduledRunOptions) !ScheduledRunResult {
        try self.validateRunOptions(options);
        var result = ScheduledRunResult{};

        var syncer_step: usize = 1;
        while (syncer_step <= options.total_syncer_steps) : (syncer_step += 1) {
            if (options.cancellation) |hook| {
                if (hook.is_cancelled(hook.ctx)) return error.Cancelled;
            }

            const fragment_id = self.syncer.scheduledFragmentId(syncer_step) orelse continue;
            const step_result = try self.runFragmentStep(syncer_step, fragment_id, options.stats, options.fragment_options);

            result.completed_syncer_steps = syncer_step;
            result.executed_steps += 1;
            result.train_examples += step_result.train_examples;
            if (step_result.average_loss) |average_loss| result.final_loss = average_loss;

            if (options.step_hook) |hook| {
                try hook.on_step(hook.ctx, .{
                    .step = step_result,
                    .completed_syncer_steps = result.completed_syncer_steps,
                    .train_examples = result.train_examples,
                    .final_loss = result.final_loss,
                });
            }
        }

        return result;
    }

    pub fn stopWorkers(self: *Self, label: []const u8) void {
        stopWorkerLoops(self.allocator, self.worker_fabric, self.request_id, self.worker_ids, label);
    }

    pub fn runFragmentStep(
        self: *Self,
        syncer_step: usize,
        fragment_id: usize,
        stats: *DecoupledRuntimeStats,
        options: FragmentStepOptions,
    ) !FragmentStepResult {
        try protocol_invariants.validateFragmentIndex(fragment_id, self.syncer.fragment_plan.fragments.len);
        protocol_invariants.assertFragmentIndex(fragment_id, self.syncer.fragment_plan.fragments.len);

        const fragment = self.syncer.fragment_plan.fragments[fragment_id];
        const fragment_bytes = common.decoupledFragmentByteCount(fragment);

        var metadata = try self.collectLearnerMetadata();
        defer {
            for (metadata.items) |*msg| msg.deinit(self.allocator);
            metadata.deinit();
        }
        if (metadata.items.len < self.min_quorum) return error.QuorumNotMet;
        std.debug.assert(metadata.items.len <= self.worker_ids.len);

        try self.sendFragmentPulls(metadata.items, fragment_id, syncer_step);

        var responses = try self.collectFragmentResponses(metadata.items.len, fragment_id, syncer_step);
        defer {
            for (responses.items) |*msg| msg.deinit(self.allocator);
            responses.deinit();
        }
        if (responses.items.len < self.min_quorum) return error.QuorumNotMet;
        std.debug.assert(responses.items.len <= metadata.items.len);

        var parsed_updates = try self.parseLearnerUpdates(responses.items, fragment, options);
        defer parsed_updates.deinit(self.allocator);

        const sync_result = try self.syncer.applyFragmentStepWithAvailability(
            syncer_step,
            fragment_id,
            parsed_updates.learner_updates,
            parsed_updates.availability,
        );
        stats.recordStep(sync_result);
        stats.learner_to_syncer_fragment_bytes += fragment_bytes * responses.items.len;
        try self.copyAndBroadcastReady(fragment_id, syncer_step);
        stats.syncer_to_learner_fragment_bytes += self.readyBroadcastBytes(fragment_bytes);

        return .{
            .syncer_step = syncer_step,
            .fragment_id = fragment_id,
            .metadata_count = metadata.items.len,
            .response_count = responses.items.len,
            .participant_count = sync_result.participant_count,
            .version = sync_result.version,
            .average_loss = parsed_updates.average_loss,
            .train_examples = parsed_updates.train_examples,
        };
    }

    fn validateRunOptions(self: *Self, options: ScheduledRunOptions) !void {
        try protocol_invariants.validateQuorum(self.min_quorum, self.worker_ids.len);
        protocol_invariants.assertQuorum(self.min_quorum, self.worker_ids.len);
        if (options.total_syncer_steps == 0) return error.InvalidSyncerSteps;
        if (options.total_syncer_steps > protocol_limits.syncer_steps_max) return error.SyncerStepsTooLarge;
        if (self.request_id == 0) return error.InvalidRequestId;
        if (self.master_weights.len == 0) return error.InvalidByteCount;
    }

    fn collectLearnerMetadata(self: *Self) !std.ArrayList(common.TimedDecoupledMessage) {
        var budget = protocol_limits.boundedCollectionBudget(self.worker_ids.len, self.collect_timeout_ms);
        budget.reason = "decoupled_learner_metadata";
        try budget.validate();

        return try common.collectDecoupledMessagesUniqueTimed(
            self.worker_fabric,
            self.allocator,
            .{
                .msg_type = message.MessageType.DECOUPLED_LEARNER_METADATA,
                .request_id = self.request_id,
            },
            self.worker_ids,
            self.min_quorum,
            budget.expected_count,
            budget.timeout_ms,
            self.syncer.currentGraceWindowMs(),
        );
    }

    fn sendFragmentPulls(
        self: *Self,
        metadata: []const common.TimedDecoupledMessage,
        fragment_id: usize,
        syncer_step: usize,
    ) !void {
        const fragment = self.syncer.fragment_plan.fragments[fragment_id];
        var pull_payload = std.json.ObjectMap.init(self.allocator);
        defer pull_payload.deinit();
        try pull_payload.put(message.DecoupledDiLoCoField.FRAGMENT_ID, .{ .integer = @intCast(fragment_id) });
        try pull_payload.put(message.DecoupledDiLoCoField.FRAGMENT_ROUND, .{ .integer = @intCast(syncer_step) });
        try pull_payload.put(message.DecoupledDiLoCoField.SYNCER_STEP, .{ .integer = @intCast(syncer_step) });

        var tensor_indices = std.json.Array.init(self.allocator);
        defer tensor_indices.deinit();
        for (fragment.tensors) |range| {
            try tensor_indices.append(.{ .integer = @intCast(range.tensor_index) });
        }
        try pull_payload.put("tensor_indices", .{ .array = tensor_indices });

        for (metadata) |meta_msg| {
            try self.worker_fabric.sendToWorkerWithContext(
                meta_msg.envelope.sender_node,
                message.MessageType.DECOUPLED_FRAGMENT_PULL,
                .{ .object = pull_payload },
                .{
                    .request_id = self.request_id,
                    .round_id = @intCast(syncer_step),
                    .task_id = @intCast(fragment_id + 1),
                },
            );
        }
    }

    fn collectFragmentResponses(
        self: *Self,
        expected_count: usize,
        fragment_id: usize,
        syncer_step: usize,
    ) !std.ArrayList(common.TimedDecoupledMessage) {
        var budget = protocol_limits.boundedCollectionBudget(expected_count, self.collect_timeout_ms);
        budget.reason = "decoupled_fragment_update";
        try budget.validate();

        return try common.collectDecoupledMessagesTimed(
            self.worker_fabric,
            self.allocator,
            .{
                .msg_type = message.MessageType.DECOUPLED_FRAGMENT_UPDATE,
                .request_id = self.request_id,
                .round_id = @intCast(syncer_step),
                .task_id = @intCast(fragment_id + 1),
            },
            self.min_quorum,
            budget.expected_count,
            budget.timeout_ms,
            self.syncer.currentGraceWindowMs(),
        );
    }

    fn parseLearnerUpdates(
        self: *Self,
        responses: []const common.TimedDecoupledMessage,
        fragment: common.fragments.Fragment,
        options: FragmentStepOptions,
    ) !ParsedLearnerUpdates {
        var parsed = try self.allocateParsedUpdates(responses.len);
        errdefer parsed.deinit(self.allocator);

        var step_loss: f32 = 0.0;
        var loss_count: usize = 0;
        for (responses) |response| {
            parsed.owned_updates[parsed.owned_count] = try common.parseDecoupledLearnerUpdate(
                self.allocator,
                response.envelope,
                fragment,
                response.arrival_ms,
            );
            parsed.learner_updates[parsed.owned_count] = parsed.owned_updates[parsed.owned_count].update;
            parsed.owned_count += 1;
            recordResponseMetrics(response.envelope, options, &step_loss, &loss_count, &parsed.train_examples);
        }

        for (self.worker_ids, 0..) |worker_id, idx| {
            parsed.availability[idx] = .{ .learner_id = worker_id, .status = .active };
        }
        parsed.average_loss = if (loss_count > 0) step_loss / @as(f32, @floatFromInt(loss_count)) else null;
        return parsed;
    }

    fn allocateParsedUpdates(self: *Self, response_count: usize) !ParsedLearnerUpdates {
        const owned_updates = try self.allocator.alloc(common.OwnedDecoupledLearnerUpdate, response_count);
        errdefer self.allocator.free(owned_updates);
        const learner_updates = try self.allocator.alloc(decoupled_diloco.LearnerFragmentUpdate, response_count);
        errdefer self.allocator.free(learner_updates);
        const availability = try self.allocator.alloc(decoupled_diloco.LearnerAvailability, self.worker_ids.len);
        return .{
            .owned_updates = owned_updates,
            .owned_count = 0,
            .learner_updates = learner_updates,
            .availability = availability,
            .average_loss = null,
            .train_examples = 0,
        };
    }

    fn recordResponseMetrics(
        envelope: message.MessageEnvelope,
        options: FragmentStepOptions,
        step_loss: *f32,
        loss_count: *usize,
        train_examples: *usize,
    ) void {
        if (!options.collect_response_metrics) return;
        const payload = switch (envelope.data) {
            .object => |object| object,
            else => {
                std.log.warn("Decoupled fragment update without metrics object from worker {}", .{envelope.sender_node});
                return;
            },
        };
        if (common.parsePayloadF32(payload, "loss")) |loss| {
            step_loss.* += loss;
            loss_count.* += 1;
        } else {
            std.log.warn("Decoupled fragment update missing loss from worker {}", .{envelope.sender_node});
        }
        if (common.parsePayloadUsize(payload, message.DecoupledDiLoCoField.TRAIN_EXAMPLES)) |examples| {
            train_examples.* += examples;
        } else {
            std.log.warn("Decoupled fragment update missing train_examples from worker {}", .{envelope.sender_node});
        }
    }

    fn copyAndBroadcastReady(self: *Self, fragment_id: usize, syncer_step: usize) !void {
        try common.copySyncerFragmentToWeightBlob(self.syncer, self.parameter_shapes, self.master_weights, fragment_id);
        if (self.target_weights) |target| {
            if (self.target_ema_momentum) |momentum| {
                common.applyPrivateTargetEma(target, self.master_weights, momentum);
            }
        }

        var ready = try common.makeDecoupledReadyValue(
            self.allocator,
            self.syncer,
            self.parameter_shapes,
            self.target_weights,
            fragment_id,
            syncer_step,
        );
        defer ready.deinit();
        try self.worker_fabric.broadcastToWorkerIdsWithContext(
            self.worker_ids,
            message.MessageType.DECOUPLED_FRAGMENT_READY,
            ready.value,
            .{
                .request_id = self.request_id,
                .round_id = @intCast(syncer_step),
                .task_id = @intCast(fragment_id + 1),
            },
        );
    }

    fn readyBroadcastBytes(self: *const Self, fragment_bytes: usize) usize {
        const ready_fragment_multiplier: usize = if (self.target_weights != null) 2 else 1;
        return fragment_bytes * ready_fragment_multiplier * self.worker_ids.len;
    }
};

pub fn stopWorkerLoops(
    allocator: Allocator,
    worker_fabric: *WorkerFabricController,
    request_id: message.RequestId,
    worker_ids: []const NodeId,
    label: []const u8,
) void {
    for (worker_ids) |worker_id| {
        var stop_payload = std.json.ObjectMap.init(allocator);
        defer stop_payload.deinit();
        worker_fabric.sendToWorkerWithContext(
            worker_id,
            message.MessageType.STOP_DECOUPLED_DILOCO_LOOP,
            .{ .object = stop_payload },
            .{ .request_id = request_id },
        ) catch |err| {
            std.log.warn("Failed to stop {s} worker {}: {}", .{ label, worker_id, err });
        };
    }
}
