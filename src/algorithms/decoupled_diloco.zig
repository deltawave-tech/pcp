const std = @import("std");
const fragments_mod = @import("fragments.zig");
const decoupled_merge = @import("decoupled_merge.zig");
const decoupled_learner = @import("decoupled_learner.zig");
const vector_clock_mod = @import("vector_clock.zig");
const event_tape_mod = @import("event_tape.zig");

const Allocator = std.mem.Allocator;
const FragmentPlan = fragments_mod.FragmentPlan;
const FragmentStrategy = fragments_mod.FragmentStrategy;
const TensorRange = fragments_mod.TensorRange;
const TensorSpec = fragments_mod.TensorSpec;
pub const NodeId = vector_clock_mod.NodeId;

pub const DecoupledDiLoCoStatus = enum {
    not_started,
    initializing,
    running,
    completed,
    failed,
    paused,
};

pub const LearnerAvailabilityStatus = enum {
    active,
    stale,
    disconnected,
    failed,
    recovering,
};

pub const LearnerAvailability = struct {
    learner_id: NodeId,
    status: LearnerAvailabilityStatus = .active,
};

pub const SyncTimingMetrics = struct {
    step_duration_ema_ms: f64 = 0.0,
    quorum_wait_duration_ema_ms: f64 = 0.0,
    sync_send_duration_ema_ms: f64 = 0.0,
    last_participant_count: usize = 0,
    last_grace_inclusion_count: usize = 0,
    last_skipped_learner_count: usize = 0,
    last_grace_window_ms: u64 = 0,
    last_quorum_wait_ms: u64 = 0,
};

pub const MergeStrategy = enum {
    weighted_average,
    rda,
    avg_embedding_rda_model,
    rda_embedding_avg_model,
};

pub const OuterGradientCompression = enum {
    none,
    int4,
};

const OuterNesterov = struct {
    allocator: Allocator,
    learning_rate: f32,
    momentum: f32,
    velocities: [][]f32 = &[_][]f32{},

    fn init(allocator: Allocator, learning_rate: f32, momentum: f32) OuterNesterov {
        return .{
            .allocator = allocator,
            .learning_rate = learning_rate,
            .momentum = momentum,
        };
    }

    fn initFromMasterParams(self: *OuterNesterov, master_params: []const []const f32) !void {
        self.velocities = try self.allocator.alloc([]f32, master_params.len);
        var initialized: usize = 0;
        errdefer {
            for (self.velocities[0..initialized]) |velocity| self.allocator.free(velocity);
            self.allocator.free(self.velocities);
            self.velocities = &[_][]f32{};
        }

        for (master_params, 0..) |param, i| {
            self.velocities[i] = try self.allocator.alloc(f32, param.len);
            @memset(self.velocities[i], 0.0);
            initialized += 1;
        }
    }

    fn deinit(self: *OuterNesterov) void {
        for (self.velocities) |velocity| self.allocator.free(velocity);
        if (self.velocities.len > 0) self.allocator.free(self.velocities);
        self.velocities = &[_][]f32{};
    }

    fn update(self: *OuterNesterov, tensor_index: usize, master_param: []f32, outer_gradient: []const f32) !void {
        if (tensor_index >= self.velocities.len) return error.OutOfBounds;
        if (master_param.len != outer_gradient.len) return error.DimensionMismatch;

        const velocity = self.velocities[tensor_index];
        if (velocity.len != master_param.len) return error.DimensionMismatch;

        for (master_param, 0..) |*param, i| {
            const grad = outer_gradient[i];
            const v_new = self.momentum * velocity[i] + grad;
            velocity[i] = v_new;
            param.* -= self.learning_rate * (grad + self.momentum * v_new);
        }
    }
};

pub const DecoupledDiLoCoConfig = struct {
    num_fragments: usize = 24,
    sync_interval_h: usize = 24,
    overlap_tau: usize = 2,
    min_quorum: usize = 1,
    fragment_strategy: FragmentStrategy = .balanced_tensor,
    merge_strategy: MergeStrategy = .avg_embedding_rda_model,
    embedding_tensor_indices: []const usize = &.{},
    learner_alpha: f32 = 0.0,
    outer_gradient_compression: OuterGradientCompression = .none,
    max_recovery_syncer_steps: usize = 24,
    outer_learning_rate: f32 = 1.0,
    outer_momentum: f32 = 0.0,
    grace_window_ms: u64 = 0,
    grace_gamma: f64 = 1.0,
    max_grace_steps: usize = 1,
    adaptive_grace_enabled: bool = false,
    timing_ema_alpha: f64 = 0.125,

    pub fn validate(self: DecoupledDiLoCoConfig) !void {
        if (self.num_fragments == 0) return error.InvalidFragmentCount;
        if (self.sync_interval_h == 0) return error.InvalidSyncInterval;
        if (self.overlap_tau == 0) return error.InvalidOverlapTau;
        if (self.min_quorum == 0) return error.InvalidQuorum;
        if (self.learner_alpha < 0.0 or self.learner_alpha > 1.0) return error.InvalidLearnerAlpha;
        if (self.grace_gamma < 0.0 or !std.math.isFinite(self.grace_gamma)) return error.InvalidGraceGamma;
        if (self.max_grace_steps == 0) return error.InvalidMaxGraceSteps;
        if (self.timing_ema_alpha <= 0.0 or self.timing_ema_alpha > 1.0 or !std.math.isFinite(self.timing_ema_alpha)) return error.InvalidTimingEmaAlpha;
    }
};

pub const TensorUpdate = struct {
    tensor_index: usize,
    values: []const f32,
};

pub const LearnerFragmentUpdate = struct {
    learner_id: NodeId,
    learner_step: usize,
    steps_since_fragment_update: usize,
    tokens_since_fragment_update: usize,
    arrival_ms: u64 = 0,
    tensors: []const TensorUpdate,

    pub fn mergeWeight(self: LearnerFragmentUpdate) f32 {
        return decoupled_merge.tokenStepWeight(self.tokens_since_fragment_update, self.steps_since_fragment_update);
    }
};

pub const QuorumSelectionMetrics = struct {
    participant_count: usize = 0,
    grace_inclusion_count: usize = 0,
    skipped_learner_count: usize = 0,
    quorum_wait_ms: u64 = 0,
    grace_window_ms: u64 = 0,
};

pub const QuorumSelection = struct {
    updates: []LearnerFragmentUpdate,
    metrics: QuorumSelectionMetrics,

    pub fn deinit(self: *QuorumSelection, allocator: Allocator) void {
        allocator.free(self.updates);
        self.* = undefined;
    }
};

pub const FragmentSyncState = struct {
    fragment_id: usize,
    offset: usize,
    version: usize = 0,
    last_syncer_step: usize = 0,
    last_participant_count: usize = 0,
    last_grace_inclusion_count: usize = 0,
    last_skipped_learner_count: usize = 0,
    last_quorum_wait_ms: u64 = 0,
    last_grace_window_ms: u64 = 0,
    last_total_weight: f32 = 0.0,
    last_compressed_values: usize = 0,
    last_compression_max_abs_error: f32 = 0.0,
};

pub const FragmentSyncResult = struct {
    did_sync: bool,
    syncer_step: usize,
    fragment_id: usize,
    participant_count: usize,
    total_weight: f32,
    sanitized_values: usize,
    version: usize,
    grace_inclusion_count: usize = 0,
    skipped_learner_count: usize = 0,
    quorum_wait_ms: u64 = 0,
    grace_window_ms: u64 = 0,
    compressed_values: usize = 0,
    compression_max_abs_error: f32 = 0.0,
};

pub const SyncerCheckpoint = struct {
    allocator: Allocator,
    global_step: usize,
    master_params: [][]f32,
    optimizer_velocities: [][]f32,
    fragment_versions: []usize,
    vector_clock: vector_clock_mod.VectorClock,
    event_tape: event_tape_mod.EventTapeSnapshot,
    num_fragments: usize,
    sync_interval_h: usize,
    fragment_strategy: FragmentStrategy,
    merge_strategy: MergeStrategy,
    event_tape_cursor: usize,

    pub fn deinit(self: *SyncerCheckpoint) void {
        freeParamMatrix(self.allocator, self.master_params);
        freeParamMatrix(self.allocator, self.optimizer_velocities);
        self.allocator.free(self.fragment_versions);
        self.vector_clock.deinit();
        self.event_tape.deinit(self.allocator);
        self.* = undefined;
    }
};

pub const DecoupledDiLoCo = struct {
    allocator: Allocator,
    config: DecoupledDiLoCoConfig,
    status: DecoupledDiLoCoStatus,
    fragment_plan: FragmentPlan,
    fragment_states: []FragmentSyncState,
    host_optimizer: OuterNesterov,
    master_params: [][]f32,
    embedding_tensor_index_overrides: []usize,
    vector_clock: vector_clock_mod.VectorClock,
    event_tape: event_tape_mod.EventTape,
    timing_metrics: SyncTimingMetrics,
    global_step: usize,

    const Self = @This();

    pub fn initWithMaster(
        allocator: Allocator,
        config: DecoupledDiLoCoConfig,
        tensor_specs: []const TensorSpec,
        initial_master_params: []const []const f32,
    ) !Self {
        try config.validate();
        try validateTensorSpecsAgainstMaster(tensor_specs, initial_master_params);
        try validateEmbeddingTensorOverrides(config.embedding_tensor_indices, initial_master_params.len);

        var plan = try fragments_mod.buildFragmentPlan(
            allocator,
            config.fragment_strategy,
            tensor_specs,
            config.num_fragments,
            config.sync_interval_h,
        );
        errdefer plan.deinit();

        var fragment_states = try allocator.alloc(FragmentSyncState, plan.fragments.len);
        errdefer allocator.free(fragment_states);
        for (plan.fragments, 0..) |fragment, i| {
            fragment_states[i] = .{
                .fragment_id = fragment.id,
                .offset = fragment.offset,
            };
        }

        var master_params = try allocator.alloc([]f32, initial_master_params.len);
        var initialized_master: usize = 0;
        errdefer {
            for (master_params[0..initialized_master]) |param| allocator.free(param);
            allocator.free(master_params);
        }
        for (initial_master_params, 0..) |param, i| {
            master_params[i] = try allocator.dupe(f32, param);
            initialized_master += 1;
        }

        var host_optimizer = OuterNesterov.init(allocator, config.outer_learning_rate, config.outer_momentum);
        errdefer host_optimizer.deinit();
        try host_optimizer.initFromMasterParams(initial_master_params);

        const embedding_tensor_index_overrides = try allocator.dupe(usize, config.embedding_tensor_indices);
        errdefer allocator.free(embedding_tensor_index_overrides);
        var owned_config = config;
        owned_config.embedding_tensor_indices = embedding_tensor_index_overrides;

        var vector_clock = vector_clock_mod.VectorClock.init(allocator);
        errdefer vector_clock.deinit();
        try vector_clock.observe(0, 0);

        var event_tape = event_tape_mod.EventTape.init(allocator);
        errdefer event_tape.deinit();

        return .{
            .allocator = allocator,
            .config = owned_config,
            .status = .not_started,
            .fragment_plan = plan,
            .fragment_states = fragment_states,
            .host_optimizer = host_optimizer,
            .master_params = master_params,
            .embedding_tensor_index_overrides = embedding_tensor_index_overrides,
            .vector_clock = vector_clock,
            .event_tape = event_tape,
            .timing_metrics = .{},
            .global_step = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.host_optimizer.deinit();
        for (self.master_params) |param| self.allocator.free(param);
        self.allocator.free(self.master_params);
        self.allocator.free(self.embedding_tensor_index_overrides);
        self.vector_clock.deinit();
        self.event_tape.deinit();
        self.allocator.free(self.fragment_states);
        self.fragment_plan.deinit();
        self.* = undefined;
    }

    pub fn getName(_: *Self) []const u8 {
        return "DecoupledDiLoCo";
    }

    pub fn getStatus(self: *Self) DecoupledDiLoCoStatus {
        return self.status;
    }

    pub fn scheduledFragmentId(self: *Self, syncer_step: usize) ?usize {
        const step_offset = syncer_step % self.config.sync_interval_h;
        for (self.fragment_plan.fragments) |fragment| {
            if (fragment.offset == step_offset) return fragment.id;
        }
        return null;
    }

    pub fn recordStepDuration(self: *Self, duration_ms: f64) void {
        self.timing_metrics.step_duration_ema_ms = updateEma(
            self.timing_metrics.step_duration_ema_ms,
            duration_ms,
            self.config.timing_ema_alpha,
        );
    }

    pub fn recordQuorumWaitDuration(self: *Self, duration_ms: f64) void {
        self.timing_metrics.quorum_wait_duration_ema_ms = updateEma(
            self.timing_metrics.quorum_wait_duration_ema_ms,
            duration_ms,
            self.config.timing_ema_alpha,
        );
    }

    pub fn recordSyncSendDuration(self: *Self, duration_ms: f64) void {
        self.timing_metrics.sync_send_duration_ema_ms = updateEma(
            self.timing_metrics.sync_send_duration_ema_ms,
            duration_ms,
            self.config.timing_ema_alpha,
        );
    }

    pub fn currentGraceWindowMs(self: *Self) u64 {
        if (!self.config.adaptive_grace_enabled) return self.config.grace_window_ms;
        if (self.timing_metrics.step_duration_ema_ms <= 0.0) return 0;

        const overlap_tau: f64 = @floatFromInt(self.config.overlap_tau);
        const slack_ms = overlap_tau * self.timing_metrics.step_duration_ema_ms -
            (self.timing_metrics.quorum_wait_duration_ema_ms + self.timing_metrics.sync_send_duration_ema_ms);
        if (slack_ms <= 0.0 or !std.math.isFinite(slack_ms)) return 0;

        const max_grace_steps: f64 = @floatFromInt(self.config.max_grace_steps);
        const max_grace_ms = max_grace_steps * self.timing_metrics.step_duration_ema_ms;
        const adaptive_ms = self.config.grace_gamma * slack_ms;
        const clamped_ms = @min(adaptive_ms, max_grace_ms);
        if (clamped_ms <= 0.0 or !std.math.isFinite(clamped_ms)) return 0;
        if (clamped_ms >= @as(f64, @floatFromInt(std.math.maxInt(u64)))) return std.math.maxInt(u64);

        return @intFromFloat(@floor(clamped_ms));
    }

    pub fn applyScheduledStep(
        self: *Self,
        syncer_step: usize,
        learner_updates: []const LearnerFragmentUpdate,
    ) !?FragmentSyncResult {
        const fragment_id = self.scheduledFragmentId(syncer_step) orelse return null;
        return try self.applyFragmentStep(syncer_step, fragment_id, learner_updates);
    }

    pub fn selectLearnerUpdates(
        self: *Self,
        learner_updates: []const LearnerFragmentUpdate,
        availability: []const LearnerAvailability,
    ) !QuorumSelection {
        var candidates = std.ArrayList(LearnerFragmentUpdate).init(self.allocator);
        errdefer candidates.deinit();

        var skipped: usize = 0;
        if (availability.len == 0) {
            try candidates.appendSlice(learner_updates);
        } else {
            for (availability) |learner| {
                if (learner.status != .active) {
                    skipped += 1;
                    continue;
                }
                if (earliestUpdateForLearner(learner_updates, learner.learner_id)) |update| {
                    try candidates.append(update);
                } else {
                    skipped += 1;
                }
            }
        }

        if (candidates.items.len < self.config.min_quorum) return error.QuorumNotMet;
        std.mem.sort(LearnerFragmentUpdate, candidates.items, {}, learnerArrivalLessThan);

        const quorum_wait_ms = candidates.items[self.config.min_quorum - 1].arrival_ms;
        const grace_window_ms = self.currentGraceWindowMs();
        const deadline_ms = saturatingAddU64(quorum_wait_ms, grace_window_ms);

        var selected_count: usize = 0;
        var grace_inclusion_count: usize = 0;
        for (candidates.items) |candidate| {
            if (candidate.arrival_ms > deadline_ms) break;
            if (candidate.arrival_ms > quorum_wait_ms) grace_inclusion_count += 1;
            selected_count += 1;
        }

        if (selected_count < self.config.min_quorum) return error.QuorumNotMet;
        skipped += candidates.items.len - selected_count;

        const selected = try self.allocator.dupe(LearnerFragmentUpdate, candidates.items[0..selected_count]);
        candidates.deinit();

        return .{
            .updates = selected,
            .metrics = .{
                .participant_count = selected_count,
                .grace_inclusion_count = grace_inclusion_count,
                .skipped_learner_count = skipped,
                .quorum_wait_ms = quorum_wait_ms,
                .grace_window_ms = grace_window_ms,
            },
        };
    }

    pub fn applyFragmentStepWithAvailability(
        self: *Self,
        syncer_step: usize,
        fragment_id: usize,
        learner_updates: []const LearnerFragmentUpdate,
        availability: []const LearnerAvailability,
    ) !FragmentSyncResult {
        var selection = try self.selectLearnerUpdates(learner_updates, availability);
        defer selection.deinit(self.allocator);
        return try self.applyFragmentStepSelected(syncer_step, fragment_id, selection.updates, selection.metrics);
    }

    pub fn selectReplayLearnerUpdates(
        self: *Self,
        syncer_step: usize,
        fragment_id: usize,
        learner_updates: []const LearnerFragmentUpdate,
    ) !QuorumSelection {
        const entry = self.event_tape.replayFragmentSync(syncer_step, fragment_id) orelse return error.MissingReplayEntry;
        var selected = try self.allocator.alloc(LearnerFragmentUpdate, entry.participants.len);
        var selected_count: usize = 0;
        errdefer self.allocator.free(selected);

        for (entry.participants) |participant| {
            const update = findLearnerUpdate(learner_updates, participant.learner_id) orelse return error.MissingReplayUpdate;
            selected[selected_count] = update;
            selected_count += 1;
        }

        return .{
            .updates = selected,
            .metrics = .{
                .participant_count = selected_count,
                .skipped_learner_count = learner_updates.len - selected_count,
            },
        };
    }

    pub fn applyFragmentStep(
        self: *Self,
        syncer_step: usize,
        fragment_id: usize,
        learner_updates: []const LearnerFragmentUpdate,
    ) !FragmentSyncResult {
        const metrics = QuorumSelectionMetrics{ .participant_count = learner_updates.len };
        return try self.applyFragmentStepSelected(syncer_step, fragment_id, learner_updates, metrics);
    }

    fn applyFragmentStepSelected(
        self: *Self,
        syncer_step: usize,
        fragment_id: usize,
        learner_updates: []const LearnerFragmentUpdate,
        selection_metrics: QuorumSelectionMetrics,
    ) !FragmentSyncResult {
        if (fragment_id >= self.fragment_plan.fragments.len) return error.InvalidFragmentId;
        if (learner_updates.len < self.config.min_quorum) return error.QuorumNotMet;

        const fragment = self.fragment_plan.fragments[fragment_id];
        var sanitized_values: usize = 0;
        var last_total_weight: f32 = 0.0;
        var compressed_values: usize = 0;
        var compression_max_abs_error: f32 = 0.0;

        for (fragment.tensors) |range| {
            var weighted = try self.allocator.alloc(decoupled_merge.WeightedFragment, learner_updates.len);
            defer self.allocator.free(weighted);

            for (learner_updates, 0..) |learner, i| {
                const tensor_update = findTensorUpdate(learner.tensors, range.tensor_index) orelse return error.MissingTensorUpdate;
                if (tensor_update.values.len != range.value_count) return error.FragmentLengthMismatch;
                weighted[i] = .{
                    .values = tensor_update.values,
                    .weight = learner.mergeWeight(),
                };
            }

            const master = self.master_params[range.tensor_index];
            if (master.len != range.value_count) return error.FragmentLengthMismatch;

            const outer_gradient = try self.allocator.alloc(f32, range.value_count);
            defer self.allocator.free(outer_gradient);

            const merge_stats = switch (self.strategyForTensor(range)) {
                .weighted_average => try decoupled_merge.mergeDirectWeightedOuterGradient(outer_gradient, master, weighted),
                .rda => try decoupled_merge.mergeRdaWeightedOuterGradient(outer_gradient, master, weighted),
                .avg_embedding_rda_model => unreachable,
                .rda_embedding_avg_model => unreachable,
            };
            sanitized_values += merge_stats.sanitized_values;
            last_total_weight = merge_stats.total_weight;

            switch (self.config.outer_gradient_compression) {
                .none => {},
                .int4 => {
                    const compression_stats = try decoupled_merge.compressDecompressInt4InPlace(self.allocator, outer_gradient);
                    compressed_values += compression_stats.value_count;
                    compression_max_abs_error = @max(compression_max_abs_error, compression_stats.max_abs_error);
                    sanitized_values += compression_stats.sanitized_values;
                },
            }

            try self.host_optimizer.update(range.tensor_index, master, outer_gradient);
        }

        const state = &self.fragment_states[fragment_id];
        state.version += 1;
        state.last_syncer_step = syncer_step;
        state.last_participant_count = learner_updates.len;
        state.last_grace_inclusion_count = selection_metrics.grace_inclusion_count;
        state.last_skipped_learner_count = selection_metrics.skipped_learner_count;
        state.last_quorum_wait_ms = selection_metrics.quorum_wait_ms;
        state.last_grace_window_ms = selection_metrics.grace_window_ms;
        state.last_total_weight = last_total_weight;
        state.last_compressed_values = compressed_values;
        state.last_compression_max_abs_error = compression_max_abs_error;
        self.global_step = syncer_step;
        self.timing_metrics.quorum_wait_duration_ema_ms = updateEma(
            self.timing_metrics.quorum_wait_duration_ema_ms,
            @floatFromInt(selection_metrics.quorum_wait_ms),
            self.config.timing_ema_alpha,
        );
        self.timing_metrics.last_participant_count = learner_updates.len;
        self.timing_metrics.last_grace_inclusion_count = selection_metrics.grace_inclusion_count;
        self.timing_metrics.last_skipped_learner_count = selection_metrics.skipped_learner_count;
        self.timing_metrics.last_quorum_wait_ms = selection_metrics.quorum_wait_ms;
        self.timing_metrics.last_grace_window_ms = selection_metrics.grace_window_ms;

        _ = try self.vector_clock.tick(0);
        const participant_records = try buildParticipantRecords(self.allocator, learner_updates);
        defer self.allocator.free(participant_records);
        _ = try self.event_tape.appendFragmentSync(syncer_step, fragment_id, participant_records, self.vector_clock);

        return .{
            .did_sync = true,
            .syncer_step = syncer_step,
            .fragment_id = fragment_id,
            .participant_count = learner_updates.len,
            .total_weight = last_total_weight,
            .sanitized_values = sanitized_values,
            .version = state.version,
            .grace_inclusion_count = selection_metrics.grace_inclusion_count,
            .skipped_learner_count = selection_metrics.skipped_learner_count,
            .quorum_wait_ms = selection_metrics.quorum_wait_ms,
            .grace_window_ms = selection_metrics.grace_window_ms,
            .compressed_values = compressed_values,
            .compression_max_abs_error = compression_max_abs_error,
        };
    }

    pub fn masterTensor(self: *Self, tensor_index: usize) []const f32 {
        return self.master_params[tensor_index];
    }

    pub fn checkpoint(self: *Self, allocator: Allocator) !SyncerCheckpoint {
        const master_params = try cloneParamMatrix(allocator, self.master_params);
        errdefer freeParamMatrix(allocator, master_params);
        const optimizer_velocities = try cloneParamMatrix(allocator, self.host_optimizer.velocities);
        errdefer freeParamMatrix(allocator, optimizer_velocities);
        const fragment_versions = try allocator.alloc(usize, self.fragment_states.len);
        errdefer allocator.free(fragment_versions);
        for (self.fragment_states, 0..) |state, i| {
            fragment_versions[i] = state.version;
        }
        var clock = try self.vector_clock.clone(allocator);
        errdefer clock.deinit();
        var tape = try self.event_tape.snapshot(allocator);
        errdefer tape.deinit(allocator);

        return .{
            .allocator = allocator,
            .global_step = self.global_step,
            .master_params = master_params,
            .optimizer_velocities = optimizer_velocities,
            .fragment_versions = fragment_versions,
            .vector_clock = clock,
            .event_tape = tape,
            .num_fragments = self.config.num_fragments,
            .sync_interval_h = self.config.sync_interval_h,
            .fragment_strategy = self.config.fragment_strategy,
            .merge_strategy = self.config.merge_strategy,
            .event_tape_cursor = self.event_tape.replay_cursor,
        };
    }

    pub fn restoreCheckpoint(self: *Self, checkpoint_value: SyncerCheckpoint) !void {
        try copyParamMatrix(self.master_params, checkpoint_value.master_params);
        try copyParamMatrix(self.host_optimizer.velocities, checkpoint_value.optimizer_velocities);
        if (checkpoint_value.fragment_versions.len != self.fragment_states.len) return error.FragmentCountMismatch;
        for (checkpoint_value.fragment_versions, 0..) |version, i| {
            self.fragment_states[i].version = version;
        }
        self.global_step = checkpoint_value.global_step;

        self.vector_clock.deinit();
        self.vector_clock = try checkpoint_value.vector_clock.clone(self.allocator);
        try self.event_tape.restoreSnapshot(checkpoint_value.event_tape);
    }

    fn strategyForTensor(self: *Self, range: TensorRange) MergeStrategy {
        const is_embedding = self.isEmbeddingTensor(range);
        return switch (self.config.merge_strategy) {
            .weighted_average => .weighted_average,
            .rda => .rda,
            .avg_embedding_rda_model => if (is_embedding) .weighted_average else .rda,
            .rda_embedding_avg_model => if (is_embedding) .rda else .weighted_average,
        };
    }

    fn isEmbeddingTensor(self: *Self, range: TensorRange) bool {
        if (range.role == .embedding) return true;
        for (self.embedding_tensor_index_overrides) |tensor_index| {
            if (tensor_index == range.tensor_index) return true;
        }
        return false;
    }
};

pub fn tensorSpecsFromShapes(allocator: Allocator, parameter_shapes: []const []const i64, bytes_per_value: usize) ![]TensorSpec {
    var specs = try allocator.alloc(TensorSpec, parameter_shapes.len);
    errdefer allocator.free(specs);

    for (parameter_shapes, 0..) |shape, i| {
        var value_count: usize = 1;
        for (shape) |dim| {
            if (dim <= 0) return error.InvalidTensorShape;
            value_count = try std.math.mul(usize, value_count, @intCast(dim));
        }
        specs[i] = try TensorSpec.fromElementCount(i, value_count, bytes_per_value);
    }

    return specs;
}

fn validateTensorSpecsAgainstMaster(tensor_specs: []const TensorSpec, initial_master_params: []const []const f32) !void {
    if (tensor_specs.len == 0) return error.NoTensorSpecs;

    for (tensor_specs) |spec| {
        if (spec.index >= initial_master_params.len) return error.InvalidTensorIndex;
        if (initial_master_params[spec.index].len != spec.value_count) return error.FragmentLengthMismatch;
    }
}

fn validateEmbeddingTensorOverrides(indices: []const usize, tensor_count: usize) !void {
    for (indices, 0..) |index, i| {
        if (index >= tensor_count) return error.InvalidEmbeddingTensorIndex;
        for (indices[0..i]) |previous| {
            if (previous == index) return error.DuplicateEmbeddingTensorIndex;
        }
    }
}

fn findTensorUpdate(updates: []const TensorUpdate, tensor_index: usize) ?TensorUpdate {
    for (updates) |update| {
        if (update.tensor_index == tensor_index) return update;
    }
    return null;
}

fn findLearnerUpdate(updates: []const LearnerFragmentUpdate, learner_id: NodeId) ?LearnerFragmentUpdate {
    for (updates) |update| {
        if (update.learner_id == learner_id) return update;
    }
    return null;
}

fn buildParticipantRecords(allocator: Allocator, learner_updates: []const LearnerFragmentUpdate) ![]event_tape_mod.ParticipantRecord {
    const records = try allocator.alloc(event_tape_mod.ParticipantRecord, learner_updates.len);
    errdefer allocator.free(records);
    for (learner_updates, 0..) |update, i| {
        records[i] = .{
            .learner_id = update.learner_id,
            .learner_step = update.learner_step,
            .steps_since_fragment_update = update.steps_since_fragment_update,
            .tokens_since_fragment_update = update.tokens_since_fragment_update,
            .merge_weight = update.mergeWeight(),
        };
    }
    return records;
}

fn cloneParamMatrix(allocator: Allocator, params: []const []const f32) ![][]f32 {
    const clone = try allocator.alloc([]f32, params.len);
    var initialized: usize = 0;
    errdefer {
        for (clone[0..initialized]) |param| allocator.free(param);
        allocator.free(clone);
    }
    for (params, 0..) |param, i| {
        clone[i] = try allocator.dupe(f32, param);
        initialized += 1;
    }
    return clone;
}

fn freeParamMatrix(allocator: Allocator, params: [][]f32) void {
    for (params) |param| allocator.free(param);
    allocator.free(params);
}

fn copyParamMatrix(dst: [][]f32, src: []const []const f32) !void {
    if (dst.len != src.len) return error.ParameterCountMismatch;
    for (dst, src) |dst_param, src_param| {
        if (dst_param.len != src_param.len) return error.ParameterShapeMismatch;
        @memcpy(dst_param, src_param);
    }
}

fn earliestUpdateForLearner(updates: []const LearnerFragmentUpdate, learner_id: NodeId) ?LearnerFragmentUpdate {
    var earliest: ?LearnerFragmentUpdate = null;
    for (updates) |update| {
        if (update.learner_id != learner_id) continue;
        if (earliest == null or update.arrival_ms < earliest.?.arrival_ms) {
            earliest = update;
        }
    }
    return earliest;
}

fn learnerArrivalLessThan(_: void, lhs: LearnerFragmentUpdate, rhs: LearnerFragmentUpdate) bool {
    if (lhs.arrival_ms == rhs.arrival_ms) return lhs.learner_id < rhs.learner_id;
    return lhs.arrival_ms < rhs.arrival_ms;
}

fn saturatingAddU64(lhs: u64, rhs: u64) u64 {
    const result = @addWithOverflow(lhs, rhs);
    return if (result[1] != 0) std.math.maxInt(u64) else result[0];
}

fn updateEma(previous: f64, sample: f64, alpha: f64) f64 {
    if (!std.math.isFinite(sample)) return previous;
    if (previous <= 0.0 or !std.math.isFinite(previous)) return sample;
    return alpha * sample + (1.0 - alpha) * previous;
}

test "syncer schedules one fragment per step when P equals H" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{
        try TensorSpec.fromElementCount(0, 1, 4),
        try TensorSpec.fromElementCount(1, 1, 4),
        try TensorSpec.fromElementCount(2, 1, 4),
    };
    const master_0 = [_]f32{1.0};
    const master_1 = [_]f32{2.0};
    const master_2 = [_]f32{3.0};
    const master = [_][]const f32{ &master_0, &master_1, &master_2 };
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 3,
        .sync_interval_h = 3,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    try std.testing.expectEqual(@as(?usize, 1), syncer.scheduledFragmentId(1));
    try std.testing.expectEqual(@as(?usize, 2), syncer.scheduledFragmentId(2));
    try std.testing.expectEqual(@as(?usize, 0), syncer.scheduledFragmentId(3));
}

test "syncer direct merge with full quorum matches synchronous averaging" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{
        try TensorSpec.fromElementCount(0, 2, 4),
        try TensorSpec.fromElementCount(1, 1, 4),
        try TensorSpec.fromElementCount(2, 1, 4),
    };
    const master_0 = [_]f32{ 10.0, 20.0 };
    const master_1 = [_]f32{100.0};
    const master_2 = [_]f32{30.0};
    const master = [_][]const f32{ &master_0, &master_1, &master_2 };
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 2,
        .sync_interval_h = 2,
        .min_quorum = 2,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_a_0 = [_]f32{ 8.0, 18.0 };
    const learner_a_2 = [_]f32{25.0};
    const learner_b_0 = [_]f32{ 6.0, 22.0 };
    const learner_b_2 = [_]f32{29.0};
    const learner_a_tensors = [_]TensorUpdate{
        .{ .tensor_index = 0, .values = &learner_a_0 },
        .{ .tensor_index = 2, .values = &learner_a_2 },
    };
    const learner_b_tensors = [_]TensorUpdate{
        .{ .tensor_index = 0, .values = &learner_b_0 },
        .{ .tensor_index = 2, .values = &learner_b_2 },
    };
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 4, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_a_tensors },
        .{ .learner_id = 2, .learner_step = 4, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_b_tensors },
    };

    const result = try syncer.applyScheduledStep(2, &learners);

    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 0), result.?.fragment_id);
    try std.testing.expectEqual(@as(usize, 2), result.?.participant_count);
    try std.testing.expectEqual(@as(usize, 1), result.?.version);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), syncer.masterTensor(0)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), syncer.masterTensor(0)[1], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), syncer.masterTensor(1)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 27.0), syncer.masterTensor(2)[0], 0.000001);
}

test "syncer enforces minimum quorum" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{10.0};
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 2,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_tensor = [_]f32{8.0};
    const learner_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_tensor }};
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_tensors },
    };

    try std.testing.expectError(error.QuorumNotMet, syncer.applyFragmentStep(1, 0, &learners));
}

test "syncer applies token weighted merge" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{10.0};
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_a_0 = [_]f32{8.0};
    const learner_b_0 = [_]f32{6.0};
    const learner_a_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_a_0 }};
    const learner_b_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_b_0 }};
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_a_tensors },
        .{ .learner_id = 2, .learner_step = 1, .steps_since_fragment_update = 2, .tokens_since_fragment_update = 10, .tensors = &learner_b_tensors },
    };

    _ = try syncer.applyScheduledStep(1, &learners);

    try std.testing.expectApproxEqAbs(@as(f32, 7.3333335), syncer.masterTensor(0)[0], 0.000001);
}

test "paper profile nanochat-shaped decoupled local smoke" {
    const allocator = std.testing.allocator;

    var embedding_spec = try TensorSpec.fromElementCount(0, 8, 4);
    embedding_spec.role = .embedding;
    const tensor_specs = [_]TensorSpec{
        embedding_spec,
        try TensorSpec.fromElementCount(1, 6, 4),
        try TensorSpec.fromElementCount(2, 4, 4),
        try TensorSpec.fromElementCount(3, 5, 4),
    };
    const master_0 = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const master_1 = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const master_2 = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const master_3 = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const master = [_][]const f32{ &master_0, &master_1, &master_2, &master_3 };

    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 4,
        .sync_interval_h = 4,
        .overlap_tau = 2,
        .min_quorum = 1,
        .fragment_strategy = .balanced_tensor,
        .merge_strategy = .avg_embedding_rda_model,
        .outer_gradient_compression = .none,
        .learner_alpha = 0.0,
        .grace_window_ms = 5,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    try std.testing.expectEqual(@as(?usize, 1), syncer.scheduledFragmentId(1));
    try std.testing.expectEqual(@as(?usize, 2), syncer.scheduledFragmentId(2));
    try std.testing.expectEqual(@as(?usize, 3), syncer.scheduledFragmentId(3));
    try std.testing.expectEqual(@as(?usize, 0), syncer.scheduledFragmentId(4));
    try std.testing.expectEqual(@as(?usize, 0), syncer.fragment_plan.fragmentForTensor(0));
    try std.testing.expectEqual(@as(?usize, 1), syncer.fragment_plan.fragmentForTensor(1));
    try std.testing.expectEqual(@as(?usize, 3), syncer.fragment_plan.fragmentForTensor(2));
    try std.testing.expectEqual(@as(?usize, 2), syncer.fragment_plan.fragmentForTensor(3));

    var learner_1 = try decoupled_learner.DecoupledLearnerState.init(allocator, .{
        .num_fragments = 4,
        .sync_interval_h = 4,
        .overlap_tau = 2,
        .learner_alpha = 0.0,
        .max_syncer_steps = 8,
        .local_node_id = 1,
    });
    defer learner_1.deinit();
    var learner_2 = try decoupled_learner.DecoupledLearnerState.init(allocator, .{
        .num_fragments = 4,
        .sync_interval_h = 4,
        .overlap_tau = 2,
        .learner_alpha = 0.0,
        .max_syncer_steps = 8,
        .local_node_id = 2,
    });
    defer learner_2.deinit();
    var learner_3 = try decoupled_learner.DecoupledLearnerState.init(allocator, .{
        .num_fragments = 4,
        .sync_interval_h = 4,
        .overlap_tau = 2,
        .learner_alpha = 0.0,
        .max_syncer_steps = 8,
        .local_node_id = 3,
    });
    defer learner_3.deinit();

    try learner_1.recordLocalStep(10);
    try learner_2.recordLocalStep(10);
    try learner_3.recordLocalStep(10);

    const l1_tensor_1 = [_]f32{ -1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const l2_tensor_1 = [_]f32{ 0.0, -1.0, 0.0, 0.0, 0.0, 0.0 };
    const l3_tensor_1 = [_]f32{ -100.0, -100.0, -100.0, -100.0, -100.0, -100.0 };
    const l1_step1_tensors = [_]TensorUpdate{.{ .tensor_index = 1, .values = &l1_tensor_1 }};
    const l2_step1_tensors = [_]TensorUpdate{.{ .tensor_index = 1, .values = &l2_tensor_1 }};
    const l3_step1_tensors = [_]TensorUpdate{.{ .tensor_index = 1, .values = &l3_tensor_1 }};
    const step1_updates = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = learner_1.learner_step, .steps_since_fragment_update = learner_1.steps_since_update[1], .tokens_since_fragment_update = learner_1.tokens_since_update[1], .arrival_ms = 0, .tensors = &l1_step1_tensors },
        .{ .learner_id = 2, .learner_step = learner_2.learner_step, .steps_since_fragment_update = learner_2.steps_since_update[1], .tokens_since_fragment_update = learner_2.tokens_since_update[1], .arrival_ms = 4, .tensors = &l2_step1_tensors },
        .{ .learner_id = 3, .learner_step = learner_3.learner_step, .steps_since_fragment_update = learner_3.steps_since_update[1], .tokens_since_fragment_update = learner_3.tokens_since_update[1], .arrival_ms = 9, .tensors = &l3_step1_tensors },
    };
    const all_active = [_]LearnerAvailability{
        .{ .learner_id = 1, .status = .active },
        .{ .learner_id = 2, .status = .active },
        .{ .learner_id = 3, .status = .active },
    };

    const step1 = try syncer.applyFragmentStepWithAvailability(1, 1, &step1_updates, &all_active);
    try std.testing.expectEqual(@as(usize, 2), step1.participant_count);
    try std.testing.expectEqual(@as(usize, 1), step1.grace_inclusion_count);
    try std.testing.expectEqual(@as(usize, 1), step1.skipped_learner_count);
    try std.testing.expectEqual(@as(usize, 0), step1.compressed_values);
    try std.testing.expectApproxEqAbs(@as(f32, -0.70710677), syncer.masterTensor(1)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.70710677), syncer.masterTensor(1)[1], 0.000001);
    try std.testing.expectEqual(@as(usize, 1), syncer.event_tape.entries.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), syncer.event_tape.entries.items[0].participants[0].merge_weight, 0.000001);

    try learner_1.applySyncerFragment(1, 1);
    try learner_2.applySyncerFragment(1, 1);
    try learner_3.applySyncerFragment(1, 1);
    try std.testing.expectEqual(@as(usize, 0), learner_1.steps_since_update[1]);
    try std.testing.expectEqual(@as(usize, 1), learner_1.steps_since_update[0]);
    try std.testing.expectEqual(@as(vector_clock_mod.Counter, 1), learner_1.vector_clock.counter(1));
    try std.testing.expectEqual(@as(vector_clock_mod.Counter, 1), learner_1.vector_clock.counter(0));

    try learner_1.recordLocalStep(10);
    try learner_2.recordLocalStep(10);
    try learner_3.recordLocalStep(10);

    const l1_tensor_3 = [_]f32{ -2.0, 0.0, 0.0, 0.0, 0.0 };
    const l1_step2_tensors = [_]TensorUpdate{.{ .tensor_index = 3, .values = &l1_tensor_3 }};
    const step2_updates = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = learner_1.learner_step, .steps_since_fragment_update = learner_1.steps_since_update[2], .tokens_since_fragment_update = learner_1.tokens_since_update[2], .arrival_ms = 0, .tensors = &l1_step2_tensors },
    };
    const degraded = [_]LearnerAvailability{
        .{ .learner_id = 1, .status = .active },
        .{ .learner_id = 2, .status = .failed },
        .{ .learner_id = 3, .status = .disconnected },
    };
    const step2 = try syncer.applyFragmentStepWithAvailability(2, 2, &step2_updates, &degraded);
    try std.testing.expectEqual(@as(usize, 1), step2.participant_count);
    try std.testing.expectEqual(@as(usize, 2), step2.skipped_learner_count);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), syncer.masterTensor(3)[0], 0.000001);

    try learner_1.applySyncerFragment(2, 2);
    try std.testing.expectEqual(@as(usize, 0), learner_1.steps_since_update[2]);
    var checkpoint_value = try syncer.checkpoint(allocator);
    defer checkpoint_value.deinit();

    const l1_tensor_2 = [_]f32{ -3.0, 0.0, 0.0, 0.0 };
    const l1_step3_tensors = [_]TensorUpdate{.{ .tensor_index = 2, .values = &l1_tensor_2 }};
    const step3_updates = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = learner_1.learner_step, .steps_since_fragment_update = @max(learner_1.steps_since_update[3], 1), .tokens_since_fragment_update = @max(learner_1.tokens_since_update[3], 10), .arrival_ms = 0, .tensors = &l1_step3_tensors },
    };
    _ = try syncer.applyFragmentStepWithAvailability(3, 3, &step3_updates, &.{.{ .learner_id = 1, .status = .active }});
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), syncer.masterTensor(2)[0], 0.000001);

    try syncer.restoreCheckpoint(checkpoint_value);
    try std.testing.expectEqual(@as(usize, 2), syncer.global_step);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), syncer.masterTensor(2)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), syncer.masterTensor(3)[0], 0.000001);
    try std.testing.expectEqual(@as(usize, 2), syncer.event_tape.entries.items.len);
    try std.testing.expectEqual(@as(vector_clock_mod.Counter, 2), syncer.vector_clock.counter(0));

    syncer.event_tape.resetReplay();
    const replay_candidates = [_]LearnerFragmentUpdate{ step1_updates[2], step1_updates[1], step1_updates[0] };
    var replay = try syncer.selectReplayLearnerUpdates(1, 1, &replay_candidates);
    defer replay.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), replay.updates.len);
    try std.testing.expectEqual(@as(NodeId, 1), replay.updates[0].learner_id);
    try std.testing.expectEqual(@as(NodeId, 2), replay.updates[1].learner_id);

    const l1_embedding = [_]f32{ -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const l2_embedding = [_]f32{ 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const l3_embedding = [_]f32{ -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0 };
    const l1_step4_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &l1_embedding }};
    const l2_step4_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &l2_embedding }};
    const l3_step4_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &l3_embedding }};
    const step4_updates = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 3, .steps_since_fragment_update = 3, .tokens_since_fragment_update = 30, .arrival_ms = 0, .tensors = &l1_step4_tensors },
        .{ .learner_id = 2, .learner_step = 3, .steps_since_fragment_update = 3, .tokens_since_fragment_update = 30, .arrival_ms = 2, .tensors = &l2_step4_tensors },
        .{ .learner_id = 3, .learner_step = 3, .steps_since_fragment_update = 3, .tokens_since_fragment_update = 30, .arrival_ms = 20, .tensors = &l3_step4_tensors },
    };
    const step4 = try syncer.applyFragmentStepWithAvailability(4, 0, &step4_updates, &all_active);
    try std.testing.expectEqual(@as(usize, 2), step4.participant_count);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), syncer.masterTensor(0)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), syncer.masterTensor(0)[1], 0.000001);
}

test "syncer records event tape participants and replay selects the same learners" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{10.0};
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_a_0 = [_]f32{8.0};
    const learner_b_0 = [_]f32{6.0};
    const learner_c_0 = [_]f32{4.0};
    const learner_a_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_a_0 }};
    const learner_b_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_b_0 }};
    const learner_c_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_c_0 }};
    const accepted = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 4, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_a_tensors },
        .{ .learner_id = 2, .learner_step = 3, .steps_since_fragment_update = 2, .tokens_since_fragment_update = 10, .tensors = &learner_b_tensors },
    };
    const replay_candidates = [_]LearnerFragmentUpdate{
        accepted[0],
        accepted[1],
        .{ .learner_id = 3, .learner_step = 5, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_c_tensors },
    };

    _ = try syncer.applyFragmentStep(1, 0, &accepted);

    try std.testing.expectEqual(@as(usize, 1), syncer.event_tape.entries.items.len);
    const entry = syncer.event_tape.entries.items[0];
    try std.testing.expectEqual(event_tape_mod.EventKind.fragment_sync, entry.kind);
    try std.testing.expectEqual(@as(usize, 2), entry.participants.len);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), entry.participants[1].merge_weight, 0.000001);
    try std.testing.expectEqual(@as(vector_clock_mod.Counter, 1), syncer.vector_clock.counter(0));

    var replay = try syncer.selectReplayLearnerUpdates(1, 0, &replay_candidates);
    defer replay.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), replay.updates.len);
    try std.testing.expectEqual(@as(NodeId, 1), replay.updates[0].learner_id);
    try std.testing.expectEqual(@as(NodeId, 2), replay.updates[1].learner_id);
    try std.testing.expectEqual(@as(usize, 1), replay.metrics.skipped_learner_count);
}

test "syncer checkpoint restores master parameters optimizer state and event tape" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{10.0};
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
        .outer_momentum = 0.5,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_a_0 = [_]f32{8.0};
    const learner_b_0 = [_]f32{5.0};
    const learner_a_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_a_0 }};
    const learner_b_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_b_0 }};
    const first = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_a_tensors },
    };
    const second = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 2, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_b_tensors },
    };

    _ = try syncer.applyFragmentStep(1, 0, &first);
    var checkpoint_value = try syncer.checkpoint(allocator);
    defer checkpoint_value.deinit();
    _ = try syncer.applyFragmentStep(2, 0, &second);

    try syncer.restoreCheckpoint(checkpoint_value);

    try std.testing.expectEqual(@as(usize, 1), syncer.global_step);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), syncer.masterTensor(0)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), syncer.host_optimizer.velocities[0][0], 0.000001);
    try std.testing.expectEqual(@as(usize, 1), syncer.fragment_states[0].version);
    try std.testing.expectEqual(@as(usize, 1), syncer.event_tape.entries.items.len);
    try std.testing.expectEqual(@as(vector_clock_mod.Counter, 1), syncer.vector_clock.counter(0));
}

test "embedding tensor index override selects direct average for paper merge profile" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{
        try TensorSpec.fromElementCount(0, 2, 4),
        try TensorSpec.fromElementCount(1, 2, 4),
    };
    const master_0 = [_]f32{ 0.0, 0.0 };
    const master_1 = [_]f32{ 0.0, 0.0 };
    const master = [_][]const f32{ &master_0, &master_1 };
    const embedding_indices = [_]usize{0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .avg_embedding_rda_model,
        .embedding_tensor_indices = &embedding_indices,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_a_0 = [_]f32{ -1.0, 0.0 };
    const learner_a_1 = [_]f32{ -1.0, 0.0 };
    const learner_b_0 = [_]f32{ 0.0, -1.0 };
    const learner_b_1 = [_]f32{ 0.0, -1.0 };
    const learner_a_tensors = [_]TensorUpdate{
        .{ .tensor_index = 0, .values = &learner_a_0 },
        .{ .tensor_index = 1, .values = &learner_a_1 },
    };
    const learner_b_tensors = [_]TensorUpdate{
        .{ .tensor_index = 0, .values = &learner_b_0 },
        .{ .tensor_index = 1, .values = &learner_b_1 },
    };
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_a_tensors },
        .{ .learner_id = 2, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_b_tensors },
    };

    _ = try syncer.applyFragmentStep(1, 0, &learners);

    try std.testing.expectApproxEqAbs(@as(f32, -0.5), syncer.masterTensor(0)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), syncer.masterTensor(0)[1], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.70710677), syncer.masterTensor(1)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.70710677), syncer.masterTensor(1)[1], 0.000001);
}

test "syncer applies int4 outer gradient compression only when explicitly enabled" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 2, 4)};
    const master_0 = [_]f32{ 0.0, 0.0 };
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
        .outer_gradient_compression = .int4,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_0 = [_]f32{ -1.0, -0.5 };
    const learner_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_0 }};
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_tensors },
    };

    const result = try syncer.applyFragmentStep(1, 0, &learners);

    try std.testing.expectEqual(@as(usize, 2), result.compressed_values);
    try std.testing.expect(result.compression_max_abs_error <= (1.0 / 7.0) / 2.0 + 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), syncer.masterTensor(0)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0 / 7.0), syncer.masterTensor(0)[1], 0.000001);
}

test "syncer rejects duplicate embedding tensor overrides" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{0.0};
    const master = [_][]const f32{&master_0};
    const embedding_indices = [_]usize{ 0, 0 };

    try std.testing.expectError(error.DuplicateEmbeddingTensorIndex, DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .embedding_tensor_indices = &embedding_indices,
    }, &tensor_specs, &master));
}

test "fixed grace includes delayed active learners and skips unavailable learners" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{10.0};
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
        .grace_window_ms = 5,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_a_0 = [_]f32{8.0};
    const learner_b_0 = [_]f32{4.0};
    const failed_learner_0 = [_]f32{-100.0};
    const learner_a_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_a_0 }};
    const learner_b_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_b_0 }};
    const failed_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &failed_learner_0 }};
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .arrival_ms = 0, .tensors = &learner_a_tensors },
        .{ .learner_id = 2, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .arrival_ms = 4, .tensors = &learner_b_tensors },
        .{ .learner_id = 3, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .arrival_ms = 0, .tensors = &failed_tensors },
    };
    const availability = [_]LearnerAvailability{
        .{ .learner_id = 1, .status = .active },
        .{ .learner_id = 2, .status = .active },
        .{ .learner_id = 3, .status = .failed },
    };

    const result = try syncer.applyFragmentStepWithAvailability(1, 0, &learners, &availability);

    try std.testing.expectEqual(@as(usize, 2), result.participant_count);
    try std.testing.expectEqual(@as(usize, 1), result.grace_inclusion_count);
    try std.testing.expectEqual(@as(usize, 1), result.skipped_learner_count);
    try std.testing.expectEqual(@as(u64, 0), result.quorum_wait_ms);
    try std.testing.expectEqual(@as(u64, 5), result.grace_window_ms);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), syncer.masterTensor(0)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 200.0), result.total_weight, 0.000001);
    try std.testing.expectEqual(@as(usize, 2), syncer.timing_metrics.last_participant_count);
    try std.testing.expectEqual(@as(usize, 1), syncer.timing_metrics.last_grace_inclusion_count);
    try std.testing.expectEqual(@as(usize, 1), syncer.timing_metrics.last_skipped_learner_count);
}

test "fixed grace skips active learners that arrive after the deadline" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{10.0};
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
        .grace_window_ms = 5,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_a_0 = [_]f32{8.0};
    const learner_b_0 = [_]f32{4.0};
    const learner_a_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_a_0 }};
    const learner_b_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_b_0 }};
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .arrival_ms = 0, .tensors = &learner_a_tensors },
        .{ .learner_id = 2, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .arrival_ms = 9, .tensors = &learner_b_tensors },
    };
    const availability = [_]LearnerAvailability{
        .{ .learner_id = 1, .status = .active },
        .{ .learner_id = 2, .status = .active },
    };

    const result = try syncer.applyFragmentStepWithAvailability(1, 0, &learners, &availability);

    try std.testing.expectEqual(@as(usize, 1), result.participant_count);
    try std.testing.expectEqual(@as(usize, 0), result.grace_inclusion_count);
    try std.testing.expectEqual(@as(usize, 1), result.skipped_learner_count);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), syncer.masterTensor(0)[0], 0.000001);
}

test "adaptive grace uses overlap slack and max step clamp" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{10.0};
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .overlap_tau = 2,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
        .grace_gamma = 0.5,
        .max_grace_steps = 1,
        .adaptive_grace_enabled = true,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    syncer.recordStepDuration(10.0);
    syncer.recordQuorumWaitDuration(2.0);
    syncer.recordSyncSendDuration(1.0);

    try std.testing.expectEqual(@as(u64, 8), syncer.currentGraceWindowMs());

    const learner_a_0 = [_]f32{8.0};
    const learner_b_0 = [_]f32{4.0};
    const learner_c_0 = [_]f32{0.0};
    const learner_a_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_a_0 }};
    const learner_b_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_b_0 }};
    const learner_c_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_c_0 }};
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .arrival_ms = 0, .tensors = &learner_a_tensors },
        .{ .learner_id = 2, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .arrival_ms = 8, .tensors = &learner_b_tensors },
        .{ .learner_id = 3, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .arrival_ms = 9, .tensors = &learner_c_tensors },
    };
    const availability = [_]LearnerAvailability{
        .{ .learner_id = 1, .status = .active },
        .{ .learner_id = 2, .status = .active },
        .{ .learner_id = 3, .status = .active },
    };

    const result = try syncer.applyFragmentStepWithAvailability(1, 0, &learners, &availability);

    try std.testing.expectEqual(@as(usize, 2), result.participant_count);
    try std.testing.expectEqual(@as(usize, 1), result.grace_inclusion_count);
    try std.testing.expectEqual(@as(usize, 1), result.skipped_learner_count);
    try std.testing.expectEqual(@as(u64, 8), result.grace_window_ms);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), syncer.masterTensor(0)[0], 0.000001);
}

test "adaptive grace clamps negative slack to zero" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 1, 4)};
    const master_0 = [_]f32{10.0};
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .overlap_tau = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .weighted_average,
        .adaptive_grace_enabled = true,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    syncer.recordStepDuration(4.0);
    syncer.recordQuorumWaitDuration(5.0);
    syncer.recordSyncSendDuration(1.0);

    try std.testing.expectEqual(@as(u64, 0), syncer.currentGraceWindowMs());
}

test "syncer RDA merge updates with norm-stable orthogonal direction" {
    const allocator = std.testing.allocator;
    const tensor_specs = [_]TensorSpec{try TensorSpec.fromElementCount(0, 2, 4)};
    const master_0 = [_]f32{ 0.0, 0.0 };
    const master = [_][]const f32{&master_0};
    var syncer = try DecoupledDiLoCo.initWithMaster(allocator, .{
        .num_fragments = 1,
        .sync_interval_h = 1,
        .min_quorum = 1,
        .fragment_strategy = .strided,
        .merge_strategy = .rda,
    }, &tensor_specs, &master);
    defer syncer.deinit();

    const learner_a_0 = [_]f32{ -1.0, 0.0 };
    const learner_b_0 = [_]f32{ 0.0, -1.0 };
    const learner_a_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_a_0 }};
    const learner_b_tensors = [_]TensorUpdate{.{ .tensor_index = 0, .values = &learner_b_0 }};
    const learners = [_]LearnerFragmentUpdate{
        .{ .learner_id = 1, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_a_tensors },
        .{ .learner_id = 2, .learner_step = 1, .steps_since_fragment_update = 1, .tokens_since_fragment_update = 10, .tensors = &learner_b_tensors },
    };

    _ = try syncer.applyScheduledStep(1, &learners);

    try std.testing.expectApproxEqAbs(@as(f32, -0.70710677), syncer.masterTensor(0)[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.70710677), syncer.masterTensor(0)[1], 0.000001);
}

test "tensor specs can be built from parameter shapes" {
    const allocator = std.testing.allocator;
    const shape_0 = [_]i64{ 2, 3 };
    const shape_1 = [_]i64{4};
    const shapes = [_][]const i64{ &shape_0, &shape_1 };

    const specs = try tensorSpecsFromShapes(allocator, &shapes, 4);
    defer allocator.free(specs);

    try std.testing.expectEqual(@as(usize, 0), specs[0].index);
    try std.testing.expectEqual(@as(usize, 6), specs[0].value_count);
    try std.testing.expectEqual(@as(usize, 24), specs[0].byte_count);
    try std.testing.expectEqual(@as(usize, 1), specs[1].index);
    try std.testing.expectEqual(@as(usize, 4), specs[1].value_count);
    try std.testing.expectEqual(@as(usize, 16), specs[1].byte_count);
}
