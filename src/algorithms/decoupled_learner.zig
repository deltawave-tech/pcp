const std = @import("std");
const decoupled_fields = @import("../protocol/decoupled_fields.zig");
const json_util = @import("../protocol/json_util.zig");
const vector_clock_mod = @import("vector_clock.zig");

const Allocator = std.mem.Allocator;
const NodeId = vector_clock_mod.NodeId;

pub const PayloadField = decoupled_fields.DecoupledDiLoCoField;

pub const DecoupledLearnerConfig = struct {
    num_fragments: usize,
    sync_interval_h: usize,
    overlap_tau: usize = 2,
    learner_alpha: f32 = 0.0,
    max_syncer_steps: usize,
    local_node_id: ?NodeId = null,

    pub fn validate(self: DecoupledLearnerConfig) !void {
        if (self.num_fragments == 0) return error.InvalidFragmentCount;
        if (self.sync_interval_h == 0) return error.InvalidSyncInterval;
        if (self.overlap_tau == 0) return error.InvalidOverlapTau;
        if (self.learner_alpha < 0.0 or self.learner_alpha > 1.0) return error.InvalidLearnerAlpha;
    }
};

pub const MetadataSnapshot = struct {
    learner_step: usize,
    global_step: usize,
    steps_since_update: []const usize,
    tokens_since_update: []const usize,
    vector_clock: []const vector_clock_mod.Entry,
};

pub const LearnerCheckpoint = struct {
    allocator: Allocator,
    learner_step: usize,
    observed_global_step: usize,
    steps_since_update: []usize,
    tokens_since_update: []usize,
    vector_clock: vector_clock_mod.VectorClock,

    pub fn deinit(self: *LearnerCheckpoint) void {
        self.allocator.free(self.steps_since_update);
        self.allocator.free(self.tokens_since_update);
        self.vector_clock.deinit();
        self.* = undefined;
    }
};

pub const DecoupledLearnerState = struct {
    allocator: Allocator,
    config: DecoupledLearnerConfig,
    learner_step: usize = 0,
    observed_global_step: usize = 0,
    steps_since_update: []usize,
    tokens_since_update: []usize,
    vector_clock: vector_clock_mod.VectorClock,

    pub fn init(allocator: Allocator, config: DecoupledLearnerConfig) !DecoupledLearnerState {
        try config.validate();

        const steps = try allocator.alloc(usize, config.num_fragments);
        errdefer allocator.free(steps);
        const tokens = try allocator.alloc(usize, config.num_fragments);
        errdefer allocator.free(tokens);

        var vector_clock = vector_clock_mod.VectorClock.init(allocator);
        errdefer vector_clock.deinit();
        if (config.local_node_id) |node_id| {
            try vector_clock.observe(node_id, 0);
        }

        @memset(steps, 0);
        @memset(tokens, 0);

        return .{
            .allocator = allocator,
            .config = config,
            .steps_since_update = steps,
            .tokens_since_update = tokens,
            .vector_clock = vector_clock,
        };
    }

    pub fn deinit(self: *DecoupledLearnerState) void {
        self.allocator.free(self.steps_since_update);
        self.allocator.free(self.tokens_since_update);
        self.vector_clock.deinit();
        self.* = undefined;
    }

    pub fn recordLocalStep(self: *DecoupledLearnerState, tokens_processed: usize) !void {
        self.learner_step = try std.math.add(usize, self.learner_step, 1);
        if (self.config.local_node_id) |node_id| {
            _ = try self.vector_clock.tick(node_id);
        }
        for (self.steps_since_update) |*value| value.* = try std.math.add(usize, value.*, 1);
        for (self.tokens_since_update) |*value| value.* = try std.math.add(usize, value.*, tokens_processed);
    }

    pub fn applySyncerFragment(self: *DecoupledLearnerState, fragment_id: usize, syncer_step: usize) !void {
        if (fragment_id >= self.config.num_fragments) return error.InvalidFragmentId;
        self.steps_since_update[fragment_id] = 0;
        self.tokens_since_update[fragment_id] = 0;
        self.observed_global_step = @max(self.observed_global_step, syncer_step);
        try self.vector_clock.observe(0, @intCast(syncer_step));
    }

    pub fn shouldContinue(self: DecoupledLearnerState) bool {
        if (self.config.max_syncer_steps == 0) return true;
        return self.observed_global_step < self.config.max_syncer_steps;
    }

    pub fn metadata(self: DecoupledLearnerState) MetadataSnapshot {
        return .{
            .learner_step = self.learner_step,
            .global_step = self.observed_global_step,
            .steps_since_update = self.steps_since_update,
            .tokens_since_update = self.tokens_since_update,
            .vector_clock = self.vector_clock.entries,
        };
    }

    pub fn buildMetadataPayload(self: DecoupledLearnerState, allocator: Allocator, train_examples: usize) !std.json.Value {
        var steps = std.json.Array.init(allocator);
        errdefer steps.deinit();
        for (self.steps_since_update) |value| {
            try steps.append(.{ .integer = @intCast(value) });
        }

        var tokens = std.json.Array.init(allocator);
        errdefer tokens.deinit();
        for (self.tokens_since_update) |value| {
            try tokens.append(.{ .integer = @intCast(value) });
        }

        const vector_clock = try self.vector_clock.toJsonArray(allocator);

        var payload = std.json.ObjectMap.init(allocator);
        errdefer payload.deinit();
        try payload.put(PayloadField.LEARNER_STEP, .{ .integer = @intCast(self.learner_step) });
        try payload.put(PayloadField.GLOBAL_STEP, .{ .integer = @intCast(self.observed_global_step) });
        try payload.put(PayloadField.TRAIN_EXAMPLES, .{ .integer = @intCast(train_examples) });
        try payload.put(PayloadField.STEPS_SINCE_FRAGMENT_UPDATE, .{ .array = steps });
        try payload.put(PayloadField.TOKENS_SINCE_FRAGMENT_UPDATE, .{ .array = tokens });
        try payload.put(PayloadField.VECTOR_CLOCK, vector_clock);

        return .{ .object = payload };
    }

    pub fn checkpoint(self: DecoupledLearnerState, allocator: Allocator) !LearnerCheckpoint {
        const steps = try allocator.dupe(usize, self.steps_since_update);
        errdefer allocator.free(steps);
        const tokens = try allocator.dupe(usize, self.tokens_since_update);
        errdefer allocator.free(tokens);
        var clock = try self.vector_clock.clone(allocator);
        errdefer clock.deinit();

        return .{
            .allocator = allocator,
            .learner_step = self.learner_step,
            .observed_global_step = self.observed_global_step,
            .steps_since_update = steps,
            .tokens_since_update = tokens,
            .vector_clock = clock,
        };
    }

    pub fn restoreCheckpoint(self: *DecoupledLearnerState, checkpoint_value: LearnerCheckpoint) !void {
        if (checkpoint_value.steps_since_update.len != self.steps_since_update.len) return error.FragmentCountMismatch;
        if (checkpoint_value.tokens_since_update.len != self.tokens_since_update.len) return error.FragmentCountMismatch;

        @memcpy(self.steps_since_update, checkpoint_value.steps_since_update);
        @memcpy(self.tokens_since_update, checkpoint_value.tokens_since_update);
        self.learner_step = checkpoint_value.learner_step;
        self.observed_global_step = checkpoint_value.observed_global_step;

        self.vector_clock.deinit();
        self.vector_clock = try checkpoint_value.vector_clock.clone(self.allocator);
    }
};

pub fn blendFragmentInPlace(local_values: []f32, global_values: []const f32, alpha: f32) !void {
    if (local_values.len != global_values.len) return error.FragmentLengthMismatch;
    if (alpha < 0.0 or alpha > 1.0) return error.InvalidLearnerAlpha;

    for (local_values, 0..) |*local, i| {
        local.* = alpha * local.* + (1.0 - alpha) * global_values[i];
        if (!std.math.isFinite(local.*)) local.* = 0.0;
    }
}

pub fn fragmentOffset(fragment_id: usize, sync_interval_h: usize) !usize {
    if (sync_interval_h == 0) return error.InvalidSyncInterval;
    return fragment_id % sync_interval_h;
}

test "learner counters increment every fragment on each local step" {
    const allocator = std.testing.allocator;
    var state = try DecoupledLearnerState.init(allocator, .{
        .num_fragments = 3,
        .sync_interval_h = 3,
        .max_syncer_steps = 10,
    });
    defer state.deinit();

    try state.recordLocalStep(8);
    try state.recordLocalStep(12);

    try std.testing.expectEqual(@as(usize, 2), state.learner_step);
    for (state.steps_since_update) |value| try std.testing.expectEqual(@as(usize, 2), value);
    for (state.tokens_since_update) |value| try std.testing.expectEqual(@as(usize, 20), value);
}

test "syncer fragment update resets only that fragment counters" {
    const allocator = std.testing.allocator;
    var state = try DecoupledLearnerState.init(allocator, .{
        .num_fragments = 3,
        .sync_interval_h = 3,
        .max_syncer_steps = 10,
    });
    defer state.deinit();

    try state.recordLocalStep(5);
    try state.recordLocalStep(5);
    try state.applySyncerFragment(1, 4);

    try std.testing.expectEqual(@as(usize, 2), state.steps_since_update[0]);
    try std.testing.expectEqual(@as(usize, 0), state.steps_since_update[1]);
    try std.testing.expectEqual(@as(usize, 2), state.steps_since_update[2]);
    try std.testing.expectEqual(@as(usize, 10), state.tokens_since_update[0]);
    try std.testing.expectEqual(@as(usize, 0), state.tokens_since_update[1]);
    try std.testing.expectEqual(@as(usize, 10), state.tokens_since_update[2]);
    try std.testing.expectEqual(@as(usize, 4), state.observed_global_step);
}

test "learner stop condition follows syncer global step" {
    const allocator = std.testing.allocator;
    var state = try DecoupledLearnerState.init(allocator, .{
        .num_fragments = 2,
        .sync_interval_h = 2,
        .max_syncer_steps = 3,
    });
    defer state.deinit();

    try std.testing.expect(state.shouldContinue());
    try state.applySyncerFragment(0, 2);
    try std.testing.expect(state.shouldContinue());
    try state.applySyncerFragment(1, 3);
    try std.testing.expect(!state.shouldContinue());
}

test "metadata payload contains learner step and per-fragment counters" {
    const allocator = std.testing.allocator;
    var state = try DecoupledLearnerState.init(allocator, .{
        .num_fragments = 2,
        .sync_interval_h = 2,
        .max_syncer_steps = 10,
        .local_node_id = 9,
    });
    defer state.deinit();

    try state.recordLocalStep(16);
    try state.applySyncerFragment(1, 7);

    const payload_value = try state.buildMetadataPayload(allocator, 4);
    defer json_util.deinitJsonContainers(payload_value);
    const payload = payload_value.object;

    try std.testing.expectEqual(@as(i64, 1), payload.get(PayloadField.LEARNER_STEP).?.integer);
    try std.testing.expectEqual(@as(i64, 7), payload.get(PayloadField.GLOBAL_STEP).?.integer);
    try std.testing.expectEqual(@as(i64, 4), payload.get(PayloadField.TRAIN_EXAMPLES).?.integer);
    try std.testing.expectEqual(@as(i64, 1), payload.get(PayloadField.STEPS_SINCE_FRAGMENT_UPDATE).?.array.items[0].integer);
    try std.testing.expectEqual(@as(i64, 0), payload.get(PayloadField.STEPS_SINCE_FRAGMENT_UPDATE).?.array.items[1].integer);
    try std.testing.expectEqual(@as(i64, 16), payload.get(PayloadField.TOKENS_SINCE_FRAGMENT_UPDATE).?.array.items[0].integer);
    try std.testing.expectEqual(@as(i64, 0), payload.get(PayloadField.TOKENS_SINCE_FRAGMENT_UPDATE).?.array.items[1].integer);
    try std.testing.expectEqual(@as(usize, 2), payload.get(PayloadField.VECTOR_CLOCK).?.array.items.len);
    const learner_clock = payload.get(PayloadField.VECTOR_CLOCK).?.array.items[1].object;
    try std.testing.expectEqual(@as(i64, 9), learner_clock.get("node_id").?.integer);
    try std.testing.expectEqual(@as(i64, 1), learner_clock.get("counter").?.integer);
}

test "fragment blending supports overwrite and interpolation" {
    var local = [_]f32{ 1.0, 3.0 };
    const global = [_]f32{ 5.0, 7.0 };

    try blendFragmentInPlace(&local, &global, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), local[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), local[1], 0.000001);

    const next_global = [_]f32{ 1.0, 3.0 };
    try blendFragmentInPlace(&local, &next_global, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), local[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), local[1], 0.000001);
}

test "learner checkpoint preserves counters and vector clock" {
    const allocator = std.testing.allocator;
    var state = try DecoupledLearnerState.init(allocator, .{
        .num_fragments = 2,
        .sync_interval_h = 2,
        .max_syncer_steps = 10,
        .local_node_id = 4,
    });
    defer state.deinit();

    try state.recordLocalStep(8);
    try state.applySyncerFragment(1, 3);
    var checkpoint_value = try state.checkpoint(allocator);
    defer checkpoint_value.deinit();

    try state.recordLocalStep(12);
    try state.restoreCheckpoint(checkpoint_value);

    try std.testing.expectEqual(@as(usize, 1), state.learner_step);
    try std.testing.expectEqual(@as(usize, 1), state.steps_since_update[0]);
    try std.testing.expectEqual(@as(usize, 0), state.steps_since_update[1]);
    try std.testing.expectEqual(@as(usize, 8), state.tokens_since_update[0]);
    try std.testing.expectEqual(@as(vector_clock_mod.Counter, 1), state.vector_clock.counter(4));
    try std.testing.expectEqual(@as(vector_clock_mod.Counter, 3), state.vector_clock.counter(0));
}
