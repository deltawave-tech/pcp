const std = @import("std");

const real_merge = @import("../../../algorithms/decoupled_merge.zig");

pub const max_dimensions = 8;
pub const max_updates = 8;
pub const epsilon = 0.000001;

pub const LearnerId = u16;

pub const MergeKind = enum {
    direct_average,
    token_weighted,
    rda,
};

pub const Violation = enum {
    none,
    invalid_dimensions,
    duplicate_update,
    below_quorum,
    output_outside_envelope,
    output_mismatch,
    rda_direction_mismatch,
    rda_norm_mismatch,
    incompatible_local_views,
};

pub const Window = struct {
    syncer_step: usize,
    fragment_id: usize,
    min_quorum: usize,
    grace_deadline_tick: usize,
    dimensions: usize,
};

pub const Update = struct {
    learner_id: LearnerId,
    syncer_step: usize,
    fragment_id: usize,
    arrival_tick: usize,
    values: []const i64,
    tokens_since_update: usize = 1,
    steps_since_update: usize = 1,
};

pub const MergeResult = struct {
    values: [max_dimensions]f64 = [_]f64{0.0} ** max_dimensions,
    dimensions: usize = 0,
    participant_count: usize = 0,
    non_participant_count: usize = 0,
    participant_mask: u64 = 0,
    total_weight: f64 = 0.0,
    direction_weight: f64 = 0.0,
    mean_norm: f64 = 0.0,
    direction_norm: f64 = 0.0,
    output_norm: f64 = 0.0,
    violation: Violation = .none,

    pub fn valueSlice(self: *const MergeResult) []const f64 {
        return self.values[0..self.dimensions];
    }
};

pub const LocalView = struct {
    syncer_step: usize,
    fragment_id: usize,
    participant_mask: u64,
    output: []const f64,
};

pub const ClaimedTrace = struct {
    window: Window,
    kind: MergeKind,
    updates: []const Update,
    claimed_output: []const f64,
};

pub const ShrinkResult = struct {
    retained_indices: [max_updates]usize = [_]usize{0} ** max_updates,
    retained_count: usize = 0,
    violation: Violation = .none,
};

const Participant = struct {
    update: Update,
    weight: f64,
};

pub fn compute(window: Window, kind: MergeKind, updates: []const Update) MergeResult {
    var result = MergeResult{ .dimensions = window.dimensions };
    if (window.dimensions == 0 or window.dimensions > max_dimensions) {
        result.violation = .invalid_dimensions;
        return result;
    }
    if (updates.len > max_updates) {
        result.violation = .invalid_dimensions;
        return result;
    }

    var participants: [max_updates]Participant = undefined;
    const collected = collectParticipants(window, kind, updates, &participants);
    result.participant_count = collected.participant_count;
    result.non_participant_count = collected.non_participant_count;
    result.participant_mask = collected.participant_mask;
    result.total_weight = collected.total_weight;
    result.violation = collected.violation;
    if (result.violation != .none) return result;

    if (result.participant_count < window.min_quorum) {
        result.violation = .below_quorum;
        return result;
    }

    switch (kind) {
        .direct_average, .token_weighted => {
            for (0..window.dimensions) |dimension| {
                var weighted_sum: f64 = 0.0;
                for (participants[0..result.participant_count]) |participant| {
                    weighted_sum += @as(f64, @floatFromInt(participant.update.values[dimension])) * participant.weight;
                }
                result.values[dimension] = weighted_sum / result.total_weight;
            }
        },
        .rda => computeRda(&result, participants[0..result.participant_count]),
    }

    return result;
}

pub fn checkClaimed(trace_value: ClaimedTrace) Violation {
    const result = compute(trace_value.window, trace_value.kind, trace_value.updates);
    if (result.violation != .none) return result.violation;
    if (trace_value.claimed_output.len != result.dimensions) return .invalid_dimensions;

    switch (trace_value.kind) {
        .direct_average, .token_weighted => {
            if (!claimedInsideEnvelope(trace_value.window, trace_value.updates, result.participant_mask, trace_value.claimed_output)) {
                return .output_outside_envelope;
            }
        },
        .rda => {
            const carrier_violation = checkRdaCarrier(trace_value.window, trace_value.updates, trace_value.claimed_output);
            if (carrier_violation != .none) return carrier_violation;
        },
    }

    for (result.valueSlice(), trace_value.claimed_output) |expected, actual| {
        if (!approxEq(expected, actual)) return .output_mismatch;
    }
    return .none;
}

pub fn checkRdaCarrier(window: Window, updates: []const Update, claimed_output: []const f64) Violation {
    if (claimed_output.len != window.dimensions) return .invalid_dimensions;
    if (window.dimensions == 0 or window.dimensions > max_dimensions) return .invalid_dimensions;
    if (updates.len > max_updates) return .invalid_dimensions;

    var participants: [max_updates]Participant = undefined;
    const collected = collectParticipants(window, .rda, updates, &participants);
    if (collected.violation != .none) return collected.violation;
    if (collected.participant_count < window.min_quorum) return .below_quorum;

    return checkRdaCarrierForParticipants(window.dimensions, participants[0..collected.participant_count], claimed_output);
}

pub fn checkCompatibleLocalViews(left: LocalView, right: LocalView) Violation {
    if (left.syncer_step != right.syncer_step) return .none;
    if (left.fragment_id != right.fragment_id) return .none;
    if (left.participant_mask != right.participant_mask) return .none;
    if (left.output.len != right.output.len) return .invalid_dimensions;

    for (left.output, right.output) |a, b| {
        if (!approxEq(a, b)) return .incompatible_local_views;
    }
    return .none;
}

pub fn shrinkFailingTrace(trace_value: ClaimedTrace) ShrinkResult {
    const target = checkClaimed(trace_value);
    var result = ShrinkResult{ .violation = target };
    if (target == .none) return result;
    if (trace_value.updates.len > max_updates) return result;

    var retained: [max_updates]bool = [_]bool{false} ** max_updates;
    for (trace_value.updates, 0..) |_, index| {
        retained[index] = true;
    }

    var changed = true;
    while (changed) {
        changed = false;
        for (trace_value.updates, 0..) |_, candidate| {
            if (!retained[candidate]) continue;

            retained[candidate] = false;
            var compact: [max_updates]Update = undefined;
            const count = compactRetained(trace_value.updates, retained[0..trace_value.updates.len], &compact);
            const shrunk = ClaimedTrace{
                .window = trace_value.window,
                .kind = trace_value.kind,
                .updates = compact[0..count],
                .claimed_output = trace_value.claimed_output,
            };

            if (checkClaimed(shrunk) == target) {
                changed = true;
                break;
            }

            retained[candidate] = true;
        }
    }

    result.retained_count = compactRetainedIndices(retained[0..trace_value.updates.len], &result.retained_indices);
    return result;
}

const CollectResult = struct {
    participant_count: usize = 0,
    non_participant_count: usize = 0,
    participant_mask: u64 = 0,
    total_weight: f64 = 0.0,
    violation: Violation = .none,
};

fn collectParticipants(
    window: Window,
    kind: MergeKind,
    updates: []const Update,
    out: *[max_updates]Participant,
) CollectResult {
    var result = CollectResult{};
    var seen_mask: u64 = 0;

    for (updates) |update| {
        if (!matchesWindow(window, update)) {
            result.non_participant_count += 1;
            continue;
        }
        if (update.arrival_tick > window.grace_deadline_tick) {
            result.non_participant_count += 1;
            continue;
        }
        if (update.values.len != window.dimensions) {
            result.violation = .invalid_dimensions;
            return result;
        }
        if (update.learner_id >= 64) {
            result.violation = .invalid_dimensions;
            return result;
        }

        const bit = @as(u64, 1) << @intCast(update.learner_id);
        if ((seen_mask & bit) != 0) {
            result.violation = .duplicate_update;
            return result;
        }

        const weight = updateWeight(kind, update);
        if (weight <= 0.0 or !std.math.isFinite(weight)) {
            result.non_participant_count += 1;
            continue;
        }

        seen_mask |= bit;
        out[result.participant_count] = .{ .update = update, .weight = weight };
        result.participant_count += 1;
        result.participant_mask |= bit;
        result.total_weight += weight;
    }

    return result;
}

fn claimedInsideEnvelope(
    window: Window,
    updates: []const Update,
    participant_mask: u64,
    claimed_output: []const f64,
) bool {
    if (claimed_output.len != window.dimensions) return false;

    for (0..window.dimensions) |dimension| {
        var found = false;
        var min_value: i64 = 0;
        var max_value: i64 = 0;

        for (updates) |update| {
            if (update.learner_id >= 64) continue;
            const bit = @as(u64, 1) << @intCast(update.learner_id);
            if ((participant_mask & bit) == 0) continue;

            const value = update.values[dimension];
            if (!found) {
                found = true;
                min_value = value;
                max_value = value;
            } else {
                min_value = @min(min_value, value);
                max_value = @max(max_value, value);
            }
        }

        if (!found) return false;
        const actual = claimed_output[dimension];
        if (actual < @as(f64, @floatFromInt(min_value)) - epsilon) return false;
        if (actual > @as(f64, @floatFromInt(max_value)) + epsilon) return false;
    }

    return true;
}

fn updateWeight(kind: MergeKind, update: Update) f64 {
    return switch (kind) {
        .direct_average => 1.0,
        .token_weighted, .rda => @floatCast(real_merge.tokenStepWeight(update.tokens_since_update, update.steps_since_update)),
    };
}

const RdaCarrier = struct {
    direction: [max_dimensions]f64 = [_]f64{0.0} ** max_dimensions,
    total_weight: f64 = 0.0,
    direction_weight: f64 = 0.0,
    mean_norm: f64 = 0.0,
    direction_norm: f64 = 0.0,
    degenerate: bool = false,
};

fn computeRda(result: *MergeResult, participants: []const Participant) void {
    var global_before = [_]f32{0.0} ** max_dimensions;
    var learner_values = [_][max_dimensions]f32{[_]f32{0.0} ** max_dimensions} ** max_updates;
    var weighted: [max_updates]real_merge.WeightedFragment = undefined;

    for (participants, 0..) |participant, participant_index| {
        for (0..result.dimensions) |dimension| {
            learner_values[participant_index][dimension] = -@as(f32, @floatFromInt(participant.update.values[dimension]));
        }
        weighted[participant_index] = .{
            .values = learner_values[participant_index][0..result.dimensions],
            .weight = @floatCast(participant.weight),
        };
    }

    var out = [_]f32{0.0} ** max_dimensions;
    const stats = real_merge.mergeRdaWeightedOuterGradient(
        out[0..result.dimensions],
        global_before[0..result.dimensions],
        weighted[0..participants.len],
    ) catch {
        result.violation = .invalid_dimensions;
        return;
    };

    for (0..result.dimensions) |dimension| {
        result.values[dimension] = @floatCast(out[dimension]);
    }

    const carrier = rdaCarrierForParticipants(result.dimensions, participants);
    result.total_weight = @floatCast(stats.total_weight);
    result.direction_weight = @floatCast(stats.direction_weight);
    result.mean_norm = carrier.mean_norm;
    result.direction_norm = carrier.direction_norm;
    result.output_norm = vectorNormF64(result.valueSlice());
}

fn checkRdaCarrierForParticipants(
    dimensions: usize,
    participants: []const Participant,
    claimed_output: []const f64,
) Violation {
    if (claimed_output.len != dimensions) return .invalid_dimensions;

    const carrier = rdaCarrierForParticipants(dimensions, participants);
    const output_norm = vectorNormF64(claimed_output);

    if (carrier.degenerate) {
        for (claimed_output) |value| {
            if (!approxEq(value, 0.0)) return .rda_direction_mismatch;
        }
        return .none;
    }

    if (!approxEq(output_norm, carrier.mean_norm)) return .rda_norm_mismatch;
    if (output_norm == 0.0 or !std.math.isFinite(output_norm)) return .rda_direction_mismatch;

    for (0..dimensions) |dimension| {
        const expected_direction = carrier.direction[dimension] / carrier.direction_norm;
        const actual_direction = claimed_output[dimension] / output_norm;
        if (!approxEq(expected_direction, actual_direction)) return .rda_direction_mismatch;
    }

    return .none;
}

fn rdaCarrierForParticipants(dimensions: usize, participants: []const Participant) RdaCarrier {
    var carrier = RdaCarrier{};
    var weighted_norm_sum: f64 = 0.0;

    for (participants) |participant| {
        const weight = participant.weight;
        carrier.total_weight += weight;

        const norm = vectorNormI64(participant.update.values[0..dimensions]);
        weighted_norm_sum += weight * norm;
        if (norm == 0.0) continue;

        carrier.direction_weight += weight;
        for (0..dimensions) |dimension| {
            carrier.direction[dimension] += weight * @as(f64, @floatFromInt(participant.update.values[dimension])) / norm;
        }
    }

    if (!isUsableWeight(carrier.total_weight) or !isUsableWeight(carrier.direction_weight)) {
        carrier.degenerate = true;
        return carrier;
    }

    carrier.mean_norm = weighted_norm_sum / carrier.total_weight;
    carrier.direction_norm = vectorNormF64(carrier.direction[0..dimensions]);
    if (carrier.direction_norm == 0.0 or !std.math.isFinite(carrier.direction_norm)) {
        carrier.degenerate = true;
    }

    return carrier;
}

fn matchesWindow(window: Window, update: Update) bool {
    return update.syncer_step == window.syncer_step and update.fragment_id == window.fragment_id;
}

fn approxEq(a: f64, b: f64) bool {
    return @abs(a - b) <= epsilon;
}

fn isUsableWeight(weight: f64) bool {
    return weight > 0.0 and std.math.isFinite(weight);
}

fn vectorNormI64(values: []const i64) f64 {
    var squared_norm: f64 = 0.0;
    for (values) |value| {
        const value64: f64 = @floatFromInt(value);
        squared_norm += value64 * value64;
    }
    return @sqrt(squared_norm);
}

fn vectorNormF64(values: []const f64) f64 {
    var squared_norm: f64 = 0.0;
    for (values) |value| {
        squared_norm += value * value;
    }
    return @sqrt(squared_norm);
}

fn compactRetained(updates: []const Update, retained: []const bool, out: *[max_updates]Update) usize {
    var count: usize = 0;
    for (updates, retained) |update, keep| {
        if (!keep) continue;
        out[count] = update;
        count += 1;
    }
    return count;
}

fn compactRetainedIndices(retained: []const bool, out: *[max_updates]usize) usize {
    var count: usize = 0;
    for (retained, 0..) |keep, index| {
        if (!keep) continue;
        out[count] = index;
        count += 1;
    }
    return count;
}

test "direct average outputs stay inside participant envelope" {
    const window = Window{ .syncer_step = 1, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 2 };
    const a = [_]i64{ 0, 10 };
    const b = [_]i64{ 10, 20 };
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 2, .values = &b },
    };

    const result = compute(window, .direct_average, &updates);

    try std.testing.expectEqual(Violation.none, result.violation);
    try std.testing.expectEqual(@as(usize, 2), result.participant_count);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result.values[0], epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), result.values[1], epsilon);
    try std.testing.expect(claimedInsideEnvelope(window, &updates, result.participant_mask, result.valueSlice()));
}

test "token weighted output uses real decoupled weighting formula" {
    const window = Window{ .syncer_step = 1, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 1 };
    const a = [_]i64{0};
    const b = [_]i64{10};
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &a, .tokens_since_update = 10, .steps_since_update = 1 },
        .{ .learner_id = 1, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 2, .values = &b, .tokens_since_update = 10, .steps_since_update = 2 },
    };

    const result = compute(window, .token_weighted, &updates);

    try std.testing.expectEqual(Violation.none, result.violation);
    try std.testing.expectApproxEqAbs(@as(f64, 150.0), result.total_weight, epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 3.3333333333333335), result.values[0], epsilon);
}

test "RDA output can leave coordinate envelope while satisfying direction norm carrier" {
    const window = Window{ .syncer_step = 1, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 2 };
    const a = [_]i64{ 1, 1 };
    const b = [_]i64{ 1, -1 };
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 2, .values = &b },
    };

    const result = compute(window, .rda, &updates);

    try std.testing.expectEqual(Violation.none, result.violation);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 2.0)), result.output_norm, epsilon);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 2.0)), result.values[0], epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.values[1], epsilon);
    try std.testing.expect(!claimedInsideEnvelope(window, &updates, result.participant_mask, result.valueSlice()));
    try std.testing.expectEqual(Violation.none, checkRdaCarrier(window, &updates, result.valueSlice()));
    try std.testing.expectEqual(Violation.none, checkClaimed(.{
        .window = window,
        .kind = .rda,
        .updates = &updates,
        .claimed_output = result.valueSlice(),
    }));
}

test "RDA carrier rejects direct average and wrong direction outputs" {
    const window = Window{ .syncer_step = 1, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 2 };
    const a = [_]i64{ 1, 1 };
    const b = [_]i64{ 1, -1 };
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 2, .values = &b },
    };
    const direct_average = [_]f64{ 1.0, 0.0 };
    const wrong_direction = [_]f64{ 0.0, @sqrt(@as(f64, 2.0)) };

    try std.testing.expectEqual(Violation.rda_norm_mismatch, checkRdaCarrier(window, &updates, &direct_average));
    try std.testing.expectEqual(Violation.rda_direction_mismatch, checkRdaCarrier(window, &updates, &wrong_direction));
}

test "RDA uses token weights for both direction and mean norm" {
    const window = Window{ .syncer_step = 2, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 2 };
    const a = [_]i64{ 2, 0 };
    const b = [_]i64{ 0, 2 };
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 2, .fragment_id = 0, .arrival_tick = 1, .values = &a, .tokens_since_update = 10, .steps_since_update = 1 },
        .{ .learner_id = 1, .syncer_step = 2, .fragment_id = 0, .arrival_tick = 2, .values = &b, .tokens_since_update = 10, .steps_since_update = 2 },
    };

    const result = compute(window, .rda, &updates);

    try std.testing.expectEqual(Violation.none, result.violation);
    try std.testing.expectApproxEqAbs(@as(f64, 150.0), result.total_weight, epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 150.0), result.direction_weight, epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result.mean_norm, epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result.output_norm, epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 1.78885436), result.values[0], epsilon);
    try std.testing.expectApproxEqAbs(@as(f64, 0.89442718), result.values[1], epsilon);
    try std.testing.expect(result.values[0] > result.values[1]);
}

test "RDA degenerate direction returns zero carrier output" {
    const window = Window{ .syncer_step = 3, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 2 };
    const a = [_]i64{ 1, 0 };
    const b = [_]i64{ -1, 0 };
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 3, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 3, .fragment_id = 0, .arrival_tick = 2, .values = &b },
    };
    const zero = [_]f64{ 0.0, 0.0 };
    const nonzero = [_]f64{ 1.0, 0.0 };

    const result = compute(window, .rda, &updates);

    try std.testing.expectEqual(Violation.none, result.violation);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.output_norm, epsilon);
    try std.testing.expectEqual(Violation.none, checkRdaCarrier(window, &updates, &zero));
    try std.testing.expectEqual(Violation.rda_direction_mismatch, checkRdaCarrier(window, &updates, &nonzero));
}

test "late and stale learners are excluded without moving output" {
    const window = Window{ .syncer_step = 2, .fragment_id = 1, .min_quorum = 2, .grace_deadline_tick = 4, .dimensions = 1 };
    const a = [_]i64{0};
    const b = [_]i64{10};
    const late = [_]i64{1000};
    const stale = [_]i64{-1000};
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 2, .fragment_id = 1, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 2, .fragment_id = 1, .arrival_tick = 2, .values = &b },
        .{ .learner_id = 2, .syncer_step = 2, .fragment_id = 1, .arrival_tick = 5, .values = &late },
        .{ .learner_id = 3, .syncer_step = 1, .fragment_id = 1, .arrival_tick = 2, .values = &stale },
    };

    const result = compute(window, .direct_average, &updates);

    try std.testing.expectEqual(Violation.none, result.violation);
    try std.testing.expectEqual(@as(usize, 2), result.participant_count);
    try std.testing.expectEqual(@as(usize, 2), result.non_participant_count);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result.values[0], epsilon);
}

test "quorum and grace window are checked in the same trace" {
    const window = Window{ .syncer_step = 3, .fragment_id = 0, .min_quorum = 3, .grace_deadline_tick = 3, .dimensions = 1 };
    const a = [_]i64{0};
    const b = [_]i64{10};
    const late = [_]i64{20};
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 3, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 3, .fragment_id = 0, .arrival_tick = 2, .values = &b },
        .{ .learner_id = 2, .syncer_step = 3, .fragment_id = 0, .arrival_tick = 4, .values = &late },
    };

    const result = compute(window, .direct_average, &updates);

    try std.testing.expectEqual(Violation.below_quorum, result.violation);
    try std.testing.expectEqual(@as(usize, 2), result.participant_count);
    try std.testing.expectEqual(@as(usize, 1), result.non_participant_count);
}

test "duplicate learner updates are rejected" {
    const window = Window{ .syncer_step = 1, .fragment_id = 0, .min_quorum = 1, .grace_deadline_tick = 5, .dimensions = 1 };
    const a = [_]i64{0};
    const b = [_]i64{10};
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 2, .values = &b },
    };

    const result = compute(window, .direct_average, &updates);

    try std.testing.expectEqual(Violation.duplicate_update, result.violation);
}

test "claimed outputs must match envelope and exact merge result" {
    const window = Window{ .syncer_step = 1, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 1 };
    const a = [_]i64{0};
    const b = [_]i64{10};
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 2, .values = &b },
    };
    const outside = [_]f64{100.0};
    const inside_but_wrong = [_]f64{4.0};
    const exact = [_]f64{5.0};

    try std.testing.expectEqual(Violation.output_outside_envelope, checkClaimed(.{
        .window = window,
        .kind = .direct_average,
        .updates = &updates,
        .claimed_output = &outside,
    }));
    try std.testing.expectEqual(Violation.output_mismatch, checkClaimed(.{
        .window = window,
        .kind = .direct_average,
        .updates = &updates,
        .claimed_output = &inside_but_wrong,
    }));
    try std.testing.expectEqual(Violation.none, checkClaimed(.{
        .window = window,
        .kind = .direct_average,
        .updates = &updates,
        .claimed_output = &exact,
    }));
}

test "compatible local views must agree on output" {
    const a = [_]f64{ 1.0, 2.0 };
    const b = [_]f64{ 1.0, 2.5 };

    try std.testing.expectEqual(Violation.incompatible_local_views, checkCompatibleLocalViews(
        .{ .syncer_step = 4, .fragment_id = 1, .participant_mask = 0b11, .output = &a },
        .{ .syncer_step = 4, .fragment_id = 1, .participant_mask = 0b11, .output = &b },
    ));
}

test "failing merge trace shrinks to smallest quorum-preserving participant set" {
    const window = Window{ .syncer_step = 1, .fragment_id = 0, .min_quorum = 2, .grace_deadline_tick = 5, .dimensions = 1 };
    const a = [_]i64{0};
    const b = [_]i64{10};
    const c = [_]i64{20};
    const d = [_]i64{30};
    const updates = [_]Update{
        .{ .learner_id = 0, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &a },
        .{ .learner_id = 1, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &b },
        .{ .learner_id = 2, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &c },
        .{ .learner_id = 3, .syncer_step = 1, .fragment_id = 0, .arrival_tick = 1, .values = &d },
    };
    const impossible = [_]f64{1000.0};

    const shrunk = shrinkFailingTrace(.{
        .window = window,
        .kind = .direct_average,
        .updates = &updates,
        .claimed_output = &impossible,
    });

    try std.testing.expectEqual(Violation.output_outside_envelope, shrunk.violation);
    try std.testing.expectEqual(@as(usize, 2), shrunk.retained_count);
}
