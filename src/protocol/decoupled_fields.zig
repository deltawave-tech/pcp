pub const DecoupledDiLoCoField = struct {
    pub const FRAGMENT_ID = "fragment_id";
    pub const FRAGMENT_ROUND = "fragment_round";
    pub const FRAGMENT_OFFSET = "fragment_offset";
    pub const SYNC_INTERVAL_H = "sync_interval_h";
    pub const OVERLAP_TAU = "overlap_tau";
    pub const GLOBAL_STEP = "global_step";
    pub const SYNCER_STEP = "syncer_step";
    pub const LEARNER_STEP = "learner_step";
    pub const STEPS_SINCE_FRAGMENT_UPDATE = "steps_since_fragment_update";
    pub const TOKENS_SINCE_FRAGMENT_UPDATE = "tokens_since_fragment_update";
    pub const TRAIN_EXAMPLES = "train_examples";
    pub const MERGE_WEIGHT = "merge_weight";
    pub const MIN_QUORUM = "min_quorum";
    pub const GRACE_WINDOW_MS = "grace_window_ms";
    pub const GRACE_GAMMA = "grace_gamma";
    pub const MAX_GRACE_STEPS = "max_grace_steps";
    pub const ADAPTIVE_GRACE_ENABLED = "adaptive_grace_enabled";
    pub const QUORUM_PARTICIPANT_COUNT = "quorum_participant_count";
    pub const GRACE_INCLUSION_COUNT = "grace_inclusion_count";
    pub const SKIPPED_LEARNER_COUNT = "skipped_learner_count";
    pub const STEP_DURATION_EMA_MS = "step_duration_ema_ms";
    pub const QUORUM_WAIT_EMA_MS = "quorum_wait_ema_ms";
    pub const SYNC_SEND_EMA_MS = "sync_send_ema_ms";
    pub const VECTOR_CLOCK = "vector_clock";
    pub const EVENT_TAPE_ID = "event_tape_id";
    pub const SNAPSHOT_ID = "snapshot_id";
    pub const RECOVERY_TARGET_SYNCER_STEP = "recovery_target_syncer_step";
    pub const UPDATES = "updates";
};

test "decoupled field constants keep protocol payload names stable" {
    const std = @import("std");

    try std.testing.expectEqualStrings("fragment_id", DecoupledDiLoCoField.FRAGMENT_ID);
    try std.testing.expectEqualStrings("syncer_step", DecoupledDiLoCoField.SYNCER_STEP);
    try std.testing.expectEqualStrings("tokens_since_fragment_update", DecoupledDiLoCoField.TOKENS_SINCE_FRAGMENT_UPDATE);
    try std.testing.expectEqualStrings("updates", DecoupledDiLoCoField.UPDATES);
}
