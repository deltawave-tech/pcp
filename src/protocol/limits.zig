const std = @import("std");

pub const worker_count_max: usize = 1024;
pub const fragment_count_max: usize = 8192;
pub const tensor_count_max: usize = 1_000_000;
pub const syncer_steps_max: usize = 1_000_000;
pub const local_steps_max: usize = 1_000_000_000;
pub const message_queue_depth_max: usize = 65_536;
pub const request_body_bytes_max: usize = 64 * 1024 * 1024;
pub const chunk_bytes_max: usize = 64 * 1024 * 1024;

pub const worker_collection_timeout_ms: u64 = 14_400_000;
pub const worker_collection_poll_ms: u64 = 50;
pub const worker_join_timeout_ms: u64 = 14_400_000;
pub const worker_join_poll_ms: u64 = 100;
pub const generation_worker_ready_timeout_ms: u64 = 14_400_000;
pub const generation_worker_ready_poll_ms: u64 = 100;
pub const async_collection_poll_ms: u64 = 10;
pub const decoupled_collection_timeout_ms: u64 = 30_000;

pub const WaitBudget = struct {
    timeout_ms: u64,
    poll_interval_ms: u64,
    expected_count: usize,
    reason: []const u8,
    allow_partial: bool = false,

    pub fn validate(self: @This()) !void {
        if (self.reason.len == 0) return error.MissingWaitReason;
        if (self.expected_count > message_queue_depth_max) return error.ExpectedCountTooLarge;
        if (self.timeout_ms == 0) return;
        if (self.poll_interval_ms == 0) return error.InvalidPollInterval;
        if (self.poll_interval_ms > self.timeout_ms) return error.InvalidPollInterval;
    }

    pub fn elapsedMs(self: @This(), start_ms: i64) u64 {
        _ = self;
        return @intCast(@max(std.time.milliTimestamp() - start_ms, 0));
    }

    pub fn expired(self: @This(), start_ms: i64) bool {
        if (self.timeout_ms == 0) return true;
        return self.elapsedMs(start_ms) >= self.timeout_ms;
    }

    pub fn sleepMs(self: @This(), start_ms: i64) u64 {
        if (self.timeout_ms == 0) return 0;
        const elapsed_ms = self.elapsedMs(start_ms);
        if (elapsed_ms >= self.timeout_ms) return 0;
        return @min(self.poll_interval_ms, self.timeout_ms - elapsed_ms);
    }
};

pub fn workerCollectionBudget(expected_count: usize) WaitBudget {
    return .{
        .timeout_ms = worker_collection_timeout_ms,
        .poll_interval_ms = worker_collection_poll_ms,
        .expected_count = expected_count,
        .reason = "worker_result_collection",
        .allow_partial = true,
    };
}

pub fn workerJoinBudget(expected_count: usize) WaitBudget {
    return .{
        .timeout_ms = worker_join_timeout_ms,
        .poll_interval_ms = worker_join_poll_ms,
        .expected_count = expected_count,
        .reason = "worker_join",
    };
}

pub fn generationWorkerReadyBudget(expected_count: usize) WaitBudget {
    return .{
        .timeout_ms = generation_worker_ready_timeout_ms,
        .poll_interval_ms = generation_worker_ready_poll_ms,
        .expected_count = expected_count,
        .reason = "generation_worker_readiness",
    };
}

pub fn boundedCollectionBudget(expected_count: usize, timeout_ms: u64) WaitBudget {
    return .{
        .timeout_ms = timeout_ms,
        .poll_interval_ms = async_collection_poll_ms,
        .expected_count = expected_count,
        .reason = "bounded_message_collection",
        .allow_partial = true,
    };
}

pub fn decoupledCollectionBudget(expected_count: usize) WaitBudget {
    return .{
        .timeout_ms = decoupled_collection_timeout_ms,
        .poll_interval_ms = async_collection_poll_ms,
        .expected_count = expected_count,
        .reason = "decoupled_fragment_collection",
    };
}

test "wait budget validates named bounded waits" {
    try workerCollectionBudget(4).validate();
    try workerJoinBudget(4).validate();
    try generationWorkerReadyBudget(4).validate();
    try boundedCollectionBudget(2, 10).validate();
    try decoupledCollectionBudget(1).validate();
}

test "wait budget rejects missing reason and impossible poll interval" {
    try std.testing.expectError(error.MissingWaitReason, (WaitBudget{
        .timeout_ms = 10,
        .poll_interval_ms = 1,
        .expected_count = 1,
        .reason = "",
    }).validate());

    try std.testing.expectError(error.InvalidPollInterval, (WaitBudget{
        .timeout_ms = 10,
        .poll_interval_ms = 11,
        .expected_count = 1,
        .reason = "invalid_poll",
    }).validate());
}
