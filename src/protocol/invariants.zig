const std = @import("std");
const limits = @import("limits.zig");

pub fn validateWorkerCount(worker_count: usize) !void {
    if (worker_count == 0) return error.InvalidWorkerCount;
    if (worker_count > limits.worker_count_max) return error.WorkerCountTooLarge;
}

pub fn assertWorkerCount(worker_count: usize) void {
    std.debug.assert(worker_count > 0);
    std.debug.assert(worker_count <= limits.worker_count_max);
}

pub fn validateQuorum(min_quorum: usize, worker_count: usize) !void {
    try validateWorkerCount(worker_count);
    if (min_quorum == 0) return error.InvalidQuorum;
    if (min_quorum > worker_count) return error.InvalidQuorum;
}

pub fn assertQuorum(min_quorum: usize, worker_count: usize) void {
    assertWorkerCount(worker_count);
    std.debug.assert(min_quorum > 0);
    std.debug.assert(min_quorum <= worker_count);
}

pub fn validateFragmentIndex(fragment_id: usize, fragment_count: usize) !void {
    try validateFragmentCount(fragment_count);
    if (fragment_id >= fragment_count) return error.InvalidFragmentIndex;
}

pub fn validateFragmentCount(fragment_count: usize) !void {
    if (fragment_count == 0) return error.InvalidFragmentCount;
    if (fragment_count > limits.fragment_count_max) return error.FragmentCountTooLarge;
}

pub fn assertFragmentIndex(fragment_id: usize, fragment_count: usize) void {
    std.debug.assert(fragment_count > 0);
    std.debug.assert(fragment_count <= limits.fragment_count_max);
    std.debug.assert(fragment_id < fragment_count);
}

pub fn assertFragmentCount(fragment_count: usize) void {
    std.debug.assert(fragment_count > 0);
    std.debug.assert(fragment_count <= limits.fragment_count_max);
}

pub fn validateByteCount(actual_bytes: usize, expected_bytes: usize) !void {
    if (expected_bytes == 0) return error.InvalidByteCount;
    if (actual_bytes != expected_bytes) return error.ByteCountMismatch;
}

pub fn assertByteCount(actual_bytes: usize, expected_bytes: usize) void {
    std.debug.assert(expected_bytes > 0);
    std.debug.assert(actual_bytes == expected_bytes);
}

pub fn validateRequestContext(expected_request_id: u64, actual_request_id: u64) !void {
    if (expected_request_id == 0) return error.InvalidRequestId;
    if (actual_request_id != expected_request_id) return error.RequestIdMismatch;
}

pub fn assertRequestContext(expected_request_id: u64, actual_request_id: u64) void {
    std.debug.assert(expected_request_id != 0);
    std.debug.assert(actual_request_id == expected_request_id);
}

test "quorum validation accepts only positive quorum within worker count" {
    try validateQuorum(1, 1);
    try validateQuorum(2, 3);
    try std.testing.expectError(error.InvalidWorkerCount, validateQuorum(1, 0));
    try std.testing.expectError(error.InvalidQuorum, validateQuorum(0, 2));
    try std.testing.expectError(error.InvalidQuorum, validateQuorum(3, 2));
}

test "fragment validation requires an in-range index" {
    try validateFragmentCount(1);
    try validateFragmentIndex(0, 1);
    try validateFragmentIndex(7, 8);
    try std.testing.expectError(error.InvalidFragmentCount, validateFragmentCount(0));
    try std.testing.expectError(error.InvalidFragmentCount, validateFragmentIndex(0, 0));
    try std.testing.expectError(error.InvalidFragmentIndex, validateFragmentIndex(8, 8));
}
