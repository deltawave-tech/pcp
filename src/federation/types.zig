const std = @import("std");

pub const MutationBatchItem = struct {
    sequence_no: u64,
    mutation_id: []const u8,
    namespace_id: []const u8,
    mutation_type: []const u8,
    target_id: []const u8,
    payload_json: []const u8,
    visibility: []const u8,
    provenance_json: []const u8,
    timestamp: i64,
};

pub const MutationBatchRequest = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    last_sequence_no: u64,
    last_replicated_sequence: u64,
    mutations: []const MutationBatchItem,
};

pub const MutationBatchAck = struct {
    accepted: bool,
    acked_sequence_no: u64,
    applied_count: usize,
    duplicate_count: usize,
};
