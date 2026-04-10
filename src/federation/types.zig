const std = @import("std");

pub const NamespacePolicySnapshot = struct {
    namespace_id: []const u8,
    default_visibility: []const u8,
    allow_global_replication: bool,
    allow_raw_payload_export: bool,
    updated_at: i64,
};

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
    namespace_policies: []const NamespacePolicySnapshot = &.{},
    mutations: []const MutationBatchItem,
};

pub const MutationBatchAck = struct {
    accepted: bool,
    acked_sequence_no: u64,
    applied_count: usize,
    duplicate_count: usize,
};
