const std = @import("std");
const message = @import("../network/message.zig");

pub const RequestId = message.RequestId;
pub const RoundId = message.RoundId;
pub const TaskId = message.TaskId;
pub const NodeId = message.NodeId;

pub const SessionState = enum {
    idle,
    active,
    expired,
};

pub const ModelRecord = struct {
    model_id: []const u8,
    generation_vmfb_path: []const u8,
    generation_mlir_path: []const u8,
    weights_path: []const u8,
    tokenizer_path: []const u8,
    max_context_tokens: usize,
    default_max_output_tokens: usize,
    pool_name: []const u8,
};

pub const SessionRecord = struct {
    session_id: []const u8,
    model_id: []const u8,
    bound_worker: ?NodeId,
    next_round_id: RoundId,
    last_prompt_hash: u64,
    last_prompt_tokens: ?[]i64,
    last_access_ts: i64,
    expires_at: i64,
    state: SessionState,
};

pub const ActiveGeneration = struct {
    request_id: RequestId,
    task_id: TaskId,
    session_id: []const u8,
    worker_id: NodeId,
    model_id: []const u8,
    client_stream_id: []const u8,
    started_at: i64,
    deadline_at: i64,
    prompt_tokens: usize,
    completion_tokens: usize,
    cancelled: bool,
};

pub const PendingRequest = struct {
    request_id: RequestId,
    session_id: []const u8,
    model_id: []const u8,
    created_at: i64,
};

pub const RoutingDecision = struct {
    worker_id: ?NodeId,
    reuse_session: bool,
    queued: bool,
};
