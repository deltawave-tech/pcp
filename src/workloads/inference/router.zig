const std = @import("std");
const types = @import("types.zig");

pub const Router = struct {
    const Self = @This();

    pub fn init() Self {
        return Self{};
    }

    pub fn decide(
        _: *Self,
        session: ?types.SessionRecord,
        ready_workers: []const types.NodeId,
    ) types.RoutingDecision {
        if (session) |record| {
            if (record.bound_worker) |worker_id| {
                for (ready_workers) |ready| {
                    if (ready == worker_id) {
                        return .{ .worker_id = worker_id, .reuse_session = true, .queued = false };
                    }
                }
            }
        }

        if (ready_workers.len > 0) {
            return .{ .worker_id = ready_workers[0], .reuse_session = false, .queued = false };
        }

        return .{ .worker_id = null, .reuse_session = false, .queued = true };
    }
};
