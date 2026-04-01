const std = @import("std");
const Allocator = std.mem.Allocator;

const Shepherd = @import("shepherd.zig").Shepherd;
const inference_types = @import("../../inference/types.zig");
const model_registry = @import("../../inference/model_registry.zig");
const session_manager = @import("../../inference/session_manager.zig");
const router = @import("../../inference/router.zig");

const ArrayList = std.ArrayList;

pub const InferenceShepherd = struct {
    base: Shepherd,
    models: model_registry.ModelRegistry,
    sessions: session_manager.SessionManager,
    router: router.Router,
    active: std.AutoHashMap(inference_types.RequestId, inference_types.ActiveGeneration),
    pending: ArrayList(inference_types.PendingRequest),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .base = Shepherd.init(allocator),
            .models = model_registry.ModelRegistry.init(allocator),
            .sessions = session_manager.SessionManager.init(allocator),
            .router = router.Router.init(),
            .active = std.AutoHashMap(inference_types.RequestId, inference_types.ActiveGeneration).init(allocator),
            .pending = ArrayList(inference_types.PendingRequest).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.pending.deinit();
        self.active.deinit();
        self.sessions.deinit();
        self.models.deinit();
        self.base.deinit();
    }

    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        try self.base.listen(host, port);
    }
};
