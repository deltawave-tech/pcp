const std = @import("std");
const Allocator = std.mem.Allocator;
const Shepherd = @import("shepherd.zig").Shepherd;
const MessageType = @import("../../network/message.zig").MessageType;
const MessageEnvelope = @import("../../network/message.zig").MessageEnvelope;
const ArrayList = std.ArrayList;

pub const RolloutData = struct {
    prompt: []const i64,
    completion: []const i64,
    allocator: Allocator,

    pub fn deinit(self: *RolloutData) void {
        self.allocator.free(self.prompt);
        self.allocator.free(self.completion);
    }
};

pub const RLShepherd = struct {
    base: Shepherd, // Composition

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .base = Shepherd.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.base.deinit();
    }

    /// Passthrough to base listen
    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        try self.base.listen(host, port);
    }

    /// Send a rollout request to a worker
    pub fn requestRollout(self: *Self, prompt: []const i64) !void {
        // Round-robin or random assignment
        // For MVP, broadcast to first available worker

        var payload = std.json.ObjectMap.init(self.base.allocator);
        defer payload.deinit();

        var prompt_arr = std.json.Array.init(self.base.allocator);
        defer prompt_arr.deinit();
        for (prompt) |t| try prompt_arr.append(.{ .integer = t });

        try payload.put("prompt", .{ .array = prompt_arr });

        // Use base functionality to broadcast (or sendToWorker)
        try self.base.broadcastToWorkers(MessageType.START_ROLLOUT, .{ .object = payload });
    }

    /// Collect rollout results
    pub fn collectRollouts(self: *Self, expected: usize) !ArrayList(RolloutData) {
        // Use base collect functionality
        const msgs = try self.base.collectFromWorkers(MessageType.ROLLOUT_COMPLETE, expected);
        defer {
            for (msgs.items) |msg| {
                msg.deinitClone(self.base.allocator);
            }
            msgs.deinit();
        }

        var results = ArrayList(RolloutData).init(self.base.allocator);
        errdefer {
            for (results.items) |*r| r.deinit();
            results.deinit();
        }

        for (msgs.items) |msg| {
            // Parse completion from msg.data
            const obj = switch (msg.data) {
                .object => |o| o,
                else => continue,
            };

            const completion_value = obj.get("completion") orelse continue;
            const completion_array = switch (completion_value) {
                .array => |arr| arr,
                else => continue,
            };

            var comp_list = ArrayList(i64).init(self.base.allocator);
            errdefer comp_list.deinit();

            for (completion_array.items) |item| {
                switch (item) {
                    .integer => |i| try comp_list.append(i),
                    else => {},
                }
            }

            const completion_slice = try comp_list.toOwnedSlice();

            // Create placeholder prompt (in real implementation, track this)
            const placeholder_prompt = try self.base.allocator.alloc(i64, 0);

            try results.append(.{
                .prompt = placeholder_prompt,
                .completion = completion_slice,
                .allocator = self.base.allocator,
            });
        }

        return results;
    }
};
