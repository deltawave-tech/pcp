const std = @import("std");
const Allocator = std.mem.Allocator;
const json = std.json;

pub const MessageEnvelope = struct {
    recipient: NodeId,
    sender: NodeId,
    msg_type: []const u8,
    msg_id: MessageId,
    data: std.json.Value,

    const Self = @This();

    pub fn asJsonString(self: Self, allocator: Allocator) error{OutOfMemory}!std.ArrayList(u8) {
        var buffer = std.ArrayList(u8).init(allocator);
        errdefer buffer.deinit();
        try std.json.stringify(self, .{}, buffer.writer());
        return buffer;
    }

    pub fn fromJsonString(json_str: []u8, allocator: Allocator) json.ParseError(json.Scanner)!json.Parsed(MessageEnvelope) {
        return try json.parseFromSlice(MessageEnvelope, allocator, json_str, .{});
    }
};

/// A (possibly random) node id.
pub const NodeId = u8;

/// A (possibly random) message id.
pub const MessageId = u8;

pub const MessageHandler = *const fn (MessageEnvelope) anyerror!void;

pub const MessageHandlerRegistry = struct {
    registry: MapT,

    const MapT = std.StringHashMap(MessageHandler);
    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{ .registry = MapT.init(allocator) };
    }

    pub fn deinit(self: *Self) void {
        self.registry.deinit();
    }

    pub fn register(self: *Self, msg_type: []const u8, handler: MessageHandler) error{OutOfMemory}!void {
        try self.registry.put(msg_type, handler);
    }

    pub fn handle(self: Self, msg: MessageEnvelope) anyerror!void {
        const handler = self.registry.get(msg.msg_type).?;
        try handler(msg);
    }
};

test "MessageEnvelope round trip JSON" {
    const allocator = std.testing.allocator;

    const original = MessageEnvelope{
        .recipient = 1,
        .sender = 2,
        .msg_type = "text",
        .msg_id = 3,
        .data = std.json.Value{ .string = "foo" },
    };

    var registry = MessageHandlerRegistry.init(allocator);
    defer registry.deinit();
    // A handler to parse JSON string back to MessageEnvelope
    const myHandler = struct {
        fn test_handler(msg: MessageEnvelope) !void {
            //// Assertions
            try std.testing.expectEqual(original.recipient, msg.recipient);
            try std.testing.expectEqual(original.sender, msg.sender);
            try std.testing.expectEqualStrings(original.msg_type, msg.msg_type);
            try std.testing.expectEqual(original.msg_id, msg.msg_id);
            try std.testing.expectEqual("string", @tagName(msg.data));
            try std.testing.expectEqualStrings(original.data.string, msg.data.string);
        }
    }.test_handler;
    try registry.register("text", myHandler);

    // Serialize to JSON string
    const buffer = try original.asJsonString(allocator);
    defer buffer.deinit();
    const json_str = buffer.items;

    // Debug print (optional)
    std.debug.print("Serialized JSON: {s}\n", .{json_str});

    const parsed = try MessageEnvelope.fromJsonString(json_str, allocator);
    defer parsed.deinit();

    try registry.handle(parsed.value);
}
