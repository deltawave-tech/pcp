const std = @import("std");
const Allocator = std.mem.Allocator;
const json = std.json;

pub const MessageEnvelope = struct {
    recipient_node: NodeId,
    recipient_service: ServiceId,
    sender_node: NodeId,
    sender_service: ServiceId,
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

    /// Deep clone a MessageEnvelope with owned copies of all string/JSON data
    pub fn clone(self: Self, allocator: Allocator) !Self {
        // Clone string fields
        const msg_type_copy = try allocator.dupe(u8, self.msg_type);
        errdefer allocator.free(msg_type_copy);

        const recipient_service_copy = try allocator.dupe(u8, self.recipient_service);
        errdefer allocator.free(recipient_service_copy);

        const sender_service_copy = try allocator.dupe(u8, self.sender_service);
        errdefer allocator.free(sender_service_copy);

        // Deep clone JSON value
        const data_copy = try cloneJsonValue(self.data, allocator);

        return Self{
            .recipient_node = self.recipient_node,
            .recipient_service = recipient_service_copy,
            .sender_node = self.sender_node,
            .sender_service = sender_service_copy,
            .msg_type = msg_type_copy,
            .msg_id = self.msg_id,
            .data = data_copy,
        };
    }

    /// Free owned copies created by clone()
    pub fn deinitClone(self: Self, allocator: Allocator) void {
        allocator.free(self.msg_type);
        allocator.free(self.recipient_service);
        allocator.free(self.sender_service);
        freeJsonValue(self.data, allocator);
    }

    /// Helper to deep clone a JSON value
    fn cloneJsonValue(value: std.json.Value, allocator: Allocator) !std.json.Value {
        return switch (value) {
            .null => .null,
            .bool => |b| .{ .bool = b },
            .integer => |i| .{ .integer = i },
            .float => |f| .{ .float = f },
            .number_string => |s| .{ .number_string = try allocator.dupe(u8, s) },
            .string => |s| .{ .string = try allocator.dupe(u8, s) },
            .array => |arr| {
                var new_array = std.json.Array.init(allocator);
                errdefer new_array.deinit();
                for (arr.items) |item| {
                    try new_array.append(try cloneJsonValue(item, allocator));
                }
                return .{ .array = new_array };
            },
            .object => |obj| {
                var new_obj = std.json.ObjectMap.init(allocator);
                errdefer new_obj.deinit();
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
                    const value_copy = try cloneJsonValue(entry.value_ptr.*, allocator);
                    try new_obj.put(key_copy, value_copy);
                }
                return .{ .object = new_obj };
            },
        };
    }

    /// Helper to free a cloned JSON value
    fn freeJsonValue(value: std.json.Value, allocator: Allocator) void {
        switch (value) {
            .null, .bool, .integer, .float => {},
            .number_string => |s| allocator.free(s),
            .string => |s| allocator.free(s),
            .array => |arr| {
                for (arr.items) |item| {
                    freeJsonValue(item, allocator);
                }
                // Need to cast to mutable to call deinit
                var arr_mut = arr;
                arr_mut.deinit();
            },
            .object => |obj| {
                // Need to cast to mutable to iterate and deinit
                var obj_mut = obj;
                var it = obj_mut.iterator();
                while (it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    freeJsonValue(entry.value_ptr.*, allocator);
                }
                obj_mut.deinit();
            },
        }
    }
    
    /// Create MessageEnvelope with binary data (Base64 encoded)
    pub fn createWithBinaryData(
        recipient_node: NodeId,
        recipient_service: ServiceId,
        sender_node: NodeId,
        sender_service: ServiceId,
        msg_type: []const u8,
        msg_id: MessageId,
        binary_data: []const u8,
        allocator: Allocator,
    ) !Self {
        // Base64 encode the binary data
        const b64_len = std.base64.standard.Encoder.calcSize(binary_data.len);
        const b64_encoded = try allocator.alloc(u8, b64_len);
        defer allocator.free(b64_encoded);
        
        const encoded_len = std.base64.standard.Encoder.encode(b64_encoded, binary_data).len;
        const final_encoded = try allocator.dupe(u8, b64_encoded[0..encoded_len]);
        
        return Self{
            .recipient_node = recipient_node,
            .recipient_service = recipient_service,
            .sender_node = sender_node,
            .sender_service = sender_service,
            .msg_type = msg_type,
            .msg_id = msg_id,
            .data = std.json.Value{ .string = final_encoded },
        };
    }
    
    /// Extract binary data from MessageEnvelope (Base64 decode)
    pub fn extractBinaryData(self: Self, allocator: Allocator) ![]u8 {
        const b64_string = switch (self.data) {
            .string => |s| s,
            else => return error.InvalidMessageFormat,
        };
        
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_string);
        const decoded = try allocator.alloc(u8, decoded_len);
        
        try std.base64.standard.Decoder.decode(decoded, b64_string);
        return decoded;
    }
};

/// A (possibly random) node id.
pub const NodeId = u8;
pub const ServiceId = []const u8;

/// A (possibly random) message id.
pub const MessageId = u8;

/// Message types for DiLoCo distributed training
pub const MessageType = struct {
    pub const JOIN_REQUEST = "JoinRequest";
    pub const JOIN_ACCEPT = "JoinAccept";
    pub const INITIALIZE_GRAPH = "InitializeGraph";
    pub const START_INNER_LOOP = "StartInnerLoop";
    pub const INNER_LOOP_COMPLETE = "InnerLoopComplete";
    pub const SHUTDOWN = "Shutdown";
    pub const HEARTBEAT = "Heartbeat";

    // Supervisor Control Protocol
    pub const SUPERVISOR_HANDSHAKE = "SupervisorHandshake"; // Supervisor -> Shepherd (ID: u64)
    pub const RESTART_WORKER = "RestartWorker";           // Shepherd -> Supervisor (Force restart)

    // RL / GRPO Protocol
    pub const START_ROLLOUT = "StartRollout";       // Shepherd -> Worker: "Generate text for these prompts"
    pub const ROLLOUT_COMPLETE = "RolloutComplete"; // Worker -> Shepherd: "Here are the token IDs and LogProbs"
    pub const UPDATE_WEIGHTS = "UpdateWeights";     // Shepherd -> Worker: "Here are the new model weights"
};

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

pub fn expectEqualMessages(expected: MessageEnvelope, actual: MessageEnvelope) error{TestExpectedEqual}!void {
    try std.testing.expectEqual(expected.recipient_node, actual.recipient_node);
    try std.testing.expectEqualStrings(expected.recipient_service, actual.recipient_service);
    try std.testing.expectEqual(expected.sender_node, actual.sender_node);
    try std.testing.expectEqualStrings(expected.sender_service, actual.sender_service);
    try std.testing.expectEqualStrings(expected.msg_type, actual.msg_type);
    try std.testing.expectEqual(expected.msg_id, actual.msg_id);

    try std.testing.expectEqual(@tagName(expected.data), @tagName(actual.data));
    switch (actual.data) {
        json.Value.string => {
            try std.testing.expectEqualStrings(expected.data.string, actual.data.string);
        },
        else => {
            _ = undefined;
        },
    }
}

test "MessageEnvelope round trip JSON" {
    const allocator = std.testing.allocator;

    const original = MessageEnvelope{
        .recipient_node = 1,
        .recipient_service = "serviceA",
        .sender_node = 2,
        .sender_service = "serviceB",
        .msg_type = "text",
        .msg_id = 3,
        .data = std.json.Value{ .string = "foo" },
    };

    var registry = MessageHandlerRegistry.init(allocator);
    defer registry.deinit();
    // A handler to parse JSON string back to MessageEnvelope
    const myHandler = struct {
        fn test_handler(msg: MessageEnvelope) error{TestExpectedEqual}!void {
            try expectEqualMessages(original, msg);
        }
    }.test_handler;
    try registry.register("text", myHandler);

    // Serialize to JSON string
    const buffer = try original.asJsonString(allocator);
    defer buffer.deinit();
    const json_str = buffer.items;

    // Debug print (optional)
    // std.debug.print("Serialized JSON: {s}\n", .{json_str});

    const parsed = try MessageEnvelope.fromJsonString(json_str, allocator);
    defer parsed.deinit();

    try registry.handle(parsed.value);
}
