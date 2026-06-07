const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub fn validateMessages(messages: std.json.Array, vision_enabled: bool) !void {
    for (messages.items) |msg| {
        const obj = switch (msg) {
            .object => |o| o,
            else => return error.InvalidMessage,
        };
        const role_val = obj.get("role") orelse return error.MissingRole;
        const content_val = obj.get("content") orelse return error.MissingContent;
        const role = switch (role_val) {
            .string => |s| s,
            else => return error.InvalidRole,
        };
        if (!std.mem.eql(u8, role, "system") and
            !std.mem.eql(u8, role, "user") and
            !std.mem.eql(u8, role, "assistant"))
        {
            return error.UnsupportedMessageRole;
        }

        switch (content_val) {
            .string => {},
            .array => {
                if (!vision_enabled) return error.MultimodalMessageContentUnsupported;
                return error.MultimodalMessageContentUnsupported;
            },
            else => return error.InvalidMessageContent,
        }
    }
}

pub fn render(allocator: Allocator, template: []const u8, messages: std.json.Array, vision_enabled: bool) ![]u8 {
    try validateMessages(messages, vision_enabled);
    if (std.mem.eql(u8, template, "qwen_text_only")) {
        return renderQwenTextOnly(allocator, messages);
    }
    if (std.mem.eql(u8, template, "legacy")) {
        return renderLegacy(allocator, messages);
    }
    return error.UnsupportedChatTemplate;
}

pub fn renderQwenTextOnly(allocator: Allocator, messages: std.json.Array) ![]u8 {
    var buf = ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    for (messages.items) |msg| {
        const obj = msg.object;
        const role = obj.get("role").?.string;
        const content = obj.get("content").?.string;

        try buf.appendSlice("<|im_start|>");
        try buf.appendSlice(role);
        try buf.append('\n');
        try buf.appendSlice(content);
        try buf.appendSlice("<|im_end|>\n");
    }

    try buf.appendSlice("<|im_start|>assistant\n");
    return buf.toOwnedSlice();
}

pub fn renderLegacy(allocator: Allocator, messages: std.json.Array) ![]u8 {
    var buf = ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    for (messages.items) |msg| {
        const obj = msg.object;
        const role = obj.get("role").?.string;
        const content = obj.get("content").?.string;
        if (std.mem.eql(u8, role, "system")) {
            try buf.appendSlice("System: ");
        } else if (std.mem.eql(u8, role, "user")) {
            try buf.appendSlice("User: ");
        } else if (std.mem.eql(u8, role, "assistant")) {
            try buf.appendSlice("Assistant: ");
        }
        try buf.appendSlice(content);
        try buf.appendSlice("\n");
    }
    try buf.appendSlice("Assistant: ");
    return buf.toOwnedSlice();
}

test "qwen text-only chat template renders generation prompt" {
    var messages = std.json.Array.init(std.testing.allocator);
    defer messages.deinit();

    var user = std.json.ObjectMap.init(std.testing.allocator);
    defer user.deinit();
    try user.put("role", .{ .string = "user" });
    try user.put("content", .{ .string = "Hi" });
    try messages.append(.{ .object = user });

    const rendered = try render(std.testing.allocator, "qwen_text_only", messages, false);
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", rendered);
}

test "qwen text-only chat template rejects multimodal content while vision is disabled" {
    var messages = std.json.Array.init(std.testing.allocator);
    defer messages.deinit();

    var parts = std.json.Array.init(std.testing.allocator);
    defer parts.deinit();
    try parts.append(.{ .string = "Hi" });

    var user = std.json.ObjectMap.init(std.testing.allocator);
    defer user.deinit();
    try user.put("role", .{ .string = "user" });
    try user.put("content", .{ .array = parts });
    try messages.append(.{ .object = user });

    try std.testing.expectError(error.MultimodalMessageContentUnsupported, validateMessages(messages, false));
}

test "qwen text-only chat template rejects unsupported roles" {
    var messages = std.json.Array.init(std.testing.allocator);
    defer messages.deinit();

    var tool = std.json.ObjectMap.init(std.testing.allocator);
    defer tool.deinit();
    try tool.put("role", .{ .string = "tool" });
    try tool.put("content", .{ .string = "payload" });
    try messages.append(.{ .object = tool });

    try std.testing.expectError(error.UnsupportedMessageRole, validateMessages(messages, false));
}
