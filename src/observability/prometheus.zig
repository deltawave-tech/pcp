const std = @import("std");

pub const Label = struct {
    name: []const u8,
    value: []const u8,
};

pub fn writeHelpAndType(writer: anytype, name: []const u8, help: []const u8, metric_type: []const u8) !void {
    try writer.print("# HELP {s} {s}\n", .{ name, help });
    try writer.print("# TYPE {s} {s}\n", .{ name, metric_type });
}

pub fn writeSampleInt(writer: anytype, name: []const u8, labels: []const Label, value: anytype) !void {
    try writeNameAndLabels(writer, name, labels);
    try writer.print(" {d}\n", .{value});
}

pub fn writeSampleFloat(writer: anytype, name: []const u8, labels: []const Label, value: f64) !void {
    try writeNameAndLabels(writer, name, labels);
    try writer.print(" {d:.6}\n", .{value});
}

fn writeNameAndLabels(writer: anytype, name: []const u8, labels: []const Label) !void {
    try writer.writeAll(name);
    if (labels.len == 0) return;

    try writer.writeByte('{');
    for (labels, 0..) |label, idx| {
        if (idx > 0) try writer.writeByte(',');
        try writer.print("{s}=\"", .{label.name});
        try writeEscapedLabelValue(writer, label.value);
        try writer.writeByte('"');
    }
    try writer.writeByte('}');
}

fn writeEscapedLabelValue(writer: anytype, value: []const u8) !void {
    for (value) |byte| {
        switch (byte) {
            '\\' => try writer.writeAll("\\\\"),
            '"' => try writer.writeAll("\\\""),
            '\n' => try writer.writeAll("\\n"),
            else => try writer.writeByte(byte),
        }
    }
}

test "prometheus label values are escaped" {
    var buf = std.ArrayList(u8).init(std.testing.allocator);
    defer buf.deinit();

    try writeSampleInt(buf.writer(), "pcp_test_metric", &.{
        .{ .name = "path", .value = "/v1/\"quoted\"\n" },
    }, 1);

    try std.testing.expectEqualStrings(
        "pcp_test_metric{path=\"/v1/\\\"quoted\\\"\\n\"} 1\n",
        buf.items,
    );
}
