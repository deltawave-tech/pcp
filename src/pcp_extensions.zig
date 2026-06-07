const backend_selection = @import("backends/selection.zig");

pub const CliMode = enum { none };

pub const CliArgs = struct {
    mode: CliMode = .none,

    pub fn parseFlag(_: *CliArgs, _: [][:0]u8, _: *usize) !bool {
        return false;
    }
};

pub fn printUsage() void {}

pub fn printExamples() void {}

pub fn runIfSelected(_: std.mem.Allocator, _: CliArgs, _: backend_selection.Backend, _: u32) !bool {
    return false;
}

const std = @import("std");
