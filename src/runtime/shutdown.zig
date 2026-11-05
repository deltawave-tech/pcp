const std = @import("std");
const builtin = @import("builtin");

var requested = std.atomic.Value(u8).init(0);

fn handleSignal(_: i32) callconv(.C) void {
    requested.store(1, .release);
}

pub fn installSignalHandlers() !void {
    if (builtin.os.tag == .windows) return;

    const action = std.posix.Sigaction{
        .handler = .{ .handler = handleSignal },
        .mask = std.posix.empty_sigset,
        .flags = 0,
    };
    try std.posix.sigaction(std.posix.SIG.TERM, &action, null);
    try std.posix.sigaction(std.posix.SIG.INT, &action, null);
}

pub fn request() void {
    requested.store(1, .release);
}

pub fn isRequested() bool {
    return requested.load(.acquire) == 1;
}

pub fn resetForTest() void {
    requested.store(0, .release);
}

pub fn waitUntilRequested() void {
    while (!isRequested()) {
        std.time.sleep(100 * std.time.ns_per_ms);
    }
}

test "shutdown request flag is observable" {
    resetForTest();
    try std.testing.expect(!isRequested());
    request();
    try std.testing.expect(isRequested());
}
