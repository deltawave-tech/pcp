const std = @import("std");

const Allocator = std.mem.Allocator;

pub fn deriveExecutorId(
    allocator: Allocator,
    gateway_id: []const u8,
    service_type: []const u8,
    service_id: []const u8,
) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}:{s}:{s}", .{ gateway_id, service_type, service_id });
}

pub fn currentJobId(allocator: Allocator, executor_id: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}:current", .{executor_id});
}

pub fn queuedJobId(allocator: Allocator, executor_id: []const u8, sequence: usize) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}:job-{d}", .{ executor_id, sequence });
}

pub fn reservationId(allocator: Allocator, executor_id: []const u8, sequence: usize) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}:reservation-{d}", .{ executor_id, sequence });
}

pub fn jobBelongsToExecutor(job_id: []const u8, executor_id: []const u8) bool {
    return std.mem.startsWith(u8, job_id, executor_id) and
        job_id.len > executor_id.len and
        job_id[executor_id.len] == ':';
}

pub fn reservationBelongsToExecutor(reservation_id: []const u8, executor_id: []const u8) bool {
    return std.mem.startsWith(u8, reservation_id, executor_id) and
        reservation_id.len > executor_id.len and
        reservation_id[executor_id.len] == ':';
}
