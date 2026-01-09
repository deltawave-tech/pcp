const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ModelSanitizer = struct {
    /// Applies numerical stability patches to MLIR source.
    /// Injects epsilon guards for log operations and division operations.
    pub fn applyStabilityPatches(allocator: Allocator, source: []const u8) ![]u8 {
        var out_buffer = std.ArrayList(u8).init(allocator);
        errdefer out_buffer.deinit();

        var lines = std.mem.splitScalar(u8, source, '\n');
        var patch_count: usize = 0;

        while (lines.next()) |line| {
            if (std.mem.indexOf(u8, line, "stablehlo.log") != null and
                std.mem.indexOf(u8, line, "tensor<2048x1xf32>") != null)
            {
                try patchLogOp(&out_buffer, allocator, line, &patch_count, "tensor<2048x1xf32>");
                continue;
            }

            if (std.mem.indexOf(u8, line, "stablehlo.divide") != null and
                std.mem.indexOf(u8, line, "tensor<64x4x32x32xf32>") != null)
            {
                try patchDivOp(&out_buffer, allocator, line, &patch_count, "tensor<64x4x32x32xf32>");
                continue;
            }

            try appendLine(&out_buffer, line);
        }

        return out_buffer.toOwnedSlice();
    }

    fn patchLogOp(buf: *std.ArrayList(u8), allocator: Allocator, line: []const u8, counter: *usize, type_str: []const u8) !void {
        var parts = std.mem.tokenizeAny(u8, line, " =:");
        const res_var = parts.next() orelse {
            try appendLine(buf, line);
            return;
        };
        _ = parts.next();
        const arg_var = parts.next() orelse {
            try appendLine(buf, line);
            return;
        };

        const eps_var = try std.fmt.allocPrint(allocator, "%eps_log_{d}", .{counter.*});
        defer allocator.free(eps_var);
        try appendFmt(buf, "    {s} = stablehlo.constant dense<1.000000e-6> : {s}\n", .{ eps_var, type_str });

        const safe_var = try std.fmt.allocPrint(allocator, "%safe_log_{d}", .{counter.*});
        defer allocator.free(safe_var);
        try appendFmt(buf, "    {s} = stablehlo.add {s}, {s} : {s}\n", .{ safe_var, arg_var, eps_var, type_str });

        try appendFmt(buf, "    {s} = stablehlo.log {s} : {s}\n", .{ res_var, safe_var, type_str });

        counter.* += 1;
        std.log.info("ModelSanitizer: Patched LOG op {s}", .{res_var});
    }

    fn patchDivOp(buf: *std.ArrayList(u8), allocator: Allocator, line: []const u8, counter: *usize, type_str: []const u8) !void {
        var parts = std.mem.tokenizeAny(u8, line, " =:");
        const res_var = parts.next() orelse {
            try appendLine(buf, line);
            return;
        };
        _ = parts.next();
        const lhs_var = parts.next() orelse {
            try appendLine(buf, line);
            return;
        };

        const rhs_raw = parts.next() orelse {
            try appendLine(buf, line);
            return;
        };

        const lhs_clean = std.mem.trimRight(u8, lhs_var, ",");
        const rhs_clean = std.mem.trimRight(u8, rhs_raw, ",");

        const eps_var = try std.fmt.allocPrint(allocator, "%eps_div_{d}", .{counter.*});
        defer allocator.free(eps_var);
        try appendFmt(buf, "    {s} = stablehlo.constant dense<1.000000e-6> : {s}\n", .{ eps_var, type_str });

        const safe_rhs = try std.fmt.allocPrint(allocator, "%safe_denom_{d}", .{counter.*});
        defer allocator.free(safe_rhs);
        try appendFmt(buf, "    {s} = stablehlo.add {s}, {s} : {s}\n", .{ safe_rhs, rhs_clean, eps_var, type_str });

        try appendFmt(buf, "    {s} = stablehlo.divide {s}, {s} : {s}\n", .{ res_var, lhs_clean, safe_rhs, type_str });

        counter.* += 1;
        std.log.info("ModelSanitizer: Patched DIVIDE op {s}", .{res_var});
    }

    fn appendLine(buf: *std.ArrayList(u8), line: []const u8) !void {
        try buf.appendSlice(line);
        try buf.append('\n');
    }

    fn appendFmt(buf: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) !void {
        const str = try std.fmt.allocPrint(buf.allocator, fmt, args);
        defer buf.allocator.free(str);
        try buf.appendSlice(str);
    }
};
