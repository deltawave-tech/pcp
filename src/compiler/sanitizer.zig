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

    /// Convert 32-bit Hex Floats (e.g., 0x3F800000) to Decimal (1.000000e+00).
    /// This allows the MLIR parser to read the value into a bf16 type (via truncation)
    /// whereas keeping the 32-bit hex literal would cause an "out of range" error.
    /// Special values (inf, -inf, NaN) are converted to MLIR-compatible hex bf16 literals.
    /// IMPORTANT: Stops processing at dialect_resources section to avoid corrupting binary blobs.
    pub fn sanitizeHexFloats(allocator: Allocator, source: []const u8) ![]u8 {
        var out = std.ArrayList(u8).init(allocator);
        errdefer out.deinit();

        // Find where dialect_resources section starts (if present)
        // We must NOT process binary blob data in this section
        const resources_marker = "dialect_resources:";
        const safe_end = if (std.mem.indexOf(u8, source, resources_marker)) |pos| pos else source.len;

        var i: usize = 0;
        while (i < safe_end) {
            // Look for "0x" followed by 8 hex digits (standard f32 hex length)
            if (i + 10 <= safe_end and std.mem.eql(u8, source[i .. i + 2], "0x")) {
                const hex_slice = source[i + 2 .. i + 10];
                var is_valid_hex = true;
                for (hex_slice) |c| {
                    if (!std.ascii.isHex(c)) {
                        is_valid_hex = false;
                        break;
                    }
                }

                // If valid f32 hex string
                if (is_valid_hex) {
                    // Parse hex to u32 -> bitcast to f32
                    const int_val = std.fmt.parseInt(u32, hex_slice, 16) catch 0;
                    const float_val: f32 = @bitCast(int_val);

                    // Handle special float values that can't be represented as decimal
                    if (std.math.isNan(float_val)) {
                        // NaN: bf16 representation 0x7FC0 (quiet NaN)
                        try out.appendSlice("0x7FC0");
                    } else if (std.math.isPositiveInf(float_val)) {
                        // +Inf: bf16 representation 0x7F80
                        try out.appendSlice("0x7F80");
                    } else if (std.math.isNegativeInf(float_val)) {
                        // -Inf: bf16 representation 0xFF80
                        try out.appendSlice("0xFF80");
                    } else {
                        // Normal float: format as scientific decimal
                        // This is safe for bf16 (parser handles the truncation)
                        const s = try std.fmt.allocPrint(allocator, "{e:.6}", .{float_val});
                        defer allocator.free(s);
                        try out.appendSlice(s);
                    }

                    i += 10; // Skip "0x" + 8 digits
                    continue;
                }
            }

            // Otherwise copy character as-is
            try out.append(source[i]);
            i += 1;
        }

        // Append the rest of the file (dialect_resources section) unchanged
        if (safe_end < source.len) {
            try out.appendSlice(source[safe_end..]);
        }

        return out.toOwnedSlice();
    }

    /// Converts f32 types to bf16 in the MLIR source text.
    /// This is a text-based pass that runs before MLIR parsing.
    pub fn convertF32ToBF16(allocator: Allocator, source: []const u8) ![]u8 {
        // We perform sequential replacements to cover standard MLIR type formats.
        // 1. Ranked tensors: tensor<...xf32> -> tensor<...xbf16>
        // 2. Scalars/Returns: : f32 -> : bf16
        // 3. Unranked/Generic: <f32> -> <bf16>

        // Pass 1: Handle Ranked Tensors (most common)
        const s1 = try std.mem.replaceOwned(u8, allocator, source, "xf32>", "xbf16>");
        defer allocator.free(s1);

        // Pass 2: Handle Scalars and Return types (e.g. "%0 = ... : f32")
        const s2 = try std.mem.replaceOwned(u8, allocator, s1, ": f32", ": bf16");
        defer allocator.free(s2);

        // Pass 3: Handle Template/Generic types (e.g. "dense<...> : tensor<f32>")
        const s3 = try std.mem.replaceOwned(u8, allocator, s2, "<f32>", "<bf16>");

        // Caller owns s3
        return s3;
    }
};
