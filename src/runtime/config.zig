const std = @import("std");

const Allocator = std.mem.Allocator;

pub fn isProductionMode() bool {
    if (std.posix.getenv("PCP_ENV")) |value| {
        if (std.ascii.eqlIgnoreCase(value, "production") or std.ascii.eqlIgnoreCase(value, "prod")) return true;
    }
    if (std.posix.getenv("PCP_MODE")) |value| {
        if (std.ascii.eqlIgnoreCase(value, "production") or std.ascii.eqlIgnoreCase(value, "prod")) return true;
    }
    if (std.posix.getenv("PCP_PRODUCTION")) |value| {
        if (std.mem.eql(u8, value, "1") or std.ascii.eqlIgnoreCase(value, "true")) return true;
    }
    return std.posix.getenv("KUBERNETES_SERVICE_HOST") != null;
}

pub fn stateDir(allocator: Allocator) ![]u8 {
    return std.process.getEnvVarOwned(allocator, "STATE_DIR") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => if (isProductionMode())
            try allocator.dupe(u8, "/var/lib/pcp")
        else
            try allocator.dupe(u8, "/tmp/pcp-dev"),
        else => err,
    };
}

pub fn statePath(allocator: Allocator, parts: []const []const u8) ![]u8 {
    const root = try stateDir(allocator);
    defer allocator.free(root);

    var all_parts = try allocator.alloc([]const u8, parts.len + 1);
    defer allocator.free(all_parts);
    all_parts[0] = root;
    for (parts, 0..) |part, i| all_parts[i + 1] = part;
    return try std.fs.path.join(allocator, all_parts);
}

pub fn parsePositiveUsizeEnv(allocator: Allocator, name: []const u8, default_value: ?usize, required: bool) !usize {
    const raw = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => {
            if (default_value) |value| return value;
            if (required) {
                std.log.err("Missing required environment variable {s}", .{name});
                return error.MissingRequiredEnv;
            }
            return error.EnvironmentVariableNotFound;
        },
        else => return err,
    };
    defer allocator.free(raw);

    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const value = std.fmt.parseInt(usize, trimmed, 10) catch |err| {
        std.log.err("Invalid integer in {s}='{s}': {}", .{ name, trimmed, err });
        return error.InvalidEnvironmentValue;
    };
    if (value == 0) {
        std.log.err("{s} must be greater than zero", .{name});
        return error.InvalidEnvironmentValue;
    }
    return value;
}

pub fn parseOptionalPortEnv(allocator: Allocator, name: []const u8) !?u16 {
    const raw = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return err,
    };
    defer allocator.free(raw);

    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    return std.fmt.parseInt(u16, trimmed, 10) catch |err| {
        std.log.err("Invalid port in {s}='{s}': {}", .{ name, trimmed, err });
        return error.InvalidEnvironmentValue;
    };
}

pub fn fileEnvName(allocator: Allocator, env_name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{s}_FILE", .{env_name});
}

pub fn loadSecretFromEnvOrFile(
    allocator: Allocator,
    env_name: ?[]const u8,
    canonical_file_env: ?[]const u8,
    required: bool,
) !?[]u8 {
    const name = env_name orelse {
        if (required) return error.MissingSecretEnvName;
        return null;
    };

    const derived_file_env = try fileEnvName(allocator, name);
    defer allocator.free(derived_file_env);

    const file_env = canonical_file_env orelse derived_file_env;
    if (try loadSecretFromFileEnv(allocator, file_env)) |secret| return secret;
    if (canonical_file_env != null and !std.mem.eql(u8, file_env, derived_file_env)) {
        if (try loadSecretFromFileEnv(allocator, derived_file_env)) |secret| return secret;
    }

    return std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => {
            if (required) {
                std.log.err("Missing required token: set {s} or {s}", .{ name, file_env });
                return error.MissingApiToken;
            }
            return null;
        },
        else => err,
    };
}

fn loadSecretFromFileEnv(allocator: Allocator, file_env: []const u8) !?[]u8 {
    const path = std.process.getEnvVarOwned(allocator, file_env) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return err,
    };
    defer allocator.free(path);
    if (path.len == 0) return null;

    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    errdefer allocator.free(bytes);
    return try trimOwned(allocator, bytes);
}

fn trimOwned(allocator: Allocator, bytes: []u8) ![]u8 {
    const trimmed = std.mem.trim(u8, bytes, " \t\r\n");
    if (trimmed.ptr == bytes.ptr and trimmed.len == bytes.len) return bytes;
    const out = try allocator.dupe(u8, trimmed);
    allocator.free(bytes);
    return out;
}

test "state dir defaults by mode-independent env" {
    const path = try statePath(std.testing.allocator, &.{ "compiler", "x.mlir" });
    defer std.testing.allocator.free(path);
    try std.testing.expect(std.mem.indexOf(u8, path, "compiler") != null);
}
