const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Neo4jConfig = struct {
    uri: []const u8,
    http_uri: ?[]const u8 = null,
    user: []const u8,
    password_env: ?[]const u8 = null,
    database: ?[]const u8 = null,
    query_timeout_ms: u32 = 5_000,
    bootstrap_on_connect: bool = true,
};

pub const ApiConfig = struct {
    host: ?[]const u8 = null,
    port: ?u16 = null,
    token_env: ?[]const u8 = null,
};

pub const FederationConfig = struct {
    enabled: bool = false,
    upstream: ?[]const u8 = null,
};

pub const SharingDefaults = struct {
    default_visibility: []const u8 = "local",
};

pub const GatewayConfig = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    graph_backend: []const u8 = "memory",
    neo4j: ?Neo4jConfig = null,
    api: ?ApiConfig = null,
    federation: ?FederationConfig = null,
    sharing_defaults: ?SharingDefaults = null,
    global_controller_endpoint: ?[]const u8 = null,
    api_token_env: ?[]const u8 = null,

    pub fn resolvedApiTokenEnv(self: GatewayConfig) ?[]const u8 {
        if (self.api) |api| {
            if (api.token_env) |token_env| return token_env;
        }
        return self.api_token_env;
    }

    pub fn resolvedGlobalControllerEndpoint(self: GatewayConfig) ?[]const u8 {
        if (self.federation) |federation| {
            if (federation.upstream) |upstream| return upstream;
        }
        return self.global_controller_endpoint;
    }

    pub fn validate(self: GatewayConfig) !void {
        if (!std.mem.eql(u8, self.graph_backend, "memory") and !std.mem.eql(u8, self.graph_backend, "neo4j")) {
            return error.UnsupportedGraphBackend;
        }

        if (std.mem.eql(u8, self.graph_backend, "neo4j")) {
            const neo4j = self.neo4j orelse return error.Neo4jConfigRequired;
            if (neo4j.uri.len == 0 or neo4j.user.len == 0) {
                return error.InvalidNeo4jConfig;
            }
        }

        const default_visibility = if (self.sharing_defaults) |sharing|
            sharing.default_visibility
        else
            "local";
        if (!std.mem.eql(u8, default_visibility, "local") and
            !std.mem.eql(u8, default_visibility, "shared") and
            !std.mem.eql(u8, default_visibility, "global"))
        {
            return error.InvalidDefaultVisibility;
        }
    }
};

pub const ConfigResult = struct {
    config: GatewayConfig,
    parsed: ?std.json.Parsed(GatewayConfig),
    json_data: ?[]u8,
    allocator: Allocator,

    pub fn deinit(self: *@This()) void {
        if (self.parsed) |*p| {
            p.deinit();
        }
        if (self.json_data) |data| {
            self.allocator.free(data);
        }
    }
};

pub fn loadGatewayConfig(allocator: Allocator, path: []const u8) !ConfigResult {
    std.log.info("Loading gateway config from: {s}", .{path});
    const data = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    const parsed = try std.json.parseFromSlice(GatewayConfig, allocator, data, .{ .ignore_unknown_fields = true });
    try parsed.value.validate();
    return .{
        .config = parsed.value,
        .parsed = parsed,
        .json_data = data,
        .allocator = allocator,
    };
}
