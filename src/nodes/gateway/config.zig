const std = @import("std");

const Allocator = std.mem.Allocator;
const scheduling = @import("scheduling.zig");
const extension_config = @import("config_extensions.zig");

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
    internal_token_env: ?[]const u8 = null,
};

pub const FederationConfig = struct {
    enabled: bool = false,
    upstream: ?[]const u8 = null,
    token_env: ?[]const u8 = null,
    heartbeat_interval_ms: u64 = 5_000,
};

pub const WorkerFabricConfig = struct {
    host: ?[]const u8 = null,
    port: ?u16 = null,
};

pub const ControllerEndpointConfig = struct {
    host: ?[]const u8 = null,
    port: ?u16 = null,
};

pub const EmbeddedInferenceControllerConfig = struct {
    enabled: bool = false,
    config_path: []const u8,
    service_id: ?[]const u8 = null,
    api: ?ControllerEndpointConfig = null,
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    reserved_workers: ?usize = null,
    max_workers: ?usize = null,
};

pub const EmbeddedTrainingControllerConfig = struct {
    enabled: bool = false,
    config_path: []const u8,
    service_id: ?[]const u8 = null,
    api: ?ControllerEndpointConfig = null,
    workers: ?usize = null,
    should_resume: bool = false,
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    reserved_workers: ?usize = null,
    max_workers: ?usize = null,
};

pub const EmbeddedRLControllerConfig = struct {
    enabled: bool = false,
    config_path: []const u8,
    service_id: ?[]const u8 = null,
    api: ?ControllerEndpointConfig = null,
    workers: ?usize = null,
    backend: []const u8 = "cpu",
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    reserved_workers: ?usize = null,
    max_workers: ?usize = null,
};

pub const ControllersConfig = struct {
    inference: ?EmbeddedInferenceControllerConfig = null,
    training: ?EmbeddedTrainingControllerConfig = null,
    rl: ?EmbeddedRLControllerConfig = null,
    extensions: ?extension_config.ControllersConfig = null,
};

pub const SharingDefaults = struct {
    default_visibility: []const u8 = "local",
};

pub const GatewayConfig = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    graph_backend: []const u8 = "memory",
    policy_store_path: ?[]const u8 = null,
    neo4j: ?Neo4jConfig = null,
    api: ?ApiConfig = null,
    federation: ?FederationConfig = null,
    worker_fabric: ?WorkerFabricConfig = null,
    controllers: ?ControllersConfig = null,
    sharing_defaults: ?SharingDefaults = null,
    federation_hub_endpoint: ?[]const u8 = null,
    api_token_env: ?[]const u8 = null,
    internal_api_token_env: ?[]const u8 = null,

    pub fn resolvedApiTokenEnv(self: GatewayConfig) ?[]const u8 {
        if (self.api) |api| {
            if (api.token_env) |token_env| return token_env;
        }
        return self.api_token_env;
    }

    pub fn resolvedFederationHubEndpoint(self: GatewayConfig) ?[]const u8 {
        if (self.federation) |federation| {
            if (federation.upstream) |upstream| return upstream;
        }
        return self.federation_hub_endpoint;
    }

    pub fn resolvedFederationTokenEnv(self: GatewayConfig) ?[]const u8 {
        if (self.federation) |federation| {
            if (federation.token_env) |token_env| return token_env;
        }
        return null;
    }

    pub fn resolvedHeartbeatIntervalMs(self: GatewayConfig) u64 {
        if (self.federation) |federation| return federation.heartbeat_interval_ms;
        return 5_000;
    }

    pub fn resolvedInternalApiTokenEnv(self: GatewayConfig) ?[]const u8 {
        if (self.api) |api| {
            if (api.internal_token_env) |token_env| return token_env;
        }
        return self.internal_api_token_env;
    }

    pub fn resolvedWorkerFabricHost(self: GatewayConfig, fallback: []const u8) []const u8 {
        if (self.worker_fabric) |worker_fabric| {
            if (worker_fabric.host) |host| return host;
        }
        return fallback;
    }

    pub fn resolvedWorkerFabricPort(self: GatewayConfig, fallback: u16) u16 {
        if (self.worker_fabric) |worker_fabric| {
            if (worker_fabric.port) |port| return port;
        }
        return fallback;
    }

    pub fn resolvedEmbeddedInference(self: GatewayConfig) ?EmbeddedInferenceControllerConfig {
        if (self.controllers) |controllers| {
            if (controllers.inference) |inference| {
                if (inference.enabled) return inference;
            }
        }
        return null;
    }

    pub fn resolvedEmbeddedTraining(self: GatewayConfig) ?EmbeddedTrainingControllerConfig {
        if (self.controllers) |controllers| {
            if (controllers.training) |training| {
                if (training.enabled) return training;
            }
        }
        return null;
    }

    pub fn resolvedEmbeddedRL(self: GatewayConfig) ?EmbeddedRLControllerConfig {
        if (self.controllers) |controllers| {
            if (controllers.rl) |rl| {
                if (rl.enabled) return rl;
            }
        }
        return null;
    }

    pub fn enabledEmbeddedControllerCount(self: GatewayConfig) usize {
        var count: usize = 0;
        if (self.resolvedEmbeddedInference() != null) count += 1;
        if (self.resolvedEmbeddedTraining() != null) count += 1;
        if (self.resolvedEmbeddedRL() != null) count += 1;
        count += extension_config.enabledEmbeddedControllerCount(self);
        return count;
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

        if (self.federation) |federation| {
            if (federation.enabled and self.resolvedFederationHubEndpoint() == null) {
                return error.FederationHubEndpointRequired;
            }
        }

        if (self.resolvedEmbeddedInference()) |inference| {
            if (inference.config_path.len == 0) {
                return error.InvalidEmbeddedInferenceConfig;
            }
            try validateWorkerClass(inference.worker_class);
            try validateTargetArch(inference.target_arch);
            try validateSchedulingBounds(null, inference.reserved_workers, inference.max_workers);
        }
        if (self.resolvedEmbeddedTraining()) |training| {
            if (training.config_path.len == 0) {
                return error.InvalidEmbeddedTrainingConfig;
            }
            try validateWorkerClass(training.worker_class);
            try validateTargetArch(training.target_arch);
            try validateSchedulingBounds(training.workers, training.reserved_workers, training.max_workers);
        }
        if (self.resolvedEmbeddedRL()) |rl| {
            if (rl.config_path.len == 0) {
                return error.InvalidEmbeddedRLConfig;
            }
            try validateWorkerClass(rl.worker_class);
            try validateTargetArch(rl.target_arch);
            try validateSchedulingBounds(rl.workers, rl.reserved_workers, rl.max_workers);
        }
        try extension_config.validate(self);
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

fn validateWorkerClass(raw: ?[]const u8) !void {
    const value = raw orelse return;
    if (scheduling.WorkerClass.parse(value) == null) {
        return error.InvalidWorkerClass;
    }
}

fn validateTargetArch(raw: ?[]const u8) !void {
    const value = raw orelse return;
    if (value.len == 0) {
        return error.InvalidTargetArch;
    }
}

fn validateSchedulingBounds(required_workers: ?usize, reserved_workers: ?usize, max_workers: ?usize) !void {
    if (max_workers) |max_value| {
        if ((reserved_workers orelse 0) > max_value) {
            return error.InvalidWorkerReservation;
        }
        if ((required_workers orelse 0) > max_value) {
            return error.InvalidWorkerLimit;
        }
    }
}
