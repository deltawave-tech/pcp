const std = @import("std");

const Allocator = std.mem.Allocator;
const gateway_config = @import("config.zig");
const service_registry = @import("service_registry.zig");
const graph_adapter = @import("graph_adapter.zig");

pub const Gateway = struct {
    allocator: Allocator,
    config: gateway_config.GatewayConfig,
    service_registry: service_registry.ServiceRegistry,
    graph: graph_adapter.MemoryGraph,
    started_at: i64,

    const Self = @This();

    pub fn init(allocator: Allocator, config: gateway_config.GatewayConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .service_registry = service_registry.ServiceRegistry.init(allocator),
            .graph = graph_adapter.MemoryGraph.init(allocator, config.graph_backend),
            .started_at = std.time.timestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.service_registry.deinit();
    }

    pub fn renderReadyJson(self: *Self, allocator: Allocator, auth_enabled: bool) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .ready = true,
            .mode = "gateway",
            .gateway_id = self.config.gateway_id,
            .lab_id = self.config.lab_id,
            .graph_backend = self.config.graph_backend,
            .registered_services = self.service_registry.count(),
            .auth_enabled = auth_enabled,
        }, .{});
    }

    pub fn renderControllerJson(self: *Self, allocator: Allocator, auth_enabled: bool) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .mode = "gateway",
            .status = "running",
            .gateway_id = self.config.gateway_id,
            .lab_id = self.config.lab_id,
            .graph_backend = self.config.graph_backend,
            .started_at = self.started_at,
            .auth_enabled = auth_enabled,
            .registered_services = self.service_registry.count(),
            .global_controller_endpoint = self.config.global_controller_endpoint,
        }, .{});
    }

    pub fn renderCapabilitiesJson(self: *Self, allocator: Allocator) ![]u8 {
        _ = self;
        return std.json.stringifyAlloc(allocator, .{
            .mode = "gateway",
            .service_registry = true,
            .graph_status = true,
            .federation_status = true,
            .graph_query = false,
            .graph_mutation = false,
            .federation_replication = false,
            .supported_service_types = &[_][]const u8{ "inference", "rl", "training" },
            .endpoints = &[_][]const u8{
                "/healthz",
                "/readyz",
                "/v1/controller",
                "/v1/capabilities",
                "/v1/services",
                "/v1/services/register",
                "/v1/federation/status",
                "/v1/graph/status",
            },
        }, .{});
    }

    pub fn renderFederationStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .gateway_id = self.config.gateway_id,
            .lab_id = self.config.lab_id,
            .connected = false,
            .mode = "stub",
            .global_controller_endpoint = self.config.global_controller_endpoint,
            .replication_enabled = false,
        }, .{});
    }
};
