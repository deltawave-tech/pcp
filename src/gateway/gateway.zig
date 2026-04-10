const std = @import("std");

const Allocator = std.mem.Allocator;
const gateway_config = @import("config.zig");
const service_registry = @import("service_registry.zig");
const graph_adapter = @import("graph_adapter.zig");
const event_ingest = @import("event_ingest.zig");
const graph_types = @import("../graph/types.zig");

pub const Gateway = struct {
    allocator: Allocator,
    config: gateway_config.GatewayConfig,
    service_registry: service_registry.ServiceRegistry,
    graph: graph_adapter.GatewayGraph,
    event_ingester: event_ingest.EventIngester,
    started_at: i64,

    const Self = @This();

    pub fn init(allocator: Allocator, config: gateway_config.GatewayConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .service_registry = service_registry.ServiceRegistry.init(allocator),
            .graph = graph_adapter.GatewayGraph.init(allocator, config) catch @panic("failed to initialize graph store"),
            .event_ingester = event_ingest.EventIngester.init(allocator, .{
                .gateway_id = config.gateway_id,
                .lab_id = config.lab_id,
                .default_visibility = if (config.sharing_defaults) |sharing|
                    graphVisibility(sharing.default_visibility)
                else
                    .local,
            }),
            .started_at = std.time.timestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.graph.deinit();
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
            .global_controller_endpoint = self.config.resolvedGlobalControllerEndpoint(),
        }, .{});
    }

    pub fn renderCapabilitiesJson(self: *Self, allocator: Allocator) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .gateway_id = self.config.gateway_id,
            .lab_id = self.config.lab_id,
            .capabilities = &[_][]const u8{
                "graph.query",
                "graph.mutate",
                "service.registry",
                "service.inference.proxy",
                "service.rl.control",
                "service.training.control",
                "federation.status",
                "events.ingest",
            },
        }, .{});
    }

    pub fn renderFederationStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .gateway_id = self.config.gateway_id,
            .lab_id = self.config.lab_id,
            .connected = false,
            .mode = "stub",
            .global_controller_endpoint = self.config.resolvedGlobalControllerEndpoint(),
            .replication_enabled = false,
        }, .{});
    }
};

fn graphVisibility(raw: []const u8) graph_types.Visibility {
    return graph_types.Visibility.parse(raw) orelse .local;
}
