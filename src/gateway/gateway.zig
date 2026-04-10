const std = @import("std");

const Allocator = std.mem.Allocator;
const gateway_config = @import("config.zig");
const service_registry = @import("service_registry.zig");
const graph_adapter = @import("graph_adapter.zig");
const event_ingest = @import("event_ingest.zig");
const policy_store = @import("../graph/policy_store.zig");
const graph_types = @import("../graph/types.zig");

pub const FederationPeer = struct {
    gateway_id: []u8,
    lab_id: []u8,
    base_url: []u8,
    graph_backend: []u8,
    status: []u8,
    registered_services: usize,
    last_sequence_no: u64,
    last_replicated_sequence: u64,
    connected_at: i64,
    last_seen_at: i64,

    pub fn deinit(self: *FederationPeer, allocator: Allocator) void {
        allocator.free(self.gateway_id);
        allocator.free(self.lab_id);
        allocator.free(self.base_url);
        allocator.free(self.graph_backend);
        allocator.free(self.status);
    }
};

pub const FederationState = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    connected: bool,
    upstream_endpoint: ?[]u8,
    status_text: []u8,
    last_sync_at: ?i64,
    last_error: ?[]u8,
    last_sequence_no: u64,
    last_replicated_sequence: u64,
    pending_mutations: usize,
    peers: std.ArrayList(FederationPeer),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .connected = false,
            .upstream_endpoint = null,
            .status_text = allocator.dupe(u8, "disconnected") catch @panic("oom"),
            .last_sync_at = null,
            .last_error = null,
            .last_sequence_no = 0,
            .last_replicated_sequence = 0,
            .pending_mutations = 0,
            .peers = std.ArrayList(FederationPeer).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.status_text);
        if (self.upstream_endpoint) |endpoint| self.allocator.free(endpoint);
        if (self.last_error) |value| self.allocator.free(value);
        self.clearPeersLocked();
        self.peers.deinit();
    }

    pub fn updateFromConnectResponse(self: *Self, allocator: Allocator, endpoint: []const u8, response: std.json.Value) !void {
        const root = switch (response) {
            .object => |object| object,
            else => return error.InvalidFederationResponse,
        };

        var peers = std.ArrayList(FederationPeer).init(allocator);
        errdefer {
            for (peers.items) |*peer| peer.deinit(allocator);
            peers.deinit();
        }

        const peers_value = root.get("peers") orelse return error.InvalidFederationResponse;
        const peers_root = switch (peers_value) {
            .object => |object| object,
            else => return error.InvalidFederationResponse,
        };
        const gateways_value = peers_root.get("gateways") orelse return error.InvalidFederationResponse;
        const gateways = switch (gateways_value) {
            .array => |array| array.items,
            else => return error.InvalidFederationResponse,
        };

        for (gateways) |item| {
            const object = switch (item) {
                .object => |object| object,
                else => return error.InvalidFederationResponse,
            };
            try peers.append(.{
                .gateway_id = try allocator.dupe(u8, stringField(object, "gateway_id") orelse return error.InvalidFederationResponse),
                .lab_id = try allocator.dupe(u8, stringField(object, "lab_id") orelse return error.InvalidFederationResponse),
                .base_url = try allocator.dupe(u8, stringField(object, "base_url") orelse return error.InvalidFederationResponse),
                .graph_backend = try allocator.dupe(u8, stringField(object, "graph_backend") orelse return error.InvalidFederationResponse),
                .status = try allocator.dupe(u8, stringField(object, "status") orelse return error.InvalidFederationResponse),
                .registered_services = usizeField(object, "registered_services") orelse 0,
                .last_sequence_no = u64Field(object, "last_sequence_no") orelse 0,
                .last_replicated_sequence = u64Field(object, "last_replicated_sequence") orelse 0,
                .connected_at = intField(object, "connected_at") orelse std.time.timestamp(),
                .last_seen_at = intField(object, "last_seen_at") orelse std.time.timestamp(),
            });
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        try replaceOptionalStringLocked(self, &self.upstream_endpoint, endpoint);
        try replaceStringLocked(self, &self.status_text, "connected");
        self.connected = true;
        self.last_sync_at = std.time.timestamp();
        if (self.last_error) |value| {
            self.allocator.free(value);
            self.last_error = null;
        }
        self.replacePeersLocked(peers);
    }

    pub fn setDisconnected(self: *Self, err: anyerror) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.connected = false;
        replaceStringLocked(self, &self.status_text, "disconnected") catch {};
        if (self.last_error) |value| self.allocator.free(value);
        self.last_error = self.allocator.dupe(u8, @errorName(err)) catch null;
    }

    pub fn updateReplicationState(self: *Self, last_sequence_no: u64, last_replicated_sequence: u64, pending_mutations: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.last_sequence_no = last_sequence_no;
        self.last_replicated_sequence = last_replicated_sequence;
        self.pending_mutations = pending_mutations;
    }

    pub fn renderStatusJson(self: *Self, allocator: Allocator, gateway_id: []const u8, lab_id: []const u8, configured_endpoint: ?[]const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        return std.json.stringifyAlloc(allocator, .{
            .gateway_id = gateway_id,
            .lab_id = lab_id,
            .connected = self.connected,
            .status = self.status_text,
            .peer_count = self.peers.items.len,
            .last_sync_at = self.last_sync_at,
            .last_error = self.last_error,
            .global_controller_endpoint = self.upstream_endpoint orelse configured_endpoint,
            .replication_enabled = false,
            .last_sequence_no = self.last_sequence_no,
            .last_replicated_sequence = self.last_replicated_sequence,
            .pending_mutations = self.pending_mutations,
        }, .{});
    }

    pub fn renderPeersJson(self: *Self, allocator: Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const ResponsePeer = struct {
            gateway_id: []const u8,
            lab_id: []const u8,
            base_url: []const u8,
            graph_backend: []const u8,
            status: []const u8,
            registered_services: usize,
            last_sequence_no: u64,
            last_replicated_sequence: u64,
            connected_at: i64,
            last_seen_at: i64,
        };

        var response_peers = std.ArrayList(ResponsePeer).init(allocator);
        defer response_peers.deinit();
        for (self.peers.items) |peer| {
            try response_peers.append(.{
                .gateway_id = peer.gateway_id,
                .lab_id = peer.lab_id,
                .base_url = peer.base_url,
                .graph_backend = peer.graph_backend,
                .status = peer.status,
                .registered_services = peer.registered_services,
                .last_sequence_no = peer.last_sequence_no,
                .last_replicated_sequence = peer.last_replicated_sequence,
                .connected_at = peer.connected_at,
                .last_seen_at = peer.last_seen_at,
            });
        }

        return std.json.stringifyAlloc(allocator, .{
            .gateways = response_peers.items,
        }, .{});
    }

    fn replacePeersLocked(self: *Self, peers: std.ArrayList(FederationPeer)) void {
        self.clearPeersLocked();
        self.peers.deinit();
        self.peers = peers;
    }

    fn clearPeersLocked(self: *Self) void {
        for (self.peers.items) |*peer| {
            peer.deinit(self.allocator);
        }
        self.peers.clearRetainingCapacity();
    }
};

pub const Gateway = struct {
    allocator: Allocator,
    config: gateway_config.GatewayConfig,
    service_registry: service_registry.ServiceRegistry,
    graph: graph_adapter.GatewayGraph,
    policy_store: policy_store.GraphPolicyStore,
    event_ingester: event_ingest.EventIngester,
    federation: FederationState,
    started_at: i64,

    const Self = @This();

    pub fn init(allocator: Allocator, config: gateway_config.GatewayConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .service_registry = service_registry.ServiceRegistry.init(allocator),
            .graph = graph_adapter.GatewayGraph.init(allocator, config) catch @panic("failed to initialize graph store"),
            .policy_store = policy_store.GraphPolicyStore.init(allocator, config.policy_store_path) catch @panic("failed to initialize policy store"),
            .event_ingester = event_ingest.EventIngester.init(allocator, .{
                .gateway_id = config.gateway_id,
                .lab_id = config.lab_id,
                .default_visibility = if (config.sharing_defaults) |sharing|
                    graphVisibility(sharing.default_visibility)
                else
                    .local,
            }),
            .federation = FederationState.init(allocator),
            .started_at = std.time.timestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.graph.deinit();
        self.policy_store.deinit();
        self.service_registry.deinit();
        self.federation.deinit();
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
                "service.rl.proxy",
                "service.training.proxy",
                "federation.status",
                "events.ingest",
            },
        }, .{});
    }

    pub fn renderFederationStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        self.federation.updateReplicationState(
            self.graph.lastSequence(),
            self.graph.lastReplicatedSequence(),
            self.graph.countPendingReplications(),
        );
        return self.federation.renderStatusJson(
            allocator,
            self.config.gateway_id,
            self.config.lab_id,
            self.config.resolvedGlobalControllerEndpoint(),
        );
    }
};

fn graphVisibility(raw: []const u8) graph_types.Visibility {
    return graph_types.Visibility.parse(raw) orelse .local;
}

pub fn defaultPolicyVisibility(config: gateway_config.GatewayConfig) graph_types.Visibility {
    return if (config.sharing_defaults) |sharing|
        graphVisibility(sharing.default_visibility)
    else
        .local;
}

fn replaceStringLocked(state: *FederationState, target: *[]u8, value: []const u8) !void {
    state.allocator.free(target.*);
    target.* = try state.allocator.dupe(u8, value);
}

fn replaceOptionalStringLocked(state: *FederationState, target: *?[]u8, value: []const u8) !void {
    if (target.*) |existing| state.allocator.free(existing);
    target.* = try state.allocator.dupe(u8, value);
}

fn stringField(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |inner| inner,
        else => null,
    };
}

fn intField(object: std.json.ObjectMap, key: []const u8) ?i64 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |inner| inner,
        else => null,
    };
}

fn usizeField(object: std.json.ObjectMap, key: []const u8) ?usize {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |inner| @intCast(inner),
        else => null,
    };
}

fn u64Field(object: std.json.ObjectMap, key: []const u8) ?u64 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |inner| @intCast(inner),
        else => null,
    };
}
