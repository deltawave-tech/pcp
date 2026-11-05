const std = @import("std");

const Allocator = std.mem.Allocator;
const gateway_config = @import("config.zig");
const identity = @import("identity.zig");
const service_registry = @import("service_registry.zig");
const graph_adapter = @import("graph_adapter.zig");
const event_ingest = @import("event_ingest.zig");
const policy_store = @import("../../protocol/federation/graph/policy_store.zig");
const graph_types = @import("../../protocol/federation/graph/types.zig");
const mutation_log = @import("../../protocol/federation/graph/mutation_log.zig");
const federation_types = @import("../../protocol/federation/types.zig");
const message_registry = @import("../../protocol/message_registry.zig");
const training_workload = @import("../../workloads/training/workload.zig");
const prometheus = @import("../../observability/prometheus.zig");
const custom_extensions = @import("custom_extensions.zig");

const GatewayReadinessRequirement = struct {
    service_type: []const u8,
    service_id: []const u8,
    required_workers: usize,
    workers_available: usize,
    ready: bool,
};

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
    services: []federation_types.ServiceRecord,

    pub fn deinit(self: *FederationPeer, allocator: Allocator) void {
        allocator.free(self.gateway_id);
        allocator.free(self.lab_id);
        allocator.free(self.base_url);
        allocator.free(self.graph_backend);
        allocator.free(self.status);
        federation_types.deinitServiceRecords(allocator, self.services);
    }
};

const NamespaceReplicationStatus = struct {
    namespace_id: []const u8,
    default_visibility: []const u8,
    allow_global_replication: bool,
    allow_raw_payload_export: bool,
    policy_updated_at: i64,
    total_mutations: usize,
    pending_mutations: usize,
    local_mutations: usize,
    shared_mutations: usize,
    global_mutations: usize,
    entity_mutations: usize,
    relation_mutations: usize,
    observation_mutations: usize,
    policy_mutations: usize,
    last_sequence_no: u64,
    replicated_through_sequence: u64,
    replication_lag: u64,
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

    pub fn updateFromConnectResponse(
        self: *Self,
        allocator: Allocator,
        endpoint: []const u8,
        response: federation_types.ConnectResponse,
    ) !void {
        var peers = std.ArrayList(FederationPeer).init(allocator);
        errdefer {
            for (peers.items) |*peer| peer.deinit(allocator);
            peers.deinit();
        }

        for (response.peers.gateways) |gateway| {
            const services = try federation_types.ownedServicesFromAdvertisements(allocator, gateway.services);
            errdefer federation_types.deinitServiceRecords(allocator, services);

            try peers.append(.{
                .gateway_id = try allocator.dupe(u8, gateway.gateway_id),
                .lab_id = try allocator.dupe(u8, gateway.lab_id),
                .base_url = try allocator.dupe(u8, gateway.base_url),
                .graph_backend = try allocator.dupe(u8, gateway.graph_backend),
                .status = try allocator.dupe(u8, gateway.status),
                .registered_services = if (gateway.services.len > 0) gateway.services.len else gateway.registered_services,
                .last_sequence_no = gateway.last_sequence_no,
                .last_replicated_sequence = gateway.last_replicated_sequence,
                .connected_at = gateway.connected_at,
                .last_seen_at = gateway.last_seen_at,
                .services = services,
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
        replaceStringLocked(self, &self.status_text, "disconnected") catch |status_err| {
            std.log.warn("Failed to update federation gateway status after {s}: {}", .{ @errorName(err), status_err });
        };
        if (self.last_error) |value| self.allocator.free(value);
        self.last_error = self.allocator.dupe(u8, @errorName(err)) catch |alloc_err| blk: {
            std.log.warn("Failed to store federation gateway error {s}: {}", .{ @errorName(err), alloc_err });
            break :blk null;
        };
    }

    pub fn updateReplicationState(self: *Self, last_sequence_no: u64, last_replicated_sequence: u64, pending_mutations: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.last_sequence_no = last_sequence_no;
        self.last_replicated_sequence = last_replicated_sequence;
        self.pending_mutations = pending_mutations;
    }

    pub fn renderStatusJson(
        self: *Self,
        allocator: Allocator,
        gateway_id: []const u8,
        lab_id: []const u8,
        configured_endpoint: ?[]const u8,
        replication_enabled: bool,
    ) ![]u8 {
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
            .federation_hub_endpoint = self.upstream_endpoint orelse configured_endpoint,
            .replication_enabled = replication_enabled,
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
            services: []const federation_types.ServiceAdvertisement,
        };

        var response_peers = std.ArrayList(ResponsePeer).init(allocator);
        defer response_peers.deinit();
        var response_service_slices = std.ArrayList([]federation_types.ServiceAdvertisement).init(allocator);
        defer {
            for (response_service_slices.items) |services| allocator.free(services);
            response_service_slices.deinit();
        }
        for (self.peers.items) |peer| {
            const response_services = try federation_types.advertisementsFromServiceRecords(allocator, peer.services);
            try response_service_slices.append(response_services);
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
                .services = response_services,
            });
        }

        return std.json.stringifyAlloc(allocator, .{
            .gateways = response_peers.items,
        }, .{});
    }

    pub fn renderServicesJson(self: *Self, allocator: Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const ResponseService = struct {
            gateway_id: []const u8,
            lab_id: []const u8,
            gateway_base_url: []const u8,
            gateway_status: []const u8,
            service_id: []const u8,
            executor_id: []const u8,
            service_type: []const u8,
            base_url: []const u8,
            auth_mode: []const u8,
            health_status: []const u8,
            job_status: []const u8,
            worker_count: usize,
            ready_worker_count: usize,
            workers_connected: usize,
            workers_ready: usize,
            workers_available: usize,
            workers_dispatchable: usize,
            worker_class: ?[]const u8,
            target_arch: ?[]const u8,
            reserved_workers: ?usize,
            max_workers: ?usize,
            capabilities: []const []const u8,
            registered_at: i64,
            updated_at: i64,
        };

        var response_services = std.ArrayList(ResponseService).init(allocator);
        defer response_services.deinit();

        for (self.peers.items) |peer| {
            for (peer.services) |service| {
                try response_services.append(.{
                    .gateway_id = peer.gateway_id,
                    .lab_id = peer.lab_id,
                    .gateway_base_url = peer.base_url,
                    .gateway_status = peer.status,
                    .service_id = service.service_id,
                    .executor_id = service.executor_id,
                    .service_type = service.service_type,
                    .base_url = service.base_url,
                    .auth_mode = service.auth_mode,
                    .health_status = service.health_status,
                    .job_status = service.job_status,
                    .worker_count = service.worker_count,
                    .ready_worker_count = service.ready_worker_count,
                    .workers_connected = service.workers_connected,
                    .workers_ready = service.workers_ready,
                    .workers_available = service.workers_available,
                    .workers_dispatchable = service.workers_available,
                    .worker_class = service.worker_class,
                    .target_arch = service.target_arch,
                    .reserved_workers = service.reserved_workers,
                    .max_workers = service.max_workers,
                    .capabilities = service.capabilities.items,
                    .registered_at = service.registered_at,
                    .updated_at = service.updated_at,
                });
            }
        }

        return std.json.stringifyAlloc(allocator, .{
            .services = response_services.items,
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

pub const LocalJobEnvelope = struct {
    allocator: Allocator,
    job_id: []u8,
    service_id: []u8,
    executor_id: []u8,
    service_type: []u8,
    job_json: []u8,

    pub fn deinit(self: *LocalJobEnvelope) void {
        self.allocator.free(self.job_id);
        self.allocator.free(self.service_id);
        self.allocator.free(self.executor_id);
        self.allocator.free(self.service_type);
        self.allocator.free(self.job_json);
    }
};

pub const LocalJobSubmitResult = struct {
    envelope: LocalJobEnvelope,
    accepted: bool = true,
    queue_position: usize = 0,

    pub fn deinit(self: *LocalJobSubmitResult) void {
        self.envelope.deinit();
    }
};

pub const LocalReservationRequest = struct {
    workers_required: ?usize = null,
};

pub const LocalReservationResult = struct {
    allocator: Allocator,
    accepted: bool,
    reservation_id: []u8,
    service_id: []u8,
    executor_id: []u8,
    service_type: []u8,
    workers_required: usize,
    reserved_at: i64,

    pub fn deinit(self: *LocalReservationResult) void {
        self.allocator.free(self.reservation_id);
        self.allocator.free(self.service_id);
        self.allocator.free(self.executor_id);
        self.allocator.free(self.service_type);
    }
};

pub const LocalJobCancelResult = struct {
    allocator: Allocator,
    accepted: bool,
    status: []u8,
    job_id: []u8,
    service_id: []u8,
    executor_id: []u8,
    service_type: []u8,

    pub fn deinit(self: *LocalJobCancelResult) void {
        self.allocator.free(self.status);
        self.allocator.free(self.job_id);
        self.allocator.free(self.service_id);
        self.allocator.free(self.executor_id);
        self.allocator.free(self.service_type);
    }
};

pub const ReadyResult = struct {
    ready: bool,
    body: []u8,
};

pub const ApiMetricsSnapshot = struct {
    requests_total: u64 = 0,
    errors_total: u64 = 0,
    duration_ns_total: u64 = 0,
};

pub const LocalReservationReleaseResult = struct {
    allocator: Allocator,
    accepted: bool,
    status: []u8,
    reservation_id: []u8,
    service_id: []u8,
    executor_id: []u8,
    service_type: []u8,

    pub fn deinit(self: *LocalReservationReleaseResult) void {
        self.allocator.free(self.status);
        self.allocator.free(self.reservation_id);
        self.allocator.free(self.service_id);
        self.allocator.free(self.executor_id);
        self.allocator.free(self.service_type);
    }
};

pub const LocalJobHook = struct {
    service_id: []const u8,
    executor_id: []const u8,
    service_type: []const u8,
    ctx: *anyopaque,
    submit: *const fn (ctx: *anyopaque, allocator: Allocator, request_body: []const u8) anyerror!LocalJobSubmitResult,
    list: *const fn (ctx: *anyopaque, allocator: Allocator) anyerror![]LocalJobEnvelope,
    lookup: *const fn (ctx: *anyopaque, allocator: Allocator, job_id: []const u8) anyerror!?LocalJobEnvelope,
    cancel: *const fn (ctx: *anyopaque, allocator: Allocator, job_id: []const u8) anyerror!?LocalJobCancelResult,
    reserve: ?*const fn (ctx: *anyopaque, allocator: Allocator, request: LocalReservationRequest) anyerror!LocalReservationResult = null,
    commit_reservation: ?*const fn (ctx: *anyopaque, allocator: Allocator, reservation_id: []const u8, request_body: []const u8) anyerror!?LocalJobSubmitResult = null,
    release_reservation: ?*const fn (ctx: *anyopaque, allocator: Allocator, reservation_id: []const u8) anyerror!?LocalReservationReleaseResult = null,
};

pub const Gateway = struct {
    allocator: Allocator,
    config: gateway_config.GatewayConfig,
    service_registry: service_registry.ServiceRegistry,
    graph: graph_adapter.GatewayGraph,
    policy_store: policy_store.GraphPolicyStore,
    event_ingester: event_ingest.EventIngester,
    federation: FederationState,
    training_jobs: ?LocalJobHook,
    rl_jobs: ?LocalJobHook,
    custom_training_jobs: ?LocalJobHook,
    started_at: i64,
    draining: std.atomic.Value(bool),

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
            .training_jobs = null,
            .rl_jobs = null,
            .custom_training_jobs = null,
            .started_at = std.time.timestamp(),
            .draining = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *Self) void {
        self.graph.deinit();
        self.policy_store.deinit();
        self.service_registry.deinit();
        self.federation.deinit();
    }

    pub fn beginDrain(self: *Self) void {
        self.draining.store(true, .release);
    }

    pub fn isDraining(self: *Self) bool {
        return self.draining.load(.acquire);
    }

    pub fn renderReadyJson(self: *Self, allocator: Allocator, auth_enabled: bool) !ReadyResult {
        const services = try self.service_registry.listServices(allocator);
        defer service_registry.ServiceRegistry.deinitServiceList(allocator, services);

        var requirements = std.ArrayList(GatewayReadinessRequirement).init(allocator);
        defer requirements.deinit();
        try appendGatewayReadinessRequirements(&requirements, self.config);

        for (requirements.items) |*requirement| {
            for (services) |service| {
                if (!std.mem.eql(u8, service.service_type, requirement.service_type)) continue;
                if (!std.mem.eql(u8, service.service_id, requirement.service_id)) continue;
                requirement.workers_available = service.workers_available;
                requirement.ready = service.workers_available >= requirement.required_workers;
                break;
            }
        }

        const draining = self.isDraining();
        var ready = !draining;
        for (requirements.items) |requirement| {
            if (!requirement.ready) {
                ready = false;
                break;
            }
        }

        const ResponseRequirement = struct {
            service_type: []const u8,
            service_id: []const u8,
            required_workers: usize,
            workers_available: usize,
            ready: bool,
        };
        var response_requirements = std.ArrayList(ResponseRequirement).init(allocator);
        defer response_requirements.deinit();
        for (requirements.items) |requirement| {
            try response_requirements.append(.{
                .service_type = requirement.service_type,
                .service_id = requirement.service_id,
                .required_workers = requirement.required_workers,
                .workers_available = requirement.workers_available,
                .ready = requirement.ready,
            });
        }

        const body = try std.json.stringifyAlloc(allocator, .{
            .ready = ready,
            .mode = "gateway",
            .gateway_id = self.config.gateway_id,
            .lab_id = self.config.lab_id,
            .graph_backend = self.config.graph_backend,
            .registered_services = services.len,
            .required_services = requirements.items.len,
            .auth_enabled = auth_enabled,
            .draining = draining,
            .requirements = response_requirements.items,
        }, .{});
        return .{ .ready = ready, .body = body };
    }

    pub fn renderPrometheusMetrics(self: *Self, allocator: Allocator, api_metrics: ApiMetricsSnapshot) ![]u8 {
        var body = std.ArrayList(u8).init(allocator);
        errdefer body.deinit();
        const writer = body.writer();

        try prometheus.writeHelpAndType(writer, "pcp_gateway_info", "Gateway identity. Labels carry gateway, lab, and graph backend.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_gateway_info", &.{
            .{ .name = "gateway_id", .value = self.config.gateway_id },
            .{ .name = "lab_id", .value = self.config.lab_id },
            .{ .name = "graph_backend", .value = self.config.graph_backend },
        }, 1);

        try prometheus.writeHelpAndType(writer, "pcp_gateway_draining", "1 when the gateway is draining after shutdown signal.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_gateway_draining", &.{}, @as(u8, if (self.isDraining()) 1 else 0));

        const services = try self.service_registry.listServices(allocator);
        defer service_registry.ServiceRegistry.deinitServiceList(allocator, services);

        try prometheus.writeHelpAndType(writer, "pcp_gateway_registered_services", "Registered local gateway services.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_gateway_registered_services", &.{}, services.len);

        try prometheus.writeHelpAndType(writer, "pcp_gateway_service_workers", "Worker counts by service and state.", "gauge");
        try prometheus.writeHelpAndType(writer, "pcp_gateway_service_reserved_workers", "Workers reserved by service.", "gauge");
        try prometheus.writeHelpAndType(writer, "pcp_gateway_service_max_workers", "Configured max workers by service; -1 means unset.", "gauge");
        try prometheus.writeHelpAndType(writer, "pcp_gateway_service_job_status", "Current service job status as a labelled gauge.", "gauge");
        for (services) |service| {
            const base_labels = [_]prometheus.Label{
                .{ .name = "service_id", .value = service.service_id },
                .{ .name = "executor_id", .value = service.executor_id },
                .{ .name = "service_type", .value = service.service_type },
                .{ .name = "worker_class", .value = service.worker_class orelse "any" },
                .{ .name = "target_arch", .value = service.target_arch orelse "default" },
            };
            try prometheus.writeSampleInt(writer, "pcp_gateway_service_workers", &(base_labels ++ .{.{ .name = "state", .value = "connected" }}), service.workers_connected);
            try prometheus.writeSampleInt(writer, "pcp_gateway_service_workers", &(base_labels ++ .{.{ .name = "state", .value = "ready" }}), service.workers_ready);
            try prometheus.writeSampleInt(writer, "pcp_gateway_service_workers", &(base_labels ++ .{.{ .name = "state", .value = "available" }}), service.workers_available);
            try prometheus.writeSampleInt(writer, "pcp_gateway_service_reserved_workers", &base_labels, service.reserved_workers orelse 0);
            const max_workers_value: isize = if (service.max_workers) |value| @intCast(value) else -1;
            try prometheus.writeSampleInt(writer, "pcp_gateway_service_max_workers", &base_labels, max_workers_value);
            try prometheus.writeSampleInt(writer, "pcp_gateway_service_job_status", &(base_labels ++ .{.{ .name = "status", .value = service.job_status }}), 1);
        }

        self.federation.mutex.lock();
        const federation_connected = self.federation.connected;
        const peer_count = self.federation.peers.items.len;
        const pending_mutations = self.federation.pending_mutations;
        const replication_lag = self.federation.last_sequence_no -| self.federation.last_replicated_sequence;
        self.federation.mutex.unlock();

        try prometheus.writeHelpAndType(writer, "pcp_gateway_federation_connected", "1 when connected to the federation hub.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_gateway_federation_connected", &.{}, @as(u8, if (federation_connected) 1 else 0));
        try prometheus.writeHelpAndType(writer, "pcp_gateway_federation_peers", "Federation peers visible to this gateway.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_gateway_federation_peers", &.{}, peer_count);
        try prometheus.writeHelpAndType(writer, "pcp_gateway_federation_pending_mutations", "Graph mutations pending federation replication.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_gateway_federation_pending_mutations", &.{}, pending_mutations);
        try prometheus.writeHelpAndType(writer, "pcp_gateway_federation_replication_lag", "Last local graph sequence minus last replicated sequence.", "gauge");
        try prometheus.writeSampleInt(writer, "pcp_gateway_federation_replication_lag", &.{}, replication_lag);

        try prometheus.writeHelpAndType(writer, "pcp_gateway_api_requests_total", "Gateway API requests handled by this process.", "counter");
        try prometheus.writeSampleInt(writer, "pcp_gateway_api_requests_total", &.{}, api_metrics.requests_total);
        try prometheus.writeHelpAndType(writer, "pcp_gateway_api_errors_total", "Gateway API requests that returned through the error path.", "counter");
        try prometheus.writeSampleInt(writer, "pcp_gateway_api_errors_total", &.{}, api_metrics.errors_total);
        try prometheus.writeHelpAndType(writer, "pcp_gateway_api_request_duration_seconds", "Gateway API request handling duration.", "summary");
        try prometheus.writeSampleInt(writer, "pcp_gateway_api_request_duration_seconds_count", &.{}, api_metrics.requests_total);
        try prometheus.writeSampleFloat(writer, "pcp_gateway_api_request_duration_seconds_sum", &.{}, @as(f64, @floatFromInt(api_metrics.duration_ns_total)) / @as(f64, @floatFromInt(std.time.ns_per_s)));

        return body.toOwnedSlice();
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
            .federation_hub_endpoint = self.config.resolvedFederationHubEndpoint(),
        }, .{});
    }

    pub fn renderCapabilitiesJson(self: *Self, allocator: Allocator) ![]u8 {
        const public_capabilities = [_][]const u8{
            "graph.query",
            "graph.mutate",
            "service.registry",
            "service.inference.proxy",
            "service.inference.query_then_infer",
            "service.rl.proxy",
            "service.training.proxy",
            message_registry.capability_inference,
            message_registry.capability_rl,
            message_registry.capability_regular_training,
            message_registry.capability_decoupled_training,
            message_registry.capability_federation,
            training_workload.capability_regular,
            training_workload.capability_decoupled_diloco,
            "federation.status",
            "federation.replication.inspect",
            "events.ingest",
        };
        const capabilities = public_capabilities ++ custom_extensions.capabilities;

        return std.json.stringifyAlloc(allocator, .{
            .gateway_id = self.config.gateway_id,
            .lab_id = self.config.lab_id,
            .capabilities = &capabilities,
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
            self.config.resolvedFederationHubEndpoint(),
            federationReplicationEnabled(self.config),
        );
    }

    pub fn renderFederationReplicationJson(self: *Self, allocator: Allocator) ![]u8 {
        const last_sequence_no = self.graph.lastSequence();
        const last_replicated_sequence = self.graph.lastReplicatedSequence();
        const pending_mutations = self.graph.countPendingReplications();
        self.federation.updateReplicationState(last_sequence_no, last_replicated_sequence, pending_mutations);

        const stats = try self.graph.listNamespaceMutationStats(allocator, last_replicated_sequence);
        defer graph_adapter.GatewayGraph.deinitNamespaceMutationStats(allocator, stats);

        const policies = try self.policy_store.listSnapshots(allocator);
        defer allocator.free(policies);

        var namespaces = std.ArrayList(NamespaceReplicationStatus).init(allocator);
        defer namespaces.deinit();

        for (stats) |stat| {
            const snapshot = self.policy_store.getSnapshot(stat.namespace_id, defaultPolicyVisibility(self.config));
            try namespaces.append(namespaceResponseFromStats(stat, snapshot));
        }

        for (policies) |policy| {
            if (containsNamespace(namespaces.items, policy.namespace_id)) continue;
            try namespaces.append(.{
                .namespace_id = policy.namespace_id,
                .default_visibility = policy.default_visibility,
                .allow_global_replication = policy.allow_global_replication,
                .allow_raw_payload_export = policy.allow_raw_payload_export,
                .policy_updated_at = policy.updated_at,
                .total_mutations = 0,
                .pending_mutations = 0,
                .local_mutations = 0,
                .shared_mutations = 0,
                .global_mutations = 0,
                .entity_mutations = 0,
                .relation_mutations = 0,
                .observation_mutations = 0,
                .policy_mutations = 0,
                .last_sequence_no = 0,
                .replicated_through_sequence = 0,
                .replication_lag = 0,
            });
        }

        self.federation.mutex.lock();
        defer self.federation.mutex.unlock();

        return std.json.stringifyAlloc(allocator, .{
            .gateway_id = self.config.gateway_id,
            .lab_id = self.config.lab_id,
            .connected = self.federation.connected,
            .status = self.federation.status_text,
            .federation_hub_endpoint = self.federation.upstream_endpoint orelse self.config.resolvedFederationHubEndpoint(),
            .last_sync_at = self.federation.last_sync_at,
            .last_error = self.federation.last_error,
            .summary = .{
                .namespace_count = namespaces.items.len,
                .last_sequence_no = last_sequence_no,
                .last_replicated_sequence = last_replicated_sequence,
                .pending_mutations = pending_mutations,
                .replication_lag = last_sequence_no -| last_replicated_sequence,
            },
            .namespaces = namespaces.items,
        }, .{});
    }

    pub fn setLocalJobHook(self: *Self, service_type: []const u8, hook: ?LocalJobHook) void {
        if (std.mem.eql(u8, service_type, service_registry.ServiceType.training.asString())) {
            self.training_jobs = hook;
        } else if (std.mem.eql(u8, service_type, service_registry.ServiceType.rl.asString())) {
            self.rl_jobs = hook;
        } else if (custom_extensions.isCustomTrainingServiceType(service_type)) {
            self.custom_training_jobs = hook;
        }
    }

    pub fn localJobHookForExecutor(self: *Self, executor_id: []const u8, service_type: []const u8) ?LocalJobHook {
        const hook = if (std.mem.eql(u8, service_type, service_registry.ServiceType.training.asString()))
            self.training_jobs
        else if (std.mem.eql(u8, service_type, service_registry.ServiceType.rl.asString()))
            self.rl_jobs
        else if (custom_extensions.isCustomTrainingServiceType(service_type))
            self.custom_training_jobs
        else
            null;
        const selected = hook orelse return null;

        if (!std.mem.eql(u8, selected.executor_id, executor_id)) return null;
        return selected;
    }

    pub fn localJobHookByJobId(self: *Self, job_id: []const u8) ?LocalJobHook {
        if (self.training_jobs) |hook| {
            if (identity.jobBelongsToExecutor(job_id, hook.executor_id)) {
                return hook;
            }
        }
        if (self.rl_jobs) |hook| {
            if (identity.jobBelongsToExecutor(job_id, hook.executor_id)) {
                return hook;
            }
        }
        if (self.custom_training_jobs) |hook| {
            if (identity.jobBelongsToExecutor(job_id, hook.executor_id)) {
                return hook;
            }
        }
        return null;
    }

    pub fn localJobHookByReservationId(self: *Self, reservation_id: []const u8) ?LocalJobHook {
        if (self.training_jobs) |hook| {
            if (identity.reservationBelongsToExecutor(reservation_id, hook.executor_id)) {
                return hook;
            }
        }
        if (self.rl_jobs) |hook| {
            if (identity.reservationBelongsToExecutor(reservation_id, hook.executor_id)) {
                return hook;
            }
        }
        if (self.custom_training_jobs) |hook| {
            if (identity.reservationBelongsToExecutor(reservation_id, hook.executor_id)) {
                return hook;
            }
        }
        return null;
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

fn containsNamespace(namespaces: []const NamespaceReplicationStatus, namespace_id: []const u8) bool {
    for (namespaces) |namespace| {
        if (std.mem.eql(u8, namespace.namespace_id, namespace_id)) return true;
    }
    return false;
}

fn appendGatewayReadinessRequirements(
    requirements: *std.ArrayList(GatewayReadinessRequirement),
    config: gateway_config.GatewayConfig,
) !void {
    if (config.resolvedEmbeddedInference()) |inference| {
        try requirements.append(.{
            .service_type = service_registry.ServiceType.inference.asString(),
            .service_id = inference.service_id orelse "inference-main",
            .required_workers = 1,
            .workers_available = 0,
            .ready = false,
        });
    }
    if (config.resolvedEmbeddedTraining()) |training| {
        try requirements.append(.{
            .service_type = service_registry.ServiceType.training.asString(),
            .service_id = training.service_id orelse "training-main",
            .required_workers = training.workers orelse 1,
            .workers_available = 0,
            .ready = false,
        });
    }
    if (config.resolvedEmbeddedRL()) |rl| {
        try requirements.append(.{
            .service_type = service_registry.ServiceType.rl.asString(),
            .service_id = rl.service_id orelse "rl-main",
            .required_workers = rl.workers orelse 1,
            .workers_available = 0,
            .ready = false,
        });
    }
    try custom_extensions.appendReadinessRequirements(requirements, config);
}

fn federationReplicationEnabled(config: gateway_config.GatewayConfig) bool {
    const federation = config.federation orelse return false;
    return federation.enabled and config.resolvedFederationHubEndpoint() != null;
}

fn namespaceResponseFromStats(
    stat: mutation_log.NamespaceMutationStats,
    snapshot: policy_store.NamespacePolicySnapshot,
) NamespaceReplicationStatus {
    return .{
        .namespace_id = stat.namespace_id,
        .default_visibility = snapshot.default_visibility,
        .allow_global_replication = snapshot.allow_global_replication,
        .allow_raw_payload_export = snapshot.allow_raw_payload_export,
        .policy_updated_at = snapshot.updated_at,
        .total_mutations = stat.total_mutations,
        .pending_mutations = stat.pending_mutations,
        .local_mutations = stat.local_mutations,
        .shared_mutations = stat.shared_mutations,
        .global_mutations = stat.global_mutations,
        .entity_mutations = stat.entity_mutations,
        .relation_mutations = stat.relation_mutations,
        .observation_mutations = stat.observation_mutations,
        .policy_mutations = stat.policy_mutations,
        .last_sequence_no = stat.last_sequence_no,
        .replicated_through_sequence = stat.replicated_through_sequence,
        .replication_lag = stat.last_sequence_no -| stat.replicated_through_sequence,
    };
}
