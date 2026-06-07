const std = @import("std");

const Allocator = std.mem.Allocator;
const federation_types = @import("../../protocol/federation/types.zig");
const gateway_registry = @import("gateway_registry.zig");
const graph_store = @import("../../protocol/federation/graph/store.zig");
const mutation_log = @import("../../protocol/federation/graph/mutation_log.zig");
const policy_store = @import("../../protocol/federation/graph/policy_store.zig");
const graph_types = @import("../../protocol/federation/graph/types.zig");

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

pub const GlobalJobRecord = struct {
    global_job_id: []u8,
    gateway_id: []u8,
    lab_id: []u8,
    gateway_base_url: []u8,
    service_id: []u8,
    executor_id: []u8,
    service_type: []u8,
    child_job_id: []u8,
    submitted_at: i64,
    updated_at: i64,

    pub fn deinit(self: *GlobalJobRecord, allocator: Allocator) void {
        allocator.free(self.global_job_id);
        allocator.free(self.gateway_id);
        allocator.free(self.lab_id);
        allocator.free(self.gateway_base_url);
        allocator.free(self.service_id);
        allocator.free(self.executor_id);
        allocator.free(self.service_type);
        allocator.free(self.child_job_id);
    }

    pub fn clone(self: GlobalJobRecord, allocator: Allocator) !GlobalJobRecord {
        return .{
            .global_job_id = try allocator.dupe(u8, self.global_job_id),
            .gateway_id = try allocator.dupe(u8, self.gateway_id),
            .lab_id = try allocator.dupe(u8, self.lab_id),
            .gateway_base_url = try allocator.dupe(u8, self.gateway_base_url),
            .service_id = try allocator.dupe(u8, self.service_id),
            .executor_id = try allocator.dupe(u8, self.executor_id),
            .service_type = try allocator.dupe(u8, self.service_type),
            .child_job_id = try allocator.dupe(u8, self.child_job_id),
            .submitted_at = self.submitted_at,
            .updated_at = self.updated_at,
        };
    }
};

pub const GlobalJobStore = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    next_sequence: usize,
    jobs: std.ArrayList(GlobalJobRecord),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .next_sequence = 1,
            .jobs = std.ArrayList(GlobalJobRecord).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.jobs.items) |*job| {
            job.deinit(self.allocator);
        }
        self.jobs.deinit();
    }

    pub fn create(
        self: *Self,
        allocator: Allocator,
        gateway_id: []const u8,
        lab_id: []const u8,
        gateway_base_url: []const u8,
        service_id: []const u8,
        executor_id: []const u8,
        service_type: []const u8,
        child_job_id: []const u8,
    ) !GlobalJobRecord {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.timestamp();
        const record = GlobalJobRecord{
            .global_job_id = try std.fmt.allocPrint(self.allocator, "global-job-{d}", .{self.next_sequence}),
            .gateway_id = try self.allocator.dupe(u8, gateway_id),
            .lab_id = try self.allocator.dupe(u8, lab_id),
            .gateway_base_url = try self.allocator.dupe(u8, gateway_base_url),
            .service_id = try self.allocator.dupe(u8, service_id),
            .executor_id = try self.allocator.dupe(u8, executor_id),
            .service_type = try self.allocator.dupe(u8, service_type),
            .child_job_id = try self.allocator.dupe(u8, child_job_id),
            .submitted_at = now,
            .updated_at = now,
        };
        self.next_sequence += 1;
        try self.jobs.append(record);
        return try record.clone(allocator);
    }

    pub fn find(self: *Self, allocator: Allocator, global_job_id: []const u8) !?GlobalJobRecord {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.jobs.items) |job| {
            if (std.mem.eql(u8, job.global_job_id, global_job_id)) {
                return try job.clone(allocator);
            }
        }
        return null;
    }

    pub fn list(self: *Self, allocator: Allocator) ![]GlobalJobRecord {
        self.mutex.lock();
        defer self.mutex.unlock();

        const out = try allocator.alloc(GlobalJobRecord, self.jobs.items.len);
        errdefer allocator.free(out);
        for (self.jobs.items, 0..) |job, idx| {
            out[idx] = try job.clone(allocator);
        }
        return out;
    }

    pub fn deinitList(allocator: Allocator, jobs: []GlobalJobRecord) void {
        for (jobs) |*job| {
            job.deinit(allocator);
        }
        allocator.free(jobs);
    }
};

pub const FederationHub = struct {
    allocator: Allocator,
    registry: gateway_registry.GatewayRegistry,
    graph: graph_store.GraphStore,
    policy_store: policy_store.GraphPolicyStore,
    jobs: GlobalJobStore,
    started_at: i64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .registry = gateway_registry.GatewayRegistry.init(allocator),
            .graph = graph_store.GraphStore.init(allocator, .{
                .backend = "memory",
                .gateway_id = "federation-hub",
                .lab_id = "federation",
                .neo4j = null,
            }) catch @panic("failed to initialize federation graph store"),
            .policy_store = policy_store.GraphPolicyStore.init(allocator, null) catch @panic("failed to initialize federation policy store"),
            .jobs = GlobalJobStore.init(allocator),
            .started_at = std.time.timestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.graph.deinit();
        self.policy_store.deinit();
        self.registry.deinit();
        self.jobs.deinit();
    }

    pub fn renderReadyJson(self: *Self, allocator: Allocator, auth_enabled: bool) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .ready = true,
            .mode = "federation-hub",
            .registered_gateways = self.registry.count(),
            .auth_enabled = auth_enabled,
        }, .{});
    }

    pub fn renderControllerJson(self: *Self, allocator: Allocator, auth_enabled: bool) ![]u8 {
        return std.json.stringifyAlloc(allocator, .{
            .mode = "federation-hub",
            .status = "running",
            .registered_gateways = self.registry.count(),
            .started_at = self.started_at,
            .auth_enabled = auth_enabled,
        }, .{});
    }

    pub fn renderGlobalGraphStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        const counts = self.graph.counts();
        return std.json.stringifyAlloc(allocator, .{
            .mode = "federation-hub",
            .registered_gateways = self.registry.count(),
            .replication_enabled = true,
            .max_replication_lag = self.registry.maxReplicationLag(),
            .global_graph = .{
                .namespaces = counts.namespaces,
                .entities = counts.entities,
                .relations = counts.relations,
                .observations = counts.observations,
            },
        }, .{});
    }

    pub fn renderGlobalGraphQueryJson(self: *Self, allocator: Allocator, request: graph_types.QueryRequest) ![]u8 {
        return self.graph.renderQueryJson(allocator, request);
    }

    pub fn renderPoliciesJson(self: *Self, allocator: Allocator) ![]u8 {
        return self.policy_store.renderJson(allocator);
    }

    pub fn renderServicesJson(self: *Self, allocator: Allocator) ![]u8 {
        return self.registry.renderServicesJson(allocator);
    }

    pub fn renderReplicationJson(self: *Self, allocator: Allocator) ![]u8 {
        const stats = try self.graph.listNamespaceMutationStats(allocator, null);
        defer graph_store.GraphStore.deinitNamespaceMutationStats(allocator, stats);

        const policies = try self.policy_store.listSnapshots(allocator);
        defer allocator.free(policies);

        const gateways = try self.registry.list(allocator);
        defer gateway_registry.GatewayRegistry.deinitList(allocator, gateways);

        var namespaces = std.ArrayList(NamespaceReplicationStatus).init(allocator);
        defer namespaces.deinit();

        for (stats) |stat| {
            const snapshot = self.policy_store.getSnapshot(stat.namespace_id, .local);
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

        const ResponseGateway = struct {
            gateway_id: []const u8,
            lab_id: []const u8,
            base_url: []const u8,
            graph_backend: []const u8,
            status: []const u8,
            registered_services: usize,
            last_sequence_no: u64,
            last_replicated_sequence: u64,
            replication_lag: u64,
            connected_at: i64,
            last_seen_at: i64,
        };

        var response_gateways = std.ArrayList(ResponseGateway).init(allocator);
        defer response_gateways.deinit();
        for (gateways) |gateway| {
            try response_gateways.append(.{
                .gateway_id = gateway.gateway_id,
                .lab_id = gateway.lab_id,
                .base_url = gateway.base_url,
                .graph_backend = gateway.graph_backend,
                .status = gateway.status,
                .registered_services = gateway.registered_services,
                .last_sequence_no = gateway.last_sequence_no,
                .last_replicated_sequence = gateway.last_replicated_sequence,
                .replication_lag = gateway.last_sequence_no -| gateway.last_replicated_sequence,
                .connected_at = gateway.connected_at,
                .last_seen_at = gateway.last_seen_at,
            });
        }

        return std.json.stringifyAlloc(allocator, .{
            .mode = "federation-hub",
            .registered_gateways = gateways.len,
            .max_replication_lag = self.registry.maxReplicationLag(),
            .global_graph = .{
                .namespaces = namespaces.items.len,
                .mutations = totalMutations(namespaces.items),
            },
            .gateways = response_gateways.items,
            .namespaces = namespaces.items,
        }, .{});
    }

    pub fn applyMutationBatch(self: *Self, allocator: Allocator, request: federation_types.MutationBatchRequest) !federation_types.MutationBatchAck {
        for (request.namespace_policies) |snapshot| {
            try self.policy_store.applySnapshot(.{
                .namespace_id = snapshot.namespace_id,
                .default_visibility = snapshot.default_visibility,
                .allow_global_replication = snapshot.allow_global_replication,
                .allow_raw_payload_export = snapshot.allow_raw_payload_export,
                .updated_at = snapshot.updated_at,
            });
        }

        for (request.mutations) |mutation| {
            const visibility = graph_types.Visibility.parse(mutation.visibility) orelse return error.InvalidVisibility;
            if (!self.policy_store.allowsIncoming(
                mutation.namespace_id,
                .local,
                visibility,
                std.mem.eql(u8, mutation.mutation_type, "append_observation"),
            )) {
                return error.PolicyRejected;
            }
        }

        const ack = try self.graph.applyReplicatedBatch(allocator, request.gateway_id, request.lab_id, request.mutations);
        if (!self.registry.markReplicated(request.gateway_id, ack.acked_sequence_no, request.last_sequence_no)) {
            return error.UnknownGateway;
        }
        return ack;
    }
};

fn containsNamespace(namespaces: []const NamespaceReplicationStatus, namespace_id: []const u8) bool {
    for (namespaces) |namespace| {
        if (std.mem.eql(u8, namespace.namespace_id, namespace_id)) return true;
    }
    return false;
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
        .pending_mutations = 0,
        .local_mutations = stat.local_mutations,
        .shared_mutations = stat.shared_mutations,
        .global_mutations = stat.global_mutations,
        .entity_mutations = stat.entity_mutations,
        .relation_mutations = stat.relation_mutations,
        .observation_mutations = stat.observation_mutations,
        .policy_mutations = stat.policy_mutations,
        .last_sequence_no = stat.last_sequence_no,
        .replicated_through_sequence = stat.last_sequence_no,
        .replication_lag = 0,
    };
}

fn totalMutations(namespaces: []const NamespaceReplicationStatus) usize {
    var total: usize = 0;
    for (namespaces) |namespace| total += namespace.total_mutations;
    return total;
}
