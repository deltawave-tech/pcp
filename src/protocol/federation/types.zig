const std = @import("std");

const Allocator = std.mem.Allocator;

pub const NamespacePolicySnapshot = struct {
    namespace_id: []const u8,
    default_visibility: []const u8,
    allow_global_replication: bool,
    allow_raw_payload_export: bool,
    updated_at: i64,
};

pub const MutationBatchItem = struct {
    sequence_no: u64,
    mutation_id: []const u8,
    namespace_id: []const u8,
    mutation_type: []const u8,
    target_id: []const u8,
    payload_json: []const u8,
    visibility: []const u8,
    provenance_json: []const u8,
    timestamp: i64,
};

pub const MutationBatchRequest = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    last_sequence_no: u64,
    last_replicated_sequence: u64,
    namespace_policies: []const NamespacePolicySnapshot = &.{},
    mutations: []const MutationBatchItem,
};

pub const MutationBatchAck = struct {
    accepted: bool,
    acked_sequence_no: u64,
    applied_count: usize,
    duplicate_count: usize,
};

pub const PlacementRequest = struct {
    gateway_id: ?[]const u8 = null,
    service_id: ?[]const u8 = null,
    executor_id: ?[]const u8 = null,
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    workers_required: ?usize = null,
};

pub const ServiceAdvertisement = struct {
    service_id: []const u8,
    executor_id: []const u8,
    service_type: []const u8,
    base_url: []const u8,
    auth_mode: []const u8 = "none",
    health_status: []const u8 = "unknown",
    job_status: []const u8 = "unknown",
    worker_count: usize = 0,
    ready_worker_count: usize = 0,
    workers_connected: ?usize = null,
    workers_ready: ?usize = null,
    workers_available: ?usize = null,
    workers_dispatchable: ?usize = null,
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    reserved_workers: ?usize = null,
    max_workers: ?usize = null,
    capabilities: []const []const u8 = &.{},
    registered_at: i64 = 0,
    updated_at: i64 = 0,
};

pub const ServiceRecord = struct {
    service_id: []u8,
    executor_id: []u8,
    service_type: []u8,
    base_url: []u8,
    auth_mode: []u8,
    health_status: []u8,
    job_status: []u8,
    worker_count: usize,
    ready_worker_count: usize,
    workers_connected: usize,
    workers_ready: usize,
    workers_available: usize,
    worker_class: ?[]u8,
    target_arch: ?[]u8,
    reserved_workers: ?usize,
    max_workers: ?usize,
    capabilities: std.ArrayList([]u8),
    registered_at: i64,
    updated_at: i64,

    pub fn deinit(self: *ServiceRecord, allocator: Allocator) void {
        allocator.free(self.service_id);
        allocator.free(self.executor_id);
        allocator.free(self.service_type);
        allocator.free(self.base_url);
        allocator.free(self.auth_mode);
        allocator.free(self.health_status);
        allocator.free(self.job_status);
        if (self.worker_class) |value| allocator.free(value);
        if (self.target_arch) |value| allocator.free(value);
        for (self.capabilities.items) |capability| {
            allocator.free(capability);
        }
        self.capabilities.deinit();
    }

    pub fn clone(self: ServiceRecord, allocator: Allocator) !ServiceRecord {
        var capabilities = std.ArrayList([]u8).init(allocator);
        errdefer {
            for (capabilities.items) |capability| allocator.free(capability);
            capabilities.deinit();
        }
        for (self.capabilities.items) |capability| {
            try capabilities.append(try allocator.dupe(u8, capability));
        }

        return .{
            .service_id = try allocator.dupe(u8, self.service_id),
            .executor_id = try allocator.dupe(u8, self.executor_id),
            .service_type = try allocator.dupe(u8, self.service_type),
            .base_url = try allocator.dupe(u8, self.base_url),
            .auth_mode = try allocator.dupe(u8, self.auth_mode),
            .health_status = try allocator.dupe(u8, self.health_status),
            .job_status = try allocator.dupe(u8, self.job_status),
            .worker_count = self.worker_count,
            .ready_worker_count = self.ready_worker_count,
            .workers_connected = self.workers_connected,
            .workers_ready = self.workers_ready,
            .workers_available = self.workers_available,
            .worker_class = if (self.worker_class) |value| try allocator.dupe(u8, value) else null,
            .target_arch = if (self.target_arch) |value| try allocator.dupe(u8, value) else null,
            .reserved_workers = self.reserved_workers,
            .max_workers = self.max_workers,
            .capabilities = capabilities,
            .registered_at = self.registered_at,
            .updated_at = self.updated_at,
        };
    }
};

pub const GatewayAdvertisement = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    base_url: []const u8,
    graph_backend: []const u8,
    status: []const u8,
    registered_services: usize = 0,
    last_sequence_no: u64 = 0,
    last_replicated_sequence: u64 = 0,
    connected_at: i64 = 0,
    last_seen_at: i64 = 0,
    services: []const ServiceAdvertisement = &.{},
};

pub const PeersResponse = struct {
    gateways: []const GatewayAdvertisement = &.{},
};

pub const ConnectResponse = struct {
    accepted: bool,
    gateway: GatewayAdvertisement,
    peers: PeersResponse,
};

pub fn ownedServicesFromAdvertisements(
    allocator: Allocator,
    services: []const ServiceAdvertisement,
) ![]ServiceRecord {
    const records = try allocator.alloc(ServiceRecord, services.len);
    errdefer allocator.free(records);

    var initialized: usize = 0;
    errdefer {
        for (records[0..initialized]) |*record| record.deinit(allocator);
    }

    for (services, 0..) |service, idx| {
        records[idx] = try ownedServiceFromAdvertisement(allocator, service);
        initialized += 1;
    }
    return records;
}

pub fn cloneServiceRecords(
    allocator: Allocator,
    services: []const ServiceRecord,
) ![]ServiceRecord {
    const clone = try allocator.alloc(ServiceRecord, services.len);
    errdefer allocator.free(clone);

    var initialized: usize = 0;
    errdefer {
        for (clone[0..initialized]) |*service| service.deinit(allocator);
    }

    for (services, 0..) |service, idx| {
        clone[idx] = try service.clone(allocator);
        initialized += 1;
    }
    return clone;
}

pub fn deinitServiceRecords(allocator: Allocator, services: []ServiceRecord) void {
    for (services) |*service| service.deinit(allocator);
    allocator.free(services);
}

pub fn advertisementsFromServiceRecords(
    allocator: Allocator,
    services: []const ServiceRecord,
) ![]ServiceAdvertisement {
    const advertisements = try allocator.alloc(ServiceAdvertisement, services.len);
    errdefer allocator.free(advertisements);

    for (services, 0..) |service, idx| {
        advertisements[idx] = .{
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
        };
    }

    return advertisements;
}

fn ownedServiceFromAdvertisement(
    allocator: Allocator,
    service: ServiceAdvertisement,
) !ServiceRecord {
    const counts = normalizeWorkerCounts(service);
    var capabilities = std.ArrayList([]u8).init(allocator);
    errdefer {
        for (capabilities.items) |capability| allocator.free(capability);
        capabilities.deinit();
    }
    for (service.capabilities) |capability| {
        try capabilities.append(try allocator.dupe(u8, capability));
    }

    return .{
        .service_id = try allocator.dupe(u8, service.service_id),
        .executor_id = try allocator.dupe(u8, service.executor_id),
        .service_type = try allocator.dupe(u8, service.service_type),
        .base_url = try allocator.dupe(u8, service.base_url),
        .auth_mode = try allocator.dupe(u8, service.auth_mode),
        .health_status = try allocator.dupe(u8, service.health_status),
        .job_status = try allocator.dupe(u8, service.job_status),
        .worker_count = counts.workers_connected,
        .ready_worker_count = counts.workers_available,
        .workers_connected = counts.workers_connected,
        .workers_ready = counts.workers_ready,
        .workers_available = counts.workers_available,
        .worker_class = if (service.worker_class) |value| try allocator.dupe(u8, value) else null,
        .target_arch = if (service.target_arch) |value| try allocator.dupe(u8, value) else null,
        .reserved_workers = service.reserved_workers,
        .max_workers = service.max_workers,
        .capabilities = capabilities,
        .registered_at = service.registered_at,
        .updated_at = service.updated_at,
    };
}

const NormalizedWorkerCounts = struct {
    workers_connected: usize,
    workers_ready: usize,
    workers_available: usize,
};

fn normalizeWorkerCounts(service: ServiceAdvertisement) NormalizedWorkerCounts {
    var workers_connected = service.workers_connected orelse service.worker_count;
    var workers_ready = service.workers_ready orelse service.ready_worker_count;
    const workers_available = service.workers_dispatchable orelse service.workers_available orelse service.ready_worker_count;

    if (workers_ready < workers_available) workers_ready = workers_available;
    if (workers_connected < workers_ready) workers_connected = workers_ready;

    return .{
        .workers_connected = workers_connected,
        .workers_ready = workers_ready,
        .workers_available = workers_available,
    };
}
