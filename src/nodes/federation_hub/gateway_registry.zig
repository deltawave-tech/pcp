const std = @import("std");

const Allocator = std.mem.Allocator;
const federation_types = @import("../../protocol/federation/types.zig");

pub const ConnectRequest = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    base_url: []const u8,
    graph_backend: []const u8,
    registered_services: usize = 0,
    services: []const federation_types.ServiceAdvertisement = &.{},
    last_sequence_no: u64 = 0,
    last_replicated_sequence: u64 = 0,
    status: ?[]const u8 = null,
};

pub const RegisteredGateway = struct {
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

    pub fn deinit(self: *RegisteredGateway, allocator: Allocator) void {
        allocator.free(self.gateway_id);
        allocator.free(self.lab_id);
        allocator.free(self.base_url);
        allocator.free(self.graph_backend);
        allocator.free(self.status);
        federation_types.deinitServiceRecords(allocator, self.services);
    }
};

pub const SelectedService = struct {
    gateway_id: []u8,
    lab_id: []u8,
    gateway_base_url: []u8,
    service: federation_types.ServiceRecord,

    pub fn deinit(self: *SelectedService, allocator: Allocator) void {
        allocator.free(self.gateway_id);
        allocator.free(self.lab_id);
        allocator.free(self.gateway_base_url);
        self.service.deinit(allocator);
    }
};

pub const GatewayRegistry = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    gateways: std.ArrayList(RegisteredGateway),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .gateways = std.ArrayList(RegisteredGateway).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.gateways.items) |*gateway| {
            gateway.deinit(self.allocator);
        }
        self.gateways.deinit();
    }

    pub fn upsert(self: *Self, request: ConnectRequest) !RegisteredGateway {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.timestamp();
        for (self.gateways.items) |*gateway| {
            if (!std.mem.eql(u8, gateway.gateway_id, request.gateway_id)) continue;
            try replaceString(self.allocator, &gateway.lab_id, request.lab_id);
            try replaceString(self.allocator, &gateway.base_url, request.base_url);
            try replaceString(self.allocator, &gateway.graph_backend, request.graph_backend);
            try replaceString(self.allocator, &gateway.status, request.status orelse "connected");
            try replaceServices(self.allocator, &gateway.services, request.services);
            gateway.registered_services = if (request.services.len > 0) request.services.len else request.registered_services;
            gateway.last_sequence_no = request.last_sequence_no;
            gateway.last_replicated_sequence = @max(gateway.last_replicated_sequence, request.last_replicated_sequence);
            gateway.last_seen_at = now;
            return try cloneGateway(self.allocator, gateway.*);
        }

        const services = try federation_types.ownedServicesFromAdvertisements(self.allocator, request.services);
        errdefer federation_types.deinitServiceRecords(self.allocator, services);

        const gateway = RegisteredGateway{
            .gateway_id = try self.allocator.dupe(u8, request.gateway_id),
            .lab_id = try self.allocator.dupe(u8, request.lab_id),
            .base_url = try self.allocator.dupe(u8, request.base_url),
            .graph_backend = try self.allocator.dupe(u8, request.graph_backend),
            .status = try self.allocator.dupe(u8, request.status orelse "connected"),
            .registered_services = if (request.services.len > 0) request.services.len else request.registered_services,
            .last_sequence_no = request.last_sequence_no,
            .last_replicated_sequence = request.last_replicated_sequence,
            .connected_at = now,
            .last_seen_at = now,
            .services = services,
        };
        try self.gateways.append(gateway);
        return try cloneGateway(self.allocator, gateway);
    }

    pub fn count(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.gateways.items.len;
    }

    pub fn list(self: *Self, allocator: Allocator) ![]RegisteredGateway {
        self.mutex.lock();
        defer self.mutex.unlock();

        const gateways = try allocator.alloc(RegisteredGateway, self.gateways.items.len);
        errdefer allocator.free(gateways);
        for (self.gateways.items, 0..) |gateway, idx| {
            gateways[idx] = try cloneGateway(allocator, gateway);
        }
        return gateways;
    }

    pub fn selectServiceByType(
        self: *Self,
        allocator: Allocator,
        service_type: []const u8,
        placement: federation_types.PlacementRequest,
    ) !?SelectedService {
        self.mutex.lock();
        defer self.mutex.unlock();

        var best_gateway_index: ?usize = null;
        var best_service_index: usize = 0;

        for (self.gateways.items, 0..) |gateway, gateway_idx| {
            if (placement.gateway_id) |gateway_id| {
                if (!std.mem.eql(u8, gateway.gateway_id, gateway_id)) continue;
            }
            if (!std.mem.eql(u8, gateway.status, "connected")) continue;

            for (gateway.services, 0..) |service, service_idx| {
                if (!std.mem.eql(u8, service.service_type, service_type)) continue;
                if (!matchesPlacement(service, placement)) continue;

                if (best_gateway_index == null or isBetterCandidate(
                    service,
                    self.gateways.items[best_gateway_index.?].services[best_service_index],
                    placement.workers_required,
                )) {
                    best_gateway_index = gateway_idx;
                    best_service_index = service_idx;
                }
            }
        }

        const gateway_idx = best_gateway_index orelse return null;
        const gateway = self.gateways.items[gateway_idx];
        const service = gateway.services[best_service_index];
        return .{
            .gateway_id = try allocator.dupe(u8, gateway.gateway_id),
            .lab_id = try allocator.dupe(u8, gateway.lab_id),
            .gateway_base_url = try allocator.dupe(u8, gateway.base_url),
            .service = try service.clone(allocator),
        };
    }

    pub fn selectServiceByCapability(
        self: *Self,
        allocator: Allocator,
        required_capability: []const u8,
        placement: federation_types.PlacementRequest,
    ) !?SelectedService {
        self.mutex.lock();
        defer self.mutex.unlock();

        var best_gateway_index: ?usize = null;
        var best_service_index: usize = 0;

        for (self.gateways.items, 0..) |gateway, gateway_idx| {
            if (placement.gateway_id) |gateway_id| {
                if (!std.mem.eql(u8, gateway.gateway_id, gateway_id)) continue;
            }
            if (!std.mem.eql(u8, gateway.status, "connected")) continue;

            for (gateway.services, 0..) |service, service_idx| {
                if (!serviceHasCapability(service, required_capability)) continue;
                if (!matchesPlacement(service, placement)) continue;

                if (best_gateway_index == null or isBetterCandidate(
                    service,
                    self.gateways.items[best_gateway_index.?].services[best_service_index],
                    placement.workers_required,
                )) {
                    best_gateway_index = gateway_idx;
                    best_service_index = service_idx;
                }
            }
        }

        const gateway_idx = best_gateway_index orelse return null;
        const gateway = self.gateways.items[gateway_idx];
        const service = gateway.services[best_service_index];
        return .{
            .gateway_id = try allocator.dupe(u8, gateway.gateway_id),
            .lab_id = try allocator.dupe(u8, gateway.lab_id),
            .gateway_base_url = try allocator.dupe(u8, gateway.base_url),
            .service = try service.clone(allocator),
        };
    }

    pub fn deinitList(allocator: Allocator, gateways: []RegisteredGateway) void {
        for (gateways) |*gateway| {
            gateway.deinit(allocator);
        }
        allocator.free(gateways);
    }

    pub fn renderPeersJson(self: *Self, allocator: Allocator) ![]u8 {
        const gateways = try self.list(allocator);
        defer deinitList(allocator, gateways);

        const ResponseGateway = struct {
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

        var response_gateways = std.ArrayList(ResponseGateway).init(allocator);
        defer response_gateways.deinit();
        var response_service_slices = std.ArrayList([]federation_types.ServiceAdvertisement).init(allocator);
        defer {
            for (response_service_slices.items) |services| allocator.free(services);
            response_service_slices.deinit();
        }
        for (gateways) |gateway| {
            const response_services = try federation_types.advertisementsFromServiceRecords(allocator, gateway.services);
            try response_service_slices.append(response_services);
            try response_gateways.append(.{
                .gateway_id = gateway.gateway_id,
                .lab_id = gateway.lab_id,
                .base_url = gateway.base_url,
                .graph_backend = gateway.graph_backend,
                .status = gateway.status,
                .registered_services = gateway.registered_services,
                .last_sequence_no = gateway.last_sequence_no,
                .last_replicated_sequence = gateway.last_replicated_sequence,
                .connected_at = gateway.connected_at,
                .last_seen_at = gateway.last_seen_at,
                .services = response_services,
            });
        }

        return std.json.stringifyAlloc(allocator, .{
            .gateways = response_gateways.items,
        }, .{});
    }

    pub fn renderServicesJson(self: *Self, allocator: Allocator) ![]u8 {
        const gateways = try self.list(allocator);
        defer deinitList(allocator, gateways);

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

        for (gateways) |gateway| {
            for (gateway.services) |service| {
                try response_services.append(.{
                    .gateway_id = gateway.gateway_id,
                    .lab_id = gateway.lab_id,
                    .gateway_base_url = gateway.base_url,
                    .gateway_status = gateway.status,
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

    pub fn markReplicated(self: *Self, gateway_id: []const u8, acked_sequence_no: u64, last_sequence_no: u64) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.gateways.items) |*gateway| {
            if (!std.mem.eql(u8, gateway.gateway_id, gateway_id)) continue;
            gateway.last_replicated_sequence = @max(gateway.last_replicated_sequence, acked_sequence_no);
            gateway.last_sequence_no = @max(gateway.last_sequence_no, last_sequence_no);
            gateway.last_seen_at = std.time.timestamp();
            return true;
        }
        return false;
    }

    pub fn maxReplicationLag(self: *Self) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var max_lag: u64 = 0;
        for (self.gateways.items) |gateway| {
            const lag = gateway.last_sequence_no -| gateway.last_replicated_sequence;
            max_lag = @max(max_lag, lag);
        }
        return max_lag;
    }
};

fn cloneGateway(allocator: Allocator, gateway: RegisteredGateway) !RegisteredGateway {
    return .{
        .gateway_id = try allocator.dupe(u8, gateway.gateway_id),
        .lab_id = try allocator.dupe(u8, gateway.lab_id),
        .base_url = try allocator.dupe(u8, gateway.base_url),
        .graph_backend = try allocator.dupe(u8, gateway.graph_backend),
        .status = try allocator.dupe(u8, gateway.status),
        .registered_services = gateway.registered_services,
        .last_sequence_no = gateway.last_sequence_no,
        .last_replicated_sequence = gateway.last_replicated_sequence,
        .connected_at = gateway.connected_at,
        .last_seen_at = gateway.last_seen_at,
        .services = try federation_types.cloneServiceRecords(allocator, gateway.services),
    };
}

fn replaceString(allocator: Allocator, target: *[]u8, value: []const u8) !void {
    allocator.free(target.*);
    target.* = try allocator.dupe(u8, value);
}

fn replaceServices(
    allocator: Allocator,
    target: *[]federation_types.ServiceRecord,
    services: []const federation_types.ServiceAdvertisement,
) !void {
    const replacement = try federation_types.ownedServicesFromAdvertisements(allocator, services);
    federation_types.deinitServiceRecords(allocator, target.*);
    target.* = replacement;
}

fn matchesPlacement(service: federation_types.ServiceRecord, placement: federation_types.PlacementRequest) bool {
    if (placement.service_id) |service_id| {
        if (!std.mem.eql(u8, service.service_id, service_id)) return false;
    }
    if (placement.executor_id) |executor_id| {
        if (!std.mem.eql(u8, service.executor_id, executor_id)) return false;
    }
    if (placement.worker_class) |worker_class| {
        const value = service.worker_class orelse return false;
        if (!std.mem.eql(u8, value, worker_class)) return false;
    }
    if (placement.target_arch) |target_arch| {
        const value = service.target_arch orelse return false;
        if (!std.mem.eql(u8, value, target_arch)) return false;
    }
    return true;
}

fn serviceHasCapability(service: federation_types.ServiceRecord, required_capability: []const u8) bool {
    for (service.capabilities.items) |capability| {
        if (std.mem.eql(u8, capability, required_capability)) return true;
    }
    return false;
}

fn isBetterCandidate(
    candidate: federation_types.ServiceRecord,
    current: federation_types.ServiceRecord,
    workers_required: ?usize,
) bool {
    const candidate_ok = std.mem.eql(u8, candidate.health_status, "ok");
    const current_ok = std.mem.eql(u8, current.health_status, "ok");
    if (candidate_ok != current_ok) return candidate_ok;

    if (workers_required) |required| {
        const candidate_sufficient = candidate.workers_available >= required;
        const current_sufficient = current.workers_available >= required;
        if (candidate_sufficient != current_sufficient) return candidate_sufficient;
    }

    if (candidate.workers_available != current.workers_available) {
        return candidate.workers_available > current.workers_available;
    }

    if (candidate.workers_ready != current.workers_ready) {
        return candidate.workers_ready > current.workers_ready;
    }

    return candidate.updated_at > current.updated_at;
}
