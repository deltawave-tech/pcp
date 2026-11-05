const std = @import("std");

const Allocator = std.mem.Allocator;

pub const ServiceType = enum {
    inference,
    rl,
    training,

    pub fn parse(value: []const u8) ?ServiceType {
        if (std.mem.eql(u8, value, "inference")) return .inference;
        if (std.mem.eql(u8, value, "rl")) return .rl;
        if (std.mem.eql(u8, value, "training")) return .training;
        return null;
    }

    pub fn asString(self: ServiceType) []const u8 {
        return switch (self) {
            .inference => "inference",
            .rl => "rl",
            .training => "training",
        };
    }
};

pub fn isValidServiceTypeName(value: []const u8) bool {
    if (value.len == 0 or value.len > 128) return false;
    for (value) |char| {
        switch (char) {
            'a'...'z', 'A'...'Z', '0'...'9', '_', '-', '.' => {},
            else => return false,
        }
    }
    return true;
}

pub const RegisterRequest = struct {
    service_id: []const u8,
    executor_id: ?[]const u8 = null,
    service_type: []const u8,
    base_url: ?[]const u8 = null,
    endpoint: ?[]const u8 = null,
    auth_mode: ?[]const u8 = null,
    health_status: ?[]const u8 = null,
    job_status: ?[]const u8 = null,
    worker_count: ?usize = null,
    ready_worker_count: ?usize = null,
    workers_connected: ?usize = null,
    workers_ready: ?usize = null,
    workers_available: ?usize = null,
    workers_dispatchable: ?usize = null,
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    reserved_workers: ?usize = null,
    max_workers: ?usize = null,
    capabilities: ?[]const []const u8 = null,
};

pub const RegisteredService = struct {
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

    pub fn deinit(self: *RegisteredService, allocator: Allocator) void {
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
};

pub const ServiceRegistry = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    services: std.ArrayList(RegisteredService),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .services = std.ArrayList(RegisteredService).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.services.items) |*service| {
            service.deinit(self.allocator);
        }
        self.services.deinit();
    }

    pub fn register(self: *Self, request: RegisterRequest) !RegisteredService {
        if (!isValidServiceTypeName(request.service_type)) return error.InvalidServiceType;
        const base_url = request.base_url orelse request.endpoint orelse return error.MissingServiceBaseUrl;
        const executor_id = request.executor_id orelse return error.MissingExecutorId;
        const counts = normalizeWorkerCounts(request);

        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.services.items) |*service| {
            if (std.mem.eql(u8, service.executor_id, executor_id)) {
                try replaceString(self.allocator, &service.service_id, request.service_id);
                try replaceString(self.allocator, &service.executor_id, executor_id);
                try replaceString(self.allocator, &service.base_url, base_url);
                try replaceString(self.allocator, &service.auth_mode, request.auth_mode orelse "none");
                try replaceString(self.allocator, &service.health_status, request.health_status orelse "ok");
                try replaceString(self.allocator, &service.job_status, request.job_status orelse "running");
                try replaceOptionalString(self.allocator, &service.worker_class, request.worker_class);
                try replaceOptionalString(self.allocator, &service.target_arch, request.target_arch);
                try replaceString(self.allocator, &service.service_type, request.service_type);
                service.worker_count = counts.workers_connected;
                service.ready_worker_count = counts.workers_available;
                service.workers_connected = counts.workers_connected;
                service.workers_ready = counts.workers_ready;
                service.workers_available = counts.workers_available;
                service.reserved_workers = request.reserved_workers;
                service.max_workers = request.max_workers;
                service.updated_at = std.time.timestamp();
                try replaceCapabilities(self.allocator, &service.capabilities, request.capabilities);
                return try cloneService(self.allocator, service.*);
            }
        }

        var capabilities = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (capabilities.items) |capability| self.allocator.free(capability);
            capabilities.deinit();
        }
        try appendCapabilities(self.allocator, &capabilities, request.capabilities);

        const now = std.time.timestamp();
        const service = RegisteredService{
            .service_id = try self.allocator.dupe(u8, request.service_id),
            .executor_id = try self.allocator.dupe(u8, executor_id),
            .service_type = try self.allocator.dupe(u8, request.service_type),
            .base_url = try self.allocator.dupe(u8, base_url),
            .auth_mode = try self.allocator.dupe(u8, request.auth_mode orelse "none"),
            .health_status = try self.allocator.dupe(u8, request.health_status orelse "ok"),
            .job_status = try self.allocator.dupe(u8, request.job_status orelse "running"),
            .worker_count = counts.workers_connected,
            .ready_worker_count = counts.workers_available,
            .workers_connected = counts.workers_connected,
            .workers_ready = counts.workers_ready,
            .workers_available = counts.workers_available,
            .worker_class = if (request.worker_class) |value| try self.allocator.dupe(u8, value) else null,
            .target_arch = if (request.target_arch) |value| try self.allocator.dupe(u8, value) else null,
            .reserved_workers = request.reserved_workers,
            .max_workers = request.max_workers,
            .capabilities = capabilities,
            .registered_at = now,
            .updated_at = now,
        };
        try self.services.append(service);
        return try cloneService(self.allocator, service);
    }

    pub fn count(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.services.items.len;
    }

    pub fn renderJson(self: *Self, allocator: Allocator) ![]u8 {
        const ResponseService = struct {
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

        self.mutex.lock();
        defer self.mutex.unlock();

        var response_services = std.ArrayList(ResponseService).init(allocator);
        defer response_services.deinit();

        for (self.services.items) |service| {
            try response_services.append(.{
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

        return std.json.stringifyAlloc(allocator, .{
            .services = response_services.items,
        }, .{});
    }

    pub fn renderServiceJson(self: *Self, allocator: Allocator, executor_id: []const u8) !?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.services.items) |service| {
            if (!std.mem.eql(u8, service.executor_id, executor_id)) continue;
            return try std.json.stringifyAlloc(allocator, .{
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
            }, .{});
        }

        return null;
    }

    pub fn findService(self: *Self, allocator: Allocator, executor_id: []const u8) !?RegisteredService {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.services.items) |service| {
            if (std.mem.eql(u8, service.executor_id, executor_id)) {
                return try cloneService(allocator, service);
            }
        }

        return null;
    }

    pub fn selectServiceByType(
        self: *Self,
        allocator: Allocator,
        service_type: []const u8,
        preferred_service_id: ?[]const u8,
        preferred_executor_id: ?[]const u8,
    ) !?RegisteredService {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (preferred_executor_id) |executor_id| {
            for (self.services.items) |service| {
                if (!std.mem.eql(u8, service.service_type, service_type)) continue;
                if (!std.mem.eql(u8, service.executor_id, executor_id)) continue;
                return try cloneService(allocator, service);
            }
            return null;
        }

        if (preferred_service_id) |service_id| {
            var best_index: ?usize = null;
            for (self.services.items, 0..) |service, idx| {
                if (!std.mem.eql(u8, service.service_type, service_type)) continue;
                if (!std.mem.eql(u8, service.service_id, service_id)) continue;
                if (best_index == null or isBetterCandidate(service, self.services.items[best_index.?])) {
                    best_index = idx;
                }
            }

            if (best_index) |idx| {
                return try cloneService(allocator, self.services.items[idx]);
            }
            return null;
        }

        var best_index: ?usize = null;
        for (self.services.items, 0..) |service, idx| {
            if (!std.mem.eql(u8, service.service_type, service_type)) continue;
            if (best_index == null or isBetterCandidate(service, self.services.items[best_index.?])) {
                best_index = idx;
            }
        }

        if (best_index) |idx| {
            return try cloneService(allocator, self.services.items[idx]);
        }

        return null;
    }

    pub fn listServices(self: *Self, allocator: Allocator) ![]RegisteredService {
        self.mutex.lock();
        defer self.mutex.unlock();

        const services = try allocator.alloc(RegisteredService, self.services.items.len);
        errdefer allocator.free(services);

        for (self.services.items, 0..) |service, idx| {
            services[idx] = try cloneService(allocator, service);
        }

        return services;
    }

    pub fn deinitServiceList(allocator: Allocator, services: []RegisteredService) void {
        for (services) |*service| {
            service.deinit(allocator);
        }
        allocator.free(services);
    }

    fn cloneService(allocator: Allocator, service: RegisteredService) !RegisteredService {
        var capabilities = std.ArrayList([]u8).init(allocator);
        errdefer {
            for (capabilities.items) |capability| allocator.free(capability);
            capabilities.deinit();
        }
        for (service.capabilities.items) |capability| {
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
            .worker_count = service.worker_count,
            .ready_worker_count = service.ready_worker_count,
            .workers_connected = service.workers_connected,
            .workers_ready = service.workers_ready,
            .workers_available = service.workers_available,
            .worker_class = if (service.worker_class) |value| try allocator.dupe(u8, value) else null,
            .target_arch = if (service.target_arch) |value| try allocator.dupe(u8, value) else null,
            .reserved_workers = service.reserved_workers,
            .max_workers = service.max_workers,
            .capabilities = capabilities,
            .registered_at = service.registered_at,
            .updated_at = service.updated_at,
        };
    }

    fn replaceString(allocator: Allocator, slot: *[]u8, value: []const u8) !void {
        allocator.free(slot.*);
        slot.* = try allocator.dupe(u8, value);
    }

    fn replaceOptionalString(allocator: Allocator, slot: *?[]u8, value: ?[]const u8) !void {
        if (slot.*) |existing| allocator.free(existing);
        slot.* = if (value) |inner| try allocator.dupe(u8, inner) else null;
    }

    fn replaceCapabilities(allocator: Allocator, capabilities: *std.ArrayList([]u8), values: ?[]const []const u8) !void {
        for (capabilities.items) |capability| {
            allocator.free(capability);
        }
        capabilities.clearRetainingCapacity();
        try appendCapabilities(allocator, capabilities, values);
    }

    fn appendCapabilities(allocator: Allocator, capabilities: *std.ArrayList([]u8), values: ?[]const []const u8) !void {
        const items = values orelse return;
        for (items) |capability| {
            try capabilities.append(try allocator.dupe(u8, capability));
        }
    }

    fn isBetterCandidate(candidate: RegisteredService, current: RegisteredService) bool {
        const candidate_ok = std.mem.eql(u8, candidate.health_status, "ok");
        const current_ok = std.mem.eql(u8, current.health_status, "ok");
        if (candidate_ok != current_ok) return candidate_ok;

        if (candidate.workers_available != current.workers_available) {
            return candidate.workers_available > current.workers_available;
        }

        return candidate.updated_at > current.updated_at;
    }
};

const NormalizedWorkerCounts = struct {
    workers_connected: usize,
    workers_ready: usize,
    workers_available: usize,
};

fn normalizeWorkerCounts(request: RegisterRequest) NormalizedWorkerCounts {
    var workers_connected = request.workers_connected orelse request.worker_count orelse 0;
    var workers_ready = request.workers_ready orelse request.ready_worker_count orelse request.workers_available orelse request.workers_dispatchable orelse workers_connected;
    const workers_available = request.workers_dispatchable orelse request.workers_available orelse request.ready_worker_count orelse request.workers_ready orelse workers_connected;
    if (workers_ready > workers_connected) workers_connected = workers_ready;
    if (workers_available > workers_ready) workers_ready = workers_available;
    return .{
        .workers_connected = workers_connected,
        .workers_ready = workers_ready,
        .workers_available = workers_available,
    };
}
