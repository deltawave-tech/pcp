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

pub const RegisterRequest = struct {
    service_id: []const u8,
    service_type: []const u8,
    base_url: ?[]const u8 = null,
    endpoint: ?[]const u8 = null,
    auth_mode: ?[]const u8 = null,
    health_status: ?[]const u8 = null,
    job_status: ?[]const u8 = null,
    worker_count: ?usize = null,
    ready_worker_count: ?usize = null,
    capabilities: ?[]const []const u8 = null,
};

pub const RegisteredService = struct {
    service_id: []u8,
    service_type: ServiceType,
    base_url: []u8,
    auth_mode: []u8,
    health_status: []u8,
    job_status: []u8,
    worker_count: usize,
    ready_worker_count: usize,
    capabilities: std.ArrayList([]u8),
    registered_at: i64,
    updated_at: i64,

    pub fn deinit(self: *RegisteredService, allocator: Allocator) void {
        allocator.free(self.service_id);
        allocator.free(self.base_url);
        allocator.free(self.auth_mode);
        allocator.free(self.health_status);
        allocator.free(self.job_status);
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
        const service_type = ServiceType.parse(request.service_type) orelse return error.InvalidServiceType;
        const base_url = request.base_url orelse request.endpoint orelse return error.MissingServiceBaseUrl;

        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.services.items) |*service| {
            if (std.mem.eql(u8, service.service_id, request.service_id)) {
                try replaceString(self.allocator, &service.base_url, base_url);
                try replaceString(self.allocator, &service.auth_mode, request.auth_mode orelse "none");
                try replaceString(self.allocator, &service.health_status, request.health_status orelse "ok");
                try replaceString(self.allocator, &service.job_status, request.job_status orelse "running");
                service.service_type = service_type;
                service.worker_count = request.worker_count orelse 0;
                service.ready_worker_count = request.ready_worker_count orelse service.worker_count;
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
            .service_type = service_type,
            .base_url = try self.allocator.dupe(u8, base_url),
            .auth_mode = try self.allocator.dupe(u8, request.auth_mode orelse "none"),
            .health_status = try self.allocator.dupe(u8, request.health_status orelse "ok"),
            .job_status = try self.allocator.dupe(u8, request.job_status orelse "running"),
            .worker_count = request.worker_count orelse 0,
            .ready_worker_count = request.ready_worker_count orelse request.worker_count orelse 0,
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
            service_type: []const u8,
            base_url: []const u8,
            auth_mode: []const u8,
            health_status: []const u8,
            job_status: []const u8,
            worker_count: usize,
            ready_worker_count: usize,
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
                .service_type = service.service_type.asString(),
                .base_url = service.base_url,
                .auth_mode = service.auth_mode,
                .health_status = service.health_status,
                .job_status = service.job_status,
                .worker_count = service.worker_count,
                .ready_worker_count = service.ready_worker_count,
                .capabilities = service.capabilities.items,
                .registered_at = service.registered_at,
                .updated_at = service.updated_at,
            });
        }

        return std.json.stringifyAlloc(allocator, .{
            .services = response_services.items,
        }, .{});
    }

    pub fn renderServiceJson(self: *Self, allocator: Allocator, service_id: []const u8) !?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.services.items) |service| {
            if (!std.mem.eql(u8, service.service_id, service_id)) continue;
            return try std.json.stringifyAlloc(allocator, .{
                .service_id = service.service_id,
                .service_type = service.service_type.asString(),
                .base_url = service.base_url,
                .auth_mode = service.auth_mode,
                .health_status = service.health_status,
                .job_status = service.job_status,
                .worker_count = service.worker_count,
                .ready_worker_count = service.ready_worker_count,
                .capabilities = service.capabilities.items,
                .registered_at = service.registered_at,
                .updated_at = service.updated_at,
            }, .{});
        }

        return null;
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
            .service_type = service.service_type,
            .base_url = try allocator.dupe(u8, service.base_url),
            .auth_mode = try allocator.dupe(u8, service.auth_mode),
            .health_status = try allocator.dupe(u8, service.health_status),
            .job_status = try allocator.dupe(u8, service.job_status),
            .worker_count = service.worker_count,
            .ready_worker_count = service.ready_worker_count,
            .capabilities = capabilities,
            .registered_at = service.registered_at,
            .updated_at = service.updated_at,
        };
    }

    fn replaceString(allocator: Allocator, slot: *[]u8, value: []const u8) !void {
        allocator.free(slot.*);
        slot.* = try allocator.dupe(u8, value);
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
};
