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
    endpoint: []const u8,
    status: ?[]const u8 = null,
    worker_count: ?usize = null,
    capabilities: ?[]const []const u8 = null,
};

pub const RegisteredService = struct {
    service_id: []u8,
    service_type: ServiceType,
    endpoint: []u8,
    status: []u8,
    worker_count: usize,
    capabilities: std.ArrayList([]u8),
    registered_at: i64,
    updated_at: i64,

    pub fn deinit(self: *RegisteredService, allocator: Allocator) void {
        allocator.free(self.service_id);
        allocator.free(self.endpoint);
        allocator.free(self.status);
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

        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.services.items) |*service| {
            if (std.mem.eql(u8, service.service_id, request.service_id)) {
                try replaceString(self.allocator, &service.endpoint, request.endpoint);
                try replaceString(self.allocator, &service.status, request.status orelse "running");
                service.service_type = service_type;
                service.worker_count = request.worker_count orelse 0;
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
            .endpoint = try self.allocator.dupe(u8, request.endpoint),
            .status = try self.allocator.dupe(u8, request.status orelse "running"),
            .worker_count = request.worker_count orelse 0,
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
            endpoint: []const u8,
            status: []const u8,
            worker_count: usize,
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
                .endpoint = service.endpoint,
                .status = service.status,
                .worker_count = service.worker_count,
                .capabilities = service.capabilities.items,
                .registered_at = service.registered_at,
                .updated_at = service.updated_at,
            });
        }

        return std.json.stringifyAlloc(allocator, .{
            .count = self.services.items.len,
            .services = response_services.items,
        }, .{});
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
            .endpoint = try allocator.dupe(u8, service.endpoint),
            .status = try allocator.dupe(u8, service.status),
            .worker_count = service.worker_count,
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
