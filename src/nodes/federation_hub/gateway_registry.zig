const std = @import("std");

const Allocator = std.mem.Allocator;

pub const ConnectRequest = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    base_url: []const u8,
    graph_backend: []const u8,
    registered_services: usize = 0,
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

    pub fn deinit(self: *RegisteredGateway, allocator: Allocator) void {
        allocator.free(self.gateway_id);
        allocator.free(self.lab_id);
        allocator.free(self.base_url);
        allocator.free(self.graph_backend);
        allocator.free(self.status);
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
            gateway.registered_services = request.registered_services;
            gateway.last_sequence_no = request.last_sequence_no;
            gateway.last_replicated_sequence = @max(gateway.last_replicated_sequence, request.last_replicated_sequence);
            gateway.last_seen_at = now;
            return try cloneGateway(self.allocator, gateway.*);
        }

        const gateway = RegisteredGateway{
            .gateway_id = try self.allocator.dupe(u8, request.gateway_id),
            .lab_id = try self.allocator.dupe(u8, request.lab_id),
            .base_url = try self.allocator.dupe(u8, request.base_url),
            .graph_backend = try self.allocator.dupe(u8, request.graph_backend),
            .status = try self.allocator.dupe(u8, request.status orelse "connected"),
            .registered_services = request.registered_services,
            .last_sequence_no = request.last_sequence_no,
            .last_replicated_sequence = request.last_replicated_sequence,
            .connected_at = now,
            .last_seen_at = now,
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
                .connected_at = gateway.connected_at,
                .last_seen_at = gateway.last_seen_at,
            });
        }

        return std.json.stringifyAlloc(allocator, .{
            .gateways = response_gateways.items,
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
    };
}

fn replaceString(allocator: Allocator, target: *[]u8, value: []const u8) !void {
    allocator.free(target.*);
    target.* = try allocator.dupe(u8, value);
}
