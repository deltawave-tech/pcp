const std = @import("std");

const Allocator = std.mem.Allocator;

pub const GatewayClient = struct {
    allocator: Allocator,
    gateway_url: []u8,
    external_token: ?[]u8,
    internal_token: ?[]u8,
    service_id: []u8,
    namespace_id: ?[]u8,

    const Self = @This();

    pub fn initFromEnv(allocator: Allocator, default_service_id: []const u8) !?Self {
        const gateway_url = std.process.getEnvVarOwned(allocator, "PCP_GATEWAY_URL") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => return null,
            else => return err,
        };
        errdefer allocator.free(gateway_url);

        const external_token = std.process.getEnvVarOwned(allocator, "PCP_GATEWAY_TOKEN") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => null,
            else => return err,
        };
        errdefer if (external_token) |token| allocator.free(token);

        const internal_token = std.process.getEnvVarOwned(allocator, "PCP_GATEWAY_INTERNAL_TOKEN") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => if (external_token) |token| try allocator.dupe(u8, token) else null,
            else => return err,
        };
        errdefer if (internal_token) |token| allocator.free(token);

        const service_id = std.process.getEnvVarOwned(allocator, "PCP_GATEWAY_SERVICE_ID") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => try allocator.dupe(u8, default_service_id),
            else => return err,
        };
        errdefer allocator.free(service_id);

        const namespace_id = std.process.getEnvVarOwned(allocator, "PCP_GATEWAY_NAMESPACE") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => null,
            else => return err,
        };
        errdefer if (namespace_id) |value| allocator.free(value);

        return .{
            .allocator = allocator,
            .gateway_url = gateway_url,
            .external_token = external_token,
            .internal_token = internal_token,
            .service_id = service_id,
            .namespace_id = namespace_id,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.gateway_url);
        if (self.external_token) |token| self.allocator.free(token);
        if (self.internal_token) |token| self.allocator.free(token);
        self.allocator.free(self.service_id);
        if (self.namespace_id) |value| self.allocator.free(value);
    }

    pub fn registerService(
        self: *const Self,
        service_type: []const u8,
        base_url: []const u8,
        capabilities: []const []const u8,
        worker_count: usize,
        ready_worker_count: usize,
        job_status: []const u8,
        health_status: []const u8,
    ) !void {
        const body = try std.json.stringifyAlloc(self.allocator, .{
            .service_id = self.service_id,
            .service_type = service_type,
            .base_url = base_url,
            .auth_mode = if (self.external_token != null) "bearer" else "none",
            .health_status = health_status,
            .job_status = job_status,
            .worker_count = worker_count,
            .ready_worker_count = ready_worker_count,
            .capabilities = capabilities,
        }, .{});
        defer self.allocator.free(body);

        try self.postJson("/v1/services/register", self.external_token orelse self.internal_token, body);
    }

    pub fn emitEvent(
        self: *const Self,
        event_id: []const u8,
        event_type: []const u8,
        job_id: ?[]const u8,
        payload_json: []const u8,
        provenance_json: ?[]const u8,
    ) !void {
        var payload = try std.json.parseFromSlice(std.json.Value, self.allocator, payload_json, .{});
        defer payload.deinit();
        var provenance: ?std.json.Parsed(std.json.Value) = null;
        defer if (provenance) |*value| value.deinit();
        if (provenance_json) |json| {
            provenance = try std.json.parseFromSlice(std.json.Value, self.allocator, json, .{});
        }

        const body = try std.json.stringifyAlloc(self.allocator, .{
            .events = &[_]struct {
                event_id: []const u8,
                service_id: []const u8,
                event_type: []const u8,
                namespace_id: ?[]const u8,
                job_id: ?[]const u8,
                payload: std.json.Value,
                provenance: ?std.json.Value,
            }{
                .{
                    .event_id = event_id,
                    .service_id = self.service_id,
                    .event_type = event_type,
                    .namespace_id = self.namespace_id,
                    .job_id = job_id,
                    .payload = payload.value,
                    .provenance = if (provenance) |value| value.value else null,
                },
            },
        }, .{});
        defer self.allocator.free(body);

        try self.postJson("/v1/internal/events", self.internal_token, body);
    }

    fn postJson(self: *const Self, path: []const u8, token: ?[]const u8, body: []const u8) !void {
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ std.mem.trimRight(u8, self.gateway_url, "/"), path });
        defer self.allocator.free(url);

        var auth_header_value: ?[]u8 = null;
        defer if (auth_header_value) |value| self.allocator.free(value);

        var headers = std.ArrayList(std.http.Header).init(self.allocator);
        defer headers.deinit();
        try headers.append(.{ .name = "content-type", .value = "application/json" });
        try headers.append(.{ .name = "accept", .value = "application/json" });
        if (token) |value| {
            auth_header_value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{value});
            try headers.append(.{ .name = "authorization", .value = auth_header_value.? });
        }

        var response_body = std.ArrayList(u8).init(self.allocator);
        defer response_body.deinit();

        const result = try client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = body,
            .extra_headers = headers.items,
            .response_storage = .{ .dynamic = &response_body },
        });

        if (result.status != .ok and result.status != .accepted and result.status != .created) {
            return error.GatewayRequestFailed;
        }
    }
};
