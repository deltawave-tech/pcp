const std = @import("std");

const Allocator = std.mem.Allocator;
const federation_types = @import("../federation/types.zig");
const mutation_log = @import("../graph/mutation_log.zig");
const gateway_mod = @import("gateway.zig");

pub const FederationClient = struct {
    allocator: Allocator,
    gateway: *gateway_mod.Gateway,
    endpoint: []u8,
    auth_token: ?[]u8,
    gateway_base_url: []u8,
    heartbeat_interval_ms: u64,
    is_running: std.atomic.Value(u8),

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        gateway: *gateway_mod.Gateway,
        endpoint: []const u8,
        auth_token: ?[]const u8,
        gateway_base_url: []const u8,
        heartbeat_interval_ms: u64,
    ) !Self {
        return .{
            .allocator = allocator,
            .gateway = gateway,
            .endpoint = try allocator.dupe(u8, std.mem.trimRight(u8, endpoint, "/")),
            .auth_token = if (auth_token) |token| try allocator.dupe(u8, token) else null,
            .gateway_base_url = try allocator.dupe(u8, gateway_base_url),
            .heartbeat_interval_ms = heartbeat_interval_ms,
            .is_running = std.atomic.Value(u8).init(1),
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.endpoint);
        if (self.auth_token) |token| self.allocator.free(token);
        self.allocator.free(self.gateway_base_url);
    }

    pub fn stop(self: *Self) void {
        self.is_running.store(0, .release);
    }

    pub fn run(self: *Self) !void {
        while (self.is_running.load(.acquire) == 1) {
            self.syncOnce() catch |err| {
                self.gateway.federation.setDisconnected(err);
            };
            self.replicatePending() catch |err| {
                self.gateway.federation.setDisconnected(err);
            };

            self.updateReplicationState();

            var remaining = self.heartbeat_interval_ms;
            while (self.is_running.load(.acquire) == 1 and remaining > 0) {
                const slice_ms: u64 = @min(remaining, @as(u64, 250));
                std.time.sleep(slice_ms * std.time.ns_per_ms);
                remaining -= slice_ms;
            }
        }
    }

    pub fn syncOnce(self: *Self) !void {
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/federation/connect", .{self.endpoint});
        defer self.allocator.free(url);

        const body = try std.json.stringifyAlloc(self.allocator, .{
            .gateway_id = self.gateway.config.gateway_id,
            .lab_id = self.gateway.config.lab_id,
            .base_url = self.gateway_base_url,
            .graph_backend = self.gateway.config.graph_backend,
            .registered_services = self.gateway.service_registry.count(),
            .last_sequence_no = self.gateway.graph.lastSequence(),
            .last_replicated_sequence = self.gateway.graph.lastReplicatedSequence(),
            .status = "connected",
        }, .{});
        defer self.allocator.free(body);

        var auth_header_value: ?[]u8 = null;
        defer if (auth_header_value) |value| self.allocator.free(value);

        var headers = std.ArrayList(std.http.Header).init(self.allocator);
        defer headers.deinit();
        try headers.append(.{ .name = "content-type", .value = "application/json" });
        try headers.append(.{ .name = "accept", .value = "application/json" });
        if (self.auth_token) |token| {
            auth_header_value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
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
            return error.GlobalControllerRequestFailed;
        }

        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, response_body.items, .{});
        defer parsed.deinit();
        try self.gateway.federation.updateFromConnectResponse(self.allocator, self.endpoint, parsed.value);
    }

    fn replicatePending(self: *Self) !void {
        while (true) {
            const last_replicated = self.gateway.graph.lastReplicatedSequence();
            const records = try self.gateway.graph.snapshotMutationsFrom(self.allocator, last_replicated, 128);
            defer deinitRecords(self.allocator, records);
            if (records.len == 0) break;

            var batch = std.ArrayList(federation_types.MutationBatchItem).init(self.allocator);
            defer batch.deinit();

            var highest_processed = last_replicated;
            for (records) |record| {
                highest_processed = record.sequence_no;
                if (record.visibility == .local) continue;
                try batch.append(.{
                    .sequence_no = record.sequence_no,
                    .mutation_id = record.mutation_id,
                    .namespace_id = record.namespace_id,
                    .mutation_type = record.mutation_type.asString(),
                    .target_id = record.target_id,
                    .payload_json = record.payload_json,
                    .visibility = record.visibility.asString(),
                    .provenance_json = record.provenance_json,
                    .timestamp = record.timestamp,
                });
                if (batch.items.len >= 64) break;
            }

            if (batch.items.len == 0) {
                self.gateway.graph.markReplicatedThrough(highest_processed);
                self.updateReplicationState();
                continue;
            }

            const ack = try self.postBatch(.{
                .gateway_id = self.gateway.config.gateway_id,
                .lab_id = self.gateway.config.lab_id,
                .last_sequence_no = self.gateway.graph.lastSequence(),
                .last_replicated_sequence = last_replicated,
                .mutations = batch.items,
            });
            if (!ack.accepted) return error.InvalidFederationAck;

            self.gateway.graph.markReplicatedThrough(highest_processed);
            self.updateReplicationState();

            if (records.len < 128) break;
        }
    }

    fn postBatch(self: *Self, request: federation_types.MutationBatchRequest) !federation_types.MutationBatchAck {
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/federation/mutations", .{self.endpoint});
        defer self.allocator.free(url);

        const body = try std.json.stringifyAlloc(self.allocator, request, .{});
        defer self.allocator.free(body);

        var auth_header_value: ?[]u8 = null;
        defer if (auth_header_value) |value| self.allocator.free(value);

        var headers = std.ArrayList(std.http.Header).init(self.allocator);
        defer headers.deinit();
        try headers.append(.{ .name = "content-type", .value = "application/json" });
        try headers.append(.{ .name = "accept", .value = "application/json" });
        if (self.auth_token) |token| {
            auth_header_value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
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
            return error.GlobalControllerRequestFailed;
        }

        const parsed = try std.json.parseFromSlice(federation_types.MutationBatchAck, self.allocator, response_body.items, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        return parsed.value;
    }

    fn updateReplicationState(self: *Self) void {
        self.gateway.federation.updateReplicationState(
            self.gateway.graph.lastSequence(),
            self.gateway.graph.lastReplicatedSequence(),
            self.gateway.graph.countPendingReplications(),
        );
    }
};

fn deinitRecords(allocator: Allocator, records: []mutation_log.MutationRecord) void {
    for (records) |*record| record.deinit(allocator);
    allocator.free(records);
}
