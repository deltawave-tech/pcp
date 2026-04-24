const std = @import("std");

const Allocator = std.mem.Allocator;
const graph_adapter = @import("graph_adapter.zig");
const graph_types = @import("../../graph/types.zig");

pub const InternalEventRequest = struct {
    event_id: []const u8,
    service_id: []const u8,
    event_type: []const u8,
    namespace_id: ?[]const u8 = null,
    job_id: ?[]const u8 = null,
    timestamp: ?i64 = null,
    payload: ?std.json.Value = null,
    provenance: ?std.json.Value = null,
};

pub const InternalEventsRequest = struct {
    events: []const InternalEventRequest,
};

pub const EventIngestOptions = struct {
    gateway_id: []const u8,
    lab_id: []const u8,
    default_visibility: graph_types.Visibility = .local,
};

pub const EventIngester = struct {
    allocator: Allocator,
    options: EventIngestOptions,

    const Self = @This();

    pub fn init(allocator: Allocator, options: EventIngestOptions) Self {
        return .{
            .allocator = allocator,
            .options = options,
        };
    }

    pub fn ingestJson(self: *Self, allocator: Allocator, graph: *graph_adapter.GatewayGraph, request: InternalEventsRequest) ![]u8 {
        var accepted = std.ArrayList(EventAcceptResult).init(allocator);
        defer {
            for (accepted.items) |result| {
                for (result.mutation_ids) |mutation_id| allocator.free(mutation_id);
                allocator.free(result.mutation_ids);
            }
            accepted.deinit();
        }

        for (request.events) |event| {
            const result = try self.ingestOne(allocator, graph, event);
            try accepted.append(result);
        }

        return std.json.stringifyAlloc(allocator, .{
            .accepted = true,
            .events = accepted.items,
        }, .{});
    }

    fn ingestOne(self: *Self, allocator: Allocator, graph: *graph_adapter.GatewayGraph, event: InternalEventRequest) !EventAcceptResult {
        const namespace_id = event.namespace_id orelse try std.fmt.allocPrint(allocator, "{s}/shared", .{self.options.lab_id});
        defer if (event.namespace_id == null) allocator.free(namespace_id);

        const timestamp = event.timestamp orelse std.time.timestamp();
        const visibility = self.options.default_visibility;
        const subject_entity_id = try eventSubjectId(allocator, event);
        defer allocator.free(subject_entity_id);
        var mutations = std.ArrayList(graph_types.MutationRequest).init(allocator);
        defer mutations.deinit();

        try buildEventMutations(
            allocator,
            &mutations,
            self.options,
            namespace_id,
            subject_entity_id,
            visibility,
            timestamp,
            event,
        );
        const body = try graph.applyMutationsJson(allocator, .{ .mutations = mutations.items });
        defer allocator.free(body);

        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();

        const mutation_ids = try extractMutationIds(allocator, parsed.value);

        return .{
            .event_id = event.event_id,
            .event_type = event.event_type,
            .service_id = event.service_id,
            .mutation_ids = mutation_ids,
        };
    }
};

const EventAcceptResult = struct {
    event_id: []const u8,
    event_type: []const u8,
    service_id: []const u8,
    mutation_ids: []const []const u8,
};

fn buildEventMutations(
    allocator: Allocator,
    mutations: *std.ArrayList(graph_types.MutationRequest),
    options: EventIngestOptions,
    namespace_id: []const u8,
    subject_entity_id: []const u8,
    visibility: graph_types.Visibility,
    timestamp: i64,
    event: InternalEventRequest,
) !void {
    const subject_entity_type = eventSubjectType(event.event_type);
    const subject_display_name = eventSubjectDisplayName(event, subject_entity_id);
    const subject_payload = try buildSubjectPayload(allocator, event, subject_entity_type, subject_display_name);
    const provenance = try buildProvenance(allocator, options, event, timestamp, event.event_type);
    const observation_payload = try buildObservationPayload(allocator, event, subject_entity_id);

    try mutations.append(.{
        .mutation_id = try std.fmt.allocPrint(allocator, "{s}:subject", .{event.event_id}),
        .mutation_type = "upsert_entity",
        .namespace_id = namespace_id,
        .target_id = subject_entity_id,
        .payload = subject_payload,
        .visibility = visibility.asString(),
        .provenance = provenance,
    });

    try mutations.append(.{
        .mutation_id = try std.fmt.allocPrint(allocator, "{s}:observation", .{event.event_id}),
        .mutation_type = "append_observation",
        .namespace_id = namespace_id,
        .target_id = try allocator.dupe(u8, event.event_id),
        .payload = observation_payload,
        .visibility = visibility.asString(),
        .provenance = try cloneJsonValue(allocator, provenance),
    });
}

fn buildSubjectPayload(
    allocator: Allocator,
    event: InternalEventRequest,
    subject_entity_type: []const u8,
    subject_display_name: []const u8,
) !std.json.Value {
    var properties = std.json.ObjectMap.init(allocator);
    errdefer properties.deinit();
    try properties.put("event_type", .{ .string = try allocator.dupe(u8, event.event_type) });
    try properties.put("service_id", .{ .string = try allocator.dupe(u8, event.service_id) });
    if (event.job_id) |job_id| {
        try properties.put("job_id", .{ .string = try allocator.dupe(u8, job_id) });
    }

    var payload = std.json.ObjectMap.init(allocator);
    errdefer payload.deinit();
    try payload.put("entity_type", .{ .string = try allocator.dupe(u8, subject_entity_type) });
    try payload.put("display_name", .{ .string = try allocator.dupe(u8, subject_display_name) });
    try payload.put("properties", .{ .object = properties });
    return .{ .object = payload };
}

fn buildObservationPayload(allocator: Allocator, event: InternalEventRequest, subject_entity_id: []const u8) !std.json.Value {
    var payload_object = std.json.ObjectMap.init(allocator);
    errdefer payload_object.deinit();

    try payload_object.put("observation_type", .{ .string = event.event_type });
    try payload_object.put("subject_entity_id", .{ .string = try allocator.dupe(u8, subject_entity_id) });
    try payload_object.put("payload", try cloneJsonValue(allocator, event.payload orelse .{ .null = {} }));

    return .{ .object = payload_object };
}

fn buildProvenance(
    allocator: Allocator,
    options: EventIngestOptions,
    event: InternalEventRequest,
    timestamp: i64,
    reason: []const u8,
) !std.json.Value {
    var object = std.json.ObjectMap.init(allocator);
    errdefer object.deinit();

    try object.put("gateway_id", .{ .string = try allocator.dupe(u8, options.gateway_id) });
    try object.put("lab_id", .{ .string = try allocator.dupe(u8, options.lab_id) });
    try object.put("service_id", .{ .string = try allocator.dupe(u8, event.service_id) });
    if (event.job_id) |job_id| {
        try object.put("job_id", .{ .string = try allocator.dupe(u8, job_id) });
    }
    try object.put("source_event_id", .{ .string = try allocator.dupe(u8, event.event_id) });
    try object.put("timestamp", .{ .integer = timestamp });
    try object.put("reason", .{ .string = try allocator.dupe(u8, reason) });

    if (event.provenance) |provenance| {
        switch (provenance) {
            .object => |map| {
                var it = map.iterator();
                while (it.next()) |entry| {
                    try object.put(try allocator.dupe(u8, entry.key_ptr.*), try cloneJsonValue(allocator, entry.value_ptr.*));
                }
            },
            else => {},
        }
    }

    return .{ .object = object };
}

fn eventSubjectId(allocator: Allocator, event: InternalEventRequest) ![]u8 {
    if (event.job_id) |job_id| {
        return std.fmt.allocPrint(allocator, "job:{s}", .{job_id});
    }
    return std.fmt.allocPrint(allocator, "service:{s}", .{event.service_id});
}

fn eventSubjectType(event_type: []const u8) []const u8 {
    if (std.mem.startsWith(u8, event_type, "inference.")) return "inference_job";
    if (std.mem.startsWith(u8, event_type, "rl.")) return "rl_job";
    if (std.mem.startsWith(u8, event_type, "training.")) return "training_job";
    return "service_event";
}

fn eventSubjectDisplayName(event: InternalEventRequest, fallback: []const u8) []const u8 {
    if (event.job_id) |job_id| return job_id;
    return fallback;
}

fn extractMutationIds(allocator: Allocator, response: std.json.Value) ![]const []const u8 {
    const root = switch (response) {
        .object => |object| object,
        else => return error.InvalidGatewayResponse,
    };
    const mutation_ids_value = root.get("mutation_ids") orelse return error.InvalidGatewayResponse;
    const items = switch (mutation_ids_value) {
        .array => |array| array.items,
        else => return error.InvalidGatewayResponse,
    };

    const result = try allocator.alloc([]const u8, items.len);
    errdefer allocator.free(result);
    for (items, 0..) |item, idx| {
        result[idx] = switch (item) {
            .string => |value| try allocator.dupe(u8, value),
            else => return error.InvalidGatewayResponse,
        };
    }
    return result;
}

fn cloneJsonValue(allocator: Allocator, value: std.json.Value) !std.json.Value {
    return switch (value) {
        .null => .{ .null = {} },
        .bool => |inner| .{ .bool = inner },
        .integer => |inner| .{ .integer = inner },
        .float => |inner| .{ .float = inner },
        .number_string => |inner| .{ .number_string = try allocator.dupe(u8, inner) },
        .string => |inner| .{ .string = try allocator.dupe(u8, inner) },
        .array => |inner| blk: {
            var array = std.json.Array.init(allocator);
            for (inner.items) |item| {
                try array.append(try cloneJsonValue(allocator, item));
            }
            break :blk .{ .array = array };
        },
        .object => |inner| blk: {
            var object = std.json.ObjectMap.init(allocator);
            var it = inner.iterator();
            while (it.next()) |entry| {
                try object.put(try allocator.dupe(u8, entry.key_ptr.*), try cloneJsonValue(allocator, entry.value_ptr.*));
            }
            break :blk .{ .object = object };
        },
    };
}
