const std = @import("std");

const Allocator = std.mem.Allocator;
const mutation_log = @import("mutation_log.zig");
const types = @import("types.zig");

pub const MemoryGraphStore = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    backend: []u8,
    gateway_id: []u8,
    lab_id: []u8,
    namespaces: std.ArrayList(types.GraphNamespace),
    entities: std.ArrayList(types.GraphEntity),
    relations: std.ArrayList(types.GraphRelation),
    observations: std.ArrayList(types.GraphObservation),
    mutation_log: mutation_log.MutationLog,

    const Self = @This();

    pub fn init(allocator: Allocator, backend: []const u8, gateway_id: []const u8, lab_id: []const u8) !Self {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .backend = try allocator.dupe(u8, backend),
            .gateway_id = try allocator.dupe(u8, gateway_id),
            .lab_id = try allocator.dupe(u8, lab_id),
            .namespaces = std.ArrayList(types.GraphNamespace).init(allocator),
            .entities = std.ArrayList(types.GraphEntity).init(allocator),
            .relations = std.ArrayList(types.GraphRelation).init(allocator),
            .observations = std.ArrayList(types.GraphObservation).init(allocator),
            .mutation_log = mutation_log.MutationLog.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.mutation_log.deinit();
        for (self.observations.items) |*observation| observation.deinit(self.allocator);
        for (self.relations.items) |*relation| relation.deinit(self.allocator);
        for (self.entities.items) |*entity| entity.deinit(self.allocator);
        for (self.namespaces.items) |*namespace| namespace.deinit(self.allocator);

        self.observations.deinit();
        self.relations.deinit();
        self.entities.deinit();
        self.namespaces.deinit();
        self.allocator.free(self.lab_id);
        self.allocator.free(self.gateway_id);
        self.allocator.free(self.backend);
    }

    pub fn renderStatusJson(self: *Self, allocator: Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        return std.json.stringifyAlloc(allocator, .{
            .backend = self.backend,
            .namespaces = self.namespaces.items.len,
            .entities = self.entities.items.len,
            .relations = self.relations.items.len,
            .observations = self.observations.items.len,
            .mutations = self.mutation_log.count(),
            .pending_mutations = self.mutation_log.countPending(),
            .last_replicated_sequence = self.mutation_log.last_replicated_sequence,
            .last_sequence = self.mutation_log.lastSequence(),
            .federation_enabled = false,
        }, .{});
    }

    pub fn applyMutationsJson(self: *Self, allocator: Allocator, request: types.MutateRequest) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var mutation_ids = std.ArrayList([]const u8).init(allocator);
        defer mutation_ids.deinit();

        for (request.mutations) |mutation| {
            const mutation_type = types.MutationType.parse(mutation.mutation_type) orelse return error.InvalidMutationType;
            const visibility = try types.Visibility.parseOrDefault(mutation.visibility);
            const now = std.time.timestamp();
            const payload_json = try jsonStringifyOrDefault(allocator, mutation.payload, "null");
            defer allocator.free(payload_json);
            const provenance_json = try jsonStringifyOrDefault(allocator, mutation.provenance, "null");
            defer allocator.free(provenance_json);

            try self.ensureNamespaceLocked(mutation.namespace_id, visibility, now);

            switch (mutation_type) {
                .upsert_entity => try self.upsertEntityLocked(mutation, visibility, now, provenance_json),
                .upsert_relation => try self.upsertRelationLocked(mutation, visibility, now, provenance_json),
                .append_observation => try self.appendObservationLocked(mutation, visibility, now, provenance_json),
                .delete_entity => self.deleteEntityLocked(mutation.target_id),
                .delete_relation => self.deleteRelationLocked(mutation.target_id),
                .policy_update => {},
            }

            const append_result = try self.mutation_log.append(self.gateway_id, self.lab_id, .{
                .mutation_id = mutation.mutation_id,
                .namespace_id = mutation.namespace_id,
                .mutation_type = mutation_type,
                .target_type = targetTypeForMutation(mutation_type),
                .target_id = mutation.target_id,
                .payload_json = payload_json,
                .visibility = visibility,
                .provenance_json = provenance_json,
                .timestamp = now,
            });
            try mutation_ids.append(append_result.mutation_id);
        }

        return std.json.stringifyAlloc(allocator, .{
            .accepted = true,
            .mutation_ids = mutation_ids.items,
        }, .{});
    }

    pub fn renderQueryJson(self: *Self, allocator: Allocator, request: types.QueryRequest) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const limit = request.limit orelse 20;
        var body = std.ArrayList(u8).init(allocator);
        errdefer body.deinit();
        var writer = body.writer();

        try writer.writeAll("{\"entities\":[");
        var emitted_entities: usize = 0;
        for (self.entities.items) |entity| {
            if (!matchesEntity(request, entity)) continue;
            if (emitted_entities >= limit) break;
            if (emitted_entities > 0) try writer.writeByte(',');
            try writeEntityJson(writer, entity);
            emitted_entities += 1;
        }

        try writer.writeAll("],\"relations\":[");
        var emitted_relations: usize = 0;
        for (self.relations.items) |relation| {
            if (!matchesRelation(request, relation)) continue;
            if (emitted_relations >= limit) break;
            if (emitted_relations > 0) try writer.writeByte(',');
            try writeRelationJson(writer, relation);
            emitted_relations += 1;
        }

        try writer.writeAll("],\"observations\":[");
        var emitted_observations: usize = 0;
        for (self.observations.items) |observation| {
            if (!matchesObservation(request, observation)) continue;
            if (emitted_observations >= limit) break;
            if (emitted_observations > 0) try writer.writeByte(',');
            try writeObservationJson(writer, observation);
            emitted_observations += 1;
        }
        try writer.writeAll("]}");

        return body.toOwnedSlice();
    }

    fn ensureNamespaceLocked(self: *Self, namespace_id: []const u8, visibility: types.Visibility, now: i64) !void {
        if (self.findNamespaceIndexLocked(namespace_id) != null) return;

        try self.namespaces.append(.{
            .namespace_id = try self.allocator.dupe(u8, namespace_id),
            .owner_gateway_id = try self.allocator.dupe(u8, self.gateway_id),
            .owner_lab_id = try self.allocator.dupe(u8, self.lab_id),
            .visibility_default = visibility,
            .description = null,
            .created_at = now,
            .updated_at = now,
        });
    }

    fn upsertEntityLocked(self: *Self, mutation: types.MutationRequest, visibility: types.Visibility, now: i64, provenance_json: []const u8) !void {
        const payload = mutation.payload orelse return error.InvalidPayload;
        const payload_object = switch (payload) {
            .object => |object| object,
            else => return error.InvalidPayload,
        };

        const entity_type = try requiredString(payload_object, "entity_type");
        const display_name = optionalString(payload_object, "display_name") orelse mutation.target_id;
        const properties_json = try stringifyObjectField(self.allocator, payload_object, "properties", "{}");
        defer self.allocator.free(properties_json);

        if (self.findEntityIndexLocked(mutation.target_id)) |idx| {
            var entity = &self.entities.items[idx];
            try replaceString(self.allocator, &entity.namespace_id, mutation.namespace_id);
            try replaceString(self.allocator, &entity.entity_type, entity_type);
            try replaceString(self.allocator, &entity.display_name, display_name);
            try replaceString(self.allocator, &entity.properties_json, properties_json);
            try replaceString(self.allocator, &entity.provenance_json, provenance_json);
            entity.visibility = visibility;
            entity.updated_at = now;
            return;
        }

        try self.entities.append(.{
            .entity_id = try self.allocator.dupe(u8, mutation.target_id),
            .namespace_id = try self.allocator.dupe(u8, mutation.namespace_id),
            .entity_type = try self.allocator.dupe(u8, entity_type),
            .display_name = try self.allocator.dupe(u8, display_name),
            .properties_json = try self.allocator.dupe(u8, properties_json),
            .visibility = visibility,
            .provenance_json = try self.allocator.dupe(u8, provenance_json),
            .created_at = now,
            .updated_at = now,
        });
    }

    fn upsertRelationLocked(self: *Self, mutation: types.MutationRequest, visibility: types.Visibility, now: i64, provenance_json: []const u8) !void {
        const payload = mutation.payload orelse return error.InvalidPayload;
        const payload_object = switch (payload) {
            .object => |object| object,
            else => return error.InvalidPayload,
        };

        const relation_type = try requiredString(payload_object, "relation_type");
        const from_entity_id = try requiredString(payload_object, "from_entity_id");
        const to_entity_id = try requiredString(payload_object, "to_entity_id");
        const properties_json = try stringifyObjectField(self.allocator, payload_object, "properties", "{}");
        defer self.allocator.free(properties_json);

        if (self.findRelationIndexLocked(mutation.target_id)) |idx| {
            var relation = &self.relations.items[idx];
            try replaceString(self.allocator, &relation.namespace_id, mutation.namespace_id);
            try replaceString(self.allocator, &relation.relation_type, relation_type);
            try replaceString(self.allocator, &relation.from_entity_id, from_entity_id);
            try replaceString(self.allocator, &relation.to_entity_id, to_entity_id);
            try replaceString(self.allocator, &relation.properties_json, properties_json);
            try replaceString(self.allocator, &relation.provenance_json, provenance_json);
            relation.visibility = visibility;
            relation.updated_at = now;
            return;
        }

        try self.relations.append(.{
            .relation_id = try self.allocator.dupe(u8, mutation.target_id),
            .namespace_id = try self.allocator.dupe(u8, mutation.namespace_id),
            .relation_type = try self.allocator.dupe(u8, relation_type),
            .from_entity_id = try self.allocator.dupe(u8, from_entity_id),
            .to_entity_id = try self.allocator.dupe(u8, to_entity_id),
            .properties_json = try self.allocator.dupe(u8, properties_json),
            .visibility = visibility,
            .provenance_json = try self.allocator.dupe(u8, provenance_json),
            .created_at = now,
            .updated_at = now,
        });
    }

    fn appendObservationLocked(self: *Self, mutation: types.MutationRequest, visibility: types.Visibility, now: i64, provenance_json: []const u8) !void {
        const payload = mutation.payload orelse return error.InvalidPayload;
        const payload_object = switch (payload) {
            .object => |object| object,
            else => return error.InvalidPayload,
        };

        const observation_type = try requiredString(payload_object, "observation_type");
        const subject_entity_id = try requiredString(payload_object, "subject_entity_id");
        const nested_payload_json = try stringifyValueField(self.allocator, payload_object, "payload", payload);
        defer self.allocator.free(nested_payload_json);

        if (self.findObservationIndexLocked(mutation.target_id)) |idx| {
            var observation = &self.observations.items[idx];
            try replaceString(self.allocator, &observation.namespace_id, mutation.namespace_id);
            try replaceString(self.allocator, &observation.subject_entity_id, subject_entity_id);
            try replaceString(self.allocator, &observation.observation_type, observation_type);
            try replaceString(self.allocator, &observation.payload_json, nested_payload_json);
            try replaceString(self.allocator, &observation.provenance_json, provenance_json);
            observation.visibility = visibility;
            observation.created_at = now;
            return;
        }

        try self.observations.append(.{
            .observation_id = try self.allocator.dupe(u8, mutation.target_id),
            .namespace_id = try self.allocator.dupe(u8, mutation.namespace_id),
            .subject_entity_id = try self.allocator.dupe(u8, subject_entity_id),
            .observation_type = try self.allocator.dupe(u8, observation_type),
            .payload_json = try self.allocator.dupe(u8, nested_payload_json),
            .visibility = visibility,
            .provenance_json = try self.allocator.dupe(u8, provenance_json),
            .created_at = now,
        });
    }

    fn deleteEntityLocked(self: *Self, entity_id: []const u8) void {
        if (self.findEntityIndexLocked(entity_id)) |idx| {
            var entity = self.entities.orderedRemove(idx);
            entity.deinit(self.allocator);
        }

        var relation_idx: usize = 0;
        while (relation_idx < self.relations.items.len) {
            const relation = &self.relations.items[relation_idx];
            if (std.mem.eql(u8, relation.from_entity_id, entity_id) or std.mem.eql(u8, relation.to_entity_id, entity_id)) {
                var removed_relation = self.relations.orderedRemove(relation_idx);
                removed_relation.deinit(self.allocator);
                continue;
            }
            relation_idx += 1;
        }

        var observation_idx: usize = 0;
        while (observation_idx < self.observations.items.len) {
            const observation = &self.observations.items[observation_idx];
            if (std.mem.eql(u8, observation.subject_entity_id, entity_id)) {
                var removed_observation = self.observations.orderedRemove(observation_idx);
                removed_observation.deinit(self.allocator);
                continue;
            }
            observation_idx += 1;
        }
    }

    fn deleteRelationLocked(self: *Self, relation_id: []const u8) void {
        if (self.findRelationIndexLocked(relation_id)) |idx| {
            var relation = self.relations.orderedRemove(idx);
            relation.deinit(self.allocator);
        }
    }

    fn findNamespaceIndexLocked(self: *Self, namespace_id: []const u8) ?usize {
        for (self.namespaces.items, 0..) |namespace, idx| {
            if (std.mem.eql(u8, namespace.namespace_id, namespace_id)) return idx;
        }
        return null;
    }

    fn findEntityIndexLocked(self: *Self, entity_id: []const u8) ?usize {
        for (self.entities.items, 0..) |entity, idx| {
            if (std.mem.eql(u8, entity.entity_id, entity_id)) return idx;
        }
        return null;
    }

    fn findRelationIndexLocked(self: *Self, relation_id: []const u8) ?usize {
        for (self.relations.items, 0..) |relation, idx| {
            if (std.mem.eql(u8, relation.relation_id, relation_id)) return idx;
        }
        return null;
    }

    fn findObservationIndexLocked(self: *Self, observation_id: []const u8) ?usize {
        for (self.observations.items, 0..) |observation, idx| {
            if (std.mem.eql(u8, observation.observation_id, observation_id)) return idx;
        }
        return null;
    }
};

fn matchesEntity(request: types.QueryRequest, entity: types.GraphEntity) bool {
    if (!matchesNamespaces(request.namespaces, entity.namespace_id)) return false;
    if (!matchesEntityTypes(request.entity_types, entity.entity_type)) return false;
    return matchesText(request.text, &.{
        entity.entity_id,
        entity.entity_type,
        entity.display_name,
        entity.properties_json,
        entity.provenance_json,
    });
}

fn matchesRelation(request: types.QueryRequest, relation: types.GraphRelation) bool {
    if (!matchesNamespaces(request.namespaces, relation.namespace_id)) return false;
    if (!matchesRelationTypes(request.relations, relation.relation_type)) return false;
    return matchesText(request.text, &.{
        relation.relation_id,
        relation.relation_type,
        relation.from_entity_id,
        relation.to_entity_id,
        relation.properties_json,
        relation.provenance_json,
    });
}

fn matchesObservation(request: types.QueryRequest, observation: types.GraphObservation) bool {
    if (!matchesNamespaces(request.namespaces, observation.namespace_id)) return false;
    return matchesText(request.text, &.{
        observation.observation_id,
        observation.subject_entity_id,
        observation.observation_type,
        observation.payload_json,
        observation.provenance_json,
    });
}

fn matchesNamespaces(filter: ?[]const []const u8, namespace_id: []const u8) bool {
    const namespaces = filter orelse return true;
    for (namespaces) |candidate| {
        if (std.mem.eql(u8, candidate, namespace_id)) return true;
    }
    return false;
}

fn matchesEntityTypes(filter: ?[]const []const u8, entity_type: []const u8) bool {
    const entity_types = filter orelse return true;
    for (entity_types) |candidate| {
        if (std.mem.eql(u8, candidate, entity_type)) return true;
    }
    return false;
}

fn matchesRelationTypes(filter: ?[]const []const u8, relation_type: []const u8) bool {
    const relation_types = filter orelse return true;
    for (relation_types) |candidate| {
        if (std.mem.eql(u8, candidate, relation_type)) return true;
    }
    return false;
}

fn matchesText(filter: ?[]const u8, haystacks: []const []const u8) bool {
    const text = filter orelse return true;
    if (text.len == 0) return true;
    for (haystacks) |haystack| {
        if (std.mem.indexOf(u8, haystack, text) != null) return true;
    }
    return false;
}

fn writeEntityJson(writer: anytype, entity: types.GraphEntity) !void {
    try writer.writeAll("{\"entity_id\":");
    try std.json.stringify(entity.entity_id, .{}, writer);
    try writer.writeAll(",\"namespace_id\":");
    try std.json.stringify(entity.namespace_id, .{}, writer);
    try writer.writeAll(",\"entity_type\":");
    try std.json.stringify(entity.entity_type, .{}, writer);
    try writer.writeAll(",\"display_name\":");
    try std.json.stringify(entity.display_name, .{}, writer);
    try writer.writeAll(",\"visibility\":");
    try std.json.stringify(entity.visibility.asString(), .{}, writer);
    try writer.writeAll(",\"properties\":");
    try writer.writeAll(entity.properties_json);
    try writer.writeAll(",\"provenance\":");
    try writer.writeAll(entity.provenance_json);
    try writer.print(",\"created_at\":{d},\"updated_at\":{d}", .{ entity.created_at, entity.updated_at });
    try writer.writeByte('}');
}

fn writeRelationJson(writer: anytype, relation: types.GraphRelation) !void {
    try writer.writeAll("{\"relation_id\":");
    try std.json.stringify(relation.relation_id, .{}, writer);
    try writer.writeAll(",\"namespace_id\":");
    try std.json.stringify(relation.namespace_id, .{}, writer);
    try writer.writeAll(",\"relation_type\":");
    try std.json.stringify(relation.relation_type, .{}, writer);
    try writer.writeAll(",\"from_entity_id\":");
    try std.json.stringify(relation.from_entity_id, .{}, writer);
    try writer.writeAll(",\"to_entity_id\":");
    try std.json.stringify(relation.to_entity_id, .{}, writer);
    try writer.writeAll(",\"visibility\":");
    try std.json.stringify(relation.visibility.asString(), .{}, writer);
    try writer.writeAll(",\"properties\":");
    try writer.writeAll(relation.properties_json);
    try writer.writeAll(",\"provenance\":");
    try writer.writeAll(relation.provenance_json);
    try writer.print(",\"created_at\":{d},\"updated_at\":{d}", .{ relation.created_at, relation.updated_at });
    try writer.writeByte('}');
}

fn writeObservationJson(writer: anytype, observation: types.GraphObservation) !void {
    try writer.writeAll("{\"observation_id\":");
    try std.json.stringify(observation.observation_id, .{}, writer);
    try writer.writeAll(",\"namespace_id\":");
    try std.json.stringify(observation.namespace_id, .{}, writer);
    try writer.writeAll(",\"subject_entity_id\":");
    try std.json.stringify(observation.subject_entity_id, .{}, writer);
    try writer.writeAll(",\"observation_type\":");
    try std.json.stringify(observation.observation_type, .{}, writer);
    try writer.writeAll(",\"visibility\":");
    try std.json.stringify(observation.visibility.asString(), .{}, writer);
    try writer.writeAll(",\"payload\":");
    try writer.writeAll(observation.payload_json);
    try writer.writeAll(",\"provenance\":");
    try writer.writeAll(observation.provenance_json);
    try writer.print(",\"created_at\":{d}", .{observation.created_at});
    try writer.writeByte('}');
}

fn requiredString(object: std.json.ObjectMap, field: []const u8) ![]const u8 {
    const value = object.get(field) orelse return error.InvalidPayload;
    return switch (value) {
        .string => |string_value| string_value,
        else => error.InvalidPayload,
    };
}

fn optionalString(object: std.json.ObjectMap, field: []const u8) ?[]const u8 {
    const value = object.get(field) orelse return null;
    return switch (value) {
        .string => |string_value| string_value,
        else => null,
    };
}

fn stringifyObjectField(allocator: Allocator, object: std.json.ObjectMap, field: []const u8, default_json: []const u8) ![]u8 {
    const value = object.get(field) orelse return allocator.dupe(u8, default_json);
    return std.json.stringifyAlloc(allocator, value, .{});
}

fn stringifyValueField(allocator: Allocator, object: std.json.ObjectMap, field: []const u8, fallback: std.json.Value) ![]u8 {
    if (object.get(field)) |value| {
        return std.json.stringifyAlloc(allocator, value, .{});
    }
    return std.json.stringifyAlloc(allocator, fallback, .{});
}

fn jsonStringifyOrDefault(allocator: Allocator, value: ?std.json.Value, default_json: []const u8) ![]u8 {
    if (value) |inner| {
        return std.json.stringifyAlloc(allocator, inner, .{});
    }
    return allocator.dupe(u8, default_json);
}

fn targetTypeForMutation(mutation_type: types.MutationType) []const u8 {
    return switch (mutation_type) {
        .upsert_entity, .delete_entity => "entity",
        .upsert_relation, .delete_relation => "relation",
        .append_observation => "observation",
        .policy_update => "policy",
    };
}

fn replaceString(allocator: Allocator, slot: *[]u8, value: []const u8) !void {
    allocator.free(slot.*);
    slot.* = try allocator.dupe(u8, value);
}
