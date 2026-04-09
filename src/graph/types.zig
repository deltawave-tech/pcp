const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Visibility = enum {
    local,
    shared,
    global,

    pub fn parse(value: []const u8) ?Visibility {
        if (std.mem.eql(u8, value, "local")) return .local;
        if (std.mem.eql(u8, value, "shared")) return .shared;
        if (std.mem.eql(u8, value, "global")) return .global;
        return null;
    }

    pub fn parseOrDefault(value: ?[]const u8) !Visibility {
        const raw = value orelse return .local;
        return parse(raw) orelse error.InvalidVisibility;
    }

    pub fn asString(self: Visibility) []const u8 {
        return switch (self) {
            .local => "local",
            .shared => "shared",
            .global => "global",
        };
    }
};

pub const MutationType = enum {
    upsert_entity,
    upsert_relation,
    append_observation,
    delete_entity,
    delete_relation,
    policy_update,

    pub fn parse(value: []const u8) ?MutationType {
        if (std.mem.eql(u8, value, "upsert_entity")) return .upsert_entity;
        if (std.mem.eql(u8, value, "upsert_relation")) return .upsert_relation;
        if (std.mem.eql(u8, value, "append_observation")) return .append_observation;
        if (std.mem.eql(u8, value, "delete_entity")) return .delete_entity;
        if (std.mem.eql(u8, value, "delete_relation")) return .delete_relation;
        if (std.mem.eql(u8, value, "policy_update")) return .policy_update;
        return null;
    }

    pub fn asString(self: MutationType) []const u8 {
        return switch (self) {
            .upsert_entity => "upsert_entity",
            .upsert_relation => "upsert_relation",
            .append_observation => "append_observation",
            .delete_entity => "delete_entity",
            .delete_relation => "delete_relation",
            .policy_update => "policy_update",
        };
    }
};

pub const MutationRequest = struct {
    mutation_id: ?[]const u8 = null,
    mutation_type: []const u8,
    namespace_id: []const u8,
    target_id: []const u8,
    payload: ?std.json.Value = null,
    visibility: ?[]const u8 = null,
    provenance: ?std.json.Value = null,
};

pub const MutateRequest = struct {
    mutations: []const MutationRequest,
};

pub const QueryRequest = struct {
    namespaces: ?[]const []const u8 = null,
    entity_types: ?[]const []const u8 = null,
    relations: ?[]const []const u8 = null,
    text: ?[]const u8 = null,
    limit: ?usize = null,
};

pub const GraphNamespace = struct {
    namespace_id: []u8,
    owner_gateway_id: []u8,
    owner_lab_id: []u8,
    visibility_default: Visibility,
    description: ?[]u8,
    created_at: i64,
    updated_at: i64,

    pub fn deinit(self: *GraphNamespace, allocator: Allocator) void {
        allocator.free(self.namespace_id);
        allocator.free(self.owner_gateway_id);
        allocator.free(self.owner_lab_id);
        if (self.description) |description| allocator.free(description);
    }
};

pub const GraphEntity = struct {
    entity_id: []u8,
    namespace_id: []u8,
    entity_type: []u8,
    display_name: []u8,
    properties_json: []u8,
    visibility: Visibility,
    provenance_json: []u8,
    created_at: i64,
    updated_at: i64,

    pub fn deinit(self: *GraphEntity, allocator: Allocator) void {
        allocator.free(self.entity_id);
        allocator.free(self.namespace_id);
        allocator.free(self.entity_type);
        allocator.free(self.display_name);
        allocator.free(self.properties_json);
        allocator.free(self.provenance_json);
    }
};

pub const GraphRelation = struct {
    relation_id: []u8,
    namespace_id: []u8,
    relation_type: []u8,
    from_entity_id: []u8,
    to_entity_id: []u8,
    properties_json: []u8,
    visibility: Visibility,
    provenance_json: []u8,
    created_at: i64,
    updated_at: i64,

    pub fn deinit(self: *GraphRelation, allocator: Allocator) void {
        allocator.free(self.relation_id);
        allocator.free(self.namespace_id);
        allocator.free(self.relation_type);
        allocator.free(self.from_entity_id);
        allocator.free(self.to_entity_id);
        allocator.free(self.properties_json);
        allocator.free(self.provenance_json);
    }
};

pub const GraphObservation = struct {
    observation_id: []u8,
    namespace_id: []u8,
    subject_entity_id: []u8,
    observation_type: []u8,
    payload_json: []u8,
    visibility: Visibility,
    provenance_json: []u8,
    created_at: i64,

    pub fn deinit(self: *GraphObservation, allocator: Allocator) void {
        allocator.free(self.observation_id);
        allocator.free(self.namespace_id);
        allocator.free(self.subject_entity_id);
        allocator.free(self.observation_type);
        allocator.free(self.payload_json);
        allocator.free(self.provenance_json);
    }
};
