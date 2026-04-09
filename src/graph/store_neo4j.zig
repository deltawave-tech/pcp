const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const types = @import("types.zig");

pub const Config = struct {
    uri: []const u8,
    http_uri: ?[]const u8 = null,
    user: []const u8,
    password_env: ?[]const u8 = null,
    database: ?[]const u8 = null,
    query_timeout_ms: u32 = 5_000,
    bootstrap_on_connect: bool = true,
};

pub const Neo4jGraphStore = struct {
    allocator: Allocator,
    config: Config,
    uri: []u8,
    http_uri: []u8,
    commit_uri: []u8,
    user: []u8,
    password: ?[]u8,
    password_env: ?[]u8,
    database: ?[]u8,
    query_timeout_ms: u32,
    bootstrap_on_connect: bool,
    is_connected: bool,
    schema_ready: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, config: Config) !Self {
        if (config.uri.len == 0 or config.user.len == 0) {
            return error.InvalidNeo4jConfig;
        }

        const http_uri = try resolveHttpUri(allocator, config);
        errdefer allocator.free(http_uri);
        const database_name = config.database orelse "neo4j";
        const commit_uri = try std.fmt.allocPrint(allocator, "{s}/db/{s}/tx/commit", .{ http_uri, database_name });
        errdefer allocator.free(commit_uri);
        const password = if (config.password_env) |password_env|
            std.process.getEnvVarOwned(allocator, password_env) catch |err| switch (err) {
                error.EnvironmentVariableNotFound => return error.MissingNeo4jPassword,
                else => return err,
            }
        else
            null;
        errdefer if (password) |secret| allocator.free(secret);

        return .{
            .allocator = allocator,
            .config = config,
            .uri = try allocator.dupe(u8, config.uri),
            .http_uri = http_uri,
            .commit_uri = commit_uri,
            .user = try allocator.dupe(u8, config.user),
            .password = password,
            .password_env = if (config.password_env) |password_env| try allocator.dupe(u8, password_env) else null,
            .database = if (config.database) |database| try allocator.dupe(u8, database) else null,
            .query_timeout_ms = config.query_timeout_ms,
            .bootstrap_on_connect = config.bootstrap_on_connect,
            .is_connected = true,
            .schema_ready = !config.bootstrap_on_connect,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.uri);
        self.allocator.free(self.http_uri);
        self.allocator.free(self.commit_uri);
        self.allocator.free(self.user);
        if (self.password) |password| self.allocator.free(password);
        if (self.password_env) |password_env| self.allocator.free(password_env);
        if (self.database) |database| self.allocator.free(database);
    }

    pub fn counts(self: *Self) types.GraphCounts {
        const statement =
            \\MATCH (ns:KGNamespace)
            \\WITH count(ns) AS namespaces
            \\OPTIONAL MATCH (e:KGEntity)
            \\WHERE coalesce(e.stub, false) = false
            \\WITH namespaces, count(e) AS entities
            \\OPTIONAL MATCH ()-[r:KG_RELATION]->()
            \\WITH namespaces, entities, count(r) AS relations
            \\OPTIONAL MATCH (o:KGObservation)
            \\WITH namespaces, entities, relations, count(o) AS observations
            \\RETURN {
            \\  namespaces: namespaces,
            \\  entities: entities,
            \\  relations: relations,
            \\  observations: observations
            \\}
        ;
        const response = self.runStatement(self.allocator, statement, null) catch return .{};
        defer response.deinit();
        return parseCounts(response.value) catch .{};
    }

    pub fn applyMutation(self: *Self, gateway_id: []const u8, lab_id: []const u8, mutation: types.MutationRequest, mutation_type: types.MutationType, visibility: types.Visibility, provenance_json: []const u8, now: i64) !void {
        try self.bootstrapIfNeeded();

        const payload_json = try jsonStringifyOrDefault(self.allocator, mutation.payload, "null");
        defer self.allocator.free(payload_json);

        const parameter_names = try mutationParametersJson(
            self.allocator,
            gateway_id,
            lab_id,
            mutation,
            mutation_type,
            visibility,
            provenance_json,
            payload_json,
            now,
        );
        defer self.allocator.free(parameter_names);

        const statement = switch (mutation_type) {
            .upsert_entity => entityMutationStatement,
            .upsert_relation => relationMutationStatement,
            .append_observation => observationMutationStatement,
            .delete_entity => deleteEntityStatement,
            .delete_relation => deleteRelationStatement,
            .policy_update => return,
        };

        var params = try std.json.parseFromSlice(std.json.Value, self.allocator, parameter_names, .{});
        defer params.deinit();
        var response = try self.runStatement(self.allocator, statement, params.value);
        response.deinit();
    }

    pub fn renderQueryJson(self: *Self, allocator: Allocator, request: types.QueryRequest) ![]u8 {
        try self.bootstrapIfNeeded();

        const param_json = try queryParametersJson(allocator, request);
        defer allocator.free(param_json);
        var params = try std.json.parseFromSlice(std.json.Value, allocator, param_json, .{});
        defer params.deinit();

        var entities_resp = try self.runStatement(allocator, entityQueryStatement, params.value);
        defer entities_resp.deinit();
        var relations_resp = try self.runStatement(allocator, relationQueryStatement, params.value);
        defer relations_resp.deinit();
        var observations_resp = try self.runStatement(allocator, observationQueryStatement, params.value);
        defer observations_resp.deinit();

        const entities = try extractRowObjects(allocator, entities_resp.value);
        defer entities.deinit();
        const relations = try extractRowObjects(allocator, relations_resp.value);
        defer relations.deinit();
        const observations = try extractRowObjects(allocator, observations_resp.value);
        defer observations.deinit();

        var body = ArrayList(u8).init(allocator);
        errdefer body.deinit();
        var writer = body.writer();

        try writer.writeAll("{\"entities\":[");
        for (entities.items, 0..) |row, idx| {
            if (idx > 0) try writer.writeByte(',');
            try writeEntityRow(writer, row);
        }
        try writer.writeAll("],\"relations\":[");
        for (relations.items, 0..) |row, idx| {
            if (idx > 0) try writer.writeByte(',');
            try writeRelationRow(writer, row);
        }
        try writer.writeAll("],\"observations\":[");
        for (observations.items, 0..) |row, idx| {
            if (idx > 0) try writer.writeByte(',');
            try writeObservationRow(writer, row);
        }
        try writer.writeAll("]}");

        return body.toOwnedSlice();
    }

    fn bootstrapIfNeeded(self: *Self) !void {
        if (self.schema_ready or !self.bootstrap_on_connect) return;

        const bootstrap_statements = [_][]const u8{
            "CREATE CONSTRAINT kg_namespace_id IF NOT EXISTS FOR (n:KGNamespace) REQUIRE n.namespace_id IS UNIQUE",
            "CREATE CONSTRAINT kg_entity_id IF NOT EXISTS FOR (n:KGEntity) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT kg_observation_id IF NOT EXISTS FOR (n:KGObservation) REQUIRE n.observation_id IS UNIQUE",
        };
        for (bootstrap_statements) |statement| {
            var response = try self.runStatement(self.allocator, statement, null);
            response.deinit();
        }
        self.schema_ready = true;
    }

    fn runStatement(self: *Self, allocator: Allocator, statement: []const u8, parameters: ?std.json.Value) !std.json.Parsed(std.json.Value) {
        var client = std.http.Client{ .allocator = allocator };
        defer client.deinit();

        const payload = try statementPayloadJson(allocator, statement, parameters);
        defer allocator.free(payload);

        const auth_header = try basicAuthHeader(allocator, self.user, self.password orelse "");
        defer allocator.free(auth_header);

        var response_body = ArrayList(u8).init(allocator);
        defer response_body.deinit();

        const extra_headers = [_]std.http.Header{
            .{ .name = "authorization", .value = auth_header },
            .{ .name = "content-type", .value = "application/json" },
            .{ .name = "accept", .value = "application/json" },
        };

        const result = try client.fetch(.{
            .location = .{ .url = self.commit_uri },
            .method = .POST,
            .payload = payload,
            .extra_headers = &extra_headers,
            .response_storage = .{ .dynamic = &response_body },
        });
        if (result.status != .ok and result.status != .accepted) {
            return error.Neo4jHttpError;
        }

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, response_body.items, .{});
        errdefer parsed.deinit();
        try ensureNoErrors(parsed.value);
        return parsed;
    }
};

const entityMutationStatement =
    \\MERGE (ns:KGNamespace {namespace_id: $namespace_id})
    \\SET ns.owner_gateway_id = $gateway_id,
    \\    ns.owner_lab_id = $lab_id,
    \\    ns.visibility_default = $visibility,
    \\    ns.updated_at = $now,
    \\    ns.created_at = coalesce(ns.created_at, $now)
    \\MERGE (e:KGEntity {entity_id: $target_id})
    \\SET e.namespace_id = $namespace_id,
    \\    e.entity_type = $entity_type,
    \\    e.display_name = $display_name,
    \\    e.stub = false,
    \\    e.properties_json = $properties_json,
    \\    e.visibility = $visibility,
    \\    e.provenance_json = $provenance_json,
    \\    e.updated_at = $now,
    \\    e.created_at = coalesce(e.created_at, $now)
;

const relationMutationStatement =
    \\MERGE (ns:KGNamespace {namespace_id: $namespace_id})
    \\SET ns.owner_gateway_id = $gateway_id,
    \\    ns.owner_lab_id = $lab_id,
    \\    ns.visibility_default = $visibility,
    \\    ns.updated_at = $now,
    \\    ns.created_at = coalesce(ns.created_at, $now)
    \\MERGE (src:KGEntity {entity_id: $from_entity_id})
    \\SET src.namespace_id = coalesce(src.namespace_id, $namespace_id),
    \\    src.entity_type = coalesce(src.entity_type, "unknown"),
    \\    src.display_name = coalesce(src.display_name, $from_entity_id),
    \\    src.stub = coalesce(src.stub, true),
    \\    src.properties_json = coalesce(src.properties_json, "{}"),
    \\    src.visibility = coalesce(src.visibility, $visibility),
    \\    src.provenance_json = coalesce(src.provenance_json, "null"),
    \\    src.updated_at = $now,
    \\    src.created_at = coalesce(src.created_at, $now)
    \\MERGE (dst:KGEntity {entity_id: $to_entity_id})
    \\SET dst.namespace_id = coalesce(dst.namespace_id, $namespace_id),
    \\    dst.entity_type = coalesce(dst.entity_type, "unknown"),
    \\    dst.display_name = coalesce(dst.display_name, $to_entity_id),
    \\    dst.stub = coalesce(dst.stub, true),
    \\    dst.properties_json = coalesce(dst.properties_json, "{}"),
    \\    dst.visibility = coalesce(dst.visibility, $visibility),
    \\    dst.provenance_json = coalesce(dst.provenance_json, "null"),
    \\    dst.updated_at = $now,
    \\    dst.created_at = coalesce(dst.created_at, $now)
    \\MERGE (src)-[r:KG_RELATION {relation_id: $target_id}]->(dst)
    \\SET r.namespace_id = $namespace_id,
    \\    r.relation_type = $relation_type,
    \\    r.properties_json = $properties_json,
    \\    r.visibility = $visibility,
    \\    r.provenance_json = $provenance_json,
    \\    r.updated_at = $now,
    \\    r.created_at = coalesce(r.created_at, $now)
;

const observationMutationStatement =
    \\MERGE (ns:KGNamespace {namespace_id: $namespace_id})
    \\SET ns.owner_gateway_id = $gateway_id,
    \\    ns.owner_lab_id = $lab_id,
    \\    ns.visibility_default = $visibility,
    \\    ns.updated_at = $now,
    \\    ns.created_at = coalesce(ns.created_at, $now)
    \\MERGE (subject:KGEntity {entity_id: $subject_entity_id})
    \\SET subject.namespace_id = coalesce(subject.namespace_id, $namespace_id),
    \\    subject.entity_type = coalesce(subject.entity_type, "unknown"),
    \\    subject.display_name = coalesce(subject.display_name, $subject_entity_id),
    \\    subject.stub = coalesce(subject.stub, true),
    \\    subject.properties_json = coalesce(subject.properties_json, "{}"),
    \\    subject.visibility = coalesce(subject.visibility, $visibility),
    \\    subject.provenance_json = coalesce(subject.provenance_json, "null"),
    \\    subject.updated_at = $now,
    \\    subject.created_at = coalesce(subject.created_at, $now)
    \\MERGE (o:KGObservation {observation_id: $target_id})
    \\SET o.namespace_id = $namespace_id,
    \\    o.subject_entity_id = $subject_entity_id,
    \\    o.observation_type = $observation_type,
    \\    o.payload_json = $payload_json,
    \\    o.visibility = $visibility,
    \\    o.provenance_json = $provenance_json,
    \\    o.created_at = $now
    \\MERGE (o)-[:OBSERVED_ON]->(subject)
;

const deleteEntityStatement = "MATCH (e:KGEntity {entity_id: $target_id}) DETACH DELETE e";
const deleteRelationStatement = "MATCH ()-[r:KG_RELATION {relation_id: $target_id}]->() DELETE r";

const entityQueryStatement =
    \\MATCH (e:KGEntity)
    \\WHERE coalesce(e.stub, false) = false
    \\  AND ($namespaces IS NULL OR e.namespace_id IN $namespaces)
    \\  AND ($entity_types IS NULL OR e.entity_type IN $entity_types)
    \\  AND ($text IS NULL OR e.entity_id CONTAINS $text OR e.entity_type CONTAINS $text OR e.display_name CONTAINS $text OR e.properties_json CONTAINS $text OR e.provenance_json CONTAINS $text)
    \\RETURN {
    \\  entity_id: e.entity_id,
    \\  namespace_id: e.namespace_id,
    \\  entity_type: e.entity_type,
    \\  display_name: e.display_name,
    \\  visibility: e.visibility,
    \\  properties_json: e.properties_json,
    \\  provenance_json: e.provenance_json,
    \\  created_at: e.created_at,
    \\  updated_at: e.updated_at
    \\}
    \\ORDER BY e.updated_at DESC
    \\LIMIT $limit
;

const relationQueryStatement =
    \\MATCH (src:KGEntity)-[r:KG_RELATION]->(dst:KGEntity)
    \\WHERE ($namespaces IS NULL OR r.namespace_id IN $namespaces)
    \\  AND ($relations IS NULL OR r.relation_type IN $relations)
    \\  AND ($text IS NULL OR r.relation_id CONTAINS $text OR r.relation_type CONTAINS $text OR src.entity_id CONTAINS $text OR dst.entity_id CONTAINS $text OR r.properties_json CONTAINS $text OR r.provenance_json CONTAINS $text)
    \\RETURN {
    \\  relation_id: r.relation_id,
    \\  namespace_id: r.namespace_id,
    \\  relation_type: r.relation_type,
    \\  from_entity_id: src.entity_id,
    \\  to_entity_id: dst.entity_id,
    \\  visibility: r.visibility,
    \\  properties_json: r.properties_json,
    \\  provenance_json: r.provenance_json,
    \\  created_at: r.created_at,
    \\  updated_at: r.updated_at
    \\}
    \\ORDER BY r.updated_at DESC
    \\LIMIT $limit
;

const observationQueryStatement =
    \\MATCH (o:KGObservation)
    \\WHERE ($namespaces IS NULL OR o.namespace_id IN $namespaces)
    \\  AND ($text IS NULL OR o.observation_id CONTAINS $text OR o.observation_type CONTAINS $text OR o.subject_entity_id CONTAINS $text OR o.payload_json CONTAINS $text OR o.provenance_json CONTAINS $text)
    \\RETURN {
    \\  observation_id: o.observation_id,
    \\  namespace_id: o.namespace_id,
    \\  subject_entity_id: o.subject_entity_id,
    \\  observation_type: o.observation_type,
    \\  visibility: o.visibility,
    \\  payload_json: o.payload_json,
    \\  provenance_json: o.provenance_json,
    \\  created_at: o.created_at
    \\}
    \\ORDER BY o.created_at DESC
    \\LIMIT $limit
;

fn resolveHttpUri(allocator: Allocator, config: Config) ![]u8 {
    if (config.http_uri) |http_uri| {
        return allocator.dupe(u8, std.mem.trimRight(u8, http_uri, "/"));
    }

    const parsed = try std.Uri.parse(config.uri);
    const host = parsed.host orelse return error.InvalidNeo4jConfig;
    return std.fmt.allocPrint(allocator, "http://{host}:7474", .{ .host = host });
}

fn statementPayloadJson(allocator: Allocator, statement: []const u8, parameters: ?std.json.Value) ![]u8 {
    const Statement = struct {
        statement: []const u8,
        parameters: ?std.json.Value = null,
    };
    return std.json.stringifyAlloc(allocator, .{
        .statements = &[_]Statement{.{ .statement = statement, .parameters = parameters }},
    }, .{});
}

fn basicAuthHeader(allocator: Allocator, user: []const u8, password: []const u8) ![]u8 {
    const raw = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ user, password });
    defer allocator.free(raw);
    const encoded_len = std.base64.standard.Encoder.calcSize(raw.len);
    const header = try allocator.alloc(u8, "Basic ".len + encoded_len);
    @memcpy(header[0.."Basic ".len], "Basic ");
    _ = std.base64.standard.Encoder.encode(header["Basic ".len..], raw);
    return header;
}

fn ensureNoErrors(response: std.json.Value) !void {
    const root = switch (response) {
        .object => |object| object,
        else => return error.InvalidNeo4jResponse,
    };
    const errors_value = root.get("errors") orelse return error.InvalidNeo4jResponse;
    const errors_array = switch (errors_value) {
        .array => |items| items.items,
        else => return error.InvalidNeo4jResponse,
    };
    if (errors_array.len == 0) return;
    return error.Neo4jQueryFailed;
}

fn parseCounts(response: std.json.Value) !types.GraphCounts {
    const root = switch (response) {
        .object => |object| object,
        else => return error.InvalidNeo4jResponse,
    };
    const results = switch (root.get("results") orelse return error.InvalidNeo4jResponse) {
        .array => |items| items.items,
        else => return error.InvalidNeo4jResponse,
    };
    if (results.len == 0) return .{};
    const data = switch (switch (results[0]) {
        .object => |object| object.get("data") orelse return error.InvalidNeo4jResponse,
        else => return error.InvalidNeo4jResponse,
    }) {
        .array => |items| items.items,
        else => return error.InvalidNeo4jResponse,
    };
    if (data.len == 0) return .{};
    const row = try firstRowObject(data[0]);
    return .{
        .namespaces = try intFieldAsUsize(row, "namespaces"),
        .entities = try intFieldAsUsize(row, "entities"),
        .relations = try intFieldAsUsize(row, "relations"),
        .observations = try intFieldAsUsize(row, "observations"),
    };
}

fn extractRowObjects(allocator: Allocator, response: std.json.Value) !std.ArrayList(std.json.ObjectMap) {
    const root = switch (response) {
        .object => |object| object,
        else => return error.InvalidNeo4jResponse,
    };
    const results = switch (root.get("results") orelse return error.InvalidNeo4jResponse) {
        .array => |items| items.items,
        else => return error.InvalidNeo4jResponse,
    };

    var rows = std.ArrayList(std.json.ObjectMap).init(allocator);
    errdefer rows.deinit();
    if (results.len == 0) return rows;

    const data = switch (switch (results[0]) {
        .object => |object| object.get("data") orelse return error.InvalidNeo4jResponse,
        else => return error.InvalidNeo4jResponse,
    }) {
        .array => |items| items.items,
        else => return error.InvalidNeo4jResponse,
    };

    for (data) |entry| {
        try rows.append(try firstRowObject(entry));
    }
    return rows;
}

fn firstRowObject(entry: std.json.Value) !std.json.ObjectMap {
    const object = switch (entry) {
        .object => |obj| obj,
        else => return error.InvalidNeo4jResponse,
    };
    const row = switch (object.get("row") orelse return error.InvalidNeo4jResponse) {
        .array => |items| items.items,
        else => return error.InvalidNeo4jResponse,
    };
    if (row.len == 0) return error.InvalidNeo4jResponse;
    return switch (row[0]) {
        .object => |obj| obj,
        else => return error.InvalidNeo4jResponse,
    };
}

fn intFieldAsUsize(object: std.json.ObjectMap, name: []const u8) !usize {
    const value = object.get(name) orelse return error.InvalidNeo4jResponse;
    return switch (value) {
        .integer => |v| @intCast(v),
        .float => |v| @intFromFloat(v),
        else => error.InvalidNeo4jResponse,
    };
}

fn mutationParametersJson(allocator: Allocator, gateway_id: []const u8, lab_id: []const u8, mutation: types.MutationRequest, mutation_type: types.MutationType, visibility: types.Visibility, provenance_json: []const u8, payload_json: []const u8, now: i64) ![]u8 {
    var entity_type: []const u8 = "";
    var display_name: []const u8 = mutation.target_id;
    var properties_json: []const u8 = "{}";
    var stored_payload_json: []const u8 = payload_json;
    var relation_type: []const u8 = "";
    var from_entity_id: []const u8 = "";
    var to_entity_id: []const u8 = "";
    var observation_type: []const u8 = "";
    var subject_entity_id: []const u8 = "";

    if (mutation.payload) |payload| {
        if (payload == .object) {
            const object = payload.object;
            entity_type = optionalStringField(object, "entity_type") orelse entity_type;
            display_name = optionalStringField(object, "display_name") orelse display_name;
            relation_type = optionalStringField(object, "relation_type") orelse relation_type;
            from_entity_id = optionalStringField(object, "from_entity_id") orelse from_entity_id;
            to_entity_id = optionalStringField(object, "to_entity_id") orelse to_entity_id;
            observation_type = optionalStringField(object, "observation_type") orelse observation_type;
            subject_entity_id = optionalStringField(object, "subject_entity_id") orelse subject_entity_id;
            if (object.get("properties")) |value| {
                properties_json = try std.json.stringifyAlloc(allocator, value, .{});
            }
            if (mutation_type == .append_observation) {
                if (object.get("payload")) |value| {
                    stored_payload_json = try std.json.stringifyAlloc(allocator, value, .{});
                }
            }
        }
    }
    defer if (properties_json.ptr != payload_json.ptr and properties_json.ptr != "{}".ptr) allocator.free(@constCast(properties_json));
    defer if (stored_payload_json.ptr != payload_json.ptr) allocator.free(@constCast(stored_payload_json));

    return std.json.stringifyAlloc(allocator, .{
        .gateway_id = gateway_id,
        .lab_id = lab_id,
        .namespace_id = mutation.namespace_id,
        .target_id = mutation.target_id,
        .visibility = visibility.asString(),
        .provenance_json = provenance_json,
        .payload_json = stored_payload_json,
        .entity_type = entity_type,
        .display_name = display_name,
        .properties_json = properties_json,
        .relation_type = relation_type,
        .from_entity_id = from_entity_id,
        .to_entity_id = to_entity_id,
        .observation_type = observation_type,
        .subject_entity_id = subject_entity_id,
        .now = now,
    }, .{});
}

fn queryParametersJson(allocator: Allocator, request: types.QueryRequest) ![]u8 {
    return std.json.stringifyAlloc(allocator, .{
        .namespaces = request.namespaces,
        .entity_types = request.entity_types,
        .relations = request.relations,
        .text = request.text,
        .limit = request.limit orelse 20,
    }, .{});
}

fn optionalStringField(object: std.json.ObjectMap, field: []const u8) ?[]const u8 {
    const value = object.get(field) orelse return null;
    return switch (value) {
        .string => |string_value| string_value,
        else => null,
    };
}

fn jsonStringifyOrDefault(allocator: Allocator, value: ?std.json.Value, default_json: []const u8) ![]u8 {
    if (value) |inner| {
        return std.json.stringifyAlloc(allocator, inner, .{});
    }
    return allocator.dupe(u8, default_json);
}

fn writeEntityRow(writer: anytype, row: std.json.ObjectMap) !void {
    try writer.writeAll("{\"entity_id\":");
    try writeJsonStringField(writer, row, "entity_id");
    try writer.writeAll(",\"namespace_id\":");
    try writeJsonStringField(writer, row, "namespace_id");
    try writer.writeAll(",\"entity_type\":");
    try writeJsonStringField(writer, row, "entity_type");
    try writer.writeAll(",\"display_name\":");
    try writeJsonStringField(writer, row, "display_name");
    try writer.writeAll(",\"visibility\":");
    try writeJsonStringField(writer, row, "visibility");
    try writer.writeAll(",\"properties\":");
    try writeRawJsonStringField(writer, row, "properties_json");
    try writer.writeAll(",\"provenance\":");
    try writeRawJsonStringField(writer, row, "provenance_json");
    try writer.writeAll(",\"created_at\":");
    try writeJsonNumberField(writer, row, "created_at");
    try writer.writeAll(",\"updated_at\":");
    try writeJsonNumberField(writer, row, "updated_at");
    try writer.writeByte('}');
}

fn writeRelationRow(writer: anytype, row: std.json.ObjectMap) !void {
    try writer.writeAll("{\"relation_id\":");
    try writeJsonStringField(writer, row, "relation_id");
    try writer.writeAll(",\"namespace_id\":");
    try writeJsonStringField(writer, row, "namespace_id");
    try writer.writeAll(",\"relation_type\":");
    try writeJsonStringField(writer, row, "relation_type");
    try writer.writeAll(",\"from_entity_id\":");
    try writeJsonStringField(writer, row, "from_entity_id");
    try writer.writeAll(",\"to_entity_id\":");
    try writeJsonStringField(writer, row, "to_entity_id");
    try writer.writeAll(",\"visibility\":");
    try writeJsonStringField(writer, row, "visibility");
    try writer.writeAll(",\"properties\":");
    try writeRawJsonStringField(writer, row, "properties_json");
    try writer.writeAll(",\"provenance\":");
    try writeRawJsonStringField(writer, row, "provenance_json");
    try writer.writeAll(",\"created_at\":");
    try writeJsonNumberField(writer, row, "created_at");
    try writer.writeAll(",\"updated_at\":");
    try writeJsonNumberField(writer, row, "updated_at");
    try writer.writeByte('}');
}

fn writeObservationRow(writer: anytype, row: std.json.ObjectMap) !void {
    try writer.writeAll("{\"observation_id\":");
    try writeJsonStringField(writer, row, "observation_id");
    try writer.writeAll(",\"namespace_id\":");
    try writeJsonStringField(writer, row, "namespace_id");
    try writer.writeAll(",\"subject_entity_id\":");
    try writeJsonStringField(writer, row, "subject_entity_id");
    try writer.writeAll(",\"observation_type\":");
    try writeJsonStringField(writer, row, "observation_type");
    try writer.writeAll(",\"visibility\":");
    try writeJsonStringField(writer, row, "visibility");
    try writer.writeAll(",\"payload\":");
    try writeRawJsonStringField(writer, row, "payload_json");
    try writer.writeAll(",\"provenance\":");
    try writeRawJsonStringField(writer, row, "provenance_json");
    try writer.writeAll(",\"created_at\":");
    try writeJsonNumberField(writer, row, "created_at");
    try writer.writeByte('}');
}

fn writeJsonStringField(writer: anytype, row: std.json.ObjectMap, name: []const u8) !void {
    const value = row.get(name) orelse return error.InvalidNeo4jResponse;
    switch (value) {
        .string => |text| try std.json.stringify(text, .{}, writer),
        else => return error.InvalidNeo4jResponse,
    }
}

fn writeRawJsonStringField(writer: anytype, row: std.json.ObjectMap, name: []const u8) !void {
    const value = row.get(name) orelse return error.InvalidNeo4jResponse;
    switch (value) {
        .string => |text| try writer.writeAll(text),
        else => return error.InvalidNeo4jResponse,
    }
}

fn writeJsonNumberField(writer: anytype, row: std.json.ObjectMap, name: []const u8) !void {
    const value = row.get(name) orelse return error.InvalidNeo4jResponse;
    switch (value) {
        .integer => |number| try writer.print("{d}", .{number}),
        .float => |number| try writer.print("{d}", .{number}),
        else => return error.InvalidNeo4jResponse,
    }
}
