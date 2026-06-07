const std = @import("std");

const Allocator = std.mem.Allocator;
const graph_types = @import("types.zig");

pub const PolicyUpdateRequest = struct {
    default_visibility: ?[]const u8 = null,
    allow_global_replication: ?bool = null,
    allow_raw_payload_export: ?bool = null,
};

pub const NamespacePolicySnapshot = struct {
    namespace_id: []const u8,
    default_visibility: []const u8,
    allow_global_replication: bool,
    allow_raw_payload_export: bool,
    updated_at: i64,
};

pub const NamespacePolicy = struct {
    namespace_id: []u8,
    default_visibility: graph_types.Visibility,
    allow_global_replication: bool,
    allow_raw_payload_export: bool,
    updated_at: i64,

    pub fn deinit(self: *NamespacePolicy, allocator: Allocator) void {
        allocator.free(self.namespace_id);
    }
};

pub const GraphPolicyStore = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    persistence_path: ?[]u8,
    policies: std.ArrayList(NamespacePolicy),

    const Self = @This();

    pub fn init(allocator: Allocator, persistence_path: ?[]const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .mutex = .{},
            .persistence_path = if (persistence_path) |path| try allocator.dupe(u8, path) else null,
            .policies = std.ArrayList(NamespacePolicy).init(allocator),
        };
        errdefer self.deinit();
        try self.loadFromDisk();
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.policies.items) |*policy| {
            policy.deinit(self.allocator);
        }
        self.policies.deinit();
        if (self.persistence_path) |path| self.allocator.free(path);
    }

    pub fn upsert(self: *Self, namespace_id: []const u8, request: PolicyUpdateRequest, default_visibility: graph_types.Visibility) !NamespacePolicySnapshot {
        self.mutex.lock();
        defer self.mutex.unlock();

        const visibility = if (request.default_visibility) |raw|
            graph_types.Visibility.parse(raw) orelse return error.InvalidVisibility
        else
            default_visibility;
        const now = std.time.timestamp();

        for (self.policies.items) |*policy| {
            if (!std.mem.eql(u8, policy.namespace_id, namespace_id)) continue;
            if (request.default_visibility != null) policy.default_visibility = visibility;
            if (request.allow_global_replication) |value| policy.allow_global_replication = value;
            if (request.allow_raw_payload_export) |value| policy.allow_raw_payload_export = value;
            policy.updated_at = now;
            try self.saveLocked();
            return snapshotOf(policy.*);
        }

        const policy = NamespacePolicy{
            .namespace_id = try self.allocator.dupe(u8, namespace_id),
            .default_visibility = visibility,
            .allow_global_replication = request.allow_global_replication orelse false,
            .allow_raw_payload_export = request.allow_raw_payload_export orelse false,
            .updated_at = now,
        };
        try self.policies.append(policy);
        errdefer {
            var owned = self.policies.pop();
            owned.deinit(self.allocator);
        }
        try self.saveLocked();
        return snapshotOf(policy);
    }

    pub fn applySnapshot(self: *Self, snapshot: NamespacePolicySnapshot) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const visibility = graph_types.Visibility.parse(snapshot.default_visibility) orelse return error.InvalidVisibility;
        for (self.policies.items) |*policy| {
            if (!std.mem.eql(u8, policy.namespace_id, snapshot.namespace_id)) continue;
            if (snapshot.updated_at < policy.updated_at) return;
            policy.default_visibility = visibility;
            policy.allow_global_replication = snapshot.allow_global_replication;
            policy.allow_raw_payload_export = snapshot.allow_raw_payload_export;
            policy.updated_at = snapshot.updated_at;
            try self.saveLocked();
            return;
        }

        try self.policies.append(.{
            .namespace_id = try self.allocator.dupe(u8, snapshot.namespace_id),
            .default_visibility = visibility,
            .allow_global_replication = snapshot.allow_global_replication,
            .allow_raw_payload_export = snapshot.allow_raw_payload_export,
            .updated_at = snapshot.updated_at,
        });
        try self.saveLocked();
    }

    pub fn getSnapshot(self: *Self, namespace_id: []const u8, default_visibility: graph_types.Visibility) NamespacePolicySnapshot {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.snapshotLocked(namespace_id, default_visibility);
    }

    pub fn listSnapshots(self: *Self, allocator: Allocator) ![]NamespacePolicySnapshot {
        self.mutex.lock();
        defer self.mutex.unlock();

        const snapshots = try allocator.alloc(NamespacePolicySnapshot, self.policies.items.len);
        for (self.policies.items, 0..) |policy, idx| {
            snapshots[idx] = snapshotOf(policy);
        }
        return snapshots;
    }

    pub fn renderJson(self: *Self, allocator: Allocator) ![]u8 {
        const snapshots = try self.listSnapshots(allocator);
        defer allocator.free(snapshots);
        return std.json.stringifyAlloc(allocator, .{ .policies = snapshots }, .{});
    }

    pub fn allowsReplication(self: *Self, namespace_id: []const u8, default_visibility: graph_types.Visibility, visibility: graph_types.Visibility, is_observation: bool) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (visibility == .local) return false;
        const snapshot = self.snapshotLocked(namespace_id, default_visibility);
        if (!snapshot.allow_global_replication) return false;
        if (is_observation and !snapshot.allow_raw_payload_export) return false;
        return true;
    }

    pub fn allowsIncoming(self: *Self, namespace_id: []const u8, default_visibility: graph_types.Visibility, visibility: graph_types.Visibility, is_observation: bool) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (visibility == .local) return false;
        const snapshot = self.snapshotLocked(namespace_id, default_visibility);
        if (!snapshot.allow_global_replication) return false;
        if (is_observation and !snapshot.allow_raw_payload_export) return false;
        return true;
    }

    fn snapshotLocked(self: *Self, namespace_id: []const u8, default_visibility: graph_types.Visibility) NamespacePolicySnapshot {
        for (self.policies.items) |policy| {
            if (std.mem.eql(u8, policy.namespace_id, namespace_id)) {
                return snapshotOf(policy);
            }
        }

        return .{
            .namespace_id = namespace_id,
            .default_visibility = default_visibility.asString(),
            .allow_global_replication = false,
            .allow_raw_payload_export = false,
            .updated_at = 0,
        };
    }

    fn loadFromDisk(self: *Self) !void {
        const path = self.persistence_path orelse return;
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => return,
            else => return err,
        };
        defer file.close();

        const bytes = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(bytes);

        const PersistedRoot = struct {
            policies: []const NamespacePolicySnapshot = &.{},
        };

        var parsed = try std.json.parseFromSlice(PersistedRoot, self.allocator, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        for (parsed.value.policies) |snapshot| {
            const visibility = graph_types.Visibility.parse(snapshot.default_visibility) orelse return error.InvalidVisibility;
            try self.policies.append(.{
                .namespace_id = try self.allocator.dupe(u8, snapshot.namespace_id),
                .default_visibility = visibility,
                .allow_global_replication = snapshot.allow_global_replication,
                .allow_raw_payload_export = snapshot.allow_raw_payload_export,
                .updated_at = snapshot.updated_at,
            });
        }
    }

    fn saveLocked(self: *Self) !void {
        const path = self.persistence_path orelse return;

        const snapshots = try self.allocator.alloc(NamespacePolicySnapshot, self.policies.items.len);
        defer self.allocator.free(snapshots);
        for (self.policies.items, 0..) |policy, idx| {
            snapshots[idx] = snapshotOf(policy);
        }

        const body = try std.json.stringifyAlloc(self.allocator, .{ .policies = snapshots }, .{ .whitespace = .indent_2 });
        defer self.allocator.free(body);

        var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        try file.writeAll(body);
    }
};

fn snapshotOf(policy: NamespacePolicy) NamespacePolicySnapshot {
    return .{
        .namespace_id = policy.namespace_id,
        .default_visibility = policy.default_visibility.asString(),
        .allow_global_replication = policy.allow_global_replication,
        .allow_raw_payload_export = policy.allow_raw_payload_export,
        .updated_at = policy.updated_at,
    };
}
