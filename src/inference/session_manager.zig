const std = @import("std");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;

pub const SessionManager = struct {
    allocator: Allocator,
    sessions: StringHashMap(types.SessionRecord),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .sessions = StringHashMap(types.SessionRecord).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            self.freeRecord(entry.value_ptr.*);
        }
        self.sessions.deinit();
    }

    pub fn upsert(self: *Self, record: types.SessionRecord) !void {
        const owned = try self.cloneRecord(record);
        errdefer self.freeRecord(owned);
        if (self.sessions.fetchRemove(owned.session_id)) |kv| {
            self.freeRecord(kv.value);
        }
        try self.sessions.put(owned.session_id, owned);
    }

    pub fn get(self: *Self, session_id: []const u8) ?types.SessionRecord {
        return self.sessions.get(session_id);
    }

    pub fn getPtr(self: *Self, session_id: []const u8) ?*types.SessionRecord {
        return self.sessions.getPtr(session_id);
    }

    pub fn remove(self: *Self, session_id: []const u8) void {
        if (self.sessions.fetchRemove(session_id)) |kv| {
            self.freeRecord(kv.value);
        }
    }

    pub fn touch(self: *Self, session_id: []const u8, now_ts: i64, ttl_seconds: u64) void {
        if (self.sessions.getPtr(session_id)) |session| {
            session.last_access_ts = now_ts;
            session.expires_at = now_ts + @as(i64, @intCast(ttl_seconds));
            session.state = .active;
        }
    }

    pub fn expire(self: *Self, now_ts: i64) void {
        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.expires_at <= now_ts) {
                entry.value_ptr.state = .expired;
            }
        }
    }

    pub fn purgeExpired(self: *Self, now_ts: i64) void {
        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.expires_at <= now_ts) {
                const session_id = entry.key_ptr.*;
                if (self.sessions.fetchRemove(session_id)) |kv| {
                    self.freeRecord(kv.value);
                }
            }
        }
    }

    fn cloneRecord(self: *Self, record: types.SessionRecord) !types.SessionRecord {
        const prompt_tokens = if (record.last_prompt_tokens) |tokens|
            try self.cloneTokens(tokens)
        else
            null;
        errdefer if (prompt_tokens) |tokens| self.allocator.free(tokens);

        return .{
            .session_id = try self.allocator.dupe(u8, record.session_id),
            .model_id = try self.allocator.dupe(u8, record.model_id),
            .bound_worker = record.bound_worker,
            .next_round_id = record.next_round_id,
            .last_prompt_hash = record.last_prompt_hash,
            .last_prompt_tokens = prompt_tokens,
            .last_access_ts = record.last_access_ts,
            .expires_at = record.expires_at,
            .state = record.state,
        };
    }

    fn cloneTokens(self: *Self, tokens: []const i64) ![]i64 {
        const out = try self.allocator.alloc(i64, tokens.len);
        @memcpy(out, tokens);
        return out;
    }

    fn freeRecord(self: *Self, record: types.SessionRecord) void {
        self.allocator.free(record.session_id);
        self.allocator.free(record.model_id);
        if (record.last_prompt_tokens) |tokens| {
            self.allocator.free(tokens);
        }
    }
};
