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
        self.sessions.deinit();
    }

    pub fn upsert(self: *Self, record: types.SessionRecord) !void {
        try self.sessions.put(record.session_id, record);
    }

    pub fn get(self: *Self, session_id: []const u8) ?types.SessionRecord {
        return self.sessions.get(session_id);
    }

    pub fn remove(self: *Self, session_id: []const u8) void {
        _ = self.sessions.remove(session_id);
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
};
