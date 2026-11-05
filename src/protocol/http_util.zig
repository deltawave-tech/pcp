const std = @import("std");

pub fn bearerToken(header: []const u8) ?[]const u8 {
    const scheme = "Bearer";
    if (header.len <= scheme.len) return null;
    if (!std.ascii.eqlIgnoreCase(header[0..scheme.len], scheme)) return null;
    if (header[scheme.len] != ' ') return null;
    const token = std.mem.trimLeft(u8, header[scheme.len + 1 ..], " ");
    if (token.len == 0) return null;
    return token;
}

pub fn parseHttpMethod(method: []const u8) !std.http.Method {
    if (std.mem.eql(u8, method, "GET")) return .GET;
    if (std.mem.eql(u8, method, "POST")) return .POST;
    if (std.mem.eql(u8, method, "PUT")) return .PUT;
    if (std.mem.eql(u8, method, "DELETE")) return .DELETE;
    return error.UnsupportedMethod;
}

pub fn statusText(status: std.http.Status) []const u8 {
    return switch (status) {
        .ok => "200 OK",
        .created => "201 Created",
        .accepted => "202 Accepted",
        .bad_request => "400 Bad Request",
        .unauthorized => "401 Unauthorized",
        .forbidden => "403 Forbidden",
        .not_found => "404 Not Found",
        .method_not_allowed => "405 Method Not Allowed",
        .conflict => "409 Conflict",
        .unprocessable_entity => "422 Unprocessable Entity",
        .too_many_requests => "429 Too Many Requests",
        .internal_server_error => "500 Internal Server Error",
        .bad_gateway => "502 Bad Gateway",
        .service_unavailable => "503 Service Unavailable",
        .gateway_timeout => "504 Gateway Timeout",
        else => "500 Internal Server Error",
    };
}

test "bearerToken parses only non-empty bearer tokens" {
    try std.testing.expectEqualStrings("dev", bearerToken("Bearer dev").?);
    try std.testing.expectEqualStrings("dev", bearerToken("bearer   dev").?);
    try std.testing.expect(bearerToken("Basic dev") == null);
    try std.testing.expect(bearerToken("Bearer") == null);
    try std.testing.expect(bearerToken("Bearer   ") == null);
}

test "parseHttpMethod supports gateway and hub proxy methods" {
    try std.testing.expectEqual(std.http.Method.GET, try parseHttpMethod("GET"));
    try std.testing.expectEqual(std.http.Method.POST, try parseHttpMethod("POST"));
    try std.testing.expectEqual(std.http.Method.PUT, try parseHttpMethod("PUT"));
    try std.testing.expectEqual(std.http.Method.DELETE, try parseHttpMethod("DELETE"));
    try std.testing.expectError(error.UnsupportedMethod, parseHttpMethod("PATCH"));
}

test "statusText returns HTTP status lines" {
    try std.testing.expectEqualStrings("200 OK", statusText(.ok));
    try std.testing.expectEqualStrings("503 Service Unavailable", statusText(.service_unavailable));
}
