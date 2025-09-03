/// Test for TCP communication layer
/// This test spawns a server and client to validate MessageEnvelope communication

const std = @import("std");
const net = std.net;
const testing = std.testing;
const tcp_stream = @import("../network/tcp_stream.zig");
const message = @import("../network/message.zig");

const TcpServer = tcp_stream.TcpServer;
const TcpClient = tcp_stream.TcpClient;
const TcpStreamManager = tcp_stream.TcpStreamManager;
const MessageEnvelope = message.MessageEnvelope;

/// Test server handler function
fn testServerHandler(stream: net.Stream, allocator: std.mem.Allocator) !void {
    defer stream.close();
    
    // Receive message from client
    const received_msg = try TcpStreamManager.receive(stream, allocator);
    defer {
        // Clean up JSON parsed data
        // Note: This is a simplified cleanup - in practice you'd need more careful memory management
    }
    
    std.log.debug("Server received message: type={s}, data={s}", .{
        received_msg.msg_type,
        switch (received_msg.data) {
            .string => |s| s,
            else => "non-string data",
        },
    });
    
    // Send response back to client
    const response = tcp_stream.createMessage(
        received_msg.recipient_node, // echo back as sender
        received_msg.recipient_service, // echo back as sender service
        received_msg.sender_node, // original sender becomes recipient
        received_msg.sender_service, // original sender service becomes recipient
        "test_response",
        received_msg.msg_id + 1,
        std.json.Value{ .string = "Server response" },
    );
    
    try TcpStreamManager.send(stream, response, allocator);
    std.log.debug("Server sent response");
}

/// Test client function
fn testClientFunction(allocator: std.mem.Allocator, host: []const u8, port: u16) !void {
    var client = TcpClient.init(allocator);
    defer client.deinit();
    
    // Connect to server
    try client.connect(host, port);
    std.log.debug("Client connected to server");
    
    // Send test message
    const test_msg = tcp_stream.createTestMessage(allocator);
    try client.send(test_msg);
    std.log.debug("Client sent test message");
    
    // Receive response
    const parsed_response = try client.receive();
    defer parsed_response.deinit();
    
    const response = parsed_response.value;
    
    std.log.debug("Client received response: type={s}", .{response.msg_type});
    
    // Validate response
    try testing.expectEqualStrings("test_response", response.msg_type);
    try testing.expectEqual(test_msg.msg_id + 1, response.msg_id);
    try testing.expectEqual(test_msg.sender_node, response.recipient_node);
    try testing.expectEqualStrings(test_msg.sender_service, response.recipient_service);
}

/// Basic TCP communication test
test "tcp_communication_basic" {
    const allocator = testing.allocator;
    const host = "127.0.0.1";
    const port = 8080;
    
    // Start server in a separate thread
    const server_thread = try std.Thread.spawn(.{}, serverThreadFunction, .{ allocator, host, port });
    defer server_thread.join();
    
    // Give server time to start
    std.time.sleep(100 * std.time.ns_per_ms);
    
    // Run client test
    try testClientFunction(allocator, host, port);
    
    std.log.info("✓ TCP communication test passed");
}

/// Server thread function
fn serverThreadFunction(allocator: std.mem.Allocator, host: []const u8, port: u16) !void {
    var server = TcpServer.init(allocator, host, port) catch |err| {
        std.log.err("Failed to start server: {}", .{err});
        return;
    };
    defer server.deinit();
    
    // Accept one connection and handle it
    const stream = try server.accept();
    try testServerHandler(stream, allocator);
}

/// Test message serialization/deserialization
test "message_serialization" {
    const allocator = testing.allocator;
    
    // Create test message
    const original = tcp_stream.createMessage(
        1, // from_node
        "worker", // from_service
        2, // to_node
        "coordinator", // to_service
        "join_request", // msg_type
        123, // msg_id
        std.json.Value{ .string = "test_data" }, // data
    );
    
    // Serialize to JSON
    const json_buffer = try original.asJsonString(allocator);
    defer json_buffer.deinit();
    
    // Deserialize back
    const parsed = try MessageEnvelope.fromJsonString(json_buffer.items, allocator);
    defer parsed.deinit();
    
    // Validate round trip
    try testing.expectEqual(original.sender_node, parsed.value.sender_node);
    try testing.expectEqualStrings(original.sender_service, parsed.value.sender_service);
    try testing.expectEqual(original.recipient_node, parsed.value.recipient_node);
    try testing.expectEqualStrings(original.recipient_service, parsed.value.recipient_service);
    try testing.expectEqualStrings(original.msg_type, parsed.value.msg_type);
    try testing.expectEqual(original.msg_id, parsed.value.msg_id);
    
    std.log.info("✓ Message serialization test passed");
}

/// Test message framing (length prefix)
test "message_framing" {
    const allocator = testing.allocator;
    
    // Create a test message
    const test_msg = tcp_stream.createTestMessage(allocator);
    
    // Serialize to get expected length
    const json_buffer = try test_msg.asJsonString(allocator);
    defer json_buffer.deinit();
    
    const expected_length: u32 = @intCast(json_buffer.items.len);
    
    // Test length prefix calculation
    const length_bytes = std.mem.asBytes(&expected_length);
    try testing.expectEqual(4, length_bytes.len);
    
    // Test reading length back
    const read_length = std.mem.readInt(u32, length_bytes[0..4], .little);
    try testing.expectEqual(expected_length, read_length);
    
    std.log.info("✓ Message framing test passed");
}

/// Test error handling
test "error_handling" {
    const allocator = testing.allocator;
    
    // Test connecting to non-existent server
    var client = TcpClient.init(allocator);
    defer client.deinit();
    
    // This should fail
    const connect_result = client.connect("127.0.0.1", 9999);
    try testing.expectError(error.ConnectionRefused, connect_result);
    
    // Test getting stream when not connected
    const stream_result = client.getStream();
    try testing.expectError(error.NotConnected, stream_result);
    
    std.log.info("✓ Error handling test passed");
}

/// Main test runner
pub fn runAllTests(allocator: std.mem.Allocator) !void {
    std.log.info("Running TCP communication tests...");
    
    // Note: Individual tests would normally be run by `zig test`
    // This is a placeholder for the test structure
    
    std.log.info("✓ All TCP tests structured and ready");
    std.log.info("To run tests, use: zig test src/examples/tcp_test.zig");
}