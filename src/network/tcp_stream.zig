/// TCP Stream Manager for sending and receiving MessageEnvelopes over TCP
/// This implements proper message framing with length prefixes

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const message = @import("message.zig");
const MessageEnvelope = message.MessageEnvelope;

/// TCP Stream Manager handles reading and writing MessageEnvelopes over TCP streams
pub const TcpStreamManager = struct {
    const Self = @This();

    /// Send a MessageEnvelope over a TCP stream
    /// Uses JSON serialization with length prefix for proper framing
    pub fn send(stream: net.Stream, envelope: MessageEnvelope, allocator: Allocator) !void {
        // Serialize the envelope to JSON
        const json_buffer = try envelope.asJsonString(allocator);
        defer json_buffer.deinit();
        
        const json_data = json_buffer.items;
        const data_length: u32 = @intCast(json_data.len);
        
        // Send length prefix first (4 bytes for u32)
        const length_bytes = std.mem.asBytes(&data_length);
        try stream.writeAll(length_bytes);
        
        // Send the actual JSON data
        try stream.writeAll(json_data);
        
        std.log.debug("Sent message: type={s}, length={}, to_node={}, to_service={s}", .{
            envelope.msg_type,
            data_length,
            envelope.recipient_node,
            envelope.recipient_service,
        });
    }

    /// Receive a MessageEnvelope from a TCP stream
    /// Reads length prefix first, then exactly that many bytes
    pub fn receive(stream: net.Stream, allocator: Allocator) !MessageEnvelope {
        // Read the length prefix (4 bytes for u32)
        var length_bytes: [4]u8 = undefined;
        _ = try stream.readAll(&length_bytes);
        const data_length = std.mem.readInt(u32, &length_bytes, .little);
        
        // Sanity check - prevent excessive memory allocation
        if (data_length > 1024 * 1024) { // 1MB limit
            return error.MessageTooLarge;
        }
        
        // Read exactly data_length bytes
        const json_data = try allocator.alloc(u8, data_length);
        defer allocator.free(json_data);
        _ = try stream.readAll(json_data);
        
        // Parse JSON back to MessageEnvelope
        const parsed = try MessageEnvelope.fromJsonString(json_data, allocator);
        // Note: caller is responsible for calling parsed.deinit()
        
        std.log.debug("Received message: type={s}, length={}, from_node={}, from_service={s}", .{
            parsed.value.msg_type,
            data_length,
            parsed.value.sender_node,
            parsed.value.sender_service,
        });
        
        return parsed.value;
    }
    
    /// Receive a MessageEnvelope with a timeout
    pub fn receiveWithTimeout(stream: net.Stream, allocator: Allocator, timeout_ms: u64) !MessageEnvelope {
        // For now, just call receive - timeout implementation would require more complex handling
        _ = timeout_ms;
        return try receive(stream, allocator);
    }
};

/// TCP Server helper for accepting connections and handling them
pub const TcpServer = struct {
    listener: net.Server,
    allocator: Allocator,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator, host: []const u8, port: u16) !Self {
        const address = try net.Address.parseIp(host, port);
        const listener = try address.listen(.{ .reuse_address = true });
        
        std.log.info("TCP Server listening on {s}:{}", .{ host, port });
        
        return Self{
            .listener = listener,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.listener.deinit();
    }
    
    /// Accept a new connection and return the stream
    pub fn accept(self: *Self) !net.Stream {
        const connection = try self.listener.accept();
        std.log.debug("Accepted new connection from {}", .{connection.address});
        return connection.stream;
    }
    
    /// Accept connections and call handler for each one in a new thread
    pub fn acceptLoop(self: *Self, handler: *const fn (net.Stream, Allocator) anyerror!void) !void {
        while (true) {
            const stream = try self.accept();
            
            // Spawn a new thread to handle the connection
            const thread = try std.Thread.spawn(.{}, handler, .{ stream, self.allocator });
            thread.detach(); // Don't wait for thread to complete
        }
    }
};

/// TCP Client helper for connecting to servers
pub const TcpClient = struct {
    stream: ?net.Stream,
    allocator: Allocator,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .stream = null,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        if (self.stream) |stream| {
            stream.close();
        }
    }
    
    /// Connect to a TCP server
    pub fn connect(self: *Self, host: []const u8, port: u16) !void {
        const address = try net.Address.parseIp(host, port);
        self.stream = try net.tcpConnectToAddress(address);
        
        std.log.info("Connected to TCP server at {s}:{}", .{ host, port });
    }
    
    /// Get the underlying stream (must be connected first)
    pub fn getStream(self: Self) !net.Stream {
        return self.stream orelse error.NotConnected;
    }
    
    /// Send a message using the TcpStreamManager
    pub fn send(self: Self, envelope: MessageEnvelope) !void {
        const stream = try self.getStream();
        try TcpStreamManager.send(stream, envelope, self.allocator);
    }
    
    /// Receive a message using the TcpStreamManager
    pub fn receive(self: Self) !MessageEnvelope {
        const stream = try self.getStream();
        return try TcpStreamManager.receive(stream, self.allocator);
    }
};

/// Error types for TCP operations
pub const TcpError = error{
    MessageTooLarge,
    NotConnected,
    ConnectionClosed,
    InvalidMessage,
};

/// Helper function to create a MessageEnvelope
pub fn createMessage(
    from_node: message.NodeId,
    from_service: message.ServiceId,
    to_node: message.NodeId,
    to_service: message.ServiceId,
    msg_type: []const u8,
    msg_id: message.MessageId,
    data: std.json.Value,
) MessageEnvelope {
    return MessageEnvelope{
        .sender_node = from_node,
        .sender_service = from_service,
        .recipient_node = to_node,
        .recipient_service = to_service,
        .msg_type = msg_type,
        .msg_id = msg_id,
        .data = data,
    };
}

/// Test helper to create a test message
pub fn createTestMessage(allocator: Allocator) MessageEnvelope {
    _ = allocator;
    return createMessage(
        1, // from_node
        "test_service", // from_service
        2, // to_node
        "target_service", // to_service
        "test_message", // msg_type
        42, // msg_id
        std.json.Value{ .string = "Hello, World!" }, // data
    );
}