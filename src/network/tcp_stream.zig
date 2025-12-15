/// TCP Stream Manager for sending and receiving MessageEnvelopes over TCP
/// This implements proper message framing with length prefixes

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;

/// Return type for receive operations
pub const ReceiveResult = struct {
    parsed: std.json.Parsed(MessageEnvelope),
    buffer: []u8,
};
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
        
        std.log.debug("Worker is about to send JSON payload: {s}", .{json_data});
        
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
    pub fn receive(stream: net.Stream, allocator: Allocator) !ReceiveResult {
        // Read the length prefix (4 bytes for u32)
        var length_bytes: [4]u8 = undefined;
        const bytes_read = stream.readAll(&length_bytes) catch |err| {
            std.log.err("Failed to read length prefix from TCP stream: {}", .{err});
            return err;
        };
        std.log.debug("Successfully read {} bytes for length prefix", .{bytes_read});
        const data_length = std.mem.readInt(u32, &length_bytes, .little);
        
        // Debug: Log raw bytes and parsed length (only for invalid messages)
        if (data_length > 1024 * 1024 * 1024) {
            std.log.debug("Invalid length prefix bytes: {} {} {} {} -> parsed as {} bytes", .{
                length_bytes[0], length_bytes[1], length_bytes[2], length_bytes[3], data_length
            });
        }

        if (data_length > 1024 * 1024 * 1024) {
            std.log.err("Message exceeds size limit: {} MB (raw bytes: {} {} {} {})", .{
                data_length / (1024 * 1024), length_bytes[0], length_bytes[1], length_bytes[2], length_bytes[3]
            });
            return error.MessageTooLarge;
        }
        
        // Log large messages for monitoring
        if (data_length > 10 * 1024 * 1024) {
            std.log.info("Large message detected: {} MB", .{data_length / (1024 * 1024)});
        }
        
        // Read exactly data_length bytes
        const json_data = try allocator.alloc(u8, data_length);
        // DO NOT defer allocator.free(json_data) here; we are returning it.
        
        _ = try stream.readAll(json_data);
        
        const parsed = try std.json.parseFromSlice(MessageEnvelope, allocator, json_data, .{});
        
        std.log.debug("Received message: type={s}, length={}, from_node={}, from_service={s}", .{
            parsed.value.msg_type,
            data_length,
            parsed.value.sender_node,
            parsed.value.sender_service,
        });
        
        // Return both the parsed object and the buffer
        return .{ .parsed = parsed, .buffer = json_data };
    }
    
    /// Receive a MessageEnvelope with a timeout
    pub fn receiveWithTimeout(stream: net.Stream, allocator: Allocator, timeout_ms: u64) !std.json.Parsed(MessageEnvelope) {
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
    
    pub const Connection = struct {
        stream: net.Stream,
        address: net.Address,
    };

    /// Accept a new connection and return both the stream and address
    pub fn accept(self: *Self) !Connection {
        const connection = try self.listener.accept();
        std.log.debug("Accepted new connection from {}", .{connection.address});
        return .{
            .stream = connection.stream,
            .address = connection.address,
        };
    }
    
    /// Accept connections and call handler for each one in a new thread
    pub fn acceptLoop(self: *Self, handler: *const fn (net.Stream, Allocator) anyerror!void) !void {
        while (true) {
            const connection = try self.accept();

            // Spawn a new thread to handle the connection
            const thread = try std.Thread.spawn(.{}, handler, .{ connection.stream, self.allocator });
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

    /// Check if currently connected
    pub fn isConnected(self: Self) bool {
        return self.stream != null;
    }

    /// Disconnect and reset stream state, but keep allocator
    pub fn disconnect(self: *Self) void {
        if (self.stream) |s| {
            s.close();
            self.stream = null;
        }
    }

    /// Connect to a TCP server
    pub fn connect(self: *Self, host: []const u8, port: u16) !void {
        if (self.stream != null) self.disconnect();

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
    pub fn receive(self: Self) !ReceiveResult {
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