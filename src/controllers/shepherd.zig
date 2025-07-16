/// Shepherd Controller - Coordinates distributed training across worker nodes
/// This is the main coordinator that manages the worker pool and orchestrates DiLoCo training

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const tcp_stream = @import("../network/tcp_stream.zig");
const message = @import("../network/message.zig");
const training_algorithm = @import("../algorithms/training_algorithm.zig");

const TcpServer = tcp_stream.TcpServer;
const TcpStreamManager = tcp_stream.TcpStreamManager;
const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
const NodeId = message.NodeId;

/// Represents a connected worker
pub const WorkerConnection = struct {
    node_id: NodeId,
    stream: net.Stream,
    last_heartbeat: i64, // timestamp
    
    const Self = @This();
    
    pub fn init(node_id: NodeId, stream: net.Stream) Self {
        return Self{
            .node_id = node_id,
            .stream = stream,
            .last_heartbeat = std.time.timestamp(),
        };
    }
    
    pub fn updateHeartbeat(self: *Self) void {
        self.last_heartbeat = std.time.timestamp();
    }
    
    pub fn isAlive(self: Self, timeout_seconds: i64) bool {
        const now = std.time.timestamp();
        return (now - self.last_heartbeat) < timeout_seconds;
    }
};

/// Shepherd coordinates distributed training across workers
pub const Shepherd = struct {
    allocator: Allocator,
    server: ?TcpServer,
    worker_pool: ArrayList(WorkerConnection),
    next_node_id: NodeId,
    is_running: bool,
    algorithm: ?*training_algorithm.TrainingAlgorithm, // Training algorithm interface
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .server = null,
            .worker_pool = ArrayList(WorkerConnection).init(allocator),
            .next_node_id = 1, // Start from 1, 0 is reserved for coordinator
            .is_running = false,
            .algorithm = null,
        };
    }
    
    pub fn deinit(self: *Self) void {
        if (self.server) |*server| {
            server.deinit();
        }
        
        // Close all worker connections
        for (self.worker_pool.items) |worker| {
            worker.stream.close();
        }
        self.worker_pool.deinit();
    }
    
    /// Start listening for worker connections
    pub fn listen(self: *Self, host: []const u8, port: u16) !void {
        self.server = try TcpServer.init(self.allocator, host, port);
        self.is_running = true;
        
        std.log.info("Shepherd listening on {s}:{} for worker connections", .{ host, port });
        
        // Start accepting connections
        while (self.is_running) {
            if (self.server) |*server| {
                const stream = server.accept() catch |err| {
                    std.log.err("Failed to accept connection: {}", .{err});
                    continue;
                };
                
                // Handle connection in a new thread
                const thread = std.Thread.spawn(.{}, handleWorkerConnection, .{ self, stream }) catch |err| {
                    std.log.err("Failed to spawn worker handler thread: {}", .{err});
                    stream.close();
                    continue;
                };
                thread.detach();
            }
        }
    }
    
    /// Handle a new worker connection
    fn handleWorkerConnection(self: *Self, stream: net.Stream) !void {
        defer stream.close();
        
        // Wait for JoinRequest from worker
        const join_msg = TcpStreamManager.receive(stream, self.allocator) catch |err| {
            std.log.err("Failed to receive join message: {}", .{err});
            return;
        };
        
        // Validate it's a JoinRequest
        if (!std.mem.eql(u8, join_msg.msg_type, MessageType.JOIN_REQUEST)) {
            std.log.err("Expected JoinRequest, got: {s}", .{join_msg.msg_type});
            return;
        }
        
        // Assign new NodeId
        const assigned_node_id = self.next_node_id;
        self.next_node_id += 1;
        
        // Add worker to pool
        const worker = WorkerConnection.init(assigned_node_id, stream);
        try self.worker_pool.append(worker);
        
        std.log.info("Worker {} connected from {}", .{ assigned_node_id, stream });
        
        // Send JoinAccept response
        const join_accept = tcp_stream.createMessage(
            0, // coordinator node_id
            "shepherd", // coordinator service
            assigned_node_id, // worker node_id
            "worker", // worker service
            MessageType.JOIN_ACCEPT,
            join_msg.msg_id + 1,
            std.json.Value{ .object = std.json.ObjectMap.init(self.allocator) }, // Empty object for now
        );
        
        TcpStreamManager.send(stream, join_accept, self.allocator) catch |err| {
            std.log.err("Failed to send join accept: {}", .{err});
            return;
        };
        
        std.log.info("Worker {} successfully joined the network", .{assigned_node_id});
        
        // Enter worker message handling loop
        try self.handleWorkerMessages(assigned_node_id, stream);
    }
    
    /// Handle ongoing messages from a worker
    fn handleWorkerMessages(self: *Self, worker_id: NodeId, stream: net.Stream) !void {
        while (self.is_running) {
            const msg = TcpStreamManager.receive(stream, self.allocator) catch |err| {
                std.log.warn("Worker {} disconnected: {}", .{ worker_id, err });
                self.removeWorker(worker_id);
                return;
            };
            
            // Handle different message types
            if (std.mem.eql(u8, msg.msg_type, MessageType.HEARTBEAT)) {
                self.handleHeartbeat(worker_id);
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.INNER_LOOP_COMPLETE)) {
                try self.handleInnerLoopComplete(worker_id, msg);
            } else {
                std.log.warn("Unknown message type from worker {}: {s}", .{ worker_id, msg.msg_type });
            }
        }
    }
    
    /// Handle heartbeat from worker
    fn handleHeartbeat(self: *Self, worker_id: NodeId) void {
        for (self.worker_pool.items) |*worker| {
            if (worker.node_id == worker_id) {
                worker.updateHeartbeat();
                return;
            }
        }
    }
    
    /// Handle inner loop completion from worker
    fn handleInnerLoopComplete(self: *Self, worker_id: NodeId, msg: MessageEnvelope) !void {
        std.log.info("Worker {} completed inner loop", .{worker_id});
        
        // TODO: Process the parameters from the message
        // This would involve deserializing the model parameters and 
        // adding them to the parameter aggregation
        _ = msg;
        
        // For now, just log the completion
        std.log.debug("Processing parameters from worker {}", .{worker_id});
    }
    
    /// Remove a worker from the pool
    fn removeWorker(self: *Self, worker_id: NodeId) void {
        for (self.worker_pool.items, 0..) |worker, i| {
            if (worker.node_id == worker_id) {
                _ = self.worker_pool.swapRemove(i);
                std.log.info("Removed worker {} from pool", .{worker_id});
                return;
            }
        }
    }
    
    /// Get current number of connected workers
    pub fn getWorkerCount(self: Self) usize {
        return self.worker_pool.items.len;
    }
    
    /// Set the training algorithm
    pub fn setAlgorithm(self: *Self, algorithm: *training_algorithm.TrainingAlgorithm) void {
        self.algorithm = algorithm;
    }
    
    /// Start training once we have enough workers
    pub fn startTraining(self: *Self, required_workers: usize) !void {
        std.log.info("Waiting for {} workers to join...", .{required_workers});
        
        // Wait for enough workers
        while (self.worker_pool.items.len < required_workers) {
            std.time.sleep(100 * std.time.ns_per_ms); // 100ms polling
        }
        
        std.log.info("Got {} workers, starting training...", .{self.worker_pool.items.len});
        
        if (self.algorithm) |algo| {
            try algo.run();
            std.log.info("Training algorithm completed successfully");
        } else {
            std.log.warn("No training algorithm set");
        }
    }
    
    /// Broadcast a message to all workers
    pub fn broadcastToWorkers(self: *Self, msg_type: []const u8, data: std.json.Value) !void {
        var msg_id: u8 = 1;
        
        for (self.worker_pool.items) |worker| {
            const msg = tcp_stream.createMessage(
                0, // coordinator node_id
                "shepherd", // coordinator service
                worker.node_id, // worker node_id
                "worker", // worker service
                msg_type,
                msg_id,
                data,
            );
            
            TcpStreamManager.send(worker.stream, msg, self.allocator) catch |err| {
                std.log.err("Failed to send message to worker {}: {}", .{ worker.node_id, err });
            };
            
            msg_id += 1;
        }
        
        std.log.info("Broadcasted {s} to {} workers", .{ msg_type, self.worker_pool.items.len });
    }
    
    /// Collect responses from all workers
    pub fn collectFromWorkers(self: *Self, expected_msg_type: []const u8) !ArrayList(MessageEnvelope) {
        var responses = ArrayList(MessageEnvelope).init(self.allocator);
        var received_count: usize = 0;
        
        // This is a simplified version - in practice, you'd need more sophisticated
        // handling with timeouts, error recovery, etc.
        while (received_count < self.worker_pool.items.len) {
            // In a real implementation, you'd use select/poll or async I/O
            // For now, this is a placeholder
            std.time.sleep(10 * std.time.ns_per_ms);
            received_count += 1; // Placeholder
        }
        
        std.log.info("Collected {} responses of type {s}", .{ responses.items.len, expected_msg_type });
        return responses;
    }
    
    /// Stop the coordinator
    pub fn stop(self: *Self) void {
        self.is_running = false;
        
        // Send shutdown message to all workers
        const shutdown_data = std.json.Value{ .string = "shutdown" };
        self.broadcastToWorkers(MessageType.SHUTDOWN, shutdown_data) catch |err| {
            std.log.err("Failed to broadcast shutdown: {}", .{err});
        };
        
        std.log.info("Shepherd stopped");
    }
};

/// Test function for the Shepherd
pub fn testShepherd(allocator: Allocator) !void {
    std.log.info("Testing Shepherd coordinator...");
    
    var shepherd = Shepherd.init(allocator);
    defer shepherd.deinit();
    
    // Test basic initialization
    try std.testing.expectEqual(@as(usize, 0), shepherd.getWorkerCount());
    try std.testing.expectEqual(@as(NodeId, 1), shepherd.next_node_id);
    
    std.log.info("✓ Shepherd test completed");
}