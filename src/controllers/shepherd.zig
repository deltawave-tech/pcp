/// Shepherd Controller - Coordinates distributed training across worker nodes
/// This is the main coordinator that manages the worker pool and orchestrates DiLoCo training

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const tcp_stream = @import("../network/tcp_stream.zig");
const message = @import("../network/message.zig");
const training_algorithm = @import("../algorithms/training_algorithm.zig");
const execution = @import("../execution.zig");
const monitoring = @import("../monitoring.zig");
const backend_selection = @import("../backend_selection.zig");

const TcpServer = tcp_stream.TcpServer;
const TcpStreamManager = tcp_stream.TcpStreamManager;
const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
const NodeId = message.NodeId;
const Executor = execution.Executor;
const Backend = backend_selection.Backend;

/// Worker readiness status
pub const WorkerStatus = enum {
    Connected,        // Worker connected but not initialized
    GraphInitialized, // Worker has loaded VMFB and is ready for tensors
    Training,         // Worker is actively training
};

/// Represents a connected worker
pub const WorkerConnection = struct {
    node_id: NodeId,
    stream: net.Stream,
    last_heartbeat: i64, // timestamp
    backend: Backend,
    status: WorkerStatus,

    const Self = @This();

    pub fn init(node_id: NodeId, stream: net.Stream, backend: Backend) Self {
        return Self{
            .node_id = node_id,
            .stream = stream,
            .last_heartbeat = std.time.timestamp(),
            .backend = backend,
            .status = .Connected,
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
/// Now accepts an Executor to make algorithms backend-agnostic
pub const Shepherd = struct {
    allocator: Allocator,
    server: ?TcpServer,
    worker_pool: ArrayList(WorkerConnection),
    worker_pool_mutex: std.Thread.Mutex, // Protect worker_pool from concurrent access
    next_node_id: NodeId,
    is_running: bool,
    algorithm: ?*training_algorithm.TrainingAlgorithm, // Training algorithm interface
    executor: ?Executor, // Generic execution backend for algorithms
    compiled_artifacts: std.AutoHashMap(Backend, []const u8), // Cache: Backend -> VMFB bytes

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .server = null,
            .worker_pool = ArrayList(WorkerConnection).init(allocator),
            .worker_pool_mutex = std.Thread.Mutex{},
            .next_node_id = 1, // Start from 1, 0 is reserved for coordinator
            .is_running = false,
            .algorithm = null,
            .executor = null,
            .compiled_artifacts = std.AutoHashMap(Backend, []const u8).init(allocator),
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

        // Clean up compiled artifacts
        var artifact_iter = self.compiled_artifacts.valueIterator();
        while (artifact_iter.next()) |vmfb_bytes| {
            self.allocator.free(vmfb_bytes.*);
        }
        self.compiled_artifacts.deinit();

        // Clean up executor if owned
        if (self.executor) |executor| {
            executor.deinit();
        }
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
        const join_msg_result = TcpStreamManager.receive(stream, self.allocator) catch |err| {
            std.log.err("Failed to receive join message: {}", .{err});
            return;
        };
        defer join_msg_result.parsed.deinit();
        defer self.allocator.free(join_msg_result.buffer);
        
        const join_msg = join_msg_result.parsed.value;
        
        
        // Validate it's a JoinRequest
        if (!std.mem.eql(u8, join_msg.msg_type, MessageType.JOIN_REQUEST)) {
            std.log.err("Expected JoinRequest, got: {s}", .{join_msg.msg_type});
            return;
        }

        // NEW: Parse the backend from the join request's data payload
        const data_obj = join_msg.data.object;
        const backend_str = data_obj.get("backend").?.string;
        
        const worker_backend = if (std.mem.eql(u8, backend_str, "cuda")) backend_selection.Backend.cuda
            else if (std.mem.eql(u8, backend_str, "rocm")) backend_selection.Backend.rocm
            else if (std.mem.eql(u8, backend_str, "metal")) backend_selection.Backend.metal
            else .cpu; // Default to cpu

        std.log.info("Worker connecting with backend: {s}", .{worker_backend.toString()});

        // Assign new NodeId
        const assigned_node_id = self.next_node_id;
        self.next_node_id += 1;

        // Add worker to pool, now including its backend (protected by mutex)
        const worker = WorkerConnection.init(assigned_node_id, stream, worker_backend);
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        try self.worker_pool.append(worker);

        // Update monitoring (still holding mutex)
        monitoring.setWorkerCount(self.worker_pool.items.len);

        std.log.info("Worker {} connected", .{assigned_node_id});
        
        // Send JoinAccept response
        const empty_obj = std.json.Value{ .object = std.json.ObjectMap.init(self.allocator) };
        const join_accept = tcp_stream.createMessage(
            0, // coordinator node_id
            "shepherd", // coordinator service
            assigned_node_id, // worker node_id
            "worker", // worker service
            MessageType.JOIN_ACCEPT,
            join_msg.msg_id + 1,
            empty_obj,
        );
        
        const json_buffer = join_accept.asJsonString(self.allocator) catch |err| {
            std.log.err("Failed to serialize JOIN_ACCEPT: {}", .{err});
            return;
        };
        defer json_buffer.deinit();
        
        TcpStreamManager.send(stream, join_accept, self.allocator) catch |err| {
            std.log.err("Failed to send join accept: {}", .{err});
            return;
        };
        
        
        // Enter worker message handling loop
        try self.handleWorkerMessages(assigned_node_id, stream);
    }
    
    /// Handle ongoing messages from a worker
    fn handleWorkerMessages(self: *Self, worker_id: NodeId, stream: net.Stream) !void {
        while (self.is_running) {
            const msg_result = TcpStreamManager.receive(stream, self.allocator) catch |err| {
                std.log.warn("Worker {} disconnected: {}", .{ worker_id, err });
                self.removeWorker(worker_id);
                // Update monitoring after worker removal
                monitoring.setWorkerCount(self.worker_pool.items.len);
                return;
            };
            defer msg_result.parsed.deinit();
            defer self.allocator.free(msg_result.buffer);
            
            const msg = msg_result.parsed.value;
            
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
    fn handleInnerLoopComplete(_: *Self, _: NodeId, _: MessageEnvelope) !void {
        // TODO: Process the parameters from the message
        // This would involve deserializing the model parameters and 
        // adding them to the parameter aggregation
    }
    
    /// Remove a worker from the pool
    fn removeWorker(self: *Self, worker_id: NodeId) void {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        for (self.worker_pool.items, 0..) |worker, i| {
            if (worker.node_id == worker_id) {
                _ = self.worker_pool.swapRemove(i);
                std.log.info("Worker {} disconnected", .{worker_id});
                // Update monitoring after worker removal
                monitoring.setWorkerCount(self.worker_pool.items.len);
                return;
            }
        }
    }

    /// Get current number of connected workers
    pub fn getWorkerCount(self: *Self) usize {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        return self.worker_pool.items.len;
    }
    
    /// Set the training algorithm
    pub fn setAlgorithm(self: *Self, algorithm: *training_algorithm.TrainingAlgorithm) void {
        self.algorithm = algorithm;
    }
    
    /// Set the executor for algorithms to use
    pub fn setExecutor(self: *Self, executor: Executor) void {
        self.executor = executor;
    }
    
    /// Get the executor for algorithms to use
    pub fn getExecutor(self: *Self) ?Executor {
        return self.executor;
    }

    /// Get or compile VMFB artifact for a specific backend
    /// Caches compiled artifacts to avoid recompilation for workers with the same backend
    pub fn getArtifact(self: *Self, backend: Backend, mlir_source: []const u8) ![]const u8 {
        // Check if we have a cached artifact for this backend
        if (self.compiled_artifacts.get(backend)) |vmfb| {
            std.log.info("Using cached VMFB for backend: {s}", .{backend.toString()});
            return vmfb;
        }

        std.log.info("Compiling VMFB for backend: {s}...", .{backend.toString()});

        // Compile MLIR to VMFB for the target backend
        const target = backend.toIreeCompilationTarget();
        const mlir_ctx = @import("../mlir_ctx.zig");
        const vmfb = try mlir_ctx.MLIRContext.compileToVMFBStatic(
            self.allocator,
            mlir_source,
            target
        );

        // Cache the compiled artifact
        try self.compiled_artifacts.put(backend, vmfb);
        std.log.info("VMFB compiled and cached for backend: {s} ({} bytes)", .{backend.toString(), vmfb.len});

        return vmfb;
    }

    /// Start training once we have enough workers
    pub fn startTraining(self: *Self, required_workers: usize) !void {
        std.log.info("Waiting for {} workers to join...", .{required_workers});

        // Wait for enough workers (check under mutex)
        while (true) {
            self.worker_pool_mutex.lock();
            const current_count = self.worker_pool.items.len;
            self.worker_pool_mutex.unlock();

            if (current_count >= required_workers) break;
            std.time.sleep(100 * std.time.ns_per_ms); // 100ms polling
        }

        self.worker_pool_mutex.lock();
        const worker_count = self.worker_pool.items.len;
        self.worker_pool_mutex.unlock();
        std.log.info("Got {} workers, starting training...", .{worker_count});
        
        if (self.algorithm) |algo| {
            try algo.run();
            std.log.info("Training algorithm completed successfully", .{});
        } else {
            std.log.warn("No training algorithm set", .{});
        }
    }
    
    /// Broadcast a message to all workers
    pub fn broadcastToWorkers(self: *Self, msg_type: []const u8, data: std.json.Value) !void {
        var msg_id: u8 = 1;

        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

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

    }

    /// Send a message to a specific worker by its NodeId
    pub fn sendToWorker(self: *Self, node_id: NodeId, msg_type: []const u8, data: std.json.Value) !void {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        for (self.worker_pool.items) |worker| {
            if (worker.node_id == node_id) {
                const msg = tcp_stream.createMessage(
                    0, "shepherd",
                    worker.node_id, "worker",
                    msg_type,
                    1, // message id can be improved later
                    data,
                );

                TcpStreamManager.send(worker.stream, msg, self.allocator) catch |err| {
                    std.log.err("Failed to send message to worker {}: {}", .{ worker.node_id, err });
                };
                return;
            }
        }
        return error.WorkerNotFound;
    }
    
    /// Collect responses from all workers
    pub fn collectFromWorkers(self: *Self, _: []const u8) !ArrayList(MessageEnvelope) {
        const responses = ArrayList(MessageEnvelope).init(self.allocator);
        var received_count: usize = 0;

        // Get worker count under mutex
        self.worker_pool_mutex.lock();
        const worker_count = self.worker_pool.items.len;
        self.worker_pool_mutex.unlock();

        // This is a simplified version - in practice, you'd need more sophisticated
        // handling with timeouts, error recovery, etc.
        while (received_count < worker_count) {
            // In a real implementation, you'd use select/poll or async I/O
            // For now, this is a placeholder
            std.time.sleep(10 * std.time.ns_per_ms);
            received_count += 1; // Placeholder
        }
        
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
        
        std.log.info("Shepherd stopped", .{});
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