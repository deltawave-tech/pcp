/// Shepherd Controller - Coordinates distributed training across worker nodes
/// This is the main coordinator that manages the worker pool and orchestrates DiLoCo training

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const tcp_stream = @import("../../network/tcp_stream.zig");
const message = @import("../../network/message.zig");
const training_algorithm = @import("../../algorithms/training_algorithm.zig");
const execution = @import("../../execution.zig");
const monitoring = @import("../../ui/monitoring.zig");
const backend_selection = @import("../../backends/selection.zig");
const data_manager = @import("data_manager.zig");

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

/// Worker configuration identifier for compiled artifacts
pub const WorkerConfig = struct {
    backend: Backend,
    target_arch: ?[]const u8,

    pub fn hash(self: @This(), hasher: anytype) void {
        std.hash.autoHash(hasher, self.backend);
        if (self.target_arch) |target| {
            hasher.update(target);
        }
    }

    pub fn eql(self: @This(), other: @This()) bool {
        if (self.backend != other.backend) return false;
        if (self.target_arch == null and other.target_arch == null) return true;
        if (self.target_arch == null or other.target_arch == null) return false;
        return std.mem.eql(u8, self.target_arch.?, other.target_arch.?);
    }
};

/// Represents a connected worker
pub const WorkerConnection = struct {
    node_id: NodeId,
    stream: net.Stream,
    last_heartbeat: i64, // timestamp
    backend: Backend,
    status: WorkerStatus,
    address: net.Address,  // Worker's network address
    target_arch: ?[]const u8,  // GPU target architecture (e.g., gfx942 for MI300X, sm_80 for A100)

    const Self = @This();

    pub fn init(node_id: NodeId, stream: net.Stream, backend: Backend, address: net.Address, target_arch: ?[]const u8) Self {
        return Self{
            .node_id = node_id,
            .stream = stream,
            .last_heartbeat = std.time.timestamp(),
            .backend = backend,
            .status = .Connected,
            .address = address,
            .target_arch = target_arch,
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
    compiled_artifacts: std.HashMap(WorkerConfig, []const u8, std.hash_map.AutoContext(WorkerConfig), std.hash_map.default_max_load_percentage), // Cache: WorkerConfig -> VMFB bytes
    // Message queue for collecting worker results
    result_queue: ArrayList(MessageEnvelope),
    result_queue_mutex: std.Thread.Mutex,
    // Data manager for chunk-based strict partitioning
    data_manager: ?data_manager.DataManager,

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
            .compiled_artifacts = std.HashMap(WorkerConfig, []const u8, std.hash_map.AutoContext(WorkerConfig), std.hash_map.default_max_load_percentage).init(allocator),
            .result_queue = ArrayList(MessageEnvelope).init(allocator),
            .result_queue_mutex = std.Thread.Mutex{},
            .data_manager = null, // Initialized when training starts
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

        // Clean up result queue (free cloned messages)
        for (self.result_queue.items) |*msg| {
            msg.deinitClone(self.allocator);
        }
        self.result_queue.deinit();

        // Clean up executor if owned
        if (self.executor) |executor| {
            executor.deinit();
        }

        // Clean up data manager
        if (self.data_manager) |*dm| {
            dm.deinit();
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
                const connection = server.accept() catch |err| {
                    std.log.err("Failed to accept connection: {}", .{err});
                    continue;
                };

                // Handle connection in a new thread, passing both stream and address
                const thread = std.Thread.spawn(.{}, handleWorkerConnection, .{ self, connection.stream, connection.address }) catch |err| {
                    std.log.err("Failed to spawn worker handler thread: {}", .{err});
                    connection.stream.close();
                    continue;
                };
                thread.detach();
            }
        }
    }

    /// Handle a new worker connection
    fn handleWorkerConnection(self: *Self, stream: net.Stream, worker_address: net.Address) !void {
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

        // Parse target architecture if present
        const target_arch: ?[]const u8 = if (data_obj.get("target_arch")) |target_val|
            switch (target_val) {
                .string => |s| s,
                else => null,
            }
        else
            null;

        std.log.info("Worker connecting with backend: {s}", .{worker_backend.toString()});
        if (target_arch) |target| {
            std.log.info("  Target architecture: {s}", .{target});
        }

        // Assign new NodeId and add worker to pool (both protected by mutex)
        self.worker_pool_mutex.lock();
        const assigned_node_id = self.next_node_id;
        self.next_node_id += 1;

        const worker = WorkerConnection.init(assigned_node_id, stream, worker_backend, worker_address, target_arch);
        self.worker_pool.append(worker) catch |err| {
            self.worker_pool_mutex.unlock();
            std.log.err("Failed to append worker {}: {}", .{ assigned_node_id, err });
            return;
        };

        // Update monitoring while still holding mutex
        const worker_count = self.worker_pool.items.len;
        self.worker_pool_mutex.unlock();
        monitoring.setWorkerCount(worker_count);
        self.updateWorkerInfo();

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
                self.updateWorkerInfo();
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
    fn handleInnerLoopComplete(self: *Self, worker_id: NodeId, msg: MessageEnvelope) !void {
        std.log.info("Worker {} sent INNER_LOOP_COMPLETE result", .{worker_id});

        // Clone the message to own the data (msg contains pointers to temporary buffers)
        const msg_clone = try msg.clone(self.allocator);
        errdefer msg_clone.deinitClone(self.allocator);

        // Add the cloned message to the result queue for collection by the main thread
        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        try self.result_queue.append(msg_clone);
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
                self.updateWorkerInfoUnlocked();
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

    /// Update worker information in monitoring system (internal, assumes mutex is held)
    fn updateWorkerInfoUnlocked(self: *Self) void {
        // Create worker info array (static allocation is fine since we copy immediately)
        var worker_info_buf: [16]monitoring.WorkerInfo = undefined;
        var count: usize = 0;

        for (self.worker_pool.items) |worker| {
            if (count >= worker_info_buf.len) break;

            // Format IP address from stored address
            var ip_buf: [64]u8 = undefined;
            const ip_str = std.fmt.bufPrint(&ip_buf, "{}", .{worker.address}) catch "unknown";

            // Get backend and status strings
            const backend_str = worker.backend.toString();
            const status_str = switch (worker.status) {
                .Connected => "Connected",
                .GraphInitialized => "Initialized",
                .Training => "Training",
            };

            // Create WorkerInfo with proper string copying
            var info = monitoring.WorkerInfo{
                .node_id = worker.node_id,
                .backend = [_]u8{0} ** 16,
                .backend_len = @min(backend_str.len, 16),
                .ip_address = [_]u8{0} ** 64,
                .ip_len = @min(ip_str.len, 64),
                .status = [_]u8{0} ** 16,
                .status_len = @min(status_str.len, 16),
            };

            // Copy strings into fixed buffers
            @memcpy(info.backend[0..info.backend_len], backend_str[0..info.backend_len]);
            @memcpy(info.ip_address[0..info.ip_len], ip_str[0..info.ip_len]);
            @memcpy(info.status[0..info.status_len], status_str[0..info.status_len]);

            worker_info_buf[count] = info;
            count += 1;
        }

        // Update monitoring with worker info (monitoring will copy the data)
        monitoring.setWorkerInfo(worker_info_buf[0..count]);
    }

    /// Update worker information in monitoring system (public, acquires mutex)
    pub fn updateWorkerInfo(self: *Self) void {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        self.updateWorkerInfoUnlocked();
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

    /// Initialize the data manager for chunk-based data partitioning
    pub fn initDataManager(self: *Self, total_size: usize, chunk_size: usize) !void {
        if (self.data_manager != null) return; // Already initialized
        self.data_manager = try data_manager.DataManager.init(self.allocator, total_size, chunk_size);
        std.log.info("DataManager initialized: {} total size, {} chunk size", .{ total_size, chunk_size });
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

    /// Snapshot of currently connected workers for the current training round
    pub fn snapshotWorkers(self: *Self) !ArrayList(WorkerConnection) {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();

        var snapshot = ArrayList(WorkerConnection).init(self.allocator);
        for (self.worker_pool.items) |worker| {
            try snapshot.append(worker);
        }
        return snapshot;
    }

    /// Update worker status
    pub fn setWorkerStatus(self: *Self, node_id: NodeId, status: WorkerStatus) void {
        self.worker_pool_mutex.lock();
        defer self.worker_pool_mutex.unlock();
        for (self.worker_pool.items) |*w| {
            if (w.node_id == node_id) {
                w.status = status;
                return;
            }
        }
    }
    
    /// Collect responses from a specific number of workers
    pub fn collectFromWorkers(self: *Self, expected_msg_type: []const u8, expected_count: usize) !ArrayList(MessageEnvelope) {
        std.log.info("Collecting results (expecting {})...", .{expected_count});

        const max_wait_seconds: i64 = 60; // Reduced from 3600 to handle mid-round disconnections
        const start_time = std.time.timestamp();

        while (true) {
            self.result_queue_mutex.lock();
            const current_results = self.result_queue.items.len;
            self.result_queue_mutex.unlock();

            if (current_results >= expected_count) {
                break;
            }

            const elapsed = std.time.timestamp() - start_time;
            if (elapsed > max_wait_seconds) {
                if (current_results > 0) {
                    std.log.warn("Timeout waiting for all results. Proceeding with {}/{}", .{current_results, expected_count});
                    break;
                }
                return error.CollectionTimeout;
            }

            std.time.sleep(50 * std.time.ns_per_ms);
        }

        self.result_queue_mutex.lock();
        defer self.result_queue_mutex.unlock();

        var responses = ArrayList(MessageEnvelope).init(self.allocator);

        var i: usize = 0;
        while (i < self.result_queue.items.len) {
            const msg = self.result_queue.items[i];
            if (std.mem.eql(u8, msg.msg_type, expected_msg_type)) {
                try responses.append(msg);
                _ = self.result_queue.orderedRemove(i);
            } else {
                i += 1;
            }

            if (responses.items.len == expected_count) break;
        }

        std.log.info("Collected {} results from workers", .{responses.items.len});
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