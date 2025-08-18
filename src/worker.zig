/// Worker - Connects to Shepherd and performs distributed training
/// This is the worker node that connects to the coordinator and runs local training

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const tcp_stream = @import("network/tcp_stream.zig");
const message = @import("network/message.zig");
const worker_backend = @import("backends/worker_backend.zig");
const mlir = @import("mlir.zig");
const mlir_ctx = @import("mlir_ctx.zig");

const TcpClient = tcp_stream.TcpClient;
const TcpStreamManager = tcp_stream.TcpStreamManager;
const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
const NodeId = message.NodeId;
const WorkerBackend = worker_backend.WorkerBackend;


/// Worker state
pub const WorkerState = enum {
    disconnected,
    connecting,
    connected,
    training,
    shutting_down,
};

/// Worker that connects to Shepherd and performs training
/// Now backend-agnostic using the WorkerBackend interface
pub const Worker = struct {
    allocator: Allocator,
    client: TcpClient,
    node_id: ?NodeId,
    state: WorkerState,
    is_running: bool,
    
    // Backend abstraction - handles all MLIR compilation and execution
    backend: WorkerBackend,
    
    // MLIR context for deserializing modules
    mlir_context: ?*mlir_ctx.MLIRContext,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator, backend: WorkerBackend) Self {
        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .node_id = null,
            .state = .disconnected,
            .is_running = false,
            .backend = backend,
            .mlir_context = null,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.client.deinit();
        self.backend.deinit();
        
        if (self.mlir_context) |ctx| {
            ctx.deinit();
            self.allocator.destroy(ctx);
        }
    }
    
    /// Connect to the Shepherd coordinator
    pub fn connect(self: *Self, master_host: []const u8, master_port: u16) !void {
        self.state = .connecting;
        
        // Connect to the master
        try self.client.connect(master_host, master_port);
        std.log.info("Connected to Shepherd at {s}:{}", .{ master_host, master_port });
        
        // Send JoinRequest
        const join_request = tcp_stream.createMessage(
            0, // temporary node_id, will be assigned by shepherd
            "worker", // our service name
            0, // shepherd node_id
            "shepherd", // shepherd service
            MessageType.JOIN_REQUEST,
            1, // message id
            std.json.Value{ .object = std.json.ObjectMap.init(self.allocator) }, // Empty object
        );
        
        try self.client.send(join_request);
        std.log.debug("Sent JoinRequest to Shepherd", .{});
        
        // Wait for JoinAccept
        const join_accept = try self.client.receive();
        defer {
            // TODO: Proper cleanup of parsed JSON
        }
        
        if (!std.mem.eql(u8, join_accept.msg_type, MessageType.JOIN_ACCEPT)) {
            return error.UnexpectedMessage;
        }
        
        // Extract assigned node_id from response
        self.node_id = join_accept.recipient_node;
        self.state = .connected;
        self.is_running = true;
        
        std.log.info("Successfully joined network with NodeId {}", .{self.node_id.?});
    }
    
    /// Main worker loop - listen for commands from Shepherd
    pub fn run(self: *Self) !void {
        if (self.state != .connected) {
            return error.NotConnected;
        }
        
        std.log.info("Worker {} entering main loop", .{self.node_id.?});
        
        while (self.is_running) {
            // Receive command from Shepherd
            const msg = self.client.receive() catch |err| {
                std.log.err("Failed to receive message from Shepherd: {}", .{err});
                self.state = .disconnected;
                return;
            };
            
            // Handle different message types
            if (std.mem.eql(u8, msg.msg_type, MessageType.START_INNER_LOOP)) {
                try self.handleStartInnerLoop(msg);
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.SHUTDOWN)) {
                self.handleShutdown(msg);
                break;
            } else {
                std.log.warn("Unknown message type: {s}", .{msg.msg_type});
            }
        }
        
        std.log.info("Worker {} exiting main loop", .{self.node_id.?});
    }
    
    /// Handle StartInnerLoop command from Shepherd
    fn handleStartInnerLoop(self: *Self, msg: MessageEnvelope) !void {
        std.log.info("Worker {} received training graph.", .{self.node_id.?});
        self.state = .training;

        // Initialize MLIR context if needed
        if (self.mlir_context == null) {
            self.mlir_context = try self.allocator.create(mlir_ctx.MLIRContext);
            self.mlir_context.?.* = try mlir_ctx.MLIRContext.init(self.allocator);
        }

        // 1. Parse the delimiter-based payload (simpler than JSON)
        const payload_bytes = switch (msg.data) {
            .string => |s| s,
            else => return error.InvalidMessageFormat,
        };
        
        const delimiter = "|||PARAMS|||";
        const split_point = std.mem.indexOf(u8, payload_bytes, delimiter) orelse return error.InvalidPayload;

        const graph_str = payload_bytes[0..split_point];
        const params_bytes = payload_bytes[split_point + delimiter.len ..];

        // 2. Parse the MLIR module from the graph string
        const module = try mlir_ctx.deserializeMLIRModule(self.allocator, self.mlir_context.?.getContext(), graph_str);
        defer module.deinit();

        // 3. Generate RANDOM data for this test run on the worker
        // This simulates having a local dataset shard.
        const batch_size = 4;
        const seq_length = 12;
        const data_size_bytes = batch_size * seq_length * 4; // 4 bytes/f32

        const random_data = try self.allocator.alloc(u8, data_size_bytes);
        defer self.allocator.free(random_data);
        // In a real scenario, you'd fill this with actual token IDs.
        // For now, zeros are fine for testing the execution pipeline.
        @memset(random_data, 0);

        // 4. Package inputs for the backend: [master_params, input_ids, targets]
        // The order MUST match the function signature defined in the Shepherd!
        const inputs_array = [_][]const u8{
            params_bytes, // from shepherd
            random_data,  // input_ids
            random_data,  // targets (can be the same for this test)
        };
        const inputs: [][]const u8 = @constCast(&inputs_array);

        // 5. Execute the training step on the backend
        const outputs = try self.backend.executeTrainingStep(module, inputs);
        defer {
            for (outputs) |o| self.allocator.free(o);
            self.allocator.free(outputs);
        }

        // 6. Send BOTH outputs (updated_params, loss) back to the shepherd.
        // We'll use a simple delimiter again.
        var response_payload = std.ArrayList(u8).init(self.allocator);
        defer response_payload.deinit();

        if (outputs.len > 0) {
            try response_payload.appendSlice(outputs[0]); // Updated params
        }
        try response_payload.appendSlice("|||LOSS|||");
        if (outputs.len > 1) {
            try response_payload.appendSlice(outputs[1]); // Loss
        }

        const final_payload_bytes = try response_payload.toOwnedSlice();
        defer self.allocator.free(final_payload_bytes);

        const response = tcp_stream.createMessage(
            self.node_id.?, // our node_id
            "worker", // our service
            0, // shepherd node_id
            "shepherd", // shepherd service
            MessageType.INNER_LOOP_COMPLETE,
            msg.msg_id + 1,
            std.json.Value{ .string = final_payload_bytes },
        );
        
        try self.client.send(response);
        std.log.info("Worker {} completed inner loop and sent results.", .{self.node_id.?});
        self.state = .connected;
    }
    
    
    
    
    
    
    
    
    
    /// Handle Shutdown command from Shepherd
    fn handleShutdown(self: *Self, msg: MessageEnvelope) void {
        _ = msg;
        std.log.info("Worker {} received shutdown command", .{self.node_id.?});
        self.state = .shutting_down;
        self.is_running = false;
    }
    
    /// Send periodic heartbeat to Shepherd
    pub fn sendHeartbeat(self: *Self) !void {
        if (self.state != .connected) return;
        
        const heartbeat = tcp_stream.createMessage(
            self.node_id.?, // our node_id
            "worker", // our service
            0, // shepherd node_id
            "shepherd", // shepherd service
            MessageType.HEARTBEAT,
            0, // heartbeat message id
            std.json.Value{ .string = "alive" },
        );
        
        self.client.send(heartbeat) catch |err| {
            std.log.err("Failed to send heartbeat: {}", .{err});
        };
    }
    
    /// Get current worker state
    pub fn getState(self: Self) WorkerState {
        return self.state;
    }
    
    /// Get assigned node ID
    pub fn getNodeId(self: Self) ?NodeId {
        return self.node_id;
    }
    
    /// Disconnect from Shepherd
    pub fn disconnect(self: *Self) void {
        self.is_running = false;
        self.state = .disconnected;
        std.log.info("Worker {} disconnected", .{self.node_id orelse 0});
    }
};

/// Test function for Worker
pub fn testWorker(allocator: Allocator) !void {
    std.log.info("Testing Worker...");
    
    // Create a mock backend for testing
    const mock_backend = WorkerBackend{
        .ptr = undefined,
        .vtable = &.{
            .executeTrainingStep = mockExecuteTrainingStep,
            .deinit = mockBackendDeinit,
        },
    };
    
    var worker = Worker.init(allocator, mock_backend);
    defer worker.deinit();
    
    // Test basic initialization
    try std.testing.expectEqual(WorkerState.disconnected, worker.getState());
    try std.testing.expectEqual(@as(?NodeId, null), worker.getNodeId());
    
    std.log.info("✓ Worker test completed");
}

// Mock functions for testing
fn mockExecuteTrainingStep(ptr: *anyopaque, mlir_module: mlir.Module, inputs: [][]const u8) ![][]u8 {
    _ = ptr;
    _ = mlir_module;
    _ = inputs;
    return &[_][]u8{};
}

fn mockBackendDeinit(ptr: *anyopaque) void {
    _ = ptr;
}