/// Worker - Connects to Shepherd and performs distributed training
/// This is the worker node that connects to the coordinator and runs local training

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const tcp_stream = @import("network/tcp_stream.zig");
const message = @import("network/message.zig");
const binary_protocol = @import("network/capnp_zig_wrapper.zig");
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
    mlir_context: mlir_ctx.MLIRContext,
    
    // NEW: Cached MLIR module from graph initialization
    cached_module: ?mlir.Module,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator, backend: WorkerBackend) !Self {
        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .node_id = null,
            .state = .disconnected,
            .is_running = false,
            .backend = backend,
            .mlir_context = try mlir_ctx.MLIRContext.init(allocator),
            .cached_module = null,
        };
    }
    
    pub fn initWithBackend(allocator: Allocator, backend: WorkerBackend) !Self {
        // Get the MLIR context from the backend instead of creating a new one
        const backend_context = backend.getContext();
        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .node_id = null,
            .state = .disconnected,
            .is_running = false,
            .backend = backend,
            .mlir_context = mlir_ctx.MLIRContext.fromContext(backend_context),
            .cached_module = null,
        };
    }
    
    pub fn deinit(self: *Self) void {
        // Cleanup cached module if it exists
        if (self.cached_module) |mod| {
            mod.deinit();
            self.cached_module = null; // Prevent double-free
        }
        
        self.client.deinit();
        self.backend.deinit();
        
        self.mlir_context.deinit();
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
        
        // Wait for JoinAccept
        std.log.debug("Worker waiting for JoinAccept response from shepherd...", .{});
        const join_accept_result = self.client.receive() catch |err| {
            std.log.err("Failed to receive JoinAccept from shepherd: {}", .{err});
            return err;
        };
        defer join_accept_result.parsed.deinit();
        defer self.allocator.free(join_accept_result.buffer); // Free the buffer
        const join_accept = join_accept_result.parsed.value;
        
        if (!std.mem.eql(u8, join_accept.msg_type, MessageType.JOIN_ACCEPT)) {
            return error.UnexpectedMessage;
        }
        
        // Extract assigned node_id from response
        self.node_id = join_accept.recipient_node;
        self.state = .connected;
        self.is_running = true;
        
    }
    
    /// Main worker loop - listen for commands from Shepherd
    pub fn run(self: *Self) !void {
        if (self.state != .connected) {
            return error.NotConnected;
        }
        
        std.log.info("Worker {} entering main loop", .{self.node_id.?});
        
        while (self.is_running) {
            // Receive command from Shepherd
            const receive_result = self.client.receive() catch |err| {
                std.log.err("Failed to receive message from Shepherd: {}", .{err});
                self.state = .disconnected;
                return;
            };
            defer receive_result.parsed.deinit();
            defer self.allocator.free(receive_result.buffer); // Free the buffer
            const msg = receive_result.parsed.value;
            
            // Handle different message types
            if (std.mem.eql(u8, msg.msg_type, MessageType.INITIALIZE_GRAPH)) {
                try self.handleInitializeGraph(msg);
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.START_INNER_LOOP)) {
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
    
    /// Handles the one-time setup message, deserializing and caching the MLIR module
    fn handleInitializeGraph(self: *Self, msg: MessageEnvelope) !void {

        // De-initialize any old module
        if (self.cached_module) |mod| {
            mod.deinit();
            self.cached_module = null;
        }

        // Payload is a Base64 encoded graph string
        const b64_graph_str = switch (msg.data) {
            .string => |s| s,
            else => return error.InvalidMessageFormat,
        };

        // Base64-decode the payload to get the original MLIR data
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_graph_str);
        const graph_str = try self.allocator.alloc(u8, decoded_len);
        defer self.allocator.free(graph_str);
        try std.base64.standard.Decoder.decode(graph_str, b64_graph_str);

        // Deserialize and cache the module using the worker's own context
        const module = try mlir_ctx.deserializeMLIRModule(self.allocator, self.mlir_context.getContext(), graph_str);
        errdefer module.deinit(); // Cleanup on error to prevent leak
        
        // Validate deserialized module
        const test_op = module.op();
        if (@intFromPtr(test_op.handle) == 0) {
            std.log.err("Invalid deserialized module", .{});
            return error.InvalidDeserializedModule;
        }
        
        // Verify the module is well-formed
        const verification_result = test_op.verify();
        if (!verification_result) {
            std.log.err("Deserialized MLIR module failed verification", .{});
            std.debug.print("Dumping invalid module for debugging:\n", .{});
            test_op.dump();
            return error.ModuleVerificationFailed;
        }
        
        // Successfully validated, cache the module
        self.cached_module = module;

        std.log.info("Worker {} initialized training graph", .{self.node_id.?});
        // No response is needed. The worker is now ready for START_INNER_LOOP.
    }
    
    /// Handle StartInnerLoop command from Shepherd using Cap'n Proto
    fn handleStartInnerLoop(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;

        // 1. Ensure the graph has been initialized and cached.
        const module = self.cached_module orelse {
            std.log.err("Received StartInnerLoop before graph was initialized.", .{});
            return error.GraphNotInitialized;
        };
        
        // Validate cached module
        const verify_op = module.op();
        if (@intFromPtr(verify_op.handle) == 0) {
            std.log.err("Cached module corrupted", .{});
            return error.CorruptedCachedModule;
        }

        // 2. The payload is Base64-encoded Cap'n Proto data
        const b64_encoded_payload = switch (msg.data) {
            .string => |s| s,
            else => return error.InvalidMessageFormat,
        };

        // 3. Base64-decode the payload to get the binary Cap'n Proto data
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        
        const capnp_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_encoded_payload);
        const capnp_bytes = try arena.allocator().alloc(u8, capnp_len);
        try std.base64.standard.Decoder.decode(capnp_bytes, b64_encoded_payload);

        // 4. Deserialize the binary data using Cap'n Proto
        const reader = try binary_protocol.WorkerPayload.Reader.init(capnp_bytes);
        defer reader.deinit();
        const params_bytes = try reader.getParams();
        const input_ids_bytes = try reader.getInputIds();
        const targets_bytes = try reader.getTargets();

        // VALIDATION: Log deserialized sizes and validate data
        std.log.info("Received params: {} bytes, inputs: {} bytes, targets: {} bytes", .{
            params_bytes.len, input_ids_bytes.len, targets_bytes.len
        });
        
        if (params_bytes.len == 0) {
            std.log.err("Empty parameters received in handleStartInnerLoop", .{});
            return error.EmptyParameters;
        }
        if (input_ids_bytes.len == 0) {
            std.log.err("Empty input_ids received in handleStartInnerLoop", .{});
            return error.EmptyInputIds;
        }
        if (targets_bytes.len == 0) {
            std.log.err("Empty targets received in handleStartInnerLoop", .{});
            return error.EmptyTargets;
        }

        // 6. Package inputs for the backend: [master_params, input_ids, targets]
        var inputs_array = [_][]const u8{ params_bytes, input_ids_bytes, targets_bytes };
        const inputs: [][]const u8 = &inputs_array;

        // 7. Execute using the CACHED module.
        const outputs = try self.backend.executeTrainingStep(module, inputs);
        defer {
            for (outputs) |o| self.allocator.free(o);
            self.allocator.free(outputs);
        }

        // 8. Extract updated parameters and loss from the backend's output
        const updated_params = if (outputs.len > 0) outputs[0] else &[_]u8{};
        var loss: f32 = 0.0;
        if (outputs.len > 1 and outputs[1].len >= @sizeOf(f32)) {
            loss = @as(f32, @bitCast(std.mem.readInt(u32, outputs[1][0..4], .little)));
        }

        // 9. Create and serialize the Cap'n Proto ShepherdPayload
        const shepherd_payload = binary_protocol.ShepherdPayload{
            .updated_params = updated_params,
            .loss = loss,
        };
        const response_capnp_bytes = try shepherd_payload.serialize(self.allocator);
        defer self.allocator.free(response_capnp_bytes);

        // 10. Base64 encode the binary payload
        const b64_len = std.base64.standard.Encoder.calcSize(response_capnp_bytes.len);
        const response_b64_payload = try self.allocator.alloc(u8, b64_len);
        defer self.allocator.free(response_b64_payload);
        
        const encoded_len = std.base64.standard.Encoder.encode(response_b64_payload, response_capnp_bytes).len;

        // 11. Create and send the message
        const response = tcp_stream.createMessage(
            self.node_id.?,
            "worker",
            0,
            "shepherd",
            MessageType.INNER_LOOP_COMPLETE,
            msg.msg_id + 1,
            std.json.Value{ .string = response_b64_payload[0..encoded_len] },
        );
        
        try self.client.send(response);
        std.log.info("Worker {} completed inner loop", .{self.node_id.?});
        self.state = .connected;
    }
    
    
    
    
    
    
    
    
    
    /// Handle Shutdown command from Shepherd
    fn handleShutdown(self: *Self, msg: MessageEnvelope) void {
        _ = msg;
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
    
    var worker = try Worker.init(allocator, mock_backend);
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