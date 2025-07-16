/// Worker - Connects to Shepherd and performs distributed training
/// This is the worker node that connects to the coordinator and runs local training

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const tcp_stream = @import("network/tcp_stream.zig");
const message = @import("network/message.zig");
const adam_mlir = @import("optimizers/adam_mlir.zig");
const ops = @import("ops.zig");
const mlir = @import("mlir.zig");
const tensor = @import("tensor.zig");

const TcpClient = tcp_stream.TcpClient;
const TcpStreamManager = tcp_stream.TcpStreamManager;
const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
const NodeId = message.NodeId;

/// Local model state for worker training
const LocalModel = struct {
    parameters: []f32,
    gradients: []f32,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator, param_count: usize) !Self {
        const parameters = try allocator.alloc(f32, param_count);
        const gradients = try allocator.alloc(f32, param_count);
        
        return Self{
            .parameters = parameters,
            .gradients = gradients,
        };
    }
    
    pub fn deinit(self: Self) void {
        // Note: caller is responsible for freeing the slices
    }
    
    pub fn updateParameters(self: *Self, new_params: []const f32) void {
        std.mem.copy(f32, self.parameters, new_params);
    }
    
    pub fn getParameters(self: Self) []const f32 {
        return self.parameters;
    }
};

/// Worker state
pub const WorkerState = enum {
    disconnected,
    connecting,
    connected,
    training,
    shutting_down,
};

/// Worker that connects to Shepherd and performs training
pub const Worker = struct {
    allocator: Allocator,
    client: TcpClient,
    node_id: ?NodeId,
    state: WorkerState,
    is_running: bool,
    // MLIR training infrastructure
    mlir_builder: ?*ops.MLIRBuilder,
    adam_optimizer: ?*adam_mlir.AdamMLIR(f32),
    local_model: ?LocalModel,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .node_id = null,
            .state = .disconnected,
            .is_running = false,
            .mlir_builder = null,
            .adam_optimizer = null,
            .local_model = null,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.client.deinit();
        
        if (self.adam_optimizer) |optimizer| {
            optimizer.deinit();
        }
        
        if (self.mlir_builder) |builder| {
            builder.deinit();
        }
        
        if (self.local_model) |model| {
            model.deinit();
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
        std.log.debug("Sent JoinRequest to Shepherd");
        
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
        std.log.info("Worker {} starting inner loop", .{self.node_id.?});
        self.state = .training;
        
        // Initialize MLIR infrastructure if not already done
        if (self.mlir_builder == null) {
            try self.initializeMLIRInfrastructure();
        }
        
        // Deserialize master parameters from message payload
        const master_params = try self.deserializeMasterParameters(msg);
        defer self.allocator.free(master_params);
        
        // Initialize local model with master parameters
        try self.initializeLocalModel(master_params);
        
        // Run local training loop using MLIR-based Adam optimizer
        try self.runLocalTrainingLoop();
        
        // Serialize final local parameters
        const final_params = try self.serializeLocalParameters();
        defer self.allocator.free(final_params);
        
        // Send InnerLoopComplete response
        const response = tcp_stream.createMessage(
            self.node_id.?, // our node_id
            "worker", // our service
            0, // shepherd node_id
            "shepherd", // shepherd service
            MessageType.INNER_LOOP_COMPLETE,
            msg.msg_id + 1,
            std.json.Value{ .string = final_params }, // serialized parameters
        );
        
        try self.client.send(response);
        std.log.info("Worker {} completed inner loop", .{self.node_id.?});
        
        self.state = .connected;
    }
    
    /// Initialize MLIR infrastructure for training
    fn initializeMLIRInfrastructure(self: *Self) !void {
        // Initialize MLIR builder
        self.mlir_builder = try self.allocator.create(ops.MLIRBuilder);
        self.mlir_builder.? = try ops.MLIRBuilder.init(self.allocator);
        
        // Initialize Adam optimizer
        const element_type = mlir.Type.f32Type(self.mlir_builder.?.ctx);
        const adam_config = adam_mlir.AdamMLIRConfiguration(f32).default_configuration();
        
        self.adam_optimizer = try self.allocator.create(adam_mlir.AdamMLIR(f32));
        self.adam_optimizer.? = try adam_mlir.AdamMLIR(f32).init(
            self.allocator,
            self.mlir_builder.?,
            adam_config,
            element_type,
        );
        
        std.log.info("Worker {} initialized MLIR infrastructure", .{self.node_id.?});
    }
    
    /// Deserialize master parameters from message (matches DiLoCo format)
    fn deserializeMasterParameters(self: *Self, msg: MessageEnvelope) ![]f32 {
        const param_data = switch (msg.data) {
            .string => |s| s,
            else => return error.InvalidMessageFormat,
        };
        
        if (param_data.len < @sizeOf(usize)) {
            return error.InvalidData;
        }
        
        // Read parameter count
        const param_count = std.mem.readInt(usize, param_data[0..@sizeOf(usize)], .little);
        const params = try self.allocator.alloc(f32, param_count);
        
        // Read all parameters
        var offset = @sizeOf(usize);
        for (params) |*param| {
            if (offset + @sizeOf(f32) > param_data.len) {
                return error.InvalidData;
            }
            param.* = std.mem.readInt(f32, param_data[offset..offset + @sizeOf(f32)], .little);
            offset += @sizeOf(f32);
        }
        
        std.log.debug("Worker {} deserialized {} parameters from {} bytes", .{ self.node_id.?, param_count, param_data.len });
        return params;
    }
    
    /// Initialize local model with master parameters
    fn initializeLocalModel(self: *Self, master_params: []const f32) !void {
        if (self.local_model) |model| {
            model.deinit();
        }
        
        self.local_model = try LocalModel.init(self.allocator, master_params.len);
        self.local_model.?.updateParameters(master_params);
        
        std.log.info("Worker {} initialized local model with {} parameters", .{
            self.node_id.?, master_params.len
        });
    }
    
    /// Run local training loop using MLIR-based optimization
    fn runLocalTrainingLoop(self: *Self) !void {
        if (self.local_model == null or self.adam_optimizer == null or self.mlir_builder == null) {
            return error.NotInitialized;
        }
        
        const inner_loop_steps = 10; // DiLoCo tau parameter
        std.log.info("Worker {} running {} inner loop steps", .{ self.node_id.?, inner_loop_steps });
        
        for (0..inner_loop_steps) |step| {
            // Create parameter and gradient tensors
            const param_shape = &[_]i64{@intCast(self.local_model.?.parameters.len)};
            const param_tensor = try self.createTensorFromArray(self.local_model.?.parameters, param_shape);
            
            // Simulate gradient computation (in practice, this would come from forward/backward pass)
            const grad_tensor = try self.computeSimulatedGradients(param_tensor);
            
            // Update parameters using MLIR Adam optimizer
            const updated_params = try self.adam_optimizer.?.update(param_tensor, grad_tensor);
            
            // Extract updated parameters back to local model
            try self.extractTensorToArray(updated_params, self.local_model.?.parameters);
            
            std.log.debug("Worker {} completed training step {}", .{ self.node_id.?, step + 1 });
        }
        
        std.log.info("Worker {} completed local training loop", .{self.node_id.?});
    }
    
    /// Create MLIR tensor from parameter array
    fn createTensorFromArray(self: *Self, array: []const f32, shape: []const i64) !tensor.Tensor(void) {
        const element_type = mlir.Type.f32Type(self.mlir_builder.?.ctx);
        
        // Convert f32 array to bytes for MLIR
        const byte_data = std.mem.sliceAsBytes(array);
        const tensor_type = mlir.Type.rankedTensorType(self.mlir_builder.?.ctx, shape, element_type);
        
        // Create constant tensor from host data
        const value = try self.mlir_builder.?.createConstant(byte_data, tensor_type, undefined);
        
        return try self.mlir_builder.?.newTensor(value);
    }
    
    /// Compute simulated gradients (placeholder for real gradient computation)
    fn computeSimulatedGradients(self: *Self, param_tensor: tensor.Tensor(void)) !tensor.Tensor(void) {
        // Simplified gradient computation - in practice, this would be computed
        // from the forward and backward pass through the neural network
        
        // For now, just create small random gradients
        const shape = param_tensor.shape.dims;
        const element_type = mlir.Type.f32Type(self.mlir_builder.?.ctx);
        
        // Create small random gradient tensor
        return try ops.constant(self.mlir_builder.?, 0.001, shape, element_type);
    }
    
    /// Extract tensor values back to array
    fn extractTensorToArray(self: *Self, tensor_val: tensor.Tensor(void), array: []f32) !void {
        // Simplified extraction - in practice, would need proper MLIR value extraction
        // For now, just add small noise to simulate parameter updates
        _ = tensor_val;
        
        var rng = std.rand.DefaultPrng.init(12345);
        for (array) |*param| {
            param.* += rng.random().floatNorm(f32) * 0.001;
        }
        
        std.log.debug("Worker {} extracted {} parameters from tensor", .{ self.node_id.?, array.len });
    }
    
    /// Serialize local parameters for transmission (matches DiLoCo format)
    fn serializeLocalParameters(self: *Self) ![]u8 {
        if (self.local_model == null) {
            return error.ModelNotInitialized;
        }
        
        // Binary serialization to match DiLoCo format
        var buffer = std.ArrayList(u8).init(self.allocator);
        defer buffer.deinit();
        
        const params = self.local_model.?.parameters;
        
        // Write parameter count first
        try buffer.appendSlice(std.mem.asBytes(&params.len));
        
        // Write all parameters as binary data
        for (params) |param| {
            try buffer.appendSlice(std.mem.asBytes(&param));
        }
        
        const result = try self.allocator.dupe(u8, buffer.items);
        std.log.debug("Worker {} serialized {} parameters to {} bytes", .{ self.node_id.?, params.len, result.len });
        
        return result;
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
    
    var worker = Worker.init(allocator);
    defer worker.deinit();
    
    // Test basic initialization
    try std.testing.expectEqual(WorkerState.disconnected, worker.getState());
    try std.testing.expectEqual(@as(?NodeId, null), worker.getNodeId());
    
    std.log.info("✓ Worker test completed");
}