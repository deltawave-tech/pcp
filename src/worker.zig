/// Worker - Connects to Shepherd and performs distributed training
/// This is the worker node that connects to the coordinator and runs local training

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const tcp_stream = @import("network/tcp_stream.zig");
const message = @import("network/message.zig");
const binary_protocol = @import("network/capnp_zig_wrapper.zig");
const worker_backend = @import("backends/worker_backend.zig");
const backend_selection = @import("backend_selection.zig");

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

    // NEW: Cached VMFB binary and shapes from graph initialization
    cached_vmfb: ?[]u8,
    cached_parameter_shapes: ?[][]i64,
    cached_data_input_shapes: ?[][]i64,

    const Self = @This();
    
    pub fn init(allocator: Allocator, backend: WorkerBackend) !Self {
        return Self{
            .allocator = allocator,
            .client = TcpClient.init(allocator),
            .node_id = null,
            .state = .disconnected,
            .is_running = false,
            .backend = backend,
            .cached_vmfb = null,
            .cached_parameter_shapes = null,
            .cached_data_input_shapes = null,
        };
    }
    
    
    pub fn deinit(self: *Self) void {
        // Cleanup cached VMFB if it exists
        if (self.cached_vmfb) |vmfb| {
            self.allocator.free(vmfb);
        }
        if (self.cached_parameter_shapes) |shapes| {
            for (shapes) |s| self.allocator.free(s);
            self.allocator.free(shapes);
        }
        if (self.cached_data_input_shapes) |shapes| {
            for (shapes) |s| self.allocator.free(s);
            self.allocator.free(shapes);
        }

        self.client.deinit();
        self.backend.deinit();
    }
    
    /// Connect to the Shepherd coordinator
    pub fn connect(self: *Self, master_host: []const u8, master_port: u16) !void {
        self.state = .connecting;
        
        // Connect to the master
        try self.client.connect(master_host, master_port);
        std.log.info("Connected to Shepherd at {s}:{}", .{ master_host, master_port });
        
        // NEW: Determine our backend type at compile time
        const my_backend = backend_selection.Backend.selectDefault();

        // NEW: Create a JSON object for the payload
        var payload_map = std.json.ObjectMap.init(self.allocator);
        try payload_map.put("backend", std.json.Value{ .string = my_backend.toString() });
        const payload = std.json.Value{ .object = payload_map };

        // Send JoinRequest with the backend information in the payload
        const join_request = tcp_stream.createMessage(
            0, // temporary node_id, will be assigned by shepherd
            "worker", // our service name
            0, // shepherd node_id
            "shepherd", // shepherd service
            MessageType.JOIN_REQUEST,
            1, // message id
            payload, // Use the new payload
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
    
    /// Handles the one-time setup message, caching the VMFB binary and shapes
    fn handleInitializeGraph(self: *Self, msg: MessageEnvelope) !void {
        // Free any old data
        if (self.cached_vmfb) |vmfb| {
            self.allocator.free(vmfb);
            self.cached_vmfb = null;
        }
        if (self.cached_parameter_shapes) |shapes| {
            for (shapes) |s| self.allocator.free(s);
            self.cached_parameter_shapes = null;
        }
        if (self.cached_data_input_shapes) |shapes| {
            for (shapes) |s| self.allocator.free(s);
            self.cached_data_input_shapes = null;
        }

        // 1. Parse the JSON payload object
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };
        
        // 2. Decode and cache the VMFB
        const b64_vmfb_val = payload.get("vmfb") orelse return error.MissingVmfbField;
        const vmfb_string = switch (b64_vmfb_val) {
            .string => |s| s,
            else => return error.InvalidVmfbFormat,
        };
        
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(vmfb_string);
        const vmfb_bytes = try self.allocator.alloc(u8, decoded_len);
        try std.base64.standard.Decoder.decode(vmfb_bytes, vmfb_string);
        self.cached_vmfb = vmfb_bytes;

        // 3. Parse and cache the parameter shapes
        const param_shapes_json = payload.get("parameter_shapes") orelse return error.MissingParameterShapesField;
        const param_shapes_array = switch (param_shapes_json) {
            .array => |arr| arr,
            else => return error.InvalidParameterShapesFormat,
        };

        var param_shapes_list = std.ArrayList([]i64).init(self.allocator);
        errdefer {
            for (param_shapes_list.items) |s| self.allocator.free(s);
            param_shapes_list.deinit();
        }

        for (param_shapes_array.items) |shape_val| {
            const dim_array = switch (shape_val) {
                .array => |arr| arr,
                else => return error.InvalidParameterShapeFormat,
            };
            var dim_list = std.ArrayList(i64).init(self.allocator);
            for (dim_array.items) |dim_val| {
                const dim = switch (dim_val) {
                    .integer => |i| i,
                    else => return error.InvalidParameterDimensionFormat,
                };
                try dim_list.append(dim);
            }
            try param_shapes_list.append(try dim_list.toOwnedSlice());
        }
        self.cached_parameter_shapes = try param_shapes_list.toOwnedSlice();

        // 4. Parse and cache the data input shapes
        const shapes_json = payload.get("data_input_shapes") orelse return error.MissingShapesField;
        const shapes_array = switch (shapes_json) {
            .array => |arr| arr,
            else => return error.InvalidShapesFormat,
        };
        
        var shapes_list = std.ArrayList([]i64).init(self.allocator);
        errdefer {
            for (shapes_list.items) |s| self.allocator.free(s);
            shapes_list.deinit();
        }
        
        for (shapes_array.items) |shape_val| {
            const dim_array = switch (shape_val) {
                .array => |arr| arr,
                else => return error.InvalidShapeFormat,
            };
            var dim_list = std.ArrayList(i64).init(self.allocator);
            for (dim_array.items) |dim_val| {
                const dim = switch (dim_val) {
                    .integer => |i| i,
                    else => return error.InvalidDimensionFormat,
                };
                try dim_list.append(dim);
            }
            try shapes_list.append(try dim_list.toOwnedSlice());
        }
        self.cached_data_input_shapes = try shapes_list.toOwnedSlice();

        std.log.info("Worker {} cached VMFB, {} parameter shapes, and {} data input shapes.", .{ self.node_id.?, self.cached_parameter_shapes.?.len, self.cached_data_input_shapes.?.len });
    }
    
    /// Handle StartInnerLoop command from Shepherd using Cap'n Proto
    fn handleStartInnerLoop(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;

        // 1. Ensure the VMFB and shapes have been cached.
        const vmfb = self.cached_vmfb orelse {
            std.log.err("Received StartInnerLoop before graph was initialized.", .{});
            return error.GraphNotInitialized;
        };
        const param_shapes = self.cached_parameter_shapes orelse {
            std.log.err("Received StartInnerLoop before parameter shapes were initialized.", .{});
            return error.ParameterShapesNotInitialized;
        };
        const data_shapes = self.cached_data_input_shapes orelse {
            std.log.err("Received StartInnerLoop before data shapes were initialized.", .{});
            return error.DataShapesNotInitialized;
        };

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

        // 6. Split the concatenated params_bytes into individual parameter tensors
        var param_tensors = try self.allocator.alloc([]const u8, param_shapes.len);
        defer self.allocator.free(param_tensors);

        var offset: usize = 0;
        for (param_shapes, 0..) |shape, i| {
            // Calculate size of this parameter tensor
            var size: usize = @sizeOf(f32); // All params are f32
            for (shape) |dim| {
                size *= @intCast(dim);
            }

            // Extract slice from concatenated buffer
            if (offset + size > params_bytes.len) {
                std.log.err("Parameter buffer too small: expected {} bytes, got {}", .{ offset + size, params_bytes.len });
                return error.ParameterBufferTooSmall;
            }
            param_tensors[i] = params_bytes[offset .. offset + size];
            offset += size;
        }

        // 7. Package ALL inputs for the backend: [6 params + input_ids + targets]
        var inputs_list = std.ArrayList([]const u8).init(self.allocator);
        defer inputs_list.deinit();

        // Add all parameter tensors
        for (param_tensors) |pt| {
            try inputs_list.append(pt);
        }
        // Add data inputs
        try inputs_list.append(input_ids_bytes);
        try inputs_list.append(targets_bytes);

        const inputs = try inputs_list.toOwnedSlice();
        defer self.allocator.free(inputs);

        // 8. Combine parameter shapes and data shapes for the backend call
        var all_shapes = try self.allocator.alloc([]const i64, param_shapes.len + data_shapes.len);
        defer self.allocator.free(all_shapes);

        for (param_shapes, 0..) |shape, i| {
            all_shapes[i] = shape;
        }
        for (data_shapes, 0..) |shape, i| {
            all_shapes[param_shapes.len + i] = shape;
        }

        // 9. Execute using the backend
        const outputs = try self.backend.executeTrainingStep(vmfb, inputs, all_shapes);
        defer {
            for (outputs) |o| self.allocator.free(o);
            self.allocator.free(outputs);
        }

        // 10. Extract updated parameters and loss from the backend's output
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
fn mockExecuteTrainingStep(ptr: *anyopaque, compiled_artifact: []const u8, inputs_data: [][]const u8, input_shapes: [][]const i64) ![][]u8 {
    _ = ptr;
    _ = compiled_artifact;
    _ = inputs_data;
    _ = input_shapes;
    return &[_][]u8{};
}

fn mockBackendDeinit(ptr: *anyopaque) void {
    _ = ptr;
}