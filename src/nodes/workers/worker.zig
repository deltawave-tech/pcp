/// Worker - Connects to Shepherd and performs distributed training
/// This is the worker node that connects to the coordinator and runs local training

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const tcp_stream = @import("../../network/tcp_stream.zig");
const message = @import("../../network/message.zig");
const binary_protocol = @import("../../network/capnp_zig_wrapper.zig");
const worker_backend = @import("../../backends/worker_backend.zig");
const backend_selection = @import("../../backends/selection.zig");
const dataset_mod = @import("../../data/dataset.zig");
const loader = @import("../../data/loader.zig");

const TcpClient = tcp_stream.TcpClient;
const TcpStreamManager = tcp_stream.TcpStreamManager;
const MessageEnvelope = message.MessageEnvelope;
const MessageType = message.MessageType;
const NodeId = message.NodeId;
const WorkerBackend = worker_backend.WorkerBackend;
const Dataset = dataset_mod.Dataset;


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

    // NEW: AdamW optimizer state buffers (M and V) and timestep
    m_states: ?[][]u8, // One buffer per parameter
    v_states: ?[][]u8, // One buffer per parameter
    timestep: f32,

    // NEW: Dataset for chunk-based data loading
    dataset: ?Dataset,
    current_chunk_id: ?usize,

    // Supervisor Pattern: ID of the supervisor managing this worker
    supervisor_id: ?i64,

    const Self = @This();

    pub fn init(allocator: Allocator, backend: WorkerBackend, supervisor_id: ?i64) !Self {
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
            .m_states = null,
            .v_states = null,
            .timestep = 1.0,
            .dataset = null,
            .current_chunk_id = null,
            .supervisor_id = supervisor_id,
        };
    }
    
    
    pub fn deinit(self: *Self) void {
        // Cleanup dataset if it exists
        if (self.dataset) |ds| {
            ds.deinit();
        }

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

        // Cleanup optimizer state buffers
        if (self.m_states) |states| {
            for (states) |s| self.allocator.free(s);
            self.allocator.free(states);
        }
        if (self.v_states) |states| {
            for (states) |s| self.allocator.free(s);
            self.allocator.free(states);
        }

        self.client.deinit();
        self.backend.deinit();
    }
    
    /// Connect to the Shepherd coordinator
    pub fn connect(self: *Self, master_host: []const u8, master_port: u16, target_arch: ?[]const u8) !void {
        self.state = .connecting;

        // Connect to the master
        try self.client.connect(master_host, master_port);
        std.log.info("Connected to Shepherd at {s}:{}", .{ master_host, master_port });

        // Get our backend type from the backend instance
        const my_backend = self.backend.getBackendType();

        // Create a JSON object for the payload
        var payload_map = std.json.ObjectMap.init(self.allocator);
        defer payload_map.deinit();
        try payload_map.put("backend", std.json.Value{ .string = my_backend.toString() });

        // Add target architecture if specified
        if (target_arch) |target| {
            try payload_map.put("target_arch", std.json.Value{ .string = target });
            std.log.info("Reporting target architecture: {s}", .{target});
        }

        // Add supervisor ID if this worker is managed by a supervisor
        if (self.supervisor_id) |sid| {
            try payload_map.put("supervisor_id", std.json.Value{ .integer = @intCast(sid) });
            std.log.info("Reporting supervisor ID: {}", .{sid});
        }

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

    /// Robust entry point with automatic reconnection on failure
    pub fn runRobust(self: *Self, host: []const u8, port: u16, target_arch: ?[]const u8) !void {
        var backoff_ms: u64 = 100;
        const max_backoff_ms: u64 = 5000; // Cap at 5 seconds

        while (true) {
            // Reset state on reconnect attempts
            self.state = .connecting;

            self.connect(host, port, target_arch) catch |err| {
                // Only log warning if backoff is significant to reduce noise during quick restarts
                if (backoff_ms > 500) {
                    std.log.warn("Worker failed to connect to Shepherd: {}. Retrying in {}ms...", .{ err, backoff_ms });
                }
                std.time.sleep(backoff_ms * std.time.ns_per_ms);
                backoff_ms = @min(backoff_ms * 2, max_backoff_ms);
                continue;
            };

            // Connected! Reset backoff
            backoff_ms = 100;

            self.run() catch |err| {
                std.log.err("Worker connection dropped: {}", .{err});
            };

            if (self.state == .shutting_down) {
                break;
            }

            // Important: Explicitly disconnect client to close socket fd
            self.client.disconnect();
            self.state = .disconnected;

            // Wait a moment before immediate reconnect to allow Shepherd to recover
            std.time.sleep(500 * std.time.ns_per_ms);
        }
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

        // 5. Allocate M and V state buffers (initialized to zero)
        // Free old buffers if they exist
        if (self.m_states) |states| {
            for (states) |s| self.allocator.free(s);
            self.allocator.free(states);
        }
        if (self.v_states) |states| {
            for (states) |s| self.allocator.free(s);
            self.allocator.free(states);
        }

        const num_params = self.cached_parameter_shapes.?.len;
        const m_buffers = try self.allocator.alloc([]u8, num_params);
        errdefer {
            for (m_buffers) |buf| self.allocator.free(buf);
            self.allocator.free(m_buffers);
        }
        const v_buffers = try self.allocator.alloc([]u8, num_params);
        errdefer {
            for (v_buffers) |buf| self.allocator.free(buf);
            self.allocator.free(v_buffers);
        }

        // Allocate and zero-initialize each state buffer
        for (self.cached_parameter_shapes.?, 0..) |shape, i| {
            var size: usize = @sizeOf(f32); // f32 for all parameters
            for (shape) |dim| {
                size *= @intCast(dim);
            }

            m_buffers[i] = try self.allocator.alloc(u8, size);
            @memset(m_buffers[i], 0);

            v_buffers[i] = try self.allocator.alloc(u8, size);
            @memset(v_buffers[i], 0);
        }

        self.m_states = m_buffers;
        self.v_states = v_buffers;
        self.timestep = 1.0; // Reset timestep

        std.log.info("Worker {} cached VMFB, {} parameter shapes, {} data input shapes, and allocated optimizer state buffers.", .{ self.node_id.?, self.cached_parameter_shapes.?.len, self.cached_data_input_shapes.?.len });
    }
    
    /// Handle StartInnerLoop command from Shepherd using Cap'n Proto
    /// Implements the tau-step inner loop with AdamW state management and chunk-based data loading
    fn handleStartInnerLoop(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;

        // 1. Ensure the VMFB, shapes, and optimizer states have been cached.
        const vmfb = self.cached_vmfb orelse return error.GraphNotInitialized;
        const param_shapes = self.cached_parameter_shapes orelse return error.ParameterShapesNotInitialized;
        const data_shapes = self.cached_data_input_shapes orelse return error.DataShapesNotInitialized;
        const m_states = self.m_states orelse return error.OptimizerStateNotInitialized;
        const v_states = self.v_states orelse return error.OptimizerStateNotInitialized;

        // 2. Extract payload as JSON object
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };

        // 3. Extract chunk metadata
        const offset = switch (payload.get("offset") orelse return error.MissingOffset) {
            .integer => |i| i,
            else => return error.InvalidOffsetFormat,
        };
        const length = switch (payload.get("length") orelse return error.MissingLength) {
            .integer => |i| i,
            else => return error.InvalidLengthFormat,
        };
        const chunk_id = switch (payload.get("chunk_id") orelse return error.MissingChunkId) {
            .integer => |i| i,
            else => return error.InvalidChunkIdFormat,
        };
        const data_path = switch (payload.get("data_path") orelse return error.MissingDataPath) {
            .string => |s| s,
            else => return error.InvalidDataPathFormat,
        };
        const tau = switch (payload.get("tau") orelse return error.MissingTau) {
            .integer => |i| @as(usize, @intCast(i)),
            else => return error.InvalidTauFormat,
        };
        const tokenizer_type = if (payload.get("tokenizer")) |tok_val|
            switch (tok_val) {
                .string => |s| s,
                else => "char",
            }
        else
            "char";
        const sampling_type = if (payload.get("sampling")) |sampling_val|
            switch (sampling_val) {
                .string => |s| s,
                else => "random",
            }
        else
            "random";

        self.current_chunk_id = @intCast(chunk_id);

        std.log.info("Worker {}: Assigned chunk {} (offset={}, length={})", .{ self.node_id.?, chunk_id, offset, length });

        // 4. Initialize dataset for this chunk using Factory Pattern
        if (self.dataset) |ds| {
            ds.deinit();
        }

        if (std.mem.eql(u8, tokenizer_type, "byte")) {
            if (!std.mem.eql(u8, sampling_type, "random")) return error.UnsupportedSamplingMode;
            std.log.info("Worker {}: Using ByteTokenizer (Configured)", .{self.node_id.?});
            const byte_ds = try loader.ByteTextDataset.initChunk(
                self.allocator,
                data_path,
                @intCast(offset),
                @intCast(length),
                @intCast(self.node_id.?),
            );
            self.dataset = byte_ds.asDataset();
        } else if (std.mem.eql(u8, tokenizer_type, "u16")) {
            if (std.mem.eql(u8, sampling_type, "fifo")) {
                std.log.info("Worker {}: Using U16TokenDatasetFifo (Configured)", .{self.node_id.?});
                const u16_ds = try loader.U16TokenDatasetFifo.initChunk(
                    self.allocator,
                    data_path,
                    @intCast(offset),
                    @intCast(length),
                    @intCast(self.node_id.?),
                );
                self.dataset = u16_ds.asDataset();
            } else if (std.mem.eql(u8, sampling_type, "random")) {
                std.log.info("Worker {}: Using U16TokenDataset (Configured)", .{self.node_id.?});
                const u16_ds = try loader.U16TokenDataset.initChunk(
                    self.allocator,
                    data_path,
                    @intCast(offset),
                    @intCast(length),
                    @intCast(self.node_id.?),
                );
                self.dataset = u16_ds.asDataset();
            } else {
                return error.UnsupportedSamplingMode;
            }
        } else if (std.mem.eql(u8, tokenizer_type, "char")) {
            if (!std.mem.eql(u8, sampling_type, "random")) return error.UnsupportedSamplingMode;
            std.log.info("Worker {}: Using CharTokenizer (Configured)", .{self.node_id.?});
            const char_ds = try loader.TextDataset.initChunk(
                self.allocator,
                data_path,
                @intCast(offset),
                @intCast(length),
                @intCast(self.node_id.?),
            );
            self.dataset = char_ds.asDataset();
        } else {
            std.log.err("Unknown tokenizer type: {s}", .{tokenizer_type});
            return error.UnknownTokenizer;
        }

        // 5. Resolve params transport: inline base64 (small models) vs shared blob files (large models).
        const params_path = if (payload.get("params_path")) |val|
            switch (val) {
                .string => |s| s,
                else => return error.InvalidParamsFormat,
            }
        else
            null;

        const blob_dir = if (payload.get("blob_dir")) |val|
            switch (val) {
                .string => |s| s,
                else => return error.InvalidMessageFormat,
            }
        else
            null;

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        var initial_params_inline: ?[]const u8 = null;
        var params_path_for_delta: ?[]const u8 = null;

        if (params_path) |p| {
            params_path_for_delta = p;
        } else {
            const b64_params = switch (payload.get("params") orelse return error.MissingParams) {
                .string => |s| s,
                else => return error.InvalidParamsFormat,
            };

            const capnp_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_params);
            const capnp_bytes = try arena.allocator().alloc(u8, capnp_len);
            try std.base64.standard.Decoder.decode(capnp_bytes, b64_params);

            // 6. Deserialize params only (no batch data)
            const reader = try binary_protocol.WorkerPayload.Reader.init(capnp_bytes);
            defer reader.deinit();
            const initial_params_bytes_temp = try reader.getParams();
            initial_params_inline = initial_params_bytes_temp;

            // Validate received parameters (sample-based for large tensors).
            const f32_view = std.mem.bytesAsSlice(f32, initial_params_bytes_temp);
            const max_sample_floats: usize = 1 << 20; // 1,048,576 floats (~4MiB)
            const sample = @min(f32_view.len, max_sample_floats);
            var sum: f64 = 0.0;
            var non_zero_count: usize = 0;
            for (f32_view[0..sample]) |val| {
                if (val != 0.0) non_zero_count += 1;
                sum += @abs(val);
            }

            std.log.warn("🔍 PARAMETER INTEGRITY CHECK (sample):", .{});
            std.log.warn("   Bytes Received: {}", .{initial_params_bytes_temp.len});
            std.log.warn("   Sample Non-Zero Elements: {} / {}", .{non_zero_count, sample});
            std.log.warn("   Sample Abs Sum: {d:.4}", .{sum});

            if (non_zero_count == 0) {
                std.log.err("🚨 CRITICAL: Worker received ALL-ZERO parameters! Training will produce NaN.", .{});
                return error.ReceivedZeroParameters;
            }
        }

        if (params_path_for_delta != null and blob_dir == null) {
            std.log.err("Missing blob_dir for params_path transport", .{});
            return error.InvalidMessageFormat;
        }

        std.log.info("Starting inner loop: chunk={}", .{chunk_id});

        // 7. Persistent optimizer states
        // We do NOT reset m_states, v_states, or timestep.
        // They must persist to maintain local momentum and curvature information
        // across communication rounds (as per DiLoCo Sec 6.1).

        // Ensure timestep is initialized if it's the very first run
        if (self.timestep == 0.0) self.timestep = 1.0;

        // 8. Copy initial params into working buffers (we'll update these in the loop)
        const current_params = try self.allocator.alloc([]u8, param_shapes.len);
        defer {
            for (current_params) |buf| self.allocator.free(buf);
            self.allocator.free(current_params);
        }

        if (params_path_for_delta) |p| {
            const file = try std.fs.cwd().openFile(p, .{});
            defer file.close();
            const stat = try file.stat();
            if (stat.size > @as(u64, std.math.maxInt(usize))) return error.ParameterBufferTooSmall;
            const file_size: usize = @intCast(stat.size);

            if (payload.get("params_bytes")) |pb| {
                if (pb == .integer and pb.integer > 0) {
                    const advertised_u64: u64 = @intCast(pb.integer);
                    if (advertised_u64 <= @as(u64, std.math.maxInt(usize))) {
                        const advertised: usize = @intCast(advertised_u64);
                        if (advertised != file_size) {
                            std.log.warn("Params blob size mismatch: advertised {} bytes, file has {} bytes", .{ advertised, file_size });
                        }
                    }
                }
            }

            var reader = file.reader();
            var total_read: usize = 0;
            for (param_shapes, 0..) |shape, i| {
                var size: usize = @sizeOf(f32);
                for (shape) |dim| size *= @intCast(dim);
                current_params[i] = try self.allocator.alloc(u8, size);
                try reader.readNoEof(current_params[i]);
                total_read += size;
            }
            if (total_read != file_size) {
                std.log.warn("Params blob read {} bytes but file size is {}", .{ total_read, file_size });
            }
        } else {
            const initial_params_bytes = initial_params_inline orelse return error.MissingParams;
            var param_offset: usize = 0;
            for (param_shapes, 0..) |shape, i| {
                var size: usize = @sizeOf(f32);
                for (shape) |dim| size *= @intCast(dim);

                if (param_offset + size > initial_params_bytes.len) return error.ParameterBufferTooSmall;

                current_params[i] = try self.allocator.alloc(u8, size);
                @memcpy(current_params[i], initial_params_bytes[param_offset .. param_offset + size]);
                param_offset += size;
            }
        }

        // 9. Extract batch_size and block_size from data_shapes
        const batch_size: usize = @intCast(data_shapes[0][0]);
        const block_size: usize = @intCast(data_shapes[0][1]);

        // 10. Run the inner loop for tau steps
        var final_loss: f32 = 0.0;

        for (0..tau) |step| {
            std.log.info("Worker {} inner loop step {}/{}", .{ self.node_id.?, step + 1, tau });

            // A. Generate batch locally from dataset
            const batch = try self.dataset.?.getBatch(batch_size, block_size);
            defer batch.deinit();

            // Build input list: [Params (N), Ms (N), Vs (N), Timestep (1), Data (D)]
            var inputs_list = std.ArrayList([]const u8).init(self.allocator);
            defer inputs_list.deinit();

            // Add current parameters
            for (current_params) |p| try inputs_list.append(p);

            // Add M states
            for (m_states) |m| try inputs_list.append(m);

            // Add V states
            for (v_states) |v| try inputs_list.append(v);

            // Add timestep (as bytes of f32)
            const timestep_bytes = std.mem.asBytes(&self.timestep);
            try inputs_list.append(timestep_bytes);

            // Add data inputs (from locally generated batch)
            try inputs_list.append(batch.inputs);
            try inputs_list.append(batch.targets);

            const inputs = try inputs_list.toOwnedSlice();
            defer self.allocator.free(inputs);

            // Build shape list: [param_shapes (N), param_shapes (N), param_shapes (N), scalar, data_shapes (D)]
            var all_shapes = std.ArrayList([]const i64).init(self.allocator);
            defer all_shapes.deinit();

            // Params, M, V shapes (3 * N)
            for (0..3) |_| {
                for (param_shapes) |shape| try all_shapes.append(shape);
            }

            // Timestep shape (scalar)
            const scalar_shape = &[_]i64{};
            try all_shapes.append(scalar_shape);

            // Data shapes
            for (data_shapes) |shape| try all_shapes.append(shape);

            const shapes_slice = try all_shapes.toOwnedSlice();
            defer self.allocator.free(shapes_slice);

            // Execute the training step
            const outputs = try self.backend.executeTrainingStep(vmfb, inputs, shapes_slice);
            defer {
                for (outputs) |o| self.allocator.free(o);
                self.allocator.free(outputs);
            }

            // Parse outputs: [Params (N), Ms (N), Vs (N), Loss (1)]
            // Expected: 3*N + 1 outputs
            const expected_outputs = param_shapes.len * 3 + 1;
            if (outputs.len != expected_outputs) {
                std.log.err("Expected {} outputs, got {}", .{ expected_outputs, outputs.len });
                return error.UnexpectedOutputCount;
            }

            const num_params = param_shapes.len;

            // Update current_params with new_params from outputs[0..N]
            for (0..num_params) |i| {
                @memcpy(current_params[i], outputs[i]);
            }

            // Update m_states with new_m from outputs[N..2N]
            for (0..num_params) |i| {
                @memcpy(m_states[i], outputs[num_params + i]);
            }

            // Update v_states with new_v from outputs[2N..3N]
            for (0..num_params) |i| {
                @memcpy(v_states[i], outputs[2 * num_params + i]);
            }

            // Extract loss from outputs[3N]
            const loss_bytes = outputs[3 * num_params];
            if (loss_bytes.len >= @sizeOf(f32)) {
                final_loss = @as(f32, @bitCast(std.mem.readInt(u32, loss_bytes[0..4], .little)));
            }

            if (!std.math.isFinite(final_loss)) {
                std.log.err("Worker {} produced non-finite loss at inner step {}/{} (timestep={d})", .{
                    self.node_id.?,
                    step + 1,
                    tau,
                    self.timestep,
                });
                break;
            }

            // Increment timestep for next iteration
            self.timestep += 1.0;
        }

        std.log.info("Worker {} completed {} inner loop steps, final loss: {d:.4}", .{ self.node_id.?, tau, final_loss });

        // 11. Calculate Delta = Initial_Params - Final_Params and send it
        if (params_path_for_delta) |p| {
            const out_dir = blob_dir orelse return error.InvalidMessageFormat;
            std.fs.cwd().makePath(out_dir) catch {};

            var name_buf: [128]u8 = undefined;
            const filename = try std.fmt.bufPrint(&name_buf, "delta_worker{d}_chunk{d}.f32", .{ self.node_id.?, self.current_chunk_id.? });
            const delta_path = try std.fs.path.join(self.allocator, &[_][]const u8{ out_dir, filename });
            defer self.allocator.free(delta_path);

            const params_file = try std.fs.cwd().openFile(p, .{});
            defer params_file.close();
            var params_reader = params_file.reader();

            const delta_file = try std.fs.cwd().createFile(delta_path, .{ .truncate = true });
            defer delta_file.close();
            var delta_writer = delta_file.writer();

            const chunk_floats_max: usize = 1 << 20; // 1,048,576 f32 ~= 4MiB
            const scratch_f32 = try self.allocator.alloc(f32, chunk_floats_max);
            defer self.allocator.free(scratch_f32);
            const scratch_bytes = std.mem.sliceAsBytes(scratch_f32);

            var bytes_written: usize = 0;
            for (current_params) |final_block_bytes| {
                const final_f32 = std.mem.bytesAsSlice(f32, final_block_bytes);
                var offset_elems: usize = 0;
                while (offset_elems < final_f32.len) {
                    const remaining_elems = final_f32.len - offset_elems;
                    const chunk_elems = @min(chunk_floats_max, remaining_elems);
                    const chunk_bytes_len = chunk_elems * @sizeOf(f32);
                    const chunk_bytes = scratch_bytes[0..chunk_bytes_len];

                    try params_reader.readNoEof(chunk_bytes);

                    for (scratch_f32[0..chunk_elems], 0..) |*val, j| {
                        val.* = val.* - final_f32[offset_elems + j];
                    }

                    try delta_writer.writeAll(chunk_bytes);
                    bytes_written += chunk_bytes_len;
                    offset_elems += chunk_elems;
                }
            }

            delta_file.sync() catch {};

            var response_payload = std.json.ObjectMap.init(self.allocator);
            defer response_payload.deinit();
            try response_payload.put("delta_path", std.json.Value{ .string = delta_path });
            try response_payload.put("delta_bytes", std.json.Value{ .integer = @intCast(bytes_written) });
            try response_payload.put("loss", std.json.Value{ .float = @floatCast(final_loss) });
            try response_payload.put("chunk_id", std.json.Value{ .integer = @intCast(self.current_chunk_id.?) });

            const response = tcp_stream.createMessage(
                self.node_id.?,
                "worker",
                0,
                "shepherd",
                MessageType.INNER_LOOP_COMPLETE,
                msg.msg_id +% 1,
                std.json.Value{ .object = response_payload },
            );

            try self.client.send(response);
            self.state = .connected;
            return;
        }

        var delta_bytes = std.ArrayList(u8).init(self.allocator);
        defer delta_bytes.deinit();

        // Iterate over parameter blocks to compute delta
        var delta_offset: usize = 0;
        for (current_params) |final_block_bytes| {
            const param_block_size = final_block_bytes.len;
            const initial_params_bytes = initial_params_inline orelse return error.MissingParams;
            const initial_block_bytes = initial_params_bytes[delta_offset .. delta_offset + param_block_size];

            // Cast to f32 for arithmetic
            const initial_f32 = std.mem.bytesAsSlice(f32, initial_block_bytes);
            const final_f32 = std.mem.bytesAsSlice(f32, final_block_bytes);

            // Calculate delta for each element and append to delta_bytes
            const num_elements = param_block_size / @sizeOf(f32);
            for (0..num_elements) |j| {
                const delta_val = initial_f32[j] - final_f32[j];
                const delta_val_bytes = std.mem.asBytes(&delta_val);
                try delta_bytes.appendSlice(delta_val_bytes);
            }

            delta_offset += param_block_size;
        }

        // Send Delta instead of raw params
        const shepherd_payload = binary_protocol.ShepherdPayload{
            .updated_params = delta_bytes.items,
            .loss = final_loss,
        };
        const response_capnp_bytes = try shepherd_payload.serialize(self.allocator);
        defer self.allocator.free(response_capnp_bytes);

        // Base64 encode the params+loss
        const b64_len = std.base64.standard.Encoder.calcSize(response_capnp_bytes.len);
        const response_b64_params = try self.allocator.alloc(u8, b64_len);
        defer self.allocator.free(response_b64_params);

        const encoded_len = std.base64.standard.Encoder.encode(response_b64_params, response_capnp_bytes).len;

        // Create JSON response with params and chunk_id
        var response_payload = std.json.ObjectMap.init(self.allocator);
        defer response_payload.deinit();
        try response_payload.put("params", std.json.Value{ .string = response_b64_params[0..encoded_len] });
        try response_payload.put("chunk_id", std.json.Value{ .integer = @intCast(self.current_chunk_id.?) });

        // Send the response
        const response = tcp_stream.createMessage(
            self.node_id.?,
            "worker",
            0,
            "shepherd",
            MessageType.INNER_LOOP_COMPLETE,
            msg.msg_id +% 1,
            std.json.Value{ .object = response_payload },
        );

        try self.client.send(response);
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
    
    var worker = try Worker.init(allocator, mock_backend, null);
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
