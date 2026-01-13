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
const math = @import("../../core/math.zig");
const tensor = @import("../../core/tensor.zig");
const IreeBackend = @import("../../backends/iree.zig").IreeBackend;

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
    cached_data_input_dtypes: ?[]tensor.DType,
    cached_param_dtype: tensor.DType, // dtype for parameters (f32, bf16, etc.)

    // NEW: AdamW optimizer state buffers (M and V) and timestep
    m_states: ?[][]u8, // One buffer per parameter
    v_states: ?[][]u8, // One buffer per parameter
    timestep: f32,

    // NEW: Dataset for chunk-based data loading
    dataset: ?Dataset,
    current_chunk_id: ?usize,

    // Supervisor Pattern: ID of the supervisor managing this worker
    supervisor_id: ?i64,

    // RL / Generation State
    kv_cache: ?[][]u8,        // Persistent KV Cache buffers on Host (legacy, to be sent to device)
    sampler: math.Sampler,    // Token sampler
    weight_blob: ?[]u8,       // Flat weight buffer for RL generation (updated by Shepherd)

    // Device Residency: Persistent GPU buffers to avoid PCIe transfers
    device_weights: ?[]IreeBackend.DeviceBuffer,  // Weights on GPU (updated once per UPDATE_WEIGHTS)
    device_kv_cache: ?[]IreeBackend.DeviceBuffer, // KV cache on GPU (pointer-swapped each step)
    device_weights_dirty: bool,                    // True if weight_blob was updated, needs re-upload

    // Chunked Transfer: State for reassembling incoming weight chunks
    incoming_weight_buffer: ?std.ArrayList(u8),
    expected_chunks: usize,
    received_chunks: usize,

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
            .cached_data_input_dtypes = null,
            .cached_param_dtype = .f32, // Default, will be updated from TRAIN message
            .m_states = null,
            .v_states = null,
            .timestep = 1.0,
            .dataset = null,
            .current_chunk_id = null,
            .supervisor_id = supervisor_id,
            .kv_cache = null,
            .sampler = math.Sampler.init(@intCast(std.time.timestamp())),
            .weight_blob = null,
            .device_weights = null,
            .device_kv_cache = null,
            .device_weights_dirty = false,
            .incoming_weight_buffer = null,
            .expected_chunks = 0,
            .received_chunks = 0,
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
        if (self.cached_data_input_dtypes) |dtypes| {
            self.allocator.free(dtypes);
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

        // Cleanup KV cache buffers
        if (self.kv_cache) |cache_buffers| {
            for (cache_buffers) |buf| self.allocator.free(buf);
            self.allocator.free(cache_buffers);
        }

        // Cleanup weight blob
        if (self.weight_blob) |blob| {
            self.allocator.free(blob);
        }

        // Cleanup incoming weight buffer (chunked transfer)
        if (self.incoming_weight_buffer) |*buf| {
            buf.deinit();
        }

        // Cleanup device-resident buffers
        if (self.device_weights) |weights| {
            for (weights) |buf| buf.release();
            self.allocator.free(weights);
        }
        if (self.device_kv_cache) |cache| {
            for (cache) |buf| buf.release();
            self.allocator.free(cache);
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
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.WEIGHT_CHUNK)) {
                try self.handleWeightChunk(msg);
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.START_INNER_LOOP)) {
                try self.handleStartInnerLoop(msg);
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.START_ROLLOUT)) {
                try self.handleStartRollout(msg);
            } else if (std.mem.eql(u8, msg.msg_type, MessageType.UPDATE_WEIGHTS)) {
                try self.handleUpdateWeights(msg);
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

    /// Handle incoming weight chunks from chunked transfer protocol
    fn handleWeightChunk(self: *Self, msg: MessageEnvelope) !void {
        const payload = switch (msg.data) {
            .object => |o| o,
            else => return error.InvalidFormat,
        };

        const chunk_index: usize = @intCast(payload.get("chunk_index").?.integer);
        const total_chunks: usize = @intCast(payload.get("total_chunks").?.integer);
        const b64_data = payload.get("data").?.string;

        // Initialize buffer on first chunk
        if (chunk_index == 0) {
            if (self.incoming_weight_buffer) |*buf| buf.deinit();
            const total_bytes: usize = @intCast(payload.get("total_bytes").?.integer);
            self.incoming_weight_buffer = try std.ArrayList(u8).initCapacity(self.allocator, total_bytes);
            self.expected_chunks = total_chunks;
            self.received_chunks = 0;
            std.log.info("Worker {}: Started receiving weights ({} bytes in {} chunks)", .{ self.node_id.?, total_bytes, total_chunks });
        }

        // Decode and append chunk
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_data);
        const decoded = try self.allocator.alloc(u8, decoded_len);
        defer self.allocator.free(decoded);
        try std.base64.standard.Decoder.decode(decoded, b64_data);

        try self.incoming_weight_buffer.?.appendSlice(decoded);
        self.received_chunks += 1;

        if (self.received_chunks % 10 == 0 or self.received_chunks == self.expected_chunks) {
            std.log.info("Worker {}: Received chunk {}/{}", .{ self.node_id.?, self.received_chunks, self.expected_chunks });
        }

        // Finalize when all chunks received
        if (self.received_chunks == self.expected_chunks) {
            std.log.info("Worker {}: Reassembly complete. {} bytes received.", .{ self.node_id.?, self.incoming_weight_buffer.?.items.len });

            // Move from ArrayList to weight_blob
            if (self.weight_blob) |blob| self.allocator.free(blob);
            self.weight_blob = try self.incoming_weight_buffer.?.toOwnedSlice();

            // Clean up builder
            self.incoming_weight_buffer = null;
            self.received_chunks = 0;
            self.expected_chunks = 0;

            // Mark for GPU upload on next use
            self.device_weights_dirty = true;
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
        if (self.cached_data_input_dtypes) |dtypes| {
            self.allocator.free(dtypes);
            self.cached_data_input_dtypes = null;
        }

        // 1. Parse the JSON payload object
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };

        // 2. Load VMFB (either from path or from base64-encoded bytes)
        if (payload.get("vmfb_path")) |vmfb_path_val| {
            // Load from filesystem
            const vmfb_path = switch (vmfb_path_val) {
                .string => |s| s,
                else => return error.InvalidVmfbPathFormat,
            };
            std.log.info("Loading VMFB from path: {s}", .{vmfb_path});
            const vmfb_bytes = try std.fs.cwd().readFileAlloc(
                self.allocator,
                vmfb_path,
                10 * 1024 * 1024 * 1024, // 10GB limit
            );
            self.cached_vmfb = vmfb_bytes;
            std.log.info("✓ Loaded VMFB from disk ({} bytes)", .{vmfb_bytes.len});
        } else if (payload.get("vmfb")) |b64_vmfb_val| {
            // Legacy: decode from base64
            const vmfb_string = switch (b64_vmfb_val) {
                .string => |s| s,
                else => return error.InvalidVmfbFormat,
            };
            const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(vmfb_string);
            const vmfb_bytes = try self.allocator.alloc(u8, decoded_len);
            try std.base64.standard.Decoder.decode(vmfb_bytes, vmfb_string);
            self.cached_vmfb = vmfb_bytes;
            std.log.info("✓ Decoded VMFB from base64 ({} bytes)", .{vmfb_bytes.len});
        } else {
            return error.MissingVmfbField;
        }

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

        // 4.5. Parse and cache the data input dtypes
        if (payload.get("data_input_dtypes")) |dtypes_json| {
            const dtypes_array = switch (dtypes_json) {
                .array => |arr| arr,
                else => return error.InvalidDTypesFormat,
            };

            var dtypes_list = std.ArrayList(tensor.DType).init(self.allocator);
            errdefer dtypes_list.deinit();

            for (dtypes_array.items) |val| {
                const s = switch (val) { .string => |str| str, else => "f32" };
                const dtype: tensor.DType = if (std.mem.eql(u8, s, "i64")) .i64
                    else if (std.mem.eql(u8, s, "i32")) .i32
                    else if (std.mem.eql(u8, s, "f32")) .f32
                    else if (std.mem.eql(u8, s, "f16")) .f16
                    else if (std.mem.eql(u8, s, "f64")) .f64
                    else if (std.mem.eql(u8, s, "bf16")) .bf16
                    else if (std.mem.eql(u8, s, "bool")) .bool
                    else .f32; // Default fallback
                try dtypes_list.append(dtype);
            }
            self.cached_data_input_dtypes = try dtypes_list.toOwnedSlice();
        }

        // 5. Load weights from local path if provided (for RL generation workers)
        var is_generation_only = false;
        if (payload.get("weights_path")) |weights_path_val| {
            const weights_path = switch (weights_path_val) {
                .string => |s| s,
                else => return error.InvalidWeightsPathFormat,
            };
            std.log.info("Loading initial weights from local path: {s}", .{weights_path});

            if (self.weight_blob) |old_blob| {
                self.allocator.free(old_blob);
            }

            const file = try std.fs.cwd().openFile(weights_path, .{ .mode = .read_only });
            defer file.close();

            const stat = try file.stat();
            const size = stat.size;

            self.weight_blob = try self.allocator.alloc(u8, size);
            errdefer {
                self.allocator.free(self.weight_blob.?);
                self.weight_blob = null;
            }

            const bytes_read = try file.readAll(self.weight_blob.?);
            if (bytes_read != size) {
                return error.IncompleteWeightsRead;
            }

            self.device_weights_dirty = true;
            is_generation_only = true;

            std.log.info("✓ Loaded initial weights locally ({} MB)", .{size / (1024 * 1024)});
        }

        // 6. Allocate M and V state buffers (skip for generation-only workers)
        if (!is_generation_only) {
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

            for (self.cached_parameter_shapes.?, 0..) |shape, i| {
                var size: usize = @sizeOf(f32);
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
            self.timestep = 1.0;
        }

        if (is_generation_only) {
            std.log.info("Worker {} cached VMFB, {} parameter shapes, {} data input shapes (generation-only).", .{ self.node_id.?, self.cached_parameter_shapes.?.len, self.cached_data_input_shapes.?.len });
        } else {
            std.log.info("Worker {} cached VMFB, {} parameter shapes, {} data input shapes, and allocated optimizer state buffers.", .{ self.node_id.?, self.cached_parameter_shapes.?.len, self.cached_data_input_shapes.?.len });
        }
    }

    /// Helper: Validates numerical stability of a buffer based on dtype
    /// Returns error if NaN or Inf is detected. Handles unaligned reads safely.
    fn validateBuffer(self: *Self, name: []const u8, bytes: []const u8, dtype: tensor.DType) !void {
        _ = self;
        var nan_count: usize = 0;
        var inf_count: usize = 0;
        var max_val: f64 = 0.0;
        var sum_abs: f64 = 0.0;
        const num_elements = bytes.len / dtype.sizeInBytes();

        // Only validate floating point types
        if (dtype != .f32 and dtype != .bf16 and dtype != .f16) return;

        if (dtype == .f32) {
            var i: usize = 0;
            while (i < bytes.len) : (i += 4) {
                if (i + 4 > bytes.len) break;
                // Safe unaligned read
                const u32_val = std.mem.readInt(u32, bytes[i..][0..4], .little);
                const val: f32 = @bitCast(u32_val);

                if (std.math.isNan(val)) nan_count += 1;
                if (std.math.isInf(val)) inf_count += 1;
                const abs_v = @abs(val);
                if (abs_v > max_val) max_val = abs_v;
                sum_abs += abs_v;
            }
        } else if (dtype == .bf16) {
            var i: usize = 0;
            while (i < bytes.len) : (i += 2) {
                if (i + 2 > bytes.len) break;
                // Safe unaligned read
                const u16_val = std.mem.readInt(u16, bytes[i..][0..2], .little);
                // bf16 to f32 conversion: shift left 16 bits
                const u32_val = @as(u32, u16_val) << 16;
                const val: f32 = @bitCast(u32_val);

                if (std.math.isNan(val)) nan_count += 1;
                if (std.math.isInf(val)) inf_count += 1;
                const abs_v = @abs(val);
                if (abs_v > max_val) max_val = abs_v;
                sum_abs += abs_v;
            }
        }

        if (nan_count > 0 or inf_count > 0) {
            std.log.err("VALIDATION FAILED for {s}: {} NaNs, {} Infs (Max: {d:.4}, Avg: {d:.4})",
                .{ name, nan_count, inf_count, max_val, sum_abs / @as(f64, @floatFromInt(num_elements)) });
            return error.NumericalInstabilityDetected;
        }
    }

    /// Handle StartInnerLoop command from Shepherd using Cap'n Proto
    /// Implements the tau-step inner loop with AdamW state management and chunk-based data loading
    fn handleStartInnerLoop(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;

        // 1. Ensure the VMFB, shapes, and optimizer states have been cached.
        const vmfb = self.cached_vmfb orelse return error.GraphNotInitialized;
        const param_shapes = self.cached_parameter_shapes orelse return error.ParameterShapesNotInitialized;
        const data_shapes = self.cached_data_input_shapes orelse return error.DataShapesNotInitialized;
        // Note: m_states and v_states are obtained AFTER reallocation code below to avoid use-after-free
        if (self.m_states == null) return error.OptimizerStateNotInitialized;
        if (self.v_states == null) return error.OptimizerStateNotInitialized;

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

        // Extract dtype for precision control (bf16 halves memory usage)
        const dtype_str = if (payload.get("dtype")) |dtype_val|
            switch (dtype_val) {
                .string => |s| s,
                else => "f32",
            }
        else
            "f32";

        const param_dtype: tensor.DType = if (std.mem.eql(u8, dtype_str, "bf16"))
            .bf16
        else if (std.mem.eql(u8, dtype_str, "f16"))
            .f16
        else
            .f32;

        std.log.info("Worker {}: Training with precision {s}", .{ self.node_id.?, dtype_str });

        // Cache the param dtype for size calculations
        self.cached_param_dtype = param_dtype;

        // Reallocate M/V states if dtype changed (they were allocated with f32 at INITIALIZE)
        // Check if first buffer size matches expected size for this dtype
        if (self.m_states) |existing_m_states| {
            if (self.cached_parameter_shapes) |shapes| {
                if (shapes.len > 0 and existing_m_states.len > 0) {
                    var expected_size: usize = param_dtype.sizeInBytes();
                    for (shapes[0]) |dim| expected_size *= @intCast(dim);

                    if (existing_m_states[0].len != expected_size) {
                        std.log.info("Worker {}: Reallocating M/V states for {s} precision (was {}, need {})", .{
                            self.node_id.?,
                            dtype_str,
                            existing_m_states[0].len,
                            expected_size,
                        });

                        // Free old buffers
                        for (existing_m_states) |buf| self.allocator.free(buf);
                        self.allocator.free(existing_m_states);
                        if (self.v_states) |existing_v_states| {
                            for (existing_v_states) |buf| self.allocator.free(buf);
                            self.allocator.free(existing_v_states);
                        }

                        // Allocate new buffers with correct dtype size
                        const num_params = shapes.len;
                        const new_m = try self.allocator.alloc([]u8, num_params);
                        const new_v = try self.allocator.alloc([]u8, num_params);

                        for (shapes, 0..) |shape, i| {
                            var size: usize = param_dtype.sizeInBytes();
                            for (shape) |dim| size *= @intCast(dim);

                            new_m[i] = try self.allocator.alloc(u8, size);
                            @memset(new_m[i], 0);
                            new_v[i] = try self.allocator.alloc(u8, size);
                            @memset(new_v[i], 0);
                        }

                        self.m_states = new_m;
                        self.v_states = new_v;
                        self.timestep = 1.0;
                    }
                }
            }
        }

        // Get m_states and v_states AFTER potential reallocation to avoid use-after-free
        const m_states = self.m_states.?;
        const v_states = self.v_states.?;

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
        } else {
            std.log.err("Unknown tokenizer type: {s}", .{tokenizer_type});
            return error.UnknownTokenizer;
        }

        // 5. Get params: either from chunked transfer (weight_blob) or inline base64
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        var initial_params_bytes_temp: []const u8 = undefined;
        const using_chunked_transfer = self.weight_blob != null and payload.get("params") == null;
        const using_raw_params = if (payload.get("raw_params")) |v| v == .bool and v.bool else false;

        if (using_chunked_transfer) {
            // Use pre-received weight_blob from chunked transfer
            std.log.info("Worker {}: Using chunked transfer weights ({} bytes, raw={})", .{ self.node_id.?, self.weight_blob.?.len, using_raw_params });

            if (using_raw_params) {
                // weight_blob contains raw parameter bytes directly
                initial_params_bytes_temp = self.weight_blob.?;
            } else {
                // weight_blob contains Cap'n Proto serialized bytes
                const reader = try binary_protocol.WorkerPayload.Reader.init(self.weight_blob.?);
                // Don't use defer - copy params before deinit to avoid use-after-free
                const params_from_reader = try reader.getParams();
                const params_copy = try arena.allocator().alloc(u8, params_from_reader.len);
                @memcpy(params_copy, params_from_reader);
                initial_params_bytes_temp = params_copy;
                reader.deinit();
            }
        } else {
            // Fall back to inline base64 params (for smaller models)
            const b64_params = switch (payload.get("params") orelse return error.MissingParams) {
                .string => |s| s,
                else => return error.InvalidParamsFormat,
            };

            const capnp_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_params);
            const decoded = try arena.allocator().alloc(u8, capnp_len);
            try std.base64.standard.Decoder.decode(decoded, b64_params);

            // 6. Deserialize params from Cap'n Proto
            const reader = try binary_protocol.WorkerPayload.Reader.init(decoded);
            // Don't use defer - copy params before deinit to avoid use-after-free
            const params_from_reader = try reader.getParams();
            const params_copy = try arena.allocator().alloc(u8, params_from_reader.len);
            @memcpy(params_copy, params_from_reader);
            initial_params_bytes_temp = params_copy;
            reader.deinit();
        }

        // Validate Received Parameters using the specific dtype
        std.log.info("Worker {}: Validating initial parameters ({} bytes, dtype={s})...",
            .{ self.node_id.?, initial_params_bytes_temp.len, @tagName(param_dtype) });

        try self.validateBuffer("Initial Params", initial_params_bytes_temp, param_dtype);

        // Save a persistent copy of initial params for delta calculation later
        const initial_params_bytes = try self.allocator.dupe(u8, initial_params_bytes_temp);
        defer self.allocator.free(initial_params_bytes);

        std.log.info("Starting inner loop: params={} bytes, chunk={}", .{ initial_params_bytes.len, chunk_id });

        // 7. Persistent optimizer states
        // We do NOT reset m_states, v_states, or timestep.
        // They must persist to maintain local momentum and curvature information
        // across communication rounds (as per DiLoCo Sec 6.1).

        // Ensure timestep is initialized if it's the very first run
        if (self.timestep == 0.0) self.timestep = 1.0;

        // 8. Copy initial params into working buffers (we'll update these in the loop)
        var current_params = try self.allocator.alloc([]u8, param_shapes.len);
        defer {
            for (current_params) |buf| self.allocator.free(buf);
            self.allocator.free(current_params);
        }

        var param_offset: usize = 0;
        const bytes_per_element = self.cached_param_dtype.sizeInBytes();
        for (param_shapes, 0..) |shape, i| {
            var size: usize = bytes_per_element;
            for (shape) |dim| size *= @intCast(dim);

            if (param_offset + size > initial_params_bytes.len) return error.ParameterBufferTooSmall;

            current_params[i] = try self.allocator.alloc(u8, size);
            @memcpy(current_params[i], initial_params_bytes[param_offset .. param_offset + size]);
            param_offset += size;
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

            // Build dtypes list for explicit precision control
            // Layout: [Params (N), M-States (N), V-States (N), Timestep (1), Data Inputs (D)]
            var dtypes_list = std.ArrayList(tensor.DType).init(self.allocator);
            defer dtypes_list.deinit();

            // Params, M-States, V-States all use param_dtype (f32/bf16/f16)
            for (0..3) |_| {
                for (0..param_shapes.len) |_| try dtypes_list.append(param_dtype);
            }

            // Timestep is always f32
            try dtypes_list.append(.f32);

            // Data inputs (batch inputs, targets) are always i64
            for (0..data_shapes.len) |_| try dtypes_list.append(.i64);

            // Execute the training step with explicit dtypes
            const outputs = try self.backend.executeTrainingStep(vmfb, inputs, shapes_slice, dtypes_list.items);
            defer {
                for (outputs) |o| self.allocator.free(o);
                self.allocator.free(outputs);
            }

            // Validate outputs for NaNs immediately to pinpoint graph issues
            const num_params_chk = param_shapes.len;

            // Check updated parameters
            for (0..num_params_chk) |i| {
                if (i < outputs.len) {
                    var name_buf: [32]u8 = undefined;
                    const name = std.fmt.bufPrint(&name_buf, "Output Param {}", .{i}) catch "Output Param";
                    try self.validateBuffer(name, outputs[i], param_dtype);
                }
            }

            // Check loss (last output)
            const loss_idx = outputs.len - 1;
            try self.validateBuffer("Loss", outputs[loss_idx], param_dtype);

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

            // Extract loss from outputs[3N] - loss is always bf16 scalar from the model
            const loss_bytes = outputs[3 * num_params];
            if (loss_bytes.len >= 2) {
                // Loss is bf16, convert to f32
                const bf16_bits = std.mem.readInt(u16, loss_bytes[0..2], .little);
                const f32_bits: u32 = @as(u32, bf16_bits) << 16;
                final_loss = @bitCast(f32_bits);
            }

            // Increment timestep for next iteration
            self.timestep += 1.0;
        }

        std.log.info("Worker {} completed {} inner loop steps, final loss: {d:.4}", .{ self.node_id.?, tau, final_loss });

        // 11. Calculate Delta = Initial_Params - Final_Params and send it
        var delta_bytes = std.ArrayList(u8).init(self.allocator);
        defer delta_bytes.deinit();

        // Iterate over parameter blocks to compute delta
        var delta_offset: usize = 0;
        for (current_params) |final_block_bytes| {
            const param_block_size = final_block_bytes.len;
            const initial_block_bytes = initial_params_bytes[delta_offset .. delta_offset + param_block_size];

            const num_elements = param_block_size / bytes_per_element;

            switch (self.cached_param_dtype) {
                .bf16 => {
                    // bf16: convert to f32 for arithmetic, store delta as bf16
                    for (0..num_elements) |j| {
                        const byte_off = j * 2;
                        // Convert initial bf16 to f32
                        const init_bf16 = std.mem.readInt(u16, initial_block_bytes[byte_off..][0..2], .little);
                        const init_f32: f32 = @bitCast(@as(u32, init_bf16) << 16);
                        // Convert final bf16 to f32
                        const final_bf16 = std.mem.readInt(u16, final_block_bytes[byte_off..][0..2], .little);
                        const final_f32: f32 = @bitCast(@as(u32, final_bf16) << 16);
                        // Calculate delta and convert back to bf16
                        const delta_f32 = init_f32 - final_f32;
                        const delta_bf16: u16 = @truncate(@as(u32, @bitCast(delta_f32)) >> 16);
                        try delta_bytes.appendSlice(std.mem.asBytes(&delta_bf16));
                    }
                },
                .f16 => {
                    // f16: similar conversion (simplified, may lose precision)
                    for (0..num_elements) |j| {
                        const byte_off = j * 2;
                        const init_f16 = std.mem.readInt(u16, initial_block_bytes[byte_off..][0..2], .little);
                        const final_f16 = std.mem.readInt(u16, final_block_bytes[byte_off..][0..2], .little);
                        // Simple subtraction in f16 space (not precise but functional)
                        const delta_f16 = init_f16 -% final_f16;
                        try delta_bytes.appendSlice(std.mem.asBytes(&delta_f16));
                    }
                },
                .f32 => {
                    // f32: direct arithmetic
                    const initial_f32 = std.mem.bytesAsSlice(f32, initial_block_bytes);
                    const final_f32 = std.mem.bytesAsSlice(f32, final_block_bytes);
                    for (0..num_elements) |j| {
                        const delta_val = initial_f32[j] - final_f32[j];
                        try delta_bytes.appendSlice(std.mem.asBytes(&delta_val));
                    }
                },
                else => {
                    // Fallback: treat as f32
                    const initial_f32 = std.mem.bytesAsSlice(f32, initial_block_bytes);
                    const final_f32 = std.mem.bytesAsSlice(f32, final_block_bytes);
                    for (0..num_elements) |j| {
                        const delta_val = initial_f32[j] - final_f32[j];
                        try delta_bytes.appendSlice(std.mem.asBytes(&delta_val));
                    }
                },
            }

            delta_offset += param_block_size;
        }

        // Send Delta using chunked transfer protocol to avoid Cap'n Proto message size limits
        const data = delta_bytes.items;
        const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB per chunk (same as shepherd->worker)
        const total_size = data.len;
        const total_chunks = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        std.log.info("Worker {}: Sending update in {} chunks ({} bytes total)", .{ self.node_id.?, total_chunks, total_size });

        var send_offset: usize = 0;
        var chunk_idx: usize = 0;
        while (chunk_idx < total_chunks) : (chunk_idx += 1) {
            const end = @min(send_offset + CHUNK_SIZE, total_size);
            const chunk_slice = data[send_offset..end];

            // Base64 encode the chunk
            const b64_len = std.base64.standard.Encoder.calcSize(chunk_slice.len);
            const b64_chunk = try self.allocator.alloc(u8, b64_len);
            defer self.allocator.free(b64_chunk);
            _ = std.base64.standard.Encoder.encode(b64_chunk, chunk_slice);

            // Construct chunk payload
            var chunk_payload = std.json.ObjectMap.init(self.allocator);
            defer chunk_payload.deinit();

            try chunk_payload.put("chunk_index", std.json.Value{ .integer = @intCast(chunk_idx) });
            try chunk_payload.put("total_chunks", std.json.Value{ .integer = @intCast(total_chunks) });
            try chunk_payload.put("total_bytes", std.json.Value{ .integer = @intCast(total_size) });
            try chunk_payload.put("data", std.json.Value{ .string = b64_chunk });
            try chunk_payload.put("loss", std.json.Value{ .float = final_loss });
            try chunk_payload.put("chunk_id", std.json.Value{ .integer = @intCast(self.current_chunk_id.?) });

            // Send UPDATE_CHUNK message
            const chunk_msg = tcp_stream.createMessage(
                self.node_id.?,
                "worker",
                0,
                "shepherd",
                MessageType.UPDATE_CHUNK,
                msg.msg_id +% 1 +% @as(u8, @truncate(chunk_idx)),
                std.json.Value{ .object = chunk_payload },
            );

            try self.client.send(chunk_msg);
            send_offset = end;

            // Small sleep every 5 chunks to prevent overwhelming network buffers
            if (chunk_idx % 5 == 4) std.time.sleep(10 * std.time.ns_per_ms);
        }

        std.log.info("Worker {}: All {} chunks sent, sending completion signal", .{ self.node_id.?, total_chunks });

        // Send final INNER_LOOP_COMPLETE with just metadata (no params - they were sent via chunks)
        var response_payload = std.json.ObjectMap.init(self.allocator);
        defer response_payload.deinit();
        try response_payload.put("chunk_id", std.json.Value{ .integer = @intCast(self.current_chunk_id.?) });
        try response_payload.put("loss", std.json.Value{ .float = final_loss });
        try response_payload.put("chunked", std.json.Value{ .bool = true }); // Signal that params were sent via chunks

        const response = tcp_stream.createMessage(
            self.node_id.?,
            "worker",
            0,
            "shepherd",
            MessageType.INNER_LOOP_COMPLETE,
            msg.msg_id +% 1 +% @as(u8, @truncate(total_chunks)),
            std.json.Value{ .object = response_payload },
        );

        try self.client.send(response);
        self.state = .connected;
    }
    /// Handle weight update from Shepherd
    fn handleUpdateWeights(self: *Self, msg: MessageEnvelope) !void {
        std.log.info("Worker {}: Receiving weight update...", .{self.node_id.?});

        const payload = switch (msg.data) {
            .object => |o| o,
            else => return error.InvalidFormat,
        };

        const weights_value = payload.get("weights") orelse return error.MissingWeights;
        const b64_weights = switch (weights_value) {
            .string => |s| s,
            else => return error.InvalidWeightsFormat,
        };

        // Decode Base64
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(b64_weights);

        // Free old weight blob if exists
        if (self.weight_blob) |old_blob| {
            self.allocator.free(old_blob);
        }

        // Allocate and decode new weights
        self.weight_blob = try self.allocator.alloc(u8, decoded_len);
        errdefer {
            self.allocator.free(self.weight_blob.?);
            self.weight_blob = null;
        }

        _ = try std.base64.standard.Decoder.decode(self.weight_blob.?, b64_weights);

        // Mark device weights as dirty so they get re-uploaded on next rollout
        self.device_weights_dirty = true;

        std.log.info("Worker {}: Weights updated ({} bytes)", .{ self.node_id.?, decoded_len });
    }

    /// Handle a Rollout Request (RL Generation) with Device Residency for Weights
    /// Weights stay on GPU VRAM (~2GB saved), KV Cache uses host-side accumulation (~25MB/step)
    fn handleStartRollout(self: *Self, msg: MessageEnvelope) !void {
        self.state = .training;

        // 1. Parse Payload
        const payload = switch (msg.data) {
            .object => |obj| obj,
            else => return error.InvalidMessageFormat,
        };

        // Get Prompt (Input IDs)
        const prompt_json = payload.get("prompt") orelse return error.MissingPrompt;
        var prompt_list = std.ArrayList(i64).init(self.allocator);
        defer prompt_list.deinit();

        switch (prompt_json) {
            .array => |arr| {
                for (arr.items) |item| {
                    try prompt_list.append(item.integer);
                }
            },
            else => return error.InvalidPromptFormat,
        }

        const max_new_tokens = 128;
        var generated_tokens = std.ArrayList(i64).init(self.allocator);
        defer generated_tokens.deinit();

        // Cast backend to IreeBackend to access device residency methods
        const iree_impl: *IreeBackend = @ptrCast(@alignCast(self.backend.ptr));

        const param_shapes = self.cached_parameter_shapes orelse return error.ParameterShapesNotInitialized;
        const data_shapes = self.cached_data_input_shapes orelse return error.DataShapesNotInitialized;

        // 1.5 PRE-LOAD SESSION (JIT compile CUDA kernels while GPU memory is free)
        if (self.cached_vmfb) |vmfb| {
            std.log.info("Worker {}: Pre-loading session (JIT compiling kernels)...", .{self.node_id.?});
            try iree_impl.loadSession(vmfb);
        }

        // 2. UPLOAD WEIGHTS TO VRAM (Once per UPDATE_WEIGHTS, not every token!)
        if (self.device_weights == null or self.device_weights_dirty) {
            if (self.weight_blob == null) {
                std.log.err("Worker {}: No weights loaded for generation!", .{self.node_id.?});
                return error.WeightsNotInitialized;
            }

            std.log.info("Worker {}: Moving Weights to VRAM (one-time cost)...", .{self.node_id.?});

            // Release old device weights if they exist
            if (self.device_weights) |old_weights| {
                for (old_weights) |buf| buf.release();
                self.allocator.free(old_weights);
            }

            var dev_weights = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
            errdefer {
                for (dev_weights.items) |buf| buf.release();
                dev_weights.deinit();
            }

            var offset: usize = 0;
            for (param_shapes) |shape| {
                var size: usize = 4; // f32
                for (shape) |dim| size *= @intCast(dim);

                const slice = self.weight_blob.?[offset .. offset + size];
                const dev_buf = try iree_impl.moveToDevice(slice, shape, .f32);
                try dev_weights.append(dev_buf);

                offset += size;
            }

            self.device_weights = try dev_weights.toOwnedSlice();
            self.device_weights_dirty = false;
            std.log.info("Worker {}: Weights uploaded ({} parameters, {} bytes total)", .{
                self.node_id.?,
                param_shapes.len,
                offset,
            });
        }

        // 3. INITIALIZE KV CACHE ON HOST (model outputs single-position slices, we accumulate)
        std.log.info("Worker {}: Initializing KV cache on host ({} buffers)...", .{ self.node_id.?, data_shapes.len - 2 });
        if (self.kv_cache) |old_cache| {
            for (old_cache) |buf| self.allocator.free(buf);
            self.allocator.free(old_cache);
        }

        const kv_shapes = data_shapes[2..];
        const kv_buffers = try self.allocator.alloc([]u8, kv_shapes.len);
        for (kv_shapes, 0..) |shape, i| {
            var size: usize = 4; // f32
            for (shape) |dim| size *= @intCast(dim);
            kv_buffers[i] = try self.allocator.alloc(u8, size);
            @memset(kv_buffers[i], 0);
        }
        self.kv_cache = kv_buffers;
        std.log.info("Worker {}: KV cache initialized on host", .{self.node_id.?});

        // 4. GENERATION LOOP
        var current_pos: i64 = 0;
        const total_len = prompt_list.items.len + max_new_tokens;

        std.log.info("Worker {}: Starting generation loop (total_len={})...", .{ self.node_id.?, total_len });

        while (current_pos < total_len) {
            if (current_pos == 0) std.log.info("Worker {}: Generation step 0 - building inputs...", .{self.node_id.?});

            // Select input token
            var input_token: i64 = 0;
            if (current_pos < prompt_list.items.len) {
                input_token = prompt_list.items[@intCast(current_pos)];
            } else {
                if (generated_tokens.items.len > 0) {
                    input_token = generated_tokens.items[generated_tokens.items.len - 1];
                }
            }

            // Build Input List of DeviceBuffers
            var inputs = std.ArrayList(IreeBackend.DeviceBuffer).init(self.allocator);
            defer inputs.deinit();

            // A. Weights (Already on Device - just add refs, no transfer!)
            for (self.device_weights.?) |w| {
                w.retain();
                try inputs.append(w);
            }
            if (current_pos == 0) std.log.info("Worker {}: Added {} weight refs", .{ self.node_id.?, self.device_weights.?.len });

            // B. Token & Position (16 bytes total)
            const token_bytes = std.mem.asBytes(&input_token);
            const dev_token = try iree_impl.moveToDevice(token_bytes, data_shapes[0], .i64);
            try inputs.append(dev_token);

            const pos_bytes = std.mem.asBytes(&current_pos);
            const dev_pos = try iree_impl.moveToDevice(pos_bytes, data_shapes[1], .i64);
            try inputs.append(dev_pos);
            if (current_pos == 0) std.log.info("Worker {}: Uploaded token+pos to device", .{self.node_id.?});

            // C. KV Cache (Upload from host - ~25MB per step)
            for (self.kv_cache.?, kv_shapes) |kv_buf, shape| {
                const dev_kv = try iree_impl.moveToDevice(kv_buf, shape, .f32);
                try inputs.append(dev_kv);
            }
            if (current_pos == 0) std.log.info("Worker {}: Uploaded {} KV cache buffers, executing...", .{ self.node_id.?, kv_shapes.len });

            // EXECUTE (Weights stay on device - saves ~2GB transfer!)
            const outputs = try iree_impl.executeWithDeviceBuffers(
                self.cached_vmfb.?,
                "main",
                inputs.items,
            );
            defer self.allocator.free(outputs);
            if (current_pos == 0) std.log.info("Worker {}: First execute complete!", .{self.node_id.?});

            // Release input refs
            for (inputs.items) |buf| buf.release();

            // PROCESS OUTPUTS: [Logits, KV_0, KV_1, ...]

            // 1. Download Logits (~600KB)
            const logits_bytes = try iree_impl.readToHost(outputs[0]);
            defer self.allocator.free(logits_bytes);
            outputs[0].release();

            const logits_f32 = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, logits_bytes)));

            // 2. Sample Logic
            const is_last_prompt_token = (current_pos == prompt_list.items.len - 1);
            const is_generating = (current_pos >= prompt_list.items.len);

            if ((is_last_prompt_token and prompt_list.items.len > 0) or is_generating) {
                const next_token = self.sampler.sample(logits_f32, 0.7);
                try generated_tokens.append(next_token);

                if (next_token == 151643) {
                    for (outputs[1..]) |buf| buf.release();
                    break;
                }
                if (generated_tokens.items.len >= max_new_tokens) {
                    for (outputs[1..]) |buf| buf.release();
                    break;
                }
            }

            // 3. Update KV Cache on Host (scatter single-position outputs into full cache)
            const NUM_KV_HEADS: usize = 2;
            const HEAD_DIM: usize = 64;
            const MAX_SEQ_LEN: usize = 1024;
            const bytes_per_head_slot: usize = HEAD_DIM * 4;
            const head_stride: usize = MAX_SEQ_LEN * bytes_per_head_slot;

            for (outputs[1..], 0..) |output_buf, i| {
                const update_bytes = try iree_impl.readToHost(output_buf);
                defer self.allocator.free(update_bytes);
                output_buf.release();

                const cache_buffer = self.kv_cache.?[i];

                for (0..NUM_KV_HEADS) |head| {
                    const src_offset = head * bytes_per_head_slot;
                    const dst_offset = head * head_stride + @as(usize, @intCast(current_pos)) * bytes_per_head_slot;

                    if (dst_offset + bytes_per_head_slot <= cache_buffer.len and
                        src_offset + bytes_per_head_slot <= update_bytes.len)
                    {
                        @memcpy(
                            cache_buffer[dst_offset .. dst_offset + bytes_per_head_slot],
                            update_bytes[src_offset .. src_offset + bytes_per_head_slot],
                        );
                    }
                }
            }

            current_pos += 1;
        }

        std.log.info("Worker {d}: Generated {d} tokens: {any}", .{
            self.node_id.?,
            generated_tokens.items.len,
            generated_tokens.items,
        });

        // 5. Send Results Back
        var result_payload = std.json.ObjectMap.init(self.allocator);
        defer result_payload.deinit();

        var tokens_array = std.json.Array.init(self.allocator);
        defer tokens_array.deinit();
        for (generated_tokens.items) |t| try tokens_array.append(.{ .integer = t });

        try result_payload.put("completion", .{ .array = tokens_array });

        const response = message.MessageEnvelope{
            .recipient_node = 0,
            .recipient_service = "shepherd",
            .sender_node = self.node_id.?,
            .sender_service = "worker",
            .msg_type = MessageType.ROLLOUT_COMPLETE,
            .msg_id = msg.msg_id + 1,
            .data = .{ .object = result_payload },
        };

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