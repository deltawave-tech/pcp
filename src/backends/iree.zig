/// IREE Backend implementation for WorkerBackend interface
/// This backend uses the IREE runtime to execute VMFB modules on various hardware accelerators
const std = @import("std");
const c = @cImport({
    // Disable atomic support entirely - we don't need it for basic runtime usage
    @cDefine("IREE_SYNCHRONIZATION_DISABLE_UNSAFE", "1");
    // Enable all available drivers that we linked
    @cDefine("IREE_HAVE_HAL_LOCAL_SYNC_DRIVER_MODULE", "1");
    @cDefine("IREE_HAVE_HAL_METAL_DRIVER_MODULE", "1");
    @cDefine("IREE_HAVE_HAL_VULKAN_DRIVER_MODULE", "1");
    @cDefine("IREE_HAVE_HAL_CUDA_DRIVER_MODULE", "1");
    // Configure system allocator to use libc (matching IREE build configuration)
    @cDefine("IREE_ALLOCATOR_SYSTEM_CTL", "iree_allocator_libc_ctl");
    @cInclude("iree/base/api.h");
    @cInclude("iree/base/allocator.h");
    @cInclude("iree/runtime/api.h");
});
const WorkerBackend = @import("worker_backend.zig").WorkerBackend;
const backend_selection = @import("selection.zig");
const mlir = @import("../mlir/wrapper.zig");
const pcp = @import("../main.zig");
const DType = pcp.tensor.DType;

const ResolvedDeviceOrdinal = struct {
    ordinal: usize,
    visibility_env_name: ?[]const u8 = null,
    visibility_env_value: ?[]const u8 = null,
    visible_device_count: ?usize = null,
};

// Helper to check for IREE errors
fn ireeCheck(status: c.iree_status_t) !void {
    // In IREE, null status indicates success, non-null indicates error
    if (status != null) {
        std.debug.print("\n======== IREE RUNTIME ERROR ========\n", .{});
        // Use iree_status_fprint for full error output including annotations
        c.iree_status_fprint(c.stderr, status);
        std.debug.print("====================================\n\n", .{});
        c.iree_status_free(status);
        return error.IreeRuntimeError;
    }
}

fn logAndFreeIreeStatus(context: []const u8, status: c.iree_status_t) void {
    if (status == null) return;
    std.log.warn("{s}", .{context});
    c.iree_status_fprint(c.stderr, status);
    c.iree_status_free(status);
}

fn resolveVisibleDeviceOrdinal(backend: backend_selection.Backend, requested_device_id: usize) !ResolvedDeviceOrdinal {
    const env_name = visibleDeviceEnvName(backend) orelse return .{ .ordinal = requested_device_id };
    const env_value = std.posix.getenv(env_name) orelse return .{ .ordinal = requested_device_id };
    const visible_count = countVisibleDevices(env_value);
    if (requested_device_id >= visible_count) {
        std.log.err(
            "{s} worker requested device {} but {s} exposes only {} device(s): '{s}'",
            .{ backend.toString(), requested_device_id, env_name, visible_count, env_value },
        );
        return error.DeviceNotVisible;
    }
    return .{
        .ordinal = requested_device_id,
        .visibility_env_name = env_name,
        .visibility_env_value = env_value,
        .visible_device_count = visible_count,
    };
}

fn visibleDeviceEnvName(backend: backend_selection.Backend) ?[]const u8 {
    return switch (backend) {
        .cuda => "CUDA_VISIBLE_DEVICES",
        .rocm => rocmVisibleDeviceEnvName(),
        else => null,
    };
}

fn rocmVisibleDeviceEnvName() ?[]const u8 {
    if (std.posix.getenv("HIP_VISIBLE_DEVICES") != null) return "HIP_VISIBLE_DEVICES";
    if (std.posix.getenv("ROCR_VISIBLE_DEVICES") != null) return "ROCR_VISIBLE_DEVICES";
    if (std.posix.getenv("GPU_DEVICE_ORDINAL") != null) return "GPU_DEVICE_ORDINAL";
    return null;
}

fn countVisibleDevices(raw: []const u8) usize {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0 or std.mem.eql(u8, trimmed, "-1")) return 0;

    var count: usize = 0;
    var tokens = std.mem.splitScalar(u8, trimmed, ',');
    while (tokens.next()) |token| {
        const device = std.mem.trim(u8, token, " \t\r\n");
        if (device.len == 0) continue;
        if (std.mem.eql(u8, device, "-1")) continue;
        count += 1;
    }
    return count;
}

pub const IreeBackend = struct {
    allocator: std.mem.Allocator,
    instance: ?*c.iree_runtime_instance_t,
    device: ?*c.iree_hal_device_t,
    session: ?*c.iree_runtime_session_t,
    backend: backend_selection.Backend,
    device_id: usize,

    // Cache tracking to detect if we need to reload the module
    loaded_vmfb_ptr: ?[*]const u8 = null,
    loaded_vmfb_len: usize = 0,

    const Self = @This();

    /// A handle to a buffer resident on the device memory (VRAM)
    /// Uses IREE reference counting for lifetime management
    pub const DeviceBuffer = struct {
        handle: *c.iree_hal_buffer_view_t,

        pub fn retain(self: DeviceBuffer) void {
            c.iree_hal_buffer_view_retain(self.handle);
        }

        pub fn release(self: DeviceBuffer) void {
            c.iree_hal_buffer_view_release(self.handle);
        }

        /// Get the byte length of this buffer
        pub fn byteLength(self: DeviceBuffer) usize {
            const buf_ptr = c.iree_hal_buffer_view_buffer(self.handle);
            return @intCast(c.iree_hal_buffer_byte_length(buf_ptr));
        }

        pub fn shapeAlloc(self: DeviceBuffer, allocator: std.mem.Allocator) ![]i64 {
            const rank: usize = @intCast(c.iree_hal_buffer_view_shape_rank(self.handle));
            const shape = try allocator.alloc(i64, rank);
            errdefer allocator.free(shape);
            for (shape, 0..) |*dim, index| {
                dim.* = @intCast(c.iree_hal_buffer_view_shape_dim(self.handle, @intCast(index)));
            }
            return shape;
        }
    };

    // Helper to map DType to IREE element type
    fn getIreeType(dtype: DType) c.iree_hal_element_type_t {
        return switch (dtype) {
            .f32 => c.IREE_HAL_ELEMENT_TYPE_FLOAT_32,
            .f64 => c.IREE_HAL_ELEMENT_TYPE_FLOAT_64,
            .f16 => c.IREE_HAL_ELEMENT_TYPE_FLOAT_16,
            .bf16 => c.IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
            .i32 => c.IREE_HAL_ELEMENT_TYPE_SINT_32,
            .i64 => c.IREE_HAL_ELEMENT_TYPE_SINT_64,
            .bool => c.IREE_HAL_ELEMENT_TYPE_BOOL_8,
        };
    }

    fn transientInputBufferParams(self: *const Self) c.iree_hal_buffer_params_t {
        const memory_type: c.iree_hal_memory_type_t = @intCast(c.IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
        const usage: c.iree_hal_buffer_usage_t = @intCast(c.IREE_HAL_BUFFER_USAGE_DEFAULT);
        _ = self;
        return .{
            .type = memory_type,
            .access = c.IREE_HAL_MEMORY_ACCESS_ALL,
            .usage = usage,
        };
    }

    pub fn init(allocator: std.mem.Allocator, backend: backend_selection.Backend, device_id: usize) !*IreeBackend {
        const resolved_device = try resolveVisibleDeviceOrdinal(backend, device_id);
        var self = try allocator.create(IreeBackend);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        self.backend = backend;
        self.device_id = resolved_device.ordinal;
        self.session = null; // Session created on first execute
        self.loaded_vmfb_ptr = null;
        self.loaded_vmfb_len = 0;

        // 1. Create Instance with driver registry
        var instance_options: c.iree_runtime_instance_options_t = undefined;
        c.iree_runtime_instance_options_initialize(&instance_options);
        c.iree_runtime_instance_options_use_all_available_drivers(&instance_options);

        try ireeCheck(c.iree_runtime_instance_create(
            &instance_options,
            c.iree_allocator_system(),
            &self.instance,
        ));

        // 2. Create device using HAL driver API for proper device selection
        const driver_name = backend.toIreeDriverName();

        if (backend == .cuda or backend == .rocm) {
            // For GPU backends, use the HAL driver API to select specific device by ordinal
            if (resolved_device.visibility_env_name) |env_name| {
                std.log.info(
                    "Creating {s} device with ordinal {} from {s}={s} (requested device {}, visible count {})",
                    .{
                        backend.toString(),
                        resolved_device.ordinal,
                        env_name,
                        resolved_device.visibility_env_value.?,
                        device_id,
                        resolved_device.visible_device_count.?,
                    },
                );
            } else {
                std.log.info("Creating {s} device with ordinal {} (no visibility env set)", .{ backend.toString(), resolved_device.ordinal });
            }

            // Get the driver registry from the instance
            const registry = c.iree_runtime_instance_driver_registry(self.instance.?);
            if (registry == null) {
                std.log.err("No driver registry available", .{});
                return error.NoDriverRegistry;
            }

            // Create the driver
            var driver: ?*c.iree_hal_driver_t = null;
            try ireeCheck(c.iree_hal_driver_registry_try_create(
                registry.?,
                c.iree_string_view_t{ .data = driver_name.ptr, .size = driver_name.len },
                c.iree_allocator_system(),
                &driver,
            ));
            defer if (driver) |d| c.iree_hal_driver_release(d);

            // Create device by ordinal (0, 1, 2, ... for each GPU)
            try ireeCheck(c.iree_hal_driver_create_device_by_ordinal(
                driver.?,
                @intCast(resolved_device.ordinal), // device ordinal
                0, // param_count
                null, // params
                c.iree_allocator_system(),
                &self.device,
            ));

            std.log.info("Successfully created device {} for {s} backend", .{ resolved_device.ordinal, backend.toString() });
        } else {
            // For CPU and other backends, use default device
            std.log.info("Creating default device for {s} backend", .{backend.toString()});
            try ireeCheck(c.iree_runtime_instance_try_create_default_device(
                self.instance.?,
                c.iree_string_view_t{ .data = driver_name.ptr, .size = driver_name.len },
                &self.device,
            ));
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.unloadSession();
        self.synchronizeAndTrim();
        if (self.device) |device| c.iree_hal_device_release(device);
        if (self.instance) |instance| c.iree_runtime_instance_release(instance);
        self.allocator.destroy(self);
    }

    pub fn unloadSession(self: *Self) void {
        if (self.session) |session| c.iree_runtime_session_release(session);
        self.session = null;
        self.loaded_vmfb_ptr = null;
        self.loaded_vmfb_len = 0;
        self.synchronizeAndTrim();
    }

    pub fn synchronizeAndTrim(self: *Self) void {
        if (self.backend != .cuda) return;
        const device = self.device orelse return;

        var signal_semaphore: ?*c.iree_hal_semaphore_t = null;
        var status = c.iree_hal_semaphore_create(
            device,
            c.IREE_HAL_QUEUE_AFFINITY_ANY,
            0,
            c.IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
            &signal_semaphore,
        );
        if (status != null) {
            logAndFreeIreeStatus("IREE CUDA semaphore create failed during trim", status);
            _ = c.iree_hal_device_trim(device);
            return;
        }

        const semaphore = signal_semaphore.?;
        defer c.iree_hal_semaphore_release(semaphore);

        var signal_payload: u64 = 1;
        var semaphore_ptr: ?*c.iree_hal_semaphore_t = semaphore;
        const empty_semaphores = c.iree_hal_semaphore_list_t{
            .count = 0,
            .semaphores = null,
            .payload_values = null,
        };
        const signal_semaphores = c.iree_hal_semaphore_list_t{
            .count = 1,
            .semaphores = &semaphore_ptr,
            .payload_values = &signal_payload,
        };

        status = c.iree_hal_device_queue_barrier(
            device,
            c.IREE_HAL_QUEUE_AFFINITY_ANY,
            empty_semaphores,
            signal_semaphores,
            c.IREE_HAL_EXECUTE_FLAG_NONE,
        );
        if (status == null) {
            status = c.iree_hal_semaphore_wait(
                semaphore,
                signal_payload,
                c.iree_infinite_timeout(),
                c.IREE_HAL_WAIT_FLAG_DEFAULT,
            );
        }
        if (status != null) {
            logAndFreeIreeStatus("IREE CUDA synchronize failed during trim", status);
        }

        status = c.iree_hal_device_trim(device);
        if (status != null) {
            logAndFreeIreeStatus("IREE CUDA device trim failed", status);
        }
    }

    // ========================================================================
    // Device Residency API - Keep tensors on GPU to avoid PCIe transfers
    // ========================================================================

    /// Allocate a buffer on the device and copy host data into it immediately.
    /// The returned DeviceBuffer is owned by the caller and must be released.
    pub fn moveToDevice(self: *Self, data: []const u8, shape: []const i64, dtype: DType) !DeviceBuffer {
        // 1. Create Shape
        var iree_shape = try self.allocator.alloc(c.iree_hal_dim_t, shape.len);
        defer self.allocator.free(iree_shape);
        for (shape, 0..) |dim, k| iree_shape[k] = @intCast(dim);

        const element_type = getIreeType(dtype);

        // 2. Allocate and Copy (Host -> Device) using DEVICE_LOCAL memory (VRAM)
        var buffer_view: ?*c.iree_hal_buffer_view_t = null;

        try ireeCheck(c.iree_hal_buffer_view_allocate_buffer_copy(
            self.device.?,
            c.iree_hal_device_allocator(self.device.?),
            @intCast(shape.len),
            iree_shape.ptr,
            element_type,
            c.IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
            self.transientInputBufferParams(),
            .{ .data = data.ptr, .data_length = data.len },
            @ptrCast(&buffer_view),
        ));

        return DeviceBuffer{ .handle = buffer_view.? };
    }

    /// Allocate a zero-filled buffer directly on the device.
    /// The returned DeviceBuffer is owned by the caller and must be released.
    pub fn allocateZerosDevice(self: *Self, shape: []const i64, dtype: DType) !DeviceBuffer {
        // Calculate byte size for zero initialization
        var byte_size: usize = dtype.sizeInBytes();
        for (shape) |d| byte_size *= @intCast(d);

        // Allocate zeros on host (temporary)
        const zeros = try self.allocator.alloc(u8, byte_size);
        defer self.allocator.free(zeros);
        @memset(zeros, 0);

        return self.moveToDevice(zeros, shape, dtype);
    }

    /// Pre-load session and JIT compile CUDA kernels.
    /// Call this BEFORE uploading weights to ensure compilation has scratch memory.
    pub fn loadSession(self: *Self, vmfb_bytes: []const u8) !void {
        const need_load = self.session == null or
            self.loaded_vmfb_ptr != vmfb_bytes.ptr or
            self.loaded_vmfb_len != vmfb_bytes.len;

        if (need_load) {
            // Teardown old session if it exists
            if (self.session) |s| {
                std.log.info("IreeBackend: Artifact changed, reloading session...", .{});
                c.iree_runtime_session_release(s);
                self.session = null;
            } else {
                std.log.info("IreeBackend: Pre-loading session (JIT compiling kernels)...", .{});
            }

            // Create fresh session
            var session_options: c.iree_runtime_session_options_t = undefined;
            c.iree_runtime_session_options_initialize(&session_options);
            try ireeCheck(c.iree_runtime_session_create_with_device(
                self.instance.?,
                &session_options,
                self.device.?,
                c.iree_runtime_instance_host_allocator(self.instance.?),
                &self.session,
            ));

            // Load the bytecode module
            try ireeCheck(c.iree_runtime_session_append_bytecode_module_from_memory(
                self.session.?,
                .{ .data = vmfb_bytes.ptr, .data_length = vmfb_bytes.len },
                c.iree_allocator_null(),
            ));

            // Update tracking
            self.loaded_vmfb_ptr = vmfb_bytes.ptr;
            self.loaded_vmfb_len = vmfb_bytes.len;
            std.log.info("IreeBackend: JIT compilation complete.", .{});
        }
    }

    /// Execute using pre-allocated DeviceBuffers.
    /// Returns a list of DeviceBuffers (caller owns them and must release each).
    /// This avoids Host<->Device transfers for inputs that are already on device.
    pub fn executeWithDeviceBuffers(
        self: *Self,
        vmfb_bytes: []const u8,
        function_name: []const u8,
        inputs: []const DeviceBuffer,
    ) ![]DeviceBuffer {
        std.log.debug("IREE: executeWithDeviceBuffers starting for '{s}' with {} inputs", .{ function_name, inputs.len });
        defer self.synchronizeAndTrim();

        // 1. Ensure session is loaded
        try self.loadSession(vmfb_bytes);
        std.log.debug("IREE: Session loaded", .{});

        // 2. Initialize call
        var call: c.iree_runtime_call_t = undefined;

        const full_fn_name = try std.fmt.allocPrint(self.allocator, "module.{s}", .{function_name});
        defer self.allocator.free(full_fn_name);

        const fn_name_view = c.iree_string_view_t{ .data = full_fn_name.ptr, .size = full_fn_name.len };
        try ireeCheck(c.iree_runtime_call_initialize_by_name(self.session.?, fn_name_view, &call));
        defer c.iree_runtime_call_deinitialize(&call);
        std.log.debug("IREE: Call initialized for '{s}'", .{full_fn_name});

        // 3. Push Inputs (Cheap pointer push - no H2D copy!)
        for (inputs, 0..) |buf, idx| {
            try ireeCheck(c.iree_runtime_call_inputs_push_back_buffer_view(&call, buf.handle));
            std.log.debug("IREE: Pushed input {} ({} bytes)", .{ idx, buf.byteLength() });
        }

        // 4. Invoke (GPU Compute)
        std.log.info("IREE: Invoking GPU kernel...", .{});
        try ireeCheck(c.iree_runtime_call_invoke(&call, 0));
        std.log.info("IREE: GPU kernel completed successfully", .{});
        try ireeCheck(c.iree_vm_list_resize(c.iree_runtime_call_inputs(&call), 0));

        // 5. Collect Outputs as DeviceBuffers (No D2H copy yet!)
        var outputs_list = std.ArrayList(DeviceBuffer).init(self.allocator);
        errdefer {
            for (outputs_list.items) |buf| buf.release();
            outputs_list.deinit();
        }

        const outputs = c.iree_runtime_call_outputs(&call);
        const output_count = c.iree_vm_list_size(outputs);
        var i: c.iree_host_size_t = 0;

        std.log.debug("IREE: Collecting {} outputs", .{output_count});
        while (i < output_count) : (i += 1) {
            var output_view: ?*c.iree_hal_buffer_view_t = null;
            // Pop retains the buffer view, so we own it now
            try ireeCheck(c.iree_runtime_call_outputs_pop_front_buffer_view(&call, @ptrCast(&output_view)));
            try outputs_list.append(DeviceBuffer{ .handle = output_view.? });
        }
        std.log.debug("IREE: executeWithDeviceBuffers completed", .{});

        return outputs_list.toOwnedSlice();
    }

    /// Read a DeviceBuffer back to Host memory.
    /// Caller owns the returned slice and must free it.
    pub fn readToHost(self: *Self, device_buf: DeviceBuffer) ![]u8 {
        const buf_ptr = c.iree_hal_buffer_view_buffer(device_buf.handle);
        const length = c.iree_hal_buffer_byte_length(buf_ptr);

        const host_data = try self.allocator.alloc(u8, @intCast(length));
        errdefer self.allocator.free(host_data);

        try ireeCheck(c.iree_hal_device_transfer_d2h(
            self.device.?,
            buf_ptr,
            0,
            host_data.ptr,
            @intCast(length),
            c.IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
            c.iree_infinite_timeout(),
        ));

        return host_data;
    }

    /// Execute a function. Reuses the session if the vmfb_bytes pointer matches the previously loaded one.
    pub fn execute(
        self: *Self,
        vmfb_bytes: []const u8,
        function_name: []const u8,
        inputs_data: [][]const u8,
        input_shapes: [][]const i64,
        input_dtypes: ?[]const DType,
    ) ![][]u8 {
        std.log.info("IREE: execute starting for '{s}' with {} inputs", .{ function_name, inputs_data.len });
        defer self.synchronizeAndTrim();

        // 1. Ensure session is loaded
        try self.loadSession(vmfb_bytes);
        std.log.debug("IREE: Session loaded", .{});

        // 2. Initialize call
        var call: c.iree_runtime_call_t = undefined;

        // IREE expects the format "module.<function_name>"
        const full_fn_name = try std.fmt.allocPrint(self.allocator, "module.{s}", .{function_name});
        defer self.allocator.free(full_fn_name);

        const fn_name_view = c.iree_string_view_t{ .data = full_fn_name.ptr, .size = full_fn_name.len };
        try ireeCheck(c.iree_runtime_call_initialize_by_name(self.session.?, fn_name_view, &call));
        defer c.iree_runtime_call_deinitialize(&call);
        std.log.debug("IREE: Call initialized for '{s}'", .{full_fn_name});

        // 3. Prepare Inputs (Host -> Device)
        std.debug.assert(inputs_data.len == input_shapes.len);

        std.log.debug("IREE: Uploading {} inputs to device...", .{inputs_data.len});
        for (inputs_data, input_shapes, 0..) |input_slice, shape_slice, i| {
            // Convert shapes to IREE format
            var iree_shape = try self.allocator.alloc(c.iree_hal_dim_t, shape_slice.len);
            defer self.allocator.free(iree_shape);
            for (shape_slice, 0..) |dim, k| {
                iree_shape[k] = @intCast(dim);
            }

            // Determine element type: Use provided dtype or default to f32
            const element_type = if (input_dtypes) |dtypes| getIreeType(dtypes[i]) else c.IREE_HAL_ELEMENT_TYPE_FLOAT_32;

            // Create buffer view directly from data
            var buffer_view: ?*c.iree_hal_buffer_view_t = null;
            ireeCheck(c.iree_hal_buffer_view_allocate_buffer_copy(
                c.iree_runtime_session_device(self.session.?),
                c.iree_runtime_session_device_allocator(self.session.?),
                @intCast(shape_slice.len),
                iree_shape.ptr,
                element_type,
                c.IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                self.transientInputBufferParams(),
                .{ .data = input_slice.ptr, .data_length = input_slice.len },
                @ptrCast(&buffer_view),
            )) catch |err| {
                std.log.err("IREE: failed uploading input {} for '{s}' ({} bytes)", .{ i, function_name, input_slice.len });
                return err;
            };
            errdefer c.iree_hal_buffer_view_release(buffer_view.?);

            ireeCheck(c.iree_runtime_call_inputs_push_back_buffer_view(&call, buffer_view.?)) catch |err| {
                std.log.err("IREE: failed pushing input {} for '{s}' ({} bytes)", .{ i, function_name, input_slice.len });
                return err;
            };
            c.iree_hal_buffer_view_release(buffer_view.?);
            std.log.debug("IREE: Uploaded input {} ({} bytes)", .{ i, input_slice.len });
        }

        // 4. Invoke
        std.log.info("IREE: Invoking GPU kernel for '{s}'...", .{function_name});
        try ireeCheck(c.iree_runtime_call_invoke(&call, 0));
        std.log.info("IREE: GPU kernel '{s}' completed successfully", .{function_name});
        try ireeCheck(c.iree_vm_list_resize(c.iree_runtime_call_inputs(&call), 0));

        // 5. Read Outputs (Device -> Host)
        var outputs_list = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (outputs_list.items) |o| self.allocator.free(o);
            outputs_list.deinit();
        }

        const outputs = c.iree_runtime_call_outputs(&call);
        const output_count = c.iree_vm_list_size(outputs);
        var i: c.iree_host_size_t = 0;

        std.log.debug("IREE: Reading {} outputs from device...", .{output_count});
        while (i < output_count) : (i += 1) {
            var output_buffer_view: ?*c.iree_hal_buffer_view_t = null;
            try ireeCheck(c.iree_runtime_call_outputs_pop_front_buffer_view(&call, @ptrCast(&output_buffer_view)));
            errdefer c.iree_hal_buffer_view_release(output_buffer_view.?);

            const output_buffer = c.iree_hal_buffer_view_buffer(output_buffer_view.?);
            const buffer_byte_length = c.iree_hal_buffer_byte_length(output_buffer);

            const output_data = try self.allocator.alloc(u8, @intCast(buffer_byte_length));
            try ireeCheck(c.iree_hal_device_transfer_d2h(
                c.iree_runtime_session_device(self.session.?),
                output_buffer,
                0,
                output_data.ptr,
                @intCast(buffer_byte_length),
                c.IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                c.iree_infinite_timeout(),
            ));

            try outputs_list.append(output_data);
            c.iree_hal_buffer_view_release(output_buffer_view.?);
            std.log.debug("IREE: Downloaded output {} ({} bytes)", .{ i, buffer_byte_length });
        }

        std.log.info("IREE: execute completed for '{s}'", .{function_name});
        return outputs_list.toOwnedSlice();
    }

    /// Create WorkerBackend interface from this IREE backend
    pub fn asWorkerBackend(self: *Self) WorkerBackend {
        return WorkerBackend{
            .ptr = self,
            .vtable = &.{
                .executeTrainingStep = executeTrainingStepInterface,
                .executeFunction = executeFunctionInterface,
                .getBackendType = getBackendTypeInterface,
                .deinit = deinitInterface,
            },
        };
    }

    fn executeFunctionInterface(ptr: *anyopaque, artifact: []const u8, function_name: []const u8, data: [][]const u8, shapes: [][]const i64, dtypes: ?[]const DType) anyerror![][]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.execute(artifact, function_name, data, shapes, dtypes);
    }

    fn executeTrainingStepInterface(ptr: *anyopaque, artifact: []const u8, data: [][]const u8, shapes: [][]const i64, provided_dtypes: ?[]const DType) anyerror![][]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));

        // If explicit dtypes are provided (e.g. for RL/Generation), use them.
        if (provided_dtypes) |dtypes| {
            return self.execute(artifact, "main", data, shapes, dtypes);
        }

        // Fallback logic for DiLoCo/AdamW legacy training if dtypes is null:
        // [Params (N x f32), M_states (N x f32), V_states (N x f32), Timestep (1 x f32), Data (2 x i64)]
        var inferred_dtypes = try self.allocator.alloc(DType, data.len);
        defer self.allocator.free(inferred_dtypes);

        // All inputs except last 2 are f32 (params, M states, V states, timestep)
        for (0..data.len) |i| {
            if (i >= data.len - 2) {
                inferred_dtypes[i] = .i64; // Last 2 are i64 data inputs
            } else {
                inferred_dtypes[i] = .f32; // Everything else is f32
            }
        }

        return self.execute(artifact, "main", data, shapes, inferred_dtypes);
    }

    fn getBackendTypeInterface(ptr: *anyopaque) backend_selection.Backend {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.backend;
    }

    fn deinitInterface(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
};

test "visible device parser counts CUDA and ROCm device lists" {
    try std.testing.expectEqual(@as(usize, 0), countVisibleDevices(""));
    try std.testing.expectEqual(@as(usize, 0), countVisibleDevices("   "));
    try std.testing.expectEqual(@as(usize, 0), countVisibleDevices("-1"));
    try std.testing.expectEqual(@as(usize, 1), countVisibleDevices("0"));
    try std.testing.expectEqual(@as(usize, 1), countVisibleDevices("3"));
    try std.testing.expectEqual(@as(usize, 2), countVisibleDevices("0,1"));
    try std.testing.expectEqual(@as(usize, 2), countVisibleDevices(" 2, 5 "));
    try std.testing.expectEqual(@as(usize, 2), countVisibleDevices("GPU-a,GPU-b"));
    try std.testing.expectEqual(@as(usize, 2), countVisibleDevices("MIG-a, MIG-b"));
    try std.testing.expectEqual(@as(usize, 2), countVisibleDevices("0,,2"));
}
