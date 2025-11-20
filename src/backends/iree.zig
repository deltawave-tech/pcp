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
const backend_selection = @import("../backend_selection.zig");
const mlir = @import("../mlir.zig");
const pcp = @import("../main.zig");
const DType = pcp.tensor.DType;

// Helper to check for IREE errors
fn ireeCheck(status: c.iree_status_t) !void {
    // In IREE, null status indicates success, non-null indicates error
    if (status != null) {
        // Try to get more detailed error information
        std.debug.print("IREE Error Details:\n", .{});

        // Get error code if available
        const code = c.iree_status_code(status);
        std.debug.print("  Status code: {}\n", .{code});

        // Try to get error message
        var buffer: [1024]u8 = undefined;
        var out_length: c.iree_host_size_t = 0;
        _ = c.iree_status_format(status, buffer.len, &buffer, &out_length);
        if (out_length > 0 and out_length < buffer.len) {
            const msg_len: usize = @intCast(out_length);
            std.debug.print("  Error message: {s}\n", .{buffer[0..msg_len]});
        }

        c.iree_status_free(status);
        return error.IreeRuntimeError;
    }
}

pub const IreeBackend = struct {
    allocator: std.mem.Allocator,
    instance: ?*c.iree_runtime_instance_t,
    device: ?*c.iree_hal_device_t,
    session: ?*c.iree_runtime_session_t,
    backend: backend_selection.Backend,

    const Self = @This();

    // Helper to map DType to IREE element type
    fn getIreeType(dtype: DType) c.iree_hal_element_type_t {
        return switch (dtype) {
            .f32 => c.IREE_HAL_ELEMENT_TYPE_FLOAT_32,
            .f64 => c.IREE_HAL_ELEMENT_TYPE_FLOAT_64,
            .f16 => c.IREE_HAL_ELEMENT_TYPE_FLOAT_16,
            .i32 => c.IREE_HAL_ELEMENT_TYPE_SINT_32,
            .i64 => c.IREE_HAL_ELEMENT_TYPE_SINT_64,
            .bool => c.IREE_HAL_ELEMENT_TYPE_BOOL_8, // IREE treats i1 as 8-bit bool
        };
    }

    pub fn init(allocator: std.mem.Allocator, backend: backend_selection.Backend) !*IreeBackend {
        var self = try allocator.create(IreeBackend);
        
        // FIX 2: Add errdefer to prevent memory leak on initialization failure.
        errdefer allocator.destroy(self);

        // 1. Create instance
        var instance_options: c.iree_runtime_instance_options_t = undefined;
        c.iree_runtime_instance_options_initialize(&instance_options);
        c.iree_runtime_instance_options_use_all_available_drivers(&instance_options);

        // Use system allocator like working IREE samples
        try ireeCheck(c.iree_runtime_instance_create(
            &instance_options,
            c.iree_allocator_system(),
            &self.instance,
        ));

        // 2. Create the HAL device directly using the instance
        const driver_name = backend.toIreeDriverName();
        try ireeCheck(c.iree_runtime_instance_try_create_default_device(
            self.instance.?,
            c.iree_string_view_t{ .data = driver_name.ptr, .size = driver_name.len },
            &self.device,
        ));

        // 4. Create the runtime session
        var session_options: c.iree_runtime_session_options_t = undefined;
        c.iree_runtime_session_options_initialize(&session_options);
        try ireeCheck(c.iree_runtime_session_create_with_device(
            self.instance.?,
            &session_options,
            self.device.?,
            c.iree_runtime_instance_host_allocator(self.instance.?),
            &self.session,
        ));
        
        self.allocator = allocator;
        self.backend = backend;
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.session) |session| c.iree_runtime_session_release(session);
        if (self.device) |device| c.iree_hal_device_release(device);
        if (self.instance) |instance| c.iree_runtime_instance_release(instance);
        self.allocator.destroy(self);
    }

    /// Execute a training step with a pre-compiled VMFB artifact.
    pub fn execute(
        self: *Self,
        vmfb_bytes: []const u8,
        function_name: []const u8,
        inputs_data: [][]const u8,
        input_shapes: [][]const i64,
        input_dtypes: ?[]const DType
    ) ![][]u8 {
        // 1. Load the .vmfb module into the session
        // NOTE: In a real app, you would load this once and cache it. For now, we load it on every call.
        try ireeCheck(c.iree_runtime_session_append_bytecode_module_from_memory(
            self.session.?,
            .{ .data = vmfb_bytes.ptr, .data_length = vmfb_bytes.len },
            c.iree_allocator_null(), // We don't own the flatbuffer data
        ));

        // 2. Initialize a call to the specified function
        var call: c.iree_runtime_call_t = undefined;

        // IREE expects the format "module.<function_name>"
        const full_fn_name = try std.fmt.allocPrint(self.allocator, "module.{s}", .{function_name});
        defer self.allocator.free(full_fn_name);

        const fn_name_view = c.iree_string_view_t{ .data = full_fn_name.ptr, .size = full_fn_name.len };
        try ireeCheck(c.iree_runtime_call_initialize_by_name(self.session.?, fn_name_view, &call));
        defer c.iree_runtime_call_deinitialize(&call);

        // 3. Create IREE buffer views from input data using the simpler allocate_buffer_copy API
        std.debug.assert(inputs_data.len == input_shapes.len);

        for (inputs_data, input_shapes, 0..) |input_slice, shape_slice, i| {
            // Convert shapes to IREE format
            var iree_shape = try self.allocator.alloc(c.iree_hal_dim_t, shape_slice.len);
            defer self.allocator.free(iree_shape);
            for (shape_slice, 0..) |dim, k| {
                iree_shape[k] = @intCast(dim);
            }

            // Determine element type: Use provided dtype or default to f32
            const element_type = if (input_dtypes) |dtypes| getIreeType(dtypes[i]) else c.IREE_HAL_ELEMENT_TYPE_FLOAT_32;

            // Create buffer view directly from data - much simpler than manual mapping!
            var buffer_view: ?*c.iree_hal_buffer_view_t = null;
            try ireeCheck(c.iree_hal_buffer_view_allocate_buffer_copy(
                c.iree_runtime_session_device(self.session.?),
                c.iree_runtime_session_device_allocator(self.session.?),
                @intCast(shape_slice.len),
                iree_shape.ptr,
                element_type, // Use dynamic type
                c.IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                .{
                    .type = c.IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                    .access = c.IREE_HAL_MEMORY_ACCESS_ALL,
                    .usage = c.IREE_HAL_BUFFER_USAGE_DEFAULT,
                    .queue_affinity = 0,
                    .min_alignment = 0,
                },
                .{ .data = input_slice.ptr, .data_length = input_slice.len },
                @ptrCast(&buffer_view),
            ));
            defer c.iree_hal_buffer_view_release(buffer_view.?);
            
            try ireeCheck(c.iree_runtime_call_inputs_push_back_buffer_view(&call, buffer_view.?));
        }
        
        // 4. Invoke the function
        try ireeCheck(c.iree_runtime_call_invoke(&call, 0)); // flags = 0

        // 5. Get the outputs from the call using VM list API
        var outputs_list = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (outputs_list.items) |o| self.allocator.free(o);
            outputs_list.deinit();
        }
        
        const outputs = c.iree_runtime_call_outputs(&call);
        const output_count = c.iree_vm_list_size(outputs);
        var i: c.iree_host_size_t = 0;
        while (i < output_count) : (i += 1) {
            var output_buffer_view: ?*c.iree_hal_buffer_view_t = null;
            try ireeCheck(c.iree_runtime_call_outputs_pop_front_buffer_view(&call, @ptrCast(&output_buffer_view)));
            defer c.iree_hal_buffer_view_release(output_buffer_view.?);

            // Get the buffer from the view and copy its data
            const output_buffer = c.iree_hal_buffer_view_buffer(output_buffer_view.?);
            const buffer_byte_length = c.iree_hal_buffer_byte_length(output_buffer);
            
            // Allocate output data and read from buffer
            const output_data = try self.allocator.alloc(u8, @intCast(buffer_byte_length));
            try ireeCheck(c.iree_hal_device_transfer_d2h(
                c.iree_runtime_session_device(self.session.?),
                output_buffer,
                0, // source_offset
                output_data.ptr, // target_buffer
                @intCast(buffer_byte_length),
                c.IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                c.iree_infinite_timeout(),
            ));
            
            try outputs_list.append(output_data);
        }

        return outputs_list.toOwnedSlice();
    }


    /// Create WorkerBackend interface from this IREE backend
    pub fn asWorkerBackend(self: *Self) WorkerBackend {
        return WorkerBackend{
            .ptr = self,
            .vtable = &.{
                .executeTrainingStep = executeTrainingStepInterface,
                .deinit = deinitInterface,
            },
        };
    }

    fn executeTrainingStepInterface(ptr: *anyopaque, artifact: []const u8, data: [][]const u8, shapes: [][]const i64) anyerror![][]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        // Note: In the future, the worker protocol should include type info.
        // For now, assuming f32 defaults in the interface is consistent with previous behavior.
        return self.execute(artifact, "main", data, shapes, null);
    }

    fn deinitInterface(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
};