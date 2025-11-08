/// IREE Backend implementation for WorkerBackend interface
/// This backend uses the IREE runtime to execute VMFB modules on various hardware accelerators

const std = @import("std");
const c = @cImport({
    @cInclude("iree/runtime/api.h");
});
const WorkerBackend = @import("worker_backend.zig").WorkerBackend;
const backend_selection = @import("../backend_selection.zig");
const mlir = @import("../mlir.zig");

// Helper to check for IREE errors
fn ireeCheck(status: c.iree_status_t) !void {
    if (!c.iree_status_is_ok(status)) {
        // In a real app, you would format the error string from the status
        c.iree_status_free(status);
        return error.IreeRuntimeError;
    }
}

pub const IreeBackend = struct {
    allocator: std.mem.Allocator,
    instance: *c.iree_runtime_instance_t,
    device: *c.iree_hal_device_t,
    session: *c.iree_runtime_session_t,
    backend: backend_selection.Backend,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, backend: backend_selection.Backend) !*IreeBackend {
        var self = try allocator.create(IreeBackend);
        
        // 1. Create instance
        var instance_options: c.iree_runtime_instance_options_t = undefined;
        c.iree_runtime_instance_options_initialize(&instance_options);
        c.iree_runtime_instance_options_use_all_available_drivers(&instance_options);
        try ireeCheck(c.iree_runtime_instance_create(
            &instance_options,
            c.iree_allocator_system(),
            &self.instance,
        ));

        // 2. Create the appropriate HAL driver
        const driver_name = backend.toIreeDriverName();
        var driver: *c.iree_hal_driver_t = undefined;
        try ireeCheck(c.iree_runtime_instance_try_create_driver(
            self.instance,
            c.iree_string_view_t{ .data = driver_name.ptr, .size = driver_name.len },
            &driver,
        ));

        // 3. Create the HAL device
        try ireeCheck(c.iree_hal_driver_create_default_device(driver, c.iree_allocator_system(), &self.device));
        c.iree_hal_driver_release(driver);

        // 4. Create the runtime session
        var session_options: c.iree_runtime_session_options_t = undefined;
        c.iree_runtime_session_options_initialize(&session_options);
        try ireeCheck(c.iree_runtime_session_create_with_device(
            self.instance,
            &session_options,
            self.device,
            c.iree_runtime_instance_host_allocator(self.instance),
            &self.session,
        ));
        
        self.allocator = allocator;
        self.backend = backend;
        return self;
    }

    pub fn deinit(self: *Self) void {
        c.iree_runtime_session_release(self.session);
        c.iree_hal_device_release(self.device);
        c.iree_runtime_instance_release(self.instance);
        self.allocator.destroy(self);
    }

    /// Execute a training step with pre-compiled VMFB bytecode
    /// This is the real implementation that will be called by the Worker
    pub fn execute(self: *Self, vmfb_bytes: []const u8, inputs: [][]const u8) ![][]u8 {
        // 1. Load the .vmfb module into the session
        // NOTE: In a real app, you would load this once and cache it. For now, we load it on every call.
        var policy: c.iree_runtime_module_policy_t = undefined;
        try ireeCheck(c.iree_runtime_session_append_bytecode_module(
            self.session,
            .{ .data = vmfb_bytes.ptr, .data_length = vmfb_bytes.len },
            &policy,
        ));

        // 2. Initialize a call to the main function
        var call: c.iree_runtime_call_t = undefined;
        const main_fn_name = c.iree_string_view_t{ .data = "main", .size = 4 };
        try ireeCheck(c.iree_runtime_call_initialize_by_name(self.session, main_fn_name, &call));
        defer c.iree_runtime_call_deinitialize(&call);

        // 3. Wrap Zig input slices into IREE buffer views and push them to the call
        for (inputs) |input_slice| {
            // NOTE: This is a simplified implementation. A real implementation needs to get the
            // expected shape/dtype from the function signature. For now, we assume f32 tensors.
            
            // Create a host buffer that wraps our input data
            var host_buffer: *c.iree_hal_buffer_t = undefined;
            const buffer_size = input_slice.len;
            
            try ireeCheck(c.iree_hal_allocator_allocate_buffer(
                c.iree_runtime_session_device_allocator(self.session),
                c.IREE_HAL_MEMORY_TYPE_HOST_LOCAL | c.IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                c.IREE_HAL_BUFFER_USAGE_ALL,
                @intCast(buffer_size),
                &host_buffer,
            ));
            defer c.iree_hal_buffer_release(host_buffer);

            // Map and copy data
            var mapped_memory: c.iree_hal_buffer_mapping_t = undefined;
            try ireeCheck(c.iree_hal_buffer_map_range(
                host_buffer,
                c.IREE_HAL_MAPPING_MODE_SCOPED,
                c.IREE_HAL_MEMORY_ACCESS_WRITE,
                0,
                c.IREE_WHOLE_BUFFER,
                &mapped_memory,
            ));
            @memcpy(mapped_memory.contents.data[0..buffer_size], input_slice);
            c.iree_hal_buffer_unmap_range(&mapped_memory);

            // Create buffer view - assuming 1D f32 tensor for simplicity
            var buffer_view: *c.iree_hal_buffer_view_t = undefined;
            const shape: [1]c.iree_hal_dim_t = .{@intCast(input_slice.len / 4)}; // Assuming f32
            try ireeCheck(c.iree_hal_buffer_view_create(
                host_buffer,
                shape.ptr,
                1, // rank
                c.IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                c.IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                c.iree_runtime_session_host_allocator(self.session),
                &buffer_view,
            ));
            
            try ireeCheck(c.iree_runtime_call_inputs_push_back_buffer_view(&call, buffer_view));
            c.iree_hal_buffer_view_release(buffer_view);
        }
        
        // 4. Invoke the function
        try ireeCheck(c.iree_runtime_session_call(self.session, &call));

        // 5. Get the outputs
        var outputs_list = std.ArrayList([]u8).init(self.allocator);
        
        const output_count = c.iree_runtime_call_outputs_size(&call);
        for (0..@intCast(output_count)) |_| {
            var output_buffer_view: *c.iree_hal_buffer_view_t = undefined;
            try ireeCheck(c.iree_runtime_call_outputs_pop_front_buffer_view(&call, &output_buffer_view));
            defer c.iree_hal_buffer_view_release(output_buffer_view);

            // Get buffer from buffer view
            const output_buffer = c.iree_hal_buffer_view_buffer(output_buffer_view);
            const buffer_byte_length = c.iree_hal_buffer_byte_length(output_buffer);

            // Map and copy output data
            var output_mapped_memory: c.iree_hal_buffer_mapping_t = undefined;
            try ireeCheck(c.iree_hal_buffer_map_range(
                output_buffer,
                c.IREE_HAL_MAPPING_MODE_SCOPED,
                c.IREE_HAL_MEMORY_ACCESS_READ,
                0,
                c.IREE_WHOLE_BUFFER,
                &output_mapped_memory,
            ));
            
            const output_data = try self.allocator.alloc(u8, @intCast(buffer_byte_length));
            @memcpy(output_data, output_mapped_memory.contents.data[0..@intCast(buffer_byte_length)]);
            c.iree_hal_buffer_unmap_range(&output_mapped_memory);
            
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

    // Interface implementations
    fn executeTrainingStepInterface(ptr: *anyopaque, artifact: []const u8, inputs: [][]const u8) anyerror![][]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.execute(artifact, inputs);
    }

    fn deinitInterface(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
};