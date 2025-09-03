const std = @import("std");
const Allocator = std.mem.Allocator;

// Import the C header
const c = @cImport({
    @cInclude("capnp_bridge.h");
});

pub const WorkerPayload = struct {
    params: []const u8,
    input_ids: []const u8,
    targets: []const u8,

    pub fn serialize(self: @This(), allocator: Allocator) ![]u8 {
        const builder = c.new_worker_payload_builder();
        defer c.free_builder(builder);
        c.set_worker_payload_params(builder, self.params.ptr, self.params.len);
        c.set_worker_payload_input_ids(builder, self.input_ids.ptr, self.input_ids.len);
        c.set_worker_payload_targets(builder, self.targets.ptr, self.targets.len);

        const size = c.get_message_size(builder);
        const buffer = try allocator.alloc(u8, size);
        const bytes_written = c.message_to_bytes(builder, buffer.ptr, buffer.len);
        if (bytes_written == 0) return error.SerializationFailed;
        return buffer;
    }

    // Reader for deserialization
    pub const Reader = struct {
        reader_ptr: *c.CapnpMessageReader,
        
        pub fn init(data: []const u8) !Reader {
            const reader_ptr = c.new_message_reader(data.ptr, data.len) orelse return error.FailedToCreateReader;
            return Reader{ .reader_ptr = reader_ptr };
        }

        pub fn deinit(self: Reader) void {
            c.free_reader(self.reader_ptr);
        }

        pub fn getParams(self: Reader) ![]const u8 {
            var data_ptr: [*c]const u8 = undefined;
            var size: usize = undefined;
            if (c.get_worker_payload_params(self.reader_ptr, &data_ptr, &size) == 0) {
                return error.DeserializationFailed;
            }
            return data_ptr[0..size];
        }

        pub fn getInputIds(self: Reader) ![]const u8 {
            var data_ptr: [*c]const u8 = undefined;
            var size: usize = undefined;
            if (c.get_worker_payload_input_ids(self.reader_ptr, &data_ptr, &size) == 0) {
                return error.DeserializationFailed;
            }
            return data_ptr[0..size];
        }

        pub fn getTargets(self: Reader) ![]const u8 {
            var data_ptr: [*c]const u8 = undefined;
            var size: usize = undefined;
            if (c.get_worker_payload_targets(self.reader_ptr, &data_ptr, &size) == 0) {
                return error.DeserializationFailed;
            }
            return data_ptr[0..size];
        }
    };
};

pub const ShepherdPayload = struct {
    updated_params: []const u8,
    loss: f32,

    pub fn serialize(self: @This(), allocator: Allocator) ![]u8 {
        const builder = c.new_shepherd_payload_builder();
        defer c.free_builder(builder);
        c.set_shepherd_payload_params(builder, self.updated_params.ptr, self.updated_params.len);
        c.set_shepherd_payload_loss(builder, self.loss);

        const size = c.get_message_size(builder);
        const buffer = try allocator.alloc(u8, size);
        const bytes_written = c.message_to_bytes(builder, buffer.ptr, buffer.len);
        if (bytes_written == 0) return error.SerializationFailed;
        return buffer;
    }

    // Reader for deserialization
    pub const Reader = struct {
        reader_ptr: *c.CapnpMessageReader,
        
        pub fn init(data: []const u8) !Reader {
            const reader_ptr = c.new_message_reader(data.ptr, data.len) orelse return error.FailedToCreateReader;
            return Reader{ .reader_ptr = reader_ptr };
        }

        pub fn deinit(self: Reader) void {
            c.free_reader(self.reader_ptr);
        }

        pub fn getUpdatedParams(self: Reader) ![]const u8 {
            var data_ptr: [*c]const u8 = undefined;
            var size: usize = undefined;
            if (c.get_shepherd_payload_params(self.reader_ptr, &data_ptr, &size) == 0) {
                return error.DeserializationFailed;
            }
            return data_ptr[0..size];
        }

        pub fn getLoss(self: Reader) !f32 {
            var loss: f32 = undefined;
            if (c.get_shepherd_payload_loss(self.reader_ptr, &loss) == 0) {
                return error.DeserializationFailed;
            }
            return loss;
        }
    };
};