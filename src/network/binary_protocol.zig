/// Binary Protocol for robust serialization between Shepherd and Workers
/// Replaces fragile string delimiters with a schema-based binary format

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Worker payload from Shepherd to Worker (start inner loop)
pub const WorkerPayload = struct {
    params: []const u8,
    
    const Self = @This();
    
    /// Serialize to binary format
    pub fn serialize(self: Self, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();
        
        // Write params length (u64)
        const params_len: u64 = self.params.len;
        try buffer.appendSlice(std.mem.asBytes(&params_len));
        
        // Write params data
        try buffer.appendSlice(self.params);
        
        return try allocator.dupe(u8, buffer.items);
    }
    
    /// Deserialize from binary format
    pub fn deserialize(allocator: Allocator, data: []const u8) !Self {
        if (data.len < @sizeOf(u64)) {
            return error.InvalidData;
        }
        
        // Read params length
        const params_len = std.mem.readInt(u64, data[0..@sizeOf(u64)], .little);
        const params_start = @sizeOf(u64);
        
        if (data.len < params_start + params_len) {
            return error.InvalidData;
        }
        
        // Extract params data
        const params = try allocator.dupe(u8, data[params_start..params_start + params_len]);
        
        return Self{
            .params = params,
        };
    }
    
    pub fn deinit(self: Self, allocator: Allocator) void {
        allocator.free(self.params);
    }
};

/// Shepherd payload from Worker to Shepherd (inner loop complete)
pub const ShepherdPayload = struct {
    updated_params: []const u8,
    loss: f32,
    
    const Self = @This();
    
    /// Serialize to binary format
    pub fn serialize(self: Self, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();
        
        // Write updated_params length (u64)
        const params_len: u64 = self.updated_params.len;
        try buffer.appendSlice(std.mem.asBytes(&params_len));
        
        // Write updated_params data
        try buffer.appendSlice(self.updated_params);
        
        // Write loss (f32)
        try buffer.appendSlice(std.mem.asBytes(&self.loss));
        
        return try allocator.dupe(u8, buffer.items);
    }
    
    /// Deserialize from binary format
    pub fn deserialize(allocator: Allocator, data: []const u8) !Self {
        if (data.len < @sizeOf(u64) + @sizeOf(f32)) {
            return error.InvalidData;
        }
        
        // Read updated_params length
        const params_len = std.mem.readInt(u64, data[0..@sizeOf(u64)], .little);
        const params_start = @sizeOf(u64);
        const params_end = params_start + params_len;
        
        if (data.len < params_end + @sizeOf(f32)) {
            return error.InvalidData;
        }
        
        // Extract updated_params data
        const updated_params = try allocator.dupe(u8, data[params_start..params_end]);
        
        // Read loss
        const loss_bytes = data[params_end..params_end + @sizeOf(f32)];
        const loss = @as(f32, @bitCast(std.mem.readInt(u32, loss_bytes[0..4], .little)));
        
        return Self{
            .updated_params = updated_params,
            .loss = loss,
        };
    }
    
    pub fn deinit(self: Self, allocator: Allocator) void {
        allocator.free(self.updated_params);
    }
};

/// Test the binary protocol serialization/deserialization
pub fn testBinaryProtocol(allocator: Allocator) !void {
    std.log.info("Testing binary protocol serialization...");
    
    // Test WorkerPayload
    {
        const test_params = "test parameter data";
        const worker_payload = WorkerPayload{
            .params = test_params,
        };
        
        // Serialize
        const serialized = try worker_payload.serialize(allocator);
        defer allocator.free(serialized);
        
        // Deserialize
        const deserialized = try WorkerPayload.deserialize(allocator, serialized);
        defer deserialized.deinit(allocator);
        
        // Verify
        try std.testing.expectEqualStrings(test_params, deserialized.params);
        std.log.info("✓ WorkerPayload serialization test passed");
    }
    
    // Test ShepherdPayload
    {
        const test_params = "updated parameter data";
        const test_loss: f32 = 0.1234;
        const shepherd_payload = ShepherdPayload{
            .updated_params = test_params,
            .loss = test_loss,
        };
        
        // Serialize
        const serialized = try shepherd_payload.serialize(allocator);
        defer allocator.free(serialized);
        
        // Deserialize
        const deserialized = try ShepherdPayload.deserialize(allocator, serialized);
        defer deserialized.deinit(allocator);
        
        // Verify
        try std.testing.expectEqualStrings(test_params, deserialized.updated_params);
        try std.testing.expectApproxEqAbs(test_loss, deserialized.loss, 0.0001);
        std.log.info("✓ ShepherdPayload serialization test passed");
    }
    
    std.log.info("✓ Binary protocol test completed");
}