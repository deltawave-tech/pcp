const std = @import("std");
const tensor = @import("../tensor.zig");
const builtin = @import("builtin");

const Allocator = std.mem.Allocator;
const DataType = f32;
const Tensor = tensor.Tensor(DataType);
const DType = tensor.DType;
const Shape = tensor.Shape;
const BackendType = tensor.BackendType;

pub const MetalError = error{
    MetalInitFailed,
    DeviceNotFound,
    CommandQueueCreationFailed,
    LibraryCreationFailed,
    FunctionCreationFailed,
    PipelineCreationFailed,
    BufferCreationFailed,
    CommandBufferCreationFailed,
    EncodingFailed,
    UnsupportedPlatform,
    ThreadgroupSizeComputationFailed,
};

// C interface to Metal Framework
const MTLDevice = opaque {};
const MTLCommandQueue = opaque {};
const MTLLibrary = opaque {};
const MTLFunction = opaque {};
const MTLComputePipelineState = opaque {};
const MTLBuffer = opaque {};
const MTLCommandBuffer = opaque {};
const MTLComputeCommandEncoder = opaque {};
const NSError = opaque {};
const NSString = opaque {};
const NSURL = opaque {};

// Import C to verify our function signatures are correct
const c = @cImport({
    @cInclude("stdint.h");
});

// Add explicit alignment to ensure struct layout matches ObjC
extern fn MTLCreateSystemDefaultDevice() ?*MTLDevice;
extern fn MTLDevice_newCommandQueue(device: *MTLDevice) ?*MTLCommandQueue;
extern fn MTLDevice_newLibraryWithSource(device: *MTLDevice, source: *NSString, options: ?*anyopaque, err: *?*NSError) ?*MTLLibrary;
extern fn MTLLibrary_newFunctionWithName(library: *MTLLibrary, name: *NSString) ?*MTLFunction;
extern fn MTLDevice_newComputePipelineStateWithFunction(device: *MTLDevice, function: *MTLFunction, err: *?*NSError) ?*MTLComputePipelineState;
extern fn MTLDevice_newBufferWithBytes(device: *MTLDevice, bytes: *const anyopaque, length: usize, options: u64) ?*MTLBuffer;
extern fn MTLDevice_newBufferWithLength(device: *MTLDevice, length: usize, options: u64) ?*MTLBuffer;
extern fn MTLBuffer_contents(buffer: *MTLBuffer) *anyopaque;
extern fn MTLCommandQueue_commandBuffer(queue: *MTLCommandQueue) ?*MTLCommandBuffer;
extern fn MTLCommandBuffer_computeCommandEncoder(buffer: *MTLCommandBuffer) ?*MTLComputeCommandEncoder;
extern fn MTLComputeCommandEncoder_setComputePipelineState(encoder: *MTLComputeCommandEncoder, state: *MTLComputePipelineState) void;
extern fn MTLComputeCommandEncoder_setBuffer(encoder: *MTLComputeCommandEncoder, buffer: *MTLBuffer, offset: usize, index: usize) void;
extern fn MTLComputePipelineState_maxTotalThreadsPerThreadgroup(state: *MTLComputePipelineState) usize;
extern fn MTLComputePipelineState_threadExecutionWidth(state: *MTLComputePipelineState) usize;
// MTLSize struct that matches the Objective-C struct
const MTLSize = extern struct {
    width: c_ulong,
    height: c_ulong,
    depth: c_ulong,
};

extern fn MTLComputeCommandEncoder_dispatchThreads(encoder: *MTLComputeCommandEncoder, threads: MTLSize, threadsPerThreadgroup: MTLSize) void;

// Helper function to dispatch threads using our array format to MTLSize
fn dispatchThreads(encoder: *MTLComputeCommandEncoder, threads: [3]usize, threadsPerThreadgroup: [3]usize) void {
    const threads_mtl = MTLSize{
        .width = threads[0],
        .height = threads[1],
        .depth = threads[2],
    };

    const threads_per_group_mtl = MTLSize{
        .width = threadsPerThreadgroup[0],
        .height = threadsPerThreadgroup[1],
        .depth = threadsPerThreadgroup[2],
    };

    MTLComputeCommandEncoder_dispatchThreads(encoder, threads_mtl, threads_per_group_mtl);
}
extern fn MTLComputeCommandEncoder_endEncoding(encoder: *MTLComputeCommandEncoder) void;
extern fn MTLCommandBuffer_commit(buffer: *MTLCommandBuffer) void;
extern fn MTLCommandBuffer_waitUntilCompleted(buffer: *MTLCommandBuffer) void;

extern fn NSString_alloc() *NSString;
extern fn NSString_initWithUTF8String(self: *NSString, cString: [*:0]const u8) *NSString;
extern fn NSString_release(self: *NSString) void;

// Metal context that holds the device and command queue
pub const MetalContext = struct {
    device: ?*MTLDevice = null,
    command_queue: ?*MTLCommandQueue = null,
    library: ?*MTLLibrary = null,
    allocator: Allocator,

    // Compiled compute pipelines for different operations
    add_pipeline: ?*MTLComputePipelineState = null,
    subtract_pipeline: ?*MTLComputePipelineState = null,
    multiply_pipeline: ?*MTLComputePipelineState = null,
    relu_pipeline: ?*MTLComputePipelineState = null,
    matmul_pipeline: ?*MTLComputePipelineState = null,

    pub fn init(allocator: Allocator) !MetalContext {
        // Check if running on Apple platform
        if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
            return MetalError.UnsupportedPlatform;
        }

        // Get the default Metal device
        const device = MTLCreateSystemDefaultDevice() orelse {
            return MetalError.DeviceNotFound;
        };

        // Create a command queue
        const command_queue = MTLDevice_newCommandQueue(device) orelse {
            return MetalError.CommandQueueCreationFailed;
        };

        // Create a Metal context
        var ctx = MetalContext{
            .device = device,
            .command_queue = command_queue,
            .allocator = allocator,
        };

        // Compile Metal kernels
        try compileMetalCode(&ctx, metal_kernels);

        return ctx;
    }

    pub fn deinit(self: *MetalContext) void {
        // In a real implementation, would need to release Metal resources
        // For now, we just set them to null to avoid accidental usage
        self.device = null;
        self.command_queue = null;
        self.library = null;
        self.add_pipeline = null;
        self.subtract_pipeline = null;
        self.multiply_pipeline = null;
        self.relu_pipeline = null;
        self.matmul_pipeline = null;
    }
};

// Global Metal context
var global_metal_context: ?MetalContext = null;

// Initialize the Metal backend
pub fn init(allocator: Allocator) !void {
    if (global_metal_context != null) return;

    std.debug.print("Metal backend: Starting initialization...\n", .{});

    // Check if we're on an Apple platform
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
        std.debug.print("Metal backend: Unsupported platform! Metal requires macOS or iOS.\n", .{});
        return MetalError.UnsupportedPlatform;
    }

    std.debug.print("Metal backend: Getting default Metal device...\n", .{});
    const device = MTLCreateSystemDefaultDevice();
    if (device == null) {
        std.debug.print("Metal backend: No Metal device found!\n", .{});
        return MetalError.DeviceNotFound;
    }
    std.debug.print("Metal backend: Metal device found.\n", .{});

    std.debug.print("Metal backend: Creating command queue...\n", .{});
    const command_queue = MTLDevice_newCommandQueue(device.?) orelse {
        std.debug.print("Metal backend: Failed to create command queue!\n", .{});
        return MetalError.CommandQueueCreationFailed;
    };
    std.debug.print("Metal backend: Command queue created.\n", .{});

    // Create a simple Metal context without compiling kernels
    // This will allow us to test without the compilation step
    global_metal_context = MetalContext{
        .device = device,
        .command_queue = command_queue,
        .allocator = allocator,
    };

    std.debug.print("Metal backend: Basic initialization completed successfully.\n", .{});
    std.debug.print("Metal backend: Skipping kernel compilation for now.\n", .{});
}

// Clean up the Metal backend
pub fn deinit() void {
    if (global_metal_context) |*ctx| {
        ctx.deinit();
        global_metal_context = null;
    }
}

// Get the global Metal context
pub fn getContext() !*MetalContext {
    if (global_metal_context) |*ctx| {
        return ctx;
    }

    return MetalError.MetalInitFailed;
}

// Metal kernel strings for various operations
const metal_kernels =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void add_f32(device const float* a [[buffer(0)]],
    \\                    device const float* b [[buffer(1)]],
    \\                    device float* result [[buffer(2)]],
    \\                    uint id [[thread_position_in_grid]]) {
    \\    result[id] = a[id] + b[id];
    \\}
    \\
    \\kernel void subtract_f32(device const float* a [[buffer(0)]],
    \\                        device const float* b [[buffer(1)]],
    \\                        device float* result [[buffer(2)]],
    \\                        uint id [[thread_position_in_grid]]) {
    \\    result[id] = a[id] - b[id];
    \\}
    \\
    \\kernel void multiply_f32(device const float* a [[buffer(0)]],
    \\                        device const float* b [[buffer(1)]],
    \\                        device float* result [[buffer(2)]],
    \\                        uint id [[thread_position_in_grid]]) {
    \\    result[id] = a[id] * b[id];
    \\}
    \\
    \\kernel void divide_f32(device const float* a [[buffer(0)]],
    \\                       device const float* b [[buffer(1)]],
    \\                       device float* result [[buffer(2)]],
    \\                       uint id [[thread_position_in_grid]]) {
    \\    result[id] = a[id] / b[id];
    \\}
    \\
    \\kernel void relu_f32(device const float* input [[buffer(0)]],
    \\                     device float* result [[buffer(1)]],
    \\                     uint id [[thread_position_in_grid]]) {
    \\    result[id] = max(0.0f, input[id]);
    \\}
    \\
    \\kernel void softmax_f32(device const float* input [[buffer(0)]],
    \\                       device float* result [[buffer(1)]],
    \\                       device const uint* dims [[buffer(2)]],
    \\                       uint id [[thread_position_in_grid]]) {
    \\    // dims[0] = batch_size, dims[1] = feature_size
    \\    const uint batch_size = dims[0];
    \\    const uint feature_size = dims[1];
    \\    
    \\    // Calculate which batch this thread is processing
    \\    const uint batch_idx = id / feature_size;
    \\    const uint feature_idx = id % feature_size;
    \\    
    \\    // Skip out-of-bounds work
    \\    if (batch_idx >= batch_size) return;
    \\    
    \\    // Find max value in this row for numerical stability
    \\    float max_val = -INFINITY;
    \\    for (uint i = 0; i < feature_size; i++) {
    \\        const float val = input[batch_idx * feature_size + i];
    \\        max_val = max(max_val, val);
    \\    }
    \\    
    \\    // Compute exp(x - max) and sum
    \\    float sum = 0.0f;
    \\    for (uint i = 0; i < feature_size; i++) {
    \\        const float val = exp(input[batch_idx * feature_size + i] - max_val);
    \\        result[batch_idx * feature_size + i] = val; // Store intermediate result
    \\        sum += val;
    \\    }
    \\    
    \\    // Normalize by sum
    \\    result[batch_idx * feature_size + feature_idx] /= sum;
    \\}
    \\
    \\// Matrix multiplication for 2D matrices
    \\// Simplified version - in a real implementation, this would use shared memory and tiling
    \\kernel void matmul_f32(device const float* a [[buffer(0)]],
    \\                      device const float* b [[buffer(1)]],
    \\                      device float* result [[buffer(2)]],
    \\                      device const uint* dims [[buffer(3)]],
    \\                      uint2 gid [[thread_position_in_grid]]) {
    \\    // dims = [M, N, P] where A is MxN and B is NxP
    \\    const uint M = dims[0];
    \\    const uint N = dims[1];
    \\    const uint P = dims[2];
    \\    
    \\    // Skip out-of-bounds work
    \\    if (gid.x >= P || gid.y >= M) return;
    \\    
    \\    // Compute the matrix multiplication for this element
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < N; k++) {
    \\        sum += a[gid.y * N + k] * b[k * P + gid.x];
    \\    }
    \\    
    \\    result[gid.y * P + gid.x] = sum;
    \\}
    \\
    \\// Transpose a 2D matrix
    \\kernel void transpose_f32(device const float* input [[buffer(0)]],
    \\                         device float* result [[buffer(1)]],
    \\                         device const uint* dims [[buffer(2)]],
    \\                         uint2 gid [[thread_position_in_grid]]) {
    \\    // dims = [rows, cols]
    \\    const uint rows = dims[0];
    \\    const uint cols = dims[1];
    \\    
    \\    // Skip out-of-bounds work
    \\    if (gid.x >= rows || gid.y >= cols) return;
    \\    
    \\    // Transpose by swapping indices
    \\    result[gid.y * rows + gid.x] = input[gid.x * cols + gid.y];
    \\}
    \\
    \\// Embedding lookup operation
    \\kernel void embedding_lookup_f32(device const float* params [[buffer(0)]],
    \\                               device const float* indices [[buffer(1)]],
    \\                               device float* result [[buffer(2)]],
    \\                               device const uint* dims [[buffer(3)]],
    \\                               uint3 gid [[thread_position_in_grid]]) {
    \\    // dims = [batch_size, seq_len, vocab_size, embed_dim]
    \\    const uint batch_size = dims[0];
    \\    const uint seq_len = dims[1];
    \\    const uint vocab_size = dims[2];
    \\    const uint embed_dim = dims[3];
    \\    
    \\    // Skip out-of-bounds work
    \\    if (gid.x >= embed_dim || gid.y >= seq_len || gid.z >= batch_size) return;
    \\    
    \\    // Get the token ID for this position
    \\    uint token_id = uint(indices[gid.z * seq_len + gid.y]);
    \\    if (token_id >= vocab_size) token_id = vocab_size - 1; // Clamp to valid range
    \\    
    \\    // Lookup embedding and copy to result
    \\    result[(gid.z * seq_len + gid.y) * embed_dim + gid.x] = params[token_id * embed_dim + gid.x];
    \\}
;

// Implementation of primitive operations for Metal backend
pub fn hasPrimitive(comptime name: []const u8) bool {
    return std.mem.eql(u8, name, "add") or
        std.mem.eql(u8, name, "subtract") or
        std.mem.eql(u8, name, "multiply") or
        std.mem.eql(u8, name, "divide") or
        std.mem.eql(u8, name, "matmul") or
        std.mem.eql(u8, name, "relu") or
        std.mem.eql(u8, name, "softmax") or
        std.mem.eql(u8, name, "transpose") or
        std.mem.eql(u8, name, "embedding_lookup");
}

// Element-wise add operation implementation
pub fn add(a: Tensor, b: Tensor, result: *Tensor) void {
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
        @panic("Metal operations only supported on Apple platforms");
    }

    const ctx = getContext() catch @panic("Metal context not initialized");
    const device = ctx.device orelse @panic("Metal device not initialized");
    const command_queue = ctx.command_queue orelse @panic("Metal command queue not initialized");
    const add_pipeline = ctx.add_pipeline orelse @panic("Metal add pipeline not initialized");

    // Get element count
    const elem_count = a.shape.elemCount();

    // Create Metal buffers
    const a_buffer = MTLDevice_newBufferWithBytes(device, a.buffer.data.ptr, a.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor a");

    const b_buffer = MTLDevice_newBufferWithBytes(device, b.buffer.data.ptr, b.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor b");

    const result_buffer = MTLDevice_newBufferWithLength(device, result.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for result tensor");

    // Create a command buffer
    const command_buffer = MTLCommandQueue_commandBuffer(command_queue) orelse {
        @panic("Failed to create command buffer");
    };

    // Create a compute command encoder
    const compute_encoder = MTLCommandBuffer_computeCommandEncoder(command_buffer) orelse {
        @panic("Failed to create compute command encoder");
    };

    // Set the compute pipeline state
    MTLComputeCommandEncoder_setComputePipelineState(compute_encoder, add_pipeline);

    // Set buffers
    MTLComputeCommandEncoder_setBuffer(compute_encoder, a_buffer, 0, 0);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, b_buffer, 0, 1);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, result_buffer, 0, 2);

    // Calculate threadgroup size
    const max_threads_per_group = MTLComputePipelineState_maxTotalThreadsPerThreadgroup(add_pipeline);
    const thread_execution_width = MTLComputePipelineState_threadExecutionWidth(add_pipeline);

    const threads_per_threadgroup = [3]usize{
        @min(max_threads_per_group, thread_execution_width),
        1,
        1,
    };

    // Dispatch threads using our helper function
    dispatchThreads(compute_encoder, [3]usize{ elem_count, 1, 1 }, threads_per_threadgroup);

    // End encoding
    MTLComputeCommandEncoder_endEncoding(compute_encoder);

    // Commit and wait
    MTLCommandBuffer_commit(command_buffer);
    MTLCommandBuffer_waitUntilCompleted(command_buffer);

    // Copy result data back
    const result_data = MTLBuffer_contents(result_buffer);
    @memcpy(result.buffer.data, @as([*]u8, @ptrCast(result_data))[0..result.buffer.data.len]);
}

// Element-wise subtract operation implementation
pub fn subtract(_: type, a: Tensor, b: Tensor, result: *Tensor) void {
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
        @panic("Metal operations only supported on Apple platforms");
    }

    const ctx = getContext() catch @panic("Metal context not initialized");
    const device = ctx.device orelse @panic("Metal device not initialized");
    const command_queue = ctx.command_queue orelse @panic("Metal command queue not initialized");
    const subtract_pipeline = ctx.subtract_pipeline orelse @panic("Metal subtract pipeline not initialized");

    // Get element count
    const elem_count = a.shape.elemCount();

    // Create Metal buffers
    const a_buffer = MTLDevice_newBufferWithBytes(device, a.buffer.data.ptr, a.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor a");

    const b_buffer = MTLDevice_newBufferWithBytes(device, b.buffer.data.ptr, b.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor b");

    const result_buffer = MTLDevice_newBufferWithLength(device, result.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for result tensor");

    // Create a command buffer
    const command_buffer = MTLCommandQueue_commandBuffer(command_queue) orelse {
        @panic("Failed to create command buffer");
    };

    // Create a compute command encoder
    const compute_encoder = MTLCommandBuffer_computeCommandEncoder(command_buffer) orelse {
        @panic("Failed to create compute command encoder");
    };

    // Set the compute pipeline state
    MTLComputeCommandEncoder_setComputePipelineState(compute_encoder, subtract_pipeline);

    // Set buffers
    MTLComputeCommandEncoder_setBuffer(compute_encoder, a_buffer, 0, 0);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, b_buffer, 0, 1);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, result_buffer, 0, 2);

    // Calculate threadgroup size
    const max_threads_per_group = MTLComputePipelineState_maxTotalThreadsPerThreadgroup(subtract_pipeline);
    const thread_execution_width = MTLComputePipelineState_threadExecutionWidth(subtract_pipeline);

    const threads_per_threadgroup = [3]usize{
        @min(max_threads_per_group, thread_execution_width),
        1,
        1,
    };

    // Dispatch threads using our helper function
    dispatchThreads(compute_encoder, [3]usize{ elem_count, 1, 1 }, threads_per_threadgroup);

    // End encoding
    MTLComputeCommandEncoder_endEncoding(compute_encoder);

    // Commit and wait
    MTLCommandBuffer_commit(command_buffer);
    MTLCommandBuffer_waitUntilCompleted(command_buffer);

    // Copy result data back
    const result_data = MTLBuffer_contents(result_buffer);
    @memcpy(result.buffer.data, @as([*]u8, @ptrCast(result_data))[0..result.buffer.data.len]);
}

// Element-wise multiply operation implementation
pub fn multiply(a: Tensor, b: Tensor, result: *Tensor) void {
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
        @panic("Metal operations only supported on Apple platforms");
    }

    const ctx = getContext() catch @panic("Metal context not initialized");
    const device = ctx.device orelse @panic("Metal device not initialized");
    const command_queue = ctx.command_queue orelse @panic("Metal command queue not initialized");
    const multiply_pipeline = ctx.multiply_pipeline orelse @panic("Metal multiply pipeline not initialized");

    // Get element count
    const elem_count = a.shape.elemCount();

    // Create Metal buffers
    const a_buffer = MTLDevice_newBufferWithBytes(device, a.buffer.data.ptr, a.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor a");

    const b_buffer = MTLDevice_newBufferWithBytes(device, b.buffer.data.ptr, b.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor b");

    const result_buffer = MTLDevice_newBufferWithLength(device, result.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for result tensor");

    // Create a command buffer
    const command_buffer = MTLCommandQueue_commandBuffer(command_queue) orelse {
        @panic("Failed to create command buffer");
    };

    // Create a compute command encoder
    const compute_encoder = MTLCommandBuffer_computeCommandEncoder(command_buffer) orelse {
        @panic("Failed to create compute command encoder");
    };

    // Set the compute pipeline state
    MTLComputeCommandEncoder_setComputePipelineState(compute_encoder, multiply_pipeline);

    // Set buffers
    MTLComputeCommandEncoder_setBuffer(compute_encoder, a_buffer, 0, 0);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, b_buffer, 0, 1);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, result_buffer, 0, 2);

    // Calculate threadgroup size
    const max_threads_per_group = MTLComputePipelineState_maxTotalThreadsPerThreadgroup(multiply_pipeline);
    const thread_execution_width = MTLComputePipelineState_threadExecutionWidth(multiply_pipeline);

    const threads_per_threadgroup = [3]usize{
        @min(max_threads_per_group, thread_execution_width),
        1,
        1,
    };

    // Dispatch threads using our helper function
    dispatchThreads(compute_encoder, [3]usize{ elem_count, 1, 1 }, threads_per_threadgroup);

    // End encoding
    MTLComputeCommandEncoder_endEncoding(compute_encoder);

    // Commit and wait
    MTLCommandBuffer_commit(command_buffer);
    MTLCommandBuffer_waitUntilCompleted(command_buffer);

    // Copy result data back
    const result_data = MTLBuffer_contents(result_buffer);
    @memcpy(result.buffer.data, @as([*]u8, @ptrCast(result_data))[0..result.buffer.data.len]);
}

// Matrix multiplication
pub fn matmul(a: Tensor, b: Tensor, result: *Tensor) void {
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
        @panic("Metal operations only supported on Apple platforms");
    }

    const ctx = getContext() catch @panic("Metal context not initialized");
    const device = ctx.device orelse @panic("Metal device not initialized");
    const command_queue = ctx.command_queue orelse @panic("Metal command queue not initialized");
    const matmul_pipeline = ctx.matmul_pipeline orelse @panic("Metal matmul pipeline not initialized");

    // Get dimensions
    const m = a.shape.dims[0];
    const n = a.shape.dims[1];
    const p = b.shape.dims[1];

    // Create Metal buffers
    const a_buffer = MTLDevice_newBufferWithBytes(device, a.buffer.data.ptr, a.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor a");

    const b_buffer = MTLDevice_newBufferWithBytes(device, b.buffer.data.ptr, b.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor b");

    const result_buffer = MTLDevice_newBufferWithLength(device, result.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for result tensor");

    // Create dimensions buffer
    const dims = [_]u32{ @intCast(m), @intCast(n), @intCast(p) };
    const dims_buffer = MTLDevice_newBufferWithBytes(device, &dims, dims.len * @sizeOf(u32), 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for dimensions");

    // Create a command buffer
    const command_buffer = MTLCommandQueue_commandBuffer(command_queue) orelse {
        @panic("Failed to create command buffer");
    };

    // Create a compute command encoder
    const compute_encoder = MTLCommandBuffer_computeCommandEncoder(command_buffer) orelse {
        @panic("Failed to create compute command encoder");
    };

    // Set the compute pipeline state
    MTLComputeCommandEncoder_setComputePipelineState(compute_encoder, matmul_pipeline);

    // Set buffers
    MTLComputeCommandEncoder_setBuffer(compute_encoder, a_buffer, 0, 0);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, b_buffer, 0, 1);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, result_buffer, 0, 2);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, dims_buffer, 0, 3);

    // Calculate threadgroup size
    const max_threads_per_group = MTLComputePipelineState_maxTotalThreadsPerThreadgroup(matmul_pipeline);
    const thread_execution_width = MTLComputePipelineState_threadExecutionWidth(matmul_pipeline);

    const threads_per_threadgroup_dim = std.math.sqrt(thread_execution_width);
    const threads_per_threadgroup = [3]usize{
        @min(threads_per_threadgroup_dim, max_threads_per_group),
        @min(threads_per_threadgroup_dim, max_threads_per_group),
        1,
    };

    // Dispatch threads using our helper function
    dispatchThreads(compute_encoder, [3]usize{ p, m, 1 }, threads_per_threadgroup);

    // End encoding
    MTLComputeCommandEncoder_endEncoding(compute_encoder);

    // Commit and wait
    MTLCommandBuffer_commit(command_buffer);
    MTLCommandBuffer_waitUntilCompleted(command_buffer);

    // Copy result data back
    const result_data = MTLBuffer_contents(result_buffer);
    @memcpy(result.buffer.data, @as([*]u8, @ptrCast(result_data))[0..result.buffer.data.len]);
}

// ReLU activation
pub fn relu(a: Tensor, result: *Tensor) void {
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
        @panic("Metal operations only supported on Apple platforms");
    }

    const ctx = getContext() catch @panic("Metal context not initialized");
    const device = ctx.device orelse @panic("Metal device not initialized");
    const command_queue = ctx.command_queue orelse @panic("Metal command queue not initialized");
    const relu_pipeline = ctx.relu_pipeline orelse @panic("Metal relu pipeline not initialized");

    // Get element count
    const elem_count = a.shape.elemCount();

    // Create Metal buffers
    const a_buffer = MTLDevice_newBufferWithBytes(device, a.buffer.data.ptr, a.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for tensor a");

    const result_buffer = MTLDevice_newBufferWithLength(device, result.buffer.data.len, 0 // MTLResourceStorageModeShared
    ) orelse @panic("Failed to create buffer for result tensor");

    // Create a command buffer
    const command_buffer = MTLCommandQueue_commandBuffer(command_queue) orelse {
        @panic("Failed to create command buffer");
    };

    // Create a compute command encoder
    const compute_encoder = MTLCommandBuffer_computeCommandEncoder(command_buffer) orelse {
        @panic("Failed to create compute command encoder");
    };

    // Set the compute pipeline state
    MTLComputeCommandEncoder_setComputePipelineState(compute_encoder, relu_pipeline);

    // Set buffers
    MTLComputeCommandEncoder_setBuffer(compute_encoder, a_buffer, 0, 0);
    MTLComputeCommandEncoder_setBuffer(compute_encoder, result_buffer, 0, 1);

    // Calculate threadgroup size
    const max_threads_per_group = MTLComputePipelineState_maxTotalThreadsPerThreadgroup(relu_pipeline);
    const thread_execution_width = MTLComputePipelineState_threadExecutionWidth(relu_pipeline);

    const threads_per_threadgroup = [3]usize{
        @min(max_threads_per_group, thread_execution_width),
        1,
        1,
    };

    // Dispatch threads using our helper function
    dispatchThreads(compute_encoder, [3]usize{ elem_count, 1, 1 }, threads_per_threadgroup);

    // End encoding
    MTLComputeCommandEncoder_endEncoding(compute_encoder);

    // Commit and wait
    MTLCommandBuffer_commit(command_buffer);
    MTLCommandBuffer_waitUntilCompleted(command_buffer);

    // Copy result data back
    const result_data = MTLBuffer_contents(result_buffer);
    @memcpy(result.buffer.data, @as([*]u8, @ptrCast(result_data))[0..result.buffer.data.len]);
}

// This function compiles Metal kernel code into a library and creates compute pipelines
fn compileMetalCode(ctx: *MetalContext, code: []const u8) !void {
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) {
        return MetalError.UnsupportedPlatform;
    }

    const device = ctx.device orelse return MetalError.DeviceNotFound;

    // Create NSString from kernel code
    const source_str = NSString_alloc();
    const source = NSString_initWithUTF8String(source_str, @ptrCast(code));
    defer NSString_release(source);

    // Compile the Metal code
    var compile_error: ?*NSError = null;
    const library = MTLDevice_newLibraryWithSource(device, source, null, &compile_error) orelse {
        // Handle compile error
        return MetalError.LibraryCreationFailed;
    };
    ctx.library = library;

    // Create compute pipeline for each kernel

    // 1. Add kernel
    {
        const kernel_name = NSString_alloc();
        const add_name = NSString_initWithUTF8String(kernel_name, "add_f32");
        defer NSString_release(add_name);

        const add_function = MTLLibrary_newFunctionWithName(library, add_name) orelse {
            return MetalError.FunctionCreationFailed;
        };

        var pipeline_error: ?*NSError = null;
        const add_pipeline = MTLDevice_newComputePipelineStateWithFunction(device, add_function, &pipeline_error) orelse {
            return MetalError.PipelineCreationFailed;
        };

        ctx.add_pipeline = add_pipeline;
    }

    // 2. Subtract kernel
    {
        const kernel_name = NSString_alloc();
        const subtract_name = NSString_initWithUTF8String(kernel_name, "subtract_f32");
        defer NSString_release(subtract_name);

        const subtract_function = MTLLibrary_newFunctionWithName(library, subtract_name) orelse {
            return MetalError.FunctionCreationFailed;
        };

        var pipeline_error: ?*NSError = null;
        const subtract_pipeline = MTLDevice_newComputePipelineStateWithFunction(device, subtract_function, &pipeline_error) orelse {
            return MetalError.PipelineCreationFailed;
        };

        ctx.subtract_pipeline = subtract_pipeline;
    }

    // 3. Multiply kernel
    {
        const kernel_name = NSString_alloc();
        const multiply_name = NSString_initWithUTF8String(kernel_name, "multiply_f32");
        defer NSString_release(multiply_name);

        const multiply_function = MTLLibrary_newFunctionWithName(library, multiply_name) orelse {
            return MetalError.FunctionCreationFailed;
        };

        var pipeline_error: ?*NSError = null;
        const multiply_pipeline = MTLDevice_newComputePipelineStateWithFunction(device, multiply_function, &pipeline_error) orelse {
            return MetalError.PipelineCreationFailed;
        };

        ctx.multiply_pipeline = multiply_pipeline;
    }

    // 4. ReLU kernel
    {
        const kernel_name = NSString_alloc();
        const relu_name = NSString_initWithUTF8String(kernel_name, "relu_f32");
        defer NSString_release(relu_name);

        const relu_function = MTLLibrary_newFunctionWithName(library, relu_name) orelse {
            return MetalError.FunctionCreationFailed;
        };

        var pipeline_error: ?*NSError = null;
        const relu_pipeline = MTLDevice_newComputePipelineStateWithFunction(device, relu_function, &pipeline_error) orelse {
            return MetalError.PipelineCreationFailed;
        };

        ctx.relu_pipeline = relu_pipeline;
    }

    // 5. Matmul kernel
    {
        const kernel_name = NSString_alloc();
        const matmul_name = NSString_initWithUTF8String(kernel_name, "matmul_f32");
        defer NSString_release(matmul_name);

        const matmul_function = MTLLibrary_newFunctionWithName(library, matmul_name) orelse {
            return MetalError.FunctionCreationFailed;
        };

        var pipeline_error: ?*NSError = null;
        const matmul_pipeline = MTLDevice_newComputePipelineStateWithFunction(device, matmul_function, &pipeline_error) orelse {
            return MetalError.PipelineCreationFailed;
        };

        ctx.matmul_pipeline = matmul_pipeline;
    }
}

// Name our implementation functions so we can reference them without recursion
const metal_impl = struct {
    pub fn add(a: Tensor, b: Tensor, result: *Tensor) void {
        @import("metal.zig").add(a, b, result);
    }

    pub fn subtract(a: Tensor, b: Tensor, result: *Tensor) void {
        @import("metal.zig").subtract(a, b, result);
    }

    pub fn multiply(a: Tensor, b: Tensor, result: *Tensor) void {
        @import("metal.zig").multiply(a, b, result);
    }

    pub fn matmul(a: Tensor, b: Tensor, result: *Tensor) void {
        @import("metal.zig").matmul(a, b, result);
    }

    pub fn relu(a: Tensor, result: *Tensor) void {
        @import("metal.zig").relu(a, result);
    }
};

// Public interface for Metal operations
pub const MetalBackend = struct {
    // Check if this backend implements a specific primitive operation
    pub fn hasPrimitive(comptime name: []const u8) bool {
        return std.mem.eql(u8, name, "add") or
            std.mem.eql(u8, name, "subtract") or
            std.mem.eql(u8, name, "multiply") or
            std.mem.eql(u8, name, "divide") or
            std.mem.eql(u8, name, "matmul") or
            std.mem.eql(u8, name, "relu") or
            std.mem.eql(u8, name, "softmax") or
            std.mem.eql(u8, name, "transpose") or
            std.mem.eql(u8, name, "embedding_lookup");
    }

    // Expose the metal implementations through our struct
    pub const add = metal_impl.add;
    pub const subtract = metal_impl.subtract;
    pub const multiply = metal_impl.multiply;
    pub const matmul = metal_impl.matmul;
    pub const relu = metal_impl.relu;
};
