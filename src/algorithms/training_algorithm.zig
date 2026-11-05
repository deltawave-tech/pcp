/// Training Algorithm Interface
/// Defines the interface for distributed training algorithms like DiLoCo

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Generic training algorithm interface
pub const TrainingAlgorithm = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    
    const VTable = struct {
        run: *const fn (ptr: *anyopaque) anyerror!void,
        deinit: *const fn (ptr: *anyopaque) void,
        getName: *const fn (ptr: *anyopaque) []const u8,
        getStatus: *const fn (ptr: *anyopaque) TrainingStatus,
    };
    
    /// Run the training algorithm
    pub fn run(self: TrainingAlgorithm) !void {
        return self.vtable.run(self.ptr);
    }
    
    /// Clean up resources
    pub fn deinit(self: TrainingAlgorithm) void {
        self.vtable.deinit(self.ptr);
    }
    
    /// Get algorithm name
    pub fn getName(self: TrainingAlgorithm) []const u8 {
        return self.vtable.getName(self.ptr);
    }
    
    /// Get current training status
    pub fn getStatus(self: TrainingAlgorithm) TrainingStatus {
        return self.vtable.getStatus(self.ptr);
    }
};

/// Training status enumeration
pub const TrainingStatus = enum {
    not_started,
    initializing,
    running,
    completed,
    failed,
    paused,
};

/// Training configuration
pub const TrainingConfig = struct {
    max_epochs: usize,
    inner_loop_steps: usize,
    outer_loop_steps: usize,
    learning_rate: f32,
    batch_size: usize,
    
    pub fn default() TrainingConfig {
        return TrainingConfig{
            .max_epochs = 100,
            .inner_loop_steps = 10,
            .outer_loop_steps = 100,
            .learning_rate = 0.001,
            .batch_size = 32,
        };
    }
};

/// Training metrics
pub const TrainingMetrics = struct {
    epoch: usize,
    loss: f32,
    accuracy: f32,
    inner_loop_count: usize,
    outer_loop_count: usize,
    
    pub fn init() TrainingMetrics {
        return TrainingMetrics{
            .epoch = 0,
            .loss = 0.0,
            .accuracy = 0.0,
            .inner_loop_count = 0,
            .outer_loop_count = 0,
        };
    }
};