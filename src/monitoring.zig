/// Thread-safe monitoring system for distributed training
/// Provides real-time metrics collection and reporting for the TUI dashboard

const std = @import("std");

/// Training metrics structure
pub const TrainingMetrics = struct {
    outer_loop_step: usize = 0,
    average_loss: f32 = 0.0,
    workers_connected: usize = 0,
    training_status: TrainingStatus = .initializing,
    last_update_time: i64 = 0,
    total_parameters: usize = 0,
    learning_rate: f32 = 0.01,
    epoch_time_ms: u64 = 0,
    
    /// Loss history for graphing (circular buffer)
    loss_history: [100]f32 = [_]f32{0.0} ** 100,
    loss_history_index: usize = 0,
    loss_history_count: usize = 0,
};

/// Training status enumeration
pub const TrainingStatus = enum {
    initializing,
    running,
    paused,
    completed,
    error_state,
    
    pub fn toString(self: TrainingStatus) []const u8 {
        return switch (self) {
            .initializing => "Initializing",
            .running => "Running",
            .paused => "Paused", 
            .completed => "Completed",
            .error_state => "Error",
        };
    }
};

/// Thread-safe training monitor
pub const TrainingMonitor = struct {
    mutex: std.Thread.Mutex = .{},
    metrics: TrainingMetrics = .{},
    
    const Self = @This();
    
    /// Set core training metrics atomically
    pub fn setMetrics(self: *Self, step: usize, loss: f32, workers: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.metrics.outer_loop_step = step;
        self.metrics.average_loss = loss;
        self.metrics.workers_connected = workers;
        self.metrics.last_update_time = std.time.timestamp();
        
        // Add to loss history (circular buffer)
        self.metrics.loss_history[self.metrics.loss_history_index] = loss;
        self.metrics.loss_history_index = (self.metrics.loss_history_index + 1) % self.metrics.loss_history.len;
        if (self.metrics.loss_history_count < self.metrics.loss_history.len) {
            self.metrics.loss_history_count += 1;
        }
    }
    
    /// Set training status
    pub fn setStatus(self: *Self, status: TrainingStatus) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.metrics.training_status = status;
        self.metrics.last_update_time = std.time.timestamp();
    }
    
    /// Set worker count
    pub fn setWorkerCount(self: *Self, count: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.metrics.workers_connected = count;
        self.metrics.last_update_time = std.time.timestamp();
    }
    
    /// Set model parameters
    pub fn setModelInfo(self: *Self, total_params: usize, lr: f32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.metrics.total_parameters = total_params;
        self.metrics.learning_rate = lr;
        self.metrics.last_update_time = std.time.timestamp();
    }
    
    /// Set epoch timing
    pub fn setEpochTime(self: *Self, time_ms: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.metrics.epoch_time_ms = time_ms;
        self.metrics.last_update_time = std.time.timestamp();
    }
    
    /// Get a snapshot of current metrics (thread-safe copy)
    pub fn getMetrics(self: *Self) TrainingMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.metrics;
    }
    
    /// Get current outer loop step
    pub fn getOuterLoopStep(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.metrics.outer_loop_step;
    }
    
    /// Get current loss
    pub fn getCurrentLoss(self: *Self) f32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.metrics.average_loss;
    }
    
    /// Get current worker count
    pub fn getWorkerCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.metrics.workers_connected;
    }
    
    /// Get current training status
    pub fn getStatus(self: *Self) TrainingStatus {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.metrics.training_status;
    }
    
    /// Get loss history for visualization
    pub fn getLossHistory(self: *Self, buffer: []f32) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const count = @min(buffer.len, self.metrics.loss_history_count);
        const start_idx = if (self.metrics.loss_history_count >= self.metrics.loss_history.len)
            self.metrics.loss_history_index
        else
            0;
            
        // Copy circular buffer to linear buffer
        for (0..count) |i| {
            const idx = (start_idx + i) % self.metrics.loss_history.len;
            buffer[i] = self.metrics.loss_history[idx];
        }
        
        return count;
    }
    
    /// Reset all metrics
    pub fn reset(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.metrics = .{};
    }
};

/// Global monitor instance for the application
pub var global_monitor = TrainingMonitor{};

/// Convenience functions for global monitor access
pub fn setMetrics(step: usize, loss: f32, workers: usize) void {
    global_monitor.setMetrics(step, loss, workers);
}

pub fn setStatus(status: TrainingStatus) void {
    global_monitor.setStatus(status);
}

pub fn setWorkerCount(count: usize) void {
    global_monitor.setWorkerCount(count);
}

pub fn setModelInfo(total_params: usize, lr: f32) void {
    global_monitor.setModelInfo(total_params, lr);
}

pub fn setEpochTime(time_ms: u64) void {
    global_monitor.setEpochTime(time_ms);
}

pub fn getMetrics() TrainingMetrics {
    return global_monitor.getMetrics();
}

pub fn getStatus() TrainingStatus {
    return global_monitor.getStatus();
}

/// Format bytes in human readable format
pub fn formatBytes(bytes: usize) struct { value: f64, unit: []const u8 } {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
    var value: f64 = @floatFromInt(bytes);
    var unit_idx: usize = 0;
    
    while (value >= 1024.0 and unit_idx < units.len - 1) {
        value /= 1024.0;
        unit_idx += 1;
    }
    
    return .{ .value = value, .unit = units[unit_idx] };
}

/// Format time duration in human readable format
pub fn formatDuration(ms: u64) struct { value: f64, unit: []const u8 } {
    if (ms < 1000) {
        return .{ .value = @floatFromInt(ms), .unit = "ms" };
    } else if (ms < 60000) {
        return .{ .value = @as(f64, @floatFromInt(ms)) / 1000.0, .unit = "s" };
    } else if (ms < 3600000) {
        return .{ .value = @as(f64, @floatFromInt(ms)) / 60000.0, .unit = "m" };
    } else {
        return .{ .value = @as(f64, @floatFromInt(ms)) / 3600000.0, .unit = "h" };
    }
}

/// Test the monitoring system
pub fn testMonitoring() !void {
    std.log.info("Testing TrainingMonitor...");
    
    var monitor = TrainingMonitor{};
    
    // Test basic metric setting
    monitor.setMetrics(10, 0.5, 3);
    monitor.setStatus(.running);
    monitor.setModelInfo(1000000, 0.001);
    
    // Get metrics
    const metrics = monitor.getMetrics();
    try std.testing.expectEqual(@as(usize, 10), metrics.outer_loop_step);
    try std.testing.expectEqual(@as(f32, 0.5), metrics.average_loss);
    try std.testing.expectEqual(@as(usize, 3), metrics.workers_connected);
    try std.testing.expectEqual(TrainingStatus.running, metrics.training_status);
    try std.testing.expectEqual(@as(usize, 1000000), metrics.total_parameters);
    try std.testing.expectEqual(@as(f32, 0.001), metrics.learning_rate);
    
    // Test loss history
    for (0..10) |i| {
        monitor.setMetrics(i, @as(f32, @floatFromInt(i)) * 0.1, 3);
    }
    
    var history: [10]f32 = undefined;
    const count = monitor.getLossHistory(&history);
    try std.testing.expectEqual(@as(usize, 10), count);
    
    std.log.info("✓ TrainingMonitor test completed");
}