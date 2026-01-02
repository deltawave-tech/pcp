/// TUI Dashboard for Distributed Training Monitoring
/// Uses basic terminal control sequences for cross-platform compatibility

const std = @import("std");
const monitoring = @import("monitoring.zig");

const Allocator = std.mem.Allocator;

/// Simple TUI Dashboard using terminal control sequences
pub const Dashboard = struct {
    allocator: Allocator,
    should_quit: bool = false,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) !Self {
        return Self{
            .allocator = allocator,
        };
    }
    
    pub fn deinit(_: *Self) void {
        // Restore cursor and clear screen on exit
        std.debug.print("\x1b[?25h\x1b[2J\x1b[H", .{});
    }
    
    /// Main dashboard loop
    pub fn run(self: *Self) !void {
        // Hide cursor and enter alternate screen
        std.debug.print("\x1b[?25l\x1b[?1049h\x1b[2J\x1b[H", .{});
        
        while (!self.should_quit) {
            // Check for 'q' key press (non-blocking)
            if (self.checkForQuit()) {
                self.should_quit = true;
                break;
            }

            // Get current metrics
            const metrics = monitoring.getMetrics();

            // Move cursor to home and render the UI
            std.debug.print("\x1b[H", .{});
            try self.render(metrics);

            // Exit automatically once training is done (or errored), so shepherd can terminate cleanly.
            if (metrics.training_status == .completed or metrics.training_status == .error_state) {
                self.should_quit = true;
                break;
            }

            // Don't burn CPU, refresh every 500ms
            std.time.sleep(500 * std.time.ns_per_ms);
        }
        
        // Exit alternate screen and restore cursor
        std.debug.print("\x1b[?1049l\x1b[?25h", .{});
    }
    
    /// Check for quit key (simplified)
    fn checkForQuit(self: *Self) bool {
        _ = self;
        // For now, just return false - in a real implementation, this would check stdin
        // This is a simplified version to avoid complex terminal input handling
        return false;
    }
    
    /// Render the TUI dashboard
    fn render(_: *Self, metrics: monitoring.TrainingMetrics) !void {

        var buf: [256]u8 = undefined;

        // Clear from cursor to end of screen to avoid artifacts
        std.debug.print("\x1b[0J", .{});

        // Title with colors
        std.debug.print("\x1b[1;36m🪐 Planetary Compute Protocol - Dashboard\x1b[0m\n\n", .{});
        
        // Status Section
        drawBox("Status", 40);
        std.debug.print("  \x1b[37mWorkers Connected:\x1b[0m \x1b[32m{}\x1b[0m\n", .{metrics.workers_connected});
        std.debug.print("  \x1b[37mTraining Status:\x1b[0m  \x1b[32m{s}\x1b[0m\n", .{metrics.training_status.toString()});

        const param_formatted = monitoring.formatBytes(metrics.total_parameters * 4); // Assume f32 params
        const param_str = try std.fmt.bufPrint(&buf, "{d:.1} {s}", .{ param_formatted.value, param_formatted.unit });
        std.debug.print("  \x1b[37mModel Size:\x1b[0m       \x1b[32m{s}\x1b[0m\n", .{param_str});
        std.debug.print("\n", .{});

        // Workers Section
        drawBox("Connected Workers", 80);
        if (metrics.worker_info_count > 0) {
            std.debug.print("  \x1b[90m┌────┬──────────┬──────────────────────────┬─────────────┐\x1b[0m\n", .{});
            std.debug.print("  \x1b[90m│\x1b[0m \x1b[1;37mID\x1b[0m \x1b[90m│\x1b[0m \x1b[1;37mBackend\x1b[0m  \x1b[90m│\x1b[0m \x1b[1;37mIP Address\x1b[0m               \x1b[90m│\x1b[0m \x1b[1;37mStatus\x1b[0m      \x1b[90m│\x1b[0m\n", .{});
            std.debug.print("  \x1b[90m├────┼──────────┼──────────────────────────┼─────────────┤\x1b[0m\n", .{});

            for (0..metrics.worker_info_count) |i| {
                const worker = metrics.worker_info[i];
                const backend = worker.getBackend();
                const ip_address = worker.getIpAddress();
                const status = worker.getStatus();

                const status_color = if (std.mem.eql(u8, status, "Training"))
                    "\x1b[32m"  // Green for Training
                else if (std.mem.eql(u8, status, "Initialized"))
                    "\x1b[33m"  // Yellow for Initialized
                else
                    "\x1b[35m"; // Magenta for Connected

                std.debug.print("  \x1b[90m│\x1b[0m \x1b[36m{d:2}\x1b[0m \x1b[90m│\x1b[0m \x1b[35m{s:<8}\x1b[0m \x1b[90m│\x1b[0m \x1b[37m{s:<24}\x1b[0m \x1b[90m│\x1b[0m {s}{s:<11}\x1b[0m \x1b[90m│\x1b[0m\n", .{
                    worker.node_id,
                    backend,
                    ip_address,
                    status_color,
                    status,
                });
            }

            std.debug.print("  \x1b[90m└────┴──────────┴──────────────────────────┴─────────────┘\x1b[0m\n", .{});
        } else {
            std.debug.print("  \x1b[90mNo workers connected yet...\x1b[0m\n", .{});
        }
        std.debug.print("\n", .{});

        // Metrics Section
        drawBox("Metrics", 40);
        std.debug.print("  \x1b[37mOuter Loop Step:\x1b[0m  \x1b[32m{}\x1b[0m\n", .{metrics.outer_loop_step});
        
        const loss_str = try std.fmt.bufPrint(&buf, "{d:.6}", .{metrics.average_loss});
        std.debug.print("  \x1b[37mAverage Loss:\x1b[0m     \x1b[32m{s}\x1b[0m\n", .{loss_str});
        
        const lr_str = try std.fmt.bufPrint(&buf, "{d:.6}", .{metrics.learning_rate});
        std.debug.print("  \x1b[37mLearning Rate:\x1b[0m    \x1b[32m{s}\x1b[0m\n", .{lr_str});
        std.debug.print("\n", .{});
        
        // Performance Section
        drawBox("⚡ Performance", 60);
        
        const time_formatted = monitoring.formatDuration(metrics.epoch_time_ms);
        const time_str = try std.fmt.bufPrint(&buf, "{d:.1} {s}", .{ time_formatted.value, time_formatted.unit });
        std.debug.print("  \x1b[37mLast Epoch Time:\x1b[0m  \x1b[32m{s}\x1b[0m", .{time_str});
        
        const throughput = if (metrics.epoch_time_ms > 0) 
            @as(f64, @floatFromInt(metrics.total_parameters)) / (@as(f64, @floatFromInt(metrics.epoch_time_ms)) / 1000.0)
        else 0.0;
        const throughput_formatted = monitoring.formatBytes(@intFromFloat(throughput));
        const throughput_str = try std.fmt.bufPrint(&buf, "{d:.1} {s}/s", .{ throughput_formatted.value, throughput_formatted.unit });
        std.debug.print("     \x1b[37mThroughput:\x1b[0m \x1b[32m{s}\x1b[0m\n", .{throughput_str});
        std.debug.print("\n", .{});
        
        // Loss History Graph
        drawBox("Loss History", 70);
        try drawLossGraph(metrics);
        std.debug.print("\n", .{});
        
        // Instructions
        std.debug.print("\x1b[37mPress Ctrl+C to quit | Updates every 500ms\x1b[0m\n", .{});
    }
    
    /// Draw a simple bordered box with title
    fn drawBox(title: []const u8, width: usize) void {
        
        // Top border
        std.debug.print("\x1b[90m┌", .{});
        for (0..width - 2) |_| {
            std.debug.print("─", .{});
        }
        std.debug.print("┐\x1b[0m\n", .{});
        
        // Title
        std.debug.print("\x1b[90m│\x1b[0m \x1b[1;33m{s}\x1b[0m", .{title});
        const title_len = title.len + 1; // +1 for space
        for (title_len..width - 2) |_| {
            std.debug.print(" ", .{});
        }
        std.debug.print("\x1b[90m│\x1b[0m\n", .{});
        
        // Separator
        std.debug.print("\x1b[90m├", .{});
        for (0..width - 2) |_| {
            std.debug.print("─", .{});
        }
        std.debug.print("┤\x1b[0m\n", .{});
    }
    
    /// Draw loss history as a simple bar chart
    fn drawLossGraph(metrics: monitoring.TrainingMetrics) !void {
        
        if (metrics.loss_history_count == 0) {
            std.debug.print("  \x1b[90mNo data available yet...\x1b[0m\n", .{});
            return;
        }
        
        // Get loss history
        var history: [50]f32 = undefined;
        var monitor = &monitoring.global_monitor;
        const count = monitor.getLossHistory(history[0..@min(50, metrics.loss_history_count)]);
        
        if (count == 0) {
            std.debug.print("  \x1b[90mNo data available yet...\x1b[0m\n", .{});
            return;
        }
        
        // Find min/max for scaling
        var min_loss = history[0];
        var max_loss = history[0];
        for (history[0..count]) |loss| {
            min_loss = @min(min_loss, loss);
            max_loss = @max(max_loss, loss);
        }
        
        // Avoid division by zero
        if (max_loss - min_loss < 0.001) {
            max_loss = min_loss + 0.001;
        }
        
        const graph_height = 6;
        const graph_width = @min(count, 60);
        
        // Draw the graph
        for (0..graph_height) |row| {
            std.debug.print("  ", .{});
            for (0..graph_width) |i| {
                const idx = if (count > graph_width) 
                    (i * count) / graph_width 
                else 
                    i;
                    
                if (idx >= count) break;
                
                const normalized = (history[idx] - min_loss) / (max_loss - min_loss);
                const bar_height = @as(usize, @intFromFloat(normalized * @as(f32, @floatFromInt(graph_height))));
                
                if (bar_height > (graph_height - row - 1)) {
                    if (row < graph_height / 2) {
                        std.debug.print("\x1b[32m█\x1b[0m", .{}); // Green
                    } else {
                        std.debug.print("\x1b[33m█\x1b[0m", .{}); // Yellow
                    }
                } else {
                    std.debug.print(" ", .{});
                }
            }
            
            // Y-axis labels
            if (row == 0) {
                var buf: [16]u8 = undefined;
                const max_str = try std.fmt.bufPrint(&buf, " Max: {d:.3}", .{max_loss});
                std.debug.print("\x1b[37m{s}\x1b[0m", .{max_str});
            } else if (row == graph_height - 1) {
                var buf: [16]u8 = undefined;
                const min_str = try std.fmt.bufPrint(&buf, " Min: {d:.3}", .{min_loss});
                std.debug.print("\x1b[37m{s}\x1b[0m", .{min_str});
            }
            
            std.debug.print("\n", .{});
        }
        
        // Bottom border of graph box
        std.debug.print("\x1b[90m└", .{});
        for (0..68) |_| {
            std.debug.print("─", .{});
        }
        std.debug.print("┘\x1b[0m\n", .{});
    }
};

/// Main entry point for dashboard thread
pub fn runDashboard() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var dashboard = Dashboard.init(allocator) catch |err| {
        std.log.err("Failed to initialize dashboard: {}", .{err});
        return;
    };
    defer dashboard.deinit();

    // Hide cursor and use a stable position (don't use alternate screen to avoid interfering with logs)
    std.debug.print("\x1b[?25l", .{}); // Hide cursor
    defer std.debug.print("\x1b[?25h", .{}); // Restore cursor at end

    // Run continuously (Ctrl+C to quit)
    while (true) {
        // Get current metrics
        const metrics = monitoring.getMetrics();

        // Move cursor to home and render (without clearing entire screen buffer)
        std.debug.print("\x1b[H", .{}); // Move cursor to home position
        dashboard.render(metrics) catch |err| {
            std.log.err("Dashboard render error: {}", .{err});
            break;
        };

        // Don't burn CPU, refresh every 500ms
        std.time.sleep(500 * std.time.ns_per_ms);
    }
}
