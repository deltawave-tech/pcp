/// Demo-only version of distributed training (no MLIR dependencies)
/// This file allows testing the distributed system without StableHLO

const std = @import("std");
const print = std.debug.print;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

// No external dependencies - fully self-contained demo

/// Simple demo coordinator
const DemoShepherd = struct {
    allocator: Allocator,
    workers: ArrayList(DemoWorker),
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .workers = ArrayList(DemoWorker).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.workers.deinit();
    }
    
    pub fn addWorker(self: *Self, worker: DemoWorker) !void {
        try self.workers.append(worker);
    }
    
    pub fn runTraining(self: *Self) !void {
        // Clear screen and show TUI
        print("\x1B[2J\x1B[H", .{}); // Clear screen and move cursor to top
        
        // Simulate 1000 training rounds
        for (0..1000) |round| {
            // Clear screen and redraw TUI
            print("\x1B[2J\x1B[H", .{});
            
            // Draw TUI header
            print("┌─────────────────────────────────────────────────────────┐\n", .{});
            print("│ 🪐 PCP Distributed Training                             │\n", .{});
            print("├─────────────────────────────────────────────────────────┤\n", .{});
            print("│ 👹 Shepherd: DiLoCo Algorithm                           │\n", .{});
            print("│ 🐉 Workers: {}                                            │\n", .{self.workers.items.len});
            print("├─────────────────────────────────────────────────────────┤\n", .{});
            print("│ Training Progress                                       │\n", .{});
            print("│                                                         │\n", .{});
            
            // Progress bar
            const progress = (round + 1) * 50 / 1000; // 50 chars wide
            print("│ [", .{});
            for (0..50) |i| {
                if (i < progress) {
                    print("█", .{});
                } else {
                    print("░", .{});
                }
            }
            print("] {}/1000 │\n", .{round + 1});
            
            print("│                                                         │\n", .{});
            print("├─────────────────────────────────────────────────────────┤\n", .{});
            print("│ Worker Status                                           │\n", .{});
            
            // Show worker status
            for (self.workers.items, 0..) |worker, i| {
                print("│ 🐉 Worker {} - Loss: {d:.4}                             │\n", .{i + 1, worker.current_loss});
            }
            
            // Calculate and show average
            const avg_loss = self.calculateAverageLoss();
            print("│                                                         │\n", .{});
            print("│ 🌙 Average Loss: {d:.4}                                 │\n", .{avg_loss});
            print("│                                                         │\n", .{});
            print("└─────────────────────────────────────────────────────────┘\n", .{});
            
            if (round < 999) {
                print("\n🌙 Training Round {}/1000 in progress...\n", .{round + 1});
                
                // Simulate worker training with updates
                for (self.workers.items) |*worker| {
                    try worker.executeStep();
                }
                
                std.time.sleep(1500 * std.time.ns_per_ms); // Show dashboard for 1.5s
            }
        }
        
        print("\n🌑 Demo Training Completed Successfully!\n", .{});
    }
    
    fn calculateAverageLoss(self: *Self) f32 {
        var total: f32 = 0.0;
        for (self.workers.items) |worker| {
            total += worker.current_loss;
        }
        return total / @as(f32, @floatFromInt(self.workers.items.len));
    }
};

/// Simple demo worker
const DemoWorker = struct {
    id: u32,
    current_loss: f32,
    step_count: u32,
    
    const Self = @This();
    
    pub fn init(id: u32) Self {
        return Self{
            .id = id,
            .current_loss = 2.0, // Start with high loss
            .step_count = 0,
        };
    }
    
    pub fn executeStep(self: *Self) !void {
        self.step_count += 1;
        
        // Simulate realistic loss decay with some noise
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp() + self.id));
        const base_decay = std.math.exp(-@as(f32, @floatFromInt(self.step_count)) * 0.1);
        const noise = rng.random().floatNorm(f32) * 0.05;
        
        self.current_loss = @max(0.01, 2.0 * base_decay + noise);
        
        // Simulate execution time
        const execution_time = 80 + @mod(self.step_count * 13, 40); // 80-120ms
        std.time.sleep(execution_time * std.time.ns_per_ms);
    }
};

/// Command line arguments (simplified)
const Args = struct {
    mode: Mode,
    workers: usize,
    
    const Mode = enum {
        coordinator,
        worker,
    };
    
    pub fn parse(_: Allocator, args: [][:0]u8) Args {
        var workers: usize = 2;
        var mode: Mode = .coordinator;
        
        var i: usize = 1;
        while (i < args.len) {
            if (std.mem.eql(u8, args[i], "--workers")) {
                i += 1;
                if (i < args.len) {
                    workers = std.fmt.parseInt(usize, args[i], 10) catch 2;
                }
            } else if (std.mem.eql(u8, args[i], "--worker")) {
                mode = .worker;
            }
            i += 1;
        }
        
        return Args{
            .mode = mode,
            .workers = workers,
        };
    }
};

/// Run demo coordinator
fn runDemoCoordinator(allocator: Allocator, worker_count: usize) !void {
    print("👹 Demo Shepherd Coordinator\n", .{});
    print("   Workers: {}\n", .{worker_count});
    
    var shepherd = DemoShepherd.init(allocator);
    defer shepherd.deinit();
    
    // Create demo workers
    for (0..worker_count) |i| {
        const worker = DemoWorker.init(@intCast(i + 1));
        try shepherd.addWorker(worker);
    }
    
    // Run the training simulation
    try shepherd.runTraining();
}

/// Main function
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);
    
    // Show help
    for (process_args) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            print("PCP Distributed Training Demo (StableHLO-free)\n", .{});
            print("Usage: pcp_demo [--workers N]\n", .{});
            print("  --workers N    Number of workers to simulate (default: 2)\n", .{});
            print("  --help        Show this help\n", .{});
            return;
        }
    }
    
    const args = Args.parse(allocator, process_args);
    
    print("🪐 PCP Distributed Training Demo\n", .{});
    print("=================================\n", .{});
    
    try runDemoCoordinator(allocator, args.workers);
    
    print("\n🌙 Demo completed!\n", .{});
}