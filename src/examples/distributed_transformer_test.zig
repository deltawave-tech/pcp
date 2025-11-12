// In src/examples/distributed_transformer_test.zig
const std = @import("std");
const pcp = @import("pcp");

const Shepherd = pcp.controllers.shepherd.Shepherd;
const Worker = pcp.worker.Worker;
const DiLoCo = pcp.algorithms.diloco.DiLoCo;
const DiLoCoConfig = pcp.algorithms.diloco.DiLoCoConfig;
const Backend = pcp.backend_selection.Backend;
const createWorkerBackend = pcp.backend_selection.createWorkerBackend;
const MLIRBuilder = pcp.ops.MLIRBuilder;
const MLIRContext = pcp.mlir_ctx.MLIRContext;
const DataLoader = pcp.data_loader.DataLoader;

const HOST = "127.0.0.1";
const PORT: u16 = 8090; // Use a clean port for the test
const NUM_WORKERS: usize = 2;

const ShepherdArgs = struct { allocator: std.mem.Allocator };
const WorkerArgs = struct { allocator: std.mem.Allocator };

/// This thread runs the Shepherd coordinator. It will initialize, run, and complete training.
fn shepherdThread(args: ShepherdArgs) !void {
    var shepherd = Shepherd.init(args.allocator);
    defer shepherd.deinit();

    var mlir_ctx = try MLIRContext.init(args.allocator);
    defer mlir_ctx.deinit();

    var mlir_builder = try MLIRBuilder.init(args.allocator, mlir_ctx.getContext());
    defer mlir_builder.deinit();

    // The Shepherd uses a 'demo' executor for orchestration and compilation.
    const demo_backend = try pcp.backends.demo.DemoBackend.init(args.allocator);
    const executor = demo_backend.asExecutor();
    shepherd.setExecutor(executor);

    // ** Use the existing, functional text DataLoader **
    var data_loader = try DataLoader.init(args.allocator, "tiny_shakespeare.txt");
    defer data_loader.deinit();

    // ** Configure DiLoCo to use the Nano-Transformer model **
    var diloco_config = DiLoCoConfig.default();
    diloco_config.model_mlir_path = "models/nano_gpt.mlir";
    diloco_config.base_config.outer_loop_steps = 3; // Run for 3 full outer loops for the test

    // Initialize the DiLoCo training algorithm with the real DataLoader
    var diloco_algo = try DiLoCo.init(args.allocator, &shepherd, diloco_config, executor, &mlir_builder, &data_loader);
    defer diloco_algo.deinit();
    
    var training_algo = diloco_algo.asTrainingAlgorithm();
    shepherd.setAlgorithm(&training_algo);

    // Start listening in a background thread
    const listen_thread = try std.Thread.spawn(.{}, struct{ fn run(s: *Shepherd) !void { try s.listen(HOST, PORT); }}.run, .{&shepherd});
    listen_thread.detach();

    // This blocks until NUM_WORKERS connect and the 3 training loops are complete.
    try shepherd.startTraining(NUM_WORKERS);
    
    // After training finishes, stop the Shepherd to release the port and signal workers.
    shepherd.stop();
}

/// This thread runs a Worker node, which will connect and execute training steps.
fn workerThread(args: WorkerArgs) !void {
    const backend = Backend.selectDefault();
    const worker_backend = try createWorkerBackend(args.allocator, backend);

    var worker = try Worker.init(args.allocator, worker_backend);
    defer worker.deinit();

    // Connect to the Shepherd
    try worker.connect(HOST, PORT);

    // This blocks until the Shepherd sends a SHUTDOWN message.
    try worker.run();
}

/// The main test function orchestrating the entire distributed run.
pub fn main() !void {
    std.debug.print("\n=== 🪐 Distributed Nano-Transformer Training Demo ===\n", .{});
    
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 1. Spawn the Shepherd thread
    const shepherd_args = ShepherdArgs{ .allocator = allocator };
    const shepherd_handle = try std.Thread.spawn(.{}, shepherdThread, .{shepherd_args});
    std.debug.print("👹 Shepherd thread spawned. Waiting for connections...\n", .{});

    // Give the Shepherd a moment to start its TCP server
    std.time.sleep(200 * std.time.ns_per_ms);

    // 2. Spawn worker threads
    var worker_handles: [NUM_WORKERS]std.Thread = undefined;
    for (&worker_handles, 0..) |*handle, i| {
        const worker_args = WorkerArgs{ .allocator = allocator };
        handle.* = try std.Thread.spawn(.{}, workerThread, .{worker_args});
        std.debug.print("🐉 Worker {} thread spawned and connecting...\n", .{i + 1});
    }

    // 3. Wait for the Shepherd to complete its training run. It will exit after its work is done.
    shepherd_handle.join();
    std.debug.print("🌙 Shepherd has completed the training run and shut down.\n", .{});

    // 4. Wait for all worker threads to finish. They exit after receiving the SHUTDOWN message.
    for (worker_handles, 0..) |handle, i| {
        handle.join();
        std.debug.print("👋 Worker {} has shut down gracefully.\n", .{i + 1});
    }

    std.debug.print("\n✅ Demo Complete: Distributed training of Nano-Transformer finished successfully!\n", .{});
}