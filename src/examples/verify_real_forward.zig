const std = @import("std");
const pcp = @import("pcp");
const mlir_ctx = pcp.mlir_ctx;
const backend_selection = pcp.backend_selection;
const IreeBackend = pcp.backends.iree.IreeBackend;
const DataLoader = pcp.data_loader.DataLoader;
const DType = pcp.tensor.DType;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🔍 Starting Real-Data Forward Pass Verification...\n", .{});

    std.debug.print("Initializing DataLoader...\n", .{});
    var loader = try DataLoader.init(allocator, "data/tiny_shakespeare.txt");
    defer loader.deinit();

    std.debug.print("📊 Tokenizer Vocab Size: {}\n", .{loader.tokenizer.vocab_size});
    if (loader.tokenizer.vocab_size > 65) {
        std.debug.print("⚠️  WARNING: Vocab size {} > 65 (Model Limit). This WILL cause NaNs!\n", .{loader.tokenizer.vocab_size});
    }

    const model_path = "models/nanochat_small.mlir";
    const mlir_source = try std.fs.cwd().readFileAlloc(allocator, model_path, 10 * 1024 * 1024);
    defer allocator.free(mlir_source);

    var mlir_context = try mlir_ctx.MLIRContext.init(allocator);
    defer mlir_context.deinit();
    const backend_type = backend_selection.Backend.cuda;
    const vmfb = try mlir_context.compileToVMFB(allocator, mlir_source, backend_type.toIreeCompilationTarget(), null);
    defer allocator.free(vmfb);

    var backend = try IreeBackend.init(allocator, backend_type, 0);
    defer backend.deinit();

    const batch_size = 64;
    const block_size = 32;
    const batch = try loader.getBatch(batch_size, block_size);
    defer {
        allocator.free(batch.x);
        allocator.free(batch.y);
    }

    var max_token: u64 = 0;
    for (batch.x) |t| if (t > max_token) {
        max_token = t;
    };
    std.debug.print("📊 Max Token ID in batch: {}\n", .{max_token});
    if (max_token >= 65) std.debug.print("🚨 CRITICAL: Input batch contains OOB tokens!\n", .{});

    var inputs_list = std.ArrayList([]const u8).init(allocator);
    defer inputs_list.deinit();
    var shapes_list = std.ArrayList([]const i64).init(allocator);
    defer shapes_list.deinit();
    var dtypes_list = std.ArrayList(DType).init(allocator);
    defer dtypes_list.deinit();

    var params_store = std.ArrayList([]u8).init(allocator);
    defer {
        for (params_store.items) |buf| allocator.free(buf);
        params_store.deinit();
    }

    const param_shapes = [_][]const i64{
        &[_]i64{ 65, 64 },
        &[_]i64{ 64, 64 },
        &[_]i64{ 64, 64 },
        &[_]i64{ 64, 64 },
        &[_]i64{ 64, 64 },
        &[_]i64{ 256, 64 },
        &[_]i64{ 64, 256 },
        &[_]i64{ 64, 64 },
        &[_]i64{ 64, 64 },
        &[_]i64{ 64, 64 },
        &[_]i64{ 64, 64 },
        &[_]i64{ 256, 64 },
        &[_]i64{ 64, 256 },
        &[_]i64{ 65, 64 },
    };

    for (param_shapes) |shape| {
        var size: usize = 1;
        for (shape) |dim| size *= @intCast(dim);
        const buf = try allocator.alloc(u8, size * 4);
        const f = std.mem.bytesAsSlice(f32, buf);
        for (f) |*val| val.* = 0.02;
        try params_store.append(buf);
        try inputs_list.append(buf);
        try shapes_list.append(shape);
        try dtypes_list.append(.f32);
    }

    try inputs_list.append(std.mem.sliceAsBytes(batch.x));
    try shapes_list.append(&[_]i64{ 64, 32 });
    try dtypes_list.append(.i64);

    try inputs_list.append(std.mem.sliceAsBytes(batch.y));
    try shapes_list.append(&[_]i64{ 64, 32 });
    try dtypes_list.append(.i64);

    const outputs = try backend.execute(vmfb, "main", inputs_list.items, shapes_list.items, dtypes_list.items);
    defer {
        for (outputs) |o| allocator.free(o);
        allocator.free(outputs);
    }
    const loss = @as(f32, @bitCast(std.mem.readInt(u32, outputs[0][0..4], .little)));

    std.debug.print("📉 Resulting Loss: {d}\n", .{loss});
    if (std.math.isNan(loss)) {
        std.debug.print("❌ FAILURE: Forward pass produced NaN!\n", .{});
        return error.NaNDetected;
    } else {
        std.debug.print("✅ SUCCESS: Forward pass produced valid loss\n", .{});
    }
}
