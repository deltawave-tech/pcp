const std = @import("std");
const pcp = @import("pcp");
const data_loader = pcp.data_loader;

const CharTokenizer = data_loader.CharTokenizer;
const DataLoader = data_loader.DataLoader;

pub fn main() !void {
    std.debug.print("=== Data Pipeline Tests ===\n", .{});
    
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test 1: Tokenizer round-trip
    try testTokenizerRoundTrip(allocator);
    
    // Test 2: DataLoader batch correctness
    try testDataLoaderBatch(allocator);
    
    std.debug.print("=== All Data Pipeline Tests Passed! ===\n", .{});
}

fn testTokenizerRoundTrip(allocator: std.mem.Allocator) !void {
    std.debug.print("Running Tokenizer round-trip test...\n", .{});
    
    // Ensure tiny_shakespeare.txt exists
    var tokenizer = try CharTokenizer.initFromFile(allocator, "tiny_shakespeare.txt");
    defer tokenizer.deinit();
    
    const text_in = "hello world";
    const tokens = try tokenizer.encode(text_in);
    defer allocator.free(tokens);
    
    const text_out = try tokenizer.decode(tokens);
    defer allocator.free(text_out);

    if (!std.mem.eql(u8, text_in, text_out)) {
        return error.TokenizerRoundTripFailed;
    }
    
    std.debug.print("🌙 Tokenizer round-trip test passed.\n", .{});
}

fn testDataLoaderBatch(allocator: std.mem.Allocator) !void {
    std.debug.print("Running DataLoader batch correctness test...\n", .{});
    
    const batch_size = 4;
    const block_size = 8;

    var loader = try DataLoader.init(allocator, "tiny_shakespeare.txt");
    defer loader.deinit();

    const batch = try loader.getBatch(batch_size, block_size);
    defer allocator.free(batch.x);
    defer allocator.free(batch.y);

    // 1. Check if the shapes are correct
    if (batch.x.len != batch_size * block_size) {
        return error.IncorrectBatchXSize;
    }
    if (batch.y.len != batch_size * block_size) {
        return error.IncorrectBatchYSize;
    }

    // 2. The most important test for language modeling:
    //    Verify that y is the shifted version of x
    for (0..batch.x.len - 1) |i| {
        // Skip checks at the boundary of each sequence in the batch
        if ((i + 1) % block_size == 0) continue;
        if (batch.x[i+1] != batch.y[i]) {
            return error.IncorrectShiftedSequence;
        }
    }

    std.debug.print("🌙 DataLoader batch correctness test passed.\n", .{});
}

test "Tokenizer round-trip" {
    const allocator = std.testing.allocator;

    // Ensure tiny_shakespeare.txt exists
    var tokenizer = try CharTokenizer.initFromFile(allocator, "tiny_shakespeare.txt");
    defer tokenizer.deinit();
    
    const text_in = "hello world";
    const tokens = try tokenizer.encode(text_in);
    defer allocator.free(tokens);
    
    const text_out = try tokenizer.decode(tokens);
    defer allocator.free(text_out);

    try std.testing.expectEqualStrings(text_in, text_out);
}

test "DataLoader getBatch correctness" {
    const allocator = std.testing.allocator;
    const batch_size = 4;
    const block_size = 8;

    var loader = try DataLoader.init(allocator, "tiny_shakespeare.txt");
    defer loader.deinit();

    const batch = try loader.getBatch(batch_size, block_size);
    defer allocator.free(batch.x);
    defer allocator.free(batch.y);

    // 1. Check if the shapes are correct
    try std.testing.expectEqual(batch_size * block_size, batch.x.len);
    try std.testing.expectEqual(batch_size * block_size, batch.y.len);

    // 2. The most important test for language modeling:
    //    Verify that y is the shifted version of x
    for (0..batch.x.len - 1) |i| {
        // Skip checks at the boundary of each sequence in the batch
        if ((i + 1) % block_size == 0) continue;
        try std.testing.expectEqual(batch.x[i+1], batch.y[i]);
    }
}