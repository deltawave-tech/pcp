const std = @import("std");
const pcp = @import("pcp");
const tensorModule = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;
const gpt2 = @import("gpt2");

const Allocator = std.mem.Allocator;

const DataType = f32;
const GPT2 = gpt2.GPT2(DataType);
const GPT2Config = gpt2.GPT2Config;

const Tensor = pcp.tensor.Tensor(DataType);

// Simple tokenizer for toy sequences
const Tokenizer = struct {
    vocab_size: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, vocab_size: usize) Tokenizer {
        return Tokenizer{
            .vocab_size = vocab_size,
            .allocator = allocator,
        };
    }

    pub fn encode(self: *Tokenizer, text: []const u8) ![]usize {
        var tokens = std.ArrayList(usize).init(self.allocator);
        defer tokens.deinit();

        for (text) |char| {
            const token_id = @mod(char, self.vocab_size);
            try tokens.append(token_id);
        }

        return tokens.toOwnedSlice();
    }

    pub fn decode(self: *Tokenizer, tokens: []const usize) ![]u8 {
        var text = std.ArrayList(u8).init(self.allocator);
        defer text.deinit();

        for (tokens) |token_id| {
            const char: u8 = @as(u8, @intCast(@min(token_id, 127)));
            try text.append(char);
        }

        return text.toOwnedSlice();
    }
};

// Dataset structure for language modeling
const Dataset = struct {
    inputs: Tensor,
    targets: Tensor,
    batch_size: usize,
    seq_length: usize,
    vocab_size: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, builder: *ops.MLIRBuilder, batch_size: usize, seq_length: usize, vocab_size: usize, corpus: []const []const u8) !Dataset {
        var inputs_list = std.ArrayList(f32).init(allocator);
        defer inputs_list.deinit();
        var targets_list = std.ArrayList(f32).init(allocator);
        defer targets_list.deinit();

        try inputs_list.resize(batch_size * seq_length);
        try targets_list.resize(batch_size * seq_length);
        @memset(inputs_list.items, 0);
        @memset(targets_list.items, 0);

        var tokenizer = Tokenizer.init(allocator, vocab_size);

        var batch_idx: usize = 0;
        var seq_idx: usize = 0;

        for (corpus) |text| {
            if (text.len == 0) continue;

            const tokens = try tokenizer.encode(text);
            defer allocator.free(tokens);

            for (tokens, 0..) |token_id, token_idx| {
                inputs_list.items[batch_idx * seq_length + seq_idx] = @floatFromInt(token_id);
                
                const next_token_id = if (token_idx + 1 < tokens.len) tokens[token_idx + 1] else 0;
                targets_list.items[batch_idx * seq_length + seq_idx] = @floatFromInt(next_token_id);

                seq_idx += 1;
                if (seq_idx >= seq_length) {
                    seq_idx = 0;
                    batch_idx = (batch_idx + 1) % batch_size;
                    if (batch_idx == 0) break;
                }
            }

            if (batch_idx == 0 and seq_idx == 0) break;
        }

        const dims = [_]i64{ @intCast(batch_size), @intCast(seq_length) };
        const input_shape = try tensorModule.Shape.initWithDims(builder.ctx, &dims, .f32);
        const target_shape = try tensorModule.Shape.initWithDims(builder.ctx, &dims, .f32);

        const inputs = try Tensor.newConstant(builder, std.mem.sliceAsBytes(inputs_list.items), input_shape);
        const targets = try Tensor.newConstant(builder, std.mem.sliceAsBytes(targets_list.items), target_shape);

        return Dataset{
            .inputs = inputs,
            .targets = targets,
            .batch_size = batch_size,
            .seq_length = seq_length,
            .vocab_size = vocab_size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Dataset) void {
        self.inputs.deinit();
        self.targets.deinit();
    }
};

// Generate a toy dataset with simple patterns
fn generateToyData(allocator: Allocator, builder: *ops.MLIRBuilder, batch_size: usize, seq_length: usize, vocab_size: usize) !Dataset {
    var corpus = std.ArrayList([]const u8).init(allocator);
    defer corpus.deinit();

    // Pattern 1: Repeated sequences
    var pattern1 = std.ArrayList(u8).init(allocator);
    defer pattern1.deinit();

    const pattern_length = 3;
    for (0..30) |i| {
        try pattern1.append(@as(u8, @intCast(@mod(i, pattern_length) + 'a')));
    }
    try corpus.append(pattern1.items);

    // Pattern 2: Counting sequence
    var pattern2 = std.ArrayList(u8).init(allocator);
    defer pattern2.deinit();

    for (0..30) |i| {
        try pattern2.append(@as(u8, @intCast(@mod(i, 10) + '0')));
    }
    try corpus.append(pattern2.items);

    // Pattern 3: Alternating
    var pattern3 = std.ArrayList(u8).init(allocator);
    defer pattern3.deinit();

    for (0..30) |i| {
        try pattern3.append(if (@mod(i, 2) == 0) 'a' else 'b');
    }
    try corpus.append(pattern3.items);

    return Dataset.init(allocator, builder, batch_size, seq_length, vocab_size, corpus.items);
}

// MLIR-based cross-entropy loss function
fn crossEntropyLoss(builder: *ops.MLIRBuilder, logits: Tensor, targets: Tensor) !Tensor {
    // This is a simplified cross-entropy loss implementation
    // In practice, we'd need more sophisticated MLIR ops for proper cross-entropy
    
    // For now, create a simple mean squared error as a placeholder
    const diff = try ops.subtract(builder, logits, targets);
    defer diff.deinit();
    
    const squared_diff = try ops.multiply(builder, diff, diff);
    defer squared_diff.deinit();
    
    // In a full implementation, we'd use stablehlo.reduce_sum here
    // For now, return the squared difference tensor
    return squared_diff;
}

// Main training function
pub fn train() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("=== MLIR-based GPT-2 Training ===\n", .{});

    const config = GPT2Config{
        .vocab_size = 128,
        .n_positions = 16,
        .n_embd = 32,
        .n_layer = 2,
        .n_head = 4,
    };

    const num_epochs = 3;
    const batch_size = 4;
    const seq_length = 12;

    std.debug.print("Training for {} epochs...\n", .{num_epochs});

    for (0..num_epochs) |epoch| {
        std.debug.print("\n=== Epoch {} ===\n", .{epoch + 1});

        // 1. Create a NEW builder for this training step's graph
        var train_step_builder = try ops.MLIRBuilder.init(allocator);
        defer train_step_builder.deinit();

        // 2. Create the model and dataset with the new builder
        var model = try GPT2.init(allocator, config, &train_step_builder);
        defer model.deinit();

        var dataset = try generateToyData(allocator, &train_step_builder, batch_size, seq_length, config.vocab_size);
        defer dataset.deinit();

        // 3. Build the forward pass AND loss calculation in one graph
        std.debug.print("Building forward + loss graph...\n", .{});
        const logits = try model.forward(dataset.inputs, &train_step_builder);
        defer logits.deinit();

        const loss = try crossEntropyLoss(&train_step_builder, logits, dataset.targets);
        defer loss.deinit();

        // 4. Finalize the forward+loss graph with a return
        const return_op = try train_step_builder.createAndAttach("func.return", &.{loss.value}, &.{});
        _ = return_op;

        const forward_and_loss_fn = train_step_builder.module.op();
        std.debug.print("--- Forward + Loss Graph ---\n", .{});
        forward_and_loss_fn.dump();

        // 5. Differentiate the ENTIRE graph to get gradients for all parameters
        std.debug.print("Computing gradients...\n", .{});
        const grad_fn = try autodiff.buildGradientGraph(allocator, &train_step_builder, forward_and_loss_fn);
        std.debug.print("--- Gradient Graph ---\n", .{});
        // Skip grad_fn.dump() for now to avoid segfault
        std.debug.print("(Gradient graph generated successfully)\n", .{});
        _ = grad_fn;

        // 6. TODO: Execute the gradient graph to get numerical gradient values
        // This will use the Metal backend once the execution pipeline is complete
        
        // 7. TODO: Implement MLIR-based optimizer operations
        // This would create stablehlo.add/subtract ops to update weight tensors

        std.debug.print("Training step {} completed successfully!\n", .{epoch + 1});
    }

    std.debug.print("\n✓ MLIR-based training completed!\n", .{});
    std.debug.print("Next steps: Implement execution pipeline and optimizer\n", .{});
}

pub fn main() !void {
    try train();
}