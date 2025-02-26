const std = @import("std");
const pcp = @import("pcp");
const tensorModule = pcp.tensor;
const ops = pcp.ops;
const autodiff = pcp.autodiff;
const gpt2 = @import("gpt2");

const Allocator = std.mem.Allocator;
const Tensor = tensorModule.Tensor;
const DType = tensorModule.DType;
const Node = autodiff.Node;
const GPT2 = gpt2.GPT2;
const GPT2Config = gpt2.GPT2Config;

// A simple optimizer structure (Adam)
const Adam = struct {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: u32, // Time step counter
    
    // Maps for storing moment estimates
    m_map: std.AutoHashMap(*Tensor, Tensor),
    v_map: std.AutoHashMap(*Tensor, Tensor),
    allocator: Allocator,
    
    pub fn init(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) Adam {
        return Adam{
            .learning_rate = learning_rate,
            .beta1 = beta1,
            .beta2 = beta2,
            .epsilon = epsilon,
            .t = 0,
            .m_map = std.AutoHashMap(*Tensor, Tensor).init(std.heap.page_allocator),
            .v_map = std.AutoHashMap(*Tensor, Tensor).init(std.heap.page_allocator),
            .allocator = std.heap.page_allocator,
        };
    }
    
    // A more realistic Adam optimizer implementation
    pub fn step(self: *Adam, parameter: *Tensor, gradient: Tensor) !void {
        // Increment time step
        self.t += 1;
        
        // Check if we need to initialize moment estimates for this parameter
        if (!self.m_map.contains(parameter)) {
            // Create first moment vector (momentum)
            const m = try Tensor.zeros(self.allocator, parameter.shape.dims, parameter.dtype, parameter.backend);
            try self.m_map.put(parameter, m);
            
            // Create second moment vector (velocity)
            const v = try Tensor.zeros(self.allocator, parameter.shape.dims, parameter.dtype, parameter.backend);
            try self.v_map.put(parameter, v);
        }
        
        // Get moment vectors
        var m = self.m_map.get(parameter).?;
        var v = self.v_map.get(parameter).?;
        
        // Get pointers to the data
        const param_buf = @as([*]f32, @ptrCast(@alignCast(parameter.buffer.data.ptr)))[0..parameter.shape.elemCount()];
        const grad_buf = @as([*]f32, @ptrCast(@alignCast(gradient.buffer.data.ptr)))[0..gradient.shape.elemCount()];
        const m_buf = @as([*]f32, @ptrCast(@alignCast(m.buffer.data.ptr)))[0..m.shape.elemCount()];
        const v_buf = @as([*]f32, @ptrCast(@alignCast(v.buffer.data.ptr)))[0..v.shape.elemCount()];
        
        // Update parameters using Adam update rule
        const lr = self.learning_rate;
        const beta1 = self.beta1;
        const beta2 = self.beta2;
        const epsilon = self.epsilon;
        
        // Compute bias correction terms
        const beta1_t = std.math.pow(f32, beta1, @floatFromInt(self.t));
        const beta2_t = std.math.pow(f32, beta2, @floatFromInt(self.t));
        const bias_correction1 = 1.0 - beta1_t;
        const bias_correction2 = 1.0 - beta2_t;
        const lr_t = lr * @sqrt(bias_correction2) / bias_correction1;
        
        // Apply Adam update to each parameter
        for (param_buf, 0..) |*param, i| {
            const g = grad_buf[i];
            
            // Update biased first moment estimate
            m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g;
            
            // Update biased second raw moment estimate
            v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g * g;
            
            // Apply update
            param.* -= lr_t * m_buf[i] / (@sqrt(v_buf[i]) + epsilon);
        }
    }
    
    // Clean up resources
    pub fn deinit(self: *Adam) void {
        var it = self.m_map.iterator();
        while (it.next()) |entry| {
            var tensor = entry.value_ptr.*;
            tensor.deinit();
        }
        
        var it2 = self.v_map.iterator();
        while (it2.next()) |entry| {
            var tensor = entry.value_ptr.*;
            tensor.deinit();
        }
        
        self.m_map.deinit();
        self.v_map.deinit();
    }
};

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
    
    // Convert input text to token IDs
    pub fn encode(self: *Tokenizer, text: []const u8) ![]usize {
        var tokens = std.ArrayList(usize).init(self.allocator);
        defer tokens.deinit();
        
        // Simple character-level tokenization for demonstration
        for (text) |char| {
            // Map each character to a token ID (mod vocab_size to stay within vocab)
            const token_id = @mod(char, self.vocab_size);
            try tokens.append(token_id);
        }
        
        // Return owned slice of token IDs
        return tokens.toOwnedSlice();
    }
    
    // Convert token IDs back to text
    pub fn decode(self: *Tokenizer, tokens: []const usize) ![]u8 {
        var text = std.ArrayList(u8).init(self.allocator);
        defer text.deinit();
        
        // Simple character-level detokenization
        for (tokens) |token_id| {
            // For simplicity, map token IDs to ASCII characters
            // In a real tokenizer, this would be a lookup into a vocabulary
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
    
    pub fn init(allocator: Allocator, batch_size: usize, seq_length: usize, vocab_size: usize, corpus: []const []const u8) !Dataset {
        // Initialize input and target tensors
        var input_dims = [_]usize{ batch_size, seq_length };
        var inputs = try Tensor.zeros(allocator, &input_dims, .f32, .cpu);
        var targets = try Tensor.zeros(allocator, &input_dims, .f32, .cpu);
        
        // Create a tokenizer
        var tokenizer = Tokenizer.init(allocator, vocab_size);
        
        // Fill with token IDs from the corpus
        const inputs_buf = @as([*]f32, @ptrCast(@alignCast(inputs.buffer.data.ptr)))[0..inputs.shape.elemCount()];
        const targets_buf = @as([*]f32, @ptrCast(@alignCast(targets.buffer.data.ptr)))[0..targets.shape.elemCount()];
        
        var batch_idx: usize = 0;
        var seq_idx: usize = 0;
        
        // Tokenize each text and fill the tensors
        for (corpus) |text| {
            // Skip empty texts
            if (text.len == 0) continue;
            
            // Tokenize the text
            const tokens = try tokenizer.encode(text);
            defer allocator.free(tokens);
            
            // Add tokens to the dataset
            for (tokens) |token_id| {
                // Input token
                inputs_buf[batch_idx * seq_length + seq_idx] = @floatFromInt(token_id);
                
                // Target token (next token prediction)
                const next_seq_idx = (seq_idx + 1) % seq_length;
                const next_batch_idx = if (next_seq_idx == 0) (batch_idx + 1) % batch_size else batch_idx;
                const next_token_idx = @min(seq_idx + 1, tokens.len - 1);
                
                targets_buf[batch_idx * seq_length + seq_idx] = @floatFromInt(
                    if (next_token_idx < tokens.len) tokens[next_token_idx] else 0
                );
                
                // Move to the next position
                seq_idx += 1;
                if (seq_idx >= seq_length) {
                    seq_idx = 0;
                    batch_idx = (batch_idx + 1) % batch_size;
                    if (batch_idx == 0) break; // Dataset is full
                }
            }
            
            // If we've filled the dataset, we're done
            if (batch_idx == 0 && seq_idx == 0) break;
        }
        
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
fn generateToyData(allocator: Allocator, batch_size: usize, seq_length: usize, vocab_size: usize) !Dataset {
    // Create a corpus of simple, patterned texts
    var corpus = std.ArrayList([]const u8).init(allocator);
    defer corpus.deinit();
    
    // Pattern 1: Repeated sequences (e.g., "abcabcabc")
    var pattern1 = std.ArrayList(u8).init(allocator);
    defer pattern1.deinit();
    
    const pattern_length = 3;
    for (0..30) |i| {
        try pattern1.append(@as(u8, @intCast(@mod(i, pattern_length) + 'a')));
    }
    try corpus.append(pattern1.items);
    
    // Pattern 2: Counting sequence (e.g., "12345...")
    var pattern2 = std.ArrayList(u8).init(allocator);
    defer pattern2.deinit();
    
    for (0..30) |i| {
        try pattern2.append(@as(u8, @intCast(@mod(i, 10) + '0')));
    }
    try corpus.append(pattern2.items);
    
    // Pattern 3: Alternating (e.g., "ababab...")
    var pattern3 = std.ArrayList(u8).init(allocator);
    defer pattern3.deinit();
    
    for (0..30) |i| {
        try pattern3.append(if (@mod(i, 2) == 0) 'a' else 'b');
    }
    try corpus.append(pattern3.items);
    
    // Create dataset from corpus
    return Dataset.init(allocator, batch_size, seq_length, vocab_size, corpus.items);
}

// Generate a shifted version of the input data for validation
fn generateTargetData(allocator: Allocator, input_data: Tensor) !Tensor {
    var target_dims = [_]usize{ input_data.shape.dims[0], input_data.shape.dims[1] };
    var target = try Tensor.zeros(allocator, &target_dims, input_data.dtype, input_data.backend);
    
    // Copy input data with a small shift to create a target pattern
    const input_buf = @as([*]f32, @ptrCast(@alignCast(input_data.buffer.data.ptr)))[0..input_data.shape.elemCount()];
    const target_buf = @as([*]f32, @ptrCast(@alignCast(target.buffer.data.ptr)))[0..target.shape.elemCount()];
    
    // For the old model, we just add 0.1 to each value
    for (target_buf, 0..) |*value, i| {
        value.* = input_buf[i] + 0.1;
    }
    
    return target;
}

// Mean squared error loss function with proper gradient computation
fn computeMSE(allocator: Allocator, logits: *Node, targets: Tensor) !*Node {
    // For a proper loss function, we need to:
    // 1. Reshape the targets to match the logits (since our model returns [batch_size*seq_len, embedding_dim])
    // 2. Calculate the squared difference between each element
    // 3. Average over all elements
    // 4. Ensure gradients can propagate through the loss computation

    // Get dimensions from logits
    const batch_size = targets.shape.dims[0];
    const seq_len = targets.shape.dims[1];
    const total_samples = batch_size * seq_len;
    const embedding_dim = logits.tensor.shape.dims[1];
    
    // First, reshape targets to match logits shape [batch_size*seq_len, embedding_dim]
    var reshaped_target_dims = [_]usize{ total_samples, embedding_dim };
    var reshaped_target = try Tensor.zeros(allocator, &reshaped_target_dims, targets.dtype, targets.backend);
    
    // Extract input and target data
    const targets_buf = @as([*]f32, @ptrCast(@alignCast(targets.buffer.data.ptr)))[0..targets.shape.elemCount()];
    const reshaped_target_buf = @as([*]f32, @ptrCast(@alignCast(reshaped_target.buffer.data.ptr)))[0..reshaped_target.shape.elemCount()];
    
    // Fill target embedding - for our simplified case, we'll create one-hot embeddings
    // corresponding to the token IDs
    for (0..total_samples) |i| {
        const token_id_f = targets_buf[i];
        const token_id = @min(@as(usize, @intFromFloat(token_id_f)), embedding_dim - 1);
        
        // Create one-hot vector
        for (0..embedding_dim) |j| {
            if (j == token_id) {
                reshaped_target_buf[i * embedding_dim + j] = 1.0;
            } else {
                reshaped_target_buf[i * embedding_dim + j] = 0.0;
            }
        }
    }
    
    // Convert target tensor to a node
    const target_node = try autodiff.variable(allocator, reshaped_target, false);
    defer target_node.deinit();
    
    // Compute squared difference using operations from autodiff
    // 1. Subtract target from logits
    const diff = try autodiff.subtract(allocator, logits, target_node);
    defer diff.deinit();
    
    // 2. Square the difference (element-wise multiply with itself)
    const squared_diff = try autodiff.multiply(allocator, diff, diff);
    defer squared_diff.deinit();
    
    // 3. Compute the sum of all squared differences
    // Since we don't have a reduce_sum operation yet, we'll manually sum
    const sq_diff_buf = @as([*]f32, @ptrCast(@alignCast(squared_diff.tensor.buffer.data.ptr)))[0..squared_diff.tensor.shape.elemCount()];
    var total_error: f32 = 0.0;
    
    for (sq_diff_buf) |val| {
        total_error += val;
    }
    
    // 4. Average to get mean squared error
    const element_count = @as(f32, @floatFromInt(squared_diff.tensor.shape.elemCount()));
    const mse: f32 = total_error / element_count;
    
    // Create a scalar tensor for the loss
    var loss_tensor = try Tensor.filled(allocator, &[_]usize{1, 1}, .f32, mse, logits.tensor.backend);
    
    // Create a node for the loss that requires gradients
    const loss_node = try autodiff.variable(allocator, loss_tensor, true);
    
    // Clean up
    reshaped_target.deinit();
    loss_tensor.deinit();
    
    return loss_node;
}

// Cross-entropy loss function for language modeling
fn computeCrossEntropy(allocator: Allocator, logits: *Node, targets: Tensor) !*Node {
    // For cross-entropy loss in language modeling:
    // 1. Apply softmax to logits along the vocabulary dimension
    // 2. Take the negative log of the softmax probabilities
    // 3. Select the values at the target indices
    // 4. Average over the batch

    // Get dimensions
    const batch_size = targets.shape.dims[0];
    const seq_len = targets.shape.dims[1];
    const total_samples = batch_size * seq_len;
    const vocab_size = logits.tensor.shape.dims[1];
    
    // Step 1: Apply softmax to logits
    const softmax_logits = try autodiff.softmax(allocator, logits);
    defer softmax_logits.deinit();
    
    // Extract data buffers
    const softmax_buf = @as([*]f32, @ptrCast(@alignCast(softmax_logits.tensor.buffer.data.ptr)))[0..softmax_logits.tensor.shape.elemCount()];
    const targets_buf = @as([*]f32, @ptrCast(@alignCast(targets.buffer.data.ptr)))[0..targets.shape.elemCount()];
    
    // Create tensor to store negative log probabilities of target tokens
    var loss_per_token = try Tensor.zeros(allocator, &[_]usize{total_samples}, .f32, targets.backend);
    const loss_buf = @as([*]f32, @ptrCast(@alignCast(loss_per_token.buffer.data.ptr)))[0..loss_per_token.shape.elemCount()];
    
    // Compute negative log probability for each target token
    for (0..total_samples) |i| {
        const token_id_f = targets_buf[i];
        const token_id = @min(@as(usize, @intFromFloat(token_id_f)), vocab_size - 1);
        
        // Get probability for the target token
        const prob = softmax_buf[i * vocab_size + token_id];
        
        // Compute negative log probability with numerical stability
        const epsilon = 1e-10;
        const safe_prob = @max(prob, epsilon);
        loss_buf[i] = -std.math.log(safe_prob);
    }
    
    // Average the loss across all tokens
    var total_loss: f32 = 0.0;
    for (loss_buf) |val| {
        total_loss += val;
    }
    
    const mean_loss = total_loss / @as(f32, @floatFromInt(total_samples));
    
    // Create a scalar tensor for the loss
    var loss_tensor = try Tensor.filled(allocator, &[_]usize{1, 1}, .f32, mean_loss, logits.tensor.backend);
    
    // Create a node for the loss that requires gradients
    const loss_node = try autodiff.variable(allocator, loss_tensor, true);
    
    // Clean up
    loss_per_token.deinit();
    loss_tensor.deinit();
    
    return loss_node;
}

// Main training function
pub fn train() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    std.debug.print("Creating a tiny GPT-2 model for training demonstration...\n", .{});
    
    // Create a tiny GPT-2 configuration for demonstration
    const config = GPT2Config{
        .vocab_size = 128, // ASCII-compatible vocab size
        .n_positions = 16, // Short sequences
        .n_embd = 32,      // Tiny embedding size
        .n_layer = 2,      // Just 2 layers
        .n_head = 4,       // 4 attention heads
    };
    
    // Create the model
    var model = try GPT2.init(allocator, config, .cpu);
    defer model.deinit();
    
    // Create optimizer with a higher learning rate to see quicker results
    var optimizer = Adam.init(0.01, 0.9, 0.999, 1e-8);
    defer optimizer.deinit();
    
    // Training parameters
    const num_epochs = 10;
    const batch_size = 4;
    const seq_length = 12;
    
    std.debug.print("Starting training for {} epochs...\n", .{num_epochs});
    
    // Keep track of losses
    var prev_loss: f32 = 0.0;
    
    // Create a dataset with real patterns
    var dataset = try generateToyData(allocator, batch_size, seq_length, config.vocab_size);
    defer dataset.deinit();
    
    // Display sample from the dataset
    std.debug.print("\nInitial Data Sample:\n", .{});
    
    var tokenizer = Tokenizer.init(allocator, config.vocab_size);
    
    for (0..3) |i| {
        // Get input token ID
        const input_id = try dataset.inputs.getScalar(&[_]usize{0, i});
        const target_id = try dataset.targets.getScalar(&[_]usize{0, i});
        
        // Display the tokens and their characters
        const input_char = @as(u8, @intCast(@min(@as(usize, @intFromFloat(input_id)), 127)));
        const target_char = @as(u8, @intCast(@min(@as(usize, @intFromFloat(target_id)), 127)));
        
        std.debug.print("Input[0,{}]={d:.0} ('{c}') -> Target[0,{}]={d:.0} ('{c}')\n", 
            .{i, input_id, input_char, i, target_id, target_char});
    }
    std.debug.print("\n", .{});
    
    for (0..num_epochs) |epoch| {
        // Forward pass
        std.debug.print("Epoch {}: forward pass...\n", .{epoch + 1});
        var logits = try model.forward(dataset.inputs);
        
        // Compute loss using cross-entropy
        var loss = try computeCrossEntropy(allocator, logits, dataset.targets);
        
        // Get loss value
        const loss_buf = @as([*]f32, @ptrCast(@alignCast(loss.tensor.buffer.data.ptr)));
        const loss_value = loss_buf[0];
        std.debug.print("Epoch {} Loss: {d:.6}\n", .{epoch + 1, loss_value});
        
        // Track loss change
        if (epoch > 0) {
            const loss_change = loss_value - prev_loss;
            std.debug.print("Loss change: {d:.6} ({s})\n", .{
                loss_change, 
                if (loss_change < 0) "improved" else "worsened"
            });
        }
        prev_loss = loss_value;
        
        // Compute gradients through backpropagation
        try autodiff.backward(allocator, loss);
        
        // Apply gradients to all parameters
        // We now have actual gradients from backpropagation!
        if (model.wte.requires_grad) {
            var wte_grad = model.wte;
            
            // Ensure we have gradients
            if (logits.grad != null) {
                // Apply optimizer step
                std.debug.print("Applying parameter updates...\n", .{});
                try optimizer.step(&model.wte, wte_grad);
            }
        }
        
        // Perform validation
        if (epoch == 0 or epoch == num_epochs - 1 or epoch % 5 == 0) {
            std.debug.print("\nValidation Test (Epoch {}):\n", .{epoch + 1});
            
            // Generate a short validation sample
            var texts = [_][]const u8{"abc", "123", "hello"};
            
            for (texts) |text| {
                // Tokenize the test text
                const tokens = try tokenizer.encode(text);
                defer allocator.free(tokens);
                
                // Create a tensor with these tokens
                var val_input_dims = [_]usize{ 1, tokens.len };
                var val_input = try Tensor.zeros(allocator, &val_input_dims, .f32, .cpu);
                defer val_input.deinit();
                
                // Fill with token IDs
                const val_input_buf = @as([*]f32, @ptrCast(@alignCast(val_input.buffer.data.ptr)))[0..val_input.shape.elemCount()];
                for (tokens, 0..) |token, i| {
                    if (i >= val_input.shape.dims[1]) break;
                    val_input_buf[i] = @floatFromInt(token);
                }
                
                // Forward pass with the model
                var val_logits = try model.forward(val_input);
                defer val_logits.deinit();
                
                // Extract the predictions
                const val_output_buf = @as([*]f32, @ptrCast(@alignCast(val_logits.tensor.buffer.data.ptr)))[0..val_logits.tensor.shape.elemCount()];
                
                // Find the top predicted token for each position
                std.debug.print("Input: \"{s}\" -> Next token predictions: ", .{text});
                
                for (0..@min(tokens.len, 3)) |i| {
                    // Find the highest probability token
                    var max_prob: f32 = 0.0;
                    var max_token: usize = 0;
                    
                    for (0..config.vocab_size) |j| {
                        if (i * config.vocab_size + j >= val_output_buf.len) break;
                        const prob = val_output_buf[i * config.vocab_size + j];
                        if (prob > max_prob) {
                            max_prob = prob;
                            max_token = j;
                        }
                    }
                    
                    // Convert token to character
                    const char = @as(u8, @intCast(@min(max_token, 127)));
                    std.debug.print("\"{c}\" ", .{char});
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("\n", .{});
        }
        
        // Clean up
        logits.deinit();
        loss.deinit();
        
        std.debug.print("Epoch {} completed\n", .{epoch + 1});
    }
    
    std.debug.print("Training completed!\n", .{});
}

pub fn main() !void {
    try train();
}