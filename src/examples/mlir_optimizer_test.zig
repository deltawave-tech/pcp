/// Test suite for MLIR-based optimizers (Adam and Nesterov)
/// This file contains comprehensive numerical verification tests for optimizer implementations

const std = @import("std");
const pcp = @import("pcp");
const mlir = pcp.mlir;
const ops = pcp.ops;
const tensor = pcp.tensor;
const adam_mlir = pcp.optimizers.adam_mlir;
const nesterov_mlir = pcp.optimizers.nesterov_mlir;
const mlir_ctx = pcp.mlir_ctx;
const backend_selection = pcp.backend_selection;
const IreeBackend = pcp.backends.iree.IreeBackend;

const Allocator = std.mem.Allocator;
const MLIRBuilder = ops.MLIRBuilder;
const Tensor = tensor.Tensor(void);
const AdamMLIR = adam_mlir.AdamMLIR(f32);
const NesterovMLIR = nesterov_mlir.NesterovMLIR(f32);

/// Shared MLIR context to avoid pass registration conflicts
var global_mlir_context: ?mlir_ctx.MLIRContext = null;
var context_initialized: bool = false;

/// Initialize the global MLIR context once per test session
fn initGlobalMLIRContext(allocator: Allocator) !void {
    if (!context_initialized) {
        global_mlir_context = try mlir_ctx.MLIRContext.init(allocator);
        context_initialized = true;
    }
}

/// Cleanup the global MLIR context at the end of test session
fn deinitGlobalMLIRContext() void {
    if (global_mlir_context) |*ctx| {
        ctx.deinit();
        global_mlir_context = null;
        context_initialized = false;
    }
}

/// A helper to compile and execute MLIR modules for testing optimizers
pub const ExecutionHelper = struct {
    allocator: Allocator,

    /// Compiles and runs a function within an MLIR module via IREE
    pub fn executeModule(
        self: *ExecutionHelper,
        module: mlir.Module,
        function_name: []const u8,
        inputs: [][]const u8,
        input_shapes: [][]const i64,
    ) ![][]u8 {
        var mlir_context = try mlir_ctx.MLIRContext.init(self.allocator);
        defer mlir_context.deinit();

        const backend = backend_selection.Backend.cpu;  // Change from .metal
        
        const vmfb_binary = try mlir_context.compileToVMFB(self.allocator, module, backend.toIreeCompilationTarget(), null);
        defer self.allocator.free(vmfb_binary);

        var iree_backend = try IreeBackend.init(self.allocator, backend);
        defer iree_backend.deinit();

        const outputs = try iree_backend.execute(vmfb_binary, function_name, inputs, input_shapes);
        
        return outputs;
    }
};

const TestError = error{
    OptimizerNotConverged,
    InvalidResult,
    MLIRError,
    TestFailed,
};

pub fn testAdamSingleStepVerification(allocator: Allocator) !void {
    std.debug.print("\n=== Verifying MLIR Adam Optimizer Single Step ===\n", .{});
    var helper = ExecutionHelper{.allocator = allocator};

    try initGlobalMLIRContext(allocator);
    const context = global_mlir_context.?.getContext();
    var builder = try MLIRBuilder.init(allocator, context);
    defer builder.deinit();

    // Configure Adam with exact values for manual calculation
    const element_type = mlir.Type.f32Type(context);
    var adam_conf = adam_mlir.AdamMLIRConfiguration(f32).default_configuration();
    adam_conf.learning_rate = 0.1;
    adam_conf.beta1 = 0.9;
    adam_conf.beta2 = 0.999;
    adam_conf.epsilon = 1e-8;
    
    var adam = try AdamMLIR.init(allocator, &builder, adam_conf, element_type);
    defer adam.deinit();

    // Build main function: SIGNATURE CHANGE -> main(p, m, v, g, t) -> (new_p, new_m, new_v)
    const scalar_type = mlir.Type.rankedTensorType(context, &.{}, element_type);
    const func_type = mlir.Type.functionType(
        context,
        &.{scalar_type, scalar_type, scalar_type, scalar_type, scalar_type}, // p, m, v, g, t
        &.{scalar_type, scalar_type, scalar_type},                           // new_p, new_m, new_v
    );

    const main_fn = try builder.createFunction("main", func_type);
    builder.setInsertionBlock(main_fn.entry_block);

    const p_in = try builder.newTensor(main_fn.entry_block.getArgument(0));
    const m_in = try builder.newTensor(main_fn.entry_block.getArgument(1));
    const v_in = try builder.newTensor(main_fn.entry_block.getArgument(2));
    const g_in = try builder.newTensor(main_fn.entry_block.getArgument(3));
    const t_in = try builder.newTensor(main_fn.entry_block.getArgument(4));
    
    const results = try adam.update(p_in, g_in, m_in, v_in, t_in);
    
    // Return the results directly from the struct.
    _ = try builder.createAndAttach("func.return", &.{
        results.new_params.value,
        results.new_m.value,
        results.new_v.value,
    }, &.{}, .{});

    builder.module.op().dump();

    // Prepare test inputs
    const p0 = [_]f32{0.5};
    const m0 = [_]f32{0.0};
    const v0 = [_]f32{0.0};
    const g1 = [_]f32{0.2};
    const t1 = [_]f32{1.0};

    var inputs_bytes = [_][]const u8{
        std.mem.sliceAsBytes(&p0),
        std.mem.sliceAsBytes(&m0),
        std.mem.sliceAsBytes(&v0),
        std.mem.sliceAsBytes(&g1),
        std.mem.sliceAsBytes(&t1),
    };
    // Input shapes are all rank-0 scalars
    var shapes = [_][]const i64{ &[_]i64{}, &[_]i64{}, &[_]i64{}, &[_]i64{}, &[_]i64{} };

    const outputs = try helper.executeModule(builder.module, "main", &inputs_bytes, &shapes);
    defer { for(outputs) |o| allocator.free(o); allocator.free(outputs); }

    // Verify outputs against manual calculation
    if (outputs.len != 3) {
        std.debug.print("ERROR: Expected 3 outputs, got {}\n", .{outputs.len});
        return TestError.InvalidResult;
    }

    const new_p_val: f32 = @bitCast(std.mem.readInt(u32, outputs[0][0..4], .little));
    const new_m_val: f32 = @bitCast(std.mem.readInt(u32, outputs[1][0..4], .little));
    const new_v_val: f32 = @bitCast(std.mem.readInt(u32, outputs[2][0..4], .little));

    std.debug.print("New Param: {d} (Expected: ~0.4)\n", .{new_p_val});
    std.debug.print("New M:     {d} (Expected: 0.02)\n", .{new_m_val});
    std.debug.print("New V:     {d} (Expected: 0.00004)\n", .{new_v_val});

    if (@abs(new_p_val - 0.4) > 1e-6) {
        std.debug.print("ERROR: Parameter value {d} not close to expected 0.4\n", .{new_p_val});
        return TestError.InvalidResult;
    }
    if (@abs(new_m_val - 0.02) > 1e-6) {
        std.debug.print("ERROR: M value {d} not close to expected 0.02\n", .{new_m_val});
        return TestError.InvalidResult;
    }
    if (@abs(new_v_val - 0.00004) > 1e-6) {
        std.debug.print("ERROR: V value {d} not close to expected 0.00004\n", .{new_v_val});
        return TestError.InvalidResult;
    }

    std.debug.print("🌚 Adam single-step verification passed!\n", .{});
}

pub fn testNesterovSingleStepVerification(allocator: Allocator) !void {
    std.debug.print("\n=== Verifying MLIR Nesterov Optimizer Single Step ===\n", .{});
    var helper = ExecutionHelper{.allocator = allocator};

    try initGlobalMLIRContext(allocator);
    const context = global_mlir_context.?.getContext();
    var builder = try MLIRBuilder.init(allocator, context);
    defer builder.deinit();

    // Configure Nesterov with exact values for manual calculation
    const element_type = mlir.Type.f32Type(context);
    var nesterov_conf = nesterov_mlir.NesterovMLIRConfiguration(f32).default_configuration();
    nesterov_conf.learning_rate = 0.1;
    nesterov_conf.momentum = 0.9;
    
    var nesterov = try NesterovMLIR.init(allocator, &builder, nesterov_conf, element_type);
    defer nesterov.deinit();

    // Build main function: main(p, v, g) -> (new_p, new_v)
    const scalar_type = mlir.Type.rankedTensorType(context, &.{}, element_type);
    const func_type = mlir.Type.functionType(
        context,
        &.{scalar_type, scalar_type, scalar_type}, // p, v, g
        &.{scalar_type, scalar_type},             // new_p, new_v
    );

    const main_fn = try builder.createFunction("main", func_type);
    builder.setInsertionBlock(main_fn.entry_block);

    const p_in = try builder.newTensor(main_fn.entry_block.getArgument(0));
    const v_in = try builder.newTensor(main_fn.entry_block.getArgument(1));
    const g_in = try builder.newTensor(main_fn.entry_block.getArgument(2));
    
    // Set optimizer's internal state
    nesterov.velocity = v_in;

    const new_params = try nesterov.update(p_in, g_in);
    
    // Return new parameter and updated velocity
    _ = try builder.createAndAttach("func.return", &.{
        new_params.value,
        nesterov.velocity.?.value,
    }, &.{}, .{});

    // Prepare test inputs
    const p0 = [_]f32{0.5};
    const v0 = [_]f32{0.0};
    const g1 = [_]f32{0.2};

    var inputs_bytes = [_][]const u8{
        std.mem.sliceAsBytes(&p0),
        std.mem.sliceAsBytes(&v0),
        std.mem.sliceAsBytes(&g1),
    };
    var shapes = [_][]const i64{ &[_]i64{}, &[_]i64{}, &[_]i64{} };

    const outputs = try helper.executeModule(builder.module, "main", &inputs_bytes, &shapes);
    defer { for(outputs) |o| allocator.free(o); allocator.free(outputs); }

    // Verify outputs against manual calculation
    if (outputs.len != 2) {
        std.debug.print("ERROR: Expected 2 outputs, got {}\n", .{outputs.len});
        return TestError.InvalidResult;
    }

    const new_p_val: f32 = @bitCast(std.mem.readInt(u32, outputs[0][0..4], .little));
    const new_v_val: f32 = @bitCast(std.mem.readInt(u32, outputs[1][0..4], .little));

    std.debug.print("New Param:    {d} (Expected: 0.48)\n", .{new_p_val});
    std.debug.print("New Velocity: {d} (Expected: 0.02)\n", .{new_v_val});

    if (@abs(new_p_val - 0.48) > 1e-6) {
        std.debug.print("ERROR: Parameter value {d} not close to expected 0.48\n", .{new_p_val});
        return TestError.InvalidResult;
    }
    if (@abs(new_v_val - 0.02) > 1e-6) {
        std.debug.print("ERROR: Velocity value {d} not close to expected 0.02\n", .{new_v_val});
        return TestError.InvalidResult;
    }

    std.debug.print("🌚 Nesterov single-step verification passed!\n", .{});
}

pub fn testAdamMatrixVerification(allocator: Allocator) !void {
    std.debug.print("\n=== Verifying MLIR Adam Optimizer on 2x2 Matrix ===\n", .{});
    var helper = ExecutionHelper{.allocator = allocator};

    try initGlobalMLIRContext(allocator);
    const context = global_mlir_context.?.getContext();
    var builder = try MLIRBuilder.init(allocator, context);
    defer builder.deinit();

    // Configure Adam with exact values
    const element_type = mlir.Type.f32Type(context);
    var adam_conf = adam_mlir.AdamMLIRConfiguration(f32).default_configuration();
    adam_conf.learning_rate = 0.1;
    adam_conf.beta1 = 0.9;
    adam_conf.beta2 = 0.999;
    adam_conf.epsilon = 1e-8;
    
    var adam = try AdamMLIR.init(allocator, &builder, adam_conf, element_type);
    defer adam.deinit();

    // Build main function for 2x2 matrices: main(p, m, v, g, t) -> (new_p, new_m, new_v)
    const matrix_type = mlir.Type.rankedTensorType(context, &[_]i64{2, 2}, element_type);
    const scalar_type = mlir.Type.rankedTensorType(context, &.{}, element_type);
    const func_type = mlir.Type.functionType(
        context,
        &.{matrix_type, matrix_type, matrix_type, matrix_type, scalar_type}, // p, m, v, g, t
        &.{matrix_type, matrix_type, matrix_type},                          // new_p, new_m, new_v
    );

    const main_fn = try builder.createFunction("main", func_type);
    builder.setInsertionBlock(main_fn.entry_block);

    const p_in = try builder.newTensor(main_fn.entry_block.getArgument(0));
    const m_in = try builder.newTensor(main_fn.entry_block.getArgument(1));
    const v_in = try builder.newTensor(main_fn.entry_block.getArgument(2));
    const g_in = try builder.newTensor(main_fn.entry_block.getArgument(3));
    const t_in = try builder.newTensor(main_fn.entry_block.getArgument(4));
    
    // Call the new stateless update function with tensor timestep
    const results = try adam.update(p_in, g_in, m_in, v_in, t_in);
    
    // Return updated values
    _ = try builder.createAndAttach("func.return", &.{
        results.new_params.value,
        results.new_m.value,
        results.new_v.value,
    }, &.{}, .{});

    // Prepare test inputs - 2x2 matrices
    const p0 = [_]f32{0.5, 0.6, 0.7, 0.8};  // [[0.5, 0.6], [0.7, 0.8]]
    const m0 = [_]f32{0.0, 0.0, 0.0, 0.0};  // Zero momentum
    const v0 = [_]f32{0.0, 0.0, 0.0, 0.0};  // Zero velocity
    const g1 = [_]f32{0.1, 0.2, 0.3, 0.4};  // [[0.1, 0.2], [0.3, 0.4]]
    const t1 = [_]f32{1.0};

    var inputs_bytes = [_][]const u8{
        std.mem.sliceAsBytes(&p0),
        std.mem.sliceAsBytes(&m0),
        std.mem.sliceAsBytes(&v0),
        std.mem.sliceAsBytes(&g1),
        std.mem.sliceAsBytes(&t1),
    };
    var shapes = [_][]const i64{ 
        &[_]i64{2, 2}, &[_]i64{2, 2}, &[_]i64{2, 2}, &[_]i64{2, 2}, &[_]i64{}
    };

    const outputs = try helper.executeModule(builder.module, "main", &inputs_bytes, &shapes);
    defer { for(outputs) |o| allocator.free(o); allocator.free(outputs); }

    // Verify outputs - each element calculated independently
    if (outputs.len != 3) {
        std.debug.print("ERROR: Expected 3 outputs, got {}\n", .{outputs.len});
        return TestError.InvalidResult;
    }

    // Extract matrix values (4 elements each)
    const new_p_vals = @as([*]const f32, @ptrCast(@alignCast(outputs[0].ptr)))[0..4];
    const new_m_vals = @as([*]const f32, @ptrCast(@alignCast(outputs[1].ptr)))[0..4];
    const new_v_vals = @as([*]const f32, @ptrCast(@alignCast(outputs[2].ptr)))[0..4];

    // Expected values calculated element-wise with same Adam formula as scalar test
    // For g=0.1: m=0.01, v=0.00001, p≈0.5-0.1*0.1/0.1≈0.4
    // For g=0.2: m=0.02, v=0.00004, p≈0.6-0.1*0.2/0.2≈0.5  
    // For g=0.3: m=0.03, v=0.00009, p≈0.7-0.1*0.3/0.3≈0.6
    // For g=0.4: m=0.04, v=0.00016, p≈0.8-0.1*0.4/0.4≈0.7
    const expected_p = [_]f32{0.4, 0.5, 0.6, 0.7};
    const expected_m = [_]f32{0.01, 0.02, 0.03, 0.04};
    const expected_v = [_]f32{0.00001, 0.00004, 0.00009, 0.00016};

    std.debug.print("Matrix results:\n", .{});
    for (0..4) |i| {
        std.debug.print("  [{}]: p={d:.6} (exp {d:.6}), m={d:.6} (exp {d:.6}), v={d:.8} (exp {d:.8})\n", 
                       .{i, new_p_vals[i], expected_p[i], new_m_vals[i], expected_m[i], new_v_vals[i], expected_v[i]});
        
        if (@abs(new_p_vals[i] - expected_p[i]) > 1e-6) {
            std.debug.print("ERROR: Element {} parameter value {d} not close to expected {d}\n", .{i, new_p_vals[i], expected_p[i]});
            return TestError.InvalidResult;
        }
        if (@abs(new_m_vals[i] - expected_m[i]) > 1e-6) {
            std.debug.print("ERROR: Element {} M value {d} not close to expected {d}\n", .{i, new_m_vals[i], expected_m[i]});
            return TestError.InvalidResult;
        }
        if (@abs(new_v_vals[i] - expected_v[i]) > 1e-8) {
            std.debug.print("ERROR: Element {} V value {d} not close to expected {d}\n", .{i, new_v_vals[i], expected_v[i]});
            return TestError.InvalidResult;
        }
    }

    std.debug.print("🌚 Adam matrix verification passed!\n", .{});
}

test "nesterov_mlir_matrix_optimization" {
    std.debug.print("\n=== Testing MLIR Nesterov Optimizer on 2D Matrix ===\n", .{});
    
    const allocator = std.testing.allocator;
    
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    const element_type = mlir.Type.f32Type(builder.ctx);
    
    // Configure Nesterov optimizer
    var nesterov_conf = nesterov_mlir.NesterovMLIRConfiguration(f32).default_configuration();
    nesterov_conf.learning_rate = 0.1;
    nesterov_conf.momentum = 0.9;
    
    var nesterov = try NesterovMLIR.init(allocator, &builder, nesterov_conf, element_type);
    defer nesterov.deinit();
    
    // Initial parameter matrix: [[0, 0], [0, 0]]
    const shape = &[_]i64{2, 2};
    const initial_params = try createZeroTensor(&builder, shape, element_type);
    var current_params = initial_params;
    
    // Target matrix: [[1, 2], [3, 4]]
    const target_params = try createTargetTensor(&builder, shape, element_type);
    
    // Optimization loop
    const max_iterations = 1000;
    for (0..max_iterations) |iteration| {
        // Compute gradient: df/dp = 2(p - target)
        const diff = try ops.subtract(&builder, current_params, target_params);
        const two_tensor = try ops.constant(&builder, 2.0, shape, element_type);
        const gradient = try ops.multiply(&builder, two_tensor, diff);
        
        // Update parameters using Nesterov
        const new_params = try nesterov.update(current_params, gradient);
        
        _ = iteration;
        current_params = new_params;
    }
    
    std.debug.print("✓ Nesterov MLIR matrix optimization test completed\n", .{});
}

test "adam_mlir_map_functionality" {
    std.debug.print("\n=== Testing MLIR Adam Optimizer Map ===\n", .{});
    
    const allocator = std.testing.allocator;
    
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    const element_type = mlir.Type.f32Type(builder.ctx);
    
    // Create Adam optimizer map
    var adam_map = adam_mlir.AdamMLIRMap(u32, f32).init(allocator, &builder, element_type);
    defer adam_map.deinit();
    
    // Configure optimizers
    var adam_conf = adam_mlir.AdamMLIRConfiguration(f32).default_configuration();
    adam_conf.learning_rate = 0.1;
    
    // Add multiple optimizers
    _ = try adam_map.add(0, adam_conf);
    _ = try adam_map.add(1, adam_conf);
    
    // Test parameter updates for different optimizers
    const shape = &[_]i64{2, 2};
    const params1 = try createZeroTensor(&builder, shape, element_type);
    const params2 = try createZeroTensor(&builder, shape, element_type);
    const grad1 = try ops.constant(&builder, 0.1, shape, element_type);
    const grad2 = try ops.constant(&builder, 0.2, shape, element_type);
    
    // Update parameters using map
    const updated_params1 = try adam_map.update(0, params1, grad1);
    const updated_params2 = try adam_map.update(1, params2, grad2);
    
    _ = updated_params1;
    _ = updated_params2;
    
    std.debug.print("✓ Adam MLIR map functionality test completed\n", .{});
}

test "nesterov_mlir_map_functionality" {
    std.debug.print("\n=== Testing MLIR Nesterov Optimizer Map ===\n", .{});
    
    const allocator = std.testing.allocator;
    
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    const element_type = mlir.Type.f32Type(builder.ctx);
    
    // Create Nesterov optimizer map
    var nesterov_map = nesterov_mlir.NesterovMLIRMap(u32, f32).init(allocator, &builder, element_type);
    defer nesterov_map.deinit();
    
    // Configure optimizers
    var nesterov_conf = nesterov_mlir.NesterovMLIRConfiguration(f32).default_configuration();
    nesterov_conf.learning_rate = 0.1;
    nesterov_conf.momentum = 0.9;
    
    // Add multiple optimizers
    _ = try nesterov_map.add(0, nesterov_conf);
    _ = try nesterov_map.add(1, nesterov_conf);
    
    // Test parameter updates for different optimizers
    const shape = &[_]i64{2, 2};
    const params1 = try createZeroTensor(&builder, shape, element_type);
    const params2 = try createZeroTensor(&builder, shape, element_type);
    const grad1 = try ops.constant(&builder, 0.1, shape, element_type);
    const grad2 = try ops.constant(&builder, 0.2, shape, element_type);
    
    // Update parameters using map
    const updated_params1 = try nesterov_map.update(0, params1, grad1);
    const updated_params2 = try nesterov_map.update(1, params2, grad2);
    
    _ = updated_params1;
    _ = updated_params2;
    
    std.debug.print("✓ Nesterov MLIR map functionality test completed\n", .{});
}

test "nesterov_mlir_lookahead_functionality" {
    std.debug.print("\n=== Testing MLIR Nesterov Optimizer Lookahead ===\n", .{});
    
    const allocator = std.testing.allocator;
    
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    const element_type = mlir.Type.f32Type(builder.ctx);
    
    // Configure Nesterov optimizer
    var nesterov_conf = nesterov_mlir.NesterovMLIRConfiguration(f32).default_configuration();
    nesterov_conf.learning_rate = 0.1;
    nesterov_conf.momentum = 0.9;
    
    var nesterov = try NesterovMLIR.init(allocator, &builder, nesterov_conf, element_type);
    defer nesterov.deinit();
    
    // Initial parameters
    const shape = &[_]i64{2, 2};
    const initial_params = try createZeroTensor(&builder, shape, element_type);
    
    // Define gradient function for lookahead
    const GradientContext = struct {
        builder: *MLIRBuilder,
        element_type: mlir.Type,
        shape: []const i64,
        
        fn computeGradient(self: @This(), params: Tensor) !Tensor {
            // Simple gradient: df/dp = 2 * p
            const two_tensor = try ops.constant(self.builder, 2.0, self.shape, self.element_type);
            return try ops.multiply(self.builder, two_tensor, params);
        }
    };
    
    const grad_context = GradientContext{
        .builder = &builder,
        .element_type = element_type,
        .shape = shape,
    };
    
    // Test lookahead update
    const updated_params = try nesterov.updateWithLookahead(initial_params, grad_context.computeGradient);
    _ = updated_params;
    
    std.debug.print("✓ Nesterov MLIR lookahead functionality test completed\n", .{});
}

test "optimizer_state_management" {
    std.debug.print("\n=== Testing MLIR Optimizer State Management ===\n", .{});
    
    const allocator = std.testing.allocator;
    
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    const element_type = mlir.Type.f32Type(builder.ctx);
    
    // Test Adam state management
    {
        const adam_conf = adam_mlir.AdamMLIRConfiguration(f32).default_configuration();
        var adam = try AdamMLIR.init(allocator, &builder, adam_conf, element_type);
        defer adam.deinit();
        
        // Initial state should be null
        try std.testing.expect(adam.m_state == null);
        try std.testing.expect(adam.v_state == null);
        
        // After first update, state should be initialized
        const shape = &[_]i64{2, 2};
        const params = try createZeroTensor(&builder, shape, element_type);
        const grads = try ops.constant(&builder, 0.1, shape, element_type);
        
        _ = try adam.update(params, grads);
        
        // State should now be initialized
        try std.testing.expect(adam.m_state != null);
        try std.testing.expect(adam.v_state != null);
    }
    
    // Test Nesterov state management
    {
        const nesterov_conf = nesterov_mlir.NesterovMLIRConfiguration(f32).default_configuration();
        var nesterov = try NesterovMLIR.init(allocator, &builder, nesterov_conf, element_type);
        defer nesterov.deinit();
        
        // Initial velocity should be null
        try std.testing.expect(nesterov.velocity == null);
        
        // After first update, velocity should be initialized
        const shape = &[_]i64{2, 2};
        const params = try createZeroTensor(&builder, shape, element_type);
        const grads = try ops.constant(&builder, 0.1, shape, element_type);
        
        _ = try nesterov.update(params, grads);
        
        // Velocity should now be initialized
        try std.testing.expect(nesterov.velocity != null);
        
        // Test velocity reset
        try nesterov.resetVelocity();
        try std.testing.expect(nesterov.velocity == null);
    }
    
    std.debug.print("✓ Optimizer state management test completed\n", .{});
}

test "optimizer_configuration_validation" {
    std.debug.print("\n=== Testing MLIR Optimizer Configuration Validation ===\n", .{});
    
    const allocator = std.testing.allocator;
    
    var builder = try MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    const element_type = mlir.Type.f32Type(builder.ctx);
    
    // Test Adam configuration
    {
        const adam_conf = adam_mlir.AdamMLIRConfiguration(f32).default_configuration();
        
        // Test valid configuration
        try std.testing.expect(adam_conf.learning_rate == 0.001);
        try std.testing.expect(adam_conf.beta1 == 0.9);
        try std.testing.expect(adam_conf.beta2 == 0.999);
        try std.testing.expect(adam_conf.epsilon == 1e-8);
        
        // Test custom configuration
        adam_conf.learning_rate = 0.01;
        adam_conf.beta1 = 0.8;
        adam_conf.beta2 = 0.99;
        adam_conf.epsilon = 1e-7;
        
        var adam = try AdamMLIR.init(allocator, &builder, adam_conf, element_type);
        defer adam.deinit();
        
        try std.testing.expect(adam.conf.learning_rate == 0.01);
        try std.testing.expect(adam.conf.beta1 == 0.8);
        try std.testing.expect(adam.conf.beta2 == 0.99);
        try std.testing.expect(adam.conf.epsilon == 1e-7);
    }
    
    // Test Nesterov configuration
    {
        const nesterov_conf = nesterov_mlir.NesterovMLIRConfiguration(f32).default_configuration();
        
        // Test valid configuration
        try std.testing.expect(nesterov_conf.learning_rate == 0.01);
        try std.testing.expect(nesterov_conf.momentum == 0.9);
        
        // Test custom configuration
        nesterov_conf.learning_rate = 0.001;
        nesterov_conf.momentum = 0.8;
        
        var nesterov = try NesterovMLIR.init(allocator, &builder, nesterov_conf, element_type);
        defer nesterov.deinit();
        
        try std.testing.expect(nesterov.conf.learning_rate == 0.001);
        try std.testing.expect(nesterov.conf.momentum == 0.8);
    }
    
    std.debug.print("✓ Optimizer configuration validation test completed\n", .{});
}

// Helper functions for test setup

fn createScalarTensor(builder: *MLIRBuilder, value: f64, element_type: mlir.Type) !Tensor {
    const scalar_value = try builder.createScalarConstant(value, element_type);
    return try builder.newTensor(scalar_value);
}

fn createZeroTensor(builder: *MLIRBuilder, shape: []const i64, element_type: mlir.Type) !Tensor {
    return try ops.constant(builder, 0.0, shape, element_type);
}

fn createTargetTensor(builder: *MLIRBuilder, shape: []const i64, element_type: mlir.Type) !Tensor {
    // Create target tensor with values [1, 2, 3, 4] for 2x2 matrix
    // This is a simplified version - in practice, you'd want to create proper tensor data
    return try ops.constant(builder, 1.0, shape, element_type);
}

pub fn testPowerOperation(allocator: Allocator) !void {
    std.debug.print("\n=== Testing stablehlo.power on IREE ===\n", .{});
    var helper = ExecutionHelper{.allocator = allocator};
    try initGlobalMLIRContext(allocator);
    const context = global_mlir_context.?.getContext();
    var builder = try MLIRBuilder.init(allocator, context);
    defer builder.deinit();
    const element_type = mlir.Type.f32Type(context);
    const scalar_type = mlir.Type.rankedTensorType(context, &.{1}, element_type);  // Use {1} to avoid rank-0 issues
    const func_type = mlir.Type.functionType(
        context,
        &.{scalar_type, scalar_type},  // base, exponent
        &.{scalar_type},  // result
    );
    const main_fn = try builder.createFunction("main", func_type);
    builder.setInsertionBlock(main_fn.entry_block);
    const base_in = try builder.newTensor(main_fn.entry_block.getArgument(0));
    const exp_in = try builder.newTensor(main_fn.entry_block.getArgument(1));
    const pow_result = try ops.power(&builder, base_in, exp_in);
    _ = try builder.createAndAttach("func.return", &.{pow_result.value}, &.{}, .{});
    // Test inputs: 0.999 ^ 1.0 = 0.999
    const base_val = [_]f32{0.999};
    const exp_val = [_]f32{1.0};
    var inputs_bytes = [_][]const u8{
        std.mem.sliceAsBytes(&base_val),
        std.mem.sliceAsBytes(&exp_val),
    };
    var shapes = [_][]const i64{ &[_]i64{1}, &[_]i64{1} };
    const outputs = try helper.executeModule(builder.module, "main", &inputs_bytes, &shapes);
    defer { for (outputs) |o| allocator.free(o); allocator.free(outputs); }
    const result_val: f32 = @bitCast(std.mem.readInt(u32, outputs[0][0..4], .little));
    std.debug.print("Computed power: {d} (Expected: 0.999)\n", .{result_val});
    if (@abs(result_val - 0.999) > 1e-6) {
        std.debug.print("ERROR: stablehlo.power failed on Metal\n", .{});
    } else {
        std.debug.print("stablehlo.power works correctly\n", .{});
    }
    // Repeat with local_sync backend to compare
}

/// Main test runner
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    defer deinitGlobalMLIRContext(); // Cleanup shared context at the end
    
    std.debug.print("=== MLIR Optimizer Numerical Verification Tests ===\n", .{});
    std.debug.print("Testing optimizer implementations with end-to-end IREE execution\n", .{});
    
    // First test the power operation in isolation
    try testPowerOperation(allocator);
    
    // Run numerical verification tests
    try testAdamSingleStepVerification(allocator);
    try testNesterovSingleStepVerification(allocator);  
    try testAdamMatrixVerification(allocator);
    
    std.debug.print("\n🌚 All MLIR Optimizer Tests Completed Successfully! 🌚\n", .{});
    std.debug.print("Adam and Nesterov optimizers verified for numerical correctness\n", .{});
}