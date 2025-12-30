const std = @import("std");
const mlir = @import("wrapper.zig");
const c = @import("c.zig").c;

pub const ModelMetadata = struct {
    parameter_shapes: [][]i64,
    data_input_shapes: [][]i64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ModelMetadata) void {
        for (self.parameter_shapes) |s| self.allocator.free(s);
        self.allocator.free(self.parameter_shapes);
        for (self.data_input_shapes) |s| self.allocator.free(s);
        self.allocator.free(self.data_input_shapes);
    }
};

pub const ModelInspector = struct {
    /// Inspects an MLIR source string to extract parameter and input shapes.
    /// Assumes the entry point is named "main".
    /// Assumes the last `num_data_inputs` arguments are data, the rest are trainable parameters.
    pub fn inspect(
        allocator: std.mem.Allocator,
        ctx: mlir.Context,
        mlir_source: []const u8,
        num_data_inputs: usize,
    ) !ModelMetadata {
        // 1. Parse module temporarily
        const temp_module = try mlir.Module.parse(ctx, mlir_source);
        defer temp_module.deinit();

        // 2. Find main function
        const forward_fn = try temp_module.findFunction("main");
        const func_type = forward_fn.getType().as(mlir.FunctionType) orelse return error.NotAFunctionType;

        // 3. Validate inputs
        if (func_type.getNumInputs() < num_data_inputs) {
            return error.NotEnoughInputsInModel;
        }
        const num_params = func_type.getNumInputs() - num_data_inputs;

        // 4. Extract Parameter Shapes
        var param_shapes = std.ArrayList([]i64).init(allocator);
        errdefer {
            for (param_shapes.items) |s| allocator.free(s);
            param_shapes.deinit();
        }

        for (0..num_params) |i| {
            const input_type = func_type.getInput(i);
            const ranked_type = input_type.as(mlir.RankedTensorType) orelse return error.ParameterIsNotATensor;
            const shape = try ranked_type.getShape(allocator);
            try param_shapes.append(shape);
        }

        // 5. Extract Data Input Shapes
        var data_shapes = std.ArrayList([]i64).init(allocator);
        errdefer {
            for (data_shapes.items) |s| allocator.free(s);
            data_shapes.deinit();
        }

        for (num_params..func_type.getNumInputs()) |i| {
            const input_type = func_type.getInput(i);
            const ranked_type = input_type.as(mlir.RankedTensorType) orelse return error.DataInputIsNotATensor;
            const shape = try ranked_type.getShape(allocator);
            try data_shapes.append(shape);
        }

        return ModelMetadata{
            .parameter_shapes = try param_shapes.toOwnedSlice(),
            .data_input_shapes = try data_shapes.toOwnedSlice(),
            .allocator = allocator,
        };
    }
};
