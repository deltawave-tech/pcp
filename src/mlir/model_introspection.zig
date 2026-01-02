const std = @import("std");
const mlir = @import("wrapper.zig");
const c = @import("c.zig").c;
const tensor = @import("../core/tensor.zig");

pub const ModelMetadata = struct {
    parameter_shapes: [][]i64,
    data_input_shapes: [][]i64,
    data_input_dtypes: []tensor.DType,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ModelMetadata) void {
        for (self.parameter_shapes) |s| self.allocator.free(s);
        self.allocator.free(self.parameter_shapes);
        for (self.data_input_shapes) |s| self.allocator.free(s);
        self.allocator.free(self.data_input_shapes);
        self.allocator.free(self.data_input_dtypes);
    }

    /// Save metadata to a JSON cache file for fast loading
    pub fn saveToFile(self: ModelMetadata, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Convert dtypes to strings
        var dtypes_str = try std.ArrayList([]const u8).initCapacity(self.allocator, self.data_input_dtypes.len);
        defer dtypes_str.deinit();

        for (self.data_input_dtypes) |dt| {
            const s = switch (dt) {
                .f32 => "f32",
                .f64 => "f64",
                .f16 => "f16",
                .bf16 => "bf16",
                .i32 => "i32",
                .i64 => "i64",
                .bool => "bool",
            };
            try dtypes_str.append(s);
        }

        // Build JSON manually to avoid complex serialization
        var buf = std.ArrayList(u8).init(self.allocator);
        defer buf.deinit();
        var writer = buf.writer();

        try writer.writeAll("{\n  \"parameter_shapes\": [\n");
        for (self.parameter_shapes, 0..) |shape, i| {
            try writer.writeAll("    [");
            for (shape, 0..) |dim, j| {
                try writer.print("{d}", .{dim});
                if (j < shape.len - 1) try writer.writeAll(", ");
            }
            try writer.writeAll("]");
            if (i < self.parameter_shapes.len - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }
        try writer.writeAll("  ],\n  \"data_input_shapes\": [\n");
        for (self.data_input_shapes, 0..) |shape, i| {
            try writer.writeAll("    [");
            for (shape, 0..) |dim, j| {
                try writer.print("{d}", .{dim});
                if (j < shape.len - 1) try writer.writeAll(", ");
            }
            try writer.writeAll("]");
            if (i < self.data_input_shapes.len - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }
        try writer.writeAll("  ],\n  \"data_input_dtypes\": [\n");
        for (dtypes_str.items, 0..) |dtype_str, i| {
            try writer.print("    \"{s}\"", .{dtype_str});
            if (i < dtypes_str.items.len - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }
        try writer.writeAll("  ]\n}\n");

        try file.writeAll(buf.items);
    }

    /// Load metadata from a JSON cache file
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !ModelMetadata {
        const data = try std.fs.cwd().readFileAlloc(allocator, path, 10 * 1024 * 1024); // 10MB limit
        defer allocator.free(data);

        const Parsed = struct {
            parameter_shapes: [][]i64,
            data_input_shapes: [][]i64,
            data_input_dtypes: [][]const u8,
        };

        const parsed = try std.json.parseFromSlice(Parsed, allocator, data, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        // Deep copy everything since parsed memory is tied to arena
        var param_shapes = try allocator.alloc([]i64, parsed.value.parameter_shapes.len);
        for (parsed.value.parameter_shapes, 0..) |s, i| {
            param_shapes[i] = try allocator.dupe(i64, s);
        }

        var data_shapes = try allocator.alloc([]i64, parsed.value.data_input_shapes.len);
        for (parsed.value.data_input_shapes, 0..) |s, i| {
            data_shapes[i] = try allocator.dupe(i64, s);
        }

        var dtypes = try allocator.alloc(tensor.DType, parsed.value.data_input_dtypes.len);
        for (parsed.value.data_input_dtypes, 0..) |s, i| {
            if (std.mem.eql(u8, s, "i64")) dtypes[i] = .i64
            else if (std.mem.eql(u8, s, "i32")) dtypes[i] = .i32
            else if (std.mem.eql(u8, s, "f32")) dtypes[i] = .f32
            else if (std.mem.eql(u8, s, "f16")) dtypes[i] = .f16
            else if (std.mem.eql(u8, s, "f64")) dtypes[i] = .f64
            else if (std.mem.eql(u8, s, "bf16")) dtypes[i] = .bf16
            else if (std.mem.eql(u8, s, "bool")) dtypes[i] = .bool
            else dtypes[i] = .f32; // Default
        }

        return ModelMetadata{
            .parameter_shapes = param_shapes,
            .data_input_shapes = data_shapes,
            .data_input_dtypes = dtypes,
            .allocator = allocator,
        };
    }
};

pub const ModelInspector = struct {
    const ArgInfo = struct { shape: []i64, dtype: tensor.DType };

    /// Fast textual parser that avoids loading the full MLIR context.
    /// Finds `func.func @main(...)` and extracts types from the signature.
    /// This is orders of magnitude faster than full MLIR parsing for large models.
    pub fn inspectLite(
        allocator: std.mem.Allocator,
        mlir_source: []const u8,
        num_data_inputs: usize,
    ) !ModelMetadata {
        // Find function signature
        const sig_marker = "func.func @main";
        const idx = std.mem.indexOf(u8, mlir_source, sig_marker) orelse return error.MainFunctionNotFound;

        const open_paren = std.mem.indexOfPos(u8, mlir_source, idx, "(") orelse return error.InvalidSignature;
        // Find matching `) ->` or `) attributes`
        const close_marker = std.mem.indexOfPos(u8, mlir_source, open_paren, ")") orelse return error.InvalidSignature;

        const args_str = mlir_source[open_paren + 1 .. close_marker];

        // Parse arguments
        var all_args = std.ArrayList(ArgInfo).init(allocator);
        defer {
            // Cleanup only if we error out; otherwise ownership transfers to result
            for (all_args.items) |arg| allocator.free(arg.shape);
            all_args.deinit();
        }

        // Split by comma, but need to handle nested commas in tensor<>
        var arg_start: usize = 0;
        var depth: usize = 0;
        var i: usize = 0;

        while (i < args_str.len) : (i += 1) {
            if (args_str[i] == '<') {
                depth += 1;
            } else if (args_str[i] == '>') {
                depth -= 1;
            } else if (args_str[i] == ',' and depth == 0) {
                // Parse this argument
                const arg_text = std.mem.trim(u8, args_str[arg_start..i], " \n\t");
                if (arg_text.len > 0) {
                    if (try parseArgument(allocator, arg_text)) |arg_info| {
                        try all_args.append(arg_info);
                    }
                }
                arg_start = i + 1;
            }
        }

        // Parse last argument
        const last_arg = std.mem.trim(u8, args_str[arg_start..], " \n\t");
        if (last_arg.len > 0) {
            if (try parseArgument(allocator, last_arg)) |arg_info| {
                try all_args.append(arg_info);
            }
        }

        if (all_args.items.len < num_data_inputs) return error.NotEnoughInputs;

        const num_params = all_args.items.len - num_data_inputs;

        // Separate into params and data
        var param_shapes = try allocator.alloc([]i64, num_params);
        var data_shapes = try allocator.alloc([]i64, num_data_inputs);
        var data_dtypes = try allocator.alloc(tensor.DType, num_data_inputs);

        for (all_args.items, 0..) |arg, arg_idx| {
            if (arg_idx < num_params) {
                param_shapes[arg_idx] = arg.shape;
            } else {
                data_shapes[arg_idx - num_params] = arg.shape;
                data_dtypes[arg_idx - num_params] = arg.dtype;
            }
        }

        // Clear the defer cleanup since we've transferred ownership
        all_args.clearRetainingCapacity();

        return ModelMetadata{
            .parameter_shapes = param_shapes,
            .data_input_shapes = data_shapes,
            .data_input_dtypes = data_dtypes,
            .allocator = allocator,
        };
    }

    fn parseArgument(allocator: std.mem.Allocator, arg_text: []const u8) !?ArgInfo {
        // Expected format: %arg0: tensor<1024x1024xf32>
        const colon_idx = std.mem.indexOf(u8, arg_text, ":") orelse return null;
        const type_part = std.mem.trim(u8, arg_text[colon_idx + 1 ..], " \n\t");

        if (!std.mem.startsWith(u8, type_part, "tensor<")) return null;
        const end_angle = std.mem.lastIndexOf(u8, type_part, ">") orelse return null;
        const content = type_part[7..end_angle]; // Inside tensor<...>

        // Parse: 2x3xf32 or 2x3xi64
        var dims = std.ArrayList(i64).init(allocator);
        errdefer dims.deinit();

        var dtype: tensor.DType = .f32;
        var it = std.mem.tokenizeScalar(u8, content, 'x');

        while (it.next()) |part| {
            if (std.fmt.parseInt(i64, part, 10)) |d| {
                try dims.append(d);
            } else |_| {
                // Must be the type
                if (std.mem.eql(u8, part, "f32")) dtype = .f32
                else if (std.mem.eql(u8, part, "i64")) dtype = .i64
                else if (std.mem.eql(u8, part, "i32")) dtype = .i32
                else if (std.mem.eql(u8, part, "f16")) dtype = .f16
                else if (std.mem.eql(u8, part, "bf16")) dtype = .bf16
                else if (std.mem.eql(u8, part, "f64")) dtype = .f64
                else if (std.mem.eql(u8, part, "i1")) dtype = .bool;
            }
        }

        return ArgInfo{ .shape = try dims.toOwnedSlice(), .dtype = dtype };
    }

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
        var data_dtypes = std.ArrayList(tensor.DType).init(allocator);
        errdefer {
            for (data_shapes.items) |s| allocator.free(s);
            data_shapes.deinit();
            data_dtypes.deinit();
        }

        for (num_params..func_type.getNumInputs()) |i| {
            const input_type = func_type.getInput(i);
            const ranked_type = input_type.as(mlir.RankedTensorType) orelse return error.DataInputIsNotATensor;
            const shape = try ranked_type.getShape(allocator);
            try data_shapes.append(shape);

            // Extract DType
            const mlir_elem_type = ranked_type.getElementType();
            const dtype = tensor.DType.fromMlirType(mlir_elem_type);
            try data_dtypes.append(dtype);
        }

        return ModelMetadata{
            .parameter_shapes = try param_shapes.toOwnedSlice(),
            .data_input_shapes = try data_shapes.toOwnedSlice(),
            .data_input_dtypes = try data_dtypes.toOwnedSlice(),
            .allocator = allocator,
        };
    }

    /// Fast textual parser specifically for Generation models (Frozen weights).
    /// Assumes ALL arguments to @main are data inputs (num_params = 0).
    /// This avoids loading the MLIR module just to count arguments.
    pub fn inspectLiteGeneration(
        allocator: std.mem.Allocator,
        mlir_source: []const u8,
    ) !ModelMetadata {
        // 1. Find function signature
        const sig_start_marker = "func.func @main";
        const idx = std.mem.indexOf(u8, mlir_source, sig_start_marker) orelse return error.MainFunctionNotFound;

        const open_paren = std.mem.indexOfPos(u8, mlir_source, idx, "(") orelse return error.InvalidSignature;
        // Simple close paren find - assumes no nested parens in types for standard models
        const close_paren = std.mem.indexOfPos(u8, mlir_source, open_paren, ")") orelse return error.InvalidSignature;

        const args_str = mlir_source[open_paren + 1 .. close_paren];

        // 2. Parse arguments
        var data_shapes = std.ArrayList([]i64).init(allocator);
        var data_dtypes = std.ArrayList(tensor.DType).init(allocator);
        // Param shapes is empty for generation (frozen weights)
        var param_shapes = std.ArrayList([]i64).init(allocator);

        errdefer {
            for (data_shapes.items) |s| allocator.free(s);
            data_shapes.deinit();
            data_dtypes.deinit();
            param_shapes.deinit();
        }

        // Split by comma, but need to handle nested commas in tensor<>
        var arg_start: usize = 0;
        var depth: usize = 0;
        var i: usize = 0;

        while (i < args_str.len) : (i += 1) {
            if (args_str[i] == '<') {
                depth += 1;
            } else if (args_str[i] == '>') {
                depth -= 1;
            } else if (args_str[i] == ',' and depth == 0) {
                // Parse this argument
                const arg_text = std.mem.trim(u8, args_str[arg_start..i], " \n\t");
                if (arg_text.len > 0) {
                    if (try parseArgument(allocator, arg_text)) |arg_info| {
                        try data_shapes.append(arg_info.shape);
                        try data_dtypes.append(arg_info.dtype);
                    }
                }
                arg_start = i + 1;
            }
        }

        // Parse last argument
        const last_arg = std.mem.trim(u8, args_str[arg_start..], " \n\t");
        if (last_arg.len > 0) {
            if (try parseArgument(allocator, last_arg)) |arg_info| {
                try data_shapes.append(arg_info.shape);
                try data_dtypes.append(arg_info.dtype);
            }
        }

        return ModelMetadata{
            .parameter_shapes = try param_shapes.toOwnedSlice(),
            .data_input_shapes = try data_shapes.toOwnedSlice(),
            .data_input_dtypes = try data_dtypes.toOwnedSlice(),
            .allocator = allocator,
        };
    }
};
