const std = @import("std");
const backend_selection = @import("selection.zig");
const tensor = @import("../core/tensor.zig");

/// Generic worker backend interface for executing compiled training artifacts.
/// This interface is completely MLIR-agnostic.
pub const WorkerBackend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        /// Executes a full training step defined by the compiled artifact.
        ///
        /// - compiled_artifact: The binary blob (.vmfb) to execute.
        /// - inputs_data: An array of byte slices for all inputs (params first, then data).
        /// - input_shapes: The shapes corresponding to each input slice.
        /// - input_dtypes: Optional dtypes for inputs (if null, inferred as f32 for params, i64 for last 2 data)
        executeTrainingStep: *const fn(
            ptr: *anyopaque,
            compiled_artifact: []const u8,
            inputs_data: [][]const u8,
            input_shapes: [][]const i64,
            input_dtypes: ?[]const tensor.DType,
        ) anyerror![][]u8,

        /// Executes a specific function within the compiled artifact.
        executeFunction: *const fn(
            ptr: *anyopaque,
            compiled_artifact: []const u8,
            function_name: []const u8,
            inputs_data: [][]const u8,
            input_shapes: [][]const i64,
            input_dtypes: ?[]const tensor.DType,
        ) anyerror![][]u8,

        /// Get the backend type
        getBackendType: *const fn(ptr: *anyopaque) backend_selection.Backend,

        /// Clean up backend resources.
        deinit: *const fn(ptr: *anyopaque) void,
    };

    /// Execute a training step with the given artifact and inputs.
    pub fn executeTrainingStep(self: WorkerBackend, artifact: []const u8, data: [][]const u8, shapes: [][]const i64, dtypes: ?[]const tensor.DType) ![][]u8 {
        return self.vtable.executeTrainingStep(self.ptr, artifact, data, shapes, dtypes);
    }

    /// Execute a specific function within the artifact.
    pub fn executeFunction(self: WorkerBackend, artifact: []const u8, function_name: []const u8, data: [][]const u8, shapes: [][]const i64, dtypes: ?[]const tensor.DType) ![][]u8 {
        return self.vtable.executeFunction(self.ptr, artifact, function_name, data, shapes, dtypes);
    }

    /// Get the backend type
    pub fn getBackendType(self: WorkerBackend) backend_selection.Backend {
        return self.vtable.getBackendType(self.ptr);
    }

    /// Clean up backend resources
    pub fn deinit(self: WorkerBackend) void {
        self.vtable.deinit(self.ptr);
    }
};