const std = @import("std");

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
        executeTrainingStep: *const fn(
            ptr: *anyopaque,
            compiled_artifact: []const u8,
            inputs_data: [][]const u8,
            input_shapes: [][]const i64,
        ) anyerror![][]u8,
        
        /// Clean up backend resources.
        deinit: *const fn(ptr: *anyopaque) void,
    };

    /// Execute a training step with the given artifact and inputs.
    pub fn executeTrainingStep(self: WorkerBackend, artifact: []const u8, data: [][]const u8, shapes: [][]const i64) ![][]u8 {
        return self.vtable.executeTrainingStep(self.ptr, artifact, data, shapes);
    }
    
    /// Clean up backend resources
    pub fn deinit(self: WorkerBackend) void {
        self.vtable.deinit(self.ptr);
    }
};