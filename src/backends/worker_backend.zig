const std = @import("std");
const mlir = @import("../mlir.zig");

/// Generic worker backend interface for executing training steps.
/// This interface is responsible for executing MLIR modules received from the Shepherd.
/// The choice of Metal, CUDA, or CPU backend is made at compile-time.
pub const WorkerBackend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        /// Executes a full training step defined by the MLIR module.
        /// Takes model parameters as input, returns updated parameters and loss.
        /// 
        /// inputs: Array of input byte arrays (parameters, data, etc.)
        /// returns: Array of output byte arrays (updated parameters, loss, etc.)
        executeTrainingStep: *const fn(
            ptr: *anyopaque,
            mlir_module: mlir.Module,
            inputs: [][]const u8,
        ) anyerror![][]u8,
        
        /// Clean up backend resources
        deinit: *const fn(ptr: *anyopaque) void,
    };

    /// Execute a training step with the given MLIR module and inputs
    pub fn executeTrainingStep(self: WorkerBackend, mod: mlir.Module, inputs: [][]const u8) ![][]u8 {
        return self.vtable.executeTrainingStep(self.ptr, mod, inputs);
    }
    
    /// Clean up backend resources
    pub fn deinit(self: WorkerBackend) void {
        self.vtable.deinit(self.ptr);
    }
};