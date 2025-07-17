const std = @import("std");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor(void);

/// Generic execution interface that the Shepherd owns and passes to algorithms.
/// This decouples algorithms from specific backend implementations.
/// 
/// The key insight: algorithms need only one capability from the outside world -
/// turning symbolic tensors into concrete bytes for serialization/communication.
pub const Executor = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        /// Takes a symbolic tensor and returns its concrete data.
        /// The executor handles the actual MLIR graph compilation and execution.
        /// This is the ONE thing algorithms can't do symbolically.
        materialize: *const fn(ptr: *anyopaque, t: Tensor) anyerror![]u8,
        
        /// Optional: Clean up resources when the executor is done
        deinit: *const fn(ptr: *anyopaque) void,
    };

    /// Materialize a symbolic tensor into concrete bytes
    pub fn materialize(self: Executor, t: Tensor) ![]u8 {
        return self.vtable.materialize(self.ptr, t);
    }
    
    /// Clean up executor resources
    pub fn deinit(self: Executor) void {
        self.vtable.deinit(self.ptr);
    }
};