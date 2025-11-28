const std = @import("std");
const tensor = @import("core/tensor.zig");
const mlir = @import("mlir/wrapper.zig");
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
        /// DEPRECATED: Use materialize_module instead
        materialize: *const fn(ptr: *anyopaque, t: Tensor) anyerror![]u8,
        
        /// NEW SIGNATURE: Takes the entire module to be executed.
        materialize_module: *const fn(ptr: *anyopaque, module: mlir.Module) anyerror![]u8,
        
        /// Get the shared MLIR context for this executor
        getContext: *const fn(ptr: *anyopaque) mlir.Context,
        
        /// Optional: Clean up resources when the executor is done
        deinit: *const fn(ptr: *anyopaque) void,
    };

    /// Materialize a symbolic tensor into concrete bytes
    /// DEPRECATED: Use materializeModule instead
    pub fn materialize(self: Executor, t: Tensor) ![]u8 {
        return self.vtable.materialize(self.ptr, t);
    }

    /// Materialize a symbolic module into concrete bytes
    pub fn materializeModule(self: Executor, module: mlir.Module) ![]u8 {
        return self.vtable.materialize_module(self.ptr, module);
    }
    
    /// Get the shared MLIR context
    pub fn getContext(self: Executor) mlir.Context {
        return self.vtable.getContext(self.ptr);
    }
    
    /// Clean up executor resources
    pub fn deinit(self: Executor) void {
        self.vtable.deinit(self.ptr);
    }
};