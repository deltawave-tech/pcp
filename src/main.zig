// PCP - Planetary Compute Protocol
// A distributed tensor computation framework

// MLIR-based tensor and operations system
pub const tensor = @import("tensor.zig");
pub const ops = @import("ops.zig");
pub const mlir = @import("mlir.zig");
pub const mlir_ctx = @import("mlir_ctx.zig");

// Backend selection and implementations
pub const backend_selection = @import("backend_selection.zig");
pub const backends = struct {
    pub const iree = @import("backends/iree.zig");
};

// Data loading and processing
pub const data_loader = @import("data_loader.zig");

// Distributed training system
pub const controllers = struct {
    pub const shepherd = @import("controllers/shepherd.zig");
};
pub const worker = @import("worker.zig");
pub const algorithms = struct {
    pub const diloco = @import("algorithms/diloco.zig");
};

// Models
pub const models = struct {
};

// Legacy systems (will be phased out)
pub const autodiff = @import("autodiff.zig");

pub const optimizers = struct {
    pub const adam_mlir = @import("optimizers/adam_mlir.zig");
    pub const nesterov_mlir = @import("optimizers/nesterov_mlir.zig");
};

test {
    // Import all tests from modules
    _ = @import("tensor.zig");
    _ = @import("autodiff.zig");
    _ = @import("ops.zig");
}
