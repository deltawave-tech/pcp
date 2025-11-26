// PCP - Planetary Compute Protocol
// A distributed tensor computation framework

// MLIR-based tensor and operations system
pub const tensor = @import("core/tensor.zig");
pub const ops = @import("core/ops.zig");
pub const mlir = @import("mlir/wrapper.zig");
pub const mlir_ctx = @import("mlir/context.zig");

// Backend selection and implementations
pub const backend_selection = @import("backends/selection.zig");
pub const backends = struct {
    pub const iree = @import("backends/iree.zig");
};

// Data loading and processing
pub const data_loader = @import("data/loader.zig");

// Distributed training system
pub const controllers = struct {
    pub const shepherd = @import("nodes/controllers/shepherd.zig");
};
pub const worker = @import("nodes/workers/worker.zig");
pub const algorithms = struct {
    pub const diloco = @import("algorithms/diloco.zig");
};

// Models
pub const models = struct {
};

// Legacy systems (will be phased out)
pub const autodiff = @import("autodiff/engine.zig");

pub const optimizers = struct {
    pub const adam_mlir = @import("optimizers/adam_mlir.zig");
    pub const nesterov_mlir = @import("optimizers/nesterov_mlir.zig");
    pub const nesterov = @import("optimizers/nesterov.zig");
};

test {
    // Import all tests from modules
    _ = @import("core/tensor.zig");
    _ = @import("autodiff/engine.zig");
    _ = @import("core/ops.zig");
}
