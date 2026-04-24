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

// Distributed runtime
pub const controllers = struct {
    pub const training = @import("nodes/gateway/controllers/training_controller.zig");
    pub const rl = @import("nodes/gateway/controllers/rl_controller.zig");
    pub const inference = @import("nodes/gateway/controllers/inference_controller.zig");
};
pub const inference = @import("inference/config.zig");
pub const gateway = @import("nodes/gateway/gateway.zig");
pub const federation = struct {
    pub const hub = @import("nodes/federation_hub/hub.zig");
    pub const types = @import("federation/types.zig");
};
pub const worker = @import("nodes/workers/worker.zig");
pub const algorithms = struct {
    pub const diloco = @import("algorithms/diloco.zig");
    pub const streaming_diloco = @import("algorithms/streaming_diloco.zig");
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
