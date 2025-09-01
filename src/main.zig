// PCP - Planetary Compute Protocol
// A distributed tensor computation framework

// MLIR-based tensor and operations system
pub const tensor = @import("tensor.zig");
pub const ops = @import("ops.zig");
pub const mlir = @import("mlir.zig");
pub const mlir_ctx = @import("mlir_ctx.zig");

// Data loading and processing
pub const data_loader = @import("data_loader.zig");

// Models
pub const models = struct {
    pub const gpt2 = @import("models/gpt2.zig");
};

// Legacy systems (will be phased out)
pub const autodiff = @import("autodiff.zig");
pub const metal = @import("backends/metal.zig");

pub const optimizers = struct {
};

test {
    // Import all tests from modules
    _ = @import("tensor.zig");
    _ = @import("autodiff.zig");
    _ = @import("ops.zig");
}
