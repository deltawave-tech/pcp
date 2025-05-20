// PCP - Planetary Compute Protocol
// A distributed tensor computation framework

pub const tensor = @import("tensor.zig");
pub const autodiff = @import("autodiff.zig");
pub const ops = @import("ops.zig");
pub const metal = @import("backends/metal.zig");

test {
    // Import all tests from modules
    _ = @import("tensor.zig");
    _ = @import("autodiff.zig");
    _ = @import("ops.zig");
}
