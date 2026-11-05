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
pub const network = struct {
    pub const message = @import("network/message.zig");
    pub const message_queue = @import("network/message_queue.zig");
};
pub const controllers = struct {
    pub const training = @import("nodes/gateway/controllers/training_controller.zig");
    pub const rl = @import("nodes/gateway/controllers/rl_controller.zig");
    pub const inference = @import("nodes/gateway/controllers/inference_controller.zig");
};
pub const workloads = struct {
    pub const training = struct {
        pub const config = @import("workloads/training/config.zig");
    };
    pub const inference = struct {
        pub const config = @import("workloads/inference/config.zig");
        pub const types = @import("workloads/inference/types.zig");
        pub const router = @import("workloads/inference/router.zig");
        pub const chat_template = @import("workloads/inference/chat_template.zig");
        pub const tokenizer = @import("workloads/inference/tokenizer.zig");
        pub const model_registry = @import("workloads/inference/model_registry.zig");
        pub const session_manager = @import("workloads/inference/session_manager.zig");
    };
};

pub const gateway = @import("nodes/gateway/gateway.zig");
pub const protocol = struct {
    pub const federation = struct {
        pub const types = @import("protocol/federation/types.zig");
        pub const graph = struct {
            pub const types = @import("protocol/federation/graph/types.zig");
            pub const store = @import("protocol/federation/graph/store.zig");
            pub const mutation_log = @import("protocol/federation/graph/mutation_log.zig");
            pub const policy_store = @import("protocol/federation/graph/policy_store.zig");
        };
    };
};
pub const federation = struct {
    pub const hub = @import("nodes/federation_hub/hub.zig");
    pub const types = protocol.federation.types;
};
pub const worker = @import("nodes/workers/worker.zig");
pub const algorithms = struct {
    pub const diloco = @import("algorithms/diloco.zig");
    pub const streaming_diloco = @import("algorithms/streaming_diloco.zig");
    pub const fragments = @import("algorithms/fragments.zig");
    pub const decoupled_merge = @import("algorithms/decoupled_merge.zig");
    pub const vector_clock = @import("algorithms/vector_clock.zig");
    pub const event_tape = @import("algorithms/event_tape.zig");
    pub const decoupled_diloco = @import("algorithms/decoupled_diloco.zig");
    pub const decoupled_learner = @import("algorithms/decoupled_learner.zig");
};

pub const observability = struct {
    pub const monitoring = @import("observability/monitoring.zig");
    pub const wandb = @import("observability/wandb.zig");
};

// Models
pub const models = struct {};

// Legacy systems (will be phased out)
pub const autodiff = @import("autodiff/engine.zig");
pub const compiler = struct {
    pub const graph_builder = @import("compiler/graph_builder.zig");
};

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
    _ = @import("network/message.zig");
    _ = @import("network/message_queue.zig");
    _ = @import("algorithms/fragments.zig");
    _ = @import("algorithms/decoupled_merge.zig");
    _ = @import("algorithms/vector_clock.zig");
    _ = @import("algorithms/event_tape.zig");
    _ = @import("algorithms/decoupled_diloco.zig");
    _ = @import("algorithms/decoupled_learner.zig");
}
