# PCP Documentation

A comprehensive guide to the distributed MLIR-based tensor computation framework.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Distributed Training System](#distributed-training-system)
3. [MLIR Integration](#mlir-integration)
4. [Automatic Differentiation](#automatic-differentiation)
5. [Backend Execution](#backend-execution)
6. [Monitoring and Dashboard](#monitoring-and-dashboard)
7. [API Reference](#api-reference)

## Architecture Overview

PCP is a distributed training framework that combines models and algorithms to create MLIR computation graphs, then distributes them to workers for GPU execution:

```
┌──────────────────────────────────────────────────────────────────────┐
│                            Controller                                │
├──────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ + ┌───────────┐ = ┌─────────────────────────────────┐   │
│  │ Model   │   │ Algorithm │   │ MLIR Training Graph             │   │
│  │         │   │           │   │ (Forward + Autodiff + Update)   │   │
│  └─────────┘   └───────────┘   └─────────────────────────────────┘   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ Serialize & Send
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           Workers                                    │
├──────────────────────────────────────────────────────────────────────┤
│ Receive MLIR → Compile to GPU → Execute → Send Results Back          │
└──────────────────────────────────────────────────────────────────────┘
```

### Core Layers

1. **Distributed Coordination**: Shepherd coordinator manages worker nodes via TCP
2. **MLIR Graph Construction**: Builds complete training graphs with autodiff
3. **Backend Execution**: Cross-platform GPU execution via SPIR-V bridge
4. **Real-time Monitoring**: TUI dashboard for training metrics and system status

## Distributed Training System

### Shepherd Coordinator (`src/controllers/shepherd.zig`)

The central coordinator that manages the distributed training process:

```zig
var shepherd = Shepherd.init(allocator);
defer shepherd.deinit();

// Start TCP server and wait for workers
try shepherd.startServer(8080);
try shepherd.run();
```

**Key responsibilities:**
- Combine models and algorithms into complete MLIR training graphs
- Serialize MLIR modules and broadcast to workers
- Collect training results and coordinate parameter updates
- Manage distributed training lifecycle

### Worker Nodes (`src/worker.zig`)

Execute MLIR training graphs on local hardware:

```zig
var worker = Worker.init(allocator, backend);
defer worker.deinit();

// Connect to shepherd and enter training loop
try worker.connect("127.0.0.1", 8080);
try worker.run();
```

**Worker execution flow:**
1. Receive serialized MLIR module from shepherd
2. Deserialize and compile to GPU (SPIR-V → Metal/CUDA)
3. Execute training step with local data
4. Send updated parameters back to shepherd

### Training Algorithms (`src/algorithms/`)

PCP implements distributed training algorithms like DiLoCo:

```zig
var diloco = try DiLoCo.init(allocator, &shepherd, config, executor);
defer diloco.deinit();

// Start distributed training
try diloco.run();
```

## MLIR Integration

### Compilation Pipeline

```
┌─────────┐   ┌─────────────┐   ┌───────┐   ┌─────────────────┐
│StableHLO│ → │ GPU Dialect │ → │SPIR-V │ → │Metal/CUDA/CPU   │
└─────────┘   └─────────────┘   └───────┘   └─────────────────┘
```

### StableHLO Integration

PCP integrates StableHLO as an external LLVM project with pass registration:

```cpp
// src/mlir/pass_anchors.cpp
extern "C" void mlirForceLoadAllRequiredPasses() {
    // StableHLO passes
    mlir::stablehlo::registerAllPasses();

    // Core MLIR passes for lowering pipeline
    (void)mlir::createCanonicalizerPass;
    (void)mlir::createConvertGpuOpsToSPIRVopsPass;
    // ... additional passes
}
```

### MLIR Context Management (`src/mlir_ctx.zig`)

Manages MLIR contexts and module serialization/deserialization:

```zig
pub const MLIRContext = struct {
    context: mlir.Context,

    pub fn init(allocator: Allocator) !Self
    pub fn deinit(self: *Self) void
    pub fn getContext(self: *Self) mlir.Context
};

// Serialize MLIR modules for network transmission
pub fn serializeMLIRModule(allocator: Allocator, module: mlir.Module) ![]u8

// Deserialize MLIR modules on worker nodes
pub fn deserializeMLIRModule(allocator: Allocator, context: mlir.Context, data: []const u8) !mlir.Module
```

## Automatic Differentiation

### Function-as-a-Unit Pattern

PCP uses a Function-as-a-Unit approach where complete training functions are differentiated:

```zig
// Create forward+loss function that autodiff can process
fn buildForwardAndLossFunction(builder: *MLIRBuilder) !mlir.Operation {
    // Define function: func(params, inputs, targets) -> (loss)
    const func_op = mlir.Operation.create(builder.ctx, "func.func", .{
        .attributes = &.{
            .{ "function_type", mlir.Attribute.typeAttr(func_type) },
            .{ "sym_name", mlir.Attribute.stringAttr(builder.ctx, "forward_and_loss_fn") },
        },
    });

    // Build forward computation and loss calculation inside function
    // ...

    return func_op;
}
```

### Gradient Graph Generation (`src/autodiff.zig`)

The autodiff system generates gradient functions automatically:

```zig
// Generate gradient function from forward function
pub fn buildGradientGraph(
    allocator: Allocator,
    builder: *MLIRBuilder,
    forward_fn: mlir.Operation
) !void {
    // Apply autodiff transformation to create gradient function
    // The result is a new func.func that computes gradients
}
```

### Complete Training Graph Construction

Training algorithms build complete MLIR modules with forward, gradient, and update logic:

```zig
// Build complete worker training graph
fn buildWorkerTrainingGraph(self: *DiLoCo) ![]u8 {
    // 1. Create forward+loss function
    const forward_fn = try self.buildForwardAndLossFunction(&builder);

    // 2. Differentiate to create gradient function
    _ = try autodiff.buildGradientGraph(self.allocator, &builder, forward_fn);

    // 3. Create main orchestration function that calls both
    const main_func = try self.buildMainFunction(&builder);

    // 4. Serialize complete module for workers
    return try serializeMLIRModule(self.allocator, builder.module);
}
```

## Backend Execution

### Backend Abstraction (`src/backends/worker_backend.zig`)

PCP provides a backend-agnostic interface for executing MLIR modules:

```zig
pub const WorkerBackend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        executeTrainingStep: *const fn(ptr: *anyopaque, mlir_module: mlir.Module, inputs: [][]const u8) anyerror![][]u8,
        deinit: *const fn(ptr: *anyopaque) void,
    };
};
```

### Metal Backend (`src/backends/metal.zig`)

Cross-platform GPU execution through SPIR-V bridge:

```zig
pub const MetalBackend = struct {
    // Execution flow: MLIR → SPIR-V → Metal Shading Language → MTLLibrary
    pub fn executeTrainingStep(
        self: *Self,
        mlir_module: mlir.Module,
        inputs: [][]const u8
    ) ![][]u8 {
        // 1. Lower MLIR to SPIR-V
        const spirv_binary = try self.lowerToSPIRV(mlir_module);

        // 2. Cross-compile SPIR-V to MSL using SPIRV-Cross
        const msl_source = try self.spirvToMSL(spirv_binary);

        // 3. Compile MSL to Metal library and execute
        return try self.executeOnMetal(msl_source, inputs);
    }
};
```

### SPIR-V Bridge (`src/mlir/spirv_bridge.cpp`)

Handles MLIR to SPIR-V translation and cross-compilation:

```cpp
// Lower MLIR module to SPIR-V binary
extern "C" bool mlirLowerToSPIRV(
    MlirModule module,
    uint8_t** spirv_data,
    size_t* spirv_size
);

// Cross-compile SPIR-V to Metal Shading Language
extern "C" bool spirvCrossCompileToMSL(
    const uint8_t* spirv_data,
    size_t spirv_size,
    char** msl_source
);
```

## Monitoring and Dashboard

### TUI Dashboard (`src/dashboard.zig`)

Real-time training monitoring with terminal user interface:

```zig
pub const Dashboard = struct {
    pub fn init(allocator: Allocator) !Self
    pub fn start(self: *Self) !void
    pub fn stop(self: *Self) void

    // Updates displayed in real-time
    pub fn updateTrainingStatus(self: *Self, status: TrainingStatus) void
    pub fn updateMetrics(self: *Self, epoch: u64, loss: f32, workers: u32) void
    pub fn updateEpochTime(self: *Self, time_ms: u64) void
};
```

**Dashboard features:**
- Real-time training metrics (loss, epoch time, worker count)
- System resource monitoring
- Training algorithm status
- Network connectivity status
- MLIR compilation and execution logs

### Monitoring System (`src/monitoring.zig`)

Centralized metrics collection and reporting:

```zig
// Global monitoring state
pub fn setStatus(status: TrainingStatus) void
pub fn setMetrics(epoch: u64, loss: f32, worker_count: u32) void
pub fn setEpochTime(time_ms: u64) void
pub fn setModelInfo(param_count: usize, learning_rate: f32) void

// Retrieve current metrics
pub fn getMetrics() TrainingMetrics
pub fn getStatus() TrainingStatus
```

## API Reference

### Running Distributed Training

Start a distributed training session with shepherd and workers:

```bash
# Terminal 1: Start shepherd coordinator
zig run src/main_distributed.zig -- --shepherd --workers 2

# Terminal 2: Start first worker
zig run src/main_distributed.zig -- --worker --connect 127.0.0.1:8080

# Terminal 3: Start second worker
zig run src/main_distributed.zig -- --worker --connect 127.0.0.1:8080
```

### Key File Structure

```
src/
├── controllers/shepherd.zig      # Distributed training coordinator
├── worker.zig                    # Worker node implementation
├── algorithms/diloco.zig          # DiLoCo distributed training algorithm
├── backends/                     # Hardware execution backends
│   ├── worker_backend.zig        # Backend abstraction interface
│   └── metal.zig                 # Metal GPU backend
├── mlir/                         # MLIR integration layer
│   ├── c.zig                     # MLIR C API bindings
│   ├── pass_anchors.cpp          # StableHLO pass registration
│   └── spirv_bridge.cpp          # SPIR-V compilation bridge
├── autodiff.zig                  # Function-as-a-Unit autodiff
├── dashboard.zig                 # TUI monitoring dashboard
├── monitoring.zig                # Metrics collection system
└── execution.zig                 # Backend execution abstraction
```

### Development Workflow

1. **Model Development**: Create models that generate MLIR computation graphs
2. **Algorithm Integration**: Implement training algorithms that combine models with autodiff
3. **Backend Testing**: Verify GPU execution through SPIR-V bridge
4. **Distributed Deployment**: Scale training across multiple worker nodes
5. **Monitoring**: Use TUI dashboard for real-time training observation

This documentation covers the current distributed MLIR-based training architecture. The system is designed for scalable, cross-platform ML training with real-time monitoring and GPU acceleration.
