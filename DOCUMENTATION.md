# PCP Documentation

Planetary Compute Protocol (PCP) is a high-performance, distributed tensor computation framework written in Zig. It enables decentralized training of Large Language Models (LLMs) using the DiLoCo (Distributed Low-Communication) algorithm.

The system leverages MLIR (Multi-Level Intermediate Representation) for graph construction and IREE (Intermediate Representation Execution Environment) for universal hardware targeting (CUDA, ROCm, Metal, Vulkan, CPU).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Distributed Training System](#distributed-training-system)
- [MLIR & Compiler Pipeline](#mlir--compiler-pipeline)
- [Automatic Differentiation](#automatic-differentiation)
- [IREE Backend Execution](#iree-backend-execution)
- [Networking Protocol](#networking-protocol)
- [Monitoring](#monitoring)
- [API & Usage](#api--usage)

## Architecture Overview

PCP separates the definition of computation (MLIR) from its execution (IREE). The coordinator ("Shepherd") constructs the training graph, compiles it to a platform-agnostic bytecode (VMFB), and distributes it to workers.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Shepherd (Coordinator)                          │
├──────────────────────────────────────────────────────────────────────┤
│  Model .mlir → MLIR Builder → Autodiff → Optimizer Injection         │
│                                  ↓                                    │
│                         IREE Compiler (iree-compile)                 │
│                                  ↓                                    │
│                    VMFB Bytecode + Initial Parameters                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ TCP + Cap'n Proto
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Worker Nodes                                 │
├──────────────────────────────────────────────────────────────────────┤
│  VMFB + Tensors → IREE Runtime → Hardware HAL (CUDA/Metal/ROCm)      │
│                                  ↓                                    │
│                     Updated Parameters → Network                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Core Layers

1. **Algorithm Layer**: Implements DiLoCo (Inner loop on workers, Outer loop on Shepherd)
2. **MLIR Layer**: Manages StableHLO dialects, tensor shapes, and graph construction
3. **Execution Layer**: Uses `iree-compile` for AOT compilation and `iree_runtime` for execution
4. **Network Layer**: TCP framing with Cap'n Proto serialization for high-throughput tensor transfer

## Distributed Training System

### Shepherd Coordinator (`src/controllers/shepherd.zig`)

The Shepherd acts as the master node. It holds the "true" model parameters and orchestrates the training lifecycle.

**Key Responsibilities:**

- **Graph Construction**: Loads the model MLIR, applies Autodiff, and injects the AdamW optimizer into the graph
- **Compilation**: Invokes `iree-compile` to generate a `.vmfb` (Virtual Machine FlatBuffer) specific to the workers' target architecture
- **Parameter Aggregation**: Receives updated parameters from workers, averages them, and applies the Nesterov Momentum update locally on the CPU

```zig
var shepherd = Shepherd.init(allocator);
defer shepherd.deinit();

// Start TCP server and wait for workers
try shepherd.listen("0.0.0.0", 8080);
```

### Worker Nodes (`src/worker.zig`)

Workers are "dumb" execution units. They do not know the model topology; they simply execute the VMFB provided by the Shepherd.

**Characteristics:**

- **State**: Maintains local AdamW optimizer states (M and V moments) which persist across inner loops
- **Execution**: Runs the inner loop for τ steps (default 10) before syncing back to the Shepherd
- **Hardware Agnostic**: The same worker code runs on NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (Metal) by selecting the appropriate IREE driver at runtime

```zig
var worker = Worker.init(allocator, backend);
defer worker.deinit();

// Connect to shepherd and enter training loop
try worker.connect("127.0.0.1", 8080, amd_target);
try worker.run();
```

### Supervisor Pattern (`src/nodes/supervisor.zig`)

To ensure fault tolerance in long-running training sessions, PCP implements a Supervisor pattern. The Supervisor is a lightweight parent process responsible for the lifecycle of the heavy compute Worker process.

**Responsibilities:**

- **Control Plane Connection**: Establishes a dedicated TCP connection to the Shepherd to listen for orchestration commands (e.g., `RESTART_WORKER`).
- **Process Management**: Spawns the actual Worker process as a child.
- **Health Monitoring**: Monitors the child process exit codes. If a worker crashes (e.g., CUDA OOM, segfault), the Supervisor automatically respawns it after a backoff period.
- **Handshake**: Performs a `SupervisorHandshake` with the Shepherd, generating a unique `supervisor_id` which is passed to the spawned worker to link the control and data planes.

```zig
// Supervisor Logic (Asynchronous Model)
try supervisor.connect(host, port);
try supervisor.sendHandshake();

// Spawn worker in background thread
try supervisor.spawnWorker(); // Spawns ./pcp --worker --supervisor-id <ID>
const monitor_thread = try supervisor.startMonitoringThread();

// Main event loop handles TCP commands from Shepherd
while (running) {
    const message = try supervisor.receiveMessage(); // Non-blocking TCP receive
    if (message.type == .RESTART_WORKER) {
        try supervisor.restartWorker(); // Force restart via TCP command
    }
    // Worker monitoring and auto-restart happens asynchronously in background
}
```

### DiLoCo Algorithm (`src/algorithms/diloco.zig`)

PCP implements the DiLoCo algorithm to reduce communication overhead:

1. **Outer Loop**: Shepherd broadcasts parameters
2. **Inner Loop**: Workers perform k steps of SGD/AdamW locally without communicating
3. **Sync**: Workers send updated parameters back
4. **Update**: Shepherd averages results and applies outer momentum

## MLIR & Compiler Pipeline

### Context Management (`src/mlir_ctx.zig`)

PCP interacts with LLVM/MLIR via a C-API wrapper. It explicitly registers required dialects:

- `stablehlo`: For tensor math operations
- `func`: For function definitions
- `arith`: For basic arithmetic

### The Compilation Flow

1. **Introspection**: The system reads a user-provided `.mlir` file (e.g., `nanogpt_forward.mlir`) to determine input shapes
2. **Graph Transformation**:
   - Forward pass is cloned
   - Backward pass is generated via `autodiff.zig`
   - Optimizer update steps (AdamW) are appended to the MLIR graph
3. **IREE Compilation**: The final module is serialized to text and passed to the `iree-compile` subprocess
4. **Targeting**: Flags like `--iree-hal-target-backends=cuda` or `--iree-hip-target=gfx942` are applied automatically based on configuration

```zig
pub const MLIRContext = struct {
    context: mlir.Context,

    pub fn init(allocator: Allocator) !Self
    pub fn deinit(self: *Self) void
    pub fn compileToVMFB(
        self: *Self,
        allocator: Allocator,
        mlir_source: []const u8,
        iree_target: []const u8,
        amd_target: ?[]const u8
    ) ![]u8
};
```

## Automatic Differentiation

PCP includes a custom reverse-mode automatic differentiation engine located in `src/autodiff.zig`.

### VJP (Vector-Jacobian Product) Rules

The engine transforms a forward MLIR graph into a gradient graph by walking backward from the return statement. It supports a wide range of StableHLO operations:

**Supported Operations:**

- **Math**: `add`, `subtract`, `multiply`, `divide`, `power`, `negate`, `exp`, `log`, `rsqrt`, `tanh`
- **Matrix**: `dot_general` (MatMul with batching support)
- **Manipulation**: `reshape`, `transpose`, `broadcast_in_dim`, `slice`, `concatenate`
- **Reduction**: `reduce_sum`, `reduce_max`
- **Advanced**: `gather` (embedding lookup), `select` (masking), `convert`

The AD engine uses a "Function-as-a-Unit" pattern, generating a dedicated `*_grad` function within the MLIR module.

```zig
// Generate gradient function from forward function
pub fn buildGradientGraph(
    allocator: Allocator,
    builder: *MLIRBuilder,
    forward_fn: mlir.Operation
) !void {
    // Apply autodiff transformation to create gradient function
}
```

## IREE Backend Execution

PCP uses the IREE Runtime (`src/backends/iree.zig`) for robust, production-grade execution across all hardware targets.

### WorkerBackend Interface

The `WorkerBackend` struct provides a generic interface:

```zig
pub const VTable = struct {
    executeTrainingStep: *const fn(
        ptr: *anyopaque,
        artifact: []const u8,    // VMFB bytes
        inputs: [][]const u8,    // Raw tensor bytes
        shapes: [][]const i64    // Tensor dimensions
    ) anyerror![][]u8,
    deinit: *const fn(ptr: *anyopaque) void,
};
```

### IREE Implementation

The `IreeBackend` handles:

- **Instance & Device Creation**: Initializes drivers (`cuda`, `hip`, `vulkan`, `local-sync`)
- **Session Management**: Loads VMFB modules
- **Tensor Marshalling**: Zero-copy (where possible) transfer of data between Host and Device using `iree_hal_buffer_view`

**Supported Backends:**

- **CUDA**: NVIDIA GPUs
- **ROCm/HIP**: AMD GPUs (MI300X, MI250X, MI100)
- **Metal**: Apple Silicon
- **Vulkan**: Cross-platform
- **CPU**: Fallback (local-sync driver)

### Per-Worker Compilation

The Shepherd compiles separate VMFB artifacts for each unique `(backend, amd_target)` combination. This enables heterogeneous clusters where workers with different GPU architectures participate in the same training session.

## Networking Protocol

PCP uses a hybrid networking approach:

### Control Plane (Supervisor <-> Shepherd)

- **Format**: JSON-only messages.
- **Purpose**: Handshakes, Heartbeats, and Restart commands (`MessageType.RESTART_WORKER`).
- **Persistence**: Connection remains open even if the Worker process crashes.

### Data Plane (Worker <-> Shepherd)

- **Format**: JSON envelope with embedded Base64-encoded Cap'n Proto blobs.
- **Purpose**: Transmission of VMFB artifacts, tensor parameters, and gradients.

Heavy tensor data (Model parameters, Gradients) is serialized using Cap'n Proto (`src/network/protocol.capnp`).

**Benefits:**

- **Zero-Copy Deserialization**: Cap'n Proto allows reading data directly from the network buffer
- **Base64 Wrapping**: Binary Cap'n Proto payloads are Base64 encoded and embedded in the JSON envelope for simplified socket handling

**Message Envelope Structure:**

```zig
pub const MessageEnvelope = struct {
    sender_node: NodeId,
    recipient_node: NodeId,
    msg_type: []const u8,        // e.g., "InnerLoopComplete"
    data: std.json.Value,        // Contains Base64 encoded Cap'n Proto blob
};
```

## Monitoring

### TUI Dashboard (`src/dashboard.zig`)

A terminal-based dashboard runs on a separate thread in the Shepherd process. It renders:

- **Status Box**: Current state (Initializing, Running, Completed)
- **Workers Table**: ID, Backend (CUDA/ROCm/CPU), IP Address, and Status
- **Metrics**: Loss history graph, Throughput (Tokens/sec), Epoch time

```zig
pub const Dashboard = struct {
    pub fn init(allocator: Allocator) !Self
    pub fn runDashboard() !void
};
```

### Metrics Collection (`src/monitoring.zig`)

A thread-safe global monitor (`TrainingMonitor`) aggregates statistics from the Shepherd controller without blocking the training loop.

```zig
// Global monitoring state
pub fn setStatus(status: TrainingStatus) void
pub fn setMetrics(epoch: u64, loss: f32, worker_count: u32) void
pub fn setWorkerInfo(workers: []const WorkerInfo) void
```

## API & Usage

### Directory Structure

| Path | Description |
|------|-------------|
| `src/algorithms/` | Implementation of DiLoCo and generic training interfaces |
| `src/backends/` | `iree.zig` (Runtime wrapper) and `worker_backend.zig` (Interface) |
| `src/controllers/` | `shepherd.zig` - Main coordination logic |
| `src/mlir/` | StableHLO dialect wrappers and C-API bindings |
| `src/network/` | TCP stream handling and Cap'n Proto bridges |
| `src/optimizers/` | `adam_mlir.zig` (Device-side) and `nesterov.zig` (Host-side) |
| `src/autodiff.zig` | The custom AD engine |
| `src/mlir_ctx.zig` | MLIR context and IREE compilation interface |

### CLI Arguments

**Shepherd:**

```bash
# Start the master node
./zig-out/bin/main_distributed --shepherd \
    --model src/models/nanogpt_forward.mlir \
    --workers 2 \
    --host 0.0.0.0 \
    --port 8080
```

**Supervisor (Fault-Tolerant Worker):**

```bash
# Starts a supervisor which manages the worker process
./zig-out/bin/main_distributed --supervise -- --worker \
    --connect <SHEPHERD_IP>:8080 \
    --backend cuda \
    --target sm_80
```

**Worker (Auto-detect backend):**

```bash
./zig-out/bin/main_distributed --worker \
    --connect <SHEPHERD_IP>:8080
```

**Worker (Specific backend):**

```bash
# CPU Worker
./zig-out/bin/main_distributed --worker \
    --connect <SHEPHERD_IP>:8080 \
    --backend cpu

# CUDA Worker (NVIDIA)
./zig-out/bin/main_distributed --worker \
    --connect <SHEPHERD_IP>:8080 \
    --backend cuda

# ROCm Worker (AMD MI300X)
./zig-out/bin/main_distributed --worker \
    --connect <SHEPHERD_IP>:8080 \
    --backend rocm \
    --amd-target gfx942
```

### Running Tests

```bash
# Verify CPU Pipeline
zig build run-cpu-pipeline-test

# Verify CUDA Pipeline (requires NVIDIA GPU)
zig build run-cuda-pipeline-test

# Verify ROCm Pipeline (requires AMD GPU)
zig build run-rocm-pipeline-test

# Mixed cluster test script
./run_mi300_cluster.sh
```

## Development Workflow

1. **Model Development**: Create models that generate MLIR computation graphs
2. **Algorithm Integration**: Implement training algorithms that combine models with autodiff
3. **Backend Testing**: Verify execution through IREE runtime on target hardware
4. **Distributed Deployment**: Scale training across multiple worker nodes with heterogeneous hardware
5. **Monitoring**: Use TUI dashboard for real-time training observation

This documentation covers the current distributed MLIR-based training architecture with IREE backend execution. The system is designed for scalable, cross-platform ML training with real-time monitoring and GPU acceleration across NVIDIA, AMD, and Apple hardware.
