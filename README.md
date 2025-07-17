# PCP

PCP is a distributed tensor computation framework written in Zig. It employs MLIR as its core intermediate representation to provide a compiler-driven approach to automatic differentiation, hardware acceleration, and optimization.

## Core Architecture

PCP transforms high-level tensor operations into optimized MLIR computation graphs. The system's design is centered on a forward-pass graph construction, which is then used to derive a corresponding gradient graph for automatic differentiation.

### Modular Protocol Controller (PCP) Architecture

```
┌───────────────┐    ┌─────────────────────┐    ┌───────────────────────┐
│ Model         │    │ Training Algorithm  │    │ Distributed System    │
│ (GPT-2)       │◀──▶│ (DiLoCo)            │◀──▶│ (Shepherd + Workers)  │
├───────────────┤    ├─────────────────────┤    ├───────────────────────┤
│ • Forward     │    │ • Parameter Sync    │    │ • TCP Communication   │
│ • MLIR Graph  │    │ • MLIR Optimizers   │    │ • Message Protocol    │
│ • Autodiff    │    │ • Inner/Outer Loop  │    │ • Graph Distribution  │
└───────────────┘    └─────────────────────┘    └───────────────────────┘
       │                       │                           │
       └───────────────────────┼───────────────────────────┘
                               ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                    MLIR Compilation Pipeline                     │
    │                                                                  │
    │  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐   │
    │  │ StableHLO   │──▶│ GPU/SPIR-V   │──▶│ Backend Execution   │   │
    │  │ Graph       │   │ Lowering     │   │ (Metal/CUDA/CPU)    │   │
    │  └─────────────┘   └──────────────┘   └─────────────────────┘   │
    └──────────────────────────────────────────────────────────────────┘
```

### MLIR Lowering Pipeline

```
┌─────────────────┐    passes    ┌─────────────────┐    translation
│ StableHLO       │ ───────────▶  │ GPU Dialect     │ ──────────────▶
│ Dialect         │               │ (gpu.launch)    │
│                 │               │                 │
│ • stablehlo.add │               │ • gpu.func      │
│ • stablehlo.mul │               │ • gpu.thread_id │
│ • stablehlo.dot │               │ • gpu.memcpy    │
└─────────────────┘               └─────────────────┘
                                           │
                                           ▼ passes
                                  ┌─────────────────┐    translation
                                  │ SPIR-V Dialect  │ ──────────────▶
                                  │ (spirv.*)       │
                                  │                 │
                                  │ • spirv.func    │
                                  │ • spirv.fadd    │
                                  │ • spirv.load    │
                                  └─────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Backend Targets                    │
                    ├─────────────────┬─────────────────┬─────────────┤
                    │ Metal (MSL)     │ CUDA (PTX)      │ CPU (LLVM)  │
                    │                 │                 │             │
                    │ • MTLLibrary    │ • CUmodule      │ • JIT/AOT   │
                    │ • MTLFunction   │ • CUfunction    │ • Native    │
                    │ • MTLBuffer     │ • CUdeviceptr   │ • Vectorized│
                    └─────────────────┴─────────────────┴─────────────┘
```

This architecture is implemented through several key components:

- **MLIR as the Core IR**: Computations are represented as MLIR graphs using the stablehlo dialect.
- **Tensor Abstraction**: Tensors (`src/tensor.zig`) are symbolic handles to `mlir.Value`, representing nodes in the computation graph rather than immediate data buffers.
- **MLIRBuilder**: A stateful builder (`src/ops.zig`) constructs the MLIR graph via the MLIR C API.
- **StableHLO Dialect Wrapper**: A clean, idiomatic Zig interface (`src/mlir/dialects/stablehlo.zig`) for creating stablehlo operations.
- **Graph-to-Graph Automatic Differentiation**: The framework features a VJP-based autodiff engine (`src/autodiff.zig`). It performs a graph-to-graph transformation by applying Vector-Jacobian Product (VJP) rules to the forward-pass graph, thereby generating a new MLIR graph that computes the necessary gradients.
- **Distributed Training System**: The framework supports distributed training using a Shepherd (coordinator) and Worker model. It implements the DiLoCo algorithm (`src/algorithms/diloco.zig`) for low-communication training. Both global and local parameter updates are managed through MLIR-based optimizers (`src/optimizers/*_mlir.zig`) that construct their update logic as sub-graphs.
- **JIT-Compiled Execution Backend**: PCP includes a just-in-time (JIT) execution engine for Apple's Metal framework (`src/backends/metal.zig`). The compilation pipeline is as follows:
  - **Lowering**: The stablehlo graph is lowered through gpu and spirv dialects.
  - **Translation**: The spirv dialect module is translated to a SPIR-V binary.
  - **Cross-Compilation**: The SPIR-V binary is translated to Metal Shading Language (MSL).
  - **Runtime Compilation**: The MSL source is compiled into a native MTLLibrary for execution on the GPU.

## Project Structure

```
.
└── src
    ├── algorithms/              # Distributed training algorithms and interfaces (DiLoCo).
    ├── backends/                # Hardware execution backends (Metal).
    ├── controllers/             # The Shepherd coordinator node.
    ├── examples/                # Usage examples and component tests.
    ├── mlir/                    # MLIR C-API wrappers, dialect definitions, and bridges.
    ├── models/                  # Model definitions (GPT-2) as MLIR graphs.
    ├── network/                 # TCP communication and messaging protocol.
    ├── optimizers/              # MLIR-based optimizers (Adam, Nesterov).
    ├── autodiff.zig             # VJP-based automatic differentiation engine.
    ├── main_distributed.zig     # Entry point for the distributed system.
    ├── mlir.zig, mlir_ctx.zig   # Core MLIR context and API wrappers.
    ├── ops.zig                  # MLIR operation graph-building functions.
    ├── tensor.zig               # The symbolic Tensor data structure.
    └── worker.zig               # The worker compute node.
```

## Usage

To run the distributed training demonstration, which starts one Shepherd and two Workers:

### Start the Shepherd Node:
```bash
zig run src/main_distributed.zig -- --shepherd --workers 2
```

### Start the First Worker Node (new terminal):
```bash
zig run src/main_distributed.zig -- --worker --connect 127.0.0.1:8080
```

### Start the Second Worker Node (new terminal):
```bash
zig run src/main_distributed.zig -- --worker --connect 127.0.0.1:8080
```