# PCP

PCP is a distributed tensor computation framework written in Zig. It employs MLIR as its core intermediate representation to provide a compiler-driven approach to automatic differentiation, hardware acceleration, and optimization.

## Core Architecture

PCP transforms high-level tensor operations into optimized MLIR computation graphs. The system's design is centered on a forward-pass graph construction, which is then used to derive a corresponding gradient graph for automatic differentiation.

### Distributed Training Overview

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

### MLIR Compilation Pipeline

```
┌─────────┐   ┌─────────────┐   ┌───────┐   ┌─────────────────┐
│StableHLO│ → │ GPU Dialect │ → │SPIR-V │ → │Metal/CUDA/CPU   │
└─────────┘   └─────────────┘   └───────┘   └─────────────────┘
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

## Building the Project

### Prerequisites

Make sure you have the following tools installed:

* **macOS:** `brew install git cmake ninja capnp`
* **Ubuntu/Debian:** `sudo apt install build-essential git cmake ninja capnproto libcapnp-dev`

### Step 1: Clone the Repository

Clone the repository and all its submodules (`llvm-project`, `stablehlo`, etc.).

```sh
git clone --recursive <your-repo-url>
cd <your-repo-name>
```

### Step 2: Build C++ Dependencies (One-Time, ~1 Hour)

We provide a script that compiles LLVM, MLIR, and StableHLO. This is a long process that runs in the background.

```sh
./build_llvm_with_stablehlo.sh
```

This script will create a local `llvm-build/` directory containing the libraries and headers our project needs.

### Step 3: Set Environment Variables (Optional)

For better control over dependency detection, you can set environment variables:

```sh
# LLVM/MLIR location
export LLVM_DIR=/path/to/your/llvm-build

# Cap'n Proto location (if not in standard system paths)
export CAPNP_DIR=/path/to/your/capnp-installation
```

If not set, the build system will auto-detect dependencies in common locations:
- **LLVM:** `llvm-build/bin/llvm-config` (project-local), system installations via Homebrew, package managers, etc.
- **Cap'n Proto:** System installations via Homebrew (`/opt/homebrew`), package managers (`/usr`, `/usr/local`)

### Step 4: Build the Project
```sh
zig build
```

### Step 5: Run Tests
```sh
zig build test
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
