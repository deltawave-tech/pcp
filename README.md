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

The project uses [Nix](https://nixos.org/download/) as a build system.  It is possible to compile
dependencies locally, but that requires a considerable amount of manual work and compilation time.
We cache build and runtime dependencies and the builds themselves via
[cachix](https://app.cachix.org/cache/pcp).

### Install Nix and enable flakes

Follow one of the methodes presented in <https://nixos.org/download/>. For example:

```shell
# The following performs a single-user installation
$ sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon
```

You probably have to logout and login again, or source the init-scripts of your shell. On Ubuntu
using bash:

```shell
source /etc/bash.bashrc
```

You have to enable flakes. On a new installation do the following:

```shell
mkdir -p ~/.config/nix/
echo 'experimental-features = nix-command flakes' >> ~/.config/nix/nix.conf
```

### Install final Builds

If you just want to grap the final build, you can add cachix manually:

```shell
cat << EOF >> ~/.config/nix/nix.conf
substituters = https://cache.nixos.org https://pcp.cachix.org
trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY= pcp.cachix.org-1:D/JYXqFAnVLlvVUJEBOWoGLmJKwKW58SxPD0+m/HXZk=
```

Now, install pcp using

```shell
nix profile add github:deltawave-tech/pcp
```

### Install a build environment

Install cachix and enable the relevant cache:
```shell
nix-env -iA nixpkgs.cachix
cachix use pcp 
```

#### Build the Project
```sh
zig build
```

#### Run Tests
```sh
zig build test
```

#### Building and Running

Build everything:
```sh
zig build
```

Run the distributed training system (requires Cap'n Proto):
```sh
zig build run-distributed
```

Run the demo (no external dependencies needed):
```sh
zig build run-demo
```

Run all tests:
```sh
zig build test
```

See all available commands:
```sh
zig build --help
```

## Usage

To run the distributed training demonstration, which starts one Shepherd and two Workers:

### Start the Shepherd Node:
```bash
./zig-out/bin/main_distributed --shepherd --workers 2
```

### Start the First Worker Node (new terminal):
```bash
./zig-out/bin/main_distributed --worker --connect 127.0.0.1:8080
```

### Start the Second Worker Node (new terminal):
```bash
./zig-out/bin/main_distributed --worker --connect 127.0.0.1:8080
```
