# PCP

PCP is a distributed tensor computation framework written in Zig. It employs MLIR and the IREE compiler toolchain as its core to provide automatic differentiation, hardware acceleration, and optimization.

## Core Architecture

PCP transforms high-level tensor operations into optimized MLIR computation graphs. The system's design is centered on a forward-pass graph construction, which is then used to derive a corresponding gradient graph for automatic differentiation.

### Distributed Training Overview

The Shepherd (controller) constructs a complete MLIR training graph using the StableHLO dialect. This graph is then compiled by the IREE compiler (`iree-compile`) into a portable `.vmfb` artifact for a specific hardware target (e.g., Metal for macOS, LLVM-CPU for generic CPUs). This binary artifact is sent to workers, which execute it using the cross-platform IREE runtime.

```
┌───────────────────────────────────────────────────────────────────┐
│                            Shepherd                               │
├───────────────────────────────────────────────────────────────────┤
│ MLIR Graph (StableHLO) → IREE Compiler → *.vmfb Artifact          │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                            Workers                                │
├───────────────────────────────────────────────────────────────────┤
│ Receive *.vmfb → IREE Runtime → Execute on GPU/CPU → Send Results │
└───────────────────────────────────────────────────────────────────┘
```

### IREE Compilation Pipeline

IREE handles the entire lowering pipeline from a high-level dialect to a hardware-specific executable format, abstracting these complex details away from our framework.

```
┌─────────┐   ┌───────────────────────────┐   ┌─────────────────┐
│StableHLO│ → │    IREE Compiler          │ → │Metal/CUDA/CPU   │
│         │   │   (iree-compile tool)     │   │  (via *.vmfb)   │
└─────────┘   └───────────────────────────┘   └─────────────────┘
```

## Project Structure

```
.
└── src
    ├── algorithms/              # Distributed training algorithms and interfaces (DiLoCo).
    ├── backends/                # Hardware execution backends (IREE, Demo).
    ├── controllers/             # The Shepherd coordinator node.
    ├── examples/                # Usage examples and component tests.
    ├── mlir/                    # MLIR C-API wrappers, dialect definitions, and bridges.
    ├── models/                  # Model definitions (GPT-2) as MLIR graphs.
    ├── network/                 # TCP communication and messaging protocol (Cap'n Proto).
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

- **Zig**: Version 0.12.0 or newer.
- **System Build Tools**: `git`, `cmake`, `ninja`.
- **Cap'n Proto**: (Required for distributed functionality). The build script will try to find it automatically. If it fails, you may need to install it or set the `CAPNP_DIR` environment variable.
  - **macOS**: `brew install capnp`
  - **Ubuntu/Debian**: `sudo apt install capnproto libcapnp-dev`

### Step 1: Clone This Repository

Clone this project repository.

```sh
git clone https://github.com/deltawave-tech/pcp.git
cd pcp
git checkout iree-install-test
```

### Step 2: Build IREE from Source

PCP depends on a local build of IREE. This is a one-time setup that can take a significant amount of time (~1 hour). The build script expects IREE to be in a sibling directory.

Navigate to the parent directory of your project and clone IREE:

```sh
cd ..
git clone https://github.com/openxla/iree.git
cd iree
```

Initialize IREE's submodules:

```sh
git submodule update --init --recursive
```

Build IREE using CMake. This will create a sibling `iree-build` directory.

```sh
cmake -GNinja -B ../iree-build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_BUILD_COMPILER=ON -DIREE_ENABLE_ASSERTIONS=ON
cmake --build ../iree-build
```

After this step, your directory structure should look like this:

```
/some/path/
├── pcp/  (This project)
├── iree/               (IREE source code)
└── iree-build/         (IREE build artifacts)
```

### Step 3: Configure Environment (Optional but Recommended)

For maximum flexibility, you can set the `IREE_DIR` environment variable to point to the directory containing your `iree` and `iree-build` folders.

```sh
# Example: Add this to your ~/.zshrc or ~/.bashrc
export IREE_DIR="/path/to/parent/directory"
```

If `IREE_DIR` is not set, the build script will automatically look for the sibling directory structure described in Step 2.

### Step 4: Build and Test PCP

Navigate back to this project's directory and use Zig to build and run the verification tests.


Run the IREE pipeline verification tests:

These tests confirm that your environment is correctly configured to compile and execute MLIR graphs via IREE on your available hardware.

For macOS (testing Metal backend):

```sh
zig build run-m3-pipeline-test
```

For any platform (testing CPU backend):

```sh
zig build run-cpu-pipeline-test
```
