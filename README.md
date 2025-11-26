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
    ├── algorithms/              # Distributed training logic
    │   ├── diloco.zig          # DiLoCo implementation (Inner/Outer loops)
    │   └── training_algorithm.zig
    ├── backends/                # Hardware execution backends
    │   ├── iree.zig            # IREE runtime integration
    │   └── worker_backend.zig  # Generic backend interface
    ├── controllers/             # Coordination logic
    │   └── shepherd.zig        # The Shepherd (Master node) implementation
    ├── examples/                # Integration tests and pipelines
    │   ├── cpu_pipeline_test.zig
    │   ├── cuda_pipeline_test.zig
    │   ├── rocm_pipeline_test.zig
    │   └── ...
    ├── mlir/                    # MLIR Dialects and C-API Wrappers
    │   ├── dialects/
    │   │   └── stablehlo.zig   # Safe wrappers for StableHLO ops
    │   ├── include/            # C++ headers for MLIR integration
    │   ├── c.zig               # Auto-generated C imports
    │   └── pass_anchors.cpp    # Dialect registration hooks
    ├── network/                 # Networking layer
    │   ├── broker.zig          # Internal message routing
    │   ├── capnp_bridge.cpp    # C++ bridge for Cap'n Proto
    │   ├── protocol.capnp      # Protocol definition
    │   └── tcp_stream.zig      # TCP framing and socket management
    ├── optimizers/              # Optimizer implementations
    │   ├── adam_mlir.zig       # AdamW implemented in MLIR (runs on Worker)
    │   ├── nesterov.zig        # Nesterov (Host/Zig implementation)
    │   └── nesterov_mlir.zig   # Nesterov (MLIR implementation)
    ├── autodiff.zig             # Reverse-mode Automatic Differentiation
    ├── backend_selection.zig   # Compile-time/Runtime backend logic
    ├── dashboard.zig           # TUI Monitoring Dashboard
    ├── data_loader.zig         # Tokenizer and Batching
    ├── execution.zig           # Generic executor interfaces
    ├── main_distributed.zig    # CLI Entry point
    ├── mlir_ctx.zig            # MLIR Context management & Compiler invocation
    ├── ops.zig                 # MLIR Operation Builder
    ├── tensor.zig              # Symbolic Tensor abstraction
    └── worker.zig              # Worker node state machine
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

For Linux with NVIDIA GPU (testing CUDA backend):

```sh
zig build run-cuda-pipeline-test
```

For Linux with AMD GPU (testing ROCm backend):

```sh
zig build run-rocm-pipeline-test
```

## Usage

The distributed training system consists of one Shepherd (coordinator) and multiple Workers (compute nodes).

### Starting the Shepherd

The Shepherd coordinates training, aggregates gradients, and manages the global model state.

```sh
./zig-out/bin/main_distributed \
  --shepherd \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 2 \
  --model ./src/models/nanogpt_forward.mlir
```

**Flags:**
- `--host`: Interface to bind to (default: 127.0.0.1)
- `--port`: TCP port to listen on (default: 8080)
- `--workers`: Number of workers to wait for before starting training
- `--model`: Path to the StableHLO MLIR model file

### Starting Workers

Workers connect to the Shepherd, compile the MLIR graph for their specific hardware, and perform the inner training loops.

#### CPU Worker

```sh
./zig-out/bin/main_distributed \
  --worker \
  --connect <SHEPHERD_IP>:8080 \
  --backend cpu
```

#### NVIDIA GPU (CUDA)

```sh
./zig-out/bin/main_distributed \
  --worker \
  --connect <SHEPHERD_IP>:8080 \
  --backend cuda
```

#### Apple Silicon (Metal)

```sh
./zig-out/bin/main_distributed \
  --worker \
  --connect <SHEPHERD_IP>:8080 \
  --backend metal
```

#### AMD GPU (ROCm)

For AMD GPUs, you must specify the target architecture using the `--amd-target` flag:

- MI300X: `gfx942`
- MI250X: `gfx90a`
- MI210: `gfx90a`
- MI100: `gfx908`

```sh
./zig-out/bin/main_distributed \
  --worker \
  --connect <SHEPHERD_IP>:8080 \
  --backend rocm \
  --amd-target gfx942
```

### Example: Mixed Hardware Cluster

You can run workers with different hardware backends in the same training session. The Shepherd will compile separate VMFB artifacts for each unique (backend, amd_target) combination.

```sh
# Terminal 1: Start Shepherd
./zig-out/bin/main_distributed --shepherd --workers 3

# Terminal 2: CPU Worker
./zig-out/bin/main_distributed --worker --backend cpu

# Terminal 3: CUDA Worker
./zig-out/bin/main_distributed --worker --backend cuda

# Terminal 4: AMD MI300X Worker
./zig-out/bin/main_distributed --worker --backend rocm --amd-target gfx942
```

Alternatively, use the provided test script for AMD clusters:

```sh
./run_mi300_cluster.sh
```

## Running Tests

Pipeline tests verify the MLIR-to-Hardware toolchain without running a full distributed session.

```sh
# Verify CPU Pipeline
zig build run-cpu-pipeline-test

# Verify CUDA Pipeline (requires NVIDIA GPU)
zig build run-cuda-pipeline-test

# Verify ROCm Pipeline (requires AMD GPU)
zig build run-rocm-pipeline-test

# Verify Optimizer Numerics
zig test src/examples/mlir_optimizer_test.zig

# Verify Autodiff Gradients
zig test src/examples/isolated_vjp_tests.zig
```
