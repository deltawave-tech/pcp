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

## Building the Project via Nix

The project uses [Nix](https://nixos.org/download/) as a build system.  It is possible to compile
dependencies locally on your own, but that requires a considerable amount of manual work and
compilation time.  We cache build and runtime dependencies and the builds themselves via
[cachix](https://app.cachix.org/cache/pcp).

No matter what, you will have to [install Nix and enable flakes](#install-nix-and-enable-flakes).

If you then just want to perform training runs follow [Install final builds](#install-nix-and-enable-flakes).
If you want to actively develop on pcp follow [Install a build environment](#install-a-build-environment).


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

If you just want to grab the final build, you can add cachix manually:

```shell
cat << EOF >> ~/.config/nix/nix.conf
substituters = https://cache.nixos.org https://pcp.cachix.org
trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY= pcp.cachix.org-1:D/JYXqFAnVLlvVUJEBOWoGLmJKwKW58SxPD0+m/HXZk=
EOF
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

Fetch the repository from GitHub. At the repository root perform
```shell
nix develop
```

Nix will drop you into a shell with all required dependencies. You should now be able to compile the
project using `zig build`.

## Building the Project manually (without Nix)

### Prerequisites

- **Zig**: Version 0.13.0 .
- **System Build Tools**: `git`, `cmake`, `ninja`.
- **Cap'n Proto**: The build script will try to find it automatically. If it fails, you may need to install it or set the `CAPNP_DIR` environment variable.
  - **macOS**: `brew install capnp`
  - **Ubuntu/Debian**: `sudo apt install capnproto libcapnp-dev`

### Step 1: Clone This Repository

Clone this project repository.

```sh
git clone https://github.com/deltawave-tech/pcp.git
cd pcp
git checkout master
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

The distributed training system consists of one Shepherd (coordinator) and multiple Workers (compute nodes). PCP separates infrastructure configuration (CLI flags) from experiment configuration (JSON file), following best practices from PyTorch and DeepSpeed.

### Configuration

**Infrastructure (CLI Flags):**
- Networking: `--host`, `--port`, `--connect`
- Hardware: `--backend`, `--target`
- Topology: `--workers`

**Experiment (JSON File):**
- Model: `model_path`
- Data: `data_path`
- Hyperparameters: `learning_rate`, `batch_size`, `block_size`
- Algorithm: `tau`, `outer_loop_steps`, `nesterov_momentum`
- Logging: `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_api_key`

Create an experiment configuration file (e.g., `experiment.json`):

```json
{
    "model_path": "models/nanogpt_forward_32.mlir",
    "data_path": "data/tiny_shakespeare.txt",
    "learning_rate": 0.0006,
    "batch_size": 32,
    "block_size": 64,
    "tau": 50,
    "outer_loop_steps": 100,
    "nesterov_momentum": 0.9,
    "wandb_project": "pcp-distributed",
    "wandb_entity": null,
    "wandb_run_name": "my-experiment",
    "wandb_api_key": null
}
```

**Weights & Biases Integration:**

PCP automatically logs training metrics to [Weights & Biases](https://wandb.ai) for experiment tracking and visualization. To enable WandB logging:

1. **Set your API key** (required):
   ```sh
   export WANDB_API_KEY=your_api_key_here
   ```

2. **Configure logging parameters** in your experiment JSON:
   - `wandb_project`: Project name in WandB (default: "pcp-distributed")
   - `wandb_entity`: WandB team/username (optional, uses your default)
   - `wandb_run_name`: Custom name for this training run (optional, auto-generated if null)
   - `wandb_api_key`: API key (optional, reads from environment variable if null)

3. **Logged metrics include:**
   - Loss (average, min, max across workers)
   - Learning rate
   - Active worker count
   - Epoch time
   - Outer loop step progress

If no API key is provided, WandB logging will be disabled and training will continue normally.

### Starting the Shepherd

The Shepherd coordinates training, aggregates gradients, and manages the global model state.

```sh
./zig-out/bin/pcp \
  --shepherd \
  --config experiment.json \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 2
```

### Starting Workers (with Supervisor)

For production environments, it is recommended to run workers under a Supervisor. The Supervisor acts as a watchdog process that connects to the Shepherd via a control plane, spawns the actual Worker process, and automatically restarts it if it crashes.

```sh
# Start a Supervisor (which manages the Worker)
./zig-out/bin/pcp --supervisor --host <SHEPHERD_IP> --port 8080 --backend cuda --target sm_80
```

**Standard Worker (No Supervisor):**

```sh
# Connect directly (useful for debugging)
./zig-out/bin/pcp --worker --connect <SHEPHERD_IP>:8080 --backend cuda
```

Workers connect to the Shepherd and execute training on their local hardware. The `--target` flag specifies GPU architecture (optional, defaults: sm_80 for CUDA, gfx942 for ROCm).

**GPU Target Architectures:**

NVIDIA: A100 (sm_80), V100 (sm_70), RTX 4090 (sm_89)
AMD: MI300X (gfx942), MI250X/MI210 (gfx90a), MI100 (gfx908)

**CPU Worker:**

```sh
./zig-out/bin/pcp --worker --connect <SHEPHERD_IP>:8080 --backend cpu
```

**NVIDIA GPU:**

```sh
./zig-out/bin/pcp --worker --connect <SHEPHERD_IP>:8080 --backend cuda --target sm_80
```

**AMD GPU:**

```sh
./zig-out/bin/pcp --worker --connect <SHEPHERD_IP>:8080 --backend rocm --target gfx942
```

### Example: Mixed Hardware Cluster

You can run workers with different hardware backends in the same training session. The Shepherd will compile separate VMFB artifacts for each unique (backend, target) combination.

```sh
# Terminal 1: Start Shepherd
./zig-out/bin/pcp --shepherd --config experiment.json --workers 3

# Terminal 2: CPU Worker
./zig-out/bin/pcp --worker --connect 127.0.0.1:8080 --backend cpu

# Terminal 3: NVIDIA A100 Worker
./zig-out/bin/pcp --worker --connect 127.0.0.1:8080 --backend cuda --target sm_80

# Terminal 4: AMD MI300X Worker
./zig-out/bin/pcp --worker --connect 127.0.0.1:8080 --backend rocm --target gfx942
```

## Adding Custom Models

PCP ships with example models (NanoGPT) for demonstration and testing. When generating a new model for training, you have to go through these steps:

- Verifying the correctness of your PyTorch model implementation
- Ensuring the model outputs a scalar loss value (not logits)
- Adapting the data loader to match your dataset format and vocabulary
- Testing your model independently before integrating with PCP

PCP allows you to train arbitrary PyTorch architectures as long as they can be compiled to StableHLO.

### 1. Define Your PyTorch Model

Create a Python file (e.g., `my_gpt.py`). Your model class must adhere to the following signature for its forward pass:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers...

    # Input:
    #   idx: LongTensor of shape (Batch, BlockSize)
    #   targets: LongTensor of shape (Batch, BlockSize)
    # Output:
    #   loss: Scalar FloatTensor (must be the loss value, not logits)
    def forward(self, idx, targets):
        # ... logic ...
        # ... logits = self.head(x) ...
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return loss
```

Note: The model must return the loss as a scalar. The optimizer logic in PCP expects the final output of the graph to be the value to minimize.

### 2. Export to MLIR

We provide a utility tool to trace your PyTorch model and convert it into a "stateless" StableHLO MLIR module that PCP can execute.

Run the exporter tool:

```sh
python tools/export_model.py \
  --model-file my_gpt.py \
  --class-name MyModel \
  --out models/my_custom_model.mlir \
  --batch-size 64 \
  --block-size 64
```

What this does:

- Loads your Python class
- Wraps it in a StatelessWrapper (separating parameters from computation)
- Uses torch-mlir to compile the computation graph to StableHLO
- Saves the .mlir file to the models/ directory

### 3. Update Configuration

The MLIR file has fixed input shapes burned into it during compilation. You must ensure your runtime configuration matches these shapes.

The export script will print a JSON snippet at the end. Create or update your experiment.json file:

```json
{
    "model_path": "models/my_custom_model.mlir",
    "data_path": "data/tiny_shakespeare.txt",
    "batch_size": 64,
    "block_size": 64,
    "learning_rate": 0.0006,
    "tau": 10,
    "outer_loop_steps": 100
}
```

### 4. Run Training

Start the Shepherd with your new configuration:

```sh
./zig-out/bin/pcp --shepherd --config experiment.json --workers 2
```
