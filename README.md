# PCP

PCP is a distributed tensor computation framework written in Zig. It employs MLIR and the IREE compiler toolchain as its core to provide automatic differentiation, hardware acceleration, and optimization.

## Table of Contents
- [Core Architecture](#core-architecture)
- [Quickstart](#quickstart)
- [Building the Project via Nix](#building-the-project-via-nix)
- [Building the Project manually](#building-the-project-manually-without-nix)
- [Usage](#usage)
- [Adding Custom Models](#adding-custom-models)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [Future Work](#future-work)

## Core Architecture

PCP transforms high-level tensor operations into optimized MLIR computation graphs. The system's design is centered on a forward-pass graph construction, which is then used to derive a corresponding gradient graph for automatic differentiation. The protocol has a modular design to enable composability of components such as distributed training algorithms, optimzers, automatic differentiation methods, compute backends, and network topologies. The current state of the protocol implements a basic configuration:
- Distributed training algorithm: DiLoCo
- Optimizers: AdamW, Nesterov
- Automatic differentiation method: Reverse AD with VJP rules
- Backends: cuda, rocm, cpu, msl (experimental)

### Distributed Training Overview

The Shepherd (controller) constructs a complete MLIR training graph using the StableHLO dialect. This graph is then compiled by the IREE compiler into a portable `.vmfb` artifact for a specific hardware target (e.g., cuda, rocm, ...). This binary artifact is sent to workers, which execute it using the cross-platform IREE runtime.

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
│         │   │                           │   │  (via *.vmfb)   │
└─────────┘   └───────────────────────────┘   └─────────────────┘
```

## Quickstart

### 1. Install Nix and Cachix

```shell
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon
```

Enable flakes:
```shell
mkdir -p ~/.config/nix/
echo 'experimental-features = nix-command flakes' >> ~/.config/nix/nix.conf
```

Install cachix
```shell
nix profile install nixpkgs#cachix
cachix use pcp
```

### 2. Install PCP

Configure nix to use cachix as a substituter:
```shell
cachix use pcp
```

Install PCP:
```shell
nix profile add github:deltawave-tech/pcp
```

**NOTE** There should be **no** compilation happening at this point (except maybe pcp which should
not take longer than around five minutes).  Especially there is no reason to see LLVM or IREE
compiling.


### 3. Set Up Python Environment (only needed on Shepherd node)

PCP requires Python dependencies for WandB tracking:

```shell
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure WandB

```shell
export WANDB_API_KEY=your_api_key_here
```

### 5. Start the Shepherd

Use example experiment configuration file `experiments/nanogpt_small.json`, or use a bigger model configuration with a larger dataset: [nanogpt_medium](experiments/README.md#nanogpt-medium)

Start a supervised Shepherd expecting (8) workers:
```shell
pcp --supervise -- --shepherd --config experiments/nanogpt_small.json --host 0.0.0.0 --port 8080 --workers 8
```

### 6. Start Worker Nodes

**Single GPU node:**
```shell
pcp --node-manager --host <SHEPHERD_IP> --port 8080 --backend cuda --target sm_80
```

**Multi-GPU node (8xH100):**
```shell
pcp --node-manager --scale 8 --host <SHEPHERD_IP> --port 8080 --backend cuda --target sm_90a
```

See [Starting Workers](#starting-workers) for more details on GPU target architectures.

Training will begin automatically once all (8) workers connect.

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

### Step 3: Configure Environment

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

### Configuration

**Infrastructure (CLI Flags):**
- Networking: `--host`, `--port`
- Hardware: `--backend`, `--target`, `--device-id`
- Topology: `--workers`, `--scale`
- Supervision: `--supervise`, `--resume`
- Modes: `--shepherd`, `--worker`, `--node-manager`

**Experiment (JSON File):**
- Model: `model_path`
- Data: `data_path`
- Hyperparameters: `learning_rate`
- Algorithm: `tau`, `outer_loop_steps`, `max_epochs`, `nesterov_momentum`
- Logging: `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_api_key`
- Recovery: `checkpoint_dir`, `should_resume`

Note: Batch and block sizes are determined by the compiled MLIR model file, not the configuration.

Create an experiment configuration file (e.g., `experiment.json`):

```json
{
    "model_path": "models/nanogpt_small.mlir",
    "data_path": "data/tiny_shakespeare.txt",
    "learning_rate": 0.0006,
    "tau": 50,
    "outer_loop_steps": 100,
    "max_epochs": 10,
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

The Shepherd coordinates training, aggregates gradients, and manages the global model state. Running with `--supervise` enables automatic crash recovery.

```sh
pcp --supervise -- \
  --shepherd \
  --config experiment.json \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 2
```

To resume from a previous training run:
```sh
pcp --supervise -- \
  --shepherd \
  --config experiment.json \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 2 \
  --resume
```

### Starting Workers

Workers connect to the Shepherd and execute training on their local hardware. We use the Node Manager to launch and monitor workers. The Node Manager automatically handles process supervision, crash recovery, and GPU assignment.

**GPU Target Architectures:**

NVIDIA: A100 (sm_80), V100 (sm_70), RTX 4090 (sm_89), H100 (sm_90a)
AMD: MI300X (gfx942), MI250X/MI210 (gfx90a), MI100 (gfx908)

#### Basic Usage (Single GPU or CPU)

Run this on any worker node. By default, it spawns 1 worker on Device 0.

```sh
pcp \
  --node-manager \
  --host <SHEPHERD_IP> \
  --port 8080 \
  --backend cuda \
  --target sm_80
```

#### Multi-GPU Scaling

To utilize a multi-GPU node (e.g., 8xH100), add the `--scale` flag. The Node Manager will spawn 8 independent, supervised workers, pinning each to a specific GPU (0-7).

```sh
pcp \
  --node-manager \
  --scale 8 \
  --host <SHEPHERD_IP> \
  --port 8080 \
  --backend cuda \
  --target sm_90a
```

#### Manual/Debugging Mode

If you need to debug a specific worker process without the supervisor layer, you can run a worker directly:

```sh
pcp --worker --device-id 0 --connect <SHEPHERD_IP>:8080 --backend cuda
```

### Example: Heterogeneous Multi-Node Cluster

You can run workers with different hardware backends in the same training session.

```sh
# Shepherd Node: Start coordinator expecting 11 total workers
pcp --shepherd --config experiments/nanogpt_small.json --workers 11

# Node 1: 8xH100 server (Runs 8 workers)
pcp --node-manager --scale 8 --host <SHEPHERD_IP> --port 8080 --backend cuda --target sm_90a

# Node 2: Single A100 server (Runs 1 worker)
pcp --node-manager --host <SHEPHERD_IP> --port 8080 --backend cuda --target sm_80

# Node 3: 2xMI300X server (Runs 2 workers)
pcp --node-manager --scale 2 --host <SHEPHERD_IP> --port 8080 --backend rocm --target gfx942
```

## Adding Custom Models

PCP ships with example models (NanoGPT) for demonstration and testing. When generating a new model for training, you have to go through these steps:

- Verifying the correctness of your PyTorch model implementation
- Ensuring the model outputs a scalar loss value (not logits)
- Adapting the data loader to match your dataset format and vocabulary
- Testing your model independently before integrating with PCP

PCP allows you to train arbitrary PyTorch architectures as long as they can be compiled to StableHLO.

**Prerequisites**: Before generating custom models, ensure you have set up the Python environment with the required dependencies:

```shell
# If not already created
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies (torch, torch-mlir, wandb)
pip install -r requirements.txt
```

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

We provide a utility tool to trace your PyTorch model and convert it into a StableHLO MLIR module that PCP can execute.

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

The MLIR file has fixed input shapes (batch size and block size) burned into it during compilation. These shapes are introspected automatically by PCP at runtime.

Create or update your experiment.json file:

```json
{
    "model_path": "models/my_custom_model.mlir",
    "data_path": "data/tiny_shakespeare.txt",
    "learning_rate": 0.0006,
    "tau": 10,
    "outer_loop_steps": 100,
    "max_epochs": 10
}
```

### 4. Run Training

Start the Shepherd with your new configuration:

```sh
pcp --supervise -- --shepherd --config experiment.json --workers 2
```

## Documentation

For detailed technical documentation, architecture guides, and API references, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Limitations

PCP currently implements data parallelism only. Model size is constrained by individual worker node memory capacity, as each worker maintains a complete copy of the model parameters. This limits scalability for very large models without implementing model parallelism techniques.

The system requires manual conversion of PyTorch models to MLIR format. New architectures may require extending the operator library and corresponding VJP (vector-Jacobian product) rules for automatic differentiation. This process demands understanding of both the model architecture and the MLIR/StableHLO representation.

## Future Work

- Expand to support advanced distributed training algorithms beyond DiLoCo, including StreamingDiLoCo, NoLoCo, MuLoCo, etc

- MLIR integration will be extended with a dialect for distributed heterogeneous computing à la ([PLDI 2025](https://pldi25.sigplan.org/details/pldi-2025-src/3/An-MLIR-Dialect-for-Distributed-Heterogeneous-Computing)). This enables first-class representation of distributed computation patterns directly in the compiler infrastructure, allowing hardware-specific optimizations across heterogeneous clusters.

- We will experiment with alternative automatic differentiation methods beyond reverse-mode AD for exploration of approaches with reduced memory pressure.
