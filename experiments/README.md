# PCP Experiments

This directory contains experiment configurations for distributed training with PCP. Each experiment specifies model architecture, dataset, hyperparameters, and training configuration.

## Table of Contents

- [NanoGPT Small](#nanogpt-small)
- [NanoGPT Medium](#nanogpt-medium)
- [NanoGPT Large](#nanogpt-large)
- [Qwen RL Test](#qwen-rl-test)

## NanoGPT Small

### Model Specifications
- **Context Length**: 32 tokens
- **Embedding Dimension**: 64
- **Layers**: 2
- **Attention Heads**: 8 (head dimension = 8)
- **Vocabulary Size**: 65 (character-level, dataset-dependent)
- **Parameters**: ~85K

This is a minimal configuration suitable for:
- Quick prototyping and testing
- CI/CD validation
- Single-GPU development
- Understanding the training pipeline

### Dataset
The `tiny_shakespeare.txt` dataset is included in the PCP repository under `data/`. No additional setup is required.

- **Size**: ~1MB
- **Source**: Concatenated Shakespeare works
- **Tokenization**: Character-level (dynamic vocabulary)

### Experiment Configuration

```json
{
    "model_path": "models/nanogpt_small.mlir",
    "data_path": "data/tiny_shakespeare.txt",
    "learning_rate": 0.0006,
    "tau": 50,
    "outer_loop_steps": 100,
    "nesterov_momentum": 0.9,
    "max_epochs": 10,
    "wandb_project": "pcp-distributed",
    "wandb_entity": null,
    "wandb_run_name": "nanogpt-small-experiment",
    "wandb_api_key": null
}
```

### Running the Experiment

```bash
# Generate the MLIR model
python tools/generate_nanogpt_small.py

# Run on a single node
zig build run-shepherd -- experiments/nanogpt_small.json

# Or use the convenience script
./run_build.sh experiments/nanogpt_small.json
```

## NanoGPT Medium

### Model Specifications
- **Context Length**: 128 tokens
- **Embedding Dimension**: 256
- **Layers**: 4
- **Attention Heads**: 8 (head dimension = 32)
- **Vocabulary Size**: 256 (byte-level, dataset-agnostic)
- **Parameters**: ~3.3M

This configuration is designed for:
- Realistic distributed training workloads
- Multi-GPU clusters
- Memory and compute stress testing
- Learning meaningful patterns from larger datasets

### Dataset

The medium experiment uses **enwik8**, a 100MB dump of Wikipedia text commonly used for character-level language modeling benchmarks (Hutter Prize).

- **Size**: 100MB (100M bytes)
- **Source**: First 100MB of English Wikipedia (XML format)
- **Tokenization**: Byte-level (0-255 ASCII values)
- **Download**: Automated via setup script

#### Setting Up the Dataset

Run the following script on **each worker node** to download and prepare enwik8:

```bash
cd data/
./setup_enwik8.sh
```

The script will:
1. Download `enwik8.zip` from http://mattmahoney.net/dc/enwik8.zip
2. Extract the archive
3. Rename `enwik8` → `enwik8.txt`
4. Clean up the zip file

After setup, the file will be available at `data/enwik8.txt`, matching the `data_path` in the experiment configuration.

**Note**: Byte-level tokenization (vocab size 256) makes the model dataset-agnostic. Unlike character-level tokenization, the vocabulary is fixed and doesn't depend on the dataset content.

### Experiment Configuration

```json
{
    "model_path": "models/nanogpt_medium.mlir",
    "data_path": "data/enwik8.txt",
    "learning_rate": 0.0006,
    "tau": 10,
    "outer_loop_steps": 200,
    "nesterov_momentum": 0.9,
    "max_epochs": 10,
    "wandb_project": "pcp-distributed",
    "wandb_entity": null,
    "wandb_run_name": "nanogpt-medium-experiment",
    "wandb_api_key": null
}
```

## NanoGPT Large

TODO

## Qwen RL Test

### Overview

GRPO (Group Relative Policy Optimization) reinforcement learning with Qwen 2.5 0.5B Instruct. This experiment demonstrates distributed RL training using an RL Shepherd controller and CUDA workers for parallel rollout generation.

### Model Specifications

- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct (HuggingFace)
- **Parameters**: ~494M (1.88 GB in fp32)
- **Context Length**: 1024 tokens
- **KV Heads**: 2 (GQA)
- **Layers**: 24
- **Hidden Dim**: 896
- **Vocabulary**: 151,936 tokens

### Architecture

```
┌─────────────────┐
│   RL Shepherd   │  Coordinator: loads weights, dispatches prompts,
│                 │  collects rollouts, computes GRPO updates
└────────┬────────┘
         │ TCP (weights, prompts, completions)
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Worker │ │Worker │  CUDA workers: run generation VMFB,
│  0    │ │  1    │  return token completions
└───────┘ └───────┘
```

### Files

| File | Description |
|------|-------------|
| `experiments/qwen_rl_test.json` | Experiment configuration |
| `models/qwen_rl_generation.mlir` | Generation model (stateless RoPE, KV cache I/O) |
| `models/qwen_rl_generation.vmfb` | Compiled IREE module for CUDA (~520 MB) |
| `models/qwen_grpo_training.vmfb` | Training backward pass for GRPO updates |
| `models/*.mlir.meta.json` | Auto-generated metadata cache (parameter shapes, dtypes) |
| `checkpoints/initial_weights/qwen_flat.bin` | Flattened HuggingFace weights (1.88 GB) |
| `data/rl_prompts.bin` | Binary-encoded prompt token IDs |

**Note on meta.json files**: These are auto-generated by the RL Shepherd when it first loads an MLIR model. They cache parameter shapes and data input shapes to avoid re-parsing MLIR on subsequent runs. The `export_weights_only.py` script uses them to verify weight file alignment.

### Export Scripts

**Generation Model** (`tools/export_qwen_generation.py`):
- Exports Qwen with functional parameters (weights as inputs)
- Implements stateless RoPE (no captured buffers)
- KV cache passed in/out for autoregressive generation
- Outputs: MLIR → compile with `iree-compile` to VMFB

**Training Model** (`tools/export_qwen_forward.py`):
- Exports forward pass (loss computation) for GRPO training
- PCP's autodiff engine computes gradients from this forward pass
- Stateless RoPE (matches generation model, no buffer inputs)

**Weights** (`tools/export_weights_only.py`):
- Extracts `model.named_parameters()` to flat binary
- Order must match MLIR function signature exactly

**Prompts** (`tools/prepare_rl_dataset.py`):
- Tokenizes text prompts to binary format
- Format: `[num_prompts:u32] [len:u32 tokens:u64[]]*`

### Prompt Dataset

Prompts are loaded from `data/rl_prompts.bin` (not from the JSON config). The binary file is created by `tools/prepare_rl_dataset.py`.

Current test prompts:
```
2+2=
The capital of France is
Write a python function to add two numbers.
Is fire hot? Answer yes or no.
Solve 3 * 4 + 2.
```

### GRPO Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `num_iterations` | 1 | Number of GRPO training iterations |
| `group_size` | 4 | Completions per prompt for relative ranking |
| `learning_rate` | 1e-6 | Policy gradient learning rate |
| `beta` | 0.1 | KL penalty coefficient |
| `prompt_file` | `data/rl_prompts.bin` | Binary file containing tokenized prompts |
| `num_prompts` | 1 | Number of prompts to use from prompt_file |
| `max_tokens` | 128 | Maximum tokens per completion |
| `temperature` | 0.7 | Sampling temperature |

**Rollouts per iteration**: `num_prompts × group_size` (configurable via JSON)

The GRPO algorithm:
1. Generates `group_size` completions for each prompt
2. Scores completions (length, EOS presence, diversity)
3. Computes relative advantages within each group
4. Updates policy using advantage-weighted gradients

### Experiment Configuration

```json
{
    "model_path": "models/qwen_rl_generation.mlir",
    "training_model_path": "models/qwen_grpo_training.vmfb",
    "weights_path": "checkpoints/initial_weights/qwen_flat.bin",
    "tokenizer": "qwen",
    "learning_rate": 1e-5,
    "nesterov_momentum": 0.9,
    "grpo_config": {
        "num_iterations": 1,
        "group_size": 4,
        "learning_rate": 1e-6,
        "beta": 0.1,
        "prompt_file": "data/rl_prompts.bin",
        "num_prompts": 1
    },
    "rollout_config": {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "batch_size": 4
    },
    "wandb_project": "pcp-qwen-rl",
    "wandb_run_name": "qwen-grpo-test"
}
```

### Running the Experiment

```bash
# 1. Export models (if not already done)
python tools/export_qwen_generation.py
python tools/export_qwen_forward.py

# 2. Compile to VMFB
iree-compile models/qwen_rl_generation.mlir \
    --iree-hal-target-backends=cuda \
    -o models/qwen_rl_generation.vmfb

# 3. Export weights
python tools/export_weights_only.py

# 4. Prepare prompts
python tools/prepare_rl_dataset.py

# 5. Run (uses run_qwen_rl_test.sh convenience script)
./run_qwen_rl_test.sh
```

The test script starts an RL Shepherd and one CUDA worker. The shepherd:
1. Loads weights and broadcasts to workers
2. Sends prompts for rollout generation
3. Collects completions and computes rewards
4. Applies GRPO gradient updates
5. Broadcasts updated weights for next iteration

### Key Implementation Details

- **Stateless RoPE**: Position embeddings computed from `position_ids` input, not module buffers
- **KV Cache**: Shape `[batch, kv_heads, seq_len, head_dim]` passed as function I/O
- **Weight Injection**: All 290 parameters passed as function inputs for gradient flow
- **Causal Mask**: Computed inside the model from position IDs

## Configuration Reference

All experiment JSON files support the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `model_path` | string | Path to compiled MLIR model (.mlir file) |
| `data_path` | string | Path to training dataset (text file) |
| `learning_rate` | float | Outer loop learning rate for pseudo-gradients |
| `tau` | integer | Number of inner loop SGD steps per worker |
| `outer_loop_steps` | integer | Number of DiLoCo outer loop iterations |
| `nesterov_momentum` | float | Momentum coefficient (0.0-1.0) |
| `max_epochs` | integer | Maximum training epochs |
| `wandb_project` | string | Weights & Biases project name |
| `wandb_entity` | string/null | W&B team/user entity |
| `wandb_run_name` | string | Experiment run identifier |
| `wandb_api_key` | string/null | W&B API key (or set via environment) |


## Best Practices

- **Start Small**: Test new configurations with `nanogpt_small.json` before scaling up
- **Dataset Consistency**: Ensure all worker nodes have the same dataset at the specified `data_path`
- **Memory Planning**: Medium model is lightweight (< 1GB VRAM per worker), making it ideal for testing distributed logic without massive hardware
- **Monitoring**: Configure W&B credentials to track training progress across distributed workers
- **Tau Tuning**: Higher `tau` values reduce synchronization overhead but may hurt convergence speed
