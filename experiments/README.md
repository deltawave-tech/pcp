# PCP Experiments

This directory contains experiment configurations for distributed training with PCP. Each experiment specifies model architecture, dataset, hyperparameters, and training configuration.

## Table of Contents

- [NanoGPT Small](#nanogpt-small)
- [NanoGPT Medium](#nanogpt-medium)
- [NanoGPT Large](#nanogpt-large)

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
