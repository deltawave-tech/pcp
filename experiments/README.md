# PCP Experiments

This directory contains JSON configs consumed by gateway embedded controllers,
export scripts, and local smoke scripts. The root-level scripts create gateway
configs around these files when an end-to-end local run needs a gateway,
controller, and worker.

## Files

- `gateway_local.json`: minimal gateway config without embedded controllers.
- `inference_qwen.json`: embedded inference controller config for Qwen
  2.5 0.5B Instruct.
- `inference_qwen3_8b.json`: embedded inference controller config for
  Qwen3-8B.
- `inference_template.json`: inference config template with the same schema.
- `qwen_rl_test.json`, `qwen_rl_api_smoke.json`, `qwen3_8b_rl_test.json`: GRPO
  RL configs.
- `nanogpt_*.json`: character or byte-level NanoGPT training configs.
- `nanochat_*.json`: NanoChat training configs for `.mlir` exports and
  tokenized `.u16` datasets.
- `rope_test.json`, `toy_qwen_test.json`: small model validation configs.

## Standard Training Config

Standard training config fields are parsed by `src/experiment_config.zig`.

```json
{
  "model_path": "models/nanochat_smoke_32.mlir",
  "data_path": "data/fineweb_edu_2shards_1m.u16",
  "tokenizer": "u16",
  "sampling": "random",
  "dtype": "f32",
  "learning_rate": 0.0006,
  "tau": 2,
  "outer_loop_steps": 3,
  "nesterov_momentum": 0.9,
  "max_epochs": 10,
  "wandb_project": "pcp-distributed",
  "wandb_entity": null,
  "wandb_run_name": "nanochat-u16-smoke-32",
  "wandb_api_key": null
}
```

Supported tokenizers include `char`, `byte`, `u16`, and `qwen`. Supported
training dtypes are `f32`, `bf16`, and `f16`.

Optional fields parsed by `src/experiment_config.zig` include:

- `effective_batch_size`
- `use_in_graph_accumulation`
- `checkpoint_dir`
- `should_resume`

Some run JSONs also carry data-preparation keys such as `chunk_manifest_path`,
`chunk_shuffle`, and `seed`. Those keys are ignored by the PCP experiment parser
and are available to surrounding scripts or cluster tooling.

## Decoupled DiLoCo Profile

Set `"distributed": {"aggregation_strategy": "decoupled_diloco"}` to route
standard training through the generic Decoupled DiLoCo syncer. The tracked
paper-profile example is `nanochat_decoupled_diloco_paper.json`. It uses the
converged config blocks:

```json
{
  "distributed": {
    "aggregation_strategy": "decoupled_diloco",
    "outer_loop_steps": 2,
    "local_steps_per_round": 24
  },
  "decoupled_diloco": {
    "num_fragments": 24,
    "sync_interval_h": 24,
    "overlap_tau": 2,
    "min_quorum": 1,
    "fragment_strategy": "balanced_tensor",
    "merge_strategy": "avg_embedding_rda_model",
    "learner_alpha": 0.0,
    "adaptive_grace_enabled": true
  }
}
```

Legacy flat keys remain accepted with a warning during migration. Convert old
configs with:

```sh
tools/normalize_training_config.py experiments/nanochat_decoupled_diloco_paper.json --kind regular
```

Small smoke tests pin lower `num_fragments` and `sync_interval_h` values for
determinism. Synchronous `"diloco"` remains supported for direct comparison
runs.

Validate the example locally:

```sh
./venv/bin/python test/nanochat/test_gateway_decoupled_diloco_paper_profile_smoke.py
```

The final `training_state.json` records the aggregation, worker count, syncer
step count, fragment settings, quorum settings, merge and fragment strategies,
byte counters for learner/syncer fragment exchange, event-tape entry count,
vector-clock entry count, skipped learner count, and the last grace window.

## Gateway Execution

Run a gateway with an embedded training controller by providing a gateway config
whose controller `config_path` points at one of these experiment JSON files.
The local smoke scripts generate those gateway configs under `/tmp`.

```sh
./run_gateway_training_api.local.sh
```

For a direct gateway launch:

```sh
PCP_GATEWAY_API_TOKEN=dev \
PCP_GATEWAY_INTERNAL_TOKEN=dev \
./result/bin/pcp \
  --gateway \
  --gateway-config gateway_training.json \
  --gateway-host 127.0.0.1 \
  --gateway-port 18010 \
  --control-host 127.0.0.1 \
  --control-port 8080
```

Workers connect to the worker-fabric endpoint:

```sh
./result/bin/pcp --worker --connect 127.0.0.1:8080 --backend cpu
```

## Inference

`inference_qwen.json` is parsed by `src/inference/config.zig`. It binds the
generation VMFB, source MLIR, flattened weights, tokenizer path, context limits,
worker backend, target architecture, and API token environment variable.

Use the fixed smoke script for an end-to-end local inference run:

```sh
./run_qwen_inference_smoke.sh
```

The gateway proxy endpoint is:

```text
POST /v1/inference/chat/completions
```

## RL

The GRPO configs place rollout and training artifact paths under `grpo_config`.
The embedded RL controller is enabled from a gateway config and accepts:

```text
POST /v1/rl/jobs
```

Useful local scripts:

```sh
./run_gateway_rl_api.local.sh
./run_qwen_rl_test.sh
```

## Export And Data Tools

- `tools/generate_nanogpt_small.py`
- `tools/generate_nanogpt_medium.py`
- `tools/generate_nanochat.py`
- `tools/generate_nanochat_bf16.py`
- `tools/export_qwen_generation.py`
- `tools/export_qwen_forward.py`
- `tools/export_qwen3_generation.py`
- `tools/export_qwen3_training.py`
- `tools/prepare_nanochat_parquet_to_u16.py`
- `tools/prepare_rl_dataset.py`
