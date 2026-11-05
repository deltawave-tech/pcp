#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BLOCK_KEYS = {"distributed", "decoupled_diloco", "artifacts", "dataset", "outputs"}

REGULAR_KEYS = {
    "model_path",
    "data_path",
    "tokenizer",
    "sampling",
    "learning_rate",
    "tau",
    "outer_loop_steps",
    "nesterov_momentum",
    "max_epochs",
    "dtype",
    "aggregation_strategy",
    "num_fragments",
    "sync_interval_h",
    "overlap_tau",
    "min_quorum",
    "grace_window_ms",
    "grace_gamma",
    "max_grace_steps",
    "adaptive_grace_enabled",
    "merge_strategy",
    "fragment_strategy",
    "learner_alpha",
    "outer_learning_rate",
    "outer_gradient_compression",
    "max_recovery_syncer_steps",
    "embedding_tensor_indices",
    "effective_batch_size",
    "use_in_graph_accumulation",
    "grpo_config",
    "checkpoint_dir",
    "should_resume",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_api_key",
}

DECOUPLED_KEYS = {
    "num_fragments",
    "sync_interval_h",
    "overlap_tau",
    "min_quorum",
    "grace_window_ms",
    "grace_gamma",
    "max_grace_steps",
    "adaptive_grace_enabled",
    "merge_strategy",
    "fragment_strategy",
    "learner_alpha",
    "outer_learning_rate",
    "outer_gradient_compression",
    "max_recovery_syncer_steps",
    "embedding_tensor_indices",
}

def put_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def block(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if isinstance(value, dict):
        return dict(value)
    return {}


def value(config: dict[str, Any], block_name: str, block_key: str, legacy_key: str | None = None) -> Any:
    current_block = block(config, block_name)
    if block_key in current_block:
        return current_block[block_key]
    return config.get(legacy_key or block_key)


def carry_unknown(config: dict[str, Any], known: set[str]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in known and k not in BLOCK_KEYS}


def compact(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if not (isinstance(v, dict) and len(v) == 0)}


def normalize_regular(config: dict[str, Any]) -> dict[str, Any]:
    output = carry_unknown(config, REGULAR_KEYS)

    artifacts = block(config, "artifacts")
    put_if_present(artifacts, "model_path", value(config, "artifacts", "model_path"))

    dataset = block(config, "dataset")
    put_if_present(dataset, "path", value(config, "dataset", "path", "data_path"))
    put_if_present(dataset, "tokenizer", value(config, "dataset", "tokenizer"))
    put_if_present(dataset, "sampling", value(config, "dataset", "sampling"))

    distributed = block(config, "distributed")
    mappings = {
        "aggregation_strategy": "aggregation_strategy",
        "outer_loop_steps": "outer_loop_steps",
        "local_steps_per_round": "tau",
        "learning_rate": "learning_rate",
        "outer_momentum": "nesterov_momentum",
        "max_epochs": "max_epochs",
        "dtype": "dtype",
        "effective_batch_size": "effective_batch_size",
        "use_in_graph_accumulation": "use_in_graph_accumulation",
    }
    for block_key, legacy_key in mappings.items():
        put_if_present(distributed, block_key, value(config, "distributed", block_key, legacy_key))

    decoupled = block(config, "decoupled_diloco")
    for key in DECOUPLED_KEYS:
        put_if_present(decoupled, key, value(config, "decoupled_diloco", key))

    outputs = block(config, "outputs")
    for key in ("checkpoint_dir", "should_resume", "wandb_project", "wandb_entity", "wandb_run_name", "wandb_api_key"):
        put_if_present(outputs, key, value(config, "outputs", key))

    put_if_present(output, "grpo_config", config.get("grpo_config"))
    output.update(compact({
        "artifacts": artifacts,
        "dataset": dataset,
        "distributed": distributed,
        "decoupled_diloco": decoupled,
        "outputs": outputs,
    }))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Render legacy training configs into the converged PCP training config shape.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--kind", choices=("auto", "regular"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    normalized = normalize_regular(config)
    rendered = json.dumps(normalized, indent=2, sort_keys=False) + "\n"

    if args.in_place:
        args.config.write_text(rendered, encoding="utf-8")
    elif args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
