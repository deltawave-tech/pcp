#!/usr/bin/env python3
"""Prepare Qwen3.6-27B text-generation contract artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen36_27b_contract import (
    DEFAULT_MAX_SEQ_LEN,
    MODEL_ID,
    build_generation_contract,
    facts_from_config,
    write_contract,
    write_metadata,
    write_tiny_dynamic_update_training_mlir,
    write_tiny_hybrid_mlir,
    write_tiny_hybrid_ops_mlir,
    write_tiny_hybrid_training_mlir,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = REPO_ROOT / "models" / "qwen36_27b_text_generation.contract.json"
DEFAULT_METADATA_PATH = REPO_ROOT / "models" / "qwen36_27b_text_generation.mlir.meta.json"
DEFAULT_TINY_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_hybrid.mlir"
DEFAULT_TINY_OPS_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_hybrid_ops.mlir"
DEFAULT_TINY_TRAINING_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_hybrid_training.mlir"
DEFAULT_TINY_DYNAMIC_UPDATE_TRAINING_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_dynamic_update_training.mlir"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--state-dtype", choices=("bf16", "f16"), default="bf16")
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--tiny-mlir-path", type=Path, default=DEFAULT_TINY_MLIR_PATH)
    parser.add_argument("--tiny-ops-mlir-path", type=Path, default=DEFAULT_TINY_OPS_MLIR_PATH)
    parser.add_argument("--tiny-training-mlir-path", type=Path, default=DEFAULT_TINY_TRAINING_MLIR_PATH)
    parser.add_argument(
        "--tiny-dynamic-update-training-mlir-path",
        type=Path,
        default=DEFAULT_TINY_DYNAMIC_UPDATE_TRAINING_MLIR_PATH,
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Fetch the Hugging Face config with transformers and validate it before writing the contract.",
    )
    return parser.parse_args()


def load_hf_config(model_id: str):
    try:
        from transformers import AutoConfig  # type: ignore
    except ImportError as exc:
        raise SystemExit("transformers is required for --from-hf; install requirements.txt first") from exc
    return AutoConfig.from_pretrained(model_id, trust_remote_code=True)


def main() -> None:
    args = parse_args()
    config = load_hf_config(args.model_id) if args.from_hf else {"model_id": args.model_id}
    facts = facts_from_config(config, external_weight_dtype=args.state_dtype)
    contract = build_generation_contract(facts, max_seq_len=args.max_seq_len, state_dtype=args.state_dtype)
    write_contract(contract, args.contract_path)
    write_metadata(contract, args.metadata_path)
    write_tiny_hybrid_mlir(args.tiny_mlir_path)
    write_tiny_hybrid_ops_mlir(args.tiny_ops_mlir_path)
    write_tiny_hybrid_training_mlir(args.tiny_training_mlir_path)
    write_tiny_dynamic_update_training_mlir(args.tiny_dynamic_update_training_mlir_path)

    print(f"Wrote Qwen3.6-27B text contract: {args.contract_path}")
    print(f"Wrote Qwen3.6-27B text metadata: {args.metadata_path}")
    print(f"Wrote tiny hybrid MLIR validation artifact: {args.tiny_mlir_path}")
    print(f"Wrote tiny hybrid ops MLIR artifact: {args.tiny_ops_mlir_path}")
    print(f"Wrote tiny hybrid training MLIR artifact: {args.tiny_training_mlir_path}")
    print(f"Wrote tiny dynamic-update training MLIR artifact: {args.tiny_dynamic_update_training_mlir_path}")
    print(f"Data inputs: {len(contract.data_input_shapes)}")
    print(f"State slots: {len(contract.state_slots)}")


if __name__ == "__main__":
    main()
