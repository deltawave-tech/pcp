#!/usr/bin/env python3
"""Export Qwen3.6-27B weights as streamed BF16/FP16 parameter bytes.

The output order is the MLIR external-parameter order when the meta file
contains ``parameter_names``. Without names, this falls back to the empty HF
module's ``named_parameters()`` order, which is the order used by the PCP
Qwen3.6 exporter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM


DEFAULT_MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_META_PATH = Path("models/qwen36_27b_text_generation.mlir.meta.json")
DEFAULT_OUTPUT_PATH = Path("checkpoints/initial_weights/qwen36_27b_bf16_streamed.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dtype", choices=("bf16", "f16"), default="bf16")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def empty_parameter_order(model_id: str) -> list[str]:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return [name for name, _ in model.named_parameters()]


def expected_order(model_id: str, meta: dict[str, Any]) -> list[str]:
    names = meta.get("parameter_names")
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return names
    return empty_parameter_order(model_id)


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "f16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def bytes_per_element(name: str) -> int:
    if name in {"bf16", "f16"}:
        return 2
    raise ValueError(f"unsupported dtype: {name}")


def numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def tensor_bytes(tensor: torch.Tensor, dtype_name: str) -> bytes:
    converted = tensor.detach().to(dtype=torch_dtype(dtype_name), device="cpu").contiguous()
    return converted.view(torch.uint16).numpy().tobytes(order="C")


def checkpoint_key(parameter_name: str, weight_map: dict[str, str]) -> str | None:
    candidates = [parameter_name]
    if parameter_name.startswith("model.model."):
        candidates.append("model.language_model." + parameter_name[len("model.model.") :])
    if parameter_name.startswith("model."):
        candidates.append("model.language_model." + parameter_name[len("model.") :])
    if parameter_name.startswith("language_model."):
        candidates.append("model." + parameter_name)
    for candidate in candidates:
        if candidate in weight_map:
            return candidate
    return None


def main() -> None:
    args = parse_args()
    meta = load_meta(args.meta)
    names = expected_order(args.model_id, meta)
    expected_shapes = meta.get("parameter_shapes") or []

    snapshot = Path(
        snapshot_download(
            args.model_id,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing safetensors index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    resolved_names = [(name, checkpoint_key(name, weight_map)) for name in names]
    missing = [name for name, resolved in resolved_names if resolved is None]
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"{len(missing)} parameters are missing from safetensors index: {preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_params: list[dict[str, Any]] = []

    current_file: str | None = None
    current_tensors: dict[str, torch.Tensor] | None = None
    offset = 0

    output_file = None if args.verify_only else args.output.open("wb")
    try:
        for index_in_order, (name, resolved_name) in enumerate(resolved_names):
            assert resolved_name is not None
            shard_file = weight_map[resolved_name]
            if current_file != shard_file:
                current_tensors = load_file(str(snapshot / shard_file), device="cpu")
                current_file = shard_file

            assert current_tensors is not None
            tensor = current_tensors[resolved_name]
            shape = list(tensor.shape)
            if expected_shapes:
                expected_shape = [int(dim) for dim in expected_shapes[index_in_order]]
                if shape != expected_shape:
                    raise SystemExit(
                        f"shape mismatch for {name}: meta={expected_shape} checkpoint={shape}"
                    )

            payload_len = numel(shape) * bytes_per_element(args.dtype)
            if output_file is not None:
                output_file.write(tensor_bytes(tensor, args.dtype))

            manifest_params.append(
                {
                    "index": index_in_order,
                    "name": name,
                    "checkpoint_name": resolved_name,
                    "shape": shape,
                    "dtype": args.dtype,
                    "offset": offset,
                    "nbytes": payload_len,
                    "source_file": shard_file,
                }
            )
            offset += payload_len
    finally:
        if output_file is not None:
            output_file.close()

    manifest = {
        "model_id": args.model_id,
        "format": f"{args.dtype}_streamed",
        "weights_path": str(args.output),
        "total_bytes": offset,
        "parameter_count": len(manifest_params),
        "parameters": manifest_params,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    mode = "Verified" if args.verify_only else "Wrote"
    print(f"{mode} {len(manifest_params)} tensors, {offset / 1024**3:.2f} GiB")
    print(f"Manifest: {manifest_path}")
    if not args.verify_only:
        print(f"Weights: {args.output}")


if __name__ == "__main__":
    main()
