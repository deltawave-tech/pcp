"""
Qwen3-8B inference contract checks.

Run from the repo root:
  python3 test/qwen/test_qwen3_inference_contract.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MLIR_PATH = REPO_ROOT / "models" / "qwen3_8b_rl_generation.mlir"
META_PATH = REPO_ROOT / "models" / "qwen3_8b_rl_generation.mlir.meta.json"
CONFIG_PATH = REPO_ROOT / "experiments" / "inference_qwen3_8b.json"

EXPECTED_LAYERS = 36
EXPECTED_PARAM_COUNT = 399
EXPECTED_DATA_INPUT_COUNT = 74
EXPECTED_CACHE_INPUT_COUNT = 72
EXPECTED_CACHE_SHAPE = [1, 8, 1024, 128]
EXPECTED_CACHE_UPDATE_SHAPE = [1, 8, 1, 128]
EXPECTED_LOGITS_SHAPE = [1, 1, 151936]


def require_file(path: Path) -> None:
    if path.exists():
        return
    try:
        import pytest  # type: ignore
    except ImportError:
        raise SystemExit(f"Missing required artifact: {path}") from None
    pytest.skip(f"Missing required artifact: {path}")


def load_json(path: Path) -> dict:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def expected_parameter_shapes() -> list[list[int]]:
    layer_shapes = [
        [4096, 4096],
        [1024, 4096],
        [1024, 4096],
        [4096, 4096],
        [128],
        [128],
        [12288, 4096],
        [12288, 4096],
        [4096, 12288],
        [4096],
        [4096],
    ]
    shapes: list[list[int]] = [[151936, 4096]]
    for _ in range(EXPECTED_LAYERS):
        shapes.extend(layer_shapes)
    shapes.extend([[4096], [151936, 4096]])
    return shapes


def split_top_level_commas(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            part = value[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_tensor_type(type_text: str) -> tuple[list[int], str]:
    type_text = type_text.strip()
    if ":" in type_text:
        type_text = type_text.split(":", 1)[1].strip()
    if not type_text.startswith("tensor<") or not type_text.endswith(">"):
        raise AssertionError(f"Unsupported tensor type: {type_text}")
    inner = type_text[len("tensor<") : -1]
    pieces = inner.split("x")
    dtype = pieces[-1]
    dims = [int(piece) for piece in pieces[:-1]]
    return dims, dtype


def parse_main_signature(mlir_text: str) -> tuple[list[tuple[list[int], str]], list[tuple[list[int], str]]]:
    match = re.search(r"func\.func @main\((.*?)\) -> \((.*?)\)", mlir_text, re.S)
    if not match:
        raise AssertionError("Could not find func.func @main signature.")

    arg_types = [parse_tensor_type(arg) for arg in split_top_level_commas(match.group(1))]
    result_types = [parse_tensor_type(result) for result in split_top_level_commas(match.group(2))]
    return arg_types, result_types


def test_metadata_contract() -> None:
    meta = load_json(META_PATH)

    parameter_shapes = meta["parameter_shapes"]
    data_input_shapes = meta["data_input_shapes"]
    data_input_dtypes = meta["data_input_dtypes"]

    assert len(parameter_shapes) == EXPECTED_PARAM_COUNT
    assert parameter_shapes == expected_parameter_shapes()

    assert len(data_input_shapes) == EXPECTED_DATA_INPUT_COUNT
    assert len(data_input_dtypes) == EXPECTED_DATA_INPUT_COUNT
    assert data_input_shapes[:2] == [[1, 1], [1, 1]]
    assert data_input_dtypes[:2] == ["i64", "i64"]
    assert data_input_shapes[2:] == [EXPECTED_CACHE_SHAPE] * EXPECTED_CACHE_INPUT_COUNT
    assert data_input_dtypes[2:] == ["f32"] * EXPECTED_CACHE_INPUT_COUNT


def test_mlir_contract() -> None:
    require_file(MLIR_PATH)
    mlir_text = MLIR_PATH.read_text(encoding="utf-8", errors="ignore")
    lowered = mlir_text.lower()

    forbidden = [
        "scaled_dot_product",
        "sdpa",
        "stablehlo.custom_call",
    ]
    for marker in forbidden:
        assert marker not in lowered

    arg_types, result_types = parse_main_signature(mlir_text)
    assert len(arg_types) == EXPECTED_PARAM_COUNT + EXPECTED_DATA_INPUT_COUNT

    data_arg_types = arg_types[-EXPECTED_DATA_INPUT_COUNT:]
    assert data_arg_types[0] == ([1, 1], "i64")
    assert data_arg_types[1] == ([1, 1], "i64")
    assert data_arg_types[2:] == [(EXPECTED_CACHE_SHAPE, "f32")] * EXPECTED_CACHE_INPUT_COUNT

    assert len(result_types) == 1 + EXPECTED_CACHE_INPUT_COUNT
    assert result_types[0] == (EXPECTED_LOGITS_SHAPE, "f32")
    assert result_types[1:] == [(EXPECTED_CACHE_UPDATE_SHAPE, "f32")] * EXPECTED_CACHE_INPUT_COUNT


def test_inference_config_contract() -> None:
    config = load_json(CONFIG_PATH)

    assert config["model_id"] == "qwen3-8b"
    assert config["tokenizer_source"] == "qwen"
    assert config["tokenizer_path"] == "tokenizers/qwen3-8b"
    assert config["weights_path"] == "checkpoints/initial_weights/qwen3_8b_flat.bin"
    assert config["generation_mlir_path"] == "models/qwen3_8b_rl_generation.mlir"
    assert config["generation_vmfb_path"] == "models/qwen3_8b_rl_generation.vmfb"
    assert config["num_gen_data_inputs"] == EXPECTED_DATA_INPUT_COUNT
    assert config["max_context_tokens"] == 1024
    assert "eos_token" not in config or config["eos_token"] is None


def main() -> None:
    test_metadata_contract()
    test_mlir_contract()
    test_inference_config_contract()
    print("OK: Qwen3-8B inference contract checks passed.")


if __name__ == "__main__":
    main()
