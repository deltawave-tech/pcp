"""
Qwen3-8B GRPO export contract checks.

Run from the repo root:
  venv/bin/python test/qwen/test_qwen3_grpo_export_contract.py
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTER_PATH = REPO_ROOT / "tools" / "export_qwen3_training.py"
FORWARD_MLIR_PATH = REPO_ROOT / "models" / "qwen3_8b_grpo_training.mlir"
BACKWARD_MLIR_PATH = REPO_ROOT / "models" / "qwen3_8b_grpo_training.grpo_backward.mlir"

EXPECTED_PARAM_COUNT = 399
EXPECTED_DATA_INPUTS = [
    ([4, 512], "i32"),
    ([4, 512], "f32"),
    ([4], "f32"),
]


def require_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        try:
            import pytest  # type: ignore
        except ImportError:
            raise SystemExit("Missing torch; run this parity check with venv/bin/python.") from None
        pytest.skip("torch is required for loss parity checks")
    return torch


def decomposed_token_nll_for_test(torch, logits, labels):
    logits_f32 = logits.float()
    max_logits = torch.max(logits_f32, dim=-1, keepdim=True).values
    shifted_logits = logits_f32 - max_logits
    log_denom = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)) + max_logits
    log_probs = logits_f32 - log_denom

    flat_log_probs = log_probs.reshape(-1, 1)
    flat_labels = labels.reshape(-1)
    token_positions = torch.arange(flat_labels.shape[0], device=labels.device, dtype=torch.int32)
    linear_indices = token_positions * logits_f32.shape[-1] + flat_labels
    return -torch.nn.functional.embedding(linear_indices, flat_log_probs).reshape(labels.shape)


def require_file(path: Path) -> None:
    if path.exists():
        return
    try:
        import pytest  # type: ignore
    except ImportError:
        raise SystemExit(f"Missing required artifact: {path}") from None
    pytest.skip(f"Missing required artifact: {path}")


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
    match = re.search(r"func\.func @main\((.*?)\) -> (?:\((.*?)\)|([^{\n]+))", mlir_text, re.S)
    if not match:
        raise AssertionError("Could not find func.func @main signature.")

    arg_types = [parse_tensor_type(arg) for arg in split_top_level_commas(match.group(1))]
    result_text = match.group(2) if match.group(2) is not None else match.group(3)
    result_types = [parse_tensor_type(result) for result in split_top_level_commas(result_text)]
    return arg_types, result_types


def test_exporter_uses_decomposed_i32_loss() -> None:
    require_file(EXPORTER_PATH)
    source = EXPORTER_PATH.read_text(encoding="utf-8")

    assert "F.cross_entropy" not in source
    assert "TOKEN_DTYPE = torch.int32" in source
    assert "torch.gather(" not in source
    assert "torch.nn.functional.embedding" in source
    assert "def token_nll" in source
    assert "torch.log(torch.sum(torch.exp(" in source


def test_decomposed_loss_matches_pytorch_cross_entropy() -> None:
    torch = require_torch()
    generator = torch.Generator().manual_seed(1234)
    logits = torch.randn((2, 3, 17), generator=generator, dtype=torch.float32)
    labels = torch.tensor([[0, 3, 16], [4, 8, 9]], dtype=torch.int32)

    actual = decomposed_token_nll_for_test(torch, logits, labels)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.to(torch.int64).reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_forward_mlir_grpo_contract() -> None:
    require_file(FORWARD_MLIR_PATH)
    mlir_text = FORWARD_MLIR_PATH.read_text(encoding="utf-8", errors="ignore")
    lowered = mlir_text.lower()

    forbidden = [
        "cross_entropy",
        "scaled_dot_product",
        "sdpa",
        "stablehlo.custom_call",
    ]
    for marker in forbidden:
        assert marker not in lowered

    assert "tensor<4x512xi64>" not in mlir_text
    assert "tensor<4x512xi32>" in mlir_text
    assert "stablehlo.exponential" in mlir_text
    assert "stablehlo.log" in mlir_text
    assert "stablehlo.gather" in mlir_text
    assert "stablehlo.reduce" in mlir_text

    arg_types, result_types = parse_main_signature(mlir_text)
    assert len(arg_types) == EXPECTED_PARAM_COUNT + len(EXPECTED_DATA_INPUTS)
    assert arg_types[-len(EXPECTED_DATA_INPUTS) :] == EXPECTED_DATA_INPUTS
    assert result_types == [([], "f32")]


def test_backward_mlir_preserves_i32_data_contract() -> None:
    require_file(BACKWARD_MLIR_PATH)
    mlir_text = BACKWARD_MLIR_PATH.read_text(encoding="utf-8", errors="ignore")
    lowered = mlir_text.lower()

    forbidden = [
        "cross_entropy",
        "scaled_dot_product",
        "sdpa",
        "stablehlo.custom_call",
    ]
    for marker in forbidden:
        assert marker not in lowered

    assert "tensor<4x512xi64>" not in mlir_text
    assert "tensor<4x512xi32>" in mlir_text
    assert "stablehlo.gather" in mlir_text

    arg_types, result_types = parse_main_signature(mlir_text)
    assert len(arg_types) == EXPECTED_PARAM_COUNT + len(EXPECTED_DATA_INPUTS)
    assert arg_types[-len(EXPECTED_DATA_INPUTS) :] == EXPECTED_DATA_INPUTS
    assert len(result_types) == EXPECTED_PARAM_COUNT


def main() -> None:
    test_exporter_uses_decomposed_i32_loss()
    test_decomposed_loss_matches_pytorch_cross_entropy()
    test_forward_mlir_grpo_contract()
    test_backward_mlir_preserves_i32_data_contract()
    print("OK: Qwen3-8B GRPO export contract checks passed.")


if __name__ == "__main__":
    main()
