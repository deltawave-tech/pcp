from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen36_27b_cuda_runtime import META_PATH, WEIGHTS_PATH, env_enabled
from test_qwen36_27b_gateway_inference_validation import run_qwen36_27b_gateway_validation


ENABLE_ENV = "PCP_ENABLE_QWEN36_27B_GATEWAY_MANIFEST_VALIDATION"


def _dtype_list(meta: dict) -> list[str]:
    dtypes = meta.get("parameter_dtypes")
    if isinstance(dtypes, list) and all(isinstance(value, str) for value in dtypes):
        return dtypes
    return ["bf16"] * len(meta["parameter_shapes"])


def _write_manifest(manifest_path: Path) -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    shapes = meta["parameter_shapes"]
    dtypes = _dtype_list(meta)
    if len(shapes) != len(dtypes):
        pytest.fail(f"metadata shape/dtype count mismatch: shapes={len(shapes)} dtypes={len(dtypes)}")

    offset = 0
    parameters = []
    for index, (shape, dtype) in enumerate(zip(shapes, dtypes)):
        byte_count = 2
        for dim in shape:
            byte_count *= int(dim)
        parameters.append(
            {
                "index": index,
                "source_file": str(WEIGHTS_PATH),
                "offset": offset,
                "nbytes": byte_count,
                "shape": shape,
                "dtype": dtype,
            }
        )
        offset += byte_count

    actual_size = WEIGHTS_PATH.stat().st_size
    if offset != actual_size:
        pytest.fail(f"manifest byte count mismatch: manifest={offset} file={actual_size}")

    manifest = {
        "format": "bf16_manifest_streamed",
        "source": "pcp_qwen36_27b_single_file_manifest",
        "total_bytes": offset,
        "parameter_count": len(parameters),
        "parameters": parameters,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_qwen36_27b_gateway_manifest_validation(tmp_path: Path) -> None:
    if not env_enabled(ENABLE_ENV):
        pytest.skip(f"set {ENABLE_ENV}=1 to run the real Qwen3.6-27B gateway manifest validation")

    manifest_path = tmp_path / "qwen36_27b_bf16_manifest_streamed.json"
    _write_manifest(manifest_path)
    run_qwen36_27b_gateway_validation(
        enable_env=ENABLE_ENV,
        config_overrides={
            "weights_path": str(manifest_path),
            "weights_format": "bf16_manifest_streamed",
        },
        temp_prefix="pcp-qwen36-27b-gateway-manifest-",
    )
