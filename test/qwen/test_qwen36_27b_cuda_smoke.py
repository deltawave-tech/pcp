"""
Opt-in Qwen3.6-27B CUDA runtime validation.

Run from the repo root on an A100 VM:
  PCP_ENABLE_QWEN36_27B_CUDA_SMOKE=1 .venv/bin/python -m pytest test/qwen/test_qwen36_27b_cuda_smoke.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen36_27b_cuda_runtime import (
    env_enabled,
    require_cuda_visible,
    require_runtime_artifacts,
    run_one_token_cuda,
)


ENABLE_ENV = "PCP_ENABLE_QWEN36_27B_CUDA_SMOKE"


def test_qwen36_27b_one_token_cuda_runtime_validation(tmp_path: Path) -> None:
    if not env_enabled(ENABLE_ENV):
        pytest.skip(f"set {ENABLE_ENV}=1 to run the 27B CUDA runtime validation")

    require_cuda_visible(ENABLE_ENV)
    require_runtime_artifacts()

    result = run_one_token_cuda(tmp_path / "qwen36_27b_logits.bin")

    print(f"gpu_memory_before={result.memory_before}")
    print(f"gpu_memory_after={result.memory_after}")
    print(f"elapsed_seconds={result.elapsed_seconds:.3f}")
    print(f"logits={result.logits_count} top_index={result.top_index} top_value={result.top_value:.6f}")
    for line in result.output_lines:
        print(line)
