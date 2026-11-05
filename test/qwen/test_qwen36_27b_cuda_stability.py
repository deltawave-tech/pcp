"""
Opt-in Qwen3.6-27B CUDA repeat-run stability gate.

Run from the repo root on an A100 VM:
  PCP_ENABLE_QWEN36_27B_CUDA_STABILITY=1 .venv/bin/python -m pytest test/qwen/test_qwen36_27b_cuda_stability.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qwen36_27b_cuda_runtime import (
    assert_no_runner_processes,
    env_enabled,
    nvidia_memory_snapshot,
    require_cuda_visible,
    require_runtime_artifacts,
    run_one_token_cuda,
)


ENABLE_ENV = "PCP_ENABLE_QWEN36_27B_CUDA_STABILITY"
MAX_RESIDUAL_GPU_MEMORY_MIB = 1024


def repeat_count() -> int:
    return int(os.environ.get("PCP_QWEN36_27B_REPEAT_COUNT", "3"))


def test_qwen36_27b_cuda_repeat_run_stability(tmp_path: Path) -> None:
    if not env_enabled(ENABLE_ENV):
        pytest.skip(f"set {ENABLE_ENV}=1 to run the 27B CUDA repeat-run stability gate")

    count = repeat_count()
    assert count >= 2, "repeat-run stability requires at least two executions"

    require_cuda_visible(ENABLE_ENV)
    require_runtime_artifacts()

    results = []
    for index in range(count):
        run_dir = tmp_path / f"run_{index + 1}"
        run_dir.mkdir()
        result = run_one_token_cuda(run_dir / "qwen36_27b_logits.bin")
        results.append(result)

        assert result.logits_count == results[0].logits_count
        assert result.top_index == results[0].top_index
        assert abs(result.top_value - results[0].top_value) <= 1e-6
        assert result.memory_after_mib is not None, result.memory_after
        assert result.memory_after_mib <= MAX_RESIDUAL_GPU_MEMORY_MIB, result.memory_after
        assert_no_runner_processes()

        print(
            "run={run} elapsed_seconds={elapsed:.3f} logits={logits} "
            "top_index={top_index} top_value={top_value:.6f} gpu_memory_after={memory}".format(
                run=index + 1,
                elapsed=result.elapsed_seconds,
                logits=result.logits_count,
                top_index=result.top_index,
                top_value=result.top_value,
                memory=result.memory_after,
            )
        )

    final_memory = nvidia_memory_snapshot()
    print(f"final_gpu_memory={final_memory}")
