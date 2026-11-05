from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

from qwen36_27b_cuda_runtime import (
    REPO_ROOT,
    WEIGHTS_PATH,
    env_enabled,
    memory_used_mib,
    nvidia_memory_snapshot,
    pcp_binary,
    require_artifact,
    require_cuda_visible,
    require_runtime_artifacts,
    runner_env,
    timeout_seconds,
)


NANOCHAT_TEST_DIR = REPO_ROOT / "test" / "nanochat"
if str(NANOCHAT_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_TEST_DIR))

from pcp_integration_harness import HarnessError, PCPIntegrationHarness, find_free_port  # noqa: E402


ENABLE_ENV = "PCP_ENABLE_QWEN36_27B_GRPO_CUDA_VALIDATION"
RESTART_ENABLE_ENV = "PCP_ENABLE_QWEN36_27B_GRPO_RESTART_VALIDATION"
CONFIG_PATH = REPO_ROOT / "experiments" / "qwen36_27b_grpo_adapter_smoke.json"
TRAINING_MLIR_PATH = REPO_ROOT / "models" / "qwen36_27b_adapter_grpo_training.mlir"
TRAINING_META_PATH = REPO_ROOT / "models" / "qwen36_27b_adapter_grpo_training.mlir.meta.json"
ADAPTER_INITIAL_PATH = REPO_ROOT / "checkpoints" / "initial_weights" / "qwen36_27b_adapter_f32.bin"
PROMPT_PATH = REPO_ROOT / "data" / "rl_prompts.bin"
ADAPTER_STATE_PATH = Path("/tmp/pcp_qwen36_27b_grpo_adapter_state.bin")
ADAPTER_BYTES = 5120 * 8 * 4
OPTIMIZER_STATE_BYTES = ADAPTER_BYTES
HOST = "127.0.0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_nonzero_adapter(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for index in range(5120 * 8):
            value = ((index % 97) - 48) * 1.0e-5
            handle.write(struct.pack("<f", value))


def _require_grpo_artifacts() -> None:
    require_runtime_artifacts()
    require_artifact(CONFIG_PATH)
    require_artifact(TRAINING_MLIR_PATH)
    require_artifact(TRAINING_META_PATH)
    require_artifact(ADAPTER_INITIAL_PATH, ADAPTER_BYTES)
    require_artifact(PROMPT_PATH)


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    grpo = config["grpo_config"]
    assert grpo["grpo_weight_mode"] == "adapter_only"
    assert grpo["group_size"] == 4
    assert grpo["num_prompts"] == 1
    assert grpo["num_iterations"] == 1
    assert grpo["rollout_max_tokens"] == 1
    assert grpo["generation_weights_path"] == str(WEIGHTS_PATH.relative_to(REPO_ROOT))
    assert grpo["generation_weights_format"] == "bf16_streamed"
    assert grpo["adapter_state_path"] == str(ADAPTER_STATE_PATH)
    return config


def _finite_log_values(pattern: str, text: str) -> list[float]:
    values = [float(match) for match in re.findall(pattern, text)]
    assert values, f"missing log values for pattern: {pattern}"
    assert all(math.isfinite(value) for value in values), values
    return values


def _wait_for_gpu_memory_below(limit_mib: int, timeout: float = 120.0) -> str:
    import time

    deadline = time.monotonic() + timeout
    last_snapshot = nvidia_memory_snapshot()
    while time.monotonic() < deadline:
        last_snapshot = nvidia_memory_snapshot()
        used_mib = memory_used_mib(last_snapshot)
        if used_mib is not None and used_mib <= limit_mib:
            return last_snapshot
        time.sleep(2.0)
    pytest.fail(f"GPU memory did not return below {limit_mib} MiB after GRPO validation: {last_snapshot}")


def _optimizer_state_path(adapter_state_path: Path) -> Path:
    return Path(f"{adapter_state_path}.optimizer_state.bin")


def _run_grpo_validation_once(
    *,
    binary: Path,
    config: dict[str, Any],
    adapter_initial_path: Path,
    adapter_state_path: Path,
    token_suffix: str,
    expect_resume: bool = False,
) -> tuple[str, str, str]:
    token = f"dev-qwen36-27b-grpo-{token_suffix}"
    internal_token = f"dev-qwen36-27b-grpo-{token_suffix}-internal"
    gateway_port = find_free_port()
    fabric_port = find_free_port()
    rl_api_port = find_free_port()
    timeout = float(timeout_seconds())

    with PCPIntegrationHarness(prefix=f"pcp-qwen36-27b-grpo-{token_suffix}-", keep_temp_on_failure=True) as harness:
        harness.pcp_bin = str(binary)
        harness.base_env = {
            **harness.base_env,
            "PCP_GATEWAY_API_TOKEN": token,
            "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
        }
        run_config = json.loads(json.dumps(config))
        run_config["grpo_config"]["weights_path"] = str(adapter_initial_path)
        run_config["grpo_config"]["training_weights_path"] = str(adapter_initial_path)
        run_config["grpo_config"]["adapter_state_path"] = str(adapter_state_path)
        grpo_config_path = harness.write_json("qwen36_27b_grpo_adapter_runtime.json", run_config)
        gateway_config_path = harness.write_json(
            "gateway_qwen36_27b_grpo.json",
            {
                "gateway_id": f"qwen36-27b-grpo-{token_suffix}",
                "lab_id": "qwen36-27b-validation",
                "graph_backend": "memory",
                "api_token_env": "PCP_GATEWAY_API_TOKEN",
                "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
                "worker_fabric": {
                    "host": HOST,
                    "port": fabric_port,
                },
                "controllers": {
                    "rl": {
                        "enabled": True,
                        "config_path": str(grpo_config_path),
                        "service_id": "qwen36-27b-grpo",
                        "workers": 1,
                        "backend": "cuda",
                        "worker_class": "cuda",
                        "target_arch": "sm_80",
                        "api": {
                            "host": HOST,
                            "port": rl_api_port,
                        },
                    },
                },
            },
        )

        harness.start_gateway("gateway", gateway_config_path, gateway_port, env={})
        harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz", timeout=60.0)
        harness.wait_for_url("RL controller health", f"http://{HOST}:{rl_api_port}/healthz", timeout=60.0)
        harness.start_worker("cuda_worker", fabric_port, backend="cuda", target="sm_80", env=runner_env(binary))
        harness.wait_for_json(
            "Qwen3.6-27B GRPO service registration",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: any(
                service.get("service_id") == "qwen36-27b-grpo"
                and service.get("service_type") == "rl"
                and service.get("worker_count", 0) >= 1
                for service in payload.get("services", [])
            ),
            timeout=180.0,
            interval=1.0,
        )
        submitted = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/rl/jobs",
            token=token,
            expected_status=202,
        )
        job_id = submitted["job_id"]
        job = harness.wait_for_json(
            "Qwen3.6-27B GRPO completion",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] in ("completed", "failed", "cancelled"),
            timeout=timeout,
            interval=2.0,
        )
        diagnostics = harness.diagnostics(lines=260)
        if job["job"]["status"] != "completed":
            pytest.fail(f"Qwen3.6-27B GRPO validation did not complete: {job}\n{diagnostics}")

        optimizer_state_path = _optimizer_state_path(adapter_state_path)
        assert adapter_state_path.exists(), f"missing adapter state: {adapter_state_path}"
        assert adapter_state_path.stat().st_size == ADAPTER_BYTES
        assert optimizer_state_path.exists(), f"missing optimizer state: {optimizer_state_path}"
        assert optimizer_state_path.stat().st_size == OPTIMIZER_STATE_BYTES
        final_checksum = _sha256(adapter_state_path)
        assert max(_finite_log_values(r"GRPO adapter objective loss: ([0-9eE+\-.]+)", diagnostics)) > 0.0
        assert max(_finite_log_values(r"L2 Norm: ([0-9eE+\-.]+)", diagnostics)) > 0.0
        assert max(_finite_log_values(r"Max \|grad\|: ([0-9eE+\-.]+)", diagnostics)) > 0.0
        assert "frozen generation weights were not broadcast" in diagnostics
        assert "Adapter-only GRPO optimizer state written" in diagnostics
        assert "Broadcasting updated weights to workers" not in diagnostics
        assert "Weight broadcast complete" not in diagnostics
        if expect_resume:
            assert "Resuming adapter-only GRPO from adapter state" in diagnostics
            assert "Resumed adapter-only GRPO optimizer state" in diagnostics
        return final_checksum, diagnostics, nvidia_memory_snapshot()


def test_qwen36_27b_grpo_cuda_validation(tmp_path: Path) -> None:
    if not env_enabled(ENABLE_ENV):
        pytest.skip(f"set {ENABLE_ENV}=1 to run the real Qwen3.6-27B GRPO CUDA validation")

    require_cuda_visible(ENABLE_ENV)
    _require_grpo_artifacts()
    config = _load_config()
    binary = pcp_binary()
    if ADAPTER_STATE_PATH.exists():
        ADAPTER_STATE_PATH.unlink()
    optimizer_state_path = _optimizer_state_path(ADAPTER_STATE_PATH)
    if optimizer_state_path.exists():
        optimizer_state_path.unlink()

    memory_before = nvidia_memory_snapshot()
    memory_after = "not reached"
    adapter_initial_path = tmp_path / "qwen36_27b_adapter_f32_nonzero.bin"
    _write_nonzero_adapter(adapter_initial_path)
    initial_checksum = _sha256(adapter_initial_path)

    try:
        final_checksum, _diagnostics, memory_active = _run_grpo_validation_once(
            binary=binary,
            config=config,
            adapter_initial_path=adapter_initial_path,
            adapter_state_path=ADAPTER_STATE_PATH,
            token_suffix="validation",
        )
        assert final_checksum != initial_checksum
        print(
            json.dumps(
                {
                    "adapter_initial_sha256": initial_checksum,
                    "adapter_final_sha256": final_checksum,
                    "adapter_state_path": str(ADAPTER_STATE_PATH),
                    "optimizer_state_path": str(optimizer_state_path),
                    "gpu_memory_before": memory_before,
                    "gpu_memory_active": memory_active,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except HarnessError as exc:
        pytest.fail(f"Qwen3.6-27B GRPO validation failed:\n{exc}")
    finally:
        memory_after = _wait_for_gpu_memory_below(1024)
        print(
            json.dumps(
                {
                    "gpu_memory_before": memory_before,
                    "gpu_memory_after": memory_after,
                    "adapter_state_path": str(ADAPTER_STATE_PATH),
                },
                indent=2,
                sort_keys=True,
            )
        )


def test_qwen36_27b_grpo_restart_validation(tmp_path: Path) -> None:
    if not env_enabled(RESTART_ENABLE_ENV):
        pytest.skip(f"set {RESTART_ENABLE_ENV}=1 to run the real Qwen3.6-27B GRPO restart validation")

    require_cuda_visible(RESTART_ENABLE_ENV)
    _require_grpo_artifacts()
    config = _load_config()
    binary = pcp_binary()
    adapter_state_path = tmp_path / "qwen36_27b_grpo_adapter_state.bin"
    optimizer_state_path = _optimizer_state_path(adapter_state_path)
    adapter_initial_path = tmp_path / "qwen36_27b_adapter_f32_nonzero.bin"
    _write_nonzero_adapter(adapter_initial_path)
    initial_checksum = _sha256(adapter_initial_path)
    memory_before = nvidia_memory_snapshot()

    try:
        first_checksum, _first_diagnostics, first_memory_active = _run_grpo_validation_once(
            binary=binary,
            config=config,
            adapter_initial_path=adapter_initial_path,
            adapter_state_path=adapter_state_path,
            token_suffix="restart-first",
        )
        assert first_checksum != initial_checksum

        second_checksum, second_diagnostics, second_memory_active = _run_grpo_validation_once(
            binary=binary,
            config=config,
            adapter_initial_path=adapter_initial_path,
            adapter_state_path=adapter_state_path,
            token_suffix="restart-second",
            expect_resume=True,
        )
        assert second_checksum != first_checksum
        assert second_checksum != initial_checksum
        assert "Optimizer initialized for 1 trainable parameters" in second_diagnostics
        assert "Optimizer initialized for 2" not in second_diagnostics
        assert optimizer_state_path.exists()
        print(
            json.dumps(
                {
                    "adapter_initial_sha256": initial_checksum,
                    "adapter_first_sha256": first_checksum,
                    "adapter_second_sha256": second_checksum,
                    "adapter_state_path": str(adapter_state_path),
                    "optimizer_state_path": str(optimizer_state_path),
                    "gpu_memory_before": memory_before,
                    "gpu_memory_first_active": first_memory_active,
                    "gpu_memory_second_active": second_memory_active,
                    "gpu_memory_after": _wait_for_gpu_memory_below(1024),
                },
                indent=2,
                sort_keys=True,
            )
        )
    except HarnessError as exc:
        pytest.fail(f"Qwen3.6-27B GRPO restart validation failed:\n{exc}")
