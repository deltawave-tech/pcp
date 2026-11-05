from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NANOCHAT_TEST_DIR = REPO_ROOT / "test" / "nanochat"
if str(NANOCHAT_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_TEST_DIR))

from pcp_integration_harness import HarnessError, PCPIntegrationHarness, find_free_port  # noqa: E402
from qwen36_27b_cuda_runtime import (  # noqa: E402
    env_enabled,
    memory_used_mib,
    nvidia_memory_snapshot,
    pcp_binary,
    python_bin_dir,
    require_cuda_visible,
    require_runtime_artifacts,
    runner_env,
    timeout_seconds,
)


ENABLE_ENV = "PCP_ENABLE_QWEN36_27B_GATEWAY_VALIDATION"
CONFIG_PATH = REPO_ROOT / "experiments" / "inference_qwen36_27b.json"
HOST = "127.0.0.1"


def _load_gateway_config(config_path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tokenizer_path"] = os.environ.get("PCP_QWEN36_27B_TOKENIZER_PATH", "Qwen/Qwen3.6-27B")
    config["default_max_output_tokens"] = 1
    config["request_timeout_seconds"] = int(
        os.environ.get("PCP_QWEN36_27B_GATEWAY_REQUEST_TIMEOUT_SECONDS", str(timeout_seconds()))
    )
    if overrides:
        config.update(overrides)
    return config


def _require_tokenizer_available(tokenizer_path: str) -> None:
    bin_dir = python_bin_dir()
    python = str(Path(bin_dir) / "python") if bin_dir is not None else sys.executable
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        [
            python,
            "-c",
            (
                "from transformers import AutoTokenizer; "
                "tok = AutoTokenizer.from_pretrained("
                "r'''%s''', use_fast=True, trust_remote_code=True); "
                "assert tok.eos_token_id is not None"
            )
            % tokenizer_path,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Tokenizer path is unavailable for Qwen3.6-27B gateway validation: {tokenizer_path}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _gateway_process_env(extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    bin_dir = python_bin_dir()
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env.update(extra)
    return env


def _service_named(payload: Any, service_id: str) -> dict[str, Any] | None:
    services = payload.get("services", []) if isinstance(payload, dict) else payload
    for service in services:
        if service.get("service_id") == service_id:
            return service
    return None


def _metric(payload: dict[str, Any], key: str) -> int:
    return int(payload.get("inference", {}).get(key, 0))


def _process_rss_kib(processes: list[Any]) -> int:
    total = 0
    for managed in processes:
        pid = managed.process.pid
        try:
            stat = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        for line in stat.splitlines():
            if line.startswith("VmRSS:"):
                total += int(line.split()[1])
                break
    return total


def _wait_for_gpu_memory_below(limit_mib: int, timeout: float = 120.0) -> str:
    deadline = time.monotonic() + timeout
    last_snapshot = nvidia_memory_snapshot()
    while time.monotonic() < deadline:
        last_snapshot = nvidia_memory_snapshot()
        used_mib = memory_used_mib(last_snapshot)
        if used_mib is not None and used_mib <= limit_mib:
            return last_snapshot
        time.sleep(2.0)
    pytest.fail(f"GPU memory did not return below {limit_mib} MiB after gateway validation: {last_snapshot}")


def run_qwen36_27b_gateway_validation(
    *,
    enable_env: str = ENABLE_ENV,
    config_overrides: dict[str, Any] | None = None,
    temp_prefix: str = "pcp-qwen36-27b-gateway-",
) -> None:
    if not env_enabled(enable_env):
        pytest.skip(f"set {enable_env}=1 to run the real Qwen3.6-27B gateway validation")

    require_cuda_visible(enable_env)
    require_runtime_artifacts()
    binary = pcp_binary()
    inference_config = _load_gateway_config(CONFIG_PATH, config_overrides)
    _require_tokenizer_available(inference_config["tokenizer_path"])

    token = "dev-qwen36-27b-gateway"
    internal_token = "dev-qwen36-27b-gateway-internal"
    gateway_port = find_free_port()
    fabric_port = find_free_port()
    inference_api_port = find_free_port()
    request_timeout = float(os.environ.get("PCP_QWEN36_27B_GATEWAY_REQUEST_TIMEOUT_SECONDS", str(timeout_seconds())))
    ready_timeout = float(os.environ.get("PCP_QWEN36_27B_GATEWAY_READY_TIMEOUT_SECONDS", "900"))

    memory_before = nvidia_memory_snapshot()
    diagnostics = ""
    memory_active = "not reached"
    memory_after = "not reached"

    try:
        with PCPIntegrationHarness(prefix=temp_prefix, keep_temp_on_failure=True) as harness:
            harness.pcp_bin = str(binary)
            harness.base_env = _gateway_process_env(
                {
                    "PCP_GATEWAY_API_TOKEN": token,
                    "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
                },
            )

            inference_config_path = harness.write_json("inference_qwen36_27b_runtime.json", inference_config)
            gateway_config = {
                "gateway_id": "qwen36-27b-gateway-validation",
                "lab_id": "qwen36-27b-validation",
                "graph_backend": "memory",
                "api_token_env": "PCP_GATEWAY_API_TOKEN",
                "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
                "worker_fabric": {
                    "host": HOST,
                    "port": fabric_port,
                },
                "controllers": {
                    "inference": {
                        "enabled": True,
                        "config_path": str(inference_config_path),
                        "service_id": "inference-main",
                        "worker_class": "cuda",
                        "target_arch": "sm_80",
                        "api": {
                            "host": HOST,
                            "port": inference_api_port,
                        },
                    },
                },
            }
            gateway_config_path = harness.write_json("gateway_qwen36_27b.json", gateway_config)

            startup_started = time.monotonic()
            harness.start_gateway("gateway", gateway_config_path, gateway_port, env={})
            harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz", timeout=60.0)
            harness.wait_for_url("inference controller health", f"http://{HOST}:{inference_api_port}/healthz", timeout=60.0)

            harness.start_worker("cuda_worker", fabric_port, backend="cuda", target="sm_80", env=runner_env(binary))
            ready = harness.wait_for_json(
                "Qwen3.6-27B inference controller readiness",
                "GET",
                f"http://{HOST}:{inference_api_port}/readyz",
                token=token,
                predicate=lambda payload: (
                    payload.get("ready") is True
                    and payload.get("ready_workers") == 1
                    and payload.get("available_workers") == 1
                    and payload.get("worker_class") == "cuda"
                    and payload.get("target_arch") == "sm_80"
                ),
                timeout=ready_timeout,
                interval=2.0,
            )
            ready_elapsed_seconds = time.monotonic() - startup_started
            peak_host_rss_kib = _process_rss_kib(harness.processes)
            services = harness.wait_for_json(
                "Qwen3.6-27B service registration",
                "GET",
                f"http://{HOST}:{gateway_port}/v1/services",
                token=token,
                predicate=lambda payload: (
                    _service_named(payload, "inference-main") is not None
                    and _service_named(payload, "inference-main").get("health_status") == "ok"
                    and _service_named(payload, "inference-main").get("ready_worker_count") == 1
                ),
                timeout=120.0,
                interval=1.0,
            )
            metrics_before = harness.request_json(
                "GET",
                f"http://{HOST}:{inference_api_port}/v1/metrics",
                token=token,
            )

            requested_max_tokens = int(os.environ.get("PCP_QWEN36_27B_GATEWAY_MAX_TOKENS", "2"))
            request_started = time.monotonic()
            completion = harness.request_json(
                "POST",
                f"http://{HOST}:{gateway_port}/v1/inference/chat/completions",
                token=token,
                payload={
                    "model": "qwen36-27b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": requested_max_tokens,
                    "temperature": 0.0,
                },
                timeout=request_timeout,
            )
            request_elapsed_ms = int((time.monotonic() - request_started) * 1000)
            peak_host_rss_kib = max(peak_host_rss_kib, _process_rss_kib(harness.processes))

            usage = completion.get("usage", {})
            choices = completion.get("choices", [])
            assert completion.get("model") in ("qwen36-27b", "pcp/qwen36-27b"), completion
            assert isinstance(choices, list) and len(choices) == 1, completion
            assert choices[0].get("message", {}).get("role") == "assistant", completion
            assert isinstance(choices[0].get("message", {}).get("content"), str), completion
            assert int(usage.get("prompt_tokens", 0)) > 0, completion
            assert 1 <= int(usage.get("completion_tokens", 0)) <= requested_max_tokens, completion
            assert int(usage.get("total_tokens", 0)) == int(usage["prompt_tokens"]) + int(usage["completion_tokens"]), completion

            metrics_after = harness.wait_for_json(
                "Qwen3.6-27B inference metrics",
                "GET",
                f"http://{HOST}:{inference_api_port}/v1/metrics",
                token=token,
                predicate=lambda payload: (
                    _metric(payload, "total_requests") >= _metric(metrics_before, "total_requests") + 1
                    and _metric(payload, "completed_requests") >= _metric(metrics_before, "completed_requests") + 1
                    and _metric(payload, "tokens_generated") >= _metric(metrics_before, "tokens_generated") + 1
                    and _metric(payload, "prompt_tokens") >= _metric(metrics_before, "prompt_tokens") + int(usage["prompt_tokens"])
                ),
                timeout=30.0,
                interval=1.0,
            )
            memory_active = nvidia_memory_snapshot()
            ttft_count_delta = _metric(metrics_after, "ttft_count") - _metric(metrics_before, "ttft_count")
            ttft_total_delta = _metric(metrics_after, "ttft_total_ms") - _metric(metrics_before, "ttft_total_ms")
            first_token_latency_ms = int(ttft_total_delta / ttft_count_delta) if ttft_count_delta > 0 else None
            completion_tokens = int(usage.get("completion_tokens", 0))
            steady_decode_latency_ms = (
                int(max(request_elapsed_ms - int(first_token_latency_ms or 0), 0) / (completion_tokens - 1))
                if completion_tokens > 1
                else None
            )
            diagnostics = harness.diagnostics(lines=160)

            print(
                json.dumps(
                    {
                        "ready": ready,
                        "service": _service_named(services, "inference-main"),
                        "usage": usage,
                        "metrics_before": metrics_before.get("inference", {}),
                        "metrics_after": metrics_after.get("inference", {}),
                        "telemetry": {
                            "compile_or_load_seconds": ready_elapsed_seconds,
                            "first_token_latency_ms": first_token_latency_ms,
                            "steady_decode_latency_ms": steady_decode_latency_ms,
                            "request_elapsed_ms": request_elapsed_ms,
                            "peak_host_rss_kib": peak_host_rss_kib,
                            "peak_gpu_memory": memory_active,
                        },
                        "gpu_memory_before": memory_before,
                        "gpu_memory_active": memory_active,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    except HarnessError as exc:
        pytest.fail(f"Qwen3.6-27B gateway validation failed:\n{exc}")
    finally:
        memory_after = _wait_for_gpu_memory_below(1024)
        print(
            json.dumps(
                {
                    "gpu_memory_before": memory_before,
                    "gpu_memory_active": memory_active,
                    "gpu_memory_after": memory_after,
                    "diagnostics": diagnostics,
                },
                indent=2,
                sort_keys=True,
            )
        )


def test_qwen36_27b_gateway_inference_validation() -> None:
    run_qwen36_27b_gateway_validation()
