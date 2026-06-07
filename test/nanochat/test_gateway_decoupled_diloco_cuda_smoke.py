from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from pcp_integration_harness import PCP_ROOT, PCPIntegrationHarness, find_free_port


CUDA_TARGET_ARCH = os.environ.get("PCP_CUDA_TARGET_ARCH", "sm_80")
CUDA_DEVICE_ID = int(os.environ.get("PCP_CUDA_DEVICE_ID", "0"))


def cuda_smoke_enabled() -> bool:
    return os.environ.get("PCP_ENABLE_CUDA_DECOUPLED_SMOKE", "").lower() in ("1", "true", "yes")


def require_cuda() -> None:
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"nvidia-smi failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    if "NVIDIA" not in result.stdout:
        raise SystemExit(f"unexpected CUDA inventory: {result.stdout}")


def main() -> int:
    if not cuda_smoke_enabled():
        print(json.dumps({"ok": True, "skipped": "set PCP_ENABLE_CUDA_DECOUPLED_SMOKE=1"}, sort_keys=True))
        return 0
    require_cuda()

    gateway_port = find_free_port()
    training_api_port = find_free_port()
    worker_fabric_port = find_free_port()
    token = "dev-decoupled-diloco-cuda"

    with PCPIntegrationHarness(prefix="pcp-regular-decoupled-diloco-cuda-", keep_temp_on_failure=True) as harness:
        output_dir = harness.tmpdir / "decoupled_output"
        training_config_path = harness.write_json(
            "regular_decoupled_diloco_cuda_config.json",
            {
                "model_path": str((PCP_ROOT / "models" / "nanochat_small.mlir").resolve()),
                "data_path": str((PCP_ROOT / "data" / "tiny_shakespeare.txt").resolve()),
                "tokenizer": "char",
                "sampling": "random",
                "dtype": "f32",
                "learning_rate": 0.0006,
                "tau": 2,
                "outer_loop_steps": 2,
                "nesterov_momentum": 0.9,
                "max_epochs": 1,
                "aggregation_strategy": "decoupled_diloco",
                "num_fragments": 2,
                "sync_interval_h": 2,
                "overlap_tau": 1,
                "min_quorum": 1,
                "grace_window_ms": 0,
                "grace_gamma": 1.0,
                "adaptive_grace_enabled": False,
                "merge_strategy": "avg_embedding_rda_model",
                "fragment_strategy": "balanced_tensor",
                "learner_alpha": 0.0,
                "outer_learning_rate": 0.7,
                "outer_gradient_compression": "none",
                "checkpoint_dir": str(output_dir),
            },
        )
        gateway_config_path = harness.write_json(
            "gateway_config.json",
            {
                "gateway_id": "gateway-regular-decoupled-diloco-cuda",
                "lab_id": "lab-decoupled",
                "graph_backend": "memory",
                "api_token_env": "PCP_GATEWAY_API_TOKEN",
                "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
                "worker_fabric": {
                    "host": "127.0.0.1",
                    "port": worker_fabric_port,
                },
                "controllers": {
                    "training": {
                        "enabled": True,
                        "config_path": str(training_config_path),
                        "service_id": "regular-decoupled-diloco-cuda",
                        "workers": 1,
                        "worker_class": "cuda",
                        "target_arch": CUDA_TARGET_ARCH,
                        "api": {
                            "host": "127.0.0.1",
                            "port": training_api_port,
                        },
                    }
                },
            },
        )

        harness.start_gateway(
            name="gateway",
            gateway_config_path=gateway_config_path,
            gateway_port=gateway_port,
            env={
                "PCP_GATEWAY_API_TOKEN": token,
                "PCP_GATEWAY_INTERNAL_TOKEN": "dev-internal",
            },
        )
        harness.start_worker(
            name="worker-1",
            control_port=worker_fabric_port,
            backend="cuda",
            target=CUDA_TARGET_ARCH,
            device_id=CUDA_DEVICE_ID,
        )

        gateway_base = f"http://127.0.0.1:{gateway_port}"
        harness.wait_for_url("gateway healthz", f"{gateway_base}/healthz")
        harness.wait_for_json(
            "CUDA training worker discovery",
            "GET",
            f"{gateway_base}/v1/services",
            token=token,
            predicate=lambda payload: any(
                service.get("service_id") == "regular-decoupled-diloco-cuda"
                and service.get("worker_count", 0) >= 1
                for service in payload.get("services", [])
            ),
            timeout=180.0,
        )

        submitted = harness.request_json(
            "POST",
            f"{gateway_base}/v1/training/jobs",
            token=token,
            expected_status=202,
        )
        job_id = submitted["job_id"]
        job = harness.wait_for_json(
            "regular Decoupled DiLoCo CUDA completion",
            "GET",
            f"{gateway_base}/v1/jobs/{job_id}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] in ("completed", "failed", "cancelled"),
            timeout=480.0,
            interval=1.0,
        )
        if job["job"]["status"] != "completed":
            raise SystemExit(f"regular Decoupled DiLoCo CUDA training did not complete successfully: {job}")

        state_path = output_dir / "training_state.json"
        weights_path = output_dir / "online_weights.bin"
        if not state_path.exists() or not weights_path.exists():
            raise SystemExit(f"missing regular Decoupled DiLoCo CUDA outputs under {output_dir}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {
            "aggregation": "decoupled_diloco",
            "workers": 1,
            "syncer_steps": 2,
            "num_fragments": 2,
            "sync_interval_h": 2,
            "min_quorum": 1,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise SystemExit(f"unexpected {key}: {state}")
        if state.get("event_tape_entries") != 2:
            raise SystemExit(f"unexpected event tape entries: {state}")
        if state.get("quorum_participants_total", 0) < state["syncer_steps"]:
            raise SystemExit(f"unexpected participant accounting: {state}")
        if state.get("learner_to_syncer_fragment_bytes", 0) <= 0:
            raise SystemExit(f"missing learner-to-syncer byte accounting: {state}")
        if state.get("syncer_to_learner_fragment_bytes", 0) <= 0:
            raise SystemExit(f"missing syncer-to-learner byte accounting: {state}")
        if weights_path.stat().st_size <= 0:
            raise SystemExit(f"empty regular Decoupled DiLoCo CUDA weights: {weights_path}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "aggregation": state["aggregation"],
                    "workers": state["workers"],
                    "syncer_steps": state["syncer_steps"],
                    "target_arch": CUDA_TARGET_ARCH,
                },
                sort_keys=True,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
