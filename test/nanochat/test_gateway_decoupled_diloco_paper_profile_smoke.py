from __future__ import annotations

import json
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from pcp_integration_harness import PCP_ROOT, PCPIntegrationHarness, find_free_port


def set_regular_config_path(config: dict[str, object], block_name: str, key: str, value: str) -> None:
    block = config.get(block_name)
    if isinstance(block, dict):
        block[key] = value
        return
    config[key] = value


def main() -> int:
    gateway_port = find_free_port()
    training_api_port = find_free_port()
    worker_fabric_port = find_free_port()
    token = "dev-decoupled-diloco-paper-profile"

    example_config_path = PCP_ROOT / "experiments" / "nanochat_decoupled_diloco_paper.json"
    example_config = json.loads(example_config_path.read_text(encoding="utf-8"))

    with PCPIntegrationHarness(prefix="pcp-regular-decoupled-diloco-paper-", keep_temp_on_failure=True) as harness:
        output_dir = harness.tmpdir / "decoupled_paper_output"
        training_config = dict(example_config)
        artifacts = training_config.get("artifacts", {})
        dataset = training_config.get("dataset", {})
        model_path = artifacts.get("model_path") if isinstance(artifacts, dict) else training_config["model_path"]
        data_path = dataset.get("path") if isinstance(dataset, dict) else training_config["data_path"]
        set_regular_config_path(training_config, "artifacts", "model_path", str((PCP_ROOT / model_path).resolve()))
        set_regular_config_path(training_config, "dataset", "path", str((PCP_ROOT / data_path).resolve()))
        set_regular_config_path(training_config, "outputs", "checkpoint_dir", str(output_dir))
        training_config_path = harness.write_json("regular_decoupled_diloco_paper_config.json", training_config)

        gateway_config_path = harness.write_json(
            "gateway_config.json",
            {
                "gateway_id": "gateway-regular-decoupled-diloco-paper",
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
                        "service_id": "regular-decoupled-diloco-paper",
                        "workers": 2,
                        "worker_class": "cpu",
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
        harness.start_worker(name="worker-1", control_port=worker_fabric_port, backend="cpu")
        harness.start_worker(name="worker-2", control_port=worker_fabric_port, backend="cpu")

        gateway_base = f"http://127.0.0.1:{gateway_port}"
        harness.wait_for_url("gateway healthz", f"{gateway_base}/healthz")
        harness.wait_for_json(
            "training worker discovery",
            "GET",
            f"{gateway_base}/v1/services",
            token=token,
            predicate=lambda payload: any(
                service.get("service_id") == "regular-decoupled-diloco-paper" and service.get("worker_count", 0) >= 2
                for service in payload.get("services", [])
            ),
        )

        submitted = harness.request_json(
            "POST",
            f"{gateway_base}/v1/training/jobs",
            token=token,
            expected_status=202,
        )
        job_id = submitted["job_id"]
        job = harness.wait_for_json(
            "regular Decoupled DiLoCo paper profile completion",
            "GET",
            f"{gateway_base}/v1/jobs/{job_id}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] in ("completed", "failed", "cancelled"),
            timeout=240.0,
            interval=0.5,
        )
        if job["job"]["status"] != "completed":
            raise SystemExit(f"paper-profile Decoupled DiLoCo training did not complete successfully: {job}")

        state_path = output_dir / "training_state.json"
        weights_path = output_dir / "online_weights.bin"
        if not state_path.exists() or not weights_path.exists():
            raise SystemExit(f"missing paper-profile Decoupled DiLoCo outputs under {output_dir}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {
            "aggregation": "decoupled_diloco",
            "workers": 2,
            "syncer_steps": 2,
            "num_fragments": 24,
            "sync_interval_h": 24,
            "overlap_tau": 2,
            "min_quorum": 1,
            "merge_strategy": "avg_embedding_rda_model",
            "fragment_strategy": "balanced_tensor",
            "learner_alpha": 0.0,
            "adaptive_grace_enabled": True,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise SystemExit(f"unexpected {key}: {state}")
        if state.get("event_tape_entries") != 2:
            raise SystemExit(f"unexpected event tape entries: {state}")
        if state.get("quorum_participants_total", 0) < 2:
            raise SystemExit(f"unexpected participant accounting: {state}")
        if state.get("learner_to_syncer_fragment_bytes", 0) <= 0:
            raise SystemExit(f"missing learner-to-syncer byte accounting: {state}")
        if state.get("syncer_to_learner_fragment_bytes", 0) <= 0:
            raise SystemExit(f"missing syncer-to-learner byte accounting: {state}")
        if weights_path.stat().st_size <= 0:
            raise SystemExit(f"empty paper-profile Decoupled DiLoCo weights: {weights_path}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "aggregation": state["aggregation"],
                    "num_fragments": state["num_fragments"],
                    "sync_interval_h": state["sync_interval_h"],
                    "adaptive_grace_enabled": state["adaptive_grace_enabled"],
                },
                sort_keys=True,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
