from __future__ import annotations

import json
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from pcp_integration_harness import PCP_ROOT, PCPIntegrationHarness, find_free_port


def main() -> int:
    gateway_port = find_free_port()
    training_api_port = find_free_port()
    worker_fabric_port = find_free_port()
    token = "dev-decoupled-diloco"

    with PCPIntegrationHarness(prefix="pcp-regular-decoupled-diloco-", keep_temp_on_failure=True) as harness:
        output_dir = harness.tmpdir / "decoupled_output"
        training_config_path = harness.write_json(
            "regular_decoupled_diloco_config.json",
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
                "min_quorum": 2,
                "grace_window_ms": 250,
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
                "gateway_id": "gateway-regular-decoupled-diloco",
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
                        "service_id": "regular-decoupled-diloco",
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
                service.get("service_id") == "regular-decoupled-diloco" and service.get("worker_count", 0) >= 2
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
            "regular Decoupled DiLoCo completion",
            "GET",
            f"{gateway_base}/v1/jobs/{job_id}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] in ("completed", "failed", "cancelled"),
            timeout=240.0,
            interval=0.5,
        )
        if job["job"]["status"] != "completed":
            raise SystemExit(f"regular Decoupled DiLoCo training did not complete successfully: {job}")

        state_path = output_dir / "training_state.json"
        weights_path = output_dir / "online_weights.bin"
        optimizer_state_path = output_dir / "outer_optimizer_state.bin"
        syncer_state_path = output_dir / "syncer_state.json"
        if (
            not state_path.exists()
            or not weights_path.exists()
            or not optimizer_state_path.exists()
            or not syncer_state_path.exists()
        ):
            raise SystemExit(f"missing regular Decoupled DiLoCo outputs under {output_dir}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {
            "aggregation": "decoupled_diloco",
            "workers": 2,
            "syncer_steps": 2,
            "num_fragments": 2,
            "sync_interval_h": 2,
            "min_quorum": 2,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise SystemExit(f"unexpected {key}: {state}")
        if state.get("schema") != "pcp.training_state.v1":
            raise SystemExit(f"missing common training state schema: {state}")
        if state.get("training_kind") != "regular":
            raise SystemExit(f"unexpected training kind: {state}")
        if state.get("decoupled", {}).get("syncer_steps") != 2:
            raise SystemExit(f"missing common decoupled config: {state}")
        if state.get("regular", {}).get("model_path") != str(
            (PCP_ROOT / "models" / "nanochat_small.mlir").resolve()
        ):
            raise SystemExit(f"missing regular extension: {state}")
        artifacts = state.get("artifacts", {})
        if artifacts.get("online_weights", {}).get("byte_count") != weights_path.stat().st_size:
            raise SystemExit(f"bad online weight artifact ref: {state}")
        if artifacts.get("outer_optimizer_state", {}).get("byte_count") != optimizer_state_path.stat().st_size:
            raise SystemExit(f"bad outer optimizer artifact ref: {state}")
        if artifacts.get("syncer_state", {}).get("byte_count") != syncer_state_path.stat().st_size:
            raise SystemExit(f"bad syncer state artifact ref: {state}")
        syncer_state = json.loads(syncer_state_path.read_text(encoding="utf-8"))
        if syncer_state.get("global_step") != 2 or syncer_state.get("fragment_versions") != [1, 1]:
            raise SystemExit(f"unexpected syncer state artifact: {syncer_state}")
        if state.get("event_tape_entries") != 2:
            raise SystemExit(f"unexpected event tape entries: {state}")
        if state.get("quorum_participants_total") != 4:
            raise SystemExit(f"unexpected participant accounting: {state}")
        if state.get("skipped_learners_total") != 0:
            raise SystemExit(f"unexpected skipped learner accounting: {state}")
        if state.get("learner_to_syncer_fragment_bytes", 0) <= 0:
            raise SystemExit(f"missing learner-to-syncer byte accounting: {state}")
        if state.get("syncer_to_learner_fragment_bytes", 0) <= 0:
            raise SystemExit(f"missing syncer-to-learner byte accounting: {state}")
        if weights_path.stat().st_size <= 0:
            raise SystemExit(f"empty regular Decoupled DiLoCo weights: {weights_path}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "aggregation": state["aggregation"],
                    "workers": state["workers"],
                    "syncer_steps": state["syncer_steps"],
                },
                sort_keys=True,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
