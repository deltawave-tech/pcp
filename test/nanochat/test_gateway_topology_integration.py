"""
PCP gateway topology integration harness.

How to run (from `pcp/`):
  nix build
  ./venv/bin/python test/nanochat/test_gateway_topology_integration.py
  ./venv/bin/python test/nanochat/test_gateway_topology_integration.py --scenario shared-topology
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from pcp_integration_harness import PCP_ROOT, HarnessError, PCPIntegrationHarness, find_free_port


HOST = "127.0.0.1"
JOB_STATUSES = {
    "queued",
    "starting",
    "waiting_for_workers",
    "initializing",
    "running",
    "completed",
    "failed",
    "cancelling",
    "cancelled",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def service_named(payload: dict[str, Any], service_id: str) -> dict[str, Any] | None:
    for service in payload.get("services", []):
        if service.get("service_id") == service_id:
            return service
    return None


def service_with_executor(payload: dict[str, Any], executor_id: str) -> dict[str, Any] | None:
    for service in payload.get("services", []):
        if service.get("executor_id") == executor_id:
            return service
    return None


def job_named(payload: dict[str, Any], job_id: str) -> dict[str, Any] | None:
    for item in payload.get("jobs", []):
        if item.get("job_id") == job_id:
            return item
    return None


def global_job_named(payload: dict[str, Any], global_job_id: str) -> dict[str, Any] | None:
    for item in payload.get("jobs", []):
        if item.get("global_job_id") == global_job_id:
            return item
    return None


def gateway_named(payload: dict[str, Any], gateway_id: str) -> dict[str, Any] | None:
    for gateway in payload.get("gateways", []):
        if gateway.get("gateway_id") == gateway_id:
            return gateway
    return None


def worker_with_lease(payload: dict[str, Any], owner: str) -> dict[str, Any] | None:
    for worker in payload.get("workers", []):
        if worker.get("lease_owner") == owner:
            return worker
    return None


def assert_chat_completion(payload: dict[str, Any]) -> None:
    require(payload.get("object") == "chat.completion", f"unexpected completion payload: {payload}")
    require(bool(payload.get("session_id")), f"missing session_id: {payload}")
    choices = payload.get("choices") or []
    require(bool(choices), f"missing choices: {payload}")


def expected_executor_id(gateway_id: str, service_type: str, service_id: str) -> str:
    return f"{gateway_id}:{service_type}:{service_id}"


def require_service_capacity(service: dict[str, Any], *, connected: int, ready: int, available: int) -> None:
    require(service["worker_count"] == connected, f"service worker_count mismatch: {service}")
    require(service["ready_worker_count"] == available, f"service ready_worker_count mismatch: {service}")
    require(service["workers_connected"] == connected, f"service workers_connected mismatch: {service}")
    require(service["workers_ready"] == ready, f"service workers_ready mismatch: {service}")
    require(service["workers_available"] == available, f"service workers_available mismatch: {service}")
    require(service["workers_dispatchable"] == available, f"service workers_dispatchable mismatch: {service}")


def start_proxy_job_stub(harness: PCPIntegrationHarness, name: str, port: int, label: str, job_type: str = "training") -> None:
    script_path = harness.tmpdir / f"{name}_stub.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import json
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = int(sys.argv[1])
            LABEL = sys.argv[2]
            JOB_TYPE = sys.argv[3]
            STATE = {"status": "idle"}

            def current_job():
                return {
                    "status": STATE["status"],
                    "job_type": JOB_TYPE,
                    "executor_marker": LABEL,
                    "workers_required": 1,
                    "workers_connected": 1,
                }

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, format, *args):
                    return

                def _write_json(self, status, payload):
                    body = json.dumps(payload).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self):
                    if self.path == "/healthz":
                        body = b"ok"
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    if self.path == "/v1/job":
                        self._write_json(200, current_job())
                        return
                    self._write_json(404, {"error": "not_found"})

                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0"))
                    if length:
                        _ = self.rfile.read(length)
                    if self.path == "/v1/job":
                        STATE["status"] = "running"
                        self._write_json(202, current_job())
                        return
                    if self.path == "/v1/job/cancel":
                        STATE["status"] = "cancelled"
                        self._write_json(200, {"accepted": True, "status": "cancelled", "executor_marker": LABEL})
                        return
                    self._write_json(404, {"error": "not_found"})

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            """
        ),
        encoding="utf-8",
    )
    harness.start_process(name=name, command=[sys.executable, str(script_path), str(port), label, job_type])
    harness.wait_for_url(f"{name} health", f"http://{HOST}:{port}/healthz")


def run_training_queue_scenario(keep_temp_on_failure: bool) -> None:
    with PCPIntegrationHarness("pcp_training_queue_", keep_temp_on_failure=keep_temp_on_failure) as harness:
        gateway_id = "lab-alpha-gateway"
        gateway_port = find_free_port()
        train_api_port = find_free_port()
        fabric_port = find_free_port()
        token = "dev-topology"
        internal_token = "dev-topology-internal"

        gateway_config = {
            "gateway_id": gateway_id,
            "lab_id": "lab-alpha",
            "graph_backend": "memory",
            "api_token_env": "PCP_GATEWAY_API_TOKEN",
            "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
            "worker_fabric": {
                "host": HOST,
                "port": fabric_port,
            },
            "controllers": {
                "training": {
                    "enabled": True,
                    "config_path": str((PCP_ROOT / "experiments" / "nanochat_u16_smoke_d1_t128.json").resolve()),
                    "service_id": "training-main",
                    "workers": 2,
                    "worker_class": "cpu",
                    "api": {
                        "host": HOST,
                        "port": train_api_port,
                    },
                }
            },
        }
        gateway_config_path = harness.write_json("gateway_training.json", gateway_config)

        harness.start_gateway(
            name="gateway",
            gateway_config_path=gateway_config_path,
            gateway_port=gateway_port,
            env={
                "PCP_GATEWAY_API_TOKEN": token,
                "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
            },
        )

        harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz")
        services = harness.wait_for_json(
            "training service registration",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: service_named(payload, "training-main") is not None,
        )
        training_service = service_named(services, "training-main")
        require(training_service is not None, f"training service missing: {services}")
        require(training_service["service_type"] == "training", f"unexpected service: {training_service}")
        training_executor_id = expected_executor_id(gateway_id, "training", "training-main")
        require(training_service["executor_id"] == training_executor_id, f"training executor mismatch: {training_service}")
        _, invalid_override_text = harness.request(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/training/jobs",
            token=token,
            payload={"learning_rate": 1e-4},
            expected_status=400,
        )
        require("UnsupportedJobOverrides" in invalid_override_text, f"training override rejection mismatch: {invalid_override_text}")
        training_detail = harness.request_json(
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services/{training_executor_id}",
            token=token,
        )
        require(training_detail["executor_id"] == training_executor_id, f"training detail mismatch: {training_detail}")

        harness.start_worker(name="training_worker", control_port=fabric_port, backend="cpu")
        services = harness.wait_for_json(
            "training worker discovery",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: (
                service_named(payload, "training-main") is not None
                and service_named(payload, "training-main")["worker_count"] >= 1
            ),
        )

        submit_one = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/training/jobs",
            token=token,
            expected_status=202,
        )
        job_id_one = submit_one["job_id"]
        job_one = harness.wait_for_json(
            "training job enters waiting_for_workers",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id_one}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] == "waiting_for_workers",
        )

        submit_two = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/training/jobs",
            token=token,
            expected_status=202,
        )
        job_id_two = submit_two["job_id"]
        job_two = harness.wait_for_json(
            "second training job queues",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id_two}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] == "queued",
        )

        cancel = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id_two}/cancel",
            token=token,
            expected_status=200,
        )
        job_two = harness.wait_for_json(
            "second training job cancels",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id_two}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] == "cancelled",
        )
        jobs = harness.request_json("GET", f"http://{HOST}:{gateway_port}/v1/jobs", token=token)

        require(submit_one["accepted"] is True, f"submit_one not accepted: {submit_one}")
        require(submit_one["queue_position"] == 0, f"submit_one queue_position mismatch: {submit_one}")
        require(submit_one["executor_id"] == training_executor_id, f"submit_one executor mismatch: {submit_one}")
        require(submit_one["job_id"] == f"{training_executor_id}:job-1", f"submit_one job_id mismatch: {submit_one}")
        require(submit_one["job"]["job_type"] == "training", f"submit_one job_type mismatch: {submit_one}")
        require(submit_one["job"]["worker_class"] == "cpu", f"submit_one worker_class mismatch: {submit_one}")
        require(submit_one["job"]["workers_required"] == 2, f"submit_one workers_required mismatch: {submit_one}")
        require(job_one["job"]["workers_connected"] == 1, f"job_one workers_connected mismatch: {job_one}")
        require(job_one["job"]["status"] == "waiting_for_workers", f"job_one status mismatch: {job_one}")
        require(job_one["executor_id"] == training_executor_id, f"job_one executor mismatch: {job_one}")

        require(submit_two["accepted"] is True, f"submit_two not accepted: {submit_two}")
        require(submit_two["queue_position"] == 1, f"submit_two queue_position mismatch: {submit_two}")
        require(submit_two["executor_id"] == training_executor_id, f"submit_two executor mismatch: {submit_two}")
        require(submit_two["job_id"] == f"{training_executor_id}:job-2", f"submit_two job_id mismatch: {submit_two}")
        require(submit_two["job"]["status"] == "queued", f"submit_two status mismatch: {submit_two}")
        require(job_two["job"]["finished_at"] is not None, f"job_two missing finished_at: {job_two}")

        require(cancel["accepted"] is True, f"cancel not accepted: {cancel}")
        require(cancel["status"] == "cancelled", f"cancel status mismatch: {cancel}")
        require(cancel["job_id"] == job_id_two, f"cancel job_id mismatch: {cancel}")
        require(cancel["executor_id"] == training_executor_id, f"cancel executor mismatch: {cancel}")

        listed_one = job_named(jobs, job_id_one)
        listed_two = job_named(jobs, job_id_two)
        require(listed_one is not None, f"job one missing from jobs list: {jobs}")
        require(listed_two is not None, f"job two missing from jobs list: {jobs}")
        require(listed_one["job"]["status"] == "waiting_for_workers", f"listed job one mismatch: {listed_one}")
        require(listed_two["job"]["status"] == "cancelled", f"listed job two mismatch: {listed_two}")
        require(listed_one["executor_id"] == training_executor_id, f"listed job one executor mismatch: {listed_one}")
        require(listed_two["executor_id"] == training_executor_id, f"listed job two executor mismatch: {listed_two}")
        require(service_named(services, "training-main")["worker_count"] >= 1, f"training service mismatch: {services}")


def run_rl_queue_scenario(keep_temp_on_failure: bool) -> None:
    with PCPIntegrationHarness("pcp_rl_queue_", keep_temp_on_failure=keep_temp_on_failure) as harness:
        gateway_id = "lab-alpha-gateway"
        gateway_port = find_free_port()
        rl_api_port = find_free_port()
        fabric_port = find_free_port()
        token = "dev-topology"
        internal_token = "dev-topology-internal"

        gateway_config = {
            "gateway_id": gateway_id,
            "lab_id": "lab-alpha",
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
                    "config_path": str((PCP_ROOT / "experiments" / "qwen_rl_api_smoke.json").resolve()),
                    "service_id": "rl-main",
                    "workers": 2,
                    "backend": "cpu",
                    "worker_class": "cpu",
                    "api": {
                        "host": HOST,
                        "port": rl_api_port,
                    },
                }
            },
        }
        gateway_config_path = harness.write_json("gateway_rl.json", gateway_config)

        harness.start_gateway(
            name="gateway",
            gateway_config_path=gateway_config_path,
            gateway_port=gateway_port,
            env={
                "PCP_GATEWAY_API_TOKEN": token,
                "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
            },
        )

        harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz")
        services = harness.wait_for_json(
            "rl service registration",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: service_named(payload, "rl-main") is not None,
        )
        rl_service = service_named(services, "rl-main")
        require(rl_service is not None, f"rl service missing: {services}")
        require(rl_service["service_type"] == "rl", f"unexpected service: {rl_service}")
        rl_executor_id = expected_executor_id(gateway_id, "rl", "rl-main")
        require(rl_service["executor_id"] == rl_executor_id, f"rl executor mismatch: {rl_service}")
        _, invalid_override_text = harness.request(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/rl/jobs",
            token=token,
            payload={"reward_scale": 1.5},
            expected_status=400,
        )
        require("UnsupportedJobOverrides" in invalid_override_text, f"rl override rejection mismatch: {invalid_override_text}")
        rl_detail = harness.request_json(
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services/{rl_executor_id}",
            token=token,
        )
        require(rl_detail["executor_id"] == rl_executor_id, f"rl detail mismatch: {rl_detail}")

        harness.start_worker(name="rl_worker", control_port=fabric_port, backend="cpu")
        services = harness.wait_for_json(
            "rl worker discovery",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: (
                service_named(payload, "rl-main") is not None
                and service_named(payload, "rl-main")["worker_count"] >= 1
            ),
        )

        submit_one = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/rl/jobs",
            token=token,
            expected_status=202,
        )
        job_id_one = submit_one["job_id"]
        job_one = harness.wait_for_json(
            "rl job enters waiting_for_workers",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id_one}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] == "waiting_for_workers",
        )

        submit_two = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/rl/jobs",
            token=token,
            expected_status=202,
        )
        job_id_two = submit_two["job_id"]
        job_two = harness.wait_for_json(
            "second rl job queues",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id_two}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] == "queued",
        )

        cancel = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id_two}/cancel",
            token=token,
            expected_status=200,
        )
        job_two = harness.wait_for_json(
            "second rl job cancels",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id_two}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] == "cancelled",
        )
        jobs = harness.request_json("GET", f"http://{HOST}:{gateway_port}/v1/jobs", token=token)

        require(submit_one["accepted"] is True, f"submit_one not accepted: {submit_one}")
        require(submit_one["queue_position"] == 0, f"submit_one queue_position mismatch: {submit_one}")
        require(submit_one["executor_id"] == rl_executor_id, f"submit_one executor mismatch: {submit_one}")
        require(submit_one["job_id"] == f"{rl_executor_id}:job-1", f"submit_one job_id mismatch: {submit_one}")
        require(submit_one["job"]["job_type"] == "rl", f"submit_one job_type mismatch: {submit_one}")
        require(submit_one["job"]["worker_class"] == "cpu", f"submit_one worker_class mismatch: {submit_one}")
        require(submit_one["job"]["workers_required"] == 2, f"submit_one workers_required mismatch: {submit_one}")
        require(job_one["job"]["workers_connected"] == 1, f"job_one workers_connected mismatch: {job_one}")
        require(job_one["job"]["status"] == "waiting_for_workers", f"job_one status mismatch: {job_one}")
        require(job_one["executor_id"] == rl_executor_id, f"job_one executor mismatch: {job_one}")

        require(submit_two["accepted"] is True, f"submit_two not accepted: {submit_two}")
        require(submit_two["queue_position"] == 1, f"submit_two queue_position mismatch: {submit_two}")
        require(submit_two["executor_id"] == rl_executor_id, f"submit_two executor mismatch: {submit_two}")
        require(submit_two["job_id"] == f"{rl_executor_id}:job-2", f"submit_two job_id mismatch: {submit_two}")
        require(submit_two["job"]["status"] == "queued", f"submit_two status mismatch: {submit_two}")
        require(job_two["job"]["finished_at"] is not None, f"job_two missing finished_at: {job_two}")

        require(cancel["accepted"] is True, f"cancel not accepted: {cancel}")
        require(cancel["status"] == "cancelled", f"cancel status mismatch: {cancel}")
        require(cancel["job_id"] == job_id_two, f"cancel job_id mismatch: {cancel}")
        require(cancel["executor_id"] == rl_executor_id, f"cancel executor mismatch: {cancel}")

        listed_one = job_named(jobs, job_id_one)
        listed_two = job_named(jobs, job_id_two)
        require(listed_one is not None, f"job one missing from jobs list: {jobs}")
        require(listed_two is not None, f"job two missing from jobs list: {jobs}")
        require(listed_one["job"]["status"] == "waiting_for_workers", f"listed job one mismatch: {listed_one}")
        require(listed_two["job"]["status"] == "cancelled", f"listed job two mismatch: {listed_two}")
        require(listed_one["executor_id"] == rl_executor_id, f"listed job one executor mismatch: {listed_one}")
        require(listed_two["executor_id"] == rl_executor_id, f"listed job two executor mismatch: {listed_two}")
        require(service_named(services, "rl-main")["worker_count"] >= 1, f"rl service mismatch: {services}")


def run_proxy_identity_scenario(keep_temp_on_failure: bool) -> None:
    with PCPIntegrationHarness("pcp_proxy_identity_", keep_temp_on_failure=keep_temp_on_failure) as harness:
        gateway_id = "lab-alpha-gateway"
        gateway_port = find_free_port()
        token = "dev-proxy"
        internal_token = "dev-proxy-internal"
        proxy_a_port = find_free_port()
        proxy_b_port = find_free_port()
        logical_service_id = "training-proxy"
        executor_a = "remote-a:training:training-proxy"
        executor_b = "remote-b:training:training-proxy"

        gateway_config = {
            "gateway_id": gateway_id,
            "lab_id": "lab-alpha",
            "graph_backend": "memory",
            "api_token_env": "PCP_GATEWAY_API_TOKEN",
            "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
        }
        gateway_config_path = harness.write_json("gateway_proxy_identity.json", gateway_config)

        harness.start_gateway(
            name="gateway",
            gateway_config_path=gateway_config_path,
            gateway_port=gateway_port,
            env={
                "PCP_GATEWAY_API_TOKEN": token,
                "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
            },
        )

        harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz")
        start_proxy_job_stub(harness, "proxy_a", proxy_a_port, "proxy-a")
        start_proxy_job_stub(harness, "proxy_b", proxy_b_port, "proxy-b")

        register_payloads = [
            {
                "service_id": logical_service_id,
                "executor_id": executor_a,
                "service_type": "training",
                "base_url": f"http://{HOST}:{proxy_a_port}",
                "auth_mode": "none",
                "health_status": "ok",
                "job_status": "idle",
                "worker_count": 1,
                "ready_worker_count": 1,
                "capabilities": ["training.job"],
            },
            {
                "service_id": logical_service_id,
                "executor_id": executor_b,
                "service_type": "training",
                "base_url": f"http://{HOST}:{proxy_b_port}",
                "auth_mode": "none",
                "health_status": "ok",
                "job_status": "idle",
                "worker_count": 2,
                "ready_worker_count": 2,
                "capabilities": ["training.job"],
            },
        ]
        for payload in register_payloads:
            response = harness.request_json(
                "POST",
                f"http://{HOST}:{gateway_port}/v1/services/register",
                token=token,
                payload=payload,
            )
            require(response["accepted"] is True, f"service registration failed: {response}")

        services = harness.wait_for_json(
            "proxy services register",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: (
                service_with_executor(payload, executor_a) is not None
                and service_with_executor(payload, executor_b) is not None
            ),
        )
        matching_services = [service for service in services.get("services", []) if service.get("service_id") == logical_service_id]
        require(len(matching_services) == 2, f"expected duplicate logical service ids: {services}")
        require_service_capacity(service_with_executor(services, executor_a), connected=1, ready=1, available=1)
        require_service_capacity(service_with_executor(services, executor_b), connected=2, ready=2, available=2)

        detail = harness.request_json(
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services/{executor_b}",
            token=token,
        )
        require(detail["service_id"] == logical_service_id, f"service detail mismatch: {detail}")
        require(detail["executor_id"] == executor_b, f"service detail executor mismatch: {detail}")
        require_service_capacity(detail, connected=2, ready=2, available=2)

        default_submit = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/training/jobs",
            token=token,
            payload={"service_id": logical_service_id},
            expected_status=202,
        )
        default_job_id = default_submit["job_id"]
        require(default_submit["service_id"] == logical_service_id, f"default submit service mismatch: {default_submit}")
        require(default_submit["executor_id"] == executor_b, f"default submit executor mismatch: {default_submit}")
        require(default_job_id == f"{executor_b}:current", f"default submit job id mismatch: {default_submit}")
        require(default_submit["job"]["executor_marker"] == "proxy-b", f"default submit marker mismatch: {default_submit}")

        default_lookup = harness.request_json(
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{default_job_id}",
            token=token,
        )
        require(default_lookup["executor_id"] == executor_b, f"default lookup executor mismatch: {default_lookup}")
        require(default_lookup["job"]["executor_marker"] == "proxy-b", f"default lookup marker mismatch: {default_lookup}")

        default_cancel = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/jobs/{default_job_id}/cancel",
            token=token,
            expected_status=200,
        )
        require(default_cancel["accepted"] is True, f"default cancel rejected: {default_cancel}")
        require(default_cancel["executor_id"] == executor_b, f"default cancel executor mismatch: {default_cancel}")

        explicit_submit = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/training/jobs",
            token=token,
            payload={"service_id": logical_service_id, "executor_id": executor_a},
            expected_status=202,
        )
        explicit_job_id = explicit_submit["job_id"]
        require(explicit_submit["executor_id"] == executor_a, f"explicit submit executor mismatch: {explicit_submit}")
        require(explicit_job_id == f"{executor_a}:current", f"explicit submit job id mismatch: {explicit_submit}")
        require(explicit_submit["job"]["executor_marker"] == "proxy-a", f"explicit submit marker mismatch: {explicit_submit}")

        explicit_lookup = harness.request_json(
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{explicit_job_id}",
            token=token,
        )
        require(explicit_lookup["executor_id"] == executor_a, f"explicit lookup executor mismatch: {explicit_lookup}")
        require(explicit_lookup["job"]["executor_marker"] == "proxy-a", f"explicit lookup marker mismatch: {explicit_lookup}")

        jobs = harness.request_json("GET", f"http://{HOST}:{gateway_port}/v1/jobs", token=token)
        listed_a = job_named(jobs, f"{executor_a}:current")
        listed_b = job_named(jobs, f"{executor_b}:current")
        require(listed_a is not None, f"executor_a missing from jobs list: {jobs}")
        require(listed_b is not None, f"executor_b missing from jobs list: {jobs}")
        require(listed_a["job"]["executor_marker"] == "proxy-a", f"executor_a jobs list mismatch: {listed_a}")
        require(listed_b["job"]["executor_marker"] == "proxy-b", f"executor_b jobs list mismatch: {listed_b}")


def run_shared_topology_scenario(keep_temp_on_failure: bool) -> None:
    with PCPIntegrationHarness("pcp_shared_topology_", keep_temp_on_failure=keep_temp_on_failure) as harness:
        gateway_id = "lab-alpha-gateway"
        gateway_port = find_free_port()
        fabric_port = find_free_port()
        train_api_port = find_free_port()
        rl_api_port = find_free_port()
        inf_api_port = find_free_port()
        token = "dev-topology"
        internal_token = "dev-topology-internal"

        gateway_config = {
            "gateway_id": gateway_id,
            "lab_id": "lab-alpha",
            "graph_backend": "memory",
            "api_token_env": "PCP_GATEWAY_API_TOKEN",
            "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
            "worker_fabric": {
                "host": HOST,
                "port": fabric_port,
            },
            "controllers": {
                "training": {
                    "enabled": True,
                    "config_path": str((PCP_ROOT / "experiments" / "nanochat_u16_smoke_d1_t128.json").resolve()),
                    "service_id": "training-main",
                    "workers": 1,
                    "should_resume": False,
                    "worker_class": "cpu",
                    "api": {
                        "host": HOST,
                        "port": train_api_port,
                    },
                },
                "rl": {
                    "enabled": True,
                    "config_path": str((PCP_ROOT / "experiments" / "qwen_rl_api_smoke.json").resolve()),
                    "service_id": "rl-main",
                    "workers": 1,
                    "backend": "cpu",
                    "worker_class": "cpu",
                    "api": {
                        "host": HOST,
                        "port": rl_api_port,
                    },
                },
                "inference": {
                    "enabled": True,
                    "config_path": str((PCP_ROOT / "experiments" / "inference_qwen.json").resolve()),
                    "service_id": "inference-main",
                    "worker_class": "cuda",
                    "target_arch": "sm_80",
                    "api": {
                        "host": HOST,
                        "port": inf_api_port,
                    },
                },
            },
        }
        gateway_config_path = harness.write_json("gateway_shared_topology.json", gateway_config)

        harness.start_gateway(
            name="gateway",
            gateway_config_path=gateway_config_path,
            gateway_port=gateway_port,
            env={
                "PCP_GATEWAY_API_TOKEN": token,
                "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
            },
        )

        harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz")
        harness.wait_for_url("training controller health", f"http://{HOST}:{train_api_port}/healthz")
        harness.wait_for_url("rl controller health", f"http://{HOST}:{rl_api_port}/healthz")

        harness.start_worker(name="cpu_worker_a", control_port=fabric_port, backend="cpu")
        harness.start_worker(name="cpu_worker_b", control_port=fabric_port, backend="cpu")
        harness.wait_for_json(
            "training cpu worker visibility",
            "GET",
            f"http://{HOST}:{train_api_port}/v1/workers",
            token=token,
            predicate=lambda payload: any(worker.get("backend") == "cpu" for worker in payload.get("workers", [])),
        )
        harness.wait_for_json(
            "rl cpu worker visibility",
            "GET",
            f"http://{HOST}:{rl_api_port}/v1/workers",
            token=token,
            predicate=lambda payload: any(worker.get("backend") == "cpu" for worker in payload.get("workers", [])),
        )

        harness.start_worker(
            name="cuda_worker_wrong_arch",
            control_port=fabric_port,
            backend="cuda",
            target="sm_90",
        )

        cuda_worker = harness.start_worker(
            name="cuda_worker",
            control_port=fabric_port,
            backend="cuda",
            target="sm_80",
        )

        ready = harness.wait_for_json(
            "inference controller ready",
            "GET",
            f"http://{HOST}:{inf_api_port}/readyz",
            token=token,
            predicate=lambda payload: (
                payload.get("ready") is True
                and payload.get("available_workers") == 1
                and payload.get("worker_class") == "cuda"
                and payload.get("target_arch") == "sm_80"
            ),
            timeout=240.0,
        )
        inference_workers = harness.wait_for_json(
            "inference compatibility gating",
            "GET",
            f"http://{HOST}:{inf_api_port}/v1/workers",
            token=token,
            predicate=lambda payload: (
                any(
                    worker.get("backend") == "cuda"
                    and worker.get("target_arch") == "sm_80"
                    and worker.get("status") == "initialized"
                    for worker in payload.get("workers", [])
                )
                and any(
                    worker.get("backend") == "cuda"
                    and worker.get("target_arch") == "sm_90"
                    and worker.get("status") == "connected"
                    for worker in payload.get("workers", [])
                )
                and sum(
                    1
                    for worker in payload.get("workers", [])
                    if worker.get("backend") == "cpu" and worker.get("status") == "connected"
                )
                >= 2
            ),
            timeout=240.0,
        )
        services = harness.wait_for_json(
            "all services register",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: all(
                service_named(payload, service_id) is not None
                for service_id in ("training-main", "rl-main", "inference-main")
            ),
        )
        training_executor_id = expected_executor_id(gateway_id, "training", "training-main")
        rl_executor_id = expected_executor_id(gateway_id, "rl", "rl-main")
        inference_executor_id = expected_executor_id(gateway_id, "inference", "inference-main")
        require(service_named(services, "training-main")["executor_id"] == training_executor_id, f"training executor mismatch: {services}")
        require(service_named(services, "rl-main")["executor_id"] == rl_executor_id, f"rl executor mismatch: {services}")
        require(service_named(services, "inference-main")["executor_id"] == inference_executor_id, f"inference executor mismatch: {services}")

        submit_training = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/training/jobs",
            token=token,
            expected_status=202,
        )
        training_job_id = submit_training["job_id"]
        submit_rl = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/rl/jobs",
            token=token,
            expected_status=202,
        )
        rl_job_id = submit_rl["job_id"]

        training_workers = harness.wait_for_json(
            "training lease acquisition",
            "GET",
            f"http://{HOST}:{train_api_port}/v1/workers",
            token=token,
            predicate=lambda payload: (
                worker_with_lease(payload, "Training") is not None
                and worker_with_lease(payload, "Training")["available"] is False
            ),
        )
        rl_workers = harness.wait_for_json(
            "rl lease acquisition",
            "GET",
            f"http://{HOST}:{rl_api_port}/v1/workers",
            token=token,
            predicate=lambda payload: (
                worker_with_lease(payload, "RL") is not None
                and worker_with_lease(payload, "RL")["available"] is False
            ),
        )
        services = harness.wait_for_json(
            "shared topology publication",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: (
                service_named(payload, "training-main") is not None
                and service_named(payload, "rl-main") is not None
                and service_named(payload, "inference-main") is not None
                and service_named(payload, "training-main")["worker_count"] == 1
                and service_named(payload, "training-main").get("workers_connected") == 1
                and service_named(payload, "training-main").get("workers_ready") == 1
                and service_named(payload, "training-main")["ready_worker_count"] == 0
                and service_named(payload, "training-main").get("workers_available") == 0
                and service_named(payload, "rl-main")["worker_count"] == 1
                and service_named(payload, "rl-main").get("workers_connected") == 1
                and service_named(payload, "rl-main").get("workers_ready") == 1
                and service_named(payload, "rl-main")["ready_worker_count"] == 0
                and service_named(payload, "rl-main").get("workers_available") == 0
                and service_named(payload, "inference-main")["health_status"] == "ok"
                and service_named(payload, "inference-main").get("workers_connected") == 1
                and service_named(payload, "inference-main").get("workers_ready") == 1
                and service_named(payload, "inference-main")["ready_worker_count"] == 1
                and service_named(payload, "inference-main").get("workers_available") == 1
            ),
        )

        training_job = harness.wait_for_json(
            "training job visibility",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{training_job_id}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] in JOB_STATUSES,
        )
        rl_job = harness.wait_for_json(
            "rl job visibility",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{rl_job_id}",
            token=token,
            predicate=lambda payload: payload["job"]["status"] in JOB_STATUSES,
        )

        direct_completion = harness.request_json(
            "POST",
            f"http://{HOST}:{inf_api_port}/v1/chat/completions",
            token=token,
            payload={
                "model": "qwen2.5-0.5b-instruct",
                "messages": [{"role": "user", "content": "Reply with SHARED_GATEWAY_OK and nothing else."}],
                "max_tokens": 8,
            },
            timeout=60.0,
        )
        gateway_completion = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/inference/chat/completions",
            token=token,
            payload={
                "model": "qwen2.5-0.5b-instruct",
                "messages": [{"role": "user", "content": "Reply with SHARED_GATEWAY_OK and nothing else."}],
                "max_tokens": 8,
            },
            timeout=60.0,
        )

        require(ready["health_status"] == "ok", f"inference ready mismatch: {ready}")
        compatible_worker = next(
            worker
            for worker in inference_workers["workers"]
            if worker.get("backend") == "cuda" and worker.get("target_arch") == "sm_80"
        )
        wrong_arch_worker = next(
            worker
            for worker in inference_workers["workers"]
            if worker.get("backend") == "cuda" and worker.get("target_arch") == "sm_90"
        )
        cpu_workers = [worker for worker in inference_workers["workers"] if worker.get("backend") == "cpu"]
        require(compatible_worker["status"] == "initialized", f"compatible inference worker mismatch: {inference_workers}")
        require(wrong_arch_worker["status"] == "connected", f"wrong-arch worker should not load inference model: {inference_workers}")
        require(cpu_workers and all(worker["status"] == "connected" for worker in cpu_workers), f"cpu workers should not load inference model: {inference_workers}")
        require(service_named(services, "training-main")["job_status"] in JOB_STATUSES, f"training service mismatch: {services}")
        require(service_named(services, "rl-main")["job_status"] in JOB_STATUSES, f"rl service mismatch: {services}")
        require_service_capacity(service_named(services, "training-main"), connected=1, ready=1, available=0)
        require_service_capacity(service_named(services, "rl-main"), connected=1, ready=1, available=0)
        require_service_capacity(service_named(services, "inference-main"), connected=1, ready=1, available=1)
        require(training_job["executor_id"] == training_executor_id, f"training executor mismatch: {training_job}")
        require(training_job["job"]["job_type"] == "training", f"training job mismatch: {training_job}")
        require(training_job["job"]["workers_connected"] == 1, f"training workers_connected mismatch: {training_job}")
        require(rl_job["executor_id"] == rl_executor_id, f"rl executor mismatch: {rl_job}")
        require(rl_job["job"]["job_type"] == "rl", f"rl job mismatch: {rl_job}")
        require(rl_job["job"]["workers_connected"] == 1, f"rl workers_connected mismatch: {rl_job}")
        require(worker_with_lease(training_workers, "Training") is not None, f"training workers mismatch: {training_workers}")
        require(worker_with_lease(rl_workers, "RL") is not None, f"rl workers mismatch: {rl_workers}")
        assert_chat_completion(direct_completion)
        assert_chat_completion(gateway_completion)

        harness.stop_process(cuda_worker)
        ready = harness.wait_for_json(
            "inference worker loss propagates",
            "GET",
            f"http://{HOST}:{inf_api_port}/readyz",
            token=token,
            expected_status=503,
            predicate=lambda payload: (
                payload.get("ready") is False
                and payload.get("available_workers") == 0
                and payload.get("health_status") == "starting"
            ),
            timeout=120.0,
        )
        services = harness.wait_for_json(
            "gateway sees inference worker loss",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: (
                service_named(payload, "inference-main") is not None
                and service_named(payload, "inference-main")["ready_worker_count"] == 0
                and service_named(payload, "inference-main").get("workers_ready") == 0
                and service_named(payload, "inference-main").get("workers_available") == 0
            ),
            timeout=120.0,
        )
        require(ready["available_workers"] == 0, f"inference did not lose worker readiness: {ready}")
        require_service_capacity(service_named(services, "inference-main"), connected=0, ready=0, available=0)

        harness.start_worker(
            name="cuda_worker_restarted",
            control_port=fabric_port,
            backend="cuda",
            target="sm_80",
        )
        ready = harness.wait_for_json(
            "inference worker restart recovers",
            "GET",
            f"http://{HOST}:{inf_api_port}/readyz",
            token=token,
            predicate=lambda payload: (
                payload.get("ready") is True
                and payload.get("available_workers") == 1
                and payload.get("health_status") == "ok"
            ),
            timeout=240.0,
        )
        services = harness.wait_for_json(
            "gateway sees inference worker recovery",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=token,
            predicate=lambda payload: (
                service_named(payload, "inference-main") is not None
                and service_named(payload, "inference-main")["ready_worker_count"] == 1
                and service_named(payload, "inference-main").get("workers_ready") == 1
                and service_named(payload, "inference-main").get("workers_available") == 1
            ),
            timeout=240.0,
        )
        require(ready["target_arch"] == "sm_80", f"inference recovery target_arch mismatch: {ready}")
        require(service_named(services, "inference-main")["health_status"] == "ok", f"service publication mismatch after recovery: {services}")
        require_service_capacity(service_named(services, "inference-main"), connected=1, ready=1, available=1)


def run_federated_query_scenario(keep_temp_on_failure: bool) -> None:
    with PCPIntegrationHarness("pcp_federated_query_", keep_temp_on_failure=keep_temp_on_failure) as harness:
        gateway_id = "lab-alpha-gateway"
        gateway_port = find_free_port()
        hub_port = find_free_port()
        inf_api_port = find_free_port()
        fabric_port = find_free_port()
        gateway_token = "dev-gateway"
        internal_token = "dev-internal"
        hub_token = "dev-global"

        gateway_config = {
            "gateway_id": gateway_id,
            "lab_id": "lab-alpha",
            "graph_backend": "memory",
            "federation_hub_endpoint": f"http://{HOST}:{hub_port}",
            "federation": {
                "enabled": True,
                "upstream": f"http://{HOST}:{hub_port}",
                "token_env": "PCP_FEDERATION_HUB_TOKEN",
                "heartbeat_interval_ms": 250,
            },
            "api_token_env": "PCP_GATEWAY_API_TOKEN",
            "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
            "worker_fabric": {
                "host": HOST,
                "port": fabric_port,
            },
            "controllers": {
                "inference": {
                    "enabled": True,
                    "config_path": str((PCP_ROOT / "experiments" / "inference_qwen.json").resolve()),
                    "service_id": "inference-main",
                    "worker_class": "cuda",
                    "target_arch": "sm_80",
                    "api": {
                        "host": HOST,
                        "port": inf_api_port,
                    },
                }
            },
        }
        gateway_config_path = harness.write_json("gateway_federated.json", gateway_config)

        harness.start_hub(
            name="federation_hub",
            hub_port=hub_port,
            token_env_name="PCP_FEDERATION_HUB_TOKEN",
            token_value=hub_token,
        )
        harness.start_gateway(
            name="gateway",
            gateway_config_path=gateway_config_path,
            gateway_port=gateway_port,
            env={
                "PCP_GATEWAY_API_TOKEN": gateway_token,
                "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
                "PCP_FEDERATION_HUB_TOKEN": hub_token,
            },
        )
        harness.start_worker(
            name="cuda_worker",
            control_port=fabric_port,
            backend="cuda",
            target="sm_80",
        )

        harness.wait_for_url("hub health", f"http://{HOST}:{hub_port}/healthz")
        harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz")
        harness.wait_for_json(
            "inference ready",
            "GET",
            f"http://{HOST}:{inf_api_port}/readyz",
            token=gateway_token,
            predicate=lambda payload: (
                payload.get("ready") is True
                and payload.get("worker_class") == "cuda"
                and payload.get("target_arch") == "sm_80"
            ),
            timeout=240.0,
        )
        harness.wait_for_json(
            "federation connected",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/federation/status",
            token=gateway_token,
            predicate=lambda payload: payload.get("connected") is True and payload.get("replication_enabled") is True,
            timeout=120.0,
        )
        services = harness.wait_for_json(
            "federated inference publication",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=gateway_token,
            predicate=lambda payload: (
                service_named(payload, "inference-main") is not None
                and service_named(payload, "inference-main")["health_status"] == "ok"
                and service_named(payload, "inference-main")["ready_worker_count"] == 1
                and service_named(payload, "inference-main").get("workers_ready") == 1
                and service_named(payload, "inference-main").get("workers_available") == 1
            ),
            timeout=240.0,
        )
        require(service_named(services, "inference-main")["service_type"] == "inference", f"inference service mismatch: {services}")
        inference_executor_id = expected_executor_id(gateway_id, "inference", "inference-main")
        require(service_named(services, "inference-main")["executor_id"] == inference_executor_id, f"inference executor mismatch: {services}")
        require_service_capacity(service_named(services, "inference-main"), connected=1, ready=1, available=1)

        hub_peers = harness.wait_for_json(
            "hub peers include federated services",
            "GET",
            f"http://{HOST}:{hub_port}/v1/federation/peers",
            token=hub_token,
            predicate=lambda payload: (
                gateway_named(payload, gateway_id) is not None
                and any(
                    service.get("executor_id") == inference_executor_id
                    for service in gateway_named(payload, gateway_id).get("services", [])
                )
            ),
            timeout=120.0,
        )
        gateway_peers = harness.wait_for_json(
            "gateway peers include federated services",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/federation/peers",
            token=gateway_token,
            predicate=lambda payload: (
                gateway_named(payload, gateway_id) is not None
                and any(
                    service.get("executor_id") == inference_executor_id
                    for service in gateway_named(payload, gateway_id).get("services", [])
                )
            ),
            timeout=120.0,
        )
        hub_services = harness.wait_for_json(
            "hub federation services",
            "GET",
            f"http://{HOST}:{hub_port}/v1/federation/services",
            token=hub_token,
            predicate=lambda payload: service_with_executor(payload, inference_executor_id) is not None,
            timeout=120.0,
        )
        gateway_services = harness.wait_for_json(
            "gateway federation services",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/federation/services",
            token=gateway_token,
            predicate=lambda payload: service_with_executor(payload, inference_executor_id) is not None,
            timeout=120.0,
        )
        require(service_with_executor(hub_services, inference_executor_id)["gateway_id"] == gateway_id, f"hub federation service mismatch: {hub_services}")
        require(service_with_executor(gateway_services, inference_executor_id)["gateway_id"] == gateway_id, f"gateway federation service mismatch: {gateway_services}")
        require_service_capacity(service_with_executor(hub_services, inference_executor_id), connected=1, ready=1, available=1)
        require_service_capacity(service_with_executor(gateway_services, inference_executor_id), connected=1, ready=1, available=1)
        require(gateway_named(hub_peers, gateway_id)["registered_services"] >= 1, f"hub peers missing service count: {hub_peers}")
        require(gateway_named(gateway_peers, gateway_id)["registered_services"] >= 1, f"gateway peers missing service count: {gateway_peers}")

        harness.request_json(
            "PUT",
            f"http://{HOST}:{gateway_port}/v1/graph/policies/lab-alpha/shared",
            token=gateway_token,
            payload={
                "default_visibility": "shared",
                "allow_global_replication": True,
                "allow_raw_payload_export": True,
            },
        )
        harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/graph/mutate",
            token=gateway_token,
            payload={
                "mutations": [
                    {
                        "mutation_type": "upsert_entity",
                        "namespace_id": "lab-alpha/shared",
                        "target_id": "experiment:shared-1",
                        "payload": {
                            "entity_type": "experiment",
                            "display_name": "Shared Experiment",
                            "properties": {
                                "scope": "shared",
                                "owner": "lab-alpha",
                            },
                        },
                        "visibility": "shared",
                        "provenance": {
                            "service_id": "eve",
                            "actor_id": "eve-agent-1",
                        },
                    },
                    {
                        "mutation_type": "upsert_entity",
                        "namespace_id": "lab-alpha/shared",
                        "target_id": "experiment:local-1",
                        "payload": {
                            "entity_type": "experiment",
                            "display_name": "Local Experiment",
                            "properties": {
                                "scope": "local",
                                "owner": "lab-alpha",
                            },
                        },
                        "visibility": "local",
                        "provenance": {
                            "service_id": "eve",
                            "actor_id": "eve-agent-1",
                        },
                    },
                ],
            },
        )

        hub_status = harness.wait_for_json(
            "hub replication status",
            "GET",
            f"http://{HOST}:{hub_port}/v1/global-graph/status",
            token=hub_token,
            predicate=lambda payload: payload.get("global_graph", {}).get("entities") == 1,
            timeout=120.0,
        )
        require(hub_status["global_graph"]["entities"] == 1, f"hub entity count mismatch: {hub_status}")

        query_then_infer = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/inference/query/chat/completions",
            token=gateway_token,
            payload={
                "graph_query": {
                    "query_mode": "local_plus_global",
                    "namespaces": ["lab-alpha/shared"],
                    "entity_types": ["experiment"],
                    "limit": 10,
                },
                "inference": {
                    "model": "qwen2.5-0.5b-instruct",
                    "messages": [{"role": "user", "content": "Reply with CONTEXT_OK and nothing else."}],
                    "max_tokens": 8,
                },
            },
            timeout=60.0,
        )

        require("graph_query" in query_then_infer, f"missing graph_query: {query_then_infer}")
        require("graph_context_prompt" in query_then_infer, f"missing graph_context_prompt: {query_then_infer}")
        require("completion" in query_then_infer, f"missing completion: {query_then_infer}")
        require(query_then_infer["completion"]["object"] == "chat.completion", f"bad completion object: {query_then_infer}")
        require(bool(query_then_infer["completion"].get("session_id")), f"missing completion session_id: {query_then_infer}")
        graph_context = query_then_infer["graph_context_prompt"]
        require("experiment:shared-1" in graph_context, f"shared entity missing from graph context: {query_then_infer}")
        require("experiment:local-1" in graph_context, f"local entity missing from graph context: {query_then_infer}")
        entity_ids = {entity["entity_id"] for entity in query_then_infer["graph_query"]["entities"]}
        require("experiment:shared-1" in entity_ids, f"shared entity missing from graph query: {query_then_infer}")
        require("experiment:local-1" in entity_ids, f"local entity missing from graph query: {query_then_infer}")


def run_federated_training_jobs_scenario(keep_temp_on_failure: bool) -> None:
    with PCPIntegrationHarness("pcp_federated_training_jobs_", keep_temp_on_failure=keep_temp_on_failure) as harness:
        gateway_id = "lab-alpha-gateway"
        gateway_port = find_free_port()
        hub_port = find_free_port()
        train_api_port = find_free_port()
        fabric_port = find_free_port()
        gateway_token = "dev-gateway"
        internal_token = "dev-internal"
        hub_token = "dev-global"

        gateway_config = {
            "gateway_id": gateway_id,
            "lab_id": "lab-alpha",
            "graph_backend": "memory",
            "federation_hub_endpoint": f"http://{HOST}:{hub_port}",
            "federation": {
                "enabled": True,
                "upstream": f"http://{HOST}:{hub_port}",
                "token_env": "PCP_FEDERATION_HUB_TOKEN",
                "heartbeat_interval_ms": 250,
            },
            "api_token_env": "PCP_GATEWAY_API_TOKEN",
            "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
            "worker_fabric": {
                "host": HOST,
                "port": fabric_port,
            },
            "controllers": {
                "training": {
                    "enabled": True,
                    "config_path": str((PCP_ROOT / "experiments" / "nanochat_u16_smoke_d1_t128.json").resolve()),
                    "service_id": "training-main",
                    "workers": 2,
                    "worker_class": "cpu",
                    "api": {
                        "host": HOST,
                        "port": train_api_port,
                    },
                }
            },
        }
        gateway_config_path = harness.write_json("gateway_federated_training_jobs.json", gateway_config)

        harness.start_hub(
            name="federation_hub",
            hub_port=hub_port,
            token_env_name="PCP_FEDERATION_HUB_TOKEN",
            token_value=hub_token,
        )
        harness.start_gateway(
            name="gateway",
            gateway_config_path=gateway_config_path,
            gateway_port=gateway_port,
            env={
                "PCP_GATEWAY_API_TOKEN": gateway_token,
                "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
                "PCP_FEDERATION_HUB_TOKEN": hub_token,
            },
        )
        harness.start_worker(name="training_worker", control_port=fabric_port, backend="cpu")

        harness.wait_for_url("hub health", f"http://{HOST}:{hub_port}/healthz")
        harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz")
        training_executor_id = expected_executor_id(gateway_id, "training", "training-main")
        hub_services = harness.wait_for_json(
            "hub sees training service",
            "GET",
            f"http://{HOST}:{hub_port}/v1/federation/services",
            token=hub_token,
            predicate=lambda payload: service_with_executor(payload, training_executor_id) is not None,
            timeout=120.0,
        )
        training_service = service_with_executor(hub_services, training_executor_id)
        require(training_service is not None, f"training service missing from hub: {hub_services}")
        require(training_service["gateway_id"] == gateway_id, f"training gateway mismatch: {hub_services}")
        require(training_service["worker_class"] == "cpu", f"training worker_class mismatch: {hub_services}")
        require(training_service["target_arch"] is None, f"training target_arch mismatch: {hub_services}")

        submit_one = harness.request_json(
            "POST",
            f"http://{HOST}:{hub_port}/v1/training/jobs",
            token=hub_token,
            payload={
                "placement": {
                    "gateway_id": gateway_id,
                    "worker_class": "cpu",
                    "workers_required": 2,
                }
            },
            expected_status=202,
        )
        global_job_one = submit_one["global_job_id"]
        child_job_one = submit_one["assignment"]["child_job_id"]
        require(global_job_one.startswith("global-job-"), f"global job id mismatch: {submit_one}")
        require(submit_one["assignment"]["gateway_id"] == gateway_id, f"gateway assignment mismatch: {submit_one}")
        require(submit_one["assignment"]["executor_id"] == training_executor_id, f"executor assignment mismatch: {submit_one}")
        require(submit_one["child"]["job_id"] == child_job_one, f"child job id mismatch: {submit_one}")

        lookup_one = harness.wait_for_json(
            "hub lookup waiting_for_workers",
            "GET",
            f"http://{HOST}:{hub_port}/v1/jobs/{global_job_one}",
            token=hub_token,
            predicate=lambda payload: payload["child"]["job"]["status"] == "waiting_for_workers",
            timeout=120.0,
        )
        require(lookup_one["assignment"]["child_job_id"] == child_job_one, f"lookup child mismatch: {lookup_one}")
        require(lookup_one["child"]["job"]["workers_connected"] == 1, f"lookup workers mismatch: {lookup_one}")
        require(lookup_one["child"]["job"]["workers_required"] == 2, f"lookup workers_required mismatch: {lookup_one}")

        submit_two = harness.request_json(
            "POST",
            f"http://{HOST}:{hub_port}/v1/training/jobs",
            token=hub_token,
            payload={"placement": {"gateway_id": gateway_id, "worker_class": "cpu"}},
            expected_status=202,
        )
        global_job_two = submit_two["global_job_id"]

        lookup_two = harness.wait_for_json(
            "hub second job queues",
            "GET",
            f"http://{HOST}:{hub_port}/v1/jobs/{global_job_two}",
            token=hub_token,
            predicate=lambda payload: payload["child"]["job"]["status"] == "queued",
            timeout=120.0,
        )
        require(lookup_two["assignment"]["gateway_id"] == gateway_id, f"second job gateway mismatch: {lookup_two}")

        cancel_two = harness.request_json(
            "POST",
            f"http://{HOST}:{hub_port}/v1/jobs/{global_job_two}/cancel",
            token=hub_token,
            expected_status=200,
        )
        require(cancel_two["child"]["accepted"] is True, f"cancel accept mismatch: {cancel_two}")
        require(cancel_two["child"]["status"] == "cancelled", f"cancel status mismatch: {cancel_two}")

        jobs = harness.wait_for_json(
            "hub jobs list",
            "GET",
            f"http://{HOST}:{hub_port}/v1/jobs",
            token=hub_token,
            predicate=lambda payload: (
                global_job_named(payload, global_job_one) is not None
                and global_job_named(payload, global_job_two) is not None
            ),
            timeout=120.0,
        )
        listed_one = global_job_named(jobs, global_job_one)
        listed_two = global_job_named(jobs, global_job_two)
        require(listed_one is not None, f"missing first global job: {jobs}")
        require(listed_two is not None, f"missing second global job: {jobs}")
        require(listed_one["child"]["job"]["status"] == "waiting_for_workers", f"listed first job mismatch: {listed_one}")
        require(listed_two["child"]["job"]["status"] == "cancelled", f"listed second job mismatch: {listed_two}")


def run_federated_training_reservations_scenario(keep_temp_on_failure: bool) -> None:
    with PCPIntegrationHarness("pcp_federated_training_reservations_", keep_temp_on_failure=keep_temp_on_failure) as harness:
        gateway_id = "lab-alpha-gateway"
        gateway_port = find_free_port()
        hub_port = find_free_port()
        train_api_port = find_free_port()
        fabric_port = find_free_port()
        gateway_token = "dev-gateway"
        internal_token = "dev-internal"
        hub_token = "dev-global"

        gateway_config = {
            "gateway_id": gateway_id,
            "lab_id": "lab-alpha",
            "graph_backend": "memory",
            "federation_hub_endpoint": f"http://{HOST}:{hub_port}",
            "federation": {
                "enabled": True,
                "upstream": f"http://{HOST}:{hub_port}",
                "token_env": "PCP_FEDERATION_HUB_TOKEN",
                "heartbeat_interval_ms": 250,
            },
            "api_token_env": "PCP_GATEWAY_API_TOKEN",
            "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
            "worker_fabric": {
                "host": HOST,
                "port": fabric_port,
            },
            "controllers": {
                "training": {
                    "enabled": True,
                    "config_path": str((PCP_ROOT / "experiments" / "nanochat_u16_smoke_d1_t128.json").resolve()),
                    "service_id": "training-main",
                    "workers": 1,
                    "worker_class": "cpu",
                    "api": {
                        "host": HOST,
                        "port": train_api_port,
                    },
                }
            },
        }
        gateway_config_path = harness.write_json("gateway_federated_training_reservations.json", gateway_config)

        harness.start_hub(
            name="federation_hub",
            hub_port=hub_port,
            token_env_name="PCP_FEDERATION_HUB_TOKEN",
            token_value=hub_token,
        )
        harness.start_gateway(
            name="gateway",
            gateway_config_path=gateway_config_path,
            gateway_port=gateway_port,
            env={
                "PCP_GATEWAY_API_TOKEN": gateway_token,
                "PCP_GATEWAY_INTERNAL_TOKEN": internal_token,
                "PCP_FEDERATION_HUB_TOKEN": hub_token,
            },
        )
        harness.start_worker(name="training_worker", control_port=fabric_port, backend="cpu")

        harness.wait_for_url("hub health", f"http://{HOST}:{hub_port}/healthz")
        harness.wait_for_url("gateway health", f"http://{HOST}:{gateway_port}/healthz")
        training_executor_id = expected_executor_id(gateway_id, "training", "training-main")
        services = harness.wait_for_json(
            "training service ready for reservations",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=gateway_token,
            predicate=lambda payload: (
                service_with_executor(payload, training_executor_id) is not None
                and service_with_executor(payload, training_executor_id)["health_status"] == "ok"
                and service_with_executor(payload, training_executor_id)["workers_available"] == 1
            ),
            timeout=120.0,
        )
        require_service_capacity(service_with_executor(services, training_executor_id), connected=1, ready=1, available=1)

        reservation = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/internal/federation/training/reservations",
            token=hub_token,
            payload={"executor_id": training_executor_id, "workers_required": 1},
            expected_status=202,
        )
        reservation_id = reservation["reservation_id"]
        require(reservation["executor_id"] == training_executor_id, f"reservation executor mismatch: {reservation}")
        require(reservation["workers_required"] == 1, f"reservation worker count mismatch: {reservation}")

        services = harness.wait_for_json(
            "reservation holds capacity",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=gateway_token,
            predicate=lambda payload: (
                service_with_executor(payload, training_executor_id) is not None
                and service_with_executor(payload, training_executor_id)["workers_available"] == 0
            ),
            timeout=120.0,
        )
        require_service_capacity(service_with_executor(services, training_executor_id), connected=1, ready=0, available=0)

        status, body = harness.request(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/internal/federation/training/reservations",
            token=hub_token,
            payload={"executor_id": training_executor_id},
            expected_status=409,
        )
        require(status == 409, f"second reservation status mismatch: {status} body={body}")
        require(body == "ExecutorBusy", f"second reservation body mismatch: {body}")

        released = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/internal/federation/reservations/{reservation_id}/release",
            token=hub_token,
            expected_status=200,
        )
        require(released["accepted"] is True, f"release accept mismatch: {released}")
        require(released["status"] == "released", f"release status mismatch: {released}")

        services = harness.wait_for_json(
            "release restores capacity",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/services",
            token=gateway_token,
            predicate=lambda payload: (
                service_with_executor(payload, training_executor_id) is not None
                and service_with_executor(payload, training_executor_id)["workers_available"] == 1
            ),
            timeout=120.0,
        )
        require_service_capacity(service_with_executor(services, training_executor_id), connected=1, ready=1, available=1)

        reservation = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/internal/federation/training/reservations",
            token=hub_token,
            payload={"executor_id": training_executor_id},
            expected_status=202,
        )
        reservation_id = reservation["reservation_id"]

        committed = harness.request_json(
            "POST",
            f"http://{HOST}:{gateway_port}/v1/internal/federation/reservations/{reservation_id}/commit",
            token=hub_token,
            expected_status=202,
        )
        job_id = committed["job_id"]
        require(committed["executor_id"] == training_executor_id, f"commit executor mismatch: {committed}")
        require(committed["job_id"].startswith(f"{training_executor_id}:job-"), f"commit job id mismatch: {committed}")

        lookup = harness.wait_for_json(
            "reserved job starts without waiting",
            "GET",
            f"http://{HOST}:{gateway_port}/v1/jobs/{job_id}",
            token=gateway_token,
            predicate=lambda payload: payload["job"]["status"] not in {"queued", "waiting_for_workers"},
            timeout=120.0,
        )
        require(lookup["executor_id"] == training_executor_id, f"lookup executor mismatch: {lookup}")
        require(lookup["job"]["workers_required"] == 1, f"lookup workers_required mismatch: {lookup}")
        require(lookup["job"]["status"] in JOB_STATUSES, f"lookup status mismatch: {lookup}")


SCENARIOS: dict[str, Callable[[bool], None]] = {
    "training-queue": run_training_queue_scenario,
    "rl-queue": run_rl_queue_scenario,
    "proxy-identity": run_proxy_identity_scenario,
    "shared-topology": run_shared_topology_scenario,
    "federated-query": run_federated_query_scenario,
    "federated-training-jobs": run_federated_training_jobs_scenario,
    "federated-training-reservations": run_federated_training_reservations_scenario,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCP gateway topology integration scenarios.")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="Run only the named scenario. May be provided multiple times.",
    )
    parser.add_argument(
        "--keep-temp-on-failure",
        action="store_true",
        help="Leave tempdirs and logs on disk when a scenario fails.",
    )
    args = parser.parse_args()

    selected = args.scenario or list(SCENARIOS)
    for name in selected:
        print(f"==> {name}", flush=True)
        try:
            SCENARIOS[name](args.keep_temp_on_failure)
        except HarnessError as err:
            raise SystemExit(f"{name} failed:\n{err}") from err
        except AssertionError as err:
            raise SystemExit(f"{name} failed:\n{err}") from err
        print(f"OK: {name}", flush=True)


if __name__ == "__main__":
    main()
