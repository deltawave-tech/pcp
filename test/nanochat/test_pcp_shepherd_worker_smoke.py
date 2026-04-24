"""
PCP system smoke test: run a tiny gateway-owned training session locally.

This validates:
  - gateway -> worker-fabric orchestration
  - embedded training controller startup
  - training loop completes without NaNs in logs

How to run (from `pcp/`):
  nix develop -c ./venv/bin/python test/nanochat/test_pcp_shepherd_worker_smoke.py
"""

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from pcp_test_utils import PCP_ROOT, find_pcp_binary


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def tail(text: str, lines: int = 120) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def controller_status(api_port: int, token: str) -> str | None:
    proc = subprocess.run(
        [
            "curl",
            "-sf",
            f"http://127.0.0.1:{api_port}/v1/controller",
            "-H",
            f"Authorization: Bearer {token}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def main():
    pcp_bin = find_pcp_binary()
    worker_port = find_free_port()
    gateway_port = find_free_port()
    controller_api_port = find_free_port()

    with tempfile.TemporaryDirectory(prefix="pcp_gateway_smoke_") as tmp:
        tmpdir = Path(tmp)
        config_path = tmpdir / "smoke_config.json"
        gateway_config_path = tmpdir / "gateway_config.json"

        config = {
            "model_path": str((PCP_ROOT / "models" / "nanochat_small.mlir").resolve()),
            "data_path": str((PCP_ROOT / "data" / "tiny_shakespeare.txt").resolve()),
            "tokenizer": "char",
            "sampling": "sequence",
            "dtype": "f32",
            "learning_rate": 0.0006,
            "tau": 1,
            "outer_loop_steps": 1,
            "nesterov_momentum": 0.9,
            "max_epochs": 1,
            "wandb_project": "pcp-distributed",
            "wandb_entity": None,
            "wandb_run_name": "pcp-gateway-smoke-test",
            "wandb_api_key": None,
        }
        config_path.write_text(json.dumps(config, indent=2))

        gateway_config = {
            "gateway_id": "gateway-smoke-test",
            "lab_id": "local-smoke",
            "graph_backend": "memory",
            "api_token_env": "PCP_GATEWAY_API_TOKEN",
            "internal_api_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
            "worker_fabric": {
                "host": "127.0.0.1",
                "port": worker_port,
            },
            "api": {
                "host": "127.0.0.1",
                "port": gateway_port,
                "token_env": "PCP_GATEWAY_API_TOKEN",
                "internal_token_env": "PCP_GATEWAY_INTERNAL_TOKEN",
            },
            "controllers": {
                "training": {
                    "enabled": True,
                    "config_path": str(config_path),
                    "service_id": "training-main",
                    "workers": 1,
                    "api": {
                        "host": "127.0.0.1",
                        "port": controller_api_port,
                    },
                }
            },
        }
        gateway_config_path.write_text(json.dumps(gateway_config, indent=2))

        env = dict(os.environ)
        env["PCP_GATEWAY_API_TOKEN"] = "dev"
        env["PCP_GATEWAY_INTERNAL_TOKEN"] = "dev-internal"

        gateway_cmd = [
            pcp_bin,
            "--gateway",
            "--gateway-config",
            str(gateway_config_path),
            "--gateway-host",
            "127.0.0.1",
            "--gateway-port",
            str(gateway_port),
        ]
        worker_cmd = [
            pcp_bin,
            "--worker",
            "--connect",
            f"127.0.0.1:{worker_port}",
            "--backend",
            "cpu",
            "--device-id",
            "0",
        ]

        gateway = subprocess.Popen(
            gateway_cmd,
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            time.sleep(1.0)
            worker = subprocess.Popen(
                worker_cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            gateway.terminate()
            gateway.wait(timeout=10)
            raise

        try:
            completed = False
            last_controller = ""
            for _ in range(240):
                if gateway.poll() is not None:
                    break
                last_controller = controller_status(controller_api_port, env["PCP_GATEWAY_API_TOKEN"]) or last_controller
                if last_controller and '"status":"completed"' in last_controller:
                    completed = True
                    break
                if last_controller and '"status":"failed"' in last_controller:
                    break
                time.sleep(1)

            gateway.terminate()
            gateway_out, _ = gateway.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            gateway.kill()
            gateway_out, _ = gateway.communicate(timeout=10)
            worker.terminate()
            worker.wait(timeout=10)
            raise SystemExit("Gateway timed out.\n" + tail(gateway_out))

        worker.terminate()
        try:
            worker_out, _ = worker.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker_out, _ = worker.communicate(timeout=10)

        combined = f"{gateway_out}\n{worker_out}"

        if not completed:
            raise SystemExit(
                "Embedded training controller did not complete.\n"
                f"--- controller status ---\n{last_controller}\n"
                f"--- gateway+worker log tail ---\n{tail(combined)}"
            )

        if re.search(r"\bnan\b", combined, flags=re.IGNORECASE):
            raise SystemExit("Found NaN in logs.\n" + tail(combined))

    print("OK: gateway-owned training smoke run completed (finite loss).")


if __name__ == "__main__":
    main()
