"""
PCP system smoke test: run a tiny shepherd+worker session locally.

This validates:
  - networking + orchestration (shepherd <-> worker)
  - compile + runtime execution path
  - training loop produces finite loss (no NaNs in logs)
  - checkpoint is written

It does NOT assert numerical parity with PyTorch (use the parity tests for that).

How to run (from `pcp/`):
  nix develop -c ./venv/bin/python test/nanochat/test_pcp_shepherd_worker_smoke.py
"""

import json
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


def main():
    pcp_bin = find_pcp_binary()
    port = find_free_port()

    with tempfile.TemporaryDirectory(prefix="pcp_smoke_") as tmp:
        tmpdir = Path(tmp)
        config_path = tmpdir / "smoke_config.json"

        config = {
            "model_path": str((PCP_ROOT / "models" / "nanochat_small.mlir").resolve()),
            "data_path": str((PCP_ROOT / "data" / "tiny_shakespeare.txt").resolve()),
            "tokenizer": "char",
            "learning_rate": 0.0006,
            "tau": 1,
            "outer_loop_steps": 1,
            "nesterov_momentum": 0.9,
            "max_epochs": 1,
            "wandb_project": "pcp-distributed",
            "wandb_entity": None,
            "wandb_run_name": "pcp-smoke-test",
            "wandb_api_key": None,
        }
        config_path.write_text(json.dumps(config, indent=2))

        shepherd_cmd = [
            pcp_bin,
            "--shepherd",
            "--config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
            "--backend",
            "cpu",
            "--no-dashboard",
        ]
        worker_cmd = [
            pcp_bin,
            "--worker",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend",
            "cpu",
            "--device-id",
            "0",
        ]

        shepherd = subprocess.Popen(
            shepherd_cmd,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            time.sleep(0.5)
            worker = subprocess.Popen(
                worker_cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            shepherd.terminate()
            shepherd.wait(timeout=10)
            raise

        try:
            shepherd_out, _ = shepherd.communicate(timeout=240)
        except subprocess.TimeoutExpired:
            shepherd.kill()
            shepherd_out, _ = shepherd.communicate(timeout=10)
            worker.terminate()
            worker.wait(timeout=10)
            raise SystemExit("Shepherd timed out.\n" + tail(shepherd_out))

        # Worker uses a robust reconnect loop; shut it down after shepherd exits.
        worker.terminate()
        try:
            worker_out, _ = worker.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker_out, _ = worker.communicate(timeout=10)

        combined = f"{shepherd_out}\n{worker_out}"

        if shepherd.returncode != 0:
            raise SystemExit(
                f"Shepherd failed (exit={shepherd.returncode}).\n"
                f"--- shepherd+worker log tail ---\n{tail(combined)}"
            )

        if re.search(r"\bnan\b", combined, flags=re.IGNORECASE):
            raise SystemExit("Found NaN in logs.\n" + tail(combined))

        run_id_file = tmpdir / ".pcp_latest_run"
        if not run_id_file.exists():
            raise SystemExit("Missing .pcp_latest_run (shepherd did not persist run id).\n" + tail(combined))
        run_id = run_id_file.read_text().strip()

        checkpoint = tmpdir / "checkpoints" / run_id / "checkpoint_1.bin"
        if not checkpoint.exists():
            raise SystemExit(f"Missing checkpoint file: {checkpoint}\n" + tail(combined))
        if checkpoint.stat().st_size == 0:
            raise SystemExit(f"Empty checkpoint file: {checkpoint}\n" + tail(combined))

    print("OK: shepherd+worker smoke run completed (finite loss, checkpoint written).")


if __name__ == "__main__":
    main()
