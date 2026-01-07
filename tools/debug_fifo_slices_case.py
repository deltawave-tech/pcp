#!/usr/bin/env python3
"""
FIFO slices debug case (PCP DiLoCo).

Goal: make FIFO slice chunking fully inspectable with tiny numbers.

Defaults:
  - workers = 2
  - outer_loop_steps = 2
  - tau = 2
  - B = 4
  - T = 3

This script will:
  1) Generate a tiny `.u16` token stream where token_value == global_token_index.
  2) Export a nanochat StableHLO MLIR with data inputs shaped `[B,T]`.
  3) Emit an experiment JSON that enables:
       tokenizer="u16", sampling="fifo", chunk_size_mode="diloco_slices"
  4) (Optional) Run PCP locally with 2 CPU workers and assert chunk assignment is correct.
"""

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PCP_DIR = REPO_ROOT / "pcp"
PCP_BIN = PCP_DIR / "result" / "bin" / "pcp"
GEN_NANOCHAT = PCP_DIR / "tools" / "generate_nanochat.py"
PCP_VENV_PY = PCP_DIR / "venv" / "bin" / "python"


@dataclass(frozen=True)
class Case:
    workers: int
    outer_steps: int
    tau: int
    b: int
    t: int
    vocab_size: int

    @property
    def needed_tokens_per_inner_step(self) -> int:
        return self.b * self.t + 1

    @property
    def slice_tokens(self) -> int:
        return self.tau * self.needed_tokens_per_inner_step

    @property
    def slice_bytes(self) -> int:
        return self.slice_tokens * 2  # u16

    @property
    def total_slices(self) -> int:
        return self.workers * self.outer_steps

    @property
    def total_tokens(self) -> int:
        return self.total_slices * self.slice_tokens

    @property
    def total_bytes(self) -> int:
        return self.total_tokens * 2  # u16


def _find_libstdcpp_dir() -> Optional[str]:
    matches = glob.glob("/nix/store/*-gcc-*-lib/lib/libstdc++.so.6")
    if not matches:
        return None
    return str(Path(matches[0]).parent)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _choose_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_u16_stream(path: Path, tokens: Iterable[int]) -> None:
    # `array('H')` is native-endian uint16. Convert to little-endian bytes on big-endian hosts.
    u16 = array("H", tokens)
    if sys.byteorder != "little":
        u16.byteswap()
    with path.open("wb") as f:
        f.write(u16.tobytes())


def _read_u16_stream(path: Path) -> List[int]:
    raw = path.read_bytes()
    if len(raw) % 2 != 0:
        raise AssertionError(f"{path} is not 2-byte aligned (len={len(raw)})")
    u16 = array("H")
    u16.frombytes(raw)
    if sys.byteorder != "little":
        u16.byteswap()
    return list(u16)


def _reshape(flat: List[int], b: int, t: int) -> List[List[int]]:
    assert len(flat) == b * t
    return [flat[i * t : (i + 1) * t] for i in range(b)]


def _print_case_header(case: Case) -> None:
    print("=== CASE ===")
    print(f"workers={case.workers} outer_steps={case.outer_steps} tau={case.tau} B={case.b} T={case.t} vocab={case.vocab_size}")
    print(f"needed_tokens_per_inner_step = B*T+1 = {case.b}*{case.t}+1 = {case.needed_tokens_per_inner_step}")
    print(f"slice_tokens = tau*(B*T+1) = {case.tau}*{case.needed_tokens_per_inner_step} = {case.slice_tokens}")
    print(f"slice_bytes  = slice_tokens*2 = {case.slice_tokens}*2 = {case.slice_bytes} bytes")
    print(f"total_slices = workers*outer_steps = {case.workers}*{case.outer_steps} = {case.total_slices}")
    print(f"total_tokens = total_slices*slice_tokens = {case.total_slices}*{case.slice_tokens} = {case.total_tokens}")
    print(f"total_bytes  = total_tokens*2 = {case.total_tokens}*2 = {case.total_bytes} bytes")
    print()


def _print_expected_schedule(case: Case) -> None:
    print("=== EXPECTED FIFO SLICE SCHEDULE (token indices) ===")
    needed = case.needed_tokens_per_inner_step
    for outer in range(case.outer_steps):
        print(f"outer_round={outer}")
        for worker in range(case.workers):
            chunk_id = outer * case.workers + worker
            slice_start = chunk_id * case.slice_tokens
            slice_end = slice_start + case.slice_tokens  # half-open
            offset_bytes = chunk_id * case.slice_bytes
            print(f"  worker{k(worker)} -> chunk_id={chunk_id} offset_bytes={offset_bytes} length_bytes={case.slice_bytes} tokens=[{slice_start}:{slice_end})")
            for inner in range(case.tau):
                win_start = slice_start + inner * needed
                win_end = win_start + needed
                x_flat = list(range(win_start, win_start + case.b * case.t))
                y_flat = list(range(win_start + 1, win_start + 1 + case.b * case.t))
                x = _reshape(x_flat, case.b, case.t)
                y = _reshape(y_flat, case.b, case.t)
                print(f"    inner_step={inner} window=[{win_start}:{win_end})  (consumes {needed} tokens)")
                print(f"      x={x}")
                print(f"      y={y}")
        print()


def k(worker: int) -> str:
    # small helper to keep output aligned for worker ids >= 10.
    return f"{worker:02d}"


def _assert_stream_matches_schedule(case: Case, tokens: List[int]) -> None:
    assert len(tokens) == case.total_tokens, f"expected {case.total_tokens} tokens, got {len(tokens)}"

    for i, tok in enumerate(tokens):
        assert tok == i, f"token value mismatch at index {i}: got {tok}, want {i}"
        assert 0 <= tok < case.vocab_size, f"token {tok} out of vocab range [0,{case.vocab_size})"

    needed = case.needed_tokens_per_inner_step
    for chunk_id in range(case.total_slices):
        slice_start = chunk_id * case.slice_tokens
        slice_end = slice_start + case.slice_tokens
        assert slice_end <= len(tokens)

        for inner in range(case.tau):
            win_start = slice_start + inner * needed
            win_end = win_start + needed
            assert win_end <= slice_end, "FIFO inner window must fit exactly within slice"

            window = tokens[win_start:win_end]
            expect = list(range(win_start, win_end))
            assert window == expect, f"window mismatch chunk={chunk_id} inner={inner}: got {window}, want {expect}"


def _export_mlir(case: Case, out_mlir: Path, *, force: bool) -> None:
    if out_mlir.exists() and not force:
        print(f"MLIR exists, reusing: {out_mlir}")
        return

    if not PCP_VENV_PY.exists():
        raise SystemExit(f"Missing python venv at {PCP_VENV_PY} (did you set up pcp/venv?)")
    if not GEN_NANOCHAT.exists():
        raise SystemExit(f"Missing generator: {GEN_NANOCHAT}")

    env = os.environ.copy()
    libstdcpp = _find_libstdcpp_dir()
    if libstdcpp:
        env["LD_LIBRARY_PATH"] = f"{libstdcpp}:{env.get('LD_LIBRARY_PATH', '')}"

    # Keep the model tiny so compile is fast.
    cmd = [
        str(PCP_VENV_PY),
        str(GEN_NANOCHAT),
        "--batch-size",
        str(case.b),
        "--block-size",
        str(case.t),
        "--vocab-size",
        str(case.vocab_size),
        "--n-layer",
        "1",
        "--n-head",
        "4",
        "--n-kv-head",
        "4",
        "--n-embd",
        "32",
        "--out",
        str(out_mlir),
    ]

    print("=== EXPORT MLIR ===")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)
    assert out_mlir.exists(), f"MLIR export did not produce {out_mlir}"
    print()


def _assert_mlir_has_bt(out_mlir: Path, *, b: int, t: int) -> None:
    text = out_mlir.read_text()
    match = re.search(r"func\.func @main\(([^)]*)\) ->", text)
    assert match, "could not find func.func @main signature"

    args_str = match.group(1).strip()
    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
    assert len(args) >= 2, "expected at least 2 args (idx, targets)"

    def _arg_type(arg: str) -> str:
        return arg.split(":", 1)[1].strip() if ":" in arg else arg.strip()

    last_two = [_arg_type(args[-2]), _arg_type(args[-1])]
    want = f"tensor<{b}x{t}xi64>"
    assert last_two[0] == want and last_two[1] == want, f"MLIR data inputs are not [{b},{t}]: {last_two}"


def _write_experiment_json(case: Case, out_json: Path, out_mlir: Path, out_u16: Path) -> None:
    config = {
        "model_path": str(out_mlir),
        "data_path": str(out_u16),
        "tokenizer": "u16",
        "sampling": "fifo",
        "chunk_size_mode": "diloco_slices",
        "chunk_manifest_path": None,
        "chunk_shuffle": False,
        "seed": 1,
        "learning_rate": 0.0006,
        "tau": case.tau,
        "outer_loop_steps": case.outer_steps,
        "nesterov_momentum": 0.9,
        "max_epochs": 1,
        "wandb_project": "pcp-distributed",
        "wandb_entity": None,
        "wandb_run_name": f"debug-fifo-slices-b{case.b}-t{case.t}-tau{case.tau}-outer{case.outer_steps}",
        "wandb_api_key": None,
    }
    out_json.write_text(json.dumps(config, indent=2) + "\n")


def _run_pcp(case: Case, exp_json: Path, logs_dir: Path) -> None:
    if not PCP_BIN.exists():
        raise SystemExit(f"Missing PCP binary at {PCP_BIN}. Build it first (from pcp/: nix build -L '.#pcp').")

    _ensure_dir(logs_dir)
    port = _choose_free_local_port()

    shepherd_log = logs_dir / "shepherd.log"
    workers_logs = [logs_dir / f"worker_{i}.log" for i in range(case.workers)]

    print("=== RUN PCP (CPU shepherd + CPU workers) ===")
    print(f"port={port}")
    print(f"logs={logs_dir}")
    print()

    shepherd_cmd = [
        str(PCP_BIN),
        "--shepherd",
        "--config",
        str(exp_json),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(case.workers),
        "--no-dashboard",
    ]

    shep = subprocess.Popen(shepherd_cmd, cwd=str(PCP_DIR), stdout=shepherd_log.open("wb"), stderr=subprocess.STDOUT)
    time.sleep(1.0)

    workers: List[subprocess.Popen] = []
    try:
        for i in range(case.workers):
            cmd = [
                str(PCP_BIN),
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
            p = subprocess.Popen(cmd, cwd=str(PCP_DIR), stdout=workers_logs[i].open("wb"), stderr=subprocess.STDOUT)
            workers.append(p)
            time.sleep(0.3)

        rc = shep.wait(timeout=240)
        if rc != 0:
            raise AssertionError(f"shepherd exited with status {rc} (see {shepherd_log})")
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    _assert_logs(case, shepherd_log, workers_logs)


def _assert_logs(case: Case, shepherd_log: Path, worker_logs: List[Path]) -> None:
    shep = shepherd_log.read_text(errors="replace")
    assert f"slice chunks ({case.slice_bytes} bytes each)" in shep, "shepherd did not initialize slice chunks as expected"
    assert "Training completed successfully!" in shep, "training did not complete"

    # Extract assigned chunk lines from shepherd.
    pat = re.compile(r"Assigned chunk (\d+) \(offset: (\d+), len: (\d+)\) to worker (\d+)")
    rows = [tuple(map(int, m.groups())) for m in pat.finditer(shep)]
    assert rows, "no chunk assignments found in shepherd log"

    # We expect exactly `workers*outer_steps` unique chunk ids and the right offset math.
    seen = {}
    for chunk_id, offset, length, worker_id in rows:
        assert length == case.slice_bytes, f"unexpected slice length in shepherd log: {length} != {case.slice_bytes}"
        assert offset == chunk_id * case.slice_bytes, f"offset mismatch: chunk {chunk_id} offset {offset} != {chunk_id}*{case.slice_bytes}"
        seen[chunk_id] = (offset, length, worker_id)

    expected_ids = set(range(case.total_slices))
    got_ids = set(seen.keys())
    assert got_ids == expected_ids, f"unexpected chunk ids assigned: got={sorted(got_ids)} want={sorted(expected_ids)}"

    # Workers should report fifo dataset selection and the same assigned chunks.
    for i, log_path in enumerate(worker_logs):
        text = log_path.read_text(errors="replace")
        assert "Using U16TokenDatasetFifo" in text, f"worker log missing fifo dataset selection: {log_path}"
        # In this tiny run each worker should run tau steps twice.
        assert f"completed {case.tau} inner loop steps" in text, f"worker did not run tau={case.tau}: {log_path}"

    print("=== PCP LOG ASSERTIONS PASSED ===")
    print(f"shepherd: {shepherd_log}")
    for p in worker_logs:
        print(f"worker:   {p}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--outer", type=int, default=2, dest="outer_steps")
    parser.add_argument("--tau", type=int, default=2)
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--t", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=256, dest="vocab_size")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force-mlir", action="store_true")
    parser.add_argument("--run-pcp", action="store_true")
    args = parser.parse_args()

    case = Case(
        workers=args.workers,
        outer_steps=args.outer_steps,
        tau=args.tau,
        b=args.b,
        t=args.t,
        vocab_size=args.vocab_size,
    )

    assert case.workers > 0
    assert case.outer_steps > 0
    assert case.tau > 0
    assert case.b > 0
    assert case.t > 0
    assert case.vocab_size >= case.total_tokens, (
        "For this debug case we want token_value == token_index, so vocab must be >= total_tokens. "
        f"Got vocab={case.vocab_size}, total_tokens={case.total_tokens}."
    )

    out_dir: Path
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = Path("/tmp") / f"pcp_fifo_slices_demo_b{case.b}_t{case.t}_tau{case.tau}_outer{case.outer_steps}_w{case.workers}_{int(time.time())}"

    _ensure_dir(out_dir)
    logs_dir = out_dir / "logs"

    out_u16 = out_dir / "tokens.u16"
    out_mlir = out_dir / "model.mlir"
    out_json = out_dir / "experiment.json"

    _print_case_header(case)
    _print_expected_schedule(case)

    print("=== GENERATE TOKEN STREAM ===")
    print(f"writing: {out_u16}")
    _write_u16_stream(out_u16, range(case.total_tokens))
    assert out_u16.stat().st_size == case.total_bytes, f"unexpected u16 file size: {out_u16.stat().st_size} != {case.total_bytes}"
    tokens = _read_u16_stream(out_u16)
    _assert_stream_matches_schedule(case, tokens)
    print("token stream OK (values match indices, slice boundaries match schedule)")
    print()

    _export_mlir(case, out_mlir, force=args.force_mlir)
    _assert_mlir_has_bt(out_mlir, b=case.b, t=case.t)
    print(f"MLIR OK (data inputs are [{case.b},{case.t}]): {out_mlir}")
    print()

    _write_experiment_json(case, out_json, out_mlir, out_u16)
    print("=== EXPERIMENT JSON ===")
    print(out_json)
    print()
    print("Run manually:")
    print(f"  cd {PCP_DIR}")
    print(f"  ./result/bin/pcp --shepherd --config {out_json} --host 127.0.0.1 --port 8081 --workers {case.workers} --no-dashboard")
    print(f"  ./result/bin/pcp --worker   --host 127.0.0.1 --port 8081 --backend cpu --device-id 0  # start {case.workers} times")
    print()

    if args.run_pcp:
        _run_pcp(case, out_json, logs_dir)


if __name__ == "__main__":
    main()
