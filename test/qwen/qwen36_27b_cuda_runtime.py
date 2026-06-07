from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "zig-out" / "bin" / "qwen36_27b_smoke"
DEFAULT_PCP_BIN = REPO_ROOT / "zig-out" / "bin" / "pcp"
VMFB_PATH = REPO_ROOT / "models" / "qwen36_27b_one_token_nocache_linalg_cuda.vmfb"
META_PATH = REPO_ROOT / "models" / "qwen36_27b_one_token_nocache_linalg.mlir.meta.json"
WEIGHTS_PATH = REPO_ROOT / "checkpoints" / "initial_weights" / "qwen36_27b_bf16_streamed.bin"
EXPECTED_LOGITS = 248_320
EXPECTED_WEIGHTS_MIN_BYTES = 50 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class CudaRuntimeResult:
    elapsed_seconds: float
    logits_count: int
    top_value: float
    top_index: int
    memory_before: str
    memory_after: str
    memory_after_mib: int | None
    output_lines: list[str]


def env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def timeout_seconds() -> int:
    return int(os.environ.get("PCP_QWEN36_27B_TIMEOUT_SECONDS", "1800"))


def run_capture(
    cmd: list[str],
    timeout: int = 60,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def runner_env(binary: Path) -> dict[str, str]:
    env = dict(os.environ)
    lib_dirs = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia/current",
    ]
    nix_glibc = nix_glibc_dir(binary)
    if nix_glibc is not None:
        lib_dirs.insert(0, nix_glibc)
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        lib_dirs.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)
    return env


def python_bin_dir() -> str | None:
    for candidate in (REPO_ROOT / ".venv" / "bin", REPO_ROOT / "venv" / "bin"):
        python = candidate / "python"
        if python.exists():
            return str(candidate)
    return None


def nix_glibc_dir(binary: Path) -> str | None:
    result = run_capture(["ldd", str(binary)])
    if result.returncode != 0:
        return None
    match = re.search(r"(/nix/store/[^\s]+-glibc-[^\s]+/lib)/libc\.so\.6", result.stdout)
    return match.group(1) if match else None


def require_cuda_visible(enable_env_name: str) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        pytest.fail(f"{enable_env_name} is set, but nvidia-smi is not on PATH")

    result = run_capture([nvidia_smi, "-L"])
    if result.returncode != 0:
        pytest.fail(
            f"{enable_env_name} is set, but nvidia-smi cannot see a GPU.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    if not list(Path("/dev").glob("nvidia*")):
        pytest.fail(f"{enable_env_name} is set, but /dev/nvidia* is not visible")


def nvidia_memory_snapshot() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "nvidia-smi unavailable"
    result = run_capture(
        [
            nvidia_smi,
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return f"nvidia-smi failed: {result.stderr.strip()}"
    return result.stdout.strip()


def memory_used_mib(snapshot: str) -> int | None:
    match = re.search(r"^\s*\d+,\s*[^,]+,\s*(\d+),\s*(\d+)", snapshot)
    return int(match.group(1)) if match else None


def require_artifact(path: Path, min_size: int = 1) -> None:
    if not path.exists():
        pytest.fail(f"Missing required Qwen3.6-27B artifact: {path}")
    size = path.stat().st_size
    if size < min_size:
        pytest.fail(f"Artifact is too small: {path} has {size} bytes, expected at least {min_size}")


def require_runtime_artifacts() -> None:
    require_artifact(VMFB_PATH)
    require_artifact(META_PATH)
    require_artifact(WEIGHTS_PATH, EXPECTED_WEIGHTS_MIN_BYTES)


def runner_binary() -> Path:
    override = os.environ.get("PCP_QWEN36_27B_RUNNER_BIN") or os.environ.get("PCP_QWEN36_27B_SMOKE_BIN")
    if override:
        path = Path(override)
        if not path.exists():
            pytest.fail(f"configured Qwen3.6-27B runner does not exist: {path}")
        return path

    if DEFAULT_RUNNER.exists():
        return DEFAULT_RUNNER

    nix = shutil.which("nix")
    if nix is None and Path("/nix/var/nix/profiles/default/bin/nix").exists():
        nix = "/nix/var/nix/profiles/default/bin/nix"
    if nix is not None:
        result = run_capture([nix, "develop", "-c", "zig", "build"], timeout=600)
    else:
        result = run_capture(["zig", "build"], timeout=600)
    if result.returncode != 0:
        pytest.fail(f"Failed to build Qwen3.6-27B runner.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    if not DEFAULT_RUNNER.exists():
        pytest.fail(f"Build completed but did not produce {DEFAULT_RUNNER}")
    return DEFAULT_RUNNER


def pcp_binary() -> Path:
    override = os.environ.get("PCP_BIN")
    if override:
        path = Path(override)
        if path.exists():
            return path
        pytest.fail(f"PCP_BIN is set but does not exist: {path}")
    if DEFAULT_PCP_BIN.exists():
        return DEFAULT_PCP_BIN
    result_bin = REPO_ROOT / "result" / "bin" / "pcp"
    if result_bin.exists():
        return result_bin
    pytest.fail("PCP binary not found; expected zig-out/bin/pcp or result/bin/pcp")


def assert_logits_file(path: Path) -> tuple[int, float, int]:
    data = path.read_bytes()
    expected_bytes = EXPECTED_LOGITS * 4
    assert len(data) == expected_bytes, f"expected {expected_bytes} logits bytes, got {len(data)}"

    best_index = 0
    best_value = -math.inf
    for index, (value,) in enumerate(struct.iter_unpack("<f", data)):
        assert math.isfinite(value), f"non-finite logit at index {index}: {value}"
        if value > best_value:
            best_value = value
            best_index = index
    return EXPECTED_LOGITS, best_value, best_index


def interesting_runtime_lines(output: str) -> list[str]:
    prefixes = (
        "Qwen3.6-27B smoke: uploaded",
        "Qwen3.6-27B smoke: invoking",
        "Qwen3.6-27B smoke: logits=",
        "Qwen3.6-27B smoke: timings",
        "info: Successfully created device",
        "info: IREE: GPU kernel completed successfully",
    )
    return [line for line in output.splitlines() if line.startswith(prefixes)]


def run_one_token_cuda(logits_path: Path) -> CudaRuntimeResult:
    binary = runner_binary()
    token = os.environ.get("PCP_QWEN36_27B_TOKEN", "9707")
    position = os.environ.get("PCP_QWEN36_27B_POSITION", "0")

    before_memory = nvidia_memory_snapshot()
    started = time.monotonic()
    result = run_capture(
        [
            str(binary),
            "--backend",
            "cuda",
            "--vmfb",
            str(VMFB_PATH),
            "--meta",
            str(META_PATH),
            "--weights",
            str(WEIGHTS_PATH),
            "--token",
            token,
            "--position",
            position,
            "--logits-out",
            str(logits_path),
        ],
        timeout=timeout_seconds(),
        env=runner_env(binary),
    )
    elapsed = time.monotonic() - started
    after_memory = nvidia_memory_snapshot()

    assert result.returncode == 0, (
        "Qwen3.6-27B CUDA runtime validation failed\n"
        f"elapsed_seconds={elapsed:.3f}\n"
        f"gpu_memory_before={before_memory}\n"
        f"gpu_memory_after={after_memory}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    count, best_value, best_index = assert_logits_file(logits_path)
    return CudaRuntimeResult(
        elapsed_seconds=elapsed,
        logits_count=count,
        top_value=best_value,
        top_index=best_index,
        memory_before=before_memory,
        memory_after=after_memory,
        memory_after_mib=memory_used_mib(after_memory),
        output_lines=interesting_runtime_lines(result.stderr),
    )


def assert_no_runner_processes() -> None:
    result = run_capture(["pgrep", "-af", "qwen36_27b_smoke"])
    if result.returncode == 1:
        return
    assert result.returncode == 0, result.stderr
    pytest.fail(f"Qwen3.6-27B runner process still active:\n{result.stdout}")
