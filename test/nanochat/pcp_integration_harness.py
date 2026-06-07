from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO


PCP_ROOT = Path(__file__).resolve().parents[2]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _append_env_path(env: dict[str, str], var: str, paths: list[str]) -> None:
    existing = env.get(var, "")
    existing_parts = [part for part in existing.split(os.pathsep) if part] if existing else []
    env[var] = os.pathsep.join(_dedupe_keep_order(existing_parts + paths))


def _maybe_add_cuda_driver_to_ld_library_path(env: dict[str, str]) -> None:
    override = env.get("PCP_TEST_CUDA_LIBRARY_PATH") or env.get("CUDA_DRIVER_LIBRARY_PATH")
    override_paths = [path for path in (override.split(os.pathsep) if override else []) if path]

    libcuda_candidates: list[Path] = []
    for base in override_paths:
        libcuda_candidates.append(Path(base) / "libcuda.so")
        libcuda_candidates.append(Path(base) / "libcuda.so.1")

    libcuda_candidates.extend(
        [
            Path("/run/opengl-driver/lib/libcuda.so"),
            Path("/run/opengl-driver/lib/libcuda.so.1"),
            Path("/run/opengl-driver-32/lib/libcuda.so"),
            Path("/run/opengl-driver-32/lib/libcuda.so.1"),
            Path("/usr/lib/wsl/lib/libcuda.so"),
            Path("/usr/lib/wsl/lib/libcuda.so.1"),
            Path("/usr/lib/x86_64-linux-gnu/libcuda.so"),
            Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
            Path("/usr/lib/x86_64-linux-gnu/nvidia/current/libcuda.so"),
            Path("/usr/lib/x86_64-linux-gnu/nvidia/current/libcuda.so.1"),
            Path("/usr/lib64/libcuda.so"),
            Path("/usr/lib64/libcuda.so.1"),
            Path("/usr/local/nvidia/lib64/libcuda.so"),
            Path("/usr/local/nvidia/lib64/libcuda.so.1"),
        ]
    )

    driver_dirs: list[str] = []
    for candidate in libcuda_candidates:
        if not candidate.exists():
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        lib_dir = resolved.parent
        if (lib_dir / "libc.so.6").exists() or (lib_dir / "libm.so.6").exists():
            continue
        driver_dirs.append(str(lib_dir))

    if driver_dirs:
        _append_env_path(env, "LD_LIBRARY_PATH", _dedupe_keep_order(driver_dirs))


def build_process_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    _maybe_add_cuda_driver_to_ld_library_path(env)
    if extra:
        env.update(extra)
    return env


def find_pcp_binary() -> str:
    override = os.environ.get("PCP_BIN")
    if override:
        override_path = Path(override)
        if override_path.exists():
            return str(override_path)
        raise RuntimeError(f"PCP_BIN is set but does not exist: {override}")

    candidates = [
        PCP_ROOT / "result" / "bin" / "pcp",
        PCP_ROOT / "zig-out" / "bin" / "pcp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which("pcp")
    if found:
        return found

    raise RuntimeError("pcp binary not found (checked result/bin/pcp, zig-out/bin/pcp, PATH).")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class HarnessError(RuntimeError):
    pass


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    log_path: Path
    log_file: TextIO
    process: subprocess.Popen[str]
    expect_running: bool = True

    def stop(self, timeout: float = 20.0) -> None:
        self.expect_running = False
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log_file.close()

    def mark_stopped(self) -> None:
        self.expect_running = False

    def tail(self, lines: int = 120) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        parts = text.splitlines()
        return "\n".join(parts[-lines:])


class PCPIntegrationHarness:
    def __init__(self, prefix: str, keep_temp_on_failure: bool = False) -> None:
        self.tmpdir = Path(os.path.realpath(tempfile.mkdtemp(prefix=prefix)))
        self.keep_temp_on_failure = keep_temp_on_failure
        self.pcp_bin = find_pcp_binary()
        self.base_env = build_process_env()
        self.processes: list[ManagedProcess] = []

    def __enter__(self) -> PCPIntegrationHarness:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(cleanup_tmp=exc is None or not self.keep_temp_on_failure)

    def close(self, cleanup_tmp: bool = True) -> None:
        for process in reversed(self.processes):
            try:
                process.stop()
            except Exception:
                pass
        self.processes.clear()
        if cleanup_tmp:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def diagnostics(self, lines: int = 120) -> str:
        sections = [f"tempdir: {self.tmpdir}"]
        for process in self.processes:
            sections.append(f"--- {process.name}: {process.log_path} ---\n{process.tail(lines)}")
        return "\n".join(sections)

    def write_json(self, filename: str, payload: Any) -> Path:
        path = self.tmpdir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def start_process(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> ManagedProcess:
        log_path = self.tmpdir / f"{name}.log"
        log_file = log_path.open("w", encoding="utf-8")
        merged_env = dict(self.base_env)
        if env:
            merged_env.update(env)
        process = subprocess.Popen(
            command,
            cwd=str(cwd or PCP_ROOT),
            env=merged_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        handle = ManagedProcess(
            name=name,
            command=list(command),
            log_path=log_path,
            log_file=log_file,
            process=process,
        )
        self.processes.append(handle)
        return handle

    def start_gateway(
        self,
        name: str,
        gateway_config_path: Path,
        gateway_port: int,
        env: dict[str, str],
        host: str = "127.0.0.1",
    ) -> ManagedProcess:
        return self.start_process(
            name=name,
            command=[
                self.pcp_bin,
                "--gateway",
                "--gateway-config",
                str(gateway_config_path),
                "--gateway-host",
                host,
                "--gateway-port",
                str(gateway_port),
            ],
            env=env,
        )

    def start_hub(
        self,
        name: str,
        hub_port: int,
        token_env_name: str,
        token_value: str,
        host: str = "127.0.0.1",
    ) -> ManagedProcess:
        return self.start_process(
            name=name,
            command=[
                self.pcp_bin,
                "--federation-hub",
                "--api-host",
                host,
                "--api-port",
                str(hub_port),
                "--api-token-env",
                token_env_name,
            ],
            env={token_env_name: token_value},
        )

    def start_worker(
        self,
        name: str,
        control_port: int,
        backend: str,
        host: str = "127.0.0.1",
        target: str | None = None,
        device_id: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ManagedProcess:
        command = [
            self.pcp_bin,
            "--worker",
            "--connect",
            f"{host}:{control_port}",
            "--backend",
            backend,
        ]
        if target:
            command.extend(["--target", target])
        if device_id is not None:
            command.extend(["--device-id", str(device_id)])
        return self.start_process(name=name, command=command, env=env)

    def stop_process(self, process: ManagedProcess, timeout: float = 20.0) -> None:
        process.stop(timeout=timeout)

    def mark_process_stopped(self, process: ManagedProcess) -> None:
        process.mark_stopped()

    def _raise_on_dead_processes(self) -> None:
        dead: list[str] = []
        for process in self.processes:
            if not process.expect_running:
                continue
            if process.process.poll() is not None:
                dead.append(f"{process.name} exited with {process.process.returncode}")
        if dead:
            raise HarnessError("\n".join(dead) + "\n" + self.diagnostics())

    def wait_for(
        self,
        description: str,
        supplier: Callable[[], Any],
        predicate: Callable[[Any], bool] | None = None,
        timeout: float = 120.0,
        interval: float = 0.5,
    ) -> Any:
        deadline = time.monotonic() + timeout
        last_value: Any = None
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            self._raise_on_dead_processes()
            try:
                last_value = supplier()
                if predicate is None or predicate(last_value):
                    return last_value
            except Exception as exc:  # pragma: no cover - diagnostic path
                last_error = exc
            time.sleep(interval)

        detail_parts = [f"Timed out waiting for {description}."]
        if last_error is not None:
            detail_parts.append(f"last_error={last_error}")
        if last_value is not None:
            detail_parts.append(f"last_value={last_value}")
        detail_parts.append(self.diagnostics())
        raise HarnessError("\n".join(detail_parts))

    def request(
        self,
        method: str,
        url: str,
        token: str | None = None,
        payload: Any | None = None,
        expected_status: int = 200,
        timeout: float = 5.0,
    ) -> tuple[int, str]:
        headers: dict[str, str] = {}
        body: bytes | None = None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.getcode()
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            status = err.code
            text = err.read().decode("utf-8", errors="replace")
        except TimeoutError as err:
            raise HarnessError(f"Request timed out for {url}: {err}") from err
        except socket.timeout as err:
            raise HarnessError(f"Request timed out for {url}: {err}") from err
        except urllib.error.URLError as err:
            raise HarnessError(f"Request failed for {url}: {err}") from err

        if status != expected_status:
            raise HarnessError(f"Unexpected HTTP status for {url}: got {status}, wanted {expected_status}, body={text}")
        return status, text

    def request_json(
        self,
        method: str,
        url: str,
        token: str | None = None,
        payload: Any | None = None,
        expected_status: int = 200,
        timeout: float = 5.0,
    ) -> Any:
        _, text = self.request(
            method=method,
            url=url,
            token=token,
            payload=payload,
            expected_status=expected_status,
            timeout=timeout,
        )
        if not text:
            return None
        return json.loads(text)

    def wait_for_json(
        self,
        description: str,
        method: str,
        url: str,
        token: str | None = None,
        payload: Any | None = None,
        expected_status: int = 200,
        predicate: Callable[[Any], bool] | None = None,
        timeout: float = 120.0,
        interval: float = 0.5,
    ) -> Any:
        return self.wait_for(
            description=description,
            supplier=lambda: self.request_json(
                method=method,
                url=url,
                token=token,
                payload=payload,
                expected_status=expected_status,
            ),
            predicate=predicate,
            timeout=timeout,
            interval=interval,
        )

    def wait_for_url(self, description: str, url: str, timeout: float = 120.0, interval: float = 0.5) -> None:
        self.wait_for(
            description=description,
            supplier=lambda: self.request("GET", url, expected_status=200),
            predicate=lambda _: True,
            timeout=timeout,
            interval=interval,
        )
