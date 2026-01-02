import ast
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import torch

NANOCHAT_ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "nanochat" / "gpt.py").exists():
        NANOCHAT_ROOT = parent
        break
if NANOCHAT_ROOT is None:
    raise SystemExit("Could not locate nanochat repo root (expected nanochat/gpt.py).")
if str(NANOCHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_ROOT))

PCP_ROOT = NANOCHAT_ROOT / "pcp"

_DTYPE_TO_NPY_DESCR = {
    torch.float32: "<f4",
    torch.int64: "<i8",
}
_NPY_DESCR_TO_DTYPE = {
    "<f4": torch.float32,
    "<i8": torch.int64,
}

def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _append_env_path(env: dict[str, str], var: str, paths: list[str]) -> None:
    existing = env.get(var, "")
    existing_parts = [p for p in existing.split(os.pathsep) if p] if existing else []
    merged = _dedupe_keep_order(existing_parts + [p for p in paths if p])
    env[var] = os.pathsep.join(merged)


def _maybe_add_cuda_driver_to_ld_library_path(env: dict[str, str]) -> None:
    override = env.get("PCP_TEST_CUDA_LIBRARY_PATH") or env.get("CUDA_DRIVER_LIBRARY_PATH")
    override_paths = [p for p in (override.split(os.pathsep) if override else []) if p]

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
        # Prefer the directory containing the *real* library to avoid adding broad
        # system library roots (can break Nix binaries by shadowing glibc).
        lib_dir = resolved.parent
        if (lib_dir / "libc.so.6").exists() or (lib_dir / "libm.so.6").exists():
            continue
        driver_dirs.append(str(lib_dir))

    if driver_dirs:
        _append_env_path(env, "LD_LIBRARY_PATH", _dedupe_keep_order(driver_dirs))


def find_pcp_binary() -> str:
    override = os.environ.get("PCP_BIN")
    if override:
        path = Path(override)
        if path.exists():
            return str(path)
        raise SystemExit(f"PCP_BIN is set but does not exist: {override}")

    candidates = [
        PCP_ROOT / "zig-out" / "bin" / "pcp",
        PCP_ROOT / "result" / "bin" / "pcp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which("pcp")
    if found:
        return found
    raise SystemExit("pcp binary not found (checked result/bin/pcp, zig-out/bin/pcp, PATH).")


def run_generate_nanochat(output_path: Path) -> None:
    script = PCP_ROOT / "tools" / "generate_nanochat.py"
    cmd = [
        sys.executable,
        str(script),
        "--batch-size",
        "64",
        "--block-size",
        "32",
        "--vocab-size",
        "65",
        "--n-layer",
        "2",
        "--n-head",
        "4",
        "--n-kv-head",
        "4",
        "--n-embd",
        "64",
        "--out",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=PCP_ROOT)


def export_training_artifacts(
    out_dir: Path,
    model_path: Path,
    backend: str | None = None,
    target: str | None = None,
    config_path: Path | None = None,
) -> None:
    pcp_bin = find_pcp_binary()
    config_arg = config_path if config_path is not None else (PCP_ROOT / "experiments" / "nanochat.json")
    if backend is None:
        backend = os.environ.get("PCP_TEST_BACKEND", "cpu")
    if target is None:
        target = os.environ.get("PCP_TEST_TARGET") or None
    cmd = [
        pcp_bin,
        "--export-training-artifacts",
        str(out_dir),
        "--model",
        str(model_path),
        "--config",
        str(config_arg),
    ]
    if backend != "auto":
        cmd.extend(["--backend", backend])
    if target:
        cmd.extend(["--target", target])
    subprocess.run(cmd, check=True, cwd=PCP_ROOT)


def write_npy_from_tensor(path: Path, tensor: torch.Tensor) -> None:
    if sys.byteorder != "little":
        raise SystemExit("This test only supports little-endian hosts.")

    tensor = tensor.detach().cpu().contiguous()
    descr = _DTYPE_TO_NPY_DESCR.get(tensor.dtype)
    if descr is None:
        raise SystemExit(f"Unsupported dtype for .npy export: {tensor.dtype}")

    shape = tuple(int(d) for d in tensor.shape)
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"

    magic = b"\x93NUMPY"
    version = b"\x01\x00"  # v1.0 uses uint16 header length

    header_bytes = header.encode("ascii")
    pad_len = (16 - ((len(magic) + len(version) + 2 + len(header_bytes) + 1) % 16)) % 16
    header_bytes = header_bytes + (b" " * pad_len) + b"\n"

    storage_bytes = bytes(tensor.untyped_storage())
    offset = tensor.storage_offset() * tensor.element_size()
    nbytes = tensor.numel() * tensor.element_size()
    payload = storage_bytes[offset : offset + nbytes]

    with path.open("wb") as f:
        f.write(magic)
        f.write(version)
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def load_npy_tensor(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise SystemExit("Invalid .npy file (bad magic).")
        version = f.read(2)
        if version != b"\x01\x00":
            raise SystemExit(f"Unsupported .npy version: {version!r}")
        header_len = struct.unpack("<H", f.read(2))[0]
        header = f.read(header_len).decode("ascii")
        header_dict = ast.literal_eval(header)

        descr = header_dict.get("descr")
        dtype = _NPY_DESCR_TO_DTYPE.get(descr)
        if dtype is None:
            raise SystemExit(f"Unsupported .npy dtype: {descr}")

        shape = header_dict.get("shape", ())
        count = 1
        for dim in shape:
            count *= int(dim)

        itemsize = 4 if dtype == torch.float32 else 8
        data = f.read(count * itemsize)
        if len(data) != count * itemsize:
            raise SystemExit("Truncated .npy data.")

    flat = torch.frombuffer(memoryview(data), dtype=dtype).clone()
    return flat.reshape(shape)


def run_iree_function(
    vmfb_path: Path,
    function_name: str,
    inputs: list[torch.Tensor],
    tmpdir: Path,
    output_count: int,
    device: str | None = None,
) -> list[torch.Tensor]:
    iree_run = shutil.which("iree-run-module")
    if not iree_run:
        raise SystemExit(
            "iree-run-module not found. Run `nix develop /home/ahmet/project/nanochat/pcp` "
            "or add the IREE SDK bin directory to PATH."
        )

    env = os.environ.copy()

    input_flags = []
    for i, tensor in enumerate(inputs):
        npy_path = tmpdir / f"input_{i:03d}.npy"
        write_npy_from_tensor(npy_path, tensor)
        input_flags.append(f"--input=@{npy_path}")

    output_flags = []
    output_paths: list[Path] = []
    for i in range(output_count):
        out_path = tmpdir / f"output_{i:03d}.npy"
        output_paths.append(out_path)
        output_flags.append(f"--output=@{out_path}")

    cmd = [
        iree_run,
        f"--module={vmfb_path}",
        f"--function={function_name}",
        *input_flags,
        *output_flags,
    ]
    if device is None:
        device = os.environ.get("PCP_TEST_IREE_DEVICE") or os.environ.get("IREE_DEVICE")
    if device is None:
        backend = os.environ.get("PCP_TEST_BACKEND")
        if backend in {"cuda", "rocm", "vulkan", "metal"}:
            device = backend
    if device:
        cmd.append(f"--device={device}")

    if device and device.startswith("cuda"):
        _maybe_add_cuda_driver_to_ld_library_path(env)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        hints: list[str] = []
        stderr = result.stderr or ""
        if "CUDA driver library 'libcuda.so'" in stderr or "CUDA driver library \"libcuda.so\"" in stderr:
            hints.append(
                "Hint: libcuda.so not found. Set PCP_TEST_CUDA_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current "
                "or otherwise ensure that directory is in LD_LIBRARY_PATH."
            )
        if "CUDA_ERROR_UNSUPPORTED_PTX_VERSION" in stderr or "Unsupported .version" in stderr:
            hints.append(
                "Hint: your NVIDIA driver can't JIT the PTX version produced by IREE. "
                "Try exporting PCP_IREE_CUDA_TARGET_FEATURES=+ptx74 (for 470.x drivers) when running the test "
                "so `pcp --export-training-artifacts` compiles compatible VMFBs."
            )
        raise SystemExit(
            "iree-run-module failed.\n"
            f"cmd:\n{' '.join(cmd)}\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            + ("" if not hints else ("\n\n" + "\n".join(hints)))
        )

    return [load_npy_tensor(path) for path in output_paths]
