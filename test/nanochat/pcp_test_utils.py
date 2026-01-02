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
    backend: str = "cpu",
    target: str | None = None,
    config_path: Path | None = None,
) -> None:
    pcp_bin = find_pcp_binary()
    config_arg = config_path if config_path is not None else (PCP_ROOT / "experiments" / "nanochat.json")
    cmd = [
        pcp_bin,
        "--export-training-artifacts",
        str(out_dir),
        "--model",
        str(model_path),
        "--config",
        str(config_arg),
        "--backend",
        backend,
    ]
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
) -> list[torch.Tensor]:
    iree_run = shutil.which("iree-run-module")
    if not iree_run:
        raise SystemExit(
            "iree-run-module not found. Run `nix develop /home/ahmet/project/nanochat/pcp` "
            "or add the IREE SDK bin directory to PATH."
        )

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
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            "iree-run-module failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    return [load_npy_tensor(path) for path in output_paths]
