import ast
import re
import shutil
import subprocess
import sys
import tempfile
import struct
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.gpt import GPT, GPTConfig


_DTYPE_TO_NPY_DESCR = {
    torch.float32: "<f4",
    torch.int64: "<i8",
}


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


def load_npy_scalar_f32(path: Path) -> float:
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

        if header_dict.get("descr") != "<f4":
            raise SystemExit(f"Unexpected output dtype: {header_dict.get('descr')}")

        shape = header_dict.get("shape", ())
        count = 1
        for dim in shape:
            count *= int(dim)
        if count != 1:
            raise SystemExit(f"Expected scalar output, got shape={shape}")

        data = f.read(4)
        if len(data) != 4:
            raise SystemExit("Truncated output .npy file.")
        return struct.unpack("<f", data)[0]


def build_model_and_inputs():
    torch.manual_seed(0)
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=65,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
    )
    model = GPT(cfg).eval()

    idx = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64)
    targets = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64)

    loss_pt = model(idx, targets)

    params = dict(model.named_parameters())
    param_names = list(params.keys())
    param_values = list(params.values())

    inputs = [p.detach().cpu() for p in param_values]
    inputs.append(idx.cpu())
    inputs.append(targets.cpu())

    return cfg, loss_pt.item(), inputs, param_names


def run_generate_nanochat(output_path):
    script = REPO_ROOT / "pcp" / "tools" / "generate_nanochat.py"
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
    subprocess.run(cmd, check=True)


def compile_with_iree(mlir_path, vmfb_path):
    iree_compile = shutil.which("iree-compile")
    if not iree_compile:
        raise SystemExit(
            "iree-compile not found. Run `nix develop /home/ahmet/project/nanochat/pcp` "
            "or add the IREE SDK bin directory to PATH."
        )

    cmd = [
        iree_compile,
        str(mlir_path),
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=generic",
        "--iree-vm-target-index-bits=64",
        "--iree-stream-resource-index-bits=64",
        "--iree-input-demote-i64-to-i32=false",
        "--iree-llvmgpu-enable-prefetch=false",
        "-o",
        str(vmfb_path),
    ]
    subprocess.run(cmd, check=True)


def run_iree(vmfb_path, inputs, tmpdir):
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

    output_path = tmpdir / "output.npy"
    cmd = [
        iree_run,
        f"--module={vmfb_path}",
        "--function=main",
        *input_flags,
        f"--output=@{output_path}",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            "iree-run-module failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    if output_path.exists():
        return load_npy_scalar_f32(output_path)

    match = re.search(r"result\\[0\\].*?([-+]?\\d*\\.\\d+(?:[eE][-+]?\\d+)?)", result.stdout)
    if not match:
        raise SystemExit(
            "Unable to parse output from iree-run-module. "
            "Try rerunning with --output=@file.npy support."
        )
    return float(match.group(1))


def main():
    cfg, loss_pt, inputs, param_names = build_model_and_inputs()
    _ = cfg, param_names  # silence unused warnings for future debugging

    with tempfile.TemporaryDirectory(prefix="nanochat_mlir_") as tmp:
        tmpdir = Path(tmp)
        mlir_path = tmpdir / "nanochat_forward_32.mlir"
        vmfb_path = tmpdir / "nanochat_forward_32.vmfb"

        run_generate_nanochat(mlir_path)
        compile_with_iree(mlir_path, vmfb_path)
        loss_mlir = run_iree(vmfb_path, inputs, tmpdir)

    diff = abs(loss_pt - loss_mlir)
    tol = 1e-4
    print(f"loss_pt:   {loss_pt}")
    print(f"loss_mlir: {loss_mlir}")
    print(f"abs diff:  {diff}")
    if diff > tol:
        raise SystemExit(f"MLIR mismatch: diff={diff} tol={tol}")
    print("OK: generate_nanochat MLIR matches PyTorch loss.")


if __name__ == "__main__":
    main()
