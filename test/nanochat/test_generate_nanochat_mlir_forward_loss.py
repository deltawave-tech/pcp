"""
End-to-end MLIR forward-loss parity test for NanoChat (script-style).

This test validates the whole path:
  PyTorch weights + inputs
    → `tools/generate_nanochat.py` (StableHLO MLIR export)
    → `iree-compile` (VMFB)
    → `iree-run-module` (execute)
and checks that the scalar loss matches eager PyTorch within tolerance.

Notes:
- This is a *forward-only* check; it does not validate gradients/training parity.
- It assumes IREE CLI tools are on PATH (recommended: run under `nix develop`).
- The IREE compile target in this test is `llvm-cpu` (so it does not validate GPU codegen).

How to run (recommended, from `pcp/`):
  nix develop -c ./venv/bin/python test/nanochat/test_generate_nanochat_mlir_forward_loss.py


Results:
Compiling nanochat with layers=2, embd=64, heads=4...
Exported params: 14 tensors
Output: /tmp/nix-shell.2rkevO/nanochat_mlir_8udg9t9g/nanochat_forward_32.mlir
Signature check: Signature check passed.
Warning: MLIR uses ops without VJP coverage:
    stablehlo.and, stablehlo.ceil, stablehlo.compare, stablehlo.dynamic_iota, stablehlo.iota
batch0 loss_pt:   4.3792643547058105
batch0 loss_mlir: 4.379264831542969
batch0 abs diff:  4.76837158203125e-07
batch1 loss_pt:   4.349891185760498
batch1 loss_mlir: 4.349894046783447
batch1 abs diff:  2.86102294921875e-06
OK: generate_nanochat MLIR matches PyTorch loss (2 batches).
"""

import ast
import re
import shutil
import subprocess
import sys
import tempfile
import struct
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


def build_model_and_params():
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

    params = dict(model.named_parameters())
    param_names = list(params.keys())
    param_values = list(params.values())

    param_values = [p.detach().cpu() for p in param_values]

    return cfg, model, param_values, param_names


def make_batch_inputs(cfg: GPTConfig, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64, generator=g)
    targets = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64, generator=g)
    return idx, targets


def parse_main_signature(mlir_text: str):
    match = re.search(r"func\.func @main\(([^)]*)\) -> ([^{\n]+)", mlir_text)
    if not match:
        raise SystemExit("Could not find func.func @main signature in exported MLIR.")

    args_str = match.group(1).strip()
    ret_str = match.group(2).strip()
    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

    def arg_type(arg: str) -> str:
        return arg.split(":", 1)[1].strip() if ":" in arg else arg.strip()

    return [arg_type(a) for a in args], ret_str


def parse_tensor_type(type_str: str):
    type_str = type_str.strip()
    if not type_str.startswith("tensor<") or not type_str.endswith(">"):
        return None
    inner = type_str[len("tensor<") : -1]
    parts = inner.split("x")
    if not parts:
        return None
    elem = parts[-1]
    dims = parts[:-1]
    return dims, elem


def assert_signature_matches_inputs(mlir_path: Path, inputs):
    mlir_text = mlir_path.read_text()
    arg_types, ret_type = parse_main_signature(mlir_text)

    if len(arg_types) != len(inputs):
        raise SystemExit(
            f"MLIR @main arg count mismatch: mlir={len(arg_types)} inputs={len(inputs)}"
        )

    if "f32" not in ret_type:
        raise SystemExit(f"Unexpected return type (expected f32): {ret_type}")

    for i, tensor in enumerate(inputs):
        ty = arg_types[i]
        parsed = parse_tensor_type(ty)
        if parsed is None:
            raise SystemExit(f"Unsupported arg type syntax at arg {i}: {ty}")
        dims, elem = parsed

        expected_elem = "f32" if tensor.dtype == torch.float32 else "i64"
        if expected_elem != elem:
            raise SystemExit(
                f"Arg {i} dtype mismatch: mlir={elem} expected={expected_elem}"
            )

        is_idx_or_targets = i >= (len(inputs) - 2)
        dims_are_unranked = dims == ["*"]

        if is_idx_or_targets:
            if dims_are_unranked or len(dims) != tensor.dim():
                raise SystemExit(
                    f"Arg {i} rank mismatch: mlir={len(dims)} expected={tensor.dim()} ({ty})"
                )
            for d_mlir, d_expected in zip(dims, tensor.shape):
                if not d_mlir.isdigit() or int(d_mlir) != int(d_expected):
                    raise SystemExit(
                        f"Arg {i} shape mismatch: mlir_dim={d_mlir} expected_dim={int(d_expected)} ({ty})"
                    )
        else:
            if not dims_are_unranked and len(dims) != tensor.dim():
                raise SystemExit(
                    f"Arg {i} rank mismatch: mlir={len(dims)} expected={tensor.dim()} ({ty})"
                )
            if not dims_are_unranked:
                for d_mlir, d_expected in zip(dims, tensor.shape):
                    if d_mlir.isdigit() and int(d_mlir) != int(d_expected):
                        raise SystemExit(
                            f"Arg {i} shape mismatch: mlir_dim={d_mlir} expected_dim={int(d_expected)} ({ty})"
                        )


def run_generate_nanochat(output_path):
    script = NANOCHAT_ROOT / "pcp" / "tools" / "generate_nanochat.py"
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
    cfg, model, param_values, param_names = build_model_and_params()
    _ = cfg, param_names  # silence unused warnings for future debugging

    with tempfile.TemporaryDirectory(prefix="nanochat_mlir_") as tmp:
        tmpdir = Path(tmp)
        mlir_path = tmpdir / "nanochat_forward_32.mlir"
        vmfb_path = tmpdir / "nanochat_forward_32.vmfb"

        run_generate_nanochat(mlir_path)
        idx0, targets0 = make_batch_inputs(cfg, seed=1)
        inputs0 = param_values + [idx0.cpu(), targets0.cpu()]
        assert_signature_matches_inputs(mlir_path, inputs0)
        compile_with_iree(mlir_path, vmfb_path)

        idx1, targets1 = make_batch_inputs(cfg, seed=2)
        inputs1 = param_values + [idx1.cpu(), targets1.cpu()]

        loss_pt0 = model(idx0, targets0).item()
        loss_mlir0 = run_iree(vmfb_path, inputs0, tmpdir)
        loss_pt1 = model(idx1, targets1).item()
        loss_mlir1 = run_iree(vmfb_path, inputs1, tmpdir)

    tol = 1e-4
    for label, loss_pt, loss_mlir in [
        ("batch0", loss_pt0, loss_mlir0),
        ("batch1", loss_pt1, loss_mlir1),
    ]:
        diff = abs(loss_pt - loss_mlir)
        print(f"{label} loss_pt:   {loss_pt}")
        print(f"{label} loss_mlir: {loss_mlir}")
        print(f"{label} abs diff:  {diff}")
        if diff > tol:
            raise SystemExit(f"MLIR mismatch ({label}): diff={diff} tol={tol}")

    print("OK: generate_nanochat MLIR matches PyTorch loss (2 batches).")


if __name__ == "__main__":
    main()
