"""
Gradient parity test: PyTorch autograd vs PCP autodiff (via training VMFB).

This test:
1) Exports the NanoChat forward MLIR.
2) Uses `pcp --export-training-artifacts` to build the training VMFB + metadata.
3) Runs the gradient function from the VMFB and compares parameter grads to PyTorch.

How to run (from `pcp/`):
  nix develop -c ./venv/bin/python test/nanochat/test_pcp_gradients_parity.py
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from pcp_test_utils import (
    export_training_artifacts,
    run_generate_nanochat,
    run_iree_function,
)
from nanochat.gpt import GPT, GPTConfig


def assert_close(actual, expected, label, atol=1e-4, rtol=1e-4):
    if torch.isnan(actual).any() or torch.isnan(expected).any():
        raise SystemExit(
            f"{label} produced NaNs (actual_nan={torch.isnan(actual).any().item()} "
            f"expected_nan={torch.isnan(expected).any().item()})"
        )
    diff = (actual - expected).abs().max().item()
    ref = expected.abs().max().item()
    tol = atol + rtol * ref
    print(f"{label}: max_diff={diff} ref_max={ref} tol={tol}")
    if diff > tol:
        raise SystemExit(f"{label} mismatch: max_diff={diff} tol={tol}")


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
    param_values = [p.detach().cpu() for p in params.values()]

    return cfg, model, param_values, param_names


def make_batch_inputs(cfg: GPTConfig, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64, generator=g)
    targets = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64, generator=g)
    return idx, targets


def main():
    cfg, model, param_values, param_names = build_model_and_params()
    _ = cfg, param_names  # reserved for future debug output

    idx, targets = make_batch_inputs(cfg, seed=1)

    model.zero_grad(set_to_none=True)
    loss_pt = model(idx, targets)
    loss_pt.backward()

    grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            raise SystemExit(f"Missing gradient for parameter: {name}")
        grads.append(param.grad.detach().cpu())

    with tempfile.TemporaryDirectory(prefix="pcp_grad_parity_") as tmp:
        tmpdir = Path(tmp)
        forward_mlir = tmpdir / "nanochat_forward_32.mlir"
        artifacts_dir = tmpdir / "training_artifacts"

        run_generate_nanochat(forward_mlir)
        export_training_artifacts(artifacts_dir, forward_mlir)

        meta = json.loads((artifacts_dir / "metadata.json").read_text())
        vmfb_path = artifacts_dir / "training.vmfb"

        clip_min = float(meta["adam"]["gradient_clip_min"])
        clip_max = float(meta["adam"]["gradient_clip_max"])
        grad_fn = meta.get("grad_function", "model_forward_pass_grad")

        loss_grad = torch.tensor(1.0, dtype=torch.float32)
        inputs = param_values + [idx.cpu(), targets.cpu(), loss_grad]

        outputs = run_iree_function(
            vmfb_path,
            grad_fn,
            inputs,
            tmpdir,
            output_count=len(param_values) + 2,
        )

    for i, grad_ref in enumerate(grads):
        grad_ref = torch.clamp(grad_ref, min=clip_min, max=clip_max)
        grad_iree = outputs[i]
        assert_close(grad_iree, grad_ref, f"param_{i}_grad")

    print("OK: PCP autodiff gradients match PyTorch (clipped).")


if __name__ == "__main__":
    main()
