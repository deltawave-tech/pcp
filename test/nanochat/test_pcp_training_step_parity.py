"""
Training-step parity test: PyTorch reference vs PCP training VMFB.

This test compares a couple of full AdamW update steps (params + m + v + loss)
against the training graph produced by PCP.

How to run (from `pcp/`):
  nix develop -c ./venv/bin/python test/nanochat/test_pcp_training_step_parity.py
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
from torch.func import functional_call

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from nanochat.gpt import GPT, GPTConfig
from pcp_test_utils import (
    export_training_artifacts,
    run_generate_nanochat,
    run_iree_function,
)


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


def adamw_update(param, grad, m, v, timestep, conf):
    clip_min = conf["gradient_clip_min"]
    clip_max = conf["gradient_clip_max"]
    max_grad_norm = conf["max_grad_norm"]
    beta1 = conf["beta1"]
    beta2 = conf["beta2"]
    lr = conf["learning_rate"]
    eps = conf["epsilon"]
    weight_decay = conf["weight_decay"]

    grad = torch.clamp(grad, min=clip_min, max=clip_max)

    max_norm = grad.new_tensor(max_grad_norm)
    norm = grad.pow(2).sum().sqrt() + eps
    max_val = torch.maximum(norm, max_norm)
    clip_scale = max_norm / max_val
    grad = grad * clip_scale

    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

    m_hat = new_m / (1.0 - (beta1**timestep))
    v_hat = new_v / (1.0 - (beta2**timestep))

    ratio = m_hat / (torch.sqrt(v_hat) + eps)
    decay_term = weight_decay * param
    new_param = param - lr * (ratio + decay_term)

    return new_param, new_m, new_v


def reference_step(model, param_names, params, m_states, v_states, timestep, idx, targets, conf):
    params_req = [p.detach().clone().requires_grad_(True) for p in params]
    params_dict = {name: p for name, p in zip(param_names, params_req)}

    loss = functional_call(model, params_dict, (idx, targets))
    grads = torch.autograd.grad(loss, params_req, create_graph=False)

    new_params = []
    new_m = []
    new_v = []
    for p, g, m, v in zip(params, grads, m_states, v_states):
        p_next, m_next, v_next = adamw_update(p, g, m, v, timestep, conf)
        new_params.append(p_next.detach().cpu())
        new_m.append(m_next.detach().cpu())
        new_v.append(v_next.detach().cpu())

    return loss.item(), new_params, new_m, new_v


def main():
    cfg, model, param_values, param_names = build_model_and_params()

    with tempfile.TemporaryDirectory(prefix="pcp_train_parity_") as tmp:
        tmpdir = Path(tmp)
        forward_mlir = tmpdir / "nanochat_forward_32.mlir"
        artifacts_dir = tmpdir / "training_artifacts"

        run_generate_nanochat(forward_mlir)
        export_training_artifacts(artifacts_dir, forward_mlir)

        meta = json.loads((artifacts_dir / "metadata.json").read_text())
        vmfb_path = artifacts_dir / "training.vmfb"

        adam_conf = {
            "learning_rate": float(meta["adam"]["learning_rate"]),
            "beta1": float(meta["adam"]["beta1"]),
            "beta2": float(meta["adam"]["beta2"]),
            "epsilon": float(meta["adam"]["epsilon"]),
            "weight_decay": float(meta["adam"]["weight_decay"]),
            "max_grad_norm": float(meta["adam"]["max_grad_norm"]),
            "gradient_clip_min": float(meta["adam"]["gradient_clip_min"]),
            "gradient_clip_max": float(meta["adam"]["gradient_clip_max"]),
        }

        params = [p.detach().cpu() for p in param_values]
        m_states = [torch.zeros_like(p) for p in params]
        v_states = [torch.zeros_like(p) for p in params]
        timestep = float(meta.get("timestep_start", 1.0))

        for step, seed in enumerate([1, 2], start=1):
            idx, targets = make_batch_inputs(cfg, seed=seed)

            loss_ref, params_ref, m_ref, v_ref = reference_step(
                model,
                param_names,
                params,
                m_states,
                v_states,
                timestep,
                idx,
                targets,
                adam_conf,
            )

            timestep_tensor = torch.tensor(timestep, dtype=torch.float32)
            inputs = (
                params
                + m_states
                + v_states
                + [timestep_tensor, idx.cpu(), targets.cpu()]
            )

            outputs = run_iree_function(
                vmfb_path,
                meta.get("main_function", "main"),
                inputs,
                tmpdir,
                output_count=len(params) * 3 + 1,
            )

            num_params = len(params)
            out_params = outputs[0:num_params]
            out_m = outputs[num_params : 2 * num_params]
            out_v = outputs[2 * num_params : 3 * num_params]
            out_loss = outputs[3 * num_params].item()

            print(f"step {step} loss_ref={loss_ref} loss_pcp={out_loss}")
            if abs(out_loss - loss_ref) > 1e-4:
                raise SystemExit(
                    f"loss mismatch at step {step}: ref={loss_ref} pcp={out_loss}"
                )

            for i in range(num_params):
                assert_close(out_params[i], params_ref[i], f"step{step}_param_{i}")
                assert_close(out_m[i], m_ref[i], f"step{step}_m_{i}")
                assert_close(out_v[i], v_ref[i], f"step{step}_v_{i}")

            params = params_ref
            m_states = m_ref
            v_states = v_ref
            timestep += 1.0

    print("OK: PCP training step matches PyTorch reference (2 steps).")


if __name__ == "__main__":
    main()
