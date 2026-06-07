"""
DiLoCo outer-step parity test (multi-worker, deterministic, no dataset).

This test simulates K workers running an inner optimizer (AdamW) for H steps,
computes the averaged delta (outer gradient), then applies the host-side
Nesterov update exactly like `src/optimizers/nesterov.zig:64`.

It compares:
  - PCP training VMFB inner steps (PCP exact compiled math), vs
  - a PyTorch reference implementation of the same inner-step math,
and asserts master parameters match after 1–3 outer rounds.

Key property: worker AdamW states (m/v + timestep) persist across outer rounds,
while parameters are re-synced from master at the start of each round
(matches `src/nodes/workers/worker.zig:518`).

How to run (from `pcp/`):
  nix develop -c ./venv/bin/python test/nanochat/test_pcp_diloco_outer_step_parity.py
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

from pcp_test_utils import export_training_artifacts, run_generate_nanochat, run_iree_function
from nanochat.gpt import GPT, GPTConfig


def assert_close(actual, expected, label, atol=5e-4, rtol=5e-4):
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
    return cfg, model, param_names, param_values


def make_batch(cfg: GPTConfig, seed: int):
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


def pcp_step(vmfb_path: Path, main_fn: str, params, m_states, v_states, timestep, idx, targets, tmpdir: Path):
    timestep_tensor = torch.tensor(float(timestep), dtype=torch.float32)
    inputs = params + m_states + v_states + [timestep_tensor, idx.cpu(), targets.cpu()]
    outputs = run_iree_function(
        vmfb_path,
        main_fn,
        inputs,
        tmpdir,
        output_count=len(params) * 3 + 1,
    )

    n = len(params)
    out_params = outputs[0:n]
    out_m = outputs[n : 2 * n]
    out_v = outputs[2 * n : 3 * n]
    out_loss = float(outputs[3 * n].item())
    return out_loss, out_params, out_m, out_v


def nesterov_update(params, velocities, outer_grads, nesterov_conf):
    lr = float(nesterov_conf["learning_rate"])
    mu = float(nesterov_conf["momentum"])

    new_params = []
    new_vel = []
    for p, v, g in zip(params, velocities, outer_grads):
        v_new = mu * v + g
        p_new = p - lr * (g + mu * v_new)
        new_params.append(p_new.detach().cpu())
        new_vel.append(v_new.detach().cpu())
    return new_params, new_vel


def avg_deltas(deltas_per_worker):
    num_workers = len(deltas_per_worker)
    num_params = len(deltas_per_worker[0])
    out = []
    for i in range(num_params):
        stacked = torch.stack([deltas_per_worker[w][i] for w in range(num_workers)], dim=0)
        out.append(stacked.mean(dim=0))
    return out


def main():
    cfg, model, param_names, param_values = build_model_and_params()

    num_workers = 3
    tau = 2
    outer_rounds = 2

    with tempfile.TemporaryDirectory(prefix="pcp_outer_parity_") as tmp:
        tmpdir = Path(tmp)
        forward_mlir = tmpdir / "nanochat_forward_32.mlir"
        artifacts_dir = tmpdir / "training_artifacts"

        run_generate_nanochat(forward_mlir)
        export_training_artifacts(artifacts_dir, forward_mlir)

        meta = json.loads((artifacts_dir / "metadata.json").read_text())
        vmfb_path = artifacts_dir / "training.vmfb"
        main_fn = meta.get("main_function", "main")

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
        nesterov_conf = {
            "learning_rate": float(meta["nesterov"]["learning_rate"]),
            "momentum": float(meta["nesterov"]["momentum"]),
        }
        timestep_start = float(meta.get("timestep_start", 1.0))

        master_pcp = [p.detach().cpu() for p in param_values]
        master_ref = [p.detach().cpu() for p in param_values]

        outer_vel_pcp = [torch.zeros_like(p) for p in master_pcp]
        outer_vel_ref = [torch.zeros_like(p) for p in master_ref]

        worker_state_pcp = [
            {
                "m": [torch.zeros_like(p) for p in master_pcp],
                "v": [torch.zeros_like(p) for p in master_pcp],
                "t": timestep_start,
            }
            for _ in range(num_workers)
        ]
        worker_state_ref = [
            {
                "m": [torch.zeros_like(p) for p in master_ref],
                "v": [torch.zeros_like(p) for p in master_ref],
                "t": timestep_start,
            }
            for _ in range(num_workers)
        ]

        for outer in range(outer_rounds):
            print(f"\n=== outer_round {outer + 1}/{outer_rounds} ===")
            master_snapshot_pcp = [p.clone() for p in master_pcp]
            master_snapshot_ref = [p.clone() for p in master_ref]

            deltas_pcp = []
            deltas_ref = []
            losses_pcp = []
            losses_ref = []

            for w in range(num_workers):
                params_pcp = [p.clone() for p in master_snapshot_pcp]
                m_pcp = [t.clone() for t in worker_state_pcp[w]["m"]]
                v_pcp = [t.clone() for t in worker_state_pcp[w]["v"]]
                t_pcp = float(worker_state_pcp[w]["t"])

                params_ref = [p.clone() for p in master_snapshot_ref]
                m_ref = [t.clone() for t in worker_state_ref[w]["m"]]
                v_ref = [t.clone() for t in worker_state_ref[w]["v"]]
                t_ref = float(worker_state_ref[w]["t"])

                for inner in range(tau):
                    seed = 10_000 * (outer + 1) + 100 * (w + 1) + (inner + 1)
                    idx, targets = make_batch(cfg, seed)

                    loss_p, params_pcp, m_pcp, v_pcp = pcp_step(
                        vmfb_path,
                        main_fn,
                        params_pcp,
                        m_pcp,
                        v_pcp,
                        t_pcp,
                        idx,
                        targets,
                        tmpdir,
                    )
                    loss_r, params_ref, m_ref, v_ref = reference_step(
                        model,
                        param_names,
                        params_ref,
                        m_ref,
                        v_ref,
                        t_ref,
                        idx,
                        targets,
                        adam_conf,
                    )

                    if abs(loss_p - loss_r) > 5e-4:
                        raise SystemExit(
                            f"loss mismatch outer={outer} worker={w} inner={inner}: ref={loss_r} pcp={loss_p}"
                        )
                    t_pcp += 1.0
                    t_ref += 1.0

                # Persist m/v/timestep across outer rounds (params re-sync each round)
                worker_state_pcp[w]["m"] = [t.detach().cpu() for t in m_pcp]
                worker_state_pcp[w]["v"] = [t.detach().cpu() for t in v_pcp]
                worker_state_pcp[w]["t"] = t_pcp

                worker_state_ref[w]["m"] = [t.detach().cpu() for t in m_ref]
                worker_state_ref[w]["v"] = [t.detach().cpu() for t in v_ref]
                worker_state_ref[w]["t"] = t_ref

                delta_pcp = [a - b for a, b in zip(master_snapshot_pcp, params_pcp)]
                delta_ref = [a - b for a, b in zip(master_snapshot_ref, params_ref)]
                deltas_pcp.append(delta_pcp)
                deltas_ref.append(delta_ref)
                losses_pcp.append(loss_p)
                losses_ref.append(loss_r)

            delta_avg_pcp = avg_deltas(deltas_pcp)
            delta_avg_ref = avg_deltas(deltas_ref)

            for i in range(len(delta_avg_pcp)):
                assert_close(delta_avg_pcp[i], delta_avg_ref[i], f"outer{outer+1}_delta_{i}")

            master_pcp, outer_vel_pcp = nesterov_update(master_pcp, outer_vel_pcp, delta_avg_pcp, nesterov_conf)
            master_ref, outer_vel_ref = nesterov_update(master_ref, outer_vel_ref, delta_avg_ref, nesterov_conf)

            for i in range(len(master_pcp)):
                assert_close(master_pcp[i], master_ref[i], f"outer{outer+1}_master_param_{i}")
                assert_close(outer_vel_pcp[i], outer_vel_ref[i], f"outer{outer+1}_velocity_{i}")

            print(
                "avg loss (pcp/ref): "
                f"{sum(losses_pcp)/len(losses_pcp):.6f} / {sum(losses_ref)/len(losses_ref):.6f}"
            )

    print("OK: DiLoCo outer-step parity matches (multi-worker synthetic).")


if __name__ == "__main__":
    main()
