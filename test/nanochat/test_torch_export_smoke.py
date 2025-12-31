"""
Smoke test for `torch.export` compatibility (script-style).

This checks that NanoChat's `GPT` forward+loss can be captured by `torch.export`
and executed, producing the same scalar loss as eager PyTorch.

This is an early warning signal: if `torch.export` can’t capture the model,
downstream torch-mlir export is likely to fail or become brittle.

How to run (recommended, from `pcp/`):
  nix develop -c ./venv/bin/python test/nanochat/test_torch_export_smoke.py

Results:
loss_pt: 4.358511924743652
rloss_export: 4.358511924743652
abs diff: 0.0
OK: torch.export matches eager loss.
"""

from typing import Any


import sys
from pathlib import Path

import torch
from torch.func import functional_call

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

torch.manual_seed(0)
cfg = GPTConfig(sequence_len=32, vocab_size=65, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)
model = GPT(cfg).eval()

idx = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64)
targets = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64)

loss_pt = model(idx, targets)

params = dict(model.named_parameters())
param_names = list(params.keys())
param_values = list(params.values())

class Wrapper(torch.nn.Module):
    def __init__(self, base, names):
        super().__init__()
        self.base = base
        self.names = names
    def forward(self, *args):
        params_dict = {n: p for n, p in zip[tuple](self.names, args[:-2])}
        return functional_call(self.base, params_dict, (args[-2], args[-1]))

wrapper = Wrapper(model, param_names)
full_inputs = param_values + [idx, targets]
ep = torch.export.export(wrapper, tuple(full_inputs), strict=False)
loss_export = ep.module()(*full_inputs)

diff = (loss_pt - loss_export).abs().item()
tol = 1e-5

print("loss_pt:", loss_pt.item())
print("loss_export:", loss_export.item())
print("abs diff:", diff)

if diff > tol:
    raise SystemExit(f"torch.export mismatch: diff={diff} tol={tol}")

print("OK: torch.export matches eager loss.")
