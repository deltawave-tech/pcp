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
        params_dict = {n: p for n, p in zip(self.names, args[:-2])}
        return functional_call(self.base, params_dict, (args[-2], args[-1]))

wrapper = Wrapper(model, param_names)
full_inputs = param_values + [idx, targets]
ep = torch.export.export(wrapper, tuple(full_inputs), strict=False)
loss_export = ep.module()(*full_inputs)

print("loss_pt:", loss_pt.item())
print("loss_export:", loss_export.item())
print("abs diff:", (loss_pt - loss_export).abs().item())
