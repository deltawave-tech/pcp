import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx
from torch.func import functional_call
import math

# --- Configuration (Scaled Up) ---
BATCH_SIZE = 64
BLOCK_SIZE = 128
VOCAB_SIZE = 256
N_EMBD = 256
N_HEAD = 8
N_LAYER = 4
DROPOUT = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.n_head = N_HEAD
        self.n_embd = N_EMBD

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # Flatten for cross_entropy
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return loss


# --- Setup and Export ---
model = NanoTransformer()
model.train()

idx = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)
targets = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)

params_dict = dict(model.named_parameters())
param_names = list(params_dict.keys())
param_values = list(params_dict.values())


class StatelessWrapper(nn.Module):
    def __init__(self, base_model, param_names):
        super().__init__()
        self.base_model = base_model
        self.param_names = param_names

    def forward(self, *args):
        params = args[:-2]
        idx = args[-2]
        targets = args[-1]
        params_dict = {name: param for name, param in zip(self.param_names, params)}
        return functional_call(self.base_model, params_dict, (idx, targets))


wrapper = StatelessWrapper(model, param_names)
full_inputs = param_values + [idx, targets]

print(f"Compiling Medium NanoGPT (Layers: {N_LAYER}, Embd: {N_EMBD})...")
module = fx.export_and_import(wrapper, *full_inputs, output_type="stablehlo")

output_path = "models/nanogpt_medium.mlir"
with open(output_path, "w") as f:
    f.write(str(module.operation))
print(f"Saved to {output_path}")
