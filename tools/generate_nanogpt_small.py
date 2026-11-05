import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx
from torch.func import functional_call
import math

# --- Configuration (GPU-aligned dimensions) ---
BATCH_SIZE = 64
BLOCK_SIZE = 32
VOCAB_SIZE = 65
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.n_head = N_HEAD
        self.n_embd = N_EMBD

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Note: We use manual attention here instead of F.scaled_dot_product_attention
        # because getting the latter to export cleanly to StableHLO via torch_mlir can be tricky
        # without specific flash-attention backend support.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Create a causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x)  # Using ReLU instead of GELU for simpler StableHLO graph
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

print(f"Found {len(param_values)} parameter tensors.")


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

print("Compiling to StableHLO with GPU-aligned dimensions...")
print(f"  BLOCK_SIZE: {BLOCK_SIZE} (matches CUDA warp size)")
print(f"  N_EMBD: {N_EMBD} (head dim = {N_EMBD // N_HEAD})")
module = fx.export_and_import(wrapper, *full_inputs, output_type="stablehlo")

output_path = "models/nanogpt_forward_32.mlir"
with open(output_path, "w") as f:
    f.write(str(module.operation))
print(f"Done. Saved to {output_path}")
