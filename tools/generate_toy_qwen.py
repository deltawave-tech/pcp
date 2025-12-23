import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx
from torch.func import functional_call
import math

# --- Configuration (Mini Qwen-like Model) ---
BATCH_SIZE = 8
BLOCK_SIZE = 64
VOCAB_SIZE = 256
N_EMBD = 128
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD
N_LAYER = 2
# SwiGLU usually uses a hidden dimension of 8/3 * N_EMBD
# We use 4 * N_EMBD for simplicity in this test
INTERMEDIATE_SIZE = N_EMBD * 4


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    cos = cos.view(1, x.shape[1], 1, -1)
    sin = sin.view(1, x.shape[1], 1, -1)
    out1 = (x1 * cos) - (x2 * sin)
    out2 = (x2 * cos) + (x1 * sin)
    return torch.cat((out1, out2), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x * rsqrt(mean(x^2) + eps) * weight
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLUMLP(nn.Module):
    """
    The Qwen MLP block.
    Uses SiLU in a gated linear unit: (Linear1(x) * SiLU(Linear2(x))) -> Linear3
    """

    def __init__(self, n_embd, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(n_embd, intermediate_size, bias=False)  # Gate projection
        self.w2 = nn.Linear(n_embd, intermediate_size, bias=False)  # Up projection
        self.w3 = nn.Linear(intermediate_size, n_embd, bias=False)  # Down projection

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class QwenSelfAttention(nn.Module):
    def __init__(self, cos, sin):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.n_head = N_HEAD
        self.n_embd = N_EMBD
        self.head_dim = HEAD_DIM
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, self.cos[:T], self.sin[:T]).transpose(1, 2)
        k = apply_rotary_emb(k, self.cos[:T], self.sin[:T]).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class QwenBlock(nn.Module):
    def __init__(self, cos, sin):
        super().__init__()
        self.ln1 = RMSNorm(N_EMBD)
        self.attn = QwenSelfAttention(cos, sin)
        self.ln2 = RMSNorm(N_EMBD)
        self.mlp = SwiGLUMLP(N_EMBD, INTERMEDIATE_SIZE)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class QwenMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        cos, sin = precompute_freqs_cis(HEAD_DIM, BLOCK_SIZE)
        self.blocks = nn.Sequential(*[QwenBlock(cos, sin) for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBD)
        # Note: lm_head removed - we'll use token_embedding.weight directly for output

    def forward(self, idx, targets, embedding_weight):
        # Use provided embedding weight for both embedding lookup and output projection
        x = F.embedding(idx, embedding_weight)
        x = self.blocks(x)
        x = self.ln_f(x)
        # Use same weight for output projection (tied weights)
        logits = F.linear(x, embedding_weight)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return loss


model = QwenMini()
model.train()

idx = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)
targets = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)

# Collect parameters, deduplicating tied weights by parameter ID
params_dict = dict(model.named_parameters())
buffers_dict = dict(model.named_buffers())

# Deduplicate parameters that share the same tensor (tied weights)
seen_params = {}
unique_param_names = []
unique_param_values = []
for name, param in params_dict.items():
    param_id = id(param)
    if param_id not in seen_params:
        seen_params[param_id] = name
        unique_param_names.append(name)
        unique_param_values.append(param)

param_names = unique_param_names
param_values = unique_param_values
buffer_values = list(buffers_dict.values())

print(f"Total parameters: {len(params_dict)}, Unique parameters (after tying): {len(unique_param_names)}")


class StatelessWrapper(nn.Module):
    def __init__(self, base_model, param_names, buffer_names, embedding_weight_idx):
        super().__init__()
        self.base_model = base_model
        self.param_names = param_names
        self.buffer_names = buffer_names
        self.embedding_weight_idx = embedding_weight_idx

    def forward(self, *args):
        n_p = len(self.param_names)
        n_b = len(self.buffer_names)
        params = args[:n_p]
        buffers = args[n_p : n_p + n_b]
        idx = args[-2]
        targets = args[-1]

        # Get the embedding weight to pass explicitly for tying
        embedding_weight = params[self.embedding_weight_idx]

        # Build state dict (excluding embedding since we pass it explicitly)
        p_dict = {name: p for name, p in zip(self.param_names, params)}
        b_dict = {name: b for name, b in zip(self.buffer_names, buffers)}
        state_dict = {**p_dict, **b_dict}
        return functional_call(self.base_model, state_dict, (idx, targets, embedding_weight))


# Find the index of token_embedding.weight
embedding_weight_idx = param_names.index("token_embedding.weight")
print(f"Embedding weight at parameter index: {embedding_weight_idx}")

wrapper = StatelessWrapper(model, param_names, list(buffers_dict.keys()), embedding_weight_idx)
full_inputs = param_values + buffer_values + [idx, targets]

print(f"Compiling Qwen-Mini (SiLU/SwiGLU/RMSNorm/RoPE)...")
module = fx.export_and_import(wrapper, *full_inputs, output_type="stablehlo")

output_path = "models/tests/qwen_test.mlir"
with open(output_path, "w") as f:
    f.write(str(module.operation))

print(f"✓ Saved Qwen-Mini model to {output_path}")
