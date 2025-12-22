import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx
from torch.func import functional_call
import math

# --- Configuration (Medium Model) ---
BATCH_SIZE = 32  # Increased from 1
BLOCK_SIZE = 128  # Increased from 64
VOCAB_SIZE = 256
N_EMBD = 256  # Increased from 128
N_HEAD = 8  # 32 dims per head
HEAD_DIM = N_EMBD // N_HEAD
N_LAYER = 4  # Increased from 1
DROPOUT = 0.0


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequencies for RoPE.
    These will be stored as constant buffers in the MLIR.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Returns shape [BLOCK_SIZE, HEAD_DIM // 2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply RoPE to a tensor x of shape [B, T, NH, D]
    """
    # x shape: [Batch, Seq, Heads, Head_Dim]
    # cos/sin shape: [Seq, Head_Half_Dim]

    d = x.shape[-1]
    # Split the last dimension (Head_Dim) into two halves
    # This generates stablehlo.slice
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    # Reshape cos/sin for broadcasting across Batch and Head dimensions
    # Resulting shape: [1, Seq, 1, Head_Half_Dim]
    cos = cos.view(1, x.shape[1], 1, -1)
    sin = sin.view(1, x.shape[1], 1, -1)

    # Standard RoPE rotation formula:
    # y1 = x1 * cos - x2 * sin
    # y2 = x2 * cos + x1 * sin
    out1 = (x1 * cos) - (x2 * sin)
    out2 = (x2 * cos) + (x1 * sin)

    # Recombine halves. This generates stablehlo.concatenate
    return torch.cat((out1, out2), dim=-1)


class RoPESelfAttention(nn.Module):
    def __init__(self, cos, sin):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.n_head = N_HEAD
        self.n_embd = N_EMBD
        self.head_dim = N_EMBD // N_HEAD

        # RoPE buffers passed from the parent block
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head: [B, T, NH, HD]
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Queries and Keys
        q = apply_rotary_emb(q, self.cos[:T], self.sin[:T]).transpose(1, 2)
        k = apply_rotary_emb(k, self.cos[:T], self.sin[:T]).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # [B, NH, T, HD]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class RoPEBlock(nn.Module):
    def __init__(self, cos, sin):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = RoPESelfAttention(cos, sin)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RoPETansformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)

        # Precompute RoPE frequencies once for the whole model
        cos, sin = precompute_freqs_cis(N_EMBD // N_HEAD, BLOCK_SIZE)

        self.blocks = nn.Sequential(*[RoPEBlock(cos, sin) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets):
        x = self.token_embedding(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return loss


# --- Main Export Logic ---
model = RoPETansformer()
model.train()

# Dummy inputs for shape inference
idx = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)
targets = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)

# Separate trainable parameters from constant buffers (RoPE frequencies)
params_dict = dict(model.named_parameters())
buffers_dict = dict(model.named_buffers())
param_names = list(params_dict.keys())
param_values = list(params_dict.values())
buffer_values = list(buffers_dict.values())


class StatelessRoPEWrapper(nn.Module):
    def __init__(self, base_model, param_names, buffer_names):
        super().__init__()
        self.base_model = base_model
        self.param_names = param_names
        self.buffer_names = buffer_names

    def forward(self, *args):
        n_p = len(self.param_names)
        n_b = len(self.buffer_names)

        # Slicing: [Params, Buffers, idx, targets]
        params = args[:n_p]
        buffers = args[n_p : n_p + n_b]
        idx = args[-2]
        targets = args[-1]

        p_dict = {name: p for name, p in zip(self.param_names, params)}
        b_dict = {name: b for name, b in zip(self.buffer_names, buffers)}

        state_dict = {**p_dict, **b_dict}
        return functional_call(self.base_model, state_dict, (idx, targets))


wrapper = StatelessRoPEWrapper(model, param_names, list(buffers_dict.keys()))
# Input signature: [~40 parameter tensors, 2 RoPE buffers, 1 input_ids, 1 targets]
full_inputs = param_values + buffer_values + [idx, targets]

print(f"Compiling Medium RoPE NanoGPT...")
print(f"  Layers: {N_LAYER}")
print(f"  Embedding Dim: {N_EMBD}")
print(f"  Heads: {N_HEAD}")
print(f"  Context: {BLOCK_SIZE}")

module = fx.export_and_import(wrapper, *full_inputs, output_type="stablehlo")

output_path = "models/rope_medium.mlir"
with open(output_path, "w") as f:
    f.write(str(module.operation))

print(f"✓ Saved medium RoPE model to {output_path}")
