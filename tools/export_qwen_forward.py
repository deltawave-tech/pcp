"""
Export Qwen GRPO forward pass (loss function) to StableHLO MLIR.
PCP's autodiff engine will then compute gradients from this forward pass.

Signature: (params..., input_ids, attention_mask, advantages) -> scalar loss
- params are in the same order as model.named_parameters() (matching convert_weights.py)
- No buffers as inputs - causal_mask and RoPE are computed as constants

Uses Nested Wrapper Pattern:
- Inner Wrapper (QwenGRPOLogic): Contains model + loss logic, returns Tensor
- Outer Wrapper (StatelessExportWrapper): Stateless interface for functional_call
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from torch_mlir import fx
from torch.func import functional_call
import types
import math
import os

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "models/qwen_grpo_training.mlir"

GROUP_SIZE = 4
SEQ_LEN = 512
DTYPE = torch.float32

print(f"Loading {MODEL_ID}...")
config = AutoConfig.from_pretrained(MODEL_ID)
config.use_cache = False
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=DTYPE,
    attn_implementation="eager",
)
model.train()

# --- 1. Stateless RoPE (Global Helpers) ---
# These are baked into the monkey patches as constants
HEAD_DIM = config.hidden_size // config.num_attention_heads
ROPE_BASE = float(config.rope_theta)

inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
t = torch.arange(SEQ_LEN, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)

# Convert to Python lists to force them as literals in MLIR (stablehlo.constant)
PRECOMPUTED_COS_LIST = emb.cos().unsqueeze(0).unsqueeze(0).tolist()
PRECOMPUTED_SIN_LIST = emb.sin().unsqueeze(0).unsqueeze(0).tolist()

causal_mask_tensor = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), -1e4, dtype=DTYPE), diagonal=1)
CAUSAL_MASK_LIST = causal_mask_tensor.unsqueeze(0).unsqueeze(0).tolist()

def get_causal_mask(device):
    return torch.tensor(CAUSAL_MASK_LIST, device=device, dtype=DTYPE)

def stateless_rope(position_ids):
    cos = torch.tensor(PRECOMPUTED_COS_LIST, device=position_ids.device, dtype=DTYPE)
    sin = torch.tensor(PRECOMPUTED_SIN_LIST, device=position_ids.device, dtype=DTYPE)
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# --- 2. Monkey Patches ---
def clean_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, -1).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(1, 2)

    cos, sin = stateless_rope(position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.config.num_key_value_heads != self.config.num_attention_heads:
        key_states = key_states.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)
        value_states = value_states.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.q_proj.in_features // self.config.num_attention_heads)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    return self.o_proj(attn_output), None, None

def clean_layer_forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return (residual + hidden_states,)

def clean_qwen2_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds=None, **kwargs):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds
    for layer in self.layers:
        hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]
    return self.norm(hidden_states)

print("Applying patches...")
for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(clean_attention_forward, layer.self_attn)
    layer.forward = types.MethodType(clean_layer_forward, layer)
model.model.forward = types.MethodType(clean_qwen2_forward, model.model)

# Remove the rotary_emb buffer since we use stateless RoPE (precomputed constants)
# This prevents it from appearing as an input in the traced graph
if hasattr(model.model, 'rotary_emb'):
    delattr(model.model, 'rotary_emb')
    print("Removed rotary_emb buffer (using stateless RoPE)")

# --- 3. Logic Wrapper (Stateful) ---
# This wrapper encapsulates the Forward + Loss logic.
# functional_call will run THIS forward method, which returns a Tensor.
class QwenGRPOLogic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model # Naming matches HF structure (model.model...)

    def forward(self, input_ids, attention_mask, advantages):
        # 1. Precompute masks/pos (Logic inside graph)
        causal_mask = get_causal_mask(input_ids.device)
        position_ids = torch.arange(SEQ_LEN, device=input_ids.device).unsqueeze(0).expand(GROUP_SIZE, SEQ_LEN)

        padding_mask = (1.0 - attention_mask) * -1e4
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        final_mask = causal_mask + padding_mask

        # 2. Model Forward - Call inner components directly to bypass HF forward
        # self.model.model is Qwen2Model (patched), returns hidden_states tensor
        # self.model.lm_head is the final projection
        hidden_states = self.model.model(input_ids, final_mask, position_ids, None)
        logits = self.model.lm_head(hidden_states)

        # 3. GRPO Loss Calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        token_losses = F.cross_entropy(flat_logits, flat_labels, reduction="none").view(GROUP_SIZE, -1)

        token_losses = token_losses * shift_mask
        seq_lengths = torch.clamp(shift_mask.sum(dim=1), min=1.0)
        seq_nll = token_losses.sum(dim=1) / seq_lengths

        weighted_losses = seq_nll * advantages
        return weighted_losses.mean()

logic_wrapper = QwenGRPOLogic(model)

# --- 4. Stateless Export Wrapper ---
# This ensures inputs are (Params... -> Data...) matching the file format
class StatelessExportWrapper(nn.Module):
    def __init__(self, logic_module, param_names):
        super().__init__()
        self.logic_module = logic_module
        self.param_names = param_names
        self.num_params = len(param_names)

    def forward(self, *args):
        # 1. Split inputs
        params = args[:self.num_params]
        input_ids = args[self.num_params]
        attention_mask = args[self.num_params + 1]
        advantages = args[self.num_params + 2]

        # 2. Reconstruct State Dict
        # Prefix with "model." since logic_wrapper stores base model as self.model
        state_dict = {}
        for name, param in zip(self.param_names, params):
            state_dict["model." + name] = param

        # NOTE: We don't add buffers (like rotary_emb.inv_freq) to state_dict
        # because we're using stateless RoPE with precomputed constants

        # 3. Functional Call on the LOGIC wrapper
        # This returns the Tensor loss directly, avoiding HF object return issues
        return functional_call(
            self.logic_module,
            state_dict,
            (input_ids, attention_mask, advantages)
        )

# Prepare State
params_dict = dict(model.named_parameters())
param_names = list(params_dict.keys())
param_values = list(params_dict.values())

print(f"Exporting with {len(param_names)} parameters matching convert_weights.py")

export_wrapper = StatelessExportWrapper(logic_wrapper, param_names)

# Dummy Inputs
dummy_input_ids = torch.zeros((GROUP_SIZE, SEQ_LEN), dtype=torch.int64)
dummy_mask = torch.ones((GROUP_SIZE, SEQ_LEN), dtype=torch.float32)
dummy_advantages = torch.zeros((GROUP_SIZE,), dtype=DTYPE)

# Final Input List: [Params..., Input_IDs, Mask, Adv]
full_inputs = param_values + [dummy_input_ids, dummy_mask, dummy_advantages]

print("Tracing...")
module = fx.export_and_import(export_wrapper, *full_inputs, output_type="stablehlo")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write(str(module.operation))

print(f"✓ Saved to {OUTPUT_PATH}")
print(f"Signature: ({len(param_names)} params, input_ids, mask, advantages) -> scalar loss")
print(f"Parameter ordering matches convert_weights.py")
