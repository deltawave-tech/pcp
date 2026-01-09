#!/usr/bin/env python3
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
MODEL_ID = "Qwen/Qwen3-8B"
OUTPUT_PATH = "models/qwen3_8b_grpo_training.mlir"

GROUP_SIZE = 4
SEQ_LEN = 512
DTYPE = torch.float32

print(f"Loading {MODEL_ID}...")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.use_cache = False
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=DTYPE,
    attn_implementation="eager",
    trust_remote_code=True
)
model.train()

# --- 1. Stateless RoPE ---
HEAD_DIM = config.hidden_size // config.num_attention_heads
ROPE_BASE = float(getattr(config, "rope_theta", 1000000.0))

inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
t = torch.arange(SEQ_LEN, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)

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

# --- 2. Monkey Patches (Updated with QK-Norm) ---
def clean_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, -1)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, -1)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(1, 2)

    # QWEN3: QK-Norm
    if hasattr(self, 'q_norm') and self.q_norm is not None:
        query_states = self.q_norm(query_states)
    if hasattr(self, 'k_norm') and self.k_norm is not None:
        key_states = self.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

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

if hasattr(model.model, 'rotary_emb'):
    delattr(model.model, 'rotary_emb')

# --- 3. Logic Wrapper ---
class QwenGRPOLogic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, advantages):
        causal_mask = get_causal_mask(input_ids.device)
        position_ids = torch.arange(SEQ_LEN, device=input_ids.device).unsqueeze(0).expand(GROUP_SIZE, SEQ_LEN)

        padding_mask = (1.0 - attention_mask) * -1e4
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        final_mask = causal_mask + padding_mask

        hidden_states = self.model.model(input_ids, final_mask, position_ids, None)
        logits = self.model.lm_head(hidden_states)

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

class StatelessExportWrapper(nn.Module):
    def __init__(self, logic_module, param_names):
        super().__init__()
        self.logic_module = logic_module
        self.param_names = param_names
        self.num_params = len(param_names)

    def forward(self, *args):
        params = args[:self.num_params]
        input_ids = args[self.num_params]
        attention_mask = args[self.num_params + 1]
        advantages = args[self.num_params + 2]

        state_dict = {}
        for name, param in zip(self.param_names, params):
            state_dict["model." + name] = param

        return functional_call(
            self.logic_module,
            state_dict,
            (input_ids, attention_mask, advantages)
        )

params_dict = dict(model.named_parameters())
param_names = list(params_dict.keys())
param_values = list(params_dict.values())

print(f"Exporting with {len(param_names)} parameters...")

export_wrapper = StatelessExportWrapper(logic_wrapper, param_names)

dummy_input_ids = torch.zeros((GROUP_SIZE, SEQ_LEN), dtype=torch.int64)
dummy_mask = torch.ones((GROUP_SIZE, SEQ_LEN), dtype=torch.float32)
dummy_advantages = torch.zeros((GROUP_SIZE,), dtype=DTYPE)

full_inputs = param_values + [dummy_input_ids, dummy_mask, dummy_advantages]

print("Tracing...")
module = fx.export_and_import(export_wrapper, *full_inputs, output_type="stablehlo")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write(str(module.operation))

print(f"✓ Saved to {OUTPUT_PATH}")
