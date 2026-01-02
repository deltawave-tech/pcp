import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from torch_mlir import fx
import os
import sys
import types
import math
import functools

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "models/qwen_grpo_training.mlir"

# Training Constraints
GROUP_SIZE = 4  # Batch size for GRPO step
SEQ_LEN = 512  # Fixed Sequence Length
DTYPE = torch.float32

print(f"Loading {MODEL_ID}...")
try:
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=DTYPE,
        attn_implementation="eager",
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

model.train()

# --- MONKEY PATCHES (RoPE, Attention, Layers) ---

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if cos.dim() == 3: cos = cos.unsqueeze(1)
    if sin.dim() == 3: sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def clean_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    hidden_size = self.q_proj.in_features
    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    head_dim = hidden_size // num_heads
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    cos, sin = kwargs.get("position_embeddings", (None, None))
    if cos is None: cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if num_key_value_heads != num_heads:
        n_rep = num_heads // num_key_value_heads
        key_states = key_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, q_len, head_dim).reshape(bsz, num_heads, q_len, head_dim)
        value_states = value_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, q_len, head_dim).reshape(bsz, num_heads, q_len, head_dim)
    query_states = query_states / math.sqrt(head_dim)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    if attention_mask is not None: attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None

def clean_layer_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    attn_outputs = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, **kwargs)
    attn_output = attn_outputs[0]
    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return (hidden_states,)

def clean_qwen2_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds=None, use_cache=False, **kwargs):
    if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    for i, layer_module in enumerate(self.layers):
        layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=None, output_attentions=False, use_cache=False, position_embeddings=position_embeddings)
        hidden_states = layer_outputs[0]
    hidden_states = self.norm(hidden_states)
    return hidden_states

print("Applying Monkey Patches...")
for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(clean_attention_forward, layer.self_attn)
    layer.forward = types.MethodType(clean_layer_forward, layer)
model.model.forward = types.MethodType(clean_qwen2_forward, model.model)

# --- 5. GRPO Loss Wrapper ---
class QwenGRPOWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Pre-create the causal mask as a buffer to avoid tracing torch.where constants
        seq_idx = torch.arange(SEQ_LEN)
        # Using full() instead of where() to avoid constants being traced as inputs
        causal_mask = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), -1e4, dtype=DTYPE), diagonal=1)
        self.register_buffer('causal_mask', causal_mask.unsqueeze(0).unsqueeze(0))

    def forward(self, input_ids, attention_mask, advantages):
        # 1. Position IDs
        position_ids = torch.arange(SEQ_LEN, device=input_ids.device).unsqueeze(0).expand(GROUP_SIZE, SEQ_LEN)

        # 2. Masks - use pre-created causal mask to avoid torch.where constants
        padding_mask = (1.0 - attention_mask) * -1e4
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        final_mask = self.causal_mask + padding_mask

        # 3. Forward
        hidden_states = self.model.model(input_ids=input_ids, attention_mask=final_mask, position_ids=position_ids, past_key_values=None)
        logits = self.model.lm_head(hidden_states)

        # 4. Loss
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

wrapper = QwenGRPOWrapper(model)

# --- EXPORT STRATEGY: Pre-transformed Functional Gradients ---

# 1. Extract State
# We separate Params (which we differentiate w.r.t) from Buffers (inv_freq, etc - constants)
params = dict(wrapper.named_parameters())
buffers = dict(wrapper.named_buffers())

param_keys = list(params.keys())
buffer_keys = list(buffers.keys())

# Ensure params require grad for tracking
param_values = [p.detach().requires_grad_(True) for p in params.values()]
buffer_values = list(buffers.values())

print(f"Model has {len(param_keys)} parameters and {len(buffer_keys)} buffers")

# 2. Define the Pure Functional Loss
# This function reconstructs the model state from arguments and runs forward
def functional_loss(params_tuple, buffers_tuple, input_ids, mask, adv):
    # Reconstruct the state dictionary
    state_dict = {}
    for k, v in zip(param_keys, params_tuple):
        state_dict[k] = v
    for k, v in zip(buffer_keys, buffers_tuple):
        state_dict[k] = v

    return torch.func.functional_call(wrapper, state_dict, (input_ids, mask, adv))

# 3. Create the Gradient Function (The Magic Step)
# We apply the gradient transform HERE, in Python, before any tracing happens.
# argnums=0 means "differentiate with respect to the first argument (params_tuple)"
print("Creating functional gradient transform...")
compute_grads = torch.func.grad(functional_loss, argnums=0)

# 4. Pre-trace the gradient function using make_fx
# This creates an FX graph that contains the backward pass operations
# without needing to change autograd state during torch.export
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode

dummy_input_ids = torch.zeros((GROUP_SIZE, SEQ_LEN), dtype=torch.int64)
dummy_mask = torch.ones((GROUP_SIZE, SEQ_LEN), dtype=torch.float32)
dummy_advantages = torch.zeros((GROUP_SIZE,), dtype=DTYPE)

print(f"Pre-tracing gradient graph with make_fx (Group Size={GROUP_SIZE})...")

# Create wrapper function that takes flat inputs (including buffers)
def grad_fn_flat(*args):
    num_params = len(param_keys)
    num_buffers = len(buffer_keys)
    curr_params = args[:num_params]
    curr_buffers = args[num_params:num_params + num_buffers]
    input_ids = args[num_params + num_buffers]
    mask = args[num_params + num_buffers + 1]
    adv = args[num_params + num_buffers + 2]
    return compute_grads(curr_params, curr_buffers, input_ids, mask, adv)

# Include buffers as inputs to avoid FakeTensor issues
full_inputs = tuple(param_values + buffer_values + [dummy_input_ids, dummy_mask, dummy_advantages])

try:
    # Trace with make_fx using FakeTensorMode that allows non-fake inputs
    with FakeTensorMode(allow_non_fake_inputs=True):
        traced_grad_fn = make_fx(grad_fn_flat, tracing_mode="real")(*full_inputs)
    print("✓ Gradient graph pre-traced successfully")
    print(f"  Graph has {len(list(traced_grad_fn.graph.nodes))} nodes")

    # 5. Verify the output and export
    print("Verifying gradient output...")
    print(f"  Input count: {len(full_inputs)} (params={len(param_values)}, buffers={len(buffer_values)}, data=3)")

    # Count outputs (should be 290 gradients)
    test_out = traced_grad_fn(*full_inputs)
    if isinstance(test_out, tuple):
        print(f"  ✓ Output count: {len(test_out)} gradient tensors")
        # Print first few shapes
        for i, g in enumerate(test_out[:5]):
            print(f"    grad[{i}]: {g.shape}")
        print(f"    ...")
    else:
        print(f"  Output: single tensor of shape {test_out.shape}")

    # Free memory before MLIR export
    import gc
    del test_out
    gc.collect()

    # Try linalg first which may use less memory, then convert to stablehlo
    print("Converting to Linalg (may use less memory)...")
    prog = fx.export_and_import(
        traced_grad_fn,
        *full_inputs,
        output_type="linalg-on-tensors",
        func_name="main"
    )

    print(f"Exporting to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(str(prog))
    print("✓ Export Successful! The MLIR now returns 290 gradient tensors.")

except Exception as e:
    print(f"Export Failed: {e}")
    import traceback
    traceback.print_exc()
