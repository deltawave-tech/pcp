import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from torch_mlir import fx
import os
import sys
import types
import math

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "models/qwen_rl_generation.mlir"

BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
DTYPE = torch.float32

print(f"Loading {MODEL_ID}...")
try:
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=DTYPE,
        attn_implementation="eager",
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

model.eval()

HEAD_DIM = config.hidden_size // config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
NUM_LAYERS = config.num_hidden_layers
NUM_HEADS = config.num_attention_heads

print(
    f"Config: Layers={NUM_LAYERS}, Heads={NUM_HEADS}, KV_Heads={NUM_KV_HEADS}, Head_Dim={HEAD_DIM}"
)


# --- 1. Static Cache Shim ---
class StaticCacheShim:
    def __init__(self, key_cache_list, value_cache_list, max_length):
        self.key_cache = key_cache_list
        self.value_cache = value_cache_list
        self.max_length = max_length
        self.new_k_updates = []
        self.new_v_updates = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.new_k_updates.append(key_states)
        self.new_v_updates.append(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.max_length

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self.max_length, 0


# --- 2. RoPE Helper ---
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
    if sin.dim() == 3:
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# --- 3. Monkey Patch: Attention Forward ---
def clean_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,
):
    hidden_size = self.q_proj.in_features
    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    head_dim = hidden_size // num_heads

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(
        1, 2
    )
    value_states = value_states.view(
        bsz, q_len, num_key_value_heads, head_dim
    ).transpose(1, 2)

    cos, sin = kwargs.get("position_embeddings", (None, None))
    if cos is None:
        cos, sin = self.rotary_emb(value_states, position_ids)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx
        )

    if num_key_value_heads != num_heads:
        n_rep = num_heads // num_key_value_heads
        key_states = (
            key_states[:, :, None, :, :]
            .expand(bsz, num_key_value_heads, n_rep, MAX_SEQ_LEN, head_dim)
            .reshape(bsz, num_heads, MAX_SEQ_LEN, head_dim)
        )
        value_states = (
            value_states[:, :, None, :, :]
            .expand(bsz, num_key_value_heads, n_rep, MAX_SEQ_LEN, head_dim)
            .reshape(bsz, num_heads, MAX_SEQ_LEN, head_dim)
        )

    query_states = query_states / math.sqrt(head_dim)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# --- 4. Monkey Patch: Layer Forward ---
def clean_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )
    attn_output = attn_outputs[0]
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return (hidden_states,)


# --- 5. Monkey Patch: Inner Model Forward (Qwen2Model) ---
def clean_inner_forward(
    self,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds=None,
    use_cache=True,
    **kwargs,
):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for i, layer_module in enumerate(self.layers):
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]

    hidden_states = self.norm(hidden_states)
    return hidden_states  # Return Tensor directly


# --- 6. Monkey Patch: Outer Model Forward (Qwen2ForCausalLM) ---
# This bypasses the library wrapper completely
def clean_outer_forward(
    self,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    use_cache=True,
    **kwargs,
):
    # Call our clean inner forward
    hidden_states = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
    )
    # Proj to vocab
    logits = self.lm_head(hidden_states)
    return logits  # Return Logits directly


# APPLY ALL PATCHES
print("Applying Monkey Patches...")
for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(clean_attention_forward, layer.self_attn)
    layer.forward = types.MethodType(clean_layer_forward, layer)

model.model.forward = types.MethodType(clean_inner_forward, model.model)
model.forward = types.MethodType(clean_outer_forward, model)


# --- 7. Wrapper & Execution ---
class QwenWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, *kv_flat):
        k_list = []
        v_list = []
        for i in range(0, len(kv_flat), 2):
            k_list.append(kv_flat[i])
            v_list.append(kv_flat[i + 1])

        cache = StaticCacheShim(k_list, v_list, MAX_SEQ_LEN)

        # Static Mask
        indices = torch.arange(MAX_SEQ_LEN, device=input_ids.device).view(
            1, 1, 1, MAX_SEQ_LEN
        )
        current_pos = position_ids.view(BATCH_SIZE, 1, 1, 1)
        neg_inf = torch.tensor(-1e4, device=input_ids.device, dtype=DTYPE)
        zero = torch.tensor(0.0, device=input_ids.device, dtype=DTYPE)
        mask = torch.where(indices <= current_pos, zero, neg_inf)

        # Returns logits directly now due to patch #6
        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
        )

        flat_updates = []
        for i in range(len(cache.new_k_updates)):
            flat_updates.append(cache.new_k_updates[i])
            flat_updates.append(cache.new_v_updates[i])

        return logits, *flat_updates


# Inputs
dummy_input_ids = torch.zeros((BATCH_SIZE, 1), dtype=torch.int64)
dummy_pos_ids = torch.zeros((BATCH_SIZE, 1), dtype=torch.int64)

dummy_kv_flat = []
cache_shape = (BATCH_SIZE, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)

print(f"Creating cache tensors: {cache_shape}")
for _ in range(NUM_LAYERS):
    dummy_kv_flat.append(torch.zeros(cache_shape, dtype=DTYPE))
    dummy_kv_flat.append(torch.zeros(cache_shape, dtype=DTYPE))

wrapper = QwenWrapper(model)

print("Tracing model with torch-mlir...")
try:
    prog = fx.export_and_import(
        wrapper,
        dummy_input_ids,
        dummy_pos_ids,
        *dummy_kv_flat,
        output_type="stablehlo",
        func_name="main",
    )

    print(f"Exporting to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(str(prog))
    print("✓ Export Successful")

except Exception as e:
    print(f"Export Failed: {e}")
    import traceback

    traceback.print_exc()
