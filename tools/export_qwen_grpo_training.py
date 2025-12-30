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
OUTPUT_PATH = "models/qwen_grpo_training.mlir"

# Training Constraints
GROUP_SIZE = 4  # Batch size for GRPO step
SEQ_LEN = 512  # Fixed Sequence Length (Prompt + Completion)
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


# --- 1. RoPE Helper ---
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


# --- 2. Monkey Patch: Attention Forward ---
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

    if num_key_value_heads != num_heads:
        n_rep = num_heads // num_key_value_heads
        key_states = (
            key_states[:, :, None, :, :]
            .expand(bsz, num_key_value_heads, n_rep, q_len, head_dim)
            .reshape(bsz, num_heads, q_len, head_dim)
        )
        value_states = (
            value_states[:, :, None, :, :]
            .expand(bsz, num_key_value_heads, n_rep, q_len, head_dim)
            .reshape(bsz, num_heads, q_len, head_dim)
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

    return attn_output, None, None


# --- 3. Monkey Patch: Layer Forward ---
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


# --- 4. Monkey Patch: Model Forward ---
def clean_qwen2_forward(
    self,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds=None,
    use_cache=False,
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
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]

    hidden_states = self.norm(hidden_states)
    return hidden_states


# APPLY PATCHES
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

    def forward(self, input_ids, attention_mask, advantages):
        """
        Calculates GRPO Loss.
        """
        # 1. Create Position IDs
        position_ids = (
            torch.arange(SEQ_LEN, device=input_ids.device)
            .unsqueeze(0)
            .expand(GROUP_SIZE, SEQ_LEN)
        )

        # 2. Create Masks
        seq_idx = torch.arange(SEQ_LEN, device=input_ids.device)
        causal_mask = torch.where(
            seq_idx[None, :] <= seq_idx[:, None],
            torch.tensor(0.0, dtype=DTYPE, device=input_ids.device),
            torch.tensor(-1e4, dtype=DTYPE, device=input_ids.device),
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        padding_mask = (1.0 - attention_mask) * -1e4
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)

        final_mask = causal_mask + padding_mask

        # 3. Forward Pass
        hidden_states = self.model.model(
            input_ids=input_ids,
            attention_mask=final_mask,
            position_ids=position_ids,
            past_key_values=None,
        )

        # 4. Compute Logits
        logits = self.model.lm_head(hidden_states)

        # 5. Shift & Compute NLL
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # FIX: Use functional cross_entropy instead of creating nn.Module
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        token_losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        token_losses = token_losses.view(GROUP_SIZE, -1)

        # Mask Padding
        token_losses = token_losses * shift_mask

        # Sum Loss per Sequence
        seq_lengths = shift_mask.sum(dim=1)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        seq_nll = token_losses.sum(dim=1) / seq_lengths

        # 6. Apply Advantages
        weighted_losses = seq_nll * advantages

        # 7. Mean over Batch
        loss = weighted_losses.mean()

        return loss


# --- Execution ---

wrapper = QwenGRPOWrapper(model)
params = dict(wrapper.named_parameters())
param_keys = list(params.keys())
param_values = list(params.values())

dummy_input_ids = torch.zeros((GROUP_SIZE, SEQ_LEN), dtype=torch.int64)
dummy_mask = torch.ones((GROUP_SIZE, SEQ_LEN), dtype=torch.float32)
dummy_advantages = torch.zeros((GROUP_SIZE,), dtype=DTYPE)

print(f"Compiling GRPO Training Graph (Group Size={GROUP_SIZE})...")

try:

    class StatelessTraceWrapper(nn.Module):
        def __init__(self, wrapper, param_keys):
            super().__init__()
            self.wrapper = wrapper
            self.param_keys = param_keys

        def forward(self, *args):
            curr_params_vals = args[:-3]
            inp = args[-3]
            msk = args[-2]
            adv = args[-1]

            curr_params = {k: v for k, v in zip(self.param_keys, curr_params_vals)}
            return torch.func.functional_call(
                self.wrapper, curr_params, (inp, msk, adv)
            )

    trace_wrapper = StatelessTraceWrapper(wrapper, param_keys)
    full_inputs = param_values + [dummy_input_ids, dummy_mask, dummy_advantages]

    prog = fx.export_and_import(
        trace_wrapper, *full_inputs, output_type="stablehlo", func_name="main"
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
