#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os
import sys
import json

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3-8B"
OUTPUT_DIR = "checkpoints/initial_weights"
OUTPUT_FILE = "qwen3_8b_flat.bin"
META_FILE = "models/qwen3_8b_rl_generation.mlir.meta.json"

def get_expected_bytes_from_meta(meta_path: str):
    if not os.path.exists(meta_path):
        return None, []
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    shapes = meta.get('parameter_shapes', [])
    total_bytes = 0
    shape_sizes = []
    for shape in shapes:
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        size_bytes = num_elements * 4  # f32
        total_bytes += size_bytes
        shape_sizes.append((shape, size_bytes))
    return total_bytes, shape_sizes

def main():
    expected_bytes, shape_sizes = get_expected_bytes_from_meta(META_FILE)

    print(f"Loading {MODEL_ID} weights in float32...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True
    )

    params = list(model.named_parameters())
    buffers = list(model.named_buffers())

    print(f"\nModel Analysis: {len(params)} parameters")

    total_param_bytes = 0
    for name, param in params:
        total_param_bytes += param.numel() * 4  # f32

    print(f"Total parameter size: {total_param_bytes / 1024**3:.2f} GB (f32)")

    if expected_bytes is not None:
        print(f"Verifying against meta.json: Expected {expected_bytes}, Actual {total_param_bytes}")
        if expected_bytes != total_param_bytes:
            print("WARNING: Byte mismatch! Check buffer inclusion logic.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    print(f"Writing weights to {output_path}...")
    with open(output_path, "wb") as f:
        for name, param in params:
            data = param.detach().cpu().float().numpy().tobytes()
            f.write(data)

    print(f"✓ Export Complete: {output_path}")

if __name__ == "__main__":
    main()
