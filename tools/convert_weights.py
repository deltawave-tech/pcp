import torch
from transformers import AutoModelForCausalLM, AutoConfig
import struct
import os
import sys

# Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "checkpoints/initial_weights"
OUTPUT_FILE = "qwen_flat.bin"

print(f"Loading {MODEL_ID}...")
try:
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float32,  # Force F32 for simplicity in Zig
        attn_implementation="eager",
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

print("Flattening and writing weights...")
print(
    "NOTE: This MUST match the iteration order of model.parameters() used in the MLIR export."
)

count = 0
total_bytes = 0

with open(output_path, "wb") as f:
    # Iterate exactly how torch-mlir export iterates
    for name, param in model.named_parameters():
        # param is a Tensor. Detach, cpu, float32, numpy.
        data = param.detach().cpu().float().numpy()

        # Write raw bytes
        f.write(data.tobytes())

        byte_len = data.nbytes
        total_bytes += byte_len
        count += 1

        # Optional: Print metadata for debugging
        # print(f"{name}: {data.shape} ({byte_len} bytes)")

print(f"✓ Saved {count} tensors to {output_path}")
print(f"✓ Total Size: {total_bytes / 1024 / 1024:.2f} MB")
