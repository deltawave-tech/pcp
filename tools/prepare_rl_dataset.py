import torch
from transformers import AutoTokenizer
import struct
import os
import json

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_FILE = "data/rl_prompts.bin"

# Test Prompts (Math & Logic)
prompts = [
    "2+2=",
    "The capital of France is",
    "Write a python function to add two numbers.",
    "Is fire hot? Answer yes or no.",
    "Solve 3 * 4 + 2."
]

print(f"Loading Tokenizer {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Tokenizing {len(prompts)} prompts...")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    # Header: Number of prompts (u32)
    f.write(struct.pack("<I", len(prompts)))

    for p in prompts:
        # Encode
        ids = tokenizer.encode(p, add_special_tokens=False)

        print(f"  '{p}' -> {len(ids)} tokens: {ids}")

        # Format: Length (u32), Tokens (u64 array)
        # Note: Zig code expects i64/u64 for tokens usually
        f.write(struct.pack("<I", len(ids)))
        for token in ids:
            f.write(struct.pack("<Q", token))  # Q = unsigned long long (64 bit)

print(f"✓ Saved to {OUTPUT_FILE}")
print(f"File size: {os.path.getsize(OUTPUT_FILE)} bytes")
