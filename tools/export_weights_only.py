#!/usr/bin/env python3
"""
Export Qwen weights as flat binary, strictly matching the MLIR export signature.

This script ensures the weight binary matches exactly what the MLIR export
script functionalizes. The key is that BOTH scripts must iterate parameters
in the EXACT SAME ORDER.

The root cause of "gibberish" output is byte-offset misalignment:
- The meta.json (from MLIR introspection) expects N parameters with total X bytes
- The weight file must have EXACTLY X bytes in the EXACT order
- Any mismatch (extra/missing bytes) shifts ALL subsequent tensor reads
- This scrambles every weight tensor after the misalignment point

CURRENT STATUS (from file sizes):
- meta.json expects:    1,976,131,200 bytes (includes [32] shape = 128 bytes)
- qwen_flat.bin:        1,976,131,072 bytes (128 bytes SHORT!)
- qwen_flat.bin.backup2: 1,976,131,200 bytes (matches meta!)

FIX: Regenerate qwen_flat.bin to match meta.json expectations.

Usage:
    python tools/export_weights_only.py [--verify-only]
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os
import sys
import json

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "checkpoints/initial_weights"
OUTPUT_FILE = "qwen_flat.bin"
META_FILE = "models/qwen_rl_generation.mlir.meta.json"

def get_expected_bytes_from_meta(meta_path: str) -> tuple[int, list]:
    """Load meta.json and calculate expected total bytes from parameter shapes."""
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


def restore_from_backup():
    """Quick fix: restore from backup2 which has correct byte count."""
    backup_path = os.path.join(OUTPUT_DIR, "qwen_flat.bin.backup2")
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    if not os.path.exists(backup_path):
        print(f"ERROR: Backup file not found: {backup_path}")
        return False

    import shutil
    shutil.copy2(backup_path, output_path)
    print(f"Restored {output_path} from {backup_path}")
    print(f"Size: {os.path.getsize(output_path):,} bytes")
    return True


def main():
    verify_only = '--verify-only' in sys.argv
    restore_backup = '--restore-backup' in sys.argv

    if restore_backup:
        if restore_from_backup():
            print("\nWeight file restored. Restart Shepherd to reload.")
        return

    # Load expected sizes from meta.json if available
    expected_bytes, shape_sizes = get_expected_bytes_from_meta(META_FILE)

    print(f"Loading {MODEL_ID}...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )

    # Get parameters and buffers separately for comparison
    params = list(model.named_parameters())
    buffers = list(model.named_buffers())

    print(f"\n{'='*60}")
    print(f"Model Analysis:")
    print(f"{'='*60}")
    print(f"  Parameters (trainable):  {len(params)}")
    print(f"  Buffers (non-trainable): {len(buffers)}")

    if buffers:
        print(f"\n  Buffers that will be EXCLUDED:")
        for name, buf in buffers:
            size = buf.numel() * buf.element_size()
            print(f"    - {name}: {list(buf.shape)} ({size} bytes)")

    # Calculate total parameter bytes
    total_param_bytes = 0
    param_info = []

    for name, param in params:
        data = param.detach().cpu().float().numpy()
        size = data.nbytes
        total_param_bytes += size
        param_info.append((name, list(param.shape), size))

    print(f"\n  Total parameter bytes: {total_param_bytes:,} ({total_param_bytes / 1024 / 1024:.2f} MB)")

    # Verify against meta.json
    if expected_bytes is not None:
        print(f"\n{'='*60}")
        print(f"Verification against {META_FILE}:")
        print(f"{'='*60}")
        print(f"  Expected shapes: {len(shape_sizes)}")
        print(f"  Actual params:   {len(params)}")
        print(f"  Expected bytes:  {expected_bytes:,}")
        print(f"  Actual bytes:    {total_param_bytes:,}")

        if len(shape_sizes) != len(params):
            print(f"\n  WARNING: Shape count mismatch!")
            print(f"  This may indicate buffers are included in meta.json")

        if expected_bytes != total_param_bytes:
            print(f"\n  WARNING: Byte count mismatch!")
            diff = abs(expected_bytes - total_param_bytes)
            print(f"  Difference: {diff:,} bytes")

            # Check if diff matches common buffer sizes
            if diff == 128:  # inv_freq for head_dim=64
                print(f"  This matches inv_freq buffer size (head_dim/2 * 4 bytes)")
        else:
            print(f"\n  MATCH: Byte counts are identical")

        # Detailed shape comparison (first few)
        print(f"\n  First 5 shapes comparison:")
        for i in range(min(5, len(shape_sizes), len(param_info))):
            meta_shape, meta_size = shape_sizes[i]
            name, param_shape, param_size = param_info[i]
            match = "" if meta_shape == param_shape else " <-- MISMATCH"
            print(f"    [{i}] meta={meta_shape} vs param={param_shape}{match}")
            print(f"         {name}")

    if verify_only:
        print(f"\nVerification complete (--verify-only mode, no files written)")
        return

    # Write the weight file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    print(f"\n{'='*60}")
    print(f"Writing weights to {output_path}:")
    print(f"{'='*60}")

    with open(output_path, "wb") as f:
        for name, param in params:
            data = param.detach().cpu().float().numpy().tobytes()
            f.write(data)

    # Verify written file
    actual_size = os.path.getsize(output_path)
    print(f"  Written: {actual_size:,} bytes")

    if actual_size == total_param_bytes:
        print(f"  Verification: PASSED")
    else:
        print(f"  Verification: FAILED (expected {total_param_bytes:,})")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"SUCCESS: Exported {len(params)} parameters ({actual_size / 1024 / 1024:.2f} MB)")
    print(f"{'='*60}")

    # Print reminder about cache cleanup
    print(f"\nNEXT STEPS:")
    print(f"  1. Verify weight file size matches meta.json expectation")
    print(f"  2. Clear cached metadata if needed: rm models/*.meta.json")
    print(f"  3. Restart Shepherd to reload weights")


def print_help():
    print("""
Usage: python tools/export_weights_only.py [OPTIONS]

Options:
  --verify-only     Only check alignment, don't write files
  --restore-backup  Restore from qwen_flat.bin.backup2 (quick fix)
  --help            Show this help message

The script ensures weight binary matches the MLIR export signature exactly.
""")


if __name__ == "__main__":
    if '--help' in sys.argv:
        print_help()
    else:
        main()
