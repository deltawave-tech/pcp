#!/usr/bin/env python3
"""Write Qwen3.6-27B adapter-only GRPO smoke artifacts.

The generated adapter state is intentionally tiny compared with the frozen 27B
base. It is the only trainable tensor in the first 27B GRPO smoke.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np


DEFAULT_ADAPTER_PATH = Path("checkpoints/initial_weights/qwen36_27b_adapter_f32.bin")
DEFAULT_PROMPT_PATH = Path("data/rl_prompts.bin")


def write_adapter(path: Path, hidden_size: int = 5120, rank: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((hidden_size, rank), dtype=np.float32)
    values.tofile(path)
    print(f"Wrote adapter weights: {path} ({values.nbytes} bytes)")


def write_prompts(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompts = [[9707]]
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(prompts)))
        for prompt in prompts:
            f.write(struct.pack("<I", len(prompt)))
            for token in prompt:
                f.write(struct.pack("<Q", token))
    print(f"Wrote GRPO prompts: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-output", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--prompt-output", type=Path, default=DEFAULT_PROMPT_PATH)
    args = parser.parse_args()

    write_adapter(args.adapter_output)
    write_prompts(args.prompt_output)


if __name__ == "__main__":
    main()
