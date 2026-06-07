#!/usr/bin/env python3
"""Export a text-only one-token Qwen3.6-27B generation graph.

This is the first full-size 27B graph target for Milestone 7. It intentionally
uses a no-cache, one-token forward so the compiler path can be validated before
adding recurrent/attention state reuse.
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
from torch.func import functional_call
from torch_mlir import fx
from transformers import AutoConfig, AutoModelForCausalLM


DEFAULT_MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_OUTPUT_PATH = Path("models/qwen36_27b_one_token_nocache.mlir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dtype", choices=("bf16", "f16"), default="bf16")
    parser.add_argument("--output-type", default="stablehlo")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "f16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.int64:
        return "i64"
    raise ValueError(f"unsupported dtype: {dtype}")


def clean_text_forward(self, input_ids, position_ids):
    hidden_states = self.embed_tokens(input_ids)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer in self.layers:
        layer_outputs = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    return self.norm(hidden_states)


def clean_outer_forward(self, input_ids, position_ids):
    hidden_states = self.model(input_ids=input_ids, position_ids=position_ids)
    return self.lm_head(hidden_states).float()


class FunctionalQwenWrapper(nn.Module):
    def __init__(self, model: nn.Module, parameter_names: list[str]):
        super().__init__()
        object.__setattr__(self, "_model", model)
        self.parameter_names = parameter_names

    def forward(self, *args):
        param_count = len(self.parameter_names)
        params = args[:param_count]
        input_ids = args[param_count]
        position_ids = args[param_count + 1]
        if input_ids.ndim != 2 or position_ids.ndim != 2:
            raise RuntimeError(
                f"bad export input binding: arg_count={len(args)} param_count={param_count} "
                f"input_shape={tuple(input_ids.shape)} position_shape={tuple(position_ids.shape)}"
            )
        param_map = {name: value for name, value in zip(self.parameter_names, params)}
        return functional_call(
            self._model,
            param_map,
            (),
            {"input_ids": input_ids, "position_ids": position_ids},
        )


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)

    print(f"load start {time.asctime()}", flush=True)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    model.model.forward = types.MethodType(clean_text_forward, model.model)
    model.forward = types.MethodType(clean_outer_forward, model)

    params = dict(model.named_parameters())
    parameter_names = list(params.keys())
    parameter_values = list(params.values())
    print(f"loaded patched {time.asctime()}", flush=True)
    print(f"params {len(parameter_names)}", flush=True)

    wrapper = FunctionalQwenWrapper(model, parameter_names)
    input_ids = torch.zeros((1, 1), dtype=torch.int64)
    position_ids = torch.zeros((1, 1), dtype=torch.int64)

    print(f"export start {time.asctime()}", flush=True)
    program = fx.export_and_import(
        wrapper,
        *parameter_values,
        input_ids,
        position_ids,
        output_type=args.output_type,
        func_name="main",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(str(program), encoding="utf-8")

    meta = {
        "parameter_names": parameter_names,
        "parameter_shapes": [list(param.shape) for param in parameter_values],
        "parameter_dtypes": [dtype_name(param.dtype) for param in parameter_values],
        "data_input_shapes": [list(input_ids.shape), list(position_ids.shape)],
        "data_input_dtypes": ["i64", "i64"],
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"export done {time.asctime()}", flush=True)
    print(f"MLIR: {args.output}", flush=True)


if __name__ == "__main__":
    main()
