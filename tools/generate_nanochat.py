import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.func import functional_call
from torch_mlir import fx
from torch_mlir.extras.fx_decomp_util import get_decomposition_table


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nanochat.gpt import GPT, GPTConfig
except Exception as exc:
    raise SystemExit(
        "Failed to import nanochat.gpt. Run from the repo root and ensure "
        "the nanochat package is present. "
        f"Original error: {exc}"
    )


DEFAULTS = {
    "batch_size": 64,
    "block_size": 32,
    "vocab_size": 65,
    "n_layer": 2,
    "n_head": 4,
    "n_kv_head": 4,
    "n_embd": 64,
}


class StatelessWrapper(nn.Module):
    def __init__(self, base_model, param_names):
        super().__init__()
        self.base_model = base_model
        self.param_names = param_names

    def forward(self, *args):
        params = args[:-2]
        idx = args[-2]
        targets = args[-1]
        params_dict = {name: param for name, param in zip(self.param_names, params)}
        return functional_call(self.base_model, params_dict, (idx, targets))


def patch_rotary_cache(model, sequence_len):
    head_dim = model.config.n_embd // model.config.n_head
    model.rotary_seq_len = sequence_len
    cos, sin = model._precompute_rotary_embeddings(
        sequence_len, head_dim, device=model.get_device()
    )
    model.cos = cos
    model.sin = sin


def validate_model_contract(model):
    bad_linears = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            bad_linears.append(name)
    if bad_linears:
        raise RuntimeError(
            "nanochat must use bias-free Linear layers. "
            f"Found bias in: {', '.join(bad_linears)}"
        )
    if model.transformer["wte"].weight is model.lm_head.weight:
        raise RuntimeError("nanochat uses untied weights; wte and lm_head are tied.")


def check_main_signature(mlir_text):
    match = re.search(r"func\.func @main\(([^)]*)\) -> ([^{\n]+)", mlir_text)
    if not match:
        return False, "Could not find func.func @main signature."

    args_str = match.group(1).strip()
    ret_str = match.group(2).strip()
    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

    if len(args) < 2:
        return False, "Signature has fewer than 2 arguments."

    def arg_type(arg):
        return arg.split(":", 1)[1].strip() if ":" in arg else arg.strip()

    last_two = [arg_type(args[-2]), arg_type(args[-1])]
    if "xi64" not in last_two[0] or "xi64" not in last_two[1]:
        return False, "Last two args are not xi64 tensors."

    if "f32" not in ret_str:
        return False, f"Return type does not look like f32: {ret_str}"

    return True, "Signature check passed."


def scan_autodiff_coverage(mlir_text, repo_root):
    engine_path = repo_root / "pcp" / "src" / "autodiff" / "engine.zig"
    autodiff_ops = set()
    if engine_path.exists():
        engine_text = engine_path.read_text()
        autodiff_ops = set(re.findall(r"\"(stablehlo\.[a-zA-Z0-9_]+)\"", engine_text))

    used_ops = set(re.findall(r"stablehlo\.[a-zA-Z0-9_]+", mlir_text))
    if not autodiff_ops:
        return sorted(used_ops)

    return sorted(used_ops - autodiff_ops)


def maybe_dump_fx(wrapper, full_inputs, output_path):
    try:
        exported = torch.export.export(wrapper, tuple(full_inputs), strict=False)
        graph_text = str(exported.graph)  # pre-decomposition graph
        output_path.write_text(graph_text)
        return graph_text
    except Exception as exc:
        print(f"Warning: failed to dump FX graph ({exc}).")
        return None


def freeze_lifted_buffers(prog):
    """Work around torch-mlir's frozen-program import path for torch 2.3.

    In torch 2.3, ExportedProgram has a `.constants` dict, and torch-mlir's
    `import_frozen_program` currently only lifts `inputs_to_lifted_tensor_constants`
    when that attribute exists. nanochat's rotary caches (`base.cos`/`base.sin`) are
    treated as buffers (`inputs_to_buffers`), so they remain as placeholders and
    become extra function arguments in the StableHLO signature.

    We explicitly replace buffer placeholders with their constant tensor values so
    the exported StableHLO signature remains: `(params..., idx, targets) -> loss`.
    """

    sig = prog.graph_signature
    buffers = getattr(sig, "inputs_to_buffers", None) or {}
    if not buffers:
        return prog

    constants = getattr(prog, "constants", None) or {}
    state_dict = prog.state_dict

    arg_replacements = {}
    for input_name, state_name in buffers.items():
        if state_name in constants:
            arg_replacements[input_name] = constants[state_name]
        else:
            arg_replacements[input_name] = state_dict[state_name]

    g = prog.graph
    for node in list(g.nodes):
        if node.op != "placeholder":
            continue
        replacement = arg_replacements.get(node.name)
        if replacement is None:
            continue
        node.replace_all_uses_with(replacement)
        g.erase_node(node)

    return prog


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export nanochat GPT to StableHLO MLIR for PCP"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--block-size", type=int, default=DEFAULTS["block_size"])
    parser.add_argument("--vocab-size", type=int, default=DEFAULTS["vocab_size"])
    parser.add_argument("--n-layer", type=int, default=DEFAULTS["n_layer"])
    parser.add_argument("--n-head", type=int, default=DEFAULTS["n_head"])
    parser.add_argument("--n-kv-head", type=int, default=DEFAULTS["n_kv_head"])
    parser.add_argument("--n-embd", type=int, default=DEFAULTS["n_embd"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--dump-fx", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    config = GPTConfig(
        sequence_len=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(device)
    model.train()
    validate_model_contract(model)
    patch_rotary_cache(model, args.block_size)

    idx = torch.zeros((args.batch_size, args.block_size), dtype=torch.int64, device=device)
    targets = torch.zeros(
        (args.batch_size, args.block_size), dtype=torch.int64, device=device
    )

    params_dict = dict(model.named_parameters())
    param_names = list(params_dict.keys())
    param_values = list(params_dict.values())

    wrapper = StatelessWrapper(model, param_names)
    full_inputs = param_values + [idx, targets]

    output_path = (
        Path(args.out)
        if args.out is not None
        else REPO_ROOT / "pcp" / "models" / f"nanochat_forward_{args.block_size}.mlir"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dump_fx:
        fx_path = output_path.with_suffix(".fx.txt")
        fx_graph = maybe_dump_fx(wrapper, full_inputs, fx_path)
        if fx_graph:
            if "scaled_dot_product_attention" not in fx_graph:
                print("Warning: FX graph did not show scaled_dot_product_attention.")
            if "rms_norm" not in fx_graph:
                print("Warning: FX graph did not show rms_norm.")

    print(
        "Compiling nanochat with "
        f"layers={args.n_layer}, embd={args.n_embd}, heads={args.n_head}..."
    )
    prog = torch.export.export(wrapper, tuple(full_inputs), strict=False)
    prog = prog.run_decompositions(get_decomposition_table())
    prog = freeze_lifted_buffers(prog)
    module = fx.export_and_import(prog, output_type="stablehlo", decomposition_table={})

    mlir_text = str(module.operation)
    output_path.write_text(mlir_text)

    signature_ok, signature_msg = check_main_signature(mlir_text)
    unsupported_ops = scan_autodiff_coverage(mlir_text, REPO_ROOT)

    print(f"Exported params: {len(param_values)} tensors")
    print(f"Output: {output_path}")
    print(f"Signature check: {signature_msg}")
    if unsupported_ops:
        print("Warning: MLIR uses ops without VJP coverage:")
        print("  " + ", ".join(unsupported_ops))

    if not signature_ok:
        raise SystemExit("Signature validation failed.")


if __name__ == "__main__":
    main()
