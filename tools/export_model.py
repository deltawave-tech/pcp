import argparse
import sys
import os
import importlib.util
import torch
import torch.nn as nn
from torch_mlir import fx
from torch.func import functional_call


def load_model_from_path(path, class_name):
    """Dynamically loads a PyTorch Model class from a file."""
    spec = importlib.util.spec_from_file_location("user_model_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.append(
        os.path.dirname(path)
    )  # Allow relative imports inside the model file
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {path}")
    return getattr(module, class_name)


class StatelessWrapper(nn.Module):
    """
    Wraps a stateful nn.Module into a functional form for PCP.
    Signature: (params..., input_ids, targets) -> loss
    """

    def __init__(self, base_model, param_names):
        super().__init__()
        self.base_model = base_model
        self.param_names = param_names

    def forward(self, *args):
        # The last two arguments are data, everything before is params
        params = args[:-2]
        idx = args[-2]
        targets = args[-1]

        # Reconstruct parameter dictionary
        params_dict = {name: param for name, param in zip(self.param_names, params)}

        # Functional call: run model using these explicit parameters
        return functional_call(self.base_model, params_dict, (idx, targets))


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch models to StableHLO MLIR for PCP"
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="Path to python file containing model class (e.g., my_model.py)",
    )
    parser.add_argument(
        "--class-name",
        required=True,
        help="Name of the nn.Module class (e.g., NanoTransformer)",
    )
    parser.add_argument("--out", default="models/model.mlir", help="Output MLIR path")

    # Hyperparameters needed for tracing
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=65)

    args = parser.parse_args()

    # 1. Load User Model
    print(f"Loading {args.class_name} from {args.model_file}...")
    ModelClass = load_model_from_path(args.model_file, args.class_name)

    # Instantiate model (You might need to adjust this if your model takes init args)
    # For now, we assume models conform to the NanoGPT config style or have defaults
    try:
        model = ModelClass()
    except TypeError:
        print(
            "Error: Model __init__ requires arguments. Please modify export_model.py to pass your specific model config."
        )
        return

    model.train()

    # 2. Prepare Dummy Inputs
    print(f"Preparing inputs: Batch={args.batch_size}, Block={args.block_size}...")
    idx = torch.zeros((args.batch_size, args.block_size), dtype=torch.int64)
    targets = torch.zeros((args.batch_size, args.block_size), dtype=torch.int64)

    # 3. Extract Parameters
    params_dict = dict(model.named_parameters())
    param_names = list(params_dict.keys())
    param_values = list(params_dict.values())
    print(f"Found {len(param_values)} parameter tensors.")

    # 4. Wrap for Stateless Execution
    wrapper = StatelessWrapper(model, param_names)
    full_inputs = param_values + [idx, targets]

    # 5. Export to StableHLO
    print("Tracing and compiling to StableHLO...")
    module = fx.export_and_import(wrapper, *full_inputs, output_type="stablehlo")

    # 6. Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(str(module.operation))

    print(f"✓ Saved to {args.out}")
    print("\n--- PASTE THIS INTO YOUR experiment.json ---")
    print(
        "{"
        + f"""
    "model_path": "{args.out}",
    "batch_size": {args.batch_size},
    "block_size": {args.block_size},
    "tau": 10,
    "outer_loop_steps": 100,
    "learning_rate": 0.001
"""
        + "}"
    )


if __name__ == "__main__":
    main()
