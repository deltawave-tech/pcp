import torch
import torch.nn as nn
from torch_mlir import fx
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoConfig
import sys
import os

# --- Configuration ---
# Qwen3-0.6B (Requested by user)
MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_PATH = "models/qwen3_0_6b.mlir"

# Training constraints
BATCH_SIZE = 1 
SEQ_LEN = 128  # Context length for training

# Wrapper to ensure the model returns ONLY the scalar loss
# and simplifies the interface for StableHLO export.
class QwenTrainingAdapter(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        print(f"Loading configuration for {model_id}...")
        self.config = AutoConfig.from_pretrained(model_id)
        self.config.use_cache = False # Critical for training/export
        
        # Try loading pretrained weights, fallback to random init
        try:
            print(f"Attempting to load pretrained weights from {model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(model_id, config=self.config)
            print("Loaded pretrained weights.")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights ({e}). Initializing random weights.")
            self.model = AutoModelForCausalLM.from_config(self.config)

        # Gradient Checkpointing (Optional - disable for simpler graph first)
        self.model.gradient_checkpointing_disable()

    def forward(self, input_ids, labels):
        # Qwen forward pass with labels computes loss automatically
        outputs = self.model(input_ids=input_ids, labels=labels, use_cache=False)
        return outputs.loss

class StatelessWrapper(nn.Module):
    def __init__(self, base_model, param_names):
        super().__init__()
        self.base_model = base_model
        self.param_names = param_names

    def forward(self, *args):
        # Architecture of arguments expected by PCP:
        # 1. All parameters (flattened list)
        # 2. Input Data (input_ids)
        # 3. Target Data (labels)
        
        params = args[:-2]
        input_ids = args[-2]
        labels = args[-1]
        
        # Reconstruct parameter dictionary
        params_dict = {name: param for name, param in zip(self.param_names, params)}
        
        # Functional call to execute model with these parameters
        # We treat 'base_model' as the QwenTrainingAdapter
        return functional_call(self.base_model, params_dict, (input_ids, labels))

def main():
    print("Initializing Qwen Model...")
    adapter = QwenTrainingAdapter(MODEL_ID)
    adapter.train()

    # Create dummy inputs for tracing
    print(f"Creating dummy inputs (Batch={BATCH_SIZE}, Seq={SEQ_LEN})...")
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.int64)
    labels = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.int64)

    # Extract parameters for stateless wrapper
    params_dict = dict(adapter.named_parameters())
    param_names = list(params_dict.keys())
    param_values = list(params_dict.values())
    
    print(f"Model has {len(param_values)} parameter tensors.")

    # Create stateless wrapper
    wrapper = StatelessWrapper(adapter, param_names)
    
    # inputs: [*params, input_ids, labels]
    full_inputs = param_values + [input_ids, labels]

    print("Compiling to StableHLO via torch-mlir...")
    try:
        # fx.export_and_import traces the model and produces MLIR
        module = fx.export_and_import(wrapper, *full_inputs, output_type="stablehlo")
        
        print(f"Export successful. Saving to {OUTPUT_PATH}...")
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            f.write(str(module.operation))
        print("Done.")
        
    except Exception as e:
        print(f"Export Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
