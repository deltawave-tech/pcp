# test_nanochat.py walkthrough

## General idea
This script is a small correctness check for exporting a GPT model with Torch Export. It:
- adds the repo root to the import path so the local `nanochat` package can be imported
- builds a small GPT model and creates random input/target tensors
- computes a reference loss with the normal PyTorch module
- wraps the model so parameters are provided as explicit inputs (using `functional_call`)
- exports that wrapper with `torch.export` and runs the exported module
- compares the two losses to confirm the export matches the original computation

## Line by line (matches the snippet as provided)
1. `import sys` - Import the system module to access `sys.path`.
2. `from pathlib import Path` - Import `Path` for filesystem path handling.
3. Blank line - Separate standard library imports from third-party imports.
4. `import torch` - Import PyTorch.
5. `from torch.func import functional_call` - Import the functional API used to call a module with explicit parameters.
6. Blank line - Separate imports from setup code.
7. `REPO_ROOT = Path(__file__).resolve().parents[1]` - Compute the repo root by resolving this file path and going two directories up.
8. `if str(REPO_ROOT) not in sys.path:` - Check whether the repo root is already on the Python import path.
9. `    sys.path.insert(0, str(REPO_ROOT))` - Prepend the repo root so local imports resolve to this checkout.
10. Blank line - Separate path setup from project imports.
11. `from nanochat.gpt import GPT, GPTConfig` - Import the model class and its config from the project.
12. Blank line - Separate imports from runtime code.
13. `torch.manual_seed(0)` - Fix the random seed for reproducible inputs.
14. `cfg = GPTConfig(sequence_len=32, vocab_size=65, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)` - Create a small GPT configuration.
15. `model = GPT(cfg).eval()` - Instantiate the model and set it to eval mode.
16. Blank line - Separate model setup from input creation.
17. `idx = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64)` - Create random input token IDs (batch 64, sequence length 32).
18. `targets = torch.randint(0, cfg.vocab_size, (64, 32), dtype=torch.int64)` - Create random target token IDs with the same shape.
19. Blank line - Separate input creation from execution.
20. `loss_pt = model(idx, targets)` - Run the model normally to get the reference loss.
21. Blank line - Separate reference run from parameter extraction.
22. `params = dict(model.named_parameters())` - Collect model parameters into a dict keyed by name.
23. `param_names = list(params.keys())` - Extract parameter names into a list.
24. `param_values = list(params.values())` - Extract parameter tensors into a list in the same order.
25. Blank line - Separate parameter extraction from the wrapper class.
26. `class Wrapper(torch.nn.Module):` - Define a wrapper module.
27. `    def __init__(self, base, names):` - Initialize the wrapper with the base model and parameter names.
28. `        super().__init__()` - Call the parent `nn.Module` constructor.
29. `        self.base = base` - Store the base model.
30. `        self.names = names` - Store the parameter name list.
31. `    def forward(self, *args):` - Define a forward that accepts all parameters plus inputs as positional args.
32. `        params_dict = {n: p for n, p in zip(self.names, args[:-2])}` - Rebuild a name->tensor dict from the parameter inputs (all but the last two args).
33. `        return functional_call(self.base, params_dict, (args[-2], args[-1]))` - Call the base model with explicit params and the input/target tensors.
34. Blank line - Separate class definition from usage.
35. `wrapper = Wrapper(model, param_names)` - Create the wrapper instance.
36. `full_inputs = param_values + [idx, targets]` - Build the full input list: all params, then `idx` and `targets`.
37. `ep = torch.export.export(wrapper, tuple(full_inputs), strict=False)` - Export the wrapper to an `ExportedProgram` using those inputs; `strict=False` relaxes export constraints.
38. `loss_export = ep.module()(*full_inputs)` - Run the exported module with the same inputs.
39. Blank line - Separate export run from printing results.
40. `print("loss_pt:", loss_pt.item())` - Print the original PyTorch loss.
41. `print("loss_export:", loss_export.item())` - Print the exported module loss.
42. `print("abs diff:", (loss_pt - loss_export).abs().item())` - Print the absolute difference between the two losses.
