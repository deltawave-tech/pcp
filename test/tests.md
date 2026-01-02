# Tests (NanoChat + PCP)

This repo has two different “layers” of checks:

- **Python script tests** in `test/nanochat/`: validate NanoChat’s PyTorch model code and the *export → StableHLO MLIR → IREE execution* path.
- **PCP runs** via the Zig binary (`pcp`): exercise PCP’s own autodiff, optimizer injection, networking, and distributed training loop.

The Python tests are meant to answer: “Does the model/math/export/forward-loss match?”  
They do **not** prove that PCP’s training/gradients/distributed execution are correct.

## Prerequisites

You typically want **both** of these available when running the Python tests:

- **Nix dev shell**: provides IREE tools (`iree-compile`, `iree-run-module`) + compatible runtime libs.
- **Python venv**: provides `torch` and `torch-mlir` (`pip install -r requirements.txt`).
- **pcp binary**: the parity tests call `pcp --export-training-artifacts`, so build `./result/bin/pcp` (or `./zig-out/bin/pcp`) first.
  - The Python helpers prefer `./zig-out/bin/pcp` if it exists. Override with `PCP_BIN=/path/to/pcp`.

Recommended way to run a test from `pcp/`:

```bash
nix develop -c ./venv/bin/python test/nanochat/<test_file>.py
```

If importing `torch` fails with `OSError: libstdc++.so.6: cannot open shared object file`, prepend a GCC lib dir to `LD_LIBRARY_PATH`:

```bash
LD_LIBRARY_PATH="$(ls -d /nix/store/*gcc-*-lib/lib | head -n1):$LD_LIBRARY_PATH" \
  nix develop -c ./venv/bin/python test/nanochat/<test_file>.py
```

If you need very verbose Zig-side debug logs, rebuild PCP with:

```bash
nix develop -c zig build -Dpcp_verbose_logs=true
```

## Python Tests

### `test/nanochat/test_gpt_ops_compat.py`

**Goal:** sanity-check that key ops implemented in `nanochat.gpt` match PyTorch behavior.

**What it checks:**
- `nanochat.gpt.norm(x)` matches a manual RMSNorm reference (and `torch.nn.functional.rms_norm` if available).
- `nanochat.gpt.scaled_dot_product_attention(...)` matches `torch.nn.functional.scaled_dot_product_attention(...)`, including the GQA (`enable_gqa`) path when supported by your PyTorch.

**How to run:**
```bash
nix develop -c ./venv/bin/python test/nanochat/test_gpt_ops_compat.py
```

**What a pass looks like:** printed max diffs + `OK: gpt.py compatibility checks passed.`

---

### `test/nanochat/test_torch_export_smoke.py`

**Goal:** ensure NanoChat’s `GPT` forward+loss can be captured and executed via `torch.export`.

**Why this exists:** if `torch.export` can’t capture the model, downstream steps like torch-mlir export will be fragile or impossible.

**What it checks:**
- Builds a small deterministic `GPT`.
- Computes eager PyTorch loss.
- Exports a stateless wrapper using `torch.export.export(...)`.
- Runs the exported program and asserts the loss matches eager within tolerance.

**How to run:**
```bash
nix develop -c ./venv/bin/python test/nanochat/test_torch_export_smoke.py
```

---

### `test/nanochat/test_generate_nanochat_mlir_forward_loss.py`

**Goal:** end-to-end check that `tools/generate_nanochat.py` produces StableHLO MLIR that (when compiled and run with IREE) matches PyTorch’s **forward loss**.

**Pipeline:**
1. Build a deterministic PyTorch `GPT` and extract its parameters.
2. Run `tools/generate_nanochat.py` to generate StableHLO MLIR (`func.func @main`).
3. Validate the MLIR `@main` signature matches the inputs we will pass (param count/dtypes, and strict shape for `idx/targets`).
4. Compile MLIR → VMFB using `iree-compile`.
5. Run VMFB using `iree-run-module` and compare `loss_mlir` vs `loss_pt` on 2 different random batches.

**How to run:**
```bash
nix develop -c ./venv/bin/python test/nanochat/test_generate_nanochat_mlir_forward_loss.py
```

**Important limitation:** this is **forward-only**. It does not validate gradients or PCP’s Zig autodiff/VJP rules.
If `tools/generate_nanochat.py` prints “ops without VJP coverage”, forward correctness can still be fine, but PCP training/backprop may be incomplete until those VJPs exist.

---

### `test/nanochat/test_pcp_gradients_parity.py`

**Goal:** compare PyTorch autograd gradients to PCP’s Zig autodiff gradients (from the training VMFB).

**What it checks:**
- Uses `pcp --export-training-artifacts` to build a training VMFB that includes the gradient function.
- Runs `model_forward_pass_grad` via `iree-run-module`.
- Compares parameter gradients to PyTorch, after applying the same elementwise gradient clamp used in PCP.

**How to run:**
```bash
nix develop -c ./venv/bin/python test/nanochat/test_pcp_gradients_parity.py
```

**GPU (example NVIDIA):**
```bash
PCP_TEST_BACKEND=cuda PCP_TEST_TARGET=sm_89 PCP_TEST_IREE_DEVICE=cuda \
  nix develop -c ./venv/bin/python test/nanochat/test_pcp_gradients_parity.py
```

If you have an older NVIDIA driver (e.g., 470.x) and see `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`,
also set `PCP_IREE_CUDA_TARGET_FEATURES=+ptx74`.

---

### `test/nanochat/test_pcp_training_step_parity.py`

**Goal:** compare full AdamW training steps (params + m + v + loss) between PCP and a PyTorch reference.

**What it checks:**
- Uses the training VMFB generated by `pcp --export-training-artifacts`.
- Runs two steps of `main` with deterministic batches.
- Matches updated parameters, M/V states, and loss against a PyTorch implementation of the same AdamW math and clipping.

**How to run:**
```bash
nix develop -c ./venv/bin/python test/nanochat/test_pcp_training_step_parity.py
```

**GPU (example NVIDIA):**
```bash
PCP_TEST_BACKEND=cuda PCP_TEST_TARGET=sm_89 PCP_TEST_IREE_DEVICE=cuda \
  nix develop -c ./venv/bin/python test/nanochat/test_pcp_training_step_parity.py
```

If you have an older NVIDIA driver (e.g., 470.x) and see `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`,
also set `PCP_IREE_CUDA_TARGET_FEATURES=+ptx74`.

---

### `test/nanochat/test_pcp_diloco_outer_step_parity.py`

**Goal:** validate a full DiLoCo **outer round** (multi-worker, H inner steps, delta averaging, Nesterov update) without involving the dataset pipeline.

**What it checks:**
- Simulates `K` workers and `H` inner steps using the **PCP training VMFB** (`main`).
- Uses deterministic synthetic `(idx, targets)` batches per `(worker, step)` so runs are reproducible.
- Computes `delta_i = master - worker_final` per worker, then `delta_avg = mean(delta_i)`.
- Applies host-side Nesterov update exactly like `src/optimizers/nesterov.zig:64`.
- Compares master params (and Nesterov velocity buffers) against a PyTorch reference implementation of the same inner-step math.

**How to run:**
```bash
nix develop -c ./venv/bin/python test/nanochat/test_pcp_diloco_outer_step_parity.py
```

**What a pass looks like:** per-tensor max diffs + `OK: DiLoCo outer-step parity matches (multi-worker synthetic).`

## PCP Run (Shepherd)

Example command:

```bash
cd /home/ahmet/project/nanochat/pcp
./result/bin/pcp --shepherd --config experiments/nanochat.json --host 0.0.0.0 --port 18080 --workers 1
```

Tip: add `--no-dashboard` for non-interactive runs (CI, tmux logging, subprocess tests).

**What it does:**
- Starts PCP in **Shepherd** mode (coordinator).
- Loads hyperparameters and paths from `experiments/nanochat.json`.
- Binds a TCP server on `0.0.0.0:18080`.
- Waits until `--workers 1` worker connects, then starts the training loop.

To actually run training, you also need a worker process. For a single-machine CPU smoke run:

```bash
./result/bin/pcp --worker --host 127.0.0.1 --port 18080 --backend cpu
```

## PCP System Smoke Test (Shepherd + Worker)

### `test/nanochat/test_pcp_shepherd_worker_smoke.py`

**Goal:** exercise the real PCP networking/runtime loop end-to-end.

**What it checks:**
- Launches a local shepherd and a local worker (CPU backend).
- Runs a tiny config (1 outer step, 1 inner step).
- Asserts: shepherd exits successfully, logs contain no `NaN`, and a checkpoint file is written.
  - The shepherd is started with `--no-dashboard` so the TUI doesn’t interfere with non-interactive execution.

**How to run:**
```bash
nix develop -c ./venv/bin/python test/nanochat/test_pcp_shepherd_worker_smoke.py
```

**What a pass looks like:** `OK: shepherd+worker smoke run completed (finite loss, checkpoint written).`

On a multi-GPU node, use the node manager (spawns supervised workers):

```bash
./result/bin/pcp --node-manager --scale 8 --host <SHEPHERD_IP> --port 18080 --backend cuda --target sm_90a
```

## “Robust on a bigger machine” checklist (what to add next)

If you want confidence beyond forward-loss parity:

- **Training-step sanity:** run a few PCP training steps and assert loss stays finite (no NaNs) and changes as expected.
- **VJP/autodiff coverage gating:** treat “ops without VJP coverage” warnings as failures (or add missing VJPs).
- **Backend coverage:** at least one smoke run per backend you care about (CPU, CUDA, ROCm), because codegen/runtime differences can matter.
- **Distributed correctness:** 2+ workers, verify synchronization works and runs don’t deadlock.
- **Checkpoint/resume:** start, checkpoint, restart with `--resume`, confirm training continues from the same run state.
