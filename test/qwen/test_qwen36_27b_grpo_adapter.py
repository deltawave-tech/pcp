"""
Qwen3.6-27B adapter-only GRPO milestone checks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "experiments" / "qwen36_27b_grpo_adapter_smoke.json"
TRAINING_MLIR_PATH = REPO_ROOT / "models" / "qwen36_27b_adapter_grpo_training.mlir"
TRAINING_META_PATH = REPO_ROOT / "models" / "qwen36_27b_adapter_grpo_training.mlir.meta.json"
GRPO_PATH = REPO_ROOT / "src" / "algorithms" / "grpo.zig"
RL_CONTROLLER_PATH = REPO_ROOT / "src" / "nodes" / "gateway" / "controllers" / "rl_controller.zig"
WORKER_PATH = REPO_ROOT / "src" / "nodes" / "workers" / "worker.zig"
TRAINING_CONTROLLER_PATH = REPO_ROOT / "src" / "nodes" / "gateway" / "controllers" / "training_controller.zig"
EXPORTER_PATH = REPO_ROOT / "tools" / "export_qwen36_27b_adapter_grpo.py"


def require_file(path: Path) -> None:
    if path.exists():
        return
    try:
        import pytest  # type: ignore
    except ImportError:
        raise SystemExit(f"Missing required file: {path}") from None
    pytest.skip(f"Missing required file: {path}")


def nix_bin() -> str | None:
    return shutil.which("nix") or (
        str(Path("/nix/var/nix/profiles/default/bin/nix"))
        if Path("/nix/var/nix/profiles/default/bin/nix").exists()
        else None
    )


def test_adapter_grpo_config_separates_frozen_base_and_trainable_adapter() -> None:
    require_file(CONFIG_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    grpo = config["grpo_config"]

    assert grpo["group_size"] == 4
    assert grpo["num_prompts"] == 1
    assert grpo["num_iterations"] == 1
    assert grpo["rollout_max_tokens"] == 1
    assert grpo["grpo_weight_mode"] == "adapter_only"
    assert grpo["weights_path"] == grpo["training_weights_path"]
    assert grpo["weights_path"].endswith("qwen36_27b_adapter_f32.bin")
    assert grpo["generation_weights_path"].endswith("qwen36_27b_bf16_streamed.bin")
    assert grpo["generation_weights_format"] == "bf16_streamed"
    assert grpo["trainable_parameter_indices"] == [0]
    assert "num_gen_data_inputs" not in grpo


def test_adapter_grpo_metadata_declares_optimizer_and_frozen_base() -> None:
    require_file(TRAINING_META_PATH)
    meta = json.loads(TRAINING_META_PATH.read_text(encoding="utf-8"))
    grpo_meta = meta["grpo_metadata"]

    assert meta["parameter_shapes"] == [[5120, 8]]
    assert meta["parameter_dtypes"] == ["f32"]
    assert meta["trainable_parameter_indices"] == [0]
    assert grpo_meta["mode"] == "adapter_only"
    assert grpo_meta["frozen_base_weights"]["format"] == "bf16_streamed"
    assert grpo_meta["delta_transport"]["scope"] == "adapter_only"
    assert grpo_meta["delta_transport"]["broadcasts_frozen_base"] is False
    assert grpo_meta["optimizer_state"][0]["parameter_index"] == 0


def test_tiny_adapter_grpo_gradients_match_pytorch() -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        try:
            import pytest  # type: ignore
        except ImportError:
            raise SystemExit("torch is required for adapter parity") from None
        pytest.skip("torch is required for adapter parity")

    torch.manual_seed(7)
    adapter = torch.randn(5120, 8, dtype=torch.float32, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float32)
    loss = (adapter * adapter).sum() + (advantages * advantages).sum()
    loss.backward()

    torch.testing.assert_close(adapter.grad, 2.0 * adapter, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(loss)


def test_runtime_uses_adapter_only_refresh_and_streamed_generation_weights() -> None:
    for path in (GRPO_PATH, RL_CONTROLLER_PATH, WORKER_PATH, TRAINING_CONTROLLER_PATH):
        require_file(path)

    grpo = GRPO_PATH.read_text(encoding="utf-8")
    controller = RL_CONTROLLER_PATH.read_text(encoding="utf-8")
    worker = WORKER_PATH.read_text(encoding="utf-8")
    training_controller = TRAINING_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert 'grpo_weight_mode, "adapter_only"' in grpo
    assert "generationWeightsPath" in grpo
    assert "trainingWeightsPath" in grpo
    assert "initialTrainingWeightsPath" in grpo
    assert "adapterStatePath" in grpo
    assert "adapterOptimizerStatePath" in grpo
    assert "Resuming adapter-only GRPO from adapter state" in grpo
    assert "Resumed adapter-only GRPO optimizer state" in grpo
    assert "Adapter-only GRPO optimizer state written" in grpo
    assert "frozen generation weights were not broadcast" in grpo
    assert "generation_weights_format" in grpo
    assert "GRPO adapter objective loss" in controller
    assert "gen_param_dtypes" in controller
    assert "weights_format" in training_controller
    assert "parameter_dtypes" in training_controller
    assert "generation_weights_path" in worker
    assert "bf16_streamed" in worker
    assert "self.generation_weights_path" in worker
    assert "std.base64.standard.Encoder.encode(b64_weights, weight_data)" in controller


def test_adapter_artifact_exporter_writes_small_state_and_prompt(tmp_path: Path) -> None:
    require_file(EXPORTER_PATH)
    adapter = tmp_path / "adapter.bin"
    prompts = tmp_path / "prompts.bin"
    subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(EXPORTER_PATH),
            "--adapter-output",
            str(adapter),
            "--prompt-output",
            str(prompts),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert adapter.stat().st_size == 5120 * 8 * 4
    assert prompts.stat().st_size > 0


def test_adapter_backward_builds_and_cuda_compiles_when_tools_are_available() -> None:
    require_file(TRAINING_MLIR_PATH)
    nix = nix_bin()
    if nix is None:
        try:
            import pytest  # type: ignore
        except ImportError:
            return
        pytest.skip("nix is required for the CUDA backward compile smoke")

    backward_mlir = Path("/tmp/qwen36_27b_adapter_grpo_backward.mlir")
    subprocess.run(
        [
            nix,
            "develop",
            "-c",
            "zig",
            "build",
            "build-grpo-backward",
            "--",
            str(TRAINING_MLIR_PATH),
            str(backward_mlir),
            "1",
            "0",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert backward_mlir.exists()

    subprocess.run(
        [
            nix,
            "develop",
            "-c",
            "iree-compile",
            str(backward_mlir),
            "--iree-hal-target-backends=cuda",
            "-o",
            "/tmp/qwen36_27b_adapter_grpo_backward_cuda.vmfb",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
