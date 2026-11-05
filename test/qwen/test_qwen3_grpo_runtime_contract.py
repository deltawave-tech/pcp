"""
Qwen3-8B GRPO runtime contract checks.

Run from the repo root:
  python3 test/qwen/test_qwen3_grpo_runtime_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "experiments" / "qwen3_8b_rl_test.json"
GRPO_PATH = REPO_ROOT / "src" / "algorithms" / "grpo.zig"
CONTROLLER_PATH = REPO_ROOT / "src" / "nodes" / "gateway" / "controllers" / "rl_controller.zig"
WORKER_PATH = REPO_ROOT / "src" / "nodes" / "workers" / "worker.zig"


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")


def test_qwen3_runtime_config_uses_small_local_refresh_contract() -> None:
    require_file(CONFIG_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    grpo = config["grpo_config"]

    assert config["data_path"] == "data/tiny_shakespeare.txt"
    assert config["tokenizer"] == "qwen"
    assert config["tau"] == 1
    assert config["outer_loop_steps"] == 1
    assert config["max_epochs"] == 1
    assert grpo["weights_path"] == "checkpoints/initial_weights/qwen3_8b_flat.bin"
    assert grpo["generation_vmfb_path"] == "models/qwen3_8b_rl_generation.vmfb"
    assert grpo["generation_mlir_path"] == "models/qwen3_8b_rl_generation.mlir"
    assert grpo["training_mlir_path"] == "models/qwen3_8b_grpo_training.mlir"
    assert grpo["num_gen_data_inputs"] == 74
    assert grpo["num_iterations"] == 1
    assert grpo["group_size"] == 4
    assert grpo["num_prompts"] == 1
    assert grpo["weight_refresh_strategy"] == "local_path"
    assert grpo["updated_weights_path"].startswith("/tmp/")
    assert grpo["trainable_parameter_indices"] == [397]


def test_qwen3_runtime_refreshes_weights_by_local_path() -> None:
    for path in (GRPO_PATH, CONTROLLER_PATH, WORKER_PATH):
        require_file(path)

    grpo_source = GRPO_PATH.read_text(encoding="utf-8")
    controller_source = CONTROLLER_PATH.read_text(encoding="utf-8")
    worker_source = WORKER_PATH.read_text(encoding="utf-8")

    assert 'weight_refresh_strategy, "local_path"' in grpo_source
    assert "refreshGenerationWeightsFromLocalSnapshot" in grpo_source
    assert "broadcastNewWeights" in grpo_source
    assert "writeWeightsSnapshot" in controller_source
    assert 'payload.put("weights_path"' in controller_source
    assert 'payload.get("weights_path")' in worker_source
    assert "loadGenerationWeights(weights_path, weights_format)" in worker_source


def test_qwen3_runtime_preserves_training_prompt_and_i32_inputs() -> None:
    require_file(CONTROLLER_PATH)
    require_file(WORKER_PATH)

    controller_source = CONTROLLER_PATH.read_text(encoding="utf-8")
    worker_source = WORKER_PATH.read_text(encoding="utf-8")

    assert 'result_payload.put("prompt"' in worker_source
    assert 'obj.get("prompt")' in controller_source
    assert "training_data_dtypes" in controller_source
    assert "buildGrpoBackwardPassForTrainableIndices" in controller_source
    assert "trainable_parameter_indices" in controller_source
    assert "applySurrogateTrainableSubsetUpdate" in controller_source
    assert "Surrogate Gradient Statistics" in controller_source
    assert "writeTokenBytes(input_ids_bytes, input_dtype" in controller_source
    assert "dtypes_list.append(input_dtype)" in controller_source
    assert "UnsupportedTrainingDataDType" in controller_source


def main() -> None:
    test_qwen3_runtime_config_uses_small_local_refresh_contract()
    test_qwen3_runtime_refreshes_weights_by_local_path()
    test_qwen3_runtime_preserves_training_prompt_and_i32_inputs()
    print("OK: Qwen3-8B GRPO runtime contract checks passed.")


if __name__ == "__main__":
    main()
