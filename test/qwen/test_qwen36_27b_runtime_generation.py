"""
Qwen3.6-27B runtime generation contract checks.

Run from the repo root:
  python3 test/qwen/test_qwen36_27b_runtime_generation.py
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "experiments" / "inference_qwen36_27b.json"
GENERATION_CACHE_PATH = REPO_ROOT / "src" / "nodes" / "workers" / "generation_cache.zig"
GENERATION_ENGINE_PATH = REPO_ROOT / "src" / "nodes" / "workers" / "generation_engine.zig"
WORKER_PATH = REPO_ROOT / "src" / "nodes" / "workers" / "worker.zig"
MODEL_INTROSPECTION_PATH = REPO_ROOT / "src" / "mlir" / "model_introspection.zig"
INFERENCE_CONFIG_PATH = REPO_ROOT / "src" / "workloads" / "inference" / "config.zig"
INFERENCE_CONTROLLER_PATH = REPO_ROOT / "src" / "nodes" / "gateway" / "controllers" / "inference_controller.zig"
WEIGHTS_EXPORT_PATH = REPO_ROOT / "tools" / "export_qwen36_27b_weights.py"
ONE_TOKEN_EXPORT_PATH = REPO_ROOT / "tools" / "export_qwen36_27b_one_token.py"
SMOKE_RUNNER_PATH = REPO_ROOT / "src" / "examples" / "qwen36_27b_smoke.zig"


def require_file(path: Path) -> None:
    if path.exists():
        return
    try:
        import pytest  # type: ignore
    except ImportError:
        raise SystemExit(f"Missing required file: {path}") from None
    pytest.skip(f"Missing required file: {path}")


def test_qwen36_config_requests_streamable_bf16_weights() -> None:
    require_file(CONFIG_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["model_id"] == "qwen36-27b"
    assert config["weights_dtype"] == "bf16"
    assert config["weights_format"] == "bf16_streamed"
    assert "flat" not in config["weights_path"]
    assert "num_gen_data_inputs" not in config


def test_generation_state_copy_supports_replace_and_sequence_update() -> None:
    require_file(GENERATION_CACHE_PATH)
    source = GENERATION_CACHE_PATH.read_text(encoding="utf-8")

    assert "pub fn copyStateUpdate" in source
    assert "std.mem.eql(i64, state_shape, update_shape)" in source
    assert "@memcpy(state_buffer, update_bytes)" in source
    assert "copyCacheUpdateAtPosition" in source
    assert "generation state copy replaces recurrent state" in source


def test_generation_engine_can_stream_weights_without_single_host_blob() -> None:
    require_file(GENERATION_ENGINE_PATH)
    source = GENERATION_ENGINE_PATH.read_text(encoding="utf-8")

    assert "weights_path: ?[]const u8" in source
    assert "loadDeviceWeightsFromPath" in source
    assert "readNoEof(param_bytes)" in source
    assert "tensorByteSize(shape, dtype)" in source
    assert "WeightBufferSizeMismatch" in source
    assert "copyStateUpdate" in source


def test_parameter_dtypes_flow_to_streamed_weight_loader() -> None:
    for path in (MODEL_INTROSPECTION_PATH, INFERENCE_CONTROLLER_PATH, WORKER_PATH):
        require_file(path)

    introspection = MODEL_INTROSPECTION_PATH.read_text(encoding="utf-8")
    controller = INFERENCE_CONTROLLER_PATH.read_text(encoding="utf-8")
    worker = WORKER_PATH.read_text(encoding="utf-8")

    assert "parameter_dtypes: ?[]tensor.DType" in introspection
    assert "parameter_dtypes" in introspection
    assert 'payload.put("parameter_dtypes"' in controller
    assert 'payload.get("parameter_dtypes")' in worker
    assert ".weight_dtypes = param_dtypes" in worker


def test_worker_load_model_accepts_streaming_weight_format() -> None:
    for path in (WORKER_PATH, INFERENCE_CONFIG_PATH, INFERENCE_CONTROLLER_PATH):
        require_file(path)

    worker = WORKER_PATH.read_text(encoding="utf-8")
    inference_config = INFERENCE_CONFIG_PATH.read_text(encoding="utf-8")
    controller = INFERENCE_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "weights_format" in inference_config
    assert 'payload.put("weights_format"' in controller
    assert "stream_weights" in worker
    assert "streamed_weights.isManifestFormat(weights_format)" in worker
    assert '"sharded"' not in worker
    assert "allocator.alloc(u8, 0)" in worker
    assert "weights_path = if (stream_weights)" in worker


def test_qwen36_weight_exporter_streams_safetensor_shards() -> None:
    require_file(WEIGHTS_EXPORT_PATH)
    source = WEIGHTS_EXPORT_PATH.read_text(encoding="utf-8")

    assert "Qwen/Qwen3.6-27B" in source
    assert "model.safetensors.index.json" in source
    assert "parameter_names" in source
    assert "load_file(str(snapshot / shard_file), device=\"cpu\")" in source
    assert "converted.view(torch.uint16)" in source
    assert '"format": f"{args.dtype}_streamed"' in source


def test_full_model_one_token_export_and_smoke_paths_exist() -> None:
    for path in (ONE_TOKEN_EXPORT_PATH, SMOKE_RUNNER_PATH):
        require_file(path)

    exporter = ONE_TOKEN_EXPORT_PATH.read_text(encoding="utf-8")
    smoke = SMOKE_RUNNER_PATH.read_text(encoding="utf-8")

    assert "Qwen/Qwen3.6-27B" in exporter
    assert 'parser.add_argument("--output-type", default="stablehlo")' in exporter
    assert "object.__setattr__(self, \"_model\", model)" in exporter
    assert "self.lm_head(hidden_states).float()" in exporter
    assert '"parameter_names": parameter_names' in exporter
    assert "write_metadata(contract, args.metadata_path)" in (REPO_ROOT / "tools" / "export_qwen36_27b_generation.py").read_text(
        encoding="utf-8"
    )
    assert "MissingGenerationDataInputMetadata" in INFERENCE_CONTROLLER_PATH.read_text(encoding="utf-8")
    assert "Qwen3.6-27B smoke: uploaded" in smoke
    assert "backend.loadSession(vmfb)" in smoke
    assert "backend.moveToDevice(bytes, shape, dtype)" in smoke
    assert "executeWithDeviceBuffers(vmfb, \"main\", inputs.items)" in smoke


def main() -> None:
    test_qwen36_config_requests_streamable_bf16_weights()
    test_generation_state_copy_supports_replace_and_sequence_update()
    test_generation_engine_can_stream_weights_without_single_host_blob()
    test_parameter_dtypes_flow_to_streamed_weight_loader()
    test_worker_load_model_accepts_streaming_weight_format()
    test_qwen36_weight_exporter_streams_safetensor_shards()
    test_full_model_one_token_export_and_smoke_paths_exist()
    print("OK: Qwen3.6-27B runtime generation checks passed.")


if __name__ == "__main__":
    main()
